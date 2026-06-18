from __future__ import annotations

import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Vercel serverless freezes the process after each response, so background
# threads never complete and the 4.5 MB body limit blocks real audio files.
# Force DSP offline on Vercel so the UI stays honest about what it can do.
_ON_VERCEL = bool(os.environ.get("VERCEL"))

if _ON_VERCEL:
    DSP_AVAILABLE = False
else:
    try:
        from eightd_engine.audio_io import export_audio, load_audio, to_seekable_wav
        from eightd_engine.dsp import (
            AudioData,
            analyze_correlation,
            measure_loudness_file,
            bpm_to_premium_rotation_cpm,
            estimate_bpm,
            panning_preset_names,
            process_8d,
            render_8d_to_wav,
            render_8d_file_to_wav,
        )
        import soundfile as _sf
        from eightd_engine.stems import (
            StemData,
            StemSeparationUnavailable,
            available_stem_mode,
            process_stem_spatial_mix,
            separate_stems_from_file,
        )
        DSP_AVAILABLE = True
    except ImportError:
        DSP_AVAILABLE = False

APP_DIR = Path(tempfile.gettempdir()) / "8d_dropzone_live"
APP_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SECONDS = 60 * 60
MAX_UPLOAD_MINUTES = MAX_UPLOAD_SECONDS // 60
MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
JOBS_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="The 8D Engine")

# Allow any origin to call the API directly.
# This lets the Vercel front-end (and local dev) bypass the proxy and talk to
# Railway in a single hop — critical for large audio uploads and fast downloads.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- Social layer (additive). Guarded like the DSP imports: if its deps are
# missing or the DB can't init, the audio tool keeps working untouched. ----
SOCIAL_AVAILABLE = False
try:
    from starlette.middleware.sessions import SessionMiddleware
    from social.routes import router as social_router
    from social.realtime import rt_router as social_rt_router
    from social.billing import billing_router as social_billing_router
    from social.db import init_db as _social_init_db

    app.add_middleware(
        SessionMiddleware,
        secret_key=os.environ.get("SESSION_SECRET", "dev-insecure-change-me"),
        same_site="lax",
        https_only=bool(os.environ.get("SESSION_HTTPS")),
    )
    app.include_router(social_router)
    app.include_router(social_rt_router)
    app.include_router(social_billing_router)
    _social_init_db()

    # CSRF defense: reject cross-origin state-changing requests to /social.
    # (Belt-and-suspenders with SameSite=lax cookies.) The Stripe webhook is
    # server-to-server with no Origin and is signature-verified, so it's exempt.
    from urllib.parse import urlparse as _urlparse
    _ALLOWED_HOSTS = {h for h in [
        os.environ.get("PUBLIC_WS_BASE", ""),
        "8d-drop-zone.vercel.app",
        "luminous-endurance-production-0696.up.railway.app",
    ] if h}

    @app.middleware("http")
    async def _csrf_origin_guard(request, call_next):
        if (request.method in ("POST", "PUT", "PATCH", "DELETE")
                and request.url.path.startswith("/social")
                and request.url.path != "/social/stripe/webhook"):
            origin = request.headers.get("origin")
            if origin:
                oh = _urlparse(origin).netloc
                allowed = {request.url.netloc} | _ALLOWED_HOSTS
                if oh and oh not in allowed:
                    return JSONResponse({"detail": "Cross-origin request blocked."}, status_code=403)
        return await call_next(request)
    SOCIAL_AVAILABLE = True
    print("[social] enabled at /social", flush=True)
except Exception as _social_err:  # pragma: no cover - keeps audio app alive
    print(f"[social] disabled: {_social_err}", flush=True)

app.mount("/files", StaticFiles(directory=str(APP_DIR)), name="files")


@dataclass(frozen=True)
class MixInstructionResult:
    """Deterministic interpretation of user mix-refinement prompts.

    This intentionally stays local/no-LLM so the render is fast, repeatable, and
    private. It maps mix-engineer language to the parameters available in the
    current master-file DSP pipeline.
    """

    settings: dict
    notes: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def apply_mix_instructions(base_settings: dict, prompt: str = "") -> MixInstructionResult:
    """Map natural-language mix instructions to spatial-render parameters.

    Supported examples:
    - "keep the vocal center/front"
    - "less movement, more subtle"
    - "make the guitar more spatial"
    - "less reverb" / "bigger room"
    - "more dramatic 8D"

    Because uploads are finished stereo masters, element-specific control is
    approximate: vocals/body are handled by center-focus band anchoring, bass is
    already mono-safe, and guitars/ambience/highs are pushed via high emphasis.
    """

    settings = dict(base_settings)
    notes: list[str] = []
    text = " ".join((prompt or "").lower().split())
    if not text:
        return MixInstructionResult(settings=settings, notes=notes)

    def add_note(note: str):
        if note not in notes:
            notes.append(note)

    vocal_terms = ("vocal", "voice", "lead", "singer", "lyrics", "lyric")
    center_terms = ("center", "centre", "front", "fixed", "still", "static", "anchor", "anchored", "one position", "without motion", "no motion")
    if any(term in text for term in vocal_terms) and any(term in text for term in center_terms):
        settings["center_focus"] = max(settings.get("center_focus", 0.0), 0.90)
        settings["motion_depth"] = min(settings.get("motion_depth", 0.7), 0.56)
        settings["spatial_mix"] = min(settings.get("spatial_mix", 0.6), 0.52)
        add_note("Lead vocal/body anchored front-center; motion shifted toward highs, reverb, and ambience.")

    if any(term in text for term in ("bass", "sub", "kick", "808", "low end", "low-end", "drums", "drum")) and any(term in text for term in center_terms):
        add_note("Low end/kick remain mono center below 150 Hz.")

    if any(term in text for term in ("guitar", "guitars", "harmony", "harmonies", "delay", "echo", "echoes", "reverb tail", "ambience", "ambient", "background")):
        if any(term in text for term in ("move", "moving", "spatial", "around", "wide", "wider", "surround", "8d")):
            settings["high_emphasis"] = max(settings.get("high_emphasis", 0.6), 0.78)
            settings["room_size"] = max(settings.get("room_size", 0.18), 0.23)
            settings["spatial_mix"] = max(settings.get("spatial_mix", 0.58), 0.66)
            add_note("Guitar/ambience/high-detail material given stronger spatial motion.")

    if any(term in text for term in ("less movement", "less motion", "subtle", "gentle", "not too much", "reduce motion", "calmer")):
        settings["motion_depth"] = min(settings.get("motion_depth", 0.7), 0.55)
        settings["spatial_mix"] = min(settings.get("spatial_mix", 0.6), 0.52)
        settings["center_focus"] = max(settings.get("center_focus", 0.0), 0.78)
        add_note("Overall orbit restrained for a subtler, less seasick render.")

    if any(term in text for term in ("more movement", "more motion", "dramatic", "powerful", "stronger 8d", "more 8d", "wider", "wide")):
        settings["motion_depth"] = max(settings.get("motion_depth", 0.7), 0.82)
        settings["spatial_mix"] = max(settings.get("spatial_mix", 0.6), 0.72)
        settings["high_emphasis"] = max(settings.get("high_emphasis", 0.6), 0.74)
        add_note("Spatial orbit made wider and more dramatic while bass remains protected.")

    if any(term in text for term in ("felt", "feel", "feels", "physical", "body", "tactile", "immersive", "deep", "presence", "impact")):
        settings["felt_presence"] = max(settings.get("felt_presence", 0.62), 0.88)
        settings["high_emphasis"] = max(settings.get("high_emphasis", 0.6), 0.76)
        settings["motion_depth"] = max(settings.get("motion_depth", 0.7), 0.80)
        settings["spatial_mix"] = max(settings.get("spatial_mix", 0.6), 0.70)
        settings["room_size"] = max(settings.get("room_size", 0.18), 0.22)
        add_note("Felt-presence layer enabled: stronger pinna/air motion plus centered tactile punch.")

    if any(term in text for term in ("less reverb", "dry", "drier", "less room", "reduce room")):
        settings["room_size"] = min(settings.get("room_size", 0.18), 0.10)
        add_note("Room/reverb reduced for a drier, more direct master.")
    elif any(term in text for term in ("more reverb", "bigger room", "larger room", "more room", "wet", "wetter", "space", "spacious")):
        settings["room_size"] = max(settings.get("room_size", 0.18), 0.30)
        add_note("Room/reflection amount increased for more externalized space.")

    if any(term in text for term in ("clean", "remove static", "less static", "hiss", "noise")):
        settings["denoise_amount"] = max(settings.get("denoise_amount", 0.72), 0.82)
        add_note("Static/hiss cleanup increased before spatialization.")

    for key in ("room_size", "motion_depth", "high_emphasis", "spatial_mix", "center_focus", "denoise_amount", "felt_presence"):
        if key in settings:
            settings[key] = _clamp01(settings[key])

    if not notes:
        add_note("Prompt saved with this render; no specific DSP keyword override was detected, so the selected profile was used.")
    return MixInstructionResult(settings=settings, notes=notes)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The 8D Engine — Spatial Audio Mastering</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root{
      color-scheme: dark;
      --void:#02040c; --void2:#05070f; --deep:#070b16;
      --hair:rgba(130,180,255,0.12);
      --hair2:rgba(150,205,255,0.30);
      --ink:#eaf1ff;
      --muted:rgba(198,213,240,0.78);
      --soft:rgba(150,172,210,0.64);
      --cyan:#62e0ff; --ice:#a9ecff; --violet:#9d8bff; --gold:#e9d2a3;
      --success:#74f5c0; --danger:#ff9a9a;
      --shadow:0 40px 120px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
      --mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
      --display:'Space Grotesk', system-ui, sans-serif;
    }
    *{ box-sizing:border-box; }
    html{ min-height:100%; background:var(--void); }
    body{
      margin:0; min-height:100vh; overflow-x:hidden; color:var(--ink);
      font-family:Inter, system-ui, -apple-system, Segoe UI, sans-serif;
      background:
        radial-gradient(60vw 50vw at 50% -10%, rgba(98,224,255,.10), transparent 60%),
        radial-gradient(50vw 40vw at 88% 8%, rgba(157,139,255,.12), transparent 60%),
        linear-gradient(180deg, #02040c 0%, #04060f 45%, #060912 100%);
    }
    .space{ position:fixed; inset:0; pointer-events:none; z-index:-3; }
    .stars{
      background-image:
        radial-gradient(1px 1px at 18px 32px, rgba(255,255,255,.75), transparent),
        radial-gradient(1px 1px at 92px 132px, rgba(180,222,255,.6), transparent),
        radial-gradient(1.6px 1.6px at 168px 70px, rgba(255,255,255,.5), transparent),
        radial-gradient(1px 1px at 210px 190px, rgba(160,200,255,.5), transparent);
      background-size:240px 240px; animation:drift 140s linear infinite;
    }
    .stars2{
      opacity:.6;
      background-image:
        radial-gradient(1px 1px at 60px 80px, rgba(255,255,255,.5), transparent),
        radial-gradient(1.2px 1.2px at 140px 20px, rgba(200,230,255,.45), transparent),
        radial-gradient(1px 1px at 30px 200px, rgba(255,255,255,.4), transparent);
      background-size:320px 320px; animation:drift 90s linear infinite reverse;
    }
    .aurora{
      background:
        radial-gradient(38vw 38vw at 10% 6%, rgba(98,224,255,.14), transparent 60%),
        radial-gradient(42vw 42vw at 92% 12%, rgba(157,139,255,.14), transparent 60%),
        radial-gradient(50vw 40vw at 50% 116%, rgba(233,210,163,.07), transparent 60%);
      animation:breathe 18s ease-in-out infinite alternate;
    }
    .grid{
      z-index:-2; opacity:.5;
      background-image:
        linear-gradient(rgba(120,180,255,.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120,180,255,.045) 1px, transparent 1px);
      background-size:64px 64px;
      -webkit-mask-image: radial-gradient(circle at 50% 0%, black, transparent 72%);
      mask-image: radial-gradient(circle at 50% 0%, black, transparent 72%);
    }
    @keyframes drift{ to{ background-position:240px 480px; } }
    @keyframes breathe{ from{ opacity:.7; transform:scale(1);} to{ opacity:1; transform:scale(1.06);} }
    @keyframes spin{ to{ transform:rotate(360deg); } }
    @keyframes blink{ 0%,100%{opacity:1;} 50%{opacity:.35;} }

    .shell{ width:min(1200px, calc(100vw - 40px)); margin:0 auto; padding:30px 0 60px; position:relative; }
    .tlabel{ font-family:var(--mono); font-size:10.5px; letter-spacing:.34em; text-transform:uppercase; color:var(--soft); }

    .nav{ display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:46px;
      padding:12px 16px; border:1px solid var(--hair); border-radius:16px;
      background:linear-gradient(180deg, rgba(12,18,34,.6), rgba(7,11,22,.4)); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); }
    .brand{ display:flex; align-items:center; gap:13px; color:var(--ink); text-decoration:none; }
    .mark{ position:relative; width:34px; height:34px; border-radius:50%; display:grid; place-items:center; font-size:15px; color:#04121a;
      background:radial-gradient(circle at 35% 30%, #fff, var(--ice) 40%, var(--cyan)); box-shadow:0 0 22px rgba(98,224,255,.5), inset 0 0 8px rgba(255,255,255,.6); }
    .mark::after{ content:""; position:absolute; inset:-6px; border-radius:50%; border:1px solid rgba(98,224,255,.4); border-top-color:transparent; border-left-color:transparent; animation:spin 7s linear infinite; }
    .word{ font-family:var(--display); font-weight:600; font-size:16px; letter-spacing:.22em; }
    .nav-right{ display:flex; align-items:center; gap:18px; }
    .sys{ display:inline-flex; align-items:center; gap:8px; font-family:var(--mono); font-size:10.5px; letter-spacing:.22em; color:var(--muted); text-transform:uppercase; }
    .nav-note{ font-family:var(--mono); font-size:10.5px; letter-spacing:.24em; color:var(--soft); text-transform:uppercase; }
    .pulse{ width:7px; height:7px; border-radius:50%; background:var(--success); box-shadow:0 0 14px var(--success); animation:blink 2.4s ease-in-out infinite; }

    .hero{ display:grid; grid-template-columns:minmax(0,1.05fr) minmax(380px,.95fr); gap:30px; align-items:stretch; }
    .copy{ padding:8px 0 0; }
    .eyebrow{ display:inline-flex; align-items:center; gap:10px; padding:8px 14px; border:1px solid var(--hair); border-radius:999px; background:rgba(98,224,255,.05); font-family:var(--mono); font-size:10.5px; font-weight:500; letter-spacing:.26em; text-transform:uppercase; color:var(--muted); }
    h1{ margin:22px 0 18px; max-width:760px; font-family:var(--display); font-size:clamp(42px,6.4vw,78px); line-height:.96; letter-spacing:-.03em; font-weight:500; }
    .grad{ background:linear-gradient(100deg, #fff 6%, var(--ice) 44%, var(--cyan) 78%, var(--violet) 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .lede{ max-width:600px; color:var(--muted); font-size:clamp(16px,1.9vw,19px); line-height:1.6; margin:0 0 30px; }
    .proof{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; max-width:620px; }
    .proof div{ position:relative; border:1px solid var(--hair); background:linear-gradient(180deg, rgba(13,20,38,.55), rgba(7,11,22,.35)); border-radius:16px; padding:14px 16px; min-height:96px; overflow:hidden; }
    .proof div::before{ content:""; position:absolute; left:0; top:0; width:100%; height:1px; background:linear-gradient(90deg, transparent, var(--hair2), transparent); }
    .proof strong{ display:block; font-family:var(--display); color:var(--ink); font-size:26px; letter-spacing:-.02em; margin:8px 0 4px; }
    .proof strong i{ font-style:normal; font-size:14px; color:var(--cyan); margin-left:3px; }
    .proof span:not(.tlabel){ color:var(--soft); font-size:12px; line-height:1.4; display:block; }
    .badges{ margin-top:24px; display:flex; flex-wrap:wrap; gap:9px; }
    .badge{ font-family:var(--mono); border:1px solid var(--hair); background:rgba(255,255,255,.025); color:var(--muted); border-radius:999px; padding:8px 12px; font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; transition:border-color .2s, color .2s, box-shadow .2s; }
    .badge:hover{ border-color:var(--hair2); color:var(--ink); box-shadow:0 0 18px rgba(98,224,255,.12); }
    .badge{ cursor:pointer; }
    .badge:focus-visible{ outline:1px solid var(--cyan); outline-offset:2px; }
    .badge[aria-pressed="true"]{ border-color:var(--cyan); color:var(--ink); box-shadow:0 0 18px rgba(98,224,255,.18); background:rgba(98,224,255,.06); }
    .feat-detail{ margin-top:14px; max-width:620px; min-height:0; }
    .feat-detail.show{ border:1px solid var(--hair); border-radius:14px; padding:13px 16px; background:linear-gradient(180deg, rgba(13,20,38,.55), rgba(7,11,22,.35)); }
    .feat-detail b{ font-family:var(--display); color:var(--ink); font-weight:500; font-size:13px; }
    .feat-detail p{ color:var(--soft); font-size:12.5px; line-height:1.6; margin:6px 0 0; }

    #zone{ position:relative; overflow:hidden; min-height:640px; border:1px solid var(--hair2); border-radius:28px;
      background:linear-gradient(180deg, rgba(14,20,38,.72), rgba(7,11,22,.6)); backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
      box-shadow:var(--shadow); padding:30px; display:flex; flex-direction:column; justify-content:space-between;
      transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease; isolation:isolate; }
    #zone::before{ content:""; position:absolute; inset:-40%; z-index:-2; opacity:.5;
      background: conic-gradient(from 140deg, transparent, rgba(98,224,255,.16), transparent 35%, rgba(157,139,255,.14), transparent 70%);
      animation:spin 22s linear infinite; }
    #zone::after{ content:""; position:absolute; inset:1px; border-radius:27px; z-index:-1; pointer-events:none;
      background:linear-gradient(180deg, rgba(255,255,255,.05), transparent 30%); }
    #zone.hover{ transform:translateY(-3px); border-color:rgba(98,224,255,.7); box-shadow:0 40px 140px rgba(40,120,190,.4), var(--shadow); }
    .corner{ position:absolute; width:18px; height:18px; border:1.5px solid rgba(98,224,255,.55); z-index:2; }
    .corner.tl{ left:14px; top:14px; border-right:0; border-bottom:0; }
    .corner.tr{ right:14px; top:14px; border-left:0; border-bottom:0; }
    .corner.bl{ left:14px; bottom:14px; border-right:0; border-top:0; }
    .corner.br{ right:14px; bottom:14px; border-left:0; border-top:0; }

    .visual{ display:grid; place-items:center; padding:14px 0 4px; pointer-events:none; user-select:none; }
    .orbit{ width:min(330px,72vw); aspect-ratio:1; border-radius:50%; position:relative; display:grid; place-items:center;
      border:1px solid rgba(150,205,255,.18); background:radial-gradient(circle, rgba(98,224,255,.06), rgba(255,255,255,.01) 55%, transparent); }
    .orbit::before, .orbit::after{ content:""; position:absolute; border-radius:50%; border:1px solid rgba(157,139,255,.16); }
    .orbit::before{ inset:13%; transform:rotate(28deg) scaleY(.58); }
    .orbit::after{ inset:26%; border-color:rgba(98,224,255,.16); transform:rotate(-31deg) scaleY(.62); }
    .sweep{ position:absolute; inset:0; border-radius:50%; background:conic-gradient(from 0deg, rgba(98,224,255,.32), rgba(98,224,255,0) 28%); animation:spin 5.5s linear infinite;
      -webkit-mask:radial-gradient(circle, transparent 26%, #000 27%); mask:radial-gradient(circle, transparent 26%, #000 27%); }
    .dot{ position:absolute; width:12px; height:12px; border-radius:50%; background:#fff; box-shadow:0 0 28px rgba(169,236,255,.9), 0 0 8px var(--cyan);
      offset-path: path('M 165 24 C 247 28 307 95 304 168 C 301 246 237 306 162 302 C 83 298 26 240 27 164 C 29 86 88 23 165 24'); animation:travel 10.4s linear infinite; }
    @keyframes travel{ to{ offset-distance:100%; } }
    .wave{ width:70%; height:80px; opacity:.9; filter:drop-shadow(0 0 22px rgba(98,224,255,.18)); }

    .zone-copy{ text-align:center; max-width:560px; margin:0 auto; width:100%; }
    .kicker{ font-family:var(--mono); color:var(--cyan); font-size:10.5px; text-transform:uppercase; letter-spacing:.28em; margin-bottom:10px; }
    .title{ font-family:var(--display); font-size:clamp(26px,3vw,36px); line-height:1.02; font-weight:600; letter-spacing:-.02em; margin-bottom:12px; }
    .hint{ color:var(--muted); line-height:1.55; font-size:14.5px; }
    .controls{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:24px; }
    .field{ text-align:left; }
    .controls > .launch{ grid-column:1 / -1; }
    label{ display:block; font-family:var(--mono); color:var(--soft); font-size:10px; letter-spacing:.2em; text-transform:uppercase; margin:0 0 8px 2px; }
    select, button, textarea{ font:inherit; }
    select{ width:100%; min-height:52px; padding:0 40px 0 15px; border:1px solid var(--hair); border-radius:13px; background:rgba(4,7,15,.7); color:var(--ink); font-weight:500; outline:none;
      appearance:none; -webkit-appearance:none;
      background-image:linear-gradient(45deg, transparent 50%, var(--cyan) 50%), linear-gradient(135deg, var(--cyan) 50%, transparent 50%);
      background-position:calc(100% - 20px) 23px, calc(100% - 15px) 23px; background-size:6px 6px, 6px 6px; background-repeat:no-repeat; }
    select:focus{ border-color:rgba(98,224,255,.6); box-shadow:0 0 0 4px rgba(98,224,255,.12); }
    textarea{ width:100%; min-height:100px; resize:vertical; padding:14px 16px; border:1px solid var(--hair); border-radius:14px; background:rgba(4,7,15,.7); color:var(--ink); outline:none; line-height:1.5; }
    textarea:focus{ border-color:rgba(98,224,255,.6); box-shadow:0 0 0 4px rgba(98,224,255,.1); }
    textarea::placeholder{ color:rgba(200,214,240,.42); }
    .prompt-box{ margin-top:14px; text-align:left; }
    .prompt-help{ margin:8px 0 0 2px; color:var(--soft); font-size:11.5px; line-height:1.45; }
    button.launch{ min-height:52px; padding:0 22px; border:0; border-radius:13px; cursor:pointer; white-space:nowrap; font-family:var(--display); font-weight:600; letter-spacing:.06em;
      color:#04121a; background:linear-gradient(135deg, var(--ice), var(--cyan) 55%, var(--violet)); box-shadow:0 14px 40px rgba(98,224,255,.28), inset 0 1px 0 rgba(255,255,255,.4);
      transition:transform .18s ease, filter .18s ease, box-shadow .18s ease; }
    button.launch:hover{ transform:translateY(-1px); filter:brightness(1.05); box-shadow:0 18px 52px rgba(98,224,255,.42), inset 0 1px 0 rgba(255,255,255,.4); }
    input{ display:none; }
    .bar{ width:100%; height:7px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; margin:18px auto 0; display:none; }
    .fill{ width:0%; height:100%; background:linear-gradient(90deg, var(--cyan), var(--violet)); border-radius:999px; transition:width .15s linear; box-shadow:0 0 16px rgba(98,224,255,.5); }
    .fill.indeterminate{ width:32%; animation:load 1s infinite ease-in-out; }
    @keyframes load{ 0%{ transform:translateX(-115%);} 100%{ transform:translateX(330%);} }
    .status-card{ margin-top:18px; border:1px solid var(--hair); border-radius:16px; padding:13px 16px; background:rgba(3,6,14,.5); text-align:left; }
    .status-card .tlabel{ display:block; margin-bottom:6px; color:var(--cyan); }
    .status{ color:var(--ink); font-family:var(--mono); white-space:pre-line; font-size:12px; line-height:1.5; }
    a{ color:var(--success); font-weight:600; text-decoration:none; }
    a:hover{ text-decoration:underline; }

    .section-head{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin:64px 0 18px; }
    .section-head h2{ font-family:var(--display); font-weight:500; font-size:clamp(22px,2.6vw,30px); letter-spacing:-.02em; margin:0; }
    .systems{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }
    .systems div{ position:relative; border:1px solid var(--hair); border-radius:16px; padding:16px; background:linear-gradient(180deg, rgba(13,20,38,.5), rgba(7,11,22,.32)); transition:border-color .2s, box-shadow .2s, transform .2s; }
    .systems div:hover{ border-color:var(--hair2); box-shadow:0 0 22px rgba(98,224,255,.1); transform:translateY(-2px); }
    .systems strong{ display:block; font-family:var(--display); font-weight:600; font-size:14px; margin:6px 0 6px; }
    .systems span:not(.tlabel){ display:block; color:var(--soft); font-size:11.5px; line-height:1.4; }
    .systems .tlabel{ color:var(--cyan); }

    .profiles{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    .profile{ position:relative; display:flex; gap:14px; align-items:flex-start; border:1px solid var(--hair); border-radius:16px; padding:16px; background:linear-gradient(180deg, rgba(13,20,38,.5), rgba(7,11,22,.32)); transition:border-color .2s, box-shadow .2s, transform .2s; }
    .profile:hover{ border-color:var(--hair2); box-shadow:0 0 22px rgba(157,139,255,.12); transform:translateY(-2px); }
    .profile .idx{ font-family:var(--mono); font-size:12px; color:var(--cyan); padding-top:2px; min-width:22px; }
    .profile b{ display:block; font-family:var(--display); font-weight:600; font-size:14px; letter-spacing:-.01em; margin-bottom:4px; }
    .profile small{ color:var(--soft); font-size:11.5px; line-height:1.4; }
    .profile{ cursor:pointer; }
    .profile:focus-visible{ outline:1px solid var(--cyan); outline-offset:2px; }
    .profile[aria-pressed="true"]{ border-color:var(--cyan); box-shadow:0 0 24px rgba(80,180,255,.20); background:linear-gradient(180deg, rgba(18,30,54,.66), rgba(9,14,28,.42)); }
    .profile[aria-pressed="true"] .idx{ color:var(--cyan); }
    .profile-detail{ margin-top:14px; border:1px solid var(--hair); border-radius:16px; padding:18px 20px; background:linear-gradient(180deg, rgba(13,20,38,.6), rgba(7,11,22,.4)); }
    .profile-detail h3{ font-family:var(--display); font-weight:600; font-size:16px; margin:0; letter-spacing:-.01em; display:flex; align-items:center; gap:10px; }
    .profile-detail .pd-idx{ font-family:var(--mono); font-size:12px; color:var(--cyan); }
    .profile-detail .pd-tag{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); border:1px solid var(--hair); border-radius:999px; padding:3px 10px; margin-left:auto; }
    .profile-detail p{ color:var(--soft); font-size:13px; line-height:1.65; margin:10px 0 0; }
    .profile-detail .pd-best{ color:var(--ink); }

    .foot{ display:flex; flex-direction:column; gap:12px; margin-top:64px; padding-top:20px; border-top:1px solid var(--hair); }
    .foot-row{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }
    .foot .word{ font-family:var(--display); letter-spacing:.22em; font-size:13px; }
    .foot .credit{ font-family:var(--mono); font-size:11px; letter-spacing:.04em; color:var(--soft); }
    .foot .credit b{ color:var(--ink); font-weight:500; }

    @media (max-width:980px){
      .hero{ grid-template-columns:1fr; }
      .systems{ grid-template-columns:repeat(2,1fr); }
      .profiles{ grid-template-columns:repeat(2,1fr); }
      #zone{ min-height:auto; }
    }
    @media (max-width:560px){
      .shell{ width:min(100vw - 22px,1200px); }
      .nav-note{ display:none; }
      .controls{ grid-template-columns:1fr; }
      .proof, .systems, .profiles{ grid-template-columns:1fr; }
      .orbit{ width:260px; }
    }
    @media (prefers-reduced-motion: reduce){ .dot, .sweep, #zone::before, .stars, .stars2, .aurora, .mark::after, .fill.indeterminate{ animation:none; } }
  </style>
</head>
<body>
  <div class="space stars" aria-hidden="true"></div>
  <div class="space stars2" aria-hidden="true"></div>
  <div class="space aurora" aria-hidden="true"></div>
  <div class="space grid" aria-hidden="true"></div>
  <main class="shell">
    <nav class="nav" aria-label="Command bar">
      <a class="brand" href="/" aria-label="The 8D Engine home"><span class="mark">◌</span><span class="word">THE&nbsp;8D&nbsp;ENGINE</span></a>
      <div class="nav-right">
        <span class="sys"><span class="pulse"></span> All systems nominal</span>
        <a href="/mixer" style="font-family:var(--mono,monospace);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#9fb0c8;border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:8px 15px;text-decoration:none">Multitrack mixer</a>
        <a href="/social" style="font-family:var(--mono,monospace);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#06101c;background:linear-gradient(135deg,#62e0ff,#9d8bff);border-radius:999px;padding:8px 15px;text-decoration:none">Community / Sign in</a>
      </div>
    </nav>
    <section class="hero">
      <div class="copy">
        <div class="eyebrow"><span class="pulse"></span> Orbital spatial mastering</div>
        <h1>Spatial masters engineered <span class="grad">in the ethers</span>, tuned for depth, dimension, and soul level resonance.</h1>
        <p class="lede">Dock a track, choose a flight profile, and the engine renders a polished binaural orbit — felt-presence panning, mono-safe bass punch, subtle room, and pristine 32-bit WAV detail.</p>
        <div class="proof" aria-label="Reference telemetry">
          <div><span class="tlabel">Reference orbit</span><strong>7.7<i>s</i></strong><span>Tight, centered, polished motion path.</span></div>
          <div><span class="tlabel">Bass lock</span><strong>150<i>Hz</i></strong><span>Kick &amp; sub stay mono-centered.</span></div>
          <div><span class="tlabel">Stereo field</span><strong>0.60</strong><span>Median side/mid width target.</span></div>
        </div>
        <div class="badges" aria-label="Onboard systems">
          <span class="badge" data-feat="bpm_aware" role="button" tabindex="0" aria-pressed="false">BPM aware</span>
          <span class="badge" data-feat="static_cleanup" role="button" tabindex="0" aria-pressed="false">Static cleanup</span>
          <span class="badge" data-feat="golden_ratio" role="button" tabindex="0" aria-pressed="false">Golden ratio motion</span>
          <span class="badge" data-feat="fibonacci_timing" role="button" tabindex="0" aria-pressed="false">Fibonacci timing</span>
          <span class="badge" data-feat="felt_presence" role="button" tabindex="0" aria-pressed="false">Felt-presence panning</span>
          <span class="badge" data-feat="ai_stems" role="button" tabindex="0" aria-pressed="false">AI stem separation</span>
        </div>
        <div class="feat-detail" id="featDetail" aria-live="polite"></div>
      </div>
      <div id="zone">
        <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
        <div class="visual" aria-hidden="true">
          <div class="orbit">
            <span class="sweep"></span>
            <span class="dot"></span>
            <svg class="wave" viewBox="0 0 420 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M6 60 C34 19 64 103 96 60 C126 18 158 104 190 60 C222 17 254 103 286 60 C318 21 352 99 414 60" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/>
              <path d="M20 60 H400" stroke="rgba(255,255,255,.16)" stroke-width="1" stroke-dasharray="6 12"/>
              <defs><linearGradient id="g" x1="0" x2="420" y1="0" y2="0"><stop stop-color="#a9ecff"/><stop offset=".55" stop-color="#62e0ff"/><stop offset="1" stop-color="#9d8bff"/></linearGradient></defs>
            </svg>
          </div>
        </div>
        <div class="zone-copy">
          <div class="kicker">Docking bay // Import audio</div>
          <div class="title" id="title">Drop your track here</div>
          <div class="hint" id="hint">We analyze tempo and render a premium binaural orbit while keeping the sub-bass and kick centered. 1 hour max per upload. MP3, WAV, FLAC, M4A, and most FFmpeg-decodable files are accepted.</div>
          <div class="controls">
            <div class="field">
              <label for="preset">Flight profile</label>
              <select id="preset">
                <option value="binaural_8d" selected>8D Binaural Mix — wide 10.4s orbit, bass-safe & felt</option>
                <option value="clean_reference">Clean Reference — polished 7.7s orbit</option>
                <option value="reference_luxe">Reference Luxe — 10.4s orbit</option>
                <option value="phi_reference_orbit">Golden Ratio Reference — φ-timed orbit</option>
                <option value="fibonacci_spiral">Fibonacci Spiral — golden-angle path</option>
                <option value="golden_figure8">Golden Figure 8 — φ front/back sweep</option>
                <option value="lucas_breath">Lucas Breath — slow Fibonacci halo</option>
                <option value="fibonacci_waltz">The Fibonacci Waltz — triple-meter golden sway you feel</option>
                <option value="fibonacci_magic">Fibonacci Magic — shimmering node-hopping sparkle</option>
                <option value="opus_fibonacci">Opus Fibonacci — flagship grand orbit, deepest feel</option>
                <option value="fireflies_plus">Fireflies Plus — smooth premium orbit</option>
                <option value="cinematic_halo">Cinematic Halo — elegant atmospheric surround</option>
                <option value="figure8">Figure 8 — front/back immersive sweep</option>
                <option value="wide_orbit">Wide Orbit — powerful chorus motion</option>
                <option value="vocal_safe">Vocal Safe — clear center, gentle motion</option>
              </select>
            </div>
            <div class="field">
              <label for="stemMode">Processing core</label>
              <select id="stemMode">
                <option value="classic" selected>Classic full-mix spatial master</option>
                <option value="ai_stems">AI stem spatial mix — vocals/drums/bass/instruments</option>
              </select>
            </div>
            <div class="field">
              <label for="fmt">Export format</label>
              <select id="fmt">
                <option value="wav" selected>WAV — 32-bit float (studio)</option>
                <option value="mp3">MP3 — 320 kbps (share)</option>
                <option value="flac">FLAC — lossless compressed</option>
              </select>
            </div>
            <button class="launch" onclick="document.getElementById('file').click()">Select track</button>
          </div>
          <div class="prompt-box">
            <label for="mixPrompt">Mix command console</label>
            <textarea id="mixPrompt" placeholder="Example: Make the 8D felt by the listener, keep the vocal front-center, keep drums static, make guitar echoes wider."></textarea>
            <p class="prompt-help">Prompt the spatial renderer before upload. Controls can anchor vocal/body, protect bass/drums, widen guitars/ambience/highs, add felt-presence panning, adjust room, cleanup, and overall motion.</p>
          </div>
          <input id="file" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.aac,.ogg,.aiff">
          <div class="bar" id="bar"><div class="fill"></div></div>
          <div class="status-card"><span class="tlabel">Telemetry</span><div class="status" id="status">Standby · Ready for upload.</div></div>
          <div id="abplayer"></div>
        </div>
      </div>
    </section>

    <div class="section-head"><h2>Onboard signal chain</h2><span class="tlabel">Core systems</span></div>
    <section class="systems" aria-label="Signal chain">
      <div><span class="tlabel">DSP</span><strong>64-bit core</strong><span>High-precision internal processing before export.</span></div>
      <div><span class="tlabel">Low end</span><strong>Mono-safe bass</strong><span>Sub and kick stay centered below 150 Hz.</span></div>
      <div><span class="tlabel">Motion</span><strong>Binaural orbit</strong><span>ITD, ILD, rear shading, smooth azimuth motion.</span></div>
      <div><span class="tlabel">Feel</span><strong>Felt presence</strong><span>Pinna-air cues and centered tactile punch.</span></div>
      <div><span class="tlabel">Export</span><strong>32-bit WAV</strong><span>Float export keeps detail, avoids brittle renders.</span></div>
    </section>

    <div class="section-head"><h2>Fifteen mastering orbits</h2><span class="tlabel">Flight profiles · tap to select</span></div>
    <section class="profiles" aria-label="Flight profiles">
      <div class="profile" data-preset="binaural_8d" role="button" tabindex="0" aria-pressed="false"><span class="idx">01</span><div><b>8D Binaural Mix</b><small>Wide 10.4s orbit — bass-safe, felt.</small></div></div>
      <div class="profile" data-preset="clean_reference" role="button" tabindex="0" aria-pressed="false"><span class="idx">02</span><div><b>Clean Reference</b><small>Polished 7.7s orbit — tight and centered.</small></div></div>
      <div class="profile" data-preset="reference_luxe" role="button" tabindex="0" aria-pressed="false"><span class="idx">03</span><div><b>Reference Luxe</b><small>Expansive 10.4s premium orbit.</small></div></div>
      <div class="profile" data-preset="phi_reference_orbit" role="button" tabindex="0" aria-pressed="false"><span class="idx">04</span><div><b>Golden Ratio Reference</b><small>φ-timed orbit for organic motion.</small></div></div>
      <div class="profile" data-preset="fibonacci_spiral" role="button" tabindex="0" aria-pressed="false"><span class="idx">05</span><div><b>Fibonacci Spiral</b><small>Golden-angle spiral path.</small></div></div>
      <div class="profile" data-preset="golden_figure8" role="button" tabindex="0" aria-pressed="false"><span class="idx">06</span><div><b>Golden Figure 8</b><small>φ front/back sweep.</small></div></div>
      <div class="profile" data-preset="lucas_breath" role="button" tabindex="0" aria-pressed="false"><span class="idx">07</span><div><b>Lucas Breath</b><small>Slow Fibonacci halo motion.</small></div></div>
      <div class="profile" data-preset="fibonacci_waltz" role="button" tabindex="0" aria-pressed="false"><span class="idx">08</span><div><b>The Fibonacci Waltz</b><small>Triple-meter golden sway.</small></div></div>
      <div class="profile" data-preset="fibonacci_magic" role="button" tabindex="0" aria-pressed="false"><span class="idx">09</span><div><b>Fibonacci Magic</b><small>Shimmering node-hopping sparkle.</small></div></div>
      <div class="profile" data-preset="opus_fibonacci" role="button" tabindex="0" aria-pressed="false"><span class="idx">10</span><div><b>Opus Fibonacci</b><small>Flagship grand orbit, deepest feel.</small></div></div>
      <div class="profile" data-preset="fireflies_plus" role="button" tabindex="0" aria-pressed="false"><span class="idx">11</span><div><b>Fireflies Plus</b><small>Smooth premium orbit shimmer.</small></div></div>
      <div class="profile" data-preset="cinematic_halo" role="button" tabindex="0" aria-pressed="false"><span class="idx">12</span><div><b>Cinematic Halo</b><small>Elegant atmospheric surround.</small></div></div>
      <div class="profile" data-preset="figure8" role="button" tabindex="0" aria-pressed="false"><span class="idx">13</span><div><b>Figure 8</b><small>Front/back immersive sweep.</small></div></div>
      <div class="profile" data-preset="wide_orbit" role="button" tabindex="0" aria-pressed="false"><span class="idx">14</span><div><b>Wide Orbit</b><small>Powerful chorus-width motion.</small></div></div>
      <div class="profile" data-preset="vocal_safe" role="button" tabindex="0" aria-pressed="false"><span class="idx">15</span><div><b>Vocal Safe</b><small>Clear center, gentle motion.</small></div></div>
    </section>
    <div class="profile-detail" id="profileDetail" aria-live="polite"></div>

    <footer class="foot">
      <div class="foot-row">
        <span class="word">THE 8D ENGINE</span>
        <span class="tlabel">64-bit DSP · Mono-safe bass · 32-bit WAV export</span>
      </div>
      <div class="credit">The 8D Engine — created by <b>Christ Dejon</b> and <b>Noel De Brackinghe</b></div>
    </footer>
  </main>
<script>
const zone = document.getElementById('zone');
const file = document.getElementById('file');
const title = document.getElementById('title');
const hint = document.getElementById('hint');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const preset = document.getElementById('preset');
const stemMode = document.getElementById('stemMode');
const mixPrompt = document.getElementById('mixPrompt');
const fmtSel = document.getElementById('fmt');
let lastOriginalFile = null;

// ── Direct API routing ────────────────────────────────────────────────────────
// Uploads and downloads go straight to Railway, skipping the Vercel proxy.
// Benefits:
//   • No 4.5 MB Vercel body-size limit — files of any size upload cleanly
//   • One network hop instead of two — faster upload AND download
//   • Real-time XHR progress works correctly (no proxy buffering)
// Local dev uses the same origin so relative paths work unchanged.
const IS_DEV = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const API = IS_DEV ? '' : 'https://luminous-endurance-production-0696.up.railway.app';

// ── DSP availability check ────────────────────────────────────────────────────
let DSP_OK = true;
(async () => {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    DSP_OK = d.dsp_available;
    if (!DSP_OK) {
      document.querySelector('.kicker').textContent = 'Engine offline · try again shortly';
      title.textContent = 'Processing engine offline';
      hint.textContent = 'The spatial mastering engine is temporarily unavailable. Please try again in a few minutes, or run the engine locally.';
      statusEl.textContent = 'Engine offline.\\n\\nTo run locally: python -m uvicorn web_app:app --port 8765';
      const btn = document.querySelector('button.launch');
      btn.textContent = 'Engine unavailable';
      btn.disabled = true;
      btn.style.opacity = '0.35';
      btn.style.cursor = 'not-allowed';
      btn.onclick = null;
    }
  } catch(e) {}
})();

zone.addEventListener('dragenter', e => { e.preventDefault(); if (DSP_OK) zone.classList.add('hover'); title.textContent = DSP_OK ? 'Release to master' : 'Run locally to master tracks'; });
zone.addEventListener('dragover',  e => { e.preventDefault(); });
zone.addEventListener('dragleave', e => { if (zone.contains(e.relatedTarget)) return; zone.classList.remove('hover'); if (!file.files.length) title.textContent = DSP_OK ? 'Drop your track here' : 'Run locally to master tracks'; });
zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('hover'); title.textContent = DSP_OK ? 'Drop your track here' : 'Run locally to master tracks'; const f = e.dataTransfer.files[0]; if (f && DSP_OK) upload(f); });
file.addEventListener('change', e => { const f = e.target.files[0]; if (f && DSP_OK) upload(f); });

const MAX_UPLOAD_MB = 200;
async function upload(f) {
  if (!DSP_OK) return;
  lastOriginalFile = f;
  const mb = (f.size / 1048576).toFixed(1);
  if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
    title.textContent = 'File too large';
    hint.textContent = `${f.name} · ${mb} MB`;
    statusEl.textContent = `Please upload a track ${MAX_UPLOAD_MB} MB or smaller.`;
    return;
  }
  title.textContent = 'Uploading…';
  hint.textContent = `${f.name} · ${mb} MB`;
  statusEl.textContent = 'Sending directly to the mastering engine…';
  bar.style.display = 'block';
  document.querySelector('.fill').style.width = '0%';
  document.querySelector('.fill').classList.remove('indeterminate');
  const data = new FormData();
  data.append('file', f);
  data.append('preset', preset.value);
  data.append('stem_mode', stemMode.value);
  data.append('mix_prompt', mixPrompt.value.trim());
  data.append('fmt', fmtSel ? fmtSel.value : 'wav');
  try {
    const json = await xhrUpload(`${API}/convert`, data, pct => {
      statusEl.textContent = `Uploading: ${pct}% of ${mb} MB`;
      document.querySelector('.fill').style.width = `${pct}%`;
    });
    title.textContent = 'Rendering spatial master…';
    statusEl.textContent = 'Upload complete. Tempo analysis, cleanup, binaural panning, room, and phase guard running now.';
    document.querySelector('.fill').classList.add('indeterminate');
    await pollJob(json.job_id);
  } catch (err) {
    title.textContent = 'Render failed';
    hint.textContent = 'Check that the file is a valid audio format and try again.';
    let msg = err.message || String(err);
    try { const d = JSON.parse(msg); if (d.detail) msg = d.detail; } catch(_) {}
    statusEl.textContent = msg;
  } finally {
    bar.style.display = 'none';
  }
}

function xhrUpload(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress(Math.min(100, Math.round((e.loaded / e.total) * 100)));
    };
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) return reject(new Error(xhr.responseText));
      resolve(JSON.parse(xhr.responseText));
    };
    xhr.onerror = () => reject(new Error('Upload failed — check your connection and try again'));
    xhr.send(formData);
  });
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`${API}/jobs/${jobId}`);
    if (!res.ok) { let t = await res.text(); try { const d = JSON.parse(t); if (d.detail) t = d.detail; } catch(_) {} throw new Error(t); }
    const job = await res.json();
    if (job.status === 'complete') {
      title.textContent = 'Spatial master ready';
      // Download URL is relative to Railway — prepend API base for a direct link.
      const dlUrl = job.download_url.startsWith('http') ? job.download_url : `${API}${job.download_url}`;
      let links = `<a href="${dlUrl}" download>⬇ Download ${job.output_name}</a>`;
      if (job.stems_url) {
        const stemsUrl = job.stems_url.startsWith('http') ? job.stems_url : `${API}${job.stems_url}`;
        links += `<br><a href="${stemsUrl}" download>⬇ Download stems (${job.stem_count} files, .zip)</a>`;
      }
      hint.innerHTML = links;
      const loud = (job.lufs != null) ? `\\nLoudness: ${job.lufs} LUFS | True peak: ${job.true_peak} dBTP` : '';
      statusEl.textContent = `Profile: ${job.preset}\\nMode: ${job.stem_mode || 'classic'} (${job.stem_engine || 'full mix'})\\nPrompt: ${job.mix_notes || 'Selected profile only'}\\nBPM: ${job.bpm}\\nOrbit: ${job.rotation_cpm} cycles/min${loud}\\nCorrelation: ${job.correlation} | Side/Mid: ${job.side_mid_ratio} | ${job.phase}`;
      buildABPlayer(dlUrl, lastOriginalFile);
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || 'Render failed');
    statusEl.textContent = `${job.message || 'Rendering…'}\\nYou can leave this tab open until the download link appears.`;
    await new Promise(r => setTimeout(r, 1500));
  }
}

// ── A/B compare + waveform (Web Audio) ─────────────────────────────────────────
let _abCtx = null;
async function buildABPlayer(resultUrl, originalFile) {
  const el = document.getElementById('abplayer');
  if (!el) return;
  el.innerHTML = '<div class="muted" style="margin-top:12px;font-size:12px">Loading A/B preview…</div>';
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) { el.innerHTML = ''; return; }
    _abCtx = _abCtx || new AC();
    const ctx = _abCtx;
    const [resBuf, origBuf] = await Promise.all([
      fetch(resultUrl).then(r => r.arrayBuffer()).then(a => ctx.decodeAudioData(a)),
      originalFile ? originalFile.arrayBuffer().then(a => ctx.decodeAudioData(a)).catch(() => null) : Promise.resolve(null),
    ]);
    el.innerHTML = `
      <style>.abtab{cursor:pointer;border:1px solid rgba(255,255,255,.18);background:transparent;color:#8a93a8;border-radius:999px;padding:5px 10px;font-family:monospace;font-size:10px;letter-spacing:.08em}.abtab.on{color:#06101c;background:#62e0ff;border-color:transparent}</style>
      <div style="margin-top:14px;border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:12px 14px;background:rgba(13,20,38,.4)">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <button id="abPlay" type="button" style="cursor:pointer;border:none;border-radius:999px;width:38px;height:38px;background:linear-gradient(135deg,#62e0ff,#9d8bff);color:#06101c;font-size:15px">&#9658;</button>
          <button id="abA" type="button" class="abtab">A · Original</button>
          <button id="abB" type="button" class="abtab on">B · 8D</button>
          <span id="abTime" style="margin-left:auto;font-family:monospace;font-size:11px;color:#8a93a8">0:00</span>
        </div>
        <canvas id="abWave" style="width:100%;height:64px;display:block"></canvas>
      </div>`;
    const canvas = document.getElementById('abWave');
    const playBtn = document.getElementById('abPlay'), btnA = document.getElementById('abA'), btnB = document.getElementById('abB'), timeEl = document.getElementById('abTime');
    if (!origBuf) { btnA.disabled = true; btnA.style.opacity = 0.4; btnA.title = 'Original preview unavailable'; }
    drawWave(canvas, resBuf, 0);
    let src = null, which = 'B', playing = false, offset = 0, startedAt = 0, rafId = 0;
    const activeBuf = () => (which === 'A' && origBuf ? origBuf : resBuf);
    const pos = () => playing ? Math.min(activeBuf().duration, offset + (ctx.currentTime - startedAt)) : offset;
    const fmtT = s => { s = Math.max(0, s | 0); return (s / 60 | 0) + ':' + String(s % 60).padStart(2, '0'); };
    function frame() {
      if (!playing) return;
      const p = pos();
      timeEl.textContent = fmtT(p);
      drawWave(canvas, resBuf, p / resBuf.duration);
      if (p >= activeBuf().duration - 0.03) { stopSrc(); playing = false; offset = 0; playBtn.innerHTML = '&#9658;'; drawWave(canvas, resBuf, 0); timeEl.textContent = '0:00'; return; }
      rafId = requestAnimationFrame(frame);
    }
    function stopSrc() { if (src) { try { src.stop(); } catch (e) {} src.disconnect(); src = null; } cancelAnimationFrame(rafId); }
    function startSrc() { const b = activeBuf(); src = ctx.createBufferSource(); src.buffer = b; src.connect(ctx.destination); offset = Math.min(offset, b.duration - 0.03); startedAt = ctx.currentTime; src.start(0, Math.max(0, offset)); rafId = requestAnimationFrame(frame); }
    playBtn.onclick = () => { ctx.resume(); if (playing) { offset = pos(); stopSrc(); playing = false; playBtn.innerHTML = '&#9658;'; } else { playing = true; playBtn.innerHTML = '&#10073;&#10073;'; startSrc(); } };
    function setWhich(w) { if (w === which) return; const p = pos(), wasPlaying = playing; if (playing) stopSrc(); which = w; offset = p; btnA.classList.toggle('on', w === 'A'); btnB.classList.toggle('on', w === 'B'); if (wasPlaying) { playing = true; startSrc(); } }
    btnA.onclick = () => { if (origBuf) setWhich('A'); };
    btnB.onclick = () => setWhich('B');
  } catch (e) {
    el.innerHTML = '<div class="muted" style="margin-top:10px;font-size:12px">A/B preview unavailable for this file.</div>';
  }
}
function drawWave(canvas, buf, playhead) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 560, h = 64;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const data = buf.getChannelData(0), step = Math.max(1, (data.length / w) | 0), mid = h / 2;
  g.strokeStyle = 'rgba(98,224,255,.55)'; g.lineWidth = 1; g.beginPath();
  for (let x = 0; x < w; x++) { let mn = 1, mx = -1; const s = x * step; for (let i = 0; i < step; i++) { const v = data[s + i] || 0; if (v < mn) mn = v; if (v > mx) mx = v; } g.moveTo(x, mid + mn * mid); g.lineTo(x, mid + mx * mid); }
  g.stroke();
  if (playhead != null) { const px = Math.max(0, Math.min(1, playhead)) * w; g.strokeStyle = '#9d8bff'; g.lineWidth = 2; g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke(); }
}

/* Flight-profile cards act as a selector: tap one to choose it (syncs the
   dropdown above) and reveal its full write-up below. */
const PROFILES = {
  binaural_8d:{tag:'Fixed · ~10.4 s', detail:`<p>The flagship, modeled on the classic Owl City "Fireflies (8D)" sound — but rebuilt the right way. A smooth, wide orbit circles the air and instruments around your head about once every 10.4 seconds, while the sub-bass and kick stay locked dead-center and the lead vocal is held up front. You get the full "moving around you" effect of viral 8D without the seasick low-end swing, and it folds down to mono far more gracefully than the tracks it imitates.</p><p class="pd-best"><b>Best for:</b> almost anything — the safest and most impressive all-rounder, and the default.</p>`},
  clean_reference:{tag:'Fixed · 7.7 s', detail:`<p>The tightest, most restrained profile. A quicker 7.7-second orbit with a narrower image, a stronger center, darker air and punch-safe bass — present but subtle and broadcast-clean, with the lowest dizziness of the set.</p><p class="pd-best"><b>Best for:</b> vocal-forward pop, loudness/streaming-sensitive masters, and listeners new to 8D.</p>`},
  reference_luxe:{tag:'Fixed · ~10.4 s', detail:`<p>The premium reference sweep: a broad 10.4-second orbit with wide mids and highs, gentle non-mechanical drift, and protected mono bass. Obvious and luxurious without tipping into novelty.</p><p class="pd-best"><b>Best for:</b> full-band productions and modern pop/EDM that want a rich, clearly-moving spatial master.</p>`},
  phi_reference_orbit:{tag:'Fixed · φ-timed', detail:`<p>Reference Luxe translated through the golden ratio. The same ~10.4-second orbit, but its small timing and position offsets are divided by powers of φ so the motion never settles into an obvious loop — it stays organic and alive.</p><p class="pd-best"><b>Best for:</b> when you love the reference feel but want subtle, ever-shifting life over long listens.</p>`},
  fibonacci_spiral:{tag:'Fixed · φ spiral', detail:`<p>A spiral whose segments last 1, 1, 2, 3, 5, 8, 13 parts of each orbit and aim toward golden-angle points around you — quick darting passes interleaved with long, elegant sweeps. Its golden-angle timing keeps the path from ever quite repeating.</p><p class="pd-best"><b>Best for:</b> evolving electronic, cinematic and ambient material that rewards non-repeating motion.</p>`},
  golden_figure8:{tag:'Fixed · φ figure-8', detail:`<p>A front-to-back figure-eight with φ-spaced lobes: the left/right side passes stay smooth while the front and rear transitions breathe at golden-ratio rates. It adds depth (front-back) rather than just width.</p><p class="pd-best"><b>Best for:</b> tracks that need a sense of sound passing in front of and behind you.</p>`},
  lucas_breath:{tag:'Fixed · Lucas timing', detail:`<p>A slow expansion-and-contraction driven by Lucas numbers (2, 1, 3, 4, 7, 11) — an elegant breathing halo and the gentlest motion in the φ family, with the lowest nausea risk.</p><p class="pd-best"><b>Best for:</b> ambient, downtempo, meditation, sleep and very long listening sessions.</p>`},
  fibonacci_waltz:{tag:'Beat-synced · 3/4', detail:`<p>A triple-meter sway: golden-angle nodes grouped 3-2-1 so the image rocks "one-two-three" with a φ-rate lilt, beat-synced to your track and tuned for deep felt presence. Graceful, never dizzy.</p><p class="pd-best"><b>Best for:</b> waltz and triple-feel songs, or anything you want to physically sway to.</p>`},
  fibonacci_magic:{tag:'Beat-synced · φ', detail:`<p>Sparkling and surprising — quick golden-angle node-hopping over a wide arc, with a fast φ/φ² shimmer layered on top so the highs twinkle and dart around your head. The brightest, airiest profile.</p><p class="pd-best"><b>Best for:</b> synth-pop, bright electronic, and dreamy "magical" material.</p>`},
  opus_fibonacci:{tag:'Beat-synced · φ', detail:`<p>The grand flagship of the golden family: the widest, slowest, most theatrical orbit, with a φ-spaced front/back figure-eight overlay and the deepest felt presence and biggest room of the set — a cinematic sweep you feel in your chest.</p><p class="pd-best"><b>Best for:</b> drops, choruses, trailers and big emotional moments.</p>`},
  fireflies_plus:{tag:'Beat-synced', detail:`<p>A reference-inspired smooth premium orbit with subtle organic drift and a light shimmer — the softer, sparklier cousin of Reference Luxe.</p><p class="pd-best"><b>Best for:</b> a pretty, easygoing general-purpose spin when you're not sure which to pick.</p>`},
  cinematic_halo:{tag:'Beat-synced', detail:`<p>A slow, emotional circle with mild non-repeating drift — wide and atmospheric, and deliberately un-dizzy.</p><p class="pd-best"><b>Best for:</b> film and score, ballads, ambient, and anything that wants space and emotion over obvious movement.</p>`},
  figure8:{tag:'Beat-synced', detail:`<p>The classic figure-eight: it alternates left/right side travel with stronger front-and-back sweeps for more three-dimensional immersion than a plain circle.</p><p class="pd-best"><b>Best for:</b> headphone showcases and immersive, dynamic listening.</p>`},
  wide_orbit:{tag:'Beat-synced', detail:`<p>The biggest, most theatrical circular orbit, with the most chorus-width energy and the least center lock.</p><p class="pd-best"><b>Best for:</b> choruses, drops and big-room material where maximum width is the goal.</p>`},
  vocal_safe:{tag:'Beat-synced', detail:`<p>The most restrained motion of all — it hovers around the front and side quadrants instead of swinging hard to the rear, keeping the strongest center clarity so lyrics stay locked front-and-center.</p><p class="pd-best"><b>Best for:</b> vocal-led songs, podcasts, acoustic, and anywhere intelligibility matters most.</p>`},
};
const profileCards = Array.from(document.querySelectorAll('.profile'));
const profileDetail = document.getElementById('profileDetail');
function showProfile(val){
  const card = profileCards.find(c => c.dataset.preset === val) || profileCards[0];
  profileCards.forEach(c => c.setAttribute('aria-pressed', String(c === card)));
  const idx = card.querySelector('.idx').textContent;
  const titleTxt = card.querySelector('b').textContent;
  const info = PROFILES[card.dataset.preset] || {tag:'', detail:''};
  profileDetail.innerHTML = '<h3><span class="pd-idx">' + idx + '</span>' + titleTxt + '<span class="pd-tag">' + info.tag + '</span></h3>' + info.detail;
}
profileCards.forEach(card => {
  const choose = () => { preset.value = card.dataset.preset; showProfile(card.dataset.preset); };
  card.addEventListener('click', choose);
  card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
});
preset.addEventListener('change', () => showProfile(preset.value));
showProfile(preset.value);

/* Onboard-systems badges: tap one to read what that feature does. */
const FEATURES = {
  bpm_aware:{title:'BPM aware', body:`The engine beat-tracks your song and ties the orbit speed to its tempo, so the motion lands with the groove instead of drifting against it. Fixed-speed profiles (like 8D Binaural Mix and the reference orbits) intentionally override this with a set rate.`},
  static_cleanup:{title:'Static cleanup', body:`A gentle, loudness-matched denoise lifts hiss and background static off the track before spatializing — it removes noise without dulling the music or changing its overall level.`},
  golden_ratio:{title:'Golden ratio motion', body:`Several profiles place their movement at golden-angle (φ ≈ 137.5°) points around you and time the passes with φ-derived ratios, so the orbit feels organic and never settles into an obvious machine loop.`},
  fibonacci_timing:{title:'Fibonacci timing', body:`Orbit segments last 1, 1, 2, 3, 5, 8, 13 parts of a cycle (or Lucas 2, 1, 3, 4, 7, 11), giving quick darting passes interleaved with long, elegant sweeps that don't repeat predictably.`},
  felt_presence:{title:'Felt-presence panning', body:`Low-mid body is selectively re-anchored and lightly reinforced so the spatial movement is something you feel in the chest, not just hear — while the sub-bass and kick stay locked in the center.`},
  ai_stems:{title:'AI stem separation', body:`Optionally splits the track into vocals, drums, bass and other (via Demucs), spatializes each with role-aware settings — bass stays mono-centered, vocals anchored front, instruments and air get more motion — then recombines them, and lets you download the separated stems as a zip. Requires Demucs on the server; otherwise the app uses the classic full-mix render.`},
};
const featDetail = document.getElementById('featDetail');
const badges = Array.from(document.querySelectorAll('.badge[data-feat]'));
function showFeature(key){
  const info = FEATURES[key]; if(!info) return;
  badges.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.feat === key)));
  featDetail.className = 'feat-detail show';
  featDetail.innerHTML = '<b>' + info.title + '</b><p>' + info.body + '</p>';
}
badges.forEach(b => {
  const open = () => showFeature(b.dataset.feat);
  b.addEventListener('click', open);
  b.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

def _set_job(job_id: str, **updates):
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()

def _separate_stems_replicate(src: Path, work_dir: Path):
    """Separate stems with Demucs hosted on Replicate (no local torch/RAM).

    Sends the track to Replicate, downloads the returned stems into ``work_dir``
    (so the existing zip step packages them), and returns ``{name: StemData}``.
    Raises :class:`StemSeparationUnavailable` if the token is missing or the call
    fails, so the caller falls back to the classic render cleanly.

    Privacy note: this uploads the user's audio to Replicate, a third party.
    """
    import urllib.request

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise StemSeparationUnavailable("REPLICATE_API_TOKEN is not set")
    try:
        import replicate
    except Exception as exc:  # pragma: no cover - env-specific
        raise StemSeparationUnavailable(f"replicate client not installed: {exc}")

    model_ref = os.environ.get("REPLICATE_DEMUCS_MODEL", "ryan5453/demucs")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(src, "rb") as fh:
            output = replicate.run(model_ref, input={"audio": fh, "output_format": "wav"})
    except Exception as exc:
        raise StemSeparationUnavailable(f"Replicate separation failed: {exc}")

    if isinstance(output, dict):
        items = output
    elif isinstance(output, (list, tuple)):
        names = ["vocals", "drums", "bass", "other"]
        items = {(names[i] if i < len(names) else f"stem{i}"): v for i, v in enumerate(output)}
    else:
        raise StemSeparationUnavailable(f"Unexpected Replicate output type: {type(output).__name__}")

    stems: dict = {}
    for name, val in items.items():
        if val is None:
            continue
        dest = work_dir / f"{name}.wav"
        try:
            data = val.read()  # replicate>=0.25 FileOutput
        except AttributeError:
            with urllib.request.urlopen(str(val), timeout=300) as resp:
                data = resp.read()
        dest.write_bytes(data)
        loaded = load_audio(dest)
        stems[name] = StemData(loaded.samples, loaded.sample_rate)
    if not stems:
        raise StemSeparationUnavailable("Replicate returned no stems")
    return stems


def _separate_stems_hf_space(src: Path, work_dir: Path):
    """Separate stems with a free Hugging Face Space running Demucs.

    Calls the Space's ``/separate`` endpoint via gradio_client (no local
    torch/RAM), copies the returned stem files into ``work_dir`` (so the zip step
    packages them), and returns ``{name: StemData}``. Raises
    :class:`StemSeparationUnavailable` on any problem so the caller falls back.

    Set ``HF_SPACE_ID`` (e.g. "username/8d-demucs-stems"); ``HF_TOKEN`` optional.
    Privacy note: this uploads the user's audio to the Space (Hugging Face).
    """
    import shutil

    space_id = os.environ.get("HF_SPACE_ID")
    if not space_id:
        raise StemSeparationUnavailable("HF_SPACE_ID is not set")
    try:
        from gradio_client import Client, handle_file
    except Exception as exc:  # pragma: no cover - env-specific
        raise StemSeparationUnavailable(f"gradio_client not installed: {exc}")

    hf_token = os.environ.get("HF_TOKEN")
    try:
        client = Client(space_id, hf_token=hf_token) if hf_token else Client(space_id)
        result = client.predict(handle_file(str(src)), api_name="/separate")
    except Exception as exc:
        raise StemSeparationUnavailable(f"HF Space separation failed: {exc}")

    paths = list(result) if isinstance(result, (list, tuple)) else [result]
    names = ["vocals", "drums", "bass", "other"]
    work_dir.mkdir(parents=True, exist_ok=True)
    stems: dict = {}
    for i, fp in enumerate(paths):
        if not fp:
            continue
        name = names[i] if i < len(names) else f"stem{i}"
        dest = work_dir / f"{name}.wav"
        try:
            shutil.copyfile(str(fp), dest)
        except Exception as exc:
            raise StemSeparationUnavailable(f"Could not read returned stem {name!r}: {exc}")
        loaded = load_audio(dest)
        stems[name] = StemData(loaded.samples, loaded.sample_rate)
    if not stems:
        raise StemSeparationUnavailable("HF Space returned no stems")
    return stems


def _convert_format(wav_path: Path, fmt: str) -> Path:
    """Convert the rendered WAV to mp3/flac with ffmpeg; return the delivered file.
    Falls back to the WAV if conversion isn't possible."""
    if fmt not in ("mp3", "flac"):
        return wav_path
    try:
        from eightd_engine.audio_io import _find_ffmpeg
        import subprocess
        ff = _find_ffmpeg()
        dest = wav_path.with_name(wav_path.stem + ("." + fmt))
        codec = ["-codec:a", "libmp3lame", "-b:a", "320k"] if fmt == "mp3" else ["-codec:a", "flac"]
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path), *codec, str(dest)], check=True)
        return dest
    except Exception as exc:
        print(f"[8D] format conversion to {fmt} failed, serving WAV: {exc}", flush=True)
        return wav_path


def _process_job(job_id: str, src: Path, out: Path, preset: str = "reference_luxe", mix_prompt: str = "", stem_mode: str = "classic", fmt: str = "wav"):
    import time as _time
    _t0 = _time.time()
    def _log(msg: str) -> None:
        print(f"[8D {job_id}] +{_time.time()-_t0:.1f}s {msg}", flush=True)
    try:
        _log(f"start preset={preset} stem_mode={stem_mode}")
        _set_job(job_id, status="processing", message="Analyzing BPM…")
        # Decode to a sample-accurate, seekable float WAV (lossless) so the render
        # can stream the input block-by-block — the full song is never loaded into
        # RAM, so track length is bounded only by disk, not memory.
        src_wav, src_wav_is_temp = to_seekable_wav(src, APP_DIR)
        _info = _sf.info(str(src_wav))
        sr_in = int(_info.samplerate)
        n_in = int(_info.frames)
        duration_seconds = n_in / float(sr_in or 1)
        _log(f"loaded {duration_seconds:.1f}s audio, sr={sr_in}")
        if duration_seconds > MAX_UPLOAD_SECONDS:
            raise ValueError(
                f"Track is {duration_seconds / 60.0:.1f} minutes long. "
                f"Please upload songs {MAX_UPLOAD_MINUTES} minutes or shorter."
            )
        # BPM from a bounded leading window (tempo is ~constant; avoids loading the
        # whole track just to beat-track it).
        _bpm_window, _ = _sf.read(str(src_wav), frames=min(n_in, 150 * sr_in), dtype="float32", always_2d=True)
        bpm = estimate_bpm(AudioData(samples=_bpm_window, sample_rate=sr_in))
        del _bpm_window
        audio = None  # the full in-memory decode is loaded lazily, only for AI stems
        safe_preset = preset if preset in panning_preset_names() else "reference_luxe"
        reference_speed_presets = {"binaural_8d", "reference_luxe", "phi_reference_orbit", "fibonacci_spiral", "golden_figure8", "lucas_breath"}
        clean_speed_presets = {"clean_reference"}
        if safe_preset in clean_speed_presets:
            rotation_cpm = 7.76
        elif safe_preset in reference_speed_presets:
            rotation_cpm = 5.78
        else:
            rotation_cpm = bpm_to_premium_rotation_cpm(bpm)
        preset_settings = {
            # Mix-engineer feedback profile: keep lead/body front-center, move air,
            # guitar brightness, ambience, and generated room instead of spinning
            # the whole vocal image.
            "binaural_8d": dict(room_size=0.22, motion_depth=0.86, high_emphasis=0.62, spatial_mix=0.70, center_focus=0.66, felt_presence=0.80),
            "clean_reference": dict(room_size=0.14, motion_depth=0.58, high_emphasis=0.42, spatial_mix=0.52, center_focus=0.84, felt_presence=0.42, denoise_amount=0.84),
            "reference_luxe": dict(room_size=0.20, motion_depth=0.74, high_emphasis=0.70, spatial_mix=0.64, center_focus=0.72, felt_presence=0.72),
            "phi_reference_orbit": dict(room_size=0.20, motion_depth=0.72, high_emphasis=0.70, spatial_mix=0.64, center_focus=0.74, felt_presence=0.74),
            "fibonacci_spiral": dict(room_size=0.22, motion_depth=0.78, high_emphasis=0.74, spatial_mix=0.68, center_focus=0.62, felt_presence=0.80),
            "golden_figure8": dict(room_size=0.18, motion_depth=0.72, high_emphasis=0.68, spatial_mix=0.62, center_focus=0.70, felt_presence=0.74),
            "lucas_breath": dict(room_size=0.24, motion_depth=0.64, high_emphasis=0.64, spatial_mix=0.60, center_focus=0.78, felt_presence=0.68),
            # Felt-first Fibonacci trio (source-method panning + reverb movement),
            # tuned so the listener physically feels the motion and low end.
            "fibonacci_waltz": dict(room_size=0.22, motion_depth=0.80, high_emphasis=0.66, spatial_mix=0.66, center_focus=0.70, felt_presence=0.84),
            "fibonacci_magic": dict(room_size=0.26, motion_depth=0.88, high_emphasis=0.82, spatial_mix=0.72, center_focus=0.58, felt_presence=0.88),
            "opus_fibonacci": dict(room_size=0.30, motion_depth=0.96, high_emphasis=0.74, spatial_mix=0.78, center_focus=0.62, felt_presence=0.96),
            "wide_orbit": dict(room_size=0.20, motion_depth=0.82, high_emphasis=0.72, spatial_mix=0.70, center_focus=0.50, felt_presence=0.84),
            "vocal_safe": dict(room_size=0.14, motion_depth=0.50, high_emphasis=0.52, spatial_mix=0.48, center_focus=0.88, felt_presence=0.56),
            "cinematic_halo": dict(room_size=0.24, motion_depth=0.68, high_emphasis=0.66, spatial_mix=0.62, center_focus=0.74, felt_presence=0.74),
            "figure8": dict(room_size=0.18, motion_depth=0.72, high_emphasis=0.66, spatial_mix=0.62, center_focus=0.66, felt_presence=0.76),
        }
        settings = preset_settings.get(
            safe_preset,
            dict(room_size=0.20, motion_depth=0.74, high_emphasis=0.70, spatial_mix=0.64, center_focus=0.72, felt_presence=0.72),
        )
        settings = dict(settings)
        settings["denoise_amount"] = max(settings.get("denoise_amount", 0.0), 0.72)
        instruction_result = apply_mix_instructions(settings, mix_prompt)
        settings = instruction_result.settings
        mix_notes = " | ".join(instruction_result.notes) if instruction_result.notes else "Selected profile only"
        _log(f"BPM={bpm:.1f} cpm={rotation_cpm:.2f} denoise={settings.get('denoise_amount', 0):.2f} center_focus={settings.get('center_focus', 0):.2f}")
        _set_job(
            job_id,
            message="Cleaning static, then rendering premium spatial master…",
            bpm=round(bpm, 1),
            rotation_cpm=round(rotation_cpm, 2),
            preset=safe_preset,
            mix_prompt=mix_prompt,
            mix_notes=mix_notes,
        )
        requested_stem_mode = stem_mode if stem_mode in {"classic", "ai_stems"} else "classic"
        stem_engine = "full_mix"
        stem_count = 0
        stems_url = None
        if requested_stem_mode == "ai_stems":
            audio = load_audio(src)  # AI-stem path needs the full mix as a reference
            try:
                stem_dir = APP_DIR / f"{job_id}_stems"
                # Pick a separation engine, cheapest/preferred first; any failure
                # falls back to the classic full-mix render so a job never breaks.
                #   1. Free Hugging Face Space (HF_SPACE_ID)
                #   2. Replicate (REPLICATE_API_TOKEN, paid)
                #   3. Local Demucs install
                if os.environ.get("HF_SPACE_ID"):
                    _set_job(job_id, message="Separating stems on the free Demucs cloud (first run may take a minute)…", stem_mode=requested_stem_mode)
                    stems = _separate_stems_hf_space(src, stem_dir)
                    stem_engine = "hf_space_demucs"
                elif os.environ.get("REPLICATE_API_TOKEN"):
                    _set_job(job_id, message="Separating stems with hosted AI (Replicate)…", stem_mode=requested_stem_mode)
                    stems = _separate_stems_replicate(src, stem_dir)
                    stem_engine = "replicate_demucs"
                else:
                    mode_info = available_stem_mode()
                    if mode_info.get("mode") != "demucs":
                        raise StemSeparationUnavailable(mode_info.get("message", "AI stem separation is unavailable"))
                    _set_job(job_id, message="Separating vocals, drums, bass, and instruments with AI stems…", stem_mode=requested_stem_mode)
                    stems = separate_stems_from_file(src, work_dir=stem_dir)
                    stem_engine = "demucs"
                stem_count = len(stems)
                # Zip the separated stem WAVs for download (vocals/drums/bass/other).
                import zipfile as _zip
                zip_path = APP_DIR / f"{out.stem}_stems.zip"
                with _zip.ZipFile(zip_path, "w", _zip.ZIP_STORED) as zf:
                    for wav in sorted(stem_dir.rglob("*.wav")):
                        zf.write(wav, arcname=wav.name)
                stems_url = f"/files/{zip_path.name}"
                _log(f"zipped {stem_count} stems -> {zip_path.name}")
                _set_job(job_id, message=f"Rendering {stem_count} separated stems with role-aware spatial processing…")
                rendered = process_stem_spatial_mix(
                    stems,
                    reference=audio,
                    rotation_cpm=rotation_cpm,
                    panning_preset=safe_preset,
                )
            except StemSeparationUnavailable as exc:
                stem_engine = "hybrid_fallback"
                fallback_note = f"AI stems unavailable; used classic protected full-mix render ({exc})."
                mix_notes = f"{mix_notes} | {fallback_note}" if mix_notes else fallback_note
                _set_job(job_id, message="AI stems unavailable; falling back to classic protected full-mix render…")
                rendered = process_8d(
                    audio,
                    rotation_cpm=rotation_cpm,
                    room_size=settings["room_size"],
                    crossover_hz=150.0,
                    motion_depth=settings["motion_depth"],
                    high_emphasis=settings["high_emphasis"],
                    spatial_mix=settings["spatial_mix"],
                    denoise_amount=settings["denoise_amount"],
                    panning_preset=safe_preset,
                    preserve_quality=True,
                    youtube_master=False,
                    section_automation=True,
                    center_focus=settings["center_focus"],
                    felt_presence=settings["felt_presence"],
                )
            report = analyze_correlation(rendered.samples)
            export_audio(rendered, out)
        else:
            # Classic full-mix path streams the render straight to the WAV file
            # block-by-block, so peak RAM does not grow with the output length
            # (long tracks no longer need the whole rendered song held in memory).
            _log("calling render_8d_file_to_wav (classic, input+output streamed)")
            report = render_8d_file_to_wav(
                src_wav,
                out,
                rotation_cpm=rotation_cpm,
                room_size=settings["room_size"],
                crossover_hz=150.0,
                motion_depth=settings["motion_depth"],
                high_emphasis=settings["high_emphasis"],
                spatial_mix=settings["spatial_mix"],
                denoise_amount=settings["denoise_amount"],
                panning_preset=safe_preset,
                preserve_quality=True,
                youtube_master=False,
                section_automation=True,
                center_focus=settings["center_focus"],
                felt_presence=settings["felt_presence"],
            )
            _log("render_8d_to_wav complete")
        _set_job(job_id, message="Measuring loudness…")
        try:
            lufs, true_peak = measure_loudness_file(out)
        except Exception as _le:
            lufs, true_peak = None, None
            _log(f"loudness measure skipped: {_le}")
        _set_job(job_id, message=f"Writing {fmt.upper()} export…")
        deliver = _convert_format(out, fmt)
        _set_job(
            job_id,
            status="complete",
            message="Done.",
            output_name=deliver.name,
            download_url=f"/files/{deliver.name}",
            bpm=round(bpm, 1),
            rotation_cpm=round(rotation_cpm, 2),
            lufs=lufs,
            true_peak=true_peak,
            correlation=round(report.correlation, 3),
            side_mid_ratio=round(report.side_mid_ratio, 3),
            phase="phase warning" if report.phase_warning else "phase safe",
            preset=safe_preset,
            mix_prompt=mix_prompt,
            mix_notes=mix_notes,
            settings={k: round(float(v), 3) for k, v in settings.items()},
            stem_mode=requested_stem_mode,
            stem_engine=stem_engine,
            stem_count=stem_count,
            stems_url=stems_url,
        )
        _log("job complete")
    except Exception as exc:
        import traceback
        _log(f"FAILED: {exc}")
        traceback.print_exc()
        _set_job(job_id, status="failed", message="Render failed.", error=str(exc))
    finally:
        try:
            _lv = locals()
            if _lv.get("src_wav_is_temp") and _lv.get("src_wav") is not None:
                Path(_lv["src_wav"]).unlink(missing_ok=True)
        except Exception:
            pass

def _cgroup_mem():
    """Return (limit_mb, usage_mb) from the container cgroup, or (None, None)."""
    def _read(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except Exception:
            return None
    limit = usage = None
    v2_max, v2_cur = _read("/sys/fs/cgroup/memory.max"), _read("/sys/fs/cgroup/memory.current")
    if v2_max is not None:
        if v2_max.isdigit():
            limit = int(v2_max)
        if v2_cur and v2_cur.isdigit():
            usage = int(v2_cur)
    else:
        v1_lim, v1_use = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"), _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if v1_lim and v1_lim.isdigit():
            limit = int(v1_lim)
        if v1_use and v1_use.isdigit():
            usage = int(v1_use)
    to_mb = lambda b: round(b / 1048576) if isinstance(b, int) and b < (1 << 62) else None
    return to_mb(limit), to_mb(usage)


@app.get("/health")
async def health():
    lim, use = _cgroup_mem()
    return {
        "status": "ok",
        "dsp_available": DSP_AVAILABLE,
        "platform": "vercel" if _ON_VERCEL else "server",
        "mem_limit_mb": lim,
        "mem_usage_mb": use,
    }


@app.post("/convert")
async def convert(file: UploadFile = File(...), preset: str = Form("reference_luxe"), mix_prompt: str = Form(""), stem_mode: str = Form("classic"), fmt: str = Form("wav")):
    fmt = fmt.lower() if fmt.lower() in ("wav", "mp3", "flac") else "wav"
    if not DSP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Audio rendering is not available in this cloud environment — "
                "the DSP stack (numpy/scipy/librosa/demucs) requires a long-lived server. "
                "Run the app locally for full rendering: python -m uvicorn web_app:app --port 8765"
            ),
        )
    suffix = Path(file.filename or "audio").suffix.lower() or ".audio"
    safe_stem = Path(file.filename or "audio").stem.replace("/", "_").replace("\\", "_")[:80]
    job_id = uuid.uuid4().hex[:12]
    src = APP_DIR / f"{safe_stem}_{job_id}{suffix}"
    out = APP_DIR / f"{safe_stem}_{job_id}_8D_Final.wav"
    total = 0
    with src.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                src.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File is too large. Please upload tracks {MAX_UPLOAD_MB} MB or smaller.",
                )
            f.write(chunk)
    _set_job(job_id, status="queued", message="Upload complete. Waiting for DSP worker…", input_name=file.filename, output_name=out.name, mix_prompt=mix_prompt, stem_mode=stem_mode)
    EXECUTOR.submit(_process_job, job_id, src, out, preset, mix_prompt, stem_mode, fmt)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing", "message": "Upload accepted. DSP render started."})

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown render job")
        return dict(job, job_id=job_id)


# ── Multitrack mixer ───────────────────────────────────────────────────────────
# Separates a track into stems and serves each one individually so the browser
# can load them into a per-stem Web Audio mixer (fader/pan/EQ/mute/solo).

def _separate_stems_any(src: Path, stem_dir: Path, set_msg=None):
    """Separate into stems with the first available engine
    (HF Space → Replicate → local Demucs). Returns ``{name: StemData}`` or
    raises :class:`StemSeparationUnavailable` when none are configured/working.
    """
    def msg(m):
        if set_msg:
            set_msg(m)
    if os.environ.get("HF_SPACE_ID"):
        msg("Separating stems on the free Demucs cloud (first run may take a minute)…")
        return _separate_stems_hf_space(src, stem_dir)
    if os.environ.get("REPLICATE_API_TOKEN"):
        msg("Separating stems with hosted AI (Replicate)…")
        return _separate_stems_replicate(src, stem_dir)
    mode_info = available_stem_mode()
    if mode_info.get("mode") != "demucs":
        raise StemSeparationUnavailable(mode_info.get("message", "AI stem separation is unavailable on this server"))
    msg("Separating vocals, drums, bass, and instruments with AI…")
    return separate_stems_from_file(src, work_dir=stem_dir)


def _process_mixer_job(job_id: str, src: Path):
    import traceback
    try:
        _set_job(job_id, status="processing", message="Separating stems for the mixer…")
        stem_dir = APP_DIR / f"{job_id}_mixstems"
        stem_dir.mkdir(parents=True, exist_ok=True)
        stems = _separate_stems_any(src, stem_dir, lambda m: _set_job(job_id, message=m))
        # Preferred display order; anything else trails alphabetically.
        order = {"vocals": 0, "drums": 1, "bass": 2, "other": 3, "guitar": 4, "piano": 5}
        out_stems = []
        for name in sorted(stems, key=lambda n: (order.get(n.lower(), 99), n)):
            sd = stems[name]
            safe = "".join(c for c in name if c.isalnum() or c in "-_").lower() or "stem"
            fn = f"{job_id}_stem_{safe}.wav"
            _sf.write(str(APP_DIR / fn), sd.samples, int(sd.sample_rate))
            out_stems.append({"name": name, "url": f"/files/{fn}"})
        _set_job(job_id, status="complete", message=f"Separated {len(out_stems)} stems.", stems=out_stems, stem_count=len(out_stems))
    except StemSeparationUnavailable as exc:
        _set_job(job_id, status="failed", message="Stem separation isn't available on this server.", error=str(exc))
    except Exception as exc:  # pragma: no cover - env-specific
        traceback.print_exc()
        _set_job(job_id, status="failed", message="Stem separation failed.", error=str(exc))
    finally:
        src.unlink(missing_ok=True)


@app.post("/mixer/separate")
async def mixer_separate(file: UploadFile = File(...)):
    if not DSP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stem separation needs the long-lived server; run the app locally or on Railway.")
    suffix = Path(file.filename or "audio").suffix.lower() or ".audio"
    safe_stem = Path(file.filename or "audio").stem.replace("/", "_").replace("\\", "_")[:80]
    job_id = uuid.uuid4().hex[:12]
    src = APP_DIR / f"{safe_stem}_{job_id}_mixin{suffix}"
    total = 0
    with src.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                f.close()
                src.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File is too large. Please upload tracks {MAX_UPLOAD_MB} MB or smaller.")
            f.write(chunk)
    _set_job(job_id, status="queued", message="Upload complete. Waiting for the stem worker…", input_name=file.filename)
    EXECUTOR.submit(_process_mixer_job, job_id, src)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing"})


@app.get("/mixer", response_class=HTMLResponse)
def mixer_page():
    return MIXER_HTML


MIXER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Multitrack Mixer · The 8D Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root{ --bg:#06101c; --panel:rgba(13,20,38,.55); --hair:rgba(255,255,255,.12); --hair2:rgba(255,255,255,.22);
    --cyan:#62e0ff; --violet:#9d8bff; --soft:#8a93a8; --text:#e7edf6;
    --mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --display:'Space Grotesk', system-ui, sans-serif; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:radial-gradient(1200px 700px at 70% -10%, #11203c 0%, #06101c 55%, #050a14 100%);
    color:var(--text); font-family:Inter, system-ui, -apple-system, Segoe UI, sans-serif; min-height:100vh; }
  a{ color:inherit; }
  .shell{ max-width:1280px; margin:0 auto; padding:22px 22px 80px; }
  nav{ display:flex; align-items:center; justify-content:space-between; margin-bottom:26px; }
  .brand{ display:flex; align-items:center; gap:10px; text-decoration:none; }
  .brand .mark{ color:var(--cyan); font-size:20px; }
  .brand .word{ font-family:var(--display); font-weight:700; letter-spacing:.16em; font-size:13px; }
  .navbtn{ font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    color:#9fb0c8; border:1px solid var(--hair2); border-radius:999px; padding:8px 15px; text-decoration:none; }
  h1{ font-family:var(--display); font-weight:700; font-size:30px; margin:0 0 6px; letter-spacing:-.01em; }
  .lede{ color:var(--soft); max-width:680px; margin:0 0 24px; line-height:1.5; font-size:14px; }
  .grad{ background:linear-gradient(135deg,var(--cyan),var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .card{ border:1px solid var(--hair); border-radius:18px; background:var(--panel); padding:22px; }
  .drop{ border:1.5px dashed var(--hair2); border-radius:16px; padding:34px; text-align:center; cursor:pointer; transition:border-color .2s, background .2s; }
  .drop:hover, .drop.drag{ border-color:var(--cyan); background:rgba(98,224,255,.05); }
  .drop h3{ font-family:var(--display); margin:8px 0 4px; font-size:16px; }
  .drop p{ color:var(--soft); font-size:12.5px; margin:0; }
  .btn{ cursor:pointer; border:none; border-radius:999px; padding:10px 18px; font-family:var(--mono);
    font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:#06101c;
    background:linear-gradient(135deg,var(--cyan),var(--violet)); }
  .btn.ghost{ background:transparent; color:#9fb0c8; border:1px solid var(--hair2); }
  .btn:disabled{ opacity:.45; cursor:not-allowed; }
  #status{ margin-top:14px; color:var(--soft); font-size:12.5px; font-family:var(--mono); min-height:18px; }
  #status.err{ color:#ff8a8a; }
  /* transport */
  .transport{ display:none; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  .play{ width:46px; height:46px; border-radius:50%; border:none; cursor:pointer; font-size:17px; color:#06101c;
    background:linear-gradient(135deg,var(--cyan),var(--violet)); }
  .tcode{ font-family:var(--mono); font-size:13px; color:#9fb0c8; min-width:96px; }
  .master{ display:flex; align-items:center; gap:8px; margin-left:auto; }
  .master label{ font-family:var(--mono); font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--soft); }
  /* mixer */
  #mixer{ display:none; gap:14px; overflow-x:auto; padding-bottom:8px; }
  .strip{ flex:0 0 150px; border:1px solid var(--hair); border-radius:16px; background:linear-gradient(180deg,rgba(13,20,38,.6),rgba(7,11,22,.4));
    padding:14px 12px; display:flex; flex-direction:column; align-items:center; gap:10px; }
  .strip .name{ font-family:var(--display); font-weight:600; font-size:13px; text-transform:capitalize; }
  .meter{ width:100%; height:8px; border-radius:6px; background:rgba(255,255,255,.07); overflow:hidden; }
  .meter > i{ display:block; height:100%; width:0%; background:linear-gradient(90deg,#36d399,#e3d24a 70%,#ff6b6b); transition:width .05s linear; }
  .eq{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; width:100%; }
  .eq .knob{ display:flex; flex-direction:column; align-items:center; gap:3px; }
  .eq .knob span{ font-family:var(--mono); font-size:8.5px; letter-spacing:.1em; color:var(--soft); }
  .eq input{ width:100%; }
  .panrow{ width:100%; display:flex; flex-direction:column; align-items:center; gap:3px; }
  .panrow span{ font-family:var(--mono); font-size:8.5px; letter-spacing:.1em; color:var(--soft); }
  .fader{ -webkit-appearance:slider-vertical; writing-mode:vertical-lr; direction:rtl; width:30px; height:130px; }
  .gaindb{ font-family:var(--mono); font-size:10px; color:#9fb0c8; }
  .ms{ display:flex; gap:6px; }
  .ms button{ cursor:pointer; width:30px; height:26px; border-radius:7px; border:1px solid var(--hair2);
    background:transparent; color:var(--soft); font-family:var(--mono); font-size:11px; font-weight:600; }
  .ms button.on.m{ background:#ff6b6b; color:#160606; border-color:transparent; }
  .ms button.on.s{ background:var(--cyan); color:#06101c; border-color:transparent; }
  input[type=range]{ accent-color:var(--cyan); }
  .actions{ display:none; gap:12px; margin-top:20px; flex-wrap:wrap; }
  .hint{ color:var(--soft); font-size:12px; margin-top:10px; }
  .result{ margin-top:14px; font-size:13px; }
  .result a{ color:var(--cyan); }
</style>
</head>
<body>
  <div class="shell">
    <nav>
      <a class="brand" href="/"><span class="mark">&#9676;</span><span class="word">THE&nbsp;8D&nbsp;ENGINE</span></a>
      <a class="navbtn" href="/">&larr; Back to mastering</a>
    </nav>

    <h1>Multitrack <span class="grad">mixer</span></h1>
    <p class="lede">Drop a finished track and the engine splits it into vocals, drums, bass and instruments. Balance each
      stem with its own fader, pan, 3-band EQ and mute/solo &mdash; then bounce a fresh mixdown or send it straight into the 8D orbit.</p>

    <div class="card" id="uploadCard">
      <div class="drop" id="drop">
        <div style="font-size:26px">&#127899;</div>
        <h3>Drop a track to un-mix</h3>
        <p>WAV, MP3, FLAC, M4A &mdash; up to 200 MB. We separate the stems on the server, then mix in your browser.</p>
        <input type="file" id="file" accept="audio/*" style="display:none"/>
      </div>
      <div id="status"></div>
    </div>

    <div class="transport" id="transport">
      <button class="play" id="play" title="Play / pause">&#9658;</button>
      <button class="btn ghost" id="stop">Stop</button>
      <span class="tcode" id="tcode">0:00 / 0:00</span>
      <div class="master">
        <label>Master</label>
        <input type="range" id="masterFader" min="0" max="1.4" step="0.01" value="1"/>
      </div>
    </div>

    <div id="mixer"></div>

    <div class="actions" id="actions">
      <button class="btn" id="mixdown">&#11015; Download mixdown (WAV)</button>
      <button class="btn ghost" id="to8d">Send mix to the 8D engine &rarr;</button>
    </div>
    <div class="result" id="result"></div>
  </div>

<script>
// Talk to the long-lived Railway backend directly in production (Vercel can't
// run the DSP/separation worker). Same single-hop pattern as the home page.
const IS_DEV = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const API = IS_DEV ? '' : 'https://luminous-endurance-production-0696.up.railway.app';
const $ = id => document.getElementById(id);
const drop = $('drop'), fileIn = $('file'), statusEl = $('status');
let ctx = null, channels = [], duration = 0;
let playing = false, startedAt = 0, offset = 0, sources = [], rafId = 0, meterRaf = 0;

function setStatus(msg, err){ statusEl.textContent = msg || ''; statusEl.classList.toggle('err', !!err); }
const fmt = s => { s = Math.max(0, s|0); return (s/60|0)+':'+String(s%60).padStart(2,'0'); };

drop.onclick = () => fileIn.click();
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('drag'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', ev => { if (ev.dataTransfer.files[0]) separate(ev.dataTransfer.files[0]); });
fileIn.onchange = () => { if (fileIn.files[0]) separate(fileIn.files[0]); };

async function separate(file){
  setStatus('Uploading ' + file.name + ' ...');
  const fd = new FormData(); fd.append('file', file);
  let res;
  try { res = await fetch(API + '/mixer/separate', { method:'POST', body:fd }); }
  catch(e){ return setStatus('Upload failed: ' + e.message, true); }
  if (!res.ok){ const t = await res.text(); return setStatus('Server error: ' + t, true); }
  const { job_id } = await res.json();
  setStatus('Separating stems... this can take a minute on the free cloud worker.');
  pollSeparation(job_id);
}

async function pollSeparation(jobId){
  for (let i=0;i<600;i++){
    let job;
    try { job = await (await fetch(API + '/jobs/' + jobId)).json(); } catch(e){ await wait(2000); continue; }
    if (job.message) setStatus(job.message);
    if (job.status === 'complete'){ await loadStems(job.stems || []); return; }
    if (job.status === 'failed'){ return setStatus(job.message + ' ' + (job.error||''), true); }
    await wait(2000);
  }
  setStatus('Timed out waiting for stem separation.', true);
}
const wait = ms => new Promise(r => setTimeout(r, ms));

async function loadStems(stems){
  if (!stems.length) return setStatus('No stems were returned.', true);
  setStatus('Loading ' + stems.length + ' stems into the mixer...');
  ctx = ctx || new (window.AudioContext||window.webkitAudioContext)();
  const decoded = [];
  for (const st of stems){
    try {
      const ab = await (await fetch(st.url.startsWith('http') ? st.url : API + st.url)).arrayBuffer();
      decoded.push({ name: st.name, buffer: await ctx.decodeAudioData(ab) });
    } catch(e){ /* skip a bad stem rather than abort the whole mix */ }
  }
  if (!decoded.length) return setStatus('Could not decode the separated stems.', true);
  buildMixer(decoded);
  setStatus('');
  $('uploadCard').style.display = 'none';
}

// buildMixer is global so it can be driven with synthetic buffers in tests.
function buildMixer(decoded){
  ctx = ctx || new (window.AudioContext||window.webkitAudioContext)();
  const mixerEl = $('mixer'); mixerEl.innerHTML = ''; channels = [];
  const masterGain = ctx.createGain();
  masterGain.gain.value = parseFloat($('masterFader').value);
  masterGain.connect(ctx.destination);
  $('masterFader').oninput = e => { masterGain.gain.value = parseFloat(e.target.value); };
  duration = Math.max.apply(null, decoded.map(d => d.buffer.duration));

  decoded.forEach(d => {
    const ch = { name:d.name, buffer:d.buffer, vol:1, pan:0, mute:false, solo:false, eq:{lo:0,mid:0,hi:0} };
    // persistent node chain: src -> lo -> mid -> hi -> pan -> gain -> analyser -> master
    ch.lo = ctx.createBiquadFilter(); ch.lo.type='lowshelf'; ch.lo.frequency.value=200;
    ch.mid = ctx.createBiquadFilter(); ch.mid.type='peaking'; ch.mid.frequency.value=1000; ch.mid.Q.value=1;
    ch.hi = ctx.createBiquadFilter(); ch.hi.type='highshelf'; ch.hi.frequency.value=4000;
    ch.panner = ctx.createStereoPanner();
    ch.gain = ctx.createGain();
    ch.analyser = ctx.createAnalyser(); ch.analyser.fftSize = 256;
    ch.lo.connect(ch.mid).connect(ch.hi).connect(ch.panner).connect(ch.gain).connect(ch.analyser).connect(masterGain);

    const el = document.createElement('div'); el.className = 'strip';
    el.innerHTML =
      '<div class="name">'+ch.name+'</div>'+
      '<div class="meter"><i></i></div>'+
      '<div class="eq">'+
        '<div class="knob"><span>LO</span><input type="range" min="-12" max="12" step="0.5" value="0" data-eq="lo"></div>'+
        '<div class="knob"><span>MID</span><input type="range" min="-12" max="12" step="0.5" value="0" data-eq="mid"></div>'+
        '<div class="knob"><span>HI</span><input type="range" min="-12" max="12" step="0.5" value="0" data-eq="hi"></div>'+
      '</div>'+
      '<div class="panrow"><span>PAN</span><input type="range" class="pan" min="-1" max="1" step="0.02" value="0"></div>'+
      '<input type="range" class="fader" min="0" max="1.4" step="0.01" value="1">'+
      '<div class="gaindb">0.0 dB</div>'+
      '<div class="ms"><button class="m" title="Mute">M</button><button class="s" title="Solo">S</button></div>';
    mixerEl.appendChild(el);

    ch.meterEl = el.querySelector('.meter > i');
    ch.gaindbEl = el.querySelector('.gaindb');
    el.querySelectorAll('[data-eq]').forEach(inp => inp.oninput = e => {
      const band = e.target.dataset.eq; ch.eq[band] = parseFloat(e.target.value);
      ch[band].gain.value = ch.eq[band];
    });
    el.querySelector('.pan').oninput = e => { ch.pan = parseFloat(e.target.value); ch.panner.pan.value = ch.pan; };
    el.querySelector('.fader').oninput = e => {
      ch.vol = parseFloat(e.target.value);
      ch.gaindbEl.textContent = (ch.vol<=0 ? '-inf' : (20*Math.log10(ch.vol)).toFixed(1)) + ' dB';
      applyGains();
    };
    const mBtn = el.querySelector('.m'), sBtn = el.querySelector('.s');
    mBtn.onclick = () => { ch.mute = !ch.mute; mBtn.classList.toggle('on', ch.mute); applyGains(); };
    sBtn.onclick = () => { ch.solo = !ch.solo; sBtn.classList.toggle('on', ch.solo); applyGains(); };
    channels.push(ch);
  });

  window.__master = masterGain;
  applyGains();
  mixerEl.style.display = 'flex';
  $('transport').style.display = 'flex';
  $('actions').style.display = 'flex';
  $('tcode').textContent = '0:00 / ' + fmt(duration);
}

function applyGains(){
  const anySolo = channels.some(c => c.solo);
  channels.forEach(c => { c.gain.gain.value = (c.mute || (anySolo && !c.solo)) ? 0 : c.vol; });
}

function startSources(){
  sources = channels.map(c => { const s = ctx.createBufferSource(); s.buffer = c.buffer; s.connect(c.lo); return s; });
  const at = Math.max(0, Math.min(offset, duration - 0.02));
  startedAt = ctx.currentTime;
  sources.forEach(s => s.start(0, at));
}
function stopSources(){ sources.forEach(s => { try{ s.stop(); }catch(e){} }); sources = []; }
function pos(){ return playing ? Math.min(duration, offset + (ctx.currentTime - startedAt)) : offset; }

function frame(){
  if (!playing) return;
  const p = pos();
  $('tcode').textContent = fmt(p) + ' / ' + fmt(duration);
  if (p >= duration - 0.03){ stopAll(); return; }
  rafId = requestAnimationFrame(frame);
}
function meterLoop(){
  channels.forEach(c => {
    const buf = new Uint8Array(c.analyser.frequencyBinCount);
    c.analyser.getByteTimeDomainData(buf);
    let sum = 0; for (let i=0;i<buf.length;i++){ const v=(buf[i]-128)/128; sum += v*v; }
    const rms = Math.sqrt(sum/buf.length);
    if (c.meterEl) c.meterEl.style.width = Math.min(100, rms*180).toFixed(0) + '%';
  });
  meterRaf = requestAnimationFrame(meterLoop);
}

$('play').onclick = () => {
  if (!channels.length) return;
  ctx.resume();
  if (playing){ offset = pos(); stopSources(); playing = false; $('play').innerHTML='&#9658;'; cancelAnimationFrame(rafId); }
  else { playing = true; $('play').innerHTML='&#10073;&#10073;'; startSources(); rafId = requestAnimationFrame(frame); meterLoop(); }
};
$('stop').onclick = () => stopAll();
function stopAll(){
  stopSources(); playing = false; offset = 0;
  $('play').innerHTML = '&#9658;'; cancelAnimationFrame(rafId); cancelAnimationFrame(meterRaf);
  channels.forEach(c => { if (c.meterEl) c.meterEl.style.width = '0%'; });
  $('tcode').textContent = '0:00 / ' + fmt(duration);
}

// ── Offline mixdown ────────────────────────────────────────────────────────────
async function renderMix(){
  const sr = channels[0].buffer.sampleRate;
  const off = new OfflineAudioContext(2, Math.ceil(duration*sr), sr);
  const master = off.createGain(); master.gain.value = parseFloat($('masterFader').value); master.connect(off.destination);
  const anySolo = channels.some(c => c.solo);
  channels.forEach(c => {
    const s = off.createBufferSource(); s.buffer = c.buffer;
    const lo = off.createBiquadFilter(); lo.type='lowshelf'; lo.frequency.value=200; lo.gain.value=c.eq.lo;
    const mid = off.createBiquadFilter(); mid.type='peaking'; mid.frequency.value=1000; mid.Q.value=1; mid.gain.value=c.eq.mid;
    const hi = off.createBiquadFilter(); hi.type='highshelf'; hi.frequency.value=4000; hi.gain.value=c.eq.hi;
    const pan = off.createStereoPanner(); pan.pan.value=c.pan;
    const g = off.createGain(); g.gain.value = (c.mute || (anySolo && !c.solo)) ? 0 : c.vol;
    s.connect(lo).connect(mid).connect(hi).connect(pan).connect(g).connect(master);
    s.start(0);
  });
  return audioBufferToWav(await off.startRendering());
}
function audioBufferToWav(buf){
  const numCh = Math.min(2, buf.numberOfChannels), len = buf.length, sr = buf.sampleRate;
  const bps = 4, blockAlign = numCh*bps, dataLen = len*blockAlign;
  const ab = new ArrayBuffer(44+dataLen), dv = new DataView(ab); let o = 0;
  const ws = s => { for (let i=0;i<s.length;i++) dv.setUint8(o++, s.charCodeAt(i)); };
  ws('RIFF'); dv.setUint32(o,36+dataLen,true); o+=4; ws('WAVE'); ws('fmt ');
  dv.setUint32(o,16,true); o+=4; dv.setUint16(o,3,true); o+=2; dv.setUint16(o,numCh,true); o+=2;
  dv.setUint32(o,sr,true); o+=4; dv.setUint32(o,sr*blockAlign,true); o+=4;
  dv.setUint16(o,blockAlign,true); o+=2; dv.setUint16(o,32,true); o+=2; ws('data'); dv.setUint32(o,dataLen,true); o+=4;
  const chans = []; for (let c=0;c<numCh;c++) chans.push(buf.getChannelData(c));
  for (let i=0;i<len;i++){ for (let c=0;c<numCh;c++){ dv.setFloat32(o, chans[c][i], true); o+=4; } }
  return new Blob([ab], { type:'audio/wav' });
}

$('mixdown').onclick = async () => {
  if (!channels.length) return;
  $('mixdown').disabled = true; setStatus('Rendering mixdown...');
  try {
    const blob = await renderMix();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = '8d_mixdown.wav'; a.click();
    setStatus('Mixdown saved.');
  } catch(e){ setStatus('Mixdown failed: ' + e.message, true); }
  $('mixdown').disabled = false;
};

$('to8d').onclick = async () => {
  if (!channels.length) return;
  $('to8d').disabled = true; setStatus('Rendering mix, then sending it to the 8D engine...');
  try {
    const blob = await renderMix();
    const fd = new FormData();
    fd.append('file', new File([blob], 'mix.wav', { type:'audio/wav' }));
    fd.append('preset', 'reference_luxe'); fd.append('stem_mode', 'classic'); fd.append('fmt', 'wav');
    const res = await fetch(API + '/convert', { method:'POST', body:fd });
    if (!res.ok){ throw new Error(await res.text()); }
    const { job_id } = await res.json();
    setStatus('8D render started...');
    for (let i=0;i<400;i++){
      const job = await (await fetch(API + '/jobs/' + job_id)).json();
      if (job.message) setStatus(job.message);
      if (job.status === 'complete'){
        const u = job.download_url.startsWith('http') ? job.download_url : API + job.download_url;
        $('result').innerHTML = '<a href="'+u+'" download>&#11015; Download your 8D-spatialized mix</a>';
        setStatus('Done.'); break;
      }
      if (job.status === 'failed'){ setStatus('8D render failed: ' + (job.error||''), true); break; }
      await wait(2000);
    }
  } catch(e){ setStatus('Could not send to 8D: ' + e.message, true); }
  $('to8d').disabled = false;
};
</script>
</body>
</html>
"""
