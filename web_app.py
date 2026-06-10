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
        from eightd_engine.audio_io import export_audio, load_audio
        from eightd_engine.dsp import (
            analyze_correlation,
            bpm_to_premium_rotation_cpm,
            estimate_bpm,
            panning_preset_names,
            process_8d,
        )
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

    .foot{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-top:64px; padding-top:20px; border-top:1px solid var(--hair); }
    .foot .word{ font-family:var(--display); letter-spacing:.22em; font-size:13px; }

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
        <span class="nav-note">Spatial Audio Mastering</span>
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
          <span class="badge">BPM aware</span>
          <span class="badge">Static cleanup</span>
          <span class="badge">Golden ratio motion</span>
          <span class="badge">Fibonacci timing</span>
          <span class="badge">Felt-presence panning</span>
          <span class="badge">AI stem separation</span>
        </div>
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
                <option value="clean_reference" selected>Clean Reference — polished 7.7s orbit</option>
                <option value="reference_luxe">Reference Luxe — 10.4s orbit</option>
                <option value="phi_reference_orbit">Golden Ratio Reference — φ-timed orbit</option>
                <option value="fibonacci_spiral">Fibonacci Spiral — golden-angle path</option>
                <option value="golden_figure8">Golden Figure 8 — φ front/back sweep</option>
                <option value="lucas_breath">Lucas Breath — slow Fibonacci halo</option>
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

    <div class="section-head"><h2>Eleven mastering orbits</h2><span class="tlabel">Flight profiles</span></div>
    <section class="profiles" aria-label="Flight profiles">
      <div class="profile"><span class="idx">01</span><div><b>Clean Reference</b><small>Polished 7.7s orbit — tight and centered.</small></div></div>
      <div class="profile"><span class="idx">02</span><div><b>Reference Luxe</b><small>Expansive 10.4s premium orbit.</small></div></div>
      <div class="profile"><span class="idx">03</span><div><b>Golden Ratio Reference</b><small>φ-timed orbit for organic motion.</small></div></div>
      <div class="profile"><span class="idx">04</span><div><b>Fibonacci Spiral</b><small>Golden-angle spiral path.</small></div></div>
      <div class="profile"><span class="idx">05</span><div><b>Golden Figure 8</b><small>φ front/back sweep.</small></div></div>
      <div class="profile"><span class="idx">06</span><div><b>Lucas Breath</b><small>Slow Fibonacci halo motion.</small></div></div>
      <div class="profile"><span class="idx">07</span><div><b>Fireflies Plus</b><small>Smooth premium orbit shimmer.</small></div></div>
      <div class="profile"><span class="idx">08</span><div><b>Cinematic Halo</b><small>Elegant atmospheric surround.</small></div></div>
      <div class="profile"><span class="idx">09</span><div><b>Figure 8</b><small>Front/back immersive sweep.</small></div></div>
      <div class="profile"><span class="idx">10</span><div><b>Wide Orbit</b><small>Powerful chorus-width motion.</small></div></div>
      <div class="profile"><span class="idx">11</span><div><b>Vocal Safe</b><small>Clear center, gentle motion.</small></div></div>
    </section>

    <footer class="foot">
      <span class="word">THE 8D ENGINE</span>
      <span class="tlabel">64-bit DSP · Mono-safe bass · 32-bit WAV export</span>
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

async function upload(f) {
  if (!DSP_OK) return;
  const mb = (f.size / 1048576).toFixed(1);
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
      hint.innerHTML = `<a href="${dlUrl}" download>⬇ Download ${job.output_name}</a>`;
      statusEl.textContent = `Profile: ${job.preset}\\nMode: ${job.stem_mode || 'classic'} (${job.stem_engine || 'full mix'})\\nPrompt: ${job.mix_notes || 'Selected profile only'}\\nBPM: ${job.bpm}\\nOrbit: ${job.rotation_cpm} cycles/min\\nCorrelation: ${job.correlation} | Side/Mid: ${job.side_mid_ratio} | ${job.phase}`;
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || 'Render failed');
    statusEl.textContent = `${job.message || 'Rendering…'}\\nYou can leave this tab open until the download link appears.`;
    await new Promise(r => setTimeout(r, 1500));
  }
}
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

def _process_job(job_id: str, src: Path, out: Path, preset: str = "reference_luxe", mix_prompt: str = "", stem_mode: str = "classic"):
    import time as _time
    _t0 = _time.time()
    def _log(msg: str) -> None:
        print(f"[8D {job_id}] +{_time.time()-_t0:.1f}s {msg}", flush=True)
    try:
        _log(f"start preset={preset} stem_mode={stem_mode}")
        _set_job(job_id, status="processing", message="Analyzing BPM…")
        audio = load_audio(src)
        duration_seconds = len(audio.samples) / float(audio.sample_rate or 1)
        _log(f"loaded {duration_seconds:.1f}s audio, sr={audio.sample_rate}")
        if duration_seconds > MAX_UPLOAD_SECONDS:
            raise ValueError(
                f"Track is {duration_seconds / 60.0:.1f} minutes long. "
                f"Please upload songs {MAX_UPLOAD_MINUTES} minutes or shorter."
            )
        bpm = estimate_bpm(audio)
        safe_preset = preset if preset in panning_preset_names() else "reference_luxe"
        reference_speed_presets = {"reference_luxe", "phi_reference_orbit", "fibonacci_spiral", "golden_figure8", "lucas_breath"}
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
            "clean_reference": dict(room_size=0.14, motion_depth=0.58, high_emphasis=0.42, spatial_mix=0.52, center_focus=0.84, felt_presence=0.42, denoise_amount=0.84),
            "reference_luxe": dict(room_size=0.20, motion_depth=0.74, high_emphasis=0.70, spatial_mix=0.64, center_focus=0.72, felt_presence=0.72),
            "phi_reference_orbit": dict(room_size=0.20, motion_depth=0.72, high_emphasis=0.70, spatial_mix=0.64, center_focus=0.74, felt_presence=0.74),
            "fibonacci_spiral": dict(room_size=0.22, motion_depth=0.78, high_emphasis=0.74, spatial_mix=0.68, center_focus=0.62, felt_presence=0.80),
            "golden_figure8": dict(room_size=0.18, motion_depth=0.72, high_emphasis=0.68, spatial_mix=0.62, center_focus=0.70, felt_presence=0.74),
            "lucas_breath": dict(room_size=0.24, motion_depth=0.64, high_emphasis=0.64, spatial_mix=0.60, center_focus=0.78, felt_presence=0.68),
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
        if requested_stem_mode == "ai_stems":
            try:
                mode_info = available_stem_mode()
                if mode_info.get("mode") != "demucs":
                    raise StemSeparationUnavailable(mode_info.get("message", "AI stem separation is unavailable"))
                _set_job(job_id, message="Separating vocals, drums, bass, and instruments with AI stems…", stem_mode=requested_stem_mode)
                stems = separate_stems_from_file(src, work_dir=APP_DIR / f"{job_id}_stems")
                stem_count = len(stems)
                _set_job(job_id, message=f"Rendering {stem_count} separated stems with role-aware spatial processing…")
                rendered = process_stem_spatial_mix(
                    stems,
                    reference=audio,
                    rotation_cpm=rotation_cpm,
                    panning_preset=safe_preset,
                )
                stem_engine = "demucs"
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
        else:
            _log("calling process_8d (classic)")
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
            _log("process_8d complete")
        _set_job(job_id, message="Writing WAV export…")
        _log("exporting WAV")
        report = analyze_correlation(rendered.samples)
        export_audio(rendered, out)
        _set_job(
            job_id,
            status="complete",
            message="Done.",
            output_name=out.name,
            download_url=f"/files/{out.name}",
            bpm=round(bpm, 1),
            rotation_cpm=round(rotation_cpm, 2),
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
        )
        _log("job complete")
    except Exception as exc:
        import traceback
        _log(f"FAILED: {exc}")
        traceback.print_exc()
        _set_job(job_id, status="failed", message="Render failed.", error=str(exc))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dsp_available": DSP_AVAILABLE,
        "platform": "vercel" if _ON_VERCEL else "server",
    }


@app.post("/convert")
async def convert(file: UploadFile = File(...), preset: str = Form("reference_luxe"), mix_prompt: str = Form(""), stem_mode: str = Form("classic")):
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
    with src.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    _set_job(job_id, status="queued", message="Upload complete. Waiting for DSP worker…", input_name=file.filename, output_name=out.name, mix_prompt=mix_prompt, stem_mode=stem_mode)
    EXECUTOR.submit(_process_job, job_id, src, out, preset, mix_prompt, stem_mode)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing", "message": "Upload accepted. DSP render started."})

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown render job")
        return dict(job, job_id=job_id)
