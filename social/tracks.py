"""The distribution engine: public track pages, embeds, share cards, visualizer.

Every published 8D master gets a gorgeous public page (``/t/{slug}``) with a
spatial player, an auto-unfurling share card (so every link posted to X / FB /
WhatsApp becomes a free ad), a one-tap TikTok/Reels visualizer video, and an
embeddable player (``/embed/{slug}``). Public + cookie-free, so Vercel can proxy
these straight from the premium domain.

Heavy deps (soundfile, Pillow) are imported lazily so this module still imports
cleanly on the Vercel serverless homepage where the DSP stack is absent.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from .db import get_db
from .models import Track, User
from .ui import esc

# Same persistent media dir the rest of the social layer uses (set
# SOCIAL_MEDIA_DIR to a mounted volume in prod; temp dir is wiped on redeploy).
MEDIA_DIR = Path(os.environ.get("SOCIAL_MEDIA_DIR") or (Path(tempfile.gettempdir()) / "8d_social_media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

tracks_router = APIRouter(tags=["tracks"])

_ALPHABET = "23456789abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"  # no look-alikes


def site_url(request: Request) -> str:
    """Absolute origin for share/OG links — the premium domain in prod."""
    return (os.environ.get("PUBLIC_SITE_URL") or str(request.base_url)).rstrip("/")


def new_slug(db: Session, n: int = 8) -> str:
    for _ in range(12):
        slug = "".join(secrets.choice(_ALPHABET) for _ in range(n))
        if not db.query(Track).filter_by(slug=slug).first():
            return slug
    return "".join(secrets.choice(_ALPHABET) for _ in range(n + 4))


def compute_peaks(path: Path, buckets: int = 480) -> list[float]:
    """Downsample audio to ``buckets`` normalized 0..1 peaks for the waveform.

    Streamed in blocks so a long track never loads fully into memory.
    """
    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        return []
    try:
        info = sf.info(str(path))
        total = info.frames
        if total <= 0:
            return []
        per = max(1, total // buckets)
        peaks: list[float] = []
        with sf.SoundFile(str(path)) as f:
            while True:
                block = f.read(per, dtype="float32", always_2d=True)
                if not len(block):
                    break
                peaks.append(float(np.abs(block).max()) if block.size else 0.0)
                if len(peaks) >= buckets:
                    break
        mx = max(peaks) if peaks else 1.0
        if mx <= 0:
            mx = 1.0
        return [round(p / mx, 3) for p in peaks]
    except Exception:
        return []


def audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return round(info.frames / float(info.samplerate or 1), 2)
    except Exception:
        return 0.0


def create_track(db: Session, *, audio_path: Path, title: str, artist: str = "",
                 lufs: str = "", true_peak: str = "", preset: str = "",
                 owner_id: int | None = None, wav_path: Path | None = None,
                 allow_download: bool = False, watermarked: bool = True) -> Track:
    """Persist a published track: copy audio into the media dir, precompute the
    waveform + duration, and create the row. ``audio_path`` should already be a
    streamable mp3; ``wav_path`` is the optional lossless master."""
    slug = new_slug(db)
    ext = audio_path.suffix.lower() or ".mp3"
    audio_name = f"trk_{slug}{ext}"
    (MEDIA_DIR / audio_name).write_bytes(audio_path.read_bytes())

    wav_name = None
    if wav_path and wav_path.exists():
        wav_name = f"trk_{slug}.wav"
        (MEDIA_DIR / wav_name).write_bytes(wav_path.read_bytes())

    peaks = compute_peaks(MEDIA_DIR / audio_name)
    dur = audio_duration(MEDIA_DIR / audio_name)

    track = Track(
        slug=slug, owner_id=owner_id, title=(title or "Untitled")[:120], artist=(artist or "")[:80],
        audio_name=audio_name, wav_name=wav_name, duration=dur,
        lufs=str(lufs or ""), true_peak=str(true_peak or ""), preset=str(preset or ""),
        allow_download=bool(allow_download), watermarked=bool(watermarked),
        peaks=json.dumps(peaks),
    )
    db.add(track)
    db.commit()
    db.refresh(track)
    return track


# ── OG share card (Pillow) ──────────────────────────────────────────────────────
def _load_font(size: int):
    from PIL import ImageFont
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)   # Pillow >= 10.1 scalable default
    except Exception:
        return ImageFont.load_default()


def make_cover_png(track: Track) -> bytes | None:
    """1200x630 share card so links unfurl beautifully on every platform."""
    try:
        import math
        from PIL import Image, ImageDraw
    except Exception:
        return None
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), (6, 11, 22))
    d = ImageDraw.Draw(img)
    # vertical gradient wash
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(8 + 12 * (1 - t)), int(16 + 18 * (1 - t)), int(34 + 30 * (1 - t))))
    # orbit rings (right side)
    cx, cy = 940, 315
    for r, col in ((150, (72, 227, 255)), (110, (157, 139, 255)), (70, (72, 227, 255))):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
    for i in range(36):
        a = i / 36 * 2 * math.pi
        rr = 150
        px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(72, 227, 255))
    # text
    eyebrow = _load_font(30); big = _load_font(74); sub = _load_font(34); tag = _load_font(28)
    d.text((80, 92), "THE 8D ENGINE", font=eyebrow, fill=(72, 227, 255))
    title = (track.title or "Untitled")
    if len(title) > 28:
        title = title[:27] + "…"
    d.text((80, 150), title, font=big, fill=(231, 237, 246))
    artist = track.artist or "Spatial master"
    d.text((80, 250), artist[:34], font=sub, fill=(157, 168, 184))
    d.text((80, 510), "🎧  Listen in 8D — headphones on", font=tag, fill=(72, 227, 255))
    if getattr(track, "watermarked", True):
        d.text((80, 556), "the8dengine.com", font=tag, fill=(157, 168, 184))
    from io import BytesIO
    out = BytesIO(); img.save(out, format="PNG"); return out.getvalue()


# ── Pages ───────────────────────────────────────────────────────────────────────
def _fmt_time(s: float) -> str:
    s = int(s or 0)
    return f"{s // 60}:{s % 60:02d}"


def _meta_head(request: Request, track: Track) -> str:
    base = site_url(request)
    url = f"{base}/t/{track.slug}"
    cover = f"{base}/t/{track.slug}/cover.png"
    audio = f"{base}/t/{track.slug}/audio"
    title = esc(track.title or "Untitled")
    artist = esc(track.artist or "")
    desc = f"{artist + ' · ' if artist else ''}A spatial 8D master. Headphones on — feel it orbit your head."
    return f"""
  <title>{title} · 8D Engine</title>
  <meta name="description" content="{esc(desc)}"/>
  <link rel="canonical" href="{url}"/>
  <meta property="og:type" content="music.song"/>
  <meta property="og:title" content="{title}"/>
  <meta property="og:description" content="{esc(desc)}"/>
  <meta property="og:url" content="{url}"/>
  <meta property="og:image" content="{cover}"/>
  <meta property="og:image:width" content="1200"/>
  <meta property="og:image:height" content="630"/>
  <meta property="og:audio" content="{audio}"/>
  <meta name="twitter:card" content="summary_large_image"/>
  <meta name="twitter:title" content="{title}"/>
  <meta name="twitter:description" content="{esc(desc)}"/>
  <meta name="twitter:image" content="{cover}"/>"""


_PLAYER_CSS = """
  :root{ --bg:#06101c; --cyan:#48e3ff; --violet:#9d8bff; --soft:#8a93a8; --ink:#e7edf6;
    --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace; --display:'Space Grotesk',system-ui,sans-serif; }
  *{ box-sizing:border-box; } a{ color:inherit; }
  body{ margin:0; min-height:100vh; color:var(--ink); font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;
    background:radial-gradient(1100px 700px at 50% -10%,#16213a 0%,rgba(6,16,28,0) 60%),linear-gradient(180deg,#06101c,#04070d); }
  .wrap{ max-width:560px; margin:0 auto; padding:40px 22px 60px; text-align:center; }
  .brand{ font-family:var(--mono); font-size:11px; letter-spacing:.26em; text-transform:uppercase; color:var(--cyan); text-decoration:none; }
  .stage{ position:relative; width:300px; height:300px; margin:30px auto 8px; }
  canvas#orbit{ width:300px; height:300px; display:block; }
  .play{ position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:78px; height:78px; border-radius:50%; border:none;
    cursor:pointer; font-size:26px; color:#04141a; background:radial-gradient(circle at 50% 35%,#8aecff,#2bb6e0);
    box-shadow:0 0 30px rgba(72,227,255,.5), inset 0 1px 0 rgba(255,255,255,.4); }
  h1{ font-family:var(--display); font-weight:700; font-size:26px; margin:14px 0 2px; letter-spacing:-.01em; }
  .artist{ color:var(--soft); font-size:14px; margin:0 0 4px; }
  .phones{ font-family:var(--mono); font-size:11px; letter-spacing:.12em; color:var(--cyan); margin:10px 0 18px; }
  .wave{ width:100%; height:54px; display:block; margin:6px 0 4px; cursor:pointer; }
  .tline{ display:flex; justify-content:space-between; font-family:var(--mono); font-size:11px; color:var(--soft); }
  .meta{ display:flex; gap:14px; justify-content:center; font-family:var(--mono); font-size:11px; color:var(--soft); margin:14px 0 22px; flex-wrap:wrap; }
  .share{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-bottom:18px; }
  .sbtn{ cursor:pointer; border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.04); color:#cdd6e6; border-radius:10px;
    padding:10px 14px; font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-decoration:none; }
  .sbtn:hover{ border-color:var(--cyan); color:var(--cyan); }
  .sbtn.vid{ background:linear-gradient(135deg,var(--cyan),var(--violet)); color:#06101c; border-color:transparent; font-weight:600; }
  .cta{ display:inline-block; margin-top:8px; font-family:var(--mono); font-size:12px; letter-spacing:.1em; text-transform:uppercase;
    color:#06101c; background:linear-gradient(135deg,var(--cyan),var(--violet)); border-radius:999px; padding:12px 22px; text-decoration:none; }
  .toast{ position:fixed; left:50%; bottom:26px; transform:translateX(-50%) translateY(20px); opacity:0; transition:.25s;
    background:#0d1426; border:1px solid rgba(255,255,255,.16); color:#e7edf6; padding:10px 16px; border-radius:10px; font-size:13px; }
  .toast.on{ opacity:1; transform:translateX(-50%) translateY(0); }
  .vidnote{ font-size:11px; color:var(--soft); margin-top:8px; min-height:14px; font-family:var(--mono); }
"""


_PLAYER_BODY = r"""
const API = location.origin;
let audio = new Audio(); audio.crossOrigin = 'anonymous'; audio.src = AUDIO_URL; audio.preload = 'metadata';
let actx=null, analyser=null, srcNode=null, raf=0, played=false;
const orbit = document.getElementById('orbit');
const octx = orbit.getContext('2d');
const DPR = Math.min(2, window.devicePixelRatio||1);
orbit.width = orbit.clientWidth*DPR; orbit.height = orbit.clientHeight*DPR; octx.scale(DPR,DPR);
const OW = orbit.clientWidth, OH = orbit.clientHeight;

function ensureGraph(){
  if (actx) return;
  actx = new (window.AudioContext||window.webkitAudioContext)();
  srcNode = actx.createMediaElementSource(audio);
  analyser = actx.createAnalyser(); analyser.fftSize = 256;
  srcNode.connect(analyser); analyser.connect(actx.destination);
}
function drawOrbit(level){
  const cx=OW/2, cy=OH/2; octx.clearRect(0,0,OW,OH);
  const t = performance.now()/1000;
  for (let ring=0; ring<3; ring++){
    const r = (OW*0.2 + ring*OW*0.11) + level*40*(1+ring*0.3);
    octx.beginPath(); octx.arc(cx,cy,r,0,Math.PI*2);
    octx.strokeStyle = ring%2 ? 'rgba(157,139,255,'+(0.18+level*0.5)+')' : 'rgba(72,227,255,'+(0.2+level*0.5)+')';
    octx.lineWidth = 1.5; octx.stroke();
  }
  const dots = 40, R = OW*0.32 + level*42;
  for (let i=0;i<dots;i++){
    const a = (i/dots)*Math.PI*2 + t*(0.5+level);
    const px = cx + R*Math.cos(a), py = cy + R*Math.sin(a);
    const s = 1.6 + level*4*(0.5+0.5*Math.sin(a*3+t));
    octx.beginPath(); octx.arc(px,py,s,0,Math.PI*2);
    octx.fillStyle = i%2 ? 'rgba(157,139,255,.9)' : 'rgba(72,227,255,.95)';
    octx.shadowColor = 'rgba(72,227,255,.8)'; octx.shadowBlur = 8+level*16; octx.fill();
  }
  octx.shadowBlur = 0;
}
function loop(){
  let level = 0;
  if (analyser){ const b=new Uint8Array(analyser.frequencyBinCount); analyser.getByteTimeDomainData(b);
    let s=0; for(let i=0;i<b.length;i++){ const v=(b[i]-128)/128; s+=v*v; } level=Math.min(1,Math.sqrt(s/b.length)*2.2); }
  drawOrbit(level);
  drawWave();
  raf = requestAnimationFrame(loop);
}
const playBtn = document.getElementById('play');
playBtn.onclick = async () => {
  ensureGraph(); await actx.resume();
  if (audio.paused){ await audio.play(); playBtn.innerHTML='&#10073;&#10073;'; if(!played){ played=true; if(navigator.sendBeacon) navigator.sendBeacon(API+'/t/'+SLUG+'/play'); } if(!raf) loop(); }
  else { audio.pause(); playBtn.innerHTML='&#9658;'; }
};
audio.onended = () => { playBtn.innerHTML='&#9658;'; };

// waveform
const wave = document.getElementById('wave');
let wctx=null, wW=0, wH=54;
function setupWave(){ if(!wave) return; const r=wave.getBoundingClientRect(); wW=r.width||300; wave.width=wW*DPR; wave.height=wH*DPR; wctx=wave.getContext('2d'); wctx.setTransform(DPR,0,0,DPR,0,0); }
function drawWave(){
  if(!wctx||!PEAKS.length) return;
  const prog = audio.duration ? audio.currentTime/audio.duration : 0;
  wctx.clearRect(0,0,wW,wH); const bw=wW/PEAKS.length, mid=wH/2;
  for(let i=0;i<PEAKS.length;i++){ const x=i*bw, hh=Math.max(1,PEAKS[i]*mid*0.96);
    wctx.fillStyle = (i/PEAKS.length<prog) ? '#48e3ff' : 'rgba(255,255,255,.16)';
    wctx.fillRect(x, mid-hh, Math.max(1,bw-1), hh*2); }
}
if (wave){ wave.addEventListener('click', e => { const r=wave.getBoundingClientRect(); if(audio.duration) audio.currentTime=((e.clientX-r.left)/r.width)*audio.duration; });
  const tt=document.getElementById('tcur'), td=document.getElementById('tdur');
  audio.ontimeupdate=()=>{ if(tt)tt.textContent=fmtT(audio.currentTime); drawWave(); };
  audio.onloadedmetadata=()=>{ if(td)td.textContent=fmtT(audio.duration); }; }
function fmtT(s){ s=Math.max(0,s|0); return (s/60|0)+':'+String(s%60).padStart(2,'0'); }
window.addEventListener('resize', ()=>{ setupWave(); drawWave(); }); setupWave(); drawOrbit(0);
"""


def _player_js(track: Track, request: Request, embed: bool = False) -> str:
    base = site_url(request)
    from urllib.parse import urlparse
    domain = urlparse(base).netloc or "the8dengine.com"
    header = (
        "const AUDIO_URL=" + json.dumps(f"{base}/t/{track.slug}/audio") + ";\n"
        "const SHARE_URL=" + json.dumps(f"{base}/t/{track.slug}") + ";\n"
        "const SLUG=" + json.dumps(track.slug) + ";\n"
        "const TITLE=" + json.dumps(track.title or "Untitled") + ";\n"
        "const PEAKS=" + (track.peaks or "[]") + ";\n"
        "const EMBED=" + ("true" if embed else "false") + ";\n"
        "const DOMAIN=" + json.dumps(domain) + ";\n"
        # Free tier carries the brand watermark; paid (watermarked=False) gets a clean export.
        "const WATERMARKED=" + ("true" if getattr(track, "watermarked", True) else "false") + ";\n"
    )
    return header + _PLAYER_BODY


def _share_js() -> str:
    return r"""
function toast(m){ const t=document.getElementById('toast'); t.textContent=m; t.classList.add('on'); setTimeout(()=>t.classList.remove('on'),1800); }
function beaconShare(){ try{ navigator.sendBeacon(location.origin+'/t/'+SLUG+'/share'); }catch(e){} }
function shareTo(net){
  beaconShare();
  const u=encodeURIComponent(SHARE_URL), txt=encodeURIComponent('🎧 '+TITLE+' — listen in 8D (headphones on)');
  let url='';
  if(net==='x') url='https://twitter.com/intent/tweet?text='+txt+'&url='+u;
  else if(net==='fb') url='https://www.facebook.com/sharer/sharer.php?u='+u;
  else if(net==='wa') url='https://api.whatsapp.com/send?text='+txt+'%20'+u;
  else if(net==='rd') url='https://www.reddit.com/submit?url='+u+'&title='+txt;
  if(url) window.open(url,'_blank','noopener,width=600,height=620');
}
async function nativeShare(){
  beaconShare();
  if(navigator.share){ try{ await navigator.share({title:TITLE, text:'🎧 Listen in 8D', url:SHARE_URL}); return; }catch(e){} }
  copyLink();
}
async function copyLink(){ try{ await navigator.clipboard.writeText(SHARE_URL); toast('Link copied'); }catch(e){ toast(SHARE_URL); } beaconShare(); }

// ── TikTok/Reels visualizer video (9:16, client-side, real time) ─────────────
async function makeVideo(){
  const note=document.getElementById('vidnote');
  if(!window.MediaRecorder){ note.textContent='Video export not supported in this browser.'; return; }
  ensureGraph(); await actx.resume();
  note.textContent='Recording a 30s clip… keep this tab open.';
  const W=1080,H=1920, cv=document.createElement('canvas'); cv.width=W; cv.height=H; const c=cv.getContext('2d');
  const dest=actx.createMediaStreamDestination(); analyser.connect(dest);
  const vstream=cv.captureStream(30); dest.stream.getAudioTracks().forEach(tr=>vstream.addTrack(tr));
  let mime='video/webm;codecs=vp9,opus'; if(!MediaRecorder.isTypeSupported(mime)) mime='video/webm';
  const rec=new MediaRecorder(vstream,{mimeType:mime, videoBitsPerSecond:6000000}); const chunks=[];
  rec.ondataavailable=e=>{ if(e.data.size) chunks.push(e.data); };
  rec.onstop=()=>{ const blob=new Blob(chunks,{type:'video/webm'}); const a=document.createElement('a');
    a.href=URL.createObjectURL(blob); a.download='8d_'+SLUG+'.webm'; a.click(); note.textContent='Saved! Post it with #8DAudio 🎧'; };
  audio.currentTime=0; await audio.play();
  const t0=performance.now(); rec.start();
  (function frame(){
    const t=(performance.now()-t0)/1000;
    const b=new Uint8Array(analyser.frequencyBinCount); analyser.getByteTimeDomainData(b);
    let s=0; for(let i=0;i<b.length;i++){ const v=(b[i]-128)/128; s+=v*v; } const lv=Math.min(1,Math.sqrt(s/b.length)*2.2);
    // bg
    const g=c.createLinearGradient(0,0,0,H); g.addColorStop(0,'#0b1830'); g.addColorStop(1,'#04070d'); c.fillStyle=g; c.fillRect(0,0,W,H);
    const cx=W/2, cy=H*0.42;
    for(let ring=0;ring<4;ring++){ c.beginPath(); c.arc(cx,cy,180+ring*70+lv*120,0,Math.PI*2);
      c.strokeStyle= ring%2?'rgba(157,139,255,'+(0.15+lv*0.5)+')':'rgba(72,227,255,'+(0.18+lv*0.5)+')'; c.lineWidth=4; c.stroke(); }
    const dots=48,R=300+lv*150;
    for(let i=0;i<dots;i++){ const a=(i/dots)*Math.PI*2+t*(0.6+lv); const px=cx+R*Math.cos(a),py=cy+R*Math.sin(a);
      c.beginPath(); c.arc(px,py,5+lv*16*(0.5+0.5*Math.sin(a*3+t)),0,Math.PI*2);
      c.fillStyle=i%2?'rgba(157,139,255,.95)':'rgba(72,227,255,.98)'; c.shadowColor='rgba(72,227,255,.9)'; c.shadowBlur=20+lv*40; c.fill(); }
    c.shadowBlur=0;
    c.textAlign='center'; c.fillStyle='#e7edf6'; c.font='700 76px Space Grotesk, sans-serif';
    c.fillText(TITLE.length>20?TITLE.slice(0,19)+'…':TITLE, cx, H*0.74);
    c.fillStyle='#48e3ff'; c.font='600 38px JetBrains Mono, monospace'; c.fillText('🎧 LISTEN IN 8D', cx, H*0.79);
    if(WATERMARKED){
      c.fillStyle='rgba(255,255,255,.82)'; c.font='700 34px JetBrains Mono, monospace'; c.fillText('THE 8D ENGINE', cx, H*0.92);
      c.fillStyle='rgba(72,227,255,.9)'; c.font='600 28px JetBrains Mono, monospace'; c.fillText(DOMAIN, cx, H*0.955);
    }
    if(t>=30||audio.ended){ rec.stop(); audio.pause(); return; }
    requestAnimationFrame(frame);
  })();
}
"""


def render_player_page(request: Request, track: Track) -> str:
    plays = track.plays or 0
    loud = f"{track.lufs} LUFS" if track.lufs else ""
    return f"""<!doctype html><html lang="en"><head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  {_meta_head(request, track)}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
  <style>{_PLAYER_CSS}</style>
</head><body>
  <div class="wrap">
    <a class="brand" href="{site_url(request)}/">◌ THE 8D ENGINE</a>
    <div class="stage">
      <canvas id="orbit"></canvas>
      <button class="play" id="play" aria-label="Play">&#9658;</button>
    </div>
    <h1>{esc(track.title or 'Untitled')}</h1>
    <p class="artist">{esc(track.artist or 'Spatial master')}</p>
    <div class="phones">🎧 HEADPHONES ON — FEEL IT ORBIT</div>
    <canvas id="wave" class="wave"></canvas>
    <div class="tline"><span id="tcur">0:00</span><span id="tdur">{_fmt_time(track.duration)}</span></div>
    <div class="meta">
      <span>▶ {plays} plays</span>{('<span>'+esc(loud)+'</span>') if loud else ''}
      {('<span>'+esc(track.preset)+'</span>') if track.preset else ''}
    </div>
    <div class="share">
      <button class="sbtn vid" onclick="makeVideo()">🎬 Reels/TikTok video</button>
      <a class="sbtn" onclick="nativeShare()">Share</a>
      <a class="sbtn" onclick="shareTo('x')">X</a>
      <a class="sbtn" onclick="shareTo('wa')">WhatsApp</a>
      <a class="sbtn" onclick="shareTo('fb')">Facebook</a>
      <a class="sbtn" onclick="copyLink()">Copy link</a>
    </div>
    <div class="vidnote" id="vidnote"></div>
    <a class="cta" href="{site_url(request)}/">Make your own in 8D →</a>
  </div>
  <div class="toast" id="toast"></div>
  <script>{_player_js(track, request)}{_share_js()}</script>
</body></html>"""


def render_embed_page(request: Request, track: Track) -> str:
    return f"""<!doctype html><html lang="en"><head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{esc(track.title or 'Untitled')} · 8D Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
  <style>{_PLAYER_CSS}
  body{{ background:#06101c; }} .wrap{{ padding:18px; max-width:420px; }} .stage{{ width:200px; height:200px; margin:8px auto; }}
  canvas#orbit{{ width:200px; height:200px; }} .play{{ width:60px; height:60px; font-size:20px; }} h1{{ font-size:18px; }}
  </style>
</head><body>
  <div class="wrap">
    <div class="stage"><canvas id="orbit"></canvas><button class="play" id="play" aria-label="Play">&#9658;</button></div>
    <h1>{esc(track.title or 'Untitled')}</h1>
    <p class="artist">{esc(track.artist or '')}</p>
    <canvas id="wave" class="wave"></canvas>
    <div class="tline"><span id="tcur">0:00</span><span id="tdur">{_fmt_time(track.duration)}</span></div>
    <a class="cta" href="{site_url(request)}/t/{track.slug}" target="_blank">Open in 8D →</a>
  </div>
  <script>{_player_js(track, request, embed=True)}</script>
</body></html>"""


# ── Routes ──────────────────────────────────────────────────────────────────────
def _get(db: Session, slug: str) -> Track | None:
    return db.query(Track).filter_by(slug=slug).first()


@tracks_router.get("/t/{slug}", response_class=HTMLResponse)
def track_page(slug: str, request: Request, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if not t or not t.is_public:
        return HTMLResponse("<h1 style='font-family:sans-serif;color:#888;text-align:center;margin-top:80px'>Track not found</h1>", status_code=404)
    return HTMLResponse(render_player_page(request, t))


@tracks_router.get("/embed/{slug}", response_class=HTMLResponse)
def embed_page(slug: str, request: Request, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if not t or not t.is_public:
        return HTMLResponse("Not found", status_code=404)
    return HTMLResponse(render_embed_page(request, t))


@tracks_router.get("/t/{slug}/audio")
def track_audio(slug: str, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if not t:
        return JSONResponse({"detail": "not found"}, status_code=404)
    path = MEDIA_DIR / t.audio_name
    if not path.exists():
        return JSONResponse({"detail": "audio missing"}, status_code=404)
    mt = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(str(path), media_type=mt)


@tracks_router.get("/t/{slug}/download")
def track_download(slug: str, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if not t or not t.allow_download or not t.wav_name:
        return JSONResponse({"detail": "download not available"}, status_code=403)
    path = MEDIA_DIR / t.wav_name
    if not path.exists():
        return JSONResponse({"detail": "file missing"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav", filename=f"{(t.title or 'track')}_8D.wav")


@tracks_router.get("/t/{slug}/cover.png")
def track_cover(slug: str, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if not t:
        return JSONResponse({"detail": "not found"}, status_code=404)
    cache = MEDIA_DIR / f"cover_{slug}.png"
    if not cache.exists():
        png = make_cover_png(t)
        if not png:
            return JSONResponse({"detail": "cover unavailable"}, status_code=404)
        cache.write_bytes(png)
    return FileResponse(str(cache), media_type="image/png")


@tracks_router.post("/t/{slug}/play")
def track_play(slug: str, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if t:
        t.plays = (t.plays or 0) + 1
        db.commit()
    return Response(status_code=204)


@tracks_router.post("/t/{slug}/share")
def track_share(slug: str, db: Session = Depends(get_db)):
    t = _get(db, slug)
    if t:
        t.shares = (t.shares or 0) + 1
        db.commit()
    return Response(status_code=204)
