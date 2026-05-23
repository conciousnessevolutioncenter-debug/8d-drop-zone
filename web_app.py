from __future__ import annotations

import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from eightd_engine.audio_io import export_audio, load_audio
from eightd_engine.dsp import analyze_correlation, bpm_to_premium_rotation_cpm, estimate_bpm, panning_preset_names, process_8d

APP_DIR = Path(tempfile.gettempdir()) / "8d_dropzone_live"
APP_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}
JOBS_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="The 8D Engine")
app.mount("/files", StaticFiles(directory=str(APP_DIR)), name="files")

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The 8D Engine — Spatial Audio Mastering</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg: #050711;
      --bg-2: #0b1020;
      --panel: rgba(13, 18, 33, 0.78);
      --panel-strong: rgba(19, 26, 45, 0.92);
      --line: rgba(180, 196, 255, 0.16);
      --line-strong: rgba(189, 202, 255, 0.28);
      --ink: rgba(248, 250, 252, 0.96);
      --muted: rgba(222, 229, 240, 0.80);
      --soft: rgba(181, 193, 211, 0.72);
      --accent: #cbb7fb;
      --accent-2: #76d7ff;
      --cream: #e9e5dd;
      --success: #9ef6c9;
      --shadow: 0 30px 100px rgba(0, 0, 0, 0.52), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    * { box-sizing: border-box; }
    html { min-height: 100%; background: var(--bg); }
    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      color: var(--ink);
      font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
      background:
        radial-gradient(circle at 16% 10%, rgba(118, 215, 255, 0.18), transparent 30vw),
        radial-gradient(circle at 82% 14%, rgba(203, 183, 251, 0.20), transparent 32vw),
        linear-gradient(145deg, #050711 0%, #090d19 46%, #0f1023 100%);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.34;
      background-image:
        linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: radial-gradient(circle at 50% 12%, black, transparent 72%);
    }
    .shell { width: min(1180px, calc(100vw - 40px)); margin: 0 auto; padding: 38px 0 48px; position: relative; }
    .nav { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:54px; }
    .brand { display:flex; align-items:center; gap:12px; color:var(--ink); text-decoration:none; font-weight:700; letter-spacing:-0.03em; }
    .mark { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; color:#111827; background:linear-gradient(135deg, var(--cream), #b9d8ff 55%, var(--accent)); box-shadow:0 12px 38px rgba(118,215,255,.18); }
    .nav-note { color:var(--soft); font-size:13px; letter-spacing:.08em; text-transform:uppercase; }
    .hero { display:grid; grid-template-columns: minmax(0, 1.02fr) minmax(360px, .98fr); gap:34px; align-items:stretch; }
    .copy { padding: 16px 0 0; }
    .eyebrow { display:inline-flex; align-items:center; gap:10px; padding:8px 12px; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.035); color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .pulse { width:7px; height:7px; border-radius:50%; background:var(--success); box-shadow:0 0 24px var(--success); }
    h1 { margin: 22px 0 18px; max-width: 780px; font-size: clamp(46px, 7vw, 86px); line-height: .91; letter-spacing: -0.075em; font-weight: 540; }
    .grad { background:linear-gradient(100deg, #fff 8%, #dbe8ff 45%, var(--accent) 92%); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .lede { max-width: 620px; color: var(--muted); font-size: clamp(17px, 2vw, 20px); line-height: 1.55; letter-spacing: -0.01em; margin:0 0 28px; }
    .proof { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; max-width:620px; }
    .proof div { border:1px solid var(--line); background:rgba(255,255,255,.035); border-radius:16px; padding:14px; min-height:84px; }
    .proof strong { display:block; color:var(--ink); font-size:20px; letter-spacing:-.04em; margin-bottom:4px; }
    .proof span { color:var(--soft); font-size:12px; line-height:1.35; }
    #zone {
      position:relative;
      overflow:hidden;
      min-height: 610px;
      border: 1px solid var(--line-strong);
      border-radius: 34px;
      background: linear-gradient(180deg, rgba(18,24,42,.92), rgba(10,14,27,.82));
      box-shadow: var(--shadow);
      padding: 28px;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
      transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
      isolation:isolate;
    }
    #zone::before { content:""; position:absolute; inset:-30%; background: radial-gradient(circle, rgba(203,183,251,.16), transparent 31%), conic-gradient(from 180deg, transparent, rgba(118,215,255,.14), transparent, rgba(203,183,251,.12), transparent); opacity:.52; animation: orbitGlow 18s linear infinite; z-index:-2; }
    #zone::after { content:""; position:absolute; inset:1px; border-radius:33px; background:linear-gradient(180deg, rgba(255,255,255,.045), transparent 32%); pointer-events:none; z-index:-1; }
    #zone.hover { transform: translateY(-3px); border-color: rgba(118,215,255,.72); box-shadow:0 34px 130px rgba(53, 90, 164, .36), var(--shadow); }
    @keyframes orbitGlow { to { transform: rotate(360deg); } }
    .visual { display:grid; place-items:center; padding:20px 0 10px; }
    .orbit { width:min(330px, 74vw); aspect-ratio:1; border-radius:50%; border:1px solid rgba(255,255,255,.16); position:relative; display:grid; place-items:center; background:radial-gradient(circle, rgba(255,255,255,.06), rgba(255,255,255,.01) 55%, transparent); }
    .orbit::before, .orbit::after { content:""; position:absolute; border-radius:50%; border:1px solid rgba(203,183,251,.18); }
    .orbit::before { inset:13%; transform:rotate(28deg) scaleY(.58); }
    .orbit::after { inset:26%; border-color:rgba(118,215,255,.16); transform:rotate(-31deg) scaleY(.62); }
    .dot { position:absolute; width:12px; height:12px; border-radius:50%; background:var(--cream); box-shadow:0 0 30px rgba(233,229,221,.84); offset-path: path('M 165 24 C 247 28 307 95 304 168 C 301 246 237 306 162 302 C 83 298 26 240 27 164 C 29 86 88 23 165 24'); animation: travel 10.4s linear infinite; }
    @keyframes travel { to { offset-distance:100%; } }
    .wave { width:72%; height:86px; opacity:.88; filter:drop-shadow(0 0 22px rgba(118,215,255,.14)); }
    .zone-copy { text-align:center; max-width: 560px; margin:0 auto; }
    .kicker { color:var(--accent); font-size:12px; text-transform:uppercase; letter-spacing:.16em; font-weight:800; margin-bottom:10px; }
    .title { font-size: clamp(27px, 3vw, 38px); line-height:1; font-weight:700; letter-spacing:-.055em; margin-bottom:12px; }
    .hint { color:var(--muted); line-height:1.55; font-size:15px; }
    .controls { display:grid; grid-template-columns: 1fr auto; gap:12px; align-items:end; margin-top:24px; }
    .field { text-align:left; }
    label { display:block; color:var(--soft); font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin:0 0 8px 2px; }
    select, button { font: inherit; }
    select { width:100%; min-height:52px; padding:0 42px 0 16px; border:1px solid var(--line); border-radius:14px; background:rgba(4,7,15,.72); color:var(--ink); font-weight:650; outline:none; box-shadow: inset 0 1px 0 rgba(255,255,255,.04); }
    select:focus { border-color:rgba(203,183,251,.7); box-shadow:0 0 0 4px rgba(203,183,251,.12); }
    button { min-height:52px; padding:0 20px; border:0; border-radius:14px; background:var(--cream); color:#111827; font-weight:800; cursor:pointer; white-space:nowrap; box-shadow:0 18px 50px rgba(233,229,221,.12); transition:transform .18s ease, filter .18s ease; }
    button:hover { transform:translateY(-1px); filter:brightness(1.04); }
    input { display:none; }
    .bar { width:100%; height:8px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; margin:18px auto 0; display:none; }
    .fill { width:0%; height:100%; background:linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius:999px; transition:width .15s linear; }
    .fill.indeterminate { width:32%; animation:load 1s infinite ease-in-out; }
    @keyframes load { 0%{transform:translateX(-115%)} 100%{transform:translateX(330%)} }
    .status-card { margin-top:18px; border:1px solid var(--line); border-radius:18px; padding:14px 16px; background:rgba(3,6,14,.48); }
    .status { color:var(--ink); font-weight:650; white-space:pre-line; font-size:13px; line-height:1.45; }
    a { color:var(--success); font-weight:800; text-decoration:none; }
    a:hover { text-decoration:underline; }
    .badges { margin-top:26px; display:flex; flex-wrap:wrap; gap:10px; }
    .badge { border:1px solid var(--line); background:rgba(255,255,255,.035); color:var(--muted); border-radius:999px; padding:9px 12px; font-size:12px; font-weight:700; }
    .signal { margin-top:34px; display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:10px; }
    .signal div { border:1px solid var(--line); border-radius:18px; padding:14px; background:rgba(255,255,255,.032); }
    .signal strong { display:block; font-size:13px; margin-bottom:5px; }
    .signal span { display:block; color:var(--soft); font-size:11px; line-height:1.35; }
    .mono { font-family:'JetBrains Mono', ui-monospace, monospace; }
    @media (max-width: 920px) {
      .hero { grid-template-columns:1fr; }
      .nav { margin-bottom:32px; }
      .proof, .signal { grid-template-columns:1fr; }
      #zone { min-height: auto; }
    }
    @media (max-width: 560px) {
      .shell { width:min(100vw - 24px, 1180px); padding-top:22px; }
      .nav-note { display:none; }
      .controls { grid-template-columns:1fr; }
      button { width:100%; }
      .orbit { width:260px; }
    }
    @media (prefers-reduced-motion: reduce) { .dot, #zone::before, .fill.indeterminate { animation:none; } }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="nav" aria-label="Product">
      <a class="brand" href="/" aria-label="The 8D Engine home"><span class="mark">◌</span><span>The 8D Engine</span></a>
      <div class="nav-note">Spatial Audio Mastering</div>
    </nav>
    <section class="hero">
      <div class="copy">
        <div class="eyebrow"><span class="pulse"></span> Professional headphone-first render</div>
        <h1><span class="grad">Elegant 8D masters</span> with a stable low end.</h1>
        <p class="lede">Upload a track, choose a mastering profile, and export a polished spatial mix with reference-style movement, mono-safe bass, subtle room, and clean 32-bit WAV detail.</p>
        <div class="proof" aria-label="Reference mix findings">
          <div><strong>10.4s</strong><span>Measured reference orbit — smooth enough to avoid fatigue.</span></div>
          <div><strong>150 Hz</strong><span>Protected crossover keeps kick and sub locked center.</span></div>
          <div><strong>0.81</strong><span>Reference median side/mid width in active sections.</span></div>
        </div>
        <div class="badges" aria-label="Processing highlights">
          <span class="badge">BPM aware</span>
          <span class="badge">Static cleanup</span>
          <span class="badge">Golden Ratio motion</span>
          <span class="badge">Fibonacci timing</span>
          <span class="badge">Reference Luxe preset</span>
        </div>
      </div>
      <div id="zone">
        <div class="visual" aria-hidden="true">
          <div class="orbit">
            <span class="dot"></span>
            <svg class="wave" viewBox="0 0 420 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M6 60 C34 19 64 103 96 60 C126 18 158 104 190 60 C222 17 254 103 286 60 C318 21 352 99 414 60" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/>
              <path d="M20 60 H400" stroke="rgba(255,255,255,.16)" stroke-width="1" stroke-dasharray="6 12"/>
              <defs><linearGradient id="g" x1="0" x2="420" y1="0" y2="0"><stop stop-color="#cbb7fb"/><stop offset=".55" stop-color="#76d7ff"/><stop offset="1" stop-color="#e9e5dd"/></linearGradient></defs>
            </svg>
          </div>
        </div>
        <div class="zone-copy">
          <div class="kicker">Import audio</div>
          <div class="title" id="title">Drop your track here</div>
          <div class="hint" id="hint">We analyze tempo and render a premium binaural orbit while keeping the sub-bass and kick centered. MP3, WAV, FLAC, M4A, and most FFmpeg-decodable files are accepted.</div>
          <div class="controls">
            <div class="field">
              <label for="preset">Mastering profile</label>
              <select id="preset">
                <option value="reference_luxe" selected>Reference Luxe — 10.4s orbit</option>
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
            <button onclick="document.getElementById('file').click()">Select Track</button>
          </div>
          <input id="file" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.aac,.ogg,.aiff">
          <div class="bar" id="bar"><div class="fill"></div></div>
          <div class="status-card"><div class="status" id="status">Ready for upload.</div></div>
        </div>
      </div>
    </section>
    <section class="signal" aria-label="Signal chain">
      <div><strong>64-bit DSP</strong><span>High precision internal processing before export.</span></div>
      <div><strong>Mono-safe bass</strong><span>Sub and kick remain centered below 150 Hz.</span></div>
      <div><strong>Binaural orbit</strong><span>ITD, ILD, rear shading, and smooth azimuth motion.</span></div>
      <div><strong>Cinematic room</strong><span>Subtle wet reflections support externalization.</span></div>
      <div><strong>32-bit WAV</strong><span>Float export keeps detail and avoids brittle renders.</span></div>
    </section>
  </main>
<script>
const zone = document.getElementById('zone');
const file = document.getElementById('file');
const title = document.getElementById('title');
const hint = document.getElementById('hint');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const preset = document.getElementById('preset');

['dragenter','dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('hover'); title.textContent='Release to master'; }));
['dragleave','drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('hover'); if (!file.files.length) title.textContent='Drop your track here'; }));
zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
file.addEventListener('change', e => { const f = e.target.files[0]; if (f) upload(f); });

async function upload(f) {
  title.textContent = 'Uploading…';
  hint.textContent = f.name;
  statusEl.textContent = 'Preparing secure upload…';
  bar.style.display = 'block';
  document.querySelector('.fill').style.width = '0%';
  document.querySelector('.fill').classList.remove('indeterminate');
  const data = new FormData();
  data.append('file', f);
  data.append('preset', preset.value);
  try {
    const json = await xhrUpload('/convert', data, pct => {
      statusEl.textContent = `Uploading: ${pct}%`;
      document.querySelector('.fill').style.width = `${pct}%`;
    });
    title.textContent = 'Rendering spatial master…';
    statusEl.textContent = 'Upload complete. Tempo analysis, cleanup, panning, room, and phase guard are running now.';
    document.querySelector('.fill').classList.add('indeterminate');
    await pollJob(json.job_id);
  } catch (err) {
    title.textContent = 'Master failed';
    hint.textContent = 'Try another audio file or a shorter upload if the tunnel times out.';
    statusEl.textContent = String(err);
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
    xhr.onerror = () => reject(new Error('Network upload failed'));
    xhr.send(formData);
  });
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/jobs/${jobId}`);
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    if (job.status === 'complete') {
      title.textContent = 'Spatial master ready';
      hint.innerHTML = `<a href="${job.download_url}" download>Download ${job.output_name}</a>`;
      statusEl.textContent = `Profile: ${job.preset}\nBPM: ${job.bpm}\nOrbit: ${job.rotation_cpm} cycles/min\nCorrelation: ${job.correlation} | Side/Mid: ${job.side_mid_ratio} | ${job.phase}`;
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || 'Render failed');
    statusEl.textContent = `${job.message || 'Rendering…'}\nYou can leave this tab open until the download link appears.`;
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

def _process_job(job_id: str, src: Path, out: Path, preset: str = "reference_luxe"):
    try:
        _set_job(job_id, status="processing", message="Analyzing BPM…")
        audio = load_audio(src)
        bpm = estimate_bpm(audio)
        safe_preset = preset if preset in panning_preset_names() else "reference_luxe"
        reference_speed_presets = {"reference_luxe", "phi_reference_orbit", "fibonacci_spiral", "golden_figure8", "lucas_breath"}
        rotation_cpm = 5.78 if safe_preset in reference_speed_presets else bpm_to_premium_rotation_cpm(bpm)
        preset_settings = {
            "reference_luxe": dict(room_size=0.22, motion_depth=0.86, high_emphasis=0.72, spatial_mix=0.74),
            "phi_reference_orbit": dict(room_size=0.22, motion_depth=0.84, high_emphasis=0.72, spatial_mix=0.74),
            "fibonacci_spiral": dict(room_size=0.24, motion_depth=0.88, high_emphasis=0.76, spatial_mix=0.76),
            "golden_figure8": dict(room_size=0.20, motion_depth=0.82, high_emphasis=0.70, spatial_mix=0.72),
            "lucas_breath": dict(room_size=0.26, motion_depth=0.68, high_emphasis=0.64, spatial_mix=0.66),
            "wide_orbit": dict(room_size=0.20, motion_depth=0.84, high_emphasis=0.68, spatial_mix=0.72),
            "vocal_safe": dict(room_size=0.14, motion_depth=0.52, high_emphasis=0.48, spatial_mix=0.54),
            "cinematic_halo": dict(room_size=0.24, motion_depth=0.70, high_emphasis=0.62, spatial_mix=0.66),
            "figure8": dict(room_size=0.18, motion_depth=0.76, high_emphasis=0.62, spatial_mix=0.68),
        }
        settings = preset_settings.get(
            safe_preset,
            dict(room_size=0.18, motion_depth=0.78, high_emphasis=0.65, spatial_mix=0.68),
        )
        _set_job(
            job_id,
            message="Cleaning static, then rendering premium spatial master…",
            bpm=round(bpm, 1),
            rotation_cpm=round(rotation_cpm, 2),
            preset=safe_preset,
        )
        rendered = process_8d(
            audio,
            rotation_cpm=rotation_cpm,
            room_size=settings["room_size"],
            crossover_hz=150.0,
            motion_depth=settings["motion_depth"],
            high_emphasis=settings["high_emphasis"],
            spatial_mix=settings["spatial_mix"],
            denoise_amount=0.72,
            panning_preset=safe_preset,
            preserve_quality=True,
            youtube_master=False,
            section_automation=True,
        )
        _set_job(job_id, message="Writing WAV export…")
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
        )
    except Exception as exc:
        _set_job(job_id, status="failed", message="Render failed.", error=str(exc))

@app.post("/convert")
async def convert(file: UploadFile = File(...), preset: str = Form("reference_luxe")):
    suffix = Path(file.filename or "audio").suffix.lower() or ".audio"
    safe_stem = Path(file.filename or "audio").stem.replace("/", "_").replace("\\", "_")[:80]
    job_id = uuid.uuid4().hex[:12]
    src = APP_DIR / f"{safe_stem}_{job_id}{suffix}"
    out = APP_DIR / f"{safe_stem}_{job_id}_8D_Final.wav"
    with src.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    _set_job(job_id, status="queued", message="Upload complete. Waiting for DSP worker…", input_name=file.filename, output_name=out.name)
    EXECUTOR.submit(_process_job, job_id, src, out, preset)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing", "message": "Upload accepted. DSP render started."})

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown render job")
        return dict(job, job_id=job_id)
