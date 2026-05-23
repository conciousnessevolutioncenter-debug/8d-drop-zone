from __future__ import annotations

import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from eightd_engine.audio_io import export_audio, load_audio
from eightd_engine.dsp import analyze_correlation, bpm_to_premium_rotation_cpm, estimate_bpm, process_8d

APP_DIR = Path(tempfile.gettempdir()) / "8d_dropzone_live"
APP_DIR.mkdir(parents=True, exist_ok=True)
JOBS = {}
JOBS_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="8D Drop-Zone Live Test")
app.mount("/files", StaticFiles(directory=str(APP_DIR)), name="files")

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>8D Drop-Zone Live</title>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; min-height:100vh; background:#0b0f17; color:#f8fafc; font-family:Inter, system-ui, -apple-system, Segoe UI, sans-serif; display:flex; align-items:center; justify-content:center; }
    .shell { width:min(860px, calc(100vw - 40px)); }
    h1 { font-size:44px; margin:0 0 6px; letter-spacing:-1.5px; }
    .sub { color:#94a3b8; margin-bottom:24px; font-size:16px; }
    #zone { border:2px solid #334155; border-radius:28px; background:#111827; min-height:330px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; transition:.18s ease; box-shadow:0 24px 80px rgba(0,0,0,.35); padding:36px; }
    #zone.hover { background:#16324f; border-color:#38bdf8; transform:translateY(-2px); }
    .arrow { color:#38bdf8; font-size:76px; line-height:1; margin-bottom:8px; }
    .title { font-size:32px; font-weight:800; margin-bottom:10px; }
    .hint { color:#94a3b8; max-width:620px; line-height:1.45; }
    .bar { width:min(520px, 90%); height:10px; border-radius:999px; background:#020617; overflow:hidden; margin:24px auto 0; display:none; }
    .fill { width:0%; height:100%; background:#38bdf8; border-radius:999px; transition:width .15s linear; }
    .fill.indeterminate { width:35%; animation:load 1s infinite ease-in-out; }
    @keyframes load { 0%{transform:translateX(-110%)} 100%{transform:translateX(320%)} }
    .status { margin-top:18px; color:#f8fafc; font-weight:700; white-space:pre-line; }
    .footer { color:#64748b; margin-top:18px; font-size:13px; line-height:1.4; }
    a { color:#22c55e; font-weight:800; }
    input { display:none; }
    button { margin-top:18px; padding:12px 18px; border:0; border-radius:12px; background:#38bdf8; color:#020617; font-weight:800; cursor:pointer; }
  </style>
</head>
<body>
  <div class="shell">
    <h1>8D Drop-Zone Live</h1>
    <div class="sub">Drag. Analyze. Convert. Browser test wrapper using the same Python DSP engine.</div>
    <div id="zone">
      <div class="arrow">⬇</div>
      <div class="title" id="title">Drop Audio Here</div>
      <div class="hint" id="hint">Drop any audio file FFmpeg/soundfile can decode → ultra-fast upload handoff, BPM lock, 150 Hz mono bass protection, HRTF-inspired orbit, subtle room, YouTube-safe master. Long DJ sets are allowed; processing can take several minutes after upload.</div>
      <button onclick="document.getElementById('file').click()">Choose File</button>
      <input id="file" type="file">
      <div class="bar" id="bar"><div class="fill"></div></div>
      <div class="status" id="status">Ready.</div>
    </div>
    <div class="footer">Premium signal chain: 64-bit float DSP → 150 Hz mono bass protection → coherent high-frequency spatial cues kept attached to the musical body → mid/high binaural orbit with ITD/ILD/rear shading → subtle cinematic room → quality guard/no clipping → 32-bit float WAV export.</div>
  </div>
<script>
const zone = document.getElementById('zone');
const file = document.getElementById('file');
const title = document.getElementById('title');
const hint = document.getElementById('hint');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');

['dragenter','dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('hover'); title.textContent='Release to Convert'; }));
['dragleave','drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('hover'); }));
zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
file.addEventListener('change', e => { const f = e.target.files[0]; if (f) upload(f); });

async function upload(f) {
  title.textContent = 'Uploading…';
  hint.textContent = f.name;
  statusEl.textContent = 'Starting upload…';
  bar.style.display = 'block';
  document.querySelector('.fill').style.width = '0%';
  document.querySelector('.fill').classList.remove('indeterminate');
  const data = new FormData();
  data.append('file', f);
  try {
    const json = await xhrUpload('/convert', data, pct => {
      statusEl.textContent = `Uploading: ${pct}%`;
      document.querySelector('.fill').style.width = `${pct}%`;
    });
    title.textContent = 'Uploaded Fast — Rendering…';
    statusEl.textContent = 'Upload complete. DSP render is running in the background…';
    document.querySelector('.fill').classList.add('indeterminate');
    await pollJob(json.job_id);
  } catch (err) {
    title.textContent = 'Conversion Failed';
    hint.textContent = 'Try a shorter file if the browser/tunnel times out. Format is not pre-filtered; decode depends on FFmpeg/soundfile support.';
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
      title.textContent = 'Success!';
      hint.innerHTML = `<a href="${job.download_url}" download>Download ${job.output_name}</a>`;
      statusEl.textContent = `BPM: ${job.bpm}\n2-bar orbit: ${job.rotation_cpm} cycles/min\nCorrelation: ${job.correlation} | Side/Mid: ${job.side_mid_ratio} | ${job.phase}`;
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || 'Render failed');
    statusEl.textContent = `${job.message || 'Rendering…'}\nUpload is finished; you can leave this tab open until the download link appears.`;
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

def _process_job(job_id: str, src: Path, out: Path):
    try:
        _set_job(job_id, status="processing", message="Analyzing BPM…")
        audio = load_audio(src)
        bpm = estimate_bpm(audio)
        rotation_cpm = bpm_to_premium_rotation_cpm(bpm)
        _set_job(job_id, message="Rendering 8D orbit…", bpm=round(bpm, 1), rotation_cpm=round(rotation_cpm, 2))
        rendered = process_8d(
            audio,
            rotation_cpm=rotation_cpm,
            room_size=0.18,
            crossover_hz=150.0,
            motion_depth=0.78,
            high_emphasis=0.65,
            spatial_mix=0.68,
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
        )
    except Exception as exc:
        _set_job(job_id, status="failed", message="Render failed.", error=str(exc))

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio").suffix.lower() or ".audio"
    safe_stem = Path(file.filename or "audio").stem.replace("/", "_").replace("\\", "_")[:80]
    job_id = uuid.uuid4().hex[:12]
    src = APP_DIR / f"{safe_stem}_{job_id}{suffix}"
    out = APP_DIR / f"{safe_stem}_{job_id}_8D_Final.wav"
    with src.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    _set_job(job_id, status="queued", message="Upload complete. Waiting for DSP worker…", input_name=file.filename, output_name=out.name)
    EXECUTOR.submit(_process_job, job_id, src, out)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing", "message": "Upload accepted. DSP render started."})

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown render job")
        return dict(job, job_id=job_id)
