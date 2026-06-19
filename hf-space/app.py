"""Hugging Face Space: Demucs stem separation API for the 8D Engine.

Exposes a named endpoint `/separate` that takes one audio file and returns the
four Demucs stems (vocals, drums, bass, other) as downloadable files. Runs on the
free CPU tier (16 GB RAM is plenty for htdemucs).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
ORDER = ["vocals", "drums", "bass", "other"]


def separate(audio_path):
    if not audio_path:
        raise gr.Error("No audio file was provided.")
    out_root = Path(tempfile.mkdtemp())
    # Demucs CLI. --segment keeps peak RAM modest; htdemucs (a Transformer model)
    # caps the segment at 7.8 s, so 7 is the largest safe value. -j runs the
    # segments across all CPU cores, which roughly halves wall-time on the free
    # 2-vCPU tier and keeps full-length songs from timing out the request.
    jobs = max(1, os.cpu_count() or 1)
    cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "--segment", "7",
           "-j", str(jobs), "-o", str(out_root), str(audio_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise gr.Error((proc.stderr or proc.stdout or "Demucs failed").strip()[-800:])

    stem_root = out_root / MODEL
    subdirs = [d for d in stem_root.iterdir() if d.is_dir()] if stem_root.exists() else []
    if not subdirs:
        raise gr.Error("Demucs produced no output.")
    stem_dir = subdirs[0]
    return [str(stem_dir / f"{name}.wav") if (stem_dir / f"{name}.wav").exists() else None for name in ORDER]


with gr.Blocks(title="8D Engine — Demucs stem separation") as demo:
    gr.Markdown("## 8D Engine — Demucs stem separation\nUpload a track; get vocals / drums / bass / other.")
    inp = gr.Audio(type="filepath", label="Input track")
    btn = gr.Button("Separate stems", variant="primary")
    outs = [gr.File(label=name) for name in ORDER]
    btn.click(separate, inputs=inp, outputs=outs, api_name="separate")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
