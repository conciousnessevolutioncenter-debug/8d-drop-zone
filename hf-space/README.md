---
title: 8D Demucs Stems
emoji: 🎚️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 8D Engine — Demucs stem separation

A tiny free-tier (CPU, 16 GB RAM) Hugging Face Space that separates a track into
**vocals / drums / bass / other** with [Demucs](https://github.com/facebookresearch/demucs),
so the 8D Engine app can offer downloadable stems without running PyTorch on its
own small server.

The 8D app calls the named API endpoint **`/separate`** via `gradio_client`:

```python
from gradio_client import Client, handle_file
client = Client("YOUR_USERNAME/8d-demucs-stems")
vocals, drums, bass, other = client.predict(handle_file("song.mp3"), api_name="/separate")
```

Free CPU Spaces sleep when idle; the first request after a nap has a cold-start
delay (~1–2 min) while the model loads, then subsequent requests are faster.
