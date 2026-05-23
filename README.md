# 8D Drop-Zone

A dark, drag-and-drop Python desktop app for converting stereo audio into headphone-first professional 8D audio.

The workflow is intentionally one action:

```text
Drop MP3/WAV/audio file → automatic BPM analysis → 8D render → *_8D_Final.wav saved beside source
```

## What it does

- Native drag-and-drop UI using `tkinterdnd2`.
- Visual hover feedback: the drop card changes state to **Release to Convert**.
- Automatic export to the same folder as the source:
  - `Song.wav` → `Song_8D_Final.wav`
- Smart analysis pipeline:
  - **BPM Lock:** detects tempo with `librosa` and sets rotation so **1 orbit = 2 bars**.
  - **Bass Preservation:** isolates the low-end below **150 Hz**, folds it to mono, and keeps it centered/static.
  - **Spatial Rotation:** spins only the mid/high motion band using an HRTF-inspired binaural orbit.
  - **ITD + ILD:** combines equal-power level panning with sub-millisecond interaural delay.
  - **Rear Shading:** darkens and lowers rear positions slightly so the orbit feels more spherical.
  - **Room Integration:** adds subtle early reflections to externalize the sound.
  - **High-Frequency Motion Emphasis:** makes air/reverb/highs carry more motion than low mids.
  - **YouTube-safe master:** targets roughly `-13 dB RMS` with peak limiting around `-1 dB`.
  - **Correlation report:** checks stereo phase health after rendering.

## Install

Recommended in a virtual environment:

```bash
cd the-8d-engine
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

On Windows:

```powershell
cd the-8d-engine
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Specific drag-and-drop dependency:

```bash
pip install tkinterdnd2
```

For MP3/M4A support, install FFmpeg and make sure `ffmpeg` is on PATH.

## Run the app

```bash
python main.py
```

Then drag an audio file directly onto the drop zone.

## DSP signal flow

```text
stereo input
  ↓
smart BPM estimate via librosa
  ↓
rotation speed = BPM / 8  # 4/4 two-bar orbit
  ↓
150 Hz crossover
  ├─ low band → mono/static center foundation
  └─ mid/high band → high emphasis → binaural orbit
         ↓
         ILD equal-power gains + ITD fractional delay + rear shading
         ↓
         subtle room reflections
  ↓
recombine bass + motion + room
  ↓
YouTube-safe master target
  ↓
correlation analysis
  ↓
*_8D_Final.wav
```

## Run tests

Use module mode so Python includes the project root on the import path:

```bash
python -m pytest tests -q
```

## Build a standalone Windows .exe

On Windows, from the project directory:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
pyinstaller --onefile --windowed --name "8D Drop-Zone" main.py
```

The executable will appear under:

```text
dist/8D Drop-Zone.exe
```

If you need MP3/M4A import/export in the packaged app, bundle FFmpeg or instruct users to install FFmpeg separately.

## Project structure

```text
eightd_engine/
  dsp.py          # bass management, BPM helpers, binaural orbit, reverb, mastering
  audio_io.py     # import/export helpers
  generative.py   # helmet-album/generative music primitives retained for future expansion
  gui.py          # dark TkinterDnD2 drop-zone app
main.py           # app entrypoint
tests/            # unit tests
docs/             # implementation notes
```

## Important note

This is an HRTF-inspired renderer, not personalized measured-HRTF convolution. It creates the 8D illusion with practical psychoacoustic cues: ITD, ILD, rear filtering, frequency-dependent motion, and room reflections.
