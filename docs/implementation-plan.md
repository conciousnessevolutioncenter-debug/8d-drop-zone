# The 8D Engine Implementation Plan

> **For Hermes:** Implement locally with strict TDD for DSP behavior, then GUI/documentation integration.

**Goal:** Build a desktop Python app that imports stereo audio and exports headphone-first 8D/binaural-style audio with stable mono bass.

**Architecture:** Keep DSP code isolated in `eightd_engine/dsp.py`, file I/O in `eightd_engine/audio_io.py`, and the Tkinter app in `eightd_engine/gui.py`. The DSP path splits bass and motion bands, keeps bass mono/static, applies azimuth-driven equal-power panning plus ITD delay to the motion band, adds lightweight room reverb, then recombines and peak-normalizes.

**Tech Stack:** Python 3.10+, NumPy, SoundFile, optional Pydub/FFmpeg for MP3, Tkinter, PyInstaller.

---

## Tasks

1. Create package skeleton and failing DSP tests.
2. Implement DSP core: format normalization, crossover, equal-power gains, fractional delay, orbit processing, feedback-comb reverb.
3. Implement audio I/O: WAV/FLAC/OGG through SoundFile, MP3 via Pydub fallback.
4. Implement Tkinter GUI: import, sliders, export, background processing thread.
5. Add README, requirements, and PyInstaller instructions.
6. Verify with pytest and a generated sine-wave smoke export.

## Acceptance Criteria

- Bass below 150 Hz is mono and centered after processing.
- Mid/high content rotates around the listener with smooth azimuth automation.
- ITD uses sub-millisecond ear delay based on azimuth.
- Output is stereo, same length/sample-rate as input, peak-safe.
- GUI can import and export files without freezing.
