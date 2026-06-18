"""Guards the streaming render path (render_8d_to_wav / render_8d_file_to_wav),
which the in-memory process_8d tests don't cover. A regression here once shipped
a NameError that crashed every classic render."""
import numpy as np
import soundfile as sf

from eightd_engine.dsp import render_8d_to_wav, render_8d_file_to_wav, AudioData


def _signal(sr=44100, seconds=3.0):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    left = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 55 * t)
    right = 0.5 * np.sin(2 * np.pi * 221 * t) + 0.2 * np.sin(2 * np.pi * 55 * t)
    return np.column_stack([left, right]).astype(np.float32), sr


def test_render_8d_to_wav_produces_output_and_valid_report(tmp_path):
    x, sr = _signal()
    out = tmp_path / "o.wav"
    rep = render_8d_to_wav(AudioData(x, sr), out, rotation_cpm=5.78,
                           preserve_quality=True, panning_preset="binaural_8d", section_automation=True)
    assert out.exists()
    a, _ = sf.read(str(out), dtype="float32")
    assert a.shape[0] == len(x) and a.shape[1] == 2
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0
    assert -1.0 <= rep.correlation <= 1.0
    assert rep.side_mid_ratio >= 0.0


def test_render_8d_file_to_wav_matches_in_memory(tmp_path):
    x, sr = _signal()
    src = tmp_path / "in.wav"
    sf.write(str(src), x, sr, subtype="FLOAT")
    mem = tmp_path / "mem.wav"
    fil = tmp_path / "fil.wav"
    params = dict(rotation_cpm=5.78, preserve_quality=True, panning_preset="binaural_8d", section_automation=True)
    render_8d_to_wav(AudioData(x.copy(), sr), mem, **params)
    render_8d_file_to_wav(src, fil, **params)
    a, _ = sf.read(str(mem), dtype="float32")
    b, _ = sf.read(str(fil), dtype="float32")
    assert a.shape == b.shape
    assert float(np.max(np.abs(a - b))) < 1e-6
