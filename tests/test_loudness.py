"""LUFS + true-peak metering (BS.1770 K-weighted, streamed)."""
import numpy as np
import soundfile as sf

from eightd_engine.dsp import measure_loudness_file


def _write(tmp_path, amp, name="t.wav", sr=44100, seconds=3.0):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    tone = (amp * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    x = np.column_stack([tone, tone])
    p = tmp_path / name
    sf.write(str(p), x, sr, subtype="FLOAT")
    return p


def test_loudness_is_finite_and_in_range(tmp_path):
    lufs, tp = measure_loudness_file(_write(tmp_path, 0.5))
    assert lufs is not None and tp is not None
    assert -40.0 < lufs < 0.0          # a half-scale 1k tone sits in this range
    assert -20.0 < tp <= 1.0           # true peak near the tone's level


def test_louder_signal_reads_higher_lufs(tmp_path):
    quiet, _ = measure_loudness_file(_write(tmp_path, 0.1, "q.wav"))
    loud, _ = measure_loudness_file(_write(tmp_path, 0.6, "l.wav"))
    assert loud > quiet + 5.0          # ~+15 dB amplitude -> clearly louder LUFS
