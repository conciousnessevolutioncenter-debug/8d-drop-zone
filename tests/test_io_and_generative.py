import numpy as np

from eightd_engine.audio_io import export_audio, load_audio
from eightd_engine.dsp import AudioData
from eightd_engine.generative import GenerativeRuleSet, generate_melodic_degrees, tempo_sync_period_ms


def test_wav_export_roundtrip_preserves_stereo_shape(tmp_path):
    sr = 22050
    t = np.arange(sr // 10) / sr
    samples = np.column_stack([
        0.2 * np.sin(2 * np.pi * 440 * t),
        0.2 * np.sin(2 * np.pi * 660 * t),
    ])
    path = tmp_path / "roundtrip.wav"

    export_audio(AudioData(samples=samples, sample_rate=sr), path)
    loaded = load_audio(path)

    assert loaded.sample_rate == sr
    assert loaded.samples.shape == samples.shape
    assert np.max(np.abs(loaded.samples)) <= 1.0


def test_generative_melody_is_seeded_and_scale_constrained():
    rules = GenerativeRuleSet(root_midi=57, scale_intervals=(0, 2, 4, 7, 9), octave_span=2)

    a = generate_melodic_degrees(16, rules, seed=123)
    b = generate_melodic_degrees(16, rules, seed=123)

    assert a == b
    allowed = {57 + 12 * octave + interval for octave in range(2) for interval in rules.scale_intervals}
    assert all(note is None or note in allowed for note in a)


def test_tempo_sync_period_uses_bars_and_bpm():
    assert tempo_sync_period_ms(120, bars=1) == 2000.0
    assert tempo_sync_period_ms(120, bars=4) == 8000.0
