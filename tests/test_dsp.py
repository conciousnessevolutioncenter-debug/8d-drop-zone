import numpy as np

from eightd_engine.dsp import (
    AudioData,
    apply_fractional_delay,
    equal_power_gains,
    process_8d,
    split_bass_motion,
)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def sine(freq, sr=44100, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_equal_power_gains_keep_power_approximately_constant():
    for azimuth in [0, 45, 90, 180, 270, 359]:
        left, right = equal_power_gains(azimuth)
        assert np.isclose(left**2 + right**2, 1.0, atol=1e-9)
        assert left >= 0
        assert right >= 0


def test_fractional_delay_preserves_shape_and_delays_impulse():
    x = np.zeros(32, dtype=np.float64)
    x[4] = 1.0

    delayed = apply_fractional_delay(x, 3.0)

    assert delayed.shape == x.shape
    assert np.argmax(delayed) == 7
    assert np.isclose(delayed[7], 1.0, atol=1e-12)


def test_crossover_keeps_sub_bass_in_center_mono_and_places_highs_in_motion_band():
    sr = 44100
    low = sine(60, sr=sr, seconds=1.0, amp=0.8)
    high = sine(2000, sr=sr, seconds=1.0, amp=0.4)
    stereo = np.column_stack([low + high, low + high])

    bass, motion = split_bass_motion(stereo, sr, crossover_hz=150)

    # Low band should be mono/centered.
    assert bass.shape == stereo.shape
    assert np.max(np.abs(bass[:, 0] - bass[:, 1])) < 1e-10

    # Most 60 Hz energy belongs to bass, most 2 kHz energy belongs to motion.
    assert rms(bass.mean(axis=1)) > rms(motion.mean(axis=1))
    assert rms(motion[:, 0]) > 0.05


def test_process_8d_preserves_length_stereo_shape_and_static_low_end():
    sr = 44100
    seconds = 2.0
    low = sine(70, sr=sr, seconds=seconds, amp=0.55)
    high = sine(1500, sr=sr, seconds=seconds, amp=0.25)
    stereo = np.column_stack([low + high, low + high])

    result = process_8d(
        AudioData(samples=stereo, sample_rate=sr),
        rotation_cpm=6.0,
        room_size=0.1,
        crossover_hz=150,
    )

    assert result.sample_rate == sr
    assert result.samples.shape == stereo.shape
    assert np.max(np.abs(result.samples)) <= 1.0

    # Low-passed output should remain strongly correlated/centered despite moving highs.
    bass, _ = split_bass_motion(result.samples, sr, crossover_hz=150)
    side_bass = bass[:, 0] - bass[:, 1]
    mid_bass = bass.mean(axis=1)
    assert rms(side_bass) < rms(mid_bass) * 0.08
