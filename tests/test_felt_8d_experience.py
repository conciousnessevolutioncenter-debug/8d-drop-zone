import numpy as np

from eightd_engine.dsp import AudioData, process_8d, rms_level, split_bass_motion


def sine(freq, sr=44100, seconds=4.0, amp=0.25):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_felt_presence_increases_perceptible_air_motion_without_moving_bass():
    sr = 44100
    seconds = 6.0
    bass = sine(64, sr=sr, seconds=seconds, amp=0.36)
    vocal_body = sine(850, sr=sr, seconds=seconds, amp=0.16)
    guitar_air_l = sine(5200, sr=sr, seconds=seconds, amp=0.055)
    guitar_air_r = sine(6900, sr=sr, seconds=seconds, amp=0.050)
    source = np.column_stack([bass + vocal_body + guitar_air_l, bass + vocal_body + guitar_air_r])

    base = process_8d(
        AudioData(source, sr),
        rotation_cpm=5.78,
        room_size=0.18,
        motion_depth=0.70,
        high_emphasis=0.58,
        spatial_mix=0.60,
        panning_preset="reference_luxe",
        preserve_quality=True,
        section_automation=True,
        center_focus=0.72,
        felt_presence=0.0,
    ).samples
    felt = process_8d(
        AudioData(source, sr),
        rotation_cpm=5.78,
        room_size=0.22,
        motion_depth=0.74,
        high_emphasis=0.68,
        spatial_mix=0.64,
        panning_preset="reference_luxe",
        preserve_quality=True,
        section_automation=True,
        center_focus=0.72,
        felt_presence=0.85,
    ).samples

    base_bass, base_motion = split_bass_motion(base, sr, crossover_hz=150)
    felt_bass, felt_motion = split_bass_motion(felt, sr, crossover_hz=150)
    _, base_air = split_bass_motion(base_motion, sr, crossover_hz=4000)
    _, felt_air = split_bass_motion(felt_motion, sr, crossover_hz=4000)

    base_air_side = rms_level((base_air[:, 0] - base_air[:, 1]) * 0.5)
    felt_air_side = rms_level((felt_air[:, 0] - felt_air[:, 1]) * 0.5)
    felt_low_side = rms_level(felt_bass[:, 0] - felt_bass[:, 1])
    felt_low_mid = rms_level(felt_bass.mean(axis=1))

    assert felt_air_side > base_air_side * 1.12
    assert felt_low_side < felt_low_mid * 0.05
    assert np.max(np.abs(felt)) <= 10 ** (-1.0 / 20.0) + 1e-12


def test_felt_presence_keeps_loudness_lift_musical_not_hyped():
    sr = 44100
    seconds = 4.0
    source = np.column_stack([
        sine(70, sr=sr, seconds=seconds, amp=0.30) + sine(1200, sr=sr, seconds=seconds, amp=0.17) + sine(7600, sr=sr, seconds=seconds, amp=0.04),
        sine(70, sr=sr, seconds=seconds, amp=0.30) + sine(1250, sr=sr, seconds=seconds, amp=0.16) + sine(6800, sr=sr, seconds=seconds, amp=0.04),
    ])

    rendered = process_8d(
        AudioData(source, sr),
        rotation_cpm=5.78,
        room_size=0.24,
        motion_depth=0.78,
        high_emphasis=0.72,
        spatial_mix=0.66,
        panning_preset="phi_reference_orbit",
        preserve_quality=True,
        section_automation=True,
        center_focus=0.70,
        felt_presence=1.0,
    ).samples

    lift_db = 20 * np.log10((rms_level(rendered) + 1e-12) / (rms_level(source) + 1e-12))

    assert lift_db <= 0.45
    assert rms_level(rendered) > 0.0
