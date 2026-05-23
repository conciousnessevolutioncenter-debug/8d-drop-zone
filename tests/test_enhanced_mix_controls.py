import numpy as np

from eightd_engine.dsp import (
    AudioData,
    analyze_correlation,
    apply_youtube_master_target,
    high_frequency_emphasis,
    process_8d,
    section_automation_curve,
)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-18))


def sine(freq, sr=44100, seconds=4.0, amp=0.4):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def side_mid_ratio(stereo):
    mid = stereo.mean(axis=1)
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    return rms(side) / (rms(mid) + 1e-12)


def test_motion_depth_controls_spatial_intensity():
    sr = 44100
    source = np.column_stack([sine(1200, sr=sr), sine(1200, sr=sr)])

    shallow = process_8d(AudioData(source, sr), motion_depth=0.2, room_size=0.0).samples
    deep = process_8d(AudioData(source, sr), motion_depth=1.0, room_size=0.0).samples

    assert side_mid_ratio(deep) > side_mid_ratio(shallow) * 1.8


def test_high_frequency_emphasis_makes_highs_move_more_than_low_mids():
    sr = 44100
    low_mid = sine(300, sr=sr, amp=0.3)
    high = sine(6000, sr=sr, amp=0.3)
    source = np.column_stack([low_mid + high, low_mid + high])

    emphasized = high_frequency_emphasis(source, sr, amount=1.0, split_hz=4000)
    base_high = sine(6000, sr=sr, amp=0.3)
    base_low = sine(300, sr=sr, amp=0.3)

    assert rms(emphasized.mean(axis=1) - base_low) > rms(source.mean(axis=1) - base_low)
    assert rms(emphasized.mean(axis=1) - base_high) < rms(source.mean(axis=1) - base_high) * 1.2


def test_youtube_master_target_hits_safe_peak_and_reasonable_loudness():
    sr = 44100
    quiet = np.column_stack([sine(1000, sr=sr, amp=0.03), sine(1000, sr=sr, amp=0.03)])

    mastered = apply_youtube_master_target(quiet, target_rms_db=-13.0, peak_ceiling_db=-1.0)

    peak_db = 20 * np.log10(np.max(np.abs(mastered)) + 1e-12)
    rms_db = 20 * np.log10(rms(mastered) + 1e-12)
    assert peak_db <= -0.99
    assert -16.5 <= rms_db <= -12.0


def test_correlation_meter_reports_phase_health():
    mono = np.ones((1000, 2)) * 0.1
    anti = np.column_stack([np.ones(1000), -np.ones(1000)]) * 0.1

    assert analyze_correlation(mono).correlation > 0.99
    assert analyze_correlation(anti).correlation < -0.99
    assert analyze_correlation(anti).phase_warning is True


def test_section_automation_intro_chorus_outro_shape():
    sr = 100
    n = sr * 100
    curve = section_automation_curve(n, sr)

    intro = curve[: 10 * sr]
    chorus = curve[45 * sr : 65 * sr]
    outro = curve[90 * sr :]

    assert float(np.mean(intro)) < 0.75
    assert float(np.mean(chorus)) > 0.95
    assert float(np.mean(outro)) > float(np.mean(intro))
