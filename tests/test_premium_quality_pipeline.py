import numpy as np

from eightd_engine.audio_io import export_audio, load_audio
from eightd_engine.dsp import (
    AudioData,
    bpm_to_premium_rotation_cpm,
    high_frequency_emphasis,
    preserve_loudness_and_peak,
    process_8d,
    rms_level,
    split_bass_motion,
)


def sine(freq, sr=44100, seconds=3.0, amp=0.25):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_premium_rotation_is_smooth_not_dizzying():
    # At common pop tempos, premium mode should orbit every four bars,
    # not every two bars. 120 BPM => 7.5 cycles/min => 8 seconds/orbit.
    assert bpm_to_premium_rotation_cpm(120.0) == 7.5
    assert 6.0 <= bpm_to_premium_rotation_cpm(96.0) <= 7.0


def test_preserve_loudness_guard_prevents_unnecessary_gain_and_clipping():
    sr = 44100
    reference = np.column_stack([sine(440, sr=sr, amp=0.4), sine(660, sr=sr, amp=0.35)])
    processed_too_hot = reference * 1.8

    guarded = preserve_loudness_and_peak(processed_too_hot, reference, peak_ceiling_db=-1.0, max_rms_lift_db=0.25)

    assert np.max(np.abs(guarded)) <= 10 ** (-1.0 / 20.0) + 1e-12
    lift_db = 20 * np.log10((rms_level(guarded) + 1e-12) / (rms_level(reference) + 1e-12))
    assert lift_db <= 0.25 + 1e-6


def test_premium_process_preserves_core_tone_and_clean_safe_headroom():
    sr = 44100
    seconds = 4.0
    bass = sine(65, sr=sr, seconds=seconds, amp=0.45)
    vocal = sine(900, sr=sr, seconds=seconds, amp=0.18)
    air_l = sine(6400, sr=sr, seconds=seconds, amp=0.06)
    air_r = sine(6900, sr=sr, seconds=seconds, amp=0.055)
    original = np.column_stack([bass + vocal + air_l, bass + vocal + air_r])

    rendered = process_8d(
        AudioData(original, sr),
        rotation_cpm=bpm_to_premium_rotation_cpm(120),
        room_size=0.18,
        crossover_hz=150,
        motion_depth=0.78,
        high_emphasis=0.25,
        spatial_mix=0.68,
        preserve_quality=True,
        youtube_master=False,
        section_automation=True,
    ).samples

    assert rendered.shape == original.shape
    assert np.all(np.isfinite(rendered))
    assert np.max(np.abs(rendered)) <= 10 ** (-1.0 / 20.0) + 1e-12

    # Core mid/vocal energy should stay close to the original rather than being
    # hollowed out by the effect.
    _bass_orig, motion_orig = split_bass_motion(original, sr, crossover_hz=150)
    _bass_rendered, motion_rendered = split_bass_motion(rendered, sr, crossover_hz=150)
    motion_ratio_db = 20 * np.log10((rms_level(motion_rendered) + 1e-12) / (rms_level(motion_orig) + 1e-12))
    assert -3.5 <= motion_ratio_db <= 1.0

    # The low end remains centered and powerful.
    bass_rendered, _ = split_bass_motion(rendered, sr, crossover_hz=150)
    side_bass = bass_rendered[:, 0] - bass_rendered[:, 1]
    assert rms_level(side_bass) < rms_level(bass_rendered.mean(axis=1)) * 0.05


def test_wav_export_uses_high_resolution_float_for_detail_preservation(tmp_path):
    sr = 44100
    # Very low-level details survive float WAV export; 16-bit PCM quantization
    # would introduce errors around 1e-5 and fail this threshold.
    t = np.arange(sr // 10) / sr
    samples = np.column_stack([
        1e-5 * np.sin(2 * np.pi * 1234 * t),
        1e-5 * np.sin(2 * np.pi * 2345 * t),
    ])
    path = tmp_path / "detail.wav"

    export_audio(AudioData(samples=samples, sample_rate=sr), path)
    loaded = load_audio(path)

    assert loaded.sample_rate == sr
    assert np.max(np.abs(loaded.samples - samples)) < 1e-8


def test_high_frequency_focus_adds_air_cues_without_thinning_midrange_root():
    sr = 44100
    mid_root = sine(650, sr=sr, seconds=2.0, amp=0.28)
    air_detail = sine(7200, sr=sr, seconds=2.0, amp=0.035)
    source = np.column_stack([mid_root + air_detail, mid_root + air_detail])

    focused = high_frequency_emphasis(source, sr, amount=0.8, split_hz=4000)
    low_original, high_original = split_bass_motion(source, sr, crossover_hz=4000)
    low_focused, high_focused = split_bass_motion(focused, sr, crossover_hz=4000)

    low_change_db = 20 * np.log10((rms_level(low_focused) + 1e-12) / (rms_level(low_original) + 1e-12))
    high_change_db = 20 * np.log10((rms_level(high_focused) + 1e-12) / (rms_level(high_original) + 1e-12))

    # The root/body of a synth, vocal, or guitar should stay essentially intact.
    assert abs(low_change_db) < 0.35
    # The air/detail band gets extra cue energy for rear/height perception.
    assert high_change_db > 0.7
