import numpy as np

from eightd_engine.audio_io import export_audio, load_audio
from eightd_engine.dsp import (
    AudioData,
    binaural_orbit,
    bpm_to_premium_rotation_cpm,
    fibonacci_preset_names,
    high_frequency_emphasis,
    panning_preset_names,
    preserve_loudness_and_peak,
    process_8d,
    reduce_static_noise,
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


def test_static_noise_reduction_lowers_hiss_without_erasing_music():
    sr = 44100
    rng = np.random.default_rng(123)
    music = np.column_stack([
        sine(440, sr=sr, seconds=2.0, amp=0.22),
        sine(660, sr=sr, seconds=2.0, amp=0.20),
    ])
    hiss = rng.normal(0.0, 0.018, size=music.shape)
    noisy = music + hiss

    cleaned = reduce_static_noise(noisy, sr, amount=0.9)

    before_noise = rms_level(noisy - music)
    after_noise = rms_level(cleaned - music)
    music_retention = rms_level(cleaned) / (rms_level(music) + 1e-12)

    assert after_noise < before_noise * 0.72
    assert 0.72 <= music_retention <= 1.05


def test_premium_panning_presets_are_available_and_distinct():
    names = panning_preset_names()
    assert {"fireflies_plus", "cinematic_halo", "figure8", "wide_orbit", "vocal_safe", "reference_luxe"}.issubset(names)

    sr = 44100
    source = np.column_stack([sine(1300, sr=sr, seconds=2.0), sine(1300, sr=sr, seconds=2.0)])
    fireflies = process_8d(AudioData(source, sr), panning_preset="fireflies_plus", room_size=0.0).samples
    figure8 = process_8d(AudioData(source, sr), panning_preset="figure8", room_size=0.0).samples
    reference = process_8d(AudioData(source, sr), panning_preset="reference_luxe", room_size=0.0).samples

    assert fireflies.shape == figure8.shape == reference.shape == source.shape
    assert rms_level(fireflies - figure8) > 1e-3
    assert rms_level(reference - fireflies) > 1e-3


def test_reference_luxe_mix_keeps_reference_style_width_with_safe_bass():
    sr = 44100
    seconds = 8.0
    bass = sine(58, sr=sr, seconds=seconds, amp=0.42)
    vocal = sine(920, sr=sr, seconds=seconds, amp=0.16)
    shimmer_l = sine(6400, sr=sr, seconds=seconds, amp=0.05)
    shimmer_r = sine(7100, sr=sr, seconds=seconds, amp=0.045)
    source = np.column_stack([bass + vocal + shimmer_l, bass + vocal + shimmer_r])

    rendered = process_8d(
        AudioData(source, sr),
        rotation_cpm=5.78,
        room_size=0.22,
        crossover_hz=150,
        motion_depth=0.86,
        high_emphasis=0.72,
        spatial_mix=0.74,
        panning_preset="reference_luxe",
        preserve_quality=True,
        section_automation=True,
    ).samples

    bass_band, motion_band = split_bass_motion(rendered, sr, crossover_hz=150)
    low_side = rms_level(bass_band[:, 0] - bass_band[:, 1])
    low_mid = rms_level(bass_band.mean(axis=1))
    motion_side = rms_level((motion_band[:, 0] - motion_band[:, 1]) * 0.5)
    motion_mid = rms_level((motion_band[:, 0] + motion_band[:, 1]) * 0.5)

    assert low_side < low_mid * 0.05
    assert 0.20 <= motion_side / (motion_mid + 1e-12) <= 1.20


def test_fibonacci_golden_ratio_preset_suite_is_registered():
    names = panning_preset_names()
    fib_names = fibonacci_preset_names()

    assert fib_names == {
        "phi_reference_orbit",
        "fibonacci_spiral",
        "golden_figure8",
        "lucas_breath",
    }
    assert fib_names.issubset(names)


def test_fibonacci_presets_are_distinct_reference_width_and_bass_safe():
    sr = 44100
    seconds = 8.0
    bass = sine(62, sr=sr, seconds=seconds, amp=0.38)
    vocal = sine(880, sr=sr, seconds=seconds, amp=0.15)
    air_l = sine(6200, sr=sr, seconds=seconds, amp=0.055)
    air_r = sine(7600, sr=sr, seconds=seconds, amp=0.045)
    source = np.column_stack([bass + vocal + air_l, bass + vocal + air_r])

    renders = {}
    for preset in fibonacci_preset_names():
        rendered = process_8d(
            AudioData(source, sr),
            rotation_cpm=5.78,
            room_size=0.22,
            crossover_hz=150,
            motion_depth=0.84,
            high_emphasis=0.72,
            spatial_mix=0.74,
            panning_preset=preset,
            preserve_quality=True,
            section_automation=True,
        ).samples
        bass_band, motion_band = split_bass_motion(rendered, sr, crossover_hz=150)
        low_side = rms_level(bass_band[:, 0] - bass_band[:, 1])
        low_mid = rms_level(bass_band.mean(axis=1))
        side = rms_level((motion_band[:, 0] - motion_band[:, 1]) * 0.5)
        mid = rms_level((motion_band[:, 0] + motion_band[:, 1]) * 0.5)

        assert rendered.shape == source.shape
        assert np.max(np.abs(rendered)) <= 10 ** (-1.0 / 20.0) + 1e-12
        assert low_side < low_mid * 0.05
        assert 0.18 <= side / (mid + 1e-12) <= 1.25
        renders[preset] = rendered

    assert rms_level(renders["phi_reference_orbit"] - renders["fibonacci_spiral"]) > 1e-3
    assert rms_level(renders["golden_figure8"] - renders["lucas_breath"]) > 1e-3


def test_binaural_orbit_is_continuous_at_processing_boundaries():
    sr = 44100
    seconds = 3.0
    source = np.column_stack([sine(1200, sr=sr, seconds=seconds), sine(1200, sr=sr, seconds=seconds)])

    rendered = binaural_orbit(source, sr, rotation_cpm=5.78, motion_depth=0.9, panning_preset="reference_luxe")
    diff = np.abs(np.diff(rendered[:, 0])) + np.abs(np.diff(rendered[:, 1]))
    boundaries = np.arange(1024, len(rendered) - 1024, 1024)
    boundary_jumps = diff[boundaries]
    typical_jump = np.percentile(diff, 95) + 1e-12

    # Regression guard for the old block-local delay implementation: it reset
    # fractional-delay buffers every 1024 samples, creating zipper/static ticks
    # that followed pan movement. Boundaries should now look like ordinary audio.
    assert np.max(boundary_jumps) < typical_jump * 2.5


def test_center_focus_keeps_vocal_band_more_front_center_while_air_still_moves():
    sr = 44100
    seconds = 5.0
    bass = sine(70, sr=sr, seconds=seconds, amp=0.35)
    vocal_body = sine(950, sr=sr, seconds=seconds, amp=0.20)
    air = sine(7200, sr=sr, seconds=seconds, amp=0.06)
    source = np.column_stack([bass + vocal_body + air, bass + vocal_body + air])

    no_focus = process_8d(
        AudioData(source, sr),
        rotation_cpm=5.78,
        room_size=0.0,
        motion_depth=0.78,
        high_emphasis=0.65,
        spatial_mix=0.68,
        panning_preset="reference_luxe",
        preserve_quality=True,
        center_focus=0.0,
    ).samples
    focused = process_8d(
        AudioData(source, sr),
        rotation_cpm=5.78,
        room_size=0.0,
        motion_depth=0.72,
        high_emphasis=0.68,
        spatial_mix=0.62,
        panning_preset="reference_luxe",
        preserve_quality=True,
        center_focus=0.75,
    ).samples

    _bass_nf, moving_band_nf = split_bass_motion(no_focus, sr, crossover_hz=150)
    _bass_f, moving_band_f = split_bass_motion(focused, sr, crossover_hz=150)

    side_nf = rms_level((moving_band_nf[:, 0] - moving_band_nf[:, 1]) * 0.5)
    mid_nf = rms_level((moving_band_nf[:, 0] + moving_band_nf[:, 1]) * 0.5)
    side_f = rms_level((moving_band_f[:, 0] - moving_band_f[:, 1]) * 0.5)
    mid_f = rms_level((moving_band_f[:, 0] + moving_band_f[:, 1]) * 0.5)

    assert side_f / (mid_f + 1e-12) < (side_nf / (mid_nf + 1e-12)) * 0.55

    # The high band still has spatial difference energy, so the result does not
    # collapse to plain mono; the movement shifts to air/ambience as requested.
    _, air_band = split_bass_motion(focused, sr, crossover_hz=4000)
    air_side = rms_level((air_band[:, 0] - air_band[:, 1]) * 0.5)
    air_mid = rms_level((air_band[:, 0] + air_band[:, 1]) * 0.5)
    assert air_side / (air_mid + 1e-12) > 0.08
