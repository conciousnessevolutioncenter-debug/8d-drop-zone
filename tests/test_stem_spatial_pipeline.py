import numpy as np

from eightd_engine.dsp import AudioData
from eightd_engine.stems import (
    StemData,
    StemRole,
    available_stem_mode,
    build_stem_spatial_plan,
    process_stem_spatial_mix,
)


def sine(freq, sr=22050, seconds=1.0, amp=0.25):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def side_mid_ratio(stereo):
    mid = stereo.mean(axis=1)
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    return float(np.sqrt(np.mean(side * side) + 1e-18) / (np.sqrt(np.mean(mid * mid) + 1e-18)))


def test_stem_plan_assigns_musical_roles_without_spinning_whole_mix():
    plan = build_stem_spatial_plan({
        "vocals": StemData(np.zeros((100, 2)), 44100),
        "drums": StemData(np.zeros((100, 2)), 44100),
        "bass": StemData(np.zeros((100, 2)), 44100),
        "guitar": StemData(np.zeros((100, 2)), 44100),
        "other": StemData(np.zeros((100, 2)), 44100),
    })

    assert plan["vocals"].role is StemRole.VOCAL
    assert plan["vocals"].center_focus >= 0.9
    assert plan["vocals"].spatial_mix < plan["guitar"].spatial_mix
    assert plan["bass"].role is StemRole.BASS
    assert plan["bass"].spatial_mix == 0.0
    assert plan["drums"].role is StemRole.PERCUSSION
    assert plan["drums"].motion_depth < plan["guitar"].motion_depth
    assert plan["guitar"].room_size > plan["vocals"].room_size


def test_stem_spatial_mix_keeps_bass_mono_and_moves_instrument_space():
    sr = 22050
    bass = np.column_stack([sine(70, sr=sr), sine(70, sr=sr)])
    vocal = np.column_stack([sine(440, sr=sr, amp=0.18), sine(440, sr=sr, amp=0.18)])
    guitar = np.column_stack([sine(1320, sr=sr, amp=0.16), sine(1320, sr=sr, amp=0.16)])

    rendered = process_stem_spatial_mix(
        stems={
            "bass": StemData(bass, sr),
            "vocals": StemData(vocal, sr),
            "guitar": StemData(guitar, sr),
        },
        reference=AudioData(bass + vocal + guitar, sr),
        rotation_cpm=8.0,
        panning_preset="cinematic_halo",
    )

    assert rendered.sample_rate == sr
    assert rendered.samples.shape == bass.shape
    assert np.max(np.abs(rendered.samples)) <= 0.98
    # Low-frequency mono component stays strongly centered.
    assert side_mid_ratio(rendered.samples[: sr // 2]) < 0.65
    # The rendered stem mix has real stereo spatial information from non-bass stems.
    assert side_mid_ratio(rendered.samples) > 0.03


def test_available_stem_mode_reports_demucs_or_hybrid_fallback():
    mode = available_stem_mode()
    assert mode["mode"] in {"demucs", "hybrid_fallback"}
    assert isinstance(mode["message"], str)
    assert mode["message"]
