import importlib.util
import sys
from pathlib import Path


def load_live_module():
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_mix_prompt_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mix_prompt_keeps_vocal_center_and_reduces_global_motion():
    module = load_live_module()
    base = dict(room_size=0.20, motion_depth=0.72, high_emphasis=0.68, spatial_mix=0.62, center_focus=0.72, denoise_amount=0.72)

    result = module.apply_mix_instructions(base, "Keep the main vocal front and center without motion; keep drums static too")

    assert result.settings["center_focus"] >= 0.90
    assert result.settings["motion_depth"] <= 0.56
    assert result.settings["spatial_mix"] <= 0.52
    assert any("Lead vocal" in note for note in result.notes)
    assert any("Low end" in note for note in result.notes)


def test_mix_prompt_can_push_guitar_ambience_wider_without_unprotecting_bass():
    module = load_live_module()
    base = dict(room_size=0.18, motion_depth=0.64, high_emphasis=0.60, spatial_mix=0.58, center_focus=0.74, denoise_amount=0.72)

    result = module.apply_mix_instructions(base, "Make the guitars and echo delays wider and more spatial, bigger room")

    assert result.settings["high_emphasis"] >= 0.78
    assert result.settings["spatial_mix"] >= 0.66
    assert result.settings["room_size"] >= 0.30
    assert result.settings["center_focus"] == base["center_focus"]
    assert any("Guitar" in note for note in result.notes)
    assert any("Room" in note for note in result.notes)


def test_mix_prompt_supports_drier_subtle_renders_and_clamps_values():
    module = load_live_module()
    base = dict(room_size=0.20, motion_depth=0.72, high_emphasis=0.68, spatial_mix=0.62, center_focus=0.72, denoise_amount=0.72)

    result = module.apply_mix_instructions(base, "Make it subtle, less movement, dry and remove static noise")

    assert result.settings["motion_depth"] <= 0.55
    assert result.settings["spatial_mix"] <= 0.52
    assert result.settings["room_size"] <= 0.10
    assert result.settings["denoise_amount"] >= 0.82
    assert all(0.0 <= result.settings[k] <= 1.0 for k in ["room_size", "motion_depth", "high_emphasis", "spatial_mix", "center_focus", "denoise_amount"])


def test_mix_prompt_supports_felt_physical_immersive_renders():
    module = load_live_module()
    base = dict(room_size=0.20, motion_depth=0.72, high_emphasis=0.68, spatial_mix=0.62, center_focus=0.72, denoise_amount=0.72, felt_presence=0.62)

    result = module.apply_mix_instructions(base, "Make the 8D felt by the listener: physical, immersive, powerful, but keep bass centered")

    assert result.settings["felt_presence"] >= 0.88
    assert result.settings["high_emphasis"] >= 0.76
    assert result.settings["motion_depth"] >= 0.80
    assert result.settings["spatial_mix"] >= 0.70
    assert any("felt" in note.lower() or "presence" in note.lower() for note in result.notes)
    assert any("Low end" in note for note in result.notes)
