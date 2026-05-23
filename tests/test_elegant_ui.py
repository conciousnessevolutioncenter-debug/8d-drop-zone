import importlib.util
import sys
from pathlib import Path


def load_live_module():
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_elegant_ui_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ui_uses_premium_product_language_not_test_wrapper_copy():
    module = load_live_module()
    html = module.HTML

    assert "The 8D Engine" in html
    assert "Spatial Audio Mastering" in html
    assert "Browser test wrapper" not in html
    assert "Drop Audio Here" not in html
    assert "Select Track" in html


def test_ui_exposes_reference_luxe_preset_and_signal_chain_badges():
    module = load_live_module()
    html = module.HTML

    assert 'value="reference_luxe"' in html
    assert 'value="phi_reference_orbit"' in html
    assert 'value="fibonacci_spiral"' in html
    assert 'value="golden_figure8"' in html
    assert 'value="lucas_breath"' in html
    assert "Reference Luxe" in html
    assert "Golden Ratio" in html
    assert "Fibonacci Spiral" in html
    assert "10.4s orbit" in html
    assert "Mono-safe bass" in html
    assert "Binaural orbit" in html
    assert "32-bit WAV" in html
