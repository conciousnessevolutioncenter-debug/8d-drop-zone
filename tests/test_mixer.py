"""Multitrack mixer page + endpoints (P2)."""
import importlib.util
import sys
from pathlib import Path


def load_live_module():
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_mixer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mixer_routes_are_registered():
    module = load_live_module()
    paths = {getattr(r, "path", "") for r in module.app.routes}
    assert "/mixer" in paths
    assert "/mixer/separate" in paths


def test_mixer_page_has_web_audio_mixer_and_controls():
    module = load_live_module()
    html = module.MIXER_HTML

    # The page wires a per-stem Web Audio mixer with the core controls.
    assert "function buildMixer" in html
    assert "createStereoPanner" in html        # per-stem pan
    assert "createBiquadFilter" in html         # 3-band EQ
    assert "OfflineAudioContext" in html        # offline mixdown render
    assert "audioBufferToWav" in html           # WAV export
    # Mute/solo + the two output actions.
    assert ">M<" in html and ">S<" in html
    assert "Download mixdown" in html
    assert "Send mix to the 8D engine" in html


def test_mixer_is_a_16_channel_console():
    module = load_live_module()
    html = module.MIXER_HTML

    # Fixed 16-channel board; stems fill the first channels and the rest are
    # loadable slots for the user's own tracks.
    assert "CHANNEL_COUNT = 16" in html
    assert "Load track" in html
    assert "function loadIntoChannel" in html
    assert "16-channel console" in html


def test_homepage_links_to_the_mixer():
    module = load_live_module()
    assert 'href="/mixer"' in module.HTML


def test_separation_is_resilient_and_has_its_own_executor():
    import inspect
    module = load_live_module()

    # Dedicated pool so a long separation never blocks the 8D render queue.
    assert hasattr(module, "STEM_EXECUTOR")
    assert module.STEM_EXECUTOR is not module.EXECUTOR

    # The HF Space call takes a status callback and retries cold-starts.
    sig = inspect.signature(module._separate_stems_hf_space)
    assert "on_status" in sig.parameters
    src = inspect.getsource(module._separate_stems_hf_space)
    assert "range(3)" in src  # retry loop for a sleeping/cold worker


def test_mixer_page_has_progress_ui():
    module = load_live_module()
    html = module.MIXER_HTML
    assert 'id="progress"' in html
    assert "function startProgress" in html
    assert "function failProgress" in html
    assert 'id="progRetry"' in html       # graceful retry on failure
