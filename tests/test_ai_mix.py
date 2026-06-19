"""AI mix co-producer: env-gating, tool schema, endpoint, mixer UI wiring."""
import importlib.util
import sys
from pathlib import Path


def load_web():
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_ai_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ai_mix_module_is_env_gated(monkeypatch):
    import ai_mix
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ai_mix.available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert ai_mix.available() is True


def test_tool_schema_shapes_the_directives():
    import ai_mix
    schema = ai_mix._TOOL["input_schema"]["properties"]
    assert ai_mix._TOOL["name"] == "set_mix"
    ch = schema["channels"]["items"]["properties"]
    for field in ("stem", "gain_db", "pan", "eq_low_db", "eq_mid_db", "eq_high_db", "mute"):
        assert field in ch
    assert "orbit" in schema and "notes" in schema


def test_ai_endpoint_registered_and_503_without_key(monkeypatch):
    from fastapi.testclient import TestClient
    module = load_web()
    paths = {getattr(r, "path", "") for r in module.app.routes}
    assert "/ai/mix" in paths
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(module.app)
    r = client.post("/ai/mix", json={"prompt": "floaty vocals", "stems": ["vocals"]})
    assert r.status_code == 503


def test_mixer_ui_has_describe_the_vibe():
    module = load_web()
    h = module.MIXER_HTML
    assert 'id="vibe"' in h          # the prompt box
    assert "function askProducer" in h
    assert "function applyAiMix" in h
    assert "setFire" in h            # widgets the AI drives
    assert "/ai/mix" in h            # calls the endpoint
