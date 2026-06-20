"""Frictionless link input: rights guard + endpoint wiring + homepage UI."""
import importlib.util
import sys
from pathlib import Path

import pytest


def load_web():
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_link_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_streaming_platforms_are_rejected_for_rights():
    from fastapi import HTTPException
    module = load_web()
    for url in [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://open.spotify.com/track/abc",
        "https://soundcloud.com/artist/track",
        "https://music.apple.com/song",
    ]:
        with pytest.raises(HTTPException) as ei:
            module._fetch_audio_url(url)
        assert ei.value.status_code == 400
        assert "streaming" in ei.value.detail.lower()


def test_non_http_scheme_rejected():
    from fastapi import HTTPException
    module = load_web()
    for url in ["ftp://host/song.mp3", "file:///etc/passwd", "", "notaurl"]:
        with pytest.raises(HTTPException) as ei:
            module._fetch_audio_url(url)
        assert ei.value.status_code == 400


def test_ssrf_guard_blocks_internal_addresses():
    from fastapi import HTTPException
    module = load_web()
    # loopback, private, and the cloud-metadata link-local address
    for url in [
        "http://127.0.0.1/x.mp3",
        "http://localhost/x.wav",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/song.mp3",
        "http://192.168.1.1/song.wav",
    ]:
        with pytest.raises(HTTPException) as ei:
            module._fetch_audio_url(url)
        assert ei.value.status_code == 400


def test_convert_requires_a_file_or_link():
    from fastapi.testclient import TestClient
    module = load_web()
    if not module.DSP_AVAILABLE:
        pytest.skip("DSP stack not available")
    client = TestClient(module.app)
    r = client.post("/convert", data={"preset": "reference_luxe"})
    assert r.status_code == 400
    assert "link" in r.json()["detail"].lower()


def test_homepage_has_link_input():
    html = load_web().HTML
    assert 'id="srcUrl"' in html
    assert "function spatializeLink" in html
    assert "source_url" in html
    assert "own or are licensed" in html       # the rights notice
