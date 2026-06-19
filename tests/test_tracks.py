"""Distribution engine: public track pages, share cards, embeds, publish flow."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def _fresh_env():
    """Isolate the social DB + media dir so tests never touch real data."""
    os.environ["SOCIAL_MEDIA_DIR"] = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "t.db").replace("\\", "/")
    os.environ["PUBLIC_SITE_URL"] = "https://the8dengine.com"
    for mod in [m for m in sys.modules if m.startswith("social")]:
        del sys.modules[mod]


def _make_wav(seconds=3.0):
    sr = 44100
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = (0.4 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    p = Path(tempfile.mkdtemp()) / "src.wav"
    sf.write(str(p), np.stack([x, x * 0.8], 1), sr)
    return p


def _new_track(title="Midnight Orbit", artist="Aurora"):
    _fresh_env()
    from social.db import init_db, SessionLocal
    from social import tracks as T
    init_db()
    db = SessionLocal()
    trk = T.create_track(db, audio_path=_make_wav(), title=title, artist=artist,
                         lufs="-9.1", preset="reference_luxe")
    db.close()
    return T, trk


def test_create_track_persists_peaks_and_duration():
    T, trk = _new_track()
    assert trk.slug and len(trk.slug) >= 8
    peaks = json.loads(trk.peaks)
    assert len(peaks) > 0 and all(0.0 <= p <= 1.0 for p in peaks)
    assert trk.duration > 2.5


def test_player_page_has_meta_player_and_share():
    T, trk = _new_track()

    class R:
        base_url = "https://the8dengine.com/"

    html = T.render_player_page(R(), trk)
    # Auto-unfurling share card so every link is an ad.
    for tag in ("og:image", "og:audio", "twitter:card", "og:title"):
        assert tag in html
    # The spatial player + the TikTok/Reels video export + share targets.
    assert 'id="orbit"' in html and "makeVideo" in html
    assert "HEADPHONES ON" in html
    assert "Midnight Orbit" in html
    assert "Make your own" in html
    # Cover renders to a real PNG.
    png = T.make_cover_png(trk)
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_embed_page_is_a_compact_player():
    T, trk = _new_track()

    class R:
        base_url = "https://the8dengine.com/"

    emb = T.render_embed_page(R(), trk)
    assert 'id="orbit"' in emb and "Open in 8D" in emb


def test_track_routes_registered_and_publish_exists():
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_tracks_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    paths = {getattr(r, "path", "") for r in module.app.routes}
    assert "/t/{slug}" in paths
    assert "/embed/{slug}" in paths
    assert "/t/{slug}/audio" in paths
    assert "/t/{slug}/cover.png" in paths
    assert "/tracks/publish" in paths
    # Homepage offers the one-tap publish/share flow.
    assert "publishTrack" in module.HTML
    assert "Publish &amp; share" in module.HTML
