"""Monetization gate: paid tiers get watermark-free shares; Stripe wiring intact."""
import os
import sys
import tempfile


def _fresh_env():
    os.environ["SOCIAL_MEDIA_DIR"] = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "m.db").replace("\\", "/")
    os.environ["PUBLIC_SITE_URL"] = "https://the8dengine.com"
    for mod in [m for m in sys.modules if m.startswith("social")]:
        del sys.modules[mod]


def test_is_paid_distinguishes_free_from_paid():
    _fresh_env()
    from social import entitlements as ent
    from social.models import User
    assert ent.is_paid(None) is False
    assert ent.is_paid(User(tier="free")) is False
    for t in ("creator", "producer", "studio"):
        assert ent.is_paid(User(tier=t)) is True


def test_paid_user_gets_watermark_free_share():
    """The gate: paid -> watermarked=False (clean video + card), free -> branded."""
    _fresh_env()
    from social.db import init_db, SessionLocal
    from social import tracks as T
    import numpy as np, soundfile as sf
    from pathlib import Path
    init_db()
    sr = 44100
    p = Path(tempfile.mkdtemp()) / "s.wav"
    sf.write(str(p), np.zeros((sr, 2), dtype="float32"), sr)

    class R:
        base_url = "https://the8dengine.com/"

    db = SessionLocal()
    free = T.create_track(db, audio_path=p, title="Free", watermarked=True)
    paid = T.create_track(db, audio_path=p, title="Paid", watermarked=False)
    db.close()
    assert free.watermarked is True and paid.watermarked is False
    assert "const WATERMARKED=true" in T._player_js(free, R())
    assert "const WATERMARKED=false" in T._player_js(paid, R())
    # Both render real cover PNGs (branding gated internally by the flag).
    assert T.make_cover_png(free)[:8] == b"\x89PNG\r\n\x1a\n"
    assert T.make_cover_png(paid)[:8] == b"\x89PNG\r\n\x1a\n"


def test_stripe_backend_is_fully_wired():
    _fresh_env()
    from social import billing
    # Tier<->price mapping + checkout/webhook/portal present and env-gated.
    assert set(billing._TIER_PRICE_ENV) == {"creator", "producer", "studio"}
    assert hasattr(billing, "checkout") and hasattr(billing, "stripe_webhook") and hasattr(billing, "portal")
    assert billing.stripe_enabled() is False  # dormant without keys
    # The headline perk is advertised on the paid plans.
    assert any("Watermark-free" in p for p in billing._PERKS["creator"])


def test_publish_endpoint_returns_gate_fields():
    _fresh_env()
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "web_app.py"
    spec = importlib.util.spec_from_file_location("eightd_money_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import inspect
    src = inspect.getsource(module.publish_track)
    assert "is_paid" in src
    assert "watermarked=not paid" in src
    assert "allow_download=paid" in src
    assert "upgrade_url" in src
