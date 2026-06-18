"""HEIC/HEIF (iPhone photo) -> JPEG conversion for post images."""
import importlib.util
import io

import pytest

heic_ok = importlib.util.find_spec("pillow_heif") is not None


@pytest.mark.skipif(not heic_ok, reason="pillow-heif not installed")
def test_heic_converts_to_jpeg():
    import pillow_heif
    from PIL import Image
    from social.routes import _heic_to_jpeg

    pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    Image.new("RGB", (24, 16), (10, 120, 200)).save(buf, format="HEIF")
    jpg = _heic_to_jpeg(buf.getvalue())
    assert jpg is not None
    assert jpg[:3] == b"\xff\xd8\xff"           # JPEG magic
    assert Image.open(io.BytesIO(jpg)).size == (24, 16)


def test_heic_converter_returns_none_on_garbage():
    from social.routes import _heic_to_jpeg
    assert _heic_to_jpeg(b"not an image") is None
