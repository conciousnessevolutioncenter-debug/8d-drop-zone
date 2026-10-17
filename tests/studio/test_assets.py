import io
import numpy as np
import soundfile as sf
import pytest
from studio.assets import import_audio, waveform_peaks
from studio.config import StudioConfig

def wav(seconds=0.1, sr=8000):
    b = io.BytesIO()
    sf.write(b, np.zeros((int(seconds*sr), 2)), sr, format='WAV')
    b.seek(0)
    return b

def test_unicode_duplicate_names_are_independent(tmp_path):
    c = StudioConfig(tmp_path)
    a = import_audio(wav(), 'Tanong ü song.wav', c)
    b = import_audio(wav(), 'Tanong ü song.wav', c)
    assert a['id'] != b['id']
    assert c.resolve(a['path']).exists()
    assert len(waveform_peaks(c.resolve(a['path']))) <= 1200

@pytest.mark.parametrize('payload', [b'', b'not audio'])
def test_invalid_import_cleaned(tmp_path, payload):
    with pytest.raises(ValueError):
        import_audio(io.BytesIO(payload), 'x.wav', StudioConfig(tmp_path))
    assert not list((tmp_path / 'assets').glob('*'))

def test_limits(tmp_path):
    assert StudioConfig(tmp_path).max_bytes == 200*1024*1024
    assert StudioConfig(tmp_path).max_seconds == 3600
    with pytest.raises(ValueError):
        import_audio(wav(), 'x.wav', StudioConfig(tmp_path, max_bytes=16))
    with pytest.raises(ValueError):
        import_audio(wav(), 'x.wav', StudioConfig(tmp_path, max_seconds=0.01))
    assert not list((tmp_path / 'assets').glob('*'))

def test_compressed_decoder_is_bounded_and_validates_nan(tmp_path,monkeypatch):
    from studio import assets
    calls=[]
    def fake(cmd,**kwargs):
        calls.append(cmd)
        sf.write(cmd[-1],np.array([[float('nan'),0.0]]),8000,subtype='FLOAT')
    monkeypatch.setattr(assets,'_find_ffmpeg',lambda:'ffmpeg')
    monkeypatch.setattr(assets.subprocess,'run',fake)
    with pytest.raises(ValueError,match='invalid sample'):
        import_audio(io.BytesIO(b'compressed'), 'x.m4a', StudioConfig(tmp_path,max_seconds=3))
    assert '-t' in calls[0]
    assert float(calls[0][calls[0].index('-t')+1]) == 4
    assert not list((tmp_path/'assets').glob('*'))
