from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from studio.models import MixSettings
from studio.renderer import RenderRequest, render, RenderCancelled

def source(tmp_path, silent=False, mono=False):
    sr=8000
    t=np.arange(sr)/sr
    x=np.zeros(sr) if silent else 0.3*np.sin(2*np.pi*440*t)
    path=tmp_path/'input.wav'
    sf.write(path, x if mono else np.column_stack([x,x]), sr)
    return path

@pytest.mark.parametrize('fmt', ['wav','flac'])
def test_render_and_preview(tmp_path, fmt):
    src=source(tmp_path)
    out=tmp_path/f'final.{fmt}'
    report=render(RenderRequest(src,out,MixSettings(),fmt), lambda p: None, lambda: False)
    audio,sr=sf.read(out)
    assert sr == 8000 and audio.shape == (8000,2)
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1
    assert report['sample_rate'] == sr
    preview=tmp_path/'preview.wav'
    render(RenderRequest(src,preview,MixSettings(),'wav',0.25,0.5), lambda p: None, lambda: False)
    assert sf.info(preview).frames == 4000

@pytest.mark.parametrize('silent,mono', [(True,False),(False,True)])
def test_silent_and_mono(tmp_path, silent, mono):
    out=tmp_path/'out.wav'
    render(RenderRequest(source(tmp_path,silent,mono),out,MixSettings(),'wav'), lambda p: None, lambda: False)
    x,_=sf.read(out)
    assert np.isfinite(x).all() and x.shape[1] == 2

def test_cancel_cleans_partial(tmp_path):
    src=source(tmp_path)
    out=tmp_path/'out.wav'
    cancelled=[False]
    def progress(p): cancelled[0]=True
    with pytest.raises(RenderCancelled):
        render(RenderRequest(src,out,MixSettings(),'wav'), progress, lambda: cancelled[0])
    assert not out.exists()
    assert not list(tmp_path.glob('.render-*'))

def test_corrupt_and_write_failure(tmp_path, monkeypatch):
    from studio import renderer
    src=tmp_path/'bad.wav'; src.write_bytes(b'bad')
    out=tmp_path/'out.wav'
    with pytest.raises(Exception):
        render(RenderRequest(src,out,MixSettings(),'wav'), lambda p: None, lambda: False)
    src=source(tmp_path)
    def fail(*args,**kwargs): raise OSError('disk full')
    monkeypatch.setattr(renderer,'render_8d_file_to_wav',fail)
    with pytest.raises(OSError):
        render(RenderRequest(src,out,MixSettings(),'wav'), lambda p: None, lambda: False)
    assert not out.exists() and not list(tmp_path.glob('.render-*'))
