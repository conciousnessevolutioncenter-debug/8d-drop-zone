import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from eightd_engine.dsp import AudioData, CorrelationReport


def load_live_module():
    path = Path(__file__).resolve().parents[1] / 'web_app.py'
    spec = importlib.util.spec_from_file_location('eightd_dropzone_web_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_wav_bytes(seconds: float = 0.2, sample_rate: int = 8000) -> bytes:
    """A short, real, soundfile-native WAV so to_seekable_wav takes the no-ffmpeg
    path (it recognizes native formats and returns the file as-is) instead of
    shelling out to decode it."""
    import io
    samples = np.zeros((int(seconds * sample_rate), 2), dtype='float32')
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format='WAV', subtype='FLOAT')
    return buf.getvalue()


def _fake_render_8d_file_to_wav(delay: float = 0.0):
    """Stand-in for the real streaming renderer: writes a tiny valid WAV to
    out_path (so downstream loudness measurement / file-exists checks still
    work against a real file) and returns a CorrelationReport, matching the
    real function's contract without doing any actual DSP work."""
    def _fake(in_path, out_path, **_kwargs):
        if delay:
            time.sleep(delay)
        sf.write(str(out_path), np.zeros((100, 2), dtype='float32'), 8000, subtype='FLOAT')
        return CorrelationReport(correlation=0.95, side_mid_ratio=0.08, phase_warning=False)
    return _fake


def test_convert_returns_immediately_with_job_id_before_dsp_finishes(monkeypatch, tmp_path):
    module = load_live_module()
    module.APP_DIR = tmp_path
    module.JOBS.clear()

    monkeypatch.setattr(module, 'estimate_bpm', lambda _audio: 120.0)
    monkeypatch.setattr(module, 'render_8d_file_to_wav', _fake_render_8d_file_to_wav(delay=0.35))

    client = TestClient(module.app)
    start = time.monotonic()
    response = client.post('/convert', files={'file': ('song.wav', _tiny_wav_bytes(), 'audio/wav')})
    elapsed = time.monotonic() - start

    assert response.status_code == 202
    payload = response.json()
    assert payload['status'] == 'processing'
    assert payload['job_id']
    assert elapsed < 0.25


def test_job_status_exposes_download_when_background_render_completes(monkeypatch, tmp_path):
    module = load_live_module()
    module.APP_DIR = tmp_path
    module.JOBS.clear()

    monkeypatch.setattr(module, 'estimate_bpm', lambda _audio: 120.0)
    monkeypatch.setattr(module, 'render_8d_file_to_wav', _fake_render_8d_file_to_wav())

    client = TestClient(module.app)
    response = client.post('/convert', files={'file': ('song.wav', _tiny_wav_bytes(), 'audio/wav')})
    assert response.status_code == 202
    job_id = response.json()['job_id']

    deadline = time.time() + 2
    status = None
    while time.time() < deadline:
        status = client.get(f'/jobs/{job_id}')
        if status.json().get('status') == 'complete':
            break
        time.sleep(0.02)

    assert status.status_code == 200
    payload = status.json()
    assert payload['status'] == 'complete'
    assert payload['download_url'].endswith('_8D_Final.wav')
    assert Path(tmp_path / payload['output_name']).exists()


def test_ai_stem_mode_uses_separation_and_stem_renderer(monkeypatch, tmp_path):
    module = load_live_module()
    module.APP_DIR = tmp_path
    module.JOBS.clear()
    src = tmp_path / 'song.wav'
    sf.write(str(src), np.zeros((1600, 2), dtype='float32'), 8000, subtype='FLOAT')
    out = tmp_path / 'song_8D_Final.wav'
    calls = {'separate': 0, 'stem_render': 0, 'classic': 0}
    audio = AudioData(samples=np.ones((100, 2), dtype=float) * 0.05, sample_rate=1000)

    monkeypatch.setattr(module, 'estimate_bpm', lambda _audio: 120.0)
    monkeypatch.setattr(module, 'load_audio', lambda _path: audio)
    monkeypatch.setattr(module, 'available_stem_mode', lambda: {'mode': 'demucs', 'message': 'ok'})

    def fake_separate(_src, work_dir=None):
        calls['separate'] += 1
        return {'vocals': module.StemData(audio.samples, audio.sample_rate), 'bass': module.StemData(audio.samples, audio.sample_rate)}

    def fake_stem_render(stems, reference, **kwargs):
        calls['stem_render'] += 1
        assert set(stems) == {'vocals', 'bass'}
        assert reference.sample_rate == 1000
        return reference

    monkeypatch.setattr(module, 'separate_stems_from_file', fake_separate)
    monkeypatch.setattr(module, 'process_stem_spatial_mix', fake_stem_render)
    monkeypatch.setattr(module, 'process_8d', lambda audio, **_kwargs: calls.__setitem__('classic', calls['classic'] + 1) or audio)
    monkeypatch.setattr(module, 'export_audio', lambda _audio, path: Path(path).write_bytes(b'wav'))

    module._process_job('job-stems', src, out, stem_mode='ai_stems')

    payload = module.JOBS['job-stems']
    assert payload['status'] == 'complete'
    assert payload['stem_mode'] == 'ai_stems'
    assert payload['stem_engine'] == 'demucs'
    assert calls == {'separate': 1, 'stem_render': 1, 'classic': 0}


def test_rejects_uploaded_tracks_longer_than_20_minutes(monkeypatch, tmp_path):
    module = load_live_module()
    module.APP_DIR = tmp_path
    module.JOBS.clear()
    src = tmp_path / 'long_song.wav'
    sf.write(str(src), np.zeros((100, 2), dtype='float32'), 8000, subtype='FLOAT')
    out = tmp_path / 'long_song_8D_Final.wav'

    class _FakeInfo:
        samplerate = 1000
        frames = module.MAX_UPLOAD_SECONDS * 1000 + 1
        format = 'WAV'

    class _FakeSF:
        @staticmethod
        def info(_path):
            return _FakeInfo()

    monkeypatch.setattr(module, '_sf', _FakeSF())

    def _must_not_render(*_args, **_kwargs):
        raise AssertionError('render_8d_file_to_wav must not run past the length check')

    monkeypatch.setattr(module, 'render_8d_file_to_wav', _must_not_render)
    monkeypatch.setattr(module, 'load_audio', _must_not_render)

    module._process_job('job-too-long', src, out)

    payload = module.JOBS['job-too-long']
    assert payload['status'] == 'failed'
    assert f'{module.MAX_UPLOAD_MINUTES} minutes or shorter' in payload['error']


def test_homepage_exposes_ai_stem_mode_choice_and_upload_limit():
    module = load_live_module()
    assert 'AI stem spatial mix' in module.HTML
    assert 'stemMode' in module.HTML
    assert f'{module.MAX_UPLOAD_MINUTES} minutes' in module.HTML or '1 hour max' in module.HTML
