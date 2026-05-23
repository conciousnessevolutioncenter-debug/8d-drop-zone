import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from eightd_engine.dsp import AudioData


def load_live_module():
    path = Path(__file__).resolve().parents[1] / 'web_app.py'
    spec = importlib.util.spec_from_file_location('eightd_dropzone_web_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_convert_returns_immediately_with_job_id_before_dsp_finishes(monkeypatch, tmp_path):
    module = load_live_module()
    module.APP_DIR = tmp_path
    module.JOBS.clear()

    def slow_load_audio(_path):
        time.sleep(0.35)
        return AudioData(samples=np.zeros((100, 2), dtype=float), sample_rate=1000)

    monkeypatch.setattr(module, 'load_audio', slow_load_audio)
    monkeypatch.setattr(module, 'estimate_bpm', lambda _audio: 120.0)
    monkeypatch.setattr(module, 'process_8d', lambda audio, **_kwargs: audio)
    monkeypatch.setattr(module, 'export_audio', lambda _audio, path: Path(path).write_bytes(b'wav'))

    client = TestClient(module.app)
    start = time.monotonic()
    response = client.post('/convert', files={'file': ('song.wav', b'fake audio bytes', 'audio/wav')})
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

    monkeypatch.setattr(module, 'load_audio', lambda _path: AudioData(samples=np.zeros((100, 2), dtype=float), sample_rate=1000))
    monkeypatch.setattr(module, 'estimate_bpm', lambda _audio: 120.0)
    monkeypatch.setattr(module, 'process_8d', lambda audio, **_kwargs: audio)
    monkeypatch.setattr(module, 'export_audio', lambda _audio, path: Path(path).write_bytes(b'wav'))

    client = TestClient(module.app)
    response = client.post('/convert', files={'file': ('song.wav', b'fake audio bytes', 'audio/wav')})
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
