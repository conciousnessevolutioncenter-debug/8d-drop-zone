import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from studio.app import create_app
from studio.config import StudioConfig

def test_api_guards_and_import(tmp_path):
    app = create_app(StudioConfig(tmp_path))
    with TestClient(app, base_url='http://127.0.0.1:8765') as c:
        assert c.get('/api/health').status_code == 200
        assert c.get('/api/health', headers={'host': 'evil.example'}).status_code == 403
        assert c.post('/api/projects').status_code == 403
        h = {'X-Studio-Token': app.state.token}
        assert c.post('/api/projects', headers={**h, 'Origin': 'https://evil.example'}).status_code == 403
        assert c.post('/api/projects', headers={**h, 'Origin': 'null'}).status_code == 403
        b = io.BytesIO()
        sf.write(b, np.zeros((800, 2)), 8000, format='WAV')
        response = c.post('/api/projects', files={'file': ('song.wav', b.getvalue())}, headers=h)
        assert response.status_code == 201, response.text
        p = response.json()
        assert c.get('/api/projects').json()[0]['id'] == p['id']
        assert c.get('/api/projects/'+p['id']+'/waveform').status_code == 200
        assert c.get('/api/assets/'+p['asset']['id']).status_code == 200
        assert c.get('/api/assets/unknown').status_code == 404
        assert c.patch('/api/projects/'+p['id']+'/settings', json={'rotation_cpm': -1}, headers=h).status_code == 422

def test_render_export_restart(tmp_path):
    import time
    config=StudioConfig(tmp_path)
    app=create_app(config)
    with TestClient(app,base_url='http://127.0.0.1:8765') as c:
        h={'X-Studio-Token':app.state.token}
        b=io.BytesIO(); sf.write(b,np.zeros((800,2)),8000,format='WAV')
        p=c.post('/api/projects',files={'file':('test.wav',b.getvalue())},headers=h).json()
        response=c.post('/api/projects/'+p['id']+'/renders',json={'kind':'final','format':'wav'},headers=h)
        assert response.status_code==202, response.text
        job=response.json()
        for _ in range(200):
            result=c.get('/api/jobs/'+job['id']).json()
            if result['status'] in ('completed','failed'): break
            time.sleep(.025)
        assert result['status']=='completed', result
        assert c.get('/api/jobs/'+job['id']+'/download').content[:4]==b'RIFF'
    app2=create_app(config)
    with TestClient(app2,base_url='http://127.0.0.1:8765') as c:
        assert c.get('/api/jobs/'+job['id']).json()['status']=='completed'

def test_large_multipart_rejected_before_parsing(tmp_path):
    app=create_app(StudioConfig(tmp_path,max_bytes=10))
    with TestClient(app,base_url='http://127.0.0.1:8765') as c:
        h={'X-Studio-Token':app.state.token,'Content-Length':'999999','Content-Type':'multipart/form-data; boundary=x'}
        assert c.post('/api/projects',content=b'bad',headers=h).status_code==413

def test_page_read_is_explicit_utf8(tmp_path,monkeypatch):
    from pathlib import Path
    original=Path.read_text
    seen=[]
    def read(path,*args,**kwargs):
        if path.name=='index.html': seen.append(kwargs.get('encoding'))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    app=create_app(StudioConfig(tmp_path))
    with TestClient(app,base_url='http://127.0.0.1:8765') as c:
        assert c.get('/').status_code==200
    assert seen==['utf-8']

def test_chunked_upload_limit(tmp_path):
    app=create_app(StudioConfig(tmp_path,max_bytes=10))
    with TestClient(app,base_url='http://127.0.0.1:8765') as c:
        boundary='testboundary'
        def chunks():
            yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
            yield b'x'*100000
            yield (f'\r\n--{boundary}--\r\n').encode()
        response=c.post('/api/projects',content=chunks(),headers={'X-Studio-Token':app.state.token,'Content-Type':f'multipart/form-data; boundary={boundary}'})
        assert response.status_code==413
