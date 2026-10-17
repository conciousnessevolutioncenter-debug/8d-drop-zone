"""Loopback-only API. Existing hosted app is deliberately independent."""
from pathlib import Path
import secrets
from contextlib import asynccontextmanager
from typing import Literal
from pydantic import BaseModel, Field
from .worker import Worker
from .limits import UploadLimit
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .config import StudioConfig
from .models import MixSettings
from .store import Store
from .assets import import_audio, waveform_peaks


class RenderBody(BaseModel):
    kind: Literal['preview', 'final'] = 'final'
    format: Literal['wav', 'flac', 'mp3'] = 'wav'
    start_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    duration_seconds: float = Field(default=15, gt=0, le=30, allow_inf_nan=False)


def create_app(config: StudioConfig) -> FastAPI:
    store = Store(config.data_dir / 'studio.db')
    worker = Worker(store, config)
    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        yield
        worker.stop()
    app = FastAPI(title='8D Engine Studio', lifespan=lifespan)
    app.add_middleware(UploadLimit, max_bytes=config.max_bytes + 65536)
    app.state.store = store
    app.state.worker = worker
    app.state.token = secrets.token_urlsafe(32)
    allowed = {f'127.0.0.1:{config.port}', f'localhost:{config.port}'}

    @app.middleware('http')
    async def local_guard(request: Request, call_next):
        host = request.headers.get('host', '')
        origin = request.headers.get('origin')
        if host not in allowed or (origin is not None and origin != 'http://' + host):
            return JSONResponse({'detail': 'Only the local Studio origin is allowed.'}, status_code=403)
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            if not secrets.compare_digest(request.headers.get('x-studio-token', ''), app.state.token):
                return JSONResponse({'detail': 'Session expired. Refresh Studio.'}, status_code=403)
        if request.url.path == '/api/projects' and request.method == 'POST':
            limit = config.max_bytes + 65536  # small multipart header allowance
            declared = request.headers.get('content-length')
            if declared is not None:
                try:
                    oversized = int(declared) > limit or int(declared) < 0
                except ValueError:
                    oversized = True
                if oversized:
                    return JSONResponse({'detail': 'Upload exceeds the allowed size.'}, status_code=413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({'detail': 'Project or asset not found.'}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    @app.get('/api/health')
    def health():
        from eightd_engine.audio_io import _find_ffmpeg
        try:
            _find_ffmpeg()
            ffmpeg = True
        except RuntimeError:
            ffmpeg = False
        return {'service': '8d-engine-studio', 'status': 'degraded' if worker.degraded else 'ready', 'ffmpeg': ffmpeg, 'queue_paused': worker.paused}

    @app.get('/api/projects')
    def projects():
        return store.list_projects()

    @app.post('/api/projects', status_code=201)
    def upload(file: UploadFile = File(...)):
        asset = import_audio(file.file, file.filename or 'Untitled', config)
        return store.create_project(Path(asset['name']).stem, asset)

    @app.get('/api/projects/{project_id}')
    def project(project_id: str):
        return store.get_project(project_id)

    @app.patch('/api/projects/{project_id}/settings')
    def settings(project_id: str, body: MixSettings):
        return store.update_settings(project_id, body)

    @app.get('/api/projects/{project_id}/waveform')
    def waveform(project_id: str):
        p = store.get_project(project_id)
        return waveform_peaks(config.resolve(p['asset']['path']))

    @app.get('/api/assets/{asset_id}')
    def asset(asset_id: str):
        for p in store.list_projects():
            if p['asset']['id'] == asset_id:
                return FileResponse(config.resolve(p['asset']['path']), media_type='audio/wav')
        raise KeyError(asset_id)

    @app.post('/api/projects/{project_id}/renders', status_code=202)
    def submit_render(project_id: str, body: RenderBody):
        project = store.get_project(project_id)
        if body.kind == 'preview' and body.start_seconds >= project['asset']['duration']:
            raise ValueError('Preview starts beyond the end of the track.')
        if body.format == 'mp3':
            from eightd_engine.audio_io import _find_ffmpeg
            try:
                _find_ffmpeg()
            except RuntimeError as exc:
                raise ValueError('MP3 export requires FFmpeg. Choose WAV or FLAC.') from exc
        job = store.enqueue(project_id, body.kind, MixSettings(**project['settings']), body.model_dump())
        worker.notify()
        return job

    @app.get('/api/jobs')
    def jobs():
        return store.list_jobs()

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        return store.get_job(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        worker.cancel(job_id)
        return store.get_job(job_id)

    @app.post('/api/jobs/{job_id}/retry', status_code=202)
    def retry(job_id: str):
        return worker.retry(job_id)

    @app.post('/api/queue/resume')
    def resume():
        worker.resume_queue()
        return {'queue_paused': False}

    @app.get('/api/jobs/{job_id}/download')
    def download(job_id: str):
        job = store.get_job(job_id)
        if job['status'] != 'completed' or not job['output']:
            raise ValueError('This render is not ready to download.')
        path = config.resolve(job['output'])
        if not path.is_file():
            raise KeyError('Output file missing')
        kind = {'.wav':'audio/wav','.flac':'audio/flac','.mp3':'audio/mpeg'}[path.suffix]
        return FileResponse(path, media_type=kind, filename='8D_' + path.name)

    static = Path(__file__).parent / 'static'
    app.mount('/static', StaticFiles(directory=static), name='static')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return (static / 'index.html').read_text(encoding='utf-8').replace('__TOKEN__', app.state.token)

    return app
