"""Single durable-job consumer; restart recovery never silently reruns work."""
import logging
from threading import Event, Thread, Lock
from .models import MixSettings
from .renderer import RenderRequest, RenderCancelled, render

class Worker:
    def __init__(self, store, config, renderer=render):
        self.store, self.config, self.renderer = store, config, renderer
        self.wake = Event()
        self.stopping = Event()
        self.lock = Lock()
        self.thread = None
        self.degraded = False
        self.pending_errors = set()
        self.paused = False
        store.init_jobs()

    def start(self):
        if self.thread is not None:
            return
        self.store.recover()
        self.paused = any(j['status']=='queued' for j in self.store.list_jobs())
        self.thread = Thread(target=self._run, name='studio-render', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)

    def notify(self):
        self.wake.set()

    def resume_queue(self):
        self.paused = False
        self.notify()

    def cancel(self, job_id):
        with self.lock:
            job = self.store.get_job(job_id)
            if job['status']=='queued':
                self.store.transition(job_id,'queued','cancelled')
            elif job['status']=='running':
                self.store.transition(job_id,'running','running',cancel_requested=1)
        self.notify()

    def retry(self, job_id):
        job = self.store.get_job(job_id)
        if job['status'] not in ('failed','interrupted','cancelled'):
            raise ValueError('Only stopped or failed jobs can be retried.')
        result = self.store.enqueue(job['project_id'],job['kind'],MixSettings(**job['settings']),job['options'])
        self.notify()
        return result

    def _run(self):
        while not self.stopping.is_set():
            self.wake.wait(.25 if not self.degraded else 1)
            self.wake.clear()
            try:
                for ident in tuple(self.pending_errors):
                    self.store.transition(ident, 'running', 'failed',
                        error='Storage failed during rendering. Free disk space and retry.')
                    self.pending_errors.discard(ident)
                if self.paused:
                    continue
                for job in self.store.list_jobs():
                    if self.stopping.is_set() or self.paused:
                        break
                    with self.lock:
                        claimed = job['status']=='queued' and self.store.transition(job['id'],'queued','running')
                    if claimed:
                        try:
                            self._process(job)
                        except Exception:
                            self.pending_errors.add(job['id'])
                            raise
                self.degraded = False
            except Exception:
                if not self.degraded:
                    logging.exception('Studio worker storage unavailable; will retry')
                self.degraded = True

    def _process(self, job):
        ident = job['id']
        opts = job['options']
        fmt = 'wav' if job['kind']=='preview' else opts.get('format','wav')
        output = self.config.resolve(f'renders/{ident}.{fmt}')
        def cancelled():
            return self.stopping.is_set() or bool(self.store.get_job(ident)['cancel_requested'])
        def progress(value):
            self.store.transition(ident,'running','running',progress=value)
        try:
            project = self.store.get_project(job['project_id'])
            req = RenderRequest(self.config.resolve(project['asset']['path']), output,
                                MixSettings(**job['settings']), fmt,
                                opts.get('start_seconds',0) if job['kind']=='preview' else None,
                                opts.get('duration_seconds',15) if job['kind']=='preview' else None)
            report = self.renderer(req,progress,cancelled)
            with self.lock:
                if cancelled():
                    raise RenderCancelled()
                self.store.transition(ident,'running','completed',progress=1,
                                      output=str(output.relative_to(self.config.data_dir)),report=report)
        except RenderCancelled:
            output.unlink(missing_ok=True)
            self.store.transition(ident,'running','interrupted' if self.stopping.is_set() else 'cancelled')
        except Exception as exc:
            logging.exception('Render %s failed', ident)
            output.unlink(missing_ok=True)
            message = str(exc) if isinstance(exc,ValueError) else 'Render failed. Check the source file, free disk space and codec installation; see studio.log.'
            self.store.transition(ident,'running','failed',error=message)
