import threading
import time
import pytest
from studio.store import Store
from studio.models import MixSettings
from studio.config import StudioConfig
from studio.worker import Worker

def setup(tmp_path):
    s=Store(tmp_path/'studio.db')
    p=s.create_project('Song', {'id':'source','path':'source.wav'})
    return s,p

def test_queue_limit_snapshot_and_recovery(tmp_path):
    s,p=setup(tmp_path)
    jobs=[s.enqueue(p['id'],'final',MixSettings(),{'format':'wav'}) for _ in range(10)]
    with pytest.raises(ValueError): s.enqueue(p['id'],'final',MixSettings(),{})
    s.update_settings(p['id'],MixSettings(rotation_cpm=9))
    assert s.get_job(jobs[0]['id'])['settings']['rotation_cpm'] == 5.78
    assert s.transition(jobs[0]['id'],'queued','running')
    w=Worker(s,StudioConfig(tmp_path))
    w.start()
    try:
        assert s.get_job(jobs[0]['id'])['status']=='interrupted'
        assert all(j['status']=='queued' for j in s.list_jobs() if j['id']!=jobs[0]['id'])
        assert w.paused
    finally: w.stop()

def test_cancel_and_failure_retry(tmp_path):
    s,p=setup(tmp_path)
    w=Worker(s,StudioConfig(tmp_path))
    job=s.enqueue(p['id'],'final',MixSettings(),{'format':'wav'})
    w.cancel(job['id'])
    assert s.get_job(job['id'])['status']=='cancelled'
    job=s.enqueue(p['id'],'final',MixSettings(),{'format':'wav'})
    w.start(); w.resume_queue()
    try:
        for _ in range(200):
            if s.get_job(job['id'])['status']=='failed': break
            time.sleep(.01)
        assert s.get_job(job['id'])['status']=='failed'
        retried=w.retry(job['id'])
        assert retried['id'] != job['id']
        assert retried['settings']==job['settings']
    finally: w.stop()

def test_one_worker_and_active_cancellation(tmp_path):
    s,p=setup(tmp_path)
    entered=threading.Event()
    def fake(req,progress,cancelled):
        entered.set()
        while not cancelled(): time.sleep(.01)
        from studio.renderer import RenderCancelled
        raise RenderCancelled()
    w=Worker(s,StudioConfig(tmp_path),renderer=fake)
    w.start()
    try:
        job=s.enqueue(p['id'],'final',MixSettings(),{'format':'wav'})
        second=s.enqueue(p['id'],'final',MixSettings(),{'format':'wav'})
        w.notify()
        assert entered.wait(2)
        assert sum(j['status']=='running' for j in s.list_jobs())==1
        assert s.get_job(second['id'])['status']=='queued'
        w.cancel(job['id'])
        for _ in range(200):
            if s.get_job(job['id'])['status']=='cancelled': break
            time.sleep(.01)
        assert s.get_job(job['id'])['status']=='cancelled'
    finally: w.stop()

def test_worker_recovers_after_transient_database_failure(tmp_path,monkeypatch):
    import sqlite3
    s,p=setup(tmp_path)
    w=Worker(s,StudioConfig(tmp_path))
    w.start()
    original=s.list_jobs
    broken=threading.Event();broken.set()
    def flaky(*args,**kwargs):
        if broken.is_set(): raise sqlite3.OperationalError('database or disk is full')
        return original(*args,**kwargs)
    monkeypatch.setattr(s,'list_jobs',flaky)
    try:
        time.sleep(.4)
        assert w.thread.is_alive()
        assert w.degraded
        broken.clear()
        job=s.enqueue(p['id'],'final',MixSettings(),{'format':'wav'});w.notify()
        for _ in range(200):
            if s.get_job(job['id'])['status']=='failed':break
            time.sleep(.01)
        assert s.get_job(job['id'])['status']=='failed'
    finally:broken.clear();w.stop()
