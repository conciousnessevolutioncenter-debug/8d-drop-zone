"""SQLite persistence. Each operation owns its connection and transaction."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .models import MixSettings

def now():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, asset TEXT NOT NULL,
                    settings TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                PRAGMA user_version=1;
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def project(row):
        if row is None:
            raise KeyError('Project not found')
        data = dict(row)
        for field in ('asset', 'settings'):
            data[field] = json.loads(data[field])
        return data

    def create_project(self, title: str, asset: dict) -> dict:
        ident = uuid.uuid4().hex
        stamp = now()
        with self.connection() as db:
            db.execute('INSERT INTO projects VALUES (?,?,?,?,?,?)',
                       (ident, title[:200], json.dumps(asset), MixSettings().model_dump_json(), stamp, stamp))
        return self.get_project(ident)

    def get_project(self, project_id: str) -> dict:
        with self.connection() as db:
            return self.project(db.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone())

    def list_projects(self) -> list[dict]:
        with self.connection() as db:
            return [self.project(r) for r in db.execute('SELECT * FROM projects ORDER BY updated_at DESC')]

    def update_settings(self, project_id: str, settings: MixSettings) -> dict:
        with self.connection() as db:
            if not db.execute('UPDATE projects SET settings=?, updated_at=? WHERE id=?',
                              (settings.model_dump_json(), now(), project_id)).rowcount:
                raise KeyError('Project not found')
        return self.get_project(project_id)

    def init_jobs(self):
        with self.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL,
                settings TEXT NOT NULL, options TEXT NOT NULL, status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0, output TEXT, report TEXT,
                error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')

    @staticmethod
    def job(row):
        if row is None:
            raise KeyError('Job not found')
        result = dict(row)
        for key in ('settings', 'options', 'report'):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def enqueue(self, project_id, kind, settings, options):
        self.get_project(project_id)
        self.init_jobs()
        if kind not in ('preview', 'final'):
            raise ValueError('Unknown render kind')
        ident = uuid.uuid4().hex
        stamp = now()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0] >= 10:
                raise ValueError('Queue is full. Wait for a render to finish.')
            db.execute('''INSERT INTO jobs
                (id,project_id,kind,settings,options,status,created_at,updated_at)
                VALUES (?,?,?,?,?,'queued',?,?)''',
                (ident,project_id,kind,settings.model_dump_json(),json.dumps(options),stamp,stamp))
        return self.get_job(ident)

    def get_job(self, job_id):
        self.init_jobs()
        with self.connection() as db:
            return self.job(db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone())

    def list_jobs(self, project_id=None):
        self.init_jobs()
        with self.connection() as db:
            if project_id:
                rows = db.execute('SELECT * FROM jobs WHERE project_id=? ORDER BY created_at', (project_id,))
            else:
                rows = db.execute('SELECT * FROM jobs ORDER BY created_at')
            return [self.job(row) for row in rows]

    def transition(self, job_id, expected, target, **updates):
        allowed = {'progress','output','report','error','cancel_requested'}
        if set(updates)-allowed:
            raise ValueError('Invalid job field')
        values = {'status':target, 'updated_at':now(), **updates}
        if 'report' in values:
            values['report'] = json.dumps(values['report'])
        with self.connection() as db:
            sql = ','.join(f'{key}=?' for key in values)
            return bool(db.execute(f'UPDATE jobs SET {sql} WHERE id=? AND status=?',
                                   (*values.values(),job_id,expected)).rowcount)

    def recover(self):
        self.init_jobs()
        with self.connection() as db:
            db.execute("UPDATE jobs SET status='interrupted', error='Studio stopped before this render finished.', updated_at=? WHERE status='running'", (now(),))
