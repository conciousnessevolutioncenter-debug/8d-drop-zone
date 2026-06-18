"""Database engine + session for the social layer.

DATABASE_URL drives the backend: defaults to a local SQLite file for dev, and
should be set to a Postgres URL in production (Railway Postgres plugin) since
the container filesystem is ephemeral. SQLAlchemy 2.0 style.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_DEFAULT_SQLITE = f"sqlite:///{Path(__file__).resolve().parent.parent / 'social.db'}"
DATABASE_URL = os.environ.get("DATABASE_URL", _DEFAULT_SQLITE)

# Railway/Heroku hand out postgres:// URLs; SQLAlchemy wants postgresql://.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_columns() -> None:
    """Add any model columns missing from existing tables (idempotent).

    A lightweight forward-only migration so adding a field (e.g. tier, credits)
    never requires wiping the database. New columns are added nullable (or with a
    literal default) so existing rows stay intact. For complex migrations, move
    to Alembic — but this prevents the common 'added a column -> data lost' trap.
    """
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in tables:
            continue  # create_all just made it with the full schema
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in have:
                continue
            ddl_type = col.type.compile(dialect=engine.dialect)
            default = ""
            d = getattr(col.default, "arg", None) if col.default is not None else None
            if d is not None and not callable(d):
                if isinstance(d, bool):
                    default = f" DEFAULT {1 if d else 0}"
                elif isinstance(d, (int, float)):
                    default = f" DEFAULT {d}"
                elif isinstance(d, str):
                    default = " DEFAULT '" + d.replace("'", "''") + "'"
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}{default}'))
                print(f"[social] migrated: added {table.name}.{col.name}", flush=True)
            except Exception as exc:  # pragma: no cover - dialect-specific
                print(f"[social] could not add {table.name}.{col.name}: {exc}", flush=True)


def init_db() -> None:
    """Create tables if absent + add any newly-introduced columns."""
    from . import models  # noqa: F401  (register models on Base.metadata)
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    # Seed the global lobby room so chat works out of the box.
    db = SessionLocal()
    try:
        if not db.query(models.Room).filter_by(slug="lobby").first():
            db.add(models.Room(slug="lobby", name="Lobby", topic="Global chat for all creators"))
            db.commit()
    finally:
        db.close()
