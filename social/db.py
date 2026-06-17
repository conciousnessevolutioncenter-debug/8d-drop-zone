"""Database engine + session for the social layer.

DATABASE_URL drives the backend: defaults to a local SQLite file for dev, and
should be set to a Postgres URL in production (Railway Postgres plugin) since
the container filesystem is ephemeral. SQLAlchemy 2.0 style.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
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


def init_db() -> None:
    """Create tables if absent. Fine for dev/early prod; swap to Alembic later."""
    from . import models  # noqa: F401  (register models on Base.metadata)
    Base.metadata.create_all(bind=engine)
    # Seed the global lobby room so chat works out of the box.
    db = SessionLocal()
    try:
        if not db.query(models.Room).filter_by(slug="lobby").first():
            db.add(models.Room(slug="lobby", name="Lobby", topic="Global chat for all creators"))
            db.commit()
    finally:
        db.close()
