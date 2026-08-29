"""Database layer: SQLAlchemy 2.0 engine/session.

Default metadata store is SQLite (zero-config, for immediate `docker compose up`
testing). Set DATABASE_URL to a postgresql:// DSN to use PostgreSQL (the design's
production store). All models map to the t_* tables from 系统设计 §4.2.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


def _ensure_sqlite_parent_dir() -> None:
    """Create the parent directory of a SQLite file if it does not exist.

    The default DATABASE_URL is sqlite:////data/db/app.db (a Docker volume path).
    On a bare-metal run the directory may not exist yet, and SQLite raises
    'unable to open database file' at connect time. For SQLite we proactively
    mkdir -p the parent so the DB can be created wherever the path points.
    """
    url = settings.database_url
    if not url.startswith("sqlite"):
        return
    # sqlite:////abs/path -> '/abs/path' ; sqlite:///rel -> 'rel'
    path = url.split("///", 1)[1] if "///" in url else ""
    if not path:
        return
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        logging.getLogger("db").info("created SQLite dir: %s", parent)


_ensure_sqlite_parent_dir()

_connect_args = {}
if settings.database_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    # Import models so they register on Base.metadata
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    # SQLite's create_all never ALTERs existing tables, so columns added to the
    # ORM after a DB was first created would raise on INSERT. Migrate idempotently.
    migrate_db()


def migrate_db() -> None:
    """Idempotently add columns that exist in the ORM but may be missing from a
    previously-created SQLite (or Postgres) table.

    Each entry is (table, column, ADD COLUMN clause). We read the live column
    list via PRAGMA (SQLite) / information_schema is not portable, but for our
    SQLite-first store PRAGMA covers the deployment default; Postgres callers
    should run Alembic instead — this guard still no-ops safely there because
    the PRAGMA query is wrapped in try/except.
    """
    from sqlalchemy import text

    expected = [
        ("t_generation_request", "mode", "VARCHAR(32) NOT NULL DEFAULT 'reference'"),
        ("t_generation_request", "first_stage_resolution", "VARCHAR(16) NOT NULL DEFAULT '360P'"),
        ("t_generation_request", "video_asset_ids", "TEXT"),
        ("t_generation_request", "loras", "TEXT"),
        ("t_generation_request", "optimize_method", "VARCHAR(16) NOT NULL DEFAULT 'builtin'"),
        ("t_generation_request", "text_encoder", "VARCHAR(255) NOT NULL DEFAULT 'clip'"),
        ("t_generation_request", "template_key", "VARCHAR(64)"),
    ]
    try:
        with engine.begin() as conn:
            for table, col, ddl in expected:
                try:
                    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                except Exception:
                    # Not SQLite or table missing — skip (e.g. Postgres path).
                    continue
                existing = {r[1] for r in rows}
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
    except Exception as e:  # pragma: no cover — never block startup on migrate
        logging.getLogger("db").warning("migrate_db skipped: %s", e)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
