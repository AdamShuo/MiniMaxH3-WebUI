"""Database layer: SQLAlchemy 2.0 engine/session.

Default metadata store is SQLite (zero-config, for immediate `docker compose up`
testing). Set DATABASE_URL to a postgresql:// DSN to use PostgreSQL (the design's
production store). All models map to the t_* tables from 系统设计 §4.2.
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

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


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
