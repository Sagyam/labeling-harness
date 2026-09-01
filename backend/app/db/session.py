"""Engine construction and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def create_engine_from_settings(url: str | None = None, *, pool_size: int | None = None) -> Engine:
    """Create a SQLAlchemy engine.

    Args:
        url: Connection URL. Defaults to the configured database URL.
        pool_size: Connection pool size. Defaults to the configured pool size.
    """
    settings = get_settings()
    return create_engine(
        url or settings.database.url,
        pool_size=pool_size or settings.database.pool_size,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide engine singleton."""
    return create_engine_from_settings()


def get_sessionmaker(bind: Engine | None = None) -> sessionmaker[Session]:
    """Return a sessionmaker bound to ``bind`` or the process engine."""
    return sessionmaker(bind=bind or get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope(bind: Engine | None = None) -> Iterator[Session]:
    """Transactional session context: commit on success, roll back on any exception."""
    session = get_sessionmaker(bind)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
