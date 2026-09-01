"""Shared pytest fixtures.

Tests marked ``db`` need a live Postgres. The URL comes from ``TEST_DATABASE_URL``
and defaults to a ``harness_test`` database on localhost, which the fixture creates if absent.
Schema is built by running the real Alembic migrations, so the suite exercises what production
runs rather than ``create_all``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DB_URL = "postgresql+psycopg://harness:harness@localhost:5432/harness_test"


def test_database_url() -> str:
    """URL of the throwaway database the suite runs against."""
    return os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DB_URL)


def _ensure_database_exists(url: str) -> None:
    target = make_url(url)
    admin = target.set(database="postgres")
    engine = sa.create_engine(admin, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            sa.text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target.database}
        ).scalar()
        if not exists:
            conn.execute(sa.text(f'CREATE DATABASE "{target.database}"'))
    engine.dispose()


def alembic_config(url: str) -> Config:
    """Alembic config pointed at ``url``."""
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


@pytest.fixture(scope="session")
def db_url() -> str:
    """Test database URL, skipping the whole ``db`` suite when Postgres is unreachable."""
    url = test_database_url()
    try:
        _ensure_database_exists(url)
    except sa.exc.OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres unavailable at {make_url(url).render_as_string()}: {exc}")
    return url


@pytest.fixture(scope="session")
def db_engine(db_url: str) -> Iterator[Engine]:
    """Session-scoped engine against a database migrated to head."""
    os.environ["DATABASE_URL"] = db_url
    engine = sa.create_engine(db_url, future=True)
    cfg = alembic_config(db_url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Function-scoped session wrapped in a transaction that is rolled back after the test.

    Nested ``session.commit()`` calls inside application code commit to a SAVEPOINT, so tests
    stay isolated while still exercising real commit paths.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint", future=True)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
