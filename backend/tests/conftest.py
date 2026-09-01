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
    os.environ["HARNESS_LLM__DRY_RUN"] = "true"
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


@pytest.fixture
def object_storage(tmp_path: Path):
    """Local-filesystem storage rooted in the test's temporary directory."""
    from app.storage.local import LocalFilesystemStorage

    return LocalFilesystemStorage(root=tmp_path / "objects")


@pytest.fixture
def settings(tmp_path: Path):
    """Settings loaded from the repository configuration.

    The ingest work root is redirected into the test's temporary directory: the API writes an
    uploaded file there before the pipeline starts, and tests that stub out the pipeline never
    reach the cleanup, so the real ``data/ingest_work`` would slowly fill with test uploads.
    """
    from app.config import load_settings

    loaded = load_settings()
    return loaded.model_copy(
        update={"ingest": loaded.ingest.model_copy(update={"work_root": tmp_path / "ingest_work"})}
    )


@pytest.fixture
def client(db_session: Session, object_storage, settings):
    """TestClient wired to the test session and a temporary object store.

    Overriding the session dependency keeps API writes inside the test transaction, so the suite
    stays isolated while still exercising the real commit path.
    """
    from fastapi.testclient import TestClient

    from app.api.deps import get_config, get_object_storage, get_session
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[get_object_storage] = lambda: object_storage
    app.dependency_overrides[get_config] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def imported_episode(db_session: Session, object_storage, settings, tmp_path: Path):
    """A fully imported and queued episode: 6 segments, 3 systems, tasks built."""
    from app.services.fixtures import build_export_fixture
    from app.services.importer import import_manifest
    from app.services.queue_builder import build_queue

    root = build_export_fixture(
        tmp_path / "export_api", episode_id="api_ep001", segments=6, systems=3
    )
    import_manifest(db_session, root, storage=object_storage, settings=settings)
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    db_session.flush()
    return "api_ep001"
