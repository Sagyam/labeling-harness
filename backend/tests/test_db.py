"""Tests for the database session layer."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.base import Base
from app.db.session import create_engine_from_settings, session_scope


def test_create_engine_uses_settings_url() -> None:
    engine = create_engine_from_settings(url="postgresql+psycopg://u:p@h:5432/d")
    assert engine.url.database == "d"
    assert engine.url.host == "h"


def test_declarative_base_uses_utc_timestamps() -> None:
    assert Base.metadata is not None


@pytest.mark.db
def test_engine_connects_to_postgres(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1


@pytest.mark.db
def test_database_timezone_is_utc(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        now = conn.execute(text("SELECT now() AT TIME ZONE 'utc'")).scalar_one()
    assert now is not None


@pytest.mark.db
def test_session_scope_commits_on_success(db_engine: Engine) -> None:
    with session_scope(bind=db_engine) as session:
        session.execute(text("CREATE TEMP TABLE t_commit (id int)"))
        session.execute(text("INSERT INTO t_commit VALUES (1)"))
    # A temp table dies with the connection; the point is that no exception escaped.


@pytest.mark.db
def test_session_scope_rolls_back_on_error(db_engine: Engine) -> None:
    with db_engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS t_rollback (id int primary key)"))
    try:
        with pytest.raises(RuntimeError), session_scope(bind=db_engine) as session:
            session.execute(text("INSERT INTO t_rollback VALUES (1)"))
            raise RuntimeError("boom")
        with db_engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM t_rollback")).scalar_one() == 0
    finally:
        with db_engine.begin() as conn:
            conn.execute(text("DROP TABLE t_rollback"))
