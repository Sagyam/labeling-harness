"""The ORM and the migrations must not drift apart.

Every other test builds its schema by running the migrations, so a model change that never got a
migration would pass the whole suite and then fail only against a real deployment.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.engine import Engine

from app.db.base import Base
from app.models import *  # noqa: F403  (registers every table on Base.metadata)

pytestmark = pytest.mark.db


def test_migrated_schema_matches_the_orm(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        differences = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert differences == [], (
        "the ORM and the migrated schema disagree; generate a migration for:\n"
        + "\n".join(f"  {d}" for d in differences)
    )


def test_import_run_status_constraint_rejects_a_removed_value(db_engine: Engine) -> None:
    """``failed`` and ``dry_run`` were never written; the CHECK no longer claims them."""
    with db_engine.connect() as conn:
        transaction = conn.begin()
        with pytest.raises(sa.exc.IntegrityError):
            conn.execute(
                sa.text("INSERT INTO import_runs (source_path, status) VALUES ('x', 'failed')")
            )
        transaction.rollback()
