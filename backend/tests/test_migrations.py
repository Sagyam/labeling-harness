"""The ORM and the migrations must not drift apart.

Every other test builds its schema by running the migrations, so a model change that never got a
migration would pass the whole suite and then fail only against a real deployment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.engine import Engine

from app.db.base import Base
from app.models import *  # noqa: F403  (registers every table on Base.metadata)

pytestmark = pytest.mark.db

#: `migrations/versions/` is not an importable package -- Alembic loads revisions by path -- so a
#: test that wants to exercise one revision's SQL has to load it the same way.
_REVISIONS = Path(__file__).resolve().parents[1] / "migrations" / "versions"


def _revision(filename: str):
    spec = importlib.util.spec_from_file_location(filename, _REVISIONS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_the_pii_strip_reduces_a_stored_speaker_block_to_the_allowlist(db_engine: Engine) -> None:
    """The migration's own SQL, run against rows shaped like the ones it was written for."""
    rows = {
        "ep_pii_mixed": {
            "topic": "tech",
            "speakers": {
                "spk0": {"name": "Sushant", "role": "host", "origin": "Kathmandu"},
                "spk1": {"gender": "female", "dialect": "eastern"},
            },
        },
        # Nothing survives the allowlist, so the whole key goes.
        "ep_pii_only": {"genre": "podcast", "speakers": {"spk0": {"name": "Kusang"}}},
        # A malformed block from some upstream manifest.
        "ep_pii_malformed": {"speakers": {"spk0": "Sushant"}},
        # No speaker block at all: must come out byte-identical.
        "ep_pii_none": {"topic": "business"},
    }
    with db_engine.connect() as conn:
        transaction = conn.begin()
        for external_id, metadata in rows.items():
            conn.execute(
                sa.text(
                    "INSERT INTO episodes (external_id, split, metadata_jsonb) "
                    "VALUES (:external_id, 'train', CAST(:metadata AS jsonb))"
                ),
                {"external_id": external_id, "metadata": json.dumps(metadata)},
            )

        strip_pii = _revision("9b1c0d4e2a71_strip_speaker_pii_from_episode_metadata.py")
        conn.execute(strip_pii._STRIP.bindparams(allowed=list(strip_pii.ALLOWED)))

        stored = dict(
            conn.execute(
                sa.text(
                    "SELECT external_id, metadata_jsonb FROM episodes "
                    "WHERE external_id LIKE 'ep_pii_%'"
                )
            ).all()
        )
        transaction.rollback()

    assert stored["ep_pii_mixed"] == {
        "topic": "tech",
        "speakers": {"spk0": {"role": "host"}, "spk1": {"gender": "female"}},
    }
    assert stored["ep_pii_only"] == {"genre": "podcast"}
    assert stored["ep_pii_malformed"] == {}
    assert stored["ep_pii_none"] == {"topic": "business"}
