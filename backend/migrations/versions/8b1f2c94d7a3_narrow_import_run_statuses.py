"""Narrow import_runs.status to the values the importer can actually write.

An import is atomic: it commits as ``succeeded`` or is rolled back whole. ``failed`` and
``dry_run`` were never written by any code path, so the CHECK constraint claimed a wider
vocabulary than the schema ever holds.

Revision ID: 8b1f2c94d7a3
Revises: 04e11abc3c9c
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "8b1f2c94d7a3"
down_revision: str | Sequence[str] | None = "04e11abc3c9c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: op.f marks the name as already rendered, so the "ck_%(table_name)s_%(constraint_name)s"
#: naming convention does not prefix it a second time.
_CONSTRAINT = "ck_import_runs_status_allowed"


def upgrade() -> None:
    op.drop_constraint(op.f(_CONSTRAINT), "import_runs", type_="check")
    op.create_check_constraint(
        op.f(_CONSTRAINT), "import_runs", "status IN ('running', 'succeeded')"
    )


def downgrade() -> None:
    op.drop_constraint(op.f(_CONSTRAINT), "import_runs", type_="check")
    op.create_check_constraint(
        op.f(_CONSTRAINT),
        "import_runs",
        "status IN ('running', 'succeeded', 'failed', 'dry_run')",
    )
