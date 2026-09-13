"""add overlap spans to segments

Clip-relative stretches where two or more people talk at once, from the overlap detector that
runs over the whole episode at ingest (D77). They raise the ``speaker_overlap`` heads-up flag,
which is shown to the annotator and never scored.

Nullable, and ``[]`` is a different value from null: an empty list is a clip the detector read
and found clean, null is a clip nobody measured. Segments imported before this migration are
null until the backfill reaches them.

Revision ID: 33de0f6109bd
Revises: d72b5e1f0a42
Create Date: 2026-09-13

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "33de0f6109bd"
down_revision: str | Sequence[str] | None = "d72b5e1f0a42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "segments",
        sa.Column("overlap_spans_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("segments", "overlap_spans_jsonb")
