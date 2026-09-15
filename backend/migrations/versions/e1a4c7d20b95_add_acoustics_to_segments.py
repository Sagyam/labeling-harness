"""add acoustics to segments

Acoustic measurements of each clip's audio, taken from the episode audio at ingest or by the
backfill (roadmap item 1, D87): today its bandwidth, later its SNR and reverb. One versioned
object rather than a column per value, so a measurement can be added without a migration and a
backfill can tell which rules a clip was measured under.

Nullable: null is a clip never measured. Segments imported before this migration are null until
the backfill reaches them.

Revision ID: e1a4c7d20b95
Revises: d5578ff42b11
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e1a4c7d20b95"
down_revision: str | Sequence[str] | None = "d5578ff42b11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "segments",
        sa.Column("acoustics_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("segments", "acoustics_jsonb")
