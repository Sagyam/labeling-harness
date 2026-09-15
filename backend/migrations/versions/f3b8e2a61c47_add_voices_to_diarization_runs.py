"""add voices to diarization runs

Each run's speaker labels linked to anonymous voice ids across episodes, by the per-speaker
embeddings already stored with the run (D87). Null until a run is linked.

Revision ID: f3b8e2a61c47
Revises: e1a4c7d20b95
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f3b8e2a61c47"
down_revision: str | Sequence[str] | None = "e1a4c7d20b95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "diarization_runs",
        sa.Column("voices_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("diarization_runs", "voices_jsonb")
