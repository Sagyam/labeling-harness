"""asr_systems.kind

A system is either a recogniser that heard the audio (``asr``) or the fuser that reconciled the
recognisers' text (``fusion``, D72). Every existing system is a recogniser.

Revision ID: d72b5e1f0a42
Revises: d71a0c3e5b10
Create Date: 2026-09-10 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d72b5e1f0a42"
down_revision: str | Sequence[str] | None = "d71a0c3e5b10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "asr_systems",
        sa.Column("kind", sa.String(length=16), nullable=False, server_default="asr"),
    )
    op.create_check_constraint("kind_allowed", "asr_systems", "kind IN ('asr', 'fusion')")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("kind_allowed", "asr_systems", type_="check")
    op.drop_column("asr_systems", "kind")
