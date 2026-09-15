"""add classes to model eval clips

Each scored clip's classes, ``{axis: bucket}`` (D87), kept with the run like its overlap share,
so filtering a run's clips agrees with the run's breakdowns. Null for clips scored before
classes existed, until the runs are reclassified.

Revision ID: a6d0f4c93e18
Revises: f3b8e2a61c47
Create Date: 2026-09-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a6d0f4c93e18"
down_revision: str | Sequence[str] | None = "f3b8e2a61c47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "model_eval_clips",
        sa.Column("classes_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("model_eval_clips", "classes_jsonb")
