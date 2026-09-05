"""add vad speech spans to segments

Clip-relative speech regions from the VAD pass that cut the clip in the first place. They are the
only independent statement about where speech is: every other timing in the schema comes from a
transcriber, so without them "no system wrote anything here" cannot be told apart from "there was
nothing to write". That is what the ``missed_speech`` flag reads (D55).

Nullable, because every segment imported before this migration has no such record and inventing
one would make them all look either perfect or defective.

Revision ID: edc4533896c2
Revises: facb0b37b4f8
Create Date: 2026-09-05 22:42:47.530303

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "edc4533896c2"
down_revision: str | Sequence[str] | None = "facb0b37b4f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "segments",
        sa.Column("vad_spans_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("segments", "vad_spans_jsonb")
