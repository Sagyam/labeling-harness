"""Add a speaker label to hypothesis words.

A transcriber that diarizes reports which speaker said each word. The column is nullable and
stays null for every transcriber that does not -- which is all of them except
``gemini-3.5-transcribe`` (D36).

The label is clip-local: ``spk_1`` in one hypothesis is not ``spk_1`` in another, and neither is
``segments.speaker_id``, which names a person from an upstream manifest. What it is good for is
the comparison within one clip -- two labels mean a turn boundary the VAD segmenter assumed was
not there.

Revision ID: facb0b37b4f8
Revises: 8b1f2c94d7a3
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "facb0b37b4f8"
down_revision: str | Sequence[str] | None = "8b1f2c94d7a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("hypothesis_words", sa.Column("speaker", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("hypothesis_words", "speaker")
