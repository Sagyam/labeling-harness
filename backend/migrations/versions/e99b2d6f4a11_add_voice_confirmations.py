"""Voice confirmations and voiceprint suggestions on saved words (D99).

``voice_confirmations`` holds the owner's verdicts on stretches of clips offered as one voice's
speech alone, with the stretch's embedding for a confirmed one. ``label_words.suggested_speaker`` keeps what a
voiceprint suggested when the lanes were served, so suggestions can be scored.

Revision ID: e99b2d6f4a11
Revises: d98a1c5e7f20
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e99b2d6f4a11"
down_revision: str | Sequence[str] | None = "d98a1c5e7f20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "label_words", sa.Column("suggested_speaker", sa.String(length=64), nullable=True)
    )
    op.create_table(
        "voice_confirmations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("voice", sa.String(length=16), nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("diarization_run_id", sa.BigInteger(), nullable=True),
        sa.Column("speaker", sa.String(length=64), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("embedding_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("annotator", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "end_time > start_time", name=op.f("ck_voice_confirmations_end_after_start")
        ),
        sa.CheckConstraint(
            "verdict IN ('confirmed', 'rejected', 'cleared')",
            name=op.f("ck_voice_confirmations_verdict_allowed"),
        ),
        sa.ForeignKeyConstraint(
            ["diarization_run_id"],
            ["diarization_runs.id"],
            name=op.f("fk_voice_confirmations_diarization_run_id_diarization_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_voice_confirmations_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_voice_confirmations")),
    )
    op.create_index(
        op.f("ix_voice_confirmations_voice"), "voice_confirmations", ["voice"], unique=False
    )
    op.create_index(
        op.f("ix_voice_confirmations_segment_id"),
        "voice_confirmations",
        ["segment_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_voice_confirmations_segment_id"), table_name="voice_confirmations")
    op.drop_index(op.f("ix_voice_confirmations_voice"), table_name="voice_confirmations")
    op.drop_table("voice_confirmations")
    op.drop_column("label_words", "suggested_speaker")
