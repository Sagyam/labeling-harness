"""diarization runs and speaker turns

Who spoke when, imported from a diarizer run over the retained episode audio after export (D58,
D78). A run is one diarization of one episode; its turns are episode-relative and may overlap,
because crosstalk is kept rather than resolved to one voice. Runs are append-only: the newest per
episode is current, and the same run imported twice is refused by its checksum.

Revision ID: d2c39b3f3b6f
Revises: 33de0f6109bd
Create Date: 2026-09-13

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d2c39b3f3b6f"
down_revision: str | Sequence[str] | None = "33de0f6109bd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "diarization_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("episode_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("checksum", sa.String(length=80), nullable=False),
        sa.Column("speakers_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embeddings_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.id"],
            name=op.f("fk_diarization_runs_episode_id_episodes"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_diarization_runs")),
        sa.UniqueConstraint(
            "episode_id", "checksum", name=op.f("uq_diarization_runs_episode_id_checksum")
        ),
    )
    op.create_index("ix_diarization_runs_episode_id", "diarization_runs", ["episode_id"])
    op.create_table(
        "speaker_turns",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("speaker", sa.String(length=64), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.CheckConstraint("end_time > start_time", name=op.f("ck_speaker_turns_end_after_start")),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["diarization_runs.id"],
            name=op.f("fk_speaker_turns_run_id_diarization_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_speaker_turns")),
    )
    op.create_index("ix_speaker_turns_run_start", "speaker_turns", ["run_id", "start_time"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_speaker_turns_run_start", table_name="speaker_turns")
    op.drop_table("speaker_turns")
    op.drop_index("ix_diarization_runs_episode_id", table_name="diarization_runs")
    op.drop_table("diarization_runs")
