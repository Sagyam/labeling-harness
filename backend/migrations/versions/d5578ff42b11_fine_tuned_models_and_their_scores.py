"""fine-tuned models and their per-clip scores

A fine-tuned model is described by the card its notebook writes, and each import of its gold or
val transcripts is a run scored against the labels of that moment (D83). The model's text lives
in its own tables, never in asr_hypotheses, so it cannot reach disagreement, the queue or an
export. Each clip keeps a snapshot of the reference it was scored against.

Revision ID: d5578ff42b11
Revises: d2c39b3f3b6f
Create Date: 2026-09-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d5578ff42b11"
down_revision: str | Sequence[str] | None = "d2c39b3f3b6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _now(name: str) -> sa.Column:
    return sa.Column(
        name, sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "asr_models",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("architecture", sa.Text(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("card_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        _now("created_at"),
        _now("updated_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_asr_models")),
        sa.UniqueConstraint("slug", name=op.f("uq_asr_models_slug")),
    )
    op.create_table(
        "model_eval_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.BigInteger(), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("decoder", sa.String(length=128), nullable=True),
        sa.Column("fold_version", sa.String(length=64), nullable=False),
        sa.Column("clip_count", sa.Integer(), nullable=False),
        sa.Column("metrics_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("source_sha256", sa.String(length=80), nullable=False),
        _now("created_at"),
        sa.CheckConstraint(
            "split IN ('gold', 'val')", name=op.f("ck_model_eval_runs_split_allowed")
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["asr_models.id"],
            name=op.f("fk_model_eval_runs_model_id_asr_models"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_eval_runs")),
        sa.UniqueConstraint(
            "model_id", "source_sha256", name=op.f("uq_model_eval_runs_model_id_source_sha256")
        ),
    )
    op.create_index("ix_model_eval_runs_model_id", "model_eval_runs", ["model_id"])
    op.create_table(
        "model_eval_clips",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("ref_label_id", sa.BigInteger(), nullable=True),
        sa.Column("ref_text", sa.Text(), nullable=False),
        sa.Column("hyp_text", sa.Text(), nullable=False),
        sa.Column("ref_words", sa.Integer(), nullable=False),
        sa.Column("errors", sa.Integer(), nullable=False),
        sa.Column("substitutions", sa.Integer(), nullable=False),
        sa.Column("deletions", sa.Integer(), nullable=False),
        sa.Column("insertions", sa.Integer(), nullable=False),
        sa.Column("raw_ref_words", sa.Integer(), nullable=False),
        sa.Column("raw_errors", sa.Integer(), nullable=False),
        sa.Column("ref_chars", sa.Integer(), nullable=False),
        sa.Column("char_errors", sa.Integer(), nullable=False),
        sa.Column("is_loop", sa.Boolean(), nullable=False),
        sa.Column("compute_s", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["model_eval_runs.id"],
            name=op.f("fk_model_eval_clips_run_id_model_eval_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_model_eval_clips_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["ref_label_id"],
            ["segment_labels.id"],
            name=op.f("fk_model_eval_clips_ref_label_id_segment_labels"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_eval_clips")),
        sa.UniqueConstraint(
            "run_id", "segment_id", name=op.f("uq_model_eval_clips_run_id_segment_id")
        ),
    )
    op.create_index("ix_model_eval_clips_run_id_errors", "model_eval_clips", ["run_id", "errors"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_model_eval_clips_run_id_errors", table_name="model_eval_clips")
    op.drop_table("model_eval_clips")
    op.drop_index("ix_model_eval_runs_model_id", table_name="model_eval_runs")
    op.drop_table("model_eval_runs")
    op.drop_table("asr_models")
