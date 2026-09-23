"""Per-speaker labels: a speakers queue, label words, and the run a label's speakers belong to (D98).

A label in the ``speakers`` label version carries one ``label_words`` row per word, with its span
and the diarized speaker it was attributed to. ``segment_labels.diarization_run_id`` says which
run's labels those speakers are; it is null for every single-stream label.

Downgrading deletes what only this revision can hold: tasks in the ``speakers`` queue and every
label in the ``speakers-v1`` version, whose words go with the table. Left behind, those labels
would read as a second single-stream label for their clips.

Revision ID: d98a1c5e7f20
Revises: a6d0f4c93e18
Create Date: 2026-09-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d98a1c5e7f20"
down_revision: str | Sequence[str] | None = "a6d0f4c93e18"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_QUEUE_CHECK = "ck_annotation_tasks_queue_allowed"
_SPEAKERS_VERSION = "speakers-v1"


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(op.f(_QUEUE_CHECK), "annotation_tasks", type_="check")
    op.create_check_constraint(
        op.f(_QUEUE_CHECK),
        "annotation_tasks",
        "queue IN ('review', 'audit', 'error', 'speakers')",
    )

    op.add_column("segment_labels", sa.Column("diarization_run_id", sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        op.f("fk_segment_labels_diarization_run_id_diarization_runs"),
        "segment_labels",
        "diarization_runs",
        ["diarization_run_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "label_words",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("label_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("word", sa.Text(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("speaker", sa.String(length=64), nullable=False),
        sa.Column("proposed_speaker", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.CheckConstraint("end_time > start_time", name=op.f("ck_label_words_end_after_start")),
        sa.CheckConstraint(
            "source IN ('label', 'recogniser', 'typed', 'copy')",
            name=op.f("ck_label_words_source_allowed"),
        ),
        sa.ForeignKeyConstraint(
            ["label_id"],
            ["segment_labels.id"],
            name=op.f("fk_label_words_label_id_segment_labels"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_label_words")),
        sa.UniqueConstraint("label_id", "position", name=op.f("uq_label_words_label_id_position")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("label_words")
    op.execute(
        sa.text(
            "DELETE FROM segment_labels WHERE label_version_id IN"
            " (SELECT id FROM label_versions WHERE name = :name)"
        ).bindparams(name=_SPEAKERS_VERSION)
    )
    op.execute(
        sa.text("DELETE FROM label_versions WHERE name = :name").bindparams(name=_SPEAKERS_VERSION)
    )
    op.drop_constraint(
        op.f("fk_segment_labels_diarization_run_id_diarization_runs"),
        "segment_labels",
        type_="foreignkey",
    )
    op.drop_column("segment_labels", "diarization_run_id")

    op.execute("DELETE FROM annotation_tasks WHERE queue = 'speakers'")
    op.drop_constraint(op.f(_QUEUE_CHECK), "annotation_tasks", type_="check")
    op.create_check_constraint(
        op.f(_QUEUE_CHECK), "annotation_tasks", "queue IN ('review', 'audit', 'error')"
    )
