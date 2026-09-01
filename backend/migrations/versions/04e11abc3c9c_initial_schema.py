"""Initial schema: provenance, content, annotation and operational tables.

Implements every table in ARCHITECTURE.md "Data model". Status columns are guarded by CHECK
constraints, and at most one active annotation task per segment is guaranteed by the partial
unique index ``uq_annotation_tasks_active_segment``.

Revision ID: 04e11abc3c9c
Revises:
Create Date: 2026-09-01 15:49:19.336874

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "04e11abc3c9c"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "asr_systems",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("system_id", sa.String(length=128), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_asr_systems")),
        sa.UniqueConstraint("system_id", name=op.f("uq_asr_systems_system_id")),
    )
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("actor", sa.String(length=64), nullable=False),
        sa.Column(
            "old_values_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "new_values_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
    )
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)
    op.create_index(
        "ix_audit_logs_entity_type_entity_id",
        "audit_logs",
        ["entity_type", "entity_id"],
        unique=False,
    )
    op.create_table(
        "episodes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("show_id", sa.String(length=255), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("published_at", sa.Date(), nullable=True),
        sa.Column("source_audio_checksum", sa.String(length=128), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("split_seed", sa.Integer(), nullable=True),
        sa.Column("split_assigned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "split IN ('train', 'val', 'test', 'unassigned')",
            name=op.f("ck_episodes_split_allowed"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_episodes")),
        sa.UniqueConstraint("external_id", name=op.f("uq_episodes_external_id")),
    )
    op.create_index("ix_episodes_show_id", "episodes", ["show_id"], unique=False)
    op.create_index("ix_episodes_split", "episodes", ["split"], unique=False)
    op.create_table(
        "import_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("pipeline_version", sa.String(length=64), nullable=True),
        sa.Column("pipeline_commit", sa.String(length=64), nullable=True),
        sa.Column("segments_inserted", sa.Integer(), nullable=False),
        sa.Column("segments_skipped", sa.Integer(), nullable=False),
        sa.Column("hypotheses_inserted", sa.Integer(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'dry_run')",
            name=op.f("ck_import_runs_status_allowed"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_import_runs")),
    )
    op.create_table(
        "label_versions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_label_versions")),
        sa.UniqueConstraint("name", name=op.f("uq_label_versions_name")),
    )
    op.create_table(
        "llm_requests",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("route", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=True),
        sa.Column("input_summary", sa.Text(), nullable=True),
        sa.Column(
            "output_json",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_requests")),
    )
    op.create_index("ix_llm_requests_request_hash", "llm_requests", ["request_hash"], unique=False)
    op.create_index(
        "ix_llm_requests_route_created_at", "llm_requests", ["route", "created_at"], unique=False
    )
    op.create_table(
        "translit_cache",
        sa.Column("latin_token", sa.String(length=128), nullable=False),
        sa.Column(
            "candidates_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("latin_token", name=op.f("pk_translit_cache")),
    )
    op.create_table(
        "segments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("episode_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("speaker_id", sa.String(length=64), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("clip_object_key", sa.Text(), nullable=False),
        sa.Column("clip_checksum", sa.String(length=128), nullable=False),
        sa.Column("peaks_object_key", sa.Text(), nullable=True),
        sa.Column("p_en", sa.Float(), nullable=True),
        sa.Column("lid", sa.String(length=16), nullable=True),
        sa.Column("pipeline_status", sa.String(length=16), nullable=False),
        sa.Column("import_run_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "pipeline_status IN ('imported', 'queued', 'labeled', 'excluded')",
            name=op.f("ck_segments_status_allowed"),
        ),
        sa.CheckConstraint("end_time > start_time", name=op.f("ck_segments_end_after_start")),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["episodes.id"],
            name=op.f("fk_segments_episode_id_episodes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["import_runs.id"],
            name=op.f("fk_segments_import_run_id_import_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_segments")),
        sa.UniqueConstraint("external_id", name=op.f("uq_segments_external_id")),
    )
    op.create_index("ix_segments_episode_id", "segments", ["episode_id"], unique=False)
    op.create_index("ix_segments_pipeline_status", "segments", ["pipeline_status"], unique=False)
    op.create_table(
        "asr_hypotheses",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("asr_system_id", sa.BigInteger(), nullable=False),
        sa.Column("text_raw", sa.Text(), nullable=False),
        sa.Column("text_normalized", sa.Text(), nullable=True),
        sa.Column("avg_logprob", sa.Float(), nullable=True),
        sa.Column("no_speech_prob", sa.Float(), nullable=True),
        sa.Column(
            "metadata_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["asr_system_id"],
            ["asr_systems.id"],
            name=op.f("fk_asr_hypotheses_asr_system_id_asr_systems"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_asr_hypotheses_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_asr_hypotheses")),
        sa.UniqueConstraint("segment_id", "asr_system_id", name="uq_asr_hypotheses_segment_system"),
    )
    op.create_index("ix_asr_hypotheses_segment_id", "asr_hypotheses", ["segment_id"], unique=False)
    op.create_table(
        "segment_scores",
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("cer_between_hypotheses", sa.Float(), nullable=True),
        sa.Column("word_disagreement_rate", sa.Float(), nullable=True),
        sa.Column("script_conflict_rate", sa.Float(), nullable=True),
        sa.Column("code_switch_density", sa.Float(), nullable=True),
        sa.Column(
            "flags_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column(
            "imported_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_segment_scores_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("segment_id", name=op.f("pk_segment_scores")),
    )
    op.create_table(
        "annotation_tasks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("queue", sa.String(length=16), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=False),
        sa.Column("seed_hypothesis_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "reason_jsonb",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "queue IN ('review', 'audit', 'error')", name=op.f("ck_annotation_tasks_queue_allowed")
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'in_progress', 'done', 'skipped')",
            name=op.f("ck_annotation_tasks_status_allowed"),
        ),
        sa.ForeignKeyConstraint(
            ["seed_hypothesis_id"],
            ["asr_hypotheses.id"],
            name=op.f("fk_annotation_tasks_seed_hypothesis_id_asr_hypotheses"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_annotation_tasks_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_annotation_tasks")),
    )
    op.create_index(
        "ix_annotation_tasks_queue_status_priority_score",
        "annotation_tasks",
        ["queue", "status", "priority_score"],
        unique=False,
    )
    op.create_index(
        "ix_annotation_tasks_segment_id", "annotation_tasks", ["segment_id"], unique=False
    )
    op.create_index(
        "uq_annotation_tasks_active_segment",
        "annotation_tasks",
        ["segment_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'in_progress')"),
    )
    op.create_table(
        "hypothesis_words",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("hypothesis_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("word_raw", sa.Text(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("predicted_language", sa.String(length=16), nullable=True),
        sa.Column("predicted_script", sa.String(length=16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["hypothesis_id"],
            ["asr_hypotheses.id"],
            name=op.f("fk_hypothesis_words_hypothesis_id_asr_hypotheses"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_hypothesis_words")),
    )
    op.create_index(
        "ix_hypothesis_words_hypothesis_id", "hypothesis_words", ["hypothesis_id"], unique=False
    )
    op.create_table(
        "segment_labels",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("label_version_id", sa.BigInteger(), nullable=False),
        sa.Column("final_text", sa.Text(), nullable=True),
        sa.Column("disposition", sa.String(length=32), nullable=False),
        sa.Column("seed_hypothesis_id", sa.BigInteger(), nullable=True),
        sa.Column("annotator", sa.String(length=64), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "disposition IN ('accepted_unchanged', 'edited', 'unusable_audio', 'uncertain')",
            name=op.f("ck_segment_labels_disposition_allowed"),
        ),
        sa.ForeignKeyConstraint(
            ["label_version_id"],
            ["label_versions.id"],
            name=op.f("fk_segment_labels_label_version_id_label_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["seed_hypothesis_id"],
            ["asr_hypotheses.id"],
            name=op.f("fk_segment_labels_seed_hypothesis_id_asr_hypotheses"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_segment_labels_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_segment_labels")),
    )
    op.create_index(
        "ix_segment_labels_label_version_id", "segment_labels", ["label_version_id"], unique=False
    )
    op.create_index(
        "ix_segment_labels_segment_id_label_version_id",
        "segment_labels",
        ["segment_id", "label_version_id"],
        unique=False,
    )
    op.create_table(
        "annotation_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.BigInteger(), nullable=True),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("annotator", sa.String(length=64), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action IN ('accept', 'edit', 'skip', 'flag', 'reopen')",
            name=op.f("ck_annotation_events_action_allowed"),
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
            name=op.f("fk_annotation_events_segment_id_segments"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["annotation_tasks.id"],
            name=op.f("fk_annotation_events_task_id_annotation_tasks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_annotation_events")),
    )
    op.create_index(
        "ix_annotation_events_created_at", "annotation_events", ["created_at"], unique=False
    )
    op.create_index(
        "ix_annotation_events_segment_id", "annotation_events", ["segment_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_annotation_events_segment_id", table_name="annotation_events")
    op.drop_index("ix_annotation_events_created_at", table_name="annotation_events")
    op.drop_table("annotation_events")
    op.drop_index("ix_segment_labels_segment_id_label_version_id", table_name="segment_labels")
    op.drop_index("ix_segment_labels_label_version_id", table_name="segment_labels")
    op.drop_table("segment_labels")
    op.drop_index("ix_hypothesis_words_hypothesis_id", table_name="hypothesis_words")
    op.drop_table("hypothesis_words")
    op.drop_index(
        "uq_annotation_tasks_active_segment",
        table_name="annotation_tasks",
        postgresql_where=sa.text("status IN ('pending', 'in_progress')"),
    )
    op.drop_index("ix_annotation_tasks_segment_id", table_name="annotation_tasks")
    op.drop_index("ix_annotation_tasks_queue_status_priority_score", table_name="annotation_tasks")
    op.drop_table("annotation_tasks")
    op.drop_table("segment_scores")
    op.drop_index("ix_asr_hypotheses_segment_id", table_name="asr_hypotheses")
    op.drop_table("asr_hypotheses")
    op.drop_index("ix_segments_pipeline_status", table_name="segments")
    op.drop_index("ix_segments_episode_id", table_name="segments")
    op.drop_table("segments")
    op.drop_table("translit_cache")
    op.drop_index("ix_llm_requests_route_created_at", table_name="llm_requests")
    op.drop_index("ix_llm_requests_request_hash", table_name="llm_requests")
    op.drop_table("llm_requests")
    op.drop_table("label_versions")
    op.drop_table("import_runs")
    op.drop_index("ix_episodes_split", table_name="episodes")
    op.drop_index("ix_episodes_show_id", table_name="episodes")
    op.drop_table("episodes")
    op.drop_index("ix_audit_logs_entity_type_entity_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_table("asr_systems")
