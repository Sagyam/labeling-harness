"""Annotation state: queue tasks, label versions, labels and interaction events.

Three status fields, each with exactly one owner: ``segments.pipeline_status`` (lifecycle),
``annotation_tasks.status`` (queue state) and ``segment_labels.disposition`` (what the human
decided). There is deliberately no fourth, and no boolean that duplicates one.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utc_now_column, utc_optional_column
from app.models.content import AsrHypothesis, JsonB, Segment
from app.models.enums import (
    ACTIVE_TASK_STATUSES,
    DISPOSITIONS,
    EVENT_ACTIONS,
    QUEUE_NAMES,
    TASK_STATUSES,
    check_in,
)

_ACTIVE_TASK_PREDICATE = "status IN ('pending', 'in_progress')"


class AnnotationTask(Base):
    """One queued unit of human work for a segment.

    At most one *active* (pending or in progress) task may exist per segment; that rule is a
    partial unique index, enforced by Postgres rather than by application code.
    """

    __tablename__ = "annotation_tasks"
    __table_args__ = (
        CheckConstraint(check_in("queue", QUEUE_NAMES), name="queue_allowed"),
        CheckConstraint(check_in("status", TASK_STATUSES), name="status_allowed"),
        Index(
            "uq_annotation_tasks_active_segment",
            "segment_id",
            unique=True,
            postgresql_where=text(_ACTIVE_TASK_PREDICATE),
        ),
        Index(
            "ix_annotation_tasks_queue_status_priority_score",
            "queue",
            "status",
            "priority_score",
        ),
        Index("ix_annotation_tasks_segment_id", "segment_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    segment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )
    queue: Mapped[str] = mapped_column(String(16), nullable=False, default="review")
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    #: Which hypothesis preloads the editor. Rotated across systems for test-split segments.
    seed_hypothesis_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("asr_hypotheses.id", ondelete="SET NULL")
    )
    #: Per-component breakdown of ``priority_score``, so the UI can explain why this surfaced.
    reason_jsonb: Mapped[dict[str, Any] | None] = mapped_column(JsonB)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[dt.datetime] = utc_now_column()
    updated_at: Mapped[dt.datetime] = utc_now_column(onupdate=lambda: dt.datetime.now(dt.UTC))

    segment: Mapped[Segment] = relationship()
    seed_hypothesis: Mapped[AsrHypothesis | None] = relationship()


class LabelVersion(Base):
    """A named set of labels, carrying the transcript policy version it was produced under."""

    __tablename__ = "label_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[dt.datetime] = utc_now_column()


class SegmentLabel(Base):
    """A human decision about one segment. Append-only.

    The latest row per ``(segment_id, label_version_id)`` is current; earlier rows are history and
    are never updated or deleted.
    """

    __tablename__ = "segment_labels"
    __table_args__ = (
        CheckConstraint(check_in("disposition", DISPOSITIONS), name="disposition_allowed"),
        Index("ix_segment_labels_segment_id_label_version_id", "segment_id", "label_version_id"),
        Index("ix_segment_labels_label_version_id", "label_version_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    segment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )
    label_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("label_versions.id", ondelete="RESTRICT"), nullable=False
    )
    final_text: Mapped[str | None] = mapped_column(Text)
    disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    #: What the human was shown; nullable because a segment can have no hypothesis at all.
    seed_hypothesis_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("asr_hypotheses.id", ondelete="SET NULL")
    )
    annotator: Mapped[str] = mapped_column(String(64), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = utc_now_column()

    segment: Mapped[Segment] = relationship()
    label_version: Mapped[LabelVersion] = relationship()
    seed_hypothesis: Mapped[AsrHypothesis | None] = relationship()


class AnnotationEvent(Base):
    """Timing and action for one interaction, the raw material of the throughput report."""

    __tablename__ = "annotation_events"
    __table_args__ = (
        CheckConstraint(check_in("action", EVENT_ACTIONS), name="action_allowed"),
        Index("ix_annotation_events_segment_id", "segment_id"),
        Index("ix_annotation_events_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("annotation_tasks.id", ondelete="SET NULL")
    )
    segment_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )
    annotator: Mapped[str] = mapped_column(String(64), nullable=False)
    #: When the client fetched the task, reported by the client -- real human elapsed time.
    opened_at: Mapped[dt.datetime | None] = utc_optional_column()
    submitted_at: Mapped[dt.datetime | None] = utc_optional_column()
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[dt.datetime] = utc_now_column()


__all__ = [
    "ACTIVE_TASK_STATUSES",
    "AnnotationEvent",
    "AnnotationTask",
    "LabelVersion",
    "SegmentLabel",
]
