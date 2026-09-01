"""Writing human decisions.

Every decision writes three rows in one transaction: an append-only ``segment_labels`` row, an
``annotation_events`` row carrying the real human timing, and an ``audit_logs`` entry. A correction
never overwrites a hypothesis, and a re-label never updates an earlier label -- the latest row per
``(segment_id, label_version_id)`` is current.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import (
    AnnotationEvent,
    AnnotationTask,
    AuditLog,
    LabelVersion,
    Segment,
    SegmentLabel,
)
from app.models.enums import APPROVED_DISPOSITIONS


class LabelingError(RuntimeError):
    """A decision could not be recorded."""


@dataclass(frozen=True)
class Decision:
    """One human decision about one task."""

    disposition: str
    final_text: str | None = None
    notes: str | None = None
    annotator: str | None = None
    label_version: str | None = None
    opened_at: dt.datetime | None = None
    submitted_at: dt.datetime | None = None
    duration_ms: int | None = None


#: Which event action each disposition records.
_ACTION_FOR_DISPOSITION = {
    "accepted_unchanged": "accept",
    "edited": "edit",
    "unusable_audio": "flag",
    "uncertain": "flag",
}


def get_or_create_label_version(
    session: Session, name: str | None = None, settings: Settings | None = None
) -> LabelVersion:
    """Fetch the named label version, creating it with the configured policy version."""
    settings = settings or get_settings()
    name = name or settings.labels.default_label_version
    version = session.scalar(sa.select(LabelVersion).where(LabelVersion.name == name))
    if version is None:
        version = LabelVersion(
            name=name,
            policy_version=settings.labels.policy_version,
            description="created on first use",
        )
        session.add(version)
        session.flush()
    return version


def _timing(decision: Decision) -> tuple[dt.datetime | None, dt.datetime, int | None]:
    """Resolve event timing.

    ``duration_ms`` measures the human's elapsed time between opening the task and submitting it,
    which only the client can observe -- server processing time is not the quantity of interest.
    """
    submitted_at = decision.submitted_at or dt.datetime.now(dt.UTC)
    duration_ms = decision.duration_ms
    if duration_ms is None and decision.opened_at is not None:
        duration_ms = max(0, int((submitted_at - decision.opened_at).total_seconds() * 1000))
    return decision.opened_at, submitted_at, duration_ms


def record_decision(
    session: Session,
    task: AnnotationTask,
    decision: Decision,
    *,
    settings: Settings | None = None,
) -> SegmentLabel:
    """Record a decision about a task: label, event and audit entry, plus status updates.

    Args:
        session: Open session; the caller commits.
        task: The task being decided. Must be active.
        decision: What the human decided and how long it took.
        settings: Configuration override.

    Returns:
        The newly written :class:`SegmentLabel`.

    Raises:
        LabelingError: The task is already finished, or the disposition is unknown.
    """
    settings = settings or get_settings()
    if task.status in {"done", "skipped"}:
        raise LabelingError(f"task {task.id} is already {task.status}")
    if decision.disposition not in _ACTION_FOR_DISPOSITION:
        raise LabelingError(f"unknown disposition {decision.disposition!r}")

    version = get_or_create_label_version(session, decision.label_version, settings)
    annotator = decision.annotator or settings.labels.default_annotator
    opened_at, submitted_at, duration_ms = _timing(decision)

    final_text = decision.final_text
    if decision.disposition == "accepted_unchanged":
        if final_text is None and task.seed_hypothesis is not None:
            final_text = task.seed_hypothesis.text_raw
        if final_text is None:
            raise LabelingError(
                f"task {task.id} has no seed hypothesis, so there is nothing to accept"
            )

    label = SegmentLabel(
        segment_id=task.segment_id,
        label_version_id=version.id,
        final_text=final_text,
        disposition=decision.disposition,
        seed_hypothesis_id=task.seed_hypothesis_id,
        annotator=annotator,
        notes=decision.notes,
    )
    session.add(label)
    session.flush()

    session.add(
        AnnotationEvent(
            task_id=task.id,
            segment_id=task.segment_id,
            annotator=annotator,
            opened_at=opened_at,
            submitted_at=submitted_at,
            duration_ms=duration_ms,
            action=_ACTION_FOR_DISPOSITION[decision.disposition],
        )
    )

    task.status = "done"
    segment = session.get(Segment, task.segment_id)
    if segment is not None:
        segment.pipeline_status = (
            "labeled" if decision.disposition in APPROVED_DISPOSITIONS else "excluded"
        )

    session.add(
        AuditLog(
            entity_type="segment_labels",
            entity_id=str(label.id),
            action="insert",
            actor=annotator,
            old_values_jsonb=None,
            new_values_jsonb={
                "task_id": task.id,
                "segment_id": task.segment_id,
                "disposition": decision.disposition,
                "final_text": final_text,
                "label_version": version.name,
                "policy_version": version.policy_version,
                "seed_hypothesis_id": task.seed_hypothesis_id,
                "duration_ms": duration_ms,
            },
        )
    )
    session.flush()
    return label


def record_skip(
    session: Session,
    task: AnnotationTask,
    *,
    annotator: str | None = None,
    opened_at: dt.datetime | None = None,
    duration_ms: int | None = None,
    settings: Settings | None = None,
) -> AnnotationEvent:
    """Defer a task without deciding anything.

    A skip writes an event and an audit entry but **no label**: the annotator did not make a
    judgement about the transcript, and inventing a label row for one would corrupt every
    disposition statistic. The segment stays queued and returns on the next queue build.
    """
    settings = settings or get_settings()
    if task.status in {"done", "skipped"}:
        raise LabelingError(f"task {task.id} is already {task.status}")

    annotator = annotator or settings.labels.default_annotator
    submitted_at = dt.datetime.now(dt.UTC)
    if duration_ms is None and opened_at is not None:
        duration_ms = max(0, int((submitted_at - opened_at).total_seconds() * 1000))

    event = AnnotationEvent(
        task_id=task.id,
        segment_id=task.segment_id,
        annotator=annotator,
        opened_at=opened_at,
        submitted_at=submitted_at,
        duration_ms=duration_ms,
        action="skip",
    )
    session.add(event)
    task.status = "skipped"
    session.add(
        AuditLog(
            entity_type="annotation_tasks",
            entity_id=str(task.id),
            action="skip",
            actor=annotator,
            old_values_jsonb={"status": "pending"},
            new_values_jsonb={"status": "skipped"},
        )
    )
    session.flush()
    return event


def latest_label(
    session: Session, segment_id: int, label_version_id: int | None = None
) -> SegmentLabel | None:
    """The current label for a segment: the most recent row, since labels are append-only."""
    query = sa.select(SegmentLabel).where(SegmentLabel.segment_id == segment_id)
    if label_version_id is not None:
        query = query.where(SegmentLabel.label_version_id == label_version_id)
    return session.scalars(
        query.order_by(SegmentLabel.created_at.desc(), SegmentLabel.id.desc()).limit(1)
    ).first()
