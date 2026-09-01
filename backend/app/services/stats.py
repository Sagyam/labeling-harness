"""Progress and throughput counters.

These back both the always-visible progress display in the UI and the standalone status report, so
they are computed here once rather than in two places.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import (
    AnnotationEvent,
    AnnotationTask,
    Episode,
    Segment,
    SegmentLabel,
)
from app.models.enums import DISPOSITIONS, PIPELINE_STATUSES, TASK_STATUSES


def _counts(session: Session, column, allowed: tuple[str, ...], source) -> dict[str, int]:
    rows = session.execute(sa.select(column, sa.func.count()).select_from(source).group_by(column))
    counts = dict.fromkeys(allowed, 0)
    for value, count in rows:
        counts[value] = count
    return counts


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def latest_labels_subquery():
    """Subquery yielding the current label id per (segment, label version).

    Labels are append-only, so "current" means the newest row, not the only row.
    """
    ranked = sa.select(
        SegmentLabel.id.label("id"),
        SegmentLabel.segment_id.label("segment_id"),
        SegmentLabel.label_version_id.label("label_version_id"),
        SegmentLabel.disposition.label("disposition"),
        SegmentLabel.final_text.label("final_text"),
        sa.func.row_number()
        .over(
            partition_by=(SegmentLabel.segment_id, SegmentLabel.label_version_id),
            order_by=(SegmentLabel.created_at.desc(), SegmentLabel.id.desc()),
        )
        .label("rank"),
    ).subquery()
    return sa.select(ranked).where(ranked.c.rank == 1).subquery()


def collect_stats(session: Session, *, session_since: dt.datetime | None = None) -> dict[str, Any]:
    """Progress counters, disposition mix, throughput and a projected finish time.

    Args:
        session: Open session.
        session_since: Start of "this session" for the session counters. Defaults to the last
            24 hours, which is what a single annotator's working day looks like.
    """
    since = session_since or dt.datetime.now(dt.UTC) - dt.timedelta(hours=24)

    segments_by_status = _counts(session, Segment.pipeline_status, PIPELINE_STATUSES, Segment)
    tasks_by_status = _counts(session, AnnotationTask.status, TASK_STATUSES, AnnotationTask)
    tasks_by_queue = dict(
        session.execute(
            sa.select(AnnotationTask.queue, sa.func.count())
            .where(AnnotationTask.status.in_(("pending", "in_progress")))
            .group_by(AnnotationTask.queue)
        ).all()
    )

    current = latest_labels_subquery()
    labels_by_disposition = dict.fromkeys(DISPOSITIONS, 0)
    for disposition, count in session.execute(
        sa.select(current.c.disposition, sa.func.count()).group_by(current.c.disposition)
    ):
        labels_by_disposition[disposition] = count

    labeled_total = sum(labels_by_disposition.values())
    accepted = labels_by_disposition["accepted_unchanged"]
    accept_rate = (accepted / labeled_total) if labeled_total else None

    durations = [
        row / 1000
        for (row,) in session.execute(
            sa.select(AnnotationEvent.duration_ms).where(AnnotationEvent.duration_ms.is_not(None))
        )
    ]
    session_events = list(
        session.execute(
            sa.select(AnnotationEvent.duration_ms).where(
                AnnotationEvent.created_at >= since,
                AnnotationEvent.action != "skip",
            )
        )
    )
    session_durations = [row / 1000 for (row,) in session_events if row is not None]

    median_seconds = _median(durations)
    backlog = tasks_by_status["pending"] + tasks_by_status["in_progress"]
    projected_seconds = median_seconds * backlog if median_seconds else None

    total_seconds = session.scalar(
        sa.select(sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0))
    )

    return {
        "episodes": session.scalar(sa.select(sa.func.count()).select_from(Episode)),
        "segments": {
            "total": sum(segments_by_status.values()),
            **segments_by_status,
        },
        "audio_hours": round(float(total_seconds or 0.0) / 3600, 3),
        "tasks": {"total": sum(tasks_by_status.values()), **tasks_by_status},
        "queues": {"review": 0, "audit": 0, "error": 0, **tasks_by_queue},
        "labels": {"total": labeled_total, **labels_by_disposition},
        "accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
        "throughput": {
            "median_seconds_per_segment": round(median_seconds, 2) if median_seconds else None,
            "labeled_total": labeled_total,
            "backlog": backlog,
            "projected_seconds_to_finish": round(projected_seconds, 1)
            if projected_seconds
            else None,
        },
        "session": {
            "since": since.isoformat(),
            "completed": len(session_events),
            "median_seconds_per_segment": round(_median(session_durations), 2)
            if session_durations
            else None,
            "elapsed_seconds": round(sum(session_durations), 1) if session_durations else 0.0,
        },
    }
