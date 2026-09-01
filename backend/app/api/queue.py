"""Triage list and progress counters."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_session, require_auth
from app.api.schemas import QueueRowOut
from app.api.serializers import serialize_queue_row
from app.models import AnnotationTask, Episode, Segment
from app.services.stats import collect_stats

router = APIRouter(tags=["queue"], dependencies=[Depends(require_auth)])


@router.get("/queue", response_model=list[QueueRowOut])
def get_queue(
    session: Session = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    episode: str | None = Query(default=None, description="episode external id"),
    min_priority: float | None = Query(default=None, ge=0.0, le=1.0),
    queue: str | None = Query(default=None, pattern="^(review|audit|error)$"),
) -> list[QueueRowOut]:
    """The triage list: pending work, highest priority first."""
    query = (
        sa.select(AnnotationTask)
        .join(Segment, Segment.id == AnnotationTask.segment_id)
        .join(Episode, Episode.id == Segment.episode_id)
        .options(
            selectinload(AnnotationTask.segment).selectinload(Segment.episode),
            selectinload(AnnotationTask.segment).selectinload(Segment.scores),
            selectinload(AnnotationTask.seed_hypothesis),
        )
        .where(AnnotationTask.status.in_(("pending", "in_progress")))
        .order_by(AnnotationTask.priority_score.desc(), AnnotationTask.id)
        .limit(limit)
        .offset(offset)
    )
    if episode:
        query = query.where(Episode.external_id == episode)
    if min_priority is not None:
        query = query.where(AnnotationTask.priority_score >= min_priority)
    query = query.where(AnnotationTask.queue == (queue or "review"))
    return [serialize_queue_row(task) for task in session.scalars(query)]


@router.get("/stats")
def get_stats(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Progress counters, disposition mix, throughput and projected completion."""
    return collect_stats(session)
