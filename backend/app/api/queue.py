"""Triage list and progress counters."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_session, require_auth
from app.api.schemas import QueueRowOut
from app.api.serializers import serialize_queue_row
from app.models import AnnotationTask, Episode, Segment, SegmentScore
from app.services.inventory import collect_inventory
from app.services.report import collect_report
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
    sort_by: str = Query(
        default="priority",
        pattern="^(priority|cmi|disagreement|duration|pot)$",
        description="Field to sort by",
    ),
    sort_order: str = Query(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort direction ('asc' or 'desc')",
    ),
) -> list[QueueRowOut]:
    """The triage list: pending work, with customizable sorting."""
    query = (
        sa.select(AnnotationTask)
        .join(Segment, Segment.id == AnnotationTask.segment_id)
        .outerjoin(SegmentScore, SegmentScore.segment_id == Segment.id)
        .join(Episode, Episode.id == Segment.episode_id)
        .options(
            selectinload(AnnotationTask.segment).selectinload(Segment.episode),
            selectinload(AnnotationTask.segment).selectinload(Segment.scores),
            selectinload(AnnotationTask.seed_hypothesis),
        )
        .where(AnnotationTask.status.in_(("pending", "in_progress")))
    )
    if episode:
        query = query.where(Episode.external_id == episode)
    if min_priority is not None:
        query = query.where(AnnotationTask.priority_score >= min_priority)
    query = query.where(AnnotationTask.queue == (queue or "review"))

    is_desc = sort_order == "desc"
    order_clauses: list[Any] = []
    if sort_by == "cmi":
        col = SegmentScore.code_switch_density
        order_clauses.append(col.desc().nulls_last() if is_desc else col.asc().nulls_last())
    elif sort_by == "disagreement":
        col = SegmentScore.word_disagreement_rate
        order_clauses.append(col.desc().nulls_last() if is_desc else col.asc().nulls_last())
    elif sort_by == "duration":
        col = Segment.duration_seconds
        order_clauses.append(col.desc() if is_desc else col.asc())
    elif sort_by == "pot":
        col = sa.case((Segment.pot == "gold", 1), else_=0)
        order_clauses.append(col.desc() if is_desc else col.asc())
    else:  # priority
        col = AnnotationTask.priority_score
        order_clauses.append(col.desc() if is_desc else col.asc())

    if sort_by != "priority":
        order_clauses.append(AnnotationTask.priority_score.desc())
    order_clauses.append(AnnotationTask.id)

    query = query.order_by(*order_clauses).limit(limit).offset(offset)
    return [serialize_queue_row(task) for task in session.scalars(query)]


@router.get("/stats")
def get_stats(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Progress counters, disposition mix, throughput and projected completion."""
    return collect_stats(session)


@router.get("/stats/report")
def get_report(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Comprehensive status report: corpus, throughput, scores, split balance, and trends."""
    return collect_report(session)


@router.get("/stats/inventory")
def get_inventory(session: Session = Depends(get_session)) -> dict[str, Any]:
    """What the corpus contains, what it is missing, and what to record next (D69)."""
    return collect_inventory(session).as_dict()
