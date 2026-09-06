"""Task endpoints: serving work and recording decisions."""

from __future__ import annotations

import datetime as dt

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_config, get_session, require_auth
from app.api.schemas import (
    AcceptIn,
    BulkAcceptIn,
    BulkAcceptOut,
    DecisionOut,
    DisputeAlternativeOut,
    DisputeOut,
    FlagIn,
    LabelIn,
    SkipIn,
    TaskOut,
)
from app.api.serializers import serialize_segment
from app.config import Settings
from app.models import AnnotationTask, AsrHypothesis, Episode, Segment
from app.models.enums import ACTIVE_TASK_STATUSES
from app.services.consensus import (
    ConsensusHypothesis,
    ConsensusWord,
    build_slots,
    seed_disputes,
)
from app.services.labeling import Decision, LabelingError, record_decision, record_skip

router = APIRouter(tags=["tasks"], dependencies=[Depends(require_auth)])


def _load_task(session: Session, task_id: int) -> AnnotationTask:
    task = session.scalars(
        sa.select(AnnotationTask)
        .options(
            selectinload(AnnotationTask.segment).selectinload(Segment.episode),
            selectinload(AnnotationTask.segment).selectinload(Segment.scores),
            selectinload(AnnotationTask.segment)
            .selectinload(Segment.hypotheses)
            .selectinload(AsrHypothesis.words),
            selectinload(AnnotationTask.segment)
            .selectinload(Segment.hypotheses)
            .selectinload(AsrHypothesis.system),
            selectinload(AnnotationTask.seed_hypothesis),
        )
        .where(AnnotationTask.id == task_id)
    ).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    return task


def _disputes_for(task: AnnotationTask) -> list[DisputeOut]:
    """Where the other systems contradicted the hypothesis this task will show.

    Computed here rather than stored: it is a pure function of `hypothesis_words`, which is
    immutable once imported, and it depends on the *seed*, which is chosen per task. Persisting
    it would mean writing a per-seed answer at ingest, before the seed exists.
    """
    seed = task.seed_hypothesis
    if seed is None:
        return []
    hypotheses = [
        ConsensusHypothesis(
            system_id=h.system.system_id,
            words=[
                ConsensusWord(
                    position=w.position, word=w.word_raw, start=w.start_time, end=w.end_time
                )
                for w in h.words
            ],
        )
        for h in task.segment.hypotheses
    ]
    disputes = seed_disputes(build_slots(hypotheses), seed_system_id=seed.system.system_id)
    return [
        DisputeOut(
            seed_position=d.seed_position,
            seed_word=d.seed_word,
            start_time=d.start,
            end_time=d.end,
            alternatives=[
                DisputeAlternativeOut(system_id=a.system_id, word=a.word) for a in d.alternatives
            ],
        )
        for d in disputes
    ]


def _serialize_task(session: Session, task: AnnotationTask) -> TaskOut:
    seed = task.seed_hypothesis
    return TaskOut(
        id=task.id,
        segment_id=task.segment_id,
        queue=task.queue,
        status=task.status,
        priority_score=task.priority_score,
        reason=task.reason_jsonb,
        seed_hypothesis_id=task.seed_hypothesis_id,
        seed_system_id=seed.system.system_id if seed else None,
        served_at=dt.datetime.now(dt.UTC),
        segment=serialize_segment(session, task.segment),
        disputes=_disputes_for(task),
    )


@router.get("/tasks/next", response_model=TaskOut)
def next_task(
    session: Session = Depends(get_session),
    queue: str = Query(default="review", pattern="^(review|audit|error)$"),
    episode: str | None = Query(default=None),
) -> TaskOut:
    """Serve the highest-priority pending task and mark it in progress.

    Marking it here is what makes resume work: reopening the app returns the same task rather than
    a fresh one, so the annotator lands exactly where they left off.
    """
    query = (
        sa.select(AnnotationTask)
        .join(Segment, Segment.id == AnnotationTask.segment_id)
        .options(
            selectinload(AnnotationTask.segment).selectinload(Segment.episode),
            selectinload(AnnotationTask.segment).selectinload(Segment.scores),
            selectinload(AnnotationTask.segment)
            .selectinload(Segment.hypotheses)
            .selectinload(AsrHypothesis.words),
            selectinload(AnnotationTask.segment)
            .selectinload(Segment.hypotheses)
            .selectinload(AsrHypothesis.system),
            selectinload(AnnotationTask.seed_hypothesis),
        )
        .where(AnnotationTask.status.in_(ACTIVE_TASK_STATUSES), AnnotationTask.queue == queue)
        .order_by(
            sa.case((AnnotationTask.status == "in_progress", 0), else_=1),
            AnnotationTask.priority_score.desc(),
            AnnotationTask.id,
        )
        .limit(1)
    )
    if episode:
        query = query.join(Episode, Episode.id == Segment.episode_id).where(
            Episode.external_id == episode
        )

    task = session.scalars(query).first()
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="queue is empty")
    if task.status == "pending":
        task.status = "in_progress"
    # Committed here rather than at teardown, so the status this response reports is the status
    # the annotator's next request will read. See `_decide` for what the window costs.
    session.commit()
    return _serialize_task(session, task)


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int, session: Session = Depends(get_session)) -> TaskOut:
    """Fetch one task without changing its status."""
    return _serialize_task(session, _load_task(session, task_id))


def _decide(
    session: Session,
    task: AnnotationTask,
    decision: Decision,
    settings: Settings,
    *,
    commit: bool = True,
) -> DecisionOut:
    """Record one decision and, by default, commit it before the caller builds a response.

    The commit cannot be left to the session dependency. FastAPI closes a ``yield`` dependency
    *after* the response has been sent, so a decision that commits at teardown is still invisible
    when the browser -- which sends ``GET /tasks/next`` the moment the save resolves -- asks for
    the next clip. That read then finds the task it just finished still active, and because
    ``/tasks/next`` serves ``in_progress`` first, the annotator is handed the same clip back and
    the editor appears not to advance. Worse, serving it flips the status to ``in_progress``,
    which lands after the ``done`` from the save and overwrites it, so the next save writes a
    second label for a clip already decided.

    ``commit=False`` is for callers batching several decisions into one transaction.
    """
    try:
        label = record_decision(session, task, decision, settings=settings)
    except LabelingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if commit:
        session.commit()
    event_duration = decision.duration_ms
    if event_duration is None and decision.opened_at is not None:
        event_duration = max(
            0, int((dt.datetime.now(dt.UTC) - decision.opened_at).total_seconds() * 1000)
        )
    return DecisionOut(
        task_id=task.id,
        segment_id=task.segment_id,
        label_id=label.id,
        disposition=label.disposition,
        verification_tier=label.verification_tier,
        task_status=task.status,
        duration_ms=event_duration,
    )


@router.post("/tasks/{task_id}/accept", response_model=DecisionOut)
def accept_task(
    task_id: int,
    body: AcceptIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> DecisionOut:
    """Accept the seed hypothesis unchanged -- the dominant interaction."""
    task = _load_task(session, task_id)
    return _decide(
        session,
        task,
        Decision(
            disposition="accepted_unchanged",
            annotator=body.annotator,
            label_version=body.label_version,
            notes=body.notes,
            opened_at=body.opened_at,
            duration_ms=body.duration_ms,
            verification_tier=body.verification_tier,
        ),
        settings,
    )


@router.post("/tasks/{task_id}/label", response_model=DecisionOut)
def label_task(
    task_id: int,
    body: LabelIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> DecisionOut:
    """Save corrected transcript text."""
    task = _load_task(session, task_id)
    return _decide(
        session,
        task,
        Decision(
            disposition="edited",
            final_text=body.final_text,
            annotator=body.annotator,
            label_version=body.label_version,
            notes=body.notes,
            opened_at=body.opened_at,
            duration_ms=body.duration_ms,
            verification_tier=body.verification_tier,
        ),
        settings,
    )


@router.post("/tasks/{task_id}/flag", response_model=DecisionOut)
def flag_task(
    task_id: int,
    body: FlagIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> DecisionOut:
    """Mark a segment ``unusable_audio`` or ``uncertain``."""
    task = _load_task(session, task_id)
    return _decide(
        session,
        task,
        Decision(
            disposition=body.disposition,
            annotator=body.annotator,
            label_version=body.label_version,
            notes=body.notes,
            opened_at=body.opened_at,
            duration_ms=body.duration_ms,
            verification_tier=body.verification_tier,
        ),
        settings,
    )


@router.post("/tasks/{task_id}/skip", response_model=DecisionOut)
def skip_task(
    task_id: int,
    body: SkipIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> DecisionOut:
    """Defer a task. Writes an event but no label -- a skip is not a decision about the text."""
    task = _load_task(session, task_id)
    try:
        event = record_skip(
            session,
            task,
            annotator=body.annotator,
            opened_at=body.opened_at,
            duration_ms=body.duration_ms,
            settings=settings,
        )
    except LabelingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    session.commit()
    return DecisionOut(
        task_id=task.id,
        segment_id=task.segment_id,
        label_id=None,
        disposition=None,
        task_status=task.status,
        duration_ms=event.duration_ms,
    )


@router.post("/tasks/bulk-accept", response_model=BulkAcceptOut)
def bulk_accept(
    body: BulkAcceptIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> BulkAcceptOut:
    """Accept several tasks in one transaction: all of them, or none."""
    accepted: list[DecisionOut] = []
    for task_id in body.task_ids:
        task = _load_task(session, task_id)
        accepted.append(
            _decide(
                session,
                task,
                Decision(
                    disposition="accepted_unchanged",
                    annotator=body.annotator,
                    label_version=body.label_version,
                    notes=body.notes,
                    opened_at=body.opened_at,
                    duration_ms=body.duration_ms,
                    verification_tier=body.verification_tier,
                ),
                settings,
                # All of them or none: one commit for the batch, once every task has been
                # recorded without raising.
                commit=False,
            )
        )
    session.commit()
    return BulkAcceptOut(accepted=accepted, count=len(accepted))
