"""Episode management and deletion endpoints."""

from __future__ import annotations

import shutil
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_object_storage, get_session, require_auth
from app.models import AnnotationTask, Episode, Segment
from app.storage import ObjectStorage
from app.storage.local import LocalFilesystemStorage

router = APIRouter(tags=["episodes"], dependencies=[Depends(require_auth)])


class EpisodeSummary(BaseModel):
    id: int
    external_id: str
    title: str | None = None
    show_id: str | None = None
    duration_seconds: float | None = None
    split: str = "unassigned"
    segment_count: int = 0
    labeled_count: int = 0
    pending_count: int = 0


class EpisodeSegmentSummary(BaseModel):
    id: int
    external_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    pipeline_status: str
    task_status: str | None = None
    seed_text: str | None = None
    flags: list[str] = []
    cmi: float | None = None
    word_disagreement_rate: float | None = None
    audio_url: str
    peaks_url: str | None = None


def _find_episode(session: Session, episode_id: str) -> Episode:
    query = sa.select(Episode)
    if episode_id.isdigit():
        query = query.where(
            sa.or_(Episode.id == int(episode_id), Episode.external_id == episode_id)
        )
    else:
        query = query.where(Episode.external_id == episode_id)
    episode = session.scalar(query)
    if episode is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Episode {episode_id!r} not found",
        )
    return episode


@router.get("/episodes", response_model=list[EpisodeSummary])
def list_episodes(session: Session = Depends(get_session)) -> list[EpisodeSummary]:
    """List all episodes with segment counts and progress."""
    episodes = session.scalars(sa.select(Episode).order_by(Episode.id.desc())).all()
    if not episodes:
        return []

    # Aggregate counts per episode
    counts_query = (
        sa.select(
            Segment.episode_id,
            sa.func.count(Segment.id).label("total_segments"),
            sa.func.count(
                sa.case((Segment.pipeline_status == "labeled", Segment.id), else_=None)
            ).label("labeled_segments"),
            sa.func.count(
                sa.case(
                    (
                        sa.and_(
                            AnnotationTask.status.in_(["pending", "in_progress"]),
                            Segment.pipeline_status != "labeled",
                        ),
                        Segment.id,
                    ),
                    else_=None,
                )
            ).label("pending_segments"),
        )
        .outerjoin(AnnotationTask, AnnotationTask.segment_id == Segment.id)
        .group_by(Segment.episode_id)
    )

    counts_map = {row.episode_id: row for row in session.execute(counts_query).all()}

    results: list[EpisodeSummary] = []
    for ep in episodes:
        stats = counts_map.get(ep.id)
        results.append(
            EpisodeSummary(
                id=ep.id,
                external_id=ep.external_id,
                title=ep.title,
                show_id=ep.show_id,
                duration_seconds=ep.duration_seconds,
                split=ep.split,
                segment_count=stats.total_segments if stats else 0,
                labeled_count=stats.labeled_segments if stats else 0,
                pending_count=stats.pending_segments if stats else 0,
            )
        )
    return results


@router.get("/episodes/{episode_id}/segments", response_model=list[EpisodeSegmentSummary])
def list_episode_segments(
    episode_id: str, session: Session = Depends(get_session)
) -> list[EpisodeSegmentSummary]:
    """List all segments in an episode with audio URLs, flags, and transcripts."""
    episode = _find_episode(session, episode_id)

    segments = session.scalars(
        sa.select(Segment)
        .options(
            selectinload(Segment.scores),
            selectinload(Segment.hypotheses),
        )
        .where(Segment.episode_id == episode.id)
        .order_by(Segment.start_time.asc())
    ).all()

    # Load active tasks for these segments
    seg_ids = [s.id for s in segments]
    task_map: dict[int, str] = {}
    if seg_ids:
        tasks = session.execute(
            sa.select(AnnotationTask.segment_id, AnnotationTask.status).where(
                AnnotationTask.segment_id.in_(seg_ids)
            )
        ).all()
        task_map = {row[0]: row[1] for row in tasks}

    results: list[EpisodeSegmentSummary] = []
    for seg in segments:
        hyp = seg.hypotheses[0] if seg.hypotheses else None
        scores = seg.scores
        results.append(
            EpisodeSegmentSummary(
                id=seg.id,
                external_id=seg.external_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration_seconds=seg.duration_seconds,
                pipeline_status=seg.pipeline_status,
                task_status=task_map.get(seg.id),
                seed_text=hyp.text_raw if hyp else None,
                flags=scores.flags_jsonb if scores and scores.flags_jsonb else [],
                cmi=round(scores.code_switch_density * 100, 1)
                if (scores and scores.code_switch_density is not None)
                else None,
                word_disagreement_rate=scores.word_disagreement_rate if scores else None,
                audio_url=f"/segments/{seg.id}/audio",
                peaks_url=f"/segments/{seg.id}/peaks" if seg.peaks_object_key else None,
            )
        )
    return results


@router.delete("/episodes/{episode_id}")
def delete_episode(
    episode_id: str,
    session: Session = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
) -> dict[str, Any]:
    """Delete an episode, all its child records, and its audio/peak clips from storage."""
    episode = _find_episode(session, episode_id)
    external_id = episode.external_id
    ep_id = episode.id

    # Fetch segments to delete storage objects
    segments = session.scalars(sa.select(Segment).where(Segment.episode_id == ep_id)).all()
    deleted_segments = len(segments)

    for seg in segments:
        try:
            storage.delete(seg.clip_object_key)
            if seg.peaks_object_key:
                storage.delete(seg.peaks_object_key)
        except Exception:
            pass

    # If local storage, remove the episode directories directly
    if isinstance(storage, LocalFilesystemStorage):
        shutil.rmtree(storage.root / "clips" / external_id, ignore_errors=True)
        shutil.rmtree(storage.root / "peaks" / external_id, ignore_errors=True)

    # Delete episode (Postgres foreign keys CASCADE to segments, tasks, hypotheses, etc.)
    session.delete(episode)
    session.commit()

    return {
        "deleted": True,
        "episode_id": ep_id,
        "external_id": external_id,
        "deleted_segments": deleted_segments,
    }
