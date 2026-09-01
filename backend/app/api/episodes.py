"""Episode management and deletion endpoints."""

from __future__ import annotations

import shutil
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_config, get_object_storage, get_session, require_auth
from app.api.schemas import EpisodeSegmentSummary, EpisodeSummary
from app.api.serializers import audio_url, peaks_url
from app.config import Settings
from app.models import AnnotationTask, AuditLog, Episode, Segment
from app.models.enums import ACTIVE_TASK_STATUSES
from app.storage import ObjectStorage, delete_objects
from app.storage.local import LocalFilesystemStorage

router = APIRouter(tags=["episodes"], dependencies=[Depends(require_auth)])


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

    # Aggregate counts per episode. Every count is over DISTINCT segment ids: a segment may carry
    # more than one task row (a skipped task plus its replacement is a legal state), and the join
    # to annotation_tasks would otherwise fan out and count task rows as if they were segments.
    counts_query = (
        sa.select(
            Segment.episode_id,
            sa.func.count(sa.distinct(Segment.id)).label("total_segments"),
            sa.func.count(
                sa.distinct(sa.case((Segment.pipeline_status == "labeled", Segment.id), else_=None))
            ).label("labeled_segments"),
            sa.func.count(
                sa.distinct(
                    sa.case(
                        (
                            sa.and_(
                                AnnotationTask.status.in_(ACTIVE_TASK_STATUSES),
                                Segment.pipeline_status != "labeled",
                            ),
                            Segment.id,
                        ),
                        else_=None,
                    )
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

    # Load the active task per segment. A segment can hold several task rows -- a skipped one and
    # its replacement -- so this filters to the active statuses and takes the newest, rather than
    # collapsing an unordered result and keeping whichever row arrived last.
    seg_ids = [s.id for s in segments]
    task_map: dict[int, str] = {}
    if seg_ids:
        tasks = session.execute(
            sa.select(AnnotationTask.segment_id, AnnotationTask.status)
            .where(
                AnnotationTask.segment_id.in_(seg_ids),
                AnnotationTask.status.in_(ACTIVE_TASK_STATUSES),
            )
            .order_by(AnnotationTask.id)
        ).all()
        task_map = dict(tasks)

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
                audio_url=audio_url(seg.id),
                peaks_url=peaks_url(seg),
            )
        )
    return results


@router.delete("/episodes/{episode_id}")
def delete_episode(
    episode_id: str,
    session: Session = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
    settings: Settings = Depends(get_config),
) -> dict[str, Any]:
    """Delete an episode, all its child records, and its audio/peak clips from storage."""
    episode = _find_episode(session, episode_id)
    external_id = episode.external_id
    ep_id = episode.id

    # Fetch segments to delete storage objects
    segments = session.scalars(sa.select(Segment).where(Segment.episode_id == ep_id)).all()
    deleted_segments = len(segments)

    for seg in segments:
        delete_objects(storage, seg.clip_object_key, seg.peaks_object_key)

    # If local storage, remove the episode directories directly
    if isinstance(storage, LocalFilesystemStorage):
        shutil.rmtree(storage.root / "clips" / external_id, ignore_errors=True)
        shutil.rmtree(storage.root / "peaks" / external_id, ignore_errors=True)

    session.add(
        AuditLog(
            entity_type="episodes",
            entity_id=str(ep_id),
            action="delete",
            actor=settings.labels.default_annotator,
            old_values_jsonb={
                "external_id": external_id,
                "title": episode.title,
                "split": episode.split,
                "segments": deleted_segments,
            },
            new_values_jsonb=None,
        )
    )

    # Delete episode (Postgres foreign keys CASCADE to segments, tasks, hypotheses, etc.)
    session.delete(episode)
    session.flush()

    return {
        "deleted": True,
        "episode_id": ep_id,
        "external_id": external_id,
        "deleted_segments": deleted_segments,
    }
