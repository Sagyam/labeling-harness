"""Measure overlapped speech on episodes that were imported before the detector existed (D77).

Ingest measures overlap as it cuts the clips. Everything already in the database gets the same
treatment here, from the episode audio retained beside the clips (D62): one detection pass per
episode, the clip-relative spans written to ``segments.overlap_spans_jsonb``, and the
``speaker_overlap`` heads-up added to or removed from each clip's stored flags.

Open tasks get the new flag in the ``flags`` list of their stored reason, which is where the
editor reads its chips. Nothing else in the reason changes, and nothing in the queue moves: the
flag is not scored, so a queue rebuild would compute the same priorities -- and an episode-scoped
rebuild would re-draw that episode's audit sample, which is not this command's business.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import soundfile as sf
import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.models import AnnotationTask, AuditLog, Episode, Segment
from app.models.enums import ACTIVE_TASK_STATUSES
from app.services.flags import overlap_flag_raised
from app.services.overlap import spans_within
from app.storage.base import ObjectStorage

FLAG = "speaker_overlap"


class Detector(Protocol):
    def detect(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]] | None: ...


@dataclass
class BackfillReport:
    """What a backfill run measured and changed."""

    episodes_measured: int = 0
    episodes_skipped: int = 0
    episodes_without_audio: int = 0
    segments_updated: int = 0
    segments_flagged: int = 0
    overlap_seconds: float = 0.0
    model_unavailable: bool = False


def backfill_overlap(
    session: Session,
    storage: ObjectStorage,
    detector: Detector,
    *,
    settings: Settings | None = None,
    actor: str,
    episode_external_ids: list[str] | None = None,
    force: bool = False,
) -> BackfillReport:
    """Measure overlap on every episode with retained audio whose clips are not measured yet.

    Args:
        session: Open session; the caller commits.
        storage: Where the episode audio lives.
        detector: Anything with :meth:`OverlapDetector.detect`'s signature.
        settings: Thresholds; defaults to the loaded configuration.
        actor: Recorded on each episode's ``audit_logs`` row.
        episode_external_ids: Restrict the run to these episodes.
        force: Re-measure episodes whose clips already carry spans.

    Returns:
        A :class:`BackfillReport`. ``model_unavailable`` means the detector had no model and the
        run stopped before writing anything further.
    """
    settings = settings or get_settings()
    report = BackfillReport()
    query = (
        sa.select(Episode)
        .options(selectinload(Episode.segments).selectinload(Segment.scores))
        .order_by(Episode.id)
    )
    if episode_external_ids:
        query = query.where(Episode.external_id.in_(episode_external_ids))

    for episode in session.scalars(query):
        segments = list(episode.segments)
        if not segments or (not force and all(s.overlap_spans_jsonb is not None for s in segments)):
            report.episodes_skipped += 1
            continue
        if not episode.audio_object_key:
            report.episodes_without_audio += 1
            continue

        audio, sample_rate = sf.read(
            io.BytesIO(storage.get_bytes(episode.audio_object_key)), dtype="float32"
        )
        spans = detector.detect(audio, sample_rate)
        if spans is None:
            report.model_unavailable = True
            return report

        flagged = _apply(session, segments, spans, settings)
        seconds = round(sum(end - start for start, end in spans), 3)
        report.episodes_measured += 1
        report.segments_updated += len(segments)
        report.segments_flagged += flagged
        report.overlap_seconds += seconds
        session.add(
            AuditLog(
                entity_type="episode",
                entity_id=episode.external_id,
                action="overlap_backfill",
                actor=actor,
                new_values_jsonb={
                    "segments": len(segments),
                    "segments_flagged": flagged,
                    "overlap_seconds": seconds,
                },
            )
        )
        session.flush()
    return report


def _apply(
    session: Session,
    segments: list[Segment],
    spans: list[tuple[float, float]],
    settings: Settings,
) -> int:
    """Write each clip's spans and reconcile its flag; return how many clips carry the flag."""
    flagged = 0
    tasks: dict[int, list[AnnotationTask]] = {}
    for task in session.scalars(
        sa.select(AnnotationTask).where(
            AnnotationTask.segment_id.in_([s.id for s in segments]),
            AnnotationTask.status.in_(ACTIVE_TASK_STATUSES),
        )
    ):
        tasks.setdefault(task.segment_id, []).append(task)

    for segment in segments:
        clip = [[a, b] for a, b in spans_within(spans, segment.start_time, segment.end_time)]
        segment.overlap_spans_jsonb = clip
        raised = overlap_flag_raised(clip, settings)
        flagged += raised
        if segment.scores is not None:
            segment.scores.flags_jsonb = _with_flag(segment.scores.flags_jsonb, raised)
        for task in tasks.get(segment.id, []):
            reason = dict(task.reason_jsonb or {})
            reason["flags"] = _with_flag(reason.get("flags"), raised)
            task.reason_jsonb = reason
    return flagged


def _with_flag(flags: list[str] | None, raised: bool) -> list[str]:
    """``flags`` with the overlap heads-up present or absent, sorted; a new list, so the JSONB
    column registers the change."""
    kept = {f for f in (flags or []) if f != FLAG}
    return sorted(kept | {FLAG} if raised else kept)
