"""Measure the acoustics of clips imported before the measurement existed (D87).

Ingest measures each clip as it cuts it. Everything already in the database gets the same
treatment here, from the episode audio retained beside the clips (D62): one pass per episode,
each clip's result written to ``segments.acoustics_jsonb``. A clip the meter could add nothing to
(:meth:`~app.services.acoustics.AcousticMeter.is_current`) is left alone unless forced, so the
command is safe to re-run and picks up only what a rule change, or a model now present, made
stale.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import soundfile as sf
import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.models import AuditLog, Episode
from app.services.acoustics import ClipSpec
from app.storage.base import ObjectStorage


class Meter(Protocol):
    @property
    def version(self) -> str: ...

    def is_current(self, result: Mapping[str, object] | None) -> bool: ...

    def measure(
        self, audio: np.ndarray, sample_rate: int, clips: list[ClipSpec]
    ) -> list[dict[str, object]]: ...


@dataclass
class BackfillReport:
    """What a backfill run measured."""

    episodes_measured: int = 0
    episodes_skipped: int = 0
    episodes_without_audio: int = 0
    segments_updated: int = 0


def backfill_acoustics(
    session: Session,
    storage: ObjectStorage,
    meter: Meter,
    *,
    actor: str,
    episode_external_ids: list[str] | None = None,
    force: bool = False,
) -> BackfillReport:
    """Measure every episode with retained audio that has a clip not measured under the current
    rules.

    Args:
        session: Open session; the caller commits.
        storage: Where the episode audio lives.
        meter: Anything with :meth:`AcousticMeter.measure`'s signature.
        actor: Recorded on each episode's ``audit_logs`` row.
        episode_external_ids: Restrict the run to these episodes.
        force: Re-measure clips already measured under the current version.
    """
    report = BackfillReport()
    query = sa.select(Episode).options(selectinload(Episode.segments)).order_by(Episode.id)
    if episode_external_ids:
        query = query.where(Episode.external_id.in_(episode_external_ids))

    for episode in session.scalars(query):
        segments = sorted(episode.segments, key=lambda s: s.start_time)
        todo = (
            segments if force else [s for s in segments if not meter.is_current(s.acoustics_jsonb)]
        )
        if not todo:
            report.episodes_skipped += 1
            continue
        if not episode.audio_object_key:
            report.episodes_without_audio += 1
            continue

        audio, sample_rate = sf.read(
            io.BytesIO(storage.get_bytes(episode.audio_object_key)), dtype="float32"
        )
        results = meter.measure(
            audio, sample_rate, [(s.start_time, s.end_time, s.vad_spans_jsonb) for s in todo]
        )
        for segment, result in zip(todo, results, strict=True):
            segment.acoustics_jsonb = result
        report.episodes_measured += 1
        report.segments_updated += len(todo)
        session.add(
            AuditLog(
                entity_type="episode",
                entity_id=episode.external_id,
                action="acoustics_backfill",
                actor=actor,
                new_values_jsonb={"segments": len(todo), "version": meter.version},
            )
        )
        session.flush()
    return report
