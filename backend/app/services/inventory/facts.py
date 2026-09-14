"""Episode facts: one row per episode with everything the inventory cuts hours by."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Episode, Segment, SegmentScore
from app.services.inventory.constants import SPEAKER_KEYS
from app.services.stats import latest_labels_subquery
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeFacts:
    """One episode reduced to what the inventory counts. The unit of attribution."""

    external_id: str
    title: str | None
    show_id: str | None
    published_at: str | None
    #: ``gold`` when every clip is in gold, ``train`` when none is, ``mixed`` in between -- gold is
    #: chosen per clip since D71, so an episode no longer has a pot of its own.
    pot: str
    gold_segments: int
    split: str
    hours: float
    segments: int
    labeled_hours: float
    verified_hours: float
    screened_hours: float
    topic: str | None
    topic_source: str | None
    #: One entry per speaker the metadata records, as ``{role, gender, age_bracket}`` with a
    #: missing field absent rather than guessed.
    speakers: tuple[Mapping[str, str], ...]
    mean_cmi: float | None
    min_cmi: float | None
    max_cmi: float | None

    def values_for(self, key: str) -> frozenset[str]:
        """What this episode contributes to one dimension.

        An episode with two speakers of different genders contributes both; one whose metadata is
        silent contributes nothing, rather than a guess that would be counted as evidence.
        """
        if key == "show_id":
            return frozenset({self.show_id} if self.show_id else ())
        if key == "topic":
            return frozenset({self.topic} if self.topic else ())
        if key == "role":
            return frozenset(s["role"] for s in self.speakers if s.get("role"))
        if key in SPEAKER_KEYS:
            return frozenset(s[key] for s in self.speakers if s.get(key))
        return frozenset()


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _speakers_of(metadata: Mapping[str, Any] | None) -> tuple[Mapping[str, str], ...]:
    """Pull the speaker blocks out of episode metadata, dropping anything malformed.

    The importer has already reduced these to the D56 allowlist, so this only has to survive an
    episode whose metadata predates that or arrived shaped differently.
    """
    speakers = (metadata or {}).get("speakers")
    if not isinstance(speakers, Mapping):
        return ()
    out: list[Mapping[str, str]] = []
    for fields in speakers.values():
        if isinstance(fields, Mapping):
            out.append({k: v for k, v in fields.items() if isinstance(v, str) and v})
    return tuple(out)


def load_episode_facts(session: Session) -> list[EpisodeFacts]:
    """Every episode, with its audio, its labels and its metadata, in one pass.

    Hours come from the episode's *segments* -- the audio that actually reaches the queue -- and
    fall back to ``episodes.duration_seconds`` only for an episode whose segments are not imported
    yet.
    """
    segment_rows = {
        row.episode_id: row
        for row in session.execute(
            sa.select(
                Segment.episode_id.label("episode_id"),
                sa.func.count().label("segments"),
                sa.func.count(sa.case((Segment.pot == "gold", Segment.id), else_=None)).label(
                    "gold_segments"
                ),
                sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0).label("seconds"),
                sa.func.avg(SegmentScore.code_switch_density).label("mean_cmi"),
                sa.func.min(SegmentScore.code_switch_density).label("min_cmi"),
                sa.func.max(SegmentScore.code_switch_density).label("max_cmi"),
            )
            .select_from(Segment)
            .outerjoin(SegmentScore, SegmentScore.segment_id == Segment.id)
            .group_by(Segment.episode_id)
        )
    }

    current = latest_labels_subquery()
    labeled: dict[int, dict[str, float]] = defaultdict(lambda: {"verified": 0.0, "screened": 0.0})
    for episode_id, tier, seconds in session.execute(
        sa.select(
            Segment.episode_id,
            current.c.verification_tier,
            sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0),
        )
        .select_from(current)
        .join(Segment, Segment.id == current.c.segment_id)
        .group_by(Segment.episode_id, current.c.verification_tier)
    ):
        labeled[episode_id][tier] = float(seconds)

    facts: list[EpisodeFacts] = []
    for episode in session.scalars(sa.select(Episode).order_by(Episode.external_id)):
        row = segment_rows.get(episode.id)
        seconds = float(row.seconds) if row else 0.0
        if seconds <= 0.0:
            seconds = float(episode.duration_seconds or 0.0)
        tiers = labeled.get(episode.id, {"verified": 0.0, "screened": 0.0})
        metadata: Mapping[str, Any] = episode.metadata_jsonb or {}
        topic = metadata.get("topic")
        topic_source = metadata.get("topic_source")
        segments = row.segments if row else 0
        gold_segments = int(row.gold_segments) if row else 0
        if gold_segments and gold_segments == segments:
            pot = "gold"
        elif gold_segments:
            pot = "mixed"
        else:
            pot = "train"
        facts.append(
            EpisodeFacts(
                external_id=episode.external_id,
                title=episode.title,
                show_id=episode.show_id,
                published_at=episode.published_at.isoformat() if episode.published_at else None,
                pot=pot,
                gold_segments=gold_segments,
                split=episode.split,
                hours=seconds / 3600,
                segments=segments,
                labeled_hours=(tiers.get("verified", 0.0) + tiers.get("screened", 0.0)) / 3600,
                verified_hours=tiers.get("verified", 0.0) / 3600,
                screened_hours=tiers.get("screened", 0.0) / 3600,
                topic=topic if isinstance(topic, str) and topic else None,
                topic_source=topic_source if isinstance(topic_source, str) else None,
                speakers=_speakers_of(metadata),
                mean_cmi=float(row.mean_cmi) if row and row.mean_cmi is not None else None,
                min_cmi=float(row.min_cmi) if row and row.min_cmi is not None else None,
                max_cmi=float(row.max_cmi) if row and row.max_cmi is not None else None,
            )
        )
    return facts
