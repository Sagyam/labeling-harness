"""Clip rows: one row per clip with everything the corpus page cuts by (D91).

The unit is the clip, not the episode. A clip has one duration, one dominant voice, one bucket on
every axis, and one label tier -- so hours cut by any category sum to the corpus, which the
episode-attributed inventory of D69 could not promise for any speaker dimension.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.llm.topic import TOPIC_LABELS
from app.models import AsrHypothesis, AsrSystem, Episode, Segment
from app.models.enums import APPROVED_DISPOSITIONS
from app.services.clip_classes import classify, load_clip_facts
from app.services.normalize import WORD_TOKEN_RE
from app.services.stats import latest_labels_subquery
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeRow:
    """What the page needs to know about an episode: its recorded facts and its voices."""

    external_id: str
    title: str | None
    show_id: str | None
    genre: str | None
    topic: str | None
    published_at: str | None
    split: str
    #: The declared speaker rows, each ``{role?, gender?, age_bracket?}`` (D56 allowlist).
    declared: tuple[Mapping[str, str], ...]
    #: Whether the episode's newest diarization run exists.
    diarized: bool
    #: Voices heard in the episode's clips, most talk first. Empty when undiarized or unlinked.
    voices: tuple[str, ...]

    @property
    def topic_in_taxonomy(self) -> bool | None:
        return self.topic in TOPIC_LABELS if self.topic else None


@dataclass(frozen=True)
class ClipRow:
    """One clip reduced to what the page counts. Times in seconds."""

    segment_id: int
    episode: str
    duration: float
    #: VAD speech seconds inside the clip; ``None`` when the spans were never recorded.
    speech_seconds: float | None
    #: Reference words: the current label's text when it carries one, else the fused seed's.
    words: int | None
    text_source: str | None
    #: ``verified``, ``screened`` or ``None`` for a clip nobody has decided.
    tier: str | None
    #: ``gold`` for a gold clip, else the episode's ``train``/``val`` (D71).
    pot: str
    #: The dominant linked voice, or ``None``.
    voice: str | None
    #: Talk seconds per linked voice inside the clip.
    voice_talk: Mapping[str, float]
    cmi: float | None
    #: Buckets from :mod:`app.services.clip_classes` -- the same classes a model run is split by.
    classes: Mapping[str, str]
    #: The page's buckets, one per category key. Filled by :func:`assign_buckets`.
    buckets: Mapping[str, str] = field(default_factory=dict)

    @property
    def words_per_second(self) -> float | None:
        """Reference words per second of speech, falling back to clip seconds without VAD."""
        if not self.words:
            return None
        seconds = self.speech_seconds if self.speech_seconds else self.duration
        return self.words / seconds if seconds > 0 else None


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _declared_rows(metadata: Mapping[str, Any] | None) -> tuple[Mapping[str, str], ...]:
    speakers = (metadata or {}).get("speakers")
    if not isinstance(speakers, Mapping):
        return ()
    out: list[Mapping[str, str]] = []
    for fields in speakers.values():
        if isinstance(fields, Mapping):
            out.append({k: v for k, v in fields.items() if isinstance(v, str) and v})
    return tuple(out)


def _string(metadata: Mapping[str, Any] | None, key: str) -> str | None:
    value = (metadata or {}).get(key)
    return value if isinstance(value, str) and value else None


def count_words(text: str | None) -> int:
    """Reference words as the fold tokeniser sees them: Latin or Devanagari runs."""
    return len(WORD_TOKEN_RE.findall(text or ""))


def load_rows(session: Session) -> tuple[list[EpisodeRow], list[ClipRow]]:
    """Every episode and every clip, in a handful of queries.

    Clip classes come from :func:`app.services.clip_classes.load_clip_facts`, so what this page
    calls crosstalk or noise is exactly what a model run's breakdown calls it.
    """
    episodes = list(session.scalars(sa.select(Episode).order_by(Episode.external_id)))
    segments = list(session.scalars(sa.select(Segment).order_by(Segment.id)))
    facts = load_clip_facts(session, segments) if segments else {}

    current = latest_labels_subquery()
    labels = {
        row.segment_id: row
        for row in session.execute(
            sa.select(
                current.c.segment_id,
                current.c.verification_tier,
                current.c.disposition,
                current.c.final_text,
            )
        )
    }
    seeds = dict(
        session.execute(
            sa.select(AsrHypothesis.segment_id, AsrHypothesis.text_raw)
            .join(AsrSystem, AsrSystem.id == AsrHypothesis.asr_system_id)
            .where(AsrSystem.kind == "fusion")
        )
        .tuples()
        .all()
    )

    by_id = {e.id: e for e in episodes}
    episode_voice_talk: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    clips: list[ClipRow] = []
    for segment in segments:
        fact = facts[segment.id]
        episode = by_id[segment.episode_id]
        label = labels.get(segment.id)
        text_source: str | None = None
        text: str | None = None
        if label is not None and label.disposition in APPROVED_DISPOSITIONS and label.final_text:
            text, text_source = label.final_text, "label"
        elif seeds.get(segment.id):
            text, text_source = seeds[segment.id], "seed"
        voices = fact.voices or {}
        talk: dict[str, float] = defaultdict(float)
        for start, end, speaker in fact.turns or ():
            if (voice := voices.get(speaker)) is not None:
                talk[voice] += max(0.0, end - start)
        for voice, seconds in talk.items():
            episode_voice_talk[episode.id][voice] += seconds
        speech = (
            sum(max(0.0, end - start) for start, end in fact.vad_spans)
            if fact.vad_spans is not None
            else None
        )
        clips.append(
            ClipRow(
                segment_id=segment.id,
                episode=episode.external_id,
                duration=float(segment.duration_seconds),
                speech_seconds=speech,
                words=count_words(text) if text is not None else None,
                text_source=text_source,
                tier=label.verification_tier if label is not None else None,
                pot="gold" if segment.pot == "gold" else episode.split,
                voice=fact.voice,
                voice_talk=dict(talk),
                cmi=fact.cmi,
                classes=classify(fact),
            )
        )

    diarized = {s.episode_id for s in segments if facts[s.id].turns is not None}
    rows = [
        EpisodeRow(
            external_id=e.external_id,
            title=e.title,
            show_id=e.show_id,
            genre=_string(e.metadata_jsonb, "genre"),
            topic=_string(e.metadata_jsonb, "topic"),
            published_at=e.published_at.isoformat() if e.published_at else None,
            split=e.split,
            declared=_declared_rows(e.metadata_jsonb),
            diarized=e.id in diarized,
            voices=tuple(
                sorted(episode_voice_talk[e.id], key=lambda v: -episode_voice_talk[e.id][v])
            ),
        )
        for e in episodes
    ]
    return rows, clips


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


__all__ = ["ClipRow", "EpisodeRow", "count_words", "load_rows", "now_iso"]
