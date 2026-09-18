"""What the corpus contains, cut every way it can be cut, and what to record next (D91).

The status report next door answers *how much is done*. This package answers a different
question, and the one that decides what to do with an afternoon: **is this corpus any good for
its two purposes -- fine-tuning an ASR model and describing Nepali-English code-mixing -- and
what would make it better?**

The unit is the **clip**, and the person is the **voice**. Every clip carries one bucket on each
of sixteen categories -- gender, age, role, voice and its exposure in train; topic, genre, show
and code-mixing; speaking speed, clip length, crosstalk and speakers; noise, room and bandwidth
-- so hours cut by any of them sum to the corpus. D69 attributed an episode's hours to every
value it carried because no route diarized; every episode is diarized now (D79) and its speakers
are linked into voices across episodes (D87), so a clip has one dominant voice and the voice
has one gender, where the episode's declared rows force it (:mod:`resolve`).

Three things come out: the categories totalled up, one profile per voice followed across
episodes, and per-category advice tagged with the purpose it serves. And the clip table itself,
compact enough to ship, so the page can cut everything by anything without a round trip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.services.inventory.categories import (
    BucketStats,
    CategoryReport,
    assign_buckets,
    attributed_words,
    build_categories,
    build_category,
    clip_table,
    speaking_rate_bucket,
)
from app.services.inventory.constants import CATEGORIES, GROUPS, Category
from app.services.inventory.facts import ClipRow, EpisodeRow, count_words, load_rows, now_iso
from app.services.inventory.recommendations import Recommendation, recommend
from app.services.inventory.resolve import (
    Resolution,
    VoiceIdentity,
    resolve_corpus,
    resolve_episode,
)
from app.services.inventory.voices import VoiceProfile, build_voices, summarize_voices

__all__ = [
    "CATEGORIES",
    "GROUPS",
    "BucketStats",
    "Category",
    "CategoryReport",
    "ClipRow",
    "EpisodeRow",
    "Inventory",
    "Recommendation",
    "Resolution",
    "VoiceIdentity",
    "VoiceProfile",
    "assemble",
    "assign_buckets",
    "attributed_words",
    "build_categories",
    "build_category",
    "build_voices",
    "clip_table",
    "collect_inventory",
    "count_words",
    "load_rows",
    "recommend",
    "resolve_corpus",
    "resolve_episode",
    "speaking_rate_bucket",
    "summarize_voices",
]


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


@dataclass
class Inventory:
    """The whole answer, ready for the API."""

    generated_at: str = ""
    totals: dict[str, Any] = field(default_factory=dict)
    groups: list[dict[str, str]] = field(default_factory=list)
    categories: list[CategoryReport] = field(default_factory=list)
    voices: list[VoiceProfile] = field(default_factory=list)
    voice_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    clips: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "totals": self.totals,
            "groups": self.groups,
            "categories": [c.as_dict() for c in self.categories],
            "voices": [v.as_dict() for v in self.voices],
            "voice_summary": self.voice_summary,
            "recommendations": [asdict(r) for r in self.recommendations],
            "episodes": self.episodes,
            "records": self.records,
            "clips": self.clips,
        }


def _records(episodes: list[EpisodeRow]) -> list[dict[str, Any]]:
    """Which episode records are incomplete, field by field, and which episodes to go and fix.

    An unfilled variable is paperwork, not a fact about the world, and the two look identical in
    every chart until this table separates them.
    """
    checks = (
        ("show_id", lambda e: bool(e.show_id)),
        ("genre", lambda e: bool(e.genre)),
        ("topic", lambda e: bool(e.topic)),
        ("topic_in_taxonomy", lambda e: bool(e.topic_in_taxonomy)),
        ("speakers", lambda e: bool(e.declared)),
        ("published_at", lambda e: bool(e.published_at)),
        ("diarized", lambda e: e.diarized),
        ("voices_linked", lambda e: bool(e.voices) or not e.diarized),
    )
    total = len(episodes)
    return [
        {
            "field": name,
            "filled": total - len(missing),
            "total": total,
            "missing_episodes": missing,
        }
        for name, predicate in checks
        for missing in [[e.external_id for e in episodes if not predicate(e)]]
    ]


def assemble(
    episodes: list[EpisodeRow],
    clips: list[ClipRow],
    *,
    min_stratum_hours: float,
    min_stratum_voices: int,
    gold_target_hours: float,
) -> Inventory:
    """Everything after the database: resolve, bucket, total, profile, advise. Pure."""
    by_id = {e.external_id: e for e in episodes}
    per_episode, per_voice = resolve_corpus(episodes)
    clips = assign_buckets(clips, by_id, per_episode, per_voice)
    categories = build_categories(clips, by_id)
    voices = build_voices(clips, by_id, per_episode, per_voice)
    summary = summarize_voices(voices)

    hours = sum(c.duration for c in clips) / 3600
    verified = sum(c.duration for c in clips if c.tier == "verified") / 3600
    screened = sum(c.duration for c in clips if c.tier == "screened") / 3600
    totals = {
        "hours": _round(hours),
        "speech_hours": _round(sum(c.speech_seconds or 0.0 for c in clips) / 3600),
        "clips": len(clips),
        "episodes": len(episodes),
        "shows": len({e.show_id for e in episodes if e.show_id}),
        "words": sum(c.words or 0 for c in clips),
        "verified_hours": _round(verified),
        "screened_hours": _round(screened),
        "unlabeled_hours": _round(max(0.0, hours - verified - screened)),
        "gold_hours": _round(sum(c.duration for c in clips if c.pot == "gold") / 3600),
        "val_hours": _round(sum(c.duration for c in clips if c.pot == "val") / 3600),
        "train_hours": _round(sum(c.duration for c in clips if c.pot == "train") / 3600),
        "gold_target_hours": gold_target_hours,
        "min_stratum_hours": min_stratum_hours,
        "min_stratum_voices": min_stratum_voices,
    }
    episode_index = {e.external_id: i for i, e in enumerate(episodes)}
    voice_index = {v.voice: i for i, v in enumerate(voices)}
    return Inventory(
        generated_at=now_iso(),
        totals=totals,
        groups=[{"key": key, "label": label} for key, label in GROUPS],
        categories=list(categories.values()),
        voices=voices,
        voice_summary=summary,
        recommendations=recommend(
            categories,
            summary,
            min_stratum_hours=min_stratum_hours,
            min_stratum_voices=min_stratum_voices,
        ),
        episodes=[
            {
                "external_id": e.external_id,
                "title": e.title,
                "show_id": e.show_id,
                "genre": e.genre,
                "topic": e.topic,
                "topic_in_taxonomy": e.topic_in_taxonomy,
                "published_at": e.published_at,
                "split": e.split,
                "declared": [dict(r) for r in e.declared],
                "voices": list(e.voices),
                "diarized": e.diarized,
            }
            for e in episodes
        ],
        records=_records(episodes),
        clips=clip_table(clips, categories, episode_index, voice_index),
    )


def collect_inventory(session: Session, *, settings: Settings | None = None) -> Inventory:
    """Read the corpus and assemble the page's payload.

    Reads only. Opening the corpus page must never change what the corpus holds -- the same rule
    ``pot_status`` follows for the same reason (D63).
    """
    settings = settings or get_settings()
    episodes, clips = load_rows(session)
    return assemble(
        episodes,
        clips,
        min_stratum_hours=settings.dataset.min_stratum_hours,
        min_stratum_voices=settings.dataset.min_stratum_voices,
        gold_target_hours=settings.dataset.gold_hours_target,
    )
