"""What the corpus contains, cut every way it can be cut, pot by pot (D91, D104).

The status report next door answers *how much is done*. This package answers *what is lacking
and what is plenty*: every clip carries one bucket on each category -- gender, age, role and the
voice's exposure in train; topic, genre, show and code-mixing; speaking speed, clip length,
crosstalk and speakers; noise, room and bandwidth -- so hours cut by any of them sum to the pot.

Gold and train/val are separate views, each rated against its own floor (:mod:`coverage`),
because a benchmark and a training set want different things and drawing them on one axis made
the smaller one unreadable. Voices, followed across episodes, are served on their own
(:func:`collect_voices`) for the voices page.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
from app.services.inventory.coverage import (
    POTS,
    Floor,
    PotReport,
    build_pot,
    category_meta,
    rate_bucket,
)
from app.services.inventory.facts import ClipRow, EpisodeRow, count_words, load_rows, now_iso
from app.services.inventory.resolve import (
    Resolution,
    VoiceIdentity,
    resolve_corpus,
    resolve_episode,
)
from app.services.inventory.voices import VoiceProfile, build_voices, summarize_voices
from app.services.voice_attributes import current_voice_attributes

__all__ = [
    "CATEGORIES",
    "GROUPS",
    "POTS",
    "BucketStats",
    "Category",
    "CategoryReport",
    "ClipRow",
    "EpisodeRow",
    "Floor",
    "Inventory",
    "PotReport",
    "Resolution",
    "VoiceIdentity",
    "VoiceProfile",
    "assemble",
    "assign_buckets",
    "attributed_words",
    "build_categories",
    "build_category",
    "build_pot",
    "build_voices",
    "clip_table",
    "collect_inventory",
    "collect_voice_profiles",
    "collect_voices",
    "count_words",
    "load_rows",
    "rate_bucket",
    "resolve_corpus",
    "resolve_episode",
    "speaking_rate_bucket",
    "summarize_voices",
]

Manual = Mapping[str, Mapping[str, str | None]]


@dataclass
class Inventory:
    """The corpus page's payload."""

    generated_at: str = ""
    groups: list[dict[str, str]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)
    pots: dict[str, PotReport] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    clips: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "groups": self.groups,
            "categories": self.categories,
            "pots": {key: pot.as_dict() for key, pot in self.pots.items()},
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


def _bucketed(
    episodes: list[EpisodeRow], clips: list[ClipRow], manual: Manual | None
) -> tuple[dict[str, EpisodeRow], list[ClipRow], dict, dict]:
    by_id = {e.external_id: e for e in episodes}
    per_episode, per_voice = resolve_corpus(episodes, manual)
    return by_id, assign_buckets(clips, by_id, per_episode, per_voice), per_episode, per_voice


def assemble(
    episodes: list[EpisodeRow],
    clips: list[ClipRow],
    *,
    floors: Mapping[str, Floor],
    targets: Mapping[str, float],
    manual: Manual | None = None,
) -> Inventory:
    """Everything after the database: resolve, bucket, then total and rate each pot. Pure."""
    by_id, clips, _, _ = _bucketed(episodes, clips, manual)
    pots = {
        key: build_pot(key, clips, by_id, floors[key], target_hours=targets.get(key))
        for key, _ in POTS
    }
    # Bucket order for the clip table is the pot-independent one: the whole corpus's.
    everything = build_categories(clips, by_id)
    voices = sorted({c.voice for c in clips if c.voice})
    return Inventory(
        generated_at=now_iso(),
        groups=[{"key": key, "label": label} for key, label in GROUPS],
        categories=category_meta(),
        pots=pots,
        records=_records(episodes),
        clips=clip_table(clips, everything, {v: i for i, v in enumerate(voices)}),
    )


def _floors(settings: Settings) -> dict[str, Floor]:
    coverage = settings.dataset.coverage
    return {
        key: Floor(
            thin_hours=pot.thin_hours,
            thin_voices=pot.thin_voices,
            voice_words=pot.voice_words,
            plenty_factor=coverage.plenty_factor,
            dominant_share=coverage.dominant_share,
        )
        for key, pot in (("train", coverage.train), ("gold", coverage.gold))
    }


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
        floors=_floors(settings),
        targets={
            "train": settings.dataset.train_hours_target,
            "gold": settings.dataset.gold_hours_target,
        },
        manual=current_voice_attributes(session),
    )


def collect_voice_profiles(
    episodes: list[EpisodeRow], clips: list[ClipRow], *, manual: Manual | None = None
) -> dict[str, Any]:
    """Every voice followed across episodes, with the corpus-wide summary. Pure."""
    by_id, clips, per_episode, per_voice = _bucketed(episodes, clips, manual)
    profiles = build_voices(clips, by_id, per_episode, per_voice)
    return {
        "generated_at": now_iso(),
        "summary": summarize_voices(profiles),
        "voices": [p.as_dict() for p in profiles],
    }


def collect_voices(session: Session) -> dict[str, Any]:
    """The voices page's payload, read from the database. Reads only."""
    episodes, clips = load_rows(session)
    return collect_voice_profiles(episodes, clips, manual=current_voice_attributes(session))
