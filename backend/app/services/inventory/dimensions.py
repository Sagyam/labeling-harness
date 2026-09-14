"""Inventory dimensions: hours, episodes and segments cut by every recorded variable."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.llm.topic import TOPIC_LABELS
from app.models import Segment, SegmentScore
from app.services.inventory.constants import (
    BAND_DESCRIPTION,
    HISTOGRAM_BINS,
    LONG_EPISODE_MINUTES,
    REGISTER_BANDS,
    VOCABULARIES,
)
from app.services.inventory.facts import EpisodeFacts, _round
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DimensionValue:
    """One value of one dimension, with the audio that carries it."""

    value: str
    hours: float
    episodes: int
    segments: int
    labeled_hours: float
    #: Distinct shows carrying this value. A value carried by exactly one show is present in the
    #: corpus and not independent of it -- drop that show and the value goes too.
    shows: int
    #: Share of corpus hours. Speaker dimensions do not sum to 1; see the module docstring.
    share: float


@dataclass
class Dimension:
    """Every value of one dimension, plus what its vocabulary says should be there and is not."""

    key: str
    values: list[DimensionValue] = field(default_factory=list)
    #: Vocabulary values with no audio at all. Empty for an open dimension like ``show_id``.
    absent: list[str] = field(default_factory=list)
    #: Values present in the data that the closed vocabulary does not contain. These are not gaps
    #: but *dirt*: a free-text topic typed into the ingest form instead of chosen from the
    #: taxonomy, which cannot be reported on and cannot be stratified against (D57).
    off_vocabulary: list[str] = field(default_factory=list)
    #: Share held by the largest value, and the Herfindahl index over all shares (1.0 is one value
    #: holding everything). Two numbers because a corpus can be dominated by one show or merely
    #: spread thin, and they need different answers.
    top_share: float = 0.0
    hhi: float = 0.0
    #: Episodes carrying no value at all for this dimension -- unfilled metadata, not a gap in the
    #: world.
    unknown_episodes: int = 0
    unknown_hours: float = 0.0


def build_dimension(facts: Sequence[EpisodeFacts], key: str) -> Dimension:
    """Aggregate one dimension over the episodes, and say what its vocabulary is missing."""
    corpus_hours = sum(f.hours for f in facts)
    hours: dict[str, float] = defaultdict(float)
    labeled: dict[str, float] = defaultdict(float)
    episodes: dict[str, int] = defaultdict(int)
    segments: dict[str, int] = defaultdict(int)
    shows: dict[str, set[str]] = defaultdict(set)
    unknown_episodes = 0
    unknown_hours = 0.0

    for fact in facts:
        values = fact.values_for(key)
        if not values:
            unknown_episodes += 1
            unknown_hours += fact.hours
            continue
        for value in values:
            hours[value] += fact.hours
            labeled[value] += fact.labeled_hours
            episodes[value] += 1
            segments[value] += fact.segments
            if fact.show_id:
                shows[value].add(fact.show_id)

    values = [
        DimensionValue(
            value=value,
            hours=_round(hours[value]),
            episodes=episodes[value],
            segments=segments[value],
            labeled_hours=_round(labeled[value]),
            shows=len(shows[value]),
            share=_round(hours[value] / corpus_hours, 4) if corpus_hours else 0.0,
        )
        for value in sorted(hours, key=lambda v: (-hours[v], v))
    ]

    vocabulary = VOCABULARIES.get(key)
    total = sum(hours.values())
    return Dimension(
        key=key,
        values=values,
        absent=[v for v in vocabulary if v not in hours] if vocabulary else [],
        off_vocabulary=sorted(v for v in hours if v not in vocabulary) if vocabulary else [],
        top_share=_round(max((v.hours for v in values), default=0.0) / total, 4) if total else 0.0,
        hhi=_round(sum((h / total) ** 2 for h in hours.values()), 4) if total else 0.0,
        unknown_episodes=unknown_episodes,
        unknown_hours=_round(unknown_hours),
    )


def build_speaker_matrix(facts: Sequence[EpisodeFacts]) -> dict[str, dict[str, float]]:
    """``gender x age_bracket`` hours, over the full closed vocabulary of both.

    Every cell is present, including the empty ones -- an absent cell rendered as a blank square
    is the entire point, and a dict that omits it makes the frontend guess at the vocabulary.
    """
    matrix = {
        gender: dict.fromkeys(VOCABULARIES["age_bracket"], 0.0) for gender in VOCABULARIES["gender"]
    }
    for fact in facts:
        # Per *cell* per episode, not per speaker: two male 20-39 speakers in one episode are one
        # cell, and adding the episode's hours twice would put more audio in a cell than exists.
        cells = {
            (s.get("gender"), s.get("age_bracket"))
            for s in fact.speakers
            if s.get("gender") and s.get("age_bracket")
        }
        for gender, age in cells:
            if gender in matrix and age in matrix[gender]:
                matrix[gender][age] += fact.hours
    return {g: {a: _round(h) for a, h in ages.items()} for g, ages in matrix.items()}


def collect_register(session: Session, shows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The code-switching distribution, weighted by audio rather than by clip count.

    Weighting matters: a corpus of two-second clips and a corpus of thirty-second ones sound
    nothing alike, and a histogram over clip counts would call them the same shape.

    The clip histogram and the **spread across show means** answer different questions and only
    the second one is about sourcing. Clip density spans nearly the whole range inside any single
    show -- one sentence is all Nepali and the next is half English -- so a wide histogram is
    compatible with every speaker in the corpus being the same kind of speaker. What a
    code-switching study needs variance in is the *speaker*, and that is the show means.

    Args:
        session: Open session.
        shows: Per-show rollup from :func:`build_shows`, for the show-mean spread.
    """
    rows = session.execute(
        sa.select(SegmentScore.code_switch_density, Segment.duration_seconds)
        .select_from(SegmentScore)
        .join(Segment, Segment.id == SegmentScore.segment_id)
        .where(SegmentScore.code_switch_density.is_not(None))
    ).all()

    histogram = [
        {
            "lower": _round(index / HISTOGRAM_BINS, 2),
            "upper": _round((index + 1) / HISTOGRAM_BINS, 2),
            "hours": 0.0,
            "segments": 0,
        }
        for index in range(HISTOGRAM_BINS)
    ]
    bands = {name: {"hours": 0.0, "segments": 0} for name, _, _ in REGISTER_BANDS}
    measured_seconds = 0.0
    weighted_sum = 0.0

    for density, seconds in rows:
        density = float(density)
        seconds = float(seconds or 0.0)
        measured_seconds += seconds
        weighted_sum += density * seconds
        index = min(HISTOGRAM_BINS - 1, int(density * HISTOGRAM_BINS))
        histogram[index]["hours"] += seconds / 3600
        histogram[index]["segments"] += 1
        for name, lower, upper in REGISTER_BANDS:
            if lower <= density < upper:
                bands[name]["hours"] += seconds / 3600
                bands[name]["segments"] += 1
                break

    for bucket in histogram:
        bucket["hours"] = _round(bucket["hours"])
    for band in bands.values():
        band["hours"] = _round(band["hours"])

    show_means = [
        {"show_id": row["show_id"], "mean_cmi": row["mean_cmi"], "hours": row["hours"]}
        for row in shows
        if row["mean_cmi"] is not None
    ]
    means = [row["mean_cmi"] for row in show_means]

    return {
        "measured_hours": _round(measured_seconds / 3600),
        "mean": _round(weighted_sum / measured_seconds, 4) if measured_seconds else None,
        "show_means": sorted(show_means, key=lambda row: row["mean_cmi"]),
        "show_mean_min": _round(min(means), 4) if means else None,
        "show_mean_max": _round(max(means), 4) if means else None,
        "show_mean_spread": _round(max(means) - min(means), 4) if len(means) > 1 else None,
        "histogram": histogram,
        "bands": [
            {
                "name": name,
                "lower": lower,
                "upper": min(upper, 1.0),
                "description": BAND_DESCRIPTION[name],
                **bands[name],
            }
            for name, lower, upper in REGISTER_BANDS
        ],
    }


def build_length_profile(facts: Sequence[EpisodeFacts]) -> dict[str, Any]:
    """How the corpus's hours are distributed over episode length.

    The corpus is moving from multi-hour podcasts to short videos picked for the speaker they
    bring, so "which end of the length range are these hours coming from" is a sourcing question,
    not a curiosity.
    """
    buckets = (
        ("under_5m", 0.0, 5.0),
        ("5_20m", 5.0, 20.0),
        ("20_45m", 20.0, 45.0),
        ("45_90m", 45.0, 90.0),
        ("over_90m", 90.0, float("inf")),
    )
    counts = {name: {"episodes": 0, "hours": 0.0} for name, _, _ in buckets}
    long_hours = 0.0
    total_hours = 0.0
    minutes: list[float] = []
    for fact in facts:
        if fact.hours <= 0:
            continue
        length = fact.hours * 60
        minutes.append(length)
        total_hours += fact.hours
        if length >= LONG_EPISODE_MINUTES:
            long_hours += fact.hours
        for name, lower, upper in buckets:
            if lower <= length < upper:
                counts[name]["episodes"] += 1
                counts[name]["hours"] += fact.hours
                break
    for bucket in counts.values():
        bucket["hours"] = _round(bucket["hours"])
    minutes.sort()
    return {
        "buckets": [{"name": name, **counts[name]} for name, _, _ in buckets],
        "median_minutes": _round(minutes[len(minutes) // 2], 1) if minutes else None,
        "long_episode_hours": _round(long_hours),
        "long_episode_share": _round(long_hours / total_hours, 4) if total_hours else 0.0,
    }


def build_metadata_completeness(facts: Sequence[EpisodeFacts]) -> list[dict[str, Any]]:
    """Which episode records are incomplete, field by field, and which episodes to go and fix.

    An unfilled stratification variable is worse than a thin stratum: a thin one is a fact about
    the corpus, an unfilled one is a fact about the paperwork, and the two look identical in every
    chart until this table separates them.
    """
    checks: tuple[tuple[str, Any], ...] = (
        ("show_id", lambda f: bool(f.show_id)),
        ("published_at", lambda f: bool(f.published_at)),
        ("topic", lambda f: bool(f.topic)),
        ("topic_in_taxonomy", lambda f: f.topic in TOPIC_LABELS if f.topic else False),
        ("speakers", lambda f: bool(f.speakers)),
        ("gender", lambda f: bool(f.values_for("gender"))),
        ("age_bracket", lambda f: bool(f.values_for("age_bracket"))),
        ("role", lambda f: bool(f.values_for("role"))),
        ("segments_imported", lambda f: f.segments > 0),
    )
    total = len(facts)
    out = []
    for name, predicate in checks:
        missing = [f.external_id for f in facts if not predicate(f)]
        absent = set(missing)
        out.append(
            {
                "field": name,
                "filled": total - len(missing),
                "total": total,
                "fraction": _round((total - len(missing)) / total, 4) if total else 0.0,
                "missing_hours": _round(sum(f.hours for f in facts if f.external_id in absent)),
                "missing_episodes": missing,
            }
        )
    return out


def build_shows(facts: Sequence[EpisodeFacts]) -> list[dict[str, Any]]:
    """Per-show rollup: the row a sourcing decision is actually made at.

    A show is the unit you can go and get more of, so the CMI range here is the number that says
    whether more of it would add register variance or just more of the same.
    """
    by_show: dict[str, list[EpisodeFacts]] = defaultdict(list)
    for fact in facts:
        by_show[fact.show_id or "(no show id)"].append(fact)

    rows = []
    for show, episodes in by_show.items():
        hours = sum(f.hours for f in episodes)
        weighted = [(f.mean_cmi, f.hours) for f in episodes if f.mean_cmi is not None]
        measured = sum(h for _, h in weighted)
        rows.append(
            {
                "show_id": show,
                "episodes": len(episodes),
                "hours": _round(hours),
                "segments": sum(f.segments for f in episodes),
                "labeled_hours": _round(sum(f.labeled_hours for f in episodes)),
                "verified_hours": _round(sum(f.verified_hours for f in episodes)),
                "speaker_profiles": len(
                    {
                        (s.get("role"), s.get("gender"), s.get("age_bracket"))
                        for f in episodes
                        for s in f.speakers
                    }
                ),
                "genders": sorted({g for f in episodes for g in f.values_for("gender")}),
                "age_brackets": sorted({a for f in episodes for a in f.values_for("age_bracket")}),
                "topics": sorted({f.topic for f in episodes if f.topic}),
                "pots": sorted({f.pot for f in episodes}),
                "mean_cmi": _round(sum(c * h for c, h in weighted) / measured, 4)
                if measured
                else None,
                "min_cmi": min(
                    (f.min_cmi for f in episodes if f.min_cmi is not None), default=None
                ),
                "max_cmi": max(
                    (f.max_cmi for f in episodes if f.max_cmi is not None), default=None
                ),
            }
        )
    rows.sort(key=lambda row: -row["hours"])
    return rows


def _speaker_profiles(facts: Iterable[EpisodeFacts]) -> set[tuple[str | None, ...]]:
    """Distinct ``(show, role, gender, age)`` combinations -- a *lower bound* on individuals.

    Two guests of the same show, gender and age bracket collapse into one entry, and there is no
    way to tell them apart: no route diarizes (D52) and the corpus stores no name (D56). Reported
    as a floor rather than a count, because an undercount that says so is usable and one that
    does not is a lie.
    """
    return {
        (f.show_id, s.get("role"), s.get("gender"), s.get("age_bracket"))
        for f in facts
        for s in f.speakers
    }
