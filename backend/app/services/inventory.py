"""What the corpus contains, what it is missing, and what to go and record next (D69).

The status report next door answers *how much is done*. This module answers a different question,
and the one that actually decides what to do with an afternoon: **is this corpus any good, and
what would make it better?**

Three sections, in the order the question is asked.

* **Inventory** -- hours, episodes and segments cut by every variable the corpus records: show,
  topic, speaker gender, speaker age bracket, speaker role, code-switching band, episode length.
* **Gaps** -- strata that are empty or thin, values the metadata never filled in, values that are
  spelled outside their closed vocabulary, and how concentrated the corpus is in its largest show.
* **Recommendations** -- the gaps turned into a ranked shopping list, each row saying what to look
  for and the number that says why.

Two measurement decisions run through all of it, and both are stated here because either one read
the wrong way turns a useful number into a false one.

**Hours are attributed per episode, to every value the episode carries.** An episode with a male
host and a female guest contributes its whole duration to *both* genders, because nothing in the
schema says which speaker held the microphone for how long -- no route diarizes (D52), so
per-speaker time does not exist. Shares therefore do not sum to 1 on any speaker dimension. The
alternative, splitting an episode's hours evenly between its speakers, would invent a number that
looks precise and is not.

**A stratum is judged by absence and thinness, never against a target distribution.** There is no
principled target share for "hours of speech from 60-79 year olds", so this module does not
pretend to one. It reports what is missing entirely, what sits under
``dataset.min_stratum_hours``, and where one value dominates -- three things that are true
independent of anyone's opinion about what the ideal corpus looks like.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.llm.topic import TOPIC_LABELS
from app.models import Episode, Segment, SegmentScore
from app.services.speaker_meta import ALLOWED_VALUES
from app.services.stats import latest_labels_subquery

#: Closed vocabularies, per dimension key. A dimension with one can say what is *absent* rather
#: than only what is present -- the difference between an inventory and a gap report. ``show_id``
#: is deliberately open: there is no list of every Nepali YouTube channel.
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "topic": TOPIC_LABELS,
    "gender": tuple(sorted(ALLOWED_VALUES["gender"])),
    "age_bracket": ("under_20", "20_39", "40_59", "60_79", "80_plus"),
}

#: Speaker dimensions, in the order a gap in one hurts most. Ordering matches
#: ``dataset.coverage_keys`` for the same reason it is ordered there.
SPEAKER_KEYS: tuple[str, ...] = ("gender", "age_bracket")

#: Code-switching bands, as ``(name, lower, upper)`` over ``code_switch_density`` (the minority
#: script's share of tokens -- in practice the English share). The boundaries are where the
#: measured corpus actually sits: every show so far falls between 0.13 and 0.36, so the interesting
#: question is whether anything exists outside that, and a three-way cut answers it without
#: inventing precision.
REGISTER_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.10),
    ("mid", 0.10, 0.25),
    ("high", 0.25, 1.01),
)

#: What a band means in the world, so a recommendation can say what to search for rather than
#: printing a number at someone.
BAND_DESCRIPTION: dict[str, str] = {
    "low": "little English mixed in -- rural, older or non-media speakers, formal Nepali",
    "mid": "moderate mixing -- the register most of the corpus already sits in",
    "high": "heavily English-mixed -- tech, startup, finance or diaspora speech",
}

#: Show mean code-switch densities closer together than this leave the dependent variable of a
#: code-switching study with almost no variance to explain: every show says roughly the same thing
#: about how much English gets mixed in, so nothing in the corpus can distinguish a cause from a
#: constant. Measured across *shows*, not clips -- clip-level density spans the whole range inside
#: any single show, which says nothing about who was recorded.
NARROW_SHOW_SPREAD = 0.15

#: Bins for the code-switching histogram. Ten is enough to see a distribution's shape and few
#: enough that each bin holds something at corpus sizes of a few hours.
HISTOGRAM_BINS = 10

#: A show holding more than this share of the corpus's hours is reported as concentration. Half is
#: the point past which "the corpus" and "that show" start to mean the same thing.
DOMINANT_SHARE = 0.5

#: Episodes longer than this are the long-podcast material the corpus is moving away from: the
#: pivot is toward 5-20 minute videos chosen for the speaker they add rather than their duration.
LONG_EPISODE_MINUTES = 45.0

#: Above this share of hours sitting in long episodes, sourcing itself is the thing to change.
LONG_EPISODE_SHARE = 0.5

#: How much each kind of gap matters, before severity scales it. Speakers first: the corpus is
#: limited by how many different people are in it, not by how many hours it holds.
GAP_WEIGHT: dict[str, float] = {
    "gender": 1.0,
    "age_bracket": 0.95,
    "register": 0.9,
    "register_spread": 0.9,
    "show_concentration": 0.8,
    "gold_coverage": 0.7,
    "episode_length": 0.5,
    "topic": 0.4,
}

#: Absent topics are listed, not enumerated: sixteen labels with two carriers between them would
#: otherwise bury every recommendation that matters under a list of categories nobody is short of.
MAX_TOPIC_RECOMMENDATIONS = 3


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


@dataclass(frozen=True)
class Recommendation:
    """One line of the shopping list: what to look for, and the number that says why."""

    #: Which gap this came from; matches a key of :data:`GAP_WEIGHT`.
    kind: str
    #: What to go and find, in the words someone would use to search for it.
    target: str
    #: The measurement behind it. Always a number, never an adjective.
    reason: str
    #: 0-1. Weight of the gap kind times how severe this instance of it is.
    priority: float
    hours_present: float
    #: Hours that would close the gap, where that is a meaningful quantity.
    hours_needed: float


@dataclass
class Inventory:
    """The whole answer: what is here, what is missing, what to record next."""

    generated_at: str = ""
    totals: dict[str, Any] = field(default_factory=dict)
    dimensions: dict[str, Dimension] = field(default_factory=dict)
    #: ``gender x age_bracket`` hours, as ``{gender: {age_bracket: hours}}``. The cells that are
    #: empty are the point of it.
    speaker_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    register: dict[str, Any] = field(default_factory=dict)
    length_profile: dict[str, Any] = field(default_factory=dict)
    metadata_completeness: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    shows: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Plain JSON-serializable data, for the API and the report."""
        return {
            "generated_at": self.generated_at,
            "totals": self.totals,
            "dimensions": {key: asdict(value) for key, value in self.dimensions.items()},
            "speaker_matrix": self.speaker_matrix,
            "register": self.register,
            "length_profile": self.length_profile,
            "metadata_completeness": self.metadata_completeness,
            "episodes": self.episodes,
            "shows": self.shows,
            "recommendations": [asdict(r) for r in self.recommendations],
        }


@dataclass(frozen=True)
class EpisodeFacts:
    """One episode reduced to what the inventory counts. The unit of attribution."""

    external_id: str
    title: str | None
    show_id: str | None
    published_at: str | None
    pot: str
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
    yet, exactly as the pot assigner budgets (D63).
    """
    segment_rows = {
        row.episode_id: row
        for row in session.execute(
            sa.select(
                Segment.episode_id.label("episode_id"),
                sa.func.count().label("segments"),
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
        facts.append(
            EpisodeFacts(
                external_id=episode.external_id,
                title=episode.title,
                show_id=episode.show_id,
                published_at=episode.published_at.isoformat() if episode.published_at else None,
                pot=episode.pot,
                split=episode.split,
                hours=seconds / 3600,
                segments=row.segments if row else 0,
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


def recommend_sources(
    *,
    dimensions: Mapping[str, Dimension],
    register: Mapping[str, Any],
    length_profile: Mapping[str, Any],
    gold_coverage_gaps: Mapping[str, list[str]],
    corpus_hours: float,
    min_stratum_hours: float,
) -> list[Recommendation]:
    """Turn the gaps into a ranked list of what to go and record.

    Pure: it reads aggregates and returns advice, so it is testable without a database and cannot
    be wrong about the corpus in a way the inventory above is not already wrong.

    Priority is ``weight x severity``. Weight is how much the *kind* of gap matters
    (:data:`GAP_WEIGHT`); severity is 1.0 for something entirely absent and
    ``deficit / min_stratum_hours`` for something merely thin, so a stratum at 90% of the floor
    ranks below one at 10% of it.

    Args:
        dimensions: Output of :func:`build_dimension`, keyed by dimension.
        register: Output of :func:`collect_register`.
        length_profile: Output of :func:`build_length_profile`.
        gold_coverage_gaps: ``{key: values}`` the gold pot does not span, from ``pot_status``.
        corpus_hours: Total ingested hours, for the concentration arithmetic.
        min_stratum_hours: Hours below which a stratum counts as thin.

    Returns:
        Recommendations, highest priority first, then alphabetically for a stable order.
    """
    out: list[Recommendation] = []

    def add(kind: str, target: str, reason: str, severity: float, present: float, needed: float):
        out.append(
            Recommendation(
                kind=kind,
                target=target,
                reason=reason,
                priority=_round(min(1.0, GAP_WEIGHT[kind] * min(1.0, max(0.0, severity))), 4),
                hours_present=_round(present),
                hours_needed=_round(max(0.0, needed)),
            )
        )

    # 1. Speaker strata: absent first, then thin. These come first because the corpus is limited
    # by how many different people are in it -- token counts were already sufficient when speaker
    # coverage was not.
    for key in SPEAKER_KEYS:
        dimension = dimensions.get(key)
        if dimension is None:
            continue
        present = {value.value: value for value in dimension.values}
        for value in VOCABULARIES.get(key, ()):
            entry = present.get(value)
            hours = entry.hours if entry else 0.0
            if hours >= min_stratum_hours:
                continue
            noun = "speakers" if key == "gender" else "speakers aged"
            if entry is None:
                add(
                    key,
                    f"{noun} {value.replace('_', ' ')} -- any show, any topic",
                    f"no audio at all; the corpus has {len(dimension.values)} of "
                    f"{len(VOCABULARIES[key])} {key} values",
                    1.0,
                    0.0,
                    min_stratum_hours,
                )
            else:
                add(
                    key,
                    f"more {noun} {value.replace('_', ' ')}",
                    f"{entry.hours:.2f} h across {entry.episodes} episode(s) in "
                    f"{entry.shows} show(s), under the {min_stratum_hours} h floor",
                    (min_stratum_hours - hours) / min_stratum_hours,
                    hours,
                    min_stratum_hours - hours,
                )
        # A value carried by a single show is present but not independent: lose the show and lose
        # the value. Worth saying, at half the severity of a thin one.
        for value in dimension.values:
            if value.shows == 1 and value.hours >= min_stratum_hours:
                add(
                    key,
                    f"a second show with {value.value.replace('_', ' ')} speakers",
                    f"all {value.hours:.2f} h of it comes from one show",
                    0.5,
                    value.hours,
                    0.0,
                )
        # Imbalance among the values that *are* present. Absence is not the only way a stratum
        # fails: a dimension where one value holds five-sixths of the audio supports no comparison
        # across it, and that is invisible to a rule that only asks whether each value exists.
        if dimension.top_share > DOMINANT_SHARE and len(dimension.values) > 1:
            leader, *rest = dimension.values
            trailing = ", ".join(f"{v.value.replace('_', ' ')} {v.hours:.2f} h" for v in rest[:3])
            add(
                key,
                f"speakers who are not {leader.value.replace('_', ' ')}"
                + (f" -- currently {trailing}" if trailing else ""),
                f"{leader.value.replace('_', ' ')} carries {dimension.top_share:.0%} of"
                f" attributed {key} hours across {leader.shows} show(s)",
                dimension.top_share,
                leader.hours,
                _round(leader.hours - sum(v.hours for v in rest)),
            )

    # 2. Register. The dependent variable of a code-switching study: with every show between 0.13
    # and 0.36 there is almost no variance in it to explain.
    for band in register.get("bands", []):
        if band["hours"] >= min_stratum_hours:
            continue
        add(
            "register",
            f"a {band['name']}-code-switching source -- {band['description']}",
            f"{band['hours']:.2f} h of audio between {band['lower']:.2f} and {band['upper']:.2f}"
            f" code-switch density, against {register.get('measured_hours', 0.0):.2f} h measured",
            1.0 if band["hours"] == 0 else (min_stratum_hours - band["hours"]) / min_stratum_hours,
            band["hours"],
            min_stratum_hours - band["hours"],
        )

    # 2b. Register *spread across shows*. The bands above ask whether a pole is missing; this asks
    # whether the poles are far enough apart to be poles at all.
    spread = register.get("show_mean_spread")
    if spread is not None and spread < NARROW_SHOW_SPREAD:
        low, high = register.get("show_mean_min"), register.get("show_mean_max")
        add(
            "register_spread",
            "a source outside the band every show already sits in"
            f" -- below {low:.0%} or above {high:.0%} English",
            f"all {len(register.get('show_means', []))} shows fall between {low:.0%} and"
            f" {high:.0%} code-switch density: {spread:.0%} of spread to explain",
            (NARROW_SHOW_SPREAD - spread) / NARROW_SHOW_SPREAD,
            0.0,
            0.0,
        )

    # 3. Show concentration. A corpus that is mostly one show measures that show, which is D63's
    # argument about the benchmark applied to the whole thing.
    shows = dimensions.get("show_id")
    if shows and shows.values and shows.top_share > DOMINANT_SHARE:
        leader = shows.values[0]
        add(
            "show_concentration",
            "any new show -- breadth matters more than which one",
            f"{leader.value} holds {leader.hours:.2f} h, {shows.top_share:.0%} of the corpus,"
            f" across {len(shows.values)} show(s) total",
            shows.top_share,
            leader.hours,
            max(0.0, leader.hours / DOMINANT_SHARE - corpus_hours),
        )

    # 4. Episode length. Not a defect in itself -- it is a sourcing habit, and the cheapest one to
    # change, because a short video costs a fraction of a long one to ingest and label.
    share = length_profile.get("long_episode_share", 0.0)
    if share > LONG_EPISODE_SHARE:
        add(
            "episode_length",
            "5-20 minute videos rather than full-length podcasts",
            f"{share:.0%} of hours sit in episodes over {LONG_EPISODE_MINUTES:.0f} min"
            f" (median {length_profile.get('median_minutes')} min)",
            (share - LONG_EPISODE_SHARE) / (1.0 - LONG_EPISODE_SHARE),
            length_profile.get("long_episode_hours", 0.0),
            0.0,
        )

    # 5. What the benchmark does not span. Lower weight than the corpus-wide gaps because gold
    # refills from the corpus on the next assignment -- fix the corpus and this often fixes itself.
    for key, values in sorted(gold_coverage_gaps.items()):
        add(
            "gold_coverage",
            f"gold has no {key}: {', '.join(values[:4])}"
            + (f" (+{len(values) - 4} more)" if len(values) > 4 else ""),
            f"{len(values)} value(s) in the corpus that the benchmark does not span",
            min(1.0, len(values) / 4),
            0.0,
            0.0,
        )

    # 6. Topics, capped. A missing topic is the cheapest gap to close and the least informative
    # one, so it is listed rather than enumerated.
    topics = dimensions.get("topic")
    if topics:
        for value in topics.absent[:MAX_TOPIC_RECOMMENDATIONS]:
            add(
                "topic",
                f"an episode about {value.replace('_', ' ')}",
                f"no audio; {len(topics.values)} of {len(TOPIC_LABELS)} topics carried",
                1.0,
                0.0,
                min_stratum_hours,
            )

    out.sort(key=lambda r: (-r.priority, r.kind, r.target))
    return out


def collect_inventory(session: Session, *, settings: Settings | None = None) -> Inventory:
    """Assemble the whole inventory: what is here, what is missing, what to record next.

    Reads only. Opening the analytics page must never change what the corpus holds -- the same
    rule ``pot_status`` follows for the same reason (D63).

    Args:
        session: Open session.
        settings: Configuration override.

    Returns:
        An :class:`Inventory`; call :meth:`Inventory.as_dict` for the API payload.
    """
    settings = settings or get_settings()
    # Imported here rather than at module scope: `pots` imports from `stats`, and keeping the
    # dependency one-directional at import time is what stops a cycle when either grows.
    from app.services.pots import pot_status

    facts = load_episode_facts(session)
    corpus_hours = sum(f.hours for f in facts)
    keys = ("show_id", "topic", "gender", "age_bracket", "role")
    dimensions = {key: build_dimension(facts, key) for key in keys}
    shows = build_shows(facts)
    register = collect_register(session, shows)
    length_profile = build_length_profile(facts)
    pots = pot_status(session, settings=settings)

    profiles = _speaker_profiles(facts)
    labeled_hours = sum(f.labeled_hours for f in facts)
    verified_hours = sum(f.verified_hours for f in facts)

    totals = {
        "episodes": len(facts),
        "segments": sum(f.segments for f in facts),
        "hours": _round(corpus_hours),
        "labeled_hours": _round(labeled_hours),
        "verified_hours": _round(verified_hours),
        "screened_hours": _round(sum(f.screened_hours for f in facts)),
        "labeled_fraction": _round(labeled_hours / corpus_hours, 4) if corpus_hours else 0.0,
        "verified_fraction": _round(verified_hours / labeled_hours, 4) if labeled_hours else 0.0,
        "shows": len(dimensions["show_id"].values),
        # A floor, not a count -- see `_speaker_profiles`.
        "speaker_profiles": len(profiles),
        "gender_values": len(dimensions["gender"].values),
        "age_values": len(dimensions["age_bracket"].values),
        "topics_carried": len(dimensions["topic"].values),
        "topics_total": len(TOPIC_LABELS),
        "gold_hours": _round(pots.buckets["gold"]["hours"]),
        "gold_target_hours": pots.gold_target_hours,
        "train_hours": _round(
            pots.buckets["train"]["hours"] + pots.buckets["val"]["hours"],
        ),
        "train_target_hours": pots.train_target_hours,
        "unassigned_hours": _round(pots.buckets["unassigned"]["hours"]),
        "min_stratum_hours": settings.dataset.min_stratum_hours,
    }

    return Inventory(
        generated_at=dt.datetime.now(dt.UTC).isoformat(),
        totals=totals,
        dimensions=dimensions,
        speaker_matrix=build_speaker_matrix(facts),
        register=register,
        length_profile=length_profile,
        metadata_completeness=build_metadata_completeness(facts),
        episodes=[
            {
                "external_id": f.external_id,
                "title": f.title,
                "show_id": f.show_id,
                "published_at": f.published_at,
                "pot": f.pot,
                "split": f.split,
                "hours": _round(f.hours),
                "minutes": _round(f.hours * 60, 1),
                "segments": f.segments,
                "labeled_hours": _round(f.labeled_hours),
                "verified_hours": _round(f.verified_hours),
                "labeled_fraction": _round(f.labeled_hours / f.hours, 4) if f.hours else 0.0,
                "topic": f.topic,
                "topic_source": f.topic_source,
                "topic_in_taxonomy": f.topic in TOPIC_LABELS if f.topic else None,
                "speakers": [dict(s) for s in f.speakers],
                "mean_cmi": _round(f.mean_cmi, 4) if f.mean_cmi is not None else None,
                "min_cmi": _round(f.min_cmi, 4) if f.min_cmi is not None else None,
                "max_cmi": _round(f.max_cmi, 4) if f.max_cmi is not None else None,
            }
            for f in facts
        ],
        shows=shows,
        recommendations=recommend_sources(
            dimensions=dimensions,
            register=register,
            length_profile=length_profile,
            gold_coverage_gaps=pots.gold_coverage_gaps,
            corpus_hours=corpus_hours,
            min_stratum_hours=settings.dataset.min_stratum_hours,
        ),
    )
