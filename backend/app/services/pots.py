"""Pot assignment: which episodes are the benchmark, and which are training material (D63).

The corpus has two pots.

* **gold** -- the benchmark. Every clip in it is *verified*: played, read, decided against both.
  Exported as the ``test`` split.
* **train** -- everything else, subdivided into ``train`` and ``val``. A clip here may be verified
  or *screened* -- waved through on the cross-ASR disagreement signal without listening.

Three rules make the arrangement defensible, and each of them exists because its absence produces a
number that looks fine and is wrong.

1. **The pot is assigned to a whole episode, before any of its clips is seen.** If the pot were
   chosen per clip while looking at it, the choice would correlate with how hard the clip turned
   out to be -- route the hard ones to gold and the benchmark is pessimistic, route the quick ones
   and it is optimistic -- and nothing recorded afterwards could separate the two. Assigning ahead
   of time makes the routing independent of the content by construction.

2. **Whole episodes, never clips.** VAD cuts are contiguous and D25 pads speech at the edges, so
   consecutive clips share audio samples; splitting inside an episode puts two halves of one
   sentence on both sides of the train/test line. Within an episode the speaker, the topic and the
   rare vocabulary are also shared. This is the part of the old D5 rule that survives -- its
   *reason* was recording conditions, which D25's loudnorm and D39's resample already flatten, but
   its conclusion was right for reasons D5 barely stated.

3. **Gold is one-directional.** An episode never leaves the gold pot, and by default never enters
   it from train either: a recording that was trained on and is later promoted to the benchmark
   invalidates every number measured against it, silently. ``allow_promote_from_train`` exists for
   the period before anything has actually been trained, and says so at the call site.

Targets are **durations**. The old scheme hashed the episode id into train/val/test, which can
express a ratio but never an amount -- there was no way to ask it for five hours of benchmark
audio, and a ratio over episode *count* says nothing about duration when episodes run from minutes
to four hours. It also stratified nothing, so an all-one-show benchmark was a live possibility.

Selection is greedy on **coverage first, duration second**: of the episodes that still fit, take
the one adding the most unseen show / gender / age bracket / topic. A five-hour benchmark drawn
from one show measures that show.

The target is also capped at ``dataset.gold_max_corpus_fraction`` of the ingested corpus. A corpus
smaller than the target would otherwise be swallowed whole -- every episode locked irreversibly
into the benchmark by rule 3, nothing left to train on, and the benchmark's coverage frozen at the
point when there was least evidence about what it should span. Gold instead grows with the corpus
and reaches its target once there is enough audio to spare.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Episode, Segment
from app.services.stats import latest_labels_subquery
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Split every gold-pot episode takes. Kept as a name rather than a literal so the one place that
#: ties the two vocabularies together is greppable.
GOLD_SPLIT = "test"

_MAX = float(1 << 64)


class PotError(RuntimeError):
    """The pot assignment could not be produced."""


@dataclass(frozen=True)
class Candidate:
    """One episode the assigner may place, with everything the decision needs."""

    episode_id: int
    external_id: str
    seconds: float
    pot: str
    split: str
    #: ``{coverage key: values this episode contributes}``. An episode with two speakers
    #: contributes two genders; one with no metadata contributes nothing and is chosen on
    #: duration alone.
    coverage: Mapping[str, frozenset[str]]

    @property
    def hours(self) -> float:
        return self.seconds / 3600


@dataclass
class PotChange:
    """One episode whose pot or split the assigner moved."""

    external_id: str
    from_pot: str
    to_pot: str
    from_split: str
    to_split: str
    hours: float


@dataclass
class PotReport:
    """What an assignment run did, or would have done in a dry run."""

    gold_hours: float = 0.0
    train_hours: float = 0.0
    val_hours: float = 0.0
    unassigned_hours: float = 0.0
    gold_target_hours: float = 0.0
    #: The target actually pursued this run: the configured one, capped at a share of the ingested
    #: corpus. Below ``gold_target_hours`` while the corpus is still small.
    gold_effective_target_hours: float = 0.0
    train_target_hours: float = 0.0
    gold_episodes: int = 0
    train_episodes: int = 0
    val_episodes: int = 0
    unassigned_episodes: int = 0
    changes: list[PotChange] = field(default_factory=list)
    #: ``{coverage key: {value: episodes in gold carrying it}}`` -- what the benchmark actually
    #: spans, which is the question a reviewer asks about a test set.
    gold_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Values present in the corpus that the gold pot does not cover at all.
    gold_coverage_gaps: dict[str, list[str]] = field(default_factory=dict)
    gold_target_met: bool = False
    #: True when the corpus is too small to reach the target without swallowing it whole.
    gold_capped_by_corpus_size: bool = False
    #: Episodes the assigner refused to place in gold because they already hold screened
    #: labels. Named rather than counted: declining to promote an episode silently is how
    #: a benchmark ends up smaller than anyone intended, with no record of why.
    gold_ineligible: list[str] = field(default_factory=list)
    #: Episodes taken *out* of gold because they hold screened labels -- rule 3's one
    #: exception, for an episode that was never validly in the benchmark to begin with.
    gold_demoted: list[str] = field(default_factory=list)
    dry_run: bool = False

    def render(self) -> str:
        """A short human-readable summary, for the CLI."""
        lines = [
            f"{'DRY RUN -- ' if self.dry_run else ''}pot assignment",
            f"  gold        {self.gold_hours:.2f} h / {self.gold_target_hours:.2f} h target"
            f"  ({self.gold_episodes} episodes)"
            f"{'  TARGET MET' if self.gold_target_met else '  SHORT'}"
            + (
                f" (capped at {self.gold_effective_target_hours:.2f} h until the corpus grows)"
                if self.gold_capped_by_corpus_size
                else ""
            ),
            f"  train       {self.train_hours:.2f} h ({self.train_episodes} episodes)",
            f"  val         {self.val_hours:.2f} h ({self.val_episodes} episodes)",
            f"  unassigned  {self.unassigned_hours:.2f} h ({self.unassigned_episodes} episodes)",
            f"  changes     {len(self.changes)}",
        ]
        if self.gold_demoted:
            lines.append(f"  DEMOTED FROM GOLD ({len(self.gold_demoted)}): holds screened labels")
            for external_id in self.gold_demoted:
                lines.append(f"    {external_id}")
        if self.gold_ineligible:
            lines.append(f"  held out of gold ({len(self.gold_ineligible)}): already screened")
            for external_id in self.gold_ineligible:
                lines.append(f"    {external_id}")
        for change in self.changes:
            lines.append(
                f"    {change.external_id}  {change.from_pot}/{change.from_split}"
                f" -> {change.to_pot}/{change.to_split}  ({change.hours:.2f} h)"
            )
        if self.gold_coverage_gaps:
            lines.append("  gold coverage gaps")
            for key, values in sorted(self.gold_coverage_gaps.items()):
                lines.append(f"    {key:<12} missing {', '.join(sorted(values))}")
        return "\n".join(lines)


def _hash_unit(external_id: str, *, seed: int) -> float:
    """Map ``(external_id, seed)`` to a stable float in [0, 1).

    BLAKE2b rather than :func:`hash`, which is salted per process and would put an episode in a
    different split on every run.
    """
    digest = hashlib.blake2b(f"{external_id}:{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / _MAX


def episode_coverage(episode: Episode, *, keys: Iterable[str]) -> dict[str, frozenset[str]]:
    """What stratification values one episode contributes, per coverage key.

    ``show_id`` comes off the column; ``topic``, ``gender`` and ``age_bracket`` come out of
    ``metadata_jsonb``, whose ``speakers`` block the importer has already reduced to the D56
    allowlist. A key the episode says nothing about maps to an empty set rather than to a guess --
    an unknown gender is not a gender, and stratifying on a fabricated one is how a coverage claim
    becomes false.
    """
    metadata: Mapping[str, Any] = episode.metadata_jsonb or {}
    speakers = metadata.get("speakers")
    speaker_values: dict[str, set[str]] = {"gender": set(), "age_bracket": set()}
    if isinstance(speakers, Mapping):
        for speaker in speakers.values():
            if not isinstance(speaker, Mapping):
                continue
            for field_name in ("gender", "age_bracket"):
                value = speaker.get(field_name)
                if isinstance(value, str) and value:
                    speaker_values[field_name].add(value)

    out: dict[str, frozenset[str]] = {}
    for key in keys:
        if key == "show_id":
            out[key] = frozenset({episode.show_id} if episode.show_id else ())
        elif key == "topic":
            topic = metadata.get("topic")
            out[key] = frozenset({topic} if isinstance(topic, str) and topic else ())
        elif key in speaker_values:
            out[key] = frozenset(speaker_values[key])
        else:  # pragma: no cover - the settings vocabulary is closed
            out[key] = frozenset()
    return out


def _load_candidates(session: Session, *, keys: list[str]) -> list[Candidate]:
    """Every episode, with the duration the assigner budgets against.

    That duration is the summed length of the episode's *segments* -- the audio that actually
    reaches the queue -- falling back to ``episodes.duration_seconds`` for an episode whose
    segments are not imported yet. Budgeting against the raw recording would over-count the
    silence VAD already removed.
    """
    segment_seconds = dict(
        session.execute(
            sa.select(
                Segment.episode_id,
                sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0),
            ).group_by(Segment.episode_id)
        ).all()
    )
    episodes = session.scalars(sa.select(Episode).order_by(Episode.external_id)).all()
    candidates = []
    for episode in episodes:
        seconds = float(segment_seconds.get(episode.id) or 0.0)
        if seconds <= 0.0:
            seconds = float(episode.duration_seconds or 0.0)
        candidates.append(
            Candidate(
                episode_id=episode.id,
                external_id=episode.external_id,
                seconds=seconds,
                pot=episode.pot,
                split=episode.split,
                coverage=episode_coverage(episode, keys=keys),
            )
        )
    return candidates


def screened_episode_ids(session: Session) -> set[int]:
    """Episodes carrying at least one label that nobody listened to.

    Screening is legal on an unassigned episode and refused on a gold one, which leaves a window:
    screen an episode first, run the assigner second, and the benchmark quietly acquires rows that
    were waved through on the disagreement signal. `record_decision` cannot catch that -- the
    decision was already legal when it was made -- so the assigner has to.
    """
    current = latest_labels_subquery()
    rows = session.execute(
        sa.select(sa.distinct(Segment.episode_id))
        .join(current, current.c.segment_id == Segment.id)
        .where(current.c.verification_tier == "screened")
    ).all()
    return {row[0] for row in rows}


def _coverage_gain(candidate: Candidate, seen: Mapping[str, set[str]], keys: list[str]) -> int:
    """How many stratification values this episode would add that gold does not already have.

    Keys are weighted by their position in ``coverage_keys``: the first key listed is the one a
    gap in hurts most, so an episode bringing a new show outranks one bringing a new age bracket.
    """
    total = 0
    for rank, key in enumerate(keys):
        weight = len(keys) - rank
        total += weight * len(candidate.coverage.get(key, frozenset()) - seen.get(key, set()))
    return total


def _select_gold(
    candidates: list[Candidate],
    *,
    target_seconds: float,
    already_gold: list[Candidate],
    keys: list[str],
    seed: int,
) -> list[Candidate]:
    """Greedily fill the gold pot: most new coverage first, then longest that still fits.

    Coverage before duration is deliberate. Duration alone would take the longest episodes, which
    is the fastest way to five hours and the most likely way to five hours of one show.
    """
    seen: dict[str, set[str]] = {key: set() for key in keys}
    seconds = 0.0
    for candidate in already_gold:
        seconds += candidate.seconds
        for key in keys:
            seen[key] |= candidate.coverage.get(key, frozenset())

    remaining = [c for c in candidates if c.seconds > 0]
    chosen: list[Candidate] = []
    while seconds < target_seconds and remaining:
        deficit = target_seconds - seconds
        # An episode longer than the deficit still counts: overshooting the target by part of one
        # episode is the price of rule 2, and undershooting a benchmark is worse than exceeding it.
        best = max(
            remaining,
            key=lambda c: (
                _coverage_gain(c, seen, keys),
                # Prefer the one that lands closest to the deficit without leaving a sliver.
                -abs(c.seconds - deficit),
                # Deterministic tie-break; a plain sort on external_id would bias toward one show.
                _hash_unit(c.external_id, seed=seed),
            ),
        )
        remaining.remove(best)
        chosen.append(best)
        seconds += best.seconds
        for key in keys:
            seen[key] |= best.coverage.get(key, frozenset())
    return chosen


def assign_pots(
    session: Session,
    *,
    settings: Settings | None = None,
    gold_hours_target: float | None = None,
    gold_max_corpus_fraction: float | None = None,
    allow_promote_from_train: bool = False,
    dry_run: bool = False,
) -> PotReport:
    """Assign every episode to the gold or train pot, and subdivide train into train/val.

    Args:
        session: Open session; the caller commits.
        settings: Configuration override.
        gold_hours_target: Hours of audio the gold pot should hold. Defaults to
            ``dataset.gold_hours_target``.
        gold_max_corpus_fraction: Ceiling on gold's share of the ingested corpus, which keeps a
            corpus smaller than the target from being swallowed whole. Defaults to
            ``dataset.gold_max_corpus_fraction``.
        allow_promote_from_train: Let an episode already in the train pot move into gold. Off by
            default, and it should stay off once anything has been trained: promoting a recording
            the model has already seen turns the benchmark into a memorization test without
            changing any number that would show it.
        dry_run: Report what would change and write nothing.

    Returns:
        A :class:`PotReport` describing the assignment and what the gold pot covers.

    Raises:
        PotError: The gold target is not a positive number of hours.
    """
    settings = settings or get_settings()
    target_hours = (
        settings.dataset.gold_hours_target if gold_hours_target is None else gold_hours_target
    )
    if target_hours <= 0:
        raise PotError(f"gold_hours_target must be positive, got {target_hours}")

    keys = list(settings.dataset.coverage_keys)
    seed = settings.dataset.pot_seed
    val_fraction = settings.dataset.val_fraction

    candidates = _load_candidates(session, keys=keys)
    screened = screened_episode_ids(session)
    # Rule 3 keeps episodes in gold so a trained-on recording cannot become the benchmark. It was
    # never meant to pin an episode there that gold's own definition excludes: a clip nobody
    # listened to. An episode screened while unassigned and then placed in gold on the hours
    # target was never validly gold, so it is put back rather than left to fail at export.
    demoted = [c for c in candidates if c.pot == "gold" and c.episode_id in screened]
    already_gold = [c for c in candidates if c.pot == "gold" and c.episode_id not in screened]

    # Cap the target at a share of what has actually been ingested, so a corpus smaller than the
    # target is not swallowed whole. Episodes already in gold are past this: rule 3 means they
    # cannot be given back, so the cap governs what is taken *next*, not what is held.
    corpus_hours = sum(c.hours for c in candidates)
    max_fraction = (
        settings.dataset.gold_max_corpus_fraction
        if gold_max_corpus_fraction is None
        else gold_max_corpus_fraction
    )
    effective_target = min(target_hours, corpus_hours * max_fraction)
    effective_target = max(effective_target, sum(c.hours for c in already_gold))
    eligible = [
        c
        for c in candidates
        if c.pot == "unassigned" or (allow_promote_from_train and c.pot == "train")
    ]
    # An episode that has already been screened can never be the benchmark, whatever it would do
    # for coverage or for the hours target. Undershooting gold is recoverable; a gold pot nobody
    # listened to is not, and it fails silently at every point except the export.
    ineligible = [c for c in eligible if c.episode_id in screened]
    eligible = [c for c in eligible if c.episode_id not in screened]

    newly_gold = _select_gold(
        eligible,
        target_seconds=effective_target * 3600,
        already_gold=already_gold,
        keys=keys,
        seed=seed,
    )
    gold_ids = {c.episode_id for c in already_gold} | {c.episode_id for c in newly_gold}

    report = PotReport(
        gold_ineligible=[c.external_id for c in ineligible],
        gold_demoted=[c.external_id for c in demoted],
        gold_target_hours=target_hours,
        gold_effective_target_hours=round(effective_target, 4),
        train_target_hours=settings.dataset.train_hours_target,
        dry_run=dry_run,
    )

    episodes = {
        episode.id: episode
        for episode in session.scalars(
            sa.select(Episode).where(Episode.id.in_([c.episode_id for c in candidates]))
        )
    }
    now = dt.datetime.now(dt.UTC)

    for candidate in candidates:
        if candidate.episode_id in gold_ids:
            pot, split = "gold", GOLD_SPLIT
        elif candidate.seconds <= 0:
            # Nothing to budget with yet -- an episode whose segments have not been imported. It
            # stays unassigned rather than silently landing in train, so the next run can still
            # consider it for gold once its duration is known.
            pot, split = "unassigned", "unassigned"
        else:
            pot = "train"
            unit = _hash_unit(candidate.external_id, seed=seed)
            split = "val" if unit < val_fraction else "train"

        if pot == "gold":
            report.gold_hours += candidate.hours
            report.gold_episodes += 1
        elif split == "val":
            report.val_hours += candidate.hours
            report.val_episodes += 1
        elif pot == "train":
            report.train_hours += candidate.hours
            report.train_episodes += 1
        else:
            report.unassigned_hours += candidate.hours
            report.unassigned_episodes += 1

        if pot != candidate.pot or split != candidate.split:
            report.changes.append(
                PotChange(
                    external_id=candidate.external_id,
                    from_pot=candidate.pot,
                    to_pot=pot,
                    from_split=candidate.split,
                    to_split=split,
                    hours=candidate.hours,
                )
            )
            if not dry_run:
                episode = episodes[candidate.episode_id]
                episode.pot = pot
                episode.split = split
                episode.split_seed = seed
                episode.split_assigned_at = now
                if pot != candidate.pot:
                    episode.pot_assigned_at = now

    report.gold_hours = round(report.gold_hours, 4)
    report.train_hours = round(report.train_hours, 4)
    report.val_hours = round(report.val_hours, 4)
    report.unassigned_hours = round(report.unassigned_hours, 4)
    report.gold_target_met = report.gold_hours >= target_hours
    report.gold_capped_by_corpus_size = effective_target < target_hours

    gold_candidates = [c for c in candidates if c.episode_id in gold_ids]
    report.gold_coverage = _coverage_counts(gold_candidates, keys)
    corpus_coverage = _coverage_counts(candidates, keys)
    report.gold_coverage_gaps = {
        key: sorted(set(corpus_coverage[key]) - set(report.gold_coverage[key]))
        for key in keys
        if set(corpus_coverage[key]) - set(report.gold_coverage[key])
    }

    logger.info(
        "pots_assigned",
        gold_hours=report.gold_hours,
        train_hours=report.train_hours,
        val_hours=report.val_hours,
        changes=len(report.changes),
        dry_run=dry_run,
    )
    return report


def _coverage_counts(candidates: list[Candidate], keys: list[str]) -> dict[str, dict[str, int]]:
    """``{key: {value: how many of these episodes carry it}}``."""
    counts: dict[str, dict[str, int]] = {key: {} for key in keys}
    for candidate in candidates:
        for key in keys:
            for value in candidate.coverage.get(key, frozenset()):
                counts[key][value] = counts[key].get(value, 0) + 1
    return counts


@dataclass
class PotStatus:
    """What the pots hold right now. Read-only: assigns nothing, moves nothing."""

    gold_target_hours: float = 0.0
    gold_effective_target_hours: float = 0.0
    gold_capped_by_corpus_size: bool = False
    train_target_hours: float = 0.0
    #: ``{pot or split name: {episodes, segments, hours, labeled_hours}}`` keyed by the four names
    #: the dashboard shows: ``gold``, ``train``, ``val``, ``unassigned``.
    buckets: dict[str, dict[str, float]] = field(default_factory=dict)
    gold_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    corpus_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    gold_coverage_gaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def gold_hours(self) -> float:
        return self.buckets.get("gold", {}).get("hours", 0.0)

    @property
    def train_hours(self) -> float:
        return self.buckets.get("train", {}).get("hours", 0.0)

    @property
    def val_hours(self) -> float:
        return self.buckets.get("val", {}).get("hours", 0.0)

    @property
    def coverage_complete(self) -> bool:
        """Whether gold covers every stratification value the corpus has."""
        return not self.gold_coverage_gaps


def pot_status(session: Session, *, settings: Settings | None = None) -> PotStatus:
    """Report what the pots currently hold, without assigning anything.

    Separate from :func:`assign_pots` on purpose: the dashboard asks this on every page load, and a
    read that could move an episode between pots as a side effect of being looked at would be a
    trap. ``labeled_hours`` counts audio that has a current label of any disposition -- the
    quantity that actually moves when the annotator works, as against ``hours``, which moves only
    when something is ingested.
    """
    settings = settings or get_settings()
    keys = list(settings.dataset.coverage_keys)

    current = latest_labels_subquery()
    labeled_seconds = dict(
        session.execute(
            sa.select(
                Segment.episode_id,
                sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0),
            )
            .select_from(current)
            .join(Segment, Segment.id == current.c.segment_id)
            .group_by(Segment.episode_id)
        ).all()
    )
    segment_counts = dict(
        session.execute(
            sa.select(Segment.episode_id, sa.func.count()).group_by(Segment.episode_id)
        ).all()
    )

    status = PotStatus(
        gold_target_hours=settings.dataset.gold_hours_target,
        train_target_hours=settings.dataset.train_hours_target,
        buckets={
            name: {"episodes": 0, "segments": 0, "hours": 0.0, "labeled_hours": 0.0}
            for name in ("gold", "train", "val", "unassigned")
        },
    )

    candidates = _load_candidates(session, keys=keys)
    gold: list[Candidate] = []
    for candidate in candidates:
        if candidate.pot == "gold":
            bucket = "gold"
            gold.append(candidate)
        elif candidate.pot == "train":
            bucket = "val" if candidate.split == "val" else "train"
        else:
            bucket = "unassigned"
        entry = status.buckets[bucket]
        entry["episodes"] += 1
        entry["segments"] += segment_counts.get(candidate.episode_id, 0)
        entry["hours"] += candidate.hours
        entry["labeled_hours"] += float(labeled_seconds.get(candidate.episode_id) or 0.0) / 3600

    for entry in status.buckets.values():
        entry["hours"] = round(entry["hours"], 4)
        entry["labeled_hours"] = round(entry["labeled_hours"], 4)

    corpus_hours = sum(c.hours for c in candidates)
    effective = min(
        settings.dataset.gold_hours_target,
        corpus_hours * settings.dataset.gold_max_corpus_fraction,
    )
    status.gold_effective_target_hours = round(max(effective, status.gold_hours), 4)
    status.gold_capped_by_corpus_size = (
        status.gold_effective_target_hours < settings.dataset.gold_hours_target
    )

    status.gold_coverage = _coverage_counts(gold, keys)
    status.corpus_coverage = _coverage_counts(candidates, keys)
    status.gold_coverage_gaps = {
        key: sorted(set(status.corpus_coverage[key]) - set(status.gold_coverage[key]))
        for key in keys
        if set(status.corpus_coverage[key]) - set(status.gold_coverage[key])
    }
    return status
