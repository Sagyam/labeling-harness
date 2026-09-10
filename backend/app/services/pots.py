"""The two pots, chosen one clip at a time by hand (D71, superseding D63's assigner).

* **gold** -- the benchmark. The owner puts a clip here with one click and can take it back out.
  Every gold clip is *verified* -- played, read, decided against both -- and exports as the
  ``test`` split whatever its episode is.
* **train** -- every other clip, subdivided into ``train`` and ``val`` by its episode.

D63 assigned whole episodes to gold, greedily against an hours target, before any clip was seen.
That rule is gone: the owner decides which clips are the benchmark. Two things survive because
they are properties of a label, not of a selection rule:

1. **Gold only holds what was listened to.** A clip whose label was screened cannot be moved into
   gold, and ``record_decision`` refuses a screened decision on a clip already there.
2. **Every move is on the record.** Each one writes an ``audit_logs`` row naming the clip, both
   pots and the reason, so the benchmark's history can be reconstructed.

The cost of choosing per clip is stated here rather than discovered later. Clips from one episode
now sit on both sides of the train/test line -- the same speaker, the same topic, sometimes the
next sentence -- so a model trained on this corpus will score better on gold than on an unseen
speaker. ``pot_status`` counts exactly how much of gold that affects, and every export row says
whether its episode spans both pots, so a result can be reported both ways.

Train and val stay per episode: a deterministic hash of the episode id against
``dataset.val_fraction``, drawn at import. They hold data of the same standard, so that line may
be redrawn freely.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import AuditLog, Episode, Segment
from app.models.enums import POTS
from app.services.stats import latest_labels_subquery
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: The split every gold clip exports as. Kept as a name rather than a literal so the one place that
#: ties the two vocabularies together is greppable.
GOLD_SPLIT = "test"

_MAX = float(1 << 64)


class PotError(RuntimeError):
    """A pot move that would break what the gold pot promises."""


@dataclass(frozen=True)
class PotMove:
    """What one call to :func:`set_segment_pot` did."""

    segment_external_id: str
    from_pot: str
    to_pot: str
    changed: bool


def _hash_unit(external_id: str, *, seed: int) -> float:
    """Map ``(external_id, seed)`` to a stable float in [0, 1).

    BLAKE2b rather than :func:`hash`, which is salted per process and would put an episode in a
    different split on every run.
    """
    digest = hashlib.blake2b(f"{external_id}:{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / _MAX


def draw_split(external_id: str, *, val_fraction: float, seed: int) -> str:
    """``train`` or ``val`` for an episode: deterministic, and redrawable by changing ``seed``."""
    return "val" if _hash_unit(external_id, seed=seed) < val_fraction else "train"


def effective_split(pot: str, episode_split: str) -> str:
    """The split one clip exports as: ``test`` in gold, otherwise whatever its episode drew."""
    return GOLD_SPLIT if pot == "gold" else episode_split


def effective_split_sql() -> sa.ColumnElement[str]:
    """:func:`effective_split` as a SQL expression over ``segments`` joined to ``episodes``."""
    return sa.case((Segment.pot == "gold", sa.literal(GOLD_SPLIT)), else_=Episode.split)


def _has_screened_label(session: Session, segment_id: int) -> bool:
    current = latest_labels_subquery()
    return bool(
        session.scalar(
            sa.select(sa.func.count())
            .select_from(current)
            .where(
                current.c.segment_id == segment_id,
                current.c.verification_tier == "screened",
            )
        )
    )


def set_segment_pot(
    session: Session,
    segment: Segment,
    pot: str,
    *,
    actor: str,
    reason: str | None = None,
) -> PotMove:
    """Put one clip in ``pot``, writing an audit row if anything changed.

    Args:
        session: Open session; the caller commits.
        segment: The clip being moved.
        pot: ``gold`` or ``train``.
        actor: Who moved it, for the audit row.
        reason: Optional note carried on the audit row.

    Raises:
        PotError: ``pot`` is not a pot, or the clip is going into gold carrying a screened label.
    """
    if pot not in POTS:
        raise PotError(f"unknown pot {pot!r}; expected one of {', '.join(POTS)}")
    previous = segment.pot
    if pot == previous:
        return PotMove(segment.external_id, previous, pot, changed=False)
    if pot == "gold" and _has_screened_label(session, segment.id):
        raise PotError(
            f"segment {segment.external_id} was screened, not listened to; gold only holds"
            " verified clips, so verify it first"
        )

    segment.pot = pot
    new_values: dict[str, Any] = {"pot": pot}
    if reason:
        new_values["reason"] = reason
    session.add(
        AuditLog(
            entity_type="segment",
            entity_id=str(segment.id),
            action="pot_changed",
            actor=actor,
            old_values_jsonb={"pot": previous},
            new_values_jsonb=new_values,
        )
    )
    logger.info("segment_pot_changed", segment=segment.external_id, from_pot=previous, to_pot=pot)
    return PotMove(segment.external_id, previous, pot, changed=True)


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


def _coverage_counts(episodes: Iterable[Episode], keys: list[str]) -> dict[str, dict[str, int]]:
    """``{key: {value: how many of these episodes carry it}}``."""
    counts: dict[str, dict[str, int]] = {key: {} for key in keys}
    for episode in episodes:
        coverage = episode_coverage(episode, keys=keys)
        for key in keys:
            for value in coverage.get(key, frozenset()):
                counts[key][value] = counts[key].get(value, 0) + 1
    return counts


@dataclass
class PotStatus:
    """What the pots hold right now. Read-only: moves nothing."""

    gold_target_hours: float = 0.0
    train_target_hours: float = 0.0
    #: ``{bucket: {episodes, segments, hours, labeled_hours}}`` for ``gold``, ``train``, ``val``
    #: and ``unassigned``. Gold is counted by clip; the other three by the episode's split, over
    #: the clips that are not gold. ``episodes`` is distinct episodes with a clip in the bucket.
    buckets: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Episodes that have clips in gold *and* in train -- the speaker/topic overlap that per-clip
    #: selection allows -- and how many gold clips come from them.
    gold_episodes_spanning_pots: int = 0
    gold_segments_in_spanning_episodes: int = 0
    gold_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    corpus_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    gold_coverage_gaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def gold_hours(self) -> float:
        return self.buckets.get("gold", {}).get("hours", 0.0)

    @property
    def coverage_complete(self) -> bool:
        """Whether gold covers every stratification value the corpus has."""
        return not self.gold_coverage_gaps


def pot_status(session: Session, *, settings: Settings | None = None) -> PotStatus:
    """Report what the pots currently hold, without moving anything.

    ``labeled_hours`` counts audio that has a current label of any disposition -- the quantity
    that moves when the annotator works, as against ``hours``, which moves only when something is
    ingested or a clip changes pot.
    """
    settings = settings or get_settings()
    keys = list(settings.dataset.coverage_keys)
    current = latest_labels_subquery()
    labeled = sa.select(current.c.segment_id).distinct().subquery()

    bucket_expr = sa.case((Segment.pot == "gold", sa.literal("gold")), else_=Episode.split)
    rows = session.execute(
        sa.select(
            bucket_expr.label("bucket"),
            sa.func.count(Segment.id),
            sa.func.count(sa.distinct(Segment.episode_id)),
            sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0),
            sa.func.coalesce(
                sa.func.sum(
                    sa.case((labeled.c.segment_id.is_not(None), Segment.duration_seconds), else_=0)
                ),
                0.0,
            ),
        )
        .join(Episode, Episode.id == Segment.episode_id)
        .outerjoin(labeled, labeled.c.segment_id == Segment.id)
        .group_by(bucket_expr)
    ).all()

    status = PotStatus(
        gold_target_hours=settings.dataset.gold_hours_target,
        train_target_hours=settings.dataset.train_hours_target,
        buckets={
            name: {"episodes": 0, "segments": 0, "hours": 0.0, "labeled_hours": 0.0}
            for name in ("gold", "train", "val", "unassigned")
        },
    )
    for bucket, segments, episodes, seconds, labeled_seconds in rows:
        entry = status.buckets.setdefault(
            bucket, {"episodes": 0, "segments": 0, "hours": 0.0, "labeled_hours": 0.0}
        )
        entry["episodes"] = int(episodes)
        entry["segments"] = int(segments)
        entry["hours"] = round(float(seconds) / 3600, 4)
        entry["labeled_hours"] = round(float(labeled_seconds) / 3600, 4)

    pots_by_episode: dict[int, dict[str, int]] = {}
    for episode_id, pot, count in session.execute(
        sa.select(Segment.episode_id, Segment.pot, sa.func.count()).group_by(
            Segment.episode_id, Segment.pot
        )
    ).all():
        pots_by_episode.setdefault(episode_id, {})[pot] = int(count)
    spanning = [
        counts for counts in pots_by_episode.values() if counts.get("gold") and counts.get("train")
    ]
    status.gold_episodes_spanning_pots = len(spanning)
    status.gold_segments_in_spanning_episodes = sum(counts["gold"] for counts in spanning)

    episodes = list(session.scalars(sa.select(Episode).order_by(Episode.external_id)))
    gold_episodes = [e for e in episodes if pots_by_episode.get(e.id, {}).get("gold")]
    status.gold_coverage = _coverage_counts(gold_episodes, keys)
    status.corpus_coverage = _coverage_counts(episodes, keys)
    status.gold_coverage_gaps = {
        key: sorted(set(status.corpus_coverage[key]) - set(status.gold_coverage[key]))
        for key in keys
        if set(status.corpus_coverage[key]) - set(status.gold_coverage[key])
    }
    return status
