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
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.llm.topic import TOPIC_LABELS
from app.services.inventory.constants import DOMINANT_SHARE, NARROW_SHOW_SPREAD
from app.services.inventory.dimensions import (
    Dimension,
    DimensionValue,
    _round,
    _speaker_profiles,
    build_dimension,
    build_length_profile,
    build_metadata_completeness,
    build_shows,
    build_speaker_matrix,
    collect_register,
)
from app.services.inventory.facts import EpisodeFacts, load_episode_facts
from app.services.inventory.recommendations import Recommendation, recommend_sources

__all__ = [
    "DOMINANT_SHARE",
    "NARROW_SHOW_SPREAD",
    "Dimension",
    "DimensionValue",
    "EpisodeFacts",
    "Inventory",
    "Recommendation",
    "build_dimension",
    "build_length_profile",
    "build_metadata_completeness",
    "build_shows",
    "build_speaker_matrix",
    "collect_inventory",
    "collect_register",
    "load_episode_facts",
    "recommend_sources",
]


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
                "gold_segments": f.gold_segments,
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
