"""Ranked sourcing recommendations: the inventory's gaps turned into a shopping list."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.llm.topic import TOPIC_LABELS
from app.services.inventory.constants import (
    DOMINANT_SHARE,
    GAP_WEIGHT,
    LONG_EPISODE_MINUTES,
    LONG_EPISODE_SHARE,
    MAX_TOPIC_RECOMMENDATIONS,
    NARROW_SHOW_SPREAD,
    SPEAKER_KEYS,
    VOCABULARIES,
)
from app.services.inventory.dimensions import Dimension
from app.services.inventory.facts import _round
from app.utils.logging import get_logger

logger = get_logger(__name__)


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
