"""Ranked advice, one category at a time: what to record next, and for which purpose (D91)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.inventory.categories import CategoryReport
from app.services.inventory.constants import (
    ASR,
    CATEGORY_WEIGHT,
    DOMINANT_SHARE,
    HOSTS_WANTED,
    KIND_WEIGHT,
    MAX_TOPIC_RECOMMENDATIONS,
    MIN_VOICE_WORDS,
    PAPER,
    RECURRING_HOST_EPISODES,
    UNMEASURED_FIX,
    UNMEASURED_SHARE,
    VERIFIED_FLOOR_HOURS,
    VOICE_DOMINANT_SHARE,
)
from app.services.inventory.facts import _round


@dataclass(frozen=True)
class Recommendation:
    """One line of the shopping list.

    Attributes:
        category: The category key it came from; the page filters on it.
        bucket: The bucket, when the advice is about one.
        goal: ``asr``, ``paper`` or ``both`` -- which purpose the gap hurts.
        kind: A key of :data:`KIND_WEIGHT`.
        target: What to go and find, in the words someone would search for.
        reason: The measurement behind it. Always numbers, never adjectives.
        priority: 0-1: category weight x kind weight x severity.
        hours_present: Hours in the bucket today.
        voices_present: Usable voices in the bucket today.
        hours_needed: Hours that would close the gap, where that is meaningful.
        voices_needed: Voices that would close the gap, where that is meaningful.
    """

    category: str
    bucket: str | None
    goal: str
    kind: str
    target: str
    reason: str
    priority: float
    hours_present: float
    voices_present: int
    hours_needed: float
    voices_needed: int


def _goal(goals: Sequence[str]) -> str:
    return "both" if ASR in goals and PAPER in goals else goals[0]


_PRETTY = {
    "under_20": "under 20",
    "80_plus": "80+",
    "20_39": "20-39",
    "40_59": "40-59",
    "60_79": "60-79",
}


def _pretty(bucket: str) -> str:
    return _PRETTY.get(bucket, bucket.replace("_", " "))


def recommend(
    categories: Mapping[str, CategoryReport],
    voice_summary: Mapping[str, Any],
    *,
    min_stratum_hours: float,
    min_stratum_voices: int,
) -> list[Recommendation]:
    """Turn every category's gaps into ranked advice.

    Pure: it reads the category reports and the voice summary and returns advice, so it is
    testable from a description of a corpus and cannot be wrong about the corpus in a way the
    reports are not already wrong.

    Rules, per category and in this order: absent buckets; thin buckets (short of voices where
    the category's unit is voices, of hours where it is hours); a bucket carried by one show; a
    dominant bucket; a bucket the gold pot does not span; a bucket a paper would compare on that
    rests on screened labels; and a category whose unmeasured share is too large to trust. Then,
    for voices: the recurrence the accommodation design needs.
    """
    out: list[Recommendation] = []

    def add(
        report: CategoryReport,
        bucket: str | None,
        kind: str,
        target: str,
        reason: str,
        severity: float,
        *,
        goal: str | None = None,
        hours: float = 0.0,
        voices: int = 0,
        hours_needed: float = 0.0,
        voices_needed: int = 0,
    ) -> None:
        weight = CATEGORY_WEIGHT.get(report.key, 0.5) * KIND_WEIGHT[kind]
        out.append(
            Recommendation(
                category=report.key,
                bucket=bucket,
                goal=goal or _goal(report.goals),
                kind=kind,
                target=target,
                reason=reason,
                priority=_round(min(1.0, weight * min(1.0, max(0.0, severity))), 4),
                hours_present=_round(hours),
                voices_present=voices,
                hours_needed=_round(max(0.0, hours_needed)),
                voices_needed=max(0, voices_needed),
            )
        )

    for report in categories.values():
        skip = set(report.unknown) | set(report.defect)
        measured = [b for b in report.buckets if b.bucket not in skip]
        if report.key == "voice":
            _voice_rules(report, voice_summary, add, min_stratum_voices=min_stratum_voices)
            continue

        # 1. Absent. Only a closed vocabulary can say what is missing.
        absent = report.absent
        if report.key == "topic":
            absent = absent[:MAX_TOPIC_RECOMMENDATIONS]
        for bucket in absent:
            noun = "voices" if report.unit == "voices" else "hours"
            add(
                report, bucket, "absent",
                f"{report.label.lower()}: any {_pretty(bucket)} -- no audio at all",
                f"0 h and 0 voices; {len(measured) - len(report.absent)} of {len(measured)}"
                f" {report.label.lower()} buckets carried",
                1.0,
                hours_needed=min_stratum_hours if noun == "hours" else 0.0,
                voices_needed=min_stratum_voices if noun == "voices" else 0,
            )  # fmt: skip

        # 2. Thin, in the category's own unit. Not for shows: a show is where audio comes from,
        # not a stratum anyone fills.
        for entry in measured if report.key != "show" else ():
            if entry.hours <= 0:
                continue
            if report.unit == "voices":
                if entry.usable_voices >= min_stratum_voices:
                    continue
                add(
                    report, entry.bucket, "thin",
                    f"{report.label.lower()}: more {_pretty(entry.bucket)} speakers",
                    f"{entry.usable_voices} voice(s) with {MIN_VOICE_WORDS}+ words across"
                    f" {entry.episodes} episode(s) in {entry.shows} show(s), {entry.hours:.2f} h;"
                    f" under the {min_stratum_voices}-voice floor",
                    (min_stratum_voices - entry.usable_voices) / min_stratum_voices,
                    hours=entry.hours, voices=entry.usable_voices,
                    voices_needed=min_stratum_voices - entry.usable_voices,
                )  # fmt: skip
            elif entry.hours < min_stratum_hours:
                add(
                    report, entry.bucket, "thin",
                    f"{report.label.lower()}: more audio at {_pretty(entry.bucket)}",
                    f"{entry.hours:.2f} h in {entry.clips} clip(s) from {entry.voices} voice(s),"
                    f" under the {min_stratum_hours} h floor",
                    (min_stratum_hours - entry.hours) / min_stratum_hours,
                    hours=entry.hours, voices=entry.usable_voices,
                    hours_needed=min_stratum_hours - entry.hours,
                )  # fmt: skip

        # 3. Carried by one show: present, and not independent of that show.
        if report.key != "show":
            for entry in measured:
                thin = (
                    entry.usable_voices < min_stratum_voices
                    if report.unit == "voices"
                    else entry.hours < min_stratum_hours
                )
                if entry.shows == 1 and not thin:
                    add(
                        report, entry.bucket, "single_show",
                        f"{report.label.lower()}: a second show with {_pretty(entry.bucket)}",
                        f"all {entry.hours:.2f} h and {entry.usable_voices} voice(s) come from"
                        " one show",
                        0.5, hours=entry.hours, voices=entry.usable_voices,
                    )  # fmt: skip

        # 4. Dominance among the buckets that are present.
        present = [b for b in measured if b.hours > 0]
        if report.top_share > DOMINANT_SHARE and len(present) > 1 and report.top_bucket:
            leader = report.stats(report.top_bucket)
            assert leader is not None
            rest = sorted((b for b in present if b is not leader), key=lambda b: -b.hours)
            trailing = ", ".join(f"{_pretty(b.bucket)} {b.hours:.2f} h" for b in rest[:3])
            add(
                report, leader.bucket, "dominant",
                f"{report.label.lower()}: anything that is not {_pretty(leader.bucket)}"
                + (f" -- currently {trailing}" if trailing else ""),
                f"{_pretty(leader.bucket)} holds {report.top_share:.0%} of measured"
                f" {report.label.lower()} hours ({leader.hours:.2f} h, {leader.usable_voices}"
                " voices)",
                report.top_share, hours=leader.hours, voices=leader.usable_voices,
                hours_needed=leader.hours - sum(b.hours for b in rest),
            )  # fmt: skip

        # 5. The benchmark does not span a *condition*. Speech and acoustic categories only: gold
        # holds voices the training data never sees (D76), so a people bucket with no gold is the
        # design, and a topic or show without gold is not a condition a WER is split by.
        if ASR in report.goals and report.group in ("speech", "acoustics"):
            for entry in measured:
                if entry.hours >= min_stratum_hours and entry.gold_hours <= 0:
                    add(
                        report, entry.bucket, "no_gold",
                        f"{report.label.lower()}: gold clips at {_pretty(entry.bucket)}",
                        f"{entry.hours:.2f} h in the corpus and none of it in gold; a WER on this"
                        " condition cannot be measured",
                        0.7, goal=ASR, hours=entry.hours, voices=entry.usable_voices,
                    )  # fmt: skip

        # 6. A paper bucket resting on screened labels (docs/sociolinguistics.md, validity).
        if PAPER in report.goals:
            for entry in measured:
                if entry.hours >= min_stratum_hours and entry.verified_hours < VERIFIED_FLOOR_HOURS:
                    add(
                        report, entry.bucket, "unverified",
                        f"{report.label.lower()}: verify a sample of {_pretty(entry.bucket)}",
                        f"{entry.verified_hours:.2f} h verified of {entry.hours:.2f} h; the rest is"
                        " screened, so its script choice is the seed's convention",
                        1.0 - entry.verified_hours / VERIFIED_FLOOR_HOURS,
                        goal=PAPER, hours=entry.hours, voices=entry.usable_voices,
                        hours_needed=VERIFIED_FLOOR_HOURS - entry.verified_hours,
                    )  # fmt: skip

        # 7. Too much unmeasured to trust any gap above.
        total = report.measured_hours + report.unknown_hours
        share = report.unknown_hours / total if total else 0.0
        if share > UNMEASURED_SHARE and report.key in UNMEASURED_FIX:
            add(
                report, None, "unmeasured",
                f"{report.label.lower()}: {UNMEASURED_FIX[report.key]}",
                f"{report.unknown_hours:.2f} h ({share:.0%}) sit in"
                f" {', '.join(report.unknown)}; every gap above is measured on the rest",
                share, hours=report.unknown_hours, hours_needed=report.unknown_hours,
            )  # fmt: skip

    out.sort(key=lambda r: (-r.priority, r.category, r.target))
    return out


def _voice_rules(
    report: CategoryReport,
    summary: Mapping[str, Any],
    add,
    *,
    min_stratum_voices: int,
) -> None:
    """Recurrence: the accommodation design, and one voice being most of the corpus."""
    hosts = int(summary.get("recurring_hosts", 0))
    if hosts < HOSTS_WANTED:
        add(
            report, None, "recurrence",
            f"voices: more episodes per recurring host, and a{' third' if hosts == 2 else 'nother'}"
            f" host with {RECURRING_HOST_EPISODES}+ host-guest episodes",
            f"{hosts} host voice(s) with {RECURRING_HOST_EPISODES}+ episodes, {HOSTS_WANTED}"
            " wanted for a within-host accommodation design",
            (HOSTS_WANTED - hosts) / HOSTS_WANTED, goal=PAPER,
            voices=hosts, voices_needed=HOSTS_WANTED - hosts,
        )  # fmt: skip
    if int(summary.get("guests_on_two_shows", 0)) == 0 and summary.get("voices", 0):
        add(
            report, None, "recurrence",
            "voices: the same guest on more than one show",
            f"0 guest voices appear with two different hosts; {summary.get('recurring', 0)} of"
            f" {summary.get('voices', 0)} voices recur at all",
            1.0, goal=PAPER, voices_needed=min_stratum_voices,
        )  # fmt: skip
    share = float(summary.get("top_voice_share", 0.0))
    if share > VOICE_DOMINANT_SHARE and summary.get("top_voice"):
        add(
            report, summary["top_voice"], "dominant",
            f"voices: anyone who is not {summary['top_voice']}",
            f"{summary['top_voice']} holds {share:.0%} of all linked talk time; a model trained"
            " on the corpus learns that person",
            share, goal=ASR,
        )  # fmt: skip
    unresolved = int(summary.get("voices", 0)) - int(summary.get("gender_resolved", 0))
    if summary.get("voices", 0) and unresolved / summary["voices"] > UNMEASURED_SHARE:
        add(
            report, None, "unmeasured",
            "voices: link declared speaker rows to voices for the episodes that do not force it",
            f"{unresolved} of {summary['voices']} voices have no resolved gender;"
            f" {summary.get('conflicts', 0)} conflict between episodes",
            unresolved / summary["voices"], goal=PAPER, voices=summary.get("gender_resolved", 0),
        )  # fmt: skip


__all__ = ["Recommendation", "recommend"]
