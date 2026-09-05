"""Progress display for the dashboard: levels, streaks, daily goal and achievements (D63).

None of this changes what is exported, what is queued, or what any number in the corpus means. It
exists because the corpus is built by one person over many months, and the honest measure of that
work -- a backlog -- only ever counts down. Something that counts up is easier to keep showing up
for.

Everything here is derived, on read, from ``segment_labels``, ``annotation_events`` and
``episodes``. Nothing is stored, so there is no state to migrate, nothing to keep in sync, and no
way for a bug in here to corrupt anything that matters. Reset the database and the numbers recompute
themselves.

Two deliberate choices about what gets rewarded:

* **Levels are measured in audio, not in clicks.** Screening a thousand two-second clips should not
  outrank verifying an hour of hard ones, and a scoreboard that rewards clip count would quietly
  push toward the first.
* **A verified label is worth more than a screened one.** Not because screening is illegitimate --
  it is a designed part of the train pot -- but because the scoreboard should not pull in the
  opposite direction from the corpus's own quality claim.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import AnnotationEvent, Episode, Segment
from app.services.stats import latest_labels_subquery

#: How much one labeled second of audio is worth, by the attention it got. A verified second is
#: worth double a screened one: both are real work, and only one is a claim that someone listened.
TIER_WEIGHT: dict[str, float] = {"verified": 1.0, "screened": 0.5}

#: Days of history the activity strip shows. Twelve weeks is a quarter -- long enough to see a
#: habit form or lapse, short enough to stay legible at one row per week.
ACTIVITY_DAYS = 84


@dataclass(frozen=True)
class Achievement:
    """One milestone, unlocked or in progress."""

    id: str
    name: str
    description: str
    #: Progress toward :attr:`target`, in the same unit. Clamped at the target once unlocked.
    progress: float
    target: float
    unlocked: bool
    #: Free-form group, used only to lay the grid out: ``volume``, ``quality``, ``habit`` or
    #: ``corpus``.
    group: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GamifyReport:
    """The dashboard's progress panel."""

    level: int = 1
    level_minutes: float = 0.0
    minutes_into_level: float = 0.0
    minutes_per_level: float = 30.0
    #: 0-1 through the current level, for the ring.
    level_fraction: float = 0.0
    weighted_minutes: float = 0.0
    verified_minutes: float = 0.0
    screened_minutes: float = 0.0
    current_streak_days: int = 0
    longest_streak_days: int = 0
    #: True when today already counts toward the streak, so the UI can say "keep it" vs "extend it".
    streak_active_today: bool = False
    today_segments: int = 0
    daily_goal_segments: int = 200
    daily_goal_fraction: float = 0.0
    best_day_segments: int = 0
    best_day: str | None = None
    active_days: int = 0
    #: ``[{"day": "2026-09-06", "segments": 42}]``, oldest first, one entry per day with activity.
    activity: list[dict[str, Any]] = field(default_factory=list)
    achievements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unlocked_count(self) -> int:
        return sum(1 for a in self.achievements if a["unlocked"])


def _daily_counts(session: Session) -> dict[dt.date, int]:
    """Decisions per calendar day, all of history. Skips do not count -- a deferral is not work.

    One query rather than one per window: the streak needs every day there has ever been, the
    activity strip needs the last twelve weeks, and the personal best needs the maximum. Slicing a
    single dict is cheaper than three round trips and cannot let the three disagree.

    ``date_trunc`` is built once and reused in both the projection and the GROUP BY. Constructing
    it twice renders two different bind parameters, and Postgres then refuses the grouping because
    it cannot see that the two expressions are the same one.
    """
    day = sa.func.date_trunc("day", AnnotationEvent.created_at)
    rows = session.execute(
        sa.select(day.label("day"), sa.func.count())
        .where(AnnotationEvent.action != "skip")
        .group_by(day)
        .order_by(day)
    )
    return {(row[0].date() if isinstance(row[0], dt.datetime) else row[0]): row[1] for row in rows}


def _streaks(days: set[dt.date], *, today: dt.date) -> tuple[int, int, bool]:
    """Current streak, longest streak, and whether today is already counted.

    The current streak survives a day that has not finished yet: at 9am with nothing done, a streak
    running through yesterday is still alive, and saying otherwise would punish the morning.
    """
    if not days:
        return 0, 0, False

    longest = run = 0
    previous: dt.date | None = None
    for day in sorted(days):
        run = run + 1 if previous is not None and (day - previous).days == 1 else 1
        longest = max(longest, run)
        previous = day

    active_today = today in days
    anchor = today if active_today else today - dt.timedelta(days=1)
    if anchor not in days:
        return 0, longest, active_today
    current = 0
    cursor = anchor
    while cursor in days:
        current += 1
        cursor -= dt.timedelta(days=1)
    return current, longest, active_today


def _achievement(
    id: str, name: str, description: str, progress: float, target: float, group: str
) -> dict[str, Any]:
    return Achievement(
        id=id,
        name=name,
        description=description,
        progress=round(min(progress, target), 3),
        target=target,
        unlocked=progress >= target,
        group=group,
    ).as_dict()


def collect_gamify(
    session: Session,
    *,
    settings: Settings | None = None,
    pot_hours: dict[str, float] | None = None,
    gold_coverage_complete: bool = False,
    today: dt.date | None = None,
) -> GamifyReport:
    """Compute the progress panel.

    Args:
        session: Open session.
        settings: Configuration override.
        pot_hours: ``{"gold": h, "train": h, "val": h}`` from the pot report, so the pot-target
            achievements agree with the pot panel above them rather than recomputing and drifting.
        gold_coverage_complete: Whether the gold pot covers every stratification value the corpus
            has. Passed in for the same reason.
        today: Override for "today", for tests. Defaults to the current UTC date.

    Returns:
        A :class:`GamifyReport`.
    """
    settings = settings or get_settings()
    today = today or dt.datetime.now(dt.UTC).date()
    pot_hours = pot_hours or {}

    current = latest_labels_subquery()

    # Labeled audio, split by how much attention it got. Durations come from the segment, so this
    # is corpus time cleared rather than time spent -- an important difference: an hour of clips
    # screened in ten minutes is still an hour of corpus.
    tier_seconds = dict(
        session.execute(
            sa.select(
                current.c.verification_tier,
                sa.func.coalesce(sa.func.sum(Segment.duration_seconds), 0.0),
            )
            .select_from(current)
            .join(Segment, Segment.id == current.c.segment_id)
            .group_by(current.c.verification_tier)
        ).all()
    )
    verified_seconds = float(tier_seconds.get("verified") or 0.0)
    screened_seconds = float(tier_seconds.get("screened") or 0.0)
    weighted_seconds = sum(
        float(seconds) * TIER_WEIGHT.get(tier, 1.0) for tier, seconds in tier_seconds.items()
    )
    weighted_minutes = weighted_seconds / 60

    per_level = settings.gamify.minutes_per_level
    level = int(weighted_minutes // per_level) + 1
    minutes_into_level = weighted_minutes - (level - 1) * per_level

    day_totals = _daily_counts(session)
    current_streak, longest_streak, active_today = _streaks(set(day_totals), today=today)

    best_day, best_count = None, 0
    for day, count in sorted(day_totals.items()):
        if count > best_count:
            best_count = count
            best_day = day.isoformat()

    window_start = today - dt.timedelta(days=ACTIVITY_DAYS)
    counts = {day: count for day, count in day_totals.items() if day >= window_start}
    today_segments = day_totals.get(today, 0)
    goal = settings.gamify.daily_goal_segments

    edited = (
        session.scalar(
            sa.select(sa.func.count()).select_from(current).where(current.c.disposition == "edited")
        )
        or 0
    )
    labeled_total = session.scalar(sa.select(sa.func.count()).select_from(current)) or 0
    shows_touched = (
        session.scalar(
            sa.select(sa.func.count(sa.distinct(Episode.show_id)))
            .select_from(current)
            .join(Segment, Segment.id == current.c.segment_id)
            .join(Episode, Episode.id == Segment.episode_id)
            .where(Episode.show_id.is_not(None))
        )
        or 0
    )

    gold_hours = pot_hours.get("gold", 0.0)
    train_hours = pot_hours.get("train", 0.0) + pot_hours.get("val", 0.0)

    achievements = [
        _achievement(
            "first_light",
            "First Light",
            "Label your first segment.",
            labeled_total,
            1,
            "volume",
        ),
        _achievement(
            "first_hour",
            "One Hour Down",
            "Clear an hour of corpus audio.",
            weighted_minutes,
            60,
            "volume",
        ),
        _achievement(
            "ten_hours",
            "Double Digits",
            "Clear ten hours of corpus audio.",
            weighted_minutes,
            600,
            "volume",
        ),
        _achievement(
            "century",
            "Century",
            "Decide 100 segments in one day.",
            best_count,
            100,
            "volume",
        ),
        _achievement(
            "marathon",
            "Marathon",
            "Decide 500 segments in one day.",
            best_count,
            500,
            "volume",
        ),
        _achievement(
            "gold_target",
            "Benchmark Complete",
            f"Fill the gold pot to its {settings.dataset.gold_hours_target:g} h target.",
            gold_hours,
            settings.dataset.gold_hours_target,
            "corpus",
        ),
        _achievement(
            "train_target",
            "Training Set Complete",
            f"Fill the train pot to its {settings.dataset.train_hours_target:g} h target.",
            train_hours,
            settings.dataset.train_hours_target,
            "corpus",
        ),
        _achievement(
            "full_spectrum",
            "Full Spectrum",
            "Gold pot covers every show, gender, age bracket and topic in the corpus.",
            1 if gold_coverage_complete else 0,
            1,
            "corpus",
        ),
        _achievement(
            "corrector",
            "Corrector",
            "Fix the transcript on 100 segments.",
            edited,
            100,
            "quality",
        ),
        _achievement(
            "verified_hour",
            "Heard It Myself",
            "Verify an hour of audio by ear.",
            verified_seconds / 60,
            60,
            "quality",
        ),
        _achievement(
            "many_shows",
            "Range",
            "Label segments from five different shows.",
            shows_touched,
            5,
            "corpus",
        ),
        _achievement(
            "streak_7",
            "Week Straight",
            "Annotate seven days in a row.",
            longest_streak,
            7,
            "habit",
        ),
        _achievement(
            "streak_30",
            "Month Straight",
            "Annotate thirty days in a row.",
            longest_streak,
            30,
            "habit",
        ),
    ]

    return GamifyReport(
        level=level,
        level_minutes=round(weighted_minutes, 2),
        minutes_into_level=round(minutes_into_level, 2),
        minutes_per_level=per_level,
        level_fraction=round(min(minutes_into_level / per_level, 1.0), 4) if per_level else 0.0,
        weighted_minutes=round(weighted_minutes, 2),
        verified_minutes=round(verified_seconds / 60, 2),
        screened_minutes=round(screened_seconds / 60, 2),
        current_streak_days=current_streak,
        longest_streak_days=longest_streak,
        streak_active_today=active_today,
        today_segments=today_segments,
        daily_goal_segments=goal,
        daily_goal_fraction=round(min(today_segments / goal, 1.0), 4) if goal else 0.0,
        best_day_segments=best_count,
        best_day=best_day,
        active_days=len(day_totals),
        activity=[{"day": day.isoformat(), "segments": counts[day]} for day in sorted(counts)],
        achievements=achievements,
    )
