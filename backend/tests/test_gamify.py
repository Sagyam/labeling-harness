"""Tests for the dashboard's progress panel: levels, streaks, milestones (D63).

None of this affects the corpus, so the tests are about the two things that would make the panel
actively misleading: a level that rewards clicking over clearing audio, and a streak that punishes
the morning.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationEvent, AnnotationTask
from app.services.fixtures import build_export_fixture
from app.services.gamify import _streaks, collect_gamify
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


@pytest.fixture
def tasks(db_session: Session, tmp_path: Path, storage, settings: Settings) -> list[AnnotationTask]:
    root = build_export_fixture(
        tmp_path / "export_gam", episode_id="gam_ep001", segments=8, systems=2
    )
    import_manifest(db_session, root, storage=storage, settings=settings)
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    db_session.flush()
    return list(db_session.scalars(sa.select(AnnotationTask).order_by(AnnotationTask.id)))


# --- streaks -----------------------------------------------------------------------------


def test_a_streak_counts_consecutive_days() -> None:
    today = dt.date(2026, 9, 6)
    days = {today, today - dt.timedelta(days=1), today - dt.timedelta(days=2)}
    current, longest, active = _streaks(days, today=today)
    assert (current, longest, active) == (3, 3, True)


def test_a_streak_survives_a_morning_with_nothing_done_yet() -> None:
    """At 9am with no work yet, a streak running through yesterday is still alive."""
    today = dt.date(2026, 9, 6)
    days = {today - dt.timedelta(days=1), today - dt.timedelta(days=2)}
    current, _longest, active = _streaks(days, today=today)
    assert current == 2
    assert active is False


def test_a_streak_breaks_after_a_missed_day() -> None:
    today = dt.date(2026, 9, 6)
    days = {today - dt.timedelta(days=3), today - dt.timedelta(days=4)}
    current, longest, _ = _streaks(days, today=today)
    assert current == 0
    assert longest == 2


def test_the_longest_streak_is_remembered_after_it_breaks() -> None:
    today = dt.date(2026, 9, 20)
    days = {dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(9)}
    current, longest, _ = _streaks(days, today=today)
    assert current == 0
    assert longest == 9


def test_no_history_is_no_streak() -> None:
    assert _streaks(set(), today=dt.date(2026, 9, 6)) == (0, 0, False)


# --- levels ------------------------------------------------------------------------------


def test_an_empty_corpus_starts_at_level_one(db_session: Session, settings: Settings) -> None:
    report = collect_gamify(db_session, settings=settings)
    assert report.level == 1
    assert report.weighted_minutes == 0.0
    assert report.current_streak_days == 0


def test_a_verified_minute_counts_for_more_than_a_screened_one(
    db_session: Session, tasks: list[AnnotationTask], settings: Settings
) -> None:
    """The scoreboard must not pull against the corpus's own quality claim."""
    record_decision(
        db_session,
        tasks[0],
        Decision(disposition="accepted_unchanged", verification_tier="verified"),
        settings=settings,
    )
    db_session.flush()
    verified_only = collect_gamify(db_session, settings=settings).weighted_minutes

    record_decision(
        db_session,
        tasks[1],
        Decision(disposition="accepted_unchanged", verification_tier="screened"),
        settings=settings,
    )
    db_session.flush()
    both = collect_gamify(db_session, settings=settings)

    assert both.weighted_minutes > verified_only
    assert both.verified_minutes > 0
    assert both.screened_minutes > 0
    # The screened clip contributed half its length, so the weighted total is below the raw one.
    assert both.weighted_minutes < both.verified_minutes + both.screened_minutes


def test_levels_are_measured_in_audio_not_in_clicks(
    db_session: Session, tasks: list[AnnotationTask], settings: Settings
) -> None:
    for task in tasks[:6]:
        record_decision(
            db_session, task, Decision(disposition="accepted_unchanged"), settings=settings
        )
    db_session.flush()

    report = collect_gamify(db_session, settings=settings)
    # Six short fixture clips are nowhere near a level, however many keystrokes they took.
    assert report.level == 1
    assert 0.0 <= report.level_fraction < 1.0
    assert report.minutes_into_level == pytest.approx(report.weighted_minutes, abs=0.01)


# --- daily goal and personal best --------------------------------------------------------


def test_the_daily_goal_tracks_todays_decisions(
    db_session: Session, tasks: list[AnnotationTask], settings: Settings
) -> None:
    for task in tasks[:3]:
        record_decision(
            db_session, task, Decision(disposition="accepted_unchanged"), settings=settings
        )
    db_session.flush()

    report = collect_gamify(db_session, settings=settings)
    assert report.today_segments == 3
    assert report.best_day_segments == 3
    assert report.best_day == dt.datetime.now(dt.UTC).date().isoformat()
    assert 0 < report.daily_goal_fraction <= 1.0


def test_a_skip_is_not_work_finished(
    db_session: Session, tasks: list[AnnotationTask], settings: Settings
) -> None:
    from app.services.labeling import record_skip

    record_skip(db_session, tasks[0], settings=settings)
    db_session.flush()
    assert collect_gamify(db_session, settings=settings).today_segments == 0


def test_the_activity_strip_is_ordered_oldest_first(
    db_session: Session, tasks: list[AnnotationTask], settings: Settings
) -> None:
    record_decision(
        db_session, tasks[0], Decision(disposition="accepted_unchanged"), settings=settings
    )
    db_session.flush()
    event = db_session.scalars(sa.select(AnnotationEvent)).one()
    event.created_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=3)
    db_session.flush()

    activity = collect_gamify(db_session, settings=settings).activity
    assert activity
    assert activity == sorted(activity, key=lambda row: row["day"])


# --- milestones --------------------------------------------------------------------------


def test_every_milestone_has_a_target_and_a_progress(
    db_session: Session, settings: Settings
) -> None:
    report = collect_gamify(db_session, settings=settings)
    assert report.achievements
    for achievement in report.achievements:
        assert achievement["target"] > 0
        assert achievement["progress"] <= achievement["target"]
        assert achievement["group"] in {"volume", "quality", "habit", "corpus"}
        assert achievement["name"]
        assert achievement["description"]


def test_the_first_milestone_unlocks_on_the_first_label(
    db_session: Session, tasks: list[AnnotationTask], settings: Settings
) -> None:
    before = {
        a["id"]: a["unlocked"] for a in collect_gamify(db_session, settings=settings).achievements
    }
    assert before["first_light"] is False

    record_decision(
        db_session, tasks[0], Decision(disposition="accepted_unchanged"), settings=settings
    )
    db_session.flush()

    after = {
        a["id"]: a["unlocked"] for a in collect_gamify(db_session, settings=settings).achievements
    }
    assert after["first_light"] is True


def test_the_pot_milestones_follow_the_pot_panel(db_session: Session, settings: Settings) -> None:
    """Passed in rather than recomputed, so the badge cannot disagree with the bar above it."""
    report = collect_gamify(
        db_session,
        settings=settings,
        pot_hours={"gold": settings.dataset.gold_hours_target, "train": 0.0, "val": 0.0},
        gold_coverage_complete=True,
    )
    unlocked = {a["id"] for a in report.achievements if a["unlocked"]}
    assert "gold_target" in unlocked
    assert "full_spectrum" in unlocked


def test_progress_is_clamped_at_the_target(db_session: Session, settings: Settings) -> None:
    report = collect_gamify(
        db_session,
        settings=settings,
        pot_hours={"gold": settings.dataset.gold_hours_target * 10, "train": 0.0, "val": 0.0},
    )
    gold = next(a for a in report.achievements if a["id"] == "gold_target")
    assert gold["progress"] == gold["target"]


def test_the_unlocked_count_matches_the_milestones(db_session: Session, settings: Settings) -> None:
    report = collect_gamify(db_session, settings=settings)
    assert report.unlocked_count == sum(1 for a in report.achievements if a["unlocked"])
