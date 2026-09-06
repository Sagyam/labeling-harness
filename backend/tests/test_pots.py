"""Tests for pot assignment (D63).

The pot is what decides whether an episode is benchmark or training material, so these tests are
mostly about the three rules that make the arrangement defensible: whole episodes only, gold is
one-directional, and the gold pot is filled to a *duration* with coverage taken into account.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import Episode, Segment, SegmentLabel
from app.services.labeling import get_or_create_label_version
from app.services.pots import PotError, assign_pots, episode_coverage, pot_status

pytestmark = pytest.mark.db


@pytest.fixture
def settings() -> Settings:
    return load_settings()


def make_episode(
    session: Session,
    external_id: str,
    *,
    minutes: float,
    show_id: str = "show-a",
    topic: str | None = None,
    gender: str | None = None,
    age_bracket: str | None = None,
    segment_seconds: float = 60.0,
) -> Episode:
    """An episode with enough real segments to carry ``minutes`` of audio."""
    speakers = {}
    if gender or age_bracket:
        speaker: dict[str, str] = {"role": "host"}
        if gender:
            speaker["gender"] = gender
        if age_bracket:
            speaker["age_bracket"] = age_bracket
        speakers = {"spk:0": speaker}

    metadata: dict[str, object] = {}
    if topic:
        metadata["topic"] = topic
    if speakers:
        metadata["speakers"] = speakers

    episode = Episode(
        external_id=external_id,
        show_id=show_id,
        title=external_id,
        duration_seconds=minutes * 60,
        split="unassigned",
        pot="unassigned",
        metadata_jsonb=metadata or None,
    )
    session.add(episode)
    session.flush()

    total = minutes * 60
    cursor = 0.0
    index = 0
    while cursor < total:
        length = min(segment_seconds, total - cursor)
        session.add(
            Segment(
                episode_id=episode.id,
                external_id=f"{external_id}_seg{index:04d}",
                start_time=cursor,
                end_time=cursor + length,
                duration_seconds=length,
                clip_object_key=f"clips/{external_id}_seg{index:04d}.flac",
                clip_checksum=f"sha256:{external_id}_{index:04d}",
                pipeline_status="imported",
            )
        )
        cursor += length
        index += 1
    session.flush()
    return episode


# --- duration targeting ------------------------------------------------------------------


def test_gold_is_filled_to_an_hours_target_not_a_ratio(
    db_session: Session, settings: Settings
) -> None:
    for index in range(10):
        make_episode(db_session, f"ep{index:03d}", minutes=60, show_id=f"show-{index}")

    report = assign_pots(db_session, settings=settings, gold_hours_target=3.0)

    assert report.gold_target_met
    assert report.gold_hours == pytest.approx(3.0, abs=1.0)
    assert report.gold_episodes == 3
    assert report.train_episodes + report.val_episodes == 7


def test_a_larger_target_takes_more_episodes(db_session: Session, settings: Settings) -> None:
    for index in range(10):
        make_episode(db_session, f"ep{index:03d}", minutes=30, show_id=f"show-{index}")

    small = assign_pots(db_session, settings=settings, gold_hours_target=1.0).gold_episodes
    db_session.rollback()

    for index in range(10):
        make_episode(db_session, f"ep{index:03d}", minutes=30, show_id=f"show-{index}")
    large = assign_pots(db_session, settings=settings, gold_hours_target=4.0).gold_episodes

    assert large > small


def test_raising_the_target_after_placement_needs_new_episodes_or_the_opt_in(
    db_session: Session, settings: Settings
) -> None:
    """Rule 3, from the other side.

    Once everything is placed, raising the gold target cannot grow gold on its own: the only
    episodes left are in the train pot, and taking one back is exactly the move that would turn a
    benchmark into a memorization test. The assigner reports the shortfall rather than quietly
    reaching for training material.
    """
    for index in range(4):
        make_episode(db_session, f"ep{index:03d}", minutes=30, show_id=f"show-{index}")
    assign_pots(db_session, settings=settings, gold_hours_target=0.5)
    db_session.flush()

    stalled = assign_pots(db_session, settings=settings, gold_hours_target=1.5)
    assert not stalled.gold_target_met
    assert stalled.gold_episodes == 1

    # A newly ingested episode is fair game, because nothing has trained on it.
    make_episode(db_session, "ep_new", minutes=60, show_id="show-new")
    grown = assign_pots(db_session, settings=settings, gold_hours_target=1.5)
    assert grown.gold_episodes == 2
    assert grown.gold_target_met


def test_a_small_corpus_is_not_swallowed_whole_into_gold(
    db_session: Session, settings: Settings
) -> None:
    """A corpus smaller than the target would otherwise all be locked into the benchmark.

    Rule 3 makes that irreversible, so it would leave nothing to train on and freeze the
    benchmark's coverage at the point when the corpus was too small to know what it should span.
    """
    for index in range(10):
        make_episode(db_session, f"ep{index:03d}", minutes=30, show_id=f"show-{index}")

    report = assign_pots(
        db_session, settings=settings, gold_hours_target=50.0, gold_max_corpus_fraction=0.3
    )

    assert not report.gold_target_met
    assert report.gold_capped_by_corpus_size
    assert report.gold_effective_target_hours == pytest.approx(1.5, abs=0.01)
    assert report.gold_hours <= 2.0
    assert report.train_episodes + report.val_episodes > 0


def test_gold_grows_as_the_corpus_grows(db_session: Session, settings: Settings) -> None:
    for index in range(4):
        make_episode(db_session, f"ep{index:03d}", minutes=60, show_id=f"show-{index}")
    first = assign_pots(
        db_session, settings=settings, gold_hours_target=10.0, gold_max_corpus_fraction=0.5
    )
    db_session.flush()

    for index in range(4, 12):
        make_episode(db_session, f"ep{index:03d}", minutes=60, show_id=f"show-{index}")
    second = assign_pots(
        db_session, settings=settings, gold_hours_target=10.0, gold_max_corpus_fraction=0.5
    )

    assert second.gold_effective_target_hours > first.gold_effective_target_hours
    assert second.gold_hours > first.gold_hours


def test_the_cap_never_forces_an_episode_out_of_gold(
    db_session: Session, settings: Settings
) -> None:
    """Rule 3 outranks the cap: gold cannot give an episode back, whatever the ceiling says."""
    for index in range(4):
        make_episode(db_session, f"ep{index:03d}", minutes=60, show_id=f"show-{index}")
    assign_pots(db_session, settings=settings, gold_hours_target=3.0, gold_max_corpus_fraction=1.0)
    db_session.flush()
    held = set(db_session.scalars(sa.select(Episode.external_id).where(Episode.pot == "gold")))
    assert len(held) >= 3

    assign_pots(db_session, settings=settings, gold_hours_target=3.0, gold_max_corpus_fraction=0.1)
    db_session.flush()
    assert (
        set(db_session.scalars(sa.select(Episode.external_id).where(Episode.pot == "gold"))) == held
    )


def test_a_non_positive_target_is_refused(db_session: Session, settings: Settings) -> None:
    with pytest.raises(PotError, match="must be positive"):
        assign_pots(db_session, settings=settings, gold_hours_target=0.0)


# --- whole episodes, never clips ---------------------------------------------------------


def test_every_segment_of_an_episode_lands_in_the_same_split(
    db_session: Session, settings: Settings
) -> None:
    """Rule 2. VAD cuts are contiguous, so a clip-level split leaks adjacent audio."""
    for index in range(6):
        make_episode(db_session, f"ep{index:03d}", minutes=30, show_id=f"show-{index}")
    assign_pots(db_session, settings=settings, gold_hours_target=1.0)
    db_session.flush()

    rows = db_session.execute(
        sa.select(Episode.external_id, sa.func.count(sa.distinct(Episode.split)))
        .join(Segment, Segment.episode_id == Episode.id)
        .group_by(Episode.external_id)
    ).all()
    assert rows
    assert all(distinct_splits == 1 for _, distinct_splits in rows)


def test_the_database_refuses_a_gold_episode_that_is_not_the_test_split(
    db_session: Session, settings: Settings
) -> None:
    """The pot/split agreement is a CHECK, not just assigner discipline."""
    episode = make_episode(db_session, "ep000", minutes=10)
    episode.pot = "gold"
    episode.split = "train"
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


# --- gold is one-directional -------------------------------------------------------------


def test_an_episode_never_leaves_the_gold_pot(db_session: Session, settings: Settings) -> None:
    """Rule 3. A benchmark recording that turns up in training invalidates every number."""
    for index in range(8):
        make_episode(db_session, f"ep{index:03d}", minutes=60, show_id=f"show-{index}")

    assign_pots(db_session, settings=settings, gold_hours_target=4.0)
    db_session.flush()
    gold_before = set(
        db_session.scalars(sa.select(Episode.external_id).where(Episode.pot == "gold"))
    )
    assert gold_before

    # Re-run with a much smaller target: gold is already over it and must not shrink.
    assign_pots(db_session, settings=settings, gold_hours_target=1.0)
    db_session.flush()
    gold_after = set(
        db_session.scalars(sa.select(Episode.external_id).where(Episode.pot == "gold"))
    )
    assert gold_after == gold_before


def test_a_train_episode_is_not_promoted_into_gold_by_default(
    db_session: Session, settings: Settings
) -> None:
    make_episode(db_session, "ep000", minutes=60, show_id="show-0")
    make_episode(db_session, "ep001", minutes=60, show_id="show-1")
    assign_pots(db_session, settings=settings, gold_hours_target=1.0)
    db_session.flush()

    train_ids = set(
        db_session.scalars(sa.select(Episode.external_id).where(Episode.pot == "train"))
    )
    assert train_ids

    # Ask for far more gold than exists. Without the opt-in, the train episodes stay put even
    # though taking them is the only way to reach the target.
    report = assign_pots(db_session, settings=settings, gold_hours_target=10.0)
    db_session.flush()
    assert not report.gold_target_met
    still_train = set(
        db_session.scalars(sa.select(Episode.external_id).where(Episode.pot == "train"))
    )
    assert train_ids <= still_train


def test_promotion_from_train_happens_only_when_explicitly_allowed(
    db_session: Session, settings: Settings
) -> None:
    make_episode(db_session, "ep000", minutes=60, show_id="show-0")
    make_episode(db_session, "ep001", minutes=60, show_id="show-1")
    assign_pots(db_session, settings=settings, gold_hours_target=1.0)
    db_session.flush()

    report = assign_pots(
        db_session,
        settings=settings,
        gold_hours_target=2.0,
        gold_max_corpus_fraction=1.0,
        allow_promote_from_train=True,
    )
    assert report.gold_episodes == 2
    assert report.gold_target_met


# --- train/val may be redrawn ------------------------------------------------------------


def test_train_and_val_both_get_episodes(db_session: Session, settings: Settings) -> None:
    for index in range(40):
        make_episode(db_session, f"ep{index:03d}", minutes=10, show_id=f"show-{index % 4}")
    report = assign_pots(db_session, settings=settings, gold_hours_target=1.0)
    assert report.train_episodes > 0
    assert report.val_episodes > 0


def test_the_val_line_is_reproducible_for_a_given_seed(
    db_session: Session, settings: Settings
) -> None:
    for index in range(20):
        make_episode(db_session, f"ep{index:03d}", minutes=10, show_id=f"show-{index % 3}")
    assign_pots(db_session, settings=settings, gold_hours_target=0.5)
    db_session.flush()
    first = dict(db_session.execute(sa.select(Episode.external_id, Episode.split)).all())

    assign_pots(db_session, settings=settings, gold_hours_target=0.5)
    db_session.flush()
    second = dict(db_session.execute(sa.select(Episode.external_id, Episode.split)).all())
    assert first == second


# --- coverage ----------------------------------------------------------------------------


def test_gold_prefers_breadth_over_taking_one_show(db_session: Session, settings: Settings) -> None:
    """A five-hour benchmark drawn from one show measures that show."""
    for index in range(6):
        make_episode(db_session, f"a{index:03d}", minutes=60, show_id="show-a", topic="politics")
    make_episode(db_session, "b000", minutes=60, show_id="show-b", topic="technology")
    make_episode(db_session, "c000", minutes=60, show_id="show-c", topic="culture")

    assign_pots(db_session, settings=settings, gold_hours_target=3.0)
    db_session.flush()

    gold_shows = set(db_session.scalars(sa.select(Episode.show_id).where(Episode.pot == "gold")))
    assert gold_shows == {"show-a", "show-b", "show-c"}


def test_coverage_gaps_are_reported(db_session: Session, settings: Settings) -> None:
    make_episode(db_session, "a000", minutes=120, show_id="show-a", topic="politics")
    make_episode(db_session, "b000", minutes=10, show_id="show-b", topic="technology")

    assign_pots(db_session, settings=settings, gold_hours_target=1.0)
    db_session.flush()
    status = pot_status(db_session, settings=settings)

    if status.gold_coverage_gaps:
        assert not status.coverage_complete
        for values in status.gold_coverage_gaps.values():
            assert values
    else:
        assert status.coverage_complete


def test_episode_coverage_reads_speakers_and_topic(db_session: Session) -> None:
    episode = make_episode(
        db_session,
        "ep000",
        minutes=1,
        show_id="show-z",
        topic="politics",
        gender="female",
        age_bracket="40_59",
    )
    coverage = episode_coverage(episode, keys=["show_id", "gender", "age_bracket", "topic"])
    assert coverage["show_id"] == frozenset({"show-z"})
    assert coverage["gender"] == frozenset({"female"})
    assert coverage["age_bracket"] == frozenset({"40_59"})
    assert coverage["topic"] == frozenset({"politics"})


def test_missing_metadata_contributes_nothing_rather_than_a_guess(
    db_session: Session,
) -> None:
    episode = make_episode(db_session, "ep000", minutes=1, show_id="show-z")
    coverage = episode_coverage(episode, keys=["show_id", "gender", "topic"])
    assert coverage["show_id"] == frozenset({"show-z"})
    assert coverage["gender"] == frozenset()
    assert coverage["topic"] == frozenset()


# --- an episode with no audio yet --------------------------------------------------------


def test_an_episode_with_no_segments_stays_unassigned(
    db_session: Session, settings: Settings
) -> None:
    """It has no duration to budget with, so placing it would be guessing."""
    db_session.add(
        Episode(external_id="empty", show_id="show-a", split="unassigned", pot="unassigned")
    )
    db_session.flush()
    make_episode(db_session, "ep000", minutes=30)

    report = assign_pots(db_session, settings=settings, gold_hours_target=0.25)
    db_session.flush()

    assert report.unassigned_episodes == 1
    episode = db_session.scalar(sa.select(Episode).where(Episode.external_id == "empty"))
    assert episode.pot == "unassigned"


# --- dry run and reporting ---------------------------------------------------------------


def test_a_dry_run_writes_nothing(db_session: Session, settings: Settings) -> None:
    make_episode(db_session, "ep000", minutes=30)
    make_episode(db_session, "ep001", minutes=30)

    report = assign_pots(db_session, settings=settings, gold_hours_target=0.5, dry_run=True)
    db_session.flush()

    assert report.dry_run
    assert report.changes
    assert all(pot == "unassigned" for pot in db_session.scalars(sa.select(Episode.pot)))


def test_the_report_renders(db_session: Session, settings: Settings) -> None:
    make_episode(db_session, "ep000", minutes=30, show_id="show-a", topic="politics")
    make_episode(db_session, "ep001", minutes=30, show_id="show-b", topic="technology")
    rendered = assign_pots(db_session, settings=settings, gold_hours_target=0.5).render()
    assert "pot assignment" in rendered
    assert "gold" in rendered


def test_pot_status_reports_labeled_hours_separately_from_ingested_hours(
    db_session: Session, settings: Settings
) -> None:
    make_episode(db_session, "ep000", minutes=30)
    assign_pots(db_session, settings=settings, gold_hours_target=0.1)
    db_session.flush()

    status = pot_status(db_session, settings=settings)
    total = sum(bucket["hours"] for bucket in status.buckets.values())
    labeled = sum(bucket["labeled_hours"] for bucket in status.buckets.values())
    assert total == pytest.approx(0.5, abs=0.01)
    assert labeled == 0.0


def test_pot_status_does_not_assign_anything(db_session: Session, settings: Settings) -> None:
    make_episode(db_session, "ep000", minutes=30)
    pot_status(db_session, settings=settings)
    db_session.flush()
    assert db_session.scalar(sa.select(Episode.pot)) == "unassigned"


# --- an episode that has already been screened can never become the benchmark (D65) ------


def screen_one_segment(session: Session, episode: Episode, settings: Settings) -> None:
    """Write one screened label against this episode, as bulk triage does."""
    segment = session.scalars(
        sa.select(Segment).where(Segment.episode_id == episode.id).limit(1)
    ).one()
    version = get_or_create_label_version(session, settings.labels.default_label_version, settings)
    session.add(
        SegmentLabel(
            segment_id=segment.id,
            label_version_id=version.id,
            final_text="waved through",
            disposition="accepted_unchanged",
            verification_tier="screened",
            annotator="owner",
        )
    )
    session.flush()


def a_corpus_with_room_for_gold(session: Session) -> dict[str, Episode]:
    """Six shows of an hour each: 6 h ingested, so the 0.35 cap leaves ~2.1 h of gold to fill."""
    return {
        name: make_episode(session, name, minutes=60, show_id=f"show-{name[-1]}", topic=name)
        for name in ("ep_a", "ep_b", "ep_c", "ep_d", "ep_e", "ep_f")
    }


def test_a_screened_episode_is_never_selected_for_gold(
    db_session: Session, settings: Settings
) -> None:
    """The hole this closes: screening is legal while unassigned, and the assigner ran after.

    A gold pot whose rows were waved through is not a benchmark, and `record_decision` cannot
    catch it -- the decision was already legal when it was made.
    """
    episodes = a_corpus_with_room_for_gold(db_session)
    for episode in episodes.values():
        screen_one_segment(db_session, episode, settings)

    assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()

    db_session.expire_all()
    pots = {name: db_session.get(Episode, ep.id).pot for name, ep in episodes.items()}
    assert "gold" not in pots.values(), pots


def test_a_clean_episode_is_preferred_over_a_screened_one(
    db_session: Session, settings: Settings
) -> None:
    """The guard must not empty the benchmark; it only holds back the screened episodes."""
    episodes = a_corpus_with_room_for_gold(db_session)
    for name in ("ep_a", "ep_b", "ep_c", "ep_d"):
        screen_one_segment(db_session, episodes[name], settings)

    assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()

    db_session.expire_all()
    gold = {name for name, ep in episodes.items() if db_session.get(Episode, ep.id).pot == "gold"}
    assert gold, "the guard emptied the gold pot instead of routing around it"
    assert gold <= {"ep_e", "ep_f"}, gold


def test_a_verified_episode_is_still_eligible_for_gold(
    db_session: Session, settings: Settings
) -> None:
    """Verified labels are what gold is made of; only `screened` disqualifies."""
    episodes = a_corpus_with_room_for_gold(db_session)
    version = get_or_create_label_version(
        db_session, settings.labels.default_label_version, settings
    )
    for episode in episodes.values():
        segment = db_session.scalars(
            sa.select(Segment).where(Segment.episode_id == episode.id).limit(1)
        ).one()
        db_session.add(
            SegmentLabel(
                segment_id=segment.id,
                label_version_id=version.id,
                final_text="listened to",
                disposition="accepted_unchanged",
                verification_tier="verified",
                annotator="owner",
            )
        )
    db_session.flush()

    assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()

    db_session.expire_all()
    gold = [name for name, ep in episodes.items() if db_session.get(Episode, ep.id).pot == "gold"]
    assert gold, "verified labels wrongly disqualified an episode from the benchmark"


def test_the_report_names_the_episodes_it_held_back(
    db_session: Session, settings: Settings
) -> None:
    """Silently declining to promote an episode is the kind of thing that needs saying."""
    episodes = a_corpus_with_room_for_gold(db_session)
    screen_one_segment(db_session, episodes["ep_a"], settings)

    report = assign_pots(db_session, settings=settings, gold_hours_target=5.0, dry_run=True)

    assert "ep_a" in report.gold_ineligible
    assert "ep_a" in report.render()


def test_a_gold_episode_holding_screened_labels_is_demoted(
    db_session: Session, settings: Settings
) -> None:
    """Rule 3's one exception: an episode that was never validly gold in the first place.

    The window is real and this repository walked into it -- an episode was screened while
    unassigned, then the assigner placed it in gold on the hours target. Rule 3 forbids leaving
    gold to stop a trained-on episode becoming the benchmark; it was never meant to pin an
    episode there that the benchmark's own definition excludes.
    """
    episodes = a_corpus_with_room_for_gold(db_session)
    assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()
    db_session.expire_all()

    gold = [name for name, ep in episodes.items() if db_session.get(Episode, ep.id).pot == "gold"]
    assert gold, "fixture did not produce a gold pot to demote from"
    victim = episodes[gold[0]]
    screen_one_segment(db_session, victim, settings)

    report = assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()
    db_session.expire_all()

    assert db_session.get(Episode, victim.id).pot == "train"
    assert db_session.get(Episode, victim.id).split in {"train", "val"}
    assert victim.external_id in report.gold_demoted
    assert victim.external_id in report.render()


def test_a_clean_gold_episode_is_never_demoted(db_session: Session, settings: Settings) -> None:
    """The demotion is narrow: only screened labels move an episode, nothing else."""
    episodes = a_corpus_with_room_for_gold(db_session)
    assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()
    db_session.expire_all()
    before = {name: db_session.get(Episode, ep.id).pot for name, ep in episodes.items()}

    report = assign_pots(db_session, settings=settings, gold_hours_target=5.0)
    db_session.flush()
    db_session.expire_all()

    after = {name: db_session.get(Episode, ep.id).pot for name, ep in episodes.items()}
    assert after == before
    assert report.gold_demoted == []
