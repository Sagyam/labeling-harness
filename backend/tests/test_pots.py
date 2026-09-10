"""Tests for the two pots, chosen one clip at a time by hand (D71).

Gold is no longer assigned to whole episodes by an algorithm. The owner puts a clip in gold with
one click, and can take it out again. What survives from D63 is the part that is a property of a
label rather than of a selection rule: gold only ever holds clips a human listened to.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AuditLog, Episode, Segment, SegmentLabel
from app.services.labeling import get_or_create_label_version
from app.services.pots import (
    GOLD_SPLIT,
    PotError,
    draw_split,
    effective_split,
    episode_coverage,
    pot_status,
    set_segment_pot,
)

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
    split: str = "train",
    topic: str | None = None,
    gender: str | None = None,
    age_bracket: str | None = None,
    segment_seconds: float = 60.0,
) -> Episode:
    """An episode with enough real segments to carry ``minutes`` of audio."""
    metadata: dict[str, object] = {}
    if topic:
        metadata["topic"] = topic
    speaker = {"role": "host"}
    if gender:
        speaker["gender"] = gender
    if age_bracket:
        speaker["age_bracket"] = age_bracket
    if len(speaker) > 1:
        metadata["speakers"] = {"spk:0": speaker}

    episode = Episode(
        external_id=external_id,
        show_id=show_id,
        title=external_id,
        duration_seconds=minutes * 60,
        split=split,
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


def segments_of(session: Session, episode: Episode) -> list[Segment]:
    return list(
        session.scalars(
            sa.select(Segment).where(Segment.episode_id == episode.id).order_by(Segment.id)
        )
    )


def label(session: Session, segment: Segment, settings: Settings, *, tier: str) -> None:
    version = get_or_create_label_version(session, settings.labels.default_label_version, settings)
    session.add(
        SegmentLabel(
            segment_id=segment.id,
            label_version_id=version.id,
            final_text="text",
            disposition="accepted_unchanged",
            verification_tier=tier,
            annotator="owner",
        )
    )
    session.flush()


# --- one clip at a time ------------------------------------------------------------------


def test_a_new_segment_starts_in_the_train_pot(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=3)
    assert {s.pot for s in segments_of(db_session, episode)} == {"train"}


def test_one_clip_moves_to_gold_and_its_neighbours_stay(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=3)
    first, second, third = segments_of(db_session, episode)

    move = set_segment_pot(db_session, second, "gold", actor="owner")

    assert move.changed
    assert (move.from_pot, move.to_pot) == ("train", "gold")
    assert [s.pot for s in (first, second, third)] == ["train", "gold", "train"]


def test_a_move_writes_an_audit_row(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)

    set_segment_pot(db_session, segment, "gold", actor="owner", reason="clean monologue")
    db_session.flush()

    row = db_session.scalars(sa.select(AuditLog).where(AuditLog.entity_type == "segment")).one()
    assert row.action == "pot_changed"
    assert row.entity_id == str(segment.id)
    assert row.old_values_jsonb == {"pot": "train"}
    assert row.new_values_jsonb == {"pot": "gold", "reason": "clean monologue"}


def test_a_clip_can_be_taken_back_out_of_gold(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)
    set_segment_pot(db_session, segment, "gold", actor="owner")

    move = set_segment_pot(db_session, segment, "train", actor="owner")

    assert move.changed
    assert segment.pot == "train"


def test_setting_the_pot_it_already_has_is_a_no_op(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)

    move = set_segment_pot(db_session, segment, "train", actor="owner")
    db_session.flush()

    assert not move.changed
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AuditLog)) == 0


def test_an_unknown_pot_is_refused(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)
    with pytest.raises(PotError):
        set_segment_pot(db_session, segment, "benchmark", actor="owner")


def test_the_database_refuses_an_unknown_pot(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)
    segment.pot = "benchmark"
    with pytest.raises(IntegrityError):
        db_session.flush()


# --- gold holds only what was listened to ----------------------------------------------------


def test_a_screened_clip_cannot_enter_gold(db_session: Session, settings: Settings) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)
    label(db_session, segment, settings, tier="screened")

    with pytest.raises(PotError, match="screened"):
        set_segment_pot(db_session, segment, "gold", actor="owner")
    assert segment.pot == "train"


def test_a_verified_clip_can_enter_gold(db_session: Session, settings: Settings) -> None:
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)
    label(db_session, segment, settings, tier="verified")

    assert set_segment_pot(db_session, segment, "gold", actor="owner").changed


def test_an_unlabelled_clip_can_enter_gold(db_session: Session) -> None:
    # It will have to be listened to when it is labelled: `record_decision` refuses a screened
    # decision on a gold clip.
    episode = make_episode(db_session, "ep000", minutes=1)
    (segment,) = segments_of(db_session, episode)
    assert set_segment_pot(db_session, segment, "gold", actor="owner").changed


# --- splits ----------------------------------------------------------------------------------


def test_a_gold_clip_is_the_test_split_whatever_its_episode_is() -> None:
    assert effective_split("gold", "train") == GOLD_SPLIT
    assert effective_split("gold", "val") == GOLD_SPLIT
    assert effective_split("train", "val") == "val"
    assert effective_split("train", "train") == "train"


def test_the_train_val_draw_is_reproducible_and_roughly_the_fraction() -> None:
    draws = [draw_split(f"ep{i:04d}", val_fraction=0.1, seed=7) for i in range(2000)]
    assert draws == [draw_split(f"ep{i:04d}", val_fraction=0.1, seed=7) for i in range(2000)]
    assert set(draws) == {"train", "val"}
    assert 0.07 < draws.count("val") / len(draws) < 0.13


# --- status ----------------------------------------------------------------------------------


def test_pot_status_counts_gold_by_clip_not_by_episode(
    db_session: Session, settings: Settings
) -> None:
    episode = make_episode(db_session, "ep000", minutes=10)
    for segment in segments_of(db_session, episode)[:3]:
        set_segment_pot(db_session, segment, "gold", actor="owner")
    db_session.flush()

    status = pot_status(db_session, settings=settings)

    assert status.buckets["gold"]["segments"] == 3
    assert status.buckets["gold"]["hours"] == pytest.approx(3 / 60, abs=1e-3)
    assert status.buckets["train"]["segments"] == 7
    assert status.buckets["train"]["hours"] == pytest.approx(7 / 60, abs=1e-3)
    assert status.buckets["gold"]["episodes"] == 1


def test_pot_status_counts_gold_episodes_that_also_feed_train(
    db_session: Session, settings: Settings
) -> None:
    """The leak per-clip gold makes possible: same speaker, same topic on both sides of the line."""
    shared = make_episode(db_session, "ep_shared", minutes=3)
    whole = make_episode(db_session, "ep_whole", minutes=2)
    set_segment_pot(db_session, segments_of(db_session, shared)[0], "gold", actor="owner")
    for segment in segments_of(db_session, whole):
        set_segment_pot(db_session, segment, "gold", actor="owner")
    db_session.flush()

    status = pot_status(db_session, settings=settings)

    assert status.gold_episodes_spanning_pots == 1
    assert status.gold_segments_in_spanning_episodes == 1


def test_val_episodes_report_in_their_own_bucket(db_session: Session, settings: Settings) -> None:
    make_episode(db_session, "ep_val", minutes=2, split="val")
    status = pot_status(db_session, settings=settings)
    assert status.buckets["val"]["segments"] == 2
    assert status.buckets["train"]["segments"] == 0


def test_pot_status_reports_labeled_hours(db_session: Session, settings: Settings) -> None:
    episode = make_episode(db_session, "ep000", minutes=2)
    label(db_session, segments_of(db_session, episode)[0], settings, tier="verified")
    status = pot_status(db_session, settings=settings)
    assert status.buckets["train"]["labeled_hours"] == pytest.approx(1 / 60, abs=1e-3)


def test_gold_coverage_reads_the_episodes_gold_clips_come_from(
    db_session: Session, settings: Settings
) -> None:
    a = make_episode(db_session, "ep_a", minutes=1, show_id="show-a", gender="female")
    make_episode(db_session, "ep_b", minutes=1, show_id="show-b", gender="male")
    set_segment_pot(db_session, segments_of(db_session, a)[0], "gold", actor="owner")
    db_session.flush()

    status = pot_status(db_session, settings=settings)

    assert set(status.gold_coverage["show_id"]) == {"show-a"}
    assert status.gold_coverage_gaps["show_id"] == ["show-b"]
    assert status.gold_coverage_gaps["gender"] == ["male"]


def test_episode_coverage_reads_speakers_and_topic(db_session: Session) -> None:
    episode = make_episode(db_session, "ep000", minutes=1, topic="technology", gender="female")
    coverage = episode_coverage(episode, keys=["show_id", "gender", "topic"])
    assert coverage == {
        "show_id": frozenset({"show-a"}),
        "gender": frozenset({"female"}),
        "topic": frozenset({"technology"}),
    }
