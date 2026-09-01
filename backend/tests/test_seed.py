"""Tests for the synthetic development seed."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import AsrHypothesis, AsrSystem, Episode, Segment, SegmentScore
from app.services.seed import seed_dev_data

pytestmark = pytest.mark.db


def test_seed_creates_the_documented_shape(db_session: Session) -> None:
    summary = seed_dev_data(db_session, episodes=1, segments_per_episode=20, systems=3)
    assert summary.episodes == 1
    assert summary.segments == 20
    assert summary.systems == 3
    assert summary.hypotheses == 60
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Episode)) == 1
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Segment)) == 20
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrSystem)) == 3
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrHypothesis)) == 60


def test_every_segment_has_scores_and_hypotheses(db_session: Session) -> None:
    seed_dev_data(db_session, episodes=1, segments_per_episode=5, systems=3)
    segments = db_session.scalars(sa.select(Segment)).all()
    for segment in segments:
        assert len(segment.hypotheses) == 3
        assert segment.scores is not None
        assert segment.duration_seconds == pytest.approx(
            segment.end_time - segment.start_time, abs=1e-6
        )


def test_seed_is_deterministic_for_a_given_seed(db_session: Session) -> None:
    seed_dev_data(db_session, episodes=1, segments_per_episode=3, systems=2, seed=99)
    first = [
        h.text_raw for h in db_session.scalars(sa.select(AsrHypothesis).order_by(AsrHypothesis.id))
    ]
    scores_first = [
        s.word_disagreement_rate
        for s in db_session.scalars(sa.select(SegmentScore).order_by(SegmentScore.segment_id))
    ]
    db_session.rollback()

    seed_dev_data(db_session, episodes=1, segments_per_episode=3, systems=2, seed=99)
    second = [
        h.text_raw for h in db_session.scalars(sa.select(AsrHypothesis).order_by(AsrHypothesis.id))
    ]
    scores_second = [
        s.word_disagreement_rate
        for s in db_session.scalars(sa.select(SegmentScore).order_by(SegmentScore.segment_id))
    ]
    assert first == second
    assert scores_first == scores_second


def test_seed_is_idempotent_on_a_second_run(db_session: Session) -> None:
    seed_dev_data(db_session, episodes=1, segments_per_episode=4, systems=2)
    summary = seed_dev_data(db_session, episodes=1, segments_per_episode=4, systems=2)
    assert summary.segments == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Segment)) == 4


def test_seeded_episodes_have_a_frozen_split(db_session: Session) -> None:
    seed_dev_data(db_session, episodes=4, segments_per_episode=2, systems=2)
    episodes = db_session.scalars(sa.select(Episode)).all()
    for episode in episodes:
        assert episode.split in {"train", "val", "test"}
        assert episode.split_seed is not None
        assert episode.split_assigned_at is not None


def test_seeded_text_contains_devanagari_and_latin(db_session: Session) -> None:
    """The corpus is code-switched; the seed must look like the real thing."""
    seed_dev_data(db_session, episodes=1, segments_per_episode=20, systems=2)
    texts = " ".join(h.text_raw for h in db_session.scalars(sa.select(AsrHypothesis)))
    assert any("ऀ" <= ch <= "ॿ" for ch in texts), "no Devanagari in seeded text"
    assert any(ch.isascii() and ch.isalpha() for ch in texts), "no Latin in seeded text"
