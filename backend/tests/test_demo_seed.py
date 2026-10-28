"""Tests for the demo seed: fictional data that fills every page, for screenshots."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import (
    AsrModel,
    DiarizationRun,
    Episode,
    LlmRequest,
    Segment,
    SegmentLabel,
)
from app.services.demo_seed import DemoSeedError, require_demo_database, seed_demo

pytestmark = pytest.mark.db


def test_refuses_any_database_not_named_demo() -> None:
    with pytest.raises(DemoSeedError):
        require_demo_database("harness")
    with pytest.raises(DemoSeedError):
        require_demo_database("harness_test")
    require_demo_database("harness_demo")


def _count(session: Session, model: type) -> int:
    return session.scalar(sa.select(sa.func.count()).select_from(model)) or 0


@pytest.fixture
def seeded(db_session: Session, object_storage, settings, tmp_path):
    summary = seed_demo(
        db_session,
        storage=object_storage,
        settings=settings,
        work_root=tmp_path / "demo",
        episodes=4,
        clips_per_episode=6,
    )
    db_session.flush()
    return summary


def test_seeds_episodes_clips_and_both_pots(db_session: Session, seeded) -> None:
    assert seeded.episodes == 4
    assert _count(db_session, Episode) == 4
    assert _count(db_session, Segment) == 24
    pots = set(db_session.scalars(sa.select(Segment.pot)))
    assert pots == {"gold", "train"}
    genres = {e.metadata_jsonb.get("genre") for e in db_session.scalars(sa.select(Episode))}
    assert None not in genres


def test_labels_carry_both_tiers_and_never_screen_gold(db_session: Session, seeded) -> None:
    rows = db_session.execute(
        sa.select(SegmentLabel.verification_tier, SegmentLabel.disposition, Segment.pot).join(
            Segment, Segment.id == SegmentLabel.segment_id
        )
    ).all()
    assert {r.verification_tier for r in rows} == {"verified", "screened"}
    assert {r.disposition for r in rows} >= {"accepted_unchanged", "edited"}
    assert not [r for r in rows if r.pot == "gold" and r.verification_tier == "screened"]
    # Some clips are left for the queue, or the triage screenshot is empty.
    assert seeded.labelled < 24


def test_voices_costs_and_a_model_exist(db_session: Session, seeded) -> None:
    runs = list(db_session.scalars(sa.select(DiarizationRun)))
    assert len(runs) == 4
    assert all(run.voices_jsonb for run in runs)
    voices = {v for run in runs for v in run.voices_jsonb.values()}
    # Hosts recur across their show's episodes, so fewer voices than speakers.
    assert len(voices) < sum(len(run.voices_jsonb) for run in runs)
    assert _count(db_session, LlmRequest) > 0
    model = db_session.scalars(sa.select(AsrModel)).one()
    assert model.runs


def test_a_second_run_adds_nothing(
    db_session: Session, seeded, object_storage, settings, tmp_path
) -> None:
    again = seed_demo(
        db_session,
        storage=object_storage,
        settings=settings,
        work_root=tmp_path / "demo",
        episodes=4,
        clips_per_episode=6,
    )
    assert again.episodes == 0
    assert _count(db_session, Segment) == 24


def test_every_page_has_something_to_show(client, seeded) -> None:
    assert client.get("/queue").json()
    assert client.get("/episodes").json()
    assert client.get("/voices").json()
    assert client.get("/models").json()
    assert client.get("/stats/inventory").status_code == 200
