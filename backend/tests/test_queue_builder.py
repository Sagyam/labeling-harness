"""Tests for queue building: task creation, priorities, seed rotation and audit sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationTask, AsrHypothesis, AsrSystem, Segment
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


def import_fixture(
    session: Session,
    tmp_path: Path,
    storage,
    settings: Settings,
    *,
    episode_id: str = "q_ep001",
    segments: int = 8,
    systems: int = 3,
    **kwargs,
) -> None:
    root = build_export_fixture(
        tmp_path / f"export_{episode_id}",
        episode_id=episode_id,
        segments=segments,
        systems=systems,
        **kwargs,
    )
    import_manifest(session, root, storage=storage, settings=settings)


def tasks(session: Session) -> list[AnnotationTask]:
    return list(
        session.scalars(sa.select(AnnotationTask).order_by(AnnotationTask.priority_score.desc()))
    )


# --- task creation -----------------------------------------------------------------------


def test_build_creates_a_task_for_every_segment(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=8)
    report = build_queue(db_session, settings=settings)
    assert report.tasks_created == 8
    assert len(tasks(db_session)) == 8


def test_queued_segments_change_pipeline_status(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=4)
    build_queue(db_session, settings=settings)
    statuses = {s.pipeline_status for s in db_session.scalars(sa.select(Segment))}
    assert statuses == {"queued"}


def test_every_task_carries_a_priority_and_a_reason(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=6, with_fusion=True)
    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        assert task.reason_jsonb is not None
        assert set(task.reason_jsonb["components"]) == {
            "unsupported_rate",
            "dropped_rate",
            "asr_disagreement",
            "acoustic_gap",
            "low_confidence",
            "rule_flag_score",
        }
        offset = 1.0 if task.reason_jsonb["hazards"] else 0.0
        assert task.reason_jsonb["score"] + offset == pytest.approx(task.priority_score)


def test_the_superseded_score_is_recorded_but_does_not_rank(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """D67 travels alongside, measured against the recogniser the old queue would have used."""
    import_fixture(db_session, tmp_path, storage, settings, segments=6, with_fusion=True)
    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        legacy = task.reason_jsonb["legacy"]
        assert set(legacy["components"]) == {
            "seed_outvoted",
            "seed_orphan_rate",
            "roman_gap",
            "low_confidence",
            "rule_flag_score",
        }
        assert task.reason_jsonb["score"] == pytest.approx(
            sum(task.reason_jsonb["contributions"].values())
        )


# --- a fused seed (D74) ------------------------------------------------------------------------


def test_every_clip_is_seeded_with_its_fused_transcript(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """Gold included: the owner chose fused seeds for gold too (D74)."""
    import_fixture(db_session, tmp_path, storage, settings, segments=6, with_fusion=True)
    first = db_session.scalars(sa.select(Segment).order_by(Segment.id)).first()
    first.pot = "gold"
    db_session.flush()

    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        assert task.seed_hypothesis.system.kind == "fusion"


def test_a_clip_with_no_fused_text_falls_back_to_the_strongest_recogniser_and_is_gated(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=4)
    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        candidates = db_session.scalars(
            sa.select(AsrHypothesis).where(AsrHypothesis.segment_id == task.segment_id)
        )
        best = max(candidates, key=lambda h: (h.avg_logprob or -99, h.id))
        assert task.seed_hypothesis_id == best.id
        assert task.reason_jsonb["hazards"] == ["unfused"]
        assert task.priority_score >= 1.0


def test_an_invented_clause_is_gated_to_the_top_of_the_queue(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=8, with_fusion=True)
    victim = db_session.scalars(
        sa.select(AsrHypothesis)
        .join(AsrSystem)
        .where(AsrSystem.kind == "fusion")
        .order_by(AsrHypothesis.id)
        .offset(3)
    ).first()
    victim.text_raw = victim.text_raw + " प्रत्यक्ष निर्वाचित कार्यकारी नभइकन"
    db_session.flush()

    build_queue(db_session, settings=settings)
    top = tasks(db_session)[0]
    assert top.segment_id == victim.segment_id
    assert "invention" in top.reason_jsonb["hazards"]
    assert top.reason_jsonb["hazard_details"]["invention"].endswith("कार्यकारी नभइकन")


def test_a_gated_clip_is_never_sampled_for_audit(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """Audit measures the easy majority; a gated clip is not part of it."""
    import_fixture(db_session, tmp_path, storage, settings, segments=30)
    build_queue(db_session, settings=settings, audit_sample_rate=0.5)
    assert {t.queue for t in tasks(db_session)} == {"review"}


def test_every_task_has_a_seed_hypothesis(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=6)
    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        assert task.seed_hypothesis_id is not None
        assert task.seed_hypothesis.segment_id == task.segment_id


# --- error queue -------------------------------------------------------------------------


def test_segments_without_hypotheses_go_to_the_error_queue(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    # A manifest always carries at least one hypothesis, so a segment reaches this state only by
    # having them removed later; the queue builder must still route it away from review.
    import_fixture(db_session, tmp_path, storage, settings, episode_id="err_ep", segments=4)
    db_session.execute(sa.delete(AsrHypothesis))
    db_session.flush()

    build_queue(db_session, settings=settings)
    assert {t.queue for t in tasks(db_session)} == {"error"}
    assert all(t.seed_hypothesis_id is None for t in tasks(db_session))


def test_error_queue_segments_never_reach_review(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=3)
    segment = db_session.scalars(sa.select(Segment)).first()
    db_session.execute(sa.delete(AsrHypothesis).where(AsrHypothesis.segment_id == segment.id))
    db_session.flush()

    build_queue(db_session, settings=settings)
    task = db_session.scalars(
        sa.select(AnnotationTask).where(AnnotationTask.segment_id == segment.id)
    ).one()
    assert task.queue == "error"


# --- idempotency -------------------------------------------------------------------------


def test_rebuilding_does_not_duplicate_active_tasks(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=5)
    build_queue(db_session, settings=settings)
    report = build_queue(db_session, settings=settings)
    assert report.tasks_created == 0
    assert len(tasks(db_session)) == 5


def test_rebuilding_updates_priorities_in_place(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=4)
    build_queue(db_session, settings=settings)
    task = tasks(db_session)[0]
    task.priority_score = 0.0
    db_session.flush()

    report = build_queue(db_session, settings=settings)
    db_session.expire_all()
    refreshed = db_session.get(AnnotationTask, task.id)
    assert refreshed.priority_score > 0.0
    assert report.tasks_updated >= 1


def test_rebuilding_leaves_completed_tasks_alone(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=4)
    build_queue(db_session, settings=settings)
    done = tasks(db_session)[0]
    done.status = "done"
    done_segment_id = done.segment_id
    db_session.flush()

    build_queue(db_session, settings=settings)
    db_session.expire_all()
    rows = db_session.scalars(
        sa.select(AnnotationTask).where(AnnotationTask.segment_id == done_segment_id)
    ).all()
    assert [r.status for r in rows] == ["done"]


def test_a_new_import_adds_only_the_new_tasks(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="grow_q", segments=3)
    build_queue(db_session, settings=settings)
    import_fixture(db_session, tmp_path, storage, settings, episode_id="grow_q", segments=5)
    report = build_queue(db_session, settings=settings)
    assert report.tasks_created == 2


# --- seed hypothesis selection ---------------------------------------------------------------


def test_a_newer_fusion_system_wins_the_seed(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """A re-fusion under a new prompt version is a new system; the latest one seeds."""
    import_fixture(db_session, tmp_path, storage, settings, segments=2, with_fusion=True)
    newer = AsrSystem(system_id="fusion-gemini-3.8-flash-p2", kind="fusion")
    db_session.add(newer)
    db_session.flush()
    for segment in db_session.scalars(sa.select(Segment)):
        db_session.add(AsrHypothesis(segment_id=segment.id, asr_system_id=newer.id, text_raw="नयाँ"))
    db_session.flush()

    build_queue(db_session, settings=settings)
    assert {t.seed_hypothesis.system.system_id for t in tasks(db_session)} == {
        "fusion-gemini-3.8-flash-p2"
    }


# --- audit queue -------------------------------------------------------------------------


def test_audit_queue_samples_easy_segments(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(
        db_session, tmp_path, storage, settings, episode_id="aud_ep", segments=60, with_fusion=True
    )
    report = build_queue(db_session, settings=settings, audit_sample_rate=0.2)
    audit = [t for t in tasks(db_session) if t.queue == "audit"]
    assert report.audit_tasks > 0
    assert len(audit) == report.audit_tasks
    review = [t for t in tasks(db_session) if t.queue == "review"]
    assert max(t.priority_score for t in audit) <= max(t.priority_score for t in review)


def test_audit_sampling_is_reproducible_under_a_fixed_seed(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(
        db_session, tmp_path, storage, settings, episode_id="aud2_ep", segments=40, with_fusion=True
    )
    build_queue(db_session, settings=settings, audit_sample_rate=0.25)
    first = sorted(t.segment_id for t in tasks(db_session) if t.queue == "audit")

    db_session.execute(sa.delete(AnnotationTask))
    db_session.flush()
    build_queue(db_session, settings=settings, audit_sample_rate=0.25)
    second = sorted(t.segment_id for t in tasks(db_session) if t.queue == "audit")
    assert first == second
    assert first


def test_audit_sample_rate_of_zero_creates_no_audit_tasks(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="aud3_ep", segments=20)
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    assert [t for t in tasks(db_session) if t.queue == "audit"] == []


# --- filtering ---------------------------------------------------------------------------


def test_build_can_be_limited_to_one_episode(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="one_ep", segments=3)
    import_fixture(db_session, tmp_path, storage, settings, episode_id="two_ep", segments=3)
    report = build_queue(db_session, settings=settings, episode_external_id="one_ep")
    assert report.tasks_created == 3
    assert len(tasks(db_session)) == 3


def test_dry_run_writes_no_tasks(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=5)
    report = build_queue(db_session, settings=settings, dry_run=True)
    assert report.tasks_created == 5
    assert tasks(db_session) == []


def test_report_renders_a_readable_summary(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=5)
    text = build_queue(db_session, settings=settings).render()
    assert "review" in text
    assert "tasks" in text
