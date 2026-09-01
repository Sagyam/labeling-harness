"""Tests for queue building: task creation, priorities, seed rotation and audit sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationTask, AsrHypothesis, AsrSystem, Episode, Segment
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
    import_fixture(db_session, tmp_path, storage, settings, segments=6)
    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        assert 0.0 <= task.priority_score <= 1.0
        assert task.reason_jsonb is not None
        assert set(task.reason_jsonb["components"]) == {
            "word_disagreement_rate",
            "low_confidence",
            "code_switch_density",
            "rule_flag_score",
        }
        assert task.reason_jsonb["score"] == pytest.approx(task.priority_score)


def test_top_priority_segments_are_the_disagreeing_ones(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, segments=12)
    build_queue(db_session, settings=settings)
    ordered = tasks(db_session)
    top = [t.reason_jsonb["components"]["word_disagreement_rate"] for t in ordered[:4]]
    bottom = [t.reason_jsonb["components"]["word_disagreement_rate"] for t in ordered[-4:]]
    assert sum(top) / 4 > sum(bottom) / 4


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


# --- seed hypothesis selection -----------------------------------------------------------


def test_train_episodes_seed_with_the_strongest_hypothesis(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="q_ep001", segments=6)
    episode = db_session.scalars(sa.select(Episode)).one()
    episode.split = "train"
    db_session.flush()

    build_queue(db_session, settings=settings)
    for task in tasks(db_session):
        best = max(
            db_session.scalars(
                sa.select(AsrHypothesis).where(AsrHypothesis.segment_id == task.segment_id)
            ),
            key=lambda h: (h.avg_logprob if h.avg_logprob is not None else -99, h.id),
        )
        assert task.seed_hypothesis_id == best.id


def test_test_episode_seeds_rotate_across_systems(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """A gold set anchored to one system cannot be defended later; rotation is the whole point."""
    import_fixture(db_session, tmp_path, storage, settings, episode_id="rot_ep", segments=40)
    episode = db_session.scalars(sa.select(Episode)).one()
    episode.split = "test"
    db_session.flush()

    build_queue(db_session, settings=settings)
    seeds = db_session.execute(
        sa.select(AsrSystem.system_id, sa.func.count())
        .select_from(AnnotationTask)
        .join(AsrHypothesis, AsrHypothesis.id == AnnotationTask.seed_hypothesis_id)
        .join(AsrSystem, AsrSystem.id == AsrHypothesis.asr_system_id)
        .group_by(AsrSystem.system_id)
    ).all()
    counts = dict(seeds)
    assert len(counts) == 3, f"seeds concentrated on {counts}"
    assert min(counts.values()) >= 5


def test_test_episode_seed_rotation_is_deterministic(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="det_ep", segments=10)
    episode = db_session.scalars(sa.select(Episode)).one()
    episode.split = "test"
    db_session.flush()
    build_queue(db_session, settings=settings)
    first = {t.segment_id: t.seed_hypothesis_id for t in tasks(db_session)}

    db_session.execute(sa.delete(AnnotationTask))
    db_session.flush()
    build_queue(db_session, settings=settings)
    assert {t.segment_id: t.seed_hypothesis_id for t in tasks(db_session)} == first


# --- audit queue -------------------------------------------------------------------------


def test_audit_queue_samples_easy_segments(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="aud_ep", segments=60)
    report = build_queue(db_session, settings=settings, audit_sample_rate=0.2)
    audit = [t for t in tasks(db_session) if t.queue == "audit"]
    assert report.audit_tasks > 0
    assert len(audit) == report.audit_tasks
    review = [t for t in tasks(db_session) if t.queue == "review"]
    assert max(t.priority_score for t in audit) <= max(t.priority_score for t in review)


def test_audit_sampling_is_reproducible_under_a_fixed_seed(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    import_fixture(db_session, tmp_path, storage, settings, episode_id="aud2_ep", segments=40)
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
