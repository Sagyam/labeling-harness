"""Backfilling overlap spans and the speaker_overlap heads-up onto episodes already imported."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationTask, AuditLog, Episode, Segment
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.services.overlap_backfill import backfill_overlap
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


class _StubOverlap:
    def __init__(self, spans) -> None:
        self.spans, self.calls = spans, 0

    def detect(self, audio, sample_rate):
        self.calls += 1
        return self.spans


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


@pytest.fixture
def settings() -> Settings:
    return load_settings()


def _imported(session: Session, tmp_path: Path, storage, settings: Settings, name: str) -> Episode:
    root = build_export_fixture(tmp_path / name, episode_id=name, segments=3, systems=2)
    import_manifest(session, root, storage=storage, settings=settings)
    build_queue(session, settings=settings)
    return session.scalars(sa.select(Episode).where(Episode.external_id == name)).one()


def _segments(session: Session, episode: Episode) -> list[Segment]:
    return list(
        session.scalars(
            sa.select(Segment).where(Segment.episode_id == episode.id).order_by(Segment.start_time)
        )
    )


def test_spans_and_the_flag_are_written_to_every_clip(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    episode = _imported(db_session, tmp_path, storage, settings, "bf_ep1")
    first = _segments(db_session, episode)[0]
    detector = _StubOverlap([(first.start_time + 0.2, first.start_time + 1.2)])

    report = backfill_overlap(db_session, storage, detector, settings=settings, actor="test")

    assert report.episodes_measured == 1
    segments = _segments(db_session, episode)
    assert all(s.overlap_spans_jsonb is not None for s in segments)
    assert segments[0].overlap_spans_jsonb == [[0.2, 1.2]]
    assert "speaker_overlap" in segments[0].scores.flags_jsonb
    assert all("speaker_overlap" not in s.scores.flags_jsonb for s in segments[1:])
    assert report.segments_flagged == 1


def test_open_tasks_show_the_new_flag_without_a_queue_rebuild(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    episode = _imported(db_session, tmp_path, storage, settings, "bf_ep2")
    first = _segments(db_session, episode)[0]
    task = db_session.scalars(
        sa.select(AnnotationTask).where(AnnotationTask.segment_id == first.id)
    ).one()
    before, priority = dict(task.reason_jsonb), task.priority_score

    backfill_overlap(
        db_session,
        storage,
        _StubOverlap([(first.start_time, first.start_time + 2.0)]),
        settings=settings,
        actor="test",
    )

    db_session.refresh(task)
    assert "speaker_overlap" in task.reason_jsonb["flags"]
    assert task.priority_score == priority  # a heads-up never moves a clip in the queue
    unchanged = {k: v for k, v in task.reason_jsonb.items() if k != "flags"}
    assert unchanged == {k: v for k, v in before.items() if k != "flags"}


def test_measured_episodes_are_skipped_unless_forced(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    episode = _imported(db_session, tmp_path, storage, settings, "bf_ep3")
    first = _segments(db_session, episode)[0]
    backfill_overlap(
        db_session,
        storage,
        _StubOverlap([(first.start_time, first.start_time + 2.0)]),
        settings=settings,
        actor="test",
    )

    again = _StubOverlap([])
    report = backfill_overlap(db_session, storage, again, settings=settings, actor="test")
    assert again.calls == 0 and report.episodes_skipped == 1

    report = backfill_overlap(
        db_session, storage, again, settings=settings, actor="test", force=True
    )
    assert again.calls == 1
    first = _segments(db_session, episode)[0]
    assert first.overlap_spans_jsonb == []
    assert "speaker_overlap" not in first.scores.flags_jsonb


def test_without_a_model_nothing_changes(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    episode = _imported(db_session, tmp_path, storage, settings, "bf_ep4")
    report = backfill_overlap(db_session, storage, _StubOverlap(None), settings=settings, actor="t")
    assert report.episodes_measured == 0
    assert report.model_unavailable is True
    assert all(s.overlap_spans_jsonb is None for s in _segments(db_session, episode))


def test_every_measured_episode_leaves_an_audit_row(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    _imported(db_session, tmp_path, storage, settings, "bf_ep5")
    backfill_overlap(db_session, storage, _StubOverlap([]), settings=settings, actor="backfill")
    rows = db_session.scalars(
        sa.select(AuditLog).where(
            AuditLog.entity_type == "episode",
            AuditLog.entity_id == "bf_ep5",
            AuditLog.action == "overlap_backfill",
        )
    ).all()
    assert len(rows) == 1
    assert rows[0].actor == "backfill"
    assert rows[0].new_values_jsonb["segments"] == 3


def test_an_episode_without_retained_audio_is_skipped(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    episode = _imported(db_session, tmp_path, storage, settings, "bf_ep6")
    episode.audio_object_key = None
    db_session.flush()
    detector = _StubOverlap([])
    report = backfill_overlap(db_session, storage, detector, settings=settings, actor="t")
    assert detector.calls == 0
    assert report.episodes_without_audio == 1
