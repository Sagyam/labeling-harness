"""Backfilling acoustic measurements onto clips already imported (D87)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import load_settings
from app.models import AuditLog, Episode, Segment
from app.services.acoustics import BANDWIDTH_ONLY_VERSION, AcousticMeter
from app.services.acoustics_backfill import backfill_acoustics
from app.services.importer import import_manifest
from app.storage.local import LocalFilesystemStorage
from tests.fixtures import build_export_fixture

pytestmark = pytest.mark.db


class _CountingMeter(AcousticMeter):
    def __init__(self) -> None:
        super().__init__()
        self.clips_seen = 0

    def measure(self, audio, sample_rate, clips):
        self.clips_seen += len(clips)
        return super().measure(audio, sample_rate, clips)


def _imported(
    session: Session, tmp_path: Path, name: str
) -> tuple[Episode, LocalFilesystemStorage]:
    storage = LocalFilesystemStorage(root=tmp_path / "objects")
    root = build_export_fixture(tmp_path / name, episode_id=name, segments=3, systems=2)
    import_manifest(session, root, storage=storage, settings=load_settings())
    episode = session.scalars(sa.select(Episode).where(Episode.external_id == name)).one()
    return episode, storage


def _segments(session: Session, episode: Episode) -> list[Segment]:
    return list(session.scalars(sa.select(Segment).where(Segment.episode_id == episode.id)))


def test_every_clip_is_measured_and_versioned(db_session: Session, tmp_path: Path) -> None:
    episode, storage = _imported(db_session, tmp_path, "ac_ep1")
    assert all(s.acoustics_jsonb is None for s in _segments(db_session, episode))

    report = backfill_acoustics(db_session, storage, AcousticMeter(), actor="test")

    assert report.episodes_measured == 1
    assert report.segments_updated == 3
    for segment in _segments(db_session, episode):
        assert segment.acoustics_jsonb["version"] == BANDWIDTH_ONLY_VERSION
        assert segment.acoustics_jsonb["bandwidth_hz"] > 0
    audit = db_session.scalars(sa.select(AuditLog).where(AuditLog.action == "acoustics_backfill"))
    assert [a.entity_id for a in audit] == ["ac_ep1"]


def test_a_second_run_measures_only_what_is_stale(db_session: Session, tmp_path: Path) -> None:
    episode, storage = _imported(db_session, tmp_path, "ac_ep2")
    backfill_acoustics(db_session, storage, AcousticMeter(), actor="test")

    meter = _CountingMeter()
    again = backfill_acoustics(db_session, storage, meter, actor="test")
    assert again.episodes_skipped == 1 and meter.clips_seen == 0

    stale = _segments(db_session, episode)[0]
    stale.acoustics_jsonb = {"version": "acoustics-v0", "bandwidth_hz": 1.0}
    db_session.flush()
    backfill_acoustics(db_session, storage, meter, actor="test")
    assert meter.clips_seen == 1
    assert stale.acoustics_jsonb["version"] == BANDWIDTH_ONLY_VERSION

    backfill_acoustics(db_session, storage, meter, actor="test", force=True)
    assert meter.clips_seen == 4


def test_an_episode_without_retained_audio_is_counted_not_measured(
    db_session: Session, tmp_path: Path
) -> None:
    episode, storage = _imported(db_session, tmp_path, "ac_ep3")
    episode.audio_object_key = None
    db_session.flush()
    report = backfill_acoustics(db_session, storage, AcousticMeter(), actor="test")
    assert report.episodes_without_audio == 1
    assert all(s.acoustics_jsonb is None for s in _segments(db_session, episode))
