"""Overlap detection and diarization during ingest: measured once, never fatal."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Episode, Segment
from app.services.acoustics import ACOUSTICS_VERSION
from app.services.ingest import (
    IngestJob,
    run_pipeline,
)
from tests.ingest_support import _pipeline_job, make_test_audio

pytestmark = pytest.mark.db


class _StubOverlap:
    """Stands in for :class:`OverlapDetector`: fixed episode-relative spans, or a failure."""

    def __init__(self, spans=None, *, error: Exception | None = None) -> None:
        self.spans, self.error, self.calls = spans, error, 0

    def detect(self, audio, sample_rate):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.spans


def _ingest_with(
    db_session: Session,
    object_storage,
    settings,
    tmp_path: Path,
    detector,
    name: str,
    meter=None,
):
    job = IngestJob(
        job_id=f"test-job-{name}",
        episode_id=f"ov_{name}",
        show_id="podcast",
        title="Overlap Test Episode",
        audio_path=make_test_audio(tmp_path / f"{name}.wav", duration_seconds=6.0),
        work_dir=tmp_path / f"work_{name}",
    )
    run_pipeline(
        job,
        session_factory=lambda: db_session,
        storage=object_storage,
        settings=settings,
        overlap_detector=detector,
        acoustic_meter=meter,
    )
    assert job.error is None, job.error
    episode = db_session.scalar(sa.select(Episode).where(Episode.external_id == f"ov_{name}"))
    segments = db_session.scalars(sa.select(Segment).where(Segment.episode_id == episode.id)).all()
    return job, sorted(segments, key=lambda s: s.start_time)


def test_overlap_is_detected_once_per_episode_and_stored_per_clip(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    detector = _StubOverlap(spans=[(0.5, 1.5)])
    job, segments = _ingest_with(db_session, object_storage, settings, tmp_path, detector, "found")
    assert detector.calls == 1
    assert all(s.overlap_spans_jsonb is not None for s in segments)
    first = segments[0]
    assert first.start_time <= 0.5
    expected = [
        [round(0.5 - first.start_time, 3), round(min(1.5, first.end_time) - first.start_time, 3)]
    ]
    assert first.overlap_spans_jsonb == expected
    assert "speaker_overlap" in first.scores.flags_jsonb
    assert any("overlap" in item.message.lower() for item in job.logs)


def test_without_an_overlap_model_clips_are_left_unmeasured(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    _, segments = _ingest_with(
        db_session, object_storage, settings, tmp_path, _StubOverlap(spans=None), "absent"
    )
    assert all(s.overlap_spans_jsonb is None for s in segments)


def test_a_failing_overlap_detector_never_fails_the_ingest(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    detector = _StubOverlap(error=RuntimeError("onnx exploded"))
    job, segments = _ingest_with(db_session, object_storage, settings, tmp_path, detector, "broken")
    assert job.status == "completed"
    assert all(s.overlap_spans_jsonb is None for s in segments)
    assert any("onnx exploded" in item.message for item in job.logs)


# --- Acoustics (D87) ------------------------------------------------------------------------


class _BrokenMeter:
    def measure(self, audio, sample_rate, clips):
        raise RuntimeError("fft exploded")


def test_every_clip_is_measured_as_it_is_cut(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    _, segments = _ingest_with(
        db_session, object_storage, settings, tmp_path, _StubOverlap(spans=[]), "acoustic"
    )
    assert segments
    for segment in segments:
        assert segment.acoustics_jsonb["version"] == ACOUSTICS_VERSION
        assert segment.acoustics_jsonb["bandwidth_hz"] > 0


def test_a_failing_meter_never_fails_the_ingest(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    job, segments = _ingest_with(
        db_session,
        object_storage,
        settings,
        tmp_path,
        _StubOverlap(spans=[]),
        "nometer",
        meter=_BrokenMeter(),
    )
    assert job.status == "completed"
    assert all(s.acoustics_jsonb is None for s in segments)
    assert any("fft exploded" in item.message for item in job.logs)


# --- Remote diarization (D79) ------------------------------------------------------------


def _diarizing(settings):
    return settings.model_copy(
        update={
            "diarization": settings.diarization.model_copy(
                update={"enabled": True, "endpoint_url": "https://example.modal.run"}
            )
        }
    )


def test_the_diarizers_turns_become_the_episodes_first_run(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.models import DiarizationRun

    sent: dict = {}

    def fake_diarize(audio, *, num_speakers, settings):
        sent.update(path=audio, num_speakers=num_speakers)
        return {
            "turns": [[0.0, 3.0, "SPEAKER_00"], [2.5, 6.0, "SPEAKER_01"]],
            "labels": ["SPEAKER_00", "SPEAKER_01"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
        }

    monkeypatch.setattr("app.services.ingest.pipeline.diarize_audio", fake_diarize)
    job = _pipeline_job(tmp_path, "diarized", seconds=6.0)
    job.metadata = {"speakers": {"host": {"gender": "male"}, "guest": {"gender": "female"}}}

    run_pipeline(job, lambda: db_session, object_storage, _diarizing(settings))

    assert job.status == "completed", job.error
    assert sent["num_speakers"] == 2, "the declared speaker count is passed on"
    assert sent["path"].name.endswith("_normalized.flac"), "the whole episode, not a clip"
    ep = db_session.scalar(sa.select(Episode).where(Episode.external_id == job.episode_id))
    run = db_session.scalar(sa.select(DiarizationRun).where(DiarizationRun.episode_id == ep.id))
    assert run is not None
    assert run.source == "https://example.modal.run"
    assert [(t.start_time, t.end_time, t.speaker) for t in run.turns] == [
        (0.0, 3.0, "SPEAKER_00"),
        (2.5, 6.0, "SPEAKER_01"),
    ]
    # Voices are handed out in talk-time order, and SPEAKER_01 talks longer (D87).
    assert run.voices_jsonb == {"SPEAKER_01": "v001", "SPEAKER_00": "v002"}


def test_a_failing_diarizer_never_fails_the_ingest(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.models import DiarizationRun

    def broken(audio, *, num_speakers, settings):
        raise RuntimeError("GPU on fire")

    monkeypatch.setattr("app.services.ingest.pipeline.diarize_audio", broken)
    job = _pipeline_job(tmp_path, "undiarized", seconds=6.0)

    run_pipeline(job, lambda: db_session, object_storage, _diarizing(settings))

    assert job.status == "completed", job.error
    assert any("GPU on fire" in item.message and item.level == "warn" for item in job.logs)
    assert db_session.scalar(sa.select(sa.func.count()).select_from(DiarizationRun)) == 0
    ep = db_session.scalar(sa.select(Episode).where(Episode.external_id == job.episode_id))
    assert db_session.scalars(sa.select(Segment).where(Segment.episode_id == ep.id)).all()


def test_a_malformed_diarization_leaves_the_imported_episode_intact(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad answer is reported, and the clips imported beside it stay."""
    monkeypatch.setattr(
        "app.services.ingest.pipeline.diarize_audio",
        lambda audio, *, num_speakers, settings: {"turns": [[5.0, 1.0, "SPEAKER_00"]]},
    )
    job = _pipeline_job(tmp_path, "malformed", seconds=6.0)

    run_pipeline(job, lambda: db_session, object_storage, _diarizing(settings))

    assert job.status == "completed", job.error
    assert any("ends before it starts" in item.message for item in job.logs)
