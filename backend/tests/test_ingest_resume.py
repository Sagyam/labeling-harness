"""Surviving a power cut: a resumed episode never pays twice for what it already bought (D93)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.llm.base import AsrResult
from app.services.ingest import IngestionManager, IngestJob, run_pipeline
from app.services.ingest.checkpoint import Checkpoint
from tests.ingest_support import make_test_audio

pytestmark = pytest.mark.db


class PowerCut(BaseException):
    """The process dying: nothing downstream catches it, and no ``finally`` tidies up after it.

    ``keep_work_dir=True`` stands in for the second half -- a real power cut never reaches the
    pipeline's cleanup either.
    """


def _job(tmp_path: Path, episode_id: str, *, seconds: float = 25.0) -> IngestJob:
    audio = tmp_path / f"{episode_id}.wav"
    if not audio.exists():
        make_test_audio(audio, duration_seconds=seconds)
    return IngestJob(
        job_id=f"job-{episode_id}",
        episode_id=episode_id,
        show_id="podcast",
        title="Power cut",
        audio_path=audio,
        work_dir=tmp_path / f"work_{episode_id}",
    )


def _paid_transcribe(calls: list[str]):
    """A recogniser that is really billed: every call is counted, and none is a dry run."""

    def fake(session, audio_path, *, route, **kwargs):
        calls.append(route)
        return AsrResult(
            route=route,
            model=f"model-{route}",
            text=f"{route} heard {Path(audio_path).stem[-5:]}",
            estimated_cost_usd=Decimal("0.001"),
        )

    return fake


def _cut_at(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    def cut(*args, **kwargs):
        raise PowerCut

    monkeypatch.setattr(target, cut)


def test_a_resumed_run_reuses_every_transcript_it_paid_for(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("app.services.ingest.pipeline.transcribe", _paid_transcribe(calls))

    with monkeypatch.context() as m:
        _cut_at(m, "app.services.ingest.pipeline._run_fusion_stage")
        with pytest.raises(PowerCut):
            run_pipeline(
                _job(tmp_path, "web_cut_asr"),
                lambda: db_session,
                object_storage,
                settings,
                keep_work_dir=True,
            )
    paid = len(calls)
    assert paid > 0

    resumed = _job(tmp_path, "web_cut_asr")
    run_pipeline(resumed, lambda: db_session, object_storage, settings)

    assert resumed.status == "completed", resumed.error
    assert len(calls) == paid, "a resumed run re-bought transcripts it already had"
    assert any("reused" in entry.message.lower() for entry in resumed.logs)

    from app.models import AsrHypothesis

    texts = db_session.scalars(sa.select(AsrHypothesis.text_raw)).all()
    assert any("heard" in text for text in texts)


def test_a_resumed_run_reuses_every_fusion_window_it_paid_for(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_ingest_fusion import _echo_first

    monkeypatch.setattr("app.services.ingest.pipeline.transcribe", _paid_transcribe([]))
    fused: list[int] = []

    def counting(messages):
        fused.append(1)
        return _echo_first(messages)

    monkeypatch.setattr(
        "app.services.ingest.fusion._fusion_completer", lambda session, routes, route: counting
    )

    with monkeypatch.context() as m:
        _cut_at(m, "app.services.ingest.pipeline.import_manifest")
        with pytest.raises(PowerCut):
            run_pipeline(
                _job(tmp_path, "web_cut_fuse"),
                lambda: db_session,
                object_storage,
                settings,
                keep_work_dir=True,
            )
    paid = len(fused)
    assert paid > 0

    resumed = _job(tmp_path, "web_cut_fuse")
    run_pipeline(resumed, lambda: db_session, object_storage, settings)

    assert resumed.status == "completed", resumed.error
    assert len(fused) == paid, "a resumed run re-bought fusion windows it already had"
    assert resumed.summary["fusion"]["fused"] == resumed.summary["segments"]
    # Nothing was spent the second time, and the log does not pretend otherwise.
    assert resumed.summary["fusion"]["cost_usd"] == 0


def test_a_resumed_run_reuses_the_diarization_it_paid_for(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diarization = settings.diarization.model_copy(
        update={"enabled": True, "endpoint_url": "https://diarize.example"}
    )
    settings = settings.model_copy(update={"diarization": diarization})
    monkeypatch.setattr("app.services.ingest.pipeline.transcribe", _paid_transcribe([]))
    gpu: list[int] = []

    def diarize(path, *, num_speakers, settings):
        gpu.append(1)
        return {"labels": ["SPEAKER_00"], "turns": [{"start": 0.0, "end": 1.0, "speaker": 0}]}

    monkeypatch.setattr("app.services.ingest.pipeline.diarize_audio", diarize)

    with monkeypatch.context() as m:
        _cut_at(m, "app.services.ingest.pipeline.import_manifest")
        with pytest.raises(PowerCut):
            run_pipeline(
                _job(tmp_path, "web_cut_diar"),
                lambda: db_session,
                object_storage,
                settings,
                keep_work_dir=True,
            )
    assert gpu == [1]

    resumed = _job(tmp_path, "web_cut_diar")
    run_pipeline(resumed, lambda: db_session, object_storage, settings)
    assert resumed.status == "completed", resumed.error
    assert gpu == [1], "a resumed run diarized the episode a second time"


def test_a_finished_download_is_reused_and_a_partial_one_is_not(
    settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.ingest.pipeline import _fetch_source_audio

    downloads: list[int] = []

    def download(url, dest_dir, *, settings, on_progress):
        downloads.append(1)
        target = Path(dest_dir) / "source_audio.webm"
        target.write_bytes(b"audio" * 1000)
        return target

    monkeypatch.setattr("app.services.ingest.pipeline.download_audio", download)

    def url_job() -> IngestJob:
        work = tmp_path / "work_dl"
        work.mkdir(exist_ok=True)
        return IngestJob(
            job_id="dl",
            episode_id="dl",
            show_id="podcast",
            title="dl",
            audio_path=None,
            work_dir=work,
            source_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        )

    assert _fetch_source_audio(url_job(), settings)
    again = url_job()
    assert _fetch_source_audio(again, settings)
    assert downloads == [1]
    assert again.audio_path == tmp_path / "work_dl" / "source_audio.webm"

    # The file changed under the marker -- a torn write -- so it is fetched again.
    (tmp_path / "work_dl" / "source_audio.webm").write_bytes(b"aud")
    assert _fetch_source_audio(url_job(), settings)
    assert downloads == [1, 1]


def test_a_torn_checkpoint_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    checkpoint = Checkpoint(tmp_path)
    checkpoint.save("asr", "k1", {"text": "hello"})
    assert checkpoint.load("asr", "k1") == {"text": "hello"}

    (tmp_path / "checkpoint" / "asr" / "k2.json").write_text('{"text": "hel', encoding="utf-8")
    assert checkpoint.load("asr", "k2") is None
    assert checkpoint.load("asr", "missing") is None


# --- resuming the queue itself -----------------------------------------------------------------


def _dead_server_state(work_root: Path) -> dict[str, str]:
    from tests.test_ingest_manager import _state_left_by_a_dead_server

    return _state_left_by_a_dead_server(work_root)


def test_interrupted_jobs_resume_on_their_own_running_ones_first(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    monkeypatch.setattr(
        "app.services.ingest.pipeline.run_pipeline",
        lambda job, *args, **kwargs: ran.append(job.job_id),
    )
    ids = _dead_server_state(tmp_path)
    revived = IngestionManager()
    revived.init_state(tmp_path)
    one_slot = settings.ingest.model_copy(update={"max_concurrent_jobs": 1})

    resumed = revived.resume_interrupted(
        lambda: db_session, object_storage, settings.model_copy(update={"ingest": one_slot})
    )
    _wait_for(revived)

    assert resumed == [ids["transcribing"], ids["downloading"], ids["waiting"]]
    assert ran == resumed
    assert any("resum" in entry.message.lower() for entry in revived.get_job(ids["waiting"]).logs)


def test_a_job_whose_import_landed_before_the_cut_is_not_run_again(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.models import Episode

    ran: list[str] = []
    monkeypatch.setattr(
        "app.services.ingest.pipeline.run_pipeline",
        lambda job, *args, **kwargs: ran.append(job.job_id),
    )
    ids = _dead_server_state(tmp_path)
    db_session.add(Episode(external_id="transcribing_ep", split="train"))
    db_session.flush()

    revived = IngestionManager()
    revived.init_state(tmp_path)
    revived.resume_interrupted(lambda: db_session, object_storage, settings)
    _wait_for(revived)

    assert ids["transcribing"] not in ran
    assert revived.get_job(ids["transcribing"]).status == "completed"


def test_resuming_can_be_switched_off(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    ids = _dead_server_state(tmp_path)
    revived = IngestionManager()
    revived.init_state(tmp_path)
    off = settings.ingest.model_copy(update={"resume_interrupted": False})

    assert (
        revived.resume_interrupted(
            lambda: db_session, object_storage, settings.model_copy(update={"ingest": off})
        )
        == []
    )
    assert revived.get_job(ids["waiting"]).status == "failed"


def _wait_for(registry: IngestionManager, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with registry._lock:
            if not registry._pending and not registry._running and not registry._waiting:
                return
        time.sleep(0.01)
    raise AssertionError("queue did not drain")


def test_the_server_resumes_interrupted_jobs_when_it_starts(
    db_session: Session, object_storage, settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nobody has to open the ingest page: starting the app is enough."""
    from fastapi.testclient import TestClient

    from app.api.deps import get_config, get_object_storage, get_session_factory
    from app.main import create_app
    from app.services.ingest import manager

    ran: list[str] = []
    monkeypatch.setattr(
        "app.services.ingest.pipeline.run_pipeline",
        lambda job, *args, **kwargs: ran.append(job.job_id),
    )
    manager.reset()
    ids = _dead_server_state(settings.ingest.work_root)

    app = create_app()
    app.dependency_overrides[get_config] = lambda: settings
    app.dependency_overrides[get_session_factory] = lambda: lambda: db_session
    app.dependency_overrides[get_object_storage] = lambda: object_storage
    try:
        with TestClient(app):
            _wait_for(manager)
    finally:
        manager.reset()

    assert sorted(ran) == sorted([ids["transcribing"], ids["downloading"], ids["waiting"]])
