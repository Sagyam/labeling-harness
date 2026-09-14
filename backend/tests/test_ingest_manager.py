"""The job state machine, the AZ-5 scram and the in-process queue."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Episode, LlmRequest
from app.services.ingest import (
    IngestJob,
    LockedSession,
    manager,
    run_pipeline,
)
from tests.ingest_support import _drain, _queued_job, make_test_audio

pytestmark = pytest.mark.db


def test_locked_session_thread_safety(db_session: Session) -> None:
    import concurrent.futures

    locked = LockedSession(db_session)

    def write_row(i: int) -> None:
        locked.add(
            LlmRequest(
                route=f"thread_test_{i}",
                model="test-model",
                request_hash=f"hash_{i}",
                input_summary="summary",
                status="succeeded",
            )
        )
        locked.flush()
        locked.commit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write_row, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    rows = db_session.scalars(
        sa.select(LlmRequest).where(LlmRequest.route.like("thread_test_%"))
    ).all()
    assert len(rows) == 16


def test_finished_jobs_are_evicted_from_the_registry(tmp_path: Path) -> None:
    from app.services.ingest import IngestionManager

    registry = IngestionManager()
    registry.MAX_FINISHED_JOBS = 2
    made = []
    for index in range(5):
        job = registry.create_job(
            episode_id=f"ep{index}",
            show_id="podcast",
            title=f"Episode {index}",
            audio_path=tmp_path / "a.wav",
            work_dir=tmp_path / f"w{index}",
        )
        job.status = "completed"
        made.append(job)

    # The newest job is still 'pending' when eviction runs, so three finished ones remain at most.
    assert registry.get_job(made[0].job_id) is None
    assert registry.get_job(made[-1].job_id) is not None


def test_events_reach_a_subscriber_from_the_pipeline_thread() -> None:
    """The pipeline runs on a worker thread while the queue belongs to the server's event loop."""
    import asyncio
    import threading

    async def scenario() -> str:
        job = IngestJob(
            job_id="test-emit",
            episode_id="ep",
            show_id="podcast",
            title="Emit",
            audio_path=Path("a.wav"),
            work_dir=Path("w"),
        )
        queue: asyncio.Queue[str] = asyncio.Queue()
        job.listeners.append((asyncio.get_running_loop(), queue))

        threading.Thread(target=job.log, args=("from the worker",), daemon=True).start()
        return await asyncio.wait_for(queue.get(), timeout=5.0)

    payload = asyncio.run(scenario())
    assert "from the worker" in payload


# --- AZ-5: stopping a run in flight -----------------------------------------------------


def test_scram_before_inference_imports_nothing(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    """A run scrammed before stage 3 stops there and writes no episode.

    The point of the abort is that it is not a partial success: half an episode is a
    differently-sampled episode, so nothing reaches the corpus (D46).
    """
    raw_audio = make_test_audio(tmp_path / "scram_raw.wav", duration_seconds=6.0)
    job = IngestJob(
        job_id="test-scram-001",
        episode_id="web_scram01",
        show_id="podcast",
        title="Scrammed Before Inference",
        audio_path=raw_audio,
        work_dir=tmp_path / "work_scram01",
    )
    assert job.scram(reason="test") is True

    run_pipeline(
        job,
        session_factory=lambda: db_session,
        storage=object_storage,
        settings=settings,
    )

    assert job.status == "aborted"
    assert job.error is None
    assert job.summary is not None
    assert job.summary["imported"] is False
    assert job.summary["reason"] == "test"
    assert db_session.scalar(sa.select(Episode).where(Episode.external_id == "web_scram01")) is None


def test_scram_mid_run_stops_dispatching_inference(
    db_session: Session,
    object_storage,
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once AZ-5 is pressed, no further segment reaches a transcriber.

    This is the whole reason the button exists, so it is asserted on the call count rather than
    on the job's status alone: a scram that let the queued segments run would still bill for
    every one of them.
    """
    raw_audio = make_test_audio(tmp_path / "scram_mid.wav", duration_seconds=30.0)
    job = IngestJob(
        job_id="test-scram-002",
        episode_id="web_scram02",
        show_id="podcast",
        title="Scrammed Mid Run",
        audio_path=raw_audio,
        work_dir=tmp_path / "work_scram02",
    )

    calls: list[str] = []
    real_transcribe = __import__("app.services.ingest.pipeline", fromlist=["transcribe"]).transcribe

    def counting_transcribe(session, audio_path, **kwargs):
        calls.append(str(audio_path))
        # Press the button on the very first clip: every segment after this one must be untouched.
        job.scram(reason="test mid-run")
        return real_transcribe(session, audio_path, **kwargs)

    monkeypatch.setattr("app.services.ingest.pipeline.transcribe", counting_transcribe)
    # One segment at a time, so "no clip after the press" is an exact count rather than a count
    # plus whatever the pool already had in flight.
    serial_settings = settings.model_copy(
        update={"ingest": settings.ingest.model_copy(update={"max_segment_concurrency": 1})}
    )

    run_pipeline(
        job,
        session_factory=lambda: db_session,
        storage=object_storage,
        settings=serial_settings,
    )

    assert job.status == "aborted"
    assert job.total_segments > 1, "need more than one segment for this to prove anything"
    # One segment's worth of routes, and not a clip more.
    assert len(set(calls)) == 1
    assert db_session.scalar(sa.select(Episode).where(Episode.external_id == "web_scram02")) is None


def test_scram_is_idempotent_and_reports_who_stopped_it() -> None:
    """The button someone hits twice reports the second press honestly."""
    job = IngestJob(
        job_id="test-scram-003",
        episode_id="web_scram03",
        show_id="podcast",
        title="Double Press",
        audio_path=None,
        work_dir=Path("/tmp/nonexistent-scram03"),
    )
    assert job.scram() is True
    assert job.scrammed is True
    assert job.scram(reason="again") is False
    assert job.scram_reason == "manual SCRAM"


def test_scram_on_a_finished_run_changes_nothing() -> None:
    job = IngestJob(
        job_id="test-scram-004",
        episode_id="web_scram04",
        show_id="podcast",
        title="Already Done",
        audio_path=None,
        work_dir=Path("/tmp/nonexistent-scram04"),
    )
    job.status = "completed"
    assert job.scram() is False
    assert job.scrammed is False


def test_api_scram_endpoint_halts_a_running_job(client: TestClient, tmp_path: Path) -> None:
    job = manager.create_job(
        episode_id="scram_api_ep",
        show_id="demo",
        title="SCRAM API",
        audio_path=tmp_path / "audio.wav",
        work_dir=tmp_path / "work",
    )
    job.status = "processing"

    response = client.post(f"/ingest/{job.job_id}/scram")
    assert response.status_code == 200
    body = response.json()
    assert body["scrammed"] is True
    assert body["already_stopping"] is False

    again = client.post(f"/ingest/{job.job_id}/scram")
    assert again.status_code == 200
    assert again.json()["already_stopping"] is True

    status_body = client.get(f"/ingest/{job.job_id}").json()
    assert status_body["scrammed"] is True


def test_api_scram_unknown_job_404(client: TestClient) -> None:
    assert client.post("/ingest/nonexistent-id-000/scram").status_code == 404


# --- the ingestion queue -----------------------------------------------------------------


def test_queued_jobs_run_one_at_a_time_in_submission_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    """The whole point of the queue: two submissions must not share the machine.

    Overlap is asserted directly rather than through timing -- a concurrency bug that only shows
    up on a loaded CI box is not a test.
    """
    running = 0
    overlapped = False
    order: list[str] = []
    guard = threading.Lock()

    def fake_pipeline(job, *args) -> None:
        nonlocal running, overlapped
        with guard:
            running += 1
            overlapped = overlapped or running > 1
            order.append(job.episode_id)
        time.sleep(0.02)
        with guard:
            running -= 1
        job.status = "completed"

    monkeypatch.setattr("app.services.ingest.pipeline.run_pipeline", fake_pipeline)

    jobs = [_queued_job(f"queued_{n}", tmp_path) for n in range(4)]
    positions = [manager.submit(job, lambda: None, None, settings) for job in jobs]
    _drain()

    assert not overlapped
    assert order == [job.episode_id for job in jobs]
    # The first submission starts immediately; each later one reports the jobs ahead of it.
    assert positions[0] == 0
    assert positions == sorted(positions)


def test_a_job_scrammed_while_still_queued_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    """AZ-5 on a waiting job must cost nothing, not be discovered a stage into the run."""
    started: list[str] = []
    release = threading.Event()

    def fake_pipeline(job, *args) -> None:
        started.append(job.episode_id)
        release.wait(timeout=5.0)
        job.status = "completed"

    monkeypatch.setattr("app.services.ingest.pipeline.run_pipeline", fake_pipeline)

    first = _queued_job("scram_first", tmp_path)
    second = _queued_job("scram_second", tmp_path)
    manager.submit(first, lambda: None, None, settings)
    manager.submit(second, lambda: None, None, settings)

    second.scram("queued and no longer wanted")
    release.set()
    _drain()

    assert started == ["scram_first"]
    assert second.status == "aborted"


def test_a_crashing_job_does_not_stop_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    """One bad job must not look like the queue silently stopping."""

    def fake_pipeline(job, *args) -> None:
        if job.episode_id == "crash_me":
            raise RuntimeError("boom")
        job.status = "completed"

    monkeypatch.setattr("app.services.ingest.pipeline.run_pipeline", fake_pipeline)

    crasher = _queued_job("crash_me", tmp_path)
    survivor = _queued_job("crash_survivor", tmp_path)
    manager.submit(crasher, lambda: None, None, settings)
    manager.submit(survivor, lambda: None, None, settings)
    _drain()

    assert crasher.status == "failed"
    assert survivor.status == "completed"


def test_the_queue_endpoint_lists_what_is_waiting(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    release = threading.Event()

    def fake_pipeline(job, *args) -> None:
        job.status = "processing"
        release.wait(timeout=5.0)
        job.status = "completed"

    monkeypatch.setattr("app.services.ingest.pipeline.run_pipeline", fake_pipeline)

    first = _queued_job("listed_first", tmp_path)
    second = _queued_job("listed_second", tmp_path)
    manager.submit(first, lambda: None, None, settings)
    manager.submit(second, lambda: None, None, settings)

    deadline = time.monotonic() + 5.0
    while manager._running_id != first.job_id and time.monotonic() < deadline:
        time.sleep(0.01)

    body = client.get("/ingest").json()
    listed = [row["episode_id"] for row in body["jobs"]]
    assert listed[:2] == ["listed_first", "listed_second"]
    assert body["running"]["episode_id"] == "listed_first"
    assert client.get(f"/ingest/{second.job_id}").json()["queue_position"] == 1

    release.set()
    _drain()


def test_queue_rich_snapshot_api(client: TestClient, tmp_path: Path) -> None:
    manager.reset()
    job = manager.create_job(
        episode_id="snap_ep",
        show_id="demo",
        title="Snapshot Ep",
        work_dir=tmp_path / "snap",
    )
    job.status = "backlog"
    job.backlog(reason="youtube_bot_detected", error_message="Bot challenge")

    response = client.get("/ingest")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "upcoming" in data
    assert "backlog" in data
    assert "past" in data
    assert "counts" in data
    assert data["counts"]["backlog"] == 1
    assert data["backlog"][0]["episode_id"] == "snap_ep"


def test_retry_and_cancel_endpoints_api(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.services.ingest.pipeline.run_pipeline", lambda *args, **kwargs: None)
    manager.reset()
    work_dir = tmp_path / "retry_ep"
    work_dir.mkdir(parents=True, exist_ok=True)
    job = manager.create_job(
        episode_id="retry_ep",
        show_id="demo",
        title="Retry Ep",
        work_dir=work_dir,
    )
    job.status = "failed"
    job.error = "Simulated failure"

    # Retry single
    resp = client.post(f"/ingest/{job.job_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    _drain()

    # Cancel job
    cancel_resp = client.delete(f"/ingest/{job.job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["action"] in ("scrammed", "removed", "cancelled", "deleted")

    # Retry all matching status
    job2 = manager.create_job(
        episode_id="backlog_ep",
        show_id="demo",
        title="Backlog Ep",
        work_dir=tmp_path / "backlog_ep",
    )
    job2.status = "backlog"
    job2.stage = "backlog"
    with manager._lock:
        manager._backlog.append(job2.job_id)

    batch_retry = client.post("/ingest/retry-all", json={"status": "backlog"})
    assert batch_retry.status_code == 200
    assert len(batch_retry.json()["results"]) == 1
    assert batch_retry.json()["results"][0]["success"] is True
    _drain()

    # Clear past
    job3 = manager.create_job(
        episode_id="past_ep",
        show_id="demo",
        title="Past Ep",
        work_dir=tmp_path / "past_ep",
    )
    job3.status = "completed"
    clear_resp = client.post("/ingest/clear-past")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["cleared_count"] >= 1
    assert manager.get_job(job3.job_id) is None
    manager.reset()


# --- overlapped speech at ingest (D77) --------------------------------------------------------
