"""The ingestion HTTP endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.ingest import (
    manager,
)
from tests.ingest_support import make_test_audio

pytestmark = pytest.mark.db


# --- API Endpoints: Ingestion Service ----------------------------------------------------


def test_api_ingest_start_and_status(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = make_test_audio(tmp_path / "upload.wav", duration_seconds=3.0)
    monkeypatch.setattr(
        "app.services.ingest.pipeline.run_pipeline",
        lambda job, *args: job.set_progress("normalizing", 20.0),
    )

    with open(wav_path, "rb") as f:
        response = client.post(
            "/ingest",
            data={"episode_title": "API Test Ep", "show_id": "demo", "episode_id": "api_ep_99"},
            files={"file": ("upload.wav", f, "audio/wav")},
        )

    assert response.status_code == 202
    body = response.json()
    job_id = body["job_id"]
    assert body["episode_id"] == "api_ep_99"

    # Status check
    st_res = client.get(f"/ingest/{job_id}")
    assert st_res.status_code == 200
    st_data = st_res.json()
    assert st_data["job_id"] == job_id
    assert "status" in st_data
    assert "stage" in st_data
    assert "logs" in st_data


def test_an_upload_carries_the_forms_speaker_count(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = make_test_audio(tmp_path / "upload.wav", duration_seconds=3.0)
    # Not queued: the shared worker could pick the job up after this test's stubs are gone, and a
    # real pipeline would commit the episode outside the test's transaction.
    monkeypatch.setattr(manager, "submit", lambda job, *args, **kwargs: 0)

    with open(wav_path, "rb") as f:
        response = client.post(
            "/ingest",
            data={"episode_title": "Panel", "episode_id": "api_panel", "speaker_count": "5"},
            files={"file": ("upload.wav", f, "audio/wav")},
        )

    assert response.status_code == 202
    job = manager.get_job(response.json()["job_id"])
    assert job is not None
    assert job.metadata["speaker_count"] == 5


def test_api_ingest_unsupported_format_rejected(client: TestClient) -> None:
    dummy = io.BytesIO(b"not an audio file")
    response = client.post(
        "/ingest",
        data={"episode_title": "Bad File Ep"},
        files={"file": ("malicious.exe", dummy, "application/octet-stream")},
    )
    assert response.status_code == 422
    assert "unsupported audio format" in response.json()["detail"]


def test_api_ingest_unknown_job_404(client: TestClient) -> None:
    response = client.get("/ingest/nonexistent-id-000")
    assert response.status_code == 404


def test_api_ingest_sse_events_stream(client: TestClient, tmp_path: Path) -> None:
    job = manager.create_job(
        episode_id="sse_ep01",
        show_id="demo",
        title="SSE Test",
        audio_path=tmp_path / "audio.wav",
        work_dir=tmp_path / "work",
    )
    job.log("Starting job", "info")
    job.status = "completed"

    with client.stream("GET", f"/ingest/{job.job_id}/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        lines = [line for line in response.iter_lines() if line.strip()]
        assert any("data: " in line for line in lines)
        assert any("Starting job" in line for line in lines)
