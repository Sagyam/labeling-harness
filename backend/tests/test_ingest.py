"""Tests for the Phase 9 Web Ingestion Pipeline with Cloud ASR."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.llm.openrouter import OpenRouterClient
from app.models import AnnotationTask, Episode, LlmRequest, Segment
from app.services.analysis import analyze_transcript
from app.services.ingest import IngestJob, manager, normalize_audio, run_pipeline
from app.services.silero_vad import (
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
)

pytestmark = pytest.mark.db


def make_test_audio(path: Path, duration_seconds: float = 6.0, sample_rate: int = 16000) -> Path:
    """Generate a test audio file with alternating bursts of tone and silence."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # 440 Hz tone
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Insert 0.5s silence gap in the middle
    mid_start = int(2.5 * sample_rate)
    mid_end = int(3.0 * sample_rate)
    audio[mid_start:mid_end] = 0.0

    sf.write(str(path), audio, sample_rate, format="WAV")
    return path


# --- Stage 1: Normalization -------------------------------------------------------------


def test_normalize_audio_converts_to_16khz_mono_flac(tmp_path: Path) -> None:
    src_wav = make_test_audio(tmp_path / "raw.wav", duration_seconds=4.0)
    out_flac = tmp_path / "norm.flac"

    duration = normalize_audio(src_wav, out_flac)
    assert out_flac.is_file()
    assert duration > 3.5

    info = sf.info(str(out_flac))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.format == "FLAC"


# --- Stage 2: Silero VAD Segmentation ---------------------------------------------------


def test_silero_vad_detects_speech_and_slices_within_bounds(tmp_path: Path) -> None:
    vad = SileroVAD()
    sr = 16000
    # 25 seconds of audio to test splitting > 20.0s
    t = np.linspace(0, 25.0, int(sr * 25.0), endpoint=False)
    audio = 0.4 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    turns = vad.detect_turns(audio, sample_rate=sr)
    assert len(turns) >= 1

    slices = segment_audio_to_slices(turns, total_duration=25.0, min_seg=2.0, max_seg=20.0)
    assert len(slices) >= 2

    # Every slice must obey 2.0s <= duration <= 20.0s
    for start, end in slices:
        dur = end - start
        assert 2.0 <= dur <= 20.0, f"Slice duration {dur} out of bounds [2.0, 20.0]"

    # Test extracting clips to disk
    norm_flac = tmp_path / "test_norm.flac"
    sf.write(str(norm_flac), audio, sr, format="FLAC")

    clips_dir = tmp_path / "clips"
    segments = extract_clips(norm_flac, slices, "ep_test", clips_dir)
    assert len(segments) == len(slices)
    for seg in segments:
        assert seg.clip_path.is_file()
        assert seg.clip_checksum
        clip_info = sf.info(str(seg.clip_path))
        assert clip_info.samplerate == 16000
        assert clip_info.channels == 1


# --- Stage 3: OpenRouter Transcribe & Logging -------------------------------------------


def test_openrouter_transcribe_is_logged_to_llm_requests(
    db_session: Session, tmp_path: Path
) -> None:
    audio_path = tmp_path / "sample.flac"
    sf.write(str(audio_path), np.zeros(16000), 16000, format="FLAC")

    client = OpenRouterClient(db_session)
    result = client.transcribe(audio_path, route="asr")

    assert result.text
    assert result.model
    # Must be logged in db_session
    logged = db_session.scalars(sa.select(LlmRequest)).all()
    assert len(logged) >= 1
    req = logged[-1]
    assert req.route == "asr"
    assert req.status in ("dry_run", "succeeded")


# --- Stage 4: Token Analysis & Scoring --------------------------------------------------


def test_analyze_transcript_calculates_cmi_and_flags() -> None:
    # Code-switching: 3 Nepali words + 2 English words
    text = "हामीले project meeting गर्नु पर्छ"
    res = analyze_transcript(text, duration_seconds=3.5)

    assert res.token_count == 5
    assert res.devanagari_count == 3
    assert res.latin_count == 2
    assert res.cmi == 40.0  # 100 * (5 - 3) / 5
    assert res.code_switch_density == 0.4
    assert res.score.score > 0
    assert "code_switch_density" in res.score.components


# --- Stage 5: End-to-End Pipeline Execution ---------------------------------------------


def test_ingest_pipeline_end_to_end(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    raw_audio = make_test_audio(tmp_path / "episode_raw.wav", duration_seconds=6.0)
    work_dir = tmp_path / "work_ep01"

    job = IngestJob(
        job_id="test-job-001",
        episode_id="web_ep001",
        show_id="podcast",
        title="Web Ingestion Test Episode",
        audio_path=raw_audio,
        work_dir=work_dir,
    )

    run_pipeline(
        job,
        session_factory=lambda: db_session,
        storage=object_storage,
        settings=settings,
    )

    assert job.error is None, f"Job failed with error: {job.error}"
    assert job.status == "completed"
    assert job.stage == "complete"
    assert job.progress == 100.0
    assert job.active_segments >= 1
    assert any("Stage 1/5" in log_item.message for log_item in job.logs)
    assert any("Stage 5/5" in log_item.message for log_item in job.logs)

    # Verify Episode created in database
    ep = db_session.scalar(sa.select(Episode).where(Episode.external_id == "web_ep001"))
    assert ep is not None
    assert ep.title == "Web Ingestion Test Episode"

    # Verify Segments created
    segments = db_session.scalars(sa.select(Segment).where(Segment.episode_id == ep.id)).all()
    assert len(segments) >= 1

    # Verify AnnotationTasks created and queued
    tasks = db_session.scalars(
        sa.select(AnnotationTask).where(AnnotationTask.segment_id.in_([s.id for s in segments]))
    ).all()
    assert len(tasks) == len(segments)
    for t in tasks:
        assert t.queue in ("review", "audit", "error")
        assert t.status == "pending"
        assert t.priority_score is not None

    # Verify LlmRequests logged
    llm_logs = db_session.scalars(sa.select(LlmRequest)).all()
    assert len(llm_logs) >= len(segments)


# --- API Endpoints: Ingestion Service ----------------------------------------------------


def test_api_ingest_start_and_status(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = make_test_audio(tmp_path / "upload.wav", duration_seconds=3.0)
    monkeypatch.setattr(
        "app.api.ingest.run_pipeline",
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
