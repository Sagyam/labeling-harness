"""Tests for the Phase 9 Web Ingestion Pipeline with Cloud ASR."""

from __future__ import annotations

import difflib
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
from app.services.ingest import (
    IngestJob,
    _mean_pairwise_disagreement,
    manager,
    normalize_audio,
    run_pipeline,
)
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
    result = client.transcribe(audio_path, route="asr_whisper_large_v3", dry_run=True)

    assert result.text
    assert result.model
    # Must be logged in db_session
    logged = db_session.scalars(sa.select(LlmRequest)).all()
    assert len(logged) >= 1
    req = logged[-1]
    assert req.route == "asr_whisper_large_v3"
    assert req.status == "dry_run"


# --- Stage 3: cross-system disagreement -------------------------------------------------


def test_two_identical_hypotheses_disagree_not_at_all() -> None:
    assert _mean_pairwise_disagreement([["a", "b"], ["a", "b"]]) == 0.0


def test_one_hypothesis_cannot_disagree_with_anything() -> None:
    """The scorer reads a missing rate as 0.0; a lone system must produce the same value."""
    assert _mean_pairwise_disagreement([["a", "b"]]) == 0.0
    assert _mean_pairwise_disagreement([]) == 0.0


def test_disagreement_over_two_systems_is_the_single_comparison_between_them() -> None:
    """Adding a third transcriber must not change what two transcribers already scored."""
    pair = [["a", "b", "c"], ["a", "b", "d"]]
    expected = 1.0 - difflib.SequenceMatcher(None, pair[0], pair[1]).ratio()
    assert _mean_pairwise_disagreement(pair) == round(expected, 4)


def test_a_third_hypothesis_informs_the_rate_rather_than_being_ignored() -> None:
    """Three systems means three pairs. An outlier should move the score, not vanish."""
    agreeing = _mean_pairwise_disagreement([["a", "b"], ["a", "b"], ["a", "b"]])
    one_outlier = _mean_pairwise_disagreement([["a", "b"], ["a", "b"], ["x", "y"]])
    assert agreeing == 0.0
    assert one_outlier > 0.0


def test_the_same_helper_gives_a_character_rate_from_raw_strings() -> None:
    """cer_between_hypotheses is the same comparison at character granularity."""
    assert _mean_pairwise_disagreement(["hello", "hello"]) == 0.0
    assert _mean_pairwise_disagreement(["hello", "world"]) > 0.0


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


# --- Stage 2: the energy VAD fallback ----------------------------------------------------
#
# This is the path taken whenever onnxruntime or the ONNX file is absent, which is the most
# likely way a deployment differs from a development machine.


def energy_vad(tmp_path: Path) -> SileroVAD:
    """A VAD with no model, so detection falls back to the energy envelope."""
    vad = SileroVAD(model_path=tmp_path / "absent.onnx")
    assert vad._session is None
    return vad


def test_energy_fallback_finds_speech_either_side_of_a_silence(tmp_path: Path) -> None:
    vad = energy_vad(tmp_path)
    sample_rate = 16000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2 * sample_rate) / sample_rate)
    silence = np.zeros(sample_rate, dtype=np.float64)
    audio = np.concatenate([tone, silence, tone]).astype(np.float32)

    turns = vad.detect_turns(audio, sample_rate=sample_rate)

    assert len(turns) >= 2
    assert turns[0].start == pytest.approx(0.0, abs=0.1)
    for turn in turns:
        assert turn.end > turn.start


def test_energy_fallback_returns_one_turn_for_continuous_speech(tmp_path: Path) -> None:
    vad = energy_vad(tmp_path)
    sample_rate = 16000
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(4 * sample_rate) / sample_rate)).astype(
        np.float32
    )

    turns = vad.detect_turns(audio, sample_rate=sample_rate)

    assert len(turns) == 1
    assert turns[0].end - turns[0].start > 3.0


def test_energy_fallback_on_silence_still_yields_the_whole_clip(tmp_path: Path) -> None:
    """Silence must not swallow a segment: the clip is handed on for a human to judge."""
    vad = energy_vad(tmp_path)
    turns = vad.detect_turns(np.zeros(16000 * 4, dtype=np.float32), sample_rate=16000)

    assert len(turns) == 1
    assert turns[0].start == 0.0


def test_energy_fallback_on_audio_shorter_than_one_frame_finds_nothing(tmp_path: Path) -> None:
    vad = energy_vad(tmp_path)
    assert vad.detect_turns(np.zeros(100, dtype=np.float32), sample_rate=16000) == []


def test_a_missing_model_file_is_reported_and_does_not_raise(tmp_path: Path) -> None:
    assert SileroVAD(model_path=tmp_path / "nope.onnx")._session is None


# --- run_pipeline housekeeping -----------------------------------------------------------


def test_the_work_directory_is_removed_when_the_job_finishes(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    """The scratch dir holds the upload, the normalized FLAC and every clip -- all stored
    elsewhere by the time the run ends, so leaving it behind just grows the disk."""
    raw_audio = make_test_audio(tmp_path / "cleanup_raw.wav", duration_seconds=6.0)
    work_dir = tmp_path / "work_cleanup"
    work_dir.mkdir()

    job = IngestJob(
        job_id="test-cleanup",
        episode_id="web_cleanup",
        show_id="podcast",
        title="Cleanup",
        audio_path=raw_audio,
        work_dir=work_dir,
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)

    assert job.status == "completed"
    assert not work_dir.exists()


def test_the_work_directory_is_removed_even_when_a_stage_fails(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    work_dir = tmp_path / "work_failed"
    work_dir.mkdir()
    (work_dir / "leftover.txt").write_text("x")

    job = IngestJob(
        job_id="test-cleanup-fail",
        episode_id="web_fail",
        show_id="podcast",
        title="Fails at stage 1",
        audio_path=tmp_path / "does_not_exist.wav",
        work_dir=work_dir,
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)

    assert job.status == "failed"
    assert not work_dir.exists()


def test_the_work_directory_can_be_kept_for_debugging(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    work_dir = tmp_path / "work_kept"
    work_dir.mkdir()

    job = IngestJob(
        job_id="test-keep",
        episode_id="web_keep",
        show_id="podcast",
        title="Kept",
        audio_path=tmp_path / "does_not_exist.wav",
        work_dir=work_dir,
    )
    run_pipeline(job, lambda: db_session, object_storage, settings, keep_work_dir=True)

    assert work_dir.exists()


def test_run_pipeline_builds_its_own_storage_when_none_is_given(
    db_session: Session, settings, tmp_path: Path
) -> None:
    """The default used to call the cached zero-argument get_storage with a settings object."""
    job = IngestJob(
        job_id="test-default-storage",
        episode_id="web_default",
        show_id="podcast",
        title="Default storage",
        audio_path=tmp_path / "does_not_exist.wav",
        work_dir=tmp_path / "work_default",
    )
    run_pipeline(job, lambda: db_session, None, settings)

    # Stage 1 fails on the missing audio, but only after storage resolved without a TypeError.
    assert job.status == "failed"
    assert "Stage 1" in job.error


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


def test_mock_transcripts_are_stored_under_a_system_id_that_names_them(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    """The suite runs with HARNESS_LLM__DRY_RUN=true, so every hypothesis here is canned text.

    It must be impossible to mistake that for model output later, in the queue or at export.
    """
    from app.models import AsrSystem

    raw_audio = make_test_audio(tmp_path / "mock_raw.wav", duration_seconds=6.0)
    job = IngestJob(
        job_id="test-mock-naming",
        episode_id="web_mock",
        show_id="podcast",
        title="Mock naming",
        audio_path=raw_audio,
        work_dir=tmp_path / "work_mock",
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)

    assert job.status == "completed"
    systems = db_session.scalars(sa.select(AsrSystem.system_id)).all()
    assert systems, "the run produced no ASR system at all"
    assert all(name.startswith("mock-") for name in systems), systems


def test_a_traversing_episode_id_cannot_escape_the_work_root(
    client: TestClient, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id names a directory the pipeline later deletes, so it must be sanitised."""
    captured: dict[str, Path] = {}
    monkeypatch.setattr(
        "app.api.ingest.run_pipeline",
        lambda job, *args: captured.update(work_dir=job.work_dir),
    )
    wav_path = make_test_audio(tmp_path / "traverse.wav", duration_seconds=3.0)

    response = client.post(
        "/ingest",
        files={"file": ("traverse.wav", wav_path.read_bytes(), "audio/wav")},
        data={"episode_title": "Traversal", "episode_id": "../../../etc/passwd"},
    )
    assert response.status_code == 202
    assert ".." not in response.json()["episode_id"]

    work_root = settings.ingest.work_root.resolve()
    assert work_root in captured["work_dir"].resolve().parents


# --- Silero VAD: the model's input contract ----------------------------------------------
#
# The ONNX input length is dynamic, so feeding the wrong window size is accepted silently and
# the model then returns a near-zero probability for everything. Nothing errors; the VAD just
# stops detecting speech, and the cutter falls back to slicing on a fixed grid.


class RecordingSession:
    """Stands in for the ONNX session, capturing the windows it is handed."""

    def __init__(self, prob: float = 0.9) -> None:
        self.windows: list[np.ndarray] = []
        self.prob = prob

    def run(self, _outputs, inputs):
        self.windows.append(inputs["input"])
        # The real session returns (output, stateN); output is shaped (batch, 1).
        return np.array([[self.prob]], dtype=np.float32), inputs["state"]


def vad_with(session) -> SileroVAD:
    vad = SileroVAD(model_path=Path("/nonexistent.onnx"))
    vad._session = session
    return vad


def test_the_model_is_fed_context_plus_chunk_not_a_bare_chunk() -> None:
    from app.services.silero_vad import CHUNK_SIZE, CONTEXT_SIZE

    session = RecordingSession()
    vad_with(session).detect_turns(np.zeros(CHUNK_SIZE * 6, dtype=np.float32), sample_rate=16000)

    assert session.windows, "the model was never called"
    for window in session.windows:
        assert window.shape == (1, CONTEXT_SIZE + CHUNK_SIZE), window.shape


def test_each_window_carries_the_previous_chunk_as_its_context() -> None:
    from app.services.silero_vad import CHUNK_SIZE, CONTEXT_SIZE

    audio = np.arange(CHUNK_SIZE * 3, dtype=np.float32)
    session = RecordingSession()
    vad_with(session).detect_turns(audio, sample_rate=16000)

    # the first window has no history and is zero-padded
    assert np.array_equal(session.windows[0][0, :CONTEXT_SIZE], np.zeros(CONTEXT_SIZE))
    # every later window opens with the tail of the chunk before it
    for i in range(1, len(session.windows)):
        previous_chunk = audio[(i - 1) * CHUNK_SIZE : i * CHUNK_SIZE]
        assert np.array_equal(session.windows[i][0, :CONTEXT_SIZE], previous_chunk[-CONTEXT_SIZE:])
        assert np.array_equal(
            session.windows[i][0, CONTEXT_SIZE:], audio[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        )


def test_an_eight_kilohertz_stream_uses_the_smaller_window() -> None:
    from app.services.silero_vad import window_sizes

    assert window_sizes(16000) == (512, 64)
    assert window_sizes(8000) == (256, 32)


def test_the_real_model_separates_speech_from_digital_silence() -> None:
    """The end-to-end guard: a working VAD must not score silence the same as everything else."""
    vad = SileroVAD()
    if vad._session is None:
        pytest.skip("silero_vad.onnx is not present")

    # Read the probabilities directly: silence must sit well below the 0.5 threshold. Before
    # the context fix this was ~0.0005 for silence *and* for speech, which is the whole bug.
    from app.services.silero_vad import CHUNK_SIZE, CONTEXT_SIZE

    silence = np.zeros(16000 * 3, dtype=np.float32)

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    probs = []
    for i in range(len(silence) // CHUNK_SIZE):
        chunk = silence[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        window = np.concatenate((context, chunk))[np.newaxis, :]
        out, state = vad._session.run(
            None, {"input": window, "state": state, "sr": np.array(16000, dtype=np.int64)}
        )
        probs.append(float(out[0][0]))
        context = chunk[-CONTEXT_SIZE:]
    assert max(probs) < 0.5


def test_no_detected_speech_is_reported_rather_than_passed_off_as_one_turn() -> None:
    """Falling back to 'the whole file is one turn' is what hid a dead VAD for months."""
    session = RecordingSession(prob=0.0)
    turns = vad_with(session).detect_turns(np.zeros(16000 * 5, dtype=np.float32), 16000)

    assert len(turns) == 1
    assert turns[0].start == 0.0
    assert turns[0].end == pytest.approx(5.0, abs=0.1)


# --- clip edges must not click -----------------------------------------------------------


def test_clip_edges_are_faded_so_a_cut_cannot_click(tmp_path: Path) -> None:
    """A cut lands mid-waveform; starting or stopping on a non-zero sample is heard as a click."""
    sr = 16000
    # a loud constant-amplitude tone: every cut point is far from a zero crossing
    t = np.arange(sr * 6) / sr
    sf.write(str(tmp_path / "src.flac"), 0.8 * np.sin(2 * np.pi * 220 * t), sr, format="FLAC")

    segments = extract_clips(tmp_path / "src.flac", [(1.0, 4.0)], "ep", tmp_path / "clips")
    clip, _ = sf.read(str(segments[0].clip_path), dtype="float32")

    assert abs(clip[0]) < 0.01, f"clip starts at {clip[0]:+.4f}, which clicks"
    assert abs(clip[-1]) < 0.01, f"clip ends at {clip[-1]:+.4f}, which clicks"
    # the body of the clip is untouched
    assert np.abs(clip[sr // 2 : -sr // 2]).max() > 0.7


def test_the_fade_leaves_a_short_clip_alone(tmp_path: Path) -> None:
    from app.services.silero_vad import apply_edge_fade

    tiny = np.ones(4, dtype=np.float32)
    assert np.array_equal(apply_edge_fade(tiny, 16000), tiny)
