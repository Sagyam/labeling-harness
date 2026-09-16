"""The ingestion pipeline stages: normalization, logging, analysis and the D46 discards."""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.llm.base import LlmRequestFailed
from app.llm.openrouter import OpenRouterClient
from app.llm.vertex import VertexClient
from app.models import AnnotationTask, Episode, LlmRequest, Segment
from app.services.analysis import analyze_transcript, mean_pairwise_disagreement
from app.services.ingest import (
    IngestJob,
    normalize_audio,
    run_pipeline,
)
from tests.ingest_support import _drain, _pipeline_job, make_test_audio

pytestmark = pytest.mark.db


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


def test_normalization_discards_content_above_nyquist_instead_of_folding_it(
    tmp_path: Path,
) -> None:
    """Dropping to 16 kHz must remove everything above 8 kHz, not fold it back below.

    Whatever survives the anti-alias filter reappears mirrored around 8 kHz. Speech carries real
    energy at 8-10 kHz -- that is where sibilants live -- so a leaky filter turns every /s/ into a
    burst of near-Nyquist noise, which is heard as a click.

    The probe is a low anchor tone plus noise confined entirely above 8 kHz, and it is measured
    against a control run of the anchor alone. Every bit of that noise has to be gone, so the two
    runs must agree: whatever the probe shows above the control is alias and nothing else.
    Comparing the two rather than asserting an absolute floor keeps the test honest about the
    leakage of its own window and of loudnorm's limiter.
    """
    sr = 48000
    n = sr * 4
    rng = np.random.default_rng(0)
    anchor = 0.1 * np.sin(2 * np.pi * 200 * np.arange(n) / sr)

    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spectrum[(freqs < 8200) | (freqs > 14000)] = 0.0
    out_of_band = np.fft.irfft(spectrum, n=n)
    out_of_band *= 0.05 / np.sqrt(np.mean(out_of_band**2))

    def alias_db(name: str, samples: np.ndarray) -> float:
        src = tmp_path / f"{name}.wav"
        out = tmp_path / f"{name}.flac"
        sf.write(str(src), samples.astype(np.float32), sr, format="WAV")
        normalize_audio(src, out)

        audio, out_sr = sf.read(str(out), dtype="float64")
        audio = audio[out_sr:-out_sr]  # drop loudnorm's ramp at either end
        power = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
        bins = np.fft.rfftfreq(len(audio), 1 / out_sr)
        anchor_power = power[(bins > 150) & (bins < 250)].sum()
        away_from_anchor = power[(bins > 1000) & (bins < 8000)].sum()
        return float(10 * np.log10(away_from_anchor / anchor_power))

    probe = alias_db("probe", anchor + out_of_band)
    control = alias_db("control", anchor)

    # The shipped default (`aresample=16000`) folded this noise down at about -35 dB against a
    # control near -85 dB: a 50 dB excess, and plainly audible. A correct filter leaves no gap.
    assert probe - control < 6.0, (
        f"content above 8 kHz folded back into the clip: probe {probe:.1f} dB vs "
        f"control {control:.1f} dB, an excess of {probe - control:.1f} dB"
    )


# --- Stage 3: OpenRouter Transcribe & Logging -------------------------------------------


def test_openrouter_transcribe_is_logged_to_llm_requests(
    db_session: Session, tmp_path: Path
) -> None:
    audio_path = tmp_path / "sample.flac"
    sf.write(str(audio_path), np.zeros(16000), 16000, format="FLAC")

    client = OpenRouterClient(db_session)
    result = client.transcribe(audio_path, route="asr_mai_transcribe_2", dry_run=True)

    assert result.text
    assert result.model == "microsoft/mai-transcribe-2"
    # Must be logged in db_session
    logged = db_session.scalars(sa.select(LlmRequest)).all()
    assert len(logged) >= 1
    req = logged[-1]
    assert req.route == "asr_mai_transcribe_2"
    assert req.status == "dry_run"


def test_vertex_transcribe_is_logged_to_llm_requests(db_session: Session, tmp_path: Path) -> None:
    audio_path = tmp_path / "test.flac"
    sf.write(str(audio_path), np.zeros(16000), 16000, format="FLAC")

    client = VertexClient(db_session)
    result = client.transcribe(audio_path, route="asr_gemini_flash", dry_run=True)

    assert result.text
    assert result.model == "gemini-3.8-flash"
    logged = db_session.scalars(sa.select(LlmRequest)).all()
    assert len(logged) >= 1
    req = logged[-1]
    assert req.route == "asr_gemini_flash"
    assert req.status == "dry_run"


# --- Stage 3: cross-system disagreement -------------------------------------------------


def test_two_identical_hypotheses_disagree_not_at_all() -> None:
    assert mean_pairwise_disagreement([["a", "b"], ["a", "b"]]) == 0.0


def test_one_hypothesis_cannot_disagree_with_anything() -> None:
    """The scorer reads a missing rate as 0.0; a lone system must produce the same value."""
    assert mean_pairwise_disagreement([["a", "b"]]) == 0.0
    assert mean_pairwise_disagreement([]) == 0.0


def test_disagreement_over_two_systems_is_the_single_comparison_between_them() -> None:
    """Adding a third transcriber must not change what two transcribers already scored."""
    pair = [["a", "b", "c"], ["a", "b", "d"]]
    expected = 1.0 - difflib.SequenceMatcher(None, pair[0], pair[1]).ratio()
    assert mean_pairwise_disagreement(pair) == round(expected, 4)


def test_a_third_hypothesis_informs_the_rate_rather_than_being_ignored() -> None:
    """Three systems means three pairs. An outlier should move the score, not vanish."""
    agreeing = mean_pairwise_disagreement([["a", "b"], ["a", "b"], ["a", "b"]])
    one_outlier = mean_pairwise_disagreement([["a", "b"], ["a", "b"], ["x", "y"]])
    assert agreeing == 0.0
    assert one_outlier > 0.0


def test_the_same_helper_gives_a_character_rate_from_raw_strings() -> None:
    """cer_between_hypotheses is the same comparison at character granularity."""
    assert mean_pairwise_disagreement(["hello", "hello"]) == 0.0
    assert mean_pairwise_disagreement(["hello", "world"]) > 0.0


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
    assert res.switch_point_count == 2
    assert res.discourse_marker_count == 0

    # With discourse marker
    text_with_dm = "So हामीले project meeting गर्नु पर्छ"
    res_dm = analyze_transcript(text_with_dm, duration_seconds=4.0)
    assert res_dm.switch_point_count == 3
    assert res_dm.discourse_marker_count == 1


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
    assert any("Stage 1/6" in log_item.message for log_item in job.logs)
    assert any("Stage 6/6" in log_item.message for log_item in job.logs)

    # Verify Episode created in database
    ep = db_session.scalar(sa.select(Episode).where(Episode.external_id == "web_ep001"))
    assert ep is not None
    assert ep.title == "Web Ingestion Test Episode"

    # Verify Segments created
    segments = db_session.scalars(sa.select(Segment).where(Segment.episode_id == ep.id)).all()
    assert len(segments) >= 1

    # The episode draws train or val at import and every clip starts in the train pot; gold is
    # chosen per clip by hand later (D71), so nothing blocks the queue and ingest builds it.
    assert ep.split in ("train", "val")
    assert {s.pot for s in segments} == {"train"}

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
        "app.services.ingest.pipeline.run_pipeline",
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
    _drain()

    work_root = settings.ingest.work_root.resolve()
    assert work_root in captured["work_dir"].resolve().parents


def test_concurrent_transcription_preserves_chronological_and_hypothesis_order(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    """Multi-threaded segment and route processing must preserve chronological order."""
    # 25 seconds of audio forces multiple bounded slices (max_seg=20.0s)
    sr = 16000
    t = np.linspace(0, 25.0, int(sr * 25.0), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    raw_audio = tmp_path / "multi_ep.wav"
    sf.write(str(raw_audio), audio, sr, format="WAV")

    work_dir = tmp_path / "work_multi"
    job = IngestJob(
        job_id="test-job-multi",
        episode_id="web_ep_multi",
        show_id="podcast",
        title="Concurrent Ingest Test",
        audio_path=raw_audio,
        work_dir=work_dir,
    )

    # settings.ingest.max_segment_concurrency is 4 by default
    run_pipeline(
        job,
        session_factory=lambda: db_session,
        storage=object_storage,
        settings=settings,
        keep_work_dir=True,
    )

    assert job.status == "completed"
    assert job.error is None

    # Verify segments.jsonl order
    segments_file = work_dir / "segments.jsonl"
    assert segments_file.is_file()
    import json

    with open(segments_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) >= 2
    # Verify strict chronological order of segments
    start_times = [r["start_time"] for r in records]
    assert start_times == sorted(start_times), "Segments must be in chronological start_time order"

    # Verify each segment's primary hypothesis is the first configured route (Scribe)
    for r in records:
        assert len(r["hypotheses"]) >= 1
        assert "scribe" in r["hypotheses"][0]["system_id"]


def test_concurrent_transcription_fails_job_cleanly_on_asr_error(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch
) -> None:
    """When every concurrent ASR worker raises, the job still ends cleanly through job.fail.

    Since D46 a raising worker discards its segment rather than the episode, so this only fails
    because the clip is the whole episode. The error names the systems that cost the run; the
    exception text itself lives on the discard record.
    """
    raw_audio = make_test_audio(tmp_path / "fail_audio.wav", duration_seconds=4.0)
    work_dir = tmp_path / "work_fail"

    job = IngestJob(
        job_id="test-job-fail",
        episode_id="web_ep_fail",
        show_id="podcast",
        title="Failing Ingest Test",
        audio_path=raw_audio,
        work_dir=work_dir,
    )

    def failing_transcribe(*args, **kwargs):
        raise RuntimeError("ASR upstream service unavailable")

    monkeypatch.setattr("app.services.ingest.pipeline.transcribe", failing_transcribe)

    run_pipeline(
        job,
        session_factory=lambda: db_session,
        storage=object_storage,
        settings=settings,
    )

    assert job.status == "failed"
    assert job.stage == "failed"
    assert "all 1 segments were discarded" in str(job.error)
    assert "gemini-3.8-flash" in str(job.error)
    assert len(job.discarded) == 1
    assert "ASR upstream service unavailable" in job.discarded[0].failures[0]["error"]


# --- D46: a refused clip costs the segment, never the episode ----------------------------


def _failing_transcribe(should_fail, *, route_to_fail: str = "asr_gemini_flash"):
    """Wrap the real ``transcribe`` so one route raises on the clips ``should_fail`` picks."""
    from app.services.ingest import pipeline as ingest_module

    real = ingest_module.transcribe

    def fake(session, audio_path, *, route, **kwargs):
        if route == route_to_fail and should_fail(Path(audio_path).name):
            raise LlmRequestFailed(f"Vertex transcription failed: finishReason=SAFETY ({route})")
        return real(session, audio_path, route=route, **kwargs)

    return fake


def test_a_refused_clip_is_discarded_and_the_episode_still_lands(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of D46.

    Stage 3 dispatches every ASR route at every clip, so one refusal near the end of a long
    episode used to throw away every paid transcript that had already succeeded.
    """
    job = _pipeline_job(tmp_path, "kept")
    monkeypatch.setattr(
        "app.services.ingest.pipeline.transcribe",
        _failing_transcribe(lambda name: name.endswith("_00000.flac")),
    )

    run_pipeline(job, session_factory=lambda: db_session, storage=object_storage, settings=settings)

    assert job.status == "completed", job.error
    assert len(job.discarded) == 1
    dropped = job.discarded[0]
    assert dropped.stage == "asr"
    assert dropped.systems == ["gemini-3.8-flash"], "the summary must name who cost the segment"
    assert "SAFETY" in dropped.failures[0]["error"]

    ep = db_session.scalar(sa.select(Episode).where(Episode.external_id == job.episode_id))
    assert ep is not None
    kept = db_session.scalars(sa.select(Segment).where(Segment.episode_id == ep.id)).all()
    assert kept, "the segments that transcribed cleanly are still imported"
    assert dropped.segment_id not in {s.external_id for s in kept}


def test_a_discarded_segment_is_reported_with_its_blame_in_the_summary(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _pipeline_job(tmp_path, "summary")
    monkeypatch.setattr(
        "app.services.ingest.pipeline.transcribe",
        _failing_transcribe(lambda name: name.endswith("_00000.flac")),
    )
    run_pipeline(job, session_factory=lambda: db_session, storage=object_storage, settings=settings)

    assert job.discard_summary() == {"gemini-3.8-flash": 1}
    assert any(
        "discarded" in log_item.message and log_item.level == "warn" for log_item in job.logs
    )
    completion = [log_item for log_item in job.logs if "finished successfully" in log_item.message]
    assert completion, "a run with discards still completes"


def test_an_episode_where_every_segment_is_refused_fails_loudly(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing survived, so there is no episode to import -- and no pretending otherwise."""
    job = _pipeline_job(tmp_path, "allgone", seconds=6.0)
    monkeypatch.setattr(
        "app.services.ingest.pipeline.transcribe", _failing_transcribe(lambda name: True)
    )
    run_pipeline(job, session_factory=lambda: db_session, storage=object_storage, settings=settings)

    assert job.status == "failed"
    assert "all" in (job.error or "") and "discarded" in (job.error or "")
    assert "gemini-3.8-flash" in (job.error or ""), "the error names the system that cost the run"


def test_the_progress_bar_still_reaches_every_segment_when_some_are_discarded(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discard bumps the same counter a success does, or the bar stalls short of the end."""
    job = _pipeline_job(tmp_path, "progress")
    monkeypatch.setattr(
        "app.services.ingest.pipeline.transcribe",
        _failing_transcribe(lambda name: name.endswith("_00000.flac")),
    )
    run_pipeline(job, session_factory=lambda: db_session, storage=object_storage, settings=settings)

    assert job.status == "completed", job.error
    assert job.active_segments == job.total_segments


def test_a_discard_is_emitted_to_subscribers_as_it_happens(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page shows losses while the run is going, not only in the final summary."""
    job = _pipeline_job(tmp_path, "sse")
    seen: list[dict] = []
    monkeypatch.setattr(
        "app.services.ingest.pipeline.transcribe",
        _failing_transcribe(lambda name: name.endswith("_00000.flac")),
    )
    monkeypatch.setattr(job, "_emit", lambda event: seen.append(event))

    run_pipeline(job, session_factory=lambda: db_session, storage=object_storage, settings=settings)

    discards = [e for e in seen if e.get("type") == "discard"]
    assert len(discards) == 1
    assert discards[0]["segment"]["failures"][0]["system_id"] == "gemini-3.8-flash"


def test_a_run_whose_every_clip_fails_still_keeps_the_failed_request_rows(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 6: a failed attempt is logged, and the log survives the episode failing.

    The rows used to be flushed and never committed, because only a clip that transcribed
    committed -- so an expired key left no trace of a single refused request.
    """
    job = _pipeline_job(tmp_path, "keylog", seconds=6.0)

    def refusing(session, audio_path, *, route, **kwargs):
        session.add(
            LlmRequest(
                route=route, model="m", request_hash="h", status="failed", error_message="401"
            )
        )
        session.flush()
        raise LlmRequestFailed("HTTP 401: key expired")

    monkeypatch.setattr("app.services.ingest.pipeline.transcribe", refusing)
    run_pipeline(job, session_factory=lambda: db_session, storage=object_storage, settings=settings)

    assert job.status == "failed"
    rows = db_session.scalars(sa.select(LlmRequest).where(LlmRequest.error_message == "401")).all()
    assert rows, "the refused requests are on record"
