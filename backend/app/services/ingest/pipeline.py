"""The six-stage run itself: normalize, segment, transcribe, fuse, analyze, import.

Pure functions over one :class:`IngestJob` -- no global state beyond the manager that schedules
them -- so the pipeline is a wiring change away from a job queue (AGENTS.md, "How to work").
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime as dt
import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from sqlalchemy.orm import Session

from app.config import Settings, get_settings, load_llm_routes
from app.llm.base import AsrResult
from app.llm.transcription import (
    ASR_PROMPT,
    asr_route_names,
    disagreement_excluded_system_ids,
    system_id_for,
    transcribe,
)
from app.services.acoustics import AcousticMeter
from app.services.analysis import analyze_transcript, mean_pairwise_disagreement
from app.services.diarization import declared_speaker_count, diarize_audio
from app.services.diarization_import import import_diarization
from app.services.forced_align import ForcedAligner, align_text
from app.services.importer import import_manifest
from app.services.overlap import OverlapDetector, spans_within
from app.services.queue_builder import build_queue
from app.services.silero_vad import (
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
    speech_spans_within,
)
from app.services.speaker_meta import strip_speaker_pii
from app.services.voices import link_voices
from app.services.youtube import (
    YouTubeBotDetected,
    YouTubeError,
    download_audio,
    is_bot_detection_error,
)
from app.storage import build_storage
from app.storage.base import ObjectStorage
from app.utils.hashing import sha256_file
from app.utils.logging import get_logger

from .audio import normalize_audio
from .fusion import _classify_episode_topic, _run_fusion_stage
from .job import DiscardedSegment, IngestJob, LockedSession

logger = get_logger(__name__)


def run_pipeline(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage | None = None,
    settings: Settings | None = None,
    *,
    keep_work_dir: bool = False,
    overlap_detector: OverlapDetector | None = None,
    acoustic_meter: AcousticMeter | None = None,
) -> None:
    """Run all 5 stages synchronously inside background worker thread.

    Args:
        job: The job to run, carrying its own progress and log state.
        session_factory: Produces database sessions; the worker owns its own.
        storage: Object storage for clips and peaks. Defaults to the configured backend.
        settings: Configuration override.
        keep_work_dir: Retain the scratch directory after the run, for debugging. It holds the
            uploaded source, the normalized FLAC and every extracted clip, all of which are
            already persisted elsewhere by the time the run finishes.
        overlap_detector: Finds overlapped speech in the episode (D77). Defaults to the ONNX
            detector, which fetches its model on first use; without one, clips go unmeasured.
        acoustic_meter: Measures each clip's bandwidth, SNR and reverb (D87). Defaults to
            :meth:`AcousticMeter.default`.
    """
    settings = settings or get_settings()
    storage = storage or build_storage(settings)

    try:
        if job.audio_path is None and not _fetch_source_audio(job, settings):
            return
        _run_stages(job, session_factory, storage, settings, overlap_detector, acoustic_meter)
    finally:
        if not keep_work_dir:
            shutil.rmtree(job.work_dir, ignore_errors=True)


def _note_backlog(job: IngestJob) -> None:
    """Register a backlogged run with the manager, so the queue and retry-all can see it.

    The manager is imported inside the function because manager.py imports this module's
    ``run_pipeline`` at module level.
    """
    from .manager import manager

    with manager._lock:
        if job.job_id not in manager._backlog:
            manager._backlog.append(job.job_id)


def _fetch_source_audio(job: IngestJob, settings: Settings) -> bool:
    """Download the job's source audio, for a job started from a URL instead of an upload.

    This occupies the same slot an upload does -- it is how the source file arrives, not a sixth
    pipeline stage -- so it reports under the ``downloading`` stage and leaves the five stages
    downstream untouched. Progress is logged per decile rather than per line: yt-dlp emits
    hundreds of them and each one is an SSE frame.

    Returns:
        True when the audio is in place and the pipeline may continue.
    """
    if not job.source_url:
        job.fail("Job has neither an uploaded file nor a source URL")
        return False

    job.status = "processing"
    job.set_progress("downloading", 0.0)
    job.log(f"Fetching audio from {job.source_url} (yt-dlp)...")

    last_decile = -1

    def report(percent: float, line: str) -> None:
        nonlocal last_decile
        decile = int(percent // 10)
        if decile != last_decile:
            last_decile = decile
            job.log(line)
        # The download shares the progress bar with the pipeline it precedes, so it fills the
        # slice ahead of stage 1 rather than the whole bar.
        job.set_progress("downloading", percent * 0.05)

    try:
        job.audio_path = download_audio(
            job.source_url, job.work_dir, settings=settings, on_progress=report
        )
    except YouTubeBotDetected as exc:
        job.backlog(
            reason="YouTube bot detection / rate limit challenge",
            error_message=str(exc),
        )
        _note_backlog(job)
        return False
    except YouTubeError as exc:
        if is_bot_detection_error(str(exc)):
            job.backlog(
                reason="YouTube bot detection / rate limit challenge",
                error_message=str(exc),
            )
            _note_backlog(job)
            return False
        job.fail(f"Audio download failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive; the module raises YouTubeError
        if is_bot_detection_error(str(exc)):
            job.backlog(
                reason="YouTube bot detection / rate limit challenge",
                error_message=str(exc),
            )
            _note_backlog(job)
            return False
        job.fail(f"Audio download failed: {exc}")
        return False

    size_mb = job.audio_path.stat().st_size / (1024 * 1024)
    job.log(f"Downloaded {job.audio_path.name} ({size_mb:.1f} MB)", "success")
    return True


def _halted(job: IngestJob, *, segments_detected: int = 0, segments_transcribed: int = 0) -> bool:
    """True when AZ-5 has been hit, having already aborted the run.

    Called at every stage boundary, so a scram stops the pipeline at the next seam rather than
    wherever the flag happened to be noticed.
    """
    if not job.scrammed:
        return False
    job.abort(
        {
            "episode_id": job.episode_id,
            "reason": job.scram_reason,
            "stage_reached": job.stage,
            "segments_detected": segments_detected,
            "segments_transcribed": segments_transcribed,
            "segments_discarded": len(job.discarded),
            "discarded_by_system": job.discard_summary(),
            "imported": False,
        }
    )
    return True


def _import_speaker_turns(
    job: IngestJob,
    session: Session,
    future: concurrent.futures.Future[dict[str, Any] | None],
    settings: Settings,
) -> None:
    """Store the remote diarizer's answer as the episode's first diarization run (D79).

    Like overlap detection, a heads-up rather than a stage: a service that failed or is still
    running after the timeout is logged, and the episode completes with uncoloured words. The
    savepoint keeps a failed import from taking the manifest import down with it.
    """
    try:
        result = future.result(timeout=settings.diarization.timeout_seconds)
        if not result:
            return
        with session.begin_nested():
            report = import_diarization(
                session,
                {job.episode_id: result},
                model=settings.diarization.model,
                source=settings.diarization.endpoint_url,
                actor="ingest",
            )
            # The new run's speakers joined to the voices of every episode before it (D87).
            linked = link_voices(session, actor="ingest")
    except Exception as exc:
        job.log(f"Diarization skipped: {type(exc).__name__}: {exc}", "warn")
        return
    job.log(
        f"Diarization: {report.turns_inserted} speaker turns, "
        f"{len(result.get('labels') or [])} speakers; {linked.voices} voices in the corpus"
    )


def _detect_overlap(
    job: IngestJob,
    detector: OverlapDetector | None,
    audio: Any,
    sample_rate: int,
) -> list[tuple[float, float]] | None:
    """Overlapped speech over the whole episode, or ``None`` when it could not be measured.

    A heads-up for the annotator, not a stage the episode depends on (D77): a missing model or a
    failure is logged and the clips simply go unmeasured.
    """
    try:
        detector = detector if detector is not None else OverlapDetector()
        spans = detector.detect(audio, sample_rate)
    except Exception as exc:
        job.log(f"Overlap detection skipped: {type(exc).__name__}: {exc}")
        return None
    if spans is None:
        job.log("Overlap detection skipped: no model available")
        return None
    seconds = sum(end - start for start, end in spans)
    job.log(f"Detected {seconds:.1f}s of overlapped speech in {len(spans)} stretches")
    return spans


def _measure_acoustics(
    job: IngestJob,
    meter: AcousticMeter | None,
    audio: Any,
    sample_rate: int,
    segments: list[Any],
    turns: Any,
) -> dict[str, dict[str, Any]]:
    """Each clip's acoustic measurements, keyed by segment id; empty when they failed (D87).

    Like overlap, a covariate rather than a stage: a failure is logged and the clips go
    unmeasured, which the backfill can mend later.
    """
    try:
        meter = meter if meter is not None else AcousticMeter.default()
        results = meter.measure(
            audio,
            sample_rate,
            [
                (
                    seg.start_time,
                    seg.end_time,
                    speech_spans_within(turns, seg.start_time, seg.end_time),
                )
                for seg in segments
            ],
        )
    except Exception as exc:
        job.log(f"Acoustic measurement skipped: {type(exc).__name__}: {exc}")
        return {}
    return {seg.segment_id: result for seg, result in zip(segments, results, strict=True)}


def _run_stages(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage,
    settings: Settings,
    overlap_detector: OverlapDetector | None = None,
    acoustic_meter: AcousticMeter | None = None,
) -> None:
    """The five pipeline stages. Every failure is reported through ``job.fail`` and returns.

    Between stages, and before each segment's inference, the run checks whether AZ-5 has been
    hit (:meth:`IngestJob.scram`) and stops there rather than at the end.
    """
    job.status = "processing"
    job.log(f"Starting ingestion for '{job.title}' ({job.episode_id})")

    if job.audio_path is None:
        job.fail("Stage 1 Audio Normalization failed: no source audio for this job")
        return

    norm_flac = job.work_dir / f"{job.episode_id}_normalized.flac"

    # Stage 1: Normalize Audio
    try:
        job.set_progress("normalizing", 5.0)
        job.log("Stage 1/6: Normalizing audio (FFmpeg loudnorm, 16 kHz mono FLAC)...")
        duration = normalize_audio(job.audio_path, norm_flac)
        source_checksum = sha256_file(job.audio_path)
        job.log(f"Audio normalized: {duration:.1f}s ({duration / 60:.1f} min)")
        job.set_progress("normalizing", 20.0)
    except Exception as exc:
        job.fail(f"Stage 1 Audio Normalization failed: {exc}")
        return

    if _halted(job):
        return

    # Speaker turns come from a GPU diarizer over the whole episode (D79). It runs remotely while
    # the stages below transcribe, and is collected at import; a failure costs the colours only.
    diarize_future: concurrent.futures.Future[dict[str, Any] | None] | None = None
    if settings.diarization.enabled and settings.diarization.endpoint_url:
        num_speakers = declared_speaker_count(job.metadata)
        job.log(f"Diarizing the episode remotely (speakers: {num_speakers or 'auto'})...")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        diarize_future = executor.submit(
            diarize_audio, norm_flac, num_speakers=num_speakers, settings=settings
        )
        # The submitted call still runs to completion; this only lets its thread go when it does.
        executor.shutdown(wait=False)

    # Stage 2: Silero VAD Segmentation
    try:
        job.set_progress("segmenting", 22.0)
        job.log("Stage 2/6: Detecting speech turns via Silero VAD (2.0s - 20.0s bounds)...")
        vad = SileroVAD()
        audio_data, sr = sf.read(str(norm_flac), dtype="float32")
        turns = vad.detect_turns(audio_data, sample_rate=sr)
        job.log(f"Detected {len(turns)} raw speech turns")

        slices = segment_audio_to_slices(turns, duration, audio=audio_data, sample_rate=sr)
        job.log(f"Partitioned into {len(slices)} bounded utterances (2.0s - 20.0s)")

        overlap = _detect_overlap(job, overlap_detector, audio_data, sr)

        clips_dir = job.work_dir / "clips"
        segments = extract_clips(
            norm_flac, slices, job.episode_id, clips_dir, max_workers=settings.ingest.cpu_workers
        )
        job.total_segments = len(segments)
        job.active_segments = len(segments)
        job.log(f"Extracted {len(segments)} audio clips to disk")
        acoustics = _measure_acoustics(job, acoustic_meter, audio_data, sr, segments, turns)
        job.set_progress(
            "segmenting", 40.0, total_segments=len(segments), active_segments=len(segments)
        )
    except Exception as exc:
        job.fail(f"Stage 2 Silero VAD Segmentation failed: {exc}")
        return

    if _halted(job, segments_detected=len(segments)):
        return

    # Stage 3 & 4: Cloud ASR & Token Analysis
    segment_records: list[dict[str, Any]] = []
    try:
        job.set_progress("transcribing", 42.0)
        routes = load_llm_routes()
        asr_routes = asr_route_names(routes) or ["asr"]
        systems = ", ".join(system_id_for(r, routes.routes.get(r)) for r in asr_routes)
        job.log(
            f"Stage 3/6: Cloud ASR inference for {len(segments)} segments "
            f"across {len(asr_routes)} systems ({systems})..."
        )

        # One aligner per run: the ONNX session is loaded once and its Run() is thread-safe,
        # so every segment worker shares it. Absent model means no word spans, not a failure.
        aligning_routes = {
            r for r in asr_routes if getattr(routes.routes.get(r), "forced_align", False)
        }
        # The fusion stage aligns the fused text too, so the aligner is loaded whenever a real
        # run is going to fuse, not only when a recogniser needs it.
        wants_aligner = bool(aligning_routes) or bool(settings.fusion.route)
        aligner = ForcedAligner() if wants_aligner and not routes.dry_run else None
        if aligner is not None and not aligner.available:
            job.log("Forced aligner model not available -- word spans will be skipped.")

        with session_factory() as raw_session:
            locked_session = LockedSession(raw_session)
            progress_lock = threading.Lock()
            completed_count = 0
            records_by_idx: list[dict[str, Any] | None] = [None] * len(segments)

            # Persistent HTTP client for connection pooling across all concurrent requests
            with httpx.Client(timeout=routes.default_timeout_seconds) as http_client:

                def _bump_progress() -> None:
                    """Advance the bar by one segment, however that segment ended."""
                    nonlocal completed_count
                    completed_count += 1
                    step_progress = 40.0 + (35.0 * completed_count / len(segments))
                    job.set_progress("transcribing", step_progress, active_segments=completed_count)

                def _process_segment(seg_idx: int, seg: Any) -> dict[str, Any] | None:
                    # AZ-5 checkpoint, and the one that matters: every route of every segment is
                    # dispatched from below this line, so a scram noticed here is inference that
                    # is never billed. A segment stopped this way is not a discard -- nothing
                    # failed on it and no system is to blame for it (D46).
                    if job.scrammed:
                        return None

                    # Rec 1: Concurrent model dispatch per segment across every ASR route
                    clip_results: dict[str, AsrResult] = {}
                    route_failures: list[dict[str, str]] = []
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max(1, len(asr_routes))
                    ) as route_pool:
                        future_to_route = {
                            route_pool.submit(
                                transcribe,
                                locked_session,
                                seg.clip_path,
                                route=r_name,
                                config=routes,
                                prompt=ASR_PROMPT,
                                client=http_client,
                            ): r_name
                            for r_name in asr_routes
                        }
                        for f in concurrent.futures.as_completed(future_to_route):
                            r_name = future_to_route[f]
                            try:
                                clip_results[r_name] = f.result()
                            except Exception as exc:  # recorded, then the segment is discarded
                                route_failures.append(
                                    {
                                        "route": r_name,
                                        "system_id": system_id_for(
                                            r_name, routes.routes.get(r_name)
                                        ),
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )

                    # One clip short of a full set is discarded, not patched (D46). Every system
                    # must speak for every segment: `word_disagreement_rate` is a mean over the
                    # pairs present and carries 0.40 of the priority score, so a segment scored
                    # from three systems where its neighbours used four is not a cheaper segment,
                    # it is a differently-measured one -- and nothing downstream would ever say so.
                    if route_failures:
                        with progress_lock:
                            job.discard(
                                DiscardedSegment(
                                    segment_id=seg.segment_id,
                                    start_time=seg.start_time,
                                    end_time=seg.end_time,
                                    stage="asr",
                                    failures=route_failures,
                                )
                            )
                            _bump_progress()
                        # The clip is dead: nothing will reference it, and leaving it in the work
                        # directory would put it in the manifest's clip upload by accident.
                        with contextlib.suppress(OSError):
                            Path(seg.clip_path).unlink()
                        return None

                    # Combine results in configured hypothesis order (asr_routes order)
                    results: list[AsrResult] = [clip_results[r_name] for r_name in asr_routes]

                    # Preserve configured hypothesis order (results[0] is primary)
                    hypotheses: list[dict[str, Any]] = []
                    for r_idx, asr_res in enumerate(results):
                        route_name = asr_routes[r_idx]
                        sys_name = system_id_for(route_name, routes.routes.get(route_name))
                        if asr_res.dry_run:
                            # A dry run returns canned text. Name the system so it can never be
                            # mistaken for real model output in the queue or at export.
                            sys_name = f"mock-{sys_name}"

                        # A transcriber that reports its own timings keeps them; one that does
                        # not gets them measured locally against the clip (D32). Never on a dry
                        # run: there is no real speech behind canned text to align it to.
                        words = asr_res.words
                        if (
                            words is None
                            and aligner is not None
                            and route_name in aligning_routes
                            and not asr_res.dry_run
                        ):
                            words = align_text(aligner, seg.clip_path, asr_res.text)

                        hypothesis = {
                            "system_id": sys_name,
                            "model_id": asr_res.model,
                            "text": asr_res.text,
                            "avg_logprob": asr_res.avg_logprob,
                            "no_speech_prob": asr_res.no_speech_prob,
                            "words": words,
                        }
                        # Provenance, not a hypothesis: the importer routes every key it does
                        # not recognise into `metadata_jsonb`, so this never reaches `text_raw`,
                        # the disagreement comparison or the analysis (D41).
                        if asr_res.metadata:
                            hypothesis.update(asr_res.metadata)
                        hypotheses.append(hypothesis)

                    # Cross-system disagreement, averaged over every pair of systems. With two
                    # systems this is the single comparison between them; with three it is the mean
                    # of the three pairs, so a third hypothesis informs the queue rather than being
                    # paid for and ignored.
                    #
                    # A route flagged `exclude_from_disagreement` is held out (D39). It is still
                    # stored and exported; it just does not vote, because its disagreement is an
                    # orthography artefact rather than evidence that anything was misheard.
                    held_out = disagreement_excluded_system_ids(routes)
                    texts = [
                        h["text"]
                        for h in hypotheses
                        if h["system_id"].removeprefix("mock-") not in held_out
                    ]
                    word_disagreement_rate = mean_pairwise_disagreement([t.split() for t in texts])
                    cer_between_hyps = mean_pairwise_disagreement(texts)

                    primary_hyp = hypotheses[0]
                    analysis = analyze_transcript(
                        primary_hyp["text"],
                        duration_seconds=seg.duration,
                        no_speech_prob=primary_hyp["no_speech_prob"],
                        settings=settings,
                    )

                    # Commit per segment (Decision D20). Thread-safe under LockedSession.
                    locked_session.commit()

                    # Progress & logging under lock
                    with progress_lock:
                        _bump_progress()

                        if completed_count % 5 == 0 or completed_count == len(segments):
                            snippet = primary_hyp["text"][:30]
                            models_str = ", ".join(h["system_id"] for h in hypotheses)
                            prefix = f"[{completed_count}/{len(segments)}] {seg.segment_id}"
                            cmi_info = f"CMI={analysis.cmi}%, Disagree={word_disagreement_rate}"
                            msg = f"{prefix} ({models_str}): '{snippet}...' ({cmi_info})"
                            job.log(msg)

                    return {
                        "segment_id": seg.segment_id,
                        "episode_id": job.episode_id,
                        # One speaker per episode, always. Diarization is a post-export step
                        # against the full episode audio, not something this pipeline guesses at
                        # (D58); nothing here is in a position to tell two speakers apart.
                        "speaker_id": "spk0",
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "clip_path": seg.clip_rel_path,
                        "clip_checksum": seg.clip_checksum,
                        "vad_spans": speech_spans_within(turns, seg.start_time, seg.end_time),
                        # Absent, not empty, when the detector did not run: never measured is not
                        # the same as measured and clean (D77).
                        **(
                            {"overlap_spans": spans_within(overlap, seg.start_time, seg.end_time)}
                            if overlap is not None
                            else {}
                        ),
                        **(
                            {"acoustics": acoustics[seg.segment_id]}
                            if seg.segment_id in acoustics
                            else {}
                        ),
                        "hypotheses": hypotheses,
                        "scores": {
                            "cmi": analysis.cmi,
                            "code_switch_density": analysis.code_switch_density,
                            "switch_point_count": analysis.switch_point_count,
                            "discourse_marker_count": analysis.discourse_marker_count,
                            "word_disagreement_rate": word_disagreement_rate,
                            "cer_between_hypotheses": cer_between_hyps,
                            "avg_logprob": primary_hyp["avg_logprob"],
                            "flags": analysis.flags,
                        },
                    }

                # Rec 2: Concurrent segment processing with bounded concurrency
                max_seg_workers = min(
                    settings.ingest.max_segment_concurrency,
                    max(1, len(segments)),
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_seg_workers) as seg_pool:
                    future_to_idx = {
                        seg_pool.submit(_process_segment, idx, seg): idx
                        for idx, seg in enumerate(segments)
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        if job.scrammed:
                            # Belt to the guard's braces: a queued future that is cancelled here
                            # never enters the worker at all. Cancelling a running or finished
                            # one is a no-op, so the ones in flight still land below.
                            for queued in future_to_idx:
                                queued.cancel()
                        seg_idx = future_to_idx[future]
                        try:
                            records_by_idx[seg_idx] = future.result()
                        except concurrent.futures.CancelledError:
                            continue
                        except Exception as exc:  # costs one segment, never the episode
                            # Everything after the transcripts -- alignment, analysis, the
                            # per-segment commit. Same rule as an ASR failure: the run is worth
                            # more than the segment, and the summary says what was lost.
                            seg = segments[seg_idx]
                            with progress_lock:
                                job.discard(
                                    DiscardedSegment(
                                        segment_id=seg.segment_id,
                                        start_time=seg.start_time,
                                        end_time=seg.end_time,
                                        stage="analysis",
                                        failures=[
                                            {
                                                "route": "-",
                                                "system_id": "analysis",
                                                "error": f"{type(exc).__name__}: {exc}",
                                            }
                                        ],
                                    )
                                )
                                _bump_progress()

            segment_records = [r for r in records_by_idx if r is not None]

        if _halted(job, segments_detected=len(segments), segments_transcribed=len(segment_records)):
            return

        if job.discarded:
            by_system = job.discard_summary()
            blame = ", ".join(f"{system} ({count})" for system, count in by_system.items())
            job.log(
                f"{len(job.discarded)} of {len(segments)} segments discarded "
                f"({len(job.discarded) / len(segments):.0%}) -- {blame}",
                "warn",
            )
        if not segment_records:
            job.fail(
                f"Stage 3/4 ASR Inference / Analysis failed: all {len(segments)} segments were "
                f"discarded ({', '.join(f'{k} ({v})' for k, v in job.discard_summary().items())})"
            )
            return

    except Exception as exc:
        job.fail(f"Stage 3/4 ASR Inference / Analysis failed: {exc}")
        return

    if _halted(job, segments_detected=len(segments), segments_transcribed=len(segment_records)):
        return

    # Stage 4: Fusion. Never fails the episode: every recogniser's work is already paid for, and a
    # clip the fuser did not answer is seeded from a recogniser and sent to review instead.
    job.set_progress("fusing", 76.0)
    fusion_summary = _run_fusion_stage(
        job, segment_records, segments, session_factory, settings, routes=routes, aligner=aligner
    )

    if _halted(job, segments_detected=len(segments), segments_transcribed=len(segment_records)):
        return

    job.set_progress("analyzing", 82.0)
    job.log("Stage 5/6: Orthography analysis, CMI and rule flags completed")

    # Stage 6: Manifest Generation, Direct Import & Queue Building
    try:
        job.set_progress("importing", 85.0)
        job.log("Stage 6/6: Generating manifest and importing directly into database...")

        job_meta = strip_speaker_pii(job.metadata)
        job_meta.update(
            _classify_episode_topic(job, segment_records, session_factory, settings, routes=routes)
        )

        episode_meta = {
            "episode_id": job.episode_id,
            "show_id": job.show_id,
            "title": job.title,
            "source_uri": job.source_url or f"file://{job.audio_path.name}",
            "published_at": dt.date.today().isoformat(),
            "duration_seconds": duration,
            "source_audio_checksum": source_checksum,
            "pipeline_version": "web_v1",
            "pipeline_commit": "web",
            # The whole normalised recording, not just the clips cut from it. Kept so a serious
            # diarizer can be run over the episode after the export, which is where speaker
            # identity now belongs (D58, D62).
            "audio_path": norm_flac.name,
            **job_meta,
        }

        # Write episode.json
        with open(job.work_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump(episode_meta, f, indent=2, ensure_ascii=False)

        # Write segments.jsonl
        with open(job.work_dir / "segments.jsonl", "w", encoding="utf-8") as f:
            for rec in segment_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with session_factory() as session:
            import_report = import_manifest(
                session,
                job.work_dir,
                storage=storage,
                settings=settings,
            )
            job.log(
                f"Database import: {import_report.segments_inserted} segments, "
                f"{import_report.clips_uploaded} clips uploaded to storage"
            )

            if diarize_future is not None:
                _import_speaker_turns(job, session, diarize_future, settings)

            # Nothing stands between import and the queue any more: the episode drew its train/val
            # split at import, and gold is chosen per clip by hand from the queue itself (D71).
            queue_report = build_queue(
                session, settings=settings, episode_external_id=job.episode_id
            )
            job.log(
                f"Queue built: {queue_report.tasks_created} tasks "
                f"({queue_report.review_tasks} review, {queue_report.audit_tasks} audit)"
            )
            session.commit()

        job.complete(
            {
                "episode_id": job.episode_id,
                "duration_seconds": round(duration, 1),
                "segments": len(segment_records),
                "segments_detected": len(segments),
                "segments_discarded": len(job.discarded),
                # Which system cost what, so a bad run points at a vendor rather than at luck.
                "discarded_by_system": job.discard_summary(),
                "discarded_segments": [d.as_dict() for d in job.discarded],
                "fusion": fusion_summary,
                "tasks_created": queue_report.tasks_created if queue_report else 0,
                "review_tasks": queue_report.review_tasks if queue_report else 0,
            }
        )
    except Exception as exc:
        job.fail(f"Stage 6 Database Import & Queue Building failed: {exc}")
        return
