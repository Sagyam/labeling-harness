"""Backend ingestion service for podcast episodes.

Coordinates the 5-stage ingestion pipeline:
1. Audio normalization via FFmpeg with loudnorm (16 kHz mono FLAC)
2. Utterance segmentation via Silero VAD (2.0s - 20.0s boundaries)
3. Cloud ASR inference across every configured `asr*` route (logged to llm_requests)
4. Orthography-aware token tagging, CMI, and rule flags
5. Manifest generation and direct database import + queue building

A segment that cannot be transcribed by every configured system is discarded and the run carries
on (D46). Stage 3 is where all the money is: it dispatches every `asr*` route at every clip, so
one refused clip near the end of a two-hour episode used to throw away hours of paid inference
that had already succeeded. Only an episode with nothing left fails outright. What was dropped,
and which system dropped it, rides in the completion summary.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import datetime as dt
import json
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
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
from app.services.analysis import analyze_transcript, mean_pairwise_disagreement
from app.services.forced_align import ForcedAligner, align_text
from app.services.importer import import_manifest
from app.services.queue_builder import build_queue
from app.services.silero_vad import (
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
)
from app.services.youtube import YouTubeError, download_audio
from app.storage import build_storage
from app.storage.base import ObjectStorage
from app.utils.hashing import sha256_file
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LockedSession:
    """Thread-safe proxy for a SQLAlchemy Session.

    Protects session operations (notably .add, .flush, and .commit) with a reentrant mutex
    so that multiple worker threads can perform concurrent ASR requests while safely logging
    LlmRequest rows and committing per-segment transactions.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._lock = threading.RLock()

    def add(self, instance: Any) -> None:
        with self._lock:
            self._session.add(instance)

    def flush(self, objects: Any = None) -> None:
        with self._lock:
            self._session.flush(objects=objects)

    def commit(self) -> None:
        with self._lock:
            self._session.commit()

    def rollback(self) -> None:
        with self._lock:
            self._session.rollback()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._session, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    return attr(*args, **kwargs)

            return wrapper
        return attr


@dataclass
class IngestLog:
    timestamp: str
    level: str
    message: str


@dataclass
class DiscardedSegment:
    """One segment dropped from the run, and who dropped it.

    A discard is not an error the annotator can act on -- the clip is gone and its cost is spent.
    It is a fact about the corpus, so it is carried all the way to the ingest summary with the
    system that caused it named (D46).
    """

    segment_id: str
    start_time: float
    end_time: float
    #: ``asr`` when a transcriber failed, ``analysis`` when the segment was lost after every
    #: transcript was in hand.
    stage: str
    #: One entry per system that failed on this clip: ``{"route", "system_id", "error"}``.
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def systems(self) -> list[str]:
        return [f["system_id"] for f in self.failures]

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "stage": self.stage,
            "failures": self.failures,
        }


@dataclass
class IngestJob:
    job_id: str
    episode_id: str
    show_id: str
    title: str
    #: The source file. ``None`` until the download stage produces one, for a job started from a
    #: URL rather than an upload.
    audio_path: Path | None
    work_dir: Path
    #: Canonical URL the audio was fetched from, when the job did not start as an upload.
    source_url: str | None = None
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    stage: str = "upload"
    progress: float = 0.0
    active_segments: int = 0
    total_segments: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    logs: list[IngestLog] = field(default_factory=list)
    #: Segments dropped mid-run rather than failing the episode (D46).
    discarded: list[DiscardedSegment] = field(default_factory=list)
    #: What the run produced, once it has. Kept so a page that reattaches to an already-finished
    #: job can show the outcome without having been connected when the event went out.
    summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Each SSE subscriber registers its queue together with the loop that queue belongs to; the
    #: pipeline runs on a worker thread and must hand work back across that boundary.
    listeners: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[str]]] = field(
        default_factory=list
    )

    def log(self, message: str, level: str = "info") -> None:
        ts = dt.datetime.now(dt.UTC).strftime("%H:%M:%S")
        entry = IngestLog(timestamp=ts, level=level, message=message)
        self.logs.append(entry)
        self._emit({"type": "log", "timestamp": ts, "level": level, "message": message})

    def set_progress(
        self, stage: str, progress: float, active_segments: int = 0, total_segments: int = 0
    ) -> None:
        self.stage = stage
        self.progress = round(progress, 1)
        if active_segments:
            self.active_segments = active_segments
        if total_segments:
            self.total_segments = total_segments

        self._emit(
            {
                "type": "progress",
                "stage": self.stage,
                "progress": self.progress,
                "active_segments": self.active_segments,
                "total_segments": self.total_segments,
            }
        )

    def discard(self, record: DiscardedSegment) -> None:
        """Drop one segment from the run and say who cost it.

        Thread-safety: called from the segment worker pool, so it appends under the caller's own
        progress lock. The list is only read after that pool has joined.
        """
        self.discarded.append(record)
        who = ", ".join(record.systems) or record.stage
        detail = "; ".join(f"{f['system_id']}: {f['error']}" for f in record.failures)
        self.log(
            f"Discarded {record.segment_id} "
            f"({record.start_time:.1f}s-{record.end_time:.1f}s) -- {who} failed. {detail}",
            "warn",
        )
        self._emit({"type": "discard", "segment": record.as_dict()})

    def discard_summary(self) -> dict[str, int]:
        """How many segments each system cost, most expensive first."""
        counts: dict[str, int] = {}
        for record in self.discarded:
            for system in record.systems or [record.stage]:
                counts[system] = counts.get(system, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def complete(self, summary: dict[str, Any]) -> None:
        self.status = "completed"
        self.stage = "complete"
        self.progress = 100.0
        self.summary = summary
        self.log("Ingestion pipeline finished successfully.", "success")
        self._emit({"type": "complete", "summary": summary, "episode_id": self.episode_id})

    def fail(self, error_message: str) -> None:
        self.status = "failed"
        self.stage = "failed"
        self.error = error_message
        self.log(f"Pipeline error: {error_message}", "error")
        self._emit({"type": "error", "error": error_message})

    def _emit(self, event: dict[str, Any]) -> None:
        """Fan an event out to every SSE subscriber.

        This runs on the pipeline's worker thread while the queues belong to the server's event
        loop, and :class:`asyncio.Queue` is not thread-safe -- writing to one directly can leave a
        waiting reader unwoken. Every put is therefore scheduled onto the owning loop.
        """
        payload = json.dumps(event)
        for loop, q in list(self.listeners):
            try:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            except RuntimeError as exc:  # loop already closed: the subscriber has gone away
                logger.debug("ingest_listener_dropped", job_id=self.job_id, error=str(exc))


class IngestionManager:
    """In-memory manager tracking active and historical ingestion jobs.

    History is capped: a job keeps its whole log in memory, so an unbounded registry grows for as
    long as the process lives. Only finished jobs are evicted, oldest first, so a running pipeline
    can never lose the record it is still writing to.
    """

    #: How many finished jobs to keep for the status endpoint to look back at.
    MAX_FINISHED_JOBS = 50

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}

    def create_job(
        self,
        *,
        episode_id: str,
        show_id: str,
        title: str,
        work_dir: Path,
        audio_path: Path | None = None,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestJob:
        job_id = str(uuid.uuid4())
        job = IngestJob(
            job_id=job_id,
            episode_id=episode_id,
            show_id=show_id,
            title=title,
            audio_path=audio_path,
            work_dir=work_dir,
            source_url=source_url,
            metadata=metadata or {},
        )
        self._jobs[job_id] = job
        self._evict_finished()
        return job

    def get_job(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def _evict_finished(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("completed", "failed")]
        finished.sort(key=lambda j: j.created_at)
        for job in finished[: max(0, len(finished) - self.MAX_FINISHED_JOBS)]:
            del self._jobs[job.job_id]


manager = IngestionManager()


#: Resampling down to 16 kHz throws away everything above 8 kHz, and whatever is not filtered out
#: first does not vanish -- it folds back below 8 kHz as alias. FFmpeg's built-in resampler is the
#: wrong tool for the job here: measured against pure tones, ``aresample=16000`` passes 8.2 kHz at
#: -15 dB and 8.5 kHz at -26 dB, so a podcast with ordinary energy in the 8-10 kHz band gets that
#: band folded on top of its own 7-8 kHz. Sibilants are exactly where that energy lives, so every
#: /s/ and /ʃ/ lands a burst of near-Nyquist noise -- heard as a click, roughly once a second.
#: libsoxr rejects the same tones at -158 dB. The cutoff keeps the passband flat to 7.6 kHz.
SOXR_RESAMPLE = "aresample=resampler=soxr:precision=28:cutoff=0.95:osr=16000"

#: Used only when the FFmpeg build has no libsoxr. Lengthening the filter and pulling the cutoff in
#: takes the same 8.2 kHz tone from -15 dB to -55 dB: far better than the default, still far worse
#: than soxr, so :func:`_resample_filter` says so out loud rather than degrading quietly.
SWR_RESAMPLE_FALLBACK = "aresample=16000:filter_size=256:cutoff=0.91"


@lru_cache(maxsize=1)
def _resample_filter() -> str:
    """The resampling stage of the stage-1 filter chain, preferring libsoxr.

    Probed once per process by asking FFmpeg to convert a fraction of a second of silence, which
    is cheaper and more truthful than parsing ``-buildconf``: it fails exactly when the filter
    would fail in the pipeline.
    """
    probe = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "0.1",
            "-af",
            SOXR_RESAMPLE,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        return SOXR_RESAMPLE
    logger.warning(
        "ffmpeg_soxr_unavailable",
        detail=probe.stderr.decode("utf-8", errors="replace")[:200],
        fallback=SWR_RESAMPLE_FALLBACK,
        impact="clips will carry more resampling alias than a soxr build produces",
    )
    return SWR_RESAMPLE_FALLBACK


def normalize_audio(input_path: Path, output_path: Path) -> float:
    """Stage 1: Normalize audio using FFmpeg with loudnorm filter.

    Converts to 16 kHz mono FLAC using two-pass EBU R128 normalization.
    Pass 1 measures integrated loudness and true-peak statistics.
    Pass 2 applies linear normalization to prevent dynamic AGC gain pumping between words, then
    downmixes and resamples through :func:`_resample_filter` so that nothing above 8 kHz folds
    back into the clip as alias.
    Returns duration in seconds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: Measure loudness parameters
    cmd1 = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-23:LRA=7:tp=-2:print_format=json",
        "-f",
        "null",
        "-",
    ]
    proc1 = subprocess.run(cmd1, capture_output=True, check=False)

    measured: dict[str, Any] | None = None
    if proc1.returncode == 0:
        try:
            stderr_text = proc1.stderr.decode("utf-8", errors="replace")
            start_brace = stderr_text.rfind("{")
            end_brace = stderr_text.rfind("}")
            if start_brace != -1 and end_brace > start_brace:
                data = json.loads(stderr_text[start_brace : end_brace + 1])
                if all(k in data for k in ("input_i", "input_lra", "input_tp", "input_thresh")):
                    measured = data
        except Exception:
            measured = None

    # Pass 2: Apply linear normalization with clean mono downmix before resampling
    resample = _resample_filter()
    if measured:
        filter_str = (
            "loudnorm=I=-23:LRA=7:tp=-2"
            f":measured_I={measured['input_i']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_tp={measured['input_tp']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured.get('target_offset', '0.0')}"
            f":linear=true,aformat=channel_layouts=mono,{resample}"
        )
    else:
        filter_str = f"loudnorm=I=-23:LRA=7:tp=-2,aformat=channel_layouts=mono,{resample}"

    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_str,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_path),
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, check=False)
    if proc2.returncode != 0:
        err_msg = proc2.stderr.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"FFmpeg normalization failed (exit code {proc2.returncode}): {err_msg}")

    info = sf.info(str(output_path))
    return float(info.duration)


def run_pipeline(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage | None = None,
    settings: Settings | None = None,
    *,
    keep_work_dir: bool = False,
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
    """
    settings = settings or get_settings()
    storage = storage or build_storage(settings)

    try:
        if job.audio_path is None and not _fetch_source_audio(job, settings):
            return
        _run_stages(job, session_factory, storage, settings)
    finally:
        if not keep_work_dir:
            shutil.rmtree(job.work_dir, ignore_errors=True)


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
    except YouTubeError as exc:
        job.fail(f"Audio download failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive; the module raises YouTubeError
        job.fail(f"Audio download failed: {exc}")
        return False

    size_mb = job.audio_path.stat().st_size / (1024 * 1024)
    job.log(f"Downloaded {job.audio_path.name} ({size_mb:.1f} MB)", "success")
    return True


def _run_stages(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage,
    settings: Settings,
) -> None:
    """The five pipeline stages. Every failure is reported through ``job.fail`` and returns."""
    job.status = "processing"
    job.log(f"Starting ingestion for '{job.title}' ({job.episode_id})")

    if job.audio_path is None:
        job.fail("Stage 1 Audio Normalization failed: no source audio for this job")
        return

    norm_flac = job.work_dir / f"{job.episode_id}_normalized.flac"

    # Stage 1: Normalize Audio
    try:
        job.set_progress("normalizing", 5.0)
        job.log("Stage 1/5: Normalizing audio (FFmpeg loudnorm, 16 kHz mono FLAC)...")
        duration = normalize_audio(job.audio_path, norm_flac)
        source_checksum = sha256_file(job.audio_path)
        job.log(f"Audio normalized: {duration:.1f}s ({duration / 60:.1f} min)")
        job.set_progress("normalizing", 20.0)
    except Exception as exc:
        job.fail(f"Stage 1 Audio Normalization failed: {exc}")
        return

    # Stage 2: Silero VAD Segmentation
    try:
        job.set_progress("segmenting", 22.0)
        job.log("Stage 2/5: Detecting speech turns via Silero VAD (2.0s - 20.0s bounds)...")
        vad = SileroVAD()
        audio_data, sr = sf.read(str(norm_flac), dtype="float32")
        turns = vad.detect_turns(audio_data, sample_rate=sr)
        job.log(f"Detected {len(turns)} raw speech turns")

        slices = segment_audio_to_slices(turns, duration, audio=audio_data, sample_rate=sr)
        job.log(f"Partitioned into {len(slices)} bounded utterances (2.0s - 20.0s)")

        clips_dir = job.work_dir / "clips"
        segments = extract_clips(norm_flac, slices, job.episode_id, clips_dir)
        job.total_segments = len(segments)
        job.active_segments = len(segments)
        job.log(f"Extracted {len(segments)} audio clips to disk")
        job.set_progress(
            "segmenting", 40.0, total_segments=len(segments), active_segments=len(segments)
        )
    except Exception as exc:
        job.fail(f"Stage 2 Silero VAD Segmentation failed: {exc}")
        return

    # Stage 3 & 4: Cloud ASR & Token Analysis
    segment_records: list[dict[str, Any]] = []
    try:
        job.set_progress("transcribing", 42.0)
        routes = load_llm_routes()
        asr_routes = asr_route_names(routes) or ["asr"]
        systems = ", ".join(system_id_for(r, routes.routes.get(r)) for r in asr_routes)
        job.log(
            f"Stage 3/5: Cloud ASR inference for {len(segments)} segments "
            f"across {len(asr_routes)} systems ({systems})..."
        )

        # One aligner per run: the ONNX session is loaded once and its Run() is thread-safe,
        # so every segment worker shares it. Absent model means no word spans, not a failure.
        aligning_routes = {
            r for r in asr_routes if getattr(routes.routes.get(r), "forced_align", False)
        }
        aligner = ForcedAligner() if aligning_routes and not routes.dry_run else None
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
                        "speaker_id": "spk0",
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "clip_path": seg.clip_rel_path,
                        "clip_checksum": seg.clip_checksum,
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
                        seg_idx = future_to_idx[future]
                        try:
                            records_by_idx[seg_idx] = future.result()
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

        job.set_progress("analyzing", 80.0)
        job.log("Stage 4/5: Orthography analysis, CMI and rule flags completed")
    except Exception as exc:
        job.fail(f"Stage 3/4 ASR Inference / Analysis failed: {exc}")
        return

    # Stage 5: Manifest Generation, Direct Import & Queue Building
    try:
        job.set_progress("importing", 85.0)
        job.log("Stage 5/5: Generating manifest and importing directly into database...")

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
            **(job.metadata or {}),
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

            job.log("Building prioritized annotation queues...")
            queue_report = build_queue(
                session, episode_external_id=job.episode_id, settings=settings
            )
            session.commit()

            msg = (
                f"Queue built: {queue_report.tasks_created} tasks created "
                f"({queue_report.review_tasks} review, {queue_report.audit_tasks} audit, "
                f"{queue_report.error_tasks} error)"
            )
            job.log(msg)

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
                "tasks_created": queue_report.tasks_created,
                "review_tasks": queue_report.review_tasks,
            }
        )
    except Exception as exc:
        job.fail(f"Stage 5 Database Import & Queue Building failed: {exc}")
        return
