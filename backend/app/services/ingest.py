"""Backend ingestion service for podcast episodes.

Coordinates the 5-stage ingestion pipeline:
1. Audio normalization via FFmpeg with loudnorm (16 kHz mono FLAC)
2. Utterance segmentation via Silero VAD (2.0s - 20.0s boundaries)
3. Cloud ASR inference across every configured `asr*` route (logged to llm_requests)
4. Orthography-aware token tagging, CMI, and rule flags
5. Manifest generation and direct database import + queue building
"""

from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import itertools
import json
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
from sqlalchemy.orm import Session

from app.config import Settings, get_settings, load_llm_routes
from app.llm.transcription import (
    ASR_PROMPT,
    asr_route_names,
    system_id_for,
    transcribe,
)
from app.services.analysis import analyze_transcript
from app.services.importer import import_manifest
from app.services.queue_builder import build_queue
from app.services.silero_vad import (
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
)
from app.storage import build_storage
from app.storage.base import ObjectStorage
from app.utils.hashing import sha256_file
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestLog:
    timestamp: str
    level: str
    message: str


@dataclass
class IngestJob:
    job_id: str
    episode_id: str
    show_id: str
    title: str
    audio_path: Path
    work_dir: Path
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    stage: str = "upload"
    progress: float = 0.0
    active_segments: int = 0
    total_segments: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    logs: list[IngestLog] = field(default_factory=list)
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

    def complete(self, summary: dict[str, Any]) -> None:
        self.status = "completed"
        self.stage = "complete"
        self.progress = 100.0
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
        audio_path: Path,
        work_dir: Path,
    ) -> IngestJob:
        job_id = str(uuid.uuid4())
        job = IngestJob(
            job_id=job_id,
            episode_id=episode_id,
            show_id=show_id,
            title=title,
            audio_path=audio_path,
            work_dir=work_dir,
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


def normalize_audio(input_path: Path, output_path: Path) -> float:
    """Stage 1: Normalize audio using FFmpeg with loudnorm filter.

    Converts to 16 kHz mono FLAC. Returns duration in seconds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-23:LRA=7:tp=-2,aresample=16000",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"FFmpeg normalization failed (exit code {proc.returncode}): {err_msg}")

    info = sf.info(str(output_path))
    return float(info.duration)


def _mean_pairwise_disagreement(sequences: list[list[str]] | list[str]) -> float:
    """Mean 1 - similarity over every unordered pair of hypotheses.

    Operates on word lists or on raw strings, giving a word-level or character-level rate from
    the same comparison. Fewer than two hypotheses means nothing disagreed, which is 0.0 -- the
    same value the scorer reads for a missing rate.
    """
    if len(sequences) < 2:
        return 0.0
    ratios = [
        difflib.SequenceMatcher(None, sequences[i], sequences[j]).ratio()
        for i, j in itertools.combinations(range(len(sequences)), 2)
    ]
    return round(1.0 - (sum(ratios) / len(ratios)), 4)


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
        _run_stages(job, session_factory, storage, settings)
    finally:
        if not keep_work_dir:
            shutil.rmtree(job.work_dir, ignore_errors=True)


def _run_stages(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage,
    settings: Settings,
) -> None:
    """The five pipeline stages. Every failure is reported through ``job.fail`` and returns."""
    job.status = "processing"
    job.log(f"Starting ingestion for '{job.title}' ({job.episode_id})")

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

        slices = segment_audio_to_slices(turns, duration)
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

        with session_factory() as session:
            for idx, seg in enumerate(segments):
                step_progress = 40.0 + (35.0 * (idx + 1) / len(segments))
                job.set_progress("transcribing", step_progress, active_segments=idx + 1)

                hypotheses: list[dict[str, Any]] = []
                for route_name in asr_routes:
                    sys_name = system_id_for(route_name, routes.routes.get(route_name))
                    asr_res = transcribe(
                        session,
                        seg.clip_path,
                        route=route_name,
                        config=routes,
                        prompt=ASR_PROMPT,
                    )
                    if asr_res.dry_run:
                        # A dry run returns canned text. Name the system so it can never be
                        # mistaken for real model output in the queue or at export.
                        sys_name = f"mock-{sys_name}"
                    hypotheses.append(
                        {
                            "system_id": sys_name,
                            "model_id": asr_res.model,
                            "text": asr_res.text,
                            "avg_logprob": asr_res.avg_logprob,
                            "no_speech_prob": asr_res.no_speech_prob,
                            "words": asr_res.words,
                        }
                    )

                # Cross-system disagreement, averaged over every pair of systems. With two
                # systems this is the single comparison between them; with three it is the mean
                # of the three pairs, so a third hypothesis informs the queue rather than being
                # paid for and ignored.
                texts = [h["text"] for h in hypotheses]
                word_disagreement_rate = _mean_pairwise_disagreement([t.split() for t in texts])
                cer_between_hyps = _mean_pairwise_disagreement(texts)

                primary_hyp = hypotheses[0]
                analysis = analyze_transcript(
                    primary_hyp["text"],
                    duration_seconds=seg.duration,
                    no_speech_prob=primary_hyp["no_speech_prob"],
                    settings=settings,
                )

                if (idx + 1) % 5 == 0 or idx == len(segments) - 1:
                    snippet = primary_hyp["text"][:30]
                    models_str = ", ".join(h["system_id"] for h in hypotheses)
                    msg = (
                        f"[{idx + 1}/{len(segments)}] {seg.segment_id} ({models_str}): "
                        f"'{snippet}...' (CMI={analysis.cmi}%, Disagree={word_disagreement_rate})"
                    )
                    job.log(msg)

                segment_records.append(
                    {
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
                            "word_disagreement_rate": word_disagreement_rate,
                            "cer_between_hypotheses": cer_between_hyps,
                            "avg_logprob": primary_hyp["avg_logprob"],
                            "flags": analysis.flags,
                        },
                    }
                )
                # Commit per segment. The loop makes one network call per route per segment, so
                # a single transaction around the whole stage would hold a pooled connection for
                # the length of the episode and lose every request-log row on a late failure.
                session.commit()
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
            "source_uri": f"file://{job.audio_path.name}",
            "published_at": dt.date.today().isoformat(),
            "duration_seconds": duration,
            "source_audio_checksum": source_checksum,
            "pipeline_version": "web_v1",
            "pipeline_commit": "web",
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
                "tasks_created": queue_report.tasks_created,
                "review_tasks": queue_report.review_tasks,
            }
        )
    except Exception as exc:
        job.fail(f"Stage 5 Database Import & Queue Building failed: {exc}")
        return
