"""Backend ingestion service for podcast episodes.

Coordinates the 6-stage ingestion pipeline:
1. Audio normalization via FFmpeg with loudnorm (16 kHz mono FLAC)
2. Utterance segmentation via Silero VAD (2.0s - 20.0s boundaries)
3. Cloud ASR inference across every configured `asr*` route (logged to llm_requests)
4. Fusion: a reasoning model reconciles the recognisers into one transcript per clip (D72)
5. Orthography-aware token tagging, CMI, and rule flags -- on the fused text where there is one
6. Manifest generation and direct database import + queue building

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
import queue
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

from app.config import LlmRoutes, Settings, get_settings, load_llm_routes
from app.llm.base import AsrResult, LlmResult
from app.llm.openrouter import OpenRouterClient
from app.llm.topic import classify_topic, sample_transcript
from app.llm.transcription import (
    ASR_PROMPT,
    asr_route_names,
    disagreement_excluded_system_ids,
    system_id_for,
    transcribe,
)
from app.llm.vertex import VertexClient
from app.services.analysis import analyze_transcript, mean_pairwise_disagreement
from app.services.forced_align import ForcedAligner, align_text
from app.services.fusion_stage import FUSION_KIND, fuse_records
from app.services.importer import import_manifest
from app.services.queue_builder import build_queue
from app.services.silero_vad import (
    SileroVAD,
    extract_clips,
    segment_audio_to_slices,
    speech_spans_within,
)
from app.services.speaker_meta import strip_speaker_pii
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
    #: "pending" is also what a job waiting its turn in the run queue reports, which is why
    #: the queue does not add a status of its own: to a caller, not started is not started.
    status: str = "pending"  # "pending", "processing", "completed", "failed", "aborted", "backlog"
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
    #: Set by :meth:`scram` from the request thread and read by every pipeline worker. An
    #: :class:`threading.Event` rather than a bool because the readers are the segment pool.
    _scram: threading.Event = field(default_factory=threading.Event, repr=False)
    #: Why the run was scrammed, for the abort summary.
    scram_reason: str | None = None

    @property
    def scrammed(self) -> bool:
        """True once someone has hit AZ-5 on this run."""
        return self._scram.is_set()

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

    def backlog(self, reason: str, error_message: str) -> None:
        """Place job on backlog (e.g. YouTube bot challenge or rate limit) to retry later."""
        self.status = "backlog"
        self.stage = "backlog"
        self.error = error_message
        self.log(f"Moved to backlog: {reason} ({error_message})", "warn")
        self._emit({"type": "backlog", "reason": reason, "error": error_message})

    def scram(self, reason: str = "manual SCRAM") -> bool:
        """Drop the rods: stop this run at the next checkpoint every worker passes.

        Called from the request thread while the pipeline runs on its own. It sets a flag and
        returns; it does not kill anything. Stage 3 is the only stage that spends money, and it
        checks the flag before dispatching a segment's routes, so the requests that have not been
        made yet are the ones this cancels. Calls already in flight are left to return -- at most
        ``max_segment_concurrency`` segments' worth -- because tearing down the HTTP client under
        them would lose the ``llm_requests`` rows for inference that was billed anyway.

        Returns:
            True if this call is what stopped the run; False if it was already stopping or over.
        """
        if self.status in ("completed", "failed", "aborted") or self.scrammed:
            return False
        self.scram_reason = reason
        self._scram.set()
        self.log(f"AZ-5: {reason}. Halting at the next checkpoint.", "warn")
        self._emit({"type": "scram", "reason": reason})
        return True

    def abort(self, summary: dict[str, Any]) -> None:
        """Finish a scrammed run: no import, no queue, and a record of what it had spent.

        Deliberately not ``complete``. Stage 5 imports an episode as a whole, and half of one is
        not a cheaper episode -- it is a differently-sampled one, which is the same reason a
        segment short of a full set of hypotheses is discarded rather than patched (D46). The
        inference already paid for is reported here instead of being quietly written to the
        corpus.
        """
        self.status = "aborted"
        self.stage = "aborted"
        self.summary = summary
        self.log(
            f"Run aborted by AZ-5 after {summary.get('segments_transcribed', 0)} of "
            f"{summary.get('segments_detected', 0)} segments. Nothing was imported.",
            "warn",
        )
        self._emit({"type": "aborted", "summary": summary, "reason": self.scram_reason})

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize job state for durable persistence."""
        return {
            "job_id": self.job_id,
            "episode_id": self.episode_id,
            "show_id": self.show_id,
            "title": self.title,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "work_dir": str(self.work_dir),
            "source_url": self.source_url,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "active_segments": self.active_segments,
            "total_segments": self.total_segments,
            "error": self.error,
            "created_at": self.created_at,
            "scram_reason": self.scram_reason,
            "summary": self.summary,
            "metadata": self.metadata,
            "logs": [
                {"timestamp": entry.timestamp, "level": entry.level, "message": entry.message}
                for entry in self.logs[-50:]
            ],
            "discarded": [d.as_dict() for d in self.discarded],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestJob:
        """Deserialize job from persisted dict."""
        job = cls(
            job_id=data["job_id"],
            episode_id=data["episode_id"],
            show_id=data.get("show_id", "podcast"),
            title=data.get("title", data["episode_id"]),
            audio_path=Path(data["audio_path"]) if data.get("audio_path") else None,
            work_dir=Path(data["work_dir"]),
            source_url=data.get("source_url"),
            status=data.get("status", "pending"),
            stage=data.get("stage", "upload"),
            progress=float(data.get("progress", 0.0)),
            active_segments=int(data.get("active_segments", 0)),
            total_segments=int(data.get("total_segments", 0)),
            error=data.get("error"),
            created_at=float(data.get("created_at", time.time())),
            scram_reason=data.get("scram_reason"),
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
        )
        for log_data in data.get("logs", []):
            job.logs.append(
                IngestLog(
                    timestamp=log_data.get("timestamp", ""),
                    level=log_data.get("level", "info"),
                    message=log_data.get("message", ""),
                )
            )
        return job


@dataclass
class _QueuedRun:
    """One submitted job together with the collaborators its pipeline will need."""

    job: IngestJob
    session_factory: Callable[[], Session]
    storage: ObjectStorage
    settings: Settings


class IngestionManager:
    """In-memory manager tracking active and historical ingestion jobs, and the queue they run in.

    History is capped: a job keeps its whole log in memory, so an unbounded registry grows for as
    long as the process lives. Only finished jobs are evicted, oldest first, so a running pipeline
    can never lose the record it is still writing to.

    Submitted jobs run **one at a time**, in submission order, on a single worker thread. The
    pipeline already parallelises inside an episode -- `max_segment_concurrency` clips are in
    flight at once in stage 3 -- so running two episodes together does not finish either sooner:
    it splits the same ffmpeg cores and the same ASR rate limit across both, and doubles the peak
    disk in the work root. Queueing instead means the machine is loaded the same whether one
    episode was submitted or twenty.

    The queue lives in this process and nowhere else, which is the same limitation the job log has
    always had: if the server dies, everything still waiting dies with it and there is no record
    the work was ever asked for. Persisting it is a separate change.
    """

    #: How many finished jobs to keep for the status endpoint to look back at.
    MAX_FINISHED_JOBS = 100

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        #: Jobs submitted and not yet picked up, with everything the pipeline needs to run them.
        self._pending: queue.Queue[_QueuedRun] = queue.Queue()
        #: Ids in queue order, so a waiting job can be told its place without draining the queue.
        #: Held under ``_lock`` with the running job, because the status endpoint reads both.
        self._waiting: list[str] = []
        #: Jobs placed in backlog (e.g. YouTube bot check/rate limiting) to retry later.
        self._backlog: list[str] = []
        self._running_id: str | None = None
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._state_file: Path | None = None
        self._state_loaded: bool = False

    def init_state(self, work_root: Path) -> None:
        """Initialize state persistence file and load previous state if present."""
        with self._lock:
            self._state_file = work_root / "queue_state.json"
            if not self._state_loaded:
                work_root.mkdir(parents=True, exist_ok=True)
                self._load_state()
                self._state_loaded = True

    def reset(self) -> None:
        """Clear all in-memory jobs and state (used for testing and teardown)."""
        with self._lock:
            self._jobs.clear()
            self._waiting.clear()
            self._backlog.clear()
            self._running_id = None
            self._state_file = None
            self._state_loaded = False
            while not self._pending.empty():
                try:
                    self._pending.get_nowait()
                    self._pending.task_done()
                except Exception:
                    break

    def _save_state(self) -> None:
        """Persist jobs and queue state to disk atomically. Caller should hold _lock."""
        if self._state_file is None:
            return
        try:
            payload = {
                "waiting": list(self._waiting),
                "backlog": list(self._backlog),
                "running_id": self._running_id,
                "jobs": [job.to_dict() for job in self._jobs.values()],
            }
            temp_file = self._state_file.with_suffix(".tmp")
            state_json = json.dumps(payload, indent=2, ensure_ascii=False)
            temp_file.write_text(state_json, encoding="utf-8")
            temp_file.replace(self._state_file)
        except Exception as exc:
            logger.debug("ingest_save_state_failed", error=str(exc))

    def _load_state(self) -> None:
        """Restore jobs and queue state from disk on startup. Caller holds _lock."""
        if self._state_file is None or not self._state_file.is_file():
            return
        try:
            content = self._state_file.read_text(encoding="utf-8")
            data = json.loads(content)
            for j_data in data.get("jobs", []):
                try:
                    job = IngestJob.from_dict(j_data)
                    # If process was terminated while processing, mark as failed rather than stuck
                    if job.status == "processing":
                        job.status = "failed"
                        job.stage = "failed"
                        job.error = "Interrupted by server restart"
                    self._jobs[job.job_id] = job
                except Exception as exc:
                    logger.warning("ingest_job_restore_failed", error=str(exc))

            for jid in data.get("backlog", []):
                if jid in self._jobs and jid not in self._backlog:
                    self._backlog.append(jid)
        except Exception as exc:
            logger.warning("ingest_load_state_failed", error=str(exc))

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
        self.init_state(work_dir.parent)
        job_id = str(uuid.uuid4())
        initial_stage = "downloading" if source_url else "upload"
        job = IngestJob(
            job_id=job_id,
            episode_id=episode_id,
            show_id=show_id,
            title=title,
            audio_path=audio_path,
            work_dir=work_dir,
            source_url=source_url,
            stage=initial_stage,
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job_id] = job
            self._evict_finished()
            self._save_state()
        return job

    def get_job(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def submit(
        self,
        job: IngestJob,
        session_factory: Callable[[], Session],
        storage: ObjectStorage,
        settings: Settings,
    ) -> int:
        """Queue a job to run when the ones before it are done.

        Returns:
            How many jobs are ahead of this one: 0 means it starts immediately.
        """
        self.init_state(settings.ingest.work_root)
        with self._lock:
            if job.job_id in self._backlog:
                self._backlog.remove(job.job_id)
            if job.job_id not in self._waiting:
                self._waiting.append(job.job_id)
            ahead = len(self._waiting) - 1 + (1 if self._running_id else 0)
            self._pending.put(_QueuedRun(job, session_factory, storage, settings))
            self._ensure_worker()
            self._save_state()
        if ahead:
            job.log(f"Queued: {ahead} job(s) ahead of this one.")
        return ahead

    def retry_job(
        self,
        job_id: str,
        session_factory: Callable[[], Session],
        storage: ObjectStorage,
        settings: Settings,
    ) -> int:
        """Retry a failed, aborted, or backlogged job."""
        self.init_state(settings.ingest.work_root)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            if job.status not in ("failed", "aborted", "backlog"):
                raise ValueError(f"Job {job_id} has status '{job.status}' and cannot be retried")

            if job_id in self._backlog:
                self._backlog.remove(job_id)
            if job_id in self._waiting:
                self._waiting.remove(job_id)

            # Reset job state
            job.status = "pending"
            job.stage = "downloading" if job.source_url else "upload"
            job.progress = 0.0
            job.active_segments = 0
            job.total_segments = 0
            job.error = None
            job._scram.clear()
            job.scram_reason = None
            job.discarded = []
            job.summary = None

            # For URL jobs, ensure work directory exists
            if not job.work_dir.exists():
                job.work_dir.mkdir(parents=True, exist_ok=True)
            if job.source_url:
                job.audio_path = None

            job.log("Job requeued for retry by user", "info")
            self._waiting.append(job.job_id)
            ahead = len(self._waiting) - 1 + (1 if self._running_id else 0)
            self._pending.put(_QueuedRun(job, session_factory, storage, settings))
            self._ensure_worker()
            self._save_state()
        return ahead

    def retry_all(
        self,
        status_filter: str | None,
        session_factory: Callable[[], Session],
        storage: ObjectStorage,
        settings: Settings,
    ) -> list[dict[str, Any]]:
        """Batch retry all jobs matching status_filter (e.g. 'backlog' or 'failed')."""
        with self._lock:
            candidate_ids = [
                j.job_id
                for j in self._jobs.values()
                if (status_filter is None and j.status in ("failed", "aborted", "backlog"))
                or (status_filter and j.status == status_filter)
            ]
        results = []
        for jid in candidate_ids:
            try:
                ahead = self.retry_job(jid, session_factory, storage, settings)
                results.append({"job_id": jid, "success": True, "queue_position": ahead})
            except Exception as exc:
                results.append({"job_id": jid, "success": False, "error": str(exc)})
        return results

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a waiting/running job or remove a completed/failed job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id} not found")
            if job_id == self._running_id:
                job.scram(reason="Cancelled by user")
                self._save_state()
                return {"job_id": job_id, "action": "scrammed", "status": job.status}
            if job_id in self._waiting:
                self._waiting.remove(job_id)
                job.status = "aborted"
                job.stage = "aborted"
                job.log("Cancelled from queue", "warn")
                self._save_state()
                return {"job_id": job_id, "action": "cancelled", "status": job.status}
            if job_id in self._backlog:
                self._backlog.remove(job_id)
                job.status = "aborted"
                job.stage = "aborted"
                job.log("Removed from backlog", "warn")
                self._save_state()
                return {"job_id": job_id, "action": "removed_from_backlog", "status": job.status}

            del self._jobs[job_id]
            self._save_state()
            return {"job_id": job_id, "action": "deleted", "status": "deleted"}

    def clear_past(self) -> int:
        """Remove completed, failed, or aborted jobs from memory history."""
        with self._lock:
            past_ids = [
                jid
                for jid, j in self._jobs.items()
                if j.status in ("completed", "failed", "aborted")
                and jid != self._running_id
                and jid not in self._waiting
                and jid not in self._backlog
            ]
            for jid in past_ids:
                del self._jobs[jid]
            self._save_state()
            return len(past_ids)

    def queue_position(self, job_id: str) -> int | None:
        """Places ahead of a job still waiting, or ``None`` once it is running or finished."""
        with self._lock:
            if job_id not in self._waiting:
                return None
            return self._waiting.index(job_id) + (1 if self._running_id else 0)

    def queue_snapshot(self) -> list[dict[str, Any]]:
        """The running job and everything behind it, in the order they will run."""
        with self._lock:
            ordered = ([self._running_id] if self._running_id else []) + list(self._waiting)
        rows = []
        for position, job_id in enumerate(ordered):
            job = self._jobs.get(job_id)
            if job is None:  # pragma: no cover - evicted between snapshot and lookup
                continue
            rows.append(
                {
                    "job_id": job.job_id,
                    "episode_id": job.episode_id,
                    "title": job.title,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "queue_position": position,
                }
            )
        return rows

    def queue_snapshot_rich(self) -> dict[str, Any]:
        """Categorized snapshot of running, upcoming, backlog, and past jobs."""
        with self._lock:
            running_job = self._jobs.get(self._running_id) if self._running_id else None
            waiting_ids = list(self._waiting)
            backlog_ids = [j.job_id for j in self._jobs.values() if j.status == "backlog"]
            past_jobs = [
                j
                for j in self._jobs.values()
                if j.status in ("completed", "failed", "aborted")
                and j.job_id != self._running_id
                and j.job_id not in waiting_ids
                and j.job_id not in backlog_ids
            ]

        def _summary(job: IngestJob, pos: int | None = None) -> dict[str, Any]:
            return {
                "job_id": job.job_id,
                "episode_id": job.episode_id,
                "show_id": job.show_id,
                "title": job.title,
                "status": job.status,
                "stage": job.stage,
                "progress": job.progress,
                "active_segments": job.active_segments,
                "total_segments": job.total_segments,
                "error": job.error,
                "source_url": job.source_url,
                "created_at": job.created_at,
                "queue_position": pos,
            }

        running = _summary(running_job) if running_job else None
        upcoming = [
            _summary(self._jobs[jid], i + 1)
            for i, jid in enumerate(waiting_ids)
            if jid in self._jobs
        ]
        backlog = [_summary(self._jobs[jid]) for jid in backlog_ids if jid in self._jobs]
        past = [_summary(j) for j in sorted(past_jobs, key=lambda j: j.created_at, reverse=True)]

        return {
            "running": running,
            "upcoming": upcoming,
            "backlog": backlog,
            "past": past,
            "counts": {
                "running": 1 if running else 0,
                "upcoming": len(upcoming),
                "backlog": len(backlog),
                "past": len(past),
                "total": len(self._jobs),
            },
            "jobs": ([running] if running else []) + upcoming,
        }

    def _ensure_worker(self) -> None:
        """Start the runner thread on first submission. Caller holds ``_lock``."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._drain, name="ingest-queue", daemon=True)
        self._worker.start()

    def _drain(self) -> None:
        """Run queued jobs one at a time, forever."""
        while True:
            item = self._pending.get()
            with self._lock:
                if item.job.job_id in self._waiting:
                    self._waiting.remove(item.job.job_id)
                self._running_id = item.job.job_id
                self._save_state()
            try:
                if item.job.scrammed:
                    item.job.abort({"segments_detected": 0, "segments_transcribed": 0})
                else:
                    run_pipeline(item.job, item.session_factory, item.storage, item.settings)
            except Exception as exc:
                logger.exception("ingest_queue_job_crashed", job_id=item.job.job_id)
                with contextlib.suppress(Exception):
                    item.job.fail(f"Ingestion crashed: {exc}")
            finally:
                with self._lock:
                    if item.job.status == "backlog" and item.job.job_id not in self._backlog:
                        self._backlog.append(item.job.job_id)
                    self._running_id = None
                    self._save_state()
                self._pending.task_done()

    def _evict_finished(self) -> None:
        finished = [
            j
            for j in self._jobs.values()
            if j.status in ("completed", "failed", "aborted")
            and j.job_id != self._running_id
            and j.job_id not in self._waiting
            and j.job_id not in self._backlog
        ]
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
        "-threads",
        "0",
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
        "-threads",
        "0",
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


def _best_text(record: dict[str, Any]) -> str:
    """The fused transcript when there is one, otherwise the first recogniser's."""
    for hypothesis in record["hypotheses"]:
        if hypothesis.get("kind") == FUSION_KIND:
            return str(hypothesis.get("text") or "")
    return str(record["hypotheses"][0].get("text") or "")


def _fusion_completer(
    session: Session, routes: LlmRoutes, route_name: str
) -> Callable[[list[dict[str, Any]]], LlmResult] | None:
    """A routed, logged completion on the fusion route, or ``None`` when fusion cannot run.

    ``None`` on a dry run -- the recognisers returned canned text and there is nothing to
    reconcile -- and when the route is not configured. Every real request goes through the
    provider client, so it writes its ``llm_requests`` row like any other inference (invariant 6).
    """
    route = routes.routes.get(route_name) if route_name else None
    if route is None or routes.dry_run:
        return None
    client: VertexClient | OpenRouterClient = (
        VertexClient(session, config=routes)
        if route.provider == "vertex"
        else OpenRouterClient(session, config=routes)
    )
    return lambda messages: client.complete(route_name, messages)


def _run_fusion_stage(
    job: IngestJob,
    segment_records: list[dict[str, Any]],
    segments: list[Any],
    session_factory: Callable[[], Session],
    settings: Settings,
    *,
    routes: LlmRoutes,
    aligner: ForcedAligner | None,
) -> dict[str, Any] | None:
    """Stage 4: fuse the episode's recognisers into one hypothesis per clip (D72).

    Mutates ``segment_records`` in place. Never raises: a stage that fails outright is logged, the
    clips keep their recognisers, and the queue seeds and routes them without a fused text.
    """
    route_name = settings.fusion.route
    route = routes.routes.get(route_name) if route_name else None
    if route is None:
        job.log(
            "Stage 4/6: Fusion skipped -- no fusion route configured; seeds fall back to one "
            "recogniser",
            "warn",
        )
        return None

    clip_paths = {seg.segment_id: Path(seg.clip_path) for seg in segments}
    try:
        with session_factory() as session:
            complete = _fusion_completer(LockedSession(session), routes, route_name)
            if complete is None:
                job.log("Stage 4/6: Fusion skipped on a dry run -- canned text has nothing to fuse")
                return None
            job.log(
                f"Stage 4/6: Fusing {len(segment_records)} segments with {route.model} "
                f"(windows of ~{settings.fusion.window_target_words} words sent)..."
            )
            report = fuse_records(
                segment_records,
                complete=complete,
                route=route,
                fusion=settings.fusion,
                settings=settings,
                aligner=aligner if aligner is not None and aligner.available else None,
                clip_path_for=clip_paths.__getitem__,
                should_stop=lambda: job.scrammed,
                log=job.log,
                max_workers=settings.ingest.max_segment_concurrency,
            )
            session.commit()
    except Exception as exc:  # the recognisers' work is paid for; see the docstring
        logger.warning("fusion_stage_failed", error=str(exc))
        job.log(f"Fusion failed ({type(exc).__name__}: {exc}); seeds fall back to one recogniser",
                "warn")  # fmt: skip
        return {"error": f"{type(exc).__name__}: {exc}"}

    job.log(
        f"Fusion: {report.fused}/{report.segments} segments fused in {report.windows} window(s), "
        f"{report.requests} request(s), {report.thought_tokens:,} thought tokens, "
        f"${report.cost_usd:.3f}"
        + (f" -- {len(report.unfused)} left unfused" if report.unfused else "")
    )
    return report.as_dict()


def _classify_episode_topic(
    job: IngestJob,
    segment_records: list[dict[str, Any]],
    session_factory: Callable[[], Session],
    settings: Settings,
    *,
    routes: LlmRoutes,
) -> dict[str, Any]:
    """Ask a model what this episode is about, when nobody has said.

    Never raises and never fails the ingest. The episode's audio, transcripts and queue are all
    already produced by this point, and a metadata field is not worth throwing them away for -- a
    failed classification leaves the topic empty and an annotator can still type one.

    Returns:
        Metadata to merge into ``episode.json``: the ``topic`` when one was decided, plus
        provenance saying where it came from. Empty when the topic was already set by hand, no
        route is configured, or the attempt failed.
    """
    route = settings.ingest.topic_route
    if not route or (job.metadata or {}).get("topic"):
        return {}
    if route not in routes.routes:
        logger.warning("topic_route_not_configured", route=route)
        return {}

    excerpt = sample_transcript(
        [_best_text(record) for record in segment_records if record["hypotheses"]]
    )
    try:
        with session_factory() as session:
            topic, meta = classify_topic(
                session,
                title=job.title,
                transcript=excerpt,
                route=route,
                config=routes,
            )
            session.commit()
    except Exception as exc:  # see the docstring: a metadata field never fails an ingest
        logger.warning("topic_classification_failed", route=route, error=str(exc))
        job.log("Topic classification failed; leaving the episode's topic empty")
        return {}

    if topic:
        job.log(f"Topic classified as '{topic}'")
        return {"topic": topic, **meta}
    return meta


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
    except YouTubeBotDetected as exc:
        job.backlog(
            reason="YouTube bot detection / rate limit challenge",
            error_message=str(exc),
        )
        with manager._lock:
            if job.job_id not in manager._backlog:
                manager._backlog.append(job.job_id)
        return False
    except YouTubeError as exc:
        if is_bot_detection_error(str(exc)):
            job.backlog(
                reason="YouTube bot detection / rate limit challenge",
                error_message=str(exc),
            )
            with manager._lock:
                if job.job_id not in manager._backlog:
                    manager._backlog.append(job.job_id)
            return False
        job.fail(f"Audio download failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive; the module raises YouTubeError
        if is_bot_detection_error(str(exc)):
            job.backlog(
                reason="YouTube bot detection / rate limit challenge",
                error_message=str(exc),
            )
            with manager._lock:
                if job.job_id not in manager._backlog:
                    manager._backlog.append(job.job_id)
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


def _run_stages(
    job: IngestJob,
    session_factory: Callable[[], Session],
    storage: ObjectStorage,
    settings: Settings,
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

        clips_dir = job.work_dir / "clips"
        segments = extract_clips(
            norm_flac, slices, job.episode_id, clips_dir, max_workers=settings.ingest.cpu_workers
        )
        job.total_segments = len(segments)
        job.active_segments = len(segments)
        job.log(f"Extracted {len(segments)} audio clips to disk")
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
