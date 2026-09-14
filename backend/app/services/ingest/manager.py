"""The in-process queue that runs one ingestion job at a time, and its persistence."""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.storage.base import ObjectStorage
from app.utils.logging import get_logger

from . import pipeline as _pipeline
from .job import IngestJob

logger = get_logger(__name__)


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
                    _pipeline.run_pipeline(
                        item.job, item.session_factory, item.storage, item.settings
                    )
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
