"""The unit of ingest work: one episode's run, its state machine and its SSE listeners.

Split out of the former single-module ingest service; the pipeline itself lives in
:mod:`app.services.ingest.pipeline`, the queue that runs jobs in
:mod:`app.services.ingest.manager`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

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
