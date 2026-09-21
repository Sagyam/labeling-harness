"""Paid results kept on disk while an episode ingests, so a resumed run never buys them twice (D93).

An episode is held in memory from download to import, and only stage 6 writes it to Postgres. A
power cut before then used to lose every transcript, fusion window and diarization the run had
already paid for. Each one is now written here the moment it arrives, keyed by exactly what was
sent -- the clip's samples, or the fusion request's messages -- plus the route that answered. A
resumed run rebuilds the same requests, finds them here and skips the call; anything that
changed in between (a re-cut clip, an edited route, a new prompt) is a different key and is paid
for again, never answered from a stale result.

This lives in the job's work directory, not the database: it is only for finishing the run that
wrote it, and goes when the run finishes. What was paid for is on record in ``llm_requests`` either
way (invariant 6); a replay is not an inference call, writes no row and is reported at zero cost.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

import soundfile as sf

from app.llm.base import AsrResult, LlmResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


def write_durably(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so a power cut leaves the old file or the new one, whole.

    Written to a sibling, flushed to the disk, renamed over the target, and the rename itself
    flushed -- without the last step a crash can keep the new name pointing at nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def request_key(*parts: Any) -> str:
    """A stable key for a request, from JSON-serialisable parts."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def audio_key(path: Path | str) -> str:
    """The audio's samples, not its file bytes.

    A re-encoded FLAC carries a different encoder string in its header though every sample is
    the same -- measured: two FFmpeg versions normalising one source gave different files and
    identical samples. Keying on the bytes would re-buy a transcript for no reason.
    """
    data, rate = sf.read(str(path), dtype="int16")
    digest = hashlib.sha256(f"{rate}:{data.shape}:".encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


class Checkpoint:
    """One run's paid results, one small JSON file each, under ``<work_dir>/checkpoint/``.

    Safe to use from the segment workers at once: every entry is its own file and is written
    whole or not at all.
    """

    def __init__(self, work_dir: Path) -> None:
        self.root = work_dir / "checkpoint"
        self._lock = threading.Lock()
        self.reused: dict[str, int] = {}

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.json"

    def load(self, kind: str, key: str) -> dict[str, Any] | None:
        """The stored entry, or ``None`` when there is none or it cannot be read.

        An unreadable entry is a miss, not an error: at worst the call is paid for again.
        """
        try:
            payload = json.loads(self._path(kind, key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        with self._lock:
            self.reused[kind] = self.reused.get(kind, 0) + 1
        return payload

    def save(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        """Store an entry durably. A failure is logged and costs only the reuse, never the run."""
        try:
            write_durably(
                self._path(kind, key), json.dumps(payload, ensure_ascii=False, default=str)
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("ingest_checkpoint_write_failed", kind=kind, error=str(exc))

    # --- typed wrappers -----------------------------------------------------------------------

    def load_asr(self, key: str) -> AsrResult | None:
        payload = self.load("asr", key)
        if payload is None:
            return None
        try:
            # Spent before the interruption, and already on record there: nothing is spent now.
            return AsrResult(**{**payload, "estimated_cost_usd": Decimal(0)})
        except TypeError:
            return None

    def save_asr(self, key: str, result: AsrResult) -> None:
        self.save("asr", key, dataclasses.asdict(result))

    def load_completion(self, key: str) -> LlmResult | None:
        payload = self.load("completion", key)
        if payload is None:
            return None
        try:
            return LlmResult(**{**payload, "estimated_cost_usd": Decimal(0)})
        except TypeError:
            return None

    def save_completion(self, key: str, result: LlmResult) -> None:
        self.save("completion", key, dataclasses.asdict(result))
