"""Ingestion endpoints: upload audio, poll progress, and stream live events."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_object_storage, get_session_factory, require_auth
from app.config import Settings
from app.services.ingest import manager, run_pipeline
from app.storage.base import ObjectStorage
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_auth)])

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"}


def _slugify(text: str) -> str:
    """Generate a clean slug for episode IDs."""
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", clean)
    return slug[:50] or f"ep_{int(time.time())}"


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_ingestion(
    file: UploadFile = File(...),
    episode_title: str = Form(...),
    show_id: str = Form("podcast"),
    episode_id: str = Form(""),
    settings: Settings = Depends(get_config),
    storage: ObjectStorage = Depends(get_object_storage),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    """Start asynchronous ingestion job from uploaded audio file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="missing file name"
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unsupported audio format '{ext}'. Allowed: {allowed}",
        )

    # Slugify whichever id we end up with, never just the generated one: this value becomes a
    # directory name under the work root and a prefix of every object key, so a caller-supplied
    # "../.." would otherwise write -- and later delete -- outside the tree entirely.
    final_episode_id = _slugify(episode_id) if episode_id.strip() else _slugify(episode_title)

    # Prepare temporary directory. run_pipeline removes it when the job finishes.
    work_dir = settings.ingest.work_root / f"{final_episode_id}_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    dest_audio_path = work_dir / f"source_audio{ext}"
    with open(dest_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = manager.create_job(
        episode_id=final_episode_id,
        show_id=show_id.strip() or "podcast",
        title=episode_title.strip(),
        audio_path=dest_audio_path,
        work_dir=work_dir,
    )

    # Launch background thread
    worker = threading.Thread(
        target=run_pipeline,
        args=(job, session_factory, storage, settings),
        daemon=True,
    )
    worker.start()

    logger.info("ingest_job_started", job_id=job.job_id, episode_id=job.episode_id)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "episode_id": job.episode_id,
        "title": job.title,
    }


@router.get("/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Get current status, stage, progress, and logs of an ingestion job."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ingestion job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "active_segments": job.active_segments,
        "total_segments": job.total_segments,
        "error": job.error,
        "episode_id": job.episode_id,
        "show_id": job.show_id,
        "title": job.title,
        "logs": [
            {"timestamp": entry.timestamp, "level": entry.level, "message": entry.message}
            for entry in job.logs
        ],
    }


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events (SSE) stream for live real-time browser debugging."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ingestion job not found")

    async def event_generator() -> AsyncIterator[str]:
        q: asyncio.Queue[str] = asyncio.Queue()
        # The pipeline emits from a worker thread, so it needs this queue's loop to hand work back.
        listener = (asyncio.get_running_loop(), q)
        job.listeners.append(listener)

        try:
            # Replay historical logs
            for log in job.logs:
                payload = {
                    "type": "log",
                    "timestamp": log.timestamp,
                    "level": log.level,
                    "message": log.message,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            prog_payload = {
                "type": "progress",
                "stage": job.stage,
                "progress": job.progress,
                "active_segments": job.active_segments,
                "total_segments": job.total_segments,
            }
            yield f"data: {json.dumps(prog_payload)}\n\n"

            while True:
                if job.status in ("completed", "failed") and q.empty():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=2.0)
                    yield f"data: {msg}\n\n"
                except TimeoutError:
                    # Keep-alive comment
                    yield ": keepalive\n\n"
        finally:
            if listener in job.listeners:
                job.listeners.remove(listener)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
