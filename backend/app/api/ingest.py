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
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_object_storage, get_session_factory, require_auth
from app.config import Settings
from app.services.ingest import manager, run_pipeline
from app.services.youtube import (
    InvalidYouTubeUrl,
    VideoInfo,
    VideoTooLong,
    YouTubeUnavailable,
    canonical_url,
    check_duration,
    probe,
)
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


class YouTubeProbeIn(BaseModel):
    """A pasted YouTube URL, in whatever shape the annotator copied it."""

    url: str = Field(min_length=1, max_length=2048)


class YouTubeIngestIn(YouTubeProbeIn):
    """A YouTube URL plus the metadata the upload form asks for.

    Every field but the URL is optional: an omitted title is taken from the video, and an omitted
    episode id is slugified from whichever title wins.
    """

    episode_title: str = ""
    show_id: str = "podcast"
    episode_id: str = ""


class YouTubeProbeOut(BaseModel):
    """What the browser needs to prefill the form and show what it is about to ingest."""

    video_id: str
    url: str
    title: str
    duration_seconds: float | None = None
    uploader: str | None = None
    thumbnail: str | None = None
    upload_date: str | None = None
    is_live: bool = False
    suggested_episode_id: str


def _probe_or_http_error(url: str, settings: Settings) -> VideoInfo:
    """Look a video up, mapping every failure onto the status code it deserves.

    A bad URL and an over-long video are the caller's problem (422); a yt-dlp or network failure
    is not (502), and telling the two apart is what makes the modal's error message actionable.
    """
    try:
        info = probe(url, settings=settings)
    except InvalidYouTubeUrl as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except YouTubeUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if info.is_live:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="live streams cannot be ingested; wait for the recording to be published",
        )
    try:
        check_duration(info, settings=settings)
    except VideoTooLong as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return info


@router.post("/youtube/probe")
async def probe_youtube(
    body: YouTubeProbeIn,
    settings: Settings = Depends(get_config),
) -> YouTubeProbeOut:
    """Read a video's metadata without downloading it, so the form can prefill itself.

    Cheap and side-effect free: no file is written and no job is created. It also front-loads
    every rejection the ingest endpoint would make, so the annotator learns a video is too long
    before committing to it rather than after.
    """
    info = _probe_or_http_error(body.url, settings)
    return YouTubeProbeOut(
        video_id=info.video_id,
        url=info.url,
        title=info.title,
        duration_seconds=info.duration_seconds,
        uploader=info.uploader,
        thumbnail=info.thumbnail,
        upload_date=info.upload_date,
        is_live=info.is_live,
        suggested_episode_id=_slugify(info.title),
    )


@router.post("/youtube", status_code=status.HTTP_202_ACCEPTED)
async def start_youtube_ingestion(
    body: YouTubeIngestIn,
    settings: Settings = Depends(get_config),
    storage: ObjectStorage = Depends(get_object_storage),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    """Start an ingestion job that fetches its own audio from a YouTube URL.

    The metadata lookup happens here rather than on the worker, so a bad URL, a live stream or an
    over-long video is a 422 on this request instead of a job that fails a minute later. The
    download itself is the job's first act.
    """
    info = _probe_or_http_error(body.url, settings)

    title = body.episode_title.strip() or info.title
    # Slugified for the same reason the upload path slugifies: this becomes a directory name under
    # the work root and a prefix of every object key.
    final_episode_id = _slugify(body.episode_id) if body.episode_id.strip() else _slugify(title)

    work_dir = settings.ingest.work_root / f"{final_episode_id}_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job = manager.create_job(
        episode_id=final_episode_id,
        show_id=body.show_id.strip() or "podcast",
        title=title,
        work_dir=work_dir,
        source_url=canonical_url(body.url),
    )

    worker = threading.Thread(
        target=run_pipeline,
        args=(job, session_factory, storage, settings),
        daemon=True,
    )
    worker.start()

    logger.info(
        "ingest_job_started",
        job_id=job.job_id,
        episode_id=job.episode_id,
        source="youtube",
        video_id=info.video_id,
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "episode_id": job.episode_id,
        "title": job.title,
        "source_url": job.source_url,
    }


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
        work_dir=work_dir,
        audio_path=dest_audio_path,
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
