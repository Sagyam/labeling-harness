"""Ingestion endpoints: upload audio, poll progress, and stream live events."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import time
import unicodedata
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_object_storage, get_session_factory, require_auth
from app.config import Settings
from app.services.episode_meta import MetadataEditError, check_genre, check_topic
from app.services.ingest import manager
from app.services.speaker_meta import MAX_SPEAKERS, strip_speaker_pii
from app.services.youtube import (
    InvalidYouTubeUrl,
    VideoInfo,
    VideoTooLong,
    YouTubeBotDetected,
    YouTubeUnavailable,
    canonical_url,
    check_duration,
    probe,
)
from app.storage.base import ObjectStorage
from app.utils.logging import get_logger
from app.utils.rate_limit import provider_gate

logger = get_logger(__name__)


def _restore_queue(settings: Settings = Depends(get_config)) -> None:
    """Load the jobs a previous server process left behind, before any endpoint reads the queue.

    Otherwise they appear only once something new is submitted, and a restart looks like an
    empty history with nothing to retry.
    """
    manager.init_state(settings.ingest.work_root)


router = APIRouter(
    prefix="/ingest",
    tags=["ingest"],
    dependencies=[Depends(require_auth), Depends(_restore_queue)],
)

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"}


def _slugify(text: str) -> str:
    """Generate a clean slug for episode IDs.

    Combining marks are kept alongside word characters. ``\\w`` matches Devanagari consonants but
    not the matras attached to them (Unicode category ``M*``), so a plain ``[^\\w\\s-]`` filter
    silently rewrites the title: हिमालयन becomes हमलयन. A slug that survives as nothing but
    separators falls back to the timestamp, rather than being stored as ``_``.
    """
    kept = "".join(
        character
        if (
            character.isalnum()
            or character in "_-"
            or character.isspace()
            or unicodedata.category(character).startswith("M")
        )
        else ""
        for character in text
    )
    slug = re.sub(r"[-\s]+", "_", kept.strip().lower())
    return slug[:50].strip("_") or f"ep_{int(time.time())}"


def _episode_metadata(
    genre: str,
    topic: str,
    speakers_json: str,
    speaker_count: int = 0,
    clip_minutes: int | None = None,
) -> dict[str, Any]:
    """Build the episode's free-form metadata from the form's optional fields.

    ``speaker_count`` is the number of speaker rows on the form, kept apart from ``speakers``
    because a row whose demographics were left blank is not sent there but is still a person the
    diarizer must find (D79).

    ``clip_minutes`` is present only when the annotator ticked the clip box (D103): stage 1 then
    keeps the recording's first that many minutes, and records what it was cut from.

    Genre and topic must be on their closed lists (D57, D102): anything else is a 422, raised
    before the caller has written anything to disk.

    The speaker block is run through the allowlist here as well as at the importer. The importer
    is the guarantee; this is so a name pasted into `speakers_json` by a hand-rolled client is
    dropped before it is written to the job's `episode.json` on disk (D56).
    """
    try:
        checked_genre, checked_topic = check_genre(genre.strip()), check_topic(topic.strip())
    except MetadataEditError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metadata: dict[str, Any] = {}
    if checked_genre:
        metadata["genre"] = checked_genre
    if checked_topic:
        metadata["topic"] = checked_topic
    if speakers_json.strip():
        # A malformed speaker block loses the metadata, never the ingest.
        with contextlib.suppress(Exception):
            metadata["speakers"] = json.loads(speakers_json)
    if speaker_count > 0:
        metadata["speaker_count"] = speaker_count
    if clip_minutes is not None:
        metadata["clip"] = {"max_seconds": clip_minutes * 60}
    return strip_speaker_pii(metadata)


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
    genre: str = ""
    topic: str = ""
    speakers_json: str = ""
    speaker_count: int = Field(default=0, ge=0, le=MAX_SPEAKERS)
    #: Opt-in: keep only the video's first this many minutes (D103). Absent keeps all of it.
    clip_minutes: int | None = Field(default=None, ge=1)


class RetryAllIn(BaseModel):
    """Filter for batch retry."""

    status_filter: str | None = None  # 'backlog', 'failed', or None for all retryable


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
    #: The spend guard. A longer video is ingestible only clipped to this or less (D103), so the
    #: probe reports it rather than refusing, and the form says what to do.
    max_duration_seconds: float


def _probe_or_http_error(url: str, settings: Settings) -> VideoInfo:
    """Look a video up, mapping every failure onto the status code it deserves.

    A bad URL or a live stream is the caller's problem (422); a yt-dlp or network failure is not
    (502), and telling the two apart is what makes the modal's error message actionable. Length
    is judged separately, by :func:`_check_length`, because a clip can fix it (D103).
    """
    # A lookup is one request, not a download, so it does not queue for the download slot; but a
    # bot check it hits cools the downloads down like one a download hit (D88).
    try:
        info = probe(url, settings=settings)
    except YouTubeBotDetected as exc:
        provider_gate("youtube", settings.ingest.youtube.limit).throttled()
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
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
    return info


def _check_length(info: VideoInfo, settings: Settings, clip_minutes: int | None) -> None:
    """Refuse, with a 422, a video whose transcribed part is over the spend guard."""
    try:
        clip_seconds = clip_minutes * 60 if clip_minutes is not None else None
        check_duration(info, settings=settings, clip_seconds=clip_seconds)
    except VideoTooLong as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.post("/youtube/probe")
async def probe_youtube(
    body: YouTubeProbeIn,
    settings: Settings = Depends(get_config),
) -> YouTubeProbeOut:
    """Read a video's metadata without downloading it, so the form can prefill itself.

    Cheap and side-effect free: no file is written and no job is created. It also front-loads
    every rejection the ingest endpoint would make but one: a video over the length limit is
    reported with the limit, not refused, because clipping it makes it ingestible (D103).
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
        max_duration_seconds=settings.ingest.youtube.max_duration_seconds,
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
    _check_length(info, settings, body.clip_minutes)

    title = body.episode_title.strip() or info.title
    # Slugified for the same reason the upload path slugifies: this becomes a directory name under
    # the work root and a prefix of every object key.
    final_episode_id = _slugify(body.episode_id) if body.episode_id.strip() else _slugify(title)

    metadata = _episode_metadata(
        body.genre, body.topic, body.speakers_json, body.speaker_count, body.clip_minutes
    )

    work_dir = settings.ingest.work_root / f"{final_episode_id}_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job = manager.create_job(
        episode_id=final_episode_id,
        show_id=body.show_id.strip() or "podcast",
        title=title,
        work_dir=work_dir,
        source_url=canonical_url(body.url),
        metadata=metadata,
    )

    ahead = manager.submit(job, session_factory, storage, settings)

    logger.info(
        "ingest_job_queued",
        job_id=job.job_id,
        episode_id=job.episode_id,
        source="youtube",
        video_id=info.video_id,
        queued_behind=ahead,
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "episode_id": job.episode_id,
        "title": job.title,
        "source_url": job.source_url,
        "queue_position": ahead,
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_ingestion(
    file: UploadFile = File(...),
    episode_title: str = Form(...),
    show_id: str = Form("podcast"),
    episode_id: str = Form(""),
    genre: str = Form(""),
    topic: str = Form(""),
    speakers_json: str = Form(""),
    speaker_count: int = Form(0, ge=0, le=MAX_SPEAKERS),
    clip_minutes: int | None = Form(None, ge=1),
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

    metadata = _episode_metadata(genre, topic, speakers_json, speaker_count, clip_minutes)

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
        metadata=metadata,
    )

    # Launch background thread
    ahead = manager.submit(job, session_factory, storage, settings)

    logger.info(
        "ingest_job_queued",
        job_id=job.job_id,
        episode_id=job.episode_id,
        queued_behind=ahead,
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "episode_id": job.episode_id,
        "title": job.title,
        "queue_position": ahead,
    }


@router.get("")
async def list_queue() -> dict[str, Any]:
    """The running ingestion, upcoming queue, backlog, and past jobs."""
    return manager.queue_snapshot_rich()


@router.post("/retry-all")
async def retry_all_jobs(
    body: RetryAllIn | None = None,
    settings: Settings = Depends(get_config),
    storage: ObjectStorage = Depends(get_object_storage),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    """Retry all backlogged or failed jobs."""
    filter_status = body.status_filter if body else None
    results = manager.retry_all(filter_status, session_factory, storage, settings)
    return {
        "retried_count": len([r for r in results if r.get("success")]),
        "results": results,
    }


@router.post("/clear-past")
async def clear_past_jobs() -> dict[str, Any]:
    """Clear finished history."""
    count = manager.clear_past()
    return {"cleared_count": count}


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    settings: Settings = Depends(get_config),
    storage: ObjectStorage = Depends(get_object_storage),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
) -> dict[str, Any]:
    """Retry a failed, aborted, or backlogged job."""
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ingestion job not found")
    if job.status not in ("failed", "aborted", "backlog"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"job is {job.status} and cannot be retried",
        )
    ahead = manager.retry_job(job_id, session_factory, storage, settings)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "episode_id": job.episode_id,
        "title": job.title,
        "queue_position": ahead,
    }


@router.delete("/{job_id}")
async def cancel_or_delete_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued job or remove a finished/backlogged job."""
    try:
        return manager.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


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
        "scrammed": job.scrammed,
        "scram_reason": job.scram_reason,
        "episode_id": job.episode_id,
        "show_id": job.show_id,
        "title": job.title,
        "source_url": job.source_url,
        # None once the job is running or finished; a count of jobs ahead while it waits.
        "queue_position": manager.queue_position(job_id),
        "logs": [
            {"timestamp": entry.timestamp, "level": entry.level, "message": entry.message}
            for entry in job.logs
        ],
        # Segments dropped mid-run (D46). Present on a running job too, so a page reopened
        # halfway through shows what has already been lost rather than only at the end.
        "discarded_segments": [record.as_dict() for record in job.discarded],
        "discarded_by_system": job.discard_summary(),
        # Present once the run has finished. A page that reattaches to a job it was not watching
        # gets the outcome from here rather than from an event it missed.
        "summary": job.summary,
    }


@router.post("/{job_id}/scram")
async def scram_job(job_id: str) -> dict[str, Any]:
    """AZ-5. Stop a running ingestion at the next checkpoint.

    Named after the reactor scram it behaves like: one control, no arguments, and the only thing
    it does is stop. It is not a pause and there is no resume -- a scrammed run imports nothing
    (see :meth:`IngestJob.abort`), because half an episode is a differently-sampled episode, not
    a cheaper one.

    Returns immediately. The pipeline runs on a worker thread and stops itself at the next
    checkpoint; the requests it has already put on the wire are left to come back, so that the
    inference they bill for still reaches ``llm_requests``. Watch the job's SSE stream for the
    ``scram`` event, then ``aborted``.
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ingestion job not found")

    # Idempotent on purpose: this is the button someone hits twice.
    halted = job.scram(reason="AZ-5 pressed")
    return {
        "job_id": job.job_id,
        "scrammed": job.scrammed,
        "status": job.status,
        "stage": job.stage,
        "already_stopping": not halted,
        "detail": (
            "Halting at the next checkpoint. Nothing will be imported."
            if halted
            else "Already halting -- waiting on the requests still in flight."
            if job.scrammed and job.status == "processing"
            else f"Job is already {job.status}."
        ),
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

            for record in job.discarded:
                yield f"data: {json.dumps({'type': 'discard', 'segment': record.as_dict()})}\n\n"

            prog_payload = {
                "type": "progress",
                "stage": job.stage,
                "progress": job.progress,
                "active_segments": job.active_segments,
                "total_segments": job.total_segments,
            }
            yield f"data: {json.dumps(prog_payload)}\n\n"

            while True:
                if job.status in ("completed", "failed", "aborted") and q.empty():
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
