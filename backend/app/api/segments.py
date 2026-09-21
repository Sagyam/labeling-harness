"""Segment endpoints, including range-capable audio streaming."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_config, get_object_storage, get_session, require_auth
from app.api.schemas import BulkDeleteSegmentsIn, BulkDeleteSegmentsOut, SegmentOut
from app.api.serializers import serialize_segment
from app.config import Settings
from app.models import AsrHypothesis, AuditLog, Segment
from app.storage import ObjectNotFound, ObjectStorage, delete_objects

router = APIRouter(tags=["segments"], dependencies=[Depends(require_auth)])

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
#: Cap on a single range response, so a malformed request cannot ask for an unbounded read.
MAX_RANGE_BYTES = 8 * 1024 * 1024


def _load_segment(session: Session, segment_id: int) -> Segment:
    segment = session.scalars(
        sa.select(Segment)
        .options(
            selectinload(Segment.episode),
            selectinload(Segment.scores),
            selectinload(Segment.hypotheses).selectinload(AsrHypothesis.words),
            selectinload(Segment.hypotheses).selectinload(AsrHypothesis.system),
        )
        .where(Segment.id == segment_id)
    ).first()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="segment not found")
    return segment


#: Characters kept in the ASCII ``filename`` fallback. Anything else becomes an underscore.
_ASCII_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def content_disposition(external_id: str, *, segment_id: int) -> str:
    """``Content-Disposition`` for a clip download, safe for a non-ASCII external id.

    Header values go out latin-1 encoded, so an episode slug carrying Devanagari raises inside the
    server and the player gets a 500 instead of audio -- for every clip of that episode, while the
    JSON and peaks endpoints beside it work fine. RFC 6266 exists for this: send an ASCII
    ``filename`` that any client can read plus a percent-encoded ``filename*`` that carries the
    real name, and let the client prefer the second.
    """
    fallback = _ASCII_FILENAME.sub("_", external_id).strip("_") or f"segment-{segment_id}"
    encoded = quote(f"{external_id}.flac", safe="")
    return f"inline; filename=\"{fallback}.flac\"; filename*=UTF-8''{encoded}"


def parse_range(header: str, size: int) -> tuple[int, int]:
    """Parse a single ``bytes=`` range against a known object size.

    Returns:
        Inclusive ``(start, end)`` byte offsets.

    Raises:
        HTTPException: 416 when the header is malformed or lies outside the object.
    """
    match = _RANGE.match(header.strip())
    if not match:
        raise HTTPException(
            status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
            detail="malformed Range header",
            headers={"Content-Range": f"bytes */{size}"},
        )
    first, last = match.group(1), match.group(2)
    if not first and not last:
        raise HTTPException(
            status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
            detail="malformed Range header",
            headers={"Content-Range": f"bytes */{size}"},
        )
    if not first:  # suffix range: the last N bytes
        length = min(int(last), size)
        start, end = size - length, size - 1
    else:
        start = int(first)
        end = int(last) if last else size - 1
    end = min(end, size - 1, start + MAX_RANGE_BYTES - 1)
    if start > end or start >= size:
        raise HTTPException(
            status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
            detail="range outside object",
            headers={"Content-Range": f"bytes */{size}"},
        )
    return start, end


@router.get("/segments/{segment_id}", response_model=SegmentOut)
def get_segment(segment_id: int, session: Session = Depends(get_session)) -> SegmentOut:
    """Full segment payload: hypotheses, scores, flags and the current label."""
    return serialize_segment(session, _load_segment(session, segment_id))


@router.get("/segments/{segment_id}/audio")
def get_segment_audio(
    segment_id: int,
    range: str | None = Header(default=None),
    session: Session = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
) -> Response:
    """Stream a clip, honouring HTTP range requests so the player can seek.

    Streaming is used for both storage backends rather than redirecting to a presigned URL: it is
    one code path, it works with the local filesystem backend, and it keeps clip URLs stable.
    """
    segment = _load_segment(session, segment_id)
    key = segment.clip_object_key
    try:
        size = storage.size(key)
    except ObjectNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="clip not found in storage"
        ) from exc

    common = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": content_disposition(segment.external_id, segment_id=segment.id),
        "Cache-Control": "private, max-age=3600",
    }
    if range is None:
        return Response(
            content=storage.get_bytes(key),
            media_type="audio/flac",
            headers={**common, "Content-Length": str(size)},
        )

    start, end = parse_range(range, size)
    chunk = storage.read_range(key, start, end)
    return Response(
        content=chunk,
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type="audio/flac",
        headers={
            **common,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(len(chunk)),
        },
    )


@router.get("/segments/{segment_id}/peaks")
def get_segment_peaks(
    segment_id: int,
    session: Session = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
) -> Response:
    """Precomputed waveform peaks. The UI never decodes audio to draw a waveform."""
    segment = _load_segment(session, segment_id)
    if not segment.peaks_object_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no peaks for this segment"
        )
    try:
        payload = storage.get_bytes(segment.peaks_object_key)
    except ObjectNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="peaks not found in storage"
        ) from exc
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.delete("/segments/{segment_id}")
def delete_segment(
    segment_id: int,
    session: Session = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
    settings: Settings = Depends(get_config),
) -> dict[str, Any]:
    """Delete a single segment and its associated tasks, labels, and storage files."""
    segment = session.scalar(sa.select(Segment).where(Segment.id == segment_id))
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="segment not found")

    external_id = segment.external_id
    delete_objects(storage, segment.clip_object_key, segment.peaks_object_key)
    session.add(_delete_audit(segment, actor=settings.labels.default_annotator))
    session.delete(segment)
    session.flush()
    return {"deleted": True, "segment_id": segment_id, "external_id": external_id}


@router.post("/segments/bulk-delete", response_model=BulkDeleteSegmentsOut)
def bulk_delete_segments(
    body: BulkDeleteSegmentsIn,
    session: Session = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
    settings: Settings = Depends(get_config),
) -> BulkDeleteSegmentsOut:
    """Delete several segments in one transaction: all of them, or none.

    A gold clip anywhere in the batch refuses the whole batch with 409 (D94): deleting is a
    filter, and gold must not be filtered by the same quick hand that filters train. Storage
    objects go only after the commit, so a refused or failed batch leaves every clip playable.
    """
    ids = list(dict.fromkeys(body.segment_ids))
    segments = session.scalars(sa.select(Segment).where(Segment.id.in_(ids))).all()
    missing = sorted(set(ids) - {s.id for s in segments})
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"segments not found: {missing}"
        )
    gold = sorted(s.id for s in segments if s.pot == "gold")
    if gold:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"gold clips cannot be bulk-deleted: {gold}",
        )

    keys = [(s.clip_object_key, s.peaks_object_key) for s in segments]
    for segment in segments:
        session.add(_delete_audit(segment, actor=settings.labels.default_annotator, bulk=True))
        session.delete(segment)
    session.commit()
    for clip_key, peaks_key in keys:
        delete_objects(storage, clip_key, peaks_key)
    return BulkDeleteSegmentsOut(deleted=ids, count=len(ids))


def _delete_audit(segment: Segment, *, actor: str, bulk: bool = False) -> AuditLog:
    """The ``audit_logs`` row a segment deletion leaves behind; it outlives the segment."""
    old_values: dict[str, Any] = {
        "external_id": segment.external_id,
        "episode_id": segment.episode_id,
        "pipeline_status": segment.pipeline_status,
        "pot": segment.pot,
    }
    if bulk:
        old_values["bulk"] = True
    return AuditLog(
        entity_type="segments",
        entity_id=str(segment.id),
        action="delete",
        actor=actor,
        old_values_jsonb=old_values,
        new_values_jsonb=None,
    )
