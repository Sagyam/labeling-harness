"""Voices: where an anonymous voice speaks, its solo clips, and the owner's verdicts (D99)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_config,
    get_object_storage,
    get_session,
    get_voice_embedder,
    require_auth,
)
from app.api.schemas import (
    VoiceClipOut,
    VoiceEpisodeOut,
    VoiceOut,
    VoiceVerdictIn,
    VoiceVerdictOut,
)
from app.api.serializers import audio_url
from app.config import Settings
from app.services.voice_clips import VoiceClip, VoiceError, record_verdict, voice_page
from app.services.voiceprint import VoiceEmbedder
from app.storage import ObjectStorage

router = APIRouter(tags=["voices"], dependencies=[Depends(require_auth)])

VOICE_PATTERN = "^v[0-9]{3,6}$"


def _clip(clip: VoiceClip) -> VoiceClipOut:
    return VoiceClipOut(
        segment_id=clip.segment_id,
        external_id=clip.external_id,
        episode_external_id=clip.episode_external_id,
        pot=clip.pot,
        duration_seconds=clip.duration_seconds,
        text=clip.text,
        verdict=clip.verdict,
        audio_url=audio_url(clip.segment_id),
        start=clip.start,
        end=clip.end,
        whole=clip.whole,
    )


@router.get("/voices/{voice}", response_model=VoiceOut)
def get_voice(
    voice: str,
    session: Session = Depends(get_session),
    episode: str | None = Query(default=None, description="only this episode's clips"),
    limit: int = Query(default=60, ge=1, le=500),
) -> VoiceOut:
    """Where a voice speaks, and the clips the diarization hears it alone in, longest first."""
    if not re.match(VOICE_PATTERN, voice):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown voice")
    try:
        page = voice_page(session, voice, episode=episode, limit=limit)
    except VoiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return VoiceOut(
        voice=page.voice,
        talk_seconds=page.talk_seconds,
        episodes=[VoiceEpisodeOut(**e.__dict__) for e in page.episodes],
        clips=[_clip(c) for c in page.clips],
        total_clips=page.total_clips,
        confirmed=page.confirmed,
        rejected=page.rejected,
        print_source=page.print_source,
        reference=_clip(page.reference) if page.reference else None,
    )


@router.post("/voices/{voice}/clips/{segment_id}", response_model=VoiceVerdictOut)
def post_verdict(
    voice: str,
    segment_id: int,
    body: VoiceVerdictIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
    storage: ObjectStorage = Depends(get_object_storage),
    embedder: VoiceEmbedder = Depends(get_voice_embedder),
) -> VoiceVerdictOut:
    """Record whether a clip is this voice alone; a confirmed clip joins the voice's print."""
    try:
        row = record_verdict(
            session,
            voice,
            segment_id,
            body.verdict,
            annotator=body.annotator or settings.labels.default_annotator,
            storage=storage,
            embedder=embedder,
        )
        page = voice_page(session, voice, limit=1)
    except VoiceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    session.commit()
    return VoiceVerdictOut(
        voice=voice,
        segment_id=segment_id,
        verdict=row.verdict,
        embedded=row.embedding_jsonb is not None,
        confirmed=page.confirmed,
        rejected=page.rejected,
    )
