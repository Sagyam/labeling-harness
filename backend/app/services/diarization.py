"""Client for the serverless GPU speaker diarizer (D79).

The diarizer is pyannote's ``speaker-diarization-community-1`` behind a Modal web endpoint
(``scripts/modal_diarize.py``). It takes the whole normalised episode FLAC and answers with one
entry of the file :mod:`app.services.diarization_import` reads: ``turns``, ``labels`` and
``embeddings``. The harness sends audio and stores the answer; torch never enters the backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Modal answers a request that runs past 150 s with a 303 to a result URL that blocks for up to
# another 150 s, so each redirect buys that much more time. 20 of them is 50 minutes.
_MAX_REDIRECTS = 20


class DiarizationServiceError(RuntimeError):
    """The diarization service refused the request or failed."""


def declared_speaker_count(metadata: Mapping[str, Any] | None) -> int | None:
    """How many speakers the episode metadata declares, or ``None`` when it declares none.

    ``speaker_count`` is the ingest form's row count and wins: a row left blank is not in
    ``speakers`` but is still someone talking. Episodes from before it existed fall back to the
    size of ``speakers``, which is what the Colab notebook used as ``num_speakers`` for the runs
    D78's 97.3% agreement was measured on.
    """
    metadata = metadata or {}
    count = metadata.get("speaker_count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return count
    speakers = metadata.get("speakers")
    if isinstance(speakers, (Mapping, list)) and speakers:
        return len(speakers)
    return None


def diarize_audio(
    audio: Path | bytes,
    *,
    num_speakers: int | None = None,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Diarize one episode's audio.

    Args:
        audio: The 16 kHz mono FLAC, as a path or its bytes.
        num_speakers: Fixes the speaker count; ``None`` lets pyannote choose.
        settings: Application settings.
        client: Reused HTTP client, for tests.

    Returns:
        The diarizer's answer, or ``None`` when diarization is disabled or has no endpoint.

    Raises:
        DiarizationServiceError: The service answered with anything but 200.
    """
    settings = settings or get_settings()
    conf = settings.diarization
    if not conf.enabled or not conf.endpoint_url:
        return None

    body = audio if isinstance(audio, bytes) else audio.read_bytes()
    data = {"num_speakers": str(num_speakers)} if num_speakers and num_speakers > 0 else {}
    # A Modal proxy-auth token: its ``wk-`` id and ``ws-`` secret joined by a dot (D79).
    headers = {"Authorization": f"Bearer {conf.auth_token}"} if conf.auth_token else {}

    logger.info("diarization_request", bytes=len(body), num_speakers=num_speakers or "auto")
    own_client = client is None
    http = client or httpx.Client(
        timeout=conf.timeout_seconds, follow_redirects=True, max_redirects=_MAX_REDIRECTS
    )
    try:
        resp = http.post(
            conf.endpoint_url,
            data=data,
            files={"file": ("audio.flac", body, "audio/flac")},
            headers=headers,
        )
    finally:
        if own_client:
            http.close()
    if resp.status_code != 200:
        raise DiarizationServiceError(
            f"diarization service answered {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()
