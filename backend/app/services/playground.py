"""The Models page playground (D85): a recording from the page, transcribed by a fine-tuned model.

The model runs on this machine's CPU in the ``playground`` sidecar, from the weights 04c's CPU
export writes: ``cpu/`` (weight-only int8) when the notebook accepted it, ``best/`` (bf16)
otherwise. The owner copies that folder next to the model's card, in ``data/models/asr/<slug>/``.

A recording is prepared exactly as ingest prepares an episode -- two-pass EBU R128 loudness, mono,
16 kHz through soxr -- so the model hears a microphone the way it heard its training audio. It is
never stored: no clip, no label, no object. The ``llm_requests`` row is the only trace (invariant
6), and the corpus is untouched (invariant 7 is about imported clips).
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import LlmRoutes
from app.llm.local_asr import LocalAsrClient
from app.services.ingest.audio import normalize_audio

#: Longest recording accepted. Flex is not a long-form model: it was fine-tuned on clips of at most
#: 21 s, and the page stops recording at this length.
MAX_SECONDS = 30.0
#: Largest upload read, before any decoding: minutes of any sane codec.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
#: The CPU export's subfolders, in the order they are tried when the card names neither.
WEIGHT_DIRS = ("cpu", "best")
_WEIGHT_FILES = ("model.int8.safetensors", "model.safetensors")


class PlaygroundError(ValueError):
    """The recording cannot be transcribed: empty, too long, or not audio."""


@dataclass(frozen=True)
class PlaygroundResult:
    text: str
    audio_s: float
    latency_ms: int | None
    weights: str
    dry_run: bool
    #: The sidecar's answer: decoder timings, whether the loop retry fired, the first decode.
    raw: dict[str, Any]


def cpu_weights(folder: Path, card: Mapping[str, Any] | None) -> str | None:
    """The subfolder of a model's folder the sidecar can load, or ``None`` when there is none.

    04c names it in the card's ``cpu.export`` (``cpu/`` or ``best/``). A card from before the
    CPU export, or a folder where only the other one was copied, falls back to whichever is
    there, int8 first.
    """
    cpu = (card or {}).get("cpu")
    preferred = str(cpu.get("export") or "").strip("/") if isinstance(cpu, Mapping) else ""
    order = [preferred, *WEIGHT_DIRS] if preferred in WEIGHT_DIRS else list(WEIGHT_DIRS)
    for name in dict.fromkeys(order):
        path = folder / name
        if (path / "config.json").is_file() and any((path / f).is_file() for f in _WEIGHT_FILES):
            return name
    return None


def prepare_audio(data: bytes, filename: str | None = None) -> tuple[bytes, float]:
    """The recording as 16 kHz mono FLAC, normalised as ingest normalises an episode.

    Returns:
        The FLAC bytes and the duration in seconds.

    Raises:
        PlaygroundError: Empty, too large, not decodable, or longer than :data:`MAX_SECONDS`.
    """
    if not data:
        raise PlaygroundError("the recording is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise PlaygroundError(f"the upload is over {MAX_UPLOAD_BYTES // 2**20} MB")
    suffix = Path(filename or "").suffix[:8] or ".bin"
    with tempfile.TemporaryDirectory(prefix="playground-") as tmp:
        src, dst = Path(tmp) / f"in{suffix}", Path(tmp) / "out.flac"
        src.write_bytes(data)
        try:
            seconds = normalize_audio(src, dst)
        except RuntimeError as exc:
            raise PlaygroundError(f"not a readable recording: {exc}") from exc
        if seconds > MAX_SECONDS:
            raise PlaygroundError(
                f"{seconds:.1f} s is longer than {MAX_SECONDS:.0f} s: Flex was fine-tuned on "
                "clips of at most 21 s and is not a long-form model"
            )
        if seconds < 0.1:
            raise PlaygroundError("the recording holds no audio")
        return dst.read_bytes(), seconds


def transcribe(
    session: Session,
    slug: str,
    weights: str,
    data: bytes,
    *,
    filename: str | None = None,
    config: LlmRoutes | None = None,
    client: Any = None,
) -> PlaygroundResult:
    """Prepare one recording and transcribe it with the model in folder ``slug``.

    The ``llm_requests`` row is flushed, not committed: the caller commits it whether or not the
    call succeeded, because a failed attempt is logged too.
    """
    flac, seconds = prepare_audio(data, filename)
    result = LocalAsrClient(session, config=config, client=client).transcribe(
        flac, model=slug, weights=weights
    )
    return PlaygroundResult(
        text=result.text,
        audio_s=round(seconds, 2),
        latency_ms=result.latency_ms,
        weights=weights,
        dry_run=result.dry_run,
        raw=result.raw or {},
    )
