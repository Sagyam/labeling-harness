"""The distillation corpus's pure rules (D101).

Unlabelled audio for roadmap §B is a corpus of files, not episodes: it never enters Postgres, the
queue or a paid route. These functions decide what a source is, which sources are refused before
any work is done, and whether a source sounds like one of gold's voices. The I/O (ffmpeg, the VAD,
the voiceprint model, the database) lives in ``scripts/prepare_distill_audio.py`` and
``scripts/screen_distill_audio.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.youtube import VIDEO_ID_RE

#: What the owner may drop into the incoming folder: anything ffmpeg reads that yt-dlp or a
#: converter writes. Stage 1 turns all of it into 16 kHz mono FLAC (invariant 7).
AUDIO_SUFFIXES = (".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".flac", ".aac")
INFO_SUFFIX = ".info.json"


@dataclass(frozen=True)
class SourceInput:
    """One audio file and, when yt-dlp wrote one, its info JSON (``<stem>.info.json``)."""

    audio: Path
    info: Path | None


def pair_inputs(paths: Iterable[Path]) -> list[SourceInput]:
    """Pair every audio file with the info JSON of the same stem, sorted by audio path."""
    paths = list(paths)
    infos = {str(p)[: -len(INFO_SUFFIX)]: p for p in paths if p.name.endswith(INFO_SUFFIX)}
    audio = sorted(p for p in paths if p.suffix.lower() in AUDIO_SUFFIXES)
    return [SourceInput(audio=a, info=infos.get(str(a.with_suffix("")))) for a in audio]


@dataclass(frozen=True)
class Source:
    """A recording and its provenance. ``source_id`` names its folder in the corpus."""

    source_id: str
    video_id: str | None
    channel: str | None
    channel_id: str | None
    playlist: str | None
    title: str | None
    duration: float | None
    audio_name: str
    audio_sha256: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_source(audio: Path, info: Mapping[str, Any] | None, *, audio_sha256: str) -> Source:
    """A source from yt-dlp's info JSON; named by its audio's hash when it carries no video id.

    Args:
        audio: The downloaded file.
        info: The parsed info JSON, or None when there was none.
        audio_sha256: The audio file's digest, for sources with no usable video id.
    """
    info = info or {}
    raw_id = str(info.get("id") or "")
    video_id = raw_id if VIDEO_ID_RE.match(raw_id) else None
    duration = info.get("duration")
    hex_digest = audio_sha256.rsplit(":", 1)[-1]  # app.utils.hashing prefixes "sha256:"
    return Source(
        source_id=f"yt-{video_id}" if video_id else f"file-{hex_digest[:12]}",
        video_id=video_id,
        channel=info.get("channel") or info.get("uploader"),
        channel_id=info.get("channel_id") or info.get("uploader_id"),
        playlist=info.get("playlist_title") or info.get("playlist"),
        title=info.get("title"),
        duration=float(duration) if isinstance(duration, int | float) else None,
        audio_name=audio.name,
        audio_sha256=audio_sha256,
    )


def refusal(
    source: Source, *, known_video_ids: set[str], blocked_channels: Sequence[str]
) -> str | None:
    """Why a source may not enter the corpus before any work is done, or None when it may.

    Args:
        source: The source.
        known_video_ids: Video ids of every episode already in the harness: labelled audio, and
            possibly gold or val (D76).
        blocked_channels: Channel names or ids gold was drawn from, compared case-insensitively.
            A name blocks every channel whose name contains it: a missed gold channel costs
            more than a refused clean one. An id must match whole.
    """
    if source.video_id and source.video_id in known_video_ids:
        return f"video {source.video_id} is already an episode in the harness"
    blocked = [b.strip().casefold() for b in blocked_channels if b.strip()]
    name = (source.channel or "").casefold()
    channel_id = (source.channel_id or "").strip().casefold()
    for b in blocked:
        if b == channel_id or (name and b in name):
            which = source.channel or source.channel_id
            return f"channel {which!r} is blocked by {b!r} (distill.blocked_channels)"
    return None


# --- the clip manifest --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipSpan:
    """One clip cut from a source: what the manifest keeps of ``extract_clips``'s output."""

    segment_id: str
    start: float
    end: float
    checksum: str


def source_dir(source_id: str) -> str:
    """A source's folder, relative to the corpus root."""
    return f"sources/{source_id}"


def clip_rows(source: Source, clips: Sequence[ClipSpan]) -> list[dict[str, Any]]:
    """The corpus manifest's rows for one source's clips, paths relative to the corpus root."""
    return [
        {
            "segment_id": c.segment_id,
            "source_id": source.source_id,
            "path": f"{source_dir(source.source_id)}/clips/{c.segment_id}.flac",
            "start": c.start,
            "end": c.end,
            "duration": round(c.end - c.start, 3),
            "checksum": c.checksum,
            "channel": source.channel,
        }
        for c in clips
    ]


# --- the voiceprint screen ----------------------------------------------------------------------


def window_starts(duration: float, window: float) -> list[float]:
    """Start times of the non-overlapping windows that fit wholly inside a clip."""
    count = int(duration // window + 1e-9)
    return [round(i * window, 6) for i in range(count)]


@dataclass
class ScreenResult:
    """A source's verdict against gold's voices.

    ``voice`` and ``seconds`` name the gold voice with the most windows at or above the
    threshold; ``windows`` are that voice's closest windows as ``(clip_id, start, similarity)``,
    for the owner to hear.
    """

    verdict: str
    voice: str | None
    seconds: float
    windows: list[tuple[str, float, float]] = field(default_factory=list)


def screen_source(
    windows: Sequence[tuple[str, float, np.ndarray | None]],
    gold: Mapping[str, np.ndarray],
    *,
    threshold: float,
    window_seconds: float,
    min_seconds: float,
    keep: int = 5,
) -> ScreenResult:
    """Quarantine a source with ``min_seconds`` or more of windows close to one gold voice.

    Args:
        windows: ``(clip_id, start, unit embedding)`` per window; None for one too short to embed.
        gold: Unit centroid per gold voice id.
        threshold: Cosine similarity at which a window sounds like a voice.
        window_seconds: Length of one window.
        min_seconds: Seconds of close windows, for one voice, that quarantine the source.
        keep: How many of the closest windows to report.

    Raises:
        ValueError: There are no gold voices; a screen with nothing to compare would clear every
            source.
    """
    if not gold:
        raise ValueError("no gold voices to screen against")
    voices = list(gold)
    centroids = np.stack([gold[v] for v in voices])
    close: dict[str, list[tuple[str, float, float]]] = {v: [] for v in voices}
    for clip_id, start, vector in windows:
        if vector is None:
            continue
        sims = centroids @ vector
        for i in np.flatnonzero(sims >= threshold):
            close[voices[i]].append((clip_id, start, float(sims[i])))
    voice = max(voices, key=lambda v: (len(close[v]), max((w[2] for w in close[v]), default=0.0)))
    seconds = len(close[voice]) * window_seconds
    best = sorted(close[voice], key=lambda w: -w[2])[:keep]
    if not close[voice]:
        return ScreenResult("clear", None, 0.0)
    return ScreenResult("quarantine" if seconds >= min_seconds else "clear", voice, seconds, best)


def gold_voice_centroids(
    runs: Sequence[tuple[Mapping[str, str] | None, Mapping[str, Sequence[float]] | None]],
) -> dict[str, np.ndarray]:
    """One unit centroid per voice heard in gold's episodes: the mean of its runs' centroids.

    Args:
        runs: Per diarization run of an episode holding a gold clip, its ``voices_jsonb`` (label to
            voice id) and ``embeddings_jsonb`` (label to the diarizer's embedding). A speaker with
            no voice or no embedding gives nothing.
    """
    sums: dict[str, np.ndarray] = {}
    for voices, embeddings in runs:
        for label, voice in (voices or {}).items():
            vector = np.asarray((embeddings or {}).get(label) or [], dtype=float)
            norm = float(np.linalg.norm(vector)) if vector.size else 0.0
            if voice and norm > 0:
                sums[voice] = sums.get(voice, 0) + vector / norm
    return {v: s / np.linalg.norm(s) for v, s in sums.items() if np.linalg.norm(s) > 0}
