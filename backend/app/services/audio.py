"""Clip format validation and waveform peak precomputation.

Clips must be 16 kHz mono FLAC. The source audio is already lossy, and re-encoding the exact audio
that will be trained on is not acceptable, so anything else is rejected at import before a row is
written.

Peaks are computed here, once, at import. The UI must never decode audio client-side to draw a
waveform -- that alone makes the editor feel sluggish by the fortieth segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import Settings, get_settings

#: Version of the peaks JSON payload, so the UI can reject a format it does not understand.
PEAKS_VERSION = 1


class ClipFormatError(ValueError):
    """A clip is missing, unreadable, or not 16 kHz mono FLAC."""


@dataclass(frozen=True)
class AudioInfo:
    """What the container says about a clip."""

    path: Path
    sample_rate: int
    channels: int
    format: str
    subtype: str | None
    frames: int
    duration_seconds: float


def probe(path: Path | str) -> AudioInfo:
    """Read a clip's container metadata without decoding the samples.

    Raises:
        ClipFormatError: The file is missing or not readable as audio.
    """
    path = Path(path)
    if not path.is_file():
        raise ClipFormatError(f"clip not found: {path}")
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises RuntimeError/LibsndfileError
        raise ClipFormatError(f"{path.name}: not readable as audio ({exc})") from exc
    return AudioInfo(
        path=path,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        format=str(info.format),
        subtype=str(info.subtype) if info.subtype else None,
        frames=int(info.frames),
        duration_seconds=float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
    )


def validate_clip(path: Path | str, settings: Settings | None = None) -> AudioInfo:
    """Validate that a clip is 16 kHz mono FLAC.

    Returns:
        The probed :class:`AudioInfo` when the clip is acceptable.

    Raises:
        ClipFormatError: With a message naming the file and the specific problem.
    """
    settings = settings or get_settings()
    expected = settings.importer
    info = probe(path)

    if info.format.upper() != expected.expected_format.upper():
        raise ClipFormatError(
            f"{info.path.name}: clips must be {expected.expected_format}, found {info.format}. "
            "The source is already lossy; re-encoding the audio you will train on is not "
            "acceptable, so convert upstream instead."
        )
    if info.sample_rate != expected.expected_sample_rate:
        raise ClipFormatError(
            f"{info.path.name}: clips must be {expected.expected_sample_rate} Hz, "
            f"found {info.sample_rate} Hz"
        )
    if info.channels != expected.expected_channels:
        channels = "mono" if expected.expected_channels == 1 else f"{expected.expected_channels}ch"
        raise ClipFormatError(
            f"{info.path.name}: clips must be {channels}, found {info.channels} channels"
        )
    return info


def compute_peaks(path: Path | str, buckets: int = 1000) -> dict[str, Any]:
    """Downsample a clip into per-bucket minimum and maximum sample values.

    Args:
        path: Clip to read.
        buckets: Number of buckets. Fewer frames than buckets is fine; the shortfall is padded
            with zeros so the arrays always have the requested length.

    Returns:
        A JSON-serializable payload with ``min`` and ``max`` arrays of length ``buckets``.
    """
    path = Path(path)
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    frames = mono.shape[0]

    minima = np.zeros(buckets, dtype=np.float32)
    maxima = np.zeros(buckets, dtype=np.float32)
    if frames:
        # Bucket boundaries by index rather than reshaping, so a frame count that does not divide
        # evenly by the bucket count still produces exactly `buckets` values.
        edges = np.linspace(0, frames, buckets + 1).astype(int)
        for i in range(buckets):
            start, end = edges[i], max(edges[i + 1], edges[i] + 1)
            window = mono[start:end]
            if window.size:
                minima[i] = window.min()
                maxima[i] = window.max()

    return {
        "version": PEAKS_VERSION,
        "buckets": buckets,
        "sample_rate": int(sample_rate),
        "frames": int(frames),
        "duration_seconds": round(frames / sample_rate, 6) if sample_rate else 0.0,
        "min": [round(float(v), 4) for v in minima],
        "max": [round(float(v), 4) for v in maxima],
    }
