"""Overlapped speech: where two or more people are talking at once (D77).

pyannote's ``segmentation-3.0`` model reads 10 s windows on a 1 s hop and says, for every
~17 ms frame, which of up to three local speakers is active -- a "powerset" of seven classes:
nobody, one of three speakers, or one of three pairs. Counting the active speakers per frame and
averaging that count across the overlapping windows is pyannote's own ``speaker_count``. Overlap
is a count of two or more. No speaker identity is involved: linking local speakers across windows
needs voice embeddings and is the expensive part of diarization, which this module does not do.

This is a numpy + onnxruntime port of that computation, so it runs without ``torch`` (D32). On a
10-minute podcast excerpt it matched pyannote 4.0.7's ``speaker_count`` from
``speaker-diarization-community-1`` on every one of 35,552 frames, and it costs about 25 s of CPU
per hour of audio on four threads.

The graph is the public MIT export in ``onnx-community/pyannote-segmentation-3.0``, fetched on
first use, pinned to a commit and checked against a digest like the aligner (D42).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.utils.logging import get_logger
from app.utils.model_fetch import fetch_pinned

logger = get_logger(__name__)

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 10 * SAMPLE_RATE
STEP_SAMPLES = 1 * SAMPLE_RATE
#: The model's frame hop in seconds (its receptive-field step).
FRAME_STEP = 0.016875
#: Active speakers in each powerset class: {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}.
ACTIVE_SPEAKERS = np.array([0, 1, 1, 1, 2, 2, 2], dtype=np.float32)

MODEL_REPO = "onnx-community/pyannote-segmentation-3.0"
#: A commit, never a branch: the graph must not change under a measurement that already ran.
MODEL_REVISION = "733a93b6473d019a773298e08cefa686894b1854"
MODEL_REMOTE = "onnx/model.onnx"
MODEL_SHA256 = "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25"
MODEL_BYTES = 5_986_908
DOWNLOAD_TIMEOUT_SECONDS = 120.0

#: Set to 1/true/yes to refuse the download; clips then get no overlap spans.
DISABLE_ENV = "HARNESS_OVERLAP_NO_DOWNLOAD"
#: The directory fetched models live in. The aligner's variable names it because it came first;
#: the container points it into the bind mount so a download survives ``up``.
MODEL_DIR_ENV = "HARNESS_ALIGNER_MODEL_DIR"
DEFAULT_MODEL_PATH = (
    Path(os.environ.get(MODEL_DIR_ENV) or (Path(__file__).resolve().parent / "models"))
    / "pyannote_segmentation_3.onnx"
)


def download_disabled() -> bool:
    """Whether the environment has switched the fetch off."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def ensure_overlap_model(model_path: Path, *, timeout: float = DOWNLOAD_TIMEOUT_SECONDS) -> bool:
    """Make sure the segmentation graph exists at ``model_path``, downloading it if missing.

    Returns:
        True when the file is present afterwards. False is not an error: the caller degrades to
        "no overlap spans", as it does for an absent aligner.
    """
    if model_path.is_file():
        return True
    if download_disabled():
        logger.warning("overlap_download_disabled", reason=f"{DISABLE_ENV} is set")
        return False
    logger.info("overlap_download_started", repo=MODEL_REPO, revision=MODEL_REVISION)
    try:
        fetch_pinned(
            repo=MODEL_REPO,
            revision=MODEL_REVISION,
            remote=MODEL_REMOTE,
            destination=model_path,
            expected_sha=MODEL_SHA256,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("overlap_download_failed", error=f"{type(exc).__name__}: {exc}")
        return False
    logger.info("overlap_download_complete", path=str(model_path))
    return True


def chunk_starts(n_samples: int) -> list[int]:
    """Sample offsets of the windows pyannote would read: every hop, plus a zero-padded last
    window whenever the hops do not end exactly at the end of the audio."""
    full = max(0, (n_samples - WINDOW_SAMPLES) // STEP_SAMPLES + 1)
    starts = [i * STEP_SAMPLES for i in range(full)]
    if n_samples < WINDOW_SAMPLES or (n_samples - WINDOW_SAMPLES) % STEP_SAMPLES > 0:
        starts.append(full * STEP_SAMPLES)
    return starts


def aggregate_counts(chunk_counts: np.ndarray, starts_seconds: Sequence[float]) -> np.ndarray:
    """Average per-window speaker counts onto one frame grid and round, as pyannote does.

    Args:
        chunk_counts: ``(windows, frames)`` active-speaker counts, one row per window.
        starts_seconds: Each window's start in seconds.

    Returns:
        One integer count per global frame; frame ``t`` starts at ``t * FRAME_STEP``.
    """
    frames = chunk_counts.shape[1]
    first = np.rint(np.asarray(starts_seconds, dtype=np.float64) / FRAME_STEP).astype(int)
    total = int(first.max()) + frames
    summed = np.zeros(total)
    hits = np.zeros(total)
    for offset, counts in zip(first, chunk_counts, strict=True):
        summed[offset : offset + frames] += counts
        hits[offset : offset + frames] += 1
    return np.rint(summed / np.maximum(hits, 1e-12))


def frames_to_spans(
    mask: np.ndarray, *, frame_step: float = FRAME_STEP
) -> list[tuple[float, float]]:
    """Runs of true frames as ``(start, end)`` seconds, rounded to the millisecond."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        (round(float(s) * frame_step, 3), round(float(e) * frame_step, 3))
        for s, e in zip(starts, ends, strict=True)
    ]


def spans_within(
    spans: Sequence[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """The overlap inside one clip, clip-relative like word spans and VAD spans (D26, D55).

    Empty when the clip holds none -- a real answer, distinct from ``None`` (never measured).
    """
    out: list[tuple[float, float]] = []
    for lo, hi in spans:
        a, b = max(lo, start), min(hi, end)
        if b > a:
            out.append((round(a - start, 3), round(b - start, 3)))
    return out


class OverlapDetector:
    """Finds overlapped speech in 16 kHz mono audio; one ONNX session, reusable across episodes.

    Args:
        model_path: The segmentation graph. Omit it to use the default location, fetching the
            pinned file there when it is missing; pass one explicitly and nothing is downloaded,
            which is what keeps tests and fixtures offline.
        threads: onnxruntime intra-op threads.
    """

    def __init__(self, model_path: Path | None = None, *, threads: int = 4) -> None:
        self._session = None
        path = model_path or DEFAULT_MODEL_PATH
        if model_path is None:
            ensure_overlap_model(path)
        if not path.is_file():
            logger.warning("overlap_model_missing", path=str(path))
            return
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    @property
    def available(self) -> bool:
        return self._session is not None

    def speaker_count(
        self, audio: np.ndarray, sample_rate: int, *, batch_size: int = 32
    ) -> np.ndarray:
        """Active speakers per frame over the whole recording."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"overlap detection needs {SAMPLE_RATE} Hz audio, got {sample_rate}")
        if self._session is None:
            raise RuntimeError("no overlap model loaded")
        audio = np.asarray(audio, dtype=np.float32)
        starts = chunk_starts(len(audio))
        counts = []
        for i in range(0, len(starts), batch_size):
            batch = np.zeros((len(starts[i : i + batch_size]), 1, WINDOW_SAMPLES), dtype=np.float32)
            for row, offset in enumerate(starts[i : i + batch_size]):
                piece = audio[offset : offset + WINDOW_SAMPLES]
                batch[row, 0, : len(piece)] = piece
            logits = self._session.run(None, {"input_values": batch})[0]
            counts.append(ACTIVE_SPEAKERS[logits.argmax(-1)])
        count = aggregate_counts(np.concatenate(counts), [s / SAMPLE_RATE for s in starts])
        return count[: int(np.ceil(len(audio) / SAMPLE_RATE / FRAME_STEP))]

    def detect(
        self, audio: np.ndarray, sample_rate: int, *, batch_size: int = 32
    ) -> list[tuple[float, float]] | None:
        """Episode-relative spans with two or more speakers; ``None`` when no model is loaded."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"overlap detection needs {SAMPLE_RATE} Hz audio, got {sample_rate}")
        if self._session is None:
            return None
        count = self.speaker_count(audio, sample_rate, batch_size=batch_size)
        return frames_to_spans(count >= 2)
