"""Speech, SNR and room reverberation per frame: pyannote's Brouhaha, as ONNX (D87).

Brouhaha (Lavechin et al., 2022) reads six-second windows and gives, for every 16.875 ms frame,
the probability of speech, the speech-to-noise ratio in dB and C50 in dB -- how much of the sound
reaches the microphone in its first 50 ms rather than as echo, so a high C50 is a dry room and a
low one a reverberant hall. It is trained on simulated noise and real room impulse responses.

The graph is built once by ``scripts/export_brouhaha_onnx.py``, which rebuilds the checkpoint in
plain torch and matched pyannote 3.1's own Brouhaha on every frame of real corpus audio. The
checkpoint is gated and OpenRAIL-licensed, so unlike the overlap detector nothing is fetched: the
file is looked for where the other models live, and without it acoustic measurement falls back
to bandwidth alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 6 * SAMPLE_RATE
#: Windows overlap by five seconds, so each frame is the mean of up to six readings. At ~7 ms a
#: window on four threads this is ~25 s of CPU per hour of audio, the overlap detector's cost.
STEP_SAMPLES = 1 * SAMPLE_RATE
#: The model's frame hop in seconds (SincNet stride 10, three max-pools of 3).
FRAME_STEP = 0.016875
#: Output columns.
VAD, SNR, C50 = 0, 1, 2

#: The directory models live in; the container points it into the bind mount (see overlap.py).
MODEL_DIR_ENV = "HARNESS_ALIGNER_MODEL_DIR"
DEFAULT_MODEL_PATH = (
    Path(os.environ.get(MODEL_DIR_ENV) or (Path(__file__).resolve().parent / "models"))
    / "brouhaha.onnx"
)


def window_starts(n_samples: int) -> list[int]:
    """Sample offsets of every window: one per hop, plus a zero-padded last one that reaches
    the end whenever the hops do not."""
    full = max(0, (n_samples - WINDOW_SAMPLES) // STEP_SAMPLES + 1)
    starts = [i * STEP_SAMPLES for i in range(full)]
    if n_samples < WINDOW_SAMPLES or (n_samples - WINDOW_SAMPLES) % STEP_SAMPLES > 0:
        starts.append(full * STEP_SAMPLES)
    return starts


def aggregate(windows: np.ndarray, starts_seconds: list[float], n_frames: int) -> np.ndarray:
    """Average per-window frame outputs onto one grid of ``n_frames`` frames.

    Args:
        windows: ``(windows, frames, channels)``, one row per window.
        starts_seconds: Each window's start in seconds.
        n_frames: Frames in the whole recording; frame ``t`` starts at ``t * FRAME_STEP``.

    Returns:
        ``(n_frames, channels)``. A window's frames stop ~60 ms short of its end, so the last
        few frames of a recording may be read by none: they are NaN.
    """
    per_window = windows.shape[1]
    first = np.rint(np.asarray(starts_seconds, dtype=np.float64) / FRAME_STEP).astype(int)
    total = max(n_frames, int(first.max()) + per_window)
    summed = np.zeros((total, windows.shape[2]))
    hits = np.zeros(total)
    for offset, frames in zip(first, windows, strict=True):
        summed[offset : offset + per_window] += frames
        hits[offset : offset + per_window] += 1
    out = summed / np.maximum(hits, 1)[:, None]
    out[hits == 0] = np.nan  # past the last window's reach: never read, so never speech
    return out[:n_frames]


class Brouhaha:
    """Brouhaha over 16 kHz mono audio; one ONNX session, reusable across episodes.

    Args:
        model_path: The exported graph; the default location when omitted.
        threads: onnxruntime intra-op threads.
    """

    def __init__(self, model_path: Path | None = None, *, threads: int = 4) -> None:
        self._session = None
        path = model_path or DEFAULT_MODEL_PATH
        if not path.is_file():
            logger.info("brouhaha_model_missing", path=str(path))
            return
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    @property
    def available(self) -> bool:
        return self._session is not None

    def frames(self, audio: np.ndarray, sample_rate: int, *, batch_size: int = 32) -> np.ndarray:
        """``(frames, 3)`` of speech probability, SNR dB and C50 dB over the whole recording."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Brouhaha needs {SAMPLE_RATE} Hz audio, got {sample_rate}")
        if self._session is None:
            raise RuntimeError("no Brouhaha model loaded")
        audio = np.asarray(audio, dtype=np.float32)
        starts = window_starts(len(audio))
        outputs = []
        for i in range(0, len(starts), batch_size):
            batch = np.zeros((len(starts[i : i + batch_size]), 1, WINDOW_SAMPLES), dtype=np.float32)
            for row, offset in enumerate(starts[i : i + batch_size]):
                piece = audio[offset : offset + WINDOW_SAMPLES]
                batch[row, 0, : len(piece)] = piece
            outputs.append(self._session.run(None, {"waveforms": batch})[0])
        n_frames = int(np.ceil(len(audio) / SAMPLE_RATE / FRAME_STEP))
        return aggregate(np.concatenate(outputs), [s / SAMPLE_RATE for s in starts], n_frames)
