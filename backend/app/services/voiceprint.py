"""Voiceprints: which of a clip's speakers a stretch of audio sounds like (D99).

WeSpeaker's ResNet34-LM, as ONNX, embeds audio into the same 256-d space pyannote's
``speaker-diarization-community-1`` uses for the per-speaker centroids stored with every run --
the same weights, fed the same Kaldi fbank. Measured on 190 single-speaker clips of 12 episodes
(docs/findings.md, *Voiceprints on clean speech and in crosstalk*): a window picks the speaker
whose stored centroid is closest 95% of the time from 0.5 s, 99% from 1 s. So a voice's print is
available for every diarized speaker without a GPU, and a voice the owner has confirmed clean
clips of gets a print built from those instead.

What it cannot do is say who said a word inside crosstalk: on two voices mixed at once it follows
the louder one, a coin toss at equal level, and it is confident when it is wrong. Suggestions are
therefore made only for words no second voice is heard over, and never applied by themselves.

The model downloads itself on first use, pinned to a commit and checked by digest, like the
aligner and the overlap detector; without it there are simply no suggestions.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.utils.logging import get_logger
from app.utils.model_fetch import fetch_pinned

logger = get_logger(__name__)

SAMPLE_RATE = 16_000

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
#: A commit, never a branch: the graph must not change under a measurement that already ran.
MODEL_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
MODEL_REMOTE = "voxceleb_resnet34_LM.onnx"
MODEL_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
DOWNLOAD_TIMEOUT_SECONDS = 120.0

#: Set to 1/true/yes to refuse the download; the editor then makes no suggestions.
DISABLE_ENV = "HARNESS_VOICEPRINT_NO_DOWNLOAD"
#: Fetched models share one directory; the aligner's variable names it because it came first.
MODEL_DIR_ENV = "HARNESS_ALIGNER_MODEL_DIR"
DEFAULT_MODEL_PATH = (
    Path(os.environ.get(MODEL_DIR_ENV) or (Path(__file__).resolve().parent / "models"))
    / "wespeaker_resnet34_lm.onnx"
)

#: Seconds of audio around a word that are embedded. 1 s was 99% right on clean speech, 0.5 s 95%.
WINDOW_SECONDS = 1.0
#: How much closer the suggested speaker must be than the next one. On clean 1 s windows, 97%
#: of words cleared it and 99.5% of those were right.
MIN_MARGIN = 0.1

# --- Kaldi fbank, as WeSpeaker computes it -----------------------------------------------------

_FRAME = 400  # 25 ms
_SHIFT = 160  # 10 ms
_N_FFT = 512
_N_MELS = 80
_LOW_HZ = 20.0
_PREEMPHASIS = 0.97
_FLOOR = float(np.finfo(np.float32).eps)


def _mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz) / 700.0)


def _mel_banks() -> np.ndarray:
    low, high = _mel(_LOW_HZ), _mel(SAMPLE_RATE / 2)
    delta = (high - low) / (_N_MELS + 1)
    bins = np.arange(_N_MELS)[:, None]
    left, centre, right = low + bins * delta, low + (bins + 1) * delta, low + (bins + 2) * delta
    mel = _mel((SAMPLE_RATE / _N_FFT) * np.arange(_N_FFT // 2))[None, :]
    banks = np.maximum(
        0.0, np.minimum((mel - left) / (centre - left), (right - mel) / (right - centre))
    )
    # Kaldi's banks stop below Nyquist; the last FFT bin always weighs zero.
    return np.pad(banks, ((0, 0), (0, 1)))


_BANKS = _mel_banks()
_WINDOW = np.hamming(_FRAME)


def fbank(samples: np.ndarray) -> np.ndarray:
    """80 log-mel filterbanks per 10 ms, mean-normalised over the input: ``(frames, 80)``.

    Kaldi's ``compute-fbank-feats`` as WeSpeaker calls it (Hamming window, no dither, no energy),
    in numpy, so torch never enters the backend. Matches ``torchaudio.compliance.kaldi.fbank`` to
    within 2e-4 on the same input.
    """
    x = np.asarray(samples, dtype=np.float64) * 32768.0
    if len(x) < _FRAME:
        return np.zeros((0, _N_MELS), dtype=np.float32)
    count = 1 + (len(x) - _FRAME) // _SHIFT
    frames = x[np.arange(_FRAME)[None, :] + _SHIFT * np.arange(count)[:, None]]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = frames - _PREEMPHASIS * np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=_N_FFT)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T, _FLOOR))
    return (feats - feats.mean(axis=0, keepdims=True)).astype(np.float32)


# --- the model ---------------------------------------------------------------------------------


def download_disabled() -> bool:
    """Whether the environment has switched the fetch off."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def ensure_voiceprint_model(model_path: Path, *, timeout: float = DOWNLOAD_TIMEOUT_SECONDS) -> bool:
    """Make sure the embedding graph exists at ``model_path``, downloading it if missing.

    Returns:
        True when the file is present afterwards. False is not an error: the caller makes no
        suggestions, as it makes no overlap spans without the overlap detector.
    """
    if model_path.is_file():
        return True
    if download_disabled():
        logger.warning("voiceprint_download_disabled", reason=f"{DISABLE_ENV} is set")
        return False
    logger.info("voiceprint_download_started", repo=MODEL_REPO, revision=MODEL_REVISION)
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
        logger.warning("voiceprint_download_failed", error=f"{type(exc).__name__}: {exc}")
        return False
    logger.info("voiceprint_download_complete", path=str(model_path))
    return True


def unit(vector: Sequence[float] | np.ndarray) -> np.ndarray | None:
    """``vector`` scaled to length 1, or None when it has no direction."""
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v)) if v.size else 0.0
    return v / norm if norm > 0 and np.isfinite(norm) else None


class VoiceEmbedder:
    """Embeds audio into the diarizer's speaker space.

    Args:
        model_path: The ONNX graph. Omit it to use the default location, fetching the pinned file
            there when missing; pass one and nothing is downloaded, which keeps tests offline.
        threads: onnxruntime intra-op threads.
    """

    def __init__(self, model_path: Path | None = None, *, threads: int = 2) -> None:
        self._session = None
        path = model_path or DEFAULT_MODEL_PATH
        if model_path is None:
            ensure_voiceprint_model(path)
        if not path.is_file():
            logger.warning("voiceprint_model_missing", path=str(path))
            return
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    @property
    def available(self) -> bool:
        return self._session is not None

    def embed(self, chunks: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        """One unit embedding per chunk of 16 kHz samples; None for a chunk too short to embed.

        Chunks of one length go through the graph as one batch, which is what the editor's
        per-word windows are.
        """
        if self._session is None:
            raise RuntimeError("the voiceprint model is not available")
        out: list[np.ndarray | None] = [None] * len(chunks)
        by_length: dict[int, list[int]] = {}
        feats = [fbank(c) for c in chunks]
        for index, f in enumerate(feats):
            if len(f) >= 10:
                by_length.setdefault(len(f), []).append(index)
        for indices in by_length.values():
            batch = np.stack([feats[i] for i in indices])
            vectors = self._session.run(None, {"feats": batch})[0]
            for i, vector in zip(indices, vectors, strict=True):
                out[i] = unit(vector)
        return out


_EMBEDDER: VoiceEmbedder | None = None


def default_embedder() -> VoiceEmbedder:
    """The process's embedder, loaded once; unavailable when the model could not be had."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = VoiceEmbedder()
    return _EMBEDDER


# --- suggestions (pure) ------------------------------------------------------------------------


def word_window(
    start: float, end: float, clip_seconds: float, width: float = WINDOW_SECONDS
) -> tuple[float, float]:
    """``width`` seconds centred on the word, slid to stay inside the clip."""
    width = min(width, clip_seconds)
    left = (start + end) / 2 - width / 2
    left = min(max(0.0, left), clip_seconds - width)
    return left, left + width


@dataclass(frozen=True)
class Suggestion:
    """A speaker the voiceprint hears a word as, other than the lane it is on."""

    speaker: int
    margin: float


def suggest(
    similarities: Mapping[int, float] | None,
    *,
    current: int | None,
    overlapped: bool,
    min_margin: float = MIN_MARGIN,
) -> Suggestion | None:
    """The speaker to suggest for one word, or None.

    Nothing is suggested for a word in crosstalk -- there the print follows the louder voice --
    or when the closest print is the word's own lane, or not closer than the next by
    ``min_margin``. A word on no lane with one candidate gets that candidate.
    """
    if overlapped or not similarities:
        return None
    ranked = sorted(similarities.items(), key=lambda kv: (-kv[1], kv[0]))
    best, score = ranked[0]
    margin = score - ranked[1][1] if len(ranked) > 1 else (1.0 if current is None else 0.0)
    if best == current or margin < min_margin:
        return None
    return Suggestion(speaker=best, margin=round(margin, 3))


@dataclass
class SuggestionReport:
    """How suggestions fared against what the annotator saved, over per-speaker labels.

    Attributes:
        words: Words of the verified text on a saved lane.
        suggested: Words that carried a suggestion when they were saved.
        taken: Suggested words the annotator left on the suggested lane.
        moved: Words the annotator moved off the lane the diarization proposed.
        moved_suggested: Moved words whose suggestion named the lane they were moved to.
    """

    words: int = 0
    suggested: int = 0
    taken: int = 0
    moved: int = 0
    moved_suggested: int = 0

    @property
    def precision(self) -> float | None:
        return self.taken / self.suggested if self.suggested else None

    @property
    def recall(self) -> float | None:
        return self.moved_suggested / self.moved if self.moved else None


def suggestion_report(
    words: Sequence[tuple[str | None, str | None, str | None, str]],
) -> SuggestionReport:
    """Score suggestions from saved words: ``(speaker, proposed, suggested, source)`` each.

    Only words of the verified text count; typed, copied and recogniser words had no proposal.
    """
    report = SuggestionReport()
    for speaker, proposed, suggested, source in words:
        if source != "label":
            continue
        report.words += 1
        if suggested is not None:
            report.suggested += 1
            report.taken += speaker == suggested
        if speaker != proposed:
            report.moved += 1
            report.moved_suggested += suggested is not None and speaker == suggested
    return report
