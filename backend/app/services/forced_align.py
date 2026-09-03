"""CTC forced alignment of a known transcript onto its own clip.

Forced alignment is not recognition: it never chooses words. Given audio and the exact text
that was said, it finds the most likely monotonic placement of that text onto the waveform and
reports where each word starts and ends. That is what lets a transcriber which returns no
timestamps still contribute word spans, and what gives D27's boundary check a second opinion
that does not come from another cloud vendor (D32).

The acoustic model is a romanizing multilingual CTC head: every script is folded into one Latin
label set before alignment, which is the only arrangement that handles Devanagari and Latin
*inside the same utterance* -- the Nepanglish case. A monolingual English head cannot score the
majority of this corpus's tokens.

Runtime mirrors ``silero_vad.py`` exactly: an ONNX session pinned to one CPU thread, and a
warning rather than an exception when the model file is absent, so ingestion degrades to "no
word spans" instead of failing an episode. Spans are clip-relative, satisfying D26 by
construction -- the aligner only ever sees the clip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from app.services.alignment import DEV_RE, WordSpan, project_missing_spans
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "mms_fa.onnx"
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "models" / "mms_fa_vocab.json"

SAMPLE_RATE = 16000
#: Names a CTC vocabulary may give the blank label, in the order they are tried.
BLANK_KEYS = ("<blank>", "<pad>", "<blk>", "_")
#: Word separator, when the vocabulary has one. MMS-style romanized heads usually do not.
WORD_DELIMITER = "|"
#: Everything the romanized label set can represent. Digits and punctuation survive
#: romanization but have no acoustic label, so they are stripped here rather than silently
#: mismatched against the vocabulary later.
ROMAN_RE = re.compile(r"[^a-z']")


def romanize(token: str) -> str:
    """Fold one token into the aligner's Latin label alphabet.

    Devanagari is transliterated through Harvard-Kyoto; Latin passes through. The result is
    lowercased and stripped to the characters an acoustic label set can carry, so a token made
    only of digits or punctuation romanizes to the empty string and is not alignable.
    """
    text = token.strip()
    if not text:
        return ""
    if DEV_RE.search(text):
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
    return ROMAN_RE.sub("", text.lower())


def build_target(
    words: list[str],
    vocab: dict[str, int],
    delimiter: str | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Flatten romanized words into one CTC target sequence.

    Args:
        words: Already-romanized words, in order.
        vocab: Label to id mapping from the model's tokenizer.
        delimiter: Word separator to insert between words, when the vocabulary has one. It is
            deliberately left outside every word's range so it never claims a word's frames.

    Returns:
        The target ids, and one ``(start, end)`` half-open range into them per input word. A
        word whose characters are all outside the vocabulary gets an empty range.
    """
    ids: list[int] = []
    ranges: list[tuple[int, int]] = []
    delimiter_id = vocab.get(delimiter) if delimiter else None

    for position, word in enumerate(words):
        if position > 0 and delimiter_id is not None:
            ids.append(delimiter_id)
        start = len(ids)
        ids.extend(vocab[ch] for ch in word if ch in vocab)
        ranges.append((start, len(ids)))

    return ids, ranges


def ctc_forced_align(
    log_probs: np.ndarray,
    targets: list[int],
    blank_id: int = 0,
) -> np.ndarray:
    """Viterbi-decode the single best CTC path through a *known* target sequence.

    Args:
        log_probs: ``[T, V]`` log probabilities, one row per acoustic frame.
        targets: The label ids that were said, in order.
        blank_id: Id of the CTC blank label.

    Returns:
        ``[T]`` of state indices into the blank-interleaved sequence
        ``[blank, t0, blank, t1, ..., blank]``, so target ``l`` occupies state ``2l + 1``.
        Empty when no valid path exists -- most often fewer frames than the target needs.
    """
    n_frames = log_probs.shape[0]
    n_targets = len(targets)
    if n_frames == 0 or n_targets == 0:
        return np.empty(0, dtype=np.int64)

    n_states = 2 * n_targets + 1
    extended = np.full(n_states, blank_id, dtype=np.int64)
    extended[1::2] = targets

    # A path may skip a blank only between two *different* labels; two identical labels in a
    # row must be separated by one, or CTC would collapse them into a single emission.
    can_skip = np.zeros(n_states, dtype=bool)
    can_skip[3::2] = extended[3::2] != extended[1:-2:2]

    neg_inf = -np.inf
    alpha = np.full(n_states, neg_inf, dtype=np.float64)
    alpha[0] = log_probs[0, extended[0]]
    if n_states > 1:
        alpha[1] = log_probs[0, extended[1]]

    backptr = np.zeros((n_frames, n_states), dtype=np.int8)
    columns = np.arange(n_states)

    for frame in range(1, n_frames):
        stay = alpha
        step = np.concatenate(([neg_inf], alpha[:-1]))
        skip = np.concatenate(([neg_inf, neg_inf], alpha[:-2]))
        skip = np.where(can_skip, skip, neg_inf)

        choices = np.stack((stay, step, skip))
        chosen = np.argmax(choices, axis=0)
        alpha = choices[chosen, columns] + log_probs[frame, extended]
        backptr[frame] = chosen

    last_states = [n_states - 1] if n_states == 1 else [n_states - 1, n_states - 2]
    best_state = max(last_states, key=lambda s: alpha[s])
    if not np.isfinite(alpha[best_state]):
        return np.empty(0, dtype=np.int64)

    path = np.zeros(n_frames, dtype=np.int64)
    state = best_state
    for frame in range(n_frames - 1, 0, -1):
        path[frame] = state
        state -= int(backptr[frame, state])
    path[0] = state
    return path


def _span_for_range(
    path: np.ndarray,
    log_probs: np.ndarray,
    extended_labels: np.ndarray,
    char_range: tuple[int, int],
    stride_seconds: float,
    word: str,
) -> WordSpan | None:
    """Collect the frames a word's characters occupy into one span."""
    start_char, end_char = char_range
    if start_char >= end_char:
        return None

    states = {2 * char + 1 for char in range(start_char, end_char)}
    frames = np.array([f for f, s in enumerate(path) if int(s) in states], dtype=np.int64)
    if frames.size == 0:
        return None

    scores = np.exp(log_probs[frames, extended_labels[path[frames]]])
    return WordSpan(
        word=word,
        start=round(float(frames[0]) * stride_seconds, 3),
        end=round(float(frames[-1] + 1) * stride_seconds, 3),
        confidence=round(float(scores.mean()), 4),
    )


def align_emissions(
    log_probs: np.ndarray,
    tokens: list[str],
    vocab: dict[str, int],
    duration_seconds: float,
    blank_id: int = 0,
) -> list[WordSpan] | None:
    """Place ``tokens`` onto an emission matrix and return one clip-relative span each.

    Tokens that romanize to nothing -- bare digits, punctuation -- cannot enter the CTC target,
    so they are interpolated between their aligned neighbours afterwards and carry the reduced
    confidence ``project_missing_spans`` assigns.

    Returns ``None`` when nothing could be aligned at all.
    """
    n_frames = log_probs.shape[0]
    if not tokens or n_frames == 0 or duration_seconds <= 0.0:
        return None

    delimiter = WORD_DELIMITER if WORD_DELIMITER in vocab else None
    romanized = [romanize(tok) for tok in tokens]
    alignable = [i for i, roman in enumerate(romanized) if roman]
    if not alignable:
        return None

    target_ids, char_ranges = build_target([romanized[i] for i in alignable], vocab, delimiter)
    if not target_ids:
        return None

    path = ctc_forced_align(log_probs, target_ids, blank_id=blank_id)
    if path.size == 0:
        return None

    extended_labels = np.full(2 * len(target_ids) + 1, blank_id, dtype=np.int64)
    extended_labels[1::2] = target_ids
    stride_seconds = duration_seconds / n_frames

    spans: dict[int, WordSpan] = {}
    for position, token_index in enumerate(alignable):
        span = _span_for_range(
            path,
            log_probs,
            extended_labels,
            char_ranges[position],
            stride_seconds,
            tokens[token_index],
        )
        if span is not None:
            spans[token_index] = span

    if not spans:
        return None

    aligned = [(tok, spans.get(i)) for i, tok in enumerate(tokens)]
    return project_missing_spans(aligned, 0.0, duration_seconds)


class ForcedAligner:
    """ONNX CTC aligner, loaded lazily and absent-tolerant.

    A missing model file is a warning, not an error: the harness still has valid transcripts
    without word spans, and an episode must not fail to ingest because an optional artefact was
    never fetched.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.vocab_path = Path(vocab_path or DEFAULT_VOCAB_PATH)
        self._session: object | None = None
        self._vocab: dict[str, int] | None = None
        self._blank_id = 0
        self._input_name = ""
        self._load()

    @property
    def available(self) -> bool:
        """Whether the model and its vocabulary both loaded."""
        return self._session is not None and self._vocab is not None

    def _load(self) -> None:
        if not self.model_path.is_file() or not self.vocab_path.is_file():
            logger.warning(
                "forced_aligner_model_missing",
                model=str(self.model_path),
                vocab=str(self.vocab_path),
            )
            return
        try:
            import onnxruntime as ort

            self._vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = session.get_inputs()[0].name
            self._session = session
            self._blank_id = self._resolve_blank_id(self._vocab)
            logger.info("forced_aligner_loaded", path=str(self.model_path))
        except Exception as exc:
            logger.warning("forced_aligner_load_failed", error=str(exc))
            self._session = None
            self._vocab = None

    @staticmethod
    def _resolve_blank_id(vocab: dict[str, int]) -> int:
        for key in BLANK_KEYS:
            if key in vocab:
                return int(vocab[key])
        return 0

    def align(
        self,
        audio_path: Path | str,
        tokens: list[str],
        sample_rate: int = SAMPLE_RATE,
    ) -> list[WordSpan] | None:
        """Align ``tokens`` against a clip, returning clip-relative word spans.

        Args:
            audio_path: 16 kHz mono clip. Invariant 6 guarantees the format, so nothing is
                resampled here.
            tokens: The words that were said, in order.
            sample_rate: Expected rate, used only to convert frames back to seconds.

        Returns:
            One span per token, or ``None`` when the model is unavailable or no token could be
            placed.
        """
        if not self.available or not tokens:
            return None

        try:
            audio, file_rate = sf.read(str(audio_path), dtype="float32")
        except Exception as exc:
            logger.warning("forced_aligner_read_failed", path=str(audio_path), error=str(exc))
            return None

        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio.size == 0:
            return None

        duration = float(audio.size) / float(file_rate or sample_rate)
        waveform = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

        try:
            outputs = self._session.run(  # type: ignore[union-attr]
                None, {self._input_name: waveform[None, :].astype(np.float32)}
            )
        except Exception as exc:
            logger.warning("forced_aligner_inference_failed", error=str(exc))
            return None

        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.ndim == 3:
            logits = logits[0]
        log_probs = logits - _logsumexp(logits)

        assert self._vocab is not None
        return align_emissions(log_probs, tokens, self._vocab, duration, blank_id=self._blank_id)


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-sum-exp, kept 2-D so it broadcasts back over the label axis."""
    peak = logits.max(axis=-1, keepdims=True)
    return peak + np.log(np.exp(logits - peak).sum(axis=-1, keepdims=True))
