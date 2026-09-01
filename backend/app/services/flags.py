"""Rule-based segment flags.

These are cheap, explainable signals computed from the imported hypotheses alone -- no model, no
LLM. They feed the ``rule_flag_score`` component of the priority formula and are shown in the UI as
reason chips, so a flag must always be something a human can verify by looking at the segment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings, get_settings

#: Every flag this module can raise. The denominator of ``rule_flag_score``.
ALL_FLAGS: tuple[str, ...] = (
    "empty_transcript",
    "repeated_ngram",
    "high_no_speech_prob",
    "too_short",
    "too_long",
    "implausible_speaking_rate",
    "script_conflict",
    "hindi_intrusion",
)

_DEVANAGARI_START, _DEVANAGARI_END = "ऀ", "ॿ"
HINDI_MARKERS: frozenset[str] = frozenset(
    {
        "नहीं",
        "था",
        "थी",
        "थे",
        "रहा",
        "रही",
        "रहे",
        "होगा",
        "होगी",
        "होंगे",
        "करना",
        "करता",
        "करते",
        "करती",
        "यह",
        "वह",
        "क्या",
        "हुए",
        "हुआ",
        "हुई",
    }
)


@dataclass(frozen=True)
class FlagHypothesis:
    """The parts of a hypothesis the flag rules look at."""

    text: str
    no_speech_prob: float | None = None


def _devanagari_ratio(text: str) -> float:
    """Fraction of letters written in Devanagari, ignoring punctuation and digits."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    devanagari = sum(1 for ch in letters if _DEVANAGARI_START <= ch <= _DEVANAGARI_END)
    return devanagari / len(letters)


def _has_repeated_ngram(text: str, size: int, threshold: int) -> bool:
    """Whether any n-gram of ``size`` words appears at least ``threshold`` times.

    This is the shape ASR hallucinations take: the decoder locks into a loop and emits the same
    phrase over and over.
    """
    words = text.split()
    if len(words) < size * threshold:
        return False
    grams = Counter(tuple(words[i : i + size]) for i in range(len(words) - size + 1))
    return max(grams.values()) >= threshold


def compute_flags(
    *,
    duration_seconds: float,
    hypotheses: Sequence[FlagHypothesis],
    settings: Settings | None = None,
) -> list[str]:
    """Compute the rule flags for one segment.

    Args:
        duration_seconds: Segment length.
        hypotheses: Every imported hypothesis for the segment. May be empty.
        settings: Thresholds; defaults to the loaded configuration.

    Returns:
        Sorted, de-duplicated flag names drawn from :data:`ALL_FLAGS`.
    """
    settings = settings or get_settings()
    cfg = settings.queue
    raised: set[str] = set()

    if not hypotheses or any(not h.text.strip() for h in hypotheses):
        raised.add("empty_transcript")

    for hypothesis in hypotheses:
        if _has_repeated_ngram(
            hypothesis.text, cfg.repeated_ngram_size, cfg.repeated_ngram_threshold
        ):
            raised.add("repeated_ngram")
        if (
            hypothesis.no_speech_prob is not None
            and hypothesis.no_speech_prob >= cfg.no_speech_prob_threshold
        ):
            raised.add("high_no_speech_prob")
        if any(w in HINDI_MARKERS for w in hypothesis.text.split()):
            raised.add("hindi_intrusion")

    if duration_seconds < cfg.min_duration_seconds:
        raised.add("too_short")
    if duration_seconds > cfg.max_duration_seconds:
        raised.add("too_long")

    word_counts = [len(h.text.split()) for h in hypotheses if h.text.strip()]
    if word_counts and duration_seconds > 0:
        # The median-ish choice: the longest hypothesis, so a truncated one does not look slow.
        rate = max(word_counts) / duration_seconds
        if rate > cfg.max_speaking_rate_wps or rate < cfg.min_speaking_rate_wps:
            raised.add("implausible_speaking_rate")

    if len(hypotheses) > 1:
        ratios = [_devanagari_ratio(h.text) for h in hypotheses if h.text.strip()]
        if ratios and (max(ratios) - min(ratios)) > 0.5:
            raised.add("script_conflict")

    return sorted(raised)


def rule_flag_score(flags: Sequence[str]) -> float:
    """Fraction of the known rule flags raised, in 0-1.

    Flags the pipeline invented upstream are ignored rather than counted, so an unfamiliar flag
    cannot push the priority score past 1.
    """
    if not flags:
        return 0.0
    known = {flag for flag in flags if flag in ALL_FLAGS}
    return len(known) / len(ALL_FLAGS)
