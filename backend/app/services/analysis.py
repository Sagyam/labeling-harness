"""Token analysis and code-mixing index (CMI) for ingested segments.

Computes orthography-aware language tags, code-switch density and rule flags without
requiring an external GPU pipeline. Priority scoring is deliberately not done here: the
queue builder scores from the stored segment scores, so doing it at ingest too would mean
two implementations of one formula.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.services.flags import FlagHypothesis, compute_flags

DEV_RE = re.compile(r"[\u0900-\u097F]")
# Note: plain \w+ is wrong because Python's \w excludes combining marks (Mn).
TOK_RE = re.compile(r"[A-Za-z']+|[\u0900-\u097F]+|\d+")


@dataclass(frozen=True)
class TokenAnalysisResult:
    """Outcome of token tagging and code-mixing analysis on a transcript."""

    token_count: int
    devanagari_count: int
    latin_count: int
    cmi: float
    code_switch_density: float
    flags: list[str]


def analyze_transcript(
    text: str,
    duration_seconds: float,
    *,
    no_speech_prob: float | None = None,
    settings: Settings | None = None,
) -> TokenAnalysisResult:
    """Tag tokens by script, compute the code-mixing index, and raise the rule flags.

    Args:
        text: The transcript to analyse.
        duration_seconds: Segment length, for the duration and speaking-rate flags.
        no_speech_prob: The hypothesis\'s no-speech probability, if the model reported one.
        settings: Threshold overrides.
    """
    settings = settings or get_settings()

    tokens = TOK_RE.findall(text or "")
    devanagari_count = 0
    latin_count = 0
    tags: list[str] = []

    for tok in tokens:
        if tok.isdigit():
            continue
        normalized = unicodedata.normalize("NFKC", tok)
        if DEV_RE.search(normalized):
            devanagari_count += 1
            tags.append("ne")
        else:
            latin_count += 1
            tags.append("en")

    n = len(tags)
    cmi = 0.0
    if n >= 2:
        majority = max(devanagari_count, latin_count)
        cmi = round(100.0 * (n - majority) / n, 2)

    csd = min(1.0, max(0.0, cmi / 100.0))

    flags = compute_flags(
        duration_seconds=duration_seconds,
        hypotheses=[FlagHypothesis(text=text, no_speech_prob=no_speech_prob)],
        settings=settings,
    )

    return TokenAnalysisResult(
        token_count=n,
        devanagari_count=devanagari_count,
        latin_count=latin_count,
        cmi=cmi,
        code_switch_density=csd,
        flags=flags,
    )
