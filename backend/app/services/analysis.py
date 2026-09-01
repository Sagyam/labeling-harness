"""Token analysis, code-mixing index (CMI), and scoring for ingested segments.

Computes orthography-aware language tags, code-switch density, rule flags, and
queue priority scores without requiring an external GPU pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.services.flags import FlagHypothesis, compute_flags
from app.services.scoring import ScoreInputs, ScoreResult, priority_score

DEV_RE = re.compile(r"[\u0900-\u097F]")
# Note: plain \w+ is wrong because Python's \w excludes combining marks (Mn).
TOK_RE = re.compile(r"[A-Za-z']+|[\u0900-\u097F]+|\d+")
HINDI_MARKERS = frozenset(
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
class TokenAnalysisResult:
    """Outcome of token tagging and code-mixing analysis on a transcript."""

    token_count: int
    devanagari_count: int
    latin_count: int
    cmi: float
    spf: float
    code_switch_density: float
    flags: list[str]
    score: ScoreResult


def analyze_transcript(
    text: str,
    duration_seconds: float,
    *,
    word_disagreement_rate: float = 0.0,
    avg_logprob: float | None = None,
    no_speech_prob: float | None = None,
    settings: Settings | None = None,
) -> TokenAnalysisResult:
    """Tag tokens, calculate CMI and switch-point fraction, compute rule flags and score."""
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
    spf = 0.0

    from itertools import pairwise

    if n >= 2:
        majority = max(devanagari_count, latin_count)
        cmi = round(100.0 * (n - majority) / n, 2)
        switches = sum(1 for a, b in pairwise(tags) if a != b)
        spf = round(switches / (n - 1), 4)

    csd = min(1.0, max(0.0, cmi / 100.0))

    flag_hyps = [FlagHypothesis(text=text, no_speech_prob=no_speech_prob)]
    flags = compute_flags(
        duration_seconds=duration_seconds,
        hypotheses=flag_hyps,
        settings=settings,
    )

    if any(t in HINDI_MARKERS for t in tokens) and "hindi_intrusion" not in flags:
        flags.append("hindi_intrusion")

    inputs = ScoreInputs(
        word_disagreement_rate=word_disagreement_rate,
        avg_logprob=avg_logprob,
        code_switch_density=csd,
        flags=flags,
    )
    score_res = priority_score(inputs, settings=settings)

    return TokenAnalysisResult(
        token_count=n,
        devanagari_count=devanagari_count,
        latin_count=latin_count,
        cmi=cmi,
        spf=spf,
        code_switch_density=csd,
        flags=flags,
        score=score_res,
    )
