"""Token analysis and code-mixing index (CMI) for ingested segments.

Computes orthography-aware language tags, code-switch density and rule flags without
requiring an external GPU pipeline. Priority scoring is deliberately not done here: the
queue builder scores from the stored segment scores, so doing it at ingest too would mean
two implementations of one formula.
"""

from __future__ import annotations

import difflib
import itertools
import re
import unicodedata
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.services.flags import FlagHypothesis, compute_flags

DEV_RE = re.compile(r"[\u0900-\u097F]")
# Note: plain \w+ is wrong because Python's \w excludes combining marks (Mn).
TOK_RE = re.compile(r"[A-Za-z']+|[\u0900-\u097F]+|\d+")


DISCOURSE_MARKERS: frozenset[str] = frozenset(
    {
        "so",
        "actually",
        "basically",
        "like",
        "literally",
        "anyway",
        "anyhow",
        "meanwhile",
        "well",
        "yeah",
        "yes",
        "no",
        "okay",
        "ok",
        "right",
        "fine",
        "see",
    }
)


@dataclass(frozen=True)
class TokenAnalysisResult:
    """Outcome of token tagging and code-mixing analysis on a transcript."""

    token_count: int
    devanagari_count: int
    latin_count: int
    cmi: float
    code_switch_density: float
    flags: list[str]
    switch_point_count: int = 0
    discourse_marker_count: int = 0


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

    switch_points = sum(1 for i in range(1, len(tags)) if tags[i] != tags[i - 1])
    dm_count = sum(1 for tok in tokens if tok.lower() in DISCOURSE_MARKERS)

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
        switch_point_count=switch_points,
        discourse_marker_count=dm_count,
    )


def mean_pairwise_disagreement(sequences: list[list[str]] | list[str]) -> float:
    """Mean 1 - similarity over every unordered pair of hypotheses.

    Operates on word lists or on raw strings, giving a word-level or character-level rate from
    the same comparison. Fewer than two hypotheses means nothing disagreed, which is 0.0 -- the
    same value the scorer reads for a missing rate.
    """
    if len(sequences) < 2:
        return 0.0
    ratios = [
        difflib.SequenceMatcher(None, sequences[i], sequences[j]).ratio()
        for i, j in itertools.combinations(range(len(sequences)), 2)
    ]
    return round(1.0 - (sum(ratios) / len(ratios)), 4)
