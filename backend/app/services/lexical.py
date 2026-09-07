"""Token-set agreement between the seed and the other systems (D67).

``consensus.py`` answers "how much of the clip's *speech time* do the other systems take away
from the seed". That is the right question for how much of the transcript is wrong, and the wrong
question for whether the annotator has to touch it at all: a clip needing one two-word fix scores
near zero on speech time and still costs a full edit cycle. Measured against realized edits, the
time-weighted term ranks the top and bottom deciles well and leaves deciles 4-7 -- 40% of the
queue -- indistinguishable from the base rate.

The two signals here are unweighted by time and deliberately crude:

``seed_orphan_rate``
    The share of the seed's tokens that *no* other system produced anywhere in the clip. It is
    ``seed_outvoted`` with the clock taken out, which is what makes a single wrong word in a long
    clip visible.

``roman_gap``
    Latin-script tokens that every other system produced and the seed did not. This is the
    corpus's English-in-Latin policy (D64) turned into a ranking signal: when the others write
    ``traffic police`` and the seed writes ``ट्राफिक पुलिस``, the annotator will retype it.

Both are pure functions of ``text_raw``, so neither needs word timings -- they still rank a
segment no system reported spans for, which is exactly where ``seed_outvoted`` is blind.

Unanimity among the others is required in both. One system disagreeing is ordinary recogniser
noise -- MAI in particular transliterates English into Devanagari wholesale -- and counting it
would rank on that system's habits rather than on the seed's errors.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.services.normalize import WORD_TOKEN_RE

#: Any Latin letter. A token counts as Latin-script if it contains one.
_LATIN = re.compile(r"[A-Za-z]")

#: Where the Latin-gap term saturates. Four unanimous missing Latin tokens is already a clip the
#: annotator rewrites; counting further buys no ranking and would let one long clip dominate.
ROMAN_GAP_FULL_SCALE = 4


def _tokens(text: str | None) -> list[str]:
    return WORD_TOKEN_RE.findall(text or "")


def seed_orphan_rate(seed_text: str | None, other_texts: Sequence[str | None]) -> float:
    """Share of the seed's tokens that no other system produced.

    Args:
        seed_text: The hypothesis the annotator will edit.
        other_texts: Every other system's text for the same segment.

    Returns:
        A fraction in 0-1. Unmeasurable -- an empty seed, or no other system -- is 0.0, matching
        the convention that a missing signal must not push a segment up the queue on its own.
    """
    seed = _tokens(seed_text)
    others = [set(_tokens(t)) for t in other_texts if _tokens(t)]
    if not seed or not others:
        return 0.0
    orphaned = sum(1 for token in seed if all(token not in other for other in others))
    return orphaned / len(seed)


def roman_gap(seed_text: str | None, other_texts: Sequence[str | None]) -> float:
    """Latin-script tokens every other system agreed on and the seed does not have.

    Args:
        seed_text: The hypothesis the annotator will edit.
        other_texts: Every other system's text for the same segment.

    Returns:
        A count normalized onto 0-1 at :data:`ROMAN_GAP_FULL_SCALE`. Unmeasurable is 0.0.
    """
    seed = _tokens(seed_text)
    other_sets: list[set[str]] = []
    for text in other_texts:
        tokens = _tokens(text)
        if tokens:
            other_sets.append({t.lower() for t in tokens if _LATIN.search(t)})
    if not seed or not other_sets:
        return 0.0
    agreed: set[str] = set.intersection(*other_sets)
    in_seed = {t.lower() for t in seed if _LATIN.search(t)}
    missing = len(agreed - in_seed)
    return min(1.0, missing / ROMAN_GAP_FULL_SCALE)


def lexical_signals(
    seed_text: str | None, other_texts: Iterable[str | None]
) -> tuple[float, float]:
    """Both signals in one pass, for callers that want them together."""
    others = list(other_texts)
    return seed_orphan_rate(seed_text, others), roman_gap(seed_text, others)
