"""Score a fine-tuned model's transcripts against the labels, clip by clip (D83).

The model runs in the training notebook, on a GPU; the harness only scores what it wrote. The
numbers must be the notebook's numbers, so every rule here is the notebook's rule
(``ftkit.harness_scorer``): WER over :func:`app.services.fold.word_errors`, folded and raw, errors
pooled over words rather than averaged over clips; CER over the folded tokens joined by spaces,
Latin lower-cased; a loop is a 3-word run repeated five or more times.

Scoring is pure: :func:`score_clip` and :func:`summarize` take text and return numbers, and only
:mod:`app.services.model_import` touches the database.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.fold import Ruleset, fold_tokens, word_errors

#: Episode resamples behind a WER interval. Seeded, so the same run always reports the same one.
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 0

#: Overlap-share buckets, as in docs/findings.md (crosstalk). ``unmeasured`` is a clip the overlap
#: detector never saw; it is kept apart from ``none`` because only ``none`` is evidence of a clean
#: clip (D77).
OVERLAP_BUCKETS = ("none", "0-5%", "5-15%", ">15%", "unmeasured")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass(frozen=True)
class ClipScore:
    """The counts for one clip. Rates are derived from them, never stored beside them."""

    ref_words: int
    errors: int
    substitutions: int
    deletions: int
    insertions: int
    raw_ref_words: int
    raw_errors: int
    ref_chars: int
    char_errors: int
    is_loop: bool


@dataclass(frozen=True)
class ScoredClip:
    """A clip's score with what the breakdowns group it by."""

    episode: str
    genre: str | None
    overlap: str
    score: ClipScore


def levenshtein(a: str, b: str) -> int:
    """Character edit distance, bit-parallel (Myers 1999, in Hyyro's formulation).

    One pass over ``a`` with ``b`` held as bit masks, so a clip's CER costs a few hundred integer
    operations instead of the ~10^5 steps of the plain table. Python integers are unbounded, so
    there is no word-size limit on ``b``.
    """
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    masks: dict[str, int] = defaultdict(int)
    for i, char in enumerate(b):
        masks[char] |= 1 << i
    full = (1 << m) - 1
    top = 1 << (m - 1)
    positive, negative, distance = full, 0, m
    for char in a:
        eq = masks.get(char, 0)
        xv = eq | negative
        xh = (((eq & positive) + positive) ^ positive) | eq
        ph = negative | ~(xh | positive)
        mh = positive & xh
        if ph & top:
            distance += 1
        elif mh & top:
            distance -= 1
        # Shifting a 1 in is the edit-distance boundary D[0][j] = j (a search would shift 0).
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        positive = (mh | ~(xv | ph)) & full
        negative = ph & xv
    return distance


def is_loop(text: str) -> bool:
    """A 3-word sequence repeated 5 or more times: the decoder is stuck, not transcribing."""
    toks = text.split()
    top = Counter(zip(toks, toks[1:], toks[2:], strict=False)).most_common(1)
    return bool(top) and top[0][1] >= 5


def _chars(text: str, ruleset: Ruleset | None) -> str:
    return " ".join(t if _DEVANAGARI.search(t) else t.lower() for t in fold_tokens(text, ruleset))


def score_clip(reference: str, hypothesis: str, *, ruleset: Ruleset | None = None) -> ClipScore:
    """Count one clip's word and character errors, folded and raw."""
    folded = word_errors(reference, hypothesis, ruleset=ruleset)
    raw = word_errors(reference, hypothesis, folded=False)
    ref_chars = _chars(reference, ruleset)
    return ClipScore(
        ref_words=folded.ref_words,
        errors=folded.errors,
        substitutions=folded.substitutions,
        deletions=folded.deletions,
        insertions=folded.insertions,
        raw_ref_words=raw.ref_words,
        raw_errors=raw.errors,
        ref_chars=len(ref_chars),
        char_errors=levenshtein(ref_chars, _chars(hypothesis, ruleset)),
        is_loop=is_loop(hypothesis),
    )


def overlap_share(spans: Sequence[Sequence[float]] | None, duration: float) -> float | None:
    """The fraction of a clip spent in crosstalk; ``None`` when it was never measured."""
    if spans is None:
        return None
    if duration <= 0:
        return 0.0
    return min(1.0, sum(max(0.0, end - start) for start, end in spans) / duration)


def overlap_bucket(share: float | None) -> str:
    """The overlap bucket for a clip's overlap share (docs/findings.md)."""
    if share is None:
        return "unmeasured"
    if share <= 0:
        return "none"
    if share < 0.05:
        return "0-5%"
    if share <= 0.15:
        return "5-15%"
    return ">15%"


def _rate(errors: int, words: int) -> float:
    return 100 * errors / words if words else 0.0


def _episode_interval(clips: Sequence[ScoredClip]) -> list[float] | None:
    """95% interval of pooled WER from resampling whole episodes, or ``None`` for one episode.

    Clips of one episode share speakers and a microphone, so they are not independent; resampling
    clips would report an interval far narrower than the data supports.
    """
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for clip in clips:
        totals[clip.episode][0] += clip.score.errors
        totals[clip.episode][1] += clip.score.ref_words
    episodes = list(totals.values())
    if len(episodes) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    rates = []
    for _ in range(BOOTSTRAP_ROUNDS):
        draw = rng.choices(episodes, k=len(episodes))
        rates.append(_rate(sum(e for e, _ in draw), sum(w for _, w in draw)))
    rates.sort()
    return [rates[int(0.025 * BOOTSTRAP_ROUNDS)], rates[int(0.975 * BOOTSTRAP_ROUNDS) - 1]]


def _group(clips: Iterable[ScoredClip], total_errors: int) -> dict[str, Any]:
    clips = list(clips)
    errors = sum(c.score.errors for c in clips)
    return {
        "clips": len(clips),
        "wer": _rate(errors, sum(c.score.ref_words for c in clips)),
        "share_of_errors": errors / total_errors if total_errors else 0.0,
    }


def summarize(clips: Sequence[ScoredClip]) -> dict[str, Any]:
    """A run's headline numbers and its breakdowns by genre and by overlap share."""
    errors = sum(c.score.errors for c in clips)
    words = sum(c.score.ref_words for c in clips)
    by_genre: dict[str, list[ScoredClip]] = defaultdict(list)
    by_overlap: dict[str, list[ScoredClip]] = defaultdict(list)
    for clip in clips:
        by_genre[clip.genre or "unknown"].append(clip)
        by_overlap[clip.overlap].append(clip)
    return {
        "clips": len(clips),
        "episodes": len({c.episode for c in clips}),
        "ref_words": words,
        "errors": errors,
        "wer": _rate(errors, words),
        "wer_ci": _episode_interval(clips),
        "raw_wer": _rate(
            sum(c.score.raw_errors for c in clips), sum(c.score.raw_ref_words for c in clips)
        ),
        "cer": _rate(
            sum(c.score.char_errors for c in clips), sum(c.score.ref_chars for c in clips)
        ),
        "loops": sum(c.score.is_loop for c in clips),
        "by_genre": {name: _group(group, errors) for name, group in sorted(by_genre.items())},
        "by_overlap": {
            name: _group(by_overlap[name], errors) for name in OVERLAP_BUCKETS if name in by_overlap
        },
    }
