"""Score a fine-tuned model's transcripts against the labels, clip by clip (D83).

The model runs in the training notebook, on a GPU; the harness only scores what it wrote. The
numbers must be the notebook's numbers, so every rule here is the notebook's rule
(``ftkit.harness_scorer``): WER over :func:`app.services.fold.word_errors`, folded and raw, errors
pooled over words rather than averaged over clips; CER over the folded tokens joined by spaces,
Latin lower-cased; a loop is a 3-word run repeated five or more times.

Scoring is pure: :func:`score_clip` and :func:`summarize` take text and return numbers, and only
:mod:`app.services.model_import` touches the database.

**Classes (D87).** Every clip carries its classes (:mod:`app.services.clip_classes`), and a run
breaks down by every axis. Raw WER per bucket mixes the bucket with the episodes that happen to
fill it -- overlap lives in podcasts, and podcasts are harder anyway -- so each bucket also gets a
**within-episode rate ratio** against its axis's baseline: the Mantel-Haenszel ratio of error
rates pooled over episodes, each episode comparing its own clips in the bucket with its own
clips in the baseline. It answers the question the crosstalk study asked with a fixed-effects
Poisson fit (docs/findings.md), one axis at a time. Its interval resamples whole episodes. A
ratio whose interval holds 1 is a condition ruled out, at this sample size, as a source of
errors.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.clip_classes import AXES, Axis
from app.services.fold import Ruleset, fold_tokens, word_errors

#: Episode resamples behind a WER interval. Seeded, so the same run always reports the same one.
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 0

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
    """A clip's score with what the breakdowns group it by: its episode, genre and classes."""

    episode: str
    genre: str | None
    classes: Mapping[str, str]
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
        "cer": _rate(
            sum(c.score.char_errors for c in clips), sum(c.score.ref_chars for c in clips)
        ),
        "share_of_errors": errors / total_errors if total_errors else 0.0,
    }


#: Per episode, (errors, reference words) in one bucket.
_Totals = dict[str, tuple[int, int]]


def _totals(clips: Iterable[ScoredClip]) -> _Totals:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for clip in clips:
        out[clip.episode][0] += clip.score.errors
        out[clip.episode][1] += clip.score.ref_words
    return {episode: (e, w) for episode, (e, w) in out.items()}


def mh_rate_ratio(
    exposed: _Totals, baseline: _Totals, episodes: Iterable[str]
) -> tuple[float | None, int]:
    """Mantel-Haenszel error-rate ratio of ``exposed`` to ``baseline``, pooled over episodes.

    Returns:
        The ratio -- ``None`` when no episode holds words in both, or the baseline has no errors
        where it can be compared -- and how many episodes could compare. An episode repeated in
        ``episodes`` (a bootstrap draw) counts each time.
    """
    numerator = denominator = 0.0
    informative = 0
    for episode in episodes:
        a, t1 = exposed.get(episode, (0, 0))
        b, t0 = baseline.get(episode, (0, 0))
        if not t1 or not t0:
            continue
        informative += 1
        numerator += a * t0 / (t0 + t1)
        denominator += b * t1 / (t0 + t1)
    return (numerator / denominator if denominator > 0 else None), informative


def _ratio_interval(
    exposed: _Totals, baseline: _Totals, draws: Sequence[Sequence[str]]
) -> list[float] | None:
    """95% interval of the ratio over episode resamples; ``None`` when too few draws give one."""
    ratios = sorted(
        r for draw in draws if (r := mh_rate_ratio(exposed, baseline, draw)[0]) is not None
    )
    if len(ratios) < len(draws) // 2:
        return None
    return [ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]]


def _axis_breakdown(
    axis: Axis, clips: Sequence[ScoredClip], total_errors: int, draws: Sequence[Sequence[str]]
) -> dict[str, Any]:
    groups: dict[str, list[ScoredClip]] = defaultdict(list)
    for clip in clips:
        if axis.name in clip.classes:
            groups[clip.classes[axis.name]].append(clip)
    order = list(axis.buckets) or sorted(groups, key=lambda b: (b == axis.unmeasured, b))
    out = {bucket: _group(groups[bucket], total_errors) for bucket in order if bucket in groups}
    if axis.baseline is None or axis.baseline not in groups:
        return out
    baseline = _totals(groups[axis.baseline])
    episodes = sorted({c.episode for c in clips})
    for bucket, entry in out.items():
        if bucket in (axis.baseline, axis.unmeasured):
            continue
        exposed = _totals(groups[bucket])
        ratio, informative = mh_rate_ratio(exposed, baseline, episodes)
        entry["rate_ratio"] = ratio
        entry["rate_ratio_episodes"] = informative
        entry["rate_ratio_ci"] = (
            _ratio_interval(exposed, baseline, draws)
            if ratio is not None and informative >= 2
            else None
        )
    return out


def _draws(episodes: Sequence[str]) -> list[list[str]]:
    """The episode resamples every ratio's interval shares, seeded like the WER interval's."""
    rng = random.Random(BOOTSTRAP_SEED)
    return [rng.choices(episodes, k=len(episodes)) for _ in range(BOOTSTRAP_ROUNDS)]


def summarize(clips: Sequence[ScoredClip]) -> dict[str, Any]:
    """A run's headline numbers and its breakdowns by genre and by every class."""
    errors = sum(c.score.errors for c in clips)
    words = sum(c.score.ref_words for c in clips)
    by_genre: dict[str, list[ScoredClip]] = defaultdict(list)
    for clip in clips:
        by_genre[clip.genre or "unknown"].append(clip)
    draws = _draws(sorted({c.episode for c in clips})) if clips else []
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
        "by_class": {
            axis.name: breakdown
            for axis in AXES
            if (breakdown := _axis_breakdown(axis, clips, errors, draws))
        },
    }
