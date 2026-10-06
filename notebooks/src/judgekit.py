"""How much could a judge that picks one of the beam's candidates gain? (roadmap G1)

The LLM-judge pilot starts by measuring its own ceiling. Beam search writes K candidates per val
clip; the oracle picks, per clip, the candidate with the fewest folded errors, which is the best
any judge choosing among them could do. If oracle@K is less than `threshold` points of folded val
WER below the beam's own first choice, no judge can earn its cost and G1 stops there. The rule was
fixed before any candidate was decoded. Gold takes no part in it.

A clip's reference word count is the same for all its candidates, so the per-clip minimum of
errors is also the pooled-WER minimum: the oracle needs no search over combinations.

The pilot (D118, `08b_Judge_Pilot`) then asks whether an LLM that picks among them beats the
cross-model vote: the sample, what the judge is shown, how its answer is read, the n-gram picker
beside it and the kill rule are here too.

Pure Python plus the sweep kit (numpy); importable without torch, on Python 3.10+.
"""

from __future__ import annotations

import math
import multiprocessing
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import sweep

#: The length cap of the standard decoder: the densest training label is 12.6 tokens/s
#: (findings.md, Decoder search), and no clip runs past the 300-token ceiling.
TOKENS_PER_S = 13.0
MAX_TOKENS = 300


def caps(
    durations: Sequence[float], tokens_per_s: float = TOKENS_PER_S, ceiling: int = MAX_TOKENS
) -> list[int]:
    """Each clip's token cap: `min(ceiling, ceil(tokens_per_s * seconds))`."""
    return [min(ceiling, math.ceil(tokens_per_s * d)) for d in durations]


def rows_at_cap(made: int, clip_caps: Sequence[int], num_beams: int) -> list[int]:
    """The rows of a beam batch that have written their clip's cap after `made` new tokens.
    Under beam search row `r` belongs to clip `r // num_beams`."""
    return [r for r in range(len(clip_caps) * num_beams) if made >= clip_caps[r // num_beams]]


def group(flat: Sequence[Any], k: int) -> list[list[Any]]:
    """`generate`'s flat output (`num_return_sequences = k`) as one list per clip, best first."""
    if len(flat) % k:
        raise ValueError(f"{len(flat)} sequences do not split into groups of {k}")
    return [list(flat[i : i + k]) for i in range(0, len(flat), k)]


def oracle_index(per: Sequence[Sequence[dict]], k: int) -> list[int]:
    """Per clip, the rank of the candidate with the fewest errors among its top `k`; a tie goes
    to the higher-ranked one, so the oracle never prefers a candidate the beam scored lower for
    nothing."""
    return [min(range(min(k, len(p))), key=lambda j, p=p: (p[j]["errors"], j)) for p in per]


def decide(top1_wer: float, oracle_wer: float, threshold: float) -> str:
    """`continue` when the oracle is at least `threshold` points below the top candidate."""
    return "continue" if top1_wer - oracle_wer >= threshold - 1e-9 else "stop"


def headroom(
    per: Sequence[Sequence[dict]],
    episodes: Sequence[str],
    summarize: Callable[[list[dict]], dict],
    *,
    ks: Sequence[int],
    threshold: float,
    n: int = 2000,
) -> dict[str, Any]:
    """The headroom report: the top candidate's score, the oracle's at every k, the gain of the
    largest k over the top candidate with a 95% interval from resampling episodes, the rank the
    oracle picked from, and the decision.

    `per` holds each clip's candidates' per-clip counts (`score.per_clip`), best first."""
    if len(per) != len(episodes):
        raise ValueError("per and episodes must describe the same clips")
    top1 = [p[0] for p in per]
    oracle = {}
    for k in ks:
        idx = oracle_index(per, k)
        oracle[str(k)] = summarize([p[i] for p, i in zip(per, idx, strict=True)])
    k_max = max(ks)
    idx = oracle_index(per, k_max)
    best = [p[i] for p, i in zip(per, idx, strict=True)]
    d, lo, hi = sweep.paired_bootstrap(top1, best, episodes, n=n)
    top = summarize(top1)
    return {
        "k": k_max,
        "top1": top,
        "oracle": oracle,
        "gain": -d,
        "gain_ci": [-hi, -lo],
        "ranks": {str(r): c for r, c in sorted(Counter(idx).items())},
        "threshold": threshold,
        "decision": decide(top["wer"], oracle[str(k_max)]["wer"], threshold),
    }


def distinct(cands: Sequence[Sequence[str]], key: Callable[[str], str]) -> dict[str, Any]:
    """How many different candidates each clip has once `key` (the fold) is applied: eight beams
    that differ only by punctuation or a folded spelling are one choice, not eight."""
    per_clip = [len({key(t) for t in cs}) for cs in cands]
    return {
        "per_clip": per_clip,
        "mean": sum(per_clip) / max(len(per_clip), 1),
        "single": sum(n == 1 for n in per_clip),
    }


# --- pickers a judge has to beat that need no LLM ------------------------------------------------


def mbr_index(
    tokens: Sequence[Sequence[Sequence[str]]],
    distance: Callable[[Sequence[str], Sequence[str]], int],
    voters: Sequence[Sequence[Sequence[str]]] | None = None,
) -> list[int]:
    """Per clip, the candidate with the least total `distance` to the clip's other candidates
    (minimum Bayes risk with every candidate weighted alike); a tie goes to the higher-ranked one.

    `tokens` holds each clip's candidates as token lists, best first. It never reads the
    reference, so it is a judge anyone could run: the candidates' own consensus.

    With `voters` (each clip's other transcripts as token lists, one vote each), the cost is the
    distance to the voters instead: the cross-model vote V3 of the pilot (D118), where the
    teacher's greedy decode and the students choose among the beams."""
    if voters is not None and len(voters) != len(tokens):
        raise ValueError("voters must be given for every clip")
    out = []
    for i, cands in enumerate(tokens):
        pool = cands if voters is None else voters[i]
        cost = [sum(distance(a, b) for b in pool) for a in cands]
        out.append(min(range(len(cands)), key=lambda j, cost=cost: (cost[j], j)))
    return out


def random_index(sizes: Sequence[int], seed: int = 0) -> list[int]:
    """One candidate per clip drawn uniformly at random, repeatable from `seed`: the floor a
    judge has to clear to show it reads anything."""
    rng = random.Random(seed)
    return [rng.randrange(n) for n in sizes]


# --- the pilot (D118): what the judge is shown and how its answer is read -----------------------


def sample(ids: Sequence[str], n: int, seed: int) -> list[str]:
    """`n` clip ids drawn at random, repeatable from `seed` whatever order `ids` arrive in. Nothing
    preselects them: a sample chosen by the oracle or by how much the candidates differ would read
    the reference or the answer."""
    pool = sorted(set(ids))
    return random.Random(seed).sample(pool, min(n, len(pool)))


def present(texts: Sequence[str], key: str, seed: int) -> list[int]:
    """The candidates' ranks in the order the judge sees them: exact duplicates (up to spacing)
    shown once as their highest-ranked copy, which has the same errors, then shuffled by `seed`
    and the clip's `key`, because LLMs favour the first candidate shown."""
    seen, ranks = set(), []
    for rank, text in enumerate(texts):
        norm = " ".join(text.split())
        if norm not in seen:
            seen.add(norm)
            ranks.append(rank)
    random.Random(f"{seed}:{key}").shuffle(ranks)
    return ranks


_PROMPT = {
    False: (
        "Below are {n} candidate transcripts of one short clip of speech: Nepali, often mixed "
        "with English, written by a speech recogniser. They differ in a few words."
    ),
    True: (
        "Listen to the audio clip: Nepali speech, often mixed with English. Below are {n} "
        "candidate transcripts of it, written by a speech recogniser. They differ in a few words."
    ),
}
_ASK = (
    "Choose the one that is most likely exactly what was said: the right words, real words, and "
    "the grammar of spoken Nepali. Do not correct or rewrite any of them.\n\n{cands}\n\n"
    "Answer with the number of the best candidate only."
)


def prompt(cands: Sequence[str], *, audio: bool) -> str:
    """The judge's instruction with `cands` numbered from 1 in the order given (D118). The audio
    judge is told to listen; the clip itself goes beside the text in the chat message."""
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(cands, 1))
    return _PROMPT[audio].format(n=len(cands)) + " " + _ASK.format(cands=numbered)


def split_answer(text: str, close: str | None) -> tuple[str, bool]:
    """What the judge answered after its thinking, and whether the thinking ended. `close` is the
    model's end-of-thinking marker, or None when it was not asked to think."""
    if close is None:
        return text.strip(), True
    if close not in text:
        return "", False
    return text.rsplit(close, 1)[1].strip(), True


_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def parse_pick(answer: str, n: int) -> int | None:
    """The shown position (from 0) the judge chose, or None: the answer must hold exactly one
    integer, from 1 to `n`. A None takes the top candidate and is counted as a parse failure."""
    numbers = re.findall(r"[0-9]+", answer.translate(_DEV_DIGITS))
    if len(numbers) != 1 or not 1 <= int(numbers[0]) <= n:
        return None
    return int(numbers[0]) - 1


# --- the pilot: the n-gram picker ----------------------------------------------------------------


class KneserNey:
    """An interpolated Kneser-Ney word n-gram model, the model class KenLM builds (D118), small
    enough to fit the train labels in pure Python. A word it never saw is `<unk>`, which only the
    uniform floor under the unigrams gives any probability."""

    def __init__(self, sentences: Sequence[Sequence[str]], order: int = 3, discount: float = 0.75):
        self.order, self.discount = order, discount
        self.vocab = sorted({w for s in sentences for w in s})
        self._known = set(self.vocab)
        self._size = len(self.vocab) + 2  # every word, </s> and <unk>
        grams: Counter = Counter()
        for s in sentences:
            toks = ["<s>"] * (order - 1) + list(s) + ["</s>"]
            for i in range(order - 1, len(toks)):
                grams[tuple(toks[i - order + 1 : i + 1])] += 1
        tables: dict[int, dict[tuple, Counter]] = {order: defaultdict(Counter)}
        for g, c in grams.items():
            tables[order][g[:-1]][g[-1]] += c
        types = set(grams)
        for k in range(order - 1, 0, -1):  # lower orders count the words seen before, not tokens
            table: dict[tuple, Counter] = defaultdict(Counter)
            lower = {g[1:] for g in types}
            for g in lower:
                table[g[:-1]][g[-1]] += 1
            tables[k], types = table, lower
        self._tables = {k: dict(t) for k, t in tables.items()}
        self._totals = {
            k: {ctx: (sum(c.values()), len(c)) for ctx, c in t.items()}
            for k, t in self._tables.items()
        }

    def _p(self, k: int, ctx: tuple, w: str) -> float:
        if k == 0:
            return 1.0 / self._size
        lower = self._p(k - 1, ctx[1:], w)
        counts = self._tables[k].get(ctx)
        if not counts:
            return lower
        total, types = self._totals[k][ctx]
        own = max(counts.get(w, 0) - self.discount, 0) / total
        return own + self.discount * types / total * lower

    def _word(self, w: str, context: bool = False) -> str:
        return w if w in self._known or w == ("<s>" if context else "</s>") else "<unk>"

    def word_logprob(self, context: Sequence[str], w: str) -> float:
        """Natural log-probability of `w` after the last `order - 1` words of `context`."""
        ctx = tuple(self._word(c, context=True) for c in context)[len(context) - self.order + 1 :]
        return math.log(self._p(self.order, ctx, self._word(w)))

    def logprob(self, tokens: Sequence[str]) -> float:
        """Natural log-probability of a whole sentence, its end included."""
        toks = ["<s>"] * (self.order - 1) + list(tokens) + ["</s>"]
        return sum(
            self.word_logprob(toks[i - self.order + 1 : i], toks[i])
            for i in range(self.order - 1, len(toks))
        )


def lm_index(
    beam: Sequence[Sequence[float]],
    lm: Sequence[Sequence[float]],
    words: Sequence[Sequence[int]],
    lam: float,
) -> list[int]:
    """Per clip, the candidate with the highest beam score plus `lam` times its n-gram
    log-probability per word (its end counted as one); a tie goes to the higher-ranked one, so
    `lam = 0` is the top candidate."""
    out = []
    for b, g, n in zip(beam, lm, words, strict=True):
        total = [b[j] + lam * g[j] / (n[j] + 1) for j in range(len(b))]
        out.append(min(range(len(b)), key=lambda j, t=total: (-t[j], j)))
    return out


def choose_lambda(grid: Sequence[float], wer_of: Callable[[float], float]) -> float:
    """The grid value with the lowest WER, a tie going to the smaller value."""
    return min(grid, key=lambda lam: (round(wer_of(lam), 9), lam))


# --- the pilot: the verdict and what is reported beside it ---------------------------------------


def verdict(
    judge: Sequence[dict],
    vote: Sequence[dict],
    episodes: Sequence[str],
    *,
    level: float,
    n: int = 2000,
) -> dict[str, Any]:
    """D118's kill rule for one judge row: WER(judge) - WER(vote) in points, with a `level`
    interval from resampling episodes; the row passes only if the whole interval is below zero."""
    d, lo, hi = sweep.paired_bootstrap(vote, judge, episodes, n=n, level=level)
    return {"diff": d, "ci": [lo, hi], "level": level, "pass": hi < 0}


def agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """The share of clips on which two pickers pick the same candidate."""
    if len(a) != len(b):
        raise ValueError("a and b must describe the same clips")
    return sum(x == y for x, y in zip(a, b, strict=True)) / max(len(a), 1)


def timing(seconds: Sequence[float], durations: Sequence[float]) -> dict[str, float]:
    """A judge's cost: seconds per clip (mean, median, 90th percentile), in all, and per second
    of the audio judged."""
    s = sorted(seconds)
    total = sum(s)
    return {
        "clips": len(s),
        "total_s": total,
        "mean_s": total / max(len(s), 1),
        "median_s": float(statistics.median(s)) if s else 0.0,
        "p90_s": s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)] if s else 0.0,
        "per_audio_s": total / max(sum(durations), 1e-9),
    }


# --- is the gain a fluke? ------------------------------------------------------------------------


def spread(
    base: Sequence[dict], pick: Sequence[dict], groups: Sequence[str], *, top: int = 5
) -> dict[str, Any]:
    """How a picker's change against `base` is spread over the groups (episodes, speakers):
    how many groups it improves, leaves alone and makes worse, the net errors it removes, and the
    share of that net the `top` most improved groups carry. A gain that one or two rooms carry is
    a fact about those rooms, not about the picker."""
    if not (len(base) == len(pick) == len(groups)):
        raise ValueError("base, pick and groups must describe the same clips")
    removed: dict[str, int] = defaultdict(int)
    for b, p, g in zip(base, pick, groups, strict=True):
        removed[g] += b["errors"] - p["errors"]
    net = sum(removed.values())
    best = sorted(removed.values(), reverse=True)[:top]
    return {
        "groups": len(removed),
        "improved": sum(v > 0 for v in removed.values()),
        "unchanged": sum(v == 0 for v in removed.values()),
        "worse": sum(v < 0 for v in removed.values()),
        "net_errors_removed": net,
        "top": top,
        "top_share": sum(best) / net if net > 0 else None,
    }


# --- aligning many candidates on every core ------------------------------------------------------


def _apply(job: tuple[Callable[[Any], Any], Sequence[Any]]) -> list[Any]:
    fn, items = job
    return [fn(x) for x in items]


def parallel_map(
    fn: Callable[[Any], Any], items: Sequence[Any], *, workers: int, chunk: int = 256
) -> list[Any]:
    """`[fn(x) for x in items]` on `workers` forked processes, in order. Forked, so `fn` may be
    a function of the notebook's own namespace that reads its globals (the scorer is a closure
    and cannot be pickled; a function that calls it can)."""
    if workers <= 1 or len(items) <= chunk:
        return [fn(x) for x in items]
    jobs = [(fn, items[i : i + chunk]) for i in range(0, len(items), chunk)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_apply, jobs)
    return [y for part in parts for y in part]
