"""How much could a judge that picks one of the beam's candidates gain? (roadmap G1)

The LLM-judge pilot starts by measuring its own ceiling. Beam search writes K candidates per val
clip; the oracle picks, per clip, the candidate with the fewest folded errors, which is the best
any judge choosing among them could do. If oracle@K is less than `threshold` points of folded val
WER below the beam's own first choice, no judge can earn its cost and G1 stops there. The rule was
fixed before any candidate was decoded. Gold takes no part in it.

A clip's reference word count is the same for all its candidates, so the per-clip minimum of
errors is also the pooled-WER minimum: the oracle needs no search over combinations.

Pure Python plus the sweep kit (numpy); importable without torch, on Python 3.10+.
"""

from __future__ import annotations

import math
import multiprocessing
import random
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
) -> list[int]:
    """Per clip, the candidate with the least total `distance` to the clip's other candidates
    (minimum Bayes risk with every candidate weighted alike); a tie goes to the higher-ranked one.

    `tokens` holds each clip's candidates as token lists, best first. It never reads the
    reference, so it is a judge anyone could run: the candidates' own consensus."""
    out = []
    for cands in tokens:
        cost = [sum(distance(a, b) for b in cands) for a in cands]
        out.append(min(range(len(cands)), key=lambda j, cost=cost: (cost[j], j)))
    return out


def random_index(sizes: Sequence[int], seed: int = 0) -> list[int]:
    """One candidate per clip drawn uniformly at random, repeatable from `seed`: the floor a
    judge has to clear to show it reads anything."""
    rng = random.Random(seed)
    return [rng.randrange(n) for n in sizes]


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
