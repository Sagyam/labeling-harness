"""How many points of a WER each recording condition costs (D111).

A condition's cost is its attributable errors: the errors on its clips beyond what the same clips
would have made without it, ``errors x (1 - 1/ratio)`` for the ratio of its error rate to a
baseline's. Two ratios are reported, and they answer different questions:

- **within**: the Mantel-Haenszel ratio pooled over the set's own unit (an episode, a speaker),
  so a clip is only compared with baseline clips of its own episode -- same voices, microphone,
  room and topic. This is the estimate of what the condition itself costs.
- **floor**: the crude ratio of the condition's pooled rate to the baseline's over the whole set.
  It also charges the condition for whatever comes with it: on gold, crosstalk lives in podcasts
  and talk shows, which are harder even where one voice speaks.

Pure and importing only the standard library: the dataset's ``harness/`` copy carries it beside
``error_store.py``. Keep it importable on Python 3.10+.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

#: Per group (episode, speaker), ``(errors, reference words)`` in one bucket.
Totals = Mapping[str, tuple[int, int]]


def mh_rate_ratio(
    exposed: Totals, baseline: Totals, groups: Iterable[str]
) -> tuple[float | None, int]:
    """Mantel-Haenszel error-rate ratio of ``exposed`` to ``baseline``, pooled over groups.

    Returns:
        The ratio -- ``None`` when no group holds words in both, or the baseline has no errors
        where it can be compared -- and how many groups could compare. A group repeated in
        ``groups`` (a bootstrap draw) counts each time.
    """
    numerator = denominator = 0.0
    informative = 0
    for group in groups:
        a, t1 = exposed.get(group, (0, 0))
        b, t0 = baseline.get(group, (0, 0))
        if not t1 or not t0:
            continue
        informative += 1
        numerator += a * t0 / (t0 + t1)
        denominator += b * t1 / (t0 + t1)
    return (numerator / denominator if denominator > 0 else None), informative


# --- the card ------------------------------------------------------------------------------------
#
# Every clip falls in one cell: a crosstalk bucket when any overlap was measured; else, when it
# has none, an SNR bucket below 45 dB; else the baseline (no crosstalk, 45+ dB) or unmeasured.
# Crosstalk is compared with every crosstalk-free clip, SNR with crosstalk-free clips at 45+ dB,
# so no clip's errors are charged to two conditions. A condition takes ``E (1 - 1/ratio)`` of its
# errors; what it leaves, with the baseline's and the unmeasured clips' errors, is the rest, which
# is split by kind of error with each clip's errors weighted ``1/ratio`` (assuming a condition
# adds errors in the mix its clips already show). The rows therefore add up to the WER exactly.

#: The kinds of error the rest is split into, in the order they are assigned: a number on either
#: side first, then deletions and insertions, then substitutions by script and similarity.
KINDS = (
    "number",
    "deletion",
    "insertion",
    "script",
    "english",
    "nepali_similar",
    "nepali_other",
)
CROSSTALK = ("0-5%", "5-15%", ">15%")
#: SNR buckets compared against 45+ dB; ``error_mining.SNR_BUCKETS`` without the baseline.
SNR = ("<15 dB", "15-25 dB", "25-35 dB", "35-45 dB")
BASELINE = {"crosstalk": "none", "snr": "45+ dB"}
#: No crosstalk measured; and no crosstalk but no SNR either, which still counts in crosstalk's
#: baseline. Both are reported as unmeasured, and their errors stay in the rest.
UNMEASURED, CLEAN_UNMEASURED = "unmeasured", "none|unmeasured"
METHODS = ("within", "floor")
#: ``error_store``'s resampling: 1000 draws of the set's groups, seeded 0.
ROUNDS = 1000
SEED = 0


class ClipCounts:
    """One clip's words and errors by kind, and where it sits on both conditions."""

    __slots__ = ("crosstalk", "group", "kinds", "snr", "words")

    def __init__(
        self, group: str, crosstalk: str, snr: str, words: int, kinds: Mapping[str, int]
    ) -> None:
        self.group, self.crosstalk, self.snr, self.words = group, crosstalk, snr, words
        self.kinds = {k: int(kinds.get(k, 0)) for k in KINDS}

    @property
    def errors(self) -> int:
        return sum(self.kinds.values())

    @property
    def cell(self) -> str:
        if self.crosstalk in CROSSTALK:
            return f"crosstalk|{self.crosstalk}"
        if self.crosstalk == BASELINE["crosstalk"]:
            if self.snr in SNR:
                return f"snr|{self.snr}"
            if self.snr == BASELINE["snr"]:
                return "baseline"
            return CLEAN_UNMEASURED
        return UNMEASURED


def _per_group(clips: Iterable[ClipCounts]) -> dict[str, dict[str, list[int]]]:
    """``{group: {cell: [words, errors, *errors by kind]}}``."""
    out: dict[str, dict[str, list[int]]] = {}
    for clip in clips:
        cell = out.setdefault(clip.group, {}).setdefault(clip.cell, [0] * (2 + len(KINDS)))
        cell[0] += clip.words
        cell[1] += clip.errors
        for i, kind in enumerate(KINDS):
            cell[2 + i] += clip.kinds[kind]
    return out


def _conditions() -> list[tuple[str, str]]:
    return [("crosstalk", b) for b in CROSSTALK] + [("snr", b) for b in SNR]


def _baseline_cells(factor: str) -> tuple[str, ...]:
    """The cells a factor's buckets are compared against."""
    if factor == "crosstalk":
        return ("baseline", *(f"snr|{b}" for b in SNR), CLEAN_UNMEASURED)
    return ("baseline",)


def _comparisons(groups: dict[str, dict[str, list[int]]]) -> dict[str, tuple[Totals, Totals]]:
    """Per condition cell, its and its baseline's ``(errors, words)`` by group: fixed for the
    set, so every draw only chooses which groups it reads."""

    def totals(cells: Iterable[str]) -> dict[str, tuple[int, int]]:
        cells = list(cells)
        return {
            g: (
                sum(by_cell.get(c, [0, 0])[1] for c in cells),
                sum(by_cell.get(c, [0, 0])[0] for c in cells),
            )
            for g, by_cell in groups.items()
        }

    return {
        f"{factor}|{bucket}": (totals([f"{factor}|{bucket}"]), totals(_baseline_cells(factor)))
        for factor, bucket in _conditions()
    }


def _estimate(
    groups: dict[str, dict[str, list[int]]],
    compared: dict[str, tuple[Totals, Totals]],
    draw: Iterable[str],
) -> dict:
    """Every number of the card over one multiset of groups (all of them, or a draw)."""
    draw = list(draw)
    width = 2 + len(KINDS)
    pooled: dict[str, list[int]] = {}
    for g, times in Counter(draw).items():
        for cell, values in groups.get(g, {}).items():
            into = pooled.setdefault(cell, [0] * width)
            for i in range(width):
                into[i] += values[i] * times
    words = sum(v[0] for v in pooled.values())

    def pooled_rate(cells: Iterable[str]) -> tuple[int, int]:
        cells = list(cells)
        return sum(pooled.get(c, [0, 0])[1] for c in cells), sum(
            pooled.get(c, [0, 0])[0] for c in cells
        )

    out: dict = {"words": words, "errors": sum(v[1] for v in pooled.values()), "cells": {}}
    weights: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    for factor, bucket in _conditions():
        cell = f"{factor}|{bucket}"
        if cell not in pooled:
            continue
        w1, e1 = pooled[cell][0], pooled[cell][1]
        base_cells = _baseline_cells(factor)
        within, informative = mh_rate_ratio(*compared[cell], draw)
        e0, w0 = pooled_rate(base_cells)
        floor = (e1 / w1) / (e0 / w0) if w1 and w0 and e0 else None
        entry = {"groups": informative}
        for method, ratio in (("within", within), ("floor", floor)):
            if e1 == 0:
                attributable, weight = 0.0, 1.0
            elif ratio is None or ratio <= 0:
                attributable, weight = None, 1.0
            else:
                attributable, weight = e1 * (1 - 1 / ratio), 1 / ratio
            weights[method][cell] = weight
            entry[method] = {
                "ratio": ratio,
                "points": None if attributable is None else _points(attributable, words),
            }
        out["cells"][cell] = entry
    for method in METHODS:
        kinds = [0.0] * len(KINDS)
        for cell, values in pooled.items():
            w = weights[method].get(cell, 1.0)
            for i in range(len(KINDS)):
                kinds[i] += values[2 + i] * w
        out[method] = {
            "kinds": [_points(k, words) for k in kinds],
            "rest": _points(sum(kinds), words),
        }
    return out


def _points(errors: float, words: int) -> float:
    return 100 * errors / words if words else 0.0


def _interval(values: list[float | None], *, need: int) -> list[float] | None:
    found = sorted(v for v in values if v is not None)
    if not found or len(found) < need:
        return None
    return [found[int(0.025 * len(found))], found[int(0.975 * len(found)) - 1]]


def card(clips: Iterable[ClipCounts]) -> dict:
    """The WER, the points each recording condition costs by both methods, and the rest by kind
    of error, each with a 95% interval from resampling the set's groups (``None`` with fewer
    than two groups, or when fewer than half the draws could estimate it).

    Returns:
        ``{ref_words, errors, wer, baseline, conditions, factors, rest, unmeasured}``. Each
        condition is ``{factor, bucket, clips, ref_words, errors, wer, within, floor}`` with
        ``{ratio, ratio_ci, points, points_ci}`` per method (and ``groups``, how many could
        compare, within); a ratio that cannot be estimated is ``None`` and its clips' errors
        stay in the rest. ``rest`` is ``{points, points_ci, kinds}`` per method.
    """
    import random

    clips = list(clips)
    groups = _per_group(clips)
    names = sorted(groups)
    compared = _comparisons(groups)
    point = _estimate(groups, compared, names)
    rng = random.Random(SEED)
    draws = [rng.choices(names, k=len(names)) for _ in range(ROUNDS)] if len(names) > 1 else []
    sampled = [_estimate(groups, compared, d) for d in draws]
    need = len(draws) // 2 or 1

    def ci(read) -> list[float] | None:
        return _interval([read(s) for s in sampled], need=need) if draws else None

    conditions = []
    for factor, bucket in _conditions():
        cell = f"{factor}|{bucket}"
        if cell not in point["cells"]:
            continue
        members = [c for c in clips if c.cell == cell]
        errors = sum(c.errors for c in members)
        words = sum(c.words for c in members)
        entry = {
            "factor": factor,
            "bucket": bucket,
            "clips": len(members),
            "ref_words": words,
            "errors": errors,
            "wer": _points(errors, words),
        }
        for method in METHODS:
            found = point["cells"][cell][method]

            def read(s, key, cell=cell, method=method):
                return s["cells"].get(cell, {}).get(method, {}).get(key)

            entry[method] = {
                "ratio": found["ratio"],
                "ratio_ci": ci(lambda s, r=read: r(s, "ratio")) if found["ratio"] else None,
                "points": found["points"],
                "points_ci": ci(lambda s, r=read: r(s, "points"))
                if found["points"] is not None
                else None,
            }
        entry["within"]["groups"] = point["cells"][cell]["groups"]
        conditions.append(entry)

    def factor_points(s, factor: str, method: str) -> float:
        return sum(
            (s["cells"][c][method]["points"] or 0.0)
            for c in s["cells"]
            if c.startswith(f"{factor}|")
        )

    factors = {
        factor: {
            method: {
                "points": factor_points(point, factor, method),
                "points_ci": ci(lambda s, f=factor, m=method: factor_points(s, f, m)),
            }
            for method in METHODS
        }
        for factor in ("crosstalk", "snr")
        if any(c["factor"] == factor for c in conditions)
    }
    rest = {
        method: {
            "points": point[method]["rest"],
            "points_ci": ci(lambda s, m=method: s[m]["rest"]),
            "kinds": [
                {
                    "kind": kind,
                    "points": point[method]["kinds"][i],
                    "points_ci": ci(lambda s, m=method, i=i: s[m]["kinds"][i]),
                }
                for i, kind in enumerate(KINDS)
            ],
        }
        for method in METHODS
    }
    unmeasured = [c for c in clips if c.cell in (UNMEASURED, CLEAN_UNMEASURED)]
    return {
        "ref_words": point["words"],
        "errors": point["errors"],
        "wer": _points(point["errors"], point["words"]),
        "baseline": dict(BASELINE),
        "conditions": conditions,
        "factors": factors,
        "rest": rest,
        "unmeasured": {
            "clips": len(unmeasured),
            "ref_words": sum(c.words for c in unmeasured),
            "errors": sum(c.errors for c in unmeasured),
        },
    }


# --- against a base model ------------------------------------------------------------------------

#: ``[difference, low, high]`` in points; the interval is ``None`` with fewer than two groups.
Diff = list


def _within_values(estimate: dict) -> dict[str, float | None]:
    """The card's within-episode numbers of one estimate, flat: each condition, each factor's
    total, the rest and each kind."""
    out: dict[str, float | None] = {
        "wer": _points(estimate["errors"], estimate["words"]),
        "rest": estimate["within"]["rest"],
    }
    for i, kind in enumerate(KINDS):
        out[f"kind|{kind}"] = estimate["within"]["kinds"][i]
    for factor, bucket in _conditions():
        cell = f"{factor}|{bucket}"
        found = estimate["cells"].get(cell)
        out[cell] = None if found is None else found["within"]["points"]
    for factor in ("crosstalk", "snr"):
        cells = [c for c in estimate["cells"] if c.startswith(f"{factor}|")]
        out[f"factor|{factor}"] = (
            sum(estimate["cells"][c]["within"]["points"] or 0.0 for c in cells) if cells else None
        )
    return out


def difference(run: Iterable[ClipCounts], base: Iterable[ClipCounts]) -> dict:
    """The run's card minus the base's, within episode, row by row, on the same clips.

    ``run`` and ``base`` must hold the same clips (the caller keeps those both scored): a clip's
    cell is its audio's, so only the errors differ. Both are resampled with the same draws of
    groups, so each interval is paired. A row either cannot measure is ``[None, None, None]``.

    Returns:
        ``{wer, rest, kinds: {kind: Diff}, factors: {factor: Diff}, conditions: {cell: Diff}}``,
        each ``Diff`` being ``[run minus base, low, high]`` in points.
    """
    import random

    prepared = []
    for clips in (list(run), list(base)):
        groups = _per_group(clips)
        prepared.append((groups, _comparisons(groups)))
    names = sorted(set(prepared[0][0]) | set(prepared[1][0]))
    rng = random.Random(SEED)
    draws = [rng.choices(names, k=len(names)) for _ in range(ROUNDS)] if len(names) > 1 else []

    def diffs(draw: list[str]) -> dict[str, float | None]:
        mine, theirs = (_within_values(_estimate(g, c, draw)) for g, c in prepared)
        return {
            key: None if mine[key] is None or theirs.get(key) is None else mine[key] - theirs[key]
            for key in mine
        }

    point = diffs(names)
    sampled = [diffs(d) for d in draws]
    need = len(draws) // 2 or 1

    def entry(key: str) -> Diff:
        if point.get(key) is None:
            return [None, None, None]
        ci = _interval([s.get(key) for s in sampled], need=need) if draws else None
        return [point[key], *(ci or [None, None])]

    return {
        "wer": entry("wer"),
        "rest": entry("rest"),
        "kinds": {kind: entry(f"kind|{kind}") for kind in KINDS},
        "factors": {f: entry(f"factor|{f}") for f in ("crosstalk", "snr")},
        "conditions": {f"{f}|{b}": entry(f"{f}|{b}") for f, b in _conditions()},
    }
