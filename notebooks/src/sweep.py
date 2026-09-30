"""Which run's weights are kept, and how two runs are compared (D96, D105).

A notebook trains several runs on the same export: the crosstalk sweep's `(XTALK_P, seed)` points
(D96), the augmentation ablation's named recipes, or the weight blends of one fine-tune (D105).
Each winner is chosen by val WER alone, under a rule fixed before any result was read. For
training runs: a recipe has to beat the baseline by more than the seed noise, measured as the gap
between the baseline's two seeds, or the baseline is kept. Gold never takes part in a choice, so
every gold number stays a held-out score.

Runs are compared on gold with a paired bootstrap that resamples whole episodes: clips of one
episode share a room and voices, and resampling them one by one would pretend to more independent
evidence than gold holds (findings.md, *How big gold has to be*). Pure numpy; importable without
torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def run_name(prefix: str, p: float, seed: int) -> str:
    """`<prefix>-p30-s0` for XTALK_P = 0.3 and seed 0: the run's folder and Models page id."""
    return f"{prefix}-p{round(p * 100):02d}-s{seed}"


def _winner(base: list[dict], other: list[dict], base_name: str, label) -> tuple[dict, str]:
    """The rule both choosers share. `base` and `other` are sorted by val WER, best first."""
    if not base or not other:
        best = (base or other)[0]
        return best, "only one kind of run, so the lowest val WER"
    if len(base) >= 2:
        noise = base[1]["val_wer"] - base[0]["val_wer"]
        seeds = f"{base_name} seeds {base[0]['seed']} and {base[1]['seed']}"
        noise_note = f"seed noise {noise:.2f} ({seeds})"
    else:
        noise = 0.0
        noise_note = f"seed noise unmeasured (one {base_name} seed), so the rule is strict"
    margin = base[0]["val_wer"] - other[0]["val_wer"]
    if margin > noise:
        return other[
            0
        ], f"{label(other[0])} beats {base_name} on val by {margin:.2f} > {noise_note}"
    return base[0], f"best augmented margin {margin:.2f} is within {noise_note}; {base_name} kept"


def choose_winner(rows: Sequence[dict]) -> tuple[dict, str]:
    """The run whose weights are kept, and why, from rows with `xtalk_p`, `seed` and `val_wer`.

    p = 0 is its better seed. The noise is the gap between its two best seeds; with one p = 0 seed
    it is unmeasured and the rule is strict. The best augmented run wins only if it beats p = 0 by
    more than the noise; a tie goes to p = 0, the simpler recipe."""
    if not rows:
        raise ValueError("no runs to choose from")
    base = sorted((r for r in rows if r["xtalk_p"] == 0), key=lambda r: r["val_wer"])
    aug = sorted((r for r in rows if r["xtalk_p"] != 0), key=lambda r: r["val_wer"])
    return _winner(base, aug, "p=0", lambda r: f"p={r['xtalk_p']}")


#: The recipe every augmentation is measured against (D105): today's training, nothing added.
BASELINE = "vanilla"


def choose_recipe(rows: Sequence[dict], baseline: str = BASELINE) -> tuple[dict, str]:
    """`choose_winner` for named recipes (D105): rows carry `recipe`, `seed` and `val_wer`.

    The baseline is its better seed, and the noise is the gap between its two best seeds. Another
    recipe wins only if it beats the baseline on val by more than that; otherwise the baseline is
    kept. Gold and the public sets never take part."""
    if not rows:
        raise ValueError("no runs to choose from")
    base = sorted((r for r in rows if r["recipe"] == baseline), key=lambda r: r["val_wer"])
    other = sorted((r for r in rows if r["recipe"] != baseline), key=lambda r: r["val_wer"])
    return _winner(base, other, baseline, lambda r: r["recipe"])


def beats_baseline(
    row: dict, baseline_rows: Sequence[dict], baseline: str = BASELINE
) -> tuple[bool, str]:
    """Whether one run clears the rule on its own, and why: what decides, as each ablation run
    finishes, if its stage goes into the combined recipe and its weights are kept."""
    base = [r for r in baseline_rows if r["recipe"] == baseline]
    if not base:
        raise ValueError(f"no {baseline} run to measure against")
    winner, why = choose_recipe([*base, row], baseline)
    return winner is row, why


def choose_blend(rows: Sequence[dict], tolerance: float = 0.3) -> tuple[dict, str]:
    """The weight blend that is kept, and why, from rows with `alpha` and `val_wer`.

    A blend is `(1 - alpha) * base + alpha * fine-tuned` (WiSE-FT), so alpha = 1 is the fine-tuned
    model and must be among the rows. The rule, fixed on 2026-09-27 before any blend was scored:
    the blend closest to base whose val WER is within `tolerance` points of the fine-tuned
    model's. Base itself (alpha = 0) is never a candidate, and if no blend qualifies the
    fine-tuned model is kept. Gold and the public sets never take part."""
    tuned = [r for r in rows if r["alpha"] == 1]
    if len(tuned) != 1:
        raise ValueError("the rule needs exactly one alpha = 1 row, the fine-tuned model")
    limit = tuned[0]["val_wer"] + tolerance
    ok = sorted(
        (r for r in rows if 0 < r["alpha"] < 1 and r["val_wer"] <= limit), key=lambda r: r["alpha"]
    )
    if not ok:
        return tuned[0], f"no blend is within {tolerance:g} of the fine-tuned val WER; it is kept"
    return ok[0], (
        f"alpha={ok[0]['alpha']:g} is the blend closest to base with val WER "
        f"{ok[0]['val_wer']:.2f} <= {tuned[0]['val_wer']:.2f} + {tolerance:g}"
    )


def paired_bootstrap(
    a: Sequence[dict],
    b: Sequence[dict],
    episodes: Sequence[str],
    *,
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """WER(b) - WER(a) in points, with a 95% interval from resampling episodes.

    `a` and `b` are per-clip `{"errors", "words"}` for the same clips in the same order, and
    `episodes` names each clip's episode. WER is pooled (sum of errors over sum of words)."""
    if not (len(a) == len(b) == len(episodes)):
        raise ValueError("a, b and episodes must describe the same clips")
    ids = sorted(set(episodes))
    index = {e: i for i, e in enumerate(ids)}
    ep = np.array([index[e] for e in episodes])
    per = np.zeros((len(ids), 3))  # errors of a, errors of b, reference words
    clips = [[x["errors"], y["errors"], x["words"]] for x, y in zip(a, b, strict=True)]
    np.add.at(per, ep, np.array(clips))

    def diff(t: np.ndarray) -> float:
        return float(100 * (t[1] - t[0]) / max(t[2], 1))

    point = diff(per.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n, len(ids)))
    totals = per[draws].sum(axis=1)  # (n, 3)
    boot = 100 * (totals[:, 1] - totals[:, 0]) / np.maximum(totals[:, 2], 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)
