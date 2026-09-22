"""The crosstalk sweep (D96): which run's weights are kept, and how two runs are compared.

The notebook trains one model per (XTALK_P, seed) point on the same export. The winner is chosen by
val WER alone, under a rule fixed before any sweep result was read: augmentation has to beat p = 0
by more than the seed noise, measured as the gap between the two p = 0 seeds, or p = 0 is kept.
Gold never takes part in the choice, so every gold number stays a held-out score.

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


def choose_winner(rows: Sequence[dict]) -> tuple[dict, str]:
    """The run whose weights are kept, and why, from rows with `xtalk_p`, `seed` and `val_wer`.

    p = 0 is its better seed. The noise is the gap between its two best seeds; with one p = 0 seed
    it is unmeasured and the rule is strict. The best augmented run wins only if it beats p = 0 by
    more than the noise; a tie goes to p = 0, the simpler recipe."""
    if not rows:
        raise ValueError("no runs to choose from")
    base = sorted((r for r in rows if r["xtalk_p"] == 0), key=lambda r: r["val_wer"])
    aug = sorted((r for r in rows if r["xtalk_p"] != 0), key=lambda r: r["val_wer"])
    if not base or not aug:
        best = (base or aug)[0]
        return best, "only one kind of run, so the lowest val WER"
    if len(base) >= 2:
        noise = base[1]["val_wer"] - base[0]["val_wer"]
        noise_note = f"seed noise {noise:.2f} (p=0 seeds {base[0]['seed']} and {base[1]['seed']})"
    else:
        noise, noise_note = 0.0, "seed noise unmeasured (one p=0 seed), so the rule is strict"
    margin = base[0]["val_wer"] - aug[0]["val_wer"]
    if margin > noise:
        return aug[0], f"p={aug[0]['xtalk_p']} beats p=0 on val by {margin:.2f} > {noise_note}"
    return base[0], f"best augmented margin {margin:.2f} is within {noise_note}; p=0 kept"


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
