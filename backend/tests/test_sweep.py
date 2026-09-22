"""The crosstalk sweep's selection rule and statistics (D96): notebooks/src/sweep.py.

It runs in the Colab notebook, not in the app, but it decides which weights are kept and what the
paper can claim, and the rule was fixed before any sweep result was read, so it is tested here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "src" / "sweep.py"
_spec = importlib.util.spec_from_file_location("sweep", _PATH)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["sweep"] = sweep
_spec.loader.exec_module(sweep)


def _row(p: float, seed: int, val: float) -> dict:
    return {"xtalk_p": p, "seed": seed, "val_wer": val, "run_name": f"p{p}-s{seed}"}


# --- run names -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "seed", "name"),
    [(0.0, 0, "x-p00-s0"), (0.1, 0, "x-p10-s0"), (0.05, 1, "x-p05-s1"), (0.5, 2, "x-p50-s2")],
)
def test_run_name_encodes_share_and_seed(p, seed, name):
    assert sweep.run_name("x", p, seed) == name


def test_run_names_in_a_grid_are_unique():
    grid = [(0.0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.5, 0), (0.0, 1)]
    assert len({sweep.run_name("x", p, s) for p, s in grid}) == len(grid)


# --- the winner ----------------------------------------------------------------------------------


def test_augmentation_wins_only_by_more_than_the_seed_noise():
    # p=0 seeds differ by 0.4: that is the noise. p=0.3 beats the better p=0 by 0.5 > 0.4.
    rows = [_row(0.0, 0, 6.0), _row(0.0, 1, 6.4), _row(0.3, 0, 5.5), _row(0.1, 0, 5.9)]
    winner, why = sweep.choose_winner(rows)
    assert (winner["xtalk_p"], winner["seed"]) == (0.3, 0)
    assert "noise" in why


def test_a_margin_within_the_seed_noise_goes_to_p0():
    rows = [_row(0.0, 0, 6.0), _row(0.0, 1, 6.4), _row(0.3, 0, 5.7)]  # margin 0.3 <= noise 0.4
    winner, _ = sweep.choose_winner(rows)
    assert winner["xtalk_p"] == 0.0


def test_a_margin_equal_to_the_noise_goes_to_p0():
    rows = [_row(0.0, 0, 6.0), _row(0.0, 1, 6.5), _row(0.2, 0, 5.5)]
    assert sweep.choose_winner(rows)[0]["xtalk_p"] == 0.0


def test_p0_is_its_better_seed():
    rows = [_row(0.0, 0, 6.4), _row(0.0, 1, 6.0), _row(0.3, 0, 6.2)]
    winner, _ = sweep.choose_winner(rows)
    assert (winner["xtalk_p"], winner["seed"]) == (0.0, 1)


def test_the_best_augmented_run_is_the_candidate_not_the_first():
    rows = [_row(0.0, 0, 6.0), _row(0.0, 1, 6.1), _row(0.1, 0, 5.85), _row(0.5, 0, 5.0)]
    winner, _ = sweep.choose_winner(rows)
    assert winner["xtalk_p"] == 0.5


def test_without_a_replicate_the_noise_is_unmeasured_and_the_rule_is_strict():
    rows = [_row(0.0, 0, 6.0), _row(0.3, 0, 5.99)]
    winner, why = sweep.choose_winner(rows)
    assert winner["xtalk_p"] == 0.3
    assert "unmeasured" in why


def test_a_single_run_wins_by_itself():
    winner, _ = sweep.choose_winner([_row(0.3, 0, 6.0)])
    assert winner["xtalk_p"] == 0.3


def test_no_runs_is_an_error():
    with pytest.raises(ValueError):
        sweep.choose_winner([])


def test_the_rule_never_reads_gold():
    rows = [_row(0.0, 0, 6.0), _row(0.0, 1, 6.4), _row(0.3, 0, 5.7)]
    for r in rows:
        r["gold_wer"] = 99.0 if r["xtalk_p"] == 0.0 else 1.0
    assert sweep.choose_winner(rows)[0]["xtalk_p"] == 0.0


# --- paired bootstrap ----------------------------------------------------------------------------


def _counts(errors: list[int], words: int = 10) -> list[dict]:
    return [{"errors": e, "words": words} for e in errors]


def test_identical_runs_differ_by_zero():
    a = _counts([1, 2, 3, 0])
    diff, lo, hi = sweep.paired_bootstrap(a, a, ["e1", "e1", "e2", "e2"], n=200)
    assert (diff, lo, hi) == (0.0, 0.0, 0.0)


def test_the_point_estimate_is_the_difference_in_pooled_wer():
    a, b = _counts([1, 1, 1, 1]), _counts([2, 2, 2, 2])  # 10% vs 20%
    diff, lo, hi = sweep.paired_bootstrap(a, b, ["e1", "e2", "e3", "e4"], n=200)
    assert diff == pytest.approx(10.0)
    assert lo <= diff <= hi


def test_a_run_worse_on_every_episode_has_an_interval_above_zero():
    a = _counts([1, 2, 1, 2, 1, 2, 1, 2])
    b = _counts([3, 4, 3, 4, 3, 4, 3, 4])
    diff, lo, _ = sweep.paired_bootstrap(a, b, [f"e{i // 2}" for i in range(8)], n=500)
    assert diff > 0 and lo > 0


def test_resampling_is_by_episode_so_one_episode_cannot_look_like_many():
    # The whole difference sits in one episode. Resampled by clip, eight clips would give a tight
    # interval; resampled by episode, that episode is often left out and the interval reaches 0.
    a = _counts([0] * 8 + [1] * 8)
    b = _counts([5] * 8 + [1] * 8)
    episodes = ["hot"] * 8 + [f"e{i}" for i in range(8)]
    _, lo, _ = sweep.paired_bootstrap(a, b, episodes, n=2000)
    assert lo <= 0.0


def test_the_bootstrap_is_repeatable_from_its_seed():
    a, b = _counts([1, 3, 0, 2, 5]), _counts([2, 1, 1, 4, 3])
    eps = ["a", "b", "c", "d", "e"]
    assert sweep.paired_bootstrap(a, b, eps, n=300, seed=7) == sweep.paired_bootstrap(
        a, b, eps, n=300, seed=7
    )


def test_misaligned_inputs_are_an_error():
    with pytest.raises(ValueError):
        sweep.paired_bootstrap(_counts([1, 2]), _counts([1]), ["e", "e"])
