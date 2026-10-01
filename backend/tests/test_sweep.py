"""The selection rules and statistics of the training notebooks (D96, D105): notebooks/src/sweep.py.

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


# --- named recipes (D105) ------------------------------------------------------------------------


def _recipe(name: str, seed: int, val: float) -> dict:
    return {"recipe": name, "seed": seed, "val_wer": val, "run_name": f"{name}-s{seed}"}


def test_a_recipe_wins_only_by_more_than_the_vanilla_seed_gap():
    rows = [_recipe("vanilla", 0, 6.0), _recipe("vanilla", 1, 6.4), _recipe("aug-noise", 0, 5.5)]
    winner, why = sweep.choose_recipe(rows)
    assert winner["recipe"] == "aug-noise"
    assert "aug-noise" in why and "vanilla" in why


def test_a_recipe_within_the_seed_gap_loses_to_the_better_vanilla_seed():
    rows = [_recipe("vanilla", 0, 6.4), _recipe("vanilla", 1, 6.0), _recipe("aug-speed", 0, 5.7)]
    winner, _ = sweep.choose_recipe(rows)
    assert (winner["recipe"], winner["seed"]) == ("vanilla", 1)


def test_recipes_and_crosstalk_shares_follow_the_same_rule():
    vals = [(0, 6.0), (1, 6.3)]
    for margin, wins in ((0.31, True), (0.30, False)):
        by_name = [_recipe("vanilla", s, v) for s, v in vals] + [_recipe("aug", 0, 6.0 - margin)]
        by_share = [_row(0.0, s, v) for s, v in vals] + [_row(0.3, 0, 6.0 - margin)]
        assert (sweep.choose_recipe(by_name)[0]["recipe"] == "aug") is wins
        assert (sweep.choose_winner(by_share)[0]["xtalk_p"] == 0.3) is wins


def test_one_run_is_judged_against_the_vanilla_seeds_alone():
    base = [_recipe("vanilla", 0, 6.0), _recipe("vanilla", 1, 6.4)]
    assert sweep.beats_baseline(_recipe("aug-noise", 0, 5.5), base)[0] is True
    assert sweep.beats_baseline(_recipe("aug-codec", 0, 5.7), base)[0] is False
    # another augmented run among the rows does not change the verdict
    assert (
        sweep.beats_baseline(_recipe("aug-codec", 0, 5.7), [*base, _recipe("aug", 0, 1.0)])[0]
        is False
    )


def test_judging_a_run_needs_a_vanilla_run():
    with pytest.raises(ValueError):
        sweep.beats_baseline(_recipe("aug-noise", 0, 5.5), [])


# --- the floor under the seed gap (D109) ---------------------------------------------------------


def test_seeds_that_agree_by_luck_do_not_lower_the_bar_below_the_floor():
    # 03a's seeds: 6.95 and 6.93, a gap of 0.02. A margin of 0.08 clears the gap but not the floor.
    base = [_recipe("vanilla", 0, 6.95), _recipe("vanilla", 1, 6.93)]
    keep, why = sweep.beats_baseline(_recipe("aug-gain", 0, 6.85), base)
    assert keep is False
    assert "floor" in why
    assert sweep.beats_baseline(_recipe("aug-noise", 0, 6.77), base)[0] is True


def test_a_margin_equal_to_the_floor_keeps_vanilla():
    base = [_recipe("vanilla", 0, 6.0), _recipe("vanilla", 1, 6.0)]
    assert sweep.beats_baseline(_recipe("aug", 0, 6.0 - sweep.MIN_NOISE), base)[0] is False
    assert sweep.beats_baseline(_recipe("aug", 0, 5.84), base)[0] is True


def test_a_seed_gap_above_the_floor_is_the_bar():
    rows = [_recipe("vanilla", 0, 6.0), _recipe("vanilla", 1, 6.4), _recipe("aug", 0, 5.7)]
    winner, why = sweep.choose_recipe(rows)
    assert winner["recipe"] == "vanilla"
    assert "0.40" in why


def test_one_vanilla_seed_is_judged_against_the_floor():
    rows = [_recipe("vanilla", 0, 6.0), _recipe("aug", 0, 5.9)]
    winner, why = sweep.choose_recipe(rows)
    assert winner["recipe"] == "vanilla"
    assert "unmeasured" in why


def test_the_floor_can_be_set():
    rows = [_recipe("vanilla", 0, 6.95), _recipe("vanilla", 1, 6.93), _recipe("aug", 0, 6.85)]
    assert sweep.choose_recipe(rows, floor=0.0)[0]["recipe"] == "aug"
    base = rows[:2]
    assert sweep.beats_baseline(rows[2], base, floor=0.0)[0] is True


def test_the_crosstalk_sweep_keeps_its_own_rule_without_a_floor():
    # D96 was fixed before its run; the floor is D109's and applies to named recipes only.
    rows = [_row(0.0, 0, 6.95), _row(0.0, 1, 6.93), _row(0.3, 0, 6.85)]
    assert sweep.choose_winner(rows)[0]["xtalk_p"] == 0.3


# --- weight blends (D105) ------------------------------------------------------------------------


def _blend(alpha: float, val: float) -> dict:
    return {"alpha": alpha, "val_wer": val}


def test_the_blend_closest_to_base_within_the_tolerance_is_kept():
    # the 2026-09-27 grid: 0.5 is within 0.3 of the fine-tune (7.28), 0.25 is not
    rows = [_blend(0.25, 7.89), _blend(0.5, 7.32), _blend(0.75, 7.04), _blend(1.0, 7.28)]
    winner, why = sweep.choose_blend(rows)
    assert winner["alpha"] == 0.5
    assert "0.5" in why


def test_base_itself_is_never_a_blend():
    rows = [_blend(0.0, 7.0), _blend(0.5, 7.2), _blend(1.0, 7.28)]
    assert sweep.choose_blend(rows)[0]["alpha"] == 0.5


def test_without_a_blend_in_tolerance_the_fine_tune_is_kept():
    rows = [_blend(0.25, 9.0), _blend(0.5, 8.0), _blend(1.0, 7.0)]
    winner, why = sweep.choose_blend(rows)
    assert winner["alpha"] == 1.0
    assert "kept" in why


def test_the_tolerance_is_inclusive_and_can_be_set():
    rows = [_blend(0.5, 7.3), _blend(1.0, 7.0)]
    assert sweep.choose_blend(rows)[0]["alpha"] == 0.5
    assert sweep.choose_blend(rows, tolerance=0.1)[0]["alpha"] == 1.0


def test_a_blend_needs_the_fine_tuned_row():
    with pytest.raises(ValueError):
        sweep.choose_blend([_blend(0.25, 7.0), _blend(0.5, 7.0)])


def test_the_blend_rule_never_reads_gold():
    rows = [_blend(0.5, 7.3), _blend(0.75, 7.0), _blend(1.0, 7.0)]
    for r in rows:
        r["gold_wer"] = 1.0 if r["alpha"] == 0.75 else 99.0
    assert sweep.choose_blend(rows)[0]["alpha"] == 0.5


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
