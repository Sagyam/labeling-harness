"""The headroom measurement of the G1 pilot (roadmap G1): notebooks/src/judgekit.py.

It runs in Colab, not in the app, but its rule decides whether the LLM-judge pilot goes on at all,
and the rule was fixed before any candidate was decoded, so it is tested here. The per-clip counts
are stand-ins with the shape of ``score.per_clip``'s."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "notebooks" / "src"
sys.path.insert(0, str(_SRC))  # judgekit imports the sweep kit beside it
_spec = importlib.util.spec_from_file_location("judgekit", _SRC / "judgekit.py")
judgekit = importlib.util.module_from_spec(_spec)
sys.modules["judgekit"] = judgekit
_spec.loader.exec_module(judgekit)


def _c(errors: int, words: int = 10) -> dict:
    return {"errors": errors, "words": words, "sub": errors, "del": 0, "ins": 0}


def _summarize(clips: list[dict]) -> dict:
    words = max(sum(c["words"] for c in clips), 1)
    return {"wer": 100 * sum(c["errors"] for c in clips) / words, "clips": len(clips)}


# --- the length cap -----------------------------------------------------------------------------


def test_a_clip_is_capped_at_13_tokens_a_second_and_never_past_300() -> None:
    assert judgekit.caps([1.0, 2.5, 30.0]) == [13, 33, 300]


def test_under_beam_search_row_r_belongs_to_clip_r_over_num_beams() -> None:
    # two clips, three beams each: clip 0 may write 2 tokens, clip 1 may write 5
    assert judgekit.rows_at_cap(2, [2, 5], num_beams=3) == [0, 1, 2]
    assert judgekit.rows_at_cap(1, [2, 5], num_beams=3) == []
    assert judgekit.rows_at_cap(5, [2, 5], num_beams=3) == [0, 1, 2, 3, 4, 5]


# --- grouping and the oracle --------------------------------------------------------------------


def test_flat_beam_output_is_grouped_per_clip_in_rank_order() -> None:
    assert judgekit.group(["a0", "a1", "b0", "b1"], 2) == [["a0", "a1"], ["b0", "b1"]]
    with pytest.raises(ValueError):
        judgekit.group(["a0", "a1", "b0"], 2)


def test_the_oracle_picks_the_fewest_errors_among_the_top_k() -> None:
    per = [[_c(3), _c(1), _c(0)]]
    assert judgekit.oracle_index(per, 1) == [0]
    assert judgekit.oracle_index(per, 2) == [1]
    assert judgekit.oracle_index(per, 3) == [2]


def test_a_tie_goes_to_the_higher_ranked_candidate() -> None:
    assert judgekit.oracle_index([[_c(2), _c(1), _c(1)]], 3) == [1]


def test_k_larger_than_the_candidates_a_clip_has_uses_them_all() -> None:
    assert judgekit.oracle_index([[_c(2), _c(0)]], 8) == [1]


# --- the decision -------------------------------------------------------------------------------


def test_g1_continues_only_when_the_oracle_gains_at_least_the_threshold() -> None:
    assert judgekit.decide(top1_wer=8.0, oracle_wer=7.0, threshold=1.0) == "continue"
    assert judgekit.decide(top1_wer=8.0, oracle_wer=7.01, threshold=1.0) == "stop"


def test_the_report_scores_top1_and_every_k_and_names_the_decision() -> None:
    # two episodes; the second candidate is better on every clip
    per = [[_c(4), _c(1)], [_c(4), _c(1)], [_c(4), _c(1)], [_c(4), _c(1)]]
    episodes = ["e1", "e1", "e2", "e2"]
    report = judgekit.headroom(per, episodes, _summarize, ks=(1, 2), threshold=1.0)
    assert report["top1"]["wer"] == pytest.approx(40.0)
    assert report["oracle"]["2"]["wer"] == pytest.approx(10.0)
    assert report["oracle"]["1"]["wer"] == pytest.approx(40.0)
    assert report["gain"] == pytest.approx(30.0)
    assert report["gain_ci"][0] <= 30.0 <= report["gain_ci"][1]
    assert report["ranks"] == {"1": 4}
    assert report["decision"] == "continue"


def test_no_headroom_stops_the_pilot() -> None:
    per = [[_c(1), _c(1)], [_c(2), _c(3)]]
    report = judgekit.headroom(per, ["e1", "e2"], _summarize, ks=(1, 2), threshold=1.0)
    assert report["gain"] == pytest.approx(0.0)
    assert report["ranks"] == {"0": 2}
    assert report["decision"] == "stop"


# --- is there anything to choose between? -------------------------------------------------------


def test_candidates_that_differ_only_by_the_key_count_once() -> None:
    cands = [["a b.", "a b", "a c"], ["x", "x.", "x!"]]
    out = judgekit.distinct(cands, key=lambda t: t.strip(".!"))
    assert out["per_clip"] == [2, 1]
    assert out["mean"] == pytest.approx(1.5)
    assert out["single"] == 1


# --- pickers a judge has to beat that need no LLM ------------------------------------------------


def _lev(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def test_mbr_picks_the_candidate_closest_to_all_the_others() -> None:
    clip = [s.split() for s in ("x y z", "a b c", "a b c", "a b d")]
    assert judgekit.mbr_index([clip], _lev) == [1]


def test_mbr_ties_go_to_the_higher_ranked_candidate() -> None:
    clip = [s.split() for s in ("a b c", "a b d")]
    assert judgekit.mbr_index([clip], _lev) == [0]


def test_mbr_of_one_candidate_is_that_candidate() -> None:
    assert judgekit.mbr_index([[["a"]]], _lev) == [0]


def test_a_random_pick_is_repeatable_and_in_range() -> None:
    sizes = [8] * 200 + [1]
    first = judgekit.random_index(sizes, seed=0)
    assert first == judgekit.random_index(sizes, seed=0)
    assert first != judgekit.random_index(sizes, seed=1)
    assert all(0 <= i < n for i, n in zip(first, sizes, strict=True))
    assert first[-1] == 0
    assert len(set(first[:200])) == 8  # every rank gets drawn


# --- is the gain a fluke? ------------------------------------------------------------------------


def test_spread_counts_the_groups_a_picker_improves_and_how_concentrated_the_gain_is() -> None:
    base = [_c(5), _c(5), _c(5), _c(5), _c(2)]
    pick = [_c(1), _c(4), _c(5), _c(5), _c(3)]
    groups = ["e1", "e2", "e3", "e3", "e4"]
    out = judgekit.spread(base, pick, groups, top=1)
    assert out["groups"] == 4
    assert out["improved"] == 2
    assert out["worse"] == 1
    assert out["unchanged"] == 1
    # e1 and e2 lose 4 + 1 errors, e4 gains 1: net 4, and e1 alone removed 4 of them
    assert out["net_errors_removed"] == 4
    assert out["top_share"] == pytest.approx(1.0)


def test_spread_with_no_gain_reports_no_share() -> None:
    out = judgekit.spread([_c(1)], [_c(1)], ["e1"], top=5)
    assert out["net_errors_removed"] == 0
    assert out["top_share"] is None


# --- aligning many candidates on every core ------------------------------------------------------


def test_parallel_map_keeps_the_order_of_its_items() -> None:
    items = list(range(-50, 50))
    assert judgekit.parallel_map(abs, items, workers=3, chunk=7) == [abs(i) for i in items]
    assert judgekit.parallel_map(abs, items, workers=1) == [abs(i) for i in items]


# --- the panel vote: other models vote among the beams --------------------------------------------


def test_with_voters_mbr_picks_the_candidate_closest_to_the_voters_not_to_the_beams() -> None:
    # the beams agree with each other on "a b c"; the voters heard "a b d"
    clip = [s.split() for s in ("a b c", "a b c", "a b d")]
    voters = [[s.split() for s in ("a b d", "a b d", "x b d")]]
    assert judgekit.mbr_index([clip], _lev) == [0]
    assert judgekit.mbr_index([clip], _lev, voters=voters) == [2]


def test_a_tie_among_the_voters_goes_to_the_higher_ranked_beam() -> None:
    clip = [s.split() for s in ("a b c", "a b d")]
    voters = [[s.split() for s in ("a b c", "a b d")]]
    assert judgekit.mbr_index([clip], _lev, voters=voters) == [0]


def test_voters_must_be_given_for_every_clip() -> None:
    with pytest.raises(ValueError):
        judgekit.mbr_index([[["a"]], [["b"]]], _lev, voters=[[["a"]]])
