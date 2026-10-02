"""How many points of a WER each recording condition costs (D111)."""

from __future__ import annotations

import pytest

from app.services.attribution import KINDS, ClipCounts, card, mh_rate_ratio


def _clip(group: str, words: int, errors: int, crosstalk: str = "none", snr: str = "45+ dB",
          kind: str = "nepali_other") -> ClipCounts:  # fmt: skip
    return ClipCounts(group=group, crosstalk=crosstalk, snr=snr, words=words,
                      kinds={kind: errors})  # fmt: skip


def _condition(got: dict, factor: str, bucket: str) -> dict:
    (found,) = [c for c in got["conditions"] if (c["factor"], c["bucket"]) == (factor, bucket)]
    return found


def _sums_to_the_wer(got: dict, method: str) -> None:
    conditions = sum(c[method]["points"] or 0.0 for c in got["conditions"])
    assert conditions + got["rest"][method]["points"] == pytest.approx(got["wer"])
    assert sum(k["points"] for k in got["rest"][method]["kinds"]) == pytest.approx(
        got["rest"][method]["points"]
    )
    for factor, entry in got["factors"].items():
        members = [c for c in got["conditions"] if c["factor"] == factor]
        assert entry[method]["points"] == pytest.approx(
            sum(c[method]["points"] or 0.0 for c in members)
        )


def test_one_episode_by_hand() -> None:
    """Clean 5 errors in 100 words, crosstalk 20 in 100: ratio 4, so 15 of the 20 are its."""
    got = card([_clip("e1", 100, 5), _clip("e1", 100, 20, crosstalk=">15%")])
    assert got["wer"] == pytest.approx(12.5)
    xt = _condition(got, "crosstalk", ">15%")
    assert (xt["clips"], xt["ref_words"], xt["errors"], xt["wer"]) == (1, 100, 20, 20.0)
    for method in ("within", "floor"):
        assert xt[method]["ratio"] == pytest.approx(4.0)
        assert xt[method]["points"] == pytest.approx(7.5)  # 15 errors of 200 words
        assert got["rest"][method]["points"] == pytest.approx(5.0)
        _sums_to_the_wer(got, method)
    assert xt["within"]["groups"] == 1


def test_the_floor_also_charges_crosstalk_for_the_show_it_comes_in() -> None:
    """Crosstalk only in the hard episode: within it doubles the rate, against the set's clean
    clips it looks like 3.3 times."""
    clips = [
        _clip("easy", 100, 2),
        _clip("hard", 100, 10),
        _clip("hard", 100, 20, crosstalk="5-15%"),
    ]
    got = card(clips)
    xt = _condition(got, "crosstalk", "5-15%")
    assert xt["within"]["ratio"] == pytest.approx(2.0)
    assert xt["within"]["points"] == pytest.approx(100 * 10 / 300)
    assert xt["floor"]["ratio"] == pytest.approx(20 / 6)
    assert xt["floor"]["points"] == pytest.approx(100 * 20 * (1 - 6 / 20) / 300)
    assert xt["floor"]["points"] > xt["within"]["points"]
    for method in ("within", "floor"):
        _sums_to_the_wer(got, method)


def test_snr_is_measured_on_crosstalk_free_clips_against_45_db() -> None:
    clips = [
        _clip("e1", 100, 4),  # baseline for both
        _clip("e1", 100, 8, snr="<15 dB"),
        _clip("e1", 100, 30, crosstalk=">15%", snr="<15 dB"),  # crosstalk, not SNR
    ]
    got = card(clips)
    noisy = _condition(got, "snr", "<15 dB")
    assert (noisy["clips"], noisy["errors"]) == (1, 8)
    assert noisy["within"]["ratio"] == pytest.approx(2.0)
    assert noisy["within"]["points"] == pytest.approx(100 * 4 / 300)
    xt = _condition(got, "crosstalk", ">15%")
    # The crosstalk baseline is every crosstalk-free clip, whatever its SNR: 12 in 200.
    assert xt["floor"]["ratio"] == pytest.approx(30 / 6)
    assert got["baseline"] == {"crosstalk": "none", "snr": "45+ dB"}
    for method in ("within", "floor"):
        _sums_to_the_wer(got, method)


def test_a_crosstalk_free_clip_without_an_snr_is_still_crosstalks_baseline() -> None:
    clips = [_clip("e1", 100, 5, snr="unmeasured"), _clip("e1", 100, 20, crosstalk=">15%")]
    xt = _condition(card(clips), "crosstalk", ">15%")
    assert xt["within"]["ratio"] == pytest.approx(4.0)


def test_a_condition_no_episode_can_compare_is_not_estimated_within_and_stays_in_the_rest() -> None:
    clips = [_clip("e1", 100, 5), _clip("e2", 100, 20, crosstalk=">15%")]
    got = card(clips)
    xt = _condition(got, "crosstalk", ">15%")
    assert (xt["within"]["ratio"], xt["within"]["points"], xt["within"]["groups"]) == (
        None,
        None,
        0,
    )
    assert got["rest"]["within"]["points"] == pytest.approx(got["wer"])
    assert xt["floor"]["ratio"] == pytest.approx(4.0)
    for method in ("within", "floor"):
        _sums_to_the_wer(got, method)


def test_unmeasured_clips_are_counted_and_left_in_the_rest() -> None:
    clips = [
        _clip("e1", 100, 5),
        _clip("e1", 50, 9, crosstalk="unmeasured"),
        _clip("e1", 50, 3, snr="unmeasured"),
    ]
    got = card(clips)
    assert got["unmeasured"] == {"clips": 2, "ref_words": 100, "errors": 12}
    assert got["conditions"] == []
    assert got["rest"]["within"]["points"] == pytest.approx(got["wer"])


def test_the_rest_is_split_by_kind_of_error_with_each_conditions_share_taken_out() -> None:
    """A crosstalk clip at ratio 4 keeps a quarter of its errors in the rest, in its own mix."""
    clips = [
        ClipCounts("e1", "none", "45+ dB", 100, {"number": 2, "deletion": 3}),
        ClipCounts("e1", ">15%", "45+ dB", 100, {"insertion": 12, "deletion": 8}),
    ]
    got = card(clips)
    kinds = {k["kind"]: k["points"] for k in got["rest"]["within"]["kinds"]}
    assert list(kinds) == list(KINDS)
    assert kinds["number"] == pytest.approx(1.0)  # 2 of 200
    assert kinds["deletion"] == pytest.approx(100 * (3 + 8 / 4) / 200)
    assert kinds["insertion"] == pytest.approx(100 * (12 / 4) / 200)
    _sums_to_the_wer(got, "within")


def test_intervals_resample_the_sets_own_unit() -> None:
    clips = []
    for g in range(6):
        clips += [_clip(f"e{g}", 100, 4 + g), _clip(f"e{g}", 100, 10 + 2 * g, crosstalk=">15%")]
    got = card(clips)
    xt = _condition(got, "crosstalk", ">15%")
    for method in ("within", "floor"):
        low, high = xt[method]["points_ci"]
        assert low <= xt[method]["points"] <= high
        low, high = xt[method]["ratio_ci"]
        assert low <= xt[method]["ratio"] <= high
        low, high = got["rest"][method]["points_ci"]
        assert low <= got["rest"][method]["points"] <= high
        assert all(k["points_ci"] is not None for k in got["rest"][method]["kinds"])
    assert card(clips) == got, "seeded: the same clips give the same intervals"
    one = card([_clip("e1", 100, 5), _clip("e1", 100, 20, crosstalk=">15%")])
    assert _condition(one, "crosstalk", ">15%")["within"]["points_ci"] is None


def test_an_empty_set_has_an_empty_card() -> None:
    got = card([])
    assert (got["wer"], got["conditions"], got["rest"]["within"]["points"]) == (0.0, [], 0.0)


def test_mh_ratio_counts_a_repeated_group_each_time() -> None:
    exposed, baseline = {"a": (10, 100), "b": (4, 100)}, {"a": (5, 100), "b": (4, 100)}
    ratio, informative = mh_rate_ratio(exposed, baseline, ["a", "b"])
    assert (ratio, informative) == (pytest.approx(14 / 9), 2)
    assert mh_rate_ratio(exposed, baseline, ["a", "a"]) == (pytest.approx(2.0), 2)


# --- against a base model ------------------------------------------------------------------------


def test_against_itself_every_row_moves_by_nothing() -> None:
    from app.services.attribution import difference

    clips = []
    for g in range(4):
        clips += [_clip(f"e{g}", 100, 4 + g), _clip(f"e{g}", 100, 9 + g, crosstalk=">15%")]
    got = difference(clips, clips)
    assert got["wer"][0] == 0.0
    assert got["factors"]["crosstalk"][0] == 0.0
    assert got["rest"][0] == 0.0
    assert all(v[0] == 0.0 for v in got["kinds"].values())


def test_a_gain_on_crosstalk_clips_is_charged_to_crosstalk_by_hand() -> None:
    """Base: clean 5 in 100, crosstalk 20 in 100 (ratio 4, 7.5 points). Run: crosstalk 10 in
    100 (ratio 2, 2.5 points). The 5 points of WER it gained are all crosstalk's."""
    from app.services.attribution import difference

    base = [_clip("e1", 100, 5), _clip("e1", 100, 20, crosstalk=">15%")]
    run = [_clip("e1", 100, 5), _clip("e1", 100, 10, crosstalk=">15%")]
    got = difference(run, base)
    assert got["wer"][0] == pytest.approx(-5.0)
    assert got["factors"]["crosstalk"][0] == pytest.approx(-5.0)
    assert got["conditions"]["crosstalk|>15%"][0] == pytest.approx(-5.0)
    assert got["rest"][0] == pytest.approx(0.0)
    assert got["wer"][1:] == [None, None], "one episode: no interval"


def test_differences_carry_paired_intervals_and_skip_what_either_cannot_measure() -> None:
    from app.services.attribution import difference

    base, run = [], []
    for g in range(6):
        base += [_clip(f"e{g}", 100, 5 + g), _clip(f"e{g}", 100, 20 + g, crosstalk=">15%")]
        run += [_clip(f"e{g}", 100, 5 + g), _clip(f"e{g}", 100, 12 + g, crosstalk=">15%")]
    base.append(_clip("lonely", 100, 9, crosstalk="0-5%"))  # no clean clip in its episode
    run.append(_clip("lonely", 100, 3, crosstalk="0-5%"))
    got = difference(run, base)
    d, lo, hi = got["factors"]["crosstalk"]
    assert lo <= d <= hi < 0
    assert got["conditions"]["crosstalk|0-5%"] == [None, None, None]
