"""The speech-LLM bake-off's rules (D121): notebooks/src/bakeoffkit.py.

It runs in Colab, not in the app, but its gate decides which speech-LLMs go on to be students, and
the gate was fixed before any candidate transcribed a clip, so it is tested here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "notebooks" / "src"
_spec = importlib.util.spec_from_file_location("bakeoffkit", _SRC / "bakeoffkit.py")
bakeoffkit = importlib.util.module_from_spec(_spec)
sys.modules["bakeoffkit"] = bakeoffkit
_spec.loader.exec_module(bakeoffkit)

NEPALI = "म आज बजार गएँ"  # ma aaja bajaar gae~
MIXED = "यो phone राम्रो छ"  # yo phone raamro chha
ENGLISH = "this is a good phone"


# --- scripts -------------------------------------------------------------------------------------


def test_letters_are_counted_by_script_with_vowel_signs_as_devanagari() -> None:
    counts = bakeoffkit.script_letters("नमस्ते hello")  # namaste hello
    assert counts == {"deva": 6, "latin": 5, "other": 0}


def test_digits_punctuation_and_spaces_are_not_letters() -> None:
    assert bakeoffkit.script_letters("१२३ 123, ।?!  ") == {"deva": 0, "latin": 0, "other": 0}


def test_a_letter_of_another_script_is_other() -> None:
    assert bakeoffkit.script_letters("বাংলা 中文")["other"] == 7  # Bengali (incl. a vowel sign), Han


def test_shares_are_zero_without_letters() -> None:
    assert bakeoffkit.script_shares("") == {"deva": 0.0, "latin": 0.0, "other": 0.0}


def test_shares_sum_to_one() -> None:
    shares = bakeoffkit.script_shares(MIXED)  # 9 Devanagari letters and signs, 5 Latin
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["latin"] == pytest.approx(5 / 14)


# --- cleaning ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  यो\nphone  राम्रो\tछ ", "यो phone राम्रो छ"),
        ('"यो phone राम्रो छ"', "यो phone राम्रो छ"),
        ("“यो phone”", "यो phone"),
        ("`यो`", "यो"),
        ('"one" and "two"', '"one" and "two"'),  # quotes that do not wrap the whole answer stay
        ("", ""),
    ],
)
def test_clean_only_collapses_whitespace_and_strips_wrapping_quotes(raw: str, clean: str) -> None:
    assert bakeoffkit.clean(raw) == clean


def test_clean_keeps_a_preamble_the_gate_and_the_insertions_have_to_see() -> None:
    text = "Here is the transcription: यो phone राम्रो छ"
    assert bakeoffkit.clean(text) == text


# --- the gate ------------------------------------------------------------------------------------


def test_a_clip_counts_toward_the_gate_when_its_reference_is_mostly_devanagari() -> None:
    report = bakeoffkit.zero_shot_report([NEPALI, MIXED, ENGLISH], [NEPALI, MIXED, ENGLISH])
    assert report["clips"] == 3
    assert report["nepali_clips"] == 2  # the English reference is not a test of writing Nepali


def test_the_gate_passes_at_ninety_percent_of_nepali_clips() -> None:
    refs = [NEPALI] * 10
    hyps = [NEPALI] * 9 + [ENGLISH]
    report = bakeoffkit.zero_shot_report(refs, hyps)
    assert report["writes_nepali"] == pytest.approx(0.9)
    assert report["passes_gate"] is True


def test_the_gate_fails_below_ninety_percent() -> None:
    report = bakeoffkit.zero_shot_report([NEPALI] * 10, [NEPALI] * 8 + [ENGLISH, ""])
    assert report["writes_nepali"] == pytest.approx(0.8)
    assert report["passes_gate"] is False


def test_a_romanised_answer_fails_the_clip() -> None:
    report = bakeoffkit.zero_shot_report([NEPALI], ["ma aaja bajaar gae"])
    assert report["writes_nepali"] == 0.0


def test_an_answer_in_another_script_fails_and_is_counted() -> None:
    report = bakeoffkit.zero_shot_report([NEPALI, NEPALI], ["আমি আজ বাজারে গিয়েছিলাম", NEPALI])
    assert report["writes_nepali"] == pytest.approx(0.5)
    assert report["other_script_clips"] == 1


def test_a_set_with_no_nepali_clip_cannot_pass() -> None:
    report = bakeoffkit.zero_shot_report([ENGLISH], [ENGLISH])
    assert report["nepali_clips"] == 0
    assert report["writes_nepali"] == 0.0 and report["passes_gate"] is False


def test_empty_and_runaway_answers_are_counted() -> None:
    refs = [NEPALI] * 3
    runaway = " ".join([NEPALI] * 10)  # 40 words against 4
    report = bakeoffkit.zero_shot_report(refs, ["", runaway, NEPALI])
    assert report["empty"] == 1
    assert report["runaway"] == 1


def test_a_runaway_is_more_than_twice_the_reference_plus_ten_words() -> None:
    ref = " ".join(["क"] * 5)
    assert not bakeoffkit.is_runaway(" ".join(["क"] * 20), ref)
    assert bakeoffkit.is_runaway(" ".join(["क"] * 21), ref)


def test_latin_shares_are_letter_weighted_over_the_set() -> None:
    report = bakeoffkit.zero_shot_report([MIXED, NEPALI], [MIXED, MIXED])
    ref = [bakeoffkit.script_letters(t) for t in (MIXED, NEPALI)]
    hyp = [bakeoffkit.script_letters(MIXED)] * 2
    assert report["latin_ref"] == pytest.approx(
        sum(c["latin"] for c in ref) / sum(sum(c.values()) for c in ref)
    )
    assert report["latin_hyp"] == pytest.approx(
        sum(c["latin"] for c in hyp) / sum(sum(c.values()) for c in hyp)
    )


def test_refs_and_hyps_must_pair_up() -> None:
    with pytest.raises(ValueError):
        bakeoffkit.zero_shot_report([NEPALI], [])


# --- each candidate's prompt ---------------------------------------------------------------------


def test_the_prompt_is_the_lowest_wer_among_those_that_pass_the_gate() -> None:
    reports = {
        "a": {"passes_gate": True, "writes_nepali": 0.95, "wer": 80.0},
        "b": {"passes_gate": True, "writes_nepali": 0.91, "wer": 60.0},
        "c": {"passes_gate": False, "writes_nepali": 0.50, "wer": 40.0},
    }
    assert bakeoffkit.pick_prompt(reports) == "b"


def test_when_no_prompt_passes_the_one_closest_to_the_gate_is_kept() -> None:
    reports = {
        "a": {"passes_gate": False, "writes_nepali": 0.70, "wer": 90.0},
        "b": {"passes_gate": False, "writes_nepali": 0.85, "wer": 95.0},
    }
    assert bakeoffkit.pick_prompt(reports) == "b"


def test_a_tie_keeps_the_prompt_named_first() -> None:
    reports = {
        "first": {"passes_gate": True, "writes_nepali": 0.95, "wer": 50.0},
        "second": {"passes_gate": True, "writes_nepali": 0.99, "wer": 50.0},
    }
    assert bakeoffkit.pick_prompt(reports) == "first"


# --- what training would cost --------------------------------------------------------------------


def test_the_projection_is_every_epoch_of_both_stages_and_their_val_passes() -> None:
    p = bakeoffkit.projection(
        train_x_realtime=100.0,
        val_x_realtime=50.0,
        human_h=50.0,
        human_epochs=8,
        distill_h=130.0,
        distill_epochs=6,
        val_h=6.0,
    )
    assert p["stage1_h"] == pytest.approx(8 * (50 / 100 + 6 / 50))
    assert p["stage2_h"] == pytest.approx(6 * (130 / 100 + 6 / 50))
    assert p["total_h"] == pytest.approx(p["stage1_h"] + p["stage2_h"])


def test_the_projection_refuses_a_speed_that_was_not_measured() -> None:
    with pytest.raises(ValueError):
        bakeoffkit.projection(
            train_x_realtime=0.0,
            val_x_realtime=50.0,
            human_h=50.0,
            human_epochs=8,
            distill_h=130.0,
            distill_epochs=6,
            val_h=6.0,
        )


# --- the verdict ---------------------------------------------------------------------------------


def test_a_candidate_continues_when_it_passes_the_gate_within_budget() -> None:
    v = bakeoffkit.verdict({"passes_gate": True}, {"total_h": 20.0}, budget_h=24.0)
    assert v == {"continue": True, "reasons": []}


def test_a_candidate_stops_on_the_gate_or_the_budget_and_says_which() -> None:
    v = bakeoffkit.verdict({"passes_gate": False}, {"total_h": 30.0}, budget_h=24.0)
    assert v["continue"] is False
    assert [r.split(":")[0] for r in v["reasons"]] == ["gate", "budget"]


def test_a_candidate_with_no_training_measurement_cannot_continue() -> None:
    v = bakeoffkit.verdict({"passes_gate": True}, None, budget_h=24.0)
    assert v["continue"] is False and v["reasons"][0].startswith("budget")


def test_the_micro_batch_is_extrapolated_from_two_measured_peaks() -> None:
    """The probe never runs a batch it expects to run out of memory: OOMs in a row during the
    2026-10-07 smoke run left ~46 GiB allocated that nothing in Python held (D121)."""
    gib = 2**30
    # 30 GiB with one clip, 32 with two: 2 GiB a clip, so 30 + 2 (n - 1) <= 60 -> n = 16
    assert bakeoffkit.fit_items(30 * gib, 32 * gib, 60 * gib) == 16
    assert bakeoffkit.fit_items(30 * gib, 32 * gib, 61.9 * gib) == 16  # whole clips only


def test_the_extrapolated_micro_batch_has_a_floor_and_a_ceiling() -> None:
    gib = 2**30
    assert bakeoffkit.fit_items(70 * gib, 72 * gib, 60 * gib) == 0  # not even one clip
    # a second clip that measured no growth
    assert bakeoffkit.fit_items(30 * gib, 30 * gib, 60 * gib, ceiling=256) == 256
    assert bakeoffkit.fit_items(30 * gib, 30.001 * gib, 60 * gib, ceiling=64) == 64


def test_a_failed_training_step_is_timed_again_on_a_rerun() -> None:
    """A crash (an OOM in the 2026-10-07 smoke run) is not a measurement: the record of it must not
    make a rerun skip the candidate, or its verdict stays "stop" for good."""
    assert bakeoffkit.needs_timing(None, rescore=False)
    assert bakeoffkit.needs_timing({"error": "OutOfMemoryError: CUDA out of memory"}, rescore=False)
    assert not bakeoffkit.needs_timing({"train_x_realtime": 120.0}, rescore=False)
    assert bakeoffkit.needs_timing({"train_x_realtime": 120.0}, rescore=True)


# --- generation length ---------------------------------------------------------------------------


def test_the_token_cap_scales_with_the_clip_and_has_a_ceiling() -> None:
    assert bakeoffkit.token_cap(10.0, 4.0) == 68  # ceil(1.5 * 10 * 4) + 8
    assert bakeoffkit.token_cap(10.0, 400.0, ceiling=600) == 600


# --- the breadth-first rounds --------------------------------------------------------------------


def _p(passes: bool, wer: float = 50.0, share: float = 0.95) -> dict:
    return {"passes_gate": passes, "wer": wer, "writes_nepali": share}


def test_the_shortlist_is_every_candidate_whose_kept_prompt_passes_in_sweep_order() -> None:
    sweeps = {
        "b": {"x": _p(False, share=0.5), "y": _p(True)},
        "a": {"x": _p(False, share=0.8)},
        "c": {"x": _p(True)},
    }
    assert bakeoffkit.shortlist(sweeps) == ["b", "c"]


def test_a_candidate_not_yet_swept_is_not_shortlisted() -> None:
    assert bakeoffkit.shortlist({"a": {}, "b": {"x": _p(True)}}) == ["b"]
