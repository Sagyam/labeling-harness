"""Tests for the fused-seed hazard gates and ranking components (D74)."""

from __future__ import annotations

import pytest

from app.config import HazardSettings
from app.services.hazards import FusionEvidence, assess
from app.services.normalize import Ruleset

RULES = Ruleset(version="t", tokens={})
CFG = HazardSettings()


def evidence(fused: str | None, asr: list[str], **kw) -> FusionEvidence:
    return FusionEvidence(fused_text=fused, asr_texts=asr, **kw)


def run(ev: FusionEvidence):
    return assess(ev, config=CFG, ruleset=RULES)


# --- clean cases never fire ------------------------------------------------------------------


def test_a_fused_text_the_recognisers_agree_with_is_clean() -> None:
    report = run(evidence("आज हामी meeting मा छौं", ["आज हामी meeting मा छौं"] * 3))
    assert report.hazards == []
    assert report.unsupported_rate == 0.0
    assert report.dropped_rate == 0.0
    assert report.asr_disagreement == 0.0


def test_a_transliteration_is_not_a_disagreement() -> None:
    """Scribe's टिम and the fuser's team are one word; nothing here is unsupported."""
    report = run(evidence("हाम्रो team राम्रो", ["हाम्रो टिम राम्रो", "हाम्रो team राम्रो"] * 1))
    assert report.hazards == []
    assert report.unsupported_rate == 0.0


def test_choosing_one_recogniser_over_the_others_is_not_invention() -> None:
    report = run(
        evidence(
            "हामी निर्वाचनमा जाँदैनौँ",
            ["हामी निर्वाचनमा जाँदैनौँ", "हामी निर्वासनमा जाँदैनौँ", "हामी निर्वासनमा जाँदैनौँ"],
        )
    )
    assert "invention" not in report.hazards
    assert report.unsupported_rate == 0.0


def test_a_near_miss_spelling_counts_as_heard() -> None:
    report = run(evidence("सबै जम्मै भयो", ["सबै जमै भयो", "सबै जमै भयो"]))
    assert report.unsupported_rate == 0.0


# --- invention --------------------------------------------------------------------------------


def test_a_run_of_words_no_recogniser_heard_is_an_invention() -> None:
    report = run(
        evidence(
            "हामी प्रत्यक्ष निर्वाचित कार्यकारी नभइकन जाँदैनौँ",
            ["हामी जाँदैनौँ", "हामी जाँदैनौँ", "हामी जाँदैनौँ"],
        )
    )
    assert "invention" in report.hazards
    assert report.details["invention"] == "प्रत्यक्ष निर्वाचित कार्यकारी नभइकन"
    assert report.unsupported_rate == pytest.approx(4 / 6, abs=1e-3)


def test_scattered_unsupported_words_rank_but_do_not_gate() -> None:
    report = run(evidence("क ख ग घ ङ च", ["क X ग Y ङ Z"] * 3))
    assert "invention" not in report.hazards
    assert report.unsupported_rate == pytest.approx(3 / 6, abs=1e-3)


# --- dropped content --------------------------------------------------------------------------


def test_words_two_recognisers_agree_on_and_the_fuser_left_out_are_dropped() -> None:
    report = run(
        evidence(
            "त्यो दिन",
            ["त्यो दिन हामी सबै घर गयौं", "त्यो दिन हामी सबै घर गयौं", "त्यो दिन"],
        )
    )
    assert "dropped" in report.hazards
    assert report.details["dropped"] == "हामी सबै घर गयौं"
    assert report.dropped_rate > 0.5


def test_one_recogniser_alone_cannot_make_a_drop() -> None:
    """MAI hallucinating a tail must not make the fuser look like it dropped something."""
    report = run(evidence("त्यो दिन", ["त्यो दिन हामी सबै घर गयौं", "त्यो दिन", "त्यो दिन"]))
    assert "dropped" not in report.hazards


# --- seams ------------------------------------------------------------------------------------


def test_words_from_the_next_clip_are_bleed_not_invention() -> None:
    report = run(
        evidence(
            "म घर गएँ अनि खाना खाएँ",
            ["म घर गएँ", "म घर गएँ", "म घर गएँ"],
            neighbour_texts=["अनि खाना खाएँ", "अनि खाना खाएँ"],
        )
    )
    assert "seam_bleed" in report.hazards
    assert "invention" not in report.hazards


# --- length and silence -----------------------------------------------------------------------


def test_an_empty_fusion_over_heard_speech_is_gated() -> None:
    assert "emptied" in run(evidence("", ["म घर गएँ", "म घर गएँ", "म"])).hazards


def test_text_where_every_recogniser_heard_nothing_is_gated() -> None:
    assert "speech_over_silence" in run(evidence("धन्यवाद सबैलाई", ["", "", ""])).hazards


def test_a_fusion_far_longer_than_every_recogniser_is_gated() -> None:
    fused = "म घर गएँ " + " ".join(["अनि"] * 12)
    assert "length_outlier" in run(evidence(fused, ["म घर गएँ", "म घर गएँ", "म घर गएँ"])).hazards


def test_an_empty_clip_everyone_agrees_is_empty_is_clean() -> None:
    assert run(evidence("", ["", "", ""])).hazards == []


# --- the fuser's own word, the waveform's -----------------------------------------------------


def test_the_fuser_saying_uncertain_is_a_gate() -> None:
    report = run(evidence("म घर गएँ", ["म घर गएँ"] * 3, fused_code="u"))
    assert report.hazards == ["fuser_uncertain"]


def test_a_clip_the_fuser_never_answered_is_gated() -> None:
    assert run(evidence(None, ["म घर गएँ"] * 3)).hazards == ["unfused"]


def test_text_the_aligner_cannot_fit_into_the_clip_is_gated() -> None:
    report = run(evidence("म घर गएँ", ["म घर गएँ"] * 3, acoustic={"aligned": False}))
    assert "unaligned" in report.hazards


def test_the_acoustic_gap_ranks_on_a_zero_to_one_scale() -> None:
    fits = [{"aligned": True, "gap": g} for g in (0.0, CFG.acoustic_gap_full_scale / 2, 99.0)]
    gaps = [run(evidence("म घर गएँ", ["म घर गएँ"] * 3, acoustic=f)).acoustic_gap for f in fits]
    assert gaps == [0.0, pytest.approx(0.5), 1.0]


def test_no_acoustic_measurement_is_unmeasured_not_perfect() -> None:
    assert run(evidence("म घर गएँ", ["म घर गएँ"] * 3)).acoustic_gap is None


# --- the recognisers among themselves ---------------------------------------------------------


def test_recogniser_disagreement_is_script_folded() -> None:
    same = run(evidence("हाम्रो team", ["हाम्रो टिम", "हाम्रो team", "हाम्रो team"]))
    different = run(evidence("हाम्रो team", ["हाम्रो घर", "तिम्रो team", "हाम्रो टिम"]))
    assert same.asr_disagreement == 0.0
    assert different.asr_disagreement > 0.2
