"""Tests for the fused-seed priority score (D74)."""

from __future__ import annotations

import pytest

from app.config import load_settings
from app.services.flags import ALL_FLAGS
from app.services.scoring import ScoreInputs, normalize_low_confidence, priority_score

SETTINGS = load_settings()
COMPONENTS = {
    "unsupported_rate",
    "dropped_rate",
    "asr_disagreement",
    "acoustic_gap",
    "low_confidence",
    "rule_flag_score",
}


def inputs(**overrides) -> ScoreInputs:
    base = {
        "unsupported_rate": 0.0,
        "dropped_rate": 0.0,
        "asr_disagreement": 0.0,
        "acoustic_gap": 0.0,
        "avg_logprob": 0.0,
        "flags": [],
        "hazards": [],
    }
    return ScoreInputs(**{**base, **overrides})


def test_all_zero_inputs_score_zero() -> None:
    result = priority_score(inputs(), settings=SETTINGS)
    assert result.score == 0.0
    assert result.priority == 0.0


def test_all_maximum_inputs_score_one() -> None:
    result = priority_score(
        inputs(
            unsupported_rate=1.0,
            dropped_rate=1.0,
            asr_disagreement=1.0,
            acoustic_gap=1.0,
            avg_logprob=-5.0,
            flags=list(ALL_FLAGS),
        ),
        settings=SETTINGS,
    )
    assert result.score == pytest.approx(1.0)


def test_the_components_are_the_fused_seed_terms() -> None:
    assert set(priority_score(inputs(), settings=SETTINGS).components) == COMPONENTS


def test_score_is_the_weighted_sum_of_its_components() -> None:
    result = priority_score(inputs(unsupported_rate=0.5, dropped_rate=0.25), settings=SETTINGS)
    weights = SETTINGS.queue.weights
    assert result.score == pytest.approx(
        0.5 * weights.unsupported_rate + 0.25 * weights.dropped_rate
    )


def test_a_gate_puts_a_clip_above_every_clip_without_one() -> None:
    """Priority >= 1 means a hazard gate fired; the worst ungated clip still sorts below it."""
    gated = priority_score(inputs(hazards=["invention"]), settings=SETTINGS)
    worst_ungated = priority_score(
        inputs(unsupported_rate=1.0, dropped_rate=1.0, asr_disagreement=1.0, acoustic_gap=1.0),
        settings=SETTINGS,
    )
    assert gated.priority >= 1.0
    assert gated.priority > worst_ungated.priority


def test_missing_inputs_are_zero_not_one() -> None:
    result = priority_score(
        ScoreInputs(
            unsupported_rate=None,
            dropped_rate=None,
            asr_disagreement=None,
            acoustic_gap=None,
            avg_logprob=None,
        ),
        settings=SETTINGS,
    )
    assert result.score == 0.0


def test_out_of_range_inputs_are_clamped() -> None:
    result = priority_score(inputs(unsupported_rate=7.0), settings=SETTINGS)
    assert result.components["unsupported_rate"] == 1.0


def test_the_reason_payload_explains_the_score_and_names_the_gates() -> None:
    result = priority_score(
        inputs(unsupported_rate=0.5, hazards=["invention"]),
        settings=SETTINGS,
        hazard_details={"invention": "प्रत्यक्ष निर्वाचित"},
    )
    reason = result.as_reason()
    assert reason["hazards"] == ["invention"]
    assert reason["hazard_details"] == {"invention": "प्रत्यक्ष निर्वाचित"}
    assert set(reason["components"]) == COMPONENTS
    assert reason["contributions"]["unsupported_rate"] == pytest.approx(
        0.5 * SETTINGS.queue.weights.unsupported_rate
    )
    assert reason["score"] == pytest.approx(result.score)


def test_an_unmeasured_acoustic_gap_is_marked_as_such() -> None:
    reason = priority_score(inputs(acoustic_gap=None), settings=SETTINGS).as_reason()
    assert "acoustic_gap" in reason["unmeasured"]


def test_the_legacy_score_travels_alongside_for_comparison() -> None:
    result = priority_score(
        inputs(),
        settings=SETTINGS,
        legacy=ScoreInputs.legacy(
            seed_outvoted=1.0, seed_orphan_rate=0.5, roman_gap=0.0, avg_logprob=0.0, flags=[]
        ),
    )
    legacy = result.as_reason()["legacy"]
    assert set(legacy["components"]) == {
        "seed_outvoted",
        "seed_orphan_rate",
        "roman_gap",
        "low_confidence",
        "rule_flag_score",
    }
    weights = SETTINGS.queue.legacy_weights
    assert legacy["score"] == pytest.approx(weights.seed_outvoted + 0.5 * weights.seed_orphan_rate)
    assert result.score == 0.0


# --- low confidence --------------------------------------------------------------------------


def test_no_logprob_is_no_signal() -> None:
    assert normalize_low_confidence(None, floor=-0.5) == 0.0


def test_a_confident_hypothesis_scores_zero() -> None:
    assert normalize_low_confidence(0.0, floor=-0.5) == 0.0


def test_the_floor_and_anything_below_it_saturate() -> None:
    assert normalize_low_confidence(-0.5, floor=-0.5) == 1.0
    assert normalize_low_confidence(-3.0, floor=-0.5) == 1.0
