"""Tests for priority scoring."""

from __future__ import annotations

import pytest

from app.config import load_settings
from app.services.scoring import (
    ScoreInputs,
    normalize_low_confidence,
    priority_score,
)

SETTINGS = load_settings()


def inputs(**kwargs) -> ScoreInputs:
    base = {
        "seed_outvoted": 0.0,
        "avg_logprob": 0.0,
        "flags": [],
    }
    return ScoreInputs(**{**base, **kwargs})


def test_all_zero_inputs_score_zero() -> None:
    result = priority_score(inputs(), settings=SETTINGS)
    assert result.score == pytest.approx(0.0)


def test_all_maximum_inputs_score_one() -> None:
    from app.services.flags import ALL_FLAGS

    result = priority_score(
        inputs(seed_outvoted=1.0, avg_logprob=-5.0, flags=list(ALL_FLAGS)),
        settings=SETTINGS,
    )
    assert result.score == pytest.approx(1.0)


def test_score_is_the_weighted_sum_of_its_components() -> None:
    weights = SETTINGS.queue.weights
    result = priority_score(inputs(seed_outvoted=0.5), settings=SETTINGS)
    assert result.score == pytest.approx(weights.seed_outvoted * 0.5)


def test_disagreement_dominates_the_other_signals() -> None:
    """A segment whose seed is contradicted must outrank a merely unconfident one."""
    outvoted = priority_score(inputs(seed_outvoted=1.0), settings=SETTINGS)
    unsure = priority_score(inputs(avg_logprob=-5.0), settings=SETTINGS)
    assert outvoted.score > unsure.score


def test_every_component_can_reach_its_full_weight() -> None:
    """No term may be capped below its nominal weight.

    The formula this replaced advertised 0.20 for a code-mixing term that was structurally
    incapable of exceeding 0.5, so it never contributed more than half of what it claimed.
    """
    from app.services.flags import ALL_FLAGS

    weights = SETTINGS.queue.weights
    assert priority_score(inputs(seed_outvoted=1.0), settings=SETTINGS).score == pytest.approx(
        weights.seed_outvoted
    )
    assert priority_score(inputs(avg_logprob=-99.0), settings=SETTINGS).score == pytest.approx(
        weights.low_confidence
    )
    assert priority_score(inputs(flags=list(ALL_FLAGS)), settings=SETTINGS).score == pytest.approx(
        weights.rule_flag_score
    )


def test_missing_inputs_are_zero_not_one() -> None:
    """An absent signal must never push a segment up the queue on its own."""
    result = priority_score(
        ScoreInputs(seed_outvoted=None, avg_logprob=None, flags=[]), settings=SETTINGS
    )
    assert result.score == pytest.approx(0.0)


def test_out_of_range_inputs_are_clamped() -> None:
    result = priority_score(inputs(seed_outvoted=5.0), settings=SETTINGS)
    assert result.components["seed_outvoted"] == 1.0
    assert result.score <= 1.0


def test_the_reason_payload_explains_the_score() -> None:
    result = priority_score(inputs(seed_outvoted=0.5, avg_logprob=-0.25), settings=SETTINGS)
    reason = result.as_reason()
    assert set(reason["components"]) == set(reason["weights"]) == set(reason["contributions"])
    assert sum(reason["contributions"].values()) == pytest.approx(reason["score"])


def test_the_legacy_score_travels_alongside_for_comparison() -> None:
    """The replacement is backed by 22 labels from one episode, so both are recorded (D54)."""
    result = priority_score(
        inputs(seed_outvoted=0.5),
        settings=SETTINGS,
        legacy=ScoreInputs.legacy(word_disagreement_rate=0.4, code_switch_density=0.3),
    )
    reason = result.as_reason()
    assert "legacy" in reason
    assert reason["legacy"]["components"]["word_disagreement_rate"] == pytest.approx(0.4)
    assert reason["legacy"]["components"]["code_switch_density"] == pytest.approx(0.3)
    # The legacy number is recorded, never blended into the live score.
    assert reason["score"] == pytest.approx(SETTINGS.queue.weights.seed_outvoted * 0.5)


# --- low confidence ----------------------------------------------------------------------


def test_no_logprob_is_no_signal() -> None:
    assert normalize_low_confidence(None, floor=-0.5) == 0.0


def test_a_confident_hypothesis_scores_zero() -> None:
    assert normalize_low_confidence(0.0, floor=-0.5) == 0.0


def test_the_floor_and_anything_below_it_saturate() -> None:
    assert normalize_low_confidence(-0.5, floor=-0.5) == 1.0
    assert normalize_low_confidence(-9.0, floor=-0.5) == 1.0


def test_the_floor_spans_the_observed_range_of_the_only_system_reporting_one() -> None:
    """Scribe's avg_logprob runs about -0.68 to -0.004; a -2.0 floor used a third of 0-1."""
    floor = SETTINGS.queue.logprob_floor
    assert normalize_low_confidence(-0.11, floor=floor) > 0.15
    assert normalize_low_confidence(-0.68, floor=floor) == 1.0
