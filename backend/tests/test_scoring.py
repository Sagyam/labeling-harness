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
        "word_disagreement_rate": 0.0,
        "avg_logprob": 0.0,
        "code_switch_density": 0.0,
        "flags": [],
    }
    return ScoreInputs(**{**base, **kwargs})


def test_all_zero_inputs_score_zero() -> None:
    result = priority_score(inputs(), settings=SETTINGS)
    assert result.score == pytest.approx(0.0)


def test_all_maximum_inputs_score_one() -> None:
    from app.services.flags import ALL_FLAGS

    result = priority_score(
        inputs(
            word_disagreement_rate=1.0,
            avg_logprob=-5.0,
            code_switch_density=1.0,
            flags=list(ALL_FLAGS),
        ),
        settings=SETTINGS,
    )
    assert result.score == pytest.approx(1.0)


def test_score_is_the_weighted_sum_of_its_components() -> None:
    result = priority_score(
        inputs(word_disagreement_rate=0.5, code_switch_density=0.25), settings=SETTINGS
    )
    assert result.score == pytest.approx(0.40 * 0.5 + 0.20 * 0.25)


def test_disagreement_dominates_the_other_signals() -> None:
    """The formula must rank a disagreeing segment above a merely code-switched one."""
    disagreeing = priority_score(inputs(word_disagreement_rate=1.0), settings=SETTINGS)
    switched = priority_score(inputs(code_switch_density=1.0), settings=SETTINGS)
    assert disagreeing.score > switched.score


def test_missing_scores_are_treated_as_zero() -> None:
    result = priority_score(
        ScoreInputs(
            word_disagreement_rate=None, avg_logprob=None, code_switch_density=None, flags=[]
        ),
        settings=SETTINGS,
    )
    assert result.score == pytest.approx(0.0)
    assert result.components["word_disagreement_rate"] == 0.0


def test_out_of_range_inputs_are_clamped() -> None:
    result = priority_score(
        inputs(word_disagreement_rate=5.0, code_switch_density=-2.0), settings=SETTINGS
    )
    assert result.components["word_disagreement_rate"] == 1.0
    assert result.components["code_switch_density"] == 0.0
    assert 0.0 <= result.score <= 1.0


def test_low_confidence_normalization_is_monotonic() -> None:
    values = [normalize_low_confidence(x, floor=-2.0) for x in (0.0, -0.5, -1.0, -2.0, -4.0)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_low_confidence_of_a_missing_logprob_is_zero() -> None:
    assert normalize_low_confidence(None, floor=-2.0) == 0.0


def test_reason_records_every_component_and_weight() -> None:
    result = priority_score(
        inputs(word_disagreement_rate=0.3, avg_logprob=-1.0, flags=["too_short"]),
        settings=SETTINGS,
    )
    reason = result.as_reason()
    assert set(reason["components"]) == {
        "word_disagreement_rate",
        "low_confidence",
        "code_switch_density",
        "rule_flag_score",
    }
    assert reason["weights"]["word_disagreement_rate"] == 0.40
    assert reason["flags"] == ["too_short"]
    assert reason["score"] == pytest.approx(result.score)
    assert sum(reason["contributions"].values()) == pytest.approx(result.score)


def test_reason_explains_why_a_segment_surfaced() -> None:
    """Every priority score must be explainable, or the UI cannot show reason chips."""
    result = priority_score(inputs(word_disagreement_rate=0.9), settings=SETTINGS)
    top = max(result.as_reason()["contributions"].items(), key=lambda kv: kv[1])
    assert top[0] == "word_disagreement_rate"
