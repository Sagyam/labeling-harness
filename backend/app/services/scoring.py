"""Priority scoring for the annotation queue.

    priority_score = 0.40 * word_disagreement_rate
                   + 0.25 * low_confidence          (normalized from avg_logprob)
                   + 0.20 * code_switch_density
                   + 0.15 * rule_flag_score

Every input is normalized to 0-1 and the weights sum to 1, so the score is itself in 0-1. No LLM is
involved: multi-system disagreement and rule flags already provide the prioritization signal.

The per-component breakdown travels with the score into ``annotation_tasks.reason_jsonb``, so the
UI can always answer "why is this segment near the top?".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.services.flags import rule_flag_score


@dataclass(frozen=True)
class ScoreInputs:
    """Everything the formula reads. Any of it may be missing, and missing means zero."""

    word_disagreement_rate: float | None
    avg_logprob: float | None
    code_switch_density: float | None
    flags: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreResult:
    """A priority score with the breakdown that explains it."""

    score: float
    components: dict[str, float]
    weights: dict[str, float]
    flags: list[str]

    def as_reason(self) -> dict[str, Any]:
        """The payload stored in ``annotation_tasks.reason_jsonb``."""
        return {
            "score": round(self.score, 9),
            "components": {k: round(v, 9) for k, v in self.components.items()},
            "weights": dict(self.weights),
            "contributions": {k: round(v * self.weights[k], 9) for k, v in self.components.items()},
            "flags": list(self.flags),
        }


def _clamp(value: float | None, low: float = 0.0, high: float = 1.0) -> float:
    if value is None:
        return low
    return max(low, min(high, float(value)))


def normalize_low_confidence(avg_logprob: float | None, *, floor: float) -> float:
    """Map an average log probability onto 0-1, where 1 is least confident.

    ``avg_logprob`` is at most 0 and unbounded below. ``floor`` (a negative number) is the point at
    which confidence is treated as fully exhausted: 0 maps to 0, ``floor`` and anything below it
    maps to 1.
    """
    if avg_logprob is None:
        return 0.0
    return _clamp(float(avg_logprob) / floor)


def priority_score(inputs: ScoreInputs, *, settings: Settings | None = None) -> ScoreResult:
    """Score one segment for the review queue.

    Args:
        inputs: Imported scores plus the segment's rule flags.
        settings: Weight and threshold overrides.

    Returns:
        The score in 0-1 and its per-component breakdown.
    """
    settings = settings or get_settings()
    weights = settings.queue.weights

    components = {
        "word_disagreement_rate": _clamp(inputs.word_disagreement_rate),
        "low_confidence": normalize_low_confidence(
            inputs.avg_logprob, floor=settings.queue.logprob_floor
        ),
        "code_switch_density": _clamp(inputs.code_switch_density),
        "rule_flag_score": rule_flag_score(inputs.flags),
    }
    weight_map = {
        "word_disagreement_rate": weights.word_disagreement_rate,
        "low_confidence": weights.low_confidence,
        "code_switch_density": weights.code_switch_density,
        "rule_flag_score": weights.rule_flag_score,
    }
    score = sum(value * weight_map[name] for name, value in components.items())
    return ScoreResult(
        score=score,
        components=components,
        weights=weight_map,
        flags=list(inputs.flags),
    )
