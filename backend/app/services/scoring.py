"""Priority scoring for the annotation queue.

    priority_score = 0.60 * seed_outvoted     (share of speech time the other systems contradict)
                   + 0.25 * low_confidence    (normalized from avg_logprob)
                   + 0.15 * rule_flag_score

Every input is normalized to 0-1 and the weights sum to 1, so the score is itself in 0-1. No LLM
is involved: time-aligned disagreement between the recognisers and the rule flags already provide
the prioritization signal.

The question the score answers is "how much of what this annotator is about to be shown is
probably wrong", so it is measured against the **seed** -- the hypothesis they will actually edit
-- rather than symmetrically across systems. See D54 for the measurement behind the weights, and
for why the two terms this replaced were dropped.

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
class LegacyInputs:
    """The two terms D54 removed, kept only so the old score can be recorded beside the new one."""

    word_disagreement_rate: float | None = None
    code_switch_density: float | None = None


@dataclass(frozen=True)
class ScoreInputs:
    """Everything the formula reads. Any of it may be missing, and missing means zero."""

    seed_outvoted: float | None
    avg_logprob: float | None
    flags: Sequence[str] = field(default_factory=list)

    @staticmethod
    def legacy(
        *, word_disagreement_rate: float | None = None, code_switch_density: float | None = None
    ) -> LegacyInputs:
        """Build the comparison payload for :func:`priority_score`."""
        return LegacyInputs(
            word_disagreement_rate=word_disagreement_rate,
            code_switch_density=code_switch_density,
        )


@dataclass(frozen=True)
class ScoreResult:
    """A priority score with the breakdown that explains it."""

    score: float
    components: dict[str, float]
    weights: dict[str, float]
    flags: list[str]
    legacy: dict[str, Any] | None = None

    def as_reason(self) -> dict[str, Any]:
        """The payload stored in ``annotation_tasks.reason_jsonb``."""
        reason: dict[str, Any] = {
            "score": round(self.score, 9),
            "components": {k: round(v, 9) for k, v in self.components.items()},
            "weights": dict(self.weights),
            "contributions": {k: round(v * self.weights[k], 9) for k, v in self.components.items()},
            "flags": list(self.flags),
        }
        if self.legacy is not None:
            reason["legacy"] = self.legacy
        return reason


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


def _legacy_reason(inputs: LegacyInputs, settings: Settings) -> dict[str, Any]:
    """The superseded formula's score, recorded but never used to rank.

    The replacement rests on 22 labels from a single episode. Writing the old number beside the
    new one costs nothing and lets the first full run decide between them on its own evidence
    rather than on that pilot.
    """
    weights = settings.queue.legacy_weights
    components = {
        "word_disagreement_rate": _clamp(inputs.word_disagreement_rate),
        "code_switch_density": _clamp(inputs.code_switch_density),
    }
    weight_map = {
        "word_disagreement_rate": weights.word_disagreement_rate,
        "code_switch_density": weights.code_switch_density,
    }
    return {
        "score": round(sum(v * weight_map[k] for k, v in components.items()), 9),
        "components": {k: round(v, 9) for k, v in components.items()},
        "weights": dict(weight_map),
    }


def priority_score(
    inputs: ScoreInputs,
    *,
    settings: Settings | None = None,
    legacy: LegacyInputs | None = None,
) -> ScoreResult:
    """Score one segment for the review queue.

    Args:
        inputs: Time-aligned disagreement against the seed, the seed's confidence, and the
            segment's rule flags.
        settings: Weight and threshold overrides.
        legacy: The superseded formula's inputs. When given, its score is recorded alongside for
            comparison; it never contributes to the ranking.

    Returns:
        The score in 0-1 and its per-component breakdown.
    """
    settings = settings or get_settings()
    weights = settings.queue.weights

    components = {
        "seed_outvoted": _clamp(inputs.seed_outvoted),
        "low_confidence": normalize_low_confidence(
            inputs.avg_logprob, floor=settings.queue.logprob_floor
        ),
        "rule_flag_score": rule_flag_score(inputs.flags),
    }
    weight_map = {
        "seed_outvoted": weights.seed_outvoted,
        "low_confidence": weights.low_confidence,
        "rule_flag_score": weights.rule_flag_score,
    }
    score = sum(value * weight_map[name] for name, value in components.items())
    return ScoreResult(
        score=score,
        components=components,
        weights=weight_map,
        flags=list(inputs.flags),
        legacy=_legacy_reason(legacy, settings) if legacy is not None else None,
    )
