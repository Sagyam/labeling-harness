"""Priority scoring for a queue seeded with fused text (D74).

    priority_score = 0.30 * unsupported_rate  (fused words no recogniser heard, by sound)
                   + 0.20 * dropped_rate      (words two recognisers heard that fusion lacks)
                   + 0.20 * asr_disagreement  (script-folded disagreement among the recognisers)
                   + 0.15 * acoustic_gap      (fused text against the aligner's own reading)
                   + 0.10 * low_confidence    (Scribe avg_logprob)
                   + 0.05 * rule_flag_score

    priority = priority_score + 1   when any hazard gate fired (``hazards.py``)

Every component is in 0-1 and the weights sum to 1, so the score is in 0-1, and a gated clip's
priority is in 1-2: it sorts above every clip without a gate, and screening refuses it.

The question is no longer "how far is the seed from the recognisers". The seed was *built* from
them, so it would score near zero on that by construction and read as quality. The question is
"where does the seed say something the evidence does not support, or omit something it does" --
and "how hard is this audio", which the recognisers' disagreement among themselves still answers
honestly, because none of them sees another's output.

The weights are provisional; nothing verified has been scored against them yet. The D67 formula
travels alongside under ``legacy``, measured against the recogniser the old queue would have
seeded with, so the first labelled run can compare what each would have surfaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.services.flags import rule_flag_score

#: Added to a gated clip's score so it sorts above every clip without a gate.
GATE_OFFSET = 1.0


@dataclass(frozen=True)
class LegacyInputs:
    """D67's inputs, measured against the recogniser the old queue would have chosen."""

    seed_outvoted: float | None = None
    seed_orphan_rate: float | None = None
    roman_gap: float | None = None
    avg_logprob: float | None = None
    flags: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreInputs:
    """Everything the formula reads. A missing value means zero and is reported as unmeasured."""

    unsupported_rate: float | None
    dropped_rate: float | None
    asr_disagreement: float | None
    acoustic_gap: float | None
    avg_logprob: float | None
    flags: Sequence[str] = field(default_factory=list)
    hazards: Sequence[str] = field(default_factory=list)

    @staticmethod
    def legacy(
        *,
        seed_outvoted: float | None = None,
        seed_orphan_rate: float | None = None,
        roman_gap: float | None = None,
        avg_logprob: float | None = None,
        flags: Sequence[str] = (),
    ) -> LegacyInputs:
        """Build the comparison payload for :func:`priority_score`."""
        return LegacyInputs(
            seed_outvoted=seed_outvoted,
            seed_orphan_rate=seed_orphan_rate,
            roman_gap=roman_gap,
            avg_logprob=avg_logprob,
            flags=list(flags),
        )


@dataclass(frozen=True)
class ScoreResult:
    """A priority score with the breakdown that explains it."""

    score: float
    components: dict[str, float]
    weights: dict[str, float]
    flags: list[str]
    hazards: list[str] = field(default_factory=list)
    hazard_details: dict[str, str] = field(default_factory=dict)
    unmeasured: list[str] = field(default_factory=list)
    legacy: dict[str, Any] | None = None

    @property
    def priority(self) -> float:
        """What the queue sorts on: the score, lifted above every ungated clip by a gate."""
        return self.score + (GATE_OFFSET if self.hazards else 0.0)

    def as_reason(self) -> dict[str, Any]:
        """The payload stored in ``annotation_tasks.reason_jsonb``."""
        reason: dict[str, Any] = {
            "score": round(self.score, 9),
            "components": {k: round(v, 9) for k, v in self.components.items()},
            "weights": dict(self.weights),
            "contributions": {k: round(v * self.weights[k], 9) for k, v in self.components.items()},
            "flags": list(self.flags),
            "hazards": list(self.hazards),
            "hazard_details": dict(self.hazard_details),
            "unmeasured": list(self.unmeasured),
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
    """D67's score, recorded but never used to rank."""
    weights = settings.queue.legacy_weights
    components = {
        "seed_outvoted": _clamp(inputs.seed_outvoted),
        "seed_orphan_rate": _clamp(inputs.seed_orphan_rate),
        "roman_gap": _clamp(inputs.roman_gap),
        "low_confidence": normalize_low_confidence(
            inputs.avg_logprob, floor=settings.queue.logprob_floor
        ),
        "rule_flag_score": rule_flag_score(inputs.flags),
    }
    weight_map = {name: float(getattr(weights, name)) for name in components}
    return {
        "score": round(sum(v * weight_map[k] for k, v in components.items()), 9),
        "components": {k: round(v, 9) for k, v in components.items()},
        "weights": weight_map,
    }


def priority_score(
    inputs: ScoreInputs,
    *,
    settings: Settings | None = None,
    hazard_details: Mapping[str, str] | None = None,
    legacy: LegacyInputs | None = None,
) -> ScoreResult:
    """Score one clip for the review queue.

    Args:
        inputs: The ``hazards.py`` components, Scribe's confidence, the rule flags and any gates
            that fired.
        settings: Weight and threshold overrides.
        hazard_details: ``{gate: words}`` for the tooltip.
        legacy: D67's inputs; recorded beside the score, never ranking.

    Returns:
        The score in 0-1, its breakdown, and the gates. Sort on :attr:`ScoreResult.priority`.
    """
    settings = settings or get_settings()
    weights = settings.queue.weights

    components = {
        "unsupported_rate": _clamp(inputs.unsupported_rate),
        "dropped_rate": _clamp(inputs.dropped_rate),
        "asr_disagreement": _clamp(inputs.asr_disagreement),
        "acoustic_gap": _clamp(inputs.acoustic_gap),
        "low_confidence": normalize_low_confidence(
            inputs.avg_logprob, floor=settings.queue.logprob_floor
        ),
        "rule_flag_score": rule_flag_score(inputs.flags),
    }
    weight_map = {name: float(getattr(weights, name)) for name in components}
    unmeasured = [
        name
        for name, value in (
            ("unsupported_rate", inputs.unsupported_rate),
            ("dropped_rate", inputs.dropped_rate),
            ("asr_disagreement", inputs.asr_disagreement),
            ("acoustic_gap", inputs.acoustic_gap),
            ("low_confidence", inputs.avg_logprob),
        )
        if value is None
    ]
    return ScoreResult(
        score=sum(value * weight_map[name] for name, value in components.items()),
        components=components,
        weights=weight_map,
        flags=list(inputs.flags),
        hazards=list(inputs.hazards),
        hazard_details=dict(hazard_details or {}),
        unmeasured=unmeasured,
        legacy=_legacy_reason(legacy, settings) if legacy is not None else None,
    )
