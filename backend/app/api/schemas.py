"""Request and response models for the review API."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class HypothesisOut(BaseModel):
    """One imported ASR hypothesis."""

    id: int
    system_id: str
    model_id: str | None = None
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    word_count: int = 0


class ScoresOut(BaseModel):
    """Imported agreement scores and the segment's flags."""

    cer_between_hypotheses: float | None = None
    word_disagreement_rate: float | None = None
    script_conflict_rate: float | None = None
    code_switch_density: float | None = None
    flags: list[str] = Field(default_factory=list)


class LabelOut(BaseModel):
    """A recorded human decision."""

    id: int
    disposition: str
    final_text: str | None = None
    annotator: str
    label_version: str
    policy_version: str
    notes: str | None = None
    created_at: dt.datetime


class SegmentOut(BaseModel):
    """A segment with everything the editor needs."""

    id: int
    external_id: str
    episode_id: int
    episode_external_id: str
    split: str
    speaker_id: str | None = None
    start_time: float
    end_time: float
    duration_seconds: float
    p_en: float | None = None
    lid: str | None = None
    pipeline_status: str
    audio_url: str
    peaks_url: str
    hypotheses: list[HypothesisOut] = Field(default_factory=list)
    scores: ScoresOut | None = None
    latest_label: LabelOut | None = None


class QueueRowOut(BaseModel):
    """One row of the triage list: dense on purpose, ~15 fit on a screen."""

    task_id: int
    segment_id: int
    segment_external_id: str
    episode_external_id: str
    queue: str
    status: str
    priority_score: float
    reason: dict[str, Any] | None = None
    flags: list[str] = Field(default_factory=list)
    duration_seconds: float
    seed_hypothesis_id: int | None = None
    seed_system_id: str | None = None
    seed_text: str | None = None
    audio_url: str
    peaks_url: str


class TaskOut(BaseModel):
    """A task with its full segment payload, for the editor."""

    id: int
    segment_id: int
    queue: str
    status: str
    priority_score: float
    reason: dict[str, Any] | None = None
    seed_hypothesis_id: int | None = None
    seed_system_id: str | None = None
    served_at: dt.datetime
    segment: SegmentOut


class DecisionIn(BaseModel):
    """Common body for accept, label and flag.

    ``opened_at`` is when the client received the task. Elapsed human time is the quantity of
    interest, and only the client can observe it -- server processing time is not a proxy for it.
    """

    opened_at: dt.datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    annotator: str | None = None
    label_version: str | None = None
    notes: str | None = None


class AcceptIn(DecisionIn):
    """Accept the seed hypothesis unchanged."""


class LabelIn(DecisionIn):
    """Save corrected text."""

    final_text: str


class FlagIn(DecisionIn):
    """Mark a segment unusable or unresolved.

    ``unusable_audio`` and ``uncertain`` are deliberately distinct: the first is an audio quality
    statistic, the second an annotation difficulty statistic, and they route differently on export.
    """

    disposition: str = Field(pattern="^(unusable_audio|uncertain)$")


class SkipIn(BaseModel):
    """Defer a task without deciding anything."""

    opened_at: dt.datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    annotator: str | None = None


class BulkAcceptIn(DecisionIn):
    """Accept several tasks at once, from the triage list."""

    task_ids: list[int] = Field(min_length=1)


class DecisionOut(BaseModel):
    """What a write produced."""

    task_id: int
    segment_id: int
    label_id: int | None = None
    disposition: str | None = None
    task_status: str
    duration_ms: int | None = None


class BulkAcceptOut(BaseModel):
    """Result of a bulk accept. All or nothing: one transaction."""

    accepted: list[DecisionOut]
    count: int


class TranslitIn(BaseModel):
    """A Latin token to transliterate."""

    token: str
    limit: int | None = Field(default=None, ge=1, le=10)


class TranslitOut(BaseModel):
    """Ranked Devanagari candidates."""

    token: str
    candidates: list[str]


class TranslitChoiceIn(BaseModel):
    """Record which candidate the annotator picked, for the correction memory."""

    token: str
    devanagari: str
