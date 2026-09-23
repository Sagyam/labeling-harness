"""Request and response models for the review API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


class HypothesisWordOut(BaseModel):
    """One word-level token within an ASR hypothesis."""

    position: int
    word: str
    start_time: float | None = None
    end_time: float | None = None
    confidence: float | None = None
    predicted_language: str | None = None
    predicted_script: str | None = None


class HypothesisOut(BaseModel):
    """One imported ASR hypothesis."""

    id: int
    system_id: str
    model_id: str | None = None
    #: ``asr`` for a recogniser, ``fusion`` for the reconciled transcript (D72).
    kind: str = "asr"
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    word_count: int = 0
    words: list[HypothesisWordOut] = Field(default_factory=list)


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
    verification_tier: str = "verified"
    final_text: str | None = None
    annotator: str
    label_version: str
    policy_version: str
    notes: str | None = None
    created_at: dt.datetime


class SpeakerTurnOut(BaseModel):
    """One speaker's stretch of a clip, clip-relative. ``speaker`` is a display number, stable
    across the episode (1 = most talk time); turns of different speakers may overlap (D78)."""

    speaker: int
    start: float
    end: float


class SegmentOut(BaseModel):
    """A segment with everything the editor needs."""

    id: int
    external_id: str
    episode_id: int
    episode_external_id: str
    #: The episode's YouTube video, rebuilt from its id; null for an uploaded file. Clip times
    #: are episode-relative, so ``start_time`` is the moment in the video.
    video_url: str | None = None
    split: str
    #: ``gold`` or ``train``, for this clip (D71). The editor reads this to decide whether
    #: screening is even offered: a gold clip has to be listened to.
    pot: str = "train"
    speaker_id: str | None = None
    start_time: float
    end_time: float
    duration_seconds: float
    p_en: float | None = None
    lid: str | None = None
    pipeline_status: str
    audio_url: str
    #: None when the segment has no precomputed peaks; the UI falls back to a plain player.
    peaks_url: str | None = None
    hypotheses: list[HypothesisOut] = Field(default_factory=list)
    scores: ScoresOut | None = None
    latest_label: LabelOut | None = None
    #: Clip-relative stretches with two or more voices at once (D77). ``[]`` is measured and
    #: clean; None was never measured.
    overlap_spans: list[list[float]] | None = None
    #: Who spoke when inside the clip, from the episode's newest imported diarization (D78).
    #: Empty when the episode has none.
    speaker_turns: list[SpeakerTurnOut] = Field(default_factory=list)
    diarization_model: str | None = None


class QueueRowOut(BaseModel):
    """One row of the triage list: dense on purpose, ~15 fit on a screen."""

    task_id: int
    segment_id: int
    segment_external_id: str
    episode_external_id: str
    queue: str
    status: str
    pot: str = "train"
    priority_score: float
    reason: dict[str, Any] | None = None
    flags: list[str] = Field(default_factory=list)
    duration_seconds: float
    seed_hypothesis_id: int | None = None
    seed_system_id: str | None = None
    seed_text: str | None = None
    cmi: float | None = None
    word_disagreement_rate: float | None = None
    #: Share of the clip spent in crosstalk; null means never measured, not clean (D77).
    overlap_share: float | None = None
    audio_url: str
    peaks_url: str | None = None


class DisputeAlternativeOut(BaseModel):
    """What one other system heard at a disputed moment."""

    system_id: str
    word: str


class DisputeOut(BaseModel):
    """A seed word every other system that spoke at that moment contradicted."""

    seed_position: int
    seed_word: str
    start_time: float
    end_time: float
    alternatives: list[DisputeAlternativeOut]


class LaneSpeakerOut(BaseModel):
    """One of the episode's speakers: a lane the multitrack editor can show (D98)."""

    number: int
    label: str
    voice: str | None = None
    #: Where this speaker's voiceprint comes from (D99): ``confirmed`` clips, the ``diarizer``'s
    #: centroid, or null when it has none.
    print_source: str | None = None
    print_clips: int = 0


class LaneWordBase(BaseModel):
    """One block on the lanes. Spans are clip-relative seconds; ``speaker`` is a display number."""

    word: str = Field(min_length=1, max_length=200)
    start: float | None = None
    end: float | None = None
    speaker: int | None = None
    proposed_speaker: int | None = None
    source: str = Field(default="label", pattern="^(label|recogniser|typed|copy)$")
    #: The lane a voiceprint suggested when the lanes were served (D99); echoed back on save.
    suggested_speaker: int | None = None


class LaneWordOut(LaneWordBase):
    """A block as served; ``start``/``end`` are null for a word no span was measured for."""

    #: How much closer the suggested speaker's print was than the next one.
    suggestion_margin: float | None = None


class LaneCandidateOut(BaseModel):
    """A word the recognisers heard where the verified text has nothing."""

    word: str
    start: float
    end: float
    systems: list[str]


class LanesOut(BaseModel):
    """What the multitrack editor opens a speakers-queue task with (D98)."""

    diarization_run_id: int
    diarization_model: str
    speakers: list[LaneSpeakerOut]
    clip_speakers: list[int]
    words: list[LaneWordOut]
    candidates: list[LaneCandidateOut]
    #: ``speakers`` (an earlier per-speaker label), ``label`` (the verified text) or ``seed``.
    base: str
    #: Whether voiceprint suggestions were computed for these lanes (D99).
    voiceprint: bool = False


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
    #: Computed per request from the stored word spans, against this task's seed. Empty when the
    #: seed has no word timings, which is the honest answer rather than a claim of agreement.
    disputes: list[DisputeOut] = Field(default_factory=list)
    #: Only for a task in the speakers queue whose episode has been diarized (D98).
    lanes: LanesOut | None = None


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
    #: How much attention this decision got. ``verified`` means the clip was played and the
    #: transcript read; ``screened`` means it was accepted on the cross-ASR disagreement signal
    #: without listening. Defaults to ``verified`` so a client that does not send the field cannot
    #: quietly weaken what the corpus claims about itself. A ``screened`` decision on a gold-pot
    #: segment is refused with 409 (D63).
    verification_tier: str = Field(default="verified", pattern="^(verified|screened)$")


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


class AttributeIn(BaseModel):
    """Save a clip's words on their speakers' lanes (D98). Always a verified decision."""

    diarization_run_id: int
    words: list[LaneWordBase] = Field(min_length=1, max_length=2000)
    opened_at: dt.datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    annotator: str | None = None
    notes: str | None = None


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
    verification_tier: str | None = None
    task_status: str
    duration_ms: int | None = None


class BulkAcceptOut(BaseModel):
    """Result of a bulk accept. All or nothing: one transaction."""

    accepted: list[DecisionOut]
    count: int


class BulkDeleteSegmentsIn(BaseModel):
    """Delete several clips at once, from the episode clip list."""

    segment_ids: list[int] = Field(min_length=1)


class BulkDeleteSegmentsOut(BaseModel):
    """Result of a bulk delete. All or nothing: one transaction."""

    deleted: list[int]
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


class EpisodeSummary(BaseModel):
    """One episode in the management list, with its annotation progress."""

    id: int
    external_id: str
    title: str | None = None
    show_id: str | None = None
    duration_seconds: float | None = None
    split: str = "unassigned"
    #: Clips of this episode the owner has put in gold (D71).
    gold_count: int = 0
    segment_count: int = 0
    labeled_count: int = 0
    pending_count: int = 0


class EpisodeSegmentSummary(BaseModel):
    """One segment row in the episode detail view."""

    id: int
    external_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    pipeline_status: str
    pot: str = "train"
    #: Status of the segment's active task, or None when it has no outstanding work.
    task_status: str | None = None
    task_id: int | None = None
    seed_text: str | None = None
    flags: list[str] = Field(default_factory=list)
    cmi: float | None = None
    word_disagreement_rate: float | None = None
    audio_url: str
    peaks_url: str | None = None


class ExportIn(BaseModel):
    """Parameters to trigger a dataset export."""

    kind: str = Field(
        default="training",
        description="One of training, gold, analytics, error_mining, all",
    )
    label_version: str | None = None
    episode: str | None = None


class ExportOutItem(BaseModel):
    """Result of exporting one kind."""

    kind: str
    row_count: int
    row_counts_by_split: dict[str, int]
    data_filename: str
    manifest_filename: str
    download_url: str
    manifest_url: str
    manifest: dict[str, Any]


class ExportOut(BaseModel):
    """Payload returned by POST /export."""

    results: list[ExportOutItem]


class ExportHistoryItem(BaseModel):
    """One previously exported dataset found on disk."""

    kind: str
    data_filename: str
    manifest_filename: str
    download_url: str
    manifest_url: str
    row_count: int
    row_counts_by_split: dict[str, int]
    exported_at: str | None = None
    file_bytes: int = 0


# --- Cost Tracker Models -------------------------------------------------------------------


class CostSummaryOut(BaseModel):
    """High-level summary of AI inference spend and request counts."""

    total_cost_usd: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    dry_run_requests: int
    average_latency_ms: float | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


class VendorCostOut(BaseModel):
    """Aggregate spend and performance metrics per vendor."""

    vendor: str
    cost_usd: float
    percentage: float
    requests: int
    successful: int
    failed: int
    dry_run: int
    average_latency_ms: float | None = None


class ModelCostOut(BaseModel):
    """Aggregate spend per model / route."""

    route: str
    model: str
    vendor: str
    cost_usd: float
    requests: int
    successful: int
    failed: int
    dry_run: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    average_latency_ms: float | None = None
    effective_rate_display: str | None = None


class CostTimelinePointOut(BaseModel):
    """Daily cost data point for trend charting."""

    date: str
    cost_usd: float
    requests: int
    by_vendor: dict[str, float] = Field(default_factory=dict)


class PricingCatalogItemOut(BaseModel):
    """Published pricing reference entry."""

    vendor: str
    route: str
    model: str
    pricing_unit: str
    base_rate_usd: float | None = None
    keyterm_rate_usd: float | None = None
    input_per_m_usd: float | None = None
    output_per_m_usd: float | None = None
    effective_rate_display: str
    description: str


class CostReportOut(BaseModel):
    """Full cost report payload for the dashboard."""

    summary: CostSummaryOut
    vendor_breakdown: list[VendorCostOut]
    model_breakdown: list[ModelCostOut]
    daily_timeline: list[CostTimelinePointOut]
    pricing_catalog: list[PricingCatalogItemOut]


class LlmRequestItemOut(BaseModel):
    """One individual inference request ledger record."""

    id: int
    route: str
    model: str | None = None
    vendor: str
    status: str
    estimated_cost_usd: float
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_summary: str | None = None
    error_message: str | None = None
    created_at: dt.datetime


class CostRequestsListOut(BaseModel):
    """Paginated list of request ledger items."""

    total: int
    items: list[LlmRequestItemOut]


class SegmentPotIn(BaseModel):
    """Move one clip into or out of gold (D71)."""

    pot: Literal["gold", "train"]
    #: Carried on the audit row.
    reason: str | None = Field(default=None, max_length=500)


class SegmentPotOut(BaseModel):
    """Where the clip is now, and whether the request moved it."""

    segment_id: int
    external_id: str
    pot: str
    changed: bool


class ModelEvalRunOut(BaseModel):
    """One import of a fine-tuned model's gold or val transcripts, as scored then (D83)."""

    id: int
    split: str
    decoder: str | None = None
    fold_version: str
    clip_count: int
    #: ``wer``, ``wer_ci``, ``raw_wer``, ``cer``, ``loops``, ``by_genre``, ``by_class``,
    #: ``skipped`` -- see :func:`app.services.model_eval.summarize`.
    metrics: dict[str, Any]
    source: str | None = None
    created_at: dt.datetime


class AsrModelOut(BaseModel):
    """One fine-tuned model and every run imported for it, oldest first."""

    id: int
    slug: str
    name: str
    description: str | None = None
    architecture: str | None = None
    trained_at: dt.datetime | None = None
    card: dict[str, Any] = Field(default_factory=dict)
    runs: list[ModelEvalRunOut] = Field(default_factory=list)
    #: The subfolder of CPU weights the playground loads (``cpu`` or ``best``), or ``None`` when
    #: none has been copied next to the card (D85).
    playground: str | None = None


class PlaygroundOut(BaseModel):
    """One recording from the Models page, transcribed on the CPU (D85)."""

    text: str
    audio_s: float
    latency_ms: int | None = None
    weights: str
    dry_run: bool = False
    #: The sidecar's answer: decoder timings, whether the loop retry fired, the first decode.
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelRescanOut(BaseModel):
    """What a rescan of the model folders did."""

    models: list[str]
    runs_created: int
    runs_unchanged: int
    skipped: int


class ClassAxisOut(BaseModel):
    """One axis a run's clips are split by (D87): its buckets in display order, the bucket rate
    ratios compare against, and the bucket meaning "not measured"."""

    name: str
    label: str
    buckets: list[str]
    baseline: str | None = None
    unmeasured: str | None = None
    descriptive: bool = False


class ModelClipOut(BaseModel):
    """One clip of a run. ``segment_id`` is the harness id, for ``/segments/{id}`` and audio."""

    segment_id: int
    external_id: str
    episode_external_id: str
    genre: str
    duration_seconds: float
    overlap_share: float | None = None
    overlap_bucket: str
    ref_words: int
    errors: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    raw_errors: int
    raw_ref_words: int
    char_errors: int
    ref_chars: int
    is_loop: bool
    #: The clip's classes, ``{axis: bucket}``, as stored with the run (D87); empty before the
    #: run was classified.
    classes: dict[str, str] = Field(default_factory=dict)
    ref_text: str
    hyp_text: str


class ModelClipPageOut(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[ModelClipOut]


class AlignOpOut(BaseModel):
    """One step of the folded word alignment: ``match``, ``fold``, ``merge``, ``sub``, ``del`` or
    ``ins``. ``fold`` is one word in two scripts and ``merge`` one word split in two -- both are
    matches, not errors."""

    kind: str
    ref: list[str]
    hyp: list[str]
    similarity: float


class ModelClipDetailOut(ModelClipOut):
    ops: list[AlignOpOut]
    fold_version: str
    #: The fold rules changed since the run was imported, so ``ops`` may not add up to the
    #: stored counts.
    fold_version_changed: bool


class VoiceEpisodeOut(BaseModel):
    """An episode a voice speaks in, and which speaker it is there."""

    episode_id: int
    external_id: str
    title: str | None = None
    speaker_number: int
    talk_seconds: float
    solo_clips: int


class VoiceClipOut(BaseModel):
    """A clip the diarization hears as one voice alone, and the owner's verdict on it (D99)."""

    segment_id: int
    external_id: str
    episode_external_id: str
    pot: str
    duration_seconds: float
    #: What the seed says in the stretch, from its word spans.
    text: str | None = None
    verdict: str | None = None
    audio_url: str
    #: The stretch of this voice alone that is played and judged, clip-relative seconds.
    start: float
    end: float
    #: The stretch is the whole clip: nobody else diarized in it, no crosstalk detected.
    whole: bool


class VoiceOut(BaseModel):
    """Everything the voice page shows about one anonymous voice (D87, D99)."""

    voice: str
    talk_seconds: float
    episodes: list[VoiceEpisodeOut]
    clips: list[VoiceClipOut]
    total_clips: int
    confirmed: int
    rejected: int
    #: ``confirmed`` once the owner has confirmed a clip, else ``diarizer``.
    print_source: str
    reference: VoiceClipOut | None = None


class VoiceVerdictIn(BaseModel):
    """Only this voice (``confirmed``), not only this voice (``rejected``), or take it back."""

    verdict: str = Field(pattern="^(confirmed|rejected|cleared)$")
    annotator: str | None = None


class VoiceVerdictOut(BaseModel):
    voice: str
    segment_id: int
    verdict: str
    embedded: bool
    confirmed: int
    rejected: int
