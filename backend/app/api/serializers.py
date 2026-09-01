"""Turning ORM rows into API payloads."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.schemas import (
    HypothesisOut,
    LabelOut,
    QueueRowOut,
    ScoresOut,
    SegmentOut,
)
from app.models import AnnotationTask, AsrHypothesis, Segment
from app.services.labeling import latest_label


def audio_url(segment_id: int) -> str:
    """URL of the segment's audio stream."""
    return f"/segments/{segment_id}/audio"


def peaks_url(segment_id: int) -> str:
    """URL of the segment's precomputed waveform peaks."""
    return f"/segments/{segment_id}/peaks"


def serialize_hypothesis(hypothesis: AsrHypothesis) -> HypothesisOut:
    """Serialize one hypothesis."""
    return HypothesisOut(
        id=hypothesis.id,
        system_id=hypothesis.system.system_id,
        model_id=hypothesis.system.model_id,
        text=hypothesis.text_raw,
        avg_logprob=hypothesis.avg_logprob,
        no_speech_prob=hypothesis.no_speech_prob,
        word_count=len(hypothesis.words),
    )


def serialize_segment(session: Session, segment: Segment) -> SegmentOut:
    """Serialize a segment with hypotheses, scores and its current label."""
    label = latest_label(session, segment.id)
    return SegmentOut(
        id=segment.id,
        external_id=segment.external_id,
        episode_id=segment.episode_id,
        episode_external_id=segment.episode.external_id,
        split=segment.episode.split,
        speaker_id=segment.speaker_id,
        start_time=segment.start_time,
        end_time=segment.end_time,
        duration_seconds=segment.duration_seconds,
        p_en=segment.p_en,
        lid=segment.lid,
        pipeline_status=segment.pipeline_status,
        audio_url=audio_url(segment.id),
        peaks_url=peaks_url(segment.id),
        hypotheses=[
            serialize_hypothesis(h) for h in sorted(segment.hypotheses, key=lambda h: h.id)
        ],
        scores=(
            ScoresOut(
                cer_between_hypotheses=segment.scores.cer_between_hypotheses,
                word_disagreement_rate=segment.scores.word_disagreement_rate,
                script_conflict_rate=segment.scores.script_conflict_rate,
                code_switch_density=segment.scores.code_switch_density,
                flags=list(segment.scores.flags_jsonb or []),
            )
            if segment.scores
            else None
        ),
        latest_label=(
            LabelOut(
                id=label.id,
                disposition=label.disposition,
                final_text=label.final_text,
                annotator=label.annotator,
                label_version=label.label_version.name,
                policy_version=label.label_version.policy_version,
                notes=label.notes,
                created_at=label.created_at,
            )
            if label
            else None
        ),
    )


def serialize_queue_row(task: AnnotationTask) -> QueueRowOut:
    """Serialize one triage row: enough to decide without opening the editor."""
    segment = task.segment
    seed = task.seed_hypothesis
    return QueueRowOut(
        task_id=task.id,
        segment_id=segment.id,
        segment_external_id=segment.external_id,
        episode_external_id=segment.episode.external_id,
        queue=task.queue,
        status=task.status,
        priority_score=task.priority_score,
        reason=task.reason_jsonb,
        flags=list(segment.scores.flags_jsonb or []) if segment.scores else [],
        duration_seconds=segment.duration_seconds,
        seed_hypothesis_id=task.seed_hypothesis_id,
        seed_system_id=seed.system.system_id if seed else None,
        seed_text=seed.text_raw if seed else None,
        audio_url=audio_url(segment.id),
        peaks_url=peaks_url(segment.id),
    )
