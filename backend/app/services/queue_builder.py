"""Queue building: turn imported segments into prioritized annotation tasks.

The builder is a pure function over a batch of segments -- no global state, no background worker --
so putting it behind a job queue later is a wiring change rather than a rewrite. Re-running it is
safe: existing active tasks have their priority and reason refreshed rather than being duplicated.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.models import AnnotationTask, AsrHypothesis, AsrSystem, Episode, Segment, SegmentScore
from app.models.enums import ACTIVE_TASK_STATUSES
from app.services.consensus import (
    ConsensusHypothesis,
    ConsensusWord,
    build_slots,
    seed_outvoted_fraction,
)
from app.services.hazards import FusionEvidence, assess
from app.services.lexical import lexical_signals
from app.services.scoring import ScoreInputs, priority_score
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueueReport:
    """What a queue build did, or would have done in a dry run."""

    segments_considered: int = 0
    tasks_created: int = 0
    tasks_updated: int = 0
    review_tasks: int = 0
    audit_tasks: int = 0
    error_tasks: int = 0
    dry_run: bool = False

    def render(self) -> str:
        """A short human-readable summary, for the CLI."""
        return "\n".join(
            [
                f"{'DRY RUN -- ' if self.dry_run else ''}queue build",
                f"  segments considered  {self.segments_considered}",
                f"  tasks created        {self.tasks_created}",
                f"  tasks updated        {self.tasks_updated}",
                f"  review               {self.review_tasks}",
                f"  audit                {self.audit_tasks}",
                f"  error                {self.error_tasks}",
            ]
        )


def _is_fusion(hypothesis: AsrHypothesis) -> bool:
    return hypothesis.system.kind == "fusion"


def strongest_recogniser(hypotheses: list[AsrHypothesis]) -> AsrHypothesis | None:
    """The recogniser hypothesis the old queue would have seeded with: highest ``avg_logprob``."""
    recognisers = [h for h in hypotheses if not _is_fusion(h)]
    if not recognisers:
        return None
    return max(
        sorted(recognisers, key=lambda h: h.id),
        key=lambda h: (h.avg_logprob if h.avg_logprob is not None else float("-inf"), h.id),
    )


def select_seed_hypothesis(hypotheses: list[AsrHypothesis]) -> AsrHypothesis | None:
    """Choose which hypothesis preloads the editor: the fused transcript, for every clip (D74).

    Gold included -- the owner chose the faster seed over a benchmark independent of the fuser,
    and the anchoring cost is recorded in D74. The newest fusion system wins, so a re-fusion under
    a new prompt version takes over without touching the old hypotheses. A clip with no fused text
    falls back to the strongest recogniser, and is gated (``unfused``) so it cannot be screened.
    """
    fused = [h for h in hypotheses if _is_fusion(h)]
    if fused:
        return max(fused, key=lambda h: (h.system.id, h.id))
    return strongest_recogniser(hypotheses)


def _seed_outvoted(segment: Segment, seed: AsrHypothesis | None) -> float | None:
    """D67's speech-time term, over the recognisers only, for the recorded legacy score."""
    if seed is None:
        return None
    hypotheses = [
        ConsensusHypothesis(
            system_id=h.system.system_id,
            words=[
                ConsensusWord(
                    position=w.position, word=w.word_raw, start=w.start_time, end=w.end_time
                )
                for w in h.words
            ],
        )
        for h in segment.hypotheses
        if not _is_fusion(h)
    ]
    slots = build_slots(hypotheses)
    if not slots:
        return None
    return seed_outvoted_fraction(slots, seed_system_id=seed.system.system_id)


def _neighbour_texts(session: Session, episode_ids: set[int]) -> dict[int, list[str]]:
    """``{segment_id: recogniser texts of the clips just before and after it}``.

    Read for whole episodes, not just the segments being queued: a clip's neighbour may already
    be labelled and out of this batch, and its recognisers are still the evidence for a seam.
    """
    if not episode_ids:
        return {}
    rows = session.execute(
        sa.select(Segment.id, Segment.episode_id, Segment.start_time, AsrHypothesis.text_raw)
        .join(AsrHypothesis, AsrHypothesis.segment_id == Segment.id)
        .join(AsrSystem, AsrSystem.id == AsrHypothesis.asr_system_id)
        .where(Segment.episode_id.in_(episode_ids), AsrSystem.kind == "asr")
        .order_by(Segment.episode_id, Segment.start_time, Segment.id)
    ).all()
    texts: dict[int, list[str]] = defaultdict(list)
    order: dict[int, list[int]] = defaultdict(list)
    for segment_id, episode_id, _start, text in rows:
        if not order[episode_id] or order[episode_id][-1] != segment_id:
            order[episode_id].append(segment_id)
        texts[segment_id].append(text or "")
    out: dict[int, list[str]] = {}
    for sequence in order.values():
        for position, segment_id in enumerate(sequence):
            around = (
                sequence[max(0, position - 1) : position] + sequence[position + 1 : position + 2]
            )
            out[segment_id] = [t for neighbour in around for t in texts[neighbour]]
    return out


def _score_for(
    segment: Segment,
    scores: SegmentScore | None,
    seed: AsrHypothesis | None,
    neighbours: list[str],
    settings: Settings,
) -> tuple[float, dict[str, Any]]:
    recognisers = [h for h in segment.hypotheses if not _is_fusion(h)]
    fused = seed if seed is not None and _is_fusion(seed) else None
    metadata = (fused.metadata_jsonb or {}) if fused is not None else {}
    report = assess(
        FusionEvidence(
            fused_text=fused.text_raw if fused is not None else None,
            asr_texts=[h.text_raw for h in recognisers],
            neighbour_texts=neighbours,
            fused_code=(metadata.get("fusion") or {}).get("code"),
            acoustic=metadata.get("acoustic"),
        ),
        config=settings.queue.hazards,
    )
    flags = list(scores.flags_jsonb or []) if scores else []
    scribe_logprob = next((h.avg_logprob for h in recognisers if h.avg_logprob is not None), None)

    fallback = strongest_recogniser(list(segment.hypotheses))
    orphan_rate, latin_gap = lexical_signals(
        fallback.text_raw if fallback else None,
        [h.text_raw for h in recognisers if fallback is None or h.id != fallback.id],
    )
    result = priority_score(
        ScoreInputs(
            unsupported_rate=report.unsupported_rate if fused is not None else None,
            dropped_rate=report.dropped_rate if fused is not None else None,
            asr_disagreement=report.asr_disagreement,
            acoustic_gap=report.acoustic_gap,
            avg_logprob=scribe_logprob,
            flags=flags,
            hazards=report.hazards,
        ),
        settings=settings,
        hazard_details=report.details,
        legacy=ScoreInputs.legacy(
            seed_outvoted=_seed_outvoted(segment, fallback),
            seed_orphan_rate=orphan_rate,
            roman_gap=latin_gap,
            avg_logprob=fallback.avg_logprob if fallback else None,
            flags=flags,
        ),
    )
    return result.priority, result.as_reason()


def _audit_selection(candidates: list[tuple[int, float]], *, rate: float, seed: int) -> set[int]:
    """Pick a reproducible random sample of the easiest segments.

    Quality on the disagreeing minority is measured by the review queue itself. The audit queue
    exists so quality on the easy majority -- the segments accepted with one keystroke -- stays
    measurable too.
    """
    if rate <= 0 or not candidates:
        return set()
    ranked = sorted(candidates, key=lambda item: (item[1], item[0]))
    pool_size = max(1, len(ranked) // 2)  # the low-priority, high-agreement half
    pool = [segment_id for segment_id, _ in ranked[:pool_size]]
    sample_size = min(len(pool), max(1, round(len(candidates) * rate)))
    return set(random.Random(seed).sample(pool, sample_size))


def build_queue(
    session: Session,
    *,
    settings: Settings | None = None,
    episode_external_id: str | None = None,
    audit_sample_rate: float | None = None,
    audit_seed: int | None = None,
    requeue_done: bool = False,
    dry_run: bool = False,
) -> QueueReport:
    """Create or refresh annotation tasks for imported segments.

    Args:
        session: Open session; the caller commits.
        settings: Configuration override.
        episode_external_id: Restrict the build to a single episode.
        audit_sample_rate: Fraction of easy segments to route to the audit queue.
        audit_seed: Seed for audit sampling; the default makes the sample reproducible.
        requeue_done: Also queue segments that already have a completed task. Off by default, so
            re-running the builder never hands the annotator work they have already finished. A
            *skipped* task is different: it was deferred, so its segment does come back.
        dry_run: Report what would change and write nothing.

    Returns:
        A :class:`QueueReport` describing the build.
    """
    settings = settings or get_settings()
    rate = settings.queue.audit_sample_rate if audit_sample_rate is None else audit_sample_rate
    seed = settings.queue.audit_seed if audit_seed is None else audit_seed

    query = (
        sa.select(Segment)
        .join(Episode, Episode.id == Segment.episode_id)
        .options(
            selectinload(Segment.hypotheses).selectinload(AsrHypothesis.words),
            selectinload(Segment.hypotheses).selectinload(AsrHypothesis.system),
            selectinload(Segment.scores),
        )
        .where(Segment.pipeline_status.in_(("imported", "queued")))
        .order_by(Segment.id)
    )
    if episode_external_id:
        query = query.where(Episode.external_id == episode_external_id)
    if not requeue_done:
        query = query.where(
            ~sa.exists().where(
                AnnotationTask.segment_id == Segment.id, AnnotationTask.status == "done"
            )
        )
    segments = list(session.scalars(query))

    neighbours = _neighbour_texts(session, {s.episode_id for s in segments})

    report = QueueReport(segments_considered=len(segments), dry_run=dry_run)
    planned: list[tuple[Segment, AsrHypothesis | None, float, dict[str, Any]]] = []
    for segment in segments:
        seed_hypothesis = select_seed_hypothesis(list(segment.hypotheses))
        score, reason = _score_for(
            segment, segment.scores, seed_hypothesis, neighbours.get(segment.id, []), settings
        )
        planned.append((segment, seed_hypothesis, score, reason))

    # Segments with no hypothesis at all go to the error queue, and a gated clip is not part of
    # the easy majority audit exists to measure, so neither is a candidate for audit.
    audit_ids = _audit_selection(
        [
            (s.id, score)
            for s, hypothesis, score, reason in planned
            if hypothesis is not None and not reason.get("hazards")
        ],
        rate=rate,
        seed=seed,
    )

    active = {
        task.segment_id: task
        for task in session.scalars(
            sa.select(AnnotationTask).where(
                AnnotationTask.segment_id.in_([s.id for s in segments]),
                AnnotationTask.status.in_(ACTIVE_TASK_STATUSES),
            )
        )
    }

    for segment, seed_hypothesis, score, reason in planned:
        if seed_hypothesis is None:
            queue = "error"
        elif segment.id in audit_ids:
            queue = "audit"
        else:
            queue = "review"

        if queue == "review":
            report.review_tasks += 1
        elif queue == "audit":
            report.audit_tasks += 1
        else:
            report.error_tasks += 1

        existing = active.get(segment.id)
        if existing is not None:
            report.tasks_updated += 1
            if not dry_run:
                existing.queue = queue
                existing.priority_score = score
                existing.reason_jsonb = reason
                existing.seed_hypothesis_id = (
                    seed_hypothesis.id if seed_hypothesis is not None else None
                )
            continue

        report.tasks_created += 1
        if dry_run:
            continue
        session.add(
            AnnotationTask(
                segment_id=segment.id,
                queue=queue,
                priority_score=score,
                seed_hypothesis_id=seed_hypothesis.id if seed_hypothesis is not None else None,
                reason_jsonb=reason,
                status="pending",
            )
        )
        if segment.pipeline_status == "imported":
            segment.pipeline_status = "queued"

    if not dry_run:
        session.flush()
    logger.info(
        "queue_build_complete",
        segments=report.segments_considered,
        created=report.tasks_created,
        updated=report.tasks_updated,
        dry_run=dry_run,
    )
    return report
