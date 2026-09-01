"""Queue building: turn imported segments into prioritized annotation tasks.

The builder is a pure function over a batch of segments -- no global state, no background worker --
so putting it behind a job queue later is a wiring change rather than a rewrite. Re-running it is
safe: existing active tasks have their priority and reason refreshed rather than being duplicated.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.models import AnnotationTask, AsrHypothesis, Episode, Segment, SegmentScore
from app.models.enums import ACTIVE_TASK_STATUSES
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


def select_seed_hypothesis(
    segment: Segment, hypotheses: list[AsrHypothesis], *, split: str
) -> AsrHypothesis | None:
    """Choose which hypothesis preloads the editor.

    For ``train`` and ``val`` episodes the strongest hypothesis is fastest to accept, so it wins.
    For ``test`` episodes the seed system is rotated deterministically by hashing the segment id:
    the rotation costs the annotator nothing -- it is the same one-key accept -- but it keeps the
    gold set from being anchored to a single system, and makes a per-seed WER breakdown possible
    later. Without the recorded seed that argument cannot be made at all.
    """
    if not hypotheses:
        return None
    ordered = sorted(hypotheses, key=lambda h: h.id)
    if split == "test":
        digest = hashlib.blake2b(segment.external_id.encode(), digest_size=8).digest()
        return ordered[int.from_bytes(digest, "big") % len(ordered)]
    return max(
        ordered,
        key=lambda h: (h.avg_logprob if h.avg_logprob is not None else float("-inf"), h.id),
    )


def _score_for(
    segment: Segment,
    scores: SegmentScore | None,
    seed: AsrHypothesis | None,
    settings: Settings,
) -> tuple[float, dict[str, Any]]:
    result = priority_score(
        ScoreInputs(
            word_disagreement_rate=scores.word_disagreement_rate if scores else None,
            avg_logprob=seed.avg_logprob if seed else None,
            code_switch_density=scores.code_switch_density if scores else None,
            flags=list(scores.flags_jsonb or []) if scores else [],
        ),
        settings=settings,
    )
    return result.score, result.as_reason()


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
        .options(selectinload(Segment.hypotheses), selectinload(Segment.scores))
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

    splits = dict(
        session.execute(
            sa.select(Episode.id, Episode.split).where(
                Episode.id.in_({s.episode_id for s in segments})
            )
        ).all()
    )

    report = QueueReport(segments_considered=len(segments), dry_run=dry_run)
    planned: list[tuple[Segment, AsrHypothesis | None, float, dict[str, Any]]] = []
    for segment in segments:
        seed_hypothesis = select_seed_hypothesis(
            segment, list(segment.hypotheses), split=splits.get(segment.episode_id, "unassigned")
        )
        score, reason = _score_for(segment, segment.scores, seed_hypothesis, settings)
        planned.append((segment, seed_hypothesis, score, reason))

    # Segments with no hypothesis at all go to the error queue and are never candidates for audit.
    audit_ids = _audit_selection(
        [(s.id, score) for s, hypothesis, score, _ in planned if hypothesis is not None],
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
