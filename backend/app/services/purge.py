"""Permanent removal of one ASR system's hypotheses from the corpus.

This is the one operation in the harness that overrides D6 -- hypotheses are immutable -- and
it does so only when the owner asks for a specific system by name. Two things make that
defensible rather than merely destructive:

* **Every row is written to a JSONL dump before anything is deleted.** The delete is
  irreversible; the dump is what keeps it recoverable.
* **Affected segment scores are recomputed.** ``word_disagreement_rate`` and
  ``cer_between_hypotheses`` describe a set of hypotheses. Once one of them is gone the stored
  value describes a set that no longer exists, which is worse than either a fresh number or a
  null. The recomputation uses the same helper the ingest pipeline does, so it reproduces
  exactly what the pipeline would have written with the surviving systems.

It is deliberately not an Alembic migration. Nothing about the schema changes, and Invariant 1
requires every migration to carry a working ``downgrade`` -- which a row deletion cannot
honestly provide.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import LlmRoutes, load_llm_routes
from app.llm.transcription import disagreement_excluded_system_ids
from app.models import AsrHypothesis, AsrSystem, AuditLog, Segment, SegmentScore
from app.models.content import HypothesisWord
from app.services.analysis import mean_pairwise_disagreement
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PurgedSystemNotFound(Exception):
    """No ``asr_systems`` row carries the requested ``system_id``."""


@dataclass
class PurgeReport:
    """What a purge removed, and what it had to recompute afterwards."""

    system_id: str
    hypotheses_deleted: int = 0
    words_deleted: int = 0
    segments_affected: list[str] = field(default_factory=list)
    segments_rescored: int = 0
    dump_path: Path | None = None
    dry_run: bool = False


def _dump_rows(
    session: Session, system: AsrSystem, hypotheses: list[AsrHypothesis]
) -> list[dict[str, Any]]:
    """Serialize every hypothesis and its words, resolving segment external ids."""
    segment_ids = {h.segment_id for h in hypotheses}
    external_by_id = dict(
        session.execute(
            sa.select(Segment.id, Segment.external_id).where(Segment.id.in_(segment_ids or {-1}))
        ).all()
    )

    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        words = session.scalars(
            sa.select(HypothesisWord)
            .where(HypothesisWord.hypothesis_id == hypothesis.id)
            .order_by(HypothesisWord.position)
        ).all()
        rows.append(
            {
                "system_id": system.system_id,
                "model_id": hypothesis.system.model_id if hypothesis.system else system.model_id,
                "segment_external_id": external_by_id.get(hypothesis.segment_id),
                "text_raw": hypothesis.text_raw,
                "text_normalized": hypothesis.text_normalized,
                "avg_logprob": hypothesis.avg_logprob,
                "no_speech_prob": hypothesis.no_speech_prob,
                "metadata": hypothesis.metadata_jsonb,
                "words": [
                    {
                        "word": w.word_raw,
                        "start": w.start_time,
                        "end": w.end_time,
                        "confidence": w.confidence,
                        "predicted_language": w.predicted_language,
                        "predicted_script": w.predicted_script,
                    }
                    for w in words
                ],
            }
        )
    return rows


def _rescore(session: Session, segment_ids: set[int], config: LlmRoutes | None = None) -> int:
    """Recompute disagreement for segments whose hypothesis set just changed.

    Reads the same hold-out set as the ingest path (D39): a system excluded there and counted
    here would silently rewrite every score a purge touched.
    """
    held_out = set(disagreement_excluded_system_ids(config or load_llm_routes()))
    rescored = 0
    for segment_id in sorted(segment_ids):
        score = session.get(SegmentScore, segment_id)
        if score is None:
            continue
        texts = list(
            session.scalars(
                sa.select(AsrHypothesis.text_raw)
                .join(AsrSystem, AsrSystem.id == AsrHypothesis.asr_system_id)
                .where(
                    AsrHypothesis.segment_id == segment_id,
                    AsrSystem.system_id.not_in(held_out) if held_out else sa.true(),
                )
            ).all()
        )
        score.word_disagreement_rate = mean_pairwise_disagreement([t.split() for t in texts])
        score.cer_between_hypotheses = mean_pairwise_disagreement(texts)
        rescored += 1
    return rescored


def purge_asr_system(
    session: Session,
    system_id: str,
    *,
    dump_dir: Path | str,
    rescore: bool = True,
    actor: str = "owner",
    dry_run: bool = False,
    config: LlmRoutes | None = None,
) -> PurgeReport:
    """Delete one ASR system, its hypotheses and their word spans.

    Args:
        session: Open session; the caller commits.
        system_id: ``asr_systems.system_id`` to remove.
        dump_dir: Directory the pre-delete JSONL dump is written into.
        rescore: Recompute disagreement scores on the affected segments.
        config: Routing table, for the disagreement hold-out set. Defaults to the committed one.
        actor: Recorded on the audit entry.
        dry_run: Report what would be removed, touching nothing.

    Returns:
        A :class:`PurgeReport` describing the damage.

    Raises:
        PurgedSystemNotFound: No system carries that id.
    """
    system = session.scalar(sa.select(AsrSystem).where(AsrSystem.system_id == system_id))
    if system is None:
        raise PurgedSystemNotFound(f"no asr_systems row with system_id {system_id!r}")

    hypotheses = list(
        session.scalars(
            sa.select(AsrHypothesis).where(AsrHypothesis.asr_system_id == system.id)
        ).all()
    )
    hypothesis_ids = [h.id for h in hypotheses]
    segment_ids = {h.segment_id for h in hypotheses}

    word_count = (
        session.scalar(
            sa.select(sa.func.count())
            .select_from(HypothesisWord)
            .where(HypothesisWord.hypothesis_id.in_(hypothesis_ids))
        )
        or 0
        if hypothesis_ids
        else 0
    )

    rows = _dump_rows(session, system, hypotheses)
    externals = [str(row["segment_external_id"]) for row in rows if row["segment_external_id"]]

    report = PurgeReport(
        system_id=system_id,
        hypotheses_deleted=len(hypotheses),
        words_deleted=int(word_count),
        segments_affected=externals,
        dry_run=dry_run,
    )
    if dry_run:
        logger.info("purge_dry_run", system_id=system_id, hypotheses=len(hypotheses))
        return report

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    dump_path = Path(dump_dir) / f"purged_{system_id}_{stamp}.jsonl"
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report.dump_path = dump_path

    if hypothesis_ids:
        session.execute(
            sa.delete(HypothesisWord).where(HypothesisWord.hypothesis_id.in_(hypothesis_ids))
        )
        session.execute(sa.delete(AsrHypothesis).where(AsrHypothesis.id.in_(hypothesis_ids)))
    session.flush()
    session.delete(system)
    session.flush()

    if rescore:
        report.segments_rescored = _rescore(session, segment_ids, config)

    session.add(
        AuditLog(
            entity_type="asr_system",
            entity_id=system_id,
            action="purge",
            actor=actor,
            old_values_jsonb={
                "hypotheses_deleted": report.hypotheses_deleted,
                "words_deleted": report.words_deleted,
                "segments_affected": len(externals),
                "segments_rescored": report.segments_rescored,
                "dump_path": str(dump_path),
            },
        )
    )
    session.flush()
    logger.info(
        "purge_complete",
        system_id=system_id,
        hypotheses=report.hypotheses_deleted,
        words=report.words_deleted,
        dump=str(dump_path),
    )
    return report
