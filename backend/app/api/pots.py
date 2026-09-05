"""Pot assignment: placing episodes in the gold or train pot (D63)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_session, require_auth
from app.api.schemas import PotAssignIn, PotChangeOut, PotReportOut, PotStatusOut
from app.config import Settings
from app.services.pots import assign_pots, pot_status

router = APIRouter(tags=["pots"], dependencies=[Depends(require_auth)])


@router.get("/pots", response_model=PotStatusOut)
def get_pots(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> PotStatusOut:
    """What each pot currently holds, and what the gold pot does not yet cover.

    A pure read. Assignment is a separate POST precisely so that looking at the dashboard can never
    move an episode between pots.
    """
    status = pot_status(session, settings=settings)
    return PotStatusOut(
        gold_target_hours=status.gold_target_hours,
        gold_effective_target_hours=status.gold_effective_target_hours,
        gold_capped_by_corpus_size=status.gold_capped_by_corpus_size,
        train_target_hours=status.train_target_hours,
        buckets=status.buckets,
        gold_coverage=status.gold_coverage,
        corpus_coverage=status.corpus_coverage,
        gold_coverage_gaps=status.gold_coverage_gaps,
        coverage_complete=status.coverage_complete,
    )


@router.post("/pots/assign", response_model=PotReportOut)
def post_assign_pots(
    body: PotAssignIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> PotReportOut:
    """Place every unassigned episode in a pot, and redraw the train/val line.

    Ingestion already calls this, so the endpoint is for the cases ingestion cannot cover: raising
    the gold target after the fact, or placing episodes that arrived through the manifest importer
    rather than the web pipeline.
    """
    report = assign_pots(
        session,
        settings=settings,
        gold_hours_target=body.gold_hours_target,
        gold_max_corpus_fraction=body.gold_max_corpus_fraction,
        allow_promote_from_train=body.allow_promote_from_train,
        dry_run=body.dry_run,
    )
    return PotReportOut(
        gold_hours=report.gold_hours,
        train_hours=report.train_hours,
        val_hours=report.val_hours,
        unassigned_hours=report.unassigned_hours,
        gold_target_hours=report.gold_target_hours,
        gold_effective_target_hours=report.gold_effective_target_hours,
        gold_capped_by_corpus_size=report.gold_capped_by_corpus_size,
        train_target_hours=report.train_target_hours,
        gold_episodes=report.gold_episodes,
        train_episodes=report.train_episodes,
        val_episodes=report.val_episodes,
        unassigned_episodes=report.unassigned_episodes,
        gold_target_met=report.gold_target_met,
        gold_coverage=report.gold_coverage,
        gold_coverage_gaps=report.gold_coverage_gaps,
        dry_run=report.dry_run,
        changes=[
            PotChangeOut(
                external_id=c.external_id,
                from_pot=c.from_pot,
                to_pot=c.to_pot,
                from_split=c.from_split,
                to_split=c.to_split,
                hours=round(c.hours, 4),
            )
            for c in report.changes
        ],
    )
