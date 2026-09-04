"""Cost tracker endpoints: aggregate spend report and inference request ledger."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_session, require_auth
from app.api.schemas import CostReportOut, CostRequestsListOut
from app.services.costs import collect_cost_report, query_cost_requests

router = APIRouter(prefix="/costs", tags=["costs"], dependencies=[Depends(require_auth)])


@router.get("", response_model=CostReportOut)
def get_costs(session: Session = Depends(get_session)) -> CostReportOut:
    """Aggregate AI inference cost report across ElevenLabs, OpenRouter, and Vertex AI."""
    return CostReportOut(**collect_cost_report(session))


@router.get("/requests", response_model=CostRequestsListOut)
def get_cost_requests(
    session: Session = Depends(get_session),
    vendor: str | None = Query(default=None, description="Filter by vendor name"),
    route: str | None = Query(default=None, description="Filter by route name"),
    status: str | None = Query(
        default=None, description="Filter by status (succeeded, failed, dry_run)"
    ),
    search: str | None = Query(default=None, description="Search route, model, summary or error"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> CostRequestsListOut:
    """Paginated, filterable inference request audit ledger."""
    total, items = query_cost_requests(
        session,
        vendor=vendor,
        route=route,
        status=status,
        search=search,
        limit=limit,
        offset=offset,
    )
    return CostRequestsListOut(total=total, items=items)
