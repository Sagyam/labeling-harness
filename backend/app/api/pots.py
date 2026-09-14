"""Moving one clip into or out of gold (D71)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_session, require_auth
from app.api.schemas import SegmentPotIn, SegmentPotOut
from app.config import Settings
from app.models import Segment
from app.services.pots import PotError, set_segment_pot

router = APIRouter(tags=["pots"], dependencies=[Depends(require_auth)])


@router.post("/segments/{segment_id}/pot", response_model=SegmentPotOut)
def post_segment_pot(
    segment_id: int,
    body: SegmentPotIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> SegmentPotOut:
    """Put one clip in gold, or take it back out. 409 if a screened clip is sent to gold."""
    segment = session.get(Segment, segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail=f"segment {segment_id} not found")
    try:
        move = set_segment_pot(
            session,
            segment,
            body.pot,
            actor=settings.labels.default_annotator,
            reason=body.reason,
        )
    except PotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return SegmentPotOut(
        segment_id=segment.id,
        external_id=segment.external_id,
        pot=segment.pot,
        changed=move.changed,
    )
