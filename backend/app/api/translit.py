"""Transliteration endpoint: the annotator's input method."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_translit_service, require_auth
from app.api.schemas import TranslitChoiceIn, TranslitIn, TranslitOut
from app.translit import TransliterationService

router = APIRouter(tags=["translit"], dependencies=[Depends(require_auth)])


@router.post("/translit", response_model=TranslitOut)
def transliterate(
    body: TranslitIn,
    service: TransliterationService = Depends(get_translit_service),
) -> TranslitOut:
    """Return ranked Devanagari candidates for a Latin token.

    The cache is consulted first, so a recurring token never leaves the database. A provider
    failure degrades to the next provider rather than to an error, because an error dialog in the
    middle of typing is worse than a slightly worse candidate list.
    """
    candidates = service.suggest(body.token)
    if body.limit is not None:
        candidates = candidates[: body.limit]
    return TranslitOut(token=body.token, candidates=candidates)


@router.post("/translit/choice", response_model=TranslitOut)
def record_choice(
    body: TranslitChoiceIn,
    service: TransliterationService = Depends(get_translit_service),
) -> TranslitOut:
    """Remember which candidate was chosen, so it ranks first next time."""
    service.record_choice(body.token, body.devanagari)
    return TranslitOut(token=body.token, candidates=service.suggest(body.token))
