"""FastAPI dependencies: session, settings, storage and the optional bearer token."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import session_scope
from app.storage import ObjectStorage, get_storage
from app.translit import TransliterationService


def get_session() -> Iterator[Session]:
    """Request-scoped transactional session."""
    with session_scope() as session:
        yield session


def get_session_factory() -> Callable[[], Session]:
    """Factory yielding new database sessions."""
    from app.db.session import get_sessionmaker

    return get_sessionmaker()


def get_config() -> Settings:
    """Loaded settings."""
    return get_settings()


def get_object_storage() -> ObjectStorage:
    """Configured object storage."""
    return get_storage()


def require_auth(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_config),
) -> None:
    """Enforce a static bearer token when one is configured.

    Local development leaves ``api.auth_token`` empty, which disables authentication entirely. The
    hook exists so that putting the harness behind a token later is configuration, not code.
    """
    expected = settings.api.auth_token
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_translit_service(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> TransliterationService:
    """Cache-first transliteration service bound to the request session."""
    return TransliterationService(session, settings=settings)
