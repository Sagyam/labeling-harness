"""Health endpoint: is the process up, and can it reach Postgres and object storage."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app.config import get_settings
from app.db.session import get_engine

router = APIRouter(tags=["ops"])


def _check_database() -> tuple[bool, str | None]:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        return False, str(exc)[:200]
    return True, None


def _check_storage() -> tuple[bool, str | None]:
    from app.storage import get_storage

    try:
        get_storage().healthcheck()
    except Exception as exc:  # pragma: no cover - exercised in storage tests
        return False, str(exc)[:200]
    return True, None


@router.get("/health")
def health() -> dict[str, Any]:
    """Report process health and the reachability of Postgres and object storage."""
    settings = get_settings()
    db_ok, db_error = _check_database()
    storage_ok, storage_error = _check_storage()
    checks = {
        "database": {"ok": db_ok, "error": db_error},
        "storage": {"ok": storage_ok, "error": storage_error, "backend": settings.storage.backend},
    }
    return {
        "status": "ok" if db_ok and storage_ok else "degraded",
        "app": settings.app.name,
        "environment": settings.app.environment,
        "checks": checks,
    }
