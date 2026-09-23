"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings, load_dotenv
from app.utils.logging import configure_logging, get_logger


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Resume the ingest jobs a crash or power cut interrupted, before serving anything (D93).

    The collaborators are resolved through the app's dependency overrides, so a test that points
    the app at a scratch work root and database resumes from there, not from the real ones.
    """
    from app.api.deps import get_config, get_object_storage, get_session_factory
    from app.services.ingest import manager

    def resolve(dependency: Callable[[], Any]) -> Any:
        return app.dependency_overrides.get(dependency, dependency)()

    try:
        manager.resume_interrupted(
            resolve(get_session_factory), resolve(get_object_storage), resolve(get_config)
        )
    except Exception as exc:  # a failed resume leaves the jobs retryable; never block startup
        get_logger(__name__).warning("ingest_resume_failed", error=str(exc))
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)
    logger = get_logger(__name__)

    app = FastAPI(
        title="Nepanglish Annotation Harness",
        version="0.1.0",
        summary="Single-annotator annotation harness for a Nepali-English code-switching corpus",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api import (
        asr_models,
        costs,
        episodes,
        export,
        health,
        ingest,
        pots,
        queue,
        segments,
        tasks,
        translit,
        voices,
    )

    for module in (
        health,
        ingest,
        queue,
        tasks,
        segments,
        translit,
        episodes,
        export,
        costs,
        pots,
        asr_models,
        voices,
    ):
        app.include_router(module.router)

    logger.info("app_created", environment=settings.app.environment)
    return app


app = create_app()
