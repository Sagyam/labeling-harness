"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings, load_dotenv
from app.utils.logging import configure_logging, get_logger


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
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api import costs, episodes, export, health, ingest, queue, segments, tasks, translit

    for module in (health, ingest, queue, tasks, segments, translit, episodes, export, costs):
        app.include_router(module.router)

    logger.info("app_created", environment=settings.app.environment)
    return app


app = create_app()
