"""The Models page: fine-tuned models, their runs, each run's clips worst first (D83), and a
playground that transcribes a recording with a model on the CPU (D85)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import sqlalchemy as sa
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_config, get_session, require_auth
from app.api.schemas import (
    AsrModelOut,
    ClassAxisOut,
    ModelClipDetailOut,
    ModelClipPageOut,
    ModelEvalRunOut,
    ModelRescanOut,
    PlaygroundOut,
)
from app.config import Settings
from app.llm.base import LlmError
from app.models import AsrModel, ModelEvalRun
from app.services.clip_classes import AXES, AXIS_BY_NAME
from app.services.model_browse import ClipFilter, ClipSort, clip_detail, list_run_clips
from app.services.model_import import ModelImportError, scan_models
from app.services.playground import PlaygroundError, cpu_weights
from app.services.playground import transcribe as playground_transcribe

router = APIRouter(tags=["models"], dependencies=[Depends(require_auth)])


def _model_out(model: AsrModel, root: Path) -> AsrModelOut:
    return AsrModelOut(
        id=model.id,
        slug=model.slug,
        name=model.name,
        description=model.description,
        architecture=model.architecture,
        trained_at=model.trained_at,
        card=model.card_jsonb,
        runs=[
            ModelEvalRunOut(
                id=run.id,
                split=run.split,
                decoder=run.decoder,
                fold_version=run.fold_version,
                clip_count=run.clip_count,
                metrics=run.metrics_jsonb,
                source=run.source,
                created_at=run.created_at,
            )
            for run in model.runs
        ],
        playground=cpu_weights(root / model.slug, model.card_jsonb),
    )


def _get_run(session: Session, run_id: int) -> ModelEvalRun:
    run = session.get(ModelEvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run


@router.get("/models", response_model=list[AsrModelOut])
def get_models(
    session: Session = Depends(get_session), settings: Settings = Depends(get_config)
) -> list[AsrModelOut]:
    """Every imported model, newest trained first."""
    models = session.scalars(
        sa.select(AsrModel)
        .options(selectinload(AsrModel.runs))
        .order_by(AsrModel.trained_at.desc().nulls_last(), AsrModel.slug)
    )
    return [_model_out(model, settings.models.root) for model in models]


@router.get("/model-classes", response_model=list[ClassAxisOut])
def get_model_classes() -> list[ClassAxisOut]:
    """The axes every run is broken down by, in display order (D87)."""
    return [
        ClassAxisOut(
            name=a.name,
            label=a.label,
            buckets=list(a.buckets),
            baseline=a.baseline,
            unmeasured=a.unmeasured,
            descriptive=a.descriptive,
        )
        for a in AXES
    ]


@router.get("/models/{slug}", response_model=AsrModelOut)
def get_model(
    slug: str, session: Session = Depends(get_session), settings: Settings = Depends(get_config)
) -> AsrModelOut:
    return _model_out(_get_model(session, slug), settings.models.root)


def _get_model(session: Session, slug: str) -> AsrModel:
    model = session.scalars(
        sa.select(AsrModel).options(selectinload(AsrModel.runs)).where(AsrModel.slug == slug)
    ).first()
    if model is None:
        raise HTTPException(status_code=404, detail=f"model {slug!r} not found")
    return model


@router.post("/models/{slug}/transcribe", response_model=PlaygroundOut)
async def post_transcribe(
    slug: str,
    audio: UploadFile = File(...),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> PlaygroundOut:
    """Transcribe one recording with this model on the CPU, in the playground sidecar (D85).

    409 when no CPU weights sit in the model's folder, 422 for a recording that is empty, too
    long or not audio, 502 when the sidecar is down or fails. The ``llm_requests`` row is kept
    either way.
    """
    model = _get_model(session, slug)
    weights = cpu_weights(settings.models.root / slug, model.card_jsonb)
    if weights is None:
        raise HTTPException(
            status_code=409,
            detail=f"no CPU weights for {slug!r}: copy the notebook's cpu/ (or best/) folder into "
            f"{settings.models.root / slug}/",
        )
    data = await audio.read()
    try:
        result = playground_transcribe(session, slug, weights, data, filename=audio.filename)
    except PlaygroundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LlmError as exc:
        session.commit()  # the failed attempt's llm_requests row outlives the error
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PlaygroundOut(
        text=result.text,
        audio_s=result.audio_s,
        latency_ms=result.latency_ms,
        weights=result.weights,
        dry_run=result.dry_run,
        raw=result.raw,
    )


@router.post("/models/rescan", response_model=ModelRescanOut)
def post_rescan(
    session: Session = Depends(get_session), settings: Settings = Depends(get_config)
) -> ModelRescanOut:
    """Import every model folder under ``models.root``; files already imported are skipped.

    All or nothing: a malformed folder is a 422 and nothing from this rescan is kept.
    """
    try:
        report = scan_models(
            session,
            settings.models.root,
            settings=settings,
            actor=settings.labels.default_annotator,
        )
    except ModelImportError as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session.commit()
    return ModelRescanOut(
        models=report.models,
        runs_created=report.runs_created,
        runs_unchanged=report.runs_unchanged,
        skipped=report.skipped,
    )


@router.get("/model-runs/{run_id}/clips", response_model=ModelClipPageOut)
def get_run_clips(
    run_id: int,
    sort: ClipSort = "errors",
    order: Literal["asc", "desc"] = "desc",
    genre: str | None = None,
    overlap: Literal["none", "0-5%", "5-15%", ">15%", "unmeasured"] | None = None,
    loops_only: bool = False,
    min_errors: int = Query(default=0, ge=0),
    class_axis: str | None = None,
    class_bucket: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> ModelClipPageOut:
    """A run's clips, worst first by default. ``class_axis`` with ``class_bucket`` keeps one
    bucket of one class breakdown (D87); an axis the harness does not know is a 422."""
    _get_run(session, run_id)
    if class_axis is not None and class_axis not in AXIS_BY_NAME:
        raise HTTPException(status_code=422, detail=f"unknown class axis {class_axis!r}")
    total, rows = list_run_clips(
        session,
        run_id,
        sort=sort,
        descending=order == "desc",
        where=ClipFilter(
            genre=genre,
            overlap=overlap,
            loops_only=loops_only,
            min_errors=min_errors,
            class_axis=class_axis,
            class_bucket=class_bucket,
        ),
        offset=offset,
        limit=limit,
    )
    return ModelClipPageOut(total=total, offset=offset, limit=limit, rows=rows)


@router.get("/model-runs/{run_id}/clips/{segment_id}", response_model=ModelClipDetailOut)
def get_run_clip(
    run_id: int, segment_id: int, session: Session = Depends(get_session)
) -> ModelClipDetailOut:
    """One clip with the folded alignment behind its counts."""
    run = _get_run(session, run_id)
    detail = clip_detail(session, run_id, segment_id, run_fold_version=run.fold_version)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"segment {segment_id} is not in run {run_id}")
    return ModelClipDetailOut(**detail)
