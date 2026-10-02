"""The Models page's Errors section: a model's error-mining files and what they say
(docs/WER-Breakdown.md).

Every path read here is built from a model slug found in the database and a set name checked
against :data:`app.services.error_mining.SETS`; every filter reaches DuckDB as a parameter.
"""

from __future__ import annotations

from typing import Any, Literal

import sqlalchemy as sa
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_session, require_auth
from app.api.schemas import ErrorClipOut, ErrorFileOut, ErrorFilesOut
from app.config import Settings
from app.models import AsrModel, AuditLog, ModelEvalClip, ModelEvalRun, Segment
from app.services import error_store
from app.services.error_mining import SETS
from app.services.error_store import (
    MAX_FILE_BYTES,
    ErrorFile,
    ErrorFileError,
    ErrorFilter,
    accept_uploads,
)
from app.services.fold import fold_version

router = APIRouter(tags=["models"], dependencies=[Depends(require_auth)])

Kind = Literal["match", "fold", "merge", "sub", "del", "ins"]
Forgiven = Literal["spelling", "script", "number", "merge"]
Script = Literal["dev", "lat", "mix", "none"]
Bucket = Literal["none", "0-5%", "5-15%", ">15%", "unmeasured"]
SnrBucket = Literal["<15 dB", "15-25 dB", "25-35 dB", "35-45 dB", "45+ dB", "unmeasured"]


def _model(session: Session, slug: str) -> AsrModel:
    model = session.scalars(sa.select(AsrModel).where(AsrModel.slug == slug)).first()
    if model is None:
        raise HTTPException(status_code=404, detail=f"model {slug!r} not found")
    return model


def _file(settings: Settings, model: AsrModel, set_name: str, *, role: str = "") -> ErrorFile:
    if set_name not in SETS:
        raise HTTPException(status_code=422, detail=f"unknown set {set_name!r}")
    found = error_store.find_file(settings.models.root / model.slug, set_name)
    if found is None:
        raise HTTPException(
            status_code=404, detail=f"{role}{model.slug!r} has no error file for {set_name}"
        )
    return found


def _base_file(session: Session, settings: Settings, base: str | None, set_name: str):
    if base is None:
        return None
    return _file(settings, _model(session, base), set_name, role="base ").path


def _newest_run(session: Session, model: AsrModel, split: str) -> ModelEvalRun | None:
    return session.scalars(
        sa.select(ModelEvalRun)
        .where(ModelEvalRun.model_id == model.id, ModelEvalRun.split == split)
        .order_by(ModelEvalRun.created_at.desc(), ModelEvalRun.id.desc())
        .limit(1)
    ).first()


def _imported_wer(session: Session, model: AsrModel, set_name: str) -> float | None:
    """The imported run's WER for gold or val: scored against today's labels, where a file was
    scored against the export's. The panel shows both when they differ; neither is wrong."""
    run = _newest_run(session, model, set_name) if set_name in ("gold", "val") else None
    return run.metrics_jsonb.get("wer") if run is not None else None


def _file_out(f: ErrorFile) -> ErrorFileOut:
    return ErrorFileOut(
        set=f.set,
        run=f.run,
        fold_version=f.fold_version,
        miner_version=f.miner_version,
        created_at=f.created_at,
    )


@router.post("/models/{slug}/errors", response_model=list[ErrorFileOut])
async def post_errors(
    slug: str,
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> list[ErrorFileOut]:
    """Store error-mining files beside the model, as ``errors/<set>.parquet``.

    All or nothing: 422 for a file that is not error rows this harness reads, an unknown set, two
    files for one set, files from more than one run, or a file over 50 MB.
    """
    model = _model(session, slug)
    uploads = []
    for upload in files:
        data = await upload.read(MAX_FILE_BYTES + 1)  # one byte over is enough to refuse it
        uploads.append((upload.filename or "upload", data))
    try:
        stored = accept_uploads(settings.models.root / model.slug, uploads)
    except ErrorFileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out = [_file_out(f) for f in stored]
    session.add(
        AuditLog(
            entity_type="asr_model",
            entity_id=model.slug,
            action="model_errors_upload",
            actor=settings.labels.default_annotator,
            new_values_jsonb={"files": [f.model_dump() for f in out]},
        )
    )
    session.commit()
    return out


@router.get("/models/{slug}/errors", response_model=ErrorFilesOut)
def get_errors(
    slug: str, session: Session = Depends(get_session), settings: Settings = Depends(get_config)
) -> ErrorFilesOut:
    """The model's error files with each one's score, and the files it cannot read."""
    model = _model(session, slug)
    found, refused = error_store.list_files(settings.models.root / model.slug)
    current = fold_version()
    files = [
        _file_out(f).model_dump()
        | error_store.summary(f.path)
        | {
            "fold_current": f.fold_version == current,
            "imported_wer": _imported_wer(session, model, f.set),
        }
        for f in found
    ]
    return ErrorFilesOut(files=files, refused=[f"{name}: {why}" for name, why in refused])


@router.get("/models/{slug}/errors/{set_name}/breakdown")
def get_breakdown(
    slug: str,
    set_name: str,
    base: str | None = None,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> dict[str, Any]:
    """WER with S/D/I, block 1 (crosstalk buckets) and block 2 (numbers); with ``base``, each
    against the base model's file for the same set."""
    model = _model(session, slug)
    path = _file(settings, model, set_name).path
    base_path = _base_file(session, settings, base, set_name)
    return error_store.breakdown(path, base=base_path) | {
        "set": set_name,
        "base": base,
        "imported_wer": _imported_wer(session, model, set_name),
    }


def _filter(
    kind: list[Kind] | None = Query(default=None),
    forgiven: Forgiven | None = None,
    ref_script: Script | None = None,
    hyp_script: Script | None = None,
    number: bool | None = None,
    overlap_bucket: Bucket | None = None,
    snr_bucket: SnrBucket | None = None,
    by: str | None = None,
    similarity_min: float | None = Query(default=None, ge=0, le=1),
    similarity_max: float | None = Query(default=None, ge=0, le=1),
) -> ErrorFilter:
    return ErrorFilter(
        kind=kind,
        forgiven=forgiven,
        ref_script=ref_script,
        hyp_script=hyp_script,
        number=number,
        overlap_bucket=overlap_bucket,
        snr_bucket=snr_bucket,
        by=by,
        similarity_min=similarity_min,
        similarity_max=similarity_max,
    )


@router.get("/models/{slug}/errors/{set_name}/confusion")
def get_confusion(
    slug: str,
    set_name: str,
    flt: ErrorFilter = Depends(_filter),
    both_ways: bool = False,
    base: str | None = None,
    sort: Literal["count", "change"] = "count",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> dict[str, Any]:
    """Block 3: how often each reference text was written as each model text, paged."""
    path = _file(settings, _model(session, slug), set_name).path
    base_path = _base_file(session, settings, base, set_name)
    return error_store.confusion(
        path, flt, both_ways=both_ways, base=base_path, sort=sort, offset=offset, limit=limit
    )


@router.get("/models/{slug}/errors/{set_name}/pairs")
def get_pairs(
    slug: str,
    set_name: str,
    flt: ErrorFilter = Depends(_filter),
    ref: str | None = None,
    hyp: str | None = None,
    sample: int | None = Query(default=None, ge=1, le=500),
    seed: int = 0,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> dict[str, Any]:
    """The occurrences behind a filter or one confusion row, with context; ``sample`` draws at
    random, the same ones for the same ``seed``."""
    path = _file(settings, _model(session, slug), set_name).path
    return error_store.occurrences(
        path, flt, ref=ref, hyp=hyp, sample=sample, seed=seed, offset=offset, limit=limit
    )


@router.get("/models/{slug}/errors/{set_name}/clips/{clip_id}", response_model=ErrorClipOut)
def get_error_clip(
    slug: str,
    set_name: str,
    clip_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> ErrorClipOut:
    """One clip's alignment from the file. On gold and val, also the imported run and segment
    that open it in the clip panel, with its audio; a public set's audio is not held here."""
    model = _model(session, slug)
    ops = error_store.clip_ops(_file(settings, model, set_name).path, clip_id)
    if not ops:
        raise HTTPException(status_code=404, detail=f"{clip_id!r} is not in {set_name}")
    run_id = segment_id = None
    run = _newest_run(session, model, set_name) if set_name in ("gold", "val") else None
    if run is not None:
        segment_id = session.scalar(
            sa.select(ModelEvalClip.segment_id)
            .join(Segment, Segment.id == ModelEvalClip.segment_id)
            .where(ModelEvalClip.run_id == run.id, Segment.external_id == clip_id)
        )
        run_id = run.id if segment_id is not None else None
    return ErrorClipOut(clip_id=clip_id, ops=ops, run_id=run_id, segment_id=segment_id)
