"""Dataset export endpoints."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_session, require_auth
from app.api.schemas import ExportHistoryItem, ExportIn, ExportOut, ExportOutItem
from app.config import Settings
from app.services.export import EXPORT_KINDS, ExportError, export_dataset

router = APIRouter(tags=["export"], dependencies=[Depends(require_auth)])


@router.post("/export", response_model=ExportOut)
def run_export(
    body: ExportIn,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_config),
) -> ExportOut:
    """Generate reproducible JSONL and manifest export files for one or all kinds."""
    if body.kind != "all" and body.kind not in EXPORT_KINDS:
        detail_msg = (
            f"Unknown export kind {body.kind!r}. Choose from {sorted(EXPORT_KINDS)} or 'all'."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail_msg,
        )

    kinds = sorted(EXPORT_KINDS) if body.kind == "all" else [body.kind]
    results: list[ExportOutItem] = []

    for kind in kinds:
        try:
            res = export_dataset(
                session,
                kind=kind,
                label_version=body.label_version,
                episode=body.episode,
                settings=settings,
            )
        except ExportError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        manifest_data: dict = {}
        if res.manifest_path.is_file():
            with contextlib.suppress(Exception):
                manifest_data = json.loads(res.manifest_path.read_text(encoding="utf-8"))

        results.append(
            ExportOutItem(
                kind=kind,
                row_count=res.row_count,
                row_counts_by_split=res.row_counts_by_split,
                data_filename=res.data_path.name,
                manifest_filename=res.manifest_path.name,
                download_url=f"/export/download/{kind}/{res.data_path.name}",
                manifest_url=f"/export/download/{kind}/{res.manifest_path.name}",
                manifest=manifest_data,
            )
        )

    return ExportOut(results=results)


@router.get("/export/download/{kind}/{filename}")
def download_export_file(
    kind: str,
    filename: str,
    settings: Settings = Depends(get_config),
) -> FileResponse:
    """Download a generated export JSONL or manifest file."""
    if kind not in EXPORT_KINDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown export kind {kind!r}",
        )

    allowed_filenames = {f"{kind}.jsonl", "manifest.json"}
    if filename not in allowed_filenames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filename {filename!r} for kind {kind!r}",
        )

    file_path = Path(settings.export.output_root) / kind / filename
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {filename!r} has not been exported yet. Generate an export first.",
        )

    media_type = "application/jsonl" if filename.endswith(".jsonl") else "application/json"
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )


@router.get("/export/history", response_model=list[ExportHistoryItem])
def list_export_history(
    settings: Settings = Depends(get_config),
) -> list[ExportHistoryItem]:
    """Inspect the exports directory for previously generated exports."""
    history: list[ExportHistoryItem] = []
    output_root = Path(settings.export.output_root)

    for kind in sorted(EXPORT_KINDS):
        kind_dir = output_root / kind
        data_path = kind_dir / f"{kind}.jsonl"
        manifest_path = kind_dir / "manifest.json"

        if manifest_path.is_file() and data_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                history.append(
                    ExportHistoryItem(
                        kind=kind,
                        data_filename=data_path.name,
                        manifest_filename=manifest_path.name,
                        download_url=f"/export/download/{kind}/{data_path.name}",
                        manifest_url=f"/export/download/{kind}/{manifest_path.name}",
                        row_count=manifest.get("row_count", 0),
                        row_counts_by_split=manifest.get("row_counts_by_split", {}),
                        exported_at=manifest.get("exported_at"),
                        file_bytes=data_path.stat().st_size,
                    )
                )
            except Exception:
                continue

    return history
