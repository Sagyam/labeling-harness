"""Error files for a model imported before error mining existed (docs/WER-Breakdown.md, step 4).

The rows are derived data, so a model that has its texts has its rows. Gold and val come from
the imported runs -- each clip's reference snapshot and the model's text, as the run scored
them -- and each public set from ``benchmarks/<set>.jsonl`` in the model's folder when the
notebook's per-clip lines were copied there. The same :func:`app.services.error_mining.rows` a
notebook calls, so a backfilled file equals what the notebook would have written.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.models import AsrModel, ModelEvalClip, ModelEvalRun, Segment
from app.models.enums import EVAL_SPLITS
from app.services.error_mining import SET_BY, SETS, read_overlap, rows, unique_ids, write
from app.services.error_store import ERRORS_DIR, ErrorFile, list_files


def _newest_run(session: Session, model: AsrModel, split: str) -> ModelEvalRun | None:
    return session.scalars(
        sa.select(ModelEvalRun)
        .where(ModelEvalRun.model_id == model.id, ModelEvalRun.split == split)
        .order_by(ModelEvalRun.created_at.desc(), ModelEvalRun.id.desc())
        .limit(1)
    ).first()


def _split_clips(session: Session, run: ModelEvalRun) -> list[dict]:
    clips = session.scalars(
        sa.select(ModelEvalClip)
        .options(selectinload(ModelEvalClip.segment).selectinload(Segment.episode))
        .where(ModelEvalClip.run_id == run.id)
    ).all()
    out = [
        {
            "clip_id": c.segment.external_id,
            "group": c.segment.episode.external_id,
            "ref": c.ref_text,
            "hyp": c.hyp_text,
            "overlap_share": c.overlap_share,
        }
        for c in clips
    ]
    return sorted(out, key=lambda c: c["clip_id"])


def _overlap(name: str, places: Sequence[Path]) -> dict[str, float | None]:
    for place in places:
        if (place / f"{name}.parquet").is_file():
            return read_overlap(place / f"{name}.parquet")
    return {}


def _public_clips(lines_path: Path, name: str, places: Sequence[Path]) -> list[dict]:
    lines = [json.loads(x) for x in lines_path.read_text(encoding="utf-8").splitlines() if x]
    shares = _overlap(name, places)
    by = SET_BY.get(name)
    ids = unique_ids(str(line["id"]) for line in lines)
    return [
        {
            "clip_id": line["id"],
            "group": line["group"],
            "ref": line["ref"],
            "hyp": line["hyp"],
            "overlap_share": shares.get(clip_id),
            "by": line.get(by) if by else None,
        }
        for line, clip_id in zip(lines, ids, strict=True)
    ]


def derive_error_files(
    session: Session, model_folder: Path, *, overlap_dirs: Sequence[Path] = ()
) -> list[ErrorFile]:
    """Write ``errors/<set>.parquet`` for every set the model has texts for; returns the files.

    Args:
        session: Open session; nothing is written to it.
        model_folder: ``<models root>/<slug>`` of an imported model.
        overlap_dirs: Where else to look for ``<set>.parquet`` overlap files, after the folder's
            own ``benchmarks/overlap/``. A public set with none is ``unmeasured``.

    Raises:
        ValueError: The folder's model was never imported.
    """
    slug = model_folder.name
    model = session.scalars(sa.select(AsrModel).where(AsrModel.slug == slug)).first()
    if model is None:
        raise ValueError(f"{slug}: not imported; rescan the models first")
    run_name = str(model.card_jsonb.get("run_name") or slug)
    errors = model_folder / ERRORS_DIR
    for split in EVAL_SPLITS:
        run = _newest_run(session, model, split)
        if run is not None and (clips := _split_clips(session, run)):
            write(rows(run_name, split, clips), errors / f"{split}.parquet")
    places = [model_folder / "benchmarks" / "overlap", *overlap_dirs]
    for name in SETS:
        lines = model_folder / "benchmarks" / f"{name}.jsonl"
        if name not in EVAL_SPLITS and lines.is_file():
            write(
                rows(run_name, name, _public_clips(lines, name, places)), errors / f"{name}.parquet"
            )
    return list_files(model_folder)[0]
