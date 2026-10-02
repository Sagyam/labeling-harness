"""A model's error files, and the breakdown read from them (docs/WER-Breakdown.md).

Error-mining rows (:mod:`app.services.error_mining`) live beside the model they describe, in the
folder the Models page already reads (D83)::

    data/models/asr/<slug>/errors/<set>.parquet

one file per set, every file of a folder from one run. They arrive by upload, or are already
there when the folder was copied from the hub, or are derived from the imported runs by
``scripts/mine_errors.py``. No table holds them: they are derived, written once and only ever
grouped, so DuckDB reads them where they lie (docs/decisions.md).

A stored file's name always comes from the set its metadata names, never from the name it was
uploaded under, so nothing a client sends becomes part of a path.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.services.error_mining import SETS, MinedFileError, metadata

ERRORS_DIR = "errors"
#: The largest file an upload may carry. A run's biggest set is a few MB.
MAX_FILE_BYTES = 50 * 1024 * 1024


class ErrorFileError(ValueError):
    """An upload that cannot be stored; nothing of it was."""


@dataclass(frozen=True)
class ErrorFile:
    """One stored file and what its metadata says."""

    set: str
    run: str
    path: Path
    fold_version: str
    miner_version: str
    created_at: str


def _described(path: Path) -> ErrorFile:
    meta = metadata(path)
    return ErrorFile(
        set=meta["set"],
        run=meta["run"],
        path=path,
        fold_version=meta["fold_version"],
        miner_version=meta["miner_version"],
        created_at=meta["created_at"],
    )


def list_files(model_folder: Path) -> tuple[list[ErrorFile], list[tuple[str, str]]]:
    """The model's error files in :data:`SETS` order, and those it cannot read with why.

    A file is set aside when it is not rows this version reads, or when its name is not the set
    its metadata names (``val.parquet`` holding gold rows would be read as val).
    """
    folder = model_folder / ERRORS_DIR
    if not folder.is_dir():
        return [], []
    found: list[ErrorFile] = []
    refused: list[tuple[str, str]] = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            described = _described(path)
        except MinedFileError as exc:
            refused.append((path.name, str(exc)))
            continue
        if path.stem != described.set:
            refused.append((path.name, f"holds {described.set} rows, not {path.stem}"))
            continue
        found.append(described)
    found.sort(key=lambda f: SETS.index(f.set))
    return found, refused


def find_file(model_folder: Path, set_name: str) -> ErrorFile | None:
    """The model's file for one set, or ``None``. ``set_name`` must be one of :data:`SETS`."""
    if set_name not in SETS:
        raise ValueError(f"unknown set {set_name!r}")
    return next((f for f in list_files(model_folder)[0] if f.set == set_name), None)


def accept_uploads(model_folder: Path, uploads: Sequence[tuple[str, bytes]]) -> list[ErrorFile]:
    """Store uploaded error files as ``errors/<set>.parquet``, all or none.

    Each file is checked as :func:`app.services.error_mining.metadata` checks it, and the upload
    is refused whole on any bad file, on two files for one set, or when the folder would then
    hold files from more than one run. A file for a set already present replaces it.

    Args:
        model_folder: ``<models root>/<slug>``.
        uploads: ``(name as uploaded, content)``; the name is only quoted back in errors.

    Raises:
        ErrorFileError: With what is wrong; nothing was stored.
    """
    if not uploads:
        raise ErrorFileError("no file uploaded")
    folder = model_folder / ERRORS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []  # removed on the way out unless moved into place
    described: list[ErrorFile] = []
    try:
        for name, data in uploads:
            if len(data) > MAX_FILE_BYTES:
                raise ErrorFileError(f"{name}: larger than {MAX_FILE_BYTES // 2**20} MB")
            path = folder / f".upload-{uuid.uuid4().hex}.parquet"
            staged.append(path)
            path.write_bytes(data)
            try:
                described.append(_described(path))
            except MinedFileError as exc:
                raise ErrorFileError(f"{name}: {exc}") from exc
        sets = [d.set for d in described]
        repeated = sorted({s for s in sets if sets.count(s) > 1})
        if repeated:
            raise ErrorFileError(f"two files for {', '.join(repeated)} in one upload")
        runs = {d.run for d in described}
        if len(runs) > 1:
            raise ErrorFileError(f"one upload holds one run, not {sorted(runs)}")
        kept = {f.run for f in list_files(model_folder)[0] if f.set not in sets}
        if kept - runs:
            raise ErrorFileError(
                f"this model's other files are from {sorted(kept)}, not {sorted(runs)}: upload "
                "every set of the new run together"
            )
        stored = []
        for d in described:
            target = folder / f"{d.set}.parquet"
            os.replace(d.path, target)
            stored.append(ErrorFile(**{**vars(d), "path": target}))
        return sorted(stored, key=lambda f: SETS.index(f.set))
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
