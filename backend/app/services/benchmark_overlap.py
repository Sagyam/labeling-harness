"""Crosstalk on the public sets, measured once (docs/WER-Breakdown.md, step 3).

A set described as single-speaker read speech is checked, not assumed: every clip goes through
the detector ingest uses (:mod:`app.services.overlap`, D77). A public set's audio never changes,
so each is measured once and kept as ``benchmarks/overlap/<set>.parquet`` -- ``clip_id``,
``duration``, ``overlap_share`` and ``spans`` -- which the notebooks and the harness both read
through :func:`app.services.error_mining.read_overlap`.

Clips arrive one at a time and their audio is dropped once measured; a run that stops resumes
from the clips already written (``scripts/measure_benchmark_overlap.py``).
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

import duckdb
import numpy as np

from app.services.clip_classes import overlap_share
from app.services.error_mining import number_id
from app.services.overlap import SAMPLE_RATE

#: The file's columns and their DuckDB types.
COLUMNS = {
    "clip_id": "VARCHAR",
    "duration": "DOUBLE",
    "overlap_share": "DOUBLE",
    "spans": "DOUBLE[][]",
}


class Detector(Protocol):
    available: bool

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]] | None: ...


def measure_rows(
    detector: Detector, rows: Iterable[dict[str, Any]], *, done: Container[str] = ()
) -> Iterator[dict[str, Any]]:
    """Measure each clip's crosstalk, in order, skipping the ones in ``done``.

    Args:
        detector: An :class:`app.services.overlap.OverlapDetector`.
        rows: ``evalkit.benchmark_row`` rows: ``segment_id`` and ``audio``, 16 kHz mono int16.
        done: Clip ids already measured (a resumed run).

    Yields:
        ``{clip_id, duration, overlap_share, spans}``. A repeated ``segment_id`` is numbered as
        :func:`~app.services.error_mining.number_id` numbers it, so it joins the error rows.

    Raises:
        RuntimeError: No model is loaded: an unmeasured clip must never be written as clean.
    """
    if not detector.available:
        raise RuntimeError("no overlap model loaded: see HARNESS_OVERLAP_NO_DOWNLOAD")
    seen: dict[str, int] = {}
    for row in rows:
        clip_id = number_id(str(row["segment_id"]), seen)
        if clip_id in done:
            continue
        audio = np.asarray(row["audio"], dtype=np.float32) / 32768.0
        duration = len(audio) / SAMPLE_RATE
        spans = detector.detect(audio, SAMPLE_RATE) or []
        yield {
            "clip_id": clip_id,
            "duration": duration,
            "overlap_share": overlap_share(spans, duration),
            "spans": [[float(start), float(end)] for start, end in spans],
        }


def write_overlap(partial: Path, path: Path) -> None:
    """Turn a run's measured lines (JSON, one clip each) into the set's Parquet file."""
    columns = "{" + ", ".join(f"'{name}': '{kind}'" for name, kind in COLUMNS.items()) + "}"
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    source, target = (str(p).replace("'", "''") for p in (partial, tmp))
    with duckdb.connect() as con:
        con.execute(
            f"COPY (SELECT * FROM read_json('{source}', format='newline_delimited', "
            f"columns={columns})) TO '{target}' (FORMAT parquet, COMPRESSION zstd)"
        )
    tmp.replace(path)
