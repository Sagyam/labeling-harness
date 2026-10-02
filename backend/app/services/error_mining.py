"""Error mining: every aligned word pair of an evaluation, classified and kept (WER-Breakdown.md).

A WER is read once; the pairs behind it are what explain it. This module turns a clip's
reference and model text into one row per step of :func:`app.services.fold.align` -- matches
included, since they are the denominators -- and says what kind of step each is. It decides
nothing: folded WER is whatever ``fold.py`` says, and every row here only describes a pair.

The rows are written to Parquet, one file per run and set, and read through DuckDB: they are
derived, written once and only ever grouped, so they are files beside the model rather than
tables (docs/decisions.md). :func:`write` and :func:`read` are the format's only writer and
reader, so the notebooks and the harness cannot drift apart.

Pure, and importing only ``fold.py``, the standard library and DuckDB: the dataset's
``harness/`` copy carries this file beside ``fold.py``, so a notebook writes exactly the rows
the Models page reads. Keep it importable on Python 3.10+.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb

from app.services.fold import (
    Alignment,
    AlignOp,
    fold_version,
    is_number,
    romanized,
    same_number,
    script,
    word_errors,
)

#: Bump whenever a column's meaning changes: a file is only comparable with another of the same.
MINER_VERSION = "mine-v1"

#: What a file may hold: the corpus's two scored splits and the public sets of
#: ``notebooks/src/evalkit.py``'s ``BENCHMARKS``, in its order.
SETS = ("gold", "val", "fleurs", "slr54", "common_voice", "indicvoices", "nepali_cs")
#: A public set's own split column (``evalkit.Benchmark.by``), carried in each row's ``by``.
SET_BY = {"indicvoices": "scenario", "nepali_cs": "speech_type"}

#: Every column, in file order, with its DuckDB type.
COLUMNS: dict[str, str] = {
    "run": "VARCHAR",
    "set": "VARCHAR",
    "clip_id": "VARCHAR",
    "group": "VARCHAR",
    "pos": "INTEGER",
    "ref": "VARCHAR[]",
    "hyp": "VARCHAR[]",
    "kind": "VARCHAR",
    "identical": "BOOLEAN",
    "forgiven": "VARCHAR",
    "ref_script": "VARCHAR",
    "hyp_script": "VARCHAR",
    "number": "BOOLEAN",
    "ref_number": "BOOLEAN",
    "similarity": "DOUBLE",
    "ref_roman": "VARCHAR",
    "hyp_roman": "VARCHAR",
    "overlap_share": "DOUBLE",
    "overlap_bucket": "VARCHAR",
    "by": "VARCHAR",
}
#: The key-value metadata every file carries.
METADATA = ("run", "set", "fold_version", "miner_version", "created_at")


class MinedFileError(ValueError):
    """A file that is not error-mining rows this version can read."""


def overlap_bucket(share: float | None) -> str:
    """``clip_classes.overlap_bucket``, repeated so this module needs nothing but ``fold.py``;
    a test holds the two together."""
    if share is None:
        return "unmeasured"
    if share <= 0:
        return "none"
    if share < 0.05:
        return "0-5%"
    if share <= 0.15:
        return "5-15%"
    return ">15%"


#: ``clip_classes``'s SNR edges (D87), lowest first; at or above the last is ``45+ dB``.
SNR_EDGES = ((15, "<15 dB"), (25, "15-25 dB"), (35, "25-35 dB"), (45, "35-45 dB"))
#: Every SNR bucket in order, the baseline (``45+ dB``, where most of the corpus is) last but one.
SNR_BUCKETS = (*(b for _, b in SNR_EDGES), "45+ dB", "unmeasured")


def snr_bucket(db: float | None) -> str:
    """``clip_classes``'s speech-to-noise bucket, repeated so this module needs nothing but
    ``fold.py``; a test holds the two together."""
    if db is None:
        return "unmeasured"
    for edge, bucket in SNR_EDGES:
        if db < edge:
            return bucket
    return "45+ dB"


def _side_script(words: Sequence[str]) -> str | None:
    """One side's script: ``dev``, ``lat``, ``mix`` (both, in one word or across a merge's
    words) or ``none`` (digits only); ``None`` for the absent side of a deletion or insertion."""
    if not words:
        return None
    found = {script(w) for w in words} - {"none"}
    if not found:
        return "none"
    return found.pop() if len(found) == 1 else "mix"


def _forgiven(op: AlignOp, identical: bool) -> str | None:
    if op.kind == "match":
        return None if identical else "spelling"
    if op.kind == "fold":
        return "number" if same_number(op.ref[0], op.hyp[0]) else "script"
    if op.kind == "merge":
        return "merge"
    return None


def _row(pos: int, op: AlignOp) -> dict[str, Any]:
    identical = op.kind == "match" and op.ref == op.hyp
    return {
        "pos": pos,
        "ref": list(op.ref),
        "hyp": list(op.hyp),
        "kind": op.kind,
        "identical": identical,
        "forgiven": _forgiven(op, identical),
        "ref_script": _side_script(op.ref),
        "hyp_script": _side_script(op.hyp),
        "number": any(is_number(w) for w in (*op.ref, *op.hyp)),
        "ref_number": any(is_number(w) for w in op.ref),
        "similarity": float(op.similarity),
        "ref_roman": " ".join(romanized(w) for w in op.ref),
        "hyp_roman": " ".join(romanized(w) for w in op.hyp),
    }


def pairs(
    ref_text: str | None, hyp_text: str | None, *, alignment: Alignment | None = None
) -> list[dict[str, Any]]:
    """One row per step of the clip's folded alignment, in order.

    Args:
        ref_text: The reference transcript.
        hyp_text: The model's transcript.
        alignment: ``fold.word_errors(ref_text, hyp_text)`` when the caller already has it (the
            scorer does); the texts are then not aligned again.
    """
    if alignment is None:
        alignment = word_errors(ref_text, hyp_text)
    return [_row(pos, op) for pos, op in enumerate(alignment.ops)]


def number_id(clip_id: str, seen: dict[str, int]) -> str:
    """``clip_id``, or ``<id>#2``, ``#3``, ... when ``seen`` (updated here) already holds it.

    A set whose id columns are empty on some clips (nepali_cs: three clips are ``-None``) still
    gets one id per clip, and two runs over the same set in the same order name each alike.
    """
    seen[clip_id] = seen.get(clip_id, 0) + 1
    return clip_id if seen[clip_id] == 1 else f"{clip_id}#{seen[clip_id]}"


def unique_ids(ids: Iterable[str]) -> Iterator[str]:
    """Every id of a set in order, numbered by :func:`number_id`."""
    seen: dict[str, int] = {}
    for clip_id in ids:
        yield number_id(clip_id, seen)


def rows(run: str, set: str, clips: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every pair of every clip of one run on one set, with the clip's columns added.

    Args:
        run: The run's name (``vanilla-s1``).
        set: ``gold``, ``val`` or a public set's name (:data:`SETS`).
        clips: ``{clip_id, group, ref, hyp, overlap_share, by}`` each; ``by`` (the set's own
            split value) may be absent, and so may ``alignment``, an existing
            ``fold.word_errors(ref, hyp)``. ``overlap_share`` is ``None`` when never measured.
            A ``clip_id`` seen before in ``clips`` is numbered by :func:`number_id`.
    """
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for clip in clips:
        share = clip.get("overlap_share")
        share = None if share is None else float(share)
        clip_id = number_id(str(clip["clip_id"]), seen)
        head = {
            "run": run,
            "set": set,
            "clip_id": clip_id,
            "group": str(clip["group"]),
        }
        tail = {
            "overlap_share": share,
            "overlap_bucket": overlap_bucket(share),
            "by": None if clip.get("by") is None else str(clip["by"]),
        }
        for pair in pairs(clip["ref"], clip["hyp"], alignment=clip.get("alignment")):
            out.append({**head, **pair, **tail})
    return out


# --- the file ------------------------------------------------------------------------------------


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _select_columns(source: str) -> str:
    return ", ".join(f'CAST("{name}" AS {kind}) AS "{name}"' for name, kind in COLUMNS.items()) + (
        f" FROM {source}"
    )


def write(found: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    """Write one run's rows on one set to ``path`` as Parquet, with its metadata.

    Written beside ``path`` and renamed into place, so a reader never sees half a file.

    Raises:
        ValueError: No rows, or rows from more than one run or set.
    """
    if not found:
        raise ValueError("no rows to write: a file holds at least one pair")
    keys = {(r["run"], r["set"]) for r in found}
    if len(keys) != 1:
        raise ValueError(f"a file holds one run and one set, not {sorted(keys)}")
    ((run, set_name),) = keys
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp.jsonl")
    partial = path.with_name(path.name + ".tmp")
    meta = {
        "run": run,
        "set": set_name,
        "fold_version": fold_version(),
        "miner_version": MINER_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
    }
    try:
        with staged.open("w", encoding="utf-8") as fh:
            for row in found:
                fh.write(json.dumps({name: row[name] for name in COLUMNS}, ensure_ascii=False))
                fh.write("\n")
        columns = "{" + ", ".join(f"{_sql_string(n)}: '{t}'" for n, t in COLUMNS.items()) + "}"
        source = (
            f"read_json({_sql_string(str(staged))}, format='newline_delimited', columns={columns})"
        )
        kv = "{" + ", ".join(f"{k}: {_sql_string(v)}" for k, v in meta.items()) + "}"
        with duckdb.connect() as con:
            con.execute(
                f"COPY (SELECT {_select_columns(source)}) TO {_sql_string(str(partial))} "
                f"(FORMAT parquet, COMPRESSION zstd, KV_METADATA {kv})"
            )
        os.replace(partial, path)
    finally:
        staged.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def metadata(path: Path | str) -> dict[str, str]:
    """A file's metadata, after checking it is rows this version reads: every column, every
    metadata key, a known set, and this :data:`MINER_VERSION`.

    Raises:
        MinedFileError: With what is wrong.
    """
    source = _sql_string(str(path))
    try:
        with duckdb.connect() as con:
            kv = con.execute(f"SELECT key, value FROM parquet_kv_metadata({source})").fetchall()
            describe = f"DESCRIBE SELECT * FROM read_parquet({source})"
            names = [r[0] for r in con.execute(describe).fetchall()]
    except duckdb.Error as exc:
        raise MinedFileError(f"{Path(path).name}: not a Parquet file ({exc})") from exc
    meta = {_text(k): _text(v) for k, v in kv}
    missing = [n for n in COLUMNS if n not in names]
    if missing:
        raise MinedFileError(f"{Path(path).name}: no column {', '.join(missing)}")
    absent = [k for k in METADATA if k not in meta]
    if absent:
        raise MinedFileError(f"{Path(path).name}: no metadata {', '.join(absent)}")
    if meta["miner_version"] != MINER_VERSION:
        raise MinedFileError(
            f"{Path(path).name}: written by {meta['miner_version']}, and this harness reads "
            f"{MINER_VERSION}: derive the file again"
        )
    if meta["set"] not in SETS:
        raise MinedFileError(f"{Path(path).name}: unknown set {meta['set']!r}")
    return {k: meta[k] for k in METADATA}


def read_overlap(path: Path | str) -> dict[str, float | None]:
    """A public set's measured crosstalk (``benchmarks/overlap/<set>.parquet``, written by
    :mod:`app.services.benchmark_overlap`) as ``{clip_id: overlap_share}``."""
    with duckdb.connect() as con:
        found = con.execute(
            f"SELECT clip_id, overlap_share FROM read_parquet({_sql_string(str(path))})"
        ).fetchall()
    return dict(found)


def read_snr(path: Path | str) -> dict[str, float | None]:
    """A public set's measured speech-to-noise ratio (``benchmarks/acoustics/<set>.parquet``,
    written by :mod:`app.services.benchmark_overlap`) as ``{clip_id: snr_db}``."""
    with duckdb.connect() as con:
        found = con.execute(
            f"SELECT clip_id, snr_db FROM read_parquet({_sql_string(str(path))})"
        ).fetchall()
    return dict(found)


def _text(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def read(path: Path | str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """A file's metadata (checked as :func:`metadata` does) and its rows, in file order."""
    meta = metadata(path)
    with duckdb.connect() as con:
        cursor = con.execute(f"SELECT {_select_columns(f'read_parquet({_sql_string(str(path))})')}")
        names = [d[0] for d in cursor.description]
        found = [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]
    return meta, found
