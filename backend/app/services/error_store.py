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
import random
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from app.services import attribution
from app.services.error_mining import SETS, MinedFileError, metadata

ERRORS_DIR = "errors"
#: ``model_eval``'s resampling, repeated so this module needs nothing of the database layer: the
#: notebooks carry it in the dataset's ``harness/`` copy. A test holds the two together.
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 0
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


# --- reading -------------------------------------------------------------------------------------
#
# Every number below is a sum over rows, so a block's WER is its errors over its reference words.
# Each query opens its own in-memory DuckDB, reads one path the caller built from validated names,
# and passes every filter value as a parameter.

_ERRORS = ("sub", "del", "ins")
_BUCKETS = ("none", "0-5%", "5-15%", ">15%", "unmeasured")
#: Words either side of an occurrence, in alignment steps.
CONTEXT_STEPS = 5


@dataclass(frozen=True)
class ErrorFilter:
    """Which rows a confusion table or an occurrence list reads; ``None`` is no condition."""

    kind: Sequence[str] | None = None
    forgiven: str | None = None
    ref_script: str | None = None
    hyp_script: str | None = None
    number: bool | None = None
    overlap_bucket: str | None = None
    by: str | None = None
    similarity_min: float | None = None
    similarity_max: float | None = None

    def where(self) -> tuple[str, list[Any]]:
        """The SQL condition and its parameters."""
        terms, params = ["TRUE"], []
        if self.kind:
            terms.append(f"kind IN ({', '.join('?' for _ in self.kind)})")
            params += list(self.kind)
        for column in ("forgiven", "ref_script", "hyp_script", "overlap_bucket", "by"):
            value = getattr(self, column)
            if value is not None:
                terms.append(f'"{column}" = ?')
                params.append(value)
        if self.number is not None:
            terms.append("number = ?")
            params.append(self.number)
        if self.similarity_min is not None:
            terms.append("similarity >= ?")
            params.append(self.similarity_min)
        if self.similarity_max is not None:
            terms.append("similarity <= ?")
            params.append(self.similarity_max)
        return " AND ".join(terms), params


def _query(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    with duckdb.connect() as con:
        cursor = con.execute(sql, list(params))
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _source(path: Path) -> str:
    return "read_parquet('" + str(path).replace("'", "''") + "')"


@dataclass(frozen=True)
class _Clip:
    """One clip's counts: everything a block adds up."""

    clip_id: str
    group: str
    bucket: str
    snr_bucket: str
    by_value: str | None
    words: int
    sub: int
    dele: int
    ins: int
    number_sub: int
    number_del: int
    number_ins: int
    number_words: int
    ref_number: bool
    kinds: tuple[int, ...]

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def number_errors(self) -> int:
        return self.number_sub + self.number_del + self.number_ins


#: A substitution between two Devanagari words this alike is a "similar Nepali word": mostly a
#: suffix (``रहेको``/``रहेका``), the line WER-Breakdown.md's table of 2026-10-01 drew. A tag on
#: the card, never a judgment that the model heard it.
SIMILAR = 0.75
#: Each error's kind for the card (``attribution.KINDS``), as SQL over one row; the first that
#: holds wins, so the kinds partition the errors.
_KIND_SQL = {
    "number": "number",
    "deletion": "kind = 'del'",
    "insertion": "kind = 'ins'",
    "english": "ref_script = 'lat' AND hyp_script = 'lat'",
    "nepali_similar": f"ref_script = 'dev' AND hyp_script = 'dev' AND similarity >= {SIMILAR}",
    "nepali_other": "ref_script = 'dev' AND hyp_script = 'dev'",
    "script": "TRUE",
}


def _kind_case() -> str:
    whens = " ".join(f"WHEN {cond} THEN '{kind}'" for kind, cond in _KIND_SQL.items())
    return f"CASE {whens} END"


def _clips(path: Path) -> dict[str, _Clip]:
    counts = ", ".join(
        f"count(*) FILTER (WHERE error_kind = '{kind}') AS k_{kind}" for kind in attribution.KINDS
    )
    rows = _query(
        f"""
        SELECT clip_id, "group", any_value(overlap_bucket) AS bucket,
               any_value(snr_bucket) AS snr_bucket, any_value("by") AS by_value,
               sum(len(ref)) AS words,
               count_if(kind = 'sub') AS sub, count_if(kind = 'del') AS dele,
               count_if(kind = 'ins') AS ins,
               count_if(number AND kind = 'sub') AS number_sub,
               count_if(number AND kind = 'del') AS number_del,
               count_if(number AND kind = 'ins') AS number_ins,
               coalesce(sum(len(ref)) FILTER (WHERE number), 0) AS number_words,
               bool_or(ref_number) AS ref_number, {counts}
        FROM (
            SELECT *, CASE WHEN kind IN ('sub', 'del', 'ins') THEN {_kind_case()} END
                      AS error_kind
            FROM {_source(path)}
        )
        GROUP BY clip_id, "group" ORDER BY clip_id
        """
    )
    out = {}
    for r in rows:
        kinds = tuple(r.pop(f"k_{kind}") for kind in attribution.KINDS)
        out[r["clip_id"]] = _Clip(**r, kinds=kinds)
    return out


def _card(clips: Sequence[_Clip]) -> dict[str, Any]:
    """The attribution card (D111): points of WER per recording condition, and the rest by
    kind of error."""
    return attribution.card(
        attribution.ClipCounts(
            c.group,
            c.bucket,
            c.snr_bucket,
            c.words,
            dict(zip(attribution.KINDS, c.kinds, strict=True)),
        )
        for c in clips
    )


def _rate(errors: float, words: float) -> float:
    return 100 * errors / words if words else 0.0


def _counts(clips: Sequence[_Clip]) -> dict[str, Any]:
    words = sum(c.words for c in clips)
    sub, dele, ins = (sum(getattr(c, k) for c in clips) for k in ("sub", "dele", "ins"))
    return {
        "clips": len(clips),
        "ref_words": words,
        "errors": sub + dele + ins,
        "wer": _rate(sub + dele + ins, words),
        "sub": _rate(sub, words),
        "del": _rate(dele, words),
        "ins": _rate(ins, words),
    }


def _draws(groups: Sequence[str]) -> list[list[str]]:
    rng = random.Random(BOOTSTRAP_SEED)
    return [rng.choices(groups, k=len(groups)) for _ in range(BOOTSTRAP_ROUNDS)]


def _percentiles(values: list[float]) -> list[float]:
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def _all_words(clip: _Clip) -> tuple[int, int]:
    return clip.errors, clip.words


Measure = Callable[[_Clip], tuple[int, int]]


def _interval(clips: Sequence[_Clip], measure: Measure = _all_words) -> list[float] | None:
    """95% interval of a pooled rate, resampling the set's own unit (``group``)."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for clip in clips:
        e, w = measure(clip)
        totals[clip.group][0] += e
        totals[clip.group][1] += w
    if len(totals) < 2:
        return None
    rates = []
    for draw in _draws(sorted(totals)):
        rates.append(_rate(sum(totals[g][0] for g in draw), sum(totals[g][1] for g in draw)))
    return _percentiles(rates)


def _paired(
    run: Sequence[_Clip], base: Mapping[str, _Clip], measure: Measure = _all_words
) -> dict[str, Any] | None:
    """This run minus the base on the clips both hold: ``[difference, low, high]`` in points,
    the interval from resampling ``group`` (``None`` with fewer than two groups)."""
    shared = [c for c in run if c.clip_id in base]
    if not shared:
        return None
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for clip in shared:
        a, b = measure(clip), measure(base[clip.clip_id])
        t = totals[clip.group]
        t[0], t[1], t[2], t[3] = t[0] + a[0], t[1] + a[1], t[2] + b[0], t[3] + b[1]

    def diff(groups: Sequence[str]) -> float:
        s = [sum(totals[g][k] for g in groups) for k in range(4)]
        return _rate(s[0], s[1]) - _rate(s[2], s[3])

    point = diff(list(totals))
    if len(totals) < 2:
        return {"wer": [point, None, None], "clips": len(shared)}
    low, high = _percentiles([diff(draw) for draw in _draws(sorted(totals))])
    return {"wer": [point, low, high], "clips": len(shared)}


def summary(path: Path) -> dict[str, Any]:
    """A file's headline: clips, reference words, errors, WER and S/D/I per 100 words."""
    return _counts(list(_clips(path).values()))


def _without_numbers(c: _Clip) -> tuple[int, int]:
    return c.errors - c.number_errors, c.words - c.number_words


def _numbers(clips: Sequence[_Clip], base: Mapping[str, _Clip] | None) -> dict[str, Any]:
    errors = sum(c.errors for c in clips)
    with_ref = [c for c in clips if c.ref_number]
    out = {
        "errors": sum(c.number_errors for c in clips),
        "sub": sum(c.number_sub for c in clips),
        "del": sum(c.number_del for c in clips),
        "ins": sum(c.number_ins for c in clips),
        "share_of_errors": sum(c.number_errors for c in clips) / errors if errors else 0.0,
        "wer_without": _rate(*map(sum, zip(*(_without_numbers(c) for c in clips), strict=True)))
        if clips
        else 0.0,
        "ref_clips": len(with_ref),
        "ref_clip_words": sum(c.words for c in with_ref),
        "ref_clip_wer": _rate(sum(c.errors for c in with_ref), sum(c.words for c in with_ref)),
    }
    if base is not None:
        theirs = _numbers([base[c.clip_id] for c in clips if c.clip_id in base], None)
        out["vs_base"] = {
            "errors": out["errors"] - theirs["errors"],
            "share_of_errors": out["share_of_errors"] - theirs["share_of_errors"],
            "wer_without": out["wer_without"] - theirs["wer_without"],
            "wer_without_paired": _paired(clips, base, _without_numbers),
            "ref_clip_wer": _paired(with_ref, base),
        }
    return out


def breakdown(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    """The file's WER with S/D/I, blocks 1 (crosstalk buckets) and 2 (numbers), and the
    attribution card (``attribution``: what each recording condition costs, D111).

    With ``base`` (the base model's file for the same set), the run's WER, each bucket's and the
    number block's carry the difference against it, on the clips both scored, with an interval
    from resampling ``group`` -- the set's own unit, as ``model_eval`` resamples episodes.
    """
    clips = list(_clips(path).values())
    theirs = _clips(base) if base is not None else None
    out = _counts(clips) | {"wer_ci": _interval(clips)}
    if theirs is not None:
        out["vs_base"] = _paired(clips, theirs)
    total_words, total_errors = out["ref_words"], out["errors"]
    buckets = []
    for bucket in _BUCKETS:
        members = [c for c in clips if c.bucket == bucket]
        if not members:
            continue
        entry = {"bucket": bucket} | _counts(members)
        entry["share_of_words"] = entry["ref_words"] / total_words if total_words else 0.0
        entry["share_of_errors"] = entry["errors"] / total_errors if total_errors else 0.0
        entry["wer_ci"] = _interval(members)
        if theirs is not None:
            entry["vs_base"] = _paired(members, theirs)
        buckets.append(entry)
    out["overlap"] = buckets
    out["numbers"] = _numbers(clips, theirs)
    out["attribution"] = _card(clips)
    out["by_values"] = sorted({c.by_value for c in clips if c.by_value is not None})
    return out


def _side(column: str) -> str:
    return f"array_to_string({column}, ' ')"


def confusion(
    path: Path,
    flt: ErrorFilter,
    *,
    both_ways: bool = False,
    base: Path | None = None,
    sort: str = "count",
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Block 3: how often each reference text was written as each model text, by kind.

    Args:
        path: The run's file.
        flt: Which rows count.
        both_ways: A->B and B->A as one row, with ``forward`` and ``backward`` counts.
        base: The base model's file for the set: each row gains ``base_count`` and ``change``.
        sort: ``count`` (most frequent first) or ``change`` (grew most against base first).
        offset: Rows to skip.
        limit: Rows to return.

    Returns:
        ``{total, rows}``; each row ``{kind, ref, hyp, count, share}``, ``share`` being its share
        of the rows of its kind under the same filter.
    """
    where, params = flt.where()
    ref, hyp = _side("ref"), _side("hyp")
    if both_ways:
        key = f"least({ref}, {hyp}) AS a, greatest({ref}, {hyp}) AS b, ({ref} <= {hyp}) AS fwd"
    else:
        key = f"{ref} AS a, {hyp} AS b, TRUE AS fwd"

    def grouped(source: Path) -> str:
        return f"""
            SELECT kind, a, b, count(*) AS n, count_if(fwd) AS forward
            FROM (SELECT kind, {key} FROM {_source(source)} WHERE {where})
            GROUP BY kind, a, b
        """

    base_join = ""
    base_cols = "NULL::BIGINT AS base_count"
    base_params: list[Any] = []
    if base is not None:
        base_join = f"LEFT JOIN ({grouped(base)}) t ON t.kind = r.kind AND t.a = r.a AND t.b = r.b"
        base_cols = "coalesce(t.n, 0) AS base_count"
        base_params = params
    order = "change DESC, r.n DESC" if sort == "change" and base is not None else "r.n DESC"
    rows = _query(
        f"""
        SELECT r.kind, r.a, r.b, r.n, r.forward, {base_cols},
               r.n - coalesce({"t.n" if base is not None else "NULL"}, 0) AS change,
               r.n / sum(r.n) OVER (PARTITION BY r.kind) AS share,
               count(*) OVER () AS total
        FROM ({grouped(path)}) r {base_join}
        ORDER BY {order}, r.kind, r.a, r.b
        LIMIT ? OFFSET ?
        """,
        [*params, *base_params, limit, offset],
    )
    total = rows[0]["total"] if rows else _count_groups(path, grouped, params)
    out = []
    for r in rows:
        row = {"kind": r["kind"], "ref": r["a"], "hyp": r["b"], "count": r["n"],
               "share": float(r["share"])}  # fmt: skip
        if both_ways:
            row |= {"forward": r["forward"], "backward": r["n"] - r["forward"]}
        if base is not None:
            row |= {"base_count": r["base_count"], "change": r["change"]}
        out.append(row)
    return {"total": int(total), "rows": out}


def _count_groups(path: Path, grouped, params: list[Any]) -> int:
    return _query(f"SELECT count(*) AS n FROM ({grouped(path)})", params)[0]["n"]


def _op(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": row["kind"],
        "ref": list(row["ref"]),
        "hyp": list(row["hyp"]),
        "similarity": float(row["similarity"]),
    }


def occurrences(
    path: Path,
    flt: ErrorFilter,
    *,
    ref: str | None = None,
    hyp: str | None = None,
    sample: int | None = None,
    seed: int = 0,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """The pairs behind a filter or a confusion row, each with its context in its clip.

    ``ref``/``hyp`` pick one confusion row (words joined by a space). ``sample`` draws that many
    at random instead of the first in clip order, the same ones for the same ``seed``: this is
    how a class is read. Each row carries up to :data:`CONTEXT_STEPS` alignment steps of its
    clip before and after it.
    """
    where, params = flt.where()
    if ref is not None:
        where += f" AND {_side('ref')} = ?"
        params.append(ref)
    if hyp is not None:
        where += f" AND {_side('hyp')} = ?"
        params.append(hyp)
    if sample is not None:
        order, take = "md5(concat(?, '|', clip_id, '|', pos))", [str(seed), sample, 0]
    else:
        order, take = "clip_id, pos", [limit, offset]
    found = _query(
        f"""
        SELECT clip_id, "group", pos, kind, ref, hyp, similarity, overlap_bucket, "by",
               count(*) OVER () AS total
        FROM {_source(path)} WHERE {where}
        ORDER BY {order} LIMIT ? OFFSET ?
        """,
        [*params, *take],
    )
    if not found:
        total = _query(f"SELECT count(*) AS n FROM {_source(path)} WHERE {where}", params)
        return {"total": total[0]["n"], "rows": []}
    context = _context(path, [(r["clip_id"], r["pos"]) for r in found])
    rows = []
    for r in found:
        steps = context[r["clip_id"]]
        rows.append(
            _op(r)
            | {
                "clip_id": r["clip_id"],
                "group": r["group"],
                "pos": r["pos"],
                "overlap_bucket": r["overlap_bucket"],
                "by": r["by"],
                "before": [
                    steps[p] for p in range(r["pos"] - CONTEXT_STEPS, r["pos"]) if p in steps
                ],
                "after": [
                    steps[p]
                    for p in range(r["pos"] + 1, r["pos"] + 1 + CONTEXT_STEPS)
                    if p in steps
                ],
            }
        )
    return {"total": found[0]["total"], "rows": rows}


def _context(path: Path, at: Sequence[tuple[str, int]]) -> dict[str, dict[int, dict[str, Any]]]:
    """The steps around each ``(clip_id, pos)``, as ``{clip_id: {pos: op}}``."""
    windows = " OR ".join("(clip_id = ? AND pos BETWEEN ? AND ?)" for _ in at)
    params: list[Any] = []
    for clip_id, pos in at:
        params += [clip_id, pos - CONTEXT_STEPS, pos + CONTEXT_STEPS]
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in _query(
        f"SELECT clip_id, pos, kind, ref, hyp, similarity FROM {_source(path)} WHERE {windows}",
        params,
    ):
        out[r["clip_id"]][r["pos"]] = _op(r)
    return out


def clip_ops(path: Path, clip_id: str) -> list[dict[str, Any]]:
    """One clip's steps in order, in the shape the clip panel's aligned diff renders."""
    return [
        _op(r)
        for r in _query(
            f"SELECT kind, ref, hyp, similarity FROM {_source(path)} WHERE clip_id = ? "
            "ORDER BY pos",
            [clip_id],
        )
    ]


def report(path: Path, *, top: int = 20) -> dict[str, Any]:
    """A file's breakdown and the ``top`` rows of each error kind's confusion table: what a
    notebook writes beside a WER, from the same queries the page runs."""
    out = breakdown(path)
    out["top"] = {
        kind: confusion(path, ErrorFilter(kind=[kind]), limit=top)["rows"] for kind in _ERRORS
    }
    return out
