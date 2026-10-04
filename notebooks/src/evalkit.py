"""One evaluation for every model (D105).

Whatever was trained -- Flex, a blend of its weights, a student at any stage -- is scored by the
functions here, so a number in one notebook means what it means in another:

  * gold and val, folded and raw, split into substitutions, deletions and insertions, overall and
    per clip class, and paired clip by clip against named reference runs with episodes resampled
    (`evaluate_run`);
  * the public Nepali sets, folded, raw and plain, paired against base Flex with each set's own
    unit resampled (`run_benchmarks`).

Every split and set also keeps its errors: each aligned word pair, classified, in
`harness/errors/<set>.parquet` with each clip's crosstalk and SNR, and a `breakdown` beside its
WER (crosstalk buckets, numbers, the
most common substitutions, deletions and insertions; D110). Both come from the
harness's own `error_mining.py` and `error_store.py`, which the dataset's `harness/` copy carries
beside fold.py, so the Models page reads exactly what the notebook wrote.

The notebook hands in what is model-specific: `decode`, its batched decoder, and `score`,
`ftkit.harness_scorer`, which is the harness's own fold.py. No torch here, so the rules are tested
from backend/tests. Keep it importable on Python 3.10+.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
import unicodedata
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import distill
import sweep

SR = 16_000
#: Scored in this order: gold first, so a run stopped while scoring has the held-out number.
SPLITS = ("gold", "val")
BUCKETS = ("none", "0-5%", "5-15%", ">15%")
#: The clip classes every paired difference is also split by (D87).
REPORT_KEYS = ("overlap", "snr", "speakers", "cmi", "duration")
#: What a result row keeps of a split's metrics.
KEEP = ("wer", "raw_wer", "cer", "sub", "del", "ins", "loops", "clips", "rtf", "vs")

#: `decode(rows)` -> texts, per-clip compute seconds, and the loop retries as
#: (segment_id, first decode, retried decode).
Decode = Callable[[Sequence[dict]], tuple[list[str], list[float], list[tuple[str, str, str]]]]
Counts = Mapping[str, Sequence[int]]


@contextmanager
def timed(label: str) -> Iterator[None]:
    """`label...` before a step and `label: done in N s` after it: a stalled step names itself."""
    print(f"{label}...", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"{label}: done in {time.perf_counter() - t0:.0f} s", flush=True)


def duration(row: dict) -> float:
    return row["end_time"] - row["start_time"]


# --- provenance ----------------------------------------------------------------------------------


def check_export(manifest: Mapping[str, Any], expected: str) -> None:
    """Refuse a dataset that is not the export every model in the comparison trains on.

    `expected` is the start of the training manifest's `exported_at` ("2026-09-30"). A date, not a
    commit: the dataset repo's history is squashed after each upload, which would orphan a pinned
    commit, and the export's own timestamp survives it."""
    got = str(manifest.get("exported_at") or "")
    if not expected or not got.startswith(expected):
        raise ValueError(
            f"the dataset is the {got or 'undated'} export, not {expected!r}: every model in one "
            "comparison trains and is scored on the same export"
        )


def kit_digests(folder: Path) -> dict[str, str]:
    """The first 16 hex digits of each kit file's sha256, for the model card: which code ran."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(Path(folder).glob("*.py"))
    }


# --- transcripts and per-clip counts -------------------------------------------------------------


def read_hyps(path: Path | str) -> dict[str, str]:
    """A transcript file (`write_hyps`) as {segment_id: text}."""
    with Path(path).open(encoding="utf-8") as fh:
        return {j["segment_id"]: j["text"] for j in map(json.loads, fh)}


def write_hyps(
    path: Path, rows: Sequence[dict], texts: Sequence[str], compute: Sequence[float]
) -> None:
    """One transcript per line, the format the harness's Models page imports (D83)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r, text, c in zip(rows, texts, compute, strict=True):
            row = {"segment_id": r["segment_id"], "text": text, "compute_s": c}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def counts(rows: Sequence[dict], clips: Sequence[Mapping[str, int]]) -> dict[str, list[int]]:
    """{segment_id: [folded errors, reference words]}: all a paired comparison needs of a run."""
    return {r["segment_id"]: [c["errors"], c["words"]] for r, c in zip(rows, clips, strict=True)}


def reference_counts(rows: Sequence[dict], by_id: Mapping[str, str], score: Any) -> dict:
    """Another system's transcripts scored against these rows' labels, on the clips it has.

    For a reference that left only transcripts (a run from an earlier export): its counts are
    taken against the current labels, and `pair` compares on the clips both scored."""
    have = [r for r in rows if r["segment_id"] in by_id]
    clips = score.per_clip([r["text"] for r in have], [by_id[r["segment_id"]] for r in have])
    return counts(have, clips)


def pair(
    rows: Sequence[dict],
    a: Counts,
    b: Counts,
    *,
    keys: Sequence[str] = REPORT_KEYS,
    group: str = "episode_id",
    n: int = 2000,
) -> dict[str, Any]:
    """WER(b) - WER(a) in points with a 95% interval, on the clips both runs scored.

    `group` names the unit resampled: a clip's episode on gold and val, a speaker, sentence or
    video on a public set. Returns `clips` (how many were shared), `all`, and one entry per value
    of each clip class in `keys`, as `[difference, low, high]`. Raises when no clip is shared."""
    idx = [i for i, r in enumerate(rows) if r["segment_id"] in a and r["segment_id"] in b]
    if not idx:
        raise ValueError("the two runs share no clip")

    def diff(which: Sequence[int]) -> list[float]:
        return list(
            sweep.paired_bootstrap(
                [
                    dict(zip(("errors", "words"), a[rows[i]["segment_id"]], strict=True))
                    for i in which
                ],
                [
                    dict(zip(("errors", "words"), b[rows[i]["segment_id"]], strict=True))
                    for i in which
                ],
                [str(rows[i][group]) for i in which],
                n=n,
            )
        )

    out: dict[str, Any] = {"clips": len(idx), "all": diff(idx)}
    for key in keys:
        values: dict[str, list[int]] = {}
        for i in idx:
            value = (rows[i].get("classes") or {}).get(key)
            if value is not None:
                values.setdefault(str(value), []).append(i)
        for value, which in sorted(values.items()):
            out[f"{key}={value}"] = diff(which)
    return out


# --- error mining (D110) -------------------------------------------------------------------------


def _miner() -> tuple[Any, Any] | None:
    """`error_mining` and `error_store` from the harness copy `ftkit.harness_scorer` put on the
    path, or None -- with a line saying so -- when the dataset's `harness/` predates them."""
    try:
        from app.services import error_mining, error_store
    except ImportError as exc:
        print(f"no error mining in the dataset's harness/ ({exc}): no error files are written")
        return None
    return error_mining, error_store


def _share(row: Mapping[str, Any]) -> float | None:
    """A gold or val clip's crosstalk share: its own, or from the export's `overlap_spans`."""
    if "overlap_share" in row:
        return row["overlap_share"]
    return distill.overlap_share(row.get("overlap_spans"), duration(row))


def write_errors(
    path: Path,
    run: str,
    rows: Sequence[dict],
    texts: Sequence[str],
    clips: Sequence[Any],
    *,
    overlap: Mapping[str, float | None] | None = None,
    snr: Mapping[str, float | None] | None = None,
    by: str | None = None,
) -> dict[str, Any] | None:
    """Every aligned word pair of one split or set, classified, written to `path` (its stem is
    the set's name); returns the file's report (blocks 1-3), or None when nothing was written.

    A clip's alignment is the one `score.per_clip` counted, when it kept it. `overlap` and `snr`
    are a public set's measured crosstalk and speech-to-noise ratio by clip id; gold and val rows
    carry their own `acoustics`, and their crosstalk as the export's clip-relative
    `overlap_spans` (or an `overlap_share` already taken from them)."""
    kit = _miner()
    if kit is None:
        return None
    mining, store = kit
    ids = mining.unique_ids(str(r["segment_id"]) for r in rows)
    found = mining.rows(
        run,
        Path(path).stem,
        [
            {
                "clip_id": r["segment_id"],
                "group": r["episode_id"],
                "ref": r["text"],
                "hyp": text,
                "overlap_share": overlap.get(i) if overlap is not None else _share(r),
                "snr_db": snr.get(i)
                if snr is not None
                else (r.get("acoustics") or {}).get("snr_db"),
                "by": r.get(by) if by else None,
                "alignment": c.get("alignment") if isinstance(c, Mapping) else None,
            }
            for r, text, c, i in zip(rows, texts, clips, ids, strict=True)
        ],
    )
    if not found:
        return None
    mining.write(found, path)
    return store.report(Path(path))


#: A public set's measured recording conditions, each `benchmarks/<kind>/<set>.parquet`.
CONDITIONS = ("overlap", "acoustics")


def fetch_conditions(
    repo: str, token: str, folder: Path, names: Sequence[str] | None = None
) -> Path:
    """Each public set's measured crosstalk and acoustics, `benchmarks/{overlap,acoustics}/<set>
    .parquet` in `repo`, copied into `folder/overlap/` and `folder/acoustics/`
    (scripts/measure_benchmark_overlap.py wrote them); a set without one is unmeasured on it."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    folder = Path(folder)
    for kind in CONDITIONS:
        (folder / kind).mkdir(parents=True, exist_ok=True)
        for name in names or tuple(BENCHMARKS):
            try:
                got = hf_hub_download(repo, f"benchmarks/{kind}/{name}.parquet", token=token)
            except EntryNotFoundError:
                print(f"{name}: no measured {kind} in {repo}; its clips are unmeasured on it")
                continue
            shutil.copyfile(got, folder / kind / f"{name}.parquet")
    return folder


def print_breakdown(label: str, report: Mapping[str, Any] | None) -> None:
    """Block 1 and block 2 in two lines, under a WER line."""
    if not report:
        return
    buckets = " · ".join(
        f"{b['bucket']} {b['wer']:.2f} ({100 * b['share_of_errors']:.0f}% of errors)"
        for b in report["overlap"]
    )
    n = report["numbers"]
    print(f"    {label} crosstalk: {buckets}")
    print(
        f"    {label} numbers: {n['errors']} errors ({100 * n['share_of_errors']:.1f}%), "
        f"WER without them {n['wer_without']:.2f}"
    )


# --- gold and val --------------------------------------------------------------------------------


def score_split(
    rows: Sequence[dict],
    texts: Sequence[str],
    compute: Sequence[float],
    log: Sequence[tuple[str, str, str]],
    score: Any,
    *,
    errors: Path | None = None,
    run: str = "",
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """One split's metrics and its per-clip counts, each clip aligned once.

    Metrics: the folded, raw and character error rates with S/D/I, the real-time factor, the loop
    retries and the greedy-only score from the same decode, and every clip class (`by_class`).
    With `errors` (`harness/errors/<split>.parquet`), the split's error rows are written there and
    its `breakdown` added."""
    refs = [r["text"] for r in rows]
    clips = score.per_clip(refs, texts)
    m = score.summarize(clips)
    m["rtf"] = sum(compute) / max(sum(duration(r) for r in rows), 1e-9)
    m["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in log]
    if log:  # the same run before any retry: only the retried clips are aligned again
        first = {s: f for s, f, _ in log}
        idx = [i for i, r in enumerate(rows) if r["segment_id"] in first]
        again = score.per_clip([refs[i] for i in idx], [first[rows[i]["segment_id"]] for i in idx])
        greedy = list(clips)
        for i, c in zip(idx, again, strict=True):
            greedy[i] = c
        m["greedy_only"] = score.summarize(greedy)
    m["by_class"] = distill.by_class(rows, clips, score.summarize)
    if errors is not None:
        m["breakdown"] = write_errors(errors, run, rows, texts, clips)
    return m, counts(rows, clips)


def evaluate_run(
    out: Path,
    run: str,
    *,
    splits: Mapping[str, Sequence[dict]],
    decode: Decode,
    score: Any,
    card: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
    references: Mapping[str, Mapping[str, Counts]] | None = None,
    keys: Sequence[str] = REPORT_KEYS,
) -> dict[str, Any]:
    """Decode gold and val with the weights `decode` reads, score them, and write the run's files.

    Written under `out`: `harness/<split>.jsonl` and `harness/model_card.json` (what the Models
    page imports, D83), `harness/errors/<split>.parquet` (its error rows), `<split>_metrics.json`,
    `per_clip.json` (the counts another run is paired against) and `result.json`, whose presence
    on the hub marks the run as done. `references` is {name: {split: per-clip counts}}; each
    becomes `vs[name]`, this run minus that one. `meta` (recipe, seed, stage, best epoch, ...) goes
    into the card and the result row, which is returned."""
    out = Path(out)
    results: dict[str, dict] = {}
    per_clip: dict[str, dict] = {}
    for split in SPLITS:
        rows = splits[split]
        with timed(f"{run} {split}: decoding {len(rows)} clips"):
            texts, compute, log = decode(rows)
        with timed(f"{run} {split}: scoring, per clip class and against the references"):
            errors = out / "harness" / "errors" / f"{split}.parquet"
            m, per_clip[split] = score_split(
                rows, texts, compute, log, score, errors=errors, run=run
            )
            m["vs"] = {}
            for name, ref in (references or {}).items():
                try:
                    m["vs"][name] = pair(rows, ref.get(split) or {}, per_clip[split], keys=keys)
                except ValueError:  # the reference scored none of this split's clips
                    continue
        results[split] = m
        write_hyps(out / "harness" / f"{split}.jsonl", rows, texts, compute)
        (out / f"{split}_metrics.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    gold, val = results["gold"], results["val"]
    full_card = {
        **card,
        **(meta or {}),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "run_name": run,
        "val_wer": val["wer"],
        "gold_wer": gold["wer"],
        # folded, per 100 reference words; S + D + I = WER
        "val_sid": {k: val[k] for k in ("sub", "del", "ins")},
        "gold_sid": {k: gold[k] for k in ("sub", "del", "ins")},
    }
    (out / "harness" / "model_card.json").write_text(
        json.dumps(full_card, indent=1, ensure_ascii=False)
    )
    (out / "per_clip.json").write_text(json.dumps(per_clip))
    row = {
        "run_name": run,
        **(meta or {}),
        "val_wer": val["wer"],
        "gold_wer": gold["wer"],
        **{split: {k: results[split][k] for k in KEEP} for split in SPLITS},
        "gold_by_class": {k: gold["by_class"].get(k) for k in keys},
    }
    (out / "result.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    for split in ("val", "gold"):
        m = results[split]
        print(
            f"{run} {split}: WER {m['wer']:.2f} (S {m['sub']:.2f}  D {m['del']:.2f}  "
            f"I {m['ins']:.2f}), raw {m['raw_wer']:.2f}"
        )
        for name, vs in m["vs"].items():
            d, lo, hi = vs["all"]
            print(f"    minus {name}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] on {vs['clips']} clips")
        print_breakdown(split, m.get("breakdown"))
    return row


# --- a finished run, scored again under newer rules --------------------------------------------

#: What `evaluate_run` derives from the scores, in `result.json` and the card: everything else in
#: them is the run's recipe, kept as it was when the run is scored again.
_ROW_SCORES = ("run_name", "val_wer", "gold_wer", "gold", "val", "gold_by_class")
_CARD_SCORES = (
    "created_at",
    "rescored_at",
    "run_name",
    "val_wer",
    "gold_wer",
    "val_sid",
    "gold_sid",
)


def stored_decode(out: Path) -> Decode:
    """A `decode` that reads a finished run's transcripts, compute and loop retries from its
    files instead of running a model. A clip the run never transcribed is refused: the rows are
    from another export."""
    out = Path(out)
    texts: dict[str, tuple[str, float]] = {}
    retries: dict[str, tuple[str, str, str]] = {}
    for split in SPLITS:
        hyps = out / "harness" / f"{split}.jsonl"
        if hyps.exists():
            with hyps.open(encoding="utf-8") as fh:
                for j in map(json.loads, fh):
                    texts[j["segment_id"]] = (j["text"], j["compute_s"])
        metrics = out / f"{split}_metrics.json"
        if metrics.exists():
            for r in json.loads(metrics.read_text("utf-8")).get("retried", []):
                retries[r["segment_id"]] = (r["segment_id"], r["first"], r["retry"])

    def decode(rows: Sequence[dict]) -> tuple[list[str], list[float], list[tuple[str, str, str]]]:
        missing = [r["segment_id"] for r in rows if r["segment_id"] not in texts]
        if missing:
            raise ValueError(
                f"{len(missing)} clip(s) this run never transcribed, first {missing[0]!r}: "
                "the rows are from another export"
            )
        ids = [r["segment_id"] for r in rows]
        return (
            [texts[i][0] for i in ids],
            [texts[i][1] for i in ids],
            [retries[i] for i in ids if i in retries],
        )

    return decode


def rescore_run(
    out: Path,
    run: str,
    *,
    splits: Mapping[str, Sequence[dict]],
    score: Any,
    fold_version: str | None = None,
    references: Mapping[str, Mapping[str, Counts]] | None = None,
    keys: Sequence[str] = REPORT_KEYS,
) -> dict[str, Any]:
    """Score a finished run again from its stored transcripts, under `score`'s rules (D113).

    The run's files (`evaluate_run`) are rewritten as if it had been scored today: the metrics,
    error rows, per-clip counts, result row and card, with the recipe kept from `result.json`, the
    card's `created_at` kept and `rescored_at` added, and `fold_version` (default:
    `score.fold_version`) set. Nothing is decoded."""
    out = Path(out)
    old_row = json.loads((out / "result.json").read_text("utf-8"))
    old_card = json.loads((out / "harness" / "model_card.json").read_text("utf-8"))
    meta = {k: v for k, v in old_row.items() if k not in _ROW_SCORES}
    card = {k: v for k, v in old_card.items() if k not in _CARD_SCORES and k not in meta}
    card["fold_version"] = fold_version or score.fold_version
    row = evaluate_run(
        out, run, splits=splits, decode=stored_decode(out), score=score, card=card, meta=meta,
        references=references, keys=keys,
    )  # fmt: skip
    path = out / "harness" / "model_card.json"
    full = json.loads(path.read_text("utf-8"))
    full["created_at"] = old_card.get("created_at", full["created_at"])
    full["rescored_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
    path.write_text(json.dumps(full, indent=1, ensure_ascii=False))
    return row


def rescore_benchmark(
    out: Path,
    name: str,
    *,
    score: Any,
    run: str | None = None,
    conditions_dir: Path | None = None,
) -> dict[str, Any]:
    """Score a public set again from its stored lines (`run_benchmarks`), under `score`'s rules.

    The lines hold each clip's reference, transcript, group and `by` value, which is all a score
    needs; the hours, speed, retries and limit of the decode are kept from the old summary.
    Rewrites `<name>.json`, `<name>.jsonl` and the set's error rows."""
    out = Path(out)
    folder = out / "benchmarks"
    old = json.loads((folder / f"{name}.json").read_text("utf-8"))
    with (folder / f"{name}.jsonl").open(encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    by = BENCHMARKS[name].by
    rows = [
        {
            "segment_id": x["id"],
            "episode_id": x["group"],
            "text": x["ref"],
            "start_time": 0.0,
            "end_time": 0.0,
            **({by: x[by]} if by else {}),
        }
        for x in lines
    ]
    summary, new_lines = score_benchmark(
        name,
        rows,
        [x["hyp"] for x in lines],
        score,
        errors=out / "harness" / "errors" / f"{name}.parquet",
        run=run or out.name,
        overlap=_measured(conditions_dir, "overlap", name),
        snr=_measured(conditions_dir, "acoustics", name),
    )
    for key in ("hours", "retried", "x_realtime", "limit"):
        if key in old:
            summary[key] = old[key]
    with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in new_lines)
    (folder / f"{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"{name}: WER {old['wer']:.2f} -> {summary['wer']:.2f}")
    return summary


# --- the public sets -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    """One public Nepali test set, as it was scored on 2026-09-27 (findings.md).

    `audio`, `text` and `group` are its columns: the clip, the reference, and the unit a paired
    interval resamples (a speaker; FLEURS repeats sentences, so there it is the sentence; nepali_cs
    is resampled by video). `clip_id` names each clip, from one column or several joined by `-`.
    `by` is a column the score is also split by. `base_wer` is base Flex's folded WER on
    2026-09-27: decoding base again should land on it, which is the loader's check."""

    name: str
    what: str
    repo: str
    config: str
    split: str
    audio: str
    text: str
    group: str
    clip_id: tuple[str, ...]
    base_wer: float
    by: str | None = None
    #: "parquet": the hub's converted files. "tar": Common Voice's mirror, a TSV and a tar of MP3s.
    layout: str = "parquet"


BENCHMARKS: dict[str, Benchmark] = {
    b.name: b
    for b in (
        Benchmark(
            name="fleurs",
            what="FLEURS ne_np test: read Wikipedia sentences",
            repo="google/fleurs",
            config="ne_np",
            split="test",
            audio="audio",
            text="raw_transcription",
            group="id",
            clip_id=("path",),
            base_wer=11.10,
        ),
        Benchmark(
            name="slr54",
            what="OpenSLR 54 test: crowd-sourced read speech (the mirror's own split)",
            repo="iamTangsang/OpenSLR54-Nepali-ASR",
            config="default",
            split="test",
            audio="utterance",
            text="transcription",
            group="speaker_id",
            clip_id=("utterance_id",),
            base_wer=8.17,
        ),
        Benchmark(
            name="common_voice",
            what="Common Voice 22 ne-NP test: crowd-sourced read speech",
            repo="fsicoli/common_voice_22_0",
            config="ne-NP",
            split="test",
            audio="path",
            text="sentence",
            group="client_id",
            clip_id=("path",),
            base_wer=8.78,
            layout="tar",
        ),
        Benchmark(
            name="indicvoices",
            what="IndicVoices nepali valid: read, extempore and conversation",
            repo="ai4bharat/IndicVoices",
            config="nepali",
            split="valid",
            audio="audio_filepath",
            text="text",
            group="speaker_id",
            clip_id=(),  # it has none: clips are numbered in file order
            base_wer=12.70,
            by="scenario",
        ),
        Benchmark(
            name="nepali_cs",
            what="nepali-cs-asr test: Nepali-English code-switched lectures",
            repo="saileshbro/nepali-cs-asr",
            config="default",
            split="test",
            audio="audio",
            text="transcription",
            group="video_id",
            clip_id=("video_id", "start_ms"),
            base_wer=10.98,
            by="speech_type",
        ),
    )
}


def plain_tokens(text: str | None) -> list[str]:
    """The normalisation published Nepali results use: NFC, punctuation and symbols dropped,
    Latin lowercased, and nothing else. No fold: `टिम` and `team` are different words here."""
    text = unicodedata.normalize("NFC", text or "")
    kept = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text)
    return kept.lower().split()


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Word-level Levenshtein distance: the errors of a plain WER."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def plain_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Pooled plain WER in percent: errors over reference words, across all clips."""
    errors = words = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        r = plain_tokens(ref)
        errors += edit_distance(r, plain_tokens(hyp))
        words += len(r)
    return 100 * errors / max(words, 1)


def plain_cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Pooled plain CER in percent, over the characters of the plain text with its spaces.

    Published Nepali results report it beside WER. A word split differently costs a plain WER
    two words but a CER one space, so the two together tell hearing from spacing."""
    errors = chars = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        r = " ".join(plain_tokens(ref))
        errors += edit_distance(r, " ".join(plain_tokens(hyp)))
        chars += len(r)
    return 100 * errors / max(chars, 1)


def _audio(data: bytes) -> Any:
    """Any encoded clip as 16 kHz mono int16: libsndfile first, ffmpeg for what it cannot read."""
    import numpy as np
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g)
        return np.round(np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    except Exception:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "pipe:1",
        ]
        raw = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.int16).copy()


def benchmark_row(bench: Benchmark, record: Mapping[str, Any], index: int, audio: Any) -> dict:
    """One clip as the decoders and the scorer read a row: it carries its own audio, its group
    stands where an episode does, and `segment_id` is the id the 2026-09-27 files used."""
    if bench.clip_id:
        clip_id = "-".join(str(record[c]) for c in bench.clip_id).rsplit("/", 1)[-1]
    else:
        clip_id = f"iv-{index:05d}"
    row = {
        "segment_id": clip_id,
        "episode_id": str(record[bench.group]),
        "text": record[bench.text] or "",
        "audio": audio,
        "start_time": 0.0,
        "end_time": len(audio) / SR,
    }
    if bench.by:
        row[bench.by] = record[bench.by]
    return row


def _parquet_records(bench: Benchmark, work: Path, token: str) -> Iterator[tuple[dict, bytes]]:
    """(record, encoded audio) from the hub's parquet conversion of the split, a file at a time."""
    import pyarrow.parquet as pq
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    listing = (
        f"https://huggingface.co/api/datasets/{bench.repo}/parquet/{bench.config}/{bench.split}"
    )
    urls = requests.get(listing, headers=headers, timeout=60)
    urls.raise_for_status()
    columns = sorted(
        {bench.audio, bench.text, bench.group, *bench.clip_id, *([bench.by] if bench.by else [])}
    )
    for k, url in enumerate(urls.json()):
        path = work / f"{bench.name}-{k:04d}.parquet"
        if not path.exists():
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                r.raise_for_status()
                with path.with_suffix(".part").open("wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            path.with_suffix(".part").rename(path)
        table = pq.ParquetFile(path)
        missing = set(columns) - set(table.schema_arrow.names)
        if missing:
            raise KeyError(
                f"{bench.repo}: no column {sorted(missing)}; it has {table.schema_arrow.names}"
            )
        for batch in table.iter_batches(batch_size=64, columns=columns):
            for record in batch.to_pylist():
                yield record, record[bench.audio]["bytes"]


def _tar_records(bench: Benchmark, token: str) -> Iterator[tuple[dict, bytes]]:
    """(record, encoded audio) from the Common Voice mirror: `transcript/<locale>/<split>.tsv`
    and the MP3s in `audio/<locale>/<split>/<locale>_<split>_<n>.tar`, in the TSV's order."""
    import csv

    from huggingface_hub import hf_hub_download, list_repo_files

    def fetch(name: str) -> str:
        return hf_hub_download(bench.repo, name, repo_type="dataset", token=token)

    tsv = fetch(f"transcript/{bench.config}/{bench.split}.tsv")
    with open(tsv, encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    prefix = f"audio/{bench.config}/{bench.split}/"
    clips: dict[str, bytes] = {}
    for name in sorted(list_repo_files(bench.repo, repo_type="dataset", token=token)):
        if name.startswith(prefix) and name.endswith(".tar"):
            with tarfile.open(fetch(name)) as tf:
                for member in tf:
                    if member.isfile():
                        clips[member.name.rsplit("/", 1)[-1]] = tf.extractfile(member).read()
    for record in records:
        yield record, clips[record[bench.audio]]


def iter_benchmark(name: str, work: Path, token: str) -> Iterator[dict]:
    """A public set's clips one at a time, in file order, each a row with its audio (16 kHz mono
    int16): for a pass that must not hold the whole set in RAM."""
    bench = BENCHMARKS[name]
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    records = (
        _tar_records(bench, token)
        if bench.layout == "tar"
        else _parquet_records(bench, work, token)
    )
    for index, (record, data) in enumerate(records):
        yield benchmark_row(bench, record, index, _audio(data))


def load_benchmark(name: str, work: Path, token: str, limit: int | None = None) -> list[dict]:
    """A public set's clips as rows with their audio in RAM (16 kHz mono int16), in file order.

    `limit` keeps the first clips only, for a smoke run. Nothing here is cached beyond the
    downloaded files: a set is loaded, decoded, scored and dropped."""
    rows = []
    for row in iter_benchmark(name, work, token):
        if limit is not None and len(rows) >= limit:
            break
        rows.append(row)
    return rows


def score_benchmark(
    name: str,
    rows: Sequence[dict],
    texts: Sequence[str],
    score: Any,
    *,
    errors: Path | None = None,
    run: str = "",
    overlap: Mapping[str, float | None] | None = None,
    snr: Mapping[str, float | None] | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """A public set's summary and its per-clip lines.

    Summary: folded WER with S/D/I, raw WER and CER from the harness's scorer, plain WER and CER,
    and the score per value of the set's `by` column. A line is `{id, group, ref, hyp, errors,
    words}`: the 2026-09-27 files' fields plus the counts a paired comparison needs. With
    `errors`, the set's error rows are written there, crosstalk from `overlap` and SNR from `snr`
    (clips missing from either are unmeasured on it), and its `breakdown` added to the summary."""
    bench = BENCHMARKS[name]
    refs = [r["text"] for r in rows]
    clips = score.per_clip(refs, texts)
    summary = score.summarize(clips)
    summary["plain_wer"] = plain_wer(refs, texts)
    summary["plain_cer"] = plain_cer(refs, texts)
    summary["hours"] = sum(duration(r) for r in rows) / 3600
    if bench.by:
        values: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            values.setdefault(str(r[bench.by]), []).append(i)
        summary["by"] = {
            v: score.summarize([clips[i] for i in idx]) for v, idx in sorted(values.items())
        }
    if errors is not None:
        summary["breakdown"] = write_errors(
            errors, run, rows, texts, clips, overlap=overlap or {}, snr=snr or {}, by=bench.by
        )
    lines = []
    for r, hyp, c in zip(rows, texts, clips, strict=True):
        line = {
            "id": r["segment_id"],
            "group": r["episode_id"],
            "ref": r["text"],
            "hyp": hyp,
            "errors": c["errors"],
            "words": c["words"],
        }
        if bench.by:
            line[bench.by] = r[bench.by]
        lines.append(line)
    return summary, lines


def pair_benchmark(a: Sequence[dict], b: Sequence[dict], n: int = 2000) -> dict[str, Any]:
    """WER(b) - WER(a) on one public set from two models' per-clip lines, resampling the set's
    group; `{clips, all: [difference, low, high]}`."""
    rows = [{"segment_id": x["id"], "episode_id": x["group"]} for x in b]
    return pair(
        rows,
        {x["id"]: [x["errors"], x["words"]] for x in a},
        {x["id"]: [x["errors"], x["words"]] for x in b},
        keys=(),
        n=n,
    )


def _measured(folder: Path | None, kind: str, name: str) -> dict[str, float | None] | None:
    """`folder/<kind>/<name>.parquet` as values by clip id, or None when absent -- or when the
    harness copy predates the reader (SNR came with mine-v2)."""
    path = Path(folder) / kind / f"{name}.parquet" if folder is not None else None
    if path is None or not path.exists() or (kit := _miner()) is None:
        return None
    read = getattr(kit[0], "read_overlap" if kind == "overlap" else "read_snr", None)
    return read(path) if read is not None else None


def run_benchmarks(
    out: Path,
    *,
    decode: Decode,
    score: Any,
    work: Path,
    token: str,
    names: Sequence[str] = tuple(BENCHMARKS),
    limit: int | None = None,
    done: Collection[str] = (),
    run: str | None = None,
    conditions_dir: Path | None = None,
) -> dict[str, dict]:
    """Decode and score each public set with the weights `decode` reads, one set at a time.

    Writes `out/benchmarks/<name>.jsonl` (per clip) and `<name>.json` (summary), and the set's
    error rows to `out/harness/errors/<name>.parquet`, as each set finishes, and skips a set named
    in `done`, so a lost runtime costs one set. `run` names the rows (default: `out`'s folder);
    `conditions_dir` holds the sets' measured crosstalk and acoustics (`fetch_conditions`)."""
    run = run or Path(out).name
    folder = Path(out) / "benchmarks"
    folder.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in names:
        if name in done:
            print(f"{name}: already scored, skipped")
            continue
        with timed(f"{name}: loading"):
            rows = load_benchmark(name, work, token, limit)
        with timed(f"{name}: decoding {len(rows)} clips"):
            texts, compute, log = decode(rows)
        with timed(f"{name}: scoring"):
            summary, lines = score_benchmark(
                name,
                rows,
                texts,
                score,
                errors=Path(out) / "harness" / "errors" / f"{name}.parquet",
                run=run,
                overlap=_measured(conditions_dir, "overlap", name),
                snr=_measured(conditions_dir, "acoustics", name),
            )
        summary["retried"] = len(log)
        summary["x_realtime"] = sum(duration(r) for r in rows) / max(sum(compute), 1e-9)
        summary["limit"] = limit
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        (folder / f"{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        print(
            f"{name}: WER {summary['wer']:.2f} (S {summary['sub']:.2f}  D {summary['del']:.2f}  "
            f"I {summary['ins']:.2f}), plain {summary['plain_wer']:.2f} / CER "
            f"{summary['plain_cer']:.2f}, {len(rows)} clips, {summary['x_realtime']:.0f}x realtime"
        )
        print_breakdown(name, summary.get("breakdown"))
        summaries[name] = summary
        del rows
    return summaries
