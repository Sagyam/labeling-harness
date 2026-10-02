#!/usr/bin/env python
"""Measure each public set's recording conditions once, for the WER breakdown (WER-Breakdown.md).

Every clip goes through ingest's overlap detector (D77) and its acoustic meter (Brouhaha's SNR
and C50, and bandwidth; D87). A set's crosstalk lands in data/benchmarks/overlap/<set>.parquet
and its acoustics in data/benchmarks/acoustics/<set>.parquet; --upload puts both in the model
repo under benchmarks/, where the notebooks read them. The loaders are evalkit's, which need
packages the backend does not carry, so run it as:

  uv run --no-sync --project backend --with pyarrow --with requests --with huggingface_hub \\
      --with scipy python scripts/measure_benchmark_overlap.py [SET ...] [--only acoustics] \\
      [--listen 10] [--upload]

Clips stream one at a time and are dropped once measured; downloads go under data/benchmarks/,
never /tmp. Each measurement is a pass of its own, and a stopped pass resumes from the clips
already written. --listen N writes each set's N highest-overlap clips as FLAC under
data/benchmarks/overlap/listen/<set>/, to be heard before the set's buckets are believed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import REPO_ROOT, bootstrap
from app.services.acoustics import AcousticMeter
from app.services.benchmark_overlap import (
    measure_acoustics,
    measure_rows,
    write_acoustics,
    write_overlap,
)
from app.services.error_mining import (
    number_id,
    overlap_bucket,
    read_overlap,
    read_snr,
    snr_bucket,
)
from app.services.overlap import SAMPLE_RATE, OverlapDetector

sys.path.insert(0, str(REPO_ROOT / "notebooks" / "src"))
import evalkit

MODEL_REPO = "Sagyam/nepanglish-asr-flex-ft"
DATA = REPO_ROOT / "data" / "benchmarks"


def _token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face token: set HF_TOKEN or run `hf auth login`")
    return token


KINDS = ("overlap", "acoustics")


def measure(name: str, kind: str, instrument, token: str, out: Path) -> Path:
    """Measure one set for ``kind``, resuming from its partial file; returns its Parquet file."""
    partial, path = out / f"{name}.jsonl", out / f"{name}.parquet"
    measure_with = measure_rows if kind == "overlap" else measure_acoustics
    done = set()
    if partial.exists():
        done = {json.loads(line)["clip_id"] for line in partial.read_text("utf-8").splitlines()}
        print(f"{name}: resuming after {len(done)} clips already measured", flush=True)
    t0, seconds, count = time.perf_counter(), 0.0, 0
    clips = evalkit.iter_benchmark(name, DATA / name, token)
    with partial.open("a", encoding="utf-8") as fh:
        for found in measure_with(instrument, clips, done=done):
            fh.write(json.dumps(found) + "\n")
            fh.flush()
            count, seconds = count + 1, seconds + found["duration"]
            if count % 250 == 0:
                print(
                    f"{name}: {count} clips, {seconds / 3600:.2f} h measured in "
                    f"{time.perf_counter() - t0:.0f} s",
                    flush=True,
                )
    if kind == "overlap":
        write_overlap(partial, path)
        values, bucket = read_overlap(path), overlap_bucket
    else:
        write_acoustics(partial, path)
        values, bucket = read_snr(path), snr_bucket
    buckets: dict[str, int] = {}
    for value in values.values():
        buckets[bucket(value)] = buckets.get(bucket(value), 0) + 1
    print(f"{name}: {len(values)} clips -> {path}; {kind} buckets {buckets}", flush=True)
    return path


def listen(name: str, path: Path, token: str, out: Path, n: int) -> None:
    """Write the set's ``n`` highest-overlap clips as FLAC, named by rank, share and id. Only
    clips with some overlap: a set with two such clips has two to hear."""
    import soundfile as sf

    shares = read_overlap(path)
    top = sorted((c for c in shares if shares[c]), key=lambda c: -shares[c])[:n]
    if not top:
        print(f"{name}: no clip with any overlap, nothing to hear", flush=True)
        return
    rank = {clip_id: k for k, clip_id in enumerate(top, 1)}
    folder = out / "listen" / name
    folder.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    for row in evalkit.iter_benchmark(name, DATA / name, token):
        clip_id = number_id(str(row["segment_id"]), seen)
        if clip_id in rank:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in clip_id)
            target = folder / f"{rank[clip_id]:02d}_{100 * shares[clip_id]:.0f}pct_{safe}.flac"
            sf.write(target, row["audio"], SAMPLE_RATE)
            del rank[clip_id]
            if not rank:
                break
    print(f"{name}: the {len(top)} highest-overlap clips are in {folder}", flush=True)


def upload(name: str, kind: str, path: Path, token: str) -> None:
    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"benchmarks/{kind}/{name}.parquet",
        repo_id=MODEL_REPO,
        token=token,
        commit_message=f"{kind} for {name} (measure_benchmark_overlap.py)",
    )
    print(f"{name}: uploaded to {MODEL_REPO}/benchmarks/{kind}/{name}.parquet", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sets", nargs="*", default=list(evalkit.BENCHMARKS))
    parser.add_argument("--only", choices=KINDS, help="one measurement instead of both")
    parser.add_argument("--threads", type=int, default=4, help="onnxruntime threads")
    parser.add_argument("--force", action="store_true", help="measure a finished set again")
    parser.add_argument("--listen", type=int, default=10, help="highest-overlap clips to keep")
    parser.add_argument("--upload", action="store_true", help=f"upload each file to {MODEL_REPO}")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.sets) - set(evalkit.BENCHMARKS))
    if unknown:
        parser.error(f"unknown set(s) {unknown}; known: {list(evalkit.BENCHMARKS)}")

    bootstrap()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per hub request otherwise
    kinds = [args.only] if args.only else list(KINDS)
    instruments = {}
    if "overlap" in kinds:
        instruments["overlap"] = OverlapDetector(threads=args.threads)
        if not instruments["overlap"].available:
            print("no overlap model: see HARNESS_OVERLAP_NO_DOWNLOAD and HARNESS_ALIGNER_MODEL_DIR")
            return 1
    if "acoustics" in kinds:
        from app.services.brouhaha import Brouhaha

        instruments["acoustics"] = AcousticMeter(Brouhaha(threads=args.threads))
        if instruments["acoustics"].version.endswith("bandwidth-only"):
            print("no Brouhaha model: scripts/export_brouhaha_onnx.py builds it into data/models/")
            return 1
    token = _token()
    for name in args.sets:
        for kind in kinds:
            out = DATA / kind
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{name}.parquet"
            if args.force:
                path.unlink(missing_ok=True)
                (out / f"{name}.jsonl").unlink(missing_ok=True)
            if path.exists():
                print(f"{name}: {kind} already measured ({path}); --force to again", flush=True)
            else:
                path = measure(name, kind, instruments[kind], token, out)
            if kind == "overlap" and args.listen:
                listen(name, path, token, out, args.listen)
            if args.upload:
                upload(name, kind, path, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
