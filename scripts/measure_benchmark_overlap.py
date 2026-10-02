#!/usr/bin/env python
"""Measure crosstalk on each public set once, for the WER breakdown (docs/WER-Breakdown.md).

Every clip goes through ingest's overlap detector (D77); a set's spans land in
data/benchmarks/overlap/<set>.parquet, and --upload puts them in the model repo at
benchmarks/overlap/<set>.parquet, where the notebooks read them. The loaders are evalkit's, which
need packages the backend does not carry, so run it as:

  uv run --no-sync --project backend --with pyarrow --with requests --with huggingface_hub \\
      --with scipy python scripts/measure_benchmark_overlap.py [SET ...] [--listen 10] [--upload]

Clips stream one at a time and are dropped once measured; downloads go under data/benchmarks/,
never /tmp. A stopped run resumes from the clips already written. --listen N writes each set's N
highest-overlap clips as FLAC under data/benchmarks/overlap/listen/<set>/, to be heard before
the set's buckets are believed.
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
from app.services.benchmark_overlap import measure_rows, write_overlap
from app.services.error_mining import number_id, overlap_bucket, read_overlap
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


def measure(name: str, detector: OverlapDetector, token: str, out: Path) -> Path:
    """Measure one set, resuming from its partial file; returns the set's Parquet file."""
    partial, path = out / f"{name}.jsonl", out / f"{name}.parquet"
    done = set()
    if partial.exists():
        done = {json.loads(line)["clip_id"] for line in partial.read_text("utf-8").splitlines()}
        print(f"{name}: resuming after {len(done)} clips already measured", flush=True)
    t0, seconds, count = time.perf_counter(), 0.0, 0
    clips = evalkit.iter_benchmark(name, DATA / name, token)
    with partial.open("a", encoding="utf-8") as fh:
        for found in measure_rows(detector, clips, done=done):
            fh.write(json.dumps(found) + "\n")
            fh.flush()
            count, seconds = count + 1, seconds + found["duration"]
            if count % 250 == 0:
                print(
                    f"{name}: {count} clips, {seconds / 3600:.2f} h measured in "
                    f"{time.perf_counter() - t0:.0f} s",
                    flush=True,
                )
    write_overlap(partial, path)
    shares = read_overlap(path)
    buckets: dict[str, int] = {}
    for share in shares.values():
        buckets[overlap_bucket(share)] = buckets.get(overlap_bucket(share), 0) + 1
    print(f"{name}: {len(shares)} clips -> {path}; buckets {buckets}", flush=True)
    return path


def listen(name: str, path: Path, token: str, out: Path, n: int) -> None:
    """Write the set's ``n`` highest-overlap clips as FLAC, named by rank, share and id."""
    import soundfile as sf

    shares = read_overlap(path)
    top = sorted(shares, key=lambda c: -(shares[c] or 0.0))[:n]
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


def upload(name: str, path: Path, token: str) -> None:
    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"benchmarks/overlap/{name}.parquet",
        repo_id=MODEL_REPO,
        token=token,
        commit_message=f"overlap spans for {name} (measure_benchmark_overlap.py)",
    )
    print(f"{name}: uploaded to {MODEL_REPO}/benchmarks/overlap/{name}.parquet", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sets", nargs="*", default=list(evalkit.BENCHMARKS))
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
    detector = OverlapDetector(threads=args.threads)
    if not detector.available:
        print("no overlap model: see HARNESS_OVERLAP_NO_DOWNLOAD and HARNESS_ALIGNER_MODEL_DIR")
        return 1
    token, out = _token(), DATA / "overlap"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        path = out / f"{name}.parquet"
        if args.force:
            path.unlink(missing_ok=True)
            (out / f"{name}.jsonl").unlink(missing_ok=True)
        if path.exists():
            print(f"{name}: already measured ({path}); --force to measure again", flush=True)
        else:
            path = measure(name, detector, token, out)
        if args.listen:
            listen(name, path, token, out, args.listen)
        if args.upload:
            upload(name, path, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
