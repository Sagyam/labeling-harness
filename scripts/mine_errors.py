#!/usr/bin/env python
"""Derive a model's error-mining files from what the harness already holds (WER-Breakdown.md).

python scripts/mine_errors.py flex-2026-09-30-vanilla-s1     # one model
python scripts/mine_errors.py --all                          # every imported model folder

Gold and val come from the model's imported runs; each public set from
data/models/asr/<slug>/benchmarks/<set>.jsonl when the notebook's per-clip lines were copied
there, with its crosstalk from data/benchmarks/overlap/<set>.parquet when that was measured
(scripts/measure_benchmark_overlap.py). Files land in data/models/asr/<slug>/errors/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import REPO_ROOT, bootstrap
from app.db.session import session_scope
from app.services.error_backfill import derive_error_files
from app.services.model_import import CARD_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="*", help="model folder names under models.root")
    parser.add_argument("--all", action="store_true", help="every model folder")
    parser.add_argument(
        "--overlap",
        type=Path,
        default=REPO_ROOT / "data" / "benchmarks" / "overlap",
        help="where the public sets' measured crosstalk is",
    )
    args = parser.parse_args(argv)
    settings = bootstrap()
    root = settings.models.root
    slugs = args.slugs
    if args.all:
        slugs = sorted(p.name for p in root.iterdir() if (p / CARD_NAME).is_file())
    if not slugs:
        parser.error("name a model, or pass --all")
    with session_scope() as session:
        for slug in slugs:
            folder = root / slug
            if not folder.is_dir():
                print(f"{slug}: no folder {folder}")
                return 1
            try:
                files = derive_error_files(session, folder, overlap_dirs=[args.overlap])
            except ValueError as exc:
                print(f"refused: {exc}")
                return 1
            for f in files:
                print(f"{slug}: {f.set} ({f.run}, {f.fold_version}, {f.miner_version}) -> {f.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
