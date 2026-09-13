#!/usr/bin/env python
"""Measure overlapped speech on imported episodes that have not been measured yet (D77).

python scripts/backfill_overlap.py
python scripts/backfill_overlap.py --episode show-a_ep012 --force
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.overlap import OverlapDetector
from app.services.overlap_backfill import backfill_overlap
from app.storage import build_storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", action="append", help="restrict to this episode (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-measure episodes already measured")
    parser.add_argument("--threads", type=int, default=4, help="onnxruntime threads")
    parser.add_argument("--actor", default="backfill_overlap", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    settings = bootstrap()
    detector = OverlapDetector(threads=args.threads)
    if not detector.available:
        print("no overlap model: see HARNESS_OVERLAP_NO_DOWNLOAD and HARNESS_ALIGNER_MODEL_DIR")
        return 1

    with session_scope() as session:
        report = backfill_overlap(
            session,
            build_storage(settings),
            detector,
            settings=settings,
            actor=args.actor,
            episode_external_ids=args.episode,
            force=args.force,
        )
    print(
        f"measured {report.episodes_measured} episode(s), skipped {report.episodes_skipped} "
        f"already measured, {report.episodes_without_audio} without retained audio; "
        f"{report.segments_updated} clips updated, {report.segments_flagged} flagged "
        f"speaker_overlap; {report.overlap_seconds:.1f} s of overlap"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
