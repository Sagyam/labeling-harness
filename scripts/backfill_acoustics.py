#!/usr/bin/env python
"""Measure the acoustics of imported clips not yet measured under the current rules (D87).

python scripts/backfill_acoustics.py
python scripts/backfill_acoustics.py --episode show-a_ep012 --force
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.acoustics import AcousticMeter
from app.services.acoustics_backfill import backfill_acoustics
from app.storage import build_storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", action="append", help="restrict to this episode (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-measure clips already measured")
    parser.add_argument("--actor", default="backfill_acoustics", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    settings = bootstrap()
    with session_scope() as session:
        report = backfill_acoustics(
            session,
            build_storage(settings),
            AcousticMeter(),
            actor=args.actor,
            episode_external_ids=args.episode,
            force=args.force,
        )
    print(
        f"measured {report.episodes_measured} episode(s), skipped {report.episodes_skipped} "
        f"already current, {report.episodes_without_audio} without retained audio; "
        f"{report.segments_updated} clips updated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
