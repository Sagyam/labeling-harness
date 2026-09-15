#!/usr/bin/env python
"""Re-read every scored clip's classes and rebuild each model run's class breakdowns (D87).

Run it after a backfill or a voice link changed what is known about the clips.

python scripts/reclassify_runs.py
python scripts/reclassify_runs.py --run 5 --run 6
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.model_import import reclassify_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=int, action="append", help="restrict to this run id")
    parser.add_argument("--actor", default="reclassify_runs", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    bootstrap()
    with session_scope() as session:
        report = reclassify_runs(session, actor=args.actor, run_ids=args.run)
    print(f"reclassified {report.runs} run(s), {report.clips} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
