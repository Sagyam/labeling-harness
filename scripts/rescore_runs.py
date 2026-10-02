#!/usr/bin/env python
"""Score every imported model run made under older fold rules again, under today's (D113).

Run it after a fold version: the run keeps its transcripts and the reference it was imported
against, and its counts, breakdowns and fold version are replaced.

python scripts/rescore_runs.py
python scripts/rescore_runs.py --run 11 --run 12
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.model_import import rescore_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=int, action="append", help="restrict to this run id")
    parser.add_argument("--actor", default="rescore_runs", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    bootstrap()
    with session_scope() as session:
        report = rescore_runs(session, actor=args.actor, run_ids=args.run)
    print(
        f"rescored {report.runs} run(s), {report.clips} clips; "
        f"{report.unchanged} already under today's fold"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
