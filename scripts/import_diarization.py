#!/usr/bin/env python
"""Import speaker turns from a diarization file made over the retained episode audio (D78).

python scripts/import_diarization.py diarization.json
python scripts/import_diarization.py diarization.json --model pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.diarization_import import import_diarization

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="the diarization JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the diarizer that produced it")
    parser.add_argument("--actor", default="import_diarization", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    bootstrap()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    with session_scope() as session:
        report = import_diarization(
            session, payload, model=args.model, source=args.path.name, actor=args.actor
        )
    print(
        f"{report.runs_created} run(s) imported ({report.turns_inserted} turns), "
        f"{report.runs_unchanged} already present"
    )
    if report.unknown_episodes:
        print(f"not in the harness, skipped: {', '.join(report.unknown_episodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
