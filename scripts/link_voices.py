#!/usr/bin/env python
"""Link diarized speakers across episodes into anonymous voices (D87).

python scripts/link_voices.py            # link the runs not linked yet
python scripts/link_voices.py --relink   # forget every link and start over
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.voices import link_voices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--relink", action="store_true", help="start over; voice ids may change")
    parser.add_argument("--actor", default="link_voices", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    bootstrap()
    with session_scope() as session:
        report = link_voices(session, actor=args.actor, relink=args.relink)
    print(
        f"linked {report.runs_linked} run(s), {report.speakers_linked} speakers; "
        f"{report.voices} voices in the corpus"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
