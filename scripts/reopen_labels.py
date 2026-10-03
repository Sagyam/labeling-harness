#!/usr/bin/env python
"""Reopen labelled clips into triage for a second listen, all of a list or none (D114).

python scripts/reopen_labels.py suspects.json --note "D100 heavy-crosstalk check" --dry-run
python scripts/reopen_labels.py suspects.json --note "D100 heavy-crosstalk check"

The list is JSON: ``[{"segment_id": <external id>, "priority": 0.8, "details": "3.1 s: ..."}]``.
``priority`` sorts triage; ``details`` is what the triage tooltip shows (what to listen for).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sqlalchemy as sa

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.models import Segment
from app.services.labeling import LabelingError, reopen_for_relabel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("listing", type=Path, help="JSON list of clips to reopen")
    parser.add_argument("--note", help="why these clips are reopened, kept on each task")
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    args = parser.parse_args(argv)

    entries = json.loads(args.listing.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and "segment_id" in e for e in entries
    ):
        print(f"{args.listing} is not a list of {{segment_id, priority, details}}")
        return 2

    settings = bootstrap()
    actor = settings.labels.default_annotator
    with session_scope() as session:
        problems: list[str] = []
        for entry in entries:
            external_id = entry["segment_id"]
            segment = session.scalar(sa.select(Segment).where(Segment.external_id == external_id))
            if segment is None:
                problems.append(f"no clip {external_id!r}")
                continue
            try:
                task = reopen_for_relabel(
                    session,
                    segment,
                    actor=actor,
                    priority=float(entry.get("priority", 0.0)),
                    details=entry.get("details"),
                    note=args.note,
                )
            except LabelingError as exc:
                problems.append(str(exc))
                continue
            print(f"{external_id:<50} task {task.id}  priority {task.priority_score:.2f}")
        if problems or args.dry_run:
            session.rollback()
            for problem in problems:
                print(problem)
            if problems:
                print(f"{len(problems)} of {len(entries)} could not be reopened; nothing reopened")
                return 1
            print(f"DRY RUN -- would reopen {len(entries)} clips")
            return 0
    print(f"reopened {len(entries)} clips into triage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
