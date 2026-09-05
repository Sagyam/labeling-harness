#!/usr/bin/env python
"""Place every episode in the gold or train pot, against a duration target (D63).

python scripts/assign_pots.py --dry-run
python scripts/assign_pots.py --gold-hours 8

Ingestion runs this itself, so the script is for what ingestion cannot cover: raising the gold
target after the fact, placing episodes that arrived through import_manifest.py, or just seeing
what the assigner would do before letting it.
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.pots import PotError, assign_pots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gold-hours",
        type=float,
        help="hours the gold pot should hold (default: dataset.gold_hours_target)",
    )
    parser.add_argument(
        "--gold-max-corpus-fraction",
        type=float,
        help=(
            "ceiling on gold's share of the ingested corpus, so a corpus smaller than the target"
            " is not swallowed whole (default: dataset.gold_max_corpus_fraction)"
        ),
    )
    parser.add_argument(
        "--allow-promote-from-train",
        action="store_true",
        help=(
            "let an episode already in the train pot move into gold. Leave this off once anything"
            " has been trained -- promoting a recording the model has seen turns the benchmark"
            " into a memorization test, and no number downstream would show it"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    args = parser.parse_args(argv)

    settings = bootstrap()

    with session_scope() as session:
        try:
            report = assign_pots(
                session,
                settings=settings,
                gold_hours_target=args.gold_hours,
                gold_max_corpus_fraction=args.gold_max_corpus_fraction,
                allow_promote_from_train=args.allow_promote_from_train,
                dry_run=args.dry_run,
            )
        except PotError as exc:
            parser.error(str(exc))
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
