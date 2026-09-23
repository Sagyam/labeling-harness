#!/usr/bin/env python
"""Queue labelled clips for the multitrack editor, most crosstalk first (D98).

python scripts/queue_speakers.py --limit 30 --dry-run
python scripts/queue_speakers.py --pot gold --limit 30
python scripts/queue_speakers.py --min-crosstalk 0.0001 --min-voices 1 --limit 0   # every clip
python scripts/queue_speakers.py --reopen <segment external id>   # a clip saved by mistake
"""

from __future__ import annotations

import argparse

import sqlalchemy as sa

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.models import Segment
from app.services.labeling import LabelingError
from app.services.speaker_attribution import (
    queue_for_speakers,
    rank_for_speakers,
    reopen_for_speakers,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pot", choices=("gold", "train"), default="gold")
    parser.add_argument("--limit", type=int, default=30, help="how many clips; 0 for all")
    parser.add_argument("--min-crosstalk", type=float, help="lowest crosstalk share, 0-1")
    parser.add_argument("--max-crosstalk", type=float, help="highest crosstalk share, 0-1")
    parser.add_argument(
        "--min-voices", type=int, default=2, help="diarized speakers the clip must have"
    )
    parser.add_argument("--reopen", metavar="EXTERNAL_ID", help="reopen one saved clip")
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    args = parser.parse_args(argv)

    settings = bootstrap()
    actor = settings.labels.default_annotator

    if args.reopen:
        with session_scope() as session:
            segment = session.scalar(sa.select(Segment).where(Segment.external_id == args.reopen))
            if segment is None:
                print(f"no clip {args.reopen!r}")
                return 1
            try:
                task = reopen_for_speakers(session, segment, actor=actor)
            except LabelingError as exc:
                print(exc)
                return 1
            if args.dry_run:
                session.rollback()
                print(f"DRY RUN -- would reopen {segment.external_id}")
                return 0
            print(f"reopened {segment.external_id} as task {task.id}")
        return 0

    with session_scope() as session:
        ranked = rank_for_speakers(
            session,
            pot=args.pot,
            settings=settings,
            min_overlap=args.min_crosstalk,
            max_overlap=args.max_crosstalk,
            min_clip_speakers=args.min_voices,
        )
        chosen = ranked if args.limit <= 0 else ranked[: args.limit]
        report = queue_for_speakers(session, chosen, actor=actor, dry_run=args.dry_run)
        for candidate in chosen:
            share = candidate.overlap_share
            print(
                f"{candidate.segment.external_id:<40} crosstalk "
                f"{'--' if share is None else f'{share:.0%}':>4}  "
                f"voices {candidate.clip_speakers}"
            )
        print(f"{len(ranked)} {args.pot} clips qualify")
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
