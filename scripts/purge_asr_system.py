#!/usr/bin/env python
"""Permanently delete one ASR system's hypotheses from the corpus.

python scripts/purge_asr_system.py --system-id gemini-3.5-transcribe --dry-run
python scripts/purge_asr_system.py --system-id gemini-3.5-transcribe --yes

This overrides D6 -- hypotheses are immutable -- so it refuses to run without --yes or an
interactive confirmation, and it writes every row to a JSONL dump before deleting anything.
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.purge import PurgedSystemNotFound, purge_asr_system


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system-id", required=True, help="asr_systems.system_id to remove")
    parser.add_argument("--dump-dir", help="where to write the pre-delete dump")
    parser.add_argument("--actor", default="owner", help="recorded on the audit entry")
    parser.add_argument(
        "--no-rescore",
        action="store_true",
        help="leave segment_scores as they are (they will describe hypotheses that are gone)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be removed, delete nothing"
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    settings = bootstrap()
    dump_dir = args.dump_dir or settings.export.output_root

    with session_scope() as session:
        try:
            preview = purge_asr_system(
                session, args.system_id, dump_dir=dump_dir, dry_run=True, actor=args.actor
            )
        except PurgedSystemNotFound as exc:
            print(f"Error: {exc}")
            return 1

        print(f"System:      {preview.system_id}")
        print(f"Hypotheses:  {preview.hypotheses_deleted}")
        print(f"Word spans:  {preview.words_deleted}")
        print(f"Segments:    {len(preview.segments_affected)}")
        for external_id in preview.segments_affected[:10]:
            print(f"  - {external_id}")
        if len(preview.segments_affected) > 10:
            print(f"  ... and {len(preview.segments_affected) - 10} more")

        if args.dry_run:
            print("\nDry run: nothing was deleted.")
            return 0

        if not args.yes:
            print("\nThis is irreversible. Hypotheses are immutable by D6; this overrides that.")
            answer = input(f"Type the system id to confirm [{preview.system_id}]: ").strip()
            if answer != preview.system_id:
                print("Aborted.")
                return 1

        report = purge_asr_system(
            session,
            args.system_id,
            dump_dir=dump_dir,
            rescore=not args.no_rescore,
            actor=args.actor,
        )

    print(f"\nDumped to:   {report.dump_path}")
    print(f"Deleted:     {report.hypotheses_deleted} hypotheses, {report.words_deleted} words")
    print(f"Rescored:    {report.segments_rescored} segments")
    if args.no_rescore:
        print("Scores left untouched: they still describe the deleted hypothesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
