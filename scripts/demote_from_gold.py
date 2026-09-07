#!/usr/bin/env python
"""Move one episode out of the gold pot and into train (D68).

python scripts/demote_from_gold.py --episode ep_602_... --reason "..." --dry-run
python scripts/demote_from_gold.py --episode ep_602_... --reason "..." --yes

This overrides D63 rule 3 -- an episode never leaves gold -- so it refuses to run without --yes or
an interactive confirmation, takes exactly one episode by name, and records the reason on an
`audit_logs` row.

It is only defensible before anything has been trained on or measured against the benchmark. Rule
3 protects two things: a recording the model has seen must not become the benchmark, and the
benchmark must not be cherry-picked once its numbers are known. Neither can happen while no number
exists. **Once one does, stop using this script.**
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.pots import PotError, demote_from_gold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", required=True, help="episodes.external_id to demote")
    parser.add_argument(
        "--reason",
        required=True,
        help="why the benchmark is shrinking; recorded on the audit row",
    )
    parser.add_argument("--actor", default="owner", help="recorded on the audit entry")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    settings = bootstrap()

    with session_scope() as session:
        try:
            preview = demote_from_gold(
                session,
                args.episode,
                reason=args.reason,
                settings=settings,
                actor=args.actor,
                dry_run=True,
            )
        except PotError as exc:
            print(f"Error: {exc}")
            return 1

        print(preview.render())

        if args.dry_run:
            print("\nDry run: nothing was written.")
            return 0

        if not args.yes:
            print(
                "\nThis overrides D63 rule 3, and is only safe while nothing has been trained on"
                "\nor measured against the gold pot."
            )
            answer = input(f"Type the episode id to confirm [{preview.external_id}]: ").strip()
            if answer != preview.external_id:
                print("Aborted.")
                return 1

        report = demote_from_gold(
            session,
            args.episode,
            reason=args.reason,
            settings=settings,
            actor=args.actor,
        )

    print()
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
