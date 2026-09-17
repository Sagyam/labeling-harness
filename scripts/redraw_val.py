#!/usr/bin/env python
"""Redraw every episode's train/val split, stratified by long and short form (D90).

python scripts/redraw_val.py --seed 20260917 --dry-run
python scripts/redraw_val.py --seed 20260917
"""

from __future__ import annotations

import argparse

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.config import get_settings
from app.db.session import session_scope
from app.services.pots import redraw_val


class _DryRun(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="seed for the draw")
    parser.add_argument("--share", type=float, help="val share of hours (default: val_fraction)")
    parser.add_argument("--long-form-hours", type=float, default=0.5)
    parser.add_argument("--actor", default="redraw_val", help="recorded in audit_logs")
    parser.add_argument("--dry-run", action="store_true", help="print the draw, change nothing")
    args = parser.parse_args(argv)

    bootstrap()
    share = args.share if args.share is not None else get_settings().dataset.val_fraction
    try:
        with session_scope() as session:
            report = redraw_val(
                session,
                share=share,
                seed=args.seed,
                long_form_hours=args.long_form_hours,
                actor=args.actor,
            )
            print(
                f"val {report.val_hours:.2f} h over {len(report.val)} episode(s), "
                f"train {report.train_hours:.2f} h; {len(report.changed)} changed"
            )
            for external_id in report.val:
                print("  val:", external_id)
            if args.dry_run:
                raise _DryRun
    except _DryRun:
        print("dry run: rolled back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
