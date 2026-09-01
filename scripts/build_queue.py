#!/usr/bin/env python
"""Build or refresh the annotation queue.

python scripts/build_queue.py
python scripts/build_queue.py --episode show-a_ep012 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings, load_dotenv
from app.db.session import session_scope
from app.services.queue_builder import build_queue
from app.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", help="restrict to one episode (external id)")
    parser.add_argument("--audit-rate", type=float, help="fraction of easy segments to audit")
    parser.add_argument("--audit-seed", type=int, help="seed for reproducible audit sampling")
    parser.add_argument(
        "--requeue-done", action="store_true", help="also queue segments already labeled"
    )
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    args = parser.parse_args(argv)

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level, json_output=False)

    with session_scope() as session:
        report = build_queue(
            session,
            settings=settings,
            episode_external_id=args.episode,
            audit_sample_rate=args.audit_rate,
            audit_seed=args.audit_seed,
            requeue_done=args.requeue_done,
            dry_run=args.dry_run,
        )
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
