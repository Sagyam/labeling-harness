#!/usr/bin/env python
"""Import an upstream export directory into the harness.

    python scripts/import_manifest.py export_show-a_ep012/
    python scripts/import_manifest.py export_show-a_ep012/ --dry-run
    python scripts/import_manifest.py export_show-a_ep012/ --allow-clip-change

Exit codes: 0 success, 1 import rejected, 2 bad usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings, load_dotenv
from app.db.session import session_scope
from app.services.importer import ImportError_, import_manifest
from app.storage import build_storage
from app.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path, help="export_<episode_id>/ directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="report planned changes and write nothing"
    )
    parser.add_argument(
        "--allow-clip-change",
        action="store_true",
        help="accept clips whose checksum differs from what was imported",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level, json_output=args.json)

    storage = build_storage(settings)
    try:
        with session_scope() as session:
            report = import_manifest(
                session,
                args.export_dir,
                storage=storage,
                settings=settings,
                dry_run=args.dry_run,
                allow_clip_change=args.allow_clip_change,
            )
    except ImportError_ as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        import dataclasses
        import json

        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
