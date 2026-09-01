#!/usr/bin/env python
"""Export a reproducible dataset.

python scripts/export_dataset.py --kind training
python scripts/export_dataset.py --kind gold --label-version v1
python scripts/export_dataset.py --kind all --output-root ./exports/2026-09-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings, load_dotenv
from app.db.session import session_scope
from app.services.export import EXPORT_KINDS, ExportError, export_dataset
from app.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", default="training", choices=[*sorted(EXPORT_KINDS), "all"])
    parser.add_argument("--label-version", help="defaults to the configured label version")
    parser.add_argument("--episode", help="restrict to one episode (external id)")
    parser.add_argument("--output-root", type=Path, help="defaults to the configured export root")
    args = parser.parse_args(argv)

    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level, json_output=False)

    kinds = sorted(EXPORT_KINDS) if args.kind == "all" else [args.kind]
    try:
        with session_scope() as session:
            for kind in kinds:
                result = export_dataset(
                    session,
                    kind=kind,
                    output_root=args.output_root,
                    label_version=args.label_version,
                    episode=args.episode,
                    settings=settings,
                )
                print(result.render())
    except ExportError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
