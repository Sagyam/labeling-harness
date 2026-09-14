#!/usr/bin/env python
"""Import fine-tuned models and their gold/val transcripts from their folders (D83).

python scripts/import_models.py                          # every folder under data/models/asr/
python scripts/import_models.py data/models/asr/flex-ft  # one model
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.model_import import ModelImportError, import_model_dir, scan_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, nargs="?", help="one model folder; default: all")
    parser.add_argument("--actor", default="import_models", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    settings = bootstrap()
    try:
        with session_scope() as session:
            if args.folder is not None:
                report = import_model_dir(session, args.folder, settings=settings, actor=args.actor)
            else:
                report = scan_models(
                    session, settings.models.root, settings=settings, actor=args.actor
                )
    except ModelImportError as exc:
        print(f"refused, nothing imported: {exc}")
        return 1
    print(
        f"{len(report.models)} model(s); {report.runs_created} run(s) imported, "
        f"{report.runs_unchanged} already present, {report.skipped} clip(s) skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
