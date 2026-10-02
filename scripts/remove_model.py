#!/usr/bin/env python
"""Remove imported models: their runs and clips from the database, and their folders.

The folder goes too, or the next Rescan imports the model again.

python scripts/remove_model.py indic-transcribe-flex-ft-2026-09-16
python scripts/remove_model.py old-a old-b --keep-folder
"""

from __future__ import annotations

import argparse
import shutil

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.model_import import ModelImportError, remove_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="+", help="model folder names under models.root")
    parser.add_argument("--keep-folder", action="store_true", help="delete the rows only")
    parser.add_argument("--actor", default="remove_model", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    settings = bootstrap()
    with session_scope() as session:
        try:
            removed = {s: remove_model(session, s, actor=args.actor) for s in args.slugs}
        except ModelImportError as exc:
            print(f"refused: {exc}; nothing removed")
            return 1
    for slug, r in removed.items():
        folder = settings.models.root / slug
        if not args.keep_folder and folder.is_dir():
            shutil.rmtree(folder)
        kept = " (folder kept)" if args.keep_folder else ""
        print(f"{slug}: removed {r.runs} run(s), {r.clips} clips{kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
