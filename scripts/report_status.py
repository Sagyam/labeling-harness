#!/usr/bin/env python
"""Print a status report.

python scripts/report_status.py
python scripts/report_status.py --format html --output status.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.report import collect_report, render_html, render_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", default="text", choices=["text", "html", "json"])
    parser.add_argument("--output", type=Path, help="write to a file instead of stdout")
    args = parser.parse_args(argv)

    bootstrap()

    with session_scope() as session:
        report = collect_report(session)

    if args.format == "html":
        rendered = render_html(report)
    elif args.format == "json":
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        rendered = render_text(report)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
