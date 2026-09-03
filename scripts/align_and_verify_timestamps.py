#!/usr/bin/env python3
"""Acoustic Timestamp Cross-Verification CLI.

Thin wrapper around app.services.alignment for standalone verification of word-level
acoustic timestamps against ground-truth tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add backend directory to sys.path so app can be imported
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.alignment import format_verification_report, run_cross_verification


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-verify word-level acoustic boundaries and generate audit triage queue."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("exports/analytics/analytics.jsonl"),
        help="Path to analytics.jsonl containing hypotheses and words.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=200.0,
        help="Boundary divergence threshold in ms to flag for human review.",
    )
    parser.add_argument(
        "--output-report",
        "-o",
        type=Path,
        default=Path("reports/timestamp_verification_report.json"),
        help="Path to save structured JSON verification report.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist. Run export first.")
        return 1

    summary = run_cross_verification(args.input, divergence_threshold_ms=args.threshold)
    print(format_verification_report(summary))

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured verification report to: {args.output_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
