#!/usr/bin/env python
"""Insert a small synthetic dataset for development.

python scripts/seed_dev_data.py --episodes 1 --segments 20 --systems 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import load_dotenv
from app.db.session import session_scope
from app.services.seed import seed_dev_data
from app.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--segments", type=int, default=20, help="segments per episode")
    parser.add_argument("--systems", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    load_dotenv()
    configure_logging("INFO", json_output=False)
    logger = get_logger("seed")

    with session_scope() as session:
        summary = seed_dev_data(
            session,
            episodes=args.episodes,
            segments_per_episode=args.segments,
            systems=args.systems,
            seed=args.seed,
        )
    logger.info(
        "seed_complete",
        episodes=summary.episodes,
        segments=summary.segments,
        systems=summary.systems,
        hypotheses=summary.hypotheses,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
