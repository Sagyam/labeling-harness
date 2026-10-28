#!/usr/bin/env python
"""Create, migrate and fill a demo database with an invented corpus, for screenshots and demos.

Point every store at demo locations first, so nothing real is read or written:

    export HARNESS_DATABASE__NAME=harness_demo
    export HARNESS_STORAGE__LOCAL_ROOT=./data/demo/objects
    export HARNESS_MODELS__ROOT=./data/demo/models
    backend/.venv/bin/python scripts/seed_demo.py

It refuses any database whose name does not end in ``_demo``. Then start the backend with the
same three variables set and open the app as usual.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import sqlalchemy as sa

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import REPO_ROOT, bootstrap
from alembic import command
from alembic.config import Config
from app.db.session import session_scope
from app.services.demo_seed import DemoSeedError, require_demo_database, seed_demo
from app.storage import build_storage
from sqlalchemy.engine import make_url


def _create_and_migrate(url: str) -> None:
    """Create the database when it is missing, then bring it to the head revision."""
    target = make_url(url)
    engine = sa.create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            sa.text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target.database}
        ).scalar()
        if not exists:
            conn.execute(sa.text(f'CREATE DATABASE "{target.database}"'))
    engine.dispose()
    backend = REPO_ROOT / "backend"
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(cfg, "head")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--clips", type=int, default=24, help="clips per episode")
    args = parser.parse_args(argv)

    settings = bootstrap()
    try:
        require_demo_database(make_url(settings.database.url).database or "")
    except DemoSeedError as exc:
        print(exc)
        return 1
    _create_and_migrate(settings.database.url)
    with tempfile.TemporaryDirectory() as work, session_scope() as session:
        summary = seed_demo(
            session,
            storage=build_storage(settings),
            work_root=Path(work),
            settings=settings,
            episodes=args.episodes,
            clips_per_episode=args.clips,
        )
    if not summary.episodes:
        print("the demo database already holds episodes; nothing seeded")
    else:
        print(
            f"seeded {summary.episodes} episodes, {summary.clips} clips, "
            f"{summary.labelled} labelled"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
