"""Tests for the command-line entry points.

The scripts are how the harness is actually operated between sessions, so their argument
parsing, exit codes and output formats are worth the same scrutiny as the services they call.
Each runs against the test database through the ``DATABASE_URL`` the fixtures set.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.db


@pytest.fixture(scope="module", autouse=True)
def _isolate_module(db_engine) -> Any:
    """Keep this module hermetic.

    Unlike the rest of the suite these tests cannot run inside a rolled-back transaction: the
    scripts open their own sessions and commit for real. So the tables are emptied on the way in
    and on the way out, and logging -- which the scripts reconfigure, binding it to whatever
    stdout is current -- is rebound afterwards so later modules do not write to a dead capture
    buffer.
    """
    import sqlalchemy as sa

    from app.db.base import Base
    from app.utils.logging import configure_logging

    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

    def truncate() -> None:
        tables = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
        with db_engine.begin() as conn:
            conn.execute(sa.text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))

    truncate()
    yield
    truncate()
    configure_logging()


def load(name: str) -> Any:
    """Import a script module fresh, so each test sees its own argument parsing."""
    module = importlib.import_module(name)
    return importlib.reload(module)


@pytest.fixture
def cli(db_engine, monkeypatch: pytest.MonkeyPatch):
    """Point the scripts' own session_scope at the test database."""
    from app.db import session as session_module

    monkeypatch.setattr(session_module, "get_engine", lambda: db_engine)
    return db_engine


def test_bootstrap_returns_settings_and_configures_logging() -> None:
    from _bootstrap import bootstrap

    settings = bootstrap()
    assert settings.app.name
    assert settings.storage.local_root.is_absolute()


def test_seed_script_inserts_and_is_idempotent(cli, capsys: pytest.CaptureFixture[str]) -> None:
    script = load("seed_dev_data")
    assert script.main(["--episodes", "1", "--segments", "3", "--systems", "2"]) == 0
    assert script.main(["--episodes", "1", "--segments", "3", "--systems", "2"]) == 0


def test_build_queue_script_reports_what_it_did(cli, capsys: pytest.CaptureFixture[str]) -> None:
    load("seed_dev_data").main(["--episodes", "1", "--segments", "3", "--systems", "2"])
    capsys.readouterr()

    assert load("build_queue").main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "segments considered" in out


def test_build_queue_script_accepts_an_episode_filter(cli) -> None:
    assert load("build_queue").main(["--episode", "does-not-exist", "--dry-run"]) == 0


def test_export_script_writes_the_requested_kind(cli, tmp_path: Path) -> None:
    script = load("export_dataset")
    assert script.main(["--kind", "training", "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "training" / "training.jsonl").is_file()
    assert (tmp_path / "training" / "manifest.json").is_file()


def test_export_script_can_write_every_kind(cli, tmp_path: Path) -> None:
    assert load("export_dataset").main(["--kind", "all", "--output-root", str(tmp_path)]) == 0
    for kind in ("training", "gold", "analytics", "error_mining"):
        assert (tmp_path / kind / f"{kind}.jsonl").is_file()


def test_export_script_rejects_an_unknown_kind(cli, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load("export_dataset").main(["--kind", "nonsense", "--output-root", str(tmp_path)])


def test_report_script_renders_text(cli, capsys: pytest.CaptureFixture[str]) -> None:
    assert load("report_status").main([]) == 0
    assert "CORPUS" in capsys.readouterr().out


def test_report_script_renders_json(cli, capsys: pytest.CaptureFixture[str]) -> None:
    assert load("report_status").main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "corpus" in payload
    assert "accept_rate_by_day" in payload


def test_report_script_writes_html_to_a_file(cli, tmp_path: Path) -> None:
    out = tmp_path / "status.html"
    assert load("report_status").main(["--format", "html", "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_import_script_rejects_a_missing_export_directory(
    cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert load("import_manifest").main([str(tmp_path / "nope")]) == 1
    assert "import failed" in capsys.readouterr().err


def test_import_script_imports_a_fixture_export(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.services.fixtures import build_export_fixture
    from app.storage.local import LocalFilesystemStorage

    script = load("import_manifest")
    monkeypatch.setattr(
        script, "build_storage", lambda settings: LocalFilesystemStorage(root=tmp_path / "objects")
    )
    root = build_export_fixture(tmp_path / "export_cli", episode_id="cli_ep000", segments=2)

    assert script.main([str(root), "--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert script.main([str(root)]) == 0
    assert "cli_ep000" in capsys.readouterr().out
