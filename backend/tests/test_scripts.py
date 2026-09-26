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


def test_queue_speakers_script_reports_what_it_would_queue(
    cli, capsys: pytest.CaptureFixture[str]
) -> None:
    assert load("queue_speakers").main(["--limit", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "gold clips qualify" in out
    assert "DRY RUN -- speakers queue" in out


def test_queue_speakers_script_reports_an_unknown_clip_to_reopen(
    cli, capsys: pytest.CaptureFixture[str]
) -> None:
    assert load("queue_speakers").main(["--reopen", "no-such-clip"]) == 1
    assert "no clip" in capsys.readouterr().out


def test_voiceprint_report_script_runs_on_an_empty_corpus(
    cli, capsys: pytest.CaptureFixture[str]
) -> None:
    assert load("voiceprint_report").main([]) == 0
    assert "saved with a suggestion" in capsys.readouterr().out


@pytest.fixture
def no_aligner_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The export script builds an aligner for edited labels; the suite never fetches its 317 MB
    model, and without it those rows simply carry no label words."""
    monkeypatch.setenv("HARNESS_ALIGNER_NO_DOWNLOAD", "1")


def test_export_script_writes_the_requested_kind(cli, tmp_path: Path, no_aligner_fetch) -> None:
    script = load("export_dataset")
    assert script.main(["--kind", "training", "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "training" / "training.jsonl").is_file()
    assert (tmp_path / "training" / "manifest.json").is_file()


def test_export_script_can_write_every_kind(cli, tmp_path: Path, no_aligner_fetch) -> None:
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


def test_diarize_script_stores_a_run_and_reports_what_it_could_not_do(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sqlalchemy as sa
    from sqlalchemy.orm import Session

    from app.models import DiarizationRun, Episode
    from app.storage.local import LocalFilesystemStorage

    storage = LocalFilesystemStorage(root=tmp_path / "objects")
    key = storage.put_bytes("episodes/cli_diar/audio.flac", b"flac bytes")
    with Session(cli) as session:
        session.add_all(
            [
                Episode(
                    external_id="cli_diar",
                    audio_object_key=key,
                    metadata_jsonb={"speakers": {"host": {}, "guest": {}}},
                ),
                Episode(external_id="cli_no_audio"),
            ]
        )
        session.commit()

    sent: dict[str, Any] = {}

    def fake_diarize(audio, *, num_speakers, settings):
        sent.update(audio=audio, num_speakers=num_speakers)
        return {
            "turns": [[0.0, 1.0, "SPEAKER_00"], [1.0, 2.0, "SPEAKER_01"]],
            "labels": ["SPEAKER_00", "SPEAKER_01"],
        }

    script = load("diarize_episode")
    monkeypatch.setattr(script, "build_storage", lambda settings: storage)
    monkeypatch.setattr(script, "diarize_audio", fake_diarize)

    assert script.main(["cli_diar", "cli_no_audio", "cli_missing"]) == 1
    out = capsys.readouterr().out
    assert sent == {"audio": b"flac bytes", "num_speakers": 2}, "declared speakers by default"
    assert "cli_diar: 2 speakers (asked 2), 2 turns, stored" in out
    assert "cli_no_audio: no retained audio" in out
    assert "cli_missing: not found" in out
    with Session(cli) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(DiarizationRun)) == 1

    assert script.main(["cli_diar", "--num-speakers", "3"]) == 0
    assert sent["num_speakers"] == 3
    assert "unchanged, already stored" in capsys.readouterr().out

    # A field report interviews passers-by nobody declared: let pyannote count them.
    assert script.main(["cli_diar", "--auto"]) == 0
    assert sent["num_speakers"] is None
    assert "(asked auto)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        script.main(["cli_diar", "--auto", "--num-speakers", "2"])


def test_models_script_imports_one_folder_and_refuses_a_bad_card(
    cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad = tmp_path / "flex-ft", tmp_path / "broken"
    good.mkdir()
    bad.mkdir()
    (good / "model_card.json").write_text(json.dumps({"name": "Flex FT"}), encoding="utf-8")
    (bad / "model_card.json").write_text("{}", encoding="utf-8")
    script = load("import_models")
    assert script.main([str(good)]) == 0
    assert "1 model(s); 0 run(s) imported" in capsys.readouterr().out
    assert script.main([str(bad)]) == 1
    assert "refused" in capsys.readouterr().out
