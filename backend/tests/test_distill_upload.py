"""What scripts/upload_distill_corpus.py sends to HF (D101): the manifest and cleared sources, and
nothing while any source is unscreened. The script runs outside the backend's virtualenv, so it is
loaded by path, like the notebook kits."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "upload_distill_corpus.py"
_spec = importlib.util.spec_from_file_location("upload_distill_corpus", _PATH)
upload = importlib.util.module_from_spec(_spec)
sys.modules["upload_distill_corpus"] = upload
_spec.loader.exec_module(upload)


def _source(root: Path, name: str, verdict: str | None) -> None:
    folder = root / "sources" / name
    (folder / "clips").mkdir(parents=True)
    if verdict:
        (folder / "screen.json").write_text(json.dumps({"verdict": verdict}), encoding="utf-8")


def test_only_the_manifest_and_cleared_sources_are_sent(tmp_path: Path) -> None:
    _source(tmp_path, "yt-aaaaaaaaaaa", "clear")
    _source(tmp_path, "yt-bbbbbbbbbbb", "quarantine")
    (tmp_path / "sources" / "yt-ccccccccccc.partial").mkdir()
    (tmp_path / "incoming").mkdir()
    assert upload.allowed_patterns(tmp_path) == ["clips.jsonl", "sources/yt-aaaaaaaaaaa/*"]


def test_nothing_is_sent_while_a_source_is_unscreened(tmp_path: Path) -> None:
    _source(tmp_path, "yt-aaaaaaaaaaa", "clear")
    _source(tmp_path, "yt-ddddddddddd", None)
    with pytest.raises(upload.NotReady, match="yt-ddddddddddd"):
        upload.allowed_patterns(tmp_path)
