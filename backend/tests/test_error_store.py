"""Error files beside a model, and the breakdown read from them (docs/WER-Breakdown.md, 4-5)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from app.services import error_mining
from app.services.error_store import (
    MAX_FILE_BYTES,
    ErrorFileError,
    accept_uploads,
    list_files,
)


def _clip(clip_id: str, ref: str, hyp: str, group: str = "g1", share: float | None = 0.0):
    return {"clip_id": clip_id, "group": group, "ref": ref, "hyp": hyp, "overlap_share": share}


def _file(tmp_path: Path, run: str, set_name: str, clips=None, name: str | None = None) -> bytes:
    clips = clips or [_clip("c1", "म घर जान्छु", "म घर जान्छु")]
    path = tmp_path / (name or f"{run}-{set_name}.parquet")
    error_mining.write(error_mining.rows(run, set_name, clips), path)
    return path.read_bytes()


# --- arrival (step 4) ----------------------------------------------------------------------------


def test_an_upload_is_stored_under_its_set_whatever_the_file_was_called(tmp_path: Path) -> None:
    model = tmp_path / "model"
    data = _file(tmp_path, "vanilla-s1", "fleurs")
    stored = accept_uploads(model, [("../../evil name.parquet", data)])
    assert [(f.set, f.run) for f in stored] == [("fleurs", "vanilla-s1")]
    assert sorted(p.name for p in (model / "errors").iterdir()) == ["fleurs.parquet"]


def test_files_of_one_upload_must_name_one_run(tmp_path: Path) -> None:
    model = tmp_path / "model"
    uploads = [
        ("a.parquet", _file(tmp_path, "vanilla-s1", "fleurs")),
        ("b.parquet", _file(tmp_path, "vanilla-s0", "gold")),
    ]
    with pytest.raises(ErrorFileError, match="one run"):
        accept_uploads(model, uploads)
    assert not (model / "errors").exists() or not list((model / "errors").iterdir())


def test_a_file_from_another_run_than_the_folders_other_files_is_refused(tmp_path: Path) -> None:
    model = tmp_path / "model"
    accept_uploads(model, [("a.parquet", _file(tmp_path, "vanilla-s1", "fleurs"))])
    with pytest.raises(ErrorFileError, match="vanilla-s1"):
        accept_uploads(model, [("b.parquet", _file(tmp_path, "vanilla-s0", "gold"))])
    # Replacing the only file it disagrees with is allowed.
    accept_uploads(model, [("c.parquet", _file(tmp_path, "vanilla-s0", "fleurs"))])
    assert [f.run for f in list_files(model)[0]] == ["vanilla-s0"]


def test_two_files_for_one_set_in_one_upload_are_refused(tmp_path: Path) -> None:
    data = _file(tmp_path, "vanilla-s1", "fleurs")
    with pytest.raises(ErrorFileError, match="fleurs"):
        accept_uploads(tmp_path / "model", [("a.parquet", data), ("b.parquet", data)])


def test_an_oversized_or_malformed_file_refuses_the_whole_upload(tmp_path: Path) -> None:
    model = tmp_path / "model"
    good = ("a.parquet", _file(tmp_path, "vanilla-s1", "fleurs"))
    with pytest.raises(ErrorFileError, match="50 MB"):
        accept_uploads(model, [good, ("big.parquet", b"0" * (MAX_FILE_BYTES + 1))])
    with pytest.raises(ErrorFileError, match="not a Parquet"):
        accept_uploads(model, [good, ("bad.parquet", b"not parquet")])
    with pytest.raises(ErrorFileError, match="no file"):
        accept_uploads(model, [])
    assert not list((model / "errors").glob("*"))


def test_an_unknown_set_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    source = tmp_path / "src.parquet"
    source.write_bytes(_file(tmp_path, "vanilla-s1", "fleurs"))
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{source}')) TO '{path}' (FORMAT parquet, KV_METADATA "
        "{run: 'vanilla-s1', set: 'librispeech', fold_version: 'x', "
        f"miner_version: '{error_mining.MINER_VERSION}', created_at: 'x'}})"
    )
    with pytest.raises(ErrorFileError, match="librispeech"):
        accept_uploads(tmp_path / "model", [("x.parquet", path.read_bytes())])


def test_listing_names_each_file_and_sets_aside_what_it_cannot_read(tmp_path: Path) -> None:
    model = tmp_path / "model"
    accept_uploads(model, [("a.parquet", _file(tmp_path, "vanilla-s1", "gold"))])
    (model / "errors" / "val.parquet").write_bytes(b"broken")
    (model / "errors" / "notes.txt").write_text("ignored")
    files, refused = list_files(model)
    assert [f.set for f in files] == ["gold"]
    assert files[0].path == model / "errors" / "gold.parquet"
    assert files[0].miner_version == error_mining.MINER_VERSION
    assert [name for name, _ in refused] == ["val.parquet"]
    assert list_files(tmp_path / "no-such-model") == ([], [])


def test_a_file_whose_name_disagrees_with_its_set_is_set_aside(tmp_path: Path) -> None:
    model = tmp_path / "model"
    (model / "errors").mkdir(parents=True)
    (model / "errors" / "val.parquet").write_bytes(_file(tmp_path, "vanilla-s1", "gold"))
    files, refused = list_files(model)
    assert files == []
    assert "gold" in refused[0][1]
