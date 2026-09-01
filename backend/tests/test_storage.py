"""Tests for the object storage adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.storage import build_storage, get_storage
from app.storage.base import ObjectNotFound
from app.storage.local import LocalFilesystemStorage


@pytest.fixture
def local_storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path)


def test_put_and_get_round_trip(local_storage: LocalFilesystemStorage) -> None:
    local_storage.put_bytes("clips/a.flac", b"abc", content_type="audio/flac")
    assert local_storage.get_bytes("clips/a.flac") == b"abc"


def test_put_file_round_trip(local_storage: LocalFilesystemStorage, tmp_path: Path) -> None:
    src = tmp_path / "src.flac"
    src.write_bytes(b"0123456789")
    key = local_storage.put_file("clips/b.flac", src, content_type="audio/flac")
    assert key == "clips/b.flac"
    assert local_storage.get_bytes(key) == b"0123456789"


def test_exists_and_size(local_storage: LocalFilesystemStorage) -> None:
    assert local_storage.exists("nope") is False
    local_storage.put_bytes("k", b"12345")
    assert local_storage.exists("k") is True
    assert local_storage.size("k") == 5


def test_get_missing_object_raises(local_storage: LocalFilesystemStorage) -> None:
    with pytest.raises(ObjectNotFound):
        local_storage.get_bytes("missing")


def test_read_range_returns_requested_bytes(local_storage: LocalFilesystemStorage) -> None:
    local_storage.put_bytes("k", b"0123456789")
    assert local_storage.read_range("k", 2, 5) == b"2345"


def test_read_range_clamps_to_object_size(local_storage: LocalFilesystemStorage) -> None:
    local_storage.put_bytes("k", b"0123456789")
    assert local_storage.read_range("k", 8, 99) == b"89"


def test_delete_is_idempotent(local_storage: LocalFilesystemStorage) -> None:
    local_storage.put_bytes("k", b"x")
    local_storage.delete("k")
    local_storage.delete("k")
    assert local_storage.exists("k") is False


def test_keys_cannot_escape_the_root(local_storage: LocalFilesystemStorage) -> None:
    with pytest.raises(ValueError, match="key"):
        local_storage.put_bytes("../escape", b"x")
    with pytest.raises(ValueError, match="key"):
        local_storage.get_bytes("/absolute")


def test_healthcheck_creates_root(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(root=tmp_path / "nested" / "root")
    storage.healthcheck()
    assert (tmp_path / "nested" / "root").is_dir()


def test_build_storage_returns_local_backend(tmp_path: Path) -> None:
    from app.config import load_settings

    settings = load_settings()
    storage = build_storage(settings)
    assert isinstance(storage, LocalFilesystemStorage)


def test_get_storage_is_cached() -> None:
    assert get_storage() is get_storage()


@pytest.mark.minio
def test_minio_round_trip_and_range() -> None:
    from app.config import load_settings
    from app.storage.minio_storage import MinioStorage

    settings = load_settings()
    minio_cfg = settings.storage.minio.model_copy(
        update={
            "access_key": os.environ.get("HARNESS_STORAGE__MINIO__ACCESS_KEY", "minioadmin"),
            "secret_key": os.environ.get("HARNESS_STORAGE__MINIO__SECRET_KEY", "minioadmin"),
        }
    )
    try:
        storage = MinioStorage(minio_cfg)
        storage.healthcheck()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"MinIO unavailable: {exc}")

    storage.put_bytes("tests/range.bin", b"0123456789", content_type="application/octet-stream")
    assert storage.get_bytes("tests/range.bin") == b"0123456789"
    assert storage.read_range("tests/range.bin", 2, 5) == b"2345"
    assert storage.size("tests/range.bin") == 10
    storage.delete("tests/range.bin")
    assert storage.exists("tests/range.bin") is False


def test_delete_objects_skips_missing_keys(local_storage: LocalFilesystemStorage) -> None:
    from app.storage import delete_objects

    local_storage.put_bytes("clip", b"x")
    delete_objects(local_storage, "clip", None)
    assert local_storage.exists("clip") is False


def test_delete_objects_logs_a_failure_instead_of_aborting(
    local_storage: LocalFilesystemStorage,
) -> None:
    """A storage failure must not stop the database delete the user actually asked for."""

    class Failing(LocalFilesystemStorage):
        def delete(self, key: str) -> None:
            raise OSError("bucket unreachable")

    from app.storage import delete_objects

    delete_objects(Failing(root=local_storage.root), "clip", "peaks")
