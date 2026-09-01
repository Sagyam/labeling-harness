"""Local filesystem storage: the fallback that makes MinIO optional."""

from __future__ import annotations

import shutil
from pathlib import Path

from app.storage.base import ObjectNotFound, ObjectStorage, validate_key


class LocalFilesystemStorage(ObjectStorage):
    """Stores objects as files beneath ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / validate_key(key)

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def put_file(
        self, key: str, path: Path, *, content_type: str = "application/octet-stream"
    ) -> str:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return key

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        return path.read_bytes()

    def read_range(self, key: str, start: int, end: int) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        size = path.stat().st_size
        start = max(0, start)
        end = min(end, size - 1)
        if start > end:
            return b""
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        return path.stat().st_size

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def presigned_url(self, key: str, *, expires_seconds: int = 3600) -> str | None:
        """Local files have no external URL; the API streams them instead."""
        return None

    def healthcheck(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():  # pragma: no cover - mkdir would have raised
            raise OSError(f"storage root is not a directory: {self.root}")
