"""Object storage adapters."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.storage.base import ObjectNotFound, ObjectStorage, StorageError
from app.storage.local import LocalFilesystemStorage

__all__ = [
    "LocalFilesystemStorage",
    "ObjectNotFound",
    "ObjectStorage",
    "StorageError",
    "build_storage",
    "get_storage",
]


def build_storage(settings: Settings | None = None) -> ObjectStorage:
    """Construct the configured storage backend."""
    settings = settings or get_settings()
    if settings.storage.backend == "minio":
        from app.storage.minio_storage import MinioStorage

        return MinioStorage(settings.storage.minio)
    return LocalFilesystemStorage(root=settings.storage.local_root)


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    """Process-wide storage singleton."""
    return build_storage()
