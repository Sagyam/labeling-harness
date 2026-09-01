"""Object storage adapters."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.storage.base import ObjectNotFound, ObjectStorage, StorageError
from app.storage.local import LocalFilesystemStorage
from app.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "LocalFilesystemStorage",
    "ObjectNotFound",
    "ObjectStorage",
    "StorageError",
    "build_storage",
    "delete_objects",
    "get_storage",
]


def delete_objects(storage: ObjectStorage, *keys: str | None) -> None:
    """Best-effort delete of several keys, skipping ``None``.

    A storage failure must not abort the database delete that follows -- the row is what the user
    asked to remove -- but it does leave an orphaned object, so it is logged rather than silently
    swallowed.
    """
    for key in keys:
        if not key:
            continue
        try:
            storage.delete(key)
        except Exception as exc:
            logger.warning("storage_delete_failed", key=key, error=str(exc)[:200])


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
