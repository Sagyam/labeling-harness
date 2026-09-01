"""Object storage interface.

Clips and waveform peaks are addressed by opaque string keys such as
``clips/show-a_ep012/show-a_ep012_0042.flac``. The harness must run with MinIO stopped, so every
method here has a local-filesystem implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageError(RuntimeError):
    """Base class for storage failures."""


class ObjectNotFound(StorageError):
    """The requested key does not exist."""


class ObjectStorage(ABC):
    """Storage adapter interface."""

    @abstractmethod
    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Store ``data`` at ``key`` and return the key."""

    @abstractmethod
    def put_file(
        self, key: str, path: Path, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Store the file at ``path`` under ``key`` and return the key."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Return the full object.

        Raises:
            ObjectNotFound: No object is stored at ``key``.
        """

    @abstractmethod
    def read_range(self, key: str, start: int, end: int) -> bytes:
        """Return bytes ``start``..``end`` inclusive, clamped to the object size.

        HTTP range requests map directly onto this, which is what lets the player seek without
        downloading the whole clip.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether an object is stored at ``key``."""

    @abstractmethod
    def size(self, key: str) -> int:
        """Object size in bytes.

        Raises:
            ObjectNotFound: No object is stored at ``key``.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` if present. Idempotent."""

    @abstractmethod
    def presigned_url(self, key: str, *, expires_seconds: int = 3600) -> str | None:
        """A time-limited URL for direct client access, or ``None`` if unsupported."""

    @abstractmethod
    def healthcheck(self) -> None:
        """Raise if the backend is not usable."""


def validate_key(key: str) -> str:
    """Reject absolute keys and traversal segments before they reach a filesystem path."""
    if not key or key.startswith("/") or key.startswith("\\"):
        raise ValueError(f"invalid object key (must be relative): {key!r}")
    parts = Path(key).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"invalid object key (traversal): {key!r}")
    return key
