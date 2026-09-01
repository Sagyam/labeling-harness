"""SHA-256 helpers.

Checksums are stored and compared in the canonical form ``sha256:<64 lowercase hex chars>``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Canonical checksum of a byte string."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path | str) -> str:
    """Canonical checksum of a file, read in chunks so large clips do not load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def normalize_checksum(value: str) -> str:
    """Return ``value`` in canonical form.

    Accepts a bare hex digest or one prefixed with ``sha256:`` in any case.

    Raises:
        ValueError: The value is not a SHA-256 digest.
    """
    candidate = value.strip()
    if ":" in candidate:
        algorithm, _, candidate = candidate.partition(":")
        if algorithm.lower() != "sha256":
            raise ValueError(f"unsupported checksum algorithm: {algorithm!r}")
    candidate = candidate.lower()
    if not _HEX64.match(candidate):
        raise ValueError(f"not a sha256 checksum: {value!r}")
    return f"sha256:{candidate}"


def checksums_match(left: str | None, right: str | None) -> bool:
    """Whether two checksums are equal, ignoring prefix and case. ``None`` never matches."""
    if not left or not right:
        return False
    try:
        return normalize_checksum(left) == normalize_checksum(right)
    except ValueError:
        return False
