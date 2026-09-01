"""Tests for checksum helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.utils.hashing import checksums_match, normalize_checksum, sha256_bytes, sha256_file


def test_sha256_bytes_matches_hashlib() -> None:
    assert sha256_bytes(b"abc") == f"sha256:{hashlib.sha256(b'abc').hexdigest()}"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == f"sha256:{hashlib.sha256(b'hello world').hexdigest()}"


def test_normalize_accepts_bare_and_prefixed_digests() -> None:
    digest = "a" * 64
    assert normalize_checksum(digest) == f"sha256:{digest}"
    assert normalize_checksum(f"sha256:{digest}") == f"sha256:{digest}"
    assert normalize_checksum(f"SHA256:{digest.upper()}") == f"sha256:{digest}"


def test_normalize_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="checksum"):
        normalize_checksum("not-a-digest")


def test_checksums_match_ignores_prefix_and_case() -> None:
    digest = "b" * 64
    assert checksums_match(digest, f"sha256:{digest.upper()}") is True
    assert checksums_match(digest, "c" * 64) is False
    assert checksums_match(None, digest) is False
