"""Download one file of a model pinned to a Hugging Face revision, checked against a digest.

Shared by every model the runtime fetches for itself (the forced aligner, the overlap detector).
A model is an executable graph that runs on every clip of the corpus, so both halves of the pin
matter: a commit rather than a branch keeps the graph from changing under a finished benchmark,
and the digest turns a bad transfer into a clean failure instead of quietly wrong output.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx

from app.utils.hashing import sha256_file


def fetch_pinned(
    *,
    repo: str,
    revision: str,
    remote: str,
    destination: Path,
    expected_sha: str,
    timeout: float,
) -> None:
    """Stream ``repo@revision:remote`` into ``destination``, atomically, or leave nothing behind.

    The download lands on a temporary file beside the destination and is renamed only after its
    digest matches, so an interrupted transfer can never be mistaken for a usable model on the
    next run.

    Raises:
        httpx.HTTPError: The transfer failed.
        ValueError: The file arrived but its SHA-256 is not ``expected_sha``.
    """
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    temp_path = Path(temp_name)
    os.close(handle)

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            with temp_path.open("wb") as out:
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)

        actual = sha256_file(temp_path).removeprefix("sha256:")
        if actual != expected_sha:
            raise ValueError(f"digest mismatch for {remote}: expected {expected_sha}, got {actual}")
        temp_path.replace(destination)
        # mkstemp creates 0600, and in the container this lands in a bind mount owned by root:
        # left alone, the file the host cannot read is also one the host cannot delete.
        destination.chmod(0o644)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
