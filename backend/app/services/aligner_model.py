"""Fetch the forced-alignment acoustic model, so a missing file fixes itself.

``scripts/export_aligner_onnx.py`` builds this artefact from the PyTorch weights, and that path
still exists for provenance -- but it needs ``torch`` and ``transformers`` in a throwaway venv,
which is a lot to ask of anyone who just wants word spans, and impossible inside the runtime
container (D32 keeps that stack out of ``pyproject.toml`` on purpose).

The same network already publishes the exported graph, so the runtime downloads it instead of
building it. ``onnx/model_int8.onnx`` from the pinned revision below is the same int8 CTC head the
export script produces: one ``input_values`` input of shape ``[batch, samples]``, one ``logits``
output of ``[batch, frames, 31]``, and the 31-label romanized vocabulary the aligner expects. No
torch anywhere.

Both files are pinned to a **revision** and checked against a **digest**. A model is an executable
graph that runs on every clip of the corpus: pinning the branch name would let it change under a
finished benchmark, and skipping the digest would let a bad transfer produce quietly wrong spans
rather than a clean failure.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import httpx

from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Set to 1/true/yes to refuse the download and keep the old "warn and skip spans" behaviour.
DISABLE_ENV = "HARNESS_ALIGNER_NO_DOWNLOAD"

MODEL_REPO = "onnx-community/mms-300m-1130-forced-aligner-ONNX"
#: A commit, never a branch: the graph must not change under a benchmark that already ran.
MODEL_REVISION = "2100fb247d8e43962eef24491597fbeb8b469531"

MODEL_REMOTE = "onnx/model_int8.onnx"
MODEL_SHA256 = "2eb5c3d2f6db2ef476aa7a7e1e5800145973e9064eb5292b1b9b8ada1207712a"
MODEL_BYTES = 317_341_664

VOCAB_REMOTE = "vocab.json"
VOCAB_SHA256 = "0ff83fc9d044537f7c72d92db15b04917f1e50cada799957963bca61e4305971"
VOCAB_BYTES = 351

DOWNLOAD_TIMEOUT_SECONDS = 600.0


def download_disabled() -> bool:
    """Whether the environment has switched the fetch off."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _fetch(remote: str, destination: Path, expected_sha: str, timeout: float) -> None:
    """Stream one file into place, atomically, or leave nothing behind.

    The download lands on a temporary file beside the destination and is renamed only after its
    digest matches, so an interrupted transfer can never be mistaken for a usable model on the
    next run.
    """
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{remote}"
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

        actual = _digest(temp_path)
        if actual != expected_sha:
            raise ValueError(f"digest mismatch for {remote}: expected {expected_sha}, got {actual}")
        temp_path.replace(destination)
        # mkstemp creates 0600, and in the container this lands in a bind mount owned by root:
        # left alone, the file the host cannot read is also one the host cannot delete.
        destination.chmod(0o644)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def ensure_aligner_model(
    model_path: Path,
    vocab_path: Path,
    *,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> bool:
    """Make sure both aligner files exist, downloading whichever is missing.

    Args:
        model_path: Where the ONNX graph belongs.
        vocab_path: Where the CTC label set belongs.
        timeout: Per-request timeout, generous because the model is ~317 MB.

    Returns:
        True when both files are present afterwards. False is not an error: the caller degrades
        to "no word spans", exactly as it does for an absent VAD model (D32).
    """
    wanted = [
        (model_path, MODEL_REMOTE, MODEL_SHA256, MODEL_BYTES),
        (vocab_path, VOCAB_REMOTE, VOCAB_SHA256, VOCAB_BYTES),
    ]
    missing = [item for item in wanted if not item[0].is_file()]
    if not missing:
        return True

    if download_disabled():
        logger.warning(
            "aligner_download_disabled",
            reason=f"{DISABLE_ENV} is set",
            missing=[str(item[0]) for item in missing],
        )
        return False

    total = sum(item[3] for item in missing)
    logger.info(
        "aligner_download_started",
        repo=MODEL_REPO,
        revision=MODEL_REVISION,
        files=len(missing),
        approx_bytes=total,
    )
    for destination, remote, expected_sha, _size in missing:
        try:
            _fetch(remote, destination, expected_sha, timeout)
            logger.info("aligner_download_file_ready", path=str(destination))
        except Exception as exc:
            logger.warning(
                "aligner_download_failed",
                file=remote,
                destination=str(destination),
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

    logger.info("aligner_download_complete", model=str(model_path), vocab=str(vocab_path))
    return True
