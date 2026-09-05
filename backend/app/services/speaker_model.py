"""Fetch the speaker-embedding graph, so a missing file fixes itself.

Same contract as ``aligner_model.py`` and for the same reasons: pinned to a **revision** rather
than a branch, because an embedding model is an executable graph that decides who is speaking on
every clip of the corpus and it must not change under a diarization that already ran; and checked
against a **digest**, because a truncated transfer should fail cleanly instead of producing
embeddings that cluster badly for reasons nobody can find.

The model is WeSpeaker's CAM++ trained on VoxCeleb, Apache-2.0, 29 MB, exported to ONNX by its
authors. It takes 80-bin Kaldi filterbank features (see ``app/services/fbank.py``) and returns one
512-dimensional embedding per utterance. No torch (D32).
"""

from __future__ import annotations

import os
from pathlib import Path

from app.services.aligner_model import fetch_file
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Set to 1/true/yes to refuse the download. Diarization then degrades to one speaker per episode,
#: which is exactly what the pipeline did before it existed.
DISABLE_ENV = "HARNESS_SPEAKER_NO_DOWNLOAD"

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-campplus-LM"
#: A commit, never a branch.
MODEL_REVISION = "c5e01c6fcffcce160861e7e79782828320192b5c"

MODEL_REMOTE = "voxceleb_CAM++_LM.onnx"
MODEL_SHA256 = "1068e4ac3a76bb9c769e6816ef30bf89363f6e966f1d938210cb8ed4038f8e93"
MODEL_BYTES = 29_292_449

DOWNLOAD_TIMEOUT_SECONDS = 300.0

#: Same convention as the aligner: the file lives beside the other ONNX graphs by default, and the
#: container points this at the mounted `/app/data` volume so an `up` does not re-download it.
MODEL_DIR_ENV = "HARNESS_SPEAKER_MODEL_DIR"
DEFAULT_MODEL_DIR = Path(
    os.environ.get(MODEL_DIR_ENV) or (Path(__file__).resolve().parent / "models")
)
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "campplus.onnx"


def download_disabled() -> bool:
    """Whether the environment has switched the fetch off."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def ensure_speaker_model(model_path: Path, *, timeout: float = DOWNLOAD_TIMEOUT_SECONDS) -> bool:
    """Make sure the embedding graph exists, downloading it if not.

    Args:
        model_path: Where the ONNX graph belongs.
        timeout: Request timeout, generous because the model is ~29 MB.

    Returns:
        True when the file is present afterwards. False is not an error: the caller degrades to a
        single speaker for the episode, exactly as it does for an absent aligner (D32).
    """
    if model_path.is_file():
        return True

    if download_disabled():
        logger.warning(
            "speaker_download_disabled", reason=f"{DISABLE_ENV} is set", path=str(model_path)
        )
        return False

    logger.info(
        "speaker_download_started",
        repo=MODEL_REPO,
        revision=MODEL_REVISION,
        approx_bytes=MODEL_BYTES,
    )
    try:
        fetch_file(
            MODEL_REPO, MODEL_REVISION, MODEL_REMOTE, model_path, MODEL_SHA256, timeout=timeout
        )
    except Exception as exc:
        logger.warning("speaker_download_failed", error=str(exc), path=str(model_path))
        return False

    logger.info("speaker_model_ready", path=str(model_path))
    return True
