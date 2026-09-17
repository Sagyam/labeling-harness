"""Stage 1: FFmpeg loudnorm to 16 kHz mono FLAC (invariant 7)."""

from __future__ import annotations

import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import soundfile as sf

from app.utils.logging import get_logger

logger = get_logger(__name__)


#: Resampling down to 16 kHz throws away everything above 8 kHz, and whatever is not filtered out
#: first does not vanish -- it folds back below 8 kHz as alias. FFmpeg's built-in resampler is the
#: wrong tool for the job here: measured against pure tones, ``aresample=16000`` passes 8.2 kHz at
#: -15 dB and 8.5 kHz at -26 dB, so a podcast with ordinary energy in the 8-10 kHz band gets that
#: band folded on top of its own 7-8 kHz. Sibilants are exactly where that energy lives, so every
#: /s/ and /ʃ/ lands a burst of near-Nyquist noise -- heard as a click, roughly once a second.
#: libsoxr rejects the same tones at -158 dB. The cutoff keeps the passband flat to 7.6 kHz.
SOXR_RESAMPLE = "aresample=resampler=soxr:precision=28:cutoff=0.95:osr=16000"

#: Used only when the FFmpeg build has no libsoxr. Lengthening the filter and pulling the cutoff in
#: takes the same 8.2 kHz tone from -15 dB to -55 dB: far better than the default, still far worse
#: than soxr, so :func:`_resample_filter` says so out loud rather than degrading quietly.
SWR_RESAMPLE_FALLBACK = "aresample=16000:filter_size=256:cutoff=0.91"


@lru_cache(maxsize=1)
def _resample_filter() -> str:
    """The resampling stage of the stage-1 filter chain, preferring libsoxr.

    Probed once per process by asking FFmpeg to convert a fraction of a second of silence, which
    is cheaper and more truthful than parsing ``-buildconf``: it fails exactly when the filter
    would fail in the pipeline.
    """
    probe = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "0.1",
            "-af",
            SOXR_RESAMPLE,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        return SOXR_RESAMPLE
    logger.warning(
        "ffmpeg_soxr_unavailable",
        detail=probe.stderr.decode("utf-8", errors="replace")[:200],
        fallback=SWR_RESAMPLE_FALLBACK,
        impact="clips will carry more resampling alias than a soxr build produces",
    )
    return SWR_RESAMPLE_FALLBACK


class SilentAudioError(RuntimeError):
    """The input holds no sound at all, so there is nothing to normalise."""


def normalize_audio(input_path: Path, output_path: Path) -> float:
    """Stage 1: Normalize audio using FFmpeg with loudnorm filter.

    Converts to 16 kHz mono FLAC using two-pass EBU R128 normalization.
    Pass 1 measures integrated loudness and true-peak statistics.
    Pass 2 applies linear normalization to prevent dynamic AGC gain pumping between words, then
    downmixes and resamples through :func:`_resample_filter` so that nothing above 8 kHz folds
    back into the clip as alias.
    Returns duration in seconds.

    Raises:
        SilentAudioError: The input is digital silence.
        RuntimeError: FFmpeg could not normalise it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: Measure loudness parameters
    cmd1 = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-23:LRA=7:tp=-2:print_format=json",
        "-f",
        "null",
        "-",
    ]
    proc1 = subprocess.run(cmd1, capture_output=True, check=False)

    measured: dict[str, Any] | None = None
    if proc1.returncode == 0:
        try:
            stderr_text = proc1.stderr.decode("utf-8", errors="replace")
            start_brace = stderr_text.rfind("{")
            end_brace = stderr_text.rfind("}")
            if start_brace != -1 and end_brace > start_brace:
                data = json.loads(stderr_text[start_brace : end_brace + 1])
                if all(k in data for k in ("input_i", "input_lra", "input_tp", "input_thresh")):
                    measured = data
        except Exception:
            measured = None
    if measured is not None and not math.isfinite(float(measured["input_i"])):
        # Silence measures -inf LUFS. Pass 2 then fails on the inf offset, and single-pass
        # loudnorm would blow the silence up to full scale.
        raise SilentAudioError("the audio is silent (loudness -inf LUFS)")

    # Pass 2: Apply linear normalization with clean mono downmix before resampling
    resample = _resample_filter()
    if measured:
        filter_str = (
            "loudnorm=I=-23:LRA=7:tp=-2"
            f":measured_I={measured['input_i']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_tp={measured['input_tp']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured.get('target_offset', '0.0')}"
            f":linear=true,aformat=channel_layouts=mono,{resample}"
        )
    else:
        filter_str = f"loudnorm=I=-23:LRA=7:tp=-2,aformat=channel_layouts=mono,{resample}"

    cmd2 = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(input_path),
        "-af",
        filter_str,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "flac",
        str(output_path),
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, check=False)
    if proc2.returncode != 0:
        err_msg = proc2.stderr.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"FFmpeg normalization failed (exit code {proc2.returncode}): {err_msg}")

    info = sf.info(str(output_path))
    return float(info.duration)
