"""Fetching an episode's source audio from a YouTube URL, via yt-dlp.

This is the other half of the browser ingestion path: instead of the annotator exporting audio by
hand and uploading it, the server fetches it. Everything downstream is unchanged -- the file lands
in the job's work directory and stage 1 normalizes it like any upload.

Two rules shape this module:

* **Nothing the caller typed reaches the subprocess.** A URL is parsed down to its eleven-character
  video id, and the id is what a canonical URL is rebuilt from. So the harness cannot be turned
  into a general-purpose fetcher for arbitrary hosts, and a URL beginning with ``-`` cannot become
  a yt-dlp flag.
* **Duration is checked before bytes move.** Every configured ``asr*`` route transcribes every
  clip, so the cost of an ingest is linear in the source duration; a mistyped link to an
  eight-hour livestream is refused by :func:`probe` rather than discovered at the ASR stage.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.config import Settings, YouTubeSettings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Hosts a video id may be extracted from. Anything else is not a YouTube URL, whatever it claims.
YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
        "youtu.be",
        "www.youtu.be",
    }
)

#: Path prefixes that carry the video id as the following segment.
_ID_BEARING_PREFIXES = ("shorts", "live", "embed", "v")

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: yt-dlp's own progress lines, read from stdout with ``--newline``.
_PROGRESS_RE = re.compile(r"^\[download\]\s+([0-9.]+)%")


class YouTubeError(RuntimeError):
    """Base class for every failure in this module."""


class InvalidYouTubeUrl(YouTubeError):
    """The text given is not a YouTube video URL. The caller's fault; a 422."""


class YouTubeUnavailable(YouTubeError):
    """yt-dlp could not deliver the video. Upstream's fault or the network's; a 502."""


class VideoTooLong(YouTubeError):
    """The video exceeds ``ingest.youtube.max_duration_seconds``, the spend guard."""


@dataclass(frozen=True)
class VideoInfo:
    """What a probe learned about a video, before anything is downloaded."""

    video_id: str
    url: str
    title: str
    duration_seconds: float | None = None
    uploader: str | None = None
    thumbnail: str | None = None
    upload_date: str | None = None
    is_live: bool = False


def parse_video_id(raw_url: str) -> str:
    """Extract the video id from any accepted YouTube URL shape.

    Accepts ``watch?v=``, ``youtu.be/``, ``shorts/``, ``live/``, ``embed/`` and ``v/`` forms, on
    any of :data:`YOUTUBE_HOSTS`, with or without a scheme.

    Args:
        raw_url: The URL as the annotator pasted it.

    Returns:
        The eleven-character video id.

    Raises:
        InvalidYouTubeUrl: The text is not a YouTube video URL, or carries no usable id.
    """
    text = (raw_url or "").strip()
    if not text:
        raise InvalidYouTubeUrl("no URL given")

    # A bare "youtu.be/ID" has no scheme, and urlparse would read the host as a path.
    if "//" not in text:
        text = f"https://{text.lstrip('/')}"

    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https"):
        raise InvalidYouTubeUrl(f"unsupported URL scheme '{parsed.scheme}'")

    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise InvalidYouTubeUrl(f"'{host or raw_url}' is not a YouTube URL")

    segments = [part for part in parsed.path.split("/") if part]

    candidate = ""
    if host.endswith("youtu.be"):
        candidate = segments[0] if segments else ""
    elif segments and segments[0] in _ID_BEARING_PREFIXES:
        candidate = segments[1] if len(segments) > 1 else ""
    else:
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]

    if not VIDEO_ID_RE.match(candidate):
        raise InvalidYouTubeUrl("no YouTube video id found in the URL")
    return candidate


def canonical_url(raw_url: str) -> str:
    """Rebuild a URL from its video id alone.

    This is the only URL form that ever reaches the subprocess. Rebuilding rather than sanitizing
    drops playlist, timestamp and tracking parameters as a side effect, so pasting a link from
    inside a playlist ingests the one video rather than the whole list.
    """
    return f"https://www.youtube.com/watch?v={parse_video_id(raw_url)}"


def _base_command(settings: YouTubeSettings) -> list[str]:
    """The yt-dlp invocation shared by probing and downloading.

    Runs it as a module of the current interpreter rather than a ``yt-dlp`` on ``PATH``: the
    package is a declared dependency, so this is the copy the deployment actually installed.
    """
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "--no-colors",
    ]
    if settings.cookies_file:
        command += ["--cookies", settings.cookies_file]
    return command


def _youtube_settings(settings: Settings | None) -> YouTubeSettings:
    return (settings or get_settings()).ingest.youtube


def _fail(action: str, stderr: str, returncode: int) -> YouTubeUnavailable:
    """Turn a yt-dlp failure into an exception carrying the tail of what it said."""
    detail = " ".join(stderr.strip().splitlines()[-3:])[:400] or f"exit code {returncode}"
    return YouTubeUnavailable(f"yt-dlp {action} failed: {detail}")


def probe(raw_url: str, settings: Settings | None = None) -> VideoInfo:
    """Read a video's metadata without downloading it.

    Args:
        raw_url: The URL as pasted; only its video id is used.
        settings: Configuration override.

    Returns:
        The video's title, duration, uploader and thumbnail.

    Raises:
        InvalidYouTubeUrl: The URL is not a YouTube video URL.
        YouTubeUnavailable: yt-dlp failed, timed out, or returned something unreadable.
    """
    yt = _youtube_settings(settings)
    url = canonical_url(raw_url)
    command = [*_base_command(yt), "--dump-single-json", "--skip-download", url]

    try:
        proc = subprocess.run(
            command, capture_output=True, check=False, timeout=yt.probe_timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise YouTubeUnavailable(
            f"yt-dlp metadata lookup timed out after {yt.probe_timeout_seconds:.0f}s"
        ) from exc

    if proc.returncode != 0:
        raise _fail(
            "metadata lookup", proc.stderr.decode("utf-8", errors="replace"), proc.returncode
        )

    try:
        payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise YouTubeUnavailable("yt-dlp returned unreadable metadata") from exc

    duration = payload.get("duration")
    info = VideoInfo(
        video_id=str(payload.get("id") or parse_video_id(url)),
        url=url,
        title=str(payload.get("title") or "").strip() or f"YouTube {parse_video_id(url)}",
        duration_seconds=float(duration) if isinstance(duration, int | float) else None,
        uploader=payload.get("uploader") or payload.get("channel") or None,
        thumbnail=payload.get("thumbnail") or None,
        upload_date=payload.get("upload_date") or None,
        is_live=bool(payload.get("is_live")),
    )
    logger.info(
        "youtube_probed",
        video_id=info.video_id,
        duration_seconds=info.duration_seconds,
        is_live=info.is_live,
    )
    return info


def check_duration(info: VideoInfo, settings: Settings | None = None) -> None:
    """Refuse a video whose length would make the ASR bill unreasonable.

    A video whose duration yt-dlp could not report passes: an unknown length is not evidence of a
    long one, and every live stream is already rejected by the caller.

    Raises:
        VideoTooLong: The duration exceeds ``ingest.youtube.max_duration_seconds``.
    """
    limit = _youtube_settings(settings).max_duration_seconds
    if info.duration_seconds is not None and info.duration_seconds > limit:
        raise VideoTooLong(
            f"video is {info.duration_seconds / 60:.0f} min, over the "
            f"{limit / 60:.0f} min ingestion limit"
        )


def download_audio(
    raw_url: str,
    dest_dir: Path,
    settings: Settings | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Download a video's audio track into ``dest_dir``.

    The file keeps whichever container YouTube served (usually ``.m4a`` or ``.webm``); stage 1
    re-encodes it, so nothing here transcodes and no quality is lost to a needless round trip.

    Args:
        raw_url: The URL as pasted; only its video id is used.
        dest_dir: Directory to write into. Created if absent.
        settings: Configuration override.
        on_progress: Called with ``(percent, line)`` for each progress line yt-dlp emits.

    Returns:
        Path to the downloaded audio file.

    Raises:
        InvalidYouTubeUrl: The URL is not a YouTube video URL.
        YouTubeUnavailable: The download failed, timed out, or produced no file.
    """
    yt = _youtube_settings(settings)
    url = canonical_url(raw_url)
    dest_dir.mkdir(parents=True, exist_ok=True)

    command = [
        *_base_command(yt),
        "--newline",
        "--format",
        yt.format,
        "--output",
        str(dest_dir / "source_audio.%(ext)s"),
        url,
    ]

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:  # yt-dlp missing from the environment entirely
        raise YouTubeUnavailable(f"could not run yt-dlp: {exc}") from exc

    assert proc.stdout is not None
    for line in proc.stdout:
        match = _PROGRESS_RE.match(line.strip())
        if match and on_progress is not None:
            on_progress(float(match.group(1)), line.strip())

    try:
        _, stderr = proc.communicate(timeout=yt.download_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.communicate()
        raise YouTubeUnavailable(
            f"yt-dlp download timed out after {yt.download_timeout_seconds:.0f}s"
        ) from exc

    if proc.returncode != 0:
        raise _fail("download", stderr or "", proc.returncode)

    # ``--output`` fixes the stem, so whatever matches is what this call produced. Partial files
    # keep a ``.part`` suffix and are skipped.
    produced = sorted(p for p in dest_dir.glob("source_audio.*") if p.suffix != ".part")
    if not produced:
        raise YouTubeUnavailable("yt-dlp reported success but wrote no audio file")

    audio_path = produced[0]
    logger.info(
        "youtube_downloaded",
        url=url,
        path=str(audio_path),
        bytes=audio_path.stat().st_size,
    )
    return audio_path
