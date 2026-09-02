"""Tests for ingesting an episode straight from a YouTube URL.

Nothing here touches the network: ``yt-dlp`` is a subprocess, so both calls into it are faked at
the :mod:`subprocess` boundary and the assertions are about the command that would have run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Episode
from app.services.ingest import IngestJob, run_pipeline
from app.services.youtube import (
    InvalidYouTubeUrl,
    VideoInfo,
    VideoTooLong,
    YouTubeUnavailable,
    canonical_url,
    check_duration,
    download_audio,
    parse_video_id,
    probe,
)

VIDEO_ID = "dQw4w9WgXcQ"

PROBE_PAYLOAD: dict[str, Any] = {
    "id": VIDEO_ID,
    "title": "Nepanglish Podcast Ep 42",
    "duration": 1830.0,
    "uploader": "Kathmandu Talks",
    "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
    "upload_date": "20260101",
    "is_live": False,
}


# --- URL parsing: the only thing standing between the annotator and a subprocess -----------


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"http://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"youtu.be/{VIDEO_ID}",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
        f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
    ],
)
def test_every_shape_a_link_gets_copied_in_yields_the_same_video_id(url: str) -> None:
    assert parse_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://vimeo.com/123456789",
        "https://youtube.evil.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=way_too_long_to_be_an_id",
        "https://www.youtube.com/watch?v=has+a+plus",
        "https://www.youtube.com/playlist?list=PL1234567890",
        "https://www.youtube.com/",
        "file:///etc/passwd",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "--version",
    ],
)
def test_anything_that_is_not_a_youtube_video_url_is_refused(url: str) -> None:
    with pytest.raises(InvalidYouTubeUrl):
        parse_video_id(url)


def test_the_url_is_rebuilt_from_the_id_so_a_playlist_link_ingests_one_video() -> None:
    pasted = f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLabcdefgh&index=7&t=42s"
    assert canonical_url(pasted) == f"https://www.youtube.com/watch?v={VIDEO_ID}"


def test_a_url_that_would_read_as_a_flag_never_survives_canonicalization() -> None:
    """A leading dash is the reason the id is re-serialized rather than the URL sanitized."""
    with pytest.raises(InvalidYouTubeUrl):
        canonical_url("-o/etc/cron.d/pwn")


# --- Probing ------------------------------------------------------------------------------


def fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
    raises: Exception | None = None,
) -> list[list[str]]:
    """Replace ``subprocess.run`` in the module, recording every command it is handed."""
    calls: list[list[str]] = []

    def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr("app.services.youtube.subprocess.run", _run)
    return calls


def test_probe_reads_the_metadata_without_downloading_anything(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    calls = fake_run(monkeypatch, stdout=json.dumps(PROBE_PAYLOAD).encode())

    info = probe(f"https://youtu.be/{VIDEO_ID}?t=90", settings=settings)

    assert info == VideoInfo(
        video_id=VIDEO_ID,
        url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
        title="Nepanglish Podcast Ep 42",
        duration_seconds=1830.0,
        uploader="Kathmandu Talks",
        thumbnail="https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        upload_date="20260101",
        is_live=False,
    )
    (command,) = calls
    assert "--skip-download" in command
    assert "--no-playlist" in command
    assert command[-1] == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert "?t=90" not in " ".join(command)


def test_probe_falls_back_to_the_channel_when_there_is_no_uploader(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    payload = {**PROBE_PAYLOAD, "uploader": None, "channel": "Kathmandu Talks"}
    fake_run(monkeypatch, stdout=json.dumps(payload).encode())

    assert probe(f"https://youtu.be/{VIDEO_ID}", settings=settings).uploader == "Kathmandu Talks"


def test_probe_reports_a_missing_duration_as_unknown_rather_than_zero(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    fake_run(monkeypatch, stdout=json.dumps({**PROBE_PAYLOAD, "duration": None}).encode())

    assert probe(f"https://youtu.be/{VIDEO_ID}", settings=settings).duration_seconds is None


def test_probe_surfaces_what_yt_dlp_said_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    fake_run(
        monkeypatch,
        returncode=1,
        stderr=b"ERROR: [youtube] dQw4w9WgXcQ: Video unavailable\n",
    )

    with pytest.raises(YouTubeUnavailable, match="Video unavailable"):
        probe(f"https://youtu.be/{VIDEO_ID}", settings=settings)


def test_probe_treats_a_timeout_as_the_upstream_being_unavailable(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    fake_run(monkeypatch, raises=subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30))

    with pytest.raises(YouTubeUnavailable, match="timed out"):
        probe(f"https://youtu.be/{VIDEO_ID}", settings=settings)


def test_probe_rejects_unreadable_metadata(monkeypatch: pytest.MonkeyPatch, settings) -> None:
    fake_run(monkeypatch, stdout=b"not json at all")

    with pytest.raises(YouTubeUnavailable, match="unreadable"):
        probe(f"https://youtu.be/{VIDEO_ID}", settings=settings)


def test_a_configured_cookie_jar_is_passed_to_yt_dlp(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    with_cookies = settings.model_copy(
        update={
            "ingest": settings.ingest.model_copy(
                update={
                    "youtube": settings.ingest.youtube.model_copy(
                        update={"cookies_file": "/secrets/yt.txt"}
                    )
                }
            )
        }
    )
    calls = fake_run(monkeypatch, stdout=json.dumps(PROBE_PAYLOAD).encode())

    probe(f"https://youtu.be/{VIDEO_ID}", settings=with_cookies)

    (command,) = calls
    assert command[command.index("--cookies") + 1] == "/secrets/yt.txt"


# --- The spend guard ----------------------------------------------------------------------


def test_a_video_longer_than_the_limit_is_refused(settings) -> None:
    limit = settings.ingest.youtube.max_duration_seconds
    info = VideoInfo(video_id=VIDEO_ID, url="u", title="t", duration_seconds=limit + 1)

    with pytest.raises(VideoTooLong):
        check_duration(info, settings=settings)


def test_a_video_at_the_limit_is_allowed(settings) -> None:
    limit = settings.ingest.youtube.max_duration_seconds
    info = VideoInfo(video_id=VIDEO_ID, url="u", title="t", duration_seconds=limit)

    check_duration(info, settings=settings)


def test_an_unknown_duration_is_not_treated_as_an_over_long_one(settings) -> None:
    """An absent signal must not block an ingest on its own -- the same rule the scorer follows."""
    check_duration(VideoInfo(video_id=VIDEO_ID, url="u", title="t"), settings=settings)


# --- Downloading --------------------------------------------------------------------------


class FakePopen:
    """A ``subprocess.Popen`` stand-in that emits canned progress lines."""

    commands: ClassVar[list[list[str]]] = []
    lines: ClassVar[list[str]] = []
    writes: ClassVar[list[str]] = []
    returncode_after: int = 0
    stderr_text: str = ""
    timeout: bool = False

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        FakePopen.commands.append(command)
        self.returncode = 0
        self._killed = False
        self.stdout = iter(self.lines)
        # ``--output`` is a template; the fake produces what the real one would have produced.
        out_dir = Path(command[command.index("--output") + 1]).parent
        for name in self.writes:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / name).write_bytes(b"fake audio bytes")

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        # A real Popen raises once on the timed-out wait and then returns for the reaping call
        # that follows the kill; a fake that raises twice would hide a missing kill instead.
        if FakePopen.timeout and not self._killed:
            raise subprocess.TimeoutExpired(cmd="yt-dlp", timeout=timeout or 0)
        self.returncode = FakePopen.returncode_after
        return "", FakePopen.stderr_text

    def kill(self) -> None:
        self._killed = True


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch):
    """A configurable fake ``Popen``, reset for each test."""
    FakePopen.commands = []
    FakePopen.lines = []
    FakePopen.returncode_after = 0
    FakePopen.stderr_text = ""
    FakePopen.writes = ["source_audio.m4a"]
    FakePopen.timeout = False
    monkeypatch.setattr("app.services.youtube.subprocess.Popen", FakePopen)
    return FakePopen


def test_download_returns_the_file_yt_dlp_wrote(fake_popen, settings, tmp_path: Path) -> None:
    path = download_audio(f"https://youtu.be/{VIDEO_ID}", tmp_path / "work", settings=settings)

    assert path == tmp_path / "work" / "source_audio.m4a"
    assert path.read_bytes() == b"fake audio bytes"

    (command,) = fake_popen.commands
    assert command[-1] == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert command[command.index("--format") + 1] == settings.ingest.youtube.format
    assert "--newline" in command


def test_a_half_written_part_file_is_never_mistaken_for_the_download(
    fake_popen, settings, tmp_path: Path
) -> None:
    fake_popen.writes = ["source_audio.webm.part", "source_audio.webm"]

    path = download_audio(f"https://youtu.be/{VIDEO_ID}", tmp_path / "work", settings=settings)

    assert path.name == "source_audio.webm"


def test_progress_lines_are_reported_to_the_caller(fake_popen, settings, tmp_path: Path) -> None:
    fake_popen.lines = [
        "[youtube] Extracting URL\n",
        "[download] Destination: source_audio.m4a\n",
        "[download]   0.0% of ~45.20MiB at 1.00MiB/s ETA 00:45\n",
        "[download]  50.0% of ~45.20MiB at 3.10MiB/s ETA 00:07\n",
        "[download] 100% of 45.20MiB in 00:14\n",
    ]
    seen: list[float] = []

    download_audio(
        f"https://youtu.be/{VIDEO_ID}",
        tmp_path / "work",
        settings=settings,
        on_progress=lambda percent, line: seen.append(percent),
    )

    assert seen == [0.0, 50.0, 100.0]


def test_a_failed_download_carries_the_reason(fake_popen, settings, tmp_path: Path) -> None:
    fake_popen.returncode_after = 1
    fake_popen.stderr_text = "ERROR: Sign in to confirm you're not a bot\n"
    fake_popen.writes = []

    with pytest.raises(YouTubeUnavailable, match="not a bot"):
        download_audio(f"https://youtu.be/{VIDEO_ID}", tmp_path / "work", settings=settings)


def test_a_download_that_produced_no_file_is_a_failure(
    fake_popen, settings, tmp_path: Path
) -> None:
    fake_popen.writes = []

    with pytest.raises(YouTubeUnavailable, match="no audio file"):
        download_audio(f"https://youtu.be/{VIDEO_ID}", tmp_path / "work", settings=settings)


def test_a_download_that_hangs_is_killed_and_reported(fake_popen, settings, tmp_path: Path) -> None:
    fake_popen.timeout = True

    with pytest.raises(YouTubeUnavailable, match="timed out"):
        download_audio(f"https://youtu.be/{VIDEO_ID}", tmp_path / "work", settings=settings)


def test_the_url_is_validated_before_a_subprocess_is_started(
    fake_popen, settings, tmp_path: Path
) -> None:
    with pytest.raises(InvalidYouTubeUrl):
        download_audio("https://evil.example/x", tmp_path / "work", settings=settings)

    assert fake_popen.commands == []


# --- The pipeline: the download occupies the upload's slot ---------------------------------


@pytest.mark.db
def test_a_url_job_downloads_before_stage_one_and_records_where_it_came_from(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audio arrives by download instead of upload; everything downstream is unchanged."""
    from tests.test_ingest import make_test_audio

    source = make_test_audio(tmp_path / "yt_source.wav", duration_seconds=6.0)

    def fake_download(url: str, dest_dir: Path, settings=None, on_progress=None) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / "source_audio.wav"
        target.write_bytes(source.read_bytes())
        if on_progress:
            on_progress(100.0, "[download] 100% of 1.00MiB in 00:01")
        return target

    monkeypatch.setattr("app.services.ingest.download_audio", fake_download)

    job = IngestJob(
        job_id="yt-job-001",
        episode_id="yt_ep001",
        show_id="podcast",
        title="Downloaded Episode",
        audio_path=None,
        work_dir=tmp_path / "work_yt",
        source_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)

    assert job.error is None, job.error
    assert job.status == "completed"
    assert job.stage == "complete"
    assert any("Fetching audio from" in entry.message for entry in job.logs)
    assert any("Stage 1/5" in entry.message for entry in job.logs)

    episode = db_session.scalar(sa.select(Episode).where(Episode.external_id == "yt_ep001"))
    assert episode is not None
    assert episode.source_uri == f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.mark.db
def test_a_failed_download_fails_the_job_without_running_a_stage(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Path:
        raise YouTubeUnavailable("yt-dlp download failed: Video unavailable")

    monkeypatch.setattr("app.services.ingest.download_audio", boom)

    job = IngestJob(
        job_id="yt-job-002",
        episode_id="yt_ep002",
        show_id="podcast",
        title="Gone",
        audio_path=None,
        work_dir=tmp_path / "work_yt_fail",
        source_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)

    assert job.status == "failed"
    assert "Video unavailable" in (job.error or "")
    assert not any("Stage 1/5" in entry.message for entry in job.logs)
    assert not (tmp_path / "work_yt_fail").exists()


@pytest.mark.db
def test_a_job_with_neither_a_file_nor_a_url_fails_loudly(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    job = IngestJob(
        job_id="yt-job-003",
        episode_id="yt_ep003",
        show_id="podcast",
        title="Nothing to ingest",
        audio_path=None,
        work_dir=tmp_path / "work_empty",
        source_url=None,
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)

    assert job.status == "failed"
    assert "neither" in (job.error or "")


# --- API ----------------------------------------------------------------------------------


@pytest.fixture
def probed(monkeypatch: pytest.MonkeyPatch):
    """Stub the metadata lookup the endpoints make, and never start a real pipeline."""

    def _install(info: VideoInfo | Exception) -> None:
        def _probe(url: str, settings=None) -> VideoInfo:
            if isinstance(info, Exception):
                raise info
            return info

        monkeypatch.setattr("app.api.ingest.probe", _probe)
        monkeypatch.setattr(
            "app.api.ingest.run_pipeline",
            lambda job, *args: job.set_progress("downloading", 0.0),
        )

    return _install


def video(**overrides: Any) -> VideoInfo:
    base = {
        "video_id": VIDEO_ID,
        "url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "title": "Nepanglish Podcast Ep 42",
        "duration_seconds": 1830.0,
        "uploader": "Kathmandu Talks",
    }
    return VideoInfo(**{**base, **overrides})


@pytest.mark.db
def test_probe_endpoint_returns_what_the_form_needs_to_prefill_itself(
    client: TestClient, probed
) -> None:
    probed(video())

    response = client.post(
        "/ingest/youtube/probe", json={"url": f"https://youtu.be/{VIDEO_ID}?t=30"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == VIDEO_ID
    assert body["title"] == "Nepanglish Podcast Ep 42"
    assert body["duration_seconds"] == 1830.0
    assert body["uploader"] == "Kathmandu Talks"
    assert body["suggested_episode_id"] == "nepanglish_podcast_ep_42"


@pytest.mark.db
def test_probe_endpoint_rejects_a_url_that_is_not_youtube(client: TestClient) -> None:
    response = client.post("/ingest/youtube/probe", json={"url": "https://vimeo.com/123456789"})

    assert response.status_code == 422
    assert "not a YouTube URL" in response.json()["detail"]


@pytest.mark.db
def test_probe_endpoint_reports_an_upstream_failure_as_a_bad_gateway(
    client: TestClient, probed
) -> None:
    probed(YouTubeUnavailable("yt-dlp metadata lookup failed: Video unavailable"))

    response = client.post("/ingest/youtube/probe", json={"url": f"https://youtu.be/{VIDEO_ID}"})

    assert response.status_code == 502
    assert "Video unavailable" in response.json()["detail"]


@pytest.mark.db
def test_a_live_stream_is_refused_before_a_job_exists(client: TestClient, probed) -> None:
    probed(video(is_live=True))

    response = client.post("/ingest/youtube", json={"url": f"https://youtu.be/{VIDEO_ID}"})

    assert response.status_code == 422
    assert "live streams" in response.json()["detail"]


@pytest.mark.db
def test_an_over_long_video_is_refused_before_anything_is_downloaded(
    client: TestClient, probed, settings
) -> None:
    probed(video(duration_seconds=settings.ingest.youtube.max_duration_seconds + 60))

    response = client.post("/ingest/youtube", json={"url": f"https://youtu.be/{VIDEO_ID}"})

    assert response.status_code == 422
    assert "ingestion limit" in response.json()["detail"]


@pytest.mark.db
def test_starting_a_youtube_job_takes_its_title_from_the_video(client: TestClient, probed) -> None:
    probed(video())

    response = client.post(
        "/ingest/youtube",
        json={"url": f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLxyz", "show_id": "demo"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["title"] == "Nepanglish Podcast Ep 42"
    assert body["episode_id"] == "nepanglish_podcast_ep_42"
    assert body["source_url"] == f"https://www.youtube.com/watch?v={VIDEO_ID}"

    status_body = client.get(f"/ingest/{body['job_id']}").json()
    assert status_body["stage"] == "downloading"
    assert status_body["show_id"] == "demo"


@pytest.mark.db
def test_an_explicit_title_and_slug_win_over_the_video_metadata(client: TestClient, probed) -> None:
    probed(video())

    response = client.post(
        "/ingest/youtube",
        json={
            "url": f"https://youtu.be/{VIDEO_ID}",
            "episode_title": "Episode 42: AI in Kathmandu",
            "episode_id": "ep_042",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["title"] == "Episode 42: AI in Kathmandu"
    assert body["episode_id"] == "ep_042"


@pytest.mark.db
def test_a_traversing_episode_id_cannot_escape_the_work_root_on_the_url_path(
    client: TestClient, probed, settings
) -> None:
    """The upload path slugifies for this reason; the URL path must not be the way around it."""
    probed(video())

    response = client.post(
        "/ingest/youtube",
        json={"url": f"https://youtu.be/{VIDEO_ID}", "episode_id": "../../../../etc/pwned"},
    )

    assert response.status_code == 202
    assert ".." not in response.json()["episode_id"]
    assert not (settings.ingest.work_root.parent / "etc").exists()
