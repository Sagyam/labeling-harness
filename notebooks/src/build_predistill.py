"""Build notebooks/PreDistill.ipynb: `python notebooks/src/build_predistill.py`. Roadmap §B step 1
(D101): the owner's zip of recordings, cut exactly as ingest cuts an episode, into the corpus
Teacher.ipynb labels. Shared code lives in ftkit.py, distill.py and prekit.py, written out by
%%writefile cells; the cutting itself is the harness's own, fetched at a pinned commit."""

import hashlib
import json
import sys
from pathlib import Path

from build_finetune import code, md, notebook

HERE = Path(__file__).parent
FTKIT = (HERE / "ftkit.py").read_text()
DISTILLKIT = (HERE / "distill.py").read_text()
PREKIT = (HERE / "prekit.py").read_text()
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
REPO = HERE.parents[1]

#: The harness files the cutting imports, fetched from GitHub at HARNESS_COMMIT and checked here.
HARNESS_FILES = [
    "backend/app/services/silero_vad.py",
    "backend/app/services/ingest/audio.py",
    "backend/app/services/models/silero_vad.onnx",
    "backend/app/utils/logging.py",
    "backend/app/utils/hashing.py",
]

INTRO = """
# Distillation, step 1 — cut the owner's recordings into the corpus

Roadmap §B, D101. The owner uploads `distill.zip`, many `<channel_name>_<NN>.mp3`, to the root of
the dataset repo. This notebook cuts every recording **exactly as ingest cuts an episode**: stage 1
(two-pass loudness normalisation to 16 kHz mono FLAC) and stage 2 (Silero VAD, 2–20 s slices),
run from the harness's own modules, fetched at a pinned commit and checked by sha256, not
re-implemented. Nothing touches Postgres, the queue or a paid route.

**What it writes**, to `DATASET_REPO/distill/`: each recording whole as `episodes/<id>.flac`, its
clip times in `sources/<id>.json`, and at the end `clips.jsonl` (every clip, named and timed as the
labelled export's rows) and `summary.json`. `ftkit.AudioStore` cuts clips from the whole recordings
in RAM, as it does for the labelled export, so the repo holds one file per recording rather than
one per clip. `Teacher.ipynb` reads it next.

**Gold.** The owner vouches that these channels are new (2026-09-25), so there is no voiceprint
screen here. Gold's shows are still refused by file name, which costs nothing.

**Runtime.** CPU work, but run on the A100 runtime: it has 12 cores where a CPU runtime has 2.
One process per core, each with a single-threaded VAD.

**Small blast radius.** Files are processed a chunk at a time and each chunk is uploaded as soon
as it is cut; a rerun skips every recording already in `distill/sources/`, so a lost runtime costs
at most one chunk. A file that cannot be read is reported and skipped, never fatal.
"""

CONFIG = r"""
DATASET_REPO = "Sagyam/nepanglish-asr"   # the owner uploads ZIP_NAME to its root
ZIP_NAME = "distill.zip"
PREFIX = "distill"                        # DATASET_REPO/distill/: episodes/, sources/, clips.jsonl
BLOCKED_CHANNELS = ["chill pill", "prime television"]   # gold's shows (D101), matched by file name
HARNESS_COMMIT = "__HARNESS_COMMIT__"     # ingest's cutting code is fetched at this commit
HARNESS_SHA256 = __HARNESS_SHA256__
WORKERS = None                            # processes; None for every core
CHUNK = None                              # recordings per uploaded chunk; None for WORKERS
"""

SETUP = r"""
%pip install -q structlog onnxruntime
import json
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
# Secrets only work from a cell run in the Colab UI: run this cell by hand once when cells are
# driven from outside (the Colab MCP); the token is then kept in the hub's token file on the VM.
if IN_COLAB:
    from google.colab import userdata

    TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists()
                                  else userdata.get("HF_TOKEN"))
    if not TOKEN_FILE.exists():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(os.environ["HF_TOKEN"])
        TOKEN_FILE.chmod(0o600)
FT = Path("/content/ft") if IN_COLAB else Path.cwd() / ".cache-ft"
FT.mkdir(parents=True, exist_ok=True)
print("cores:", os.cpu_count(), "| ffmpeg:", os.popen("ffmpeg -version").readline().strip())
"""

HARNESS = r"""
# The harness's own cutting code, byte for byte, at HARNESS_COMMIT: ingest's stage 1 and 2 and the
# two helpers they import. Empty package files stand in for the rest of the backend.
import hashlib
import urllib.request

HARNESS = FT / "harness"
for rel, digest in HARNESS_SHA256.items():
    data = urllib.request.urlopen("https://raw.githubusercontent.com/Sagyam/labeling-harness/"
                                  f"{HARNESS_COMMIT}/{rel}").read()
    assert hashlib.sha256(data).hexdigest() == digest, f"{rel}: sha256 mismatch"
    path = HARNESS / rel.removeprefix("backend/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
for pkg in ("app", "app/services", "app/services/ingest", "app/utils"):
    (HARNESS / pkg / "__init__.py").touch()
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(FT))

import distill
import ftkit
import prekit
from app.services.silero_vad import SileroVAD

assert SileroVAD().available, "the Silero model did not load"
print("harness code at", HARNESS_COMMIT[:7], "checked:", len(HARNESS_SHA256), "files")
"""

FETCH = r"""
import zipfile

from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
with ftkit.timed(f"downloading {ZIP_NAME}"):
    archive = hf_hub_download(DATASET_REPO, ZIP_NAME, repo_type="dataset", token=TOKEN)
MP3 = FT / "recordings"
with ftkit.timed(f"unzipping {Path(archive).stat().st_size / 2**30:.1f} GiB"):
    with zipfile.ZipFile(archive) as z:
        z.extractall(MP3)
suffixes = {".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".flac", ".aac"}
files = sorted(p for p in MP3.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
               and not p.name.startswith("._"))
ids = {}
for p in files:
    ids.setdefault(distill.source_from_filename(p.name)[0], []).append(p.name)
clashes = {k: v for k, v in ids.items() if len(v) > 1}
assert not clashes, f"two files would share a source id: {clashes}"
channels = sorted({distill.source_from_filename(p.name)[1] for p in files})
print(f"{len(files)} recordings, {sum(p.stat().st_size for p in files) / 2**30:.1f} GiB, "
      f"from {len(channels)} channels: {', '.join(channels)}")
"""

CUT_NOTE = """
## Cut, a chunk at a time

Every recording goes through `prekit.process_file`: ingest's `normalize_audio`, then its Silero VAD
and `segment_audio_to_slices`. Each chunk is uploaded as soon as it is cut. The line after each
chunk says how long the rest will take at the rate so far.
"""

CUT = r"""
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

prefix = f"{PREFIX}/sources/"
done = {Path(f).stem for f in api.list_repo_files(DATASET_REPO, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".json")}
todo = [p for p in files if distill.source_from_filename(p.name)[0] not in done]
workers = WORKERS or os.cpu_count()
chunk = CHUNK or workers
total_bytes = sum(p.stat().st_size for p in todo)
print(f"{len(done)} recordings already cut, {len(todo)} to cut on {workers} processes, "
      f"{chunk} per uploaded chunk")
results, started, cut_bytes = [], time.perf_counter(), 0
with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
    for number, k in enumerate(range(0, len(todo), chunk), 1):
        part = todo[k:k + chunk]
        stage = FT / "stage"
        shutil.rmtree(stage, ignore_errors=True)
        t0 = time.perf_counter()
        got = list(pool.map(prekit.process_file, [str(p) for p in part], [str(stage)] * len(part),
                            [BLOCKED_CHANNELS] * len(part)))
        results += got
        cut = [g for g in got if g["status"] == "prepared"]
        cut_bytes += sum(p.stat().st_size for p in part)
        rate = cut_bytes / (time.perf_counter() - started)
        print(f"chunk {number}: {len(cut)} cut, {sum(g['status'] == 'failed' for g in got)} failed, "
              f"{sum(g['status'] == 'blocked' for g in got)} blocked | "
              f"{sum(g['duration'] for g in cut) / 3600:.1f} h audio -> "
              f"{sum(g['speech_seconds'] for g in cut) / 3600:.1f} h speech, "
              f"{sum(g['clips'] for g in cut)} clips in {time.perf_counter() - t0:.0f} s | "
              f"about {(total_bytes - cut_bytes) / rate / 60:.0f} min left", flush=True)
        if cut:
            with ftkit.timed(f"chunk {number}: uploading {len(cut)} recordings"):
                api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(stage),
                                  path_in_repo=PREFIX, commit_message=f"{PREFIX}: chunk {number}, {len(cut)} recordings")
for g in results:
    if g["status"] != "prepared":
        print(f"{g['status'].upper()} {g['file']}: {g.get('error', 'channel ' + g['channel'])}")
"""

MANIFEST_NOTE = """
## The manifest and the summary

Rebuilt from every `distill/sources/*.json` on the repo, so it is right whatever earlier runs
did.
"""

MANIFEST = r"""
names = sorted(f for f in api.list_repo_files(DATASET_REPO, repo_type="dataset")
               if f.startswith(f"{PREFIX}/sources/") and f.endswith(".json"))
with ftkit.timed(f"reading {len(names)} recordings' clip lists"):
    sources = [json.loads(Path(hf_hub_download(DATASET_REPO, n, repo_type="dataset", token=TOKEN)).read_text("utf-8"))
               for n in names]
rows = [r for s in sources for r in s["rows"]]
per_channel = {}
for s in sources:
    c = per_channel.setdefault(s["channel"], {"recordings": 0, "audio_h": 0.0, "speech_h": 0.0, "clips": 0})
    c["recordings"] += 1
    c["audio_h"] += s["duration"] / 3600
    c["speech_h"] += s["speech_seconds"] / 3600
    c["clips"] += s["clips"]
summary = {
    "recordings": len(sources),
    "clips": len(rows),
    "audio_hours": round(sum(s["duration"] for s in sources) / 3600, 2),
    "speech_hours": round(sum(r["duration"] for r in rows) / 3600, 2),
    "vad": sorted({s["vad"] for s in sources}),
    "harness_commit": HARNESS_COMMIT,
    "blocked_channels": BLOCKED_CHANNELS,
    "per_channel": {k: {x: round(y, 2) if isinstance(y, float) else y for x, y in v.items()}
                    for k, v in sorted(per_channel.items())},
}
out = FT / "manifest"
out.mkdir(exist_ok=True)
(out / "clips.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
(out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")
with ftkit.timed("uploading clips.jsonl and summary.json"):
    api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(out), path_in_repo=PREFIX,
                      commit_message=f"{PREFIX}: manifest, {len(rows)} clips, {summary['speech_hours']} h")
print(json.dumps({k: v for k, v in summary.items() if k != "per_channel"}, indent=1))
for name, c in summary["per_channel"].items():
    print(f"  {name:<40} {c['recordings']:>4} recordings {c['audio_h']:>7.1f} h audio "
          f"{c['speech_h']:>7.1f} h speech {c['clips']:>7} clips")
"""


def cells(commit: str) -> list:
    digests = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in HARNESS_FILES}
    config = CONFIG.replace("__HARNESS_COMMIT__", commit).replace(
        "__HARNESS_SHA256__", json.dumps(digests, indent=4)
    )
    return [
        md(INTRO),
        md("## Config"),
        code(config),
        md("## Setup"),
        code(SETUP),
        code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
        code("%%writefile /content/ft/distill.py\n" + DISTILLKIT),
        code("%%writefile /content/ft/prekit.py\n" + PREKIT),
        code(HARNESS),
        md("## The recordings"),
        code(FETCH),
        md(CUT_NOTE),
        code(CUT),
        md(MANIFEST_NOTE),
        code(MANIFEST),
    ]


if __name__ == "__main__":
    # The commit whose harness files the notebook fetches: HEAD, whose files must match the
    # working tree (the digests are read from it), so build after committing them.
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", *HARNESS_FILES],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        sys.exit(f"harness files differ from HEAD; commit them first:\n{dirty}")
    path = OUT_DIR / "PreDistill.ipynb"
    path.write_text(
        json.dumps(notebook(cells(commit)), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", path, len(cells(commit)), "cells, harness at", commit[:7])
