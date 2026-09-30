"""Build notebooks/04_PreDistill.ipynb: `python notebooks/src/build_predistill.py`. Step 4 of the
protocol (D101, D105): the owner's zip of recordings, cut exactly as ingest cuts an episode, into
the corpus 05_Teacher.ipynb labels, and each clip's overlapped speech measured. Shared code lives
in ftkit.py, distill.py and prekit.py, written out by %%writefile cells; the cutting and the
overlap detector are the harness's own, fetched at a pinned commit."""

import hashlib
import json
import sys
from pathlib import Path

import nbkit
from nbkit import code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
REPO = HERE.parents[1]

#: The harness files the notebook imports, fetched from GitHub at HARNESS_COMMIT and checked here.
HARNESS_FILES = [
    "backend/app/services/silero_vad.py",
    "backend/app/services/ingest/audio.py",
    "backend/app/services/models/silero_vad.onnx",
    "backend/app/services/overlap.py",
    "backend/app/utils/logging.py",
    "backend/app/utils/hashing.py",
    "backend/app/utils/model_fetch.py",
]

INTRO = """
# 04 — PreDistill: cut the unlabelled recordings, and measure their crosstalk

Step 4 of the protocol (D105); the corpus itself is D101. Two passes over the unlabelled audio the
teacher will label, both CPU work and both resumable. Nothing touches Postgres, the queue or a
paid route.

**1. Cut.** The owner uploads `distill.zip`, many `<channel_name>_<NN>.mp3`, to the root of the
dataset repo. Every recording is cut **exactly as ingest cuts an episode**: stage 1 (two-pass
loudness normalisation to 16 kHz mono FLAC) and stage 2 (Silero VAD, 2–20 s slices), run from
the harness's own modules, fetched at a pinned commit and checked by sha256, not re-implemented.
With no zip at the root there is nothing new to cut, and the notebook goes on to the second pass
over what `distill/` already holds.

**2. Measure crosstalk.** Flex, the teacher, is weakest where two people talk at once: on gold it
scores about 6.5% on clean clips, 17% at 5–15% overlap and 29% above 15% (findings.md). A
pseudo-label written over crosstalk is more often wrong, so step 5 can leave such clips out. The
harness's own overlap detector (`app/services/overlap.py`, D77: pyannote's segmentation model as
ONNX, no diarization) reads each whole recording, and every clip gets its overlapped spans and the
share of it that is overlapped, as the labelled export's clips carry them.

**This notebook measures; it drops nothing.** The report at the end prices each threshold: the
hours each channel would lose. Choose the threshold from that table and set it as
`MAX_OVERLAP_SHARE` in `05_Teacher.ipynb`, whose filter applies it. Changing it later needs no
second pass here. Round-table shows are the most overlapped and also bring the most voices, so a
strict threshold can leave one solo commentator carrying the corpus.

**What it writes**, to `DATASET_REPO/distill/`: each recording whole as `episodes/<id>.flac`, its
clip times (and, after the second pass, their overlap) in `sources/<id>.json`, and at the end
`clips.jsonl` (every clip, named and timed as the labelled export's rows) and `summary.json`.
`ftkit.AudioStore` cuts clips from the whole recordings in RAM, as it does for the labelled export,
so the repo holds one file per recording rather than one per clip.

**Gold.** The owner vouches that these channels are new (2026-09-25), so there is no voiceprint
screen here. Gold's shows are still refused by file name, which costs nothing.

**Runtime.** CPU work, but run on the A100 runtime: it has 12 cores where a CPU runtime has 2.
One process per core, each with single-threaded models. The detector costs about 25 s of CPU per
hour of audio on four threads, so 100 h is a quarter of an hour or so on twelve processes.

**Small blast radius.** Both passes work a chunk of recordings at a time and upload each chunk as
it finishes; a rerun skips every recording already cut, and every recording already measured. A
file that cannot be read is reported and skipped, never fatal. `LIMIT` measures only the first
recordings, for a smoke run.
"""

CONFIG = r"""
DATASET_REPO = "Sagyam/nepanglish-asr"   # the owner uploads ZIP_NAME to its root
ZIP_NAME = "distill.zip"                  # cut when it is there; without it only the overlap pass runs
PREFIX = "distill"                        # DATASET_REPO/distill/: episodes/, sources/, clips.jsonl
BLOCKED_CHANNELS = ["chill pill", "prime television"]   # gold's shows (D101), matched by file name
HARNESS_COMMIT = "__HARNESS_COMMIT__"     # ingest's cutting code is fetched at this commit
HARNESS_SHA256 = __HARNESS_SHA256__
WORKERS = None                            # processes; None for every core
CHUNK = None                              # recordings per uploaded chunk; None for WORKERS
THRESHOLDS = (0.05, 0.15, 0.30)           # overlap shares the report prices: the bucket edges and one above
LIMIT = None                              # measure overlap on the first N recordings only (a smoke run)
"""

SETUP_TAIL = r"""
print("cores:", os.cpu_count(), "| ffmpeg:", os.popen("ffmpeg -version").readline().strip())
"""

HARNESS = r"""
# The harness's own code, byte for byte, at HARNESS_COMMIT: ingest's stage 1 and 2, the overlap
# detector, and the helpers they import. Empty package files stand in for the rest of the backend.
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

import distill
import ftkit
import prekit
from app.services.overlap import OverlapDetector
from app.services.silero_vad import SileroVAD

assert SileroVAD().available, "the Silero model did not load"
# Loaded once here, before any worker starts: this is what downloads the pinned graph, so twelve
# processes do not race to fetch it.
assert OverlapDetector(threads=1).available, "the overlap model did not download"
print("harness code at", HARNESS_COMMIT[:7], "checked:", len(HARNESS_SHA256), "files")
"""

FETCH = r"""
import zipfile

from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
MP3 = FT / "recordings"
if api.file_exists(DATASET_REPO, ZIP_NAME, repo_type="dataset"):
    with ftkit.timed(f"downloading {ZIP_NAME}"):
        archive = hf_hub_download(DATASET_REPO, ZIP_NAME, repo_type="dataset", token=TOKEN)
    with ftkit.timed(f"unzipping {Path(archive).stat().st_size / 2**30:.1f} GiB"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(MP3)
else:
    print(f"no {ZIP_NAME} at the root of {DATASET_REPO}: nothing new to cut; "
          f"the overlap pass below works on what {PREFIX}/ already holds")
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

OVERLAP_NOTE = """
## Measure overlapped speech, a chunk at a time

Every recording in `distill/` whose clip list carries no overlap yet goes through
`prekit.measure_overlap`: the harness's detector over the whole recording, then each clip's spans
and share. A chunk's clip lists are uploaded as soon as it is measured.
"""

OVERLAP = r"""
import glob

from huggingface_hub import snapshot_download


def source_names():
    return sorted(f for f in api.list_repo_files(DATASET_REPO, repo_type="dataset")
                  if f.startswith(f"{PREFIX}/sources/") and f.endswith(".json"))


with ftkit.timed("reading which recordings are measured"):
    root = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                                  allow_patterns=[f"{PREFIX}/sources/*.json"]))
metas = {name: json.loads((root / name).read_text("utf-8")) for name in source_names()}
todo = [name for name, meta in metas.items() if not meta.get("overlap")]
if LIMIT:
    todo = todo[:LIMIT]
left = sum(metas[name]["duration"] for name in todo)
print(f"{len(metas) - len(todo)} recordings already measured or left out by LIMIT, "
      f"{len(todo)} to measure ({left / 3600:.1f} h) on {workers} processes")
results, started, measured = [], time.perf_counter(), 0.0
with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
    for number, k in enumerate(range(0, len(todo), chunk), 1):
        part = todo[k:k + chunk]
        stems = [Path(name).stem for name in part]
        stage = FT / "overlap-stage"
        shutil.rmtree(stage, ignore_errors=True)
        t0 = time.perf_counter()
        audio = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                                       allow_patterns=[f"{PREFIX}/episodes/{glob.escape(s)}.flac" for s in stems]))
        got = list(pool.map(prekit.measure_overlap,
                            [str(audio / PREFIX / "episodes" / f"{s}.flac") for s in stems],
                            [str(root / name) for name in part], [str(stage)] * len(part)))
        results += got
        ok = [g for g in got if g["status"] == "measured"]
        measured += sum(metas[name]["duration"] for name in part)
        rate = measured / (time.perf_counter() - started)
        print(f"chunk {number}: {len(ok)} measured, {len(got) - len(ok)} not | "
              f"{sum(g['overlap_seconds'] for g in ok) / 60:.1f} min overlapped in "
              f"{sum(g['duration'] for g in ok) / 3600:.1f} h, {time.perf_counter() - t0:.0f} s | "
              f"about {(left - measured) / rate / 60:.0f} min left", flush=True)
        if ok:
            with ftkit.timed(f"chunk {number}: uploading {len(ok)} clip lists"):
                api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(stage),
                                  path_in_repo=PREFIX, commit_message=f"{PREFIX}: overlap, chunk {number}, {len(ok)} recordings")
for g in results:
    if g["status"] != "measured":
        print(f"{g['status'].upper()} {g['source_id']}: {g.get('error', 'no overlap model')}")
"""

MANIFEST_NOTE = """
## The manifest, the summary, and what a threshold would cost

Rebuilt from every `distill/sources/*.json` on the repo, so it is right whatever earlier runs
did. The table prices each overlap threshold per channel: the hours that a `MAX_OVERLAP_SHARE` of
that value would drop in `05_Teacher.ipynb`. **Pick the threshold from this table.**
"""

MANIFEST = r"""
names = source_names()
with ftkit.timed(f"reading {len(names)} recordings' clip lists"):
    root = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                                  allow_patterns=[f"{PREFIX}/sources/*.json"]))
sources = [json.loads((root / n).read_text("utf-8")) for n in names]
rows = [r for s in sources for r in s["rows"]]
per_channel = {}
for s in sources:
    c = per_channel.setdefault(s["channel"], {"recordings": 0, "audio_h": 0.0, "speech_h": 0.0, "clips": 0})
    c["recordings"] += 1
    c["audio_h"] += s["duration"] / 3600
    c["speech_h"] += s["speech_seconds"] / 3600
    c["clips"] += s["clips"]
loss = distill.overlap_loss(rows, THRESHOLDS)
buckets = {}
for r in rows:
    bucket = distill.overlap_bucket(r.get("overlap_share"))
    buckets[bucket] = buckets.get(bucket, 0.0) + r["duration"] / 3600
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
    "overlap": {
        "measured_recordings": sum(bool(s.get("overlap")) for s in sources),
        "hours_by_bucket": {k: round(v, 2) for k, v in sorted(buckets.items())},
        "per_channel": {name: {"hours": round(c["hours"], 2), "unmeasured_hours": round(c["unmeasured_hours"], 2),
                               "dropped_hours": {str(t): round(h, 2) for t, h in c["dropped_hours"].items()}}
                        for name, c in loss.items()},
    },
}
out = FT / "manifest"
out.mkdir(exist_ok=True)
(out / "clips.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
(out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")
with ftkit.timed("uploading clips.jsonl and summary.json"):
    api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(out), path_in_repo=PREFIX,
                      commit_message=f"{PREFIX}: manifest, {len(rows)} clips, {summary['speech_hours']} h")
print(json.dumps({k: v for k, v in summary.items() if k not in ("per_channel", "overlap")}, indent=1))
for name, c in summary["per_channel"].items():
    print(f"  {name:<40} {c['recordings']:>4} recordings {c['audio_h']:>7.1f} h audio "
          f"{c['speech_h']:>7.1f} h speech {c['clips']:>7} clips")
print(f"\noverlap measured on {summary['overlap']['measured_recordings']} of {len(sources)} recordings; "
      "hours by bucket:", summary["overlap"]["hours_by_bucket"])
print(f"\n{'hours a threshold would drop':<40} {'speech h':>9} {'unmeasured':>11}"
      + "".join(f"{f'> {t:.0%}':>10}" for t in THRESHOLDS))
for name, c in loss.items():
    cells = "".join(f"{f'{h:.1f} ({h / max(c['hours'], 1e-9):.0%})':>10}" for h in c["dropped_hours"].values())
    print(f"  {name:<38} {c['hours']:>9.1f} {c['unmeasured_hours']:>11.1f}{cells}")
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
        nbkit.setup("structlog onnxruntime httpx", tail=SETUP_TAIL),
        *nbkit.kits("ftkit", "distill", "prekit"),
        code(HARNESS),
        md("## The recordings"),
        code(FETCH),
        md(CUT_NOTE),
        code(CUT),
        md(OVERLAP_NOTE),
        code(OVERLAP),
        md(MANIFEST_NOTE),
        code(MANIFEST),
    ]


def head_commit() -> str:
    """The commit whose harness files the notebook fetches: HEAD, whose files must match the
    working tree (the digests are read from it), so build after committing them."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--", *HARNESS_FILES)
    if dirty:
        sys.exit(f"harness files differ from HEAD; commit them first:\n{dirty}")
    return git("rev-parse", "HEAD")


def notebooks(commit: str | None = None) -> dict[str, list]:
    """{file name: cells}, fetching the harness at `commit` (HEAD when not given)."""
    return {"04_PreDistill.ipynb": cells(commit or head_commit())}


if __name__ == "__main__":
    nbkit.write(notebooks(), OUT_DIR)
