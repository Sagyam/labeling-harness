# Nepanglish Annotation Harness

A single-annotator, web-driven annotation harness for a Nepali–English code-switching
("Nepanglish") podcast ASR corpus.

Drop a podcast episode into the browser and it comes back as a prioritized queue: the harness
normalizes the audio, cuts it on speech turns, transcribes every clip with several cloud ASR
models, ranks the segments the models disagree about, and hands them to you one keystroke at a
time. What you approve becomes a versioned, reproducible dataset.

It does not train models, manage users, or serve more than one annotator. It is built for one fast
human.

## Quick start

Everything in containers:

```bash
cp .env.example .env              # defaults work locally
docker compose up --build         # postgres, minio, backend :8000, frontend :5173
```

Then open <http://localhost:5173>.

Running the backend on the host instead (what you want while developing):

```bash
cp .env.example .env
docker compose up -d postgres minio       # just the dependencies
cd backend
uv venv --python 3.13
uv pip install -e ".[dev]"
.venv/bin/alembic upgrade head            # create the schema
.venv/bin/uvicorn app.main:app --reload   # API on :8000
cd ../frontend && npm install && npm run dev
```

Requires Docker Compose v2 (the standalone `docker-compose` binary works identically), and
`ffmpeg` on the host if you ingest audio outside the container. `yt-dlp` is a Python dependency of
the backend, so YouTube ingestion needs no separate install.

## Using it

**Ingest.** `+ Ingest` in the header takes an `.mp3`, `.m4a` or `.wav`, plus a show and episode
title — or a **YouTube URL**, in which case the server fetches the audio itself with `yt-dlp` and
fills the title and slug in from the video. Five stages run in the background — normalize, segment,
transcribe, analyse, build queue — and stream their logs into the panel as they go. When it
finishes, `Start Annotating` drops you straight into the queue.

A URL is looked up before anything is downloaded, so a private, live or over-long video is refused
while you are still typing rather than after you commit to it. The four-hour ceiling is a spend
guard — `ingest.youtube.max_duration_seconds` in `config/settings.yaml` — because cost is linear in
source duration. Ingesting a video is on you as far as its licensing goes; the harness does not
check.

Transcription calls cost money. Two models transcribe every clip — ElevenLabs Scribe v2 and
Gemini 3.8 Flash — so one clip is two calls, billed against your ElevenLabs and OpenRouter
balances. Both are prepaid, which is the whole reason those two providers are the ones wired up.
Routes are configured in `config/llm_routes.yaml`; set `dry_run: true` there to exercise the
pipeline without spending anything.

**Triage** is where the time goes. A dense list, highest-priority segment first, with the reason it
surfaced shown next to it. Most segments are correct, so the dominant motion is listen, `Enter`,
move on.

**Editor** (`e`) is for the ones that are not: waveform, loopable playback, the transcript, the
other systems' hypotheses, and a live word diff against what you started from. Typing Latin and
pressing `Space` offers Devanagari candidates; `Esc` keeps what you typed. The harness remembers
which candidate you picked and ranks it first next time.

**Episodes** lets you browse what has been ingested and delete an episode or a single segment,
audio and all.

### Keyboard

| Triage | | Editor | |
|---|---|---|---|
| `j` / `k` | Move between rows | `Ctrl+Space` | Play / pause |
| `Space` | Play / pause row | `Ctrl+Enter` | Save and advance |
| `Enter` | Accept unchanged, advance | `Ctrl+Shift+Enter` | Save and stay |
| `e` | Open in editor | `Alt+1…5` | Load hypothesis 1–5 |
| `f` | Flag unusable audio | `Alt+←` / `Alt+→` | Seek ∓2 s |
| `u` | Mark uncertain | `Ctrl+L` | Toggle loop |
| `x` | Toggle row selection | `Ctrl+T` | Toggle transliteration |
| `Shift+Enter` | Accept selected rows | `Esc` | Back to triage |

In the transliteration popup: `1`–`5` pick a candidate, `Enter` takes the first, `Esc` keeps the
Latin exactly as typed. `?` opens the full list in the app.

## Command-line scripts

Run from the repository root with the backend virtualenv:

```bash
backend/.venv/bin/python scripts/import_manifest.py  export_show-a_ep012/ [--dry-run]
backend/.venv/bin/python scripts/build_queue.py      [--episode show-a_ep012]
backend/.venv/bin/python scripts/export_dataset.py   --kind training --label-version v1
backend/.venv/bin/python scripts/align_and_verify_timestamps.py  [--input exports/analytics/analytics.jsonl]
backend/.venv/bin/python scripts/report_status.py    [--format html]
backend/.venv/bin/python scripts/seed_dev_data.py    # synthetic data for development
```

`export_dataset.py` writes four kinds — `training`, `gold`, `analytics`, `error_mining` — each with
a `manifest.json` recording label version, policy version, filters, row counts, per-file checksums,
timestamp, git commit and the contributing import runs. The same inputs produce byte-identical
output. For `analytics`, `timestamp_verification_report.json` is generated automatically.

## Configuration

`config/settings.yaml` holds everything non-secret. Any value can be overridden by an environment
variable named `HARNESS_<SECTION>__<KEY>` — a double underscore separates nesting levels — for
example `HARNESS_DATABASE__PORT=5433` or `HARNESS_STORAGE__BACKEND=minio`.

Secrets come from the environment only, never from YAML:

| Variable | Purpose |
|---|---|
| `HARNESS_DATABASE__PASSWORD` or `DATABASE_URL` | Postgres credentials |
| `HARNESS_STORAGE__MINIO__ACCESS_KEY` / `__SECRET_KEY` | MinIO credentials |
| `HARNESS_API__AUTH_TOKEN` | Optional static bearer token; empty disables auth |
| `OPENROUTER_API_KEY` | OpenRouter key; carries the Gemini transcriber |
| `ELEVEN_LABS_API_KEY` | ElevenLabs key for Scribe v2; scope it to speech-to-text only |

`ingest.youtube.cookies_file` points at a Netscape-format cookie jar, for videos YouTube declines
to serve anonymously. It is a path in YAML because it is not itself a secret; the file it names is,
so keep it outside the repository.

Object storage defaults to the local filesystem, so the harness is fully usable with MinIO stopped.

## Tests

```bash
cd backend
.venv/bin/python -m pytest              # 496 tests
.venv/bin/python -m pytest -m "not db"  # skip the ones that need Postgres
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
cd ../frontend && npm run build         # tsc -b && vite build
```

Tests marked `db` run against `TEST_DATABASE_URL` (default: a `harness_test` database on
localhost), which the suite creates and migrates itself, using the real Alembic migrations rather
than `create_all`. Tests marked `minio` skip when MinIO is unreachable.

Install the pre-commit hook (lint, format check, full suite) with
`git config core.hooksPath .githooks`.

## Status

Complete and in use: ingestion, queue building, triage, editor, transliteration, export, and the
status report. The measured baseline over a 50-segment run is **1.7 seconds per segment** median, at
a 76% accept rate.

## Documentation

| Document | Contents |
|---|---|
| [AGENTS.md](AGENTS.md) | Working agreement for coding agents: rules, conventions, gotchas |
| [docs/architecture.md](docs/architecture.md) | Ingestion pipeline, schema, priority formula, API, module map |
| [docs/decisions.md](docs/decisions.md) | Every design decision, why it was made, what reversing costs |
| [docs/manifest-contract.md](docs/manifest-contract.md) | The import format, and the transcript policy |
