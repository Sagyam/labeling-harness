# Nepanglish Annotation Harness

A single-annotator, web-driven annotation harness for a Nepali–English code-switching
("Nepanglish") podcast ASR corpus.

Drop a podcast episode into the browser and it comes back as a prioritized queue: the harness
normalizes the audio, cuts it on speech turns, transcribes every clip with several cloud ASR
models, ranks the segments the models disagree about, and hands them to you one keystroke at a
time. What you approve becomes a versioned, reproducible dataset.

It does not train models, manage users, or serve more than one annotator. It is built for one fast
human.

![Ingestion pipeline: upload or YouTube URL, normalize, VAD segment, transcribe with every cloud ASR route, analyse, import and build the queue](docs/diagrams/ingest-pipeline.svg)

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
fills the title and slug in from the video. Six stages run in the background — normalize, segment,
transcribe, fuse, analyse, build queue — and stream their logs into the panel as they go. When it
finishes, `Start Annotating` drops you straight into the queue.

A URL is looked up before anything is downloaded, so a private, live or over-long video is refused
while you are still typing rather than after you commit to it. The four-hour ceiling is a spend
guard — `ingest.youtube.max_duration_seconds` in `config/settings.yaml` — because cost is linear in
source duration. Ingesting a video is on you as far as its licensing goes; the harness does not
check.

Transcription calls cost money. Three models transcribe every clip — ElevenLabs Scribe v2,
Microsoft MAI-Transcribe 2 and Google Gemini 3.8 Flash — so one clip is three calls, billed
against your ElevenLabs, OpenRouter and Google Cloud accounts. Every attempt is written to
`llm_requests`, which is the only record of what an ingest spent, so watch that table (or a
provider-side budget alert) rather than expecting the harness to stop you. Each model hears only
the audio; none is shown another's transcript, so where they disagree is a measurement rather
than an echo.

**Fusion** is the one step that reads all three. A thinking Gemini 3.8 Flash takes every
recogniser's text for about thirty minutes of clips at once — plus a little of what comes next and
its own output for what came before — and writes one verbatim transcript per clip, English in
Latin and Nepali in Devanagari. That fused text is what the editor opens with (D72). It costs one
request per ~30 minutes of audio, mostly in thinking tokens, and it never hears the audio: the
forced aligner places its words back on the clip, and the queue checks them against the
recognisers and the waveform before anything is allowed to skip listening.

Word-level timestamps do not all come from the models. Scribe, MAI and Gemini 3.5 Transcribe
report their own — and Gemini 3.5 Transcribe also says which speaker said each word, so a clip
with a turn in it can be spotted. Gemini 3.8 Flash returns none, and its spans are measured
locally by a CTC forced aligner that places its transcript back onto the clip
(`backend/app/services/forced_align.py`). That model file is ~300 MB and is not committed — build
it once with `python scripts/export_aligner_onnx.py`. Without it the pipeline still runs, just
without word spans for that system.
Routes are configured in `config/llm_routes.yaml`; set `dry_run: true` there to exercise the
pipeline without spending anything.

**Two pots.** Every clip starts in the **train** pot, subdivided into train and val by its
episode, where a clip may be *screened* — accepted on cross-ASR disagreement without listening.
The **gold** pot is the benchmark, and you choose it one clip at a time: the star on a triage row,
`g`, or "Add to gold" in the editor. The same button takes a clip back out, and every move is
written to the audit log (D71). A clip that was screened cannot go into gold. Choosing clips from
an episode that also feeds train puts the same speaker on both sides of the line; the analytics
page counts those clips and every export row carries `episode_spans_pots`, so a result can be
reported with and without them.

Every label records which it was, `verified` or `screened`, and every export row carries it. The
harness refuses to screen a gold clip, and refuses to write a gold export containing one.

**Triage** is where the time goes. A dense list, highest-priority segment first, with the reason it
surfaced shown next to it. Most segments are correct, so the dominant motion is listen, `Enter`,
move on — or `s` to screen without listening, where the pot allows it. A clip whose fused seed
tripped a **hazard gate** — words no recogniser heard, words two of them heard that it dropped, text
from the neighbouring clip, the wrong length, text the audio cannot hold — carries a red chip, sits
above everything else, and cannot be screened (D74).

**Editor** (`e`) is for the ones that are not: waveform, loopable playback, the transcript, the
other systems' hypotheses, and a live word diff against what you started from. Under the waveform
the seed transcript runs as a karaoke line — the word being spoken grows and lights up, and
clicking any word plays from there. Words the other systems all disagreed with carry a wavy
underline; clicking one shows what each of them heard at that instant, plays just that moment,
and swaps the word in one click. Typing Latin and
pressing `Space` offers Devanagari candidates; `Esc` keeps what you typed. The harness remembers
which candidate you picked and ranks it first next time.

**Episodes** lets you browse what has been ingested and delete an episode or a single segment,
audio and all.

**Corpus** answers three questions in the order you ask them: what is in the dataset, what is
missing from it, and what to go and record next. It opens with a ranked shopping list — the
speaker strata that are empty or thin, the code-switching pole nothing sits at, the show holding
too much of the corpus — each row carrying the measurement that produced it. Below that is the
evidence: a gender-by-age grid whose empty cells are drawn rather than omitted, hours by show,
topic and speaker, the distribution of English mixed in and how far apart the shows are on it, the
two pots against their targets, which episode records have unfilled or off-taxonomy fields, and a
sortable row per show and per episode. Everything is derived on read; nothing is stored.

### Keyboard

| Triage | | Editor | |
|---|---|---|---|
| `j` / `k` | Move between rows | `Ctrl+Space` | Play / pause |
| `Space` | Play / pause row | `Ctrl+Enter` | Save and advance |
| `Enter` | Accept unchanged, advance | `Ctrl+Shift+Enter` | Save and stay |
| `s` | Screen without listening | | |
| `e` | Open in editor | `Alt+1…5` | Load hypothesis 1–5 |
| `f` | Flag unusable audio | `Alt+←` / `Alt+→` | Seek ∓2 s |
| `u` | Mark uncertain | `Ctrl+L` | Toggle loop |
| `g` | Add to / remove from gold | | |
| `x` | Toggle row selection | `Ctrl+T` | Toggle transliteration |
| `Shift+Enter` | Accept selected rows | `Esc` | Back to triage |
| `Shift+S` | Screen selected rows | | |

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
a `manifest.json` recording label version, policy version, filters, row counts, the verified/screened
mix per split, per-file checksums, timestamp, git commit and the contributing import runs. The same inputs produce byte-identical
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
| `OPENROUTER_API_KEY` | OpenRouter key; carries the MAI-Transcribe 2 transcriber |
| `HARNESS_ALIGNER_NO_DOWNLOAD` | Set to 1 to refuse the forced-aligner model download and skip word spans |
| `HARNESS_ALIGNER_MODEL_DIR` | Where the aligner model is kept; the container uses `/app/data/models` |
| `VERTEX_API_KEY` | Vertex AI key, restricted to `aiplatform.googleapis.com`; carries Gemini 3.5 Transcribe and 3.8 Flash |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | Project the Vertex calls bill and quota against, and the region serving them (`global`) |

`ingest.youtube.cookies_file` points at a Netscape-format cookie jar, for videos YouTube declines
to serve anonymously. It is a path in YAML because it is not itself a secret; the file it names is,
so keep it outside the repository.

Object storage defaults to the local filesystem, so the harness is fully usable with MinIO stopped.

## Tests

```bash
cd backend
.venv/bin/python -m pytest              # 581 tests
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
| [docs/diagrams/](docs/diagrams/) | Figure sources (HTML), plus `.svg` and print-ready `.pdf` exports |
