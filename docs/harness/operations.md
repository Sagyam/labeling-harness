# Operations

Running, configuring and testing the harness. For what it does, see the
[README](../../README.md) and the [guide](guide.md); for how it is built,
[architecture.md](architecture.md).

## Development setup

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

## Command-line scripts

Run from the repository root with the backend virtualenv:

```bash
backend/.venv/bin/python scripts/build_queue.py      [--episode show-a_ep012]
backend/.venv/bin/python scripts/export_dataset.py   --kind training --label-version v1
backend/.venv/bin/python scripts/align_and_verify_timestamps.py  [--input exports/analytics/analytics.jsonl]
backend/.venv/bin/python scripts/report_status.py    [--format html]
backend/.venv/bin/python scripts/backfill_overlap.py [--episode show-a_ep012] [--force]
backend/.venv/bin/python scripts/diarize_episode.py  show-a_ep012 [--num-speakers 2]   # on the Modal GPU
backend/.venv/bin/python scripts/import_models.py    [data/models/asr/flex-ft]   # fine-tuned models (D83)
backend/.venv/bin/python scripts/seed_dev_data.py    # synthetic data for development
backend/.venv/bin/python scripts/seed_demo.py        # invented corpus for demos (harness_demo only)
```

`export_dataset.py` writes four kinds — `training`, `gold`, `analytics`, `error_mining` — each with
a `manifest.json` recording label version, policy version, filters, row counts, the verified/screened
mix per split, per-file checksums, timestamp, git commit and the contributing import runs. The same inputs produce byte-identical
output. For `analytics`, `timestamp_verification_report.json` is generated automatically. A
`training` row carries its label's timed words; edited labels are realigned on their clips, which
loads the forced aligner (`--no-realign` skips it).

### The demo corpus

`scripts/seed_demo.py` fills a separate database with three invented shows: speech-shaped audio
(it says nothing), fused seeds and recogniser hypotheses, voices, labels in both tiers, gold
clips, a cost log and one scored model. It is what the screenshots are taken from. It refuses any
database not named `*_demo`, so point every store elsewhere first:

```bash
export HARNESS_DATABASE__NAME=harness_demo
export HARNESS_STORAGE__LOCAL_ROOT=./data/demo/objects
export HARNESS_MODELS__ROOT=./data/demo/models
backend/.venv/bin/python scripts/seed_demo.py    # creates and migrates the database itself
```

Start the backend with the same variables set.

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
| `HARNESS_OVERLAP_NO_DOWNLOAD` | Set to 1 to refuse the overlap detector's model download (6 MB); clips then go unmeasured |
| `HARNESS_ALIGNER_MODEL_DIR` | Where the aligner model is kept; the container uses `/app/data/models` |
| `VERTEX_API_KEY` | Vertex AI key, restricted to `aiplatform.googleapis.com`; carries Gemini 3.5 Transcribe and 3.8 Flash |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | Project the Vertex calls bill and quota against, and the region serving them (`global`) |
| `HARNESS_DIARIZATION__AUTH_TOKEN` | Modal proxy token for the diarizer, `<wk-id>.<ws-secret>` (`modal workspace proxy-tokens create`); without it ingest skips speaker turns |

`ingest.youtube.cookies_file` points at a Netscape-format cookie jar, for videos YouTube declines
to serve anonymously. It is a path in YAML because it is not itself a secret; the file it names is,
so keep it outside the repository.

Object storage defaults to the local filesystem, so the harness is fully usable with MinIO stopped.

## Tests

```bash
cd backend
.venv/bin/python -m pytest              # 1871 tests
.venv/bin/python -m pytest -m "not db"  # skip the ones that need Postgres
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
cd ../frontend && npm run build         # tsc -b && vite build
```

Tests marked `db` run against `TEST_DATABASE_URL` (default: a `harness_test` database on
localhost), which the suite creates and migrates itself, using the real Alembic migrations rather
than `create_all`. Tests marked `minio` skip when MinIO is unreachable.

Install the git hooks with `git config core.hooksPath .githooks`. Pre-commit runs lint and, when
the staged diff touches `backend/`, `scripts/`, `config/`, `notebooks/src/` or `docker-compose.yml`,
the not-db suite; pre-push always runs the full suite (D97).
