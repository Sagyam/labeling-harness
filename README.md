# Nepanglish Annotation Harness

A single-annotator, web-driven annotation harness for a Nepali–English code-switching
("Nepanglish") podcast ASR corpus.

It does three things: **import** a manifest produced by an upstream GPU pipeline, **annotate** it
through a fast keyboard-first web UI, and **export** reproducible, versioned datasets.

It does not run ASR, train models, or manage users. Everything the harness knows arrives through
the manifest contract described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

```bash
cp .env.example .env                              # 1. configure (defaults work locally)
docker compose up -d postgres minio               # 2. start dependencies
cd backend && uv venv --python 3.13               # 3. create the virtualenv
uv pip install -e ".[dev]"                        # 4. install the backend
.venv/bin/alembic upgrade head                    # 5. create the schema
.venv/bin/python -m pytest                        # 6. run the suite
.venv/bin/uvicorn app.main:app --reload           # 7. serve the API on :8000
```

Or run the whole stack in containers:

```bash
docker compose up --build                         # postgres, minio, backend, frontend
```

`docker compose` v2 syntax is required. On systems where the Docker CLI plugin is not installed,
the standalone `docker-compose` binary works identically.

## Command-line scripts

All scripts live in `scripts/` and are run from the repository root with the backend virtualenv:

```bash
backend/.venv/bin/python scripts/import_manifest.py  export_show-a_ep012/ [--dry-run]
backend/.venv/bin/python scripts/build_queue.py      [--episode show-a_ep012]
backend/.venv/bin/python scripts/export_dataset.py   --kind training --label-version v1
backend/.venv/bin/python scripts/report_status.py    [--format html]
backend/.venv/bin/python scripts/seed_dev_data.py    # synthetic data for development
```

## Status

Backend phases 0-4, 7 and 8 are complete and verified; see [VERIFICATION.md](VERIFICATION.md) for
the criteria and their evidence. The review UI (Phase 6) is not built yet - `frontend/` is a
minimal shell - so annotation is currently done through the API.

## Configuration

`config/settings.yaml` holds non-secret configuration. Any value can be overridden with an
environment variable named `HARNESS_<SECTION>__<KEY>` (double underscore separates nesting), for
example `HARNESS_DATABASE__PORT=5433` or `HARNESS_STORAGE__BACKEND=minio`.

Secrets come from the environment only, never from YAML:

| Variable | Purpose |
|---|---|
| `HARNESS_DATABASE__PASSWORD` or `DATABASE_URL` | Postgres credentials |
| `HARNESS_STORAGE__MINIO__ACCESS_KEY` / `__SECRET_KEY` | MinIO credentials |
| `HARNESS_API__AUTH_TOKEN` | Optional static bearer token; empty disables auth |
| `OPENROUTER_API_KEY` | OpenRouter key (unused at MVP) |

Object storage defaults to the local filesystem, so the harness is fully usable with MinIO stopped.

## Tests

```bash
cd backend
.venv/bin/python -m pytest              # full suite
.venv/bin/python -m pytest -m "not db"  # skip tests that need Postgres
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
```

Tests marked `db` run against `TEST_DATABASE_URL` (default: a `harness_test` database on
localhost), which the suite creates and migrates itself. Tests marked `minio` skip when MinIO is
not reachable.

A pre-commit hook running lint and the suite is installed with `git config core.hooksPath .githooks`.

## Documentation

| File | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Manifest contract, schema, priority formula, module map |
| [DECISIONS.md](DECISIONS.md) | Design decisions and their rationale |
| [PROGRESS.md](PROGRESS.md) | Session log: what completed, what is blocked |
| [VERIFICATION.md](VERIFICATION.md) | Per-phase verification criteria and their results |
| [TODO.md](TODO.md) | The full project specification and phase plan |
