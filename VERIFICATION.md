# Verification

Status of each phase's verification criteria. A phase is not complete until every criterion here
passes; failures are recorded rather than hidden.

Legend: ✅ pass · ❌ fail · ⏳ not yet run · ⛔ deferred (out of this session's scope)

## Phase 0 — Scaffolding — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| `docker compose up` starts all four services | ✅ | postgres, minio, backend, frontend all `Up`; `docker-compose ps` clean |
| Backend connects to Postgres | ✅ | `GET /health` → `{"status":"ok","checks":{"database":{"ok":true}}}` |
| Backend connects to MinIO | ✅ | In-container put/get round trip against `MinioStorage` succeeded |
| `alembic upgrade head` and `downgrade base` both succeed | ✅ | Both exit 0 against the local Postgres |
| `pytest` passes | ✅ | 44 passed |
| `ruff check` passes | ✅ | "All checks passed!"; `ruff format --check` clean |
| README explains startup in under ten commands | ✅ | 7-command quick start, or a single `docker compose up --build` |

Note: `docker compose` (CLI plugin) is not installed on the development machine; the standalone
`docker-compose` v5.5.0 binary was used, which speaks the same Compose v2 file format.

## Phase 1 — Schema and migrations — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| Migrations run from empty and roll back cleanly | ✅ | `alembic upgrade head` → `downgrade base` → `upgrade head`, all exit 0; the suite rebuilds its schema this way on every run |
| Seed creates 1 episode, 20 segments, 3 ASR systems, hypotheses for each segment | ✅ | `scripts/seed_dev_data.py` → `episodes=1 segments=20 systems=3 hypotheses=60`; asserted in `test_seed.py` |
| A second active task for the same segment is rejected by the database, not application code | ✅ | `test_second_active_task_for_a_segment_is_rejected_by_the_database` raises `IntegrityError` from the partial unique index `uq_annotation_tasks_active_segment`; `test_a_finished_task_does_not_block_a_new_one` proves the predicate is partial |
| Foreign keys reject orphan rows | ✅ | `test_orphan_segment_is_rejected`, `test_orphan_hypothesis_is_rejected` |
| Tests cover insert and query of every core entity | ✅ | `test_models.py` round-trips all 14 tables; CHECK constraints on split, pipeline_status, queue and disposition are each asserted |
| Indexes for queue ordering, segment lookup and export filtering | ✅ | `test_required_index_exists` asserts five named indexes exist in `pg_indexes` |
| Schema documented in ARCHITECTURE.md | ✅ | "Data model" section |

86 tests passing, `ruff check` and `ruff format --check` clean.
