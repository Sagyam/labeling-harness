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

## Phase 2 — Manifest importer — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| Importing the same export twice inserts nothing the second time and reports it | ✅ | CLI run: second import reports `0 inserted, 6 unchanged` / `0 inserted, 18 unchanged`; `test_reimport_of_an_unchanged_export_inserts_nothing`, `test_reimport_does_not_duplicate_words` |
| A changed clip checksum errors unless explicitly overridden | ✅ | `test_changed_clip_checksum_is_an_error` (message names `--allow-clip-change`), `test_changed_clip_is_accepted_with_the_override` |
| A malformed manifest fails before any write; the database is unchanged | ✅ | `test_a_malformed_manifest_writes_nothing`, `test_a_rejected_import_writes_nothing` — validation and clip probing run as a planning pass before the first insert |
| Non-FLAC or non-16 kHz clips are rejected with a clear message | ✅ | `test_non_flac_clip_is_rejected`, `test_wrong_sample_rate_is_rejected`, `test_stereo_clip_is_rejected`; messages name the file and the reason |
| Peaks JSON exists for every imported segment | ✅ | `test_peaks_exist_for_every_segment` asserts the object exists in storage with the configured bucket count; supplied peaks are reused (`test_supplied_peaks_are_used_instead_of_recomputed`) |
| Split assignment is deterministic for a given seed and stable across re-imports | ✅ | `test_split_is_assigned_at_import`, `test_split_is_stable_across_reimport`, `test_split_is_deterministic_for_a_given_seed`, plus the pinned hash vector in `test_splits.py` |
| Dry-run prints planned changes and writes nothing | ✅ | CLI dry run printed the plan; `test_dry_run_writes_nothing_to_the_database`, `test_dry_run_writes_nothing_to_storage`, `test_dry_run_after_a_real_import_reports_a_no_op` |
| JSON Schema for both files, validated at import | ✅ | `app/schemas/episode.schema.json`, `app/schemas/segment.schema.json`; `test_manifest.py` covers missing ids, empty hypotheses, duplicate segment/system ids, cross-episode records and malformed lines |
| An `import_runs` row records the run | ✅ | `test_import_records_an_import_run`, `test_segments_link_to_their_import_run` |

161 tests passing, lint and format clean.
