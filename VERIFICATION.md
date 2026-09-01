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

## Phase 3 — Queue building — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| Queue builder creates tasks and is idempotent | ✅ | `test_build_creates_a_task_for_every_segment`, `test_rebuilding_does_not_duplicate_active_tasks` (second build creates 0), `test_a_new_import_adds_only_the_new_tasks`, `test_rebuilding_leaves_completed_tasks_alone` |
| Top-priority segments are visibly the disagreeing ones on the seed data | ✅ | `test_top_priority_segments_are_the_disagreeing_ones` — mean `word_disagreement_rate` of the top four exceeds the bottom four |
| Audit sampling is reproducible under a fixed seed | ✅ | `test_audit_sampling_is_reproducible_under_a_fixed_seed`; the sample is drawn from the low-priority half (`test_audit_queue_samples_easy_segments`) |
| Test-episode seeds are distributed across systems, not concentrated on one | ✅ | `test_test_episode_seeds_rotate_across_systems` — 40 segments, 3 systems, every system seeds at least 5; `test_test_episode_seed_rotation_is_deterministic` |
| Train/val episodes seed with the strongest hypothesis | ✅ | `test_train_episodes_seed_with_the_strongest_hypothesis` |
| `reason_jsonb` explains every priority score | ✅ | `test_every_task_carries_a_priority_and_a_reason` asserts all four components, weights, contributions and the score itself; `test_reason_explains_why_a_segment_surfaced` |
| Segments with zero hypotheses go to the error queue, never review | ✅ | `test_segments_without_hypotheses_go_to_the_error_queue`, `test_error_queue_segments_never_reach_review` |
| Rule flags computed at import | ✅ | `app/services/flags.py` (7 rules); wired into the importer and asserted by `test_rule_flags_are_computed_at_import`; upstream flags are preserved (`test_imported_flags_are_preserved_alongside_computed_ones`) |
| Priority formula documented in ARCHITECTURE.md | ✅ | "Priority formula" section, including the normalization of each input |

213 tests passing, lint and format clean.

## Phase 4 — Review API — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| Automated API tests cover every endpoint including error paths | ✅ | `test_api.py`, 47 tests: 404 on unknown task/segment, 409 on deciding a finished task, 422 on bad disposition/missing text/empty bulk list, 416 on bad ranges, 401 on a bad token |
| Accepting writes a label with `disposition='accepted_unchanged'` and non-null `duration_ms` | ✅ | `test_accept_writes_a_label_an_event_and_an_audit_entry` |
| Bulk accept writes one label and one event per task, transactionally | ✅ | `test_bulk_accept_writes_one_label_and_event_per_task`, `test_bulk_accept_is_all_or_nothing` (a bad id in the list rolls the whole batch back) |
| Range requests return 206 with correct byte ranges | ✅ | `test_audio_range_request_returns_206_with_the_right_bytes`, plus open-ended and suffix ranges; live check returned `content-range: bytes 0-99/167280` |
| Timing reflects real elapsed time, not server processing time | ✅ | `test_accept_records_real_elapsed_time_not_server_time` (12 s of client elapsed time recorded), `test_explicit_duration_wins_over_opened_at` |
| Every write creates a `segment_labels` row and an `annotation_events` row | ✅ | Asserted per endpoint; **exception**: skip writes an event and audit entry but no label, by design — see DECISIONS D14 |
| Every write creates an `audit_logs` entry | ✅ | `test_accept_writes_a_label_an_event_and_an_audit_entry`, `test_skip_writes_an_event_but_no_label` |
| Optional static bearer token | ✅ | `test_no_authentication_is_required_by_default`, `test_a_configured_token_is_enforced` |

## Phase 5 — Devanagari input helper (backend only) — ✅ service complete, ⛔ UI deferred

| Criterion | Status | Evidence |
|---|---|---|
| Provider interface has a mock implementation used in tests | ✅ | `app/translit/mock.py`; `test_mock_provider_satisfies_the_interface` |
| Cache hit returns without a network call | ✅ | `test_cache_hit_never_touches_a_provider` asserts the provider is consulted exactly once across repeated lookups |
| Remote provider failure falls back to offline without an error | ✅ | `test_a_failing_provider_falls_through_to_the_next`, plus remote timeout/bad-status/garbled-payload tests |
| Correction memory ranks a previous choice first | ✅ | `test_correction_memory_ranks_a_previous_choice_first` |
| Candidates cached case-insensitively, capped, hit-counted | ✅ | `test_cache_lookup_is_case_insensitive`, `test_candidates_are_capped_at_the_configured_maximum`, `test_cache_hits_are_counted` |
| `Esc` leaves the Latin token untouched | ⛔ | UI behaviour — Phase 6, deferred |
| Candidate popup fully keyboard-operable | ⛔ | UI behaviour — Phase 6, deferred |
| Manual check: twenty common Nepali words | ⛔ | Requires the owner; the live endpoint returned `["कुरा","कुर","कूरा","कूर","कउरा"]` for `kura` |

281 tests passing, lint and format clean. Live smoke test against the Docker stack: `/stats`,
`/queue`, `/tasks/next`, `/tasks/{id}/accept`, `/translit` and a ranged audio request all behaved.

## Section 3 — OpenRouter client — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| Thin client: key from env, route-based model, per-route timeout and max tokens | ✅ | `app/llm/openrouter.py`; `test_the_request_targets_openrouter_with_the_route_model` |
| Retries with backoff | ✅ | `test_a_rate_limit_is_retried_then_succeeds`, `test_retries_are_bounded`, `test_a_timeout_is_retried`, `test_a_client_error_is_not_retried` |
| Dry-run mode | ✅ | `test_dry_run_makes_no_http_call` (the transport raises if touched), `test_per_call_dry_run_overrides_the_configuration` |
| Request logging to `llm_requests` | ✅ | `test_every_call_is_logged`, `test_a_failure_is_logged_with_the_error`, `test_dry_run_is_still_logged` |
| `config/llm_routes.yaml` has `routes: {}` and `enabled: false` | ✅ | `test_the_committed_configuration_is_disabled_and_empty`; a disabled client refuses to call |
| Covered by tests against a mocked HTTP layer | ✅ | 17 tests, all through `httpx.MockTransport`; the suite never makes a paid call |

## Phase 7 — Export — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| Two consecutive exports produce identical checksums | ✅ | `test_two_consecutive_exports_are_byte_identical`; records are ordered by `segment_id` and JSON keys are sorted |
| A test-split segment appearing in a training export fails a test | ✅ | `test_a_test_segment_never_appears_in_the_training_export` — asserted against the actual test-split segment ids, not assumed |
| Row counts in the manifest match the files | ✅ | `test_manifest_row_counts_match_the_files`, `test_manifest_checksum_matches_the_data_file` |
| Export succeeds when word timestamps are entirely absent | ✅ | `test_analytics_export_succeeds_without_word_timestamps` |
| Exported text matches the current label for every segment | ✅ | `test_exported_text_matches_the_current_label`, `test_only_the_latest_label_is_exported` (labels are append-only, so "current" means newest) |
| `unusable_audio` never appears in training or gold | ✅ | `test_unusable_audio_never_appears_in_training_or_gold` |
| `disposition` and `seed_system_id` retained | ✅ | `test_disposition_and_seed_system_are_retained` |
| Manifest records version, policy, filters, counts, checksums, timestamp, commit, import runs | ✅ | `test_manifest_records_provenance` |
| All four export kinds | ✅ | training, gold, analytics, error_mining — one test class each |
| JSONL always; Parquet if cheap | ✅ / ⛔ | JSONL implemented. Parquet **not** added: it would pull in pyarrow (~100 MB) for a corpus this size, which is not "cheap"; revisit when the archive is loaded at scale |

## Phase 8 — Status report — ✅ complete (2026-09-01)

| Criterion | Status | Evidence |
|---|---|---|
| One command, no SQL knowledge required | ✅ | `python scripts/report_status.py` (`--format text|html|json`) |
| Report includes throughput and projected completion | ✅ | `test_report_includes_throughput`, `test_report_projects_completion`; live run shows median seconds/segment, segments/hour, annotator hours, backlog and projected hours |
| Report runs against an empty database without crashing | ✅ | `test_report_runs_against_an_empty_database` (text and HTML both render) |
| Episodes, audio hours, segments by status | ✅ | `test_report_counts_the_corpus` |
| Labels by disposition, accept rate over time | ✅ | `test_report_breaks_down_dispositions_and_accept_rate`, `test_report_includes_accept_rate_over_time` |
| Score distributions | ✅ | `test_report_includes_score_distributions` |
| Split balance in hours | ✅ | `test_report_includes_split_balance_in_hours` |
| Word timestamp coverage | ✅ | `test_report_includes_word_timestamp_coverage` |
| HTML output is safe | ✅ | `test_html_escapes_untrusted_text` — episode titles come from an upstream manifest and are escaped |

## Phase 6 — Review UI — ⛔ deferred

Not started, by agreement: the triage and editor modes need interactive keyboard testing with the
owner present. `frontend/` holds a minimal Vite + React shell that calls `/health` and `/stats`, so
the compose stack is real, but no triage list, editor, waveform, shortcut map or progress display
exists yet. The manual throughput baseline (50 segments end to end, median seconds per segment) and
the transliteration popup criteria in Phase 5 depend on this phase and are also outstanding.
