# Architecture

## System boundary

```text
   GPU pipeline (Colab/Kaggle, out of scope)
   normalize -> diarize -> segment -> LID -> multi-system ASR -> scores
                              |
                              v
                    export_<episode_id>/          <-- the contract
                              |
   ┌──────────────────────────┴───────────────────────────────────────┐
   │ Harness                                                          │
   │   import  ->  queue build  ->  annotate (web UI)  ->  export      │
   │   Postgres = source of truth; MinIO/local FS = clips and peaks    │
   └──────────────────────────────────────────────────────────────────┘
```

The harness never loads an ASR model, never trains, and knows nothing that did not arrive through
the manifest.

## Module map

```text
backend/app/
  config.py        YAML + env configuration, frozen pydantic models
  main.py          FastAPI application factory
  api/             HTTP routers (health, queue, tasks, segments, translit, stats)
  db/              engine, session scope, declarative base
  models/          SQLAlchemy ORM models -- one module per concept group
  services/        importer, peaks, scoring, queue builder, labeling, export, reporting
  storage/         ObjectStorage interface + local filesystem and MinIO implementations
  translit/        Latin -> Devanagari providers and the cache
  llm/             OpenRouter client (no route wired at MVP)
  utils/           logging, hashing, time
scripts/           thin CLI wrappers over services
config/            settings.yaml, llm_routes.yaml
```

Import and queue building are pure functions over a batch with no global state, so moving them
behind a job queue later is a wiring change, not a rewrite.

## Input contract: the manifest

```text
export_<episode_id>/
  episode.json          object described by app/schemas/episode.schema.json
  segments.jsonl        one object per line, app/schemas/segment.schema.json
  clips/<segment_id>.flac    16 kHz mono FLAC, rejected otherwise
  peaks/<segment_id>.json    optional; generated at import when absent
```

Both files are validated against JSON Schema before a single row is written. Import is idempotent,
keyed on `segment_id` for segments and `(segment_id, system_id)` for hypotheses. A changed clip
checksum is an error unless `--allow-clip-change` is passed.

## Data model

Postgres is the source of truth. All timestamps are `timestamptz` in UTC.

### Provenance and content

| Table | Purpose |
|---|---|
| `import_runs` | One row per import invocation, with counts and status |
| `episodes` | Episode metadata plus the **frozen** train/val/test split |
| `segments` | Time span, clip and peaks object keys, `pipeline_status` |
| `asr_systems` | One row per upstream ASR system |
| `asr_hypotheses` | Immutable imported transcripts, one per (segment, system) |
| `hypothesis_words` | Optional word-level timings, languages and scripts |
| `segment_scores` | Imported agreement scores and rule flags, one row per segment |

### Annotation

| Table | Purpose |
|---|---|
| `annotation_tasks` | Queue rows with `priority_score`, `reason_jsonb`, `seed_hypothesis_id` |
| `label_versions` | Named label sets carrying a `policy_version` |
| `segment_labels` | Append-only human decisions; latest row per (segment, label_version) is current |
| `annotation_events` | Timing and action per interaction, for throughput measurement |
| `audit_logs` | Every write, with old and new values |
| `translit_cache` | Latin token -> ranked Devanagari candidates |
| `llm_requests` | OpenRouter request log (table exists; no route wired at MVP) |

### Status discipline

Exactly three status fields, each with one owner:

- `segments.pipeline_status` — `imported | queued | labeled | excluded`
- `annotation_tasks.status` — `pending | in_progress | done | skipped`
- `segment_labels.disposition` — `accepted_unchanged | edited | unusable_audio | uncertain`

No fourth status, and no boolean that duplicates one. `segment_labels` rows are append-only.

### Frozen splits

Split assignment happens once, at episode import, hashed from `(episode_id, split_seed)` and stored
in `episodes.split`. It is never recomputed. Without a stored split the train/test division would be
recalculated at every export, so adding episodes would silently migrate segments across the boundary
and two exports of "the same" dataset would differ. Splits are at **episode** level: segments from
one episode share speaker, recording conditions and topic, so a segment-level split leaks.

## Priority formula

```text
priority_score =
    0.40 * word_disagreement_rate
  + 0.25 * low_confidence            (normalized from avg_logprob)
  + 0.20 * code_switch_density
  + 0.15 * rule_flag_score
```

Every input is normalized to 0–1 and the weights sum to 1, so the score is itself in 0–1. Weights
live in `config/settings.yaml` under `queue.weights` and are validated to sum to 1.0 at load.

- `word_disagreement_rate` — imported; missing is treated as 0.
- `low_confidence` — `clamp((logprob_floor - avg_logprob) / logprob_floor, 0, 1)` over the seed
  hypothesis, where `logprob_floor` defaults to −2.0.
- `code_switch_density` — imported; missing is treated as 0.
- `rule_flag_score` — fraction of rule flags raised for the segment (see below).

The per-component breakdown is stored in `annotation_tasks.reason_jsonb`, so the UI can always show
why a segment surfaced.

### Rule flags (computed at import)

`empty_transcript`, `repeated_ngram` (hallucination pattern), `high_no_speech_prob`,
`too_short` (< 1 s), `too_long` (> 30 s), `implausible_speaking_rate`, `script_conflict`.

### Seed hypothesis selection

- `train`/`val` episodes: the highest-scoring hypothesis, because accepting it is the fastest path.
- `test` episodes: the seed system is **rotated deterministically** by hashing `segment_id` across
  available systems, and recorded in `annotation_tasks.seed_hypothesis_id` and on the label. The
  rotation costs the annotator nothing but keeps the gold set from being anchored to one system, and
  makes a per-seed WER breakdown possible later.

Segments with zero hypotheses go to the `error` queue, never to `review`. An audit queue takes a
seeded random sample (default 5%) of low-priority, high-agreement segments so quality on the easy
majority stays measurable.

## LLM routing

All LLM inference goes through OpenRouter (`backend/app/llm/openrouter.py`), which is prepaid, so
there is no possibility of a surprise invoice. At MVP no route is wired: `config/llm_routes.yaml`
has `enabled: false` and `routes: {}`. The client and the `llm_requests` table exist so that adding
a route later inherits the billing, logging, retry and dry-run guarantees already in place.

Upstream ASR — including any commercial transcription API — runs in the GPU pipeline, outside this
codebase, and is out of scope for this rule.

## Export

Four export kinds, each writing `manifest.json` next to the data:

1. **training** — `train` + `val` splits, approved labels only.
2. **gold** — `test` split only, retaining `seed_system_id` per segment.
3. **analytics** — includes word-level fields where hypothesis words were imported.
4. **error_mining** — `uncertain` and `unusable_audio` dispositions, for pipeline debugging.

The manifest records label version, policy version, filters, split row counts, SHA-256 of each
output file, timestamp, git commit and the contributing `import_runs`. Exports are deterministic:
the same inputs and filters produce byte-identical output.
