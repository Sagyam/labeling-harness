# Agent Prompt: Nepanglish Annotation Harness

You are building a **single-annotator, web-driven annotation harness** for a Nepali–English
code-switching ("Nepanglish") podcast ASR corpus.

This will be built over many sessions. Work in phases, with explicit TODOs, verification criteria,
and reversible decisions.

---

# 0. Scope: what this system is, and what it is not

**This harness does three things:**

1. **Import** a manifest produced by an upstream GPU pipeline (segments + clips + multi-system ASR
   hypotheses + agreement scores).
2. **Annotate** — a fast web UI where one human accepts or corrects transcripts.
3. **Export** — reproducible, versioned training/test datasets.

**This harness does NOT:**

- Run ASR. Audio normalization, diarization, segmentation, language ID, multi-system transcription,
  token tagging, and agreement scoring all happen **upstream**, in Colab/Kaggle notebooks on GPU.
  The harness never loads an ASR model.
- Train models.
- Manage users, teams, or permissions.
- Do image/video annotation.
- Serve multiple tenants.
- Stream in real time.

The boundary between the pipeline and the harness is **the manifest** (Section 6). Everything the
harness knows arrives through that file.

The primary annotator is the project owner, working alone. Design for one fast human, not a team.

---

# 1. Transcript policy

Target output format — English in Latin script, Nepali in Devanagari:

```text
So today म Python मा loops बारे कुरा गर्छु।
```

The policy is not encoded as rules the system enforces. It is stored as a `policy_version` string
attached to every label so exports are interpretable later. Do not build a policy linter, a
guideline editor, or a guideline display panel.

---

# 2. Global engineering rules

## 2.1 Process

- **Test-driven development.** Write the failing test first, then the implementation. Every module
  with logic (importers, scoring, queue building, export, transliteration) gets tests before code.
- **Atomic commits.** One logical change per commit. Conventional Commits format
  (`feat:`, `fix:`, `test:`, `refactor:`, `chore:`, `docs:`). Every commit must leave the test suite
  green. Never mix a refactor with a behaviour change.
- **The owner will not be reviewing code line by line.** This raises, not lowers, the bar: tests,
  type hints, docstrings on public functions, and honest `PROGRESS.md` entries are the only
  quality signal that exists. Do not report a phase complete when its verification criteria fail.
- Small pull-request-sized chunks of work per session, each ending in a green suite.

## 2.2 Stack

- Postgres is the source of truth. Migrations for all schema changes, up and down.
- Python backend (FastAPI), typed, linted (ruff), formatted (ruff format).
- Frontend: a small single-page app. React + Vite is fine. No heavy component library.
- MinIO for audio object storage, behind a storage adapter interface with a local-filesystem
  implementation as the fallback. The harness must be usable without MinIO running.
- Docker Compose brings up Postgres, MinIO, backend, frontend.
- UTC timestamps everywhere.
- Configuration in YAML; secrets in environment variables; nothing secret committed.
- Idempotent, resumable jobs.
- Prefer boring technology.

## 2.3 Documentation

Maintain `README.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `PROGRESS.md`, `TODO.md`,
`VERIFICATION.md`. Update at the end of every session with what completed, what is blocked, what
verification passed and failed, and the next first task.

---

# 3. LLM routing rule

**All LLM inference from this codebase goes through OpenRouter.** No exceptions, no direct calls to
OpenAI, Anthropic, Google, Groq, or Mistral. The reason is billing control: OpenRouter is prepaid,
so there is no possibility of a surprise invoice.

**At MVP, the harness makes no LLM calls at all.** Multi-system disagreement and rule-based flags
already provide the prioritization signal. Do not build policy checking, hypothesis ranking, or
correction suggestion.

What to build now, and only this:

- `backend/app/llm/openrouter.py` — a thin client: API key from `OPENROUTER_API_KEY`, route-based
  model selection from config, per-route timeout and max tokens, retries with backoff, dry-run
  mode, and request logging to the `llm_requests` table.
- `config/llm_routes.yaml` with an empty `routes: {}` block and `enabled: false`.
- The `llm_requests` table in the schema, so no migration is needed later.

No route is wired to any pipeline stage. The client exists so that when a route is added later, the
billing, logging, and dry-run guarantees are already in place. Cover it with tests against a mocked
HTTP layer.

Note: upstream ASR (including any commercial transcription API) runs in the GPU pipeline, outside
this codebase, and is out of scope for this rule.

---

# 4. Core principles

1. **Human time is the binding constraint.** The dominant interaction is *accept and move on*.
   Optimize the median segment, not the hard one.
2. **Models produce hypotheses; the human label is truth.** Never overwrite a hypothesis with a
   correction.
3. **Preserve raw data.** Store imported hypotheses immutably. Labels are append-only rows.
4. **Everything reproducible.** Every export records label version, policy version, split, filters,
   row counts, checksums, and git commit.

---

# 5. Input contract: the manifest

The upstream pipeline emits a directory:

```text
export_<episode_id>/
  episode.json
  segments.jsonl
  clips/
    <segment_id>.flac
  peaks/                # optional; harness generates if absent
    <segment_id>.json
```

## episode.json

```json
{
  "episode_id": "show-a_ep012",
  "show_id": "show-a",
  "title": "Example podcast",
  "source_uri": "https://...",
  "published_at": "2026-01-01",
  "source_audio_checksum": "sha256:...",
  "duration_seconds": 4821.3,
  "pipeline_version": "nb-v3",
  "pipeline_commit": "a1b2c3d"
}
```

## segments.jsonl — one object per line

```json
{
  "segment_id": "show-a_ep012_0042",
  "episode_id": "show-a_ep012",
  "speaker_id": "SPEAKER_01",
  "start_time": 123.4,
  "end_time": 135.2,
  "clip_path": "clips/show-a_ep012_0042.flac",
  "clip_checksum": "sha256:...",
  "p_en": 0.31,
  "lid": "ne",
  "hypotheses": [
    {
      "system_id": "qwen-ne",
      "model_id": "sidskarki/Qwen3-ASR-Nepali",
      "text": "So today म Python मा loops बारे कुरा गर्छु।",
      "avg_logprob": -0.34,
      "no_speech_prob": 0.01,
      "words": [
        {"word": "So", "start": 123.4, "end": 123.7, "confidence": 0.92,
         "predicted_language": "en", "predicted_script": "latin"}
      ]
    }
  ],
  "scores": {
    "cer_between_hypotheses": 0.18,
    "word_disagreement_rate": 0.22,
    "script_conflict_rate": 0.05,
    "code_switch_density": 0.42
  },
  "flags": ["low_confidence"]
}
```

Rules:

- `hypotheses` must contain at least one entry; `words` is optional and may be absent or empty.
- `scores` may be partially absent; the harness recomputes nothing, it stores what it receives and
  treats missing scores as null.
- Clips are **16 kHz mono FLAC**. Reject WAV or MP3 clips at import with a clear error — the source
  is already lossy and re-encoding the exact audio you will train on is not acceptable. The original
  episode file is archived separately and is not needed by the harness.
- Import is idempotent, keyed on `(segment_id, system_id)` for hypotheses and `segment_id` for
  segments. Re-importing an unchanged export is a no-op. Re-importing with a changed clip checksum
  is an error unless `--allow-clip-change` is passed.

Write a JSON Schema for both files and validate at import. A malformed manifest must fail loudly
before any row is written.

---

# 6. Data model

Postgres. Concepts must exist; exact column names may evolve.

## 6.1 Provenance and content

```text
import_runs(
  id, source_path, pipeline_version, pipeline_commit,
  segments_inserted, segments_skipped, hypotheses_inserted,
  started_at, finished_at, status, notes
)

episodes(
  id, external_id, show_id, title, source_uri, published_at,
  source_audio_checksum, duration_seconds,
  split,                    -- 'train' | 'val' | 'test' | 'unassigned'
  split_seed, split_assigned_at,
  metadata_jsonb, created_at, updated_at
)

segments(
  id, episode_id, speaker_id,
  start_time, end_time, duration_seconds,
  clip_object_key, clip_checksum, peaks_object_key,
  p_en, lid,
  pipeline_status,          -- 'imported' | 'queued' | 'labeled' | 'excluded'
  import_run_id, created_at, updated_at
)

asr_systems(id, system_id, model_id, notes, created_at)

asr_hypotheses(
  id, segment_id, asr_system_id,
  text_raw, text_normalized,
  avg_logprob, no_speech_prob, metadata_jsonb, created_at
)

hypothesis_words(
  id, hypothesis_id, word_raw, start_time, end_time, confidence,
  predicted_language, predicted_script, created_at
)

segment_scores(
  segment_id PRIMARY KEY,
  cer_between_hypotheses, word_disagreement_rate,
  script_conflict_rate, code_switch_density,
  flags_jsonb, imported_at
)
```

## 6.2 Annotation

```text
annotation_tasks(
  id, segment_id UNIQUE WHERE status IN ('pending','in_progress'),
  queue,                    -- 'review' | 'audit' | 'error'
  priority_score,
  seed_hypothesis_id,       -- which hypothesis preloads the editor
  reason_jsonb,             -- why this segment scored where it did
  status,                   -- 'pending' | 'in_progress' | 'done' | 'skipped'
  created_at, updated_at
)

label_versions(id, name, description, policy_version, created_at)

segment_labels(
  id, segment_id, label_version_id,
  final_text,
  disposition,              -- see below
  seed_hypothesis_id,       -- what the human was shown, nullable
  annotator, notes,
  created_at
)

annotation_events(
  id, task_id, segment_id, annotator,
  opened_at, submitted_at, duration_ms,
  action,                   -- 'accept' | 'edit' | 'skip' | 'flag' | 'reopen'
  created_at
)

audit_logs(
  id, entity_type, entity_id, action, actor,
  old_values_jsonb, new_values_jsonb, created_at
)

translit_cache(
  latin_token PRIMARY KEY, candidates_jsonb, provider,
  hit_count, created_at, updated_at
)

llm_requests(
  id, route, model, request_hash, input_summary, output_json,
  prompt_tokens, completion_tokens, estimated_cost_usd,
  latency_ms, status, error_message, created_at
)
```

### disposition values

```text
accepted_unchanged   -- seed hypothesis was correct as-is
edited               -- human changed the text
unusable_audio       -- music, crosstalk, dead air, non-speech
uncertain            -- speech is real but the human could not resolve it
```

`unusable_audio` and `uncertain` are distinct and both matter: the first is an audio quality
statistic, the second is an annotation difficulty statistic, and they route differently on export.

### Status discipline

There are exactly three status fields and each has one owner:

- `segments.pipeline_status` — where the segment sits in the harness lifecycle.
- `annotation_tasks.status` — queue state.
- `segment_labels.disposition` — what the human decided.

Do not add a fourth. Do not add boolean flags that duplicate a status. `segment_labels` rows are
append-only; the latest row per `(segment_id, label_version_id)` is current.

## 6.3 Frozen splits

Split assignment happens **once, at episode import**, hashed from `(episode_id, split_seed)` and
written to `episodes.split`. It is never recomputed.

Why this matters: without a stored split, the train/test division is recalculated at every export.
Add ten episodes and segments silently migrate between train and test, which means a model can end
up trained on audio it was already benchmarked against, and two exports of "the same" dataset are
not the same dataset. Storing the assignment costs one column and one import step.

Splits are at **episode** level, never segment level — segments from one episode share speakers,
recording conditions, and topic, so a segment-level split leaks.

---

# 7. Phases

Do not start a phase until the previous phase's verification criteria pass.

---

## Phase 0: Scaffolding

**TODO**

- Repository layout (below).
- Docker Compose: Postgres, MinIO, backend, frontend.
- `.env.example`, config loading, structured logging.
- ruff lint + format, pytest, coverage reporting.
- Alembic migrations wired up.
- Documentation files.
- Pre-commit hook running lint and tests.

```text
/
  docker-compose.yml
  README.md  ARCHITECTURE.md  DECISIONS.md  PROGRESS.md  TODO.md  VERIFICATION.md
  backend/
    app/
      main.py  config.py
      db/  models/  api/  services/  storage/  llm/  translit/  utils/
    migrations/
    tests/
  frontend/
    package.json  src/
  scripts/
    import_manifest.py
    build_queue.py
    export_dataset.py
    report_status.py
  config/
    settings.yaml
    llm_routes.yaml
```

**Verification**

- `docker compose up` starts all four services.
- Backend connects to Postgres and MinIO.
- `alembic upgrade head` and `downgrade base` both succeed.
- `pytest` passes; `ruff check` passes.
- README explains startup in under ten commands.

---

## Phase 1: Schema and migrations

**TODO**

- Implement every table in Section 6.
- Constraints: foreign keys, the partial unique index on active tasks, enum or check constraints on
  status and disposition columns.
- Indexes for: queue ordering by `priority_score`, lookup by `segment_id`, export filtering by
  `episode.split` and `label_version_id`.
- Seed script producing a small synthetic dataset.
- Schema documentation in `ARCHITECTURE.md`.

**Verification**

- Migrations run from empty and roll back cleanly.
- Seed creates 1 episode, 20 segments, 3 ASR systems, hypotheses for each segment.
- A second active task for the same segment is rejected by the database, not by application code.
- Foreign keys reject orphan rows.
- Tests cover insert and query of every core entity.

---

## Phase 2: Manifest importer

**TODO**

- JSON Schema for `episode.json` and `segments.jsonl`; validate before writing.
- Verify clip format is 16 kHz mono FLAC; reject otherwise.
- Upload clips to object storage via the storage adapter; record object keys.
- **Precompute waveform peaks** at import (downsampled min/max arrays, ~1000 buckets per segment),
  store as JSON objects alongside clips. The UI must never decode audio client-side to draw a
  waveform — that alone will make the editor feel sluggish by the fortieth segment.
- Assign episode split from `(episode_id, split_seed)`.
- Idempotency, dry-run mode, an import report, and an `import_runs` row.

**Verification**

- Importing the same export twice inserts nothing the second time and reports it.
- A changed clip checksum errors unless explicitly overridden.
- A malformed manifest fails before any write; the database is unchanged.
- Non-FLAC or non-16 kHz clips are rejected with a clear message.
- Peaks JSON exists for every imported segment.
- Split assignment is deterministic for a given seed and stable across re-imports.
- Dry-run prints planned changes and writes nothing.

---

## Phase 3: Queue building

**TODO**

- Priority scoring from imported scores and flags only. No LLM.

```text
priority_score =
    0.40 * word_disagreement_rate
  + 0.25 * low_confidence          (normalized from avg_logprob)
  + 0.20 * code_switch_density
  + 0.15 * rule_flag_score
```

Normalize each input to 0–1. Store the component breakdown in `reason_jsonb` so the UI can show
*why* a segment surfaced.

- Rule-based flags computed at import: empty transcript, repeated n-gram (hallucination pattern),
  high `no_speech_prob`, segment shorter than 1 s or longer than 30 s, implausible speaking rate,
  script conflict between systems.
- **Seed hypothesis selection.** For each task, pick which hypothesis preloads the editor:
  - Segments in `train`/`val` episodes: seed with the highest-scoring hypothesis (fastest to accept).
  - Segments in `test` episodes: **rotate the seed system deterministically** by hashing
    `segment_id` across available systems, and record `seed_hypothesis_id`.

  The rotation costs the human nothing — it is the same one-key accept — but it means the test set
  is not anchored to a single system, and a per-seed WER breakdown can be reported later to show
  the anchoring effect is small. Without the recorded column that argument cannot be made at all.
- Audit queue: a seeded random sample (default 5%) of low-priority, high-agreement segments, so
  quality on the easy majority is measurable.
- Segments with zero hypotheses go to the `error` queue, never to `review`.
- Re-running the queue builder updates priorities without duplicating active tasks.

**Verification**

- Queue builder creates tasks and is idempotent.
- Top-priority segments are visibly the disagreeing ones on the seed data.
- Audit sampling is reproducible under a fixed seed.
- Test-episode seeds are distributed across systems, not concentrated on one.
- `reason_jsonb` explains every priority score.
- Priority formula documented in `ARCHITECTURE.md`.

---

## Phase 4: Review API

**TODO**

```text
GET  /health
GET  /stats                              -- progress counters
GET  /queue?limit=&episode=&min_priority=  -- triage list
GET  /tasks/next
GET  /tasks/{id}
GET  /segments/{id}
GET  /segments/{id}/audio                -- presigned URL or streamed with range support
GET  /segments/{id}/peaks
POST /tasks/{id}/accept                  -- disposition=accepted_unchanged
POST /tasks/{id}/label                   -- disposition=edited, body carries final_text
POST /tasks/{id}/flag                    -- unusable_audio | uncertain
POST /tasks/{id}/skip
POST /tasks/bulk-accept                  -- list of task ids
POST /translit                           -- Latin token -> Devanagari candidates
```

Requirements:

- Audio responses must support HTTP range requests so the player can seek without a full download.
- Every write creates a `segment_labels` row and an `annotation_events` row with real timing.
- Every write creates an `audit_logs` entry.
- No authentication in local dev; support an optional static bearer token via config.

**Verification**

- Automated API tests cover every endpoint including error paths.
- Accepting a task writes a label with `disposition='accepted_unchanged'` and non-null `duration_ms`.
- Bulk accept writes one label and one event per task, transactionally.
- Range requests return 206 with correct byte ranges.
- Timing on an event reflects real elapsed time between fetch and submit, not server processing time.

---

## Phase 5: Devanagari input helper

Build this **before** the editor UI. The annotator types in Latin and cannot type Devanagari
directly, so transliteration is not a convenience feature — it is the input method, and editing
throughput depends entirely on it.

**Design**

- `TranslitProvider` interface with `suggest(latin_token: str) -> list[str]` returning ranked
  Devanagari candidates.
- Two implementations:
  1. **Remote** — Google Input Tools transliteration endpoint
     (`inputtools.google.com/request` with `itc=ne-t-i0-und`), called from the **backend**, never
     the browser. It is undocumented and may change; isolate it behind the interface, set a short
     timeout, and degrade gracefully.
  2. **Offline fallback** — rule-based transliteration via `indic-transliteration` (Sanscript,
     ITRANS/HK schemes). Lower quality on casual romanization but has no network dependency.
- **Cache every lookup** in `translit_cache`, keyed on the lowercased Latin token. The same tokens
  recur constantly, so after a few hundred segments most lookups are local. The accumulated cache is
  also a by-product worth keeping: it is a romanization lexicon for this speaker community.
- Configurable provider order; cache is always consulted first.

**UI behaviour**

- Typing Latin inside the transcript editor and pressing space opens an inline candidate popup.
  Number keys `1`–`5` insert a candidate; `Enter` accepts the first; `Esc` keeps the Latin as typed
  (correct behaviour for genuine English words — the fallback must never be Devanagari).
- Selecting an existing Devanagari word and retyping in Latin replaces it through the same popup.
- A **correction memory**: if the same Latin token was previously resolved to a particular
  Devanagari form in this project, rank that first.
- A toggle to disable the helper entirely for English-heavy segments.

**Verification**

- Provider interface has a mock implementation used in tests.
- Cache hit returns without a network call; test asserts the HTTP layer is not touched.
- Remote provider failure falls back to offline without an error dialog.
- `Esc` leaves the Latin token untouched.
- Candidate popup is fully keyboard-operable; the mouse is never required.
- Manual check: type twenty common Nepali words in Latin and confirm the correct Devanagari form
  appears in the top three candidates.

---

## Phase 6: Review UI

Two modes, both keyboard-first. Mouse use should be optional everywhere.

### 6.1 Triage mode (the default, and the one that matters)

A dense list of queued segments, roughly 15 per screen, each row showing: seed hypothesis text,
duration, priority reason chips, and a play control.

- `j` / `k` — move between rows
- `Space` — play/pause the focused row
- `Enter` — accept as-is and advance
- `e` — open the focused segment in the editor
- `f` — flag unusable audio
- `u` — mark uncertain
- `Shift+Enter` — accept all selected rows

This is the workflow that matches "listen, it's fine, accept, move on." It must be possible to
process a screen of easy segments without ever leaving the list.

### 6.2 Editor mode

Opened only for segments needing correction.

- Waveform from precomputed peaks, with a playhead.
- Play/pause, loop-segment toggle, playback speed (0.75× / 1× / 1.25×), replay-last-2-seconds.
- Editable transcript textarea, preloaded with the seed hypothesis, with the transliteration helper
  active.
- All hypotheses listed below the editor with the system name; a button and number key to load any
  one into the editor.
- A visible diff between the seed hypothesis and the current text.
- Priority reason chips and rule flags.

Shortcuts (editor):

```text
Space            play/pause          (when the textarea is not focused)
Ctrl+Space       play/pause          (always)
Ctrl+Enter       save and next
Ctrl+Shift+Enter save and stay
Alt+1..5         load hypothesis N into editor
Alt+Left/Right   seek -2s / +2s
Ctrl+L           toggle loop
Esc              close popup, then unfocus editor, then exit to triage
```

Do not bind `Ctrl+S` — the browser owns it and the collision is a papercut every single time.
Every shortcut must behave predictably with the textarea focused; that is the default state.

### 6.3 Progress display

Always visible, not on a separate page:

- Segments done / total, for the current episode and overall.
- This session: count completed, elapsed, median seconds per segment.
- Projected time to finish the current queue at the current median rate.
- Accept rate — what fraction were accepted unchanged. This number is the health check on the
  upstream pipeline; if it collapses, the ASR stage regressed.
- Resume: reopening the app returns to the exact position in the queue.

**Verification**

- A full screen of segments can be triaged with keyboard only.
- Editor loads, plays, edits, and saves; the label appears in Postgres with correct disposition.
- Transliteration popup works inside the editor without breaking undo.
- Progress counters update live and survive a page reload.
- Manual test: 50 sample segments processed end to end; record the median seconds per segment in
  `VERIFICATION.md` — that number is the project's throughput baseline.

---

## Phase 7: Export

**TODO**

Four exports, each writing a manifest alongside the data.

1. **ASR training export** — train/val splits, approved labels only.
2. **Gold test export** — test split only, with `seed_system_id` retained per segment.
3. **Analytics export** — includes word-level fields where hypothesis words were imported.
4. **Error mining export** — `uncertain` and `unusable_audio` dispositions, for pipeline debugging.

Training record shape:

```json
{
  "segment_id": "show-a_ep012_0042",
  "episode_id": "show-a_ep012",
  "audio_path": "clips/show-a_ep012_0042.flac",
  "start_time": 123.4,
  "end_time": 135.2,
  "text": "So today म Python मा loops बारे कुरा गर्छु।",
  "disposition": "accepted_unchanged",
  "seed_system_id": "qwen-ne",
  "label_version": "v1",
  "policy_version": "policy_v1",
  "split": "train",
  "code_switch_density": 0.42
}
```

Export manifest must record: label version, policy version, filters applied, split, row count per
split, SHA-256 of each output file, export timestamp, git commit, and the set of `import_runs`
contributing rows.

Requirements:

- Deterministic: same inputs and filters produce byte-identical output.
- Test episodes never appear in the training export. Assert this, do not assume it.
- `unusable_audio` never appears in training or gold exports.
- JSONL always; Parquet if it is cheap to add.
- Retaining `disposition` and `seed_system_id` in the output is mandatory — they are what make the
  dataset defensible to a reviewer.

**Verification**

- Two consecutive exports produce identical checksums.
- A test-split segment appearing in a training export fails a test.
- Row counts in the manifest match the files.
- Export succeeds when word timestamps are entirely absent.
- Exported text matches the current label row for every segment.

---

## Phase 8: Status report

**TODO**

A single command producing a plain-text or HTML report:

- Episodes, total audio hours, segments, segments by pipeline status.
- Labels by disposition, and accept rate over time.
- Annotation throughput: segments/hour, median seconds/segment, total annotator hours.
- Queue backlog and projected completion.
- Score distributions: mean disagreement, script conflict rate, code-switch density.
- Split balance across train/val/test in hours.
- Word timestamp coverage.

**Verification**

- One command, no SQL knowledge required.
- Report includes throughput and projected completion.
- Report runs against an empty database without crashing.

---

# 8. Deferred — do not build now, keep the door open

The corpus is currently small, but the archive behind it is roughly 2,400 hours. Do not build any
of the following, and do not architect around them either — just avoid decisions that would make
them expensive later.

| Deferred | Keep the door open by |
|---|---|
| Job queue / worker pool | Keep import and queue-build as pure functions over a batch; no global state |
| Table partitioning | Keep `hypothesis_words` writes batched and its foreign keys clean |
| Multi-annotator, IAA, adjudication | `segment_labels` already carries `annotator`; keep rows append-only rather than updating in place |
| LLM-assisted checks | The OpenRouter client and `llm_requests` table already exist |
| Word-level language editing | `hypothesis_words` schema is retained; no UI |
| Annotation guidelines in-app | `policy_version` is stored on every label |
| Auth and user management | Optional bearer token hook already in the API layer |

If any of these becomes necessary, it should be a new phase with its own verification criteria, not
a retrofit.

---

# 9. Session protocol

At the start of each session: read `PROGRESS.md`, `TODO.md`, `VERIFICATION.md`; state the current
phase and what this session will do.

At the end: update all three; list what completed, what failed verification, what is blocked; write
the next session's first task. Commit atomically throughout, never in one lump at the end.

Do not begin a new phase until the current phase's verification passes or the owner explicitly
overrides.

---

# 10. Definition of done for the MVP

MVP is Phases 0 through 6 with:

1. An export directory from the GPU pipeline imports cleanly, idempotently, with peaks generated.
2. Episode splits are frozen at import.
3. A prioritized queue exists with explained priority scores.
4. Triage mode lets the annotator accept segments with one keystroke each.
5. Editor mode supports correction with working Latin-to-Devanagari input.
6. Progress and throughput are visible at all times and survive reload.
7. 200 segments have been annotated end to end and the median seconds per segment is recorded.
8. Test suite green, lint clean, documentation current.

Export and reporting (Phases 7–8) follow once the editor is proven not to fight the annotator.

---

# 11. First task

**Phase 0 only.** Repository layout, Docker Compose with Postgres and MinIO, backend skeleton,
config loading, Alembic wired, pytest and ruff configured, pre-commit hook, documentation files.

Write the tests for config loading and the database connection first.

Use mise if you need to install anything locally
