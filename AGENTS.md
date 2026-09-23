# AGENTS.md

Working agreement for coding agents in this repository. [README.md](README.md) explains what the
harness is and how to run it; read that first if you have not.

## Orientation

```text
backend/app/     FastAPI app: api/ db/ models/ services/ storage/ translit/ llm/ utils/ schemas/
backend/tests/   pytest suite; `db`-marked tests run against a real Postgres
backend/migrations/  Alembic revisions — the only way the schema changes
frontend/src/    Vite + React 19 + TypeScript; components/ui/ is vendored shadcn/ui
scripts/         thin CLI wrappers over services
config/          settings.yaml (non-secret), llm_routes.yaml (ASR and LLM routes)
docs/            architecture, decisions, roadmap, findings, sociolinguistics, custom-arch
notebooks/       EDA, sociolinguistics and fine-tuning (04c, Indic-Transcribe; generated from notebooks/src/)
```

Speaker diarization runs on a Modal GPU (`scripts/modal_diarize.py`, D79), not in the backend.

ASR work: what is next is in [docs/roadmap.md](docs/roadmap.md), what has been measured in
[docs/findings.md](docs/findings.md). Model-building designs parked until a tcpWER uplift are in
[docs/custom-arch.md](docs/custom-arch.md).

Deeper maps: [docs/architecture.md](docs/architecture.md) for the pipeline, schema, priority
formula and endpoint list; [docs/decisions.md](docs/decisions.md) for why things are the way they
are — read the relevant entry before arguing with a design.

## Invariants

Breaking one of these is a design change, not a refactor. Say so out loud before doing it.

1. **Postgres is the source of truth.** Every schema change is an Alembic revision with a working
   `downgrade`. Never `create_all`, never an out-of-band `ALTER`.
2. **Hypotheses are immutable; labels are append-only.** A correction never overwrites a
   hypothesis, and a re-label inserts a new `segment_labels` row. Current means newest per
   `(segment_id, label_version_id)`.
3. **Exactly three status fields**, each with one owner: `segments.pipeline_status`,
   `annotation_tasks.status`, `segment_labels.disposition`. Do not add a fourth, and do not add a
   boolean that duplicates one.
4. **Two pots, chosen per clip by hand** (D71). `segments.pot` is `gold` or `train`; a gold
   clip exports as `test` whatever its episode's `train`/`val` split is. Only
   `set_segment_pot` moves a clip, and every move writes an `audit_logs` row. No algorithm
   decides gold, and nothing may reintroduce one without a new decision entry. Train and val are
   per episode, drawn by hash at import, and may be redrawn freely.
5. **A label says how hard it was looked at.** `segment_labels.verification_tier` is `verified`
   (played and read) or `screened` (accepted on the disagreement signal without listening). It
   defaults to `verified` everywhere, so no caller can weaken the claim by omission. A screened
   decision on a gold clip is refused with 409, and so is moving a screened clip into gold.
6. **Every inference call is routed and logged.** Inference goes through a named route in
   `config/llm_routes.yaml` and a client in `app/llm/`, and writes an `llm_requests` row —
   whichever vendor served it, and whether it succeeded, failed or was a dry run. There is no
   longer a prepaid-only rule (D34): a provider is chosen for what it can transcribe, and spend is
   controlled by `dry_run`, `ingest.youtube.max_duration_seconds` and the `llm_requests` audit
   trail rather than by the shape of the vendor's billing.
7. **Clips are 16 kHz mono FLAC.** Anything else is rejected at import, before any row is written.
8. **Every decision writes three rows in one transaction**: a label, an `annotation_events` row
   with the client-reported elapsed time, and an `audit_logs` entry. The one exception is skip,
   which writes an event and an audit entry but no label (decisions D14).
9. **Configuration in YAML, secrets in the environment.** Nothing secret is committed, ever.
10. **UTC everywhere**, `timestamptz` in the database.

## How to work

- **Tests first.** Anything with logic — importers, scoring, queue building, export,
  transliteration, ingestion — gets a failing test before the implementation. The owner does not
  review line by line, so the suite is the only quality signal that exists.
- **Atomic commits**, Conventional Commits format (`feat:`, `fix:`, `test:`, `refactor:`, `chore:`,
  `docs:`). One logical change each; every commit passes lint and the not-db suite, and every push
  leaves the full suite green (D97). Never mix a refactor with
  a behaviour change.
- **Do not report something as done when its tests fail.** Say what failed.
- Type hints and docstrings on public functions. Boring technology wins.
- Import and queue building stay pure functions over a batch, with no global state — that is what
  keeps a job queue a wiring change rather than a rewrite.

### Commands

```bash
cd backend
.venv/bin/python -m pytest                     # full suite (1626 tests; needs Postgres)
.venv/bin/python -m pytest -m "not db"         # no Postgres
.venv/bin/python -m pytest tests/test_api.py -k accept
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/alembic revision -m "..." && .venv/bin/alembic upgrade head
cd ../frontend && npm run build                # tsc -b && vite build
```

`docker compose up -d postgres minio` before `db`-marked tests; the suite creates and migrates its
own `harness_test` database using the real migrations. `mise` pins Python 3.13 and Node 22; the
backend virtualenv is `backend/.venv`.

### Driving the app in a browser

A Playwright MCP server is configured in `.mcp.json` (server name: `playwright`), so
`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_console_messages`,
`browser_network_requests` and `browser_take_screenshot` are available for debugging UI against the
running app.

Prefer `browser_snapshot` (accessibility tree, cheap) over `browser_take_screenshot` for finding
and clicking elements; screenshot when the question is visual — layout, spacing, colour, overflow.

Bring the stack up first: the frontend calls the backend at `http://localhost:8000`, and with the
backend down every panel shows a "Failed to load …" toast. `docker compose up -d` for the full
stack, or `cd frontend && npm run dev` for pure layout work (expect API errors in the console).
Then navigate to `http://localhost:5173/`.

Notes: it runs headless against system Chromium (`/usr/bin/chromium`) — drop `--headless` from
`.mcp.json` to watch it. `@playwright/mcp` is pinned; unpinning risks the bundled `playwright-core`
wanting a browser build that is not installed. Snapshots and console logs land in
`.playwright-mcp/`, which is gitignored.

## Gotchas

- Ingestion spends real money: each `asr*` route transcribes every clip, and three are
  configured, so a clip costs three calls -- plus one fusion request per ~30 minutes of audio,
  which is billed mostly as thinking tokens. Use a short audio file, or `dry_run: true` in
  `config/llm_routes.yaml`, when exercising the pipeline; a dry run skips fusion entirely. The same arithmetic is why
  `ingest.youtube.max_duration_seconds` exists — cost is linear in source duration, so a YouTube
  URL is a bigger footgun than a file the annotator had to download first.
- A YouTube URL never reaches `yt-dlp` as typed. `app/services/youtube.py` parses out the
  eleven-character video id and rebuilds a canonical `watch?v=<id>` from it, which is the only form
  the subprocess sees (D23). Do not "improve" this into sanitizing the string: the rebuild is what
  keeps the harness from being a fetcher for arbitrary hosts and a leading `-` from becoming a
  flag. Its download occupies the upload's slot, not a sixth stage — the five stages and their
  numbering are unchanged. The editors' "Watch at" link follows the same rule: `youtube.video_url`
  rebuilds it from the id, so a stored string never reaches an `href`.
- OpenRouter's Batch API is text-only. A `:batch` model slug is rejected outright on the
  synchronous endpoint, and a batch carrying audio is accepted and *then* terminally fails
  validation — so a `:batch` transcriber fails an episode late rather than at startup. No ASR
  route may name one; `test_config.py` enforces it (D22).
- The **seed** is the fused transcript (D74), and it is not a measurement of anything: it was built
  to agree with the recognisers. Never score the seed *against* the recognisers -- that is what
  the D67 terms did, and they collapse to zero on a fused seed and read as quality. Compare the
  recognisers to each other, and the fused text to the recognisers and the waveform only through
  `app/services/hazards.py`. `low_confidence` reads Scribe, the only route with `avg_logprob`;
  CMI is measured on the fused text.
- **Every text comparison goes through `app/services/fold.py`.** A plain token comparison charges
  `टिम`/`team` and `गर्नुभयो`/`गर्नु भयो` as errors, which was about half of every system's
  substitutions on the gold pot. Report folded and raw WER side by side, with `fold_version()`,
  and split every WER into substitutions, deletions and insertions (per 100 folded reference
  words, summing to the WER): error mining reads all three, because a change can move errors
  between kinds without moving WER, and on crosstalk the labels charge hearing the other voice as
  insertions.
- Word spans have two sources and they must not be confused. Scribe, MAI and Gemini 3.5
  Transcribe *report* their own; Gemini Flash's are *measured* afterwards by the local CTC aligner
  in `app/services/forced_align.py`, which is what the `forced_align` flag on a route turns on.
  Never set that flag on a route that reports its own timings: overwriting them would destroy the
  independent references the D33 boundary report compares.
- **No route diarizes** (D52, reversing D49), so `hypothesis_words.speaker` is null for anything
  ingested now. Clip-local labels could not answer the question they were collected for: the
  pipeline segments before it transcribes, so a clip almost always holds one speaker. Do not turn
  the flag back on expecting speaker identity — `spk:0` in one hypothesis was never `spk:0` in
  another, and neither is `segments.speaker_id`. Speaker identity comes from a full-episode
  diarization on a remote GPU, stored in `speaker_turns` (D78, D79) and joined by time. Rows
  ingested before D52 still carry labels; do not join on them.
- Gemini runs on **Vertex AI**, not AI Studio, via `app/llm/vertex.py` (D39). Auth is one API key
  (`VERTEX_API_KEY`, restricted to `aiplatform.googleapis.com`) sent as an `x-goog-api-key`
  header, never a `?key=` query parameter — httpx puts URLs in its error strings and those are
  copied into `llm_requests.error_message`. No Application Default Credentials, gcloud token
  stores, or service-account files.
- **The recogniser's model id is `gemini-3.5-transcribe-preview` on Vertex and
  `gemini-3.5-transcribe` on AI Studio.** Asking Vertex for the bare id is a 404 in every region.
  That, not an allowlist, is what broke the first Vertex attempt. There is no Interactions API on
  Vertex: both routes are `:generateContent`, and the recogniser is configured through
  `generationConfig.audioTranscriptionConfig` (`diarization`, `wordTimestamp`, `languageCodes` —
  the other spellings are deprecated).
- **If another Vertex recogniser is ever configured, this is what was learned from the last one**
  (removed in D51; `app/llm/script_restore.py` was kept for it and then deleted in D81). It
  accepted no steering at
  all: a `systemInstruction` was a hard 400, a text part was accepted and ignored, and
  `customVocabulary` returned 200 and then silently suppressed `speakerLabel` entirely. Because
  nothing could be told to it, it **transliterated English into Devanagari** (`active` →
  `एक्टिभ`), which `restore_script` repaired afterwards on the token list — **one token in, one
  token out**, so spans survived and `forced_align` stayed false. Its `language_codes` had to be
  `[ne-NP]` alone: two or more codes made it answer HTTP 200 with no content for clips past ~15 s,
  against `MAX_SEG_SECONDS = 20.0`.
- **An empty 200 from Gemini is a failure, not a transcript**, and `_send_with_retries` cannot see
  it — `vertex.py` judges emptiness itself and retries. The exception is `audio_chat` returning
  empty with no `blockReason`: `ASR_PROMPT` asks for that when there is no intelligible speech.
- The raw Devanagari lives in the hypothesis's `metadata_jsonb` as `text_devanagari`. It is
  provenance: keep it out of `text_raw`, the disagreement comparison, the analysis and the queue.
- **A fused hypothesis is never a recogniser.** Its system has `asr_systems.kind = fusion`, and
  every disagreement computation filters on `kind = 'asr'` -- the fused text was built to agree
  with the recognisers, so counting it would make agreement meaningless. The fuser is also never
  shown an earlier fusion's output as an input. Its system id ends in the prompt version
  (`fusion-gemini-3.8-flash-p1`); change `PROMPT_VERSION` in `app/llm/fusion.py` whenever the
  instruction changes, or a re-fusion collides with the old hypotheses' unique key.
- No route is held out of the disagreement scores: D51 removed the only one that ever was, so
  `disagreement_excluded_system_ids()` returns an empty set. It is still the single source both
  computation sites -- `ingest.py` and `purge.py` -- read from. Naming a system in either place
  independently silently desynchronises the two, which is why a route that must not be scored is
  deleted rather than remembered.
- Every Vertex transcription request turns safety filtering **off** (`OFF`). This is not
  optional and not a shortcut. Google blocks on the prompt/audio by default and answers with no
  candidates — so the failure arrives as an empty hypothesis rather than an error and drags that
  clip's disagreement rate. A transcriber's job is to write down what was said.
- `backend/uv.lock` is committed and the image installs from it, hash-checked (D43). Add a
  dependency by editing `pyproject.toml` and running `uv lock`, then commit both -- a
  `pyproject.toml` change alone will not reach the container, and `uv export --frozen` fails
  loudly when the two disagree.
- **Brouhaha's ONNX (SNR and reverb, D87) never downloads itself**: the model is gated and
  OpenRAIL. `scripts/export_brouhaha_onnx.py` builds it once into `data/models/` from an account
  that accepted its terms. Without it, ingest measures bandwidth only and says so in the result's
  version; `scripts/backfill_acoustics.py` completes those clips once it is there.
- The aligner's ONNX model is gitignored (~317 MB) and **downloads itself when missing** (D42),
  pinned to a commit and digest-checked, so the container no longer needs the export.
  `scripts/export_aligner_onnx.py` remains for provenance; `torch` and `transformers` must never
  enter `backend/pyproject.toml`. A missing *or unfetchable* model is a warning and no word spans,
  never a failed episode -- same contract as `silero_vad.onnx`.
- The **voiceprint model** (WeSpeaker ResNet34-LM ONNX, 26 MB, D99) downloads itself the same way,
  on the first speakers-queue clip or voice confirmation; `HARNESS_VOICEPRINT_NO_DOWNLOAD=1` refuses
  it and the editor simply makes no suggestions. Its fbank is numpy (`voiceprint.fbank`), checked
  against torchaudio's by a stored fixture, because torch may not enter the backend. A voiceprint
  follows the louder voice in crosstalk: never use it to attribute overlapped words.
- The fetch only writes to the **default** path. Passing `model_path`/`vocab_path` explicitly
  disables it, which is what keeps fixtures and the degradation test from pulling 317 MB.
  `HARNESS_ALIGNER_MODEL_DIR` moves it (the container uses `/app/data/models`, inside the bind
  mount, so it survives `up`); `HARNESS_ALIGNER_NO_DOWNLOAD=1` refuses it.
- The transcribe stage commits per segment on purpose (D20). Do not "tidy" it into one transaction.
- **A paid result is checkpointed only after its `llm_requests` row is committed** (D93). Move a
  new paid call into ingest the same way — through `Checkpoint` in `app/services/ingest/`,
  keyed by everything that was sent — or a power cut throws it away. Interrupted jobs resume at
  startup; `ingest.resume_interrupted: false` stops that.
- `/tasks/next` marks the task `in_progress` — that is what makes resume work. A partial unique
  index enforces one active task per segment, so a second one raises `IntegrityError` from the
  database, not from application code.
- `annotation_events.duration_ms` is client-reported. Server processing time is not the quantity of
  interest, and substituting it would make the throughput baseline meaningless.
- Audio is streamed with HTTP range support, never a presigned redirect — one code path for both
  storage backends. `ObjectStorage.read_range` exists for exactly this.
- Peaks are precomputed at import and never computed in the browser.
- Anything reaching HTML output (episode titles, transcripts) comes from a user, YouTube or a
  model. Escape it.


## Documentation

Eight documents, and no others: this file, `README.md`, and the six under `docs/`. When behaviour
changes, update the document it contradicts in the same commit — a stale `docs/architecture.md` is
worse than none. Record a real design choice in `docs/decisions.md` as a new numbered entry with
its reversal cost. When a later entry overrides an old one, delete the old entry and add one line
for it to the file's "Retired entries" list, so code comments citing it still resolve; trim an
entry that is only partly superseded down to what still holds.

Do not reintroduce per-session progress, TODO or verification files. Git history is the session
log, and the test suite is the verification record.
