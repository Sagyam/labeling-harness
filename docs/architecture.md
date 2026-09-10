# Architecture

## System boundary

![Ingestion pipeline: upload or YouTube URL, normalize, VAD segment, transcribe with every cloud ASR route, analyse, import and build the queue](diagrams/ingest-pipeline.svg)

The harness provides an end-to-end web workflow: the annotator selects a podcast file directly in
the browser or pastes a YouTube URL, watches real-time progress and logs as it normalizes, segments,
and queries Cloud ASR, and immediately begins annotation in the Review UI without touching the CLI
or fragile Colab notebooks.

## Module map

```text
backend/app/
  config.py        YAML + env configuration, frozen pydantic models
  main.py          FastAPI application factory
  api/             HTTP routers (health, ingest, queue, tasks, segments, translit, episodes)
  db/              engine, session scope, declarative base
  models/          SQLAlchemy ORM models -- one module per concept group
  schemas/         JSON Schema for the manifest (episode.schema.json, segment.schema.json)
  services/        ingest pipeline (audio, silero_vad, analysis, youtube), importer, peaks,
                   scoring, queue builder, labeling, corpus, export, reporting
  storage/         ObjectStorage interface + local filesystem and MinIO implementations
  translit/        Latin -> Devanagari providers and the cache
  llm/             base (retry, dry-run, request log), openrouter, elevenlabs, and the
                   transcription dispatcher every ASR call in the pipeline goes through
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

![Core data model: episodes own segments; each segment carries ASR hypotheses, one scores row, queue tasks and append-only labels](diagrams/data-model.svg)

Postgres is the source of truth. All timestamps are `timestamptz` in UTC.

### Provenance and content

| Table | Purpose |
|---|---|
| `import_runs` | One row per import invocation, with counts and status |
| `episodes` | Episode metadata plus the **frozen** train/val/test split |
| `segments` | Time span, clip and peaks object keys, `pipeline_status`, VAD speech spans |
| `asr_systems` | One row per upstream ASR system |
| `asr_hypotheses` | Immutable imported transcripts, one per (segment, system) |
| `hypothesis_words` | Optional word-level timings, languages and scripts; times are **clip-relative** (D26) |
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
| `llm_requests` | Provider request log; every ASR attempt during ingestion is recorded here, whichever vendor served it |

### Status discipline

![Annotation task lifecycle: pending to in_progress to done, with skip returning the task to pending and every decision writing one of four dispositions](diagrams/annotation-lifecycle.svg)

Exactly three status fields, each with one owner:

- `segments.pipeline_status` — `imported | queued | labeled | excluded`
- `annotation_tasks.status` — `pending | in_progress | done | skipped`
- `segment_labels.disposition` — `accepted_unchanged | edited | unusable_audio | uncertain`

No fourth status, and no boolean that duplicates one. `segment_labels` rows are append-only.

### Two pots

`segments.pot` is `gold` or `train` (D71). The owner puts individual clips in gold by hand —
`POST /segments/{id}/pot`, from the triage star, `g`, or the editor button — and can take them back
out; `set_segment_pot` (`app/services/pots.py`) is the only writer, refuses to move a clip with a
screened label into gold, and writes an `audit_logs` row per move. A gold clip exports as the
`test` split whatever its episode is (`effective_split`).

`episodes.split` is `train`, `val` or `unassigned`, drawn at import from a BLAKE2b hash of the
episode id against `dataset.val_fraction`, and subdivides only the clips that are not gold.

Per-clip selection puts clips of one episode on both sides of the train/test line. `pot_status`
counts the gold episodes that also feed train and the gold clips they hold, and every export row
carries `episode_spans_pots`, so a gold result can be reported with and without them.

### Verification tier

`segment_labels.verification_tier` is `verified` (clip played, transcript read) or `screened`
(accepted on cross-ASR disagreement without listening). It defaults to `verified` at every layer, so
a caller that omits it cannot weaken the corpus's claim about itself. Screening a gold segment is
refused with 409 at `record_decision`, and the `gold` export refuses to write if a screened row
reaches it anyway. Every export row carries the tier and each manifest reports the mix per split.

## Priority formula

![Priority score composition](diagrams/priority-scoring.svg)

> The figure above predates D74 and shows the D54 formula; the text below is current.

```text
priority_score =
    0.30 * unsupported_rate    (fused words no recogniser heard, by sound)
  + 0.20 * dropped_rate        (words two recognisers heard that the fused text lacks)
  + 0.20 * asr_disagreement    (script-folded disagreement among the recognisers themselves)
  + 0.15 * acoustic_gap        (the fused text against the aligner's own reading of the clip)
  + 0.10 * low_confidence      (Scribe's avg_logprob)
  + 0.05 * rule_flag_score

priority = priority_score + 1  when any hazard gate fired
```

Every component is in 0–1 and the weights sum to 1 (validated at load, `queue.weights`), so the
score is in 0–1 and a gated clip's priority is in 1–2: it sorts above every ungated clip, and
screening it is refused with 409. The weights are provisional (D74).

The seed is the fused transcript, which was built to agree with the recognisers, so the question
is not "how far is the seed from them" -- that is near zero by construction -- but "where does
it say something the evidence does not support, or omit something it does", plus "how hard is
this audio". `app/services/hazards.py` answers both, comparing everything through the
script-folding normalizer (`app/services/fold.py`), so a respelling like `टिम`/`team` is never
counted as a disagreement.

- `unsupported_rate` — share of fused words that no recogniser has at that position, even as a
  near-miss spelling (romanized similarity ≥ 0.6).
- `dropped_rate` — the longest run of words two recognisers both have and the fused text lacks,
  against the recognisers' median length.
- `asr_disagreement` — mean pairwise folded word error among the recognisers. Still an independent
  measurement: no recogniser sees another's output.
- `acoustic_gap` — from the aligner's `AcousticFit`: the mean, over speech frames, of how much less
  likely the fused text makes each frame than the model's own best reading, scaled by
  `acoustic_gap_full_scale`. On hard audio both are low and the gap stays small, which is what the
  plain forced-path posterior could not tell apart from hallucination. Uncalibrated: it ranks and
  gates nothing. Unmeasured (no aligner) is recorded under `unmeasured`, not scored as a fit.
- `low_confidence` — `clamp(avg_logprob / logprob_floor, 0, 1)` over Scribe, the only recogniser
  reporting one. It describes Scribe's text, not the fused text shown; it is a difficulty prior.
- `rule_flag_score` — fraction of rule flags raised for the segment (see below).

### Hazard gates

Any gate makes a clip unscreenable and lifts it above every clip without one. They are about the
*shape* of a failure, because a three-word invention in a forty-word clip barely moves a rate:

| Gate | Fires when |
|---|---|
| `invention` | ≥ `invention_run` (3) consecutive fused words no recogniser heard |
| `dropped` | ≥ `dropped_run` (3) consecutive words that two recognisers share and fusion lacks |
| `seam_bleed` | unheard fused words that the neighbouring clip's recognisers did hear |
| `length_outlier` | fused length outside 1/1.6–1.6× the recognisers' median, and ≥ 8 characters off |
| `emptied` / `speech_over_silence` | empty over heard speech / text where nothing was heard |
| `unaligned` | the aligner cannot fit the fused text into the clip |
| `fuser_uncertain` | the fuser coded the clip `u` |
| `unfused` | no fused text; the seed fell back to a recogniser |

The words behind each gate travel in `reason_jsonb.hazard_details` and show in the triage tooltip.
D67's formula is still computed, against the recogniser the old queue would have seeded with, and
recorded under `reason_jsonb.legacy`; it ranks nothing.

### Rule flags (computed at import)

`empty_transcript`, `repeated_ngram` (hallucination pattern), `high_no_speech_prob`,
`too_short` (< 1 s), `too_long` (> 30 s), `implausible_speaking_rate`, `script_conflict`,
`missed_speech`.

These eight are the whole vocabulary and the denominator of `rule_flag_score`, so a flag name from
outside the list is stored on the segment but contributes nothing to the score.

`missed_speech` fires when more than 25% of a clip's VAD-detected speech has no word from *any*
system over it (D55). It is the one flag that does not read a transcript: `segments.vad_spans_jsonb`
is the only timing in the schema that a transcriber did not produce, so it is what separates "no
system wrote anything here" from "there was nothing to write". It cannot fire on a segment with no
stored spans, or on one where no system reported word timings. The importer
computes them itself, over every hypothesis of the segment, and unions the result with whatever
`flags` the manifest carried: `flags_jsonb = sorted(received | computed)`. That is the one place
the harness does not simply store what it receives.

### Seed hypothesis selection

Every clip, gold included, is seeded with its fused transcript (D74); when a re-fusion has added
a newer fusion system, the newest wins. A clip the fuser did not answer falls back to the
recogniser with the highest `avg_logprob` and is gated `unfused`. Gold's old per-system seed
rotation is gone: the owner chose the faster seed over a benchmark independent of the fuser, and
D74 records what that costs.

Segments with zero hypotheses go to the `error` queue, never to `review`. An audit queue takes a
seeded random sample (default 5%) of the low-priority half of the *ungated* clips, so quality on
the easy majority stays measurable.

## Ingestion and Cloud ASR

Ingestion runs inside the app, not in an upstream notebook: the annotator uploads an episode in the
browser -- or pastes a YouTube URL and lets the server fetch it (below) -- and watches it become a
queue. `POST /ingest` starts a background job and returns a job id; the six stages are:

1. **Normalize** — FFmpeg two-pass `loudnorm` to 16 kHz mono FLAC with linear normalization,
   avoiding dynamic AGC gain pumping between words. The downsample runs through libsoxr, whose
   stopband is steep enough that content above 8 kHz is discarded rather than folded back into the
   clip as alias (D39).
2. **Segment** — Silero VAD (ONNX, CPU) cuts on speech turns padded by 150 ms, bounded to 2.0 s–20.0 s,
   snapping long-turn subdivisions to low-energy pauses with a 15 ms raised-cosine edge fade so
   slices do not click.
3. **Transcribe** — every route named `asr*` in `config/llm_routes.yaml` transcribes every clip,
   producing one ASR system per route, in the order the routes are written. Transcribers for a
   segment run concurrently via a worker pool with a shared `httpx.Client` for HTTP connection
   pooling, while up to `max_segment_concurrency` segments are processed in parallel. Each attempt is
   logged to `llm_requests`, whichever provider served it. `app/llm/transcription.py` dispatches
   on the route's `provider` and `api`; see the table below.
4. **Fuse** — a reasoning model reads every recogniser's text for ~30 minutes of consecutive
   clips at a time and writes one verbatim transcript per clip (D72, `app/llm/fusion.py`,
   `app/services/fusion_stage.py`). It never hears the audio; the CTC aligner places the fused text
   back on each clip for its word spans. The result is one more hypothesis per clip, under an
   `asr_systems.kind = fusion` system -- the seed, and never a disagreement signal. Skipped on a
   dry run; a clip the fuser did not answer keeps its recognisers and falls back to one of them.
5. **Analyse** — Devanagari/Latin ratio, code-mixing index, cross-system word disagreement, script
   conflict and the rule flags below. Disagreement is the mean over every *pair* of recognisers,
   never the fused text; code-mixing is measured on the fused text where there is one, because it
   is the only transcript that follows the script policy.
6. **Import and build** — segments, hypotheses, scores and queue tasks are written in one pass, so
   "Start Annotating" works the moment the job finishes.

### Fusion windows

A window is about `fusion.window_target_seconds` (1800) of speech, and windows are balanced: a
45-minute episode is two of 22.5, never 30 + 15, because thinking cost scales with what the model
reads rather than with what it writes. Most sources are 5-20 minute videos and are one window,
with no seams at all. Around its targets a window carries `lookahead_seconds` of the following
clips' raw hypotheses and `carryover_seconds` of the fuser's own output for the clips before.
Recognisers are shown as anonymous `A`/`B`/`C`, in route order.

The contract is one JSON object per target id. A missing, duplicate or invented id, or unparseable
output, earns one retry; a truncated answer (`finishReason` other than `STOP`) is halved at once;
halving stops at `max_depth`, and what still fails is left unfused rather than guessed. A 429 is
waited out in minutes. Each request is an `llm_requests` row; the fused hypothesis's
`metadata_jsonb.fusion` carries its window, the model's code (`k`/`s`/`m`/`c`/`u`), the prompt
version and the model version.

The route bounds thinking (`thinking_budget: 24576`) because thoughts and answer share
`max_tokens`: a 30-minute window reads ~190 segments and answers in ~11k tokens, and the pilot's
dynamic budget spent 20-34k thought tokens per ~100 segments read.

### Configured transcribers

| Route | Provider | API shape | Returns | Steered by |
|---|---|---|---|---|
| `asr_scribe_v2` | ElevenLabs (direct) | `/v1/speech-to-text` | text, word spans, per-word logprob, speaker per word | `language_code: ne` — nothing else (D48) |
| `asr_mai_transcribe_2` | OpenRouter | `/audio/transcriptions` | text, word spans | `language: ne` — nothing else (D48) |
| `asr_gemini_flash` | Vertex AI (direct) | `POST …/gemini-3.8-flash:generateContent` | text only | the full policy prompt as `systemInstruction`, `language: ne` |

The fused hypothesis is what stage 5 measures the Devanagari/Latin ratio and the code-mixing
index on; the first route stands in for a clip the fuser did not answer. Rule flags are computed
at import over *all* hypotheses, fused included.

Scribe is first because it is the only configured transcriber reporting per-word log probabilities —
so it remains the source of the `low_confidence` term. Reordering the routes moves the CMI
measurement to a different model, and it also moves `low_confidence`, because a hypothesis with
no `avg_logprob` never wins the train/val "highest confidence" comparison.

Only `asr_gemini_flash` is actually steerable. It is the one route whose model reads the clip as a
chat model and takes the policy as a `systemInstruction`; the other two are dedicated recognisers,
and neither can be told anything in prose (D48).

Ingestion routes each clip across all three systems, producing a three-way disagreement signal for
queue prioritisation. Each hears only the audio -- no recogniser is ever shown another's
transcript, which is what keeps their disagreement an independent measurement rather than a
correlated one. The fuser is the only thing that reads all three, and it is not a recogniser.
Three systems is three paid calls per clip, plus one fusion request per ~30 minutes; the count is
the routing table's. It was four until D51 removed the composite, and Scribe cost a fifth call
per clip for script restoration until D73.

The three do not return the same thing. Scribe reports word spans, per-word log probabilities and
a speaker label per word (D49); MAI reports text and word spans; Gemini 3.8 Flash reports text
alone, and its word spans are measured afterwards by the local CTC forced aligner (D31, D32).
Flash is the one general-purpose model in the set -- it obeys the policy prompt, and it may
equally editorialise or hallucinate over silence, which is the price of that opinion.

No route diarizes (D52). `hypothesis_words.speaker` is null for everything ingested since, and
the column is retained for a future full-episode diarization stage rather than for any ASR route
to fill. Labels collected before D52 are clip-local -- `spk_1` in one hypothesis is not `spk_1` in
another, and neither is a `segments.speaker_id` from an upstream manifest -- so nothing should
join on them. The measurement behind the reversal is in D51 and D52: on a two-speaker episode, 51
of 68 clips were single-speaker for both diarizing systems, because the pipeline segments before
it transcribes.

### Fetching the audio instead of uploading it

A job may name a YouTube URL rather than carry a file. The download occupies the **same slot an
upload does** -- it is how the source file arrives, not a seventh stage -- so it reports under a
`downloading` stage that precedes stage 1 and leaves the six stages untouched.
`app/services/youtube.py` shells out to `yt-dlp`, and two rules shape it:

- **Nothing the annotator typed reaches the subprocess.** A URL is parsed down to its
  eleven-character video id and a canonical `https://www.youtube.com/watch?v=<id>` is rebuilt from
  that id alone. So the harness cannot be turned into a fetcher for arbitrary hosts, a URL
  beginning with `-` cannot become a yt-dlp flag, and a link copied from inside a playlist ingests
  the one video rather than the list.
- **The video is inspected before any bytes move.** `POST /ingest/youtube` looks the video up
  first, so a private, live or over-long video is a 422 on that request instead of a job that
  fails a minute later. `ingest.youtube.max_duration_seconds` (4 h by default) is a spend guard,
  not a technical limit: every `asr*` route transcribes every clip, so cost is linear in source
  duration.

`POST /ingest/youtube/probe` runs the same lookup on its own, downloading nothing and creating no
job, so the browser can prefill the title and slug and show what it is about to ingest. The
downloaded file keeps whichever container YouTube served -- stage 1 re-encodes it anyway, so
nothing transcodes twice -- and the canonical URL is stored as the episode's `source_uri`.

`GET /ingest/{id}` reports stage, progress and error state; `GET /ingest/{id}/events` streams the
same log lines the backend writes, over SSE, into a terminal panel in the browser. Clips are
committed per segment rather than in one transaction around the whole stage, so a job that fails
halfway leaves the work it already did.

The manifest importer (below) remains the other, equal-status way in: an upstream GPU pipeline can
still produce `export_<episode_id>/` and `scripts/import_manifest.py` will ingest it.

### Known gaps

Recorded here rather than left to be rediscovered. Neither is load-bearing today, and both are
behaviour changes, so neither is fixed in passing.

- **Stage 4 writes three values the importer never reads.** `ingest.py` nests `cmi`, `avg_logprob`
  and `flags` inside the segment record's `scores` object, but the importer reads `flags` from the
  record's *top level* (as the manifest contract specifies) and `SegmentScore` has no column for
  the other two. The flags survive anyway — the importer recomputes the same rules over the same
  hypotheses — and `avg_logprob` reaches the queue through the seed hypothesis, so the practical
  loss is CMI, which is only ever displayed in the ingest log.
- **The forced aligner's model file is not in the repository.** It is ~300 MB, against
  `silero_vad.onnx`'s 2.3 MB, so it is gitignored and built once by
  `scripts/export_aligner_onnx.py`. Without it, ingestion logs a line and Gemini's hypotheses
  carry no word spans; nothing else changes, and the boundary report simply finds no comparison
  source.
- **Skipping a task audit-logs the wrong old value.** `labeling.skip_task` hard-codes
  `old_values_jsonb={"status": "pending"}`, but any task opened through `/tasks/next` is already
  `in_progress` by then (decision D16). The new value and the action are correct; only the
  recorded prior state is wrong.

## Export

Four export kinds, each writing `manifest.json` next to the data:

1. **training** — `train` + `val` splits, approved labels only.
2. **gold** — `test` split only, retaining `seed_system_id` per segment.
3. **analytics** — includes word-level fields where hypothesis words were imported, episode metadata (speaker role and gender, and the episode topic — never a name or a dialect, D56), and automatically generates `timestamp_verification_report.json`. That report compares two independent timing sources — Scribe's own word spans against the forced aligner's spans over Gemini's transcript — on the tokens both agree were said, reporting agreement tolerances (<= 25 ms, <= 50 ms, <= 100 ms) and flagging divergence (> 200 ms) for human review (D33).
4. **error_mining** — `uncertain` and `unusable_audio` dispositions, for pipeline debugging.

The manifest records label version, policy version, filters, split row counts, SHA-256 of each
output file, timestamp, git commit and the contributing `import_runs`. Exports are deterministic:
the same inputs and filters produce byte-identical output.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process health plus Postgres and object storage reachability |
| `GET /stats` | Progress counters, disposition mix, accept rate, throughput, projected finish |
| `GET /queue` | Triage list; `limit`, `offset`, `episode`, `min_priority`, `queue` |
| `GET /tasks/next` | Highest-priority pending task; marks it `in_progress` so reopening resumes |
| `GET /tasks/{id}` | One task with its full segment payload; does not change status |
| `GET /segments/{id}` | Segment with all hypotheses, scores, flags and current label |
| `GET /segments/{id}/audio` | Clip stream with HTTP range support (206) |
| `GET /segments/{id}/peaks` | Precomputed waveform peaks JSON |
| `POST /tasks/{id}/accept` | `disposition=accepted_unchanged`; `verification_tier` says whether it was heard |
| `POST /tasks/{id}/label` | `disposition=edited`, body carries `final_text` |
| `POST /tasks/{id}/flag` | `unusable_audio` or `uncertain` |
| `POST /tasks/{id}/skip` | Defer; event only, no label |
| `POST /tasks/bulk-accept` | Accept many tasks in one transaction |
| `POST /translit` | Latin token → ranked Devanagari candidates |
| `POST /translit/choice` | Record the chosen form for the correction memory |
| `POST /ingest` | Upload an episode's audio; starts the pipeline, returns a job id |
| `GET /ingest` | Inspect queue status: running job, upcoming queue, bot backlog, and past jobs |
| `POST /ingest/youtube` | Ingest from a YouTube URL; the server fetches the audio itself |
| `POST /ingest/youtube/batch` | Queue multiple YouTube URLs for batch ingestion |
| `POST /ingest/youtube/probe` | Read a video's metadata; downloads nothing and creates no job |
| `POST /ingest/{id}/retry` | Requeue a failed, aborted, or backlogged ingestion job |
| `POST /ingest/retry-all` | Batch retry all jobs matching a status filter (e.g. `backlog`, `failed`) |
| `DELETE /ingest/{id}` | Cancel a queued/running job or remove a finished/backlog job |
| `POST /ingest/clear-past` | Prune finished history jobs from the manager |
| `GET /ingest/{id}` | Job stage, progress, active segment count, error state |
| `GET /ingest/{id}/events` | SSE stream of the job's log lines |
| `GET /episodes` | Episode list with per-episode segment counts and progress |
| `GET /episodes/{id}/segments` | Segments of one episode with flags, transcripts and audio URLs |
| `DELETE /episodes/{id}` | Delete an episode, its child rows and its clips and peaks |
| `DELETE /segments/{id}` | Delete one segment and its stored objects |
| `GET /stats/report` | Pipeline status: pots, coverage, verification mix, agreement, accept-rate trend |
| `GET /stats/inventory` | Corpus inventory: hours by every recorded dimension, gaps, and ranked sourcing recommendations (D69) |
| `GET /pots` | What each pot holds and what the gold pot does not cover; moves nothing |
| `POST /segments/{id}/pot` | Put one clip in gold or take it out (D71); 409 for a screened clip |
| `POST /export` | Export dataset profiles (`training`, `gold`, `analytics`, `error_mining`) |
| `GET /export/download/{kind}/{filename}` | Download exported dataset JSONL or manifest |
| `GET /export/history` | List previous exported dataset artifacts on disk |
| `GET /costs` | Aggregate AI inference cost report across ElevenLabs, OpenRouter, and Google |
| `GET /costs/requests` | Filterable, paginated audit ledger of all external AI requests and incurred spend |

Every decision writes three rows in one transaction: an append-only `segment_labels` row, an
`annotation_events` row carrying the client-reported elapsed time, and an `audit_logs` entry.
Authentication is off when `api.auth_token` is empty; setting it requires `Authorization: Bearer`.
Deletions are audited like any other write; `/health` is the only unauthenticated route.

## Transliteration

`TranslitProvider.suggest(latin_token) -> list[str]` has three implementations: the remote Google
Input Tools endpoint (called from the backend, short timeout, degrades to nothing on failure), an
offline rule-based provider built on `indic-transliteration`, and a static provider for tests.
`TransliterationService` consults `translit_cache` first, so a recurring token never leaves
Postgres, and `record_choice` promotes a previously chosen form to the front of the candidate list —
the correction memory. The accumulated cache is a romanization lexicon for this speaker community.

## Review UI

`frontend/` is a Vite + React 19 single-page app in TypeScript, styled with Tailwind v4 and
shadcn/ui components vendored into `src/components/ui/` (Radix primitives plus local styling —
copied in, not a component-library dependency). `App.tsx` holds the whole session: which queue is
active, triage or editor mode, the focused row, the multi-select set and the open task.

| Piece | File | Role |
|---|---|---|
| Triage | `components/TriageView.tsx` | Dense keyboard-first list over `/queue`; one keystroke per decision |
| Editor | `components/EditorView.tsx` | Waveform, playback, transcript editing, hypothesis switching, live diff |
| Waveform | `components/Waveform.tsx` | Draws the precomputed peaks; click to seek, playhead follows audio |
| Karaoke line | `components/KaraokeTranscript.tsx` | Lights and grows the spoken word from `hypothesis_words` spans; click a word to play it |
| Dispute panel | `components/DisputePopover.tsx` | What the other systems heard at a disputed moment; plays it, and swaps the word in one click |
| Transliteration | `components/TranslitEditor.tsx` | Inline Latin → Devanagari candidate popup over `/translit` |
| Ingest | `components/IngestModal.tsx` | Upload, 5-stage stepper, progress bar, live SSE log console |
| Episodes | `components/EpisodeManagerModal.tsx` | Browse episodes and segments, delete either |
| Progress | `components/Header.tsx` | Polls `/stats`: completed, accept rate, throughput, projected finish |

Audio is never decoded in the browser to draw a waveform (D8), and clips are streamed from
`/segments/{id}/audio` with range requests rather than fetched whole.
