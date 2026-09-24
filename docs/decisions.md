# Decisions

Each entry: the decision, why, and what it would cost to reverse.

## Retired entries

Entries a later decision overrode, or that only recorded a removal, are deleted. Code comments
still cite some of them, so each keeps one line here; the full text is in git history before the
commit that added this list. D59–D61 were never used.

- **D5** Frozen episode-level splits drawn by hash. Replaced by D63, D71 and D90: train/val is
  per episode and redrawable, gold is chosen per clip.
- **D10** All inference through OpenRouter, because it is prepaid. Replaced by D34.
- **D21** The provider rule is "prepaid", and ElevenLabs Scribe is called directly. The prepaid
  rule went in D34. What survives: retries, dry run and the `llm_requests` write live in
  `app/llm/base.py`, so every provider inherits them.
- **D24** Whisper large-v3 dropped: poor on code-switched Nepali, no word spans or confidence.
- **D28** Microsoft MAI-Transcribe 2 added, on OpenRouter's `/audio/transcriptions`.
- **D29, D30** Gemini 3.5 Transcribe on AI Studio's Live API, and VAD macro-windowing to survive
  its 100 requests/day quota (about four hours of audio a day). Replaced by D31.
- **D35** Gemini on Vertex AI through Application Default Credentials. Replaced by D39.
- **D36** Gemini 3.5 Transcribe as a fourth system, with word-level diarization; the origin of
  `hypothesis_words.speaker`. The route went in D51, diarization in D52.
- **D38** Gemini back on AI Studio under an API key. Replaced by D39 (quota is per key there).
- **D40** Gemini 3.5 Transcribe held out of disagreement for writing English in Devanagari.
  Replaced by D41, then D51.
- **D41** The Gemini composite: the recogniser heard, a text model rewrote each token's script,
  one token in and one token out, with the raw Devanagari kept as `text_devanagari`. Its two
  Gemini failure modes (one language code only; an empty 200 is a failure) are in AGENTS.md.
  Removed in D51.
- **D44** The script-restore route ran with thinking off: thinking truncated its JSON answers.
  The route is gone (D73 below, D81).
- **D49** Scribe asked to diarize, for a second source of speaker labels. Reversed by D52.
- **D50** The composite held out again, for omitting English speech. Replaced by D51.
- **D54** Priority on `seed_outvoted` (time-aligned disagreement with the seed);
  `code_switch_density` dropped from ranking because it anti-correlated with edits (ρ −0.77).
  Replaced by D67, then D74.
- **D65** Script restoration on the seed route, and screened episodes kept out of gold. Replaced
  by D73 and D71; the screened-never-gold rule stands in D63.
- **D67** Queue ranked on `seed_orphan_rate` and `roman_gap`. Replaced by D74: every term measured
  the seed against the recognisers, and a fused seed is built from them.
- **D68** One episode demoted out of gold by a one-off script. Replaced by D71.
- **D73** Script restoration taken off the seed route: the fused seed writes the script policy
  itself (D74), and a restored Scribe would have been a second, dependent vote in fusion.
- **D80** Batch URL submission removed; see D75.
- **D81** `app/llm/script_restore.py` deleted: no route used it after D73.
- **D82** The recorded D67 `legacy` score deleted from `reason_jsonb`: structurally zero on a
  fused seed.
- **D92** The playground starts with the stack; see D85.

## D1 — Postgres is the source of truth; migrations are the only schema change mechanism
Every schema change ships as an Alembic revision with a working `downgrade`. The test suite builds
its schema by running the real migrations rather than `create_all`, so a migration that works only
on paper fails the suite. **Reversal cost:** none; this is the floor.

## D2 — Sync SQLAlchemy, not async
One annotator, one browser tab. Async buys concurrency this workload does not have and costs
debuggability. FastAPI runs sync endpoints in a threadpool. **Reversal:** mechanical but broad;
would touch every service signature.

## D3 — Configuration in YAML, secrets in environment variables only
`config/settings.yaml` is committed with empty strings where secrets go. Environment variables
override YAML (`HARNESS_<SECTION>__<KEY>`), which required reordering pydantic-settings sources so
env wins over the init values carrying the YAML. **Reversal:** trivial.

## D4 — Storage behind an adapter, local filesystem as the default
The harness must be usable with MinIO stopped, so `ObjectStorage` has a local implementation and
`storage.backend` defaults to `local`. `read_range` is part of the interface rather than an
S3-specific extra, because HTTP range support in the audio endpoint depends on it. **Reversal:**
none needed; both implementations are kept.

## D6 — Hypotheses are immutable; labels are append-only
A correction never overwrites a hypothesis, and a re-label never updates a `segment_labels` row; the
latest row per `(segment_id, label_version_id)` is current. This keeps the door open for
multi-annotator agreement work without a schema migration. **Reversal:** would lose annotation
history.

## D7 — Exactly three status fields
`segments.pipeline_status`, `annotation_tasks.status`, `segment_labels.disposition`. No fourth, and
no boolean that duplicates one. Status sprawl is how these schemas rot. **Reversal:** n/a.

## D8 — Peaks are precomputed at import, never in the browser
The UI must not decode audio client-side to draw a waveform; that is what makes an editor feel
sluggish by the fortieth segment. Import writes a downsampled min/max array (default 1000 buckets)
as a JSON object next to the clip. **Reversal:** cheap, but the latency is the point.

## D9 — Clips must be 16 kHz mono FLAC; anything else is rejected at import
The source is already lossy and re-encoding the exact audio that will be trained on is not
acceptable. Rejection happens during validation, before any row is written. **Reversal:** would
require re-importing every episode.

## D11 — Validation by JSON Schema at the manifest boundary
`backend/app/schemas/episode.schema.json` and `segment.schema.json` are the executable form of the input
contract, checked before any write, so a malformed manifest fails loudly with an empty database
rather than half-importing. **Reversal:** none.

## D12 — Plain git hooks instead of the pre-commit framework
`.githooks/` holds the hooks, installed with `git config core.hooksPath .githooks` and run with the
backend virtualenv. One less dependency and one less lockfile for a single-developer project. What
each hook runs is D97. **Reversal:** trivial.

## D13 — Test isolation by transaction rollback, against a real Postgres
The suite needs real partial unique indexes and real foreign keys — application-level checks would
not prove the constraints exist. Each test runs inside a transaction rolled back afterwards, with
`join_transaction_mode="create_savepoint"` so service code can still call `commit()`. **Reversal:**
would weaken the schema guarantees the suite proves.

## D14 — A skip writes an event, but no label
The specification says every write creates a `segment_labels` row. A skip is the exception: the
annotator deferred the segment without judging the transcript, so inventing a label row for it would
corrupt every disposition statistic and every export filter that reads `disposition`. A skip
therefore writes an `annotation_events` row (`action='skip'`) and an `audit_logs` entry, sets
`annotation_tasks.status='skipped'`, and leaves `segments.pipeline_status` untouched so the next
queue build brings the segment back. **Reversal:** trivial, but it would make accept rate and
disposition counts meaningless.

## D15 — Audio is streamed with range support, never a presigned redirect
The specification allows either. Streaming is one code path that works identically for the local
filesystem and MinIO backends, keeps clip URLs stable and same-origin, and avoids leaking a
long-lived object URL. `ObjectStorage.read_range` exists precisely so a 206 costs one ranged read
rather than a full download. **Reversal:** adding a redirect later is additive; the endpoint stays.

## D16 — `/tasks/next` marks the task `in_progress`
That is what makes resume work: reopening the app returns the same task rather than a fresh one, so
the annotator lands exactly where they left off. The partial unique index guarantees there is only
ever one active task per segment, so this cannot fan out. **Reversal:** trivial.

## D17 — Elapsed time is reported by the client
`annotation_events.duration_ms` is computed from the `opened_at` the client sends, not from server
processing time, because the quantity of interest is how long the human took. An explicit
`duration_ms` in the request wins over `opened_at`, so a client that measures precisely can say so.
**Reversal:** would make the throughput baseline meaningless.

## D18 — Podcast ingestion and Cloud ASR integrated into Web UI

> **The manifest importer as a way in is superseded by D86**: `scripts/import_manifest.py` is
> deleted, and an episode enters only through the web ingest. The importer service stays as the
> ingest's last stage.

The external, fragile Colab GPU notebook is replaced with an in-app ingestion flow managed
entirely from the Web UI. The annotator uploads or selects a podcast audio file (.mp3, .m4a, .wav)
directly in the browser. The backend normalizes loudness and segments speech at natural pauses via
lightweight local VAD, routes speech recognition to Cloud ASR endpoints (including OpenRouter),
computes multi-system agreement and rule flags, and auto-populates the review queue. Progress,
system metrics, and debug logs stream live to the Web UI.
**Why:** Eliminates fragile Colab environments, GPU memory limits, and CLI friction. The annotator
never leaves the browser to ingest new episodes. The manifest importer stays, so an upstream GPU
pipeline remains a supported way in. **Reversal:** the importer path is untouched; removing the web
flow would cost only the UI.

## D19 — shadcn/ui components vendored into the repo, not a component-library dependency
The rule was "no heavy component library". shadcn/ui satisfies it in the letter that matters: the
components in `frontend/src/components/ui/` are source files in this repository, built on Radix
primitives, editable in place, with no upgrade treadmill and nothing to theme around. The
alternative was hand-rolling twenty accessible primitives (dialog, popover, tooltip, scroll area)
for a keyboard-first UI where focus management is the whole game. **Reversal:** the files are ours;
deleting the ones we do not use costs nothing.

## D20 — Ingestion writes per segment, not in one transaction
The transcribe stage makes one network call per route per segment. Wrapping the stage in a single
transaction would hold a pooled connection open for the length of an episode, and a failure at
segment 300 would discard 299 segments of paid ASR. Each segment commits as it lands.
When concurrent model dispatch and segment processing were introduced to accelerate long episodes,
the commit-per-segment invariant was preserved via `LockedSession`, ensuring that concurrent worker
threads safely commit each completed segment independently without holding long-lived global transactions.
**Reversal:** trivial, but it would make a long job all-or-nothing.

## D22 — Transcribers run on synchronous endpoints; OpenRouter's Batch API cannot carry audio
Batch pricing is half the synchronous rate, so a `:batch` slug is the obvious thing to reach for
on a corpus this size. It does not work, and it fails in the most expensive possible way:

- On `/v1/chat/completions`, a `:batch` slug is rejected with `404 "This model is only available
  through the Batch API."`
- Submitted to `/api/beta/batches` with an `input_audio` part, the batch is accepted as
  `202 validating` and *then* terminally fails: `"Batch does not support this content; remove
  audio, video, file, or other non-text content parts."`

Both were verified against the live API. The second is the dangerous one — the failure is
asynchronous, whole-batch rather than per-request, and permanent, so a retry loop that treats a
non-terminal status as "keep waiting" would burn a 24-hour completion window per attempt and
surface the problem an episode late.

The harness therefore runs every transcriber synchronously and pays the full rate, and
`config/llm_routes.yaml` names plain model slugs. A test asserts no `asr*` route ends in `:batch`,
so the constraint fails at configuration time rather than mid-ingest. **Reversal:** if OpenRouter
allows audio in batch, an `api: batch` shape would need submit/poll plus retry logic that never
resubmits a terminal batch — which is why this is written down rather than half-built.

## D23 — YouTube audio is fetched server-side, from a canonical URL rebuilt out of the video id
Every episode used to arrive as an upload, which meant the annotator downloading audio by hand
before the harness could see it. `POST /ingest/youtube` moves that step inside the app: the server
runs `yt-dlp` and the file lands in the job's work directory, where an upload would have.

Three choices are worth recording:

- **The download is not a sixth stage.** It occupies the slot an upload occupies — how the source
  file arrives — and reports under a `downloading` stage ahead of stage 1. The five stages, their
  numbering and their log lines are untouched, so nothing downstream had to learn about URLs.
- **Nothing the caller typed reaches the subprocess.** The URL is parsed down to its
  eleven-character video id and a canonical `watch?v=<id>` is rebuilt from that id alone. Sanitizing
  the string instead would leave the harness one bug away from being a general-purpose fetcher for
  arbitrary hosts (an SSRF), and a URL beginning with `-` one quoting mistake away from being a
  yt-dlp flag. Dropping playlist and timestamp parameters is a free side effect.
- **The video is inspected before bytes move.** The endpoint probes first, so a private, live or
  over-long video is a 422 rather than a job that fails a minute later — and `POST
  /ingest/youtube/probe` exposes the same lookup so the browser can prefill the form. The 4-hour
  ceiling is a spend guard, not a technical limit: every `asr*` route transcribes every clip, so
  cost is linear in source duration and a mistyped link to a livestream recording is expensive.

yt-dlp is a runtime dependency of the backend and a subprocess rather than a library import, for
the same reason FFmpeg is: it is a tool with a command line, its failures are exit codes and
stderr, and its progress is lines on stdout. **Reversal:** deleting the module and the two
endpoints leaves the upload path exactly as it was; nothing downstream and no table depends on it,
beyond `episodes.source_uri` carrying a URL instead of a `file://` name for episodes ingested this
way.

## D25 — Two-pass linear loudnorm, raised-cosine edge fade, and speech padding in VAD
Audio clips ingested through the pipeline occasionally exhibited audible clicking artifacts.
Acoustic analysis identified four contributing causes:
1. Neural VAD probability onset latency meant cuts occurred without speech padding, truncating initial
   consonants or room tone directly at high amplitude.
2. A 5 ms linear edge fade has discontinuous first derivatives at its boundaries and is shorter than a
   single pitch period (6.25 ms at 160 Hz), causing high-frequency spectral splatter heard as a click.
3. Subdivisions of speech turns longer than 20 s were sliced blindly on a fixed grid regardless of speech content.
4. Single-pass `loudnorm` acted as an adaptive gain controller that pumped gain by up to +18 dB during speech
   pauses and clamped down on subsequent word attacks.

The fix introduces:
- 150 ms pre- and post-speech padding (`speech_pad_ms`) in VAD turn detection so cuts land in silence.
- A 15 ms raised-cosine (half-Hann) ramp with zero boundary derivatives in `apply_edge_fade`.
- Energy-aware pause snapping in `segment_audio_to_slices` for turns exceeding 20 s.
- Two-pass EBU R128 `loudnorm` with `linear=true` in `normalize_audio` to preserve natural dynamics without AGC pumping.
- Audio element pause/reset guards in the frontend to avoid decoder pops during looping and task navigation.

**Reversal:** The parameters (`speech_pad_ms`, `FADE_MS`, two-pass filter) are encapsulated in
`app/services/silero_vad.py` and `app/services/ingest.py`. Reverting them restores the previous single-pass
and linear-fade behavior without schema changes.

### Anti-aliased downsampling with libsoxr (was a second D39)

The four boundary fixes above hold — measured on shipped clips, every cut lands in silence, the first and
last samples are zero, and the second pass applies a constant 1.5101x gain with no pumping
whatsoever (106 dB against a no-loudnorm control). The clicking nevertheless remained, because it
was never at the boundaries: it was spread through the body of every clip.

The cause was the last filter in stage 1, `aresample=16000`. FFmpeg's built-in resampler defaults to
a short filter with the cutoff at 0.97 of Nyquist, and its stopband is far too shallow for a 3:1
decimation. Measured with pure tones through the real pipeline, it passes 8.2 kHz at -15 dB and
8.5 kHz at -26 dB. Nothing above 8 kHz can survive a move to 16 kHz — it folds back, mirrored, on
top of the audio that is kept. The source episode carries -24 dB of its energy in the 8-10 kHz
band, which is exactly where sibilants live, so every /s/ and /ʃ/ deposited a burst of
near-Nyquist noise into the 7-8 kHz band: about 1.2 audible ticks per second, and roughly an
eighth of all energy in the clips' top octave.

The fix routes the resample through libsoxr:
`aresample=resampler=soxr:precision=28:cutoff=0.95:osr=16000`. It rejects the same tones at
-158 dB. End to end through `normalize_audio`, out-of-band content that previously folded down at
-35 dB now lands at -125 dB; on the real episode, mean alias falls from -48 dB to -139 dB. The
speech band is untouched (96 dB agreement below 6 kHz) and the passband stays flat to 7.6 kHz.

`_resample_filter` probes for libsoxr once per process by running the filter on a tenth of a second
of silence, and falls back to a lengthened built-in filter (`filter_size=256:cutoff=0.91`, which
measures -55 dB) with a loud warning rather than degrading in silence. Debian's `ffmpeg`, which the
backend image installs, is built `--enable-libsoxr`, so the fallback should stay unused.

**Clips ingested before this change carry the alias baked in and must be re-ingested to benefit.**

**Reversal:** `SOXR_RESAMPLE` and `SWR_RESAMPLE_FALLBACK` in `app/services/ingest.py` are the whole
change. `test_normalization_discards_content_above_nyquist_instead_of_folding_it` guards it by
comparing an out-of-band probe against an anchor-only control; on the old chain it fails with a
50 dB excess.

## D26 — Word timings are clip-relative; the segment's own span is episode-relative
`hypothesis_words.start_time` and `end_time` count seconds from the start of the **clip**, while
`segments.start_time` and `end_time` count seconds from the start of the **episode**. The two
timelines are deliberate, not an oversight, and the code has always written them this way — it was
`docs/manifest-contract.md` that documented the wrong one, showing a word starting at the same
`123.4` as the segment that contains it.

Three reasons the clip is the right origin:

- **The word list travels beside a clip, not beside an episode.** Every export record carries
  `audio_path: "clips/<segment>.flac"` and the word spans in the same object
  (`app/services/export.py`). Episode-relative times there are a silent trap: a reader opens a
  six-second FLAC, seeks to 123.4 s and gets nothing, with no error to explain it.
- **Clip-relative is measured; episode-relative is derived.** Scribe is handed `seg.clip_path` and
  reports offsets into that file. Expressing them on the episode timeline means adding
  `segment.start_time` — a VAD decision (`app/services/silero_vad.py`), shifted again by D25's
  150 ms speech padding and by the cut landing on a frame boundary rather than the exact float. The
  stored value should be the one that was observed, with the error-carrying sum left to whoever
  wants it.
- **A row stays readable on its own.** Interpreting a `hypothesis_words` row never requires joining
  up through `asr_hypotheses` to `segments` for an offset, and no derived value goes stale if an
  episode is deleted and re-ingested under different VAD settings.

`start_time + word.start` is the conversion when an episode timeline is wanted. The harness rebases
nothing on import, so a manifest supplying episode-relative word times is stored as written and is
wrong; the contract now says so.

**Reversal:** rebasing to episode-relative means adding `segment.start_time` in
`app/services/importer.py` where words are inserted, plus a data migration over existing
`hypothesis_words` rows. Cheap while the table is small, and there is no consumer to break today —
no API endpoint exposes word times, and only the `analytics` export emits them.

## D27 — Word-level acoustic boundary cross-verification runs automatically on analytics export
Rather than requiring annotators to manually adjust word boundary sliders in the browser (which slows
annotation down by 10x and violates the "no word-level editing UI" guidance in AGENTS.md), word-level
acoustic boundary alignment is evaluated algorithmically on export (`app/services/alignment.py`).

Whenever `POST /export` runs for `kind="analytics"`, the export service automatically executes dynamic
sequence alignment between the human-verified/gold transcript tokens and the acoustic model's word spans
(`hypothesis_words`). It calculates tolerance agreement rates (<= 25ms, <= 50ms, <= 100ms) and emits
`timestamp_verification_report.json` directly into the export directory. Segments where boundary
divergence exceeds 200ms are isolated in the report as an audit triage queue.

**Reversal:** if word-level UI editing ever becomes necessary, it can read from `hypothesis_words` directly;
the automated export report has zero runtime database dependency and runs entirely in memory over the export
batch.

## D31 — Gemini 3.8 Flash transcribes from the audio alone
`asr_gemini_flash` is an `api: audio_chat` route: a general model asked to transcribe, which
`config.py` documents as liable to editorialise or hallucinate over silence. Naming the shape
keeps that risk in configuration rather than buried in a client. It is not asked for word
timestamps; its spans come from the forced aligner (D32). It reports no `avg_logprob`. It
replaced a Live API recogniser whose quota covered about four hours of audio a day (D29, D30), and
it is served from Vertex (D39).

It is given **only the audio** — never the other hypotheses. Feeding it Scribe's and MAI's
transcripts to reconcile would collapse three independent opinions into one correlated output,
and every recogniser-to-recogniser comparison depends on them staying independent. It also caps a
hallucination at one hypothesis of three. Reconciling is fusion's job (D72), which runs afterwards
and is kept out of disagreement by `asr_systems.kind`.

**Reversal:** hand the route the other transcripts, and lose its independence.

## D32 — Word timestamps from a local CTC forced aligner, not from the model
`app/services/forced_align.py` places a known transcript back onto its own clip and reports where
each word starts and ends. Routes opt in with `forced_align: true`; today only `asr_gemini_flash`
does.

**Why:** forced alignment never chooses words. Given audio and the exact text that was said, it
finds the most likely monotonic placement of that text onto the waveform. That decouples *which
words were said* from *when they were said*, and the two questions have different best answers: a
model may be excellent at the first and unable to report the second. Once they are separate, a
transcriber returning no timestamps is no longer disqualified, and D33's boundary check gains a
timing source that is not another cloud vendor.

The acoustic model is a romanizing multilingual CTC head (`facebook/mms-300m-1130-forced-aligner`
is the reference). Every script folds into one Latin label set before alignment, which is the only
arrangement that handles Devanagari and Latin *inside a single utterance*. A monolingual English
head cannot place the majority of this corpus's tokens — the last report counted 2219 Devanagari
against 1223 Latin.

Runtime is `onnxruntime`, already a dependency, with the trellis and backtrace in numpy. **No torch
enters the service.** The pattern is `silero_vad.py` verbatim: one CPU-pinned session, and a
warning rather than an exception when the model is absent, so a missing optional artefact costs
word spans instead of failing an episode. Spans are clip-relative by construction, satisfying D26 —
the aligner only ever sees the clip.

Two known soft spots, both cheap to revisit:

- **The model file is ~300 MB** and cannot be committed the way the 2.3 MB `silero_vad.onnx` was.
  It is gitignored and built once by `scripts/export_aligner_onnx.py`, which needs `torch` and
  `transformers` in a throwaway venv.
- **Romanization is approximate.** MMS-FA was trained against `uroman` output; the first
  implementation uses `indic-transliteration`'s Harvard-Kyoto scheme, already a dependency, folded
  to the label alphabet. The measurement that decides whether that is good enough is D33's report:
  a large gap between `rate_within_50ms_devanagari` and `rate_within_50ms_latin` means the
  romanization is the problem, and the fix is to add the pure-Python `uroman` package behind the
  same one-function seam.

**Reversal:** set `forced_align: false` on the route. Hypotheses keep their text and lose their
spans; nothing else reads them.

## D33 — The word boundary report compares two sources, because it previously compared one
`verify_segment_timestamps` took a gold text and one word list, and computed its delta as
`abs(span.start - spans[idx].start)` — where `span` was itself an element of `spans`. Both sides of
every comparison were the same object; `TokenBoundaryDiff` set `ref_start == comp_start`. Whenever
the alignment was one-to-one the delta was exactly 0.0, and a non-zero value meant only that an
insertion had shifted an index.

`reports/timestamp_verification_report.json` carries the signature of it: `within_25ms_count` and
`within_50ms_count` are both `2620`, identical, because the deltas were bimodal — exactly 0 or
large, never in between. Its headline `rate_within_25ms: 76.12` was the share of tokens that
aligned at the same index, not the share of boundaries that agreed within 25 ms. **That file
predates this fix and its numbers mean nothing.**

The root cause was structural rather than a slip: only Scribe returned word spans, so there was no
second source in the corpus to compare against, and the function was written as though one list
could verify itself. The aligner (D32) supplies the missing one.

The function now takes two word lists, matches them with the existing `align_tokens_dynamic`, and
scores only pairs matched on both sides — two systems that disagree about *which word was said*
have no meaningful boundary delta between them. Tokens on one side only are counted as coverage,
not as agreement. `run_cross_verification_on_records` selects its two sides by `system_id` rather
than taking whichever hypothesis happened to carry words first, and the summary now records both
ids so a report can never again be read as a system compared against itself.

**Reversal:** none worth having. The previous behaviour was a defect, not a design.

## D34 — Drop the prepaid-only provider rule; spend is controlled by the switches, not the vendor
Invariant 5 required every inference provider to be prepaid — a balance the owner topped up,
never an invoice — and D10, D21 and D29 each argued a provider in on those terms. The rule is
removed. What remains of the invariant is the part that was doing the work: every call goes
through a named route and a client in `app/llm/`, and writes an `llm_requests` row.

**Why:** the rule stopped selecting for anything. It never bounded spend to a useful number — a
balance large enough to ingest a real episode is a balance large enough to be wasted — and the
things that actually cap a runaway ingest are unrelated to how the vendor bills: `dry_run` in
`config/llm_routes.yaml`, `ingest.youtube.max_duration_seconds`, the fixed set of `asr*` routes,
and the fact that a job transcribes a finite list of clips and then stops. Meanwhile the rule
excluded Vertex AI, which is postpaid, and which is where Google now serves its models (D35). A
provider chosen for its billing page rather than for what it can transcribe is the wrong trade on
a corpus whose whole purpose is measuring transcription quality.

The mitigation is that spend stays visible rather than capped: `llm_requests` records every
attempt, with route, model, status and latency, and it is the only spend record the harness has.
A GCP budget alert is the owner's job and lives outside this repository.

**Reversal:** restore the sentence to `AGENTS.md` and drop every postpaid route. Nothing in the
code enforced the rule — no test asserted it and no client checked it — so the reversal is
documentation plus a routing table, which is exactly why it was worth so little.

## D37 — Unified multi-vendor AI cost tracking and dashboard
A centralized pricing calculation engine (`app/llm/cost.py`) and analytics service
(`app/services/costs.py`) compute, record, and aggregate inference spend across ElevenLabs,
OpenRouter, and Google Cloud Vertex AI into `llm_requests`, backed by `GET /costs` and
`GET /costs/requests` and an interactive UI dashboard.

**Why:** External provider dashboards are delayed, fragmented across 3 separate vendor consoles,
and prior code wrote `estimated_cost_usd=None` for Vertex AI and ElevenLabs calls. D37 ensures
every request computes exact unit costs (audio duration or token consumption), logs them
atomically, and offers real-time auditability in the UI.

**Reversal:** None. Costs remain append-only in `llm_requests`.

## D39 — Gemini runs on Vertex AI under one restricted API key
Both Google models move back to Vertex AI. `app/llm/google.py` and `provider: google` are deleted;
`app/llm/vertex.py` and `provider: vertex` replace them. `asr_gemini_transcribe` calls
**`gemini-3.5-transcribe-preview`** and `asr_gemini_flash` calls `gemini-3.8-flash`, both at
`projects/{project}/locations/global/publishers/google/models/{model}:generateContent` on
`aiplatform.googleapis.com`. The `system_id` of each route is unchanged, so hypotheses recorded
before and after stay comparable.

**Why:** D38 put the models on AI Studio, whose quota is per key and cannot be raised, and stage 3
began failing with `HTTP 429 ... You exceeded your current quota` — the same tax D29 and D30 were
built around and D35 was written to escape. Vertex quota is project-scoped.

**What actually broke the first Vertex attempt (D35/D36).** Not an allowlist, and not the project.
The two surfaces name the model differently: Vertex serves `gemini-3.5-transcribe-preview` and
returns 404 for the bare `gemini-3.5-transcribe` that the old client asked for, in every region.
Probed directly: `preview` is `PUBLIC_PREVIEW` on both `global` and `us-central1` and absent from
`europe-west4`; `gemini-3.8-flash` is `GA` on the same two. D38's other correction still stands —
`interactions:create` was never a valid endpoint — but there is no Interactions API on Vertex at
all, so the fix is `:generateContent` with `generationConfig.audioTranscriptionConfig`
(`diarization`, `wordTimestamp`, `languageCodes`; every other spelling is deprecated).

**Auth is one API key, not ADC.** D35 assumed Vertex required Application Default Credentials.
It does not: a Google Cloud API key restricted to `aiplatform.googleapis.com` authenticates the
project-scoped endpoint, and responses come back `trafficType: ON_DEMAND` — paid project traffic,
not a free tier. So `google-auth`, the service-account JSON and the docker credential mount all
stay deleted, and the harness keeps the single-key shape it has for OpenRouter and ElevenLabs. The
key travels as an `x-goog-api-key` header rather than `?key=`, because httpx puts request URLs in
its error strings and `_send_with_retries` copies those into `llm_requests.error_message`. Least
privilege is the key's own restriction plus `roles/aiplatform.expressUser` on the service account
behind it — the key is refused by `generativelanguage.googleapis.com` with a 403.

**The recogniser accepts no steering, and that has a cost worth stating.** `systemInstruction` is
a hard `400 The input system_instruction is not supported.`; a text part in `contents` is accepted
and silently ignored; and `customVocabulary` is accepted with a 200 and then suppresses
`speakerLabel` entirely — measured three runs each way on one clip, labelled `spk:0` without it and
unlabelled with it. AI Studio at least answered 400 for that combination, so the constraint now
lives in the client, which drops the field and warns. Diarization is why the route exists (D36);
trading it for term biasing would be the expensive way to buy nothing.

The consequence is that `SCRIPT_POLICY` cannot reach this model by any route, and it therefore
**transliterates English into Devanagari** — `active` → `एक्टिभ`, `range` → `रेन्ज`, measured on a
real clip. That is a property of the model, not a bug, and it means this system will always score
badly on script fidelity. It is kept because it is the only transcriber reporting speaker labels,
and D36's split stands: Flash takes a `systemInstruction`, honours the no-transliteration rule and
may editorialise; the recogniser gets the script wrong and gets the spans right. Dropping the route
remains one line in `config/llm_routes.yaml` if that trade stops being worth four paid calls a clip.

**Speaker labels are per segment.** Vertex returns one `Part` per speaker turn carrying
`speakerLabel` (`spk:0`), not a label per word as the Interactions API did, so the client fans each
segment's label onto its own words to keep `hypothesis_words.speaker` a per-word column.

**Reversal:** returning to AI Studio is `provider`, the base URL, the model id and the payload
builder — and it re-enters the quota trap that caused this. Moving to ADC would mean restoring
`google-auth` and a credential mount for no gain now that a key authenticates the same endpoint.

## D42 — The aligner model downloads itself, pinned and digest-checked
`ForcedAligner` fetches `mms_fa.onnx` and its vocabulary when they are missing, instead of warning
and skipping word spans. The artefact is `onnx/model_int8.onnx` from
`onnx-community/mms-300m-1130-forced-aligner-ONNX`, pinned to commit `2100fb24` and verified
against a SHA-256 digest. `app/services/aligner_model.py` owns it.

**Why:** D32 keeps `torch` and `transformers` out of `pyproject.toml`, so the only way to obtain
the model was `scripts/export_aligner_onnx.py` in a throwaway venv — impossible inside the runtime
container, which is exactly where the file kept being absent. The community export is the same
graph the script produces: one `input_values` input of `[batch, samples]`, one `logits` output of
`[batch, frames, 31]`, and the same 31-label romanized vocabulary. Verified against a real clip
before adopting it. The export script stays for provenance and for rebuilding from the weights.

**Pinned to a commit, not a branch, and checked against a digest.** This is an executable graph
that runs over every clip in the corpus. A branch could move under a benchmark that already ran,
and a truncated transfer that still loads would put quietly wrong spans on every segment — worse
than no aligner. The download lands on a temporary file and is renamed only after its digest
matches, so a failed transfer leaves nothing that could be mistaken for a model next run.

**It fetches only into the default location.** A caller who names `model_path` is pointing at a
file they manage — a fixture, their own export, a mount — and downloading 317 MB over that choice
would be the wrong kind of helpful. `HARNESS_ALIGNER_NO_DOWNLOAD=1` refuses the fetch entirely,
for an air-gapped or bandwidth-constrained run.

**Where it lands.** `HARNESS_ALIGNER_MODEL_DIR` moves the directory; the container sets it to
`/app/data/models`, inside the existing bind mount. Keeping it out of the image means the image
stays small and the model is fetched once rather than on every `up`. The file is chmod 644 after
download because the container writes as root into a mount the host user has to be able to remove.

**A failed download is still not an error.** D32's contract is unchanged: no model means no word
spans, never a failed episode. Unreachable network, disabled fetch and digest mismatch all
degrade the same way an absent file always did.

**Reversal:** delete the module and the `_load` call; the export script alone gets you the file
back.

## D43 — The dependency lockfile is committed, and the image installs from it
`backend/uv.lock` is tracked. The container no longer runs `pip install -e .` against
`pyproject.toml`; it exports the locked set with `uv export --frozen --no-dev` and installs that
with `--require-hashes`, then installs the app itself with `--no-deps`.

**Why:** every dependency in `pyproject.toml` is an open `>=` floor with no upper bound, and the
image resolved them fresh at build time. An image rebuilt months from now would have picked up a
different `onnxruntime` or `numpy` — the two libraries that decide CTC alignment numerics and
float rounding — and the corpus exists to measure things. D5 freezes the splits, D6 makes
hypotheses immutable and D42 pins the acoustic model to a commit for exactly this reason; the
dependency set was the last unpinned input, and the one most likely to drift silently.

Hashes ride along with the versions in the export, so a substituted or corrupted wheel fails the
install rather than shipping. `--no-deps` on the editable install stops pip re-resolving anything
the lock already decided.

**A side benefit worth keeping:** the dependency layer now precedes `COPY app`, so a one-line code
change no longer reinstalls all 65 packages.

**What this does not cover:** the build still fetches `hatchling` through pip's build isolation,
and `uv` itself is pinned by image tag rather than digest. Both are build-time tools that cannot
change runtime behaviour, which is where the reproducibility argument actually bites.

**Reversal:** delete the lockfile and restore `pip install -e .`; nothing else depends on it.

## D45 — A Gemini block arrives on `finishReason`, and the audio route stops thinking too
`vertex.py` judges a response through `withheld_reason`, which reads **both**
`promptFeedback.blockReason` (the input refused) and `candidates[0].finishReason` (the output
withheld). `asr_gemini_flash` sets `reasoning_enabled: false`, and `script_restore` moves to
Vertex.

**The bug.** The empty-200 check read `promptFeedback.blockReason` only. Over 229 ad hoc calls
against the live API that field was never populated once — not even on a block forced with
`BLOCK_LOW_AND_ABOVE`. A real block on Gemini 3.8 Flash comes back as HTTP 200 with
`finishReason: "SAFETY"`, populated `safetyRatings` carrying `"blocked": true`, and a `content` of
`{"role": "model"}` with no parts. `audio_chat` is allowed to answer with an empty string —
`ASR_PROMPT` asks for one over silence — so a censored clip and a silent clip were the same
observation, and the censored one became an empty hypothesis in the corpus logged as *succeeded*.
Verified end to end: a captured block body through `VertexClient.transcribe` returned `text=''`,
raised nothing, and wrote `status='succeeded'`. `asr_gemini_composite` was never exposed
(`wants_words` makes an empty transcript fail there), only `asr_gemini_flash`.

`PROHIBITED_CONTENT`, `BLOCKLIST` and `SPII` are non-configurable filters that no `safetySettings`
threshold turns off, so they are caught here regardless of D39's `OFF`. `MAX_TOKENS` and
`RECITATION` are not safety but mean the same thing for a transcript.

**D41's spurious `SAFETY` does not reproduce.** 171 calls found none: 35 with the exact
`audio_chat` payload, and 136 replaying `script_restore`'s real prompts from the failed ingest
across four safety configurations (none, `OFF`×4, `OFF`×5 with `CIVIC_INTEGRITY`, `BLOCK_NONE`×4).
There is also no record to check it against — the retry loop only logs a *final* failure, so a
block that cleared on retry left no row at all. That is what moves `script_restore` to Vertex,
where the audio already goes. `VertexClient.complete` mirrors `OpenRouterClient.complete` so a
text route names a provider in configuration rather than in code. The cost of the move is
visibility: OpenRouter reports the model's reasoning *text*, which is how D44 was diagnosed;
Vertex reports only `thoughtsTokenCount`. Switch the route's `provider` back if that matters more.

**`OFF` is doing real work, so keep it.** At `BLOCK_LOW_AND_ABOVE` all 7 clips of a Nepali phone
review blocked, on `DANGEROUS_CONTENT` at probability LOW–MEDIUM with severity NEGLIGIBLE
(scores 0.20–0.35). That is classifier noise on low-resource-language speech, and `OFF` is what
keeps it off the corpus.

**Thinking off on the audio route.** Measured over 35 clips of 20 s: a mean 895 thought tokens
against 88 tokens of transcript — 91% of billed output, ~170k thought tokens per hour of audio —
and `thinkingBudget: 0` takes it to zero with the transcript unchanged. Unlike D44's route this one
has no `max_tokens`, so nothing made the spend visible. `reasoning_enabled` is one route field
spelled per provider: `reasoning: {enabled}` on OpenRouter, `generationConfig.thinkingConfig` on
Vertex. **Never set it on an `api: transcription` route** — the dedicated recogniser answers any
`thinkingConfig` with `400 Thinking is not enabled for this model`, and reports zero thought
tokens anyway.

**Reversal:** `withheld_reason` is a strict improvement and should stay. The route settings are
one line each.

## D46 — A refused clip costs its segment, never the episode
A segment that cannot be transcribed by **every** configured ASR system is discarded and the run
continues. Only an episode with nothing left fails. What was dropped, and which system dropped it,
rides in the ingest summary and in the SSE stream as it happens.

**Why:** stage 3 is where all the money is — it dispatches every `asr*` route at every clip — and
it was all-or-nothing. One refusal in the last minute of a two-hour episode threw away every paid
transcript that had already succeeded. The first real ingest died exactly that way on segment 34
of a 14-minute video (D44), and the same failure on a podcast would have cost hours of inference
to re-run.

**All systems or none, for that segment.** A segment scored from three systems where its
neighbours used four is not a cheaper segment, it is a differently-measured one:
`word_disagreement_rate` is a mean over the pairs present and carries 0.40 of the priority score,
and nothing downstream records which pairs those were. Dropping the segment keeps every surviving
one comparable. The clip file is unlinked so it cannot reach the manifest's upload by accident;
the `llm_requests` rows stay, because they are the spend record (D34) and that money was spent.

**No fraction threshold.** An episode that loses 60% of its segments is a bad episode, but failing
the run over it re-creates the problem this decision exists to remove. The discard rate is reported
loudly — a `warn` log line, a count in the summary, a per-system tally — and left as the
annotator's call.

**Blame is counted, not buried.** `discard_summary()` returns segments-lost per `system_id`. A run
where one vendor accounts for every discard is a vendor problem, and that is only visible if the
count is kept.

**Reversal:** re-raise instead of recording in `_process_segment`, and stage 3 is all-or-nothing
again.

## D47 — Ingestion is a page, not a dismissible modal
`IngestView` replaces `IngestModal`, reachable at `#ingest`, from the header button, and on `7`.

**Why:** an ingest runs for as long as the episode is long, and it was being watched through a
dialog that closed on a stray Escape, took the whole app hostage while it was open, and lost the
log when it went. A discard summary that only exists at the end of a run needs somewhere it can
still be read afterwards. The running job's id is parked in `localStorage`, so the page reattaches
to a pipeline it was not watching — after navigation, and across a browser reload.

**A finished job is read, not streamed.** The SSE endpoint replays a job's whole history on
connect and the server closes the stream once the job ends. `EventSource` treats a closed stream as
a failure and reconnects every few seconds — so attaching to a finished job replayed its log into
the page over and over: seven copies and growing, measured. The status endpoint now carries the
completion summary as well, so a finished job is rendered entirely from that one request and no
stream is opened; a live job opens one, and closes it on the terminal event. React's `StrictMode`
double-mount was hiding inside the same bug.

**Reversal:** the component is self-contained; nothing else reads `ACTIVE_JOB_KEY`.

## D48 — Three of the four transcribers cannot be steered, so they are no longer sent steering

`asr_scribe_v2` sends no key terms and `asr_mai_transcribe_2` sends no prompt. Both are now
steered by their language code alone, which is the whole of what a dedicated recogniser will
accept. `ASR_PROMPT` still goes to `audio_chat` routes, which is the only shape that reads it.

**Why:** the assumption was that only Gemini 3.5 Transcribe was unsteerable (D39), and that the
other two were being told the transcript policy in the ways they accept. Measured over one
code-switched episode — 23 clips, a TV review in Nepali-English — neither is true.

**MAI ignores the prompt entirely.** OpenRouter does not list `prompt` among the model's
supported parameters (`max_tokens`, `temperature`, `top_p`, `max_completion_tokens`). Four clips
were run through six arms, including a repeat of the control to establish that the model is
deterministic at all. The control, the real `ASR_PROMPT`, and an extreme "write every English word
in Latin, never Devanagari" prompt returned **byte-identical** transcripts on 4/4 clips — as did
dropping `language` altogether, so `language: ne` is also inert and the model auto-detects. The
one input that changes the output is `language: en`, which changes it on 4/4 clips and romanizes
the Nepali half (`तपाईंको living room` → `tapayko living room`). It stays `ne`.

**Scribe's key terms are worse than nothing.** Three arms over all 23 clips: no key terms, the
committed generic list, and an oracle list of 36 terms taken from the episode's own vocabulary —
the list the chicken-and-egg problem says you cannot have. The oracle list *raised*
English-written-in-Devanagari events from 19 to 23 and posted the worst agreement with the other
three systems (0.807 against 0.821 for no key terms). The effect is real but strictly local:
listed proper nouns bind (`एन्ड्रोइड टिभी` → `Android TV`), and everything else is unprotected —
on one clip the oracle list romanized a whole English sentence, `You can get better deals on this
TV` → `यू क्यान गेट बेटर डिल्स अन दिस टिभी`, a 30-point drop in Latin ratio.

**Scribe is also not deterministic**, which is why the arms are reported against a noise floor:
re-running yesterday's exact configuration reproduced it byte-for-byte on only 2 of 23 clips, mean
|ΔLatin%| 1.7. The key-term effects (1.2–3.3) sit inside that band. Only the one 30-point
regression is clearly outside it.

**Consequences.** Scribe drops to the $0.22/hr base rate; the $0.05/hr key-term surcharge is never
incurred, and `calculate_elevenlabs_cost` no longer takes a `has_keyterms` argument. MAI stops
paying for prompt tokens that were discarded. Neither transcript changes in any way the harness
can measure, so no hypothesis already imported is invalidated.

**What this does not fix.** Two of the four systems still write English in Devanagari and now
provably cannot be told not to. The composite has a repair step for it; Scribe has nothing, and
that is a property of the vendor rather than of the configuration. Script correctness is a
post-hoc repair problem or a model-choice problem, not a prompting problem.

**Not done: seeding key terms from another system's output.** It would have coupled two
hypotheses that are supposed to be independent measurements, and the coupling would be tightest at
the code-switch points the corpus exists to measure. A term hallucinated by one system and primed
into another reads as consensus, which *lowers* `word_disagreement_rate` and so hides the segment
from the review queue. Episode metadata is a clean source — this episode's YouTube tags do carry
`Hisense 65Q6Q QLED TV`, `Samsung`, `LG` — but it names the proper nouns Scribe already gets right
and none of the loanwords it fails on (`bleed`, `support`, `sound`, `ethernet`, `boost`). Moot
either way now that the lever is known not to work.

**Reversal:** cheap and local. Restore the `keyterms` parameter on `ElevenLabsClient.transcribe`
and the `DEFAULT_KEYTERMS` tuple, and drop the `dedicated` guard in `transcription.transcribe` so
the prompt reaches the transcription endpoint again. The measurements above are the reason not to.

## D51 — The Gemini composite is removed, not held out
`asr_gemini_composite` and the `script_restore` route it drove are gone from
`config/llm_routes.yaml`. Three ASR systems remain: Scribe, MAI and Flash. The hold-out set
`disagreement_excluded_system_ids()` returns is empty, which is the honest state rather than a
forgotten flag.

**Why.** Its recogniser deletes the English half of code-switched speech, deterministically.
Measured without aligning anything, Spearman(clip Latin share, token shortfall) was **+0.670**
against Scribe, **+0.612** against Flash and **+0.598** against MAI (n=68): three references with
different failure modes agree. Clips more than 10% short were 35% of both episodes measured. On
leave-one-out consensus agreement it trailed the other three (0.538 against 0.610–0.623, which sit
within noise of each other). D50 had held it out rather than removed it because it was one of two
sources of speaker labels; that argument fell when clip-local diarization proved unable to measure
speaker agreement (51 of 68 clips were single-speaker for both systems; see D52).

**Why removal rather than a disabled flag.** A route configured but unused leaves
`exclude_from_disagreement` as the only thing between its hypotheses and the score, and the two
computation sites (`ingest.py`, `purge.py`) desynchronise silently when a system is named in one
and not the other. A route that must never be used is better deleted than remembered.

**The hypotheses are evidence.** `scripts/purge_asr_system.py` wrote every row to JSONL before
deleting; the dump is the record the English-deletion finding rests on.

**Reversal:** restore the route blocks from git and re-ingest. Purged hypotheses come back only
from the dump or by paying to re-transcribe.

## D52 — No cloud ASR route is asked to diarize, reversing D49

`asr_scribe_v2.diarize` is `false`. No route in the table asks for speaker labels, so
`hypothesis_words.speaker` is null for everything ingested from here on. The column stays, and the
ElevenLabs client still honours the flag, so this is one line to reverse.

**D49 turned it on for a reason that no longer holds.** The argument was that two sources make a
label checkable: Scribe plus the composite, with MAI returning no speaker field and Flash no
timings to attach one to. D51 removed the composite, which leaves one source — and a label no
other system can be checked against was exactly what D49 said is not evidence of anything.

**The deeper problem is that clip-local labels cannot answer the question.** Measured on a
two-speaker podcast, 51 of 68 clips were single-speaker for *both* diarizing systems and agreed
trivially at 100%; only 17 clips were contested at all. The pipeline segments before it
transcribes and the VAD cutter runs to `MAX_SEG_SECONDS = 20`, so a clip almost always contains
one speaker. Ingesting more multi-speaker episodes does not raise that ceiling, because the
ceiling comes from the ordering of the stages.

**What the labels cost.** Storage on every word row, and a column that invites a join which is
always wrong: `spk:0` in one hypothesis is not `spk:0` in another, and neither is
`segments.speaker_id`. A future reader has to know that to avoid the mistake, and the only
protection was a comment. Removing the data removes the trap.

**Where speaker identity should come from instead.** A full-episode diarization pass that runs
*before* segmentation, with segments joining to it by time — a new pipeline stage, not a flag, and
therefore not built here. Source audio is retained, so both ingested episodes can be back-filled
when it exists. Word times are clip-local, so the join is
`episode_time = segment.start_time + word.start_time`, and it should be made against Scribe's
spans rather than any recogniser's self-reported ones: the D48 triangulation put Scribe and
CTC-aligned Flash within 27 ms of each other and both ~70–77 ms from the recogniser that was
removed in D51.

**Already-collected labels are left in place.** Turning the flag off stops collection; it does not
delete the 3,188 word rows already carrying a speaker. They are harmless as long as nothing joins
on them, and re-collecting them would cost a re-ingest, so deleting is the expensive direction of
an easily-reversed decision. Clear them with a single `UPDATE hypothesis_words SET speaker = NULL`
if the storage matters more than the option.

**Reversal:** set `diarize: true`. Nothing downstream changed — null already meant "not diarized",
which is what every other system has always reported.

## D53 — Cross-system disagreement is measured on the clock, and shown to the annotator

`app/services/consensus.py` groups every system's word spans for a segment into **slots** — one
stretch of time that one or more systems put a word in — and reports the seed words that every
other system present contradicted. `/tasks/*` returns those as `disputes`, and the editor
underlines them, offers what the others heard, and swaps one in a click.

**Why time rather than strings.** The existing `word_disagreement_rate` compares token sequences
with `difflib`, which has to *infer* which word corresponds to which. That inference fails on
exactly the segments worth reviewing: a repeated word matches the wrong occurrence, and a
re-ordering scores as two errors. Every system heard the same audio, so their spans are
observations of one timeline and the correspondence is a fact rather than a guess. Spans are
clip-relative by construction (D26), so no offset arithmetic is involved.

**Outvoted, not merely disputed.** A slot counts against the seed only when *every* other system
present disagrees. With three systems, one dissenter against one supporter is weak evidence and
underlining it would mark most of the transcript. Measured over the pilot episode the strict rule
underlines a median of 4 words per segment (13.5% of them), which is dense enough to be worth
looking at and sparse enough to read.

**Computed, not stored.** It is a pure function of `hypothesis_words`, which is immutable once
imported, so the answer cannot go stale; and it depends on the *seed*, which is chosen per task at
queue build, so there is no per-segment answer to persist at ingest. No migration, no re-ingest,
and the disagreement rule can be changed without touching the schema.

**Why not fuse the hypotheses instead.** A ROVER-style composite that votes per slot was measured
against the 22 human transcripts available and came out worse than every one of its inputs (0.254
mean WER against Scribe's 0.151, Flash's 0.161 and MAI's 0.205), because 44% of slots do not hold
all three systems and the vote degenerates there. It would also manufacture a transcript no model
produced, with no `asr_hypotheses` row to point `segment_labels.seed_hypothesis_id` at. Handing
the annotator the alternatives keeps the 2:1 signal and leaves the decision — and the provenance —
with a human.

**Reversal:** delete the service and the `disputes` field. Nothing else reads it, no data is
written by it, and no stored column changes.

## D55 — The VAD's speech spans are stored, so silence can be told from a dropped phrase

`segments.vad_spans_jsonb` holds the speech regions the VAD found inside each clip, clip-relative
like word timings (D26). The `missed_speech` flag fires when more than 25% of that speech has no
word from *any* system over it.

**Why it has to be stored rather than derived.** Every other timing in the schema comes from a
transcriber. So "no system wrote anything between 11.2 s and 18 s" is ambiguous from the database
alone: either nobody heard the speech, or there was none. The VAD is the only independent
statement about where speech is, and it already ran — it is what cut the clip — so this is an
intersection of turns already computed, not a second detection pass.

**Why any system counts as covering.** One system hearing a stretch proves the audio was
transcribable there, so it is not a hole in the corpus; it is a disagreement, which
`seed_outvoted` already measures. `missed_speech` is for the region every system skipped. That is
the D50 truncation failure, which was found by hand over one episode and would now raise a flag.

**Why 25%.** The VAD pads turns by 150 ms and word spans cover neither breath nor hesitation, so a
small uncovered remainder is normal. The threshold is deliberately generous: a false flag costs an
annotator a look at a fine segment, and this is the one flag no transcript can corroborate.

**Nullable, and it must stay that way.** Segments imported before this migration have no spans,
and inventing them would make every one either perfect or defective. No spans means the flag
cannot fire, which is the honest reading of an unasked question.

**On the flag vocabulary.** `ALL_FLAGS` goes from seven names to eight, so `rule_flag_score`'s
denominator changes. That term carries 0.15 and was identically zero on every labelled segment
(D54), so the effect on ranking is negligible — but it is a change to a stored number, not just
to a new column.

**Reversal:** drop the column, drop `missed_speech` from `ALL_FLAGS`, and stop passing `vad_spans`
at import. The migration has a working `downgrade`, verified up, down and up again.

## D56 — Speaker name and dialect leave the schema; a speaker allowlist replaces them

The ingest form collected a name, a gender and a `Dialect / Origin` string per speaker, and all
three went into `episodes.metadata_jsonb->'speakers'` and out through the analytics export. Name
and dialect are gone. What a speaker block may now contain is an allowlist — `role` and `gender` —
enforced in `app/services/speaker_meta.py`.

**Why the name goes.** It is identity, and being a public figure does not make it less so. The
corpus already holds this person's voice, their words and a timestamped clip of both; a name is
the join key that turns a research corpus into a dossier about a named individual who never
agreed to be in one. There is no analysis in this project that needs it: everything the harness
computes keys off `speaker_id`, which is episode-local and says nothing about who the speaker is.

**Why the dialect goes, which is the less obvious half.** The owner cannot label Nepali dialect
reliably, and neither can anyone else working on this corpus casually. That does not make the
column merely empty, it makes it *wrong*: a stratification variable is not a note, it is a
grouping that every result computed over it inherits. A dialect field would be filled in with a
guess for the episodes someone felt confident about and left blank for the rest, and the resulting
comparison — "code-switching by dialect" — would be a report on the labeller's confidence rather
than on Nepali. No column is a truthful account of what is known here. A wrong one is not.

**Why an allowlist and not a blocklist, and why at the importer.** Episode metadata arrives from
two directions: this repository's ingest form, and an upstream `episode.json` written by a
pipeline that is not in this repository. `episode.schema.json` keeps unknown properties on purpose
so a new upstream field is never silently lost — right for provenance, wrong for people, because
it means any key the upstream pipeline invents lands in the database and in every export. A
blocklist would have to be extended every time the upstream form gains a field; an allowlist
defaults to dropping. `_upsert_episode` is the one choke point both directions pass through, so
that is where it runs. The API applies it too, so a name never even reaches the job's
`episode.json` on disk, but the importer is the guarantee.

**Gender stays, as a hand-entered field.** It is a coarse variable a code-switching study
genuinely stratifies on, and it is two values on a form rather than a claim about anyone's
identity beyond what the annotator chose to record. Whether to *infer* it acoustically is a
separate question and is not settled here.

**Rows already written.** Migration `9b1c0d4e2a71` rewrites every stored speaker block down to the
same allowlist. Its `downgrade` is a no-op and says so: deleting the names is the point of the
revision, the data is kept nowhere to restore them from, and the column shape never changed, so
the previous revision's schema is still correct after downgrading.

**Reversal:** delete `speaker_meta.py` and stop calling it. The fields would then be storable
again — but the names already deleted are gone, and that is intentional.

## D57 — The episode topic is classified from the transcript, not typed into a box

`ingest.topic_route` names a text route that labels one episode with one topic, from an excerpt of
what was actually said in it, when the ingest form left the topic blank. The answer must be a
member of the closed taxonomy in `app/llm/topic.py` or it is discarded.

**Why automate it.** The free-text `Topic / Domain` box was almost always left empty, which is the
worst outcome available for a stratification variable: a column filled in for the episodes someone
had energy for cannot be reported on, and its gaps are not random. A label on every episode, from
one source, applied the same way, is worth more than an occasional hand-written one.

**Why the transcript and not the title.** A YouTube title is marketing, and an uploaded file has
no title worth the name. The transcript is the only description of the episode the harness holds
that the episode itself produced. The title is passed as a hint and nothing more.

**Why the excerpt is sampled across the episode.** Nepali podcasts open with greetings, sponsor
reads and channel promotion, so the first few minutes are systematically the least topical part of
the recording. A prefix would classify the advertising. `sample_transcript` spreads its picks from
the first segment to the last within a character budget, and the prompt says the excerpt will read
disjointedly so the model does not try to make one narrative of it.

**Why a closed taxonomy.** Free text would produce a hundred one-episode categories with no two
spelled alike, which is the same unusable column by a different route. Sixteen labels, `other`
included so an off-taxonomy episode is not forced into a wrong one. An answer outside the list is
treated as no answer: the episode keeps an empty topic that a human can fill in, which is strictly
better than a category that exists once. It is not retried — a model that answered off-taxonomy at
temperature 0 will do it again, and a billed retry loop buys nothing.

**Why it cannot fail an ingest.** By the time it runs, the audio, the clips, the transcripts and
the queue all exist. A metadata field is not worth discarding them for, so every failure — an
unconfigured route, a provider outage, an unparseable answer — is logged and leaves the topic
empty. The one call per episode is routed and logged like every other (invariant 5), so what
topic labelling costs is visible in `llm_requests` beside the transcripts.

**A hand-typed topic always wins.** The classifier only runs when the form's topic is blank, and
the stored `topic_source` says which of the two produced the value.

**Reversal:** set `ingest.topic_route` to an empty string. Nothing else changes; episodes already
labelled keep their label and their `topic_source`.

## D58 — Speaker demographics are declared by hand; neither gender nor age is inferred from audio

Both were tried against real speech and both failed, so the ingest form asks instead. What follows
is kept because it is the evidence for a field that now costs an annotator two seconds, and
without it the obvious "why not just infer this?" gets asked again every six months.

**Gender from voice does not transfer to Nepali.** `JaesungHuh/voice-gender-classifier` (MIT,
ECAPA-TDNN, 15.5M params, a 15 MB int8 ONNX build) reports **98.7% on VoxCeleb1**. Measured
through one harness on FLEURS read speech:

| Language | n | Accuracy | Male recall | Female recall |
|---|---:|---:|---|---|
| English | 50 | 96.0% | 23/25 | 25/25 |
| Hindi | 100 | 84.0% | 36/50 | 48/50 |
| **Nepali** | **100** | **65.0%** | **65/100** | *(FLEURS ne_np has no female clips)* |

English reproduces the published figure, which is what rules out a bug in the measurement: same
code, same graph, three languages. The degradation is systematic — male recall falls 92% → 72% →
65% moving to South Asia while female recall holds, so the model calls about a third of Nepali male
speakers female. Confidence thresholding answers 44% of speakers at 92% and *falls* to 85% at
`p ≥ 0.99`, so it is not calibrated at the top end here either; pooling 3-second chunks across a
speaker lifts 74% to 82%. At 65-82% this is the D56 failure exactly: a stratification variable
wrong often enough to bias everything computed over it, replacing a dropdown a human gets right.

**Age is not attempted at all**, for the same reason one step further along. Published speech age
estimation runs at 7.1-10.8 years MAE, which makes a decade bracket right about 35% of the time
before any cross-lingual penalty — and the gender result says that penalty is large, not modest.
The only credible model is `audeering/wav2vec2-large-robust-*-ft-age-gender`: PyTorch only (D32
keeps torch out), 0.3B parameters, and CC-BY-NC-SA 4.0, whose non-commercial and share-alike terms
would propagate to a corpus built from its output. Twenty-year buckets typed by someone who can
see the speaker are both cheaper and better.

**In-pipeline diarization is not the answer either, and was removed.** An episode-level clustering
stage was built and measured before being deleted. On real Nepali and Hindi speech, cosine distance
between two turns of the *same* speaker averaged 0.28 at 4-second turns against 0.66 for different
speakers — but the distributions overlap at every turn length tried (at 4 s: within-speaker p90
0.43, between-speaker p10 0.31), and the best threshold scored 80-90% correct partitions on
two-speaker episodes. Good enough to be tempting, not good enough to be a grouping variable, and
it put a second-rate diarizer in the hot path of every ingest.

So `segments.speaker_id` stays `spk0` for everything, and diarization moves **after** the export,
where a serious tool (pyannote 3.1 and its successors) can run against the full episode audio with
no time budget and be re-run when it improves. That is what D62 exports the episode audio for.

**Reversal:** none needed — the inference code is gone rather than switched off. The form fields
are ordinary metadata and the allowlist in `speaker_meta.py` is what governs them.

## D62 — The whole episode's audio is kept and exported
Only the clips used to survive an ingest: the normalised full recording was deleted with the job's
work directory, so nothing downstream could look at an episode as a whole. The importer now
uploads it to `episodes/<id>/audio.flac` and records `episodes.audio_object_key`, and every export
with an object store to read from copies each episode's recording into `episodes/` beside the
rows, listed in the manifest with a digest.

**Why.** Speaker identity needs a whole recording, not a bag of clips (D58). It is also what every
later per-episode pass reads without re-ingesting: the Modal diarizer's re-runs (D79), and the
overlap and acoustics backfills (D77, D87).

**Copied, not referenced by key**, so an export directory stays self-contained and can move to a
machine with no access to this deployment's object store. A missing object is a warning, never a
failed export: episodes imported before the column have nothing to point at, and the metadata is
complete without the audio.

**Reversal:** drop the column and the export step. Exports already written keep their copies.

## D63 — The tier records how hard a label was looked at
> **How gold is chosen is superseded by D71**: this entry also filled episode-level pots to an
> hours target with a coverage-first assigner, which is gone. What stands is below.

`segment_labels.verification_tier` is `verified` (played and read) or `screened` (accepted on
cross-ASR disagreement without listening). It defaults to `verified`, every export row carries it,
and the manifest reports the mix per split.

**The tier is the corpus's claim about itself.** The train pot is meant to be screened: that is
the only way a large corpus is affordable. But a screened row written as `accepted_unchanged` by
`annotator: owner` asserts a human verified it. That is false, and it costs twice — a reviewer
asking about the verification protocol gets a wrong answer, and the disagreement gate can never be
measured, because its decision was overwritten by a confirmation nobody made. The audit sample
(`queue.audit_sample_rate`) is the instrument: re-verify a sample of screened clips and the gate's
error rate falls out.

**Gold holds only what was listened to.** A screened decision on a gold clip is refused at
`record_decision` with a 409, moving a screened clip into gold is refused (D71), and the `gold`
export refuses to write at all if a screened row reaches it — belt and braces, because the failure
is silent and the artefact outlives the session that made it.

**Bad audio is flagged, not deleted.** `unusable_audio` keeps "what fraction of real Nepanglish
podcast audio is untranscribable", which is a publishable number. Deletion exists for junk (D94).

**Reversal:** the migration has a working `downgrade`. Dropping the tier makes every screened row
indistinguishable from a verified one, which is the thing this entry exists to prevent.

## D64 — Orthography is normalized at export, not in the database

`config/normalization.yaml` holds a reviewed table of whole-token substitutions, applied by
`app/services/normalize.py` to `segment_labels.final_text` on the way into an export. The table's
`version` is written to every manifest as `normalization_version`. The database is not touched.

**The corpus had drifted to a coin flip.** In the first 340 labels, `चैँ` appears 115 times and
`चाहिँ` 127 — the same word, both spellings, across 70 labels. Neither is a transcription error;
the owner simply prefers the fuller written form and fixed it on the clip in front of them 35 times
without going back for the other 115. Twenty-two other clusters had drifted the same way (`हरू`/`हरु`,
`भनौँ`/`भनौं`, `चिज`/`चीज`), 39 occurrences in total. Left alone this is worse than either spelling
would have been on its own: a model trained on it learns that the choice is free in identical
contexts, which is precisely what fixing it by hand was meant to prevent.

**Export, not import.** Normalizing the ASR seed before the annotator sees it would save the
retyping, and it was rejected for now. It puts text nobody wrote in front of a reader who is
measurably inclined to accept it — the accept-unchanged rate is 33% and a pre-cleaned seed reads as
more trustworthy than it is, so the errors it buys are invisible in exactly the place they matter.
Export-time normalization also re-runs over the whole corpus every time, so a rule added in month
three applies to episode one; a seed-time rewrite would leave the corpus stratified by when each
clip happened to be labeled, which is the drift this decision exists to remove.

**The database keeps what was typed.** Nothing rewrites `final_text` — D6 makes labels append-only,
and a backfill would have to insert new rows rather than update. It is also unnecessary: the table
is applied on read, so revising a rule is a config edit and a re-export, not a migration, and the
record of what a human actually wrote survives every revision of house style.

**Rules are approved, never mined.** The table was built from the corpus's own variant clusters, but
by hand. An automated pass over the edit history proposed `जे → day` — from a single clip where the
English word "day" had been transcribed as `जे` — and `जे` is a common Nepali word that appears in
three labels. Mining produces candidates; a rule is a claim about a language and needs someone who
speaks it. Nine observed clusters are listed in the file as deliberately **not** normalized
(`ऊ`/`उ` and `ई`/`इ` are separate letters, not diacritic variants; `गर्या`/`गऱ्या` differ by U+0931;
`बिचमा`/`बीचमा` are both standard with no majority to appeal to). `हामीहरूको`/`हामीहरुको` was an
exact 5–5 tie, broken toward the long `ू` because every other contested pair in the corpus votes
that way.

**Two mechanical traps, both tested.** `str.replace` fires inside longer words and would turn
`चैँको` into `चाहिँको`, so substitution is whole-token. And a Devanagari token class written as the
obvious `[ऀ-ॿ]` swallows `।` (U+0964 DANDA), which lives inside that block — every rule would then
silently stop working at the end of a sentence. `WORD_TOKEN_RE` cuts U+0964 and U+0965 out of the
range. The ruleset is also rejected at load if any rule chains into another (`A → B`, `B → C`),
because the result would depend on iteration order and would not be idempotent.

**The cost.** This teaches a model one orthography. It will not emit `चैँ` even where a speaker's
register calls for it, and WER measured against any other Nepali corpus will be inflated by pure
spelling disagreement. That is why `normalization_version` is in the manifest rather than implied:
evaluating against this dataset means applying the same table to the reference side.

**Reversal:** delete the `tokens` block and re-export. There is no migration to undo and no label
to recover, because none was ever overwritten.

## D66 — An ingest has a stop button (AZ-5)
`POST /ingest/{job_id}/scram` sets a flag. The pipeline reads it at each stage boundary and,
crucially, at the top of `_process_segment` — every route of every clip is dispatched below that
line, so a scram is inference that is never billed. Requests already on the wire are left to
return rather than having the client torn down under them, because they are billed either way and
their `llm_requests` rows are the only record of it.

A scrammed run **imports nothing**. That is the D46 argument: half an episode is not a cheaper
episode, it is a differently-sampled one, and nothing downstream would ever say so. The abort
summary reports how many segments were transcribed before the stop, so the spend is visible.

It was built during an outage in which a failing script-restore call (since removed, D73) failed
every segment of a run that could not be stopped.

**Reversal:** delete the endpoint and the `job.scrammed` checks; nothing persists a scram beyond
the job's own memory.

## D69 — The analytics page answers what to record next, and the scoreboard is deleted
> **The inventory itself is rebuilt by D91**: the unit is the clip and the person is the voice.
> What stands is below.

The dashboard's levels, streaks, daily goal and achievements (`app/services/gamify.py`) are
deleted. Nothing was stored, so there was no migration. The corpus is short of *speakers*, not
hours, and a scoreboard of cleared audio rewards the one axis that was already sufficient. A page
whose most prominent number is the wrong number is worse than one with no number, because it is
read as advice.

Three rules from the page that replaced it still hold, each because its opposite produces a number
that looks fine and is false:
- **A stratum is judged by absence, thinness and dominance, never against a target
  distribution.** There is no defensible ideal share of any stratum; there is a floor below which
  it supports no claim (`dataset.min_stratum_hours`).
- **Every recommendation carries the measurement that produced it.** A recommendation whose number
  is not visible is an opinion.
- **Off-taxonomy values are reported as dirt, not as gaps.** A free-typed topic cannot be
  stratified on, and otherwise looks exactly like a topic that is simply rare.

`dataset.train_hours_target` dropped from 50 h to 20 h in the same change: the work is domain
adaptation onto a pretrained model, which saturates roughly an order of magnitude earlier than a
from-scratch curve.

**Reversal:** restore `gamify.py` from git; it read only tables that still exist.

## D70 — The gender vocabulary narrows to two values

`ALLOWED_VALUES["gender"]` in `app/services/speaker_meta.py` goes from
`{male, female, non_binary, other}` to `{male, female}`, and the ingest form, the episode JSON
schema and the analytics grid follow it. No migration: the vocabulary is an application-level
allowlist over `episodes.metadata_jsonb`, not a database constraint, and no row has ever carried
either removed value.

**The reason is that neither was ever fillable.** Nepali-language podcasting has almost no openly
LGBT creators to source from, and the fields are typed by hand from having watched an episode
(D58) — so the only way `non_binary` could have been populated was the owner guessing at a
stranger's gender identity, which is exactly what D56 refused to let this corpus do with dialect
and for the same reason: a confidently wrong stratification tag is worse than no tag, because it is
used as a grouping variable and silently biases every result computed over it.

**An unfillable vocabulary value is not neutral, which is what forced the decision now.** D69's
sourcing page ranks gaps by absence, and absence is exactly what these two are. They came out as
the **top two recommendations at priority 1.00**, above every gap that could actually be closed —
telling the owner to go and find non-binary Nepali podcasters ahead of the age brackets, the
register pole and the show concentration that are all genuinely actionable. A page that ranks
unclosable work first is worse than no page. Narrowing the vocabulary was the honest fix; special-
casing two values inside the recommender to suppress them would have hidden the same claim behind
a filter nobody would find again.

**This narrows what the corpus records, not who it records.** A speaker neither value fits is
recorded with the gender field dropped — the same path any unknown value takes — and the episode
is then *unrecorded* on gender rather than counted as evidence for either value. The inventory
already distinguishes those two states: an unfilled field lands in `unknown_episodes`, reported as
paperwork, while a thin stratum is reported as a fact about the corpus. What is lost is the
ability to state the distinction in an export; what is kept is the ability to tell "nobody
recorded this" from "this does not exist".

**Reversal:** put the two values back in `ALLOWED_VALUES`, the schema enum and `GENDER_OPTIONS`.
Cheap and total while nothing is stored under them. It stops being cheap the moment a speaker is
recorded under one and a later narrowing would drop their record on re-import — so if the corpus
ever does reach such a speaker, widen it first and re-ingest rather than editing the row.

## D71 — Gold is chosen one clip at a time, by hand

Supersedes D63's assigner and D68's demotion script. `segments.pot` (`gold` / `train`, default
`train`) replaces `episodes.pot`; `POST /segments/{id}/pot` moves one clip, from a star button on
the triage row, `g` on the keyboard, or "Add to gold" in the editor. `episodes.split` keeps only
`train` / `val` / `unassigned`, drawn by hash at import, and a gold clip exports as `test` whatever
its episode drew. `assign_pots`, `demote_from_gold`, `POST /pots/assign`,
`dataset.gold_max_corpus_fraction` and both scripts are deleted. Ingest now builds the queue
itself, because nothing is left for it to wait on.

**Why.** The owner does not trust what the algorithm put in gold and wants to decide it. D63's
greedy coverage-first selection was never measured against anything, and D68 had already had to
override it once by hand. A benchmark whose membership the owner cannot defend clip by clip is a
worse benchmark than one they chose.

**What survives, because it is a property of a label rather than of a selection rule.** Gold holds
only what was listened to: `set_segment_pot` refuses to move a clip carrying a screened label into
gold (409 at the API), `record_decision` still refuses a screened decision on a gold clip, and the
gold export still refuses a screened row. Every move writes an `audit_logs` row
(`entity_type=segment`, `action=pot_changed`, old and new pot, optional reason), so the benchmark's
history can be rebuilt even though a clip may now leave gold.

**The cost, stated rather than discovered.** D63's two rules existed for measured reasons and this
decision gives both up:

- *Chosen while looking at the clip.* Selection can now correlate with anything the owner notices
  -- clean audio, one speaker, an easy transcript -- and gold WER then describes the clips that
  were chosen, not the corpus. Nothing recorded afterwards can separate the two.
- *Clips, not episodes.* Consecutive clips share audio padding, speaker, topic and rare
  vocabulary, so a gold clip whose neighbours are training material is partly memorised by any
  model trained on this corpus. Its gold score is optimistic in a way no metric reveals.

The second is at least countable, so it is counted. `pot_status` reports how many gold episodes
also feed train and how many gold clips they hold, the analytics page says so when it is nonzero,
and every export row carries `episode_spans_pots`. A paper can report gold WER both over all gold
clips and over those from episodes nothing was trained on -- if the second set is too small to
mean anything, that is itself the finding.

**Reversal:** mechanical but real. Reintroducing episode-level pots means deciding what an
episode with mixed clips becomes; the migration's downgrade makes any episode with a gold clip a
gold episode, which moves that episode's train clips into the benchmark. Do it before any model is
trained on this corpus, or not at all.

## D72 — A reasoning model fuses the recognisers into one transcript per clip

A sixth ingest stage, between transcription and analysis. `fuse_transcript` (Vertex, thinking
Gemini 3.8 Flash, `thinking_budget: 24576`) reads every recogniser's text for about thirty
minutes of consecutive clips and writes one verbatim transcript per clip, which is imported as one
more hypothesis under an `asr_systems.kind = fusion` system (`fusion-gemini-3.8-flash-p1`). Its
words are placed on the clip by the CTC aligner. Code-mixing is measured on it. D74 makes it the
seed.

**Why.** Each recogniser hears one clip and nothing else, so each gets wrong what only context
fixes: a name said clearly a minute earlier, a term the speaker keeps using, a sentence finishing
in the next clip. Scored on 896 gold labels, each system only where it had not seeded the label,
the pilot fused text posted 0.192 WER against 0.262 (Scribe), 0.304 (MAI) and 0.313 (Gemini). On
the 120 clips whose label matched no hypothesis -- the only reference none of them could be
anchored to -- it was at parity with Scribe, 0.245 against 0.250. The owner listened to the
disagreements and preferred the fused text. After their spot check corrected the pilot's
hallucination count, one block of words no recogniser produced remained in 896 clips, and it
was not an invention.

**Windows, not the whole episode, and not one clip.** One clip loses the context that makes fusion
work. A whole long episode is slow and makes one bad response cost everything, although the pilot
measured an entire hour of three hypotheses at ~65k tokens, so context is not what binds. Output
is: thoughts and answer share `max_tokens`, and one 40-segment pilot window spent 27,922 thought
tokens against 3,361 of answer. So windows are about 30 minutes, balanced across the episode,
with ~2 minutes of raw lookahead and ~5 of the fuser's own earlier output as carry-over, and
thinking is bounded. A 5-20 minute video is one window.

**What is refused rather than patched.** The answer must be one JSON object per target id. A broken
contract gets one retry; a truncated answer is halved at once; a window still failing at
`max_depth` leaves its clips unfused, and those clips keep their recognisers. The recognisers'
work is paid for before fusion starts, so fusion failing -- even raising -- never fails an
episode.

**Known, unmitigated, and recorded.** The fuser is a Gemini model arbitrating a slate with a
Gemini transcript in it; the systems are anonymised as A/B/C, and nothing more is done about
self-preference. Prompt order is fixed and its effect unmeasured. World knowledge is how the fuser
corrects, and it is also how it would quietly "fix" a speaker's misstated fact. The instruction
prohibits that explicitly, and nothing in the harness can see it happen. Thinking tokens are
billed as output and are most of this route's cost -- the pilot reported $0.64 when it actually
cost about $3.47, because thinking was left out of the estimate. That is fixed in the same series.

**Reversal:** set `fusion.route: ""`. Ingests go back to three recognisers and a recogniser seed;
fused hypotheses already imported stay (hypotheses are immutable) and can be ignored by kind.

## D74 — The fused transcript is the seed, and the queue looks for what fusion got catastrophically wrong

The seed -- what the editor opens with -- is the newest fused hypothesis, for every clip,
**gold included**. The priority score is rebuilt around it, hazard gates make a clip
unscreenable, and every comparison goes through a new script-folding normalizer
(`app/services/fold.py`, `fold-v1`). D67's formula is kept only as a recorded `legacy` score
(removed in D82).

**The old score cannot follow the seed.** `seed_outvoted`, `seed_orphan_rate` and `roman_gap`
measure how far the seed is from the recognisers. A transcript built to reconcile them is near
zero on all three by construction, so the queue would sort by almost nothing, and the
auto-approve rate would jump with no evidence that a single label had improved. Given that the
screening ceiling is set by the base edit rate, that is exactly the number that must not be
allowed to fool anyone.

**What replaces it asks where the seed and the evidence part company.** `unsupported_rate`: fused
words no recogniser heard, by sound. `dropped_rate`: words two recognisers heard that the fused
text lacks -- deletion, which every orphan-style signal is blind to by construction.
`asr_disagreement`: the recognisers against each other, which is still independent and still
measures how hard the audio is. `acoustic_gap`: the aligner's goodness-of-pronunciation gap
between the fused text and its own reading of the clip. That separates hard audio from
invented words, and the plain forced-path posterior could not (new-plan §4.2). Weights 0.30 /
0.20 / 0.20 / 0.15, plus 0.10 for Scribe's logprob and 0.05 for rule flags. They are provisional
-- nothing verified has been scored against them.

**Gates are about shape, and they are what actually protects screening.** A contiguous run of
unheard words (`invention`), a contiguous run two recognisers share and fusion lacks (`dropped`),
unheard words the neighbouring clip heard (`seam_bleed` -- the failure long windows invite), a
fused text of the wrong size (`length_outlier`, `emptied`, `speech_over_silence`), text the
aligner cannot fit into the clip (`unaligned`), the fuser's own `u`, and no fused text at all
(`unfused`). Any one lifts the clip's priority by 1.0 above every ungated clip and makes
`record_decision` refuse a screened decision (409). Scattered unheard words rank but do not gate:
a correction only the fuser could make is also a word no recogniser produced, and gating on that
would punish the thing fusion is for. The acoustic gap gates nothing until it is calibrated.

**The normalizer is aggressive on purpose, and only as a comparison.** Scribe and MAI
transliterate English into Devanagari whatever they are told, and fixing every instance in the
corpus is not feasible. So `fold.py` treats a Latin word and a Devanagari word as the same when
their consonant skeletons match (`एक्टिभ`/`active`, `कफी`/`coffee`, `phoneमा`/`फोनमा`). It folds
the spelling variants Nepali writers do not hold consistently (vowel length,
chandrabindu/anusvara, nukta, final virama, digit script, the D64 table), and accepts a zero-cost
two-to-one merge for spacing (`गर्नुभयो`/`गर्नु भयो`). It never matches two Devanagari words by
sound, because `गर्नु` and `गर्ने` share a skeleton and are different words. On the gold pot about
half of every system's substitutions were respellings of these kinds. The owner accepted the
cost: a WER computed this way cannot see a wrong script, and it erases the loanword/English
distinction in the metric. Labels are never rewritten by it; the D64 table still decides exported
spelling. Report raw and folded WER side by side, named by `fold_version()`.
*(Superseded in part by D84: fold-v2 also drops fillers and folds numbers, contractions and
colloquial Nepali.)*

**Gold is seeded by fusion too, and that is a real cost the owner chose.** D63 rotated gold's seed
across recognisers so the benchmark was not anchored to one of them. Gold labels will now lean
toward the fuser's reading, and a model fine-tuned on fusion-seeded train labels shares that
lean, so its gold WER will be flattered against the commercial systems by an amount nothing here
measures. The pilot measured the anchoring effect at about +0.23 WER for any system on clips it
had seeded. Gold is still 100% listened to, which bounds the damage to what a listener fails to
catch, and that is not zero.

**What no gate can see.** A knowledgeable correction of a speaker's actual misstatement sounds
right, aligns fine, agrees with at most one recogniser, and passes everything above. The fuser's
instruction forbids it, and the harness cannot check.

**Reversal:** `select_seed_hypothesis` back to the strongest recogniser, and the D67 weights back
into `queue.weights` (from git; the legacy plumbing was removed in D82). The gates are independent
of the seed and could be kept.

## D75 — Ingest runs from a persistent queue, with a backlog for YouTube bot checks
`IngestionManager` (`app/services/ingest/manager.py`) separates submission from execution. It
holds pending, running, backlogged and past jobs and writes them atomically to
`queue_state.json`, so a browser disconnect neither halts nor hides a pipeline. Several jobs run
at once since D88, and interrupted jobs resume at startup since D93.

**A bot check backlogs one job, not the queue.** YouTube periodically answers with a CAPTCHA,
"Sign in to confirm you're not a bot", or a 429. `_fetch_source_audio` traps `YouTubeBotDetected`
and moves the job to `backlog`, and the queue goes on to the next episode. Backlogged jobs are
retried one at a time (`POST /ingest/{id}/retry`) or all together (`POST /ingest/retry-all`).

**One video per submission, each with its own metadata.** A batch endpoint that took a list of
URLs with one show, genre and topic and no speakers was removed (D80): every video in it was
diarized with a guessed speaker count. The submit button stays enabled while another episode
ingests and reads "Add to queue".

**CPU work is parallel.** Clip extraction (slicing, edge fades, FLAC encoding, checksums) runs on
a `ThreadPoolExecutor` of up to `min(8, os.cpu_count())`; the VAD and aligner ONNX sessions set
`intra_op_num_threads` up to 8; loudnorm passes run FFmpeg with `-threads 0`.

**Reversal:** replace `IngestionManager` with synchronous runs; set `intra_op_num_threads = 1`
and extract clips in a plain loop.

## D76 — A training row never holds gold audio

Amends D71 without reversing it: gold stays chosen per clip, and the line between gold and
training is drawn at the **audio**, not the clip id. The `training` export (the only kind with
`clear_of_gold`) drops any train or val clip whose time span overlaps a gold clip's in the same
episode, lists the dropped ids in the manifest as `excluded_for_gold_overlap`, and raises
`GoldLeakError` -- writing nothing -- if a gold clip ever reaches its query at all.

**Why.** Clip ids were already disjoint, because a gold clip exports as `test`. Audio was not.
Measured on the 2026-09-12 export: 4 train clips shared 0.05-0.18 s (0.4 s in total) with a gold
clip, through the padding VAD puts around each cut. That is a fragment of the benchmark inside
the training data. Trimming the train clip would leave a transcript that no longer matches its
audio, so it is dropped.

**What this does not fix, on purpose.** 36 of 42 episodes still put clips on both sides, so a
gold clip's speaker, topic and neighbouring sentences are in training (D71's stated cost, still
counted by `episode_spans_pots`). Episode- or voice-level gold was measured and declined for
now: making gold whole episodes would take 19.0 of the 19.76 train hours out of training. The
owner's plan is to add voices that never appear in gold as the corpus grows, which gives a
speaker-held-out test without emptying the training set.

**Reversal:** delete `clear_of_gold`, `_gold_spans` and the manifest field. Doing so puts gold
audio back into training exports.

## D77 — Overlapped speech is measured at ingest and raised as a heads-up, never scored

Every clip now carries `segments.overlap_spans_jsonb`: the clip-relative stretches where two or
more people talk at once. A clip with at least `queue.overlap_flag_min_seconds` (0.5 s) of it
gets the `speaker_overlap` flag. The flag is a chip in the editor and a field in the export, and
nothing else: it is listed in `INFO_FLAGS`, outside `ALL_FLAGS`, so `rule_flag_score` never
counts it, and no gate, pot rule or export filter reads it.

**Why measure it.** Crosstalk is the largest error factor found in the corpus (docs/findings.md,
2026-09-13). Within an episode, each 10 points of a clip's overlap share multiplies Flex's error
rate by 1.49 [1.32, 1.69]. Podcast WER is 9.0% on clips without overlap and 24.3% above 15%,
mostly dropped words. Speaking rate and code-mixing show no effect once the episode is held
fixed. A number that large belongs next to CMI and speaking rate in every evaluation, and in
front of the annotator on the clip it affects.

**Why a heads-up and not a ranking signal.** Overlap is a property of the audio, not evidence
that the transcript is wrong. Scoring it would push overlapped clips up the queue for no reason
the annotator can act on. Gating or dropping them would bias the corpus toward single speakers
and flatter every WER computed on it, which the owner called cheating. So the clips stay in
every pot and every export, and the flag tells the annotator where the audio is hardest and
where the seed, fused from recognisers that each followed one voice, is least reliable.

**Why detection runs locally, when D58 took diarization out of ingest.** D58 removed a
clustering diarizer: speaker *identity* from voice embeddings, 80-90% correct and expensive.
Overlap *detection* needs no identity. pyannote's segmentation model alone says how many local
speakers are active in each ~17 ms frame. The embedding and clustering steps, which cost 97% of
pyannote's CPU time, only link those local speakers into episode-wide identities. So this runs:
- **cheaply:** about 25 s of CPU per hour of audio on four threads, inside the VAD stage that
  already holds the episode in memory;
- **without torch (D32):** `app/services/overlap.py` ports pyannote's `speaker_count` to numpy
  and onnxruntime, over the public MIT export `onnx-community/pyannote-segmentation-3.0`. The
  graph is fetched on first use, pinned to a commit and digest-checked, as the aligner is (D42);
- **faithfully:** on a 10-minute podcast excerpt the port matched pyannote 4.0.7's speaker count
  from `speaker-diarization-community-1` on all 35,552 frames, with the same 7.7 s of overlap.
  Speaker identity still does not come from ingest.

**`[]` is not null.** An empty list is a clip the detector read and found clean; null is a clip
nobody measured. The importer keeps them apart, unlike `vad_spans` (D55), because an evaluation
that reads overlap must not count unmeasured clips as clean ones. A missing model or a detector
failure leaves the clips null and the episode completes.

**Already-imported episodes** are measured by `scripts/backfill_overlap.py` from the episode audio
retained under D62. It writes one `audit_logs` row per episode. It updates the stored flags, and
the `flags` list in open tasks' reasons, where the editor reads its chips. It does not rebuild the
queue: the flag is not scored, and an episode-scoped rebuild would re-draw that episode's audit
sample.

**Why 0.5 s.** 25% of podcast clips have some overlap and 14% have half a second or more. Below
that the overlap is mostly a clipped backchannel, and a chip on every fourth clip stops being
read. The threshold is a setting.

**Reversal:** drop `speaker_overlap` from `INFO_FLAGS` and the overlap pass from ingest.
Downgrade the migration to drop the column. Nothing ranked on the flag, so no queue or label
changes.

## D78 — Speaker turns are stored per diarization run, never computed in the harness
> **Where and when the diarizer runs is D79**: ingest calls it on a Modal GPU for every new
> episode. Turns made outside the harness are not accepted (D86).

Two tables hold who spoke when:
- `diarization_runs`: one diarization of one episode, with its model, source file, a checksum,
  and the speakers ordered by talk time;
- `speaker_turns`: the run's episode-relative turns, which may overlap.

The editor uses the newest run of the clip's episode to colour each word by its speaker. This is
D58's plan and D52's "full-episode pass, joined by time": a word's episode time is
`segment.start_time + word.start_time`. No torch, no embeddings and no diarizer in the harness
process.

**Why it can be trusted.** pyannote's `speaker-diarization-community-1` and the EDA's ECAPA voice
prints are independent systems. They agreed on who is talking in 97.3% of 41,285 three-second
windows across 16 multi-speaker episodes, and in 99.2% of the windows pyannote calls
single-speaker. The disagreements sit in turn changes and overlap, which the editor marks
separately (D77).

**Runs are append-only, like hypotheses.** A newer diarization is a new run and the older one
stays; "current" is the newest run per episode. The same answer stored twice is a no-op, keyed by
a checksum over the model and the turns.

**Speaker numbers, not names.** A run's labels (`SPEAKER_00`) mean nothing outside that run, and
diarization cannot say which voice is the declared host. The editor shows "Speaker 1, 2, ..." in
talk-time order, stable across every clip of the episode. `segments.speaker_id` is untouched and
still `spk0` (D56, D58). Per-speaker embeddings are stored with the run; D87 links them into
voices across episodes.

**Cost.** About 1,000 turns per hour of audio: roughly 100 KB per hour in Postgres including the
index.

**Reversal:** drop the two tables (the migration's `downgrade`). Without turns the editor falls
back to uncoloured words.

## D79 — Ingest diarizes every new episode on a remote GPU

Right after stage 1, ingest sends the whole normalised episode FLAC to pyannote's
`speaker-diarization-community-1` on a Modal L4 (`scripts/modal_diarize.py`), on a background
thread. It collects the answer at stage 6 and stores it as the episode's first D78 run. New
episodes get speaker colours the moment they land. `scripts/diarize_episode.py` makes the same
call for episodes already imported. `notebooks/05-diarize.ipynb` is deleted: a Colab runtime, a
Drive mount, a manual import, and colours that lagged every ingest by however long it took
someone to rerun them.

**What this reverses, and what it keeps.** D58 and D78 kept the diarizer out of ingest. The part
of that which mattered stays: nothing heavy runs in the harness process. No torch (D32), no model
download, no GPU, and the backend adds one HTTP call. What changes is the trigger: ingest now asks
for the run instead of waiting for someone to make it. The model, pyannote version (4.0.7),
output shape and speaker-count rule are the notebook's, so D78's 97.3% agreement with the ECAPA
voice prints still describes these runs.

**A heads-up, never a stage failure** (as overlap detection is, D77). If the service is down,
refuses, times out or returns something the importer rejects, the job logs a warning and the
episode completes with uncoloured words. The import runs in a savepoint, so a bad answer cannot
roll back the clips imported beside it. `diarization.enabled: false` turns it off.

**The declared speaker count is passed as `num_speakers`, as the notebook did.** Left to choose,
pyannote split a 20-minute monologue into 8 speakers, one with 96.5% of the talk and seven
sharing the rest. Given `num_speakers=1`, it returned one. The count is the ingest form's speaker
rows, sent as `speaker_count` and stored in the episode metadata apart from `speakers`. A row whose
gender and age were left blank never reaches `speakers` but is still a voice to find, and counting
`speakers` would tell pyannote too few, which is worse than letting it guess: it has to merge two
people. The form allows eight rows, and `speakers` in the manifest schema was raised from four to
match, since web ingest imports through it. With the speaker section left closed, the episode
declares nothing and pyannote counts for itself, stray speakers and all. Episodes from before
`speaker_count` fall back to the size of `speakers`.

A speaker pyannote is told to find but never hears gets a NaN embedding, which crosses HTTP as
`null`. Such a speaker loses its vector, not the run: the endpoint sends `null` for the row and the
importer drops any vector that is not all finite numbers.

**Checked on three YouTube videos, left to choose:**

| video | declared | found | audio | wall time | turns | overlap |
|---|---|---|---|---|---|---|
| monologue | 1 | 8 (one voice has 96.5% of talk) | 20.3 min | 68 s, cold | 72 | 0.5 s |
| dialogue | 2 | 2 (88% / 12%) | 10.7 min | 22 s | 211 | 1.1 s |
| group | 5 | 5 | 21.7 min | 45 s | 943 | 202 s |

Warm, it takes about 2 s of wall time per minute of audio, upload included, and a cold start adds
about 25-30 s. A 105-minute file (the three concatenated twice, `num_speakers=5`) took 213 s and
found 5 speakers.

**Two things the endpoint needs that are easy to lose:**
- **Proxy auth.** The URL is `<workspace>--<app>-<class>-<method>.modal.run`, easy to guess, and
  an open endpoint would let anyone spend GPU time. The endpoint sets `requires_proxy_auth`, and
  ingest sends a Modal proxy token (`HARNESS_DIARIZATION__AUTH_TOKEN`, env only per D3). Without
  it the call is a 401 and a warning.
- **Redirects.** Modal answers any request still running after 150 s with a 303 to a result URL,
  which blocks for up to another 150 s. httpx does not follow redirects by default, so every
  episode over about 75 minutes would have failed. The client follows up to 20, which is 50
  minutes. The 105-minute check above went through one 303.

**Every call is a cold start, on purpose.** Modal bills a container from start-up to shutdown,
including the idle `scaledown_window` it waits in for another request. Ingest's next request comes
minutes later, after the episode has transcribed, so a warm container is never reused. The window
is 2 s, Modal's minimum, and the GPU stops as soon as it answers.

**Cost.** About 25 s of cold start plus about 2 minutes per hour of audio. At $0.000222/s for the
L4, plus 6-10% for CPU and memory, that is about 3.5 cents per hour of audio. With the old 120 s
window it was about 6.5, and Modal's billing report for the runs above agreed with that. Modal's
free $30 a month covers the whole corpus several times over. Upload is the FLAC, about 2 MB per
minute. Storage is D78's.

**Reversal:** set `diarization.enabled: false`. Episodes then land without turns until someone runs
`scripts/diarize_episode.py`. Delete the
Modal app with `modal app stop nepanglish-diarization`. No schema changed.

## D83 — Fine-tuned models are scored in the harness from the notebook's text, never run in it

A **Models** page lists every fine-tuned model with its gold and val scores and, mainly, the clips
it got wrong, worst first: play the clip, read the folded diff, filter by genre, crosstalk or
loops. The fine-tuning notebook (04c) writes `OUT/harness/`: `model_card.json` plus per-clip
`gold.jsonl` and `val.jsonl` from the best weights. The owner copies that folder to
`data/models/asr/<slug>/`, and `POST /models/rescan` (or `scripts/import_models.py`) imports it.
Three tables hold the result: `asr_models`, `model_eval_runs` (one per imported file) and
`model_eval_clips`.

**The harness scores text; it does not run the model.** The option was CPU inference of an int8
ONNX export on the owner's machine. It was declined for scoring, for two reasons:
- **Speed.** The notebook transcribes gold at ~630x realtime on the A100. The same 2.3 h on an
  8-core CPU is an unmeasured estimate of 10–30 min per run.
- **It is a different model.** Quantization shifts WER by an unmeasured amount, likely near the
  ~0.3-point run-to-run noise. CPU-scored runs would not compare like-for-like with the bf16
  numbers in `docs/findings.md`.

So no inference happens here, nothing is routed through `llm_routes.yaml`, and invariant 6 is not
touched. A future mic playground that does run a model locally needs its own decision.

**Scoring is the notebook's scoring**, so the page and the findings cannot disagree:
- WER is `fold.word_errors`, folded and raw, pooled over words;
- CER is over folded tokens, with Latin lower-cased;
- a loop is a 3-word run repeated 5 or more times (`ftkit.harness_scorer`, `ftkit.is_loop`);
- references are the current labels, normalized exactly as the export normalizes them.

On the 2026-09-12 Flex output this reproduces 11.53% folded / 14.52% raw / 8.07% CER on gold and
7.57% on val.

A run also carries a WER interval from resampling whole episodes, and breakdowns by genre and by
the crosstalk buckets in docs/findings.md. Clicking a breakdown bar filters the clips. The clip's diff is
drawn from the backend's alignment ops, so every mark is one counted error. The editor's raw
`DiffViewer` would mark `टिम`/`team`.

**Separate tables, never `asr_hypotheses`.** A fine-tuned model is not a recogniser of the ingest
pipeline. Its text must not reach disagreement, the queue or an export, and keeping it out by
table is sturdier than a new `asr_systems.kind` every query must remember to filter.

**A run is a snapshot.** Each clip keeps the reference text it was scored against, with the
label's id, and the clip's overlap share at import. Labels are append-only and gold changes by
hand (D71), so a run stays reproducible after a relabel. The page warns when `fold_version` has
changed since import.

**What is refused and what is skipped:**
- **Refused:** the same file twice is a no-op (sha256). A file naming clips the harness does not
  hold is refused whole, because it was made from another dataset.
- **Skipped and counted on the run:** clips that have left the split since the notebook ran, and
  clips whose current label carries no transcript. Nothing is silently scored as gold that is no
  longer gold.

**Cost.** About 1,100 clip rows per model (gold + val), ~1 MB with texts. Import takes ~13 s,
mostly fold alignment.

**Reversal:** drop the three tables (the migration's `downgrade`), `app/api/asr_models.py`, the
`model_*` services, `scripts/import_models.py`, the Models view and 04c's harness-folder cell.
Nothing else reads them.

## D84 — fold-v2: fillers dropped, numbers, contractions and colloquial Nepali folded

`fold.py` was studied against the errors it still charged on the Flex fine-tune's gold and val
output (3,767 errors). Five kinds of respelling were being counted as recognition errors, and
the owner asked for the most aggressive fold of each. `fold-v2` adds them:

| rule (leave-one-out) | gold | val |
|---|---|---|
| **Fillers are dropped** from both texts before alignment (`अँ`, `उम्`, `uh`, `um`, `hmm`, `mhm`...) | −0.59 | −1.26 |
| **Colloquial Nepali** in one form: `हैन`/`होइन`, `भाको`/`भएको`, `गरिराको`/`गरिरहेको`, `गर्दियो`/`गरिदियो`, `गर्या`/`गरेको`, `गर्नुस्`/`गर्नुहोस्`, `संपन्न`/`सम्पन्न`, `मात्रै`/`मात्र`... | −0.75 | −0.23 |
| **Numbers**: digits match the number spelled out in either language (`15`/`पन्ध्र`/`fifteen`, `5,000`/`पाँच हजार`, `2009`/`two thousand nine`, `1st`/`first`); `%` is `percent` | −0.36 | −0.29 |
| **English contractions** match their expansion (`you're`/`you are`, `gonna`/`going to`) | −0.19 | −0.25 |
| **Merges up to three words a side** when the joined words are one spelling or one number | −0.10 | −0.08 |

Flex (2026-09-12 weights, standard decoder): gold **11.53 → 9.65%**, val **7.57 → 5.59%**, gold
CER 8.07 → 7.62% (CER changes only through dropped fillers, quote marks and digit groups). The
recognisers on the same 706 gold clips, measured against the export's references: Scribe 13.11 →
11.40, MAI 11.99 → 10.21, Gemini 11.96 → 9.66. Every system gains 1.7–2.3 points and **Flex's
lead over Gemini disappears** (9.65 against 9.66): part of what fold-v1 charged was noise that
hurt Gemini more.

**Fillers are dropped, not folded.** Folding their spellings into one class gained a third as
much. Dropping follows Whisper's normalizer and NIST's optionally deletable hesitations: a missed
or extra `uh` is not a recognition error. Labels still keep fillers verbatim, and the model is
still trained on them; only the metric ignores them. `हम्` is a filler and Hindi `हम` is not, so
the test runs before the spelling key strips the virama. `उहुँ` and `अहँ` ("no") are answers and
stay.

**Colloquial forms are rules, not a list.** Most are productive (every verb has a `-इरहे-`/`-इरा-`
progressive), so they are rewrites of the spelling key. Each was run over the corpus's 20,777
word types, and every group it joined was read. That audit threw out two rules and narrowed two:
- **Rejected: medial nasal marks.** They join real words: `भाडा`/`भांडा` (rent/utensils),
  `बास`/`बांस`, `आउला`/`आउँला`.
- **Rejected: gemination.** `भन्ने`/`भने` are different words.
- **Narrowed:** `-या` → `-एको` needs two letters before it, because `क्या` ("what") became
  `केको` ("of what").
- **Narrowed:** the benefactive `-इदि-` skips a final `-दिन`, because `गर्दिन` ("I don't do") is
  not `गरिदिन`.

On the Flex output the rules joined 202 pairs; all were read and all were one word.
The emphatic `ै` forms (`मात्रै`, `एकदमै`) are audible, and are in by the owner's choice.

**Numbers need digits on one side.** `one` and `एक` are two languages. A speaker said one of them,
so a number word never matches a number word in the other language. `छ` ("is") matches `6` by
sound, which is the same trade the cross-script rule already makes.

**Wider merges are exact only.** Merging up to 3×3 by sound took gold to 9.86% on its own, and
the merges were junk: `थाहा नै रहेनछ` against `Thane नै रहेछ`. Beyond two words against one, the
joined text must have one spelling key or be one number.

**fold-v1 over-folds, and that was kept.** The study also found fold-v1 hiding real errors:
- **Swallowed neighbours.** A long word absorbs a neighbouring particle or filler inside the
  cross-script slack: `type को`/`type`, `information को`/`information`, `Nepal`/`नेपालको`.
- **Translations and unrelated short words.** The short-word rule matches `and`/`अनि`, `So`/`हो`,
  `You`/`यो`, `to`/`जो`.

A suffix-aware check and an inherent-`a`-only short-word rule removed these, measured at +1.08
gold / +0.81 val. The owner chose to keep fold-v1's leniency, so every fold-v2 number still
includes it. It is the next place to look if the metric reads too kind.

**Cost.** Alignment takes ~40% longer (1,109 clips: 12.3 → 17.5 s), almost all from the wider
merges. The rules live in `fold.py`, not in `config/`, so the dataset's `harness/` copy the notebook
scores with stays one file. Every stored score from before is a fold-v1 number:
- the tables in docs/findings.md;
- the Models page's imported run, which warns, and which a rescan will not rescore, because the
  import dedupes on the file's sha256;
- queue scores computed at ingest.

**Reversal:** revert `fold.py` to fold-v1; nothing is stored under the new rules except numbers
that name `fold-v2`.

## D85 — A mic playground runs a fine-tuned model on the CPU, in a sidecar, logged as a route

The Models page gets a **playground** at the top: hold a button and talk, click to start and stop,
or pick an audio file. The model the page is showing transcribes the recording, and the text
appears under a player for the take. It is for hearing a model at work on a voice it has never
met: a demo and a smell test, never a score. D83 left this out and said a playground that runs a
model locally would need its own decision. This is that decision.

**Where the model runs: a sidecar, not the backend.** D32 and D79 keep torch out of the harness
process, and that holds here. The model runs in `playground/`, a compose service with CPU PyTorch
and nothing else. The backend sends it one HTTP call per recording, as it does for the Modal
diarizer. It first sat behind a compose profile, because a loaded model holds 1.3–2.5 GB of RAM;
since D92 it starts with the stack, because `server.py` loads a model lazily and holds one at a
time, so an idle container is a Python process and nothing more.

**Which weights: 04c's CPU export, copied next to the card.**
- Weight-only int8 goes in `data/models/asr/<slug>/cpu/` when the notebook accepted it (val WER
  within +0.3 of bf16, and no more loops).
- Otherwise the bf16 `best/` goes there.

The card's `cpu.export` says which one. The page shows the playground only when one of them is
there, and says how to add it when neither is. ONNX is not used: plain PyTorch bf16 is already
realtime on the 7700X (docs/findings.md), and dynamic int8 broke Flex.

**Invariant 6 is kept, not bent.** A local model is a provider like the others:
- the route is `playground_transcribe` in `config/llm_routes.yaml`, with `provider: local`;
- the client is `app/llm/local_asr.py`;
- every attempt writes an `llm_requests` row naming the model folder, with its status, latency
  and output, and costs $0;
- `dry_run` returns canned text without calling the sidecar.

A `local` route may not be named `asr*`, so a CPU model can never become an ingest system. A failed
attempt's row is committed before the error goes back to the page (the D66 gap does not repeat
here).

**The recording is prepared as ingest prepares an episode.** The backend runs the stage-1
normalisation on it (two-pass EBU R128 loudness, mono, 16 kHz through soxr), so the model hears the
microphone the way it heard its training audio. The page turns off the browser's own echo
cancellation, noise suppression and gain control for the same reason.

**Decoder: the standard one** (docs/findings.md). Greedy, capped at 13 tokens per second of audio, and a
clip whose output loops is decoded again with a repetition penalty of 1.2 and no repeated 6-gram.
The cap applies to the first pass too: on a CPU a loop run to 300 tokens costs seconds.

**Limits.**
- **30 s per recording.** Flex was fine-tuned on clips of at most 21 s and is not a long-form
  model; the page stops at 29 s.
- **One model in memory.** Switching models reloads it.
- **One request at a time.** A second recording waits for the first.

**Measured on the first model, 2026-09-15** (`indic-transcribe-flex-ft-2026-09-15`):
- **Accuracy.** 04c accepted int8: val 5.54% against 5.58% for bf16, gold 9.58% against 9.54%,
  no loops, and the same text on 337 of 403 val clips.
- **Speed on the 7700X, 8 threads.** A 20 s gold clip decoded in 2.3 s (RTF 0.11). The first call
  after a start adds 5.2 s to load the model. The sidecar holds 1.6 GB.
- **Output.** The playground's text on that clip matched the notebook's GPU text except for one
  word, which is also not in the reference.

**Nothing is stored but the request row.** A recording is not a clip. It gets no segment, no
label and no object, so the corpus, the pots and invariant 7 are untouched. The takes shown on the
page live in the browser tab.

**Reversal:** delete `playground/`, the compose service, the route, `app/llm/local_asr.py`,
`app/services/playground.py`, the endpoint and the page's panel. The `llm_requests` rows stay; they
are the record of what ran.

## D86 — An episode enters only through the web ingest: a YouTube URL or an audio file

There are exactly two ways to add an episode, both on the Ingest page: a YouTube URL
(`POST /ingest/youtube`) and an uploaded audio file (`POST /ingest`). Every stage from loudness
normalisation to the queue runs inside the harness, so no episode arrives partly processed by
something else.

`scripts/import_manifest.py`, which imported an `export_<episode_id>/` directory produced by an
upstream GPU pipeline (D18), is deleted. So is `scripts/import_diarization.py`, which loaded speaker
turns diarized elsewhere (D78); `scripts/diarize_episode.py` re-runs the harness's own Modal
diarizer for an episode that needs it (D79). The notebooks that produced those files are gone
(D18, D79), and an outside producer's segmentation, recognisers, flags or diarizer would each be
one the harness cannot re-run or audit.

**What stays.** The manifest format and its JSON Schemas (D11), `app/services/manifest.py` and
`app/services/importer.py` are the ingest's own last stage: it writes `episode.json` and
`segments.jsonl` into the job directory and imports them, so the schemas still check every row
before it is written. The importer is an internal handoff now, not a door.

**What went with the scripts.** The importer's `dry_run` and `allow_clip_change` options existed
only for the CLI, and ingest never passed either. Both are gone, and so is everything only they
reached: the dry-run report, `ImportReport.render()`, and the branch that replaced an
already-imported clip. A clip whose checksum changed since import is now always refused.

**Reversal:** restore the two scripts, the two importer options and their tests from git, in one
revert of this commit.

## D87 — Every clip has classes, and every model run is split by all of them

Roadmap item 1 goes into the standard pipeline rather than into a one-off analysis. Every clip
falls into one bucket on each axis of `app/services/clip_classes.py`:
- crosstalk, speakers in the clip, turn changes, pause share and clip length, from spans already
  stored;
- audio bandwidth, measured at ingest (`segments.acoustics_jsonb`);
- the clip's voice and that voice's talk time in train, from voices linked across episodes
  (`diarization_runs.voices_jsonb`);
- declared gender and age, where every speaker of the episode shares them;
- CMI, as a descriptive axis.

A model run snapshots each clip's classes and reports WER, CER, share of errors and a
within-episode rate ratio per bucket. It also reports WER and CER per word class: script, number,
code-switch, clip edge. The owner's reasoning: an axis that turns out not to matter is worth
having too, because a ratio whose interval holds 1 is a condition ruled out, and keeping the
axis keeps it ruled out on every later model.

**Why a rate ratio and not bucket WER.** Buckets are filled by different episodes. Crosstalk
lives in podcasts, and podcasts are harder anyway, so pooled WER per bucket measures genre as
much as the bucket. The Mantel-Haenszel ratio compares each episode's clips in the bucket only
with the same episode's clips in the baseline, then pools over episodes. It is the one-axis form
of the crosstalk study's Poisson fit with episode fixed effects (docs/findings.md). It adjusts
for nothing but the episode: two axes that travel together within an episode, such as speakers
and crosstalk, each carry some of the other.

**Why classes never read the reference.** A class computed without the reference exists for
train clips and unlabelled audio too, so training data can be chosen or weighted by it.
Word classes are the exception by nature. They describe reference words, and exist only for
scored runs.

**Voices.** A voice is an anonymous id. A speaker joins a voice when its diarizer embedding is
within cosine 0.6 of the voice's centroid. The closest pairs are matched first, and two speakers
of one run never share a voice. On the 44 episodes, thresholds of 0.5, 0.6 and 0.7 give the same
32 voices, with the recurring hosts the EDA's ECAPA prints found. No name, gender or age is ever
attached to a voice (D56, D58). Declared gender and age reach a clip only when the whole episode
agrees, because diarization cannot say which voice is which declared speaker (D78).

**Not measured: clipping.** The stored audio is resampled to 16 kHz before anything reads it, and
the source is not retained, so a clipping detector would measure the resampler.

**SNR and reverb: Brouhaha, rebuilt without pyannote.** `pyannote/brouhaha` gives speech
probability, speech-to-noise ratio and C50 per 16.875 ms frame. Its checkpoint needs pyannote
3.1, which pins a torch and torchaudio the harness does not use. So
`scripts/export_brouhaha_onnx.py` rebuilds it in plain torch, reading the checkpoint with every
non-torch class stubbed out so none of its own code runs, and exports ONNX. The rebuild matched
pyannote 3.1.1's Brouhaha on every frame of 36 six-second windows of real corpus audio, and the
ONNX matched the rebuild to within 0.003 dB. It runs beside the overlap detector:
- 6 s windows on a 1 s hop, averaged;
- each clip gets the mean SNR and C50 of the frames Brouhaha calls speech;
- ~25 s of CPU per hour of audio, 11.8 minutes for the corpus.

The model is gated and OpenRAIL, so the export is never fetched or published. It lives in the
gitignored `data/models/`, and without it a clip is measured for bandwidth alone and stamped
`acoustics-v2-bandwidth-only`, which a backfill with the model completes. A backfill never
measures less than it finds.

**Cost.** Bandwidth is 20 s of CPU for the whole corpus, Brouhaha 12 minutes. Linking voices is milliseconds.
Reclassifying the two Flex runs takes ~18 s, mostly the ratio bootstraps. Three nullable JSONB
columns.

**Reversal:** drop the three columns (`segments.acoustics_jsonb`, `diarization_runs.voices_jsonb`,
`model_eval_clips.classes_jsonb`) with their migrations. Revert the class, acoustics and voice
services and scripts. Restore `by_overlap` in `model_eval.summarize` and the crosstalk panel on
the Models page.

## D88 — Episodes ingest a few at a time, behind one rate-limit gate per service

Up to `ingest.max_concurrent_jobs` episodes (default 3) now run at once. They start in submission
order. Every call to a rate-limited service goes through that service's one process-wide gate
(`app/utils/rate_limit.py`), whichever job makes it. Before this, jobs ran one at a time.

**Why.** Short episodes spent about 3 minutes each in the queue, and nearly all of it was waiting on
remote services. One fusion call alone is 2–4 minutes of Gemini thinking (median 210 s over 107
calls). Running jobs one at a time protected the ASR rate limits, but by idling the machine.
Nothing coordinated the requests of two jobs, and a 429 cost its clip (D46): the client retried
three times over 1.5 s, then gave up. A queue of 83 two-minute gold files would have taken about 4
hours.

**What a gate does.**
- **In-flight cap.** At most `max_in_flight` requests are open at once, summed over every job.
- **Spacing.** Starts are at least `min_interval_seconds` apart.
- **Shared cooldown.** A 429 starts a cooldown that every caller waits out. Its length is the
  `Retry-After` (or OpenRouter's `X-RateLimit-Reset`) if the service sends one. Otherwise it starts
  at `cooldown_seconds`, doubles for each further cooldown without a success between, and is capped
  at `max_cooldown_seconds`.
- **Adaptive limit.** The same refusal halves the working limit. Successes win it back one slot per
  `allowed` successes, i.e. additive increase, multiplicative decrease. Refusals that land inside a
  running cooldown count once, so twenty requests refused together are one signal.
- **Retries.** A 429 does not spend `max_retries`. The request waits out the cooldown and tries
  again, up to `rate_limit_retries` times (default 8, about 10 minutes of backoff). Only after that
  is it a failure, and so a discard. Latency in `llm_requests` is time spent in requests, excluding
  gate waits.

**Limits.** They live in `llm_routes.yaml` under `limits`, per provider: openrouter 16, elevenlabs 8,
vertex 12. Gemini ASR and fusion share the Vertex gate because they share the model's quota. The 8
for ElevenLabs is what one ingest already sent without a 429 in 7,905 calls. YouTube has its own
gate in `ingest.youtube.limit`: one download at a time, starts 5 s apart. A bot check on a download
or a lookup cools every later download down for 10 minutes. The job that hit it is backlogged as
before, and the ones behind it wait rather than trip the check again. Modal diarization is not
gated; it is at most one call per running job.

**Import still takes turns.** Stage 6 writes corpus-wide rows (voices, systems, the queue), so it runs
under one lock. The diarizer's answer is awaited before the lock is taken, so a slow GPU never holds
up another job's import.

**Fixed on the way.** A clip whose every route failed was discarded without committing, so the
`llm_requests` rows of its failed attempts were rolled back. An expired key left no trace of a
single refused call (invariant 6). The discard path now commits.

**Cost.** No extra requests: the same calls, overlapping in time. Peak CPU and disk rise with the
number of jobs, since each runs its own ONNX models and work directory (~350 MB of models, well
within 27 GB).

**Reversal:** set `max_concurrent_jobs: 1`. The gates are harmless on their own and can stay.

## D89 — fold-v3: every colloquial Nepali form folded, grouped by kind

Roadmap item 6 asked for a spelling convention for spoken Nepali: the references write a
colloquial form (`गको`, `गरिराछ`, `उल्ले`) one day and the standard form the next. The error
mining listed ten kinds of such form. There is no accepted normalizer for Nepali to follow, so the
owner chose to **fold all ten** and let a listening check by native speakers decide what to
tighten. `fold-v3` does that. It is still a comparison: no label or hypothesis changes.

**The rules are grouped by kind in `fold.py`**, one table and one rule tuple per group, applied in
the order of `_COLLOQUIAL_GROUPS`:

| group | examples |
|---|---|
| 1. contracted verb forms | `भाको`/`भएको`, `गको`/`गएको`, `भाछ`/`भएछ`, `थ्यो`/`थियो`, `थेँ`/`थिएँ`, `थिएन`/`थिइनँ`, `राखीछु`/`राखेछु` |
| 2. the western `-या` participle | `गर्या`/`गरेको`, `गर्न्या`/`गर्ने` |
| 3. the progressive | `गरिरा`/`गरिरहेको`, `भइराथ्यो`/`भइरहन्थ्यो` |
| 4. the benefactive | `गर्दिया`/`गरिदिए`, `छोड्देऊ`/`छोडिदेऊ`, `भन्दिहाल्छु`/`भनिदिइहाल्छु` |
| 5. first person plural | `भनुम्`, `भनम्`, `भनूँ`/`भनौँ`; `गरेम्`/`गर्यौँ`; `जाम्`, `जाऔँ`/`जाऊँ` |
| 6. contracted pronouns | `उल्ले`/`उसले`, `तेल्लाई`/`त्यसलाई`, `जोले`/`जसले` |
| 7. `लाउनु` for `लगाउनु` | `लाएर`/`लगाएर`, `लाउँछ`/`लगाउँछ` |
| 8. emphatic `-ै`, doubled consonants | `मै`/`मा`, `अझै`/`अझ`, `लिएरै`/`लिएर`, `सक्केसम्म`/`सकेसम्म`, `खत्रा`/`खतरा` |
| 9. loose pairs | `नि`/`पनि`, `गर्न`/`गर्नु`, `भन्दाखेरि`/`भन्दा`, `अलिकति`/`अलि`, `या`/`यहाँ`, `छुइनँ`/`छैन`, `खाइयो`/`खायो` |
| 10. forms not yet in the corpus | `जान्न`, `भन्नि`, `भनेसि`, `गर्चु` |

A last group, **joined**, lets one spoken word match two written ones through a merge:
`भाथ्यो`/`भएको थियो`, `गरिराछ`/`गरिरहेको छ`, `भाछैन`/`भएको छैन`, `गरेनि`/`गरे पनि`.

**Two fold-v2 bugs are fixed on the way.** The progressive rule ran before the `-या` rule, so
`गरिराख्या` never met `गरिराखेको`. A table word lost its fold under a case ending: `रुपियाँ` was
folded, `रुपियाँको` and `पहिलादेखि` were not. Table words now keep their fold under the common
endings.

**Two fold-v2 refusals are reversed by the owner's choice:** `गरूँ`/`गरौँ` ("let me" / "let's")
and `अलिकति`/`अलि` are now one word each. Group 9 as a whole joins words that are sometimes
different: `नि` is also a particle, `या` is also "or", `गरियो` is a passive, `दिन` is also "day".

**Audit.** As in D84, every rule was run over the corpus vocabulary (14,000 word types, plus the
fine-tune's output) and all 380 groups it newly joined were read. Two were narrowed:
- **The infinitive** skips the benefactive, so `गर्दिनु` ("to do for") does not meet `गर्दिन` ("I
  don't do").
- **`-ुँ` → `-ौँ`** needs two letters before it, so `हुँ` ("I am") does not meet `हौँ`.

One wrong join is kept: `जान्न` ("I don't know") meets `जाँदिनँ` ("I don't go"), because the D84
nasal rule already makes `जान्दिनँ` and `जाँदिनँ` one key.

**Measured** on the 2026-09-16 fine-tune and the recognisers, same references:

| system | gold fold-v2 | gold fold-v3 | gain |
|---|---|---|---|
| Flex fine-tune | 6.58 | 6.25 | 0.33 |
| Scribe | 7.08 | 6.68 | 0.40 |
| Gemini | 6.38 | 6.16 | 0.23 |
| MAI | 6.00 | 5.81 | 0.19 |

Flex val: 12.59 → 12.05. The gain is small, about a sixth of the 2.02-point ceiling item 6
estimated. Most of that ceiling is grammar (person, tense, case suffixes), which no convention
removes. **The fine-tune is not the system that gains most**, so the rules were not fitted to its
errors, although they were chosen by reading them.

**Reversal:** revert the groups to fold-v2's rules. Every number stored under fold-v2 stays valid
under its own version name.

## D90 — Val is redrawn by stratum: long and short form, capped, one podcast per show

Val was drawn by hashing each episode against `val_fraction`. With 44 episodes, from 6-minute
reviews to 2-hour roundtables, the hash gave val 3.94 h (larger than gold's 2.17 h), 849 of its 921
clips from two podcasts, and one roundtable carried 83% of val's errors. Epochs were being picked
on one recording's crosstalk, and val took longer to decode than gold.

**The draw (`pots.draw_val`, run by `scripts/redraw_val.py`).**
- **Two strata.** Long-form episodes (at least 0.5 h outside gold) and short-form ones are drawn
  separately. Each gives `val_fraction` of its hours to val, so val has the same mix as train.
- **Hash order, capped.** Within a stratum, episodes are taken in `hash(episode, seed)` order while
  val stays within 125% of the target.
- **No dominant recording.** An episode longer than half of its stratum's target is never taken.
- **One podcast per show**, so val's long-form hosts differ.
- **Gold-only episodes are left alone**: their split exports nothing.
- Every changed episode gets an `audit_logs` row (`split_changed`).

**Result, seed 20260917.** Val 2.18 h over 6 episodes: 2 podcasts from different shows
(`on_air_with_sanjay_814`, `the_bravo_delta_show s3e2`) and 4 reviews. That is 635 clips, 20% with
crosstalk (train 27%). Train is 22.30 h. The roundtable is back in train. New imports still draw by
hash; rerun the script to rebalance.

**Cost.** Val numbers from before 2026-09-17 are not comparable with later ones. Val holds only two
podcasts, the most whole episodes the corpus allows at 10%.

**Reversal:** restore the splits from `data/backups/harness_2026-09-17_pre_val_redraw.dump`, or
reverse the `split_changed` audit rows.

## D91 — The corpus page counts clips and voices, not episodes, and advises per category

`app/services/inventory/` is rewritten and the analytics page is replaced. The unit of every
count is the **clip**, and the person is the **voice**. Every clip carries one bucket on each of
sixteen categories — gender, age, role, voice and its exposure in train; topic, genre, show and
code-mixing; speaking speed, clip length, crosstalk and speakers in the clip; noise (SNR), room
(C50) and bandwidth — so hours cut by any category sum to the corpus. The payload also ships the
clip table itself (one compact row per clip, a few hundred kilobytes), and the page re-cuts every
card client-side when a bucket, a cross-tab cell, a recommendation or a voice is clicked.

**Why the episode attribution of D69 went.** It existed because no route diarized, so an episode
with a male host and a female guest had to count its whole duration on both sides. Every episode
is diarized now (D79) and its speakers are linked into anonymous voices across episodes (D87), so
a clip has one dominant voice and the voice has one gender where the episode's rows force it.
Shares sum to one, the gender-by-age grid is one cell of a general cross-tab, and "how many
people" is a count of voices rather than a floor over `(show, role, gender, age)` tuples.

**Declared rows reach voices only where the episode forces the match** (`resolve.py`): one row and
one voice; one declared host and exactly one recurring voice; or every remaining row agreeing on a
field, and never when more voices than rows remain. Gender and age travel with the voice to every
episode it appears in; two episodes that disagree make a `conflict`, reported rather than
averaged. Role stays per episode. These are the three rules the sociolinguistics notebook has
applied by hand since 2026-09-18, moved into the harness so the page and the paper agree; on the
2026-09-18 corpus they reach the same 30 of 59 usable voices with gender and 24 with age. Nothing
is inferred from audio (D58) and no name exists to link to (D56).

**Speaking speed** is reference words per second of VAD speech, from the current label's text
where a clip has one and the fused seed otherwise. It is computed here rather than in
`clip_classes.py`, whose axes never read the reference (D87). Edges 2.0 / 2.6 / 3.2 / 3.8 sit
near the 10th, 30th, 70th and 93rd percentiles of the corpus (median 2.9 w/s).

**Advice is per category and per purpose.** Each recommendation names the category and bucket it
came from and whether it hurts the recogniser, the paper or both, so the page can be read for one
purpose at a time. A bucket is thin in its own unit: `dataset.min_stratum_voices` (5, new) for
people and content, where the claim is about speakers, and `dataset.min_stratum_hours` for speech
and acoustic conditions. A voice counts toward a bucket once it has 300 reference words attributed
inside it, the notebook's floor for a usable voice. Gold is asked for only on speech and acoustic
conditions: gold holds voices train never sees (D76), so a people bucket with no gold is the
design. A paper bucket resting on screened labels is asked to verify a sample, because the mixing
variable is the script choice the fused seed made (docs/sociolinguistics.md). Shows are not asked
for more speakers; a show is where audio comes from, not a stratum anyone fills. The diarizer's
"nobody heard" clip is a defect, not a condition, and is excluded from every rule.

**Voices are followed.** One profile per voice: talk time inside the corpus's clips, attributed
words, episodes with minutes and share and the other voices in the room, shows, roles, hours by
pot, verified share, mean CMI, speaking rate, first and last date. The page draws each voice's
episodes as a strip and lets a co-voice be followed with a click. The recurrence the accommodation
design needs — hosts with five or more host–guest episodes, guests who appear with more than one
host — is measured and asked for; a monologue host in twenty episodes accommodates nobody.

**Not built: the manual voice-to-row link.** docs/sociolinguistics.md ranks it first among the
metadata worth adding. It needs a listen-and-click UI and a stored link on `diarization_runs`;
the forced rules above cover the podcasts whose structure decides it, and the page now shows,
per voice, that the rest are unresolved rather than absent.

**Cost.** One request builds everything in under a second on 7.6k clips; the payload is ~850 KB.
`/stats/report` is unchanged and no longer read by the page. The old panels are deleted, not
hidden.

**Reversal:** restore `app/services/inventory/` and `frontend/src/components/analytics/` from
before this commit; drop `dataset.min_stratum_voices`. No schema change was made.

## D93 — An interrupted ingest resumes by itself and never pays twice

A power cut on 2026-09-21 stopped three running jobs and stranded three queued ones. Postgres
recovered cleanly and nothing was half-imported: an episode is held in memory until stage 6 writes
it in one transaction. But the two episodes in flight lost every transcript and fusion window they
had paid for, and the queue could not be continued. Jobs saved as `pending` came back in no
category: not listed, not retryable, never run. The saved state was also read only on the next
submission, so after a restart the Ingest page showed nothing.

**What changed.**
- **Paid results are checkpointed** in the job's work directory as they arrive
  (`app/services/ingest/checkpoint.py`). This covers transcripts, fusion requests, the Modal
  diarization, and a marker for a finished download. Each entry is keyed by exactly what was sent,
  so a changed clip, route or prompt is a miss, paid for again, never a stale answer.
  - Transcripts are keyed on the clip's decoded samples, not its bytes. Two FFmpeg versions
    normalising one source gave different files and identical samples.
  - Fusion is keyed on the request's messages. Windows run in order, each carrying the previous
    answer forward, so a resumed run rebuilds the same requests up to the first one never
    answered.
- **Fusion's `llm_requests` rows are committed per request**, not at the end of the stage. A cut
  mid-fusion used to lose the record of every window already bought (invariant 6). A result is
  checkpointed only after its row is committed, so nothing is ever replayed that is not on record.
- **Writes are durable.** Checkpoint entries and `queue_state.json` are fsynced before they are
  renamed into place, and the directory after. A torn entry reads as a miss.
- **Interrupted jobs resume at startup** (`ingest.resume_interrupted`, default on). Running jobs
  come first, then the queue, in their old order. A job whose episode is already in the database
  is marked complete instead. With the setting off, the jobs come back failed and retryable. The
  Ingest API also loads the saved state on its first request.

**Why automatic.** The work was already asked for, and the checkpoint caps what a resume can
spend at what the dead run had not yet bought. Needing a person to notice and click retry is the
failure this is meant to remove.

**Not covered.** A replay writes no `llm_requests` row and reports zero cost; the first call's row
stands. A call cut off after the vendor answered but before its row was committed is paid and lost;
that window is a request's round trip. Topic classification ($0.002) is not checkpointed. A job
that fails normally still deletes its work directory, checkpoint included.

**Cost.** One small JSON file per paid result, gone when the job finishes. Hashing the samples
measured 1.8 ms per clip, and 0.58 s per hour of episode for the diarization key.

**Reversal:** set `resume_interrupted: false` to stop resuming. Removing the checkpoint means
reverting `checkpoint.py` and its three call sites in `pipeline.py` and `fusion.py`.

## D94 — Clips can be deleted in bulk, and never from gold

`POST /segments/bulk-delete` takes a list of segment ids and deletes them in one transaction,
all or none, from "Delete selected" on the triage toolbar, which acts on the rows already ticked
for bulk accept. Each clip leaves an `audit_logs` row (`action: delete`,
`old_values_jsonb.bulk: true`, with its pot) that outlives it. Storage
objects are removed only after the commit, so a refused batch leaves every clip playable. A gold
clip anywhere in the batch refuses the whole batch with 409, and the button stays disabled while
one is selected.

**Why.** Single-clip delete already existed (`DELETE /segments/{id}`, in the editor and on each
row), and D63's "deletion is not offered" was not true of the code. Removing junk clips one dialog
at a time was the owner's actual workflow, so this overrides that paragraph of D63.

**What it costs.** D63's two warnings still hold for anything deleted this way. A deleted clip is
missing from the untranscribable-audio count, so bad audio should still be flagged
`unusable_audio`. Deleting is also a filter applied to train and not to gold. The audit rows are
what make that filter countable after the fact. Refusing gold keeps the benchmark out of it.

**Reversal:** delete the endpoint, its schema and the triage button. Nothing persists apart from the
audit rows, and the deleted clips do not come back.

## D95 — Synthetic crosstalk is mixed on the fly, labelled as the clip's own speaker, and cut from the same room

The fine-tune notebook (`notebooks/src/xtalk.py`, on when `XTALK_P > 0`) gives a fresh 30% of the
train clips measured clean short bursts of another voice every epoch, in the DataLoader workers.
Each mixed clip keeps its label unchanged.

- **The label is the clip's own speaker.** The references in real overlap keep what was audible:
  usually the main voice, sometimes the other one, sometimes both. The owner accepts any of these
  as long as nothing is written that was not said. A fixed target always takes one acceptable
  option, the main voice. Writing the other voice's words would need them transcribed, and the only
  text for a donor excerpt is an unverified recogniser's, which would teach invented words. Flex's
  decoder has no speaker-attributed output. Measured on 8,406 synthetic bursts against Scribe's
  word spans (2026-09-22): 58% hold only word fragments, 37% at least one whole word, and 6% no
  word, so most of what the model learns to leave out could not be written as a word anyway.
- **Measured shape, not LibriMix defaults** (findings.md, *Overlap windows*). Window durations
  come from the measured deciles, with a median of 0.42 s. The level gap is within ±3 dB (median
  |gap| 1.6), and either voice can be louder. The model cannot learn "drop the quieter voice".
  It has to learn "follow the voice that holds the clip".
- **The other voice comes from the same episode** when it has another voice with a solo stretch
  long enough. Same room, microphone and level, so the channel gives nothing away. Otherwise it
  comes from another train episode. On the 2026-09-21 export, 55% of windows come from the same
  episode: 27 of the 52 train episodes have one voice.
- **Weighted toward the error buckets.** An augmented clip draws its overlapped share with weights
  40/30/30 over 1–5%, 5–15% and 15–40%. Realised on the export, the split is 30/34/36. Real overlap
  is 3% of audio, but the errors sit above 5%.
- **What is never touched.** Clips with real crosstalk keep it, and unmeasured or undiarized clips
  are not assumed clean. Donors come only from train clips' solo speech: `DonorPool` raises on a
  val or gold row (D76). Val and gold are never mixed.

**Why on the fly.** It needs no new export and no new HF upload. Every epoch sees different mixes,
it costs 2.7 ms of worker CPU per clip, and `XTALK_P = 0` gives the old run back exactly.

**How it is judged:** by the sweep in D96. Gold WER in the crosstalk buckets also charges
acceptable choices (a label that kept the other voice counts its omission as deletions), so the
S/D/I split is read with it, and the deletions on clips without overlap are the check that the
model has not learnt to drop its own speaker's short words.

**Reversal:** set `XTALK_P = 0`. To remove it entirely, delete `xtalk.py`, its cell and the
mixer lines in `collate`.

## D96 — XTALK_P is swept on one export, and the kept weights are chosen on val by a rule fixed in advance

The fine-tune notebook trains one model per `(XTALK_P, seed)` point in `SWEEP`:
`(0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.5, 0), (0, 1)`. Every point starts from the same base
weights, on the same export, with the same batch budget, and is scored on val and gold overall
and per crosstalk bucket, with S/D/I. The 2026-09-17 model is decoded on the same val and gold
under the same decoder. The results are meant for a paper, so what may be claimed is fixed here,
before any point has run.

- **The winner is chosen on val alone** (`sweep.choose_winner`). p = 0 is its better seed. The
  noise is the gap between the two p = 0 seeds. The best augmented run is kept only if it beats
  p = 0 on val by more than the noise; otherwise p = 0 is kept, as the simpler recipe. Gold never
  takes part, so gold stays a held-out score for every point. Choosing on gold would bias every
  gold number reported.
- **Two seeds at p = 0.** One run per point cannot tell an effect of p from a different random
  run. The only earlier estimate (09-16 against 09-17, +0.68 [−0.04, +1.82]) also changed the fold
  and the val draw. The second seed also reorders the batches: `make_batches` seeds its shuffle
  with `epoch + 1000 * seed`, so seed 0 keeps the order of every earlier run.
- **Up to 0.5.** 73% of train clips are eligible to be mixed. At p = 0.5, about 36% of each epoch
  is synthetic crosstalk, on top of the 27% of clips with real crosstalk, against about 3% of the
  audio in reality.
- **Gold comparisons are paired and resampled by episode** (`sweep.paired_bootstrap`). Clips of
  one episode share a room and voices, so resampling clips would overstate how much gold knows.
- **What the added data did** is p = 0 seed 0 against the 09-17 model, paired on gold. The export
  grew from 7,052 to 9,774 clips. Val grew with the new episodes too, which changes which epoch
  early stopping picks, so the gold pair is the reading.
- **Only the winner keeps its weights.** Every point uploads its metrics, transcripts, per-clip
  counts and harness folder as it finishes, so all of them can go on the Models page and a
  dropped session resumes. `best/` stays on the VM until the winner is known, and is uploaded and
  exported for the CPU only for the winner.
- **Crosstalk is a secondary analysis, not the paper's claim.** The model card says the model is
  trained for single-speaker recordings. Overlapped speech proper (cpWER, tcpWER, ORC-WER,
  diarization-conditioned models) is left to separate work.

**Run 2026-09-22: cut short, rule not applied.** Only p = 0, 0.1, 0.2 and 0.5 ran (seed 0), so
there was no seed noise to apply the rule with. No winner was chosen and no weights were kept.
Results are in findings.md.

**Reversal:** set `SWEEP` to one point, which runs a plain single fine-tune. Delete `sweep.py`, its
tests and the sweep cells to remove it entirely.

## D97 — A commit runs lint and the not-db suite; the full suite gates the push
`.githooks/pre-commit` always runs ruff, and runs `pytest -m "not db"` only when the staged diff
touches `backend/`, `scripts/`, `config/`, `notebooks/src/` or `docker-compose.yml` — the paths the
suite can see (`notebooks/src` is loaded directly by `test_sweep.py` and `test_crosstalk_mixer.py`).
`.githooks/pre-push` always runs the full suite, `db` tests included. A commit is therefore not
proof that the schema and API layers still hold; a push is. **Reversal:** trivial — folding the full
suite back into pre-commit is a one-line change to one short script.

## D98 — Per-speaker labels are words on fixed speaker lanes, stored as their own label version
> **Stopped by D100** (2026-09-24): the speakers queue is closed. The editor and the five saved labels are kept.

Roadmap A, as the owner designed it on 2026-09-23. A clip in the `speakers` queue opens in the
multitrack editor: a lane for each of the episode's diarized speakers, a block for each word at
the time it was said, on a 10 ms grid. The annotator moves words between lanes and adds the words
the transcript lacks. Nothing else.

- **Where the blocks start.** The clip's verified single-stream label, put on the fused seed's
  word spans by a plain (unfolded) word alignment: a matched or substituted word takes that
  word's span, a word the seed lacks gets none and must be placed by hand. Folding is wrong here:
  it merges a word with its neighbour at no cost, which stretches spans across two words. Each
  word starts on the speaker whose turns cover most of it (not its midpoint, which in crosstalk is
  a coin toss); a word no turn touches starts on no lane. Without a verified label the seed is
  used; an earlier per-speaker label of the same run is reopened as saved.
- **Speakers are fixed.** Lanes are the current diarization run's speakers, so the editor cannot
  create or delete one. A speaker the diarizer invented is an empty lane. Two people it merged
  cannot be split here: the clip is flagged `uncertain` with the note "diarizer merged two
  voices", for an episode-level fix.
- **Recogniser candidates.** A word at least two recognisers heard, at a moment the verified text
  has none and with no same word in the text within 0.6 s, is offered in a bottom row to drag onto
  a lane. With one recogniser enough, the first clip offered 49 candidates for 52 words; with two,
  the 30 queued clips offer 41 for 1,174.
- **Storage.** A save is one `segment_labels` row in the `speakers-v1` label version, one
  `label_words` row per word (span, the run's raw speaker label, the proposed speaker, and where
  the word came from), and the usual event and audit rows, in one transaction (invariants 2 and
  8). `diarization_run_id` names the run; a save against a run that has since been replaced is
  refused. It is always `verified`: who spoke in crosstalk cannot be screened (invariant 5). The
  disposition is `accepted_unchanged` when the lanes are saved exactly as served, `edited`
  otherwise. No new status field (invariant 3).
- **It is not a single-stream label.** `latest_label` and `latest_labels_subquery` leave the
  `speakers-v1` version out, so stats, reports, pots, the inventory and the export still count and
  export one label per clip. A speakers-queue decision never changes `pipeline_status`, and an
  accept or text label on a speakers task is refused: the lanes are saved only through
  `/tasks/{id}/attribute`. The flat text of the lanes (every word in time order, a shared word
  twice) is kept in `final_text` for reading, not for scoring.
- **Which clips.** `scripts/queue_speakers.py` ranks labelled clips of a pot with two or more
  diarized speakers by the overlap detector's crosstalk share. On 2026-09-23, 288 gold clips
  qualified; the 30 with the most crosstalk (29–50%) were queued for the owner to try the design.
  Later that day the owner asked for every gold clip with any measured crosstalk, to judge the
  diarizer on easy clips as well (sorting by crosstalk in triage): 261 in all. The script takes a
  crosstalk band and `--min-voices 1` for clips the diarizer heard as one voice.
- **A save is undone by reopening, never by deleting.** `queue_speakers.py --reopen` puts a saved
  clip back in the queue with a `reopen` event and an audit row; the editor opens from the saved
  lanes, and the next save supersedes them.
  The roadmap asked for a throwaway prototype before any harness build; the owner asked for it in
  the harness instead, so these 30 clips are the pilot, and words moved, words added and time per
  clip come from its `label_words` and `annotation_events` rows.

D95 is untouched: synthetic crosstalk in training still keeps the clip's own label.

**Reversal:** moderate. `alembic downgrade` to `a6d0f4c93e18` deletes the speakers tasks, the
`speakers-v1` labels and their words; then delete the two services, the editor, the endpoint and
the queue script.

## D99 — Voices get a page, confirmed stretches become their print, and prints only suggest
> **Dormant under D100**: suggestions were served only to the speakers queue, which is closed.

The owner asked, from the multitrack editor: who is v125, and can clean clips of a voice fix the
diarizer's mistakes about it? Measured first (findings.md, *Voiceprints on clean speech and in
crosstalk*); built on what the measurement allows.

- **Playback speed** goes down to 0.25× and is remembered in the browser across clips, in both
  editors. A preference of one viewer, so browser storage rather than the database.
- **A voice page.** Every voice (D87's anonymous id) has a page, opened from a lane's voice id in
  the multitrack editor or from the Corpus voices table: the episodes it speaks in and, for each
  clip it speaks in, its longest stretch alone (its turns, less every other speaker's turns and
  every detected overlap span), 1.5 s or longer, longest first. Whole solo clips were tried first
  and left 55 of 197 voices with nothing to hear; stretches leave 10. Each lane also has a button
  that plays the voice's best stretch without leaving the editor. Still no name (D56): knowing
  what a voice sounds like is the point, not who it is.
- **Confirmations.** The owner marks a played stretch *only this voice*, *not only*, or takes the
  verdict back: `voice_confirmations`, append-only, newest per (voice, segment) current, with the
  stretch and an audit row. The stretch is recomputed by the server, never taken from the client.
  A verdict is a judgement like a label, not a status of any row, so invariant 3 is untouched.
- **Prints.** WeSpeaker ResNet34-LM as ONNX, with a numpy Kaldi fbank (torch never enters the
  backend), downloads itself pinned and digest-checked like the aligner and the overlap detector;
  without it there are no suggestions and nothing fails. It shares pyannote's embedding space, so
  a voice's print is the mean of its confirmed stretches' embeddings, across episodes, and the
  diarizer's stored centroid until it has one.
- **Suggestions, not corrections.** When a speakers-queue clip is served, each word no second
  voice is heard over (one diarized turn at its middle, no detected overlap) is embedded over 1 s
  and scored against the prints of the clip's speakers. When another speaker's print is closer by
  0.1 or more, the block is marked with that speaker; `V` takes one suggestion, `Shift+V` all of
  them. Nothing moves by itself. Words in crosstalk get no suggestion: there the print follows the
  louder voice and is confident when wrong.
- **Scored before trusted.** A saved word keeps the suggestion it was served with
  (`label_words.suggested_speaker`). `scripts/voiceprint_report.py` reports how many suggested
  words the owner left on the suggested lane, and how many of the owner's moves a suggestion
  foresaw. Suggestions may start moving words by themselves only on a new decision entry that
  cites that report.

**What this does not do:** re-diarize an episode with the prints, or attribute words inside
crosstalk. The first is possible (constrained clustering on the Modal GPU, D79) and is the next
step if the report shows the diarizer's clean-speech errors are what costs time; the second needs
a model that separates or follows a target voice (roadmap D).

**Reversal:** cheap. `alembic downgrade d98a1c5e7f20` drops the confirmations and the stored
suggestions; delete `voiceprint.py`, `voice_clips.py`, the voices router, `VoiceDialog.tsx` and the
editor's suggestion code.

## D100 — Speaker turns are automatic metadata, not a source of truth; per-speaker labelling stops

On 2026-09-24 the owner listened to the diarizer's word attribution, with every word coloured by
its speaker (findings.md, *Diarization cannot say who said a word*), and ruled that word-level
attribution is unusable and speaker turns are hit or miss. The measurements agree:
- A speaker count does not fix it. A ceiling merges real second voices as often as the exact
  count split one, and on a two-person interview "told nothing" moves 7% of words between the
  same two voices.
- In that interview, 17% of words took their speaker from the join's tie rule, because both
  turns covered the whole word.

- **Gold is single-stream: everything said, in time order.** A clip's label is what was said,
  whoever said it. Overlapped stretches are marked by the overlap detector's spans (D77), which
  the export already carries, and not by speaker. Labels written before this, some keeping only
  the stronger voice, are not re-audited.
- **Speaker turns stay, described as what they are.** Ingest still diarizes (D79), and the
  export still carries each clip's turns and linked voices (D78, D87). They are automatic and
  cannot be trusted, neither for which speaker said a word nor for where a turn changes: changes
  in pace, excitement and loudness read as a change of speaker. They colour the editors and
  nothing more. They are never a reference, and the dataset is not described as diarized. A speaker count in a report comes from the counts declared
  at ingest, not from voice ids. Short videos were often declared as 3 by default, which a report
  must say.
- **The speakers queue is closed.** Its 257 open tasks were skipped on 2026-09-24 by
  `d100-close-speakers-queue`, each with its event and audit row. The five saved `speakers-v1`
  labels stay (labels are append-only). The code of D98 and D99 is kept on purpose (the owner's
  call): if roadmap F yields turns that can be trusted, the multitrack editor is what uses them.
  Until then, do not run `scripts/queue_speakers.py`.
- **Overlap models are on hold.** The owner judged roadmap D out of scope for the code-switching
  paper, and put it, with the custom architectures, after an experiment with the diarizer itself
  (roadmap F), since both are conditioned on diarization. If they resume, they are developed and selected on synthetic mixes, where attribution is
  known by construction (roadmap C1). They are checked on real crosstalk last: recognition against
  the single-stream gold, and attribution by grading a model's output by ear rather than labelling
  references.

**Reversal:** cheap. Re-queue clips with `scripts/queue_speakers.py`; nothing was deleted.
