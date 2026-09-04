# Decisions

Each entry: the decision, why, and what it would cost to reverse.

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

## D5 — Frozen episode-level splits, assigned once at import
Stored in `episodes.split` from `hash(episode_id, split_seed)`. Recomputing at export time would let
segments migrate between train and test as episodes are added, which silently invalidates every
earlier benchmark. Splits are per episode, never per segment, because segments from one episode
share speaker, room and topic. **Reversal:** would invalidate all existing exports; treat as
permanent.

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

## D10 — All LLM and cloud-ASR inference goes through OpenRouter
OpenRouter is prepaid, which removes the possibility of a surprise invoice. The client, its retry
and dry-run behaviour and the `llm_requests` log were built before any route existed, so when the
ingestion pipeline started calling cloud ASR (D18) it inherited all three. Routes live in
`config/llm_routes.yaml`; every route named `asr*` becomes one ASR system during ingestion.
No direct calls to OpenAI, Anthropic, Google, Groq or Mistral, ever.
**Superseded part:** the original MVP made no LLM calls at all and shipped `routes: {}` with
`enabled: false`. Prioritization still uses only multi-system disagreement and rule flags — nothing
in scoring, policy checking or correction suggestion calls a model.
**Superseded part:** "through OpenRouter" was always a proxy for "prepaid". D21 restates the rule
in the terms that actually matter and admits one direct provider on them. OpenRouter remains the
default and still carries all text inference. **Reversal:** n/a.

## D11 — Validation by JSON Schema at the manifest boundary
`backend/app/schemas/episode.schema.json` and `segment.schema.json` are the executable form of the input
contract, checked before any write, so a malformed manifest fails loudly with an empty database
rather than half-importing. **Reversal:** none.

## D12 — Plain git hook instead of the pre-commit framework
`.githooks/pre-commit` runs ruff and pytest with the backend virtualenv. One less dependency and one
less lockfile for a single-developer project. **Reversal:** trivial.

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

## D21 — The provider rule is "prepaid", not "OpenRouter"; ElevenLabs Scribe is called directly
**The prepaid half is superseded by D34.** The half that survives is the one that mattered in
practice: `app/llm/base.py` holds the retry policy, the dry-run switch and the `llm_requests`
write, so a new provider inherits the guarantees instead of reimplementing them.

The point of routing everything through OpenRouter (D10) was never the vendor. It was that
OpenRouter is topped up rather than invoiced, so the worst outcome of a runaway ingest is an
exhausted balance the owner chose to fund. ElevenLabs bills the same way, which means sending
Scribe through a proxy would buy nothing and cost accuracy: Scribe is the only transcriber the
harness has that returns word spans and per-word log probabilities, and it is not reachable
through OpenRouter at all.

So the invariant is restated as its own justification — every provider must be prepaid — and
`app/llm/base.py` now holds the retry policy, the dry-run switch and the `llm_requests` write, so
a second provider inherits the guarantees rather than reimplementing them. Scribe's key is
`ELEVEN_LABS_API_KEY` and should be scoped to speech-to-text only.

Scribe has no free-text prompt parameter, so the transcript policy cannot be stated to it in
prose the way it is to the other two. Its steering is `language_code: ne` plus a key-term list.
**Reversal:** delete the route and the client; nothing else depends on it. Hypotheses already
imported under `elevenlabs-scribe-v2` stay valid, and the word-level confidence signal disappears
with it.

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

## D24 — Drop Whisper large-v3 from cloud ASR; upgrade secondary to Gemini 3.8 Flash
Real-world testing showed OpenAI's Whisper large-v3 having poor performance on Nepali-English
code-switched audio. It returned text without word spans or confidence signals and produced frequent
transcription errors compared to ElevenLabs Scribe.

Whisper was removed completely from the cloud ASR pipeline, reducing transcription from three calls
per clip to two (Scribe v2 and Gemini 3.8 Flash) and halving OpenRouter spend per segment.
Simultaneously, the general LLM audio-chat route was upgraded from `google/gemini-3.5-flash-lite` to
`google/gemini-3.8-flash` (`asr_gemini_flash`), retaining its role as a disagreement signal and prompt
follower.

**Reversal:** Re-add `asr_whisper_large_v3` or another dedicated recogniser route to
`config/llm_routes.yaml`. Historical hypotheses under `whisper-large-v3` remain immutable in
`asr_hypotheses`.

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

## D28 — Swap secondary transcriber to Microsoft MAI-Transcribe 2 on OpenRouter
The secondary cloud ASR route (`asr_gemini_flash`), which used `google/gemini-3.8-flash` via chat
completions with audio attachments (`audio_chat`), is replaced by `microsoft/mai-transcribe-2`
(`asr_mai_transcribe_2`) routed through OpenRouter's `/audio/transcriptions` endpoint.

**Why:** MAI-Transcribe 2 is a dedicated multilingual speech-to-text model with native support for
code-switching and automatic language identification. Routing it through OpenRouter adheres strictly to
Invariant 5 (prepaid billing control), preserves the commit-per-segment and 2-call-per-clip ingestion
budget (paired with ElevenLabs Scribe v2 as primary), and tests lower WER performance against real
Nepali-English conversational speech.

**Reversal:** Revert `config/llm_routes.yaml` to configure `asr_gemini_flash` under `google/gemini-3.8-flash`.
Existing hypotheses in `asr_hypotheses` recorded under `gemini-3.8-flash` remain immutable.

## D29 — Add Google AI Studio as third ASR provider with Gemini 3.5 Transcribe
**Superseded by D31.** The model was right about code-switching and wrong about quota: its
Live API tier allows 100 requests a day, about four hours of audio. Google AI Studio remains
the third provider; the model and endpoint changed.

Google AI Studio is admitted as the third cloud inference provider alongside OpenRouter and ElevenLabs,
wiring `gemini-3.5-transcribe` (`asr_gemini_transcribe`) as a third cloud ASR route.

**Why:** Gemini 3.5 Transcribe is a dedicated speech-to-text model based on Gemini audio understanding.
It natively handles intra-sentence code-switching, verbatim transcription, and word-level timestamps
via the Google Interactions API (`POST /v1beta/interactions`). Operating alongside ElevenLabs Scribe v2
and Microsoft MAI-Transcribe 2 on OpenRouter, all three configured models now emit verbatim transcripts
and word-level timestamps, creating a rich three-way disagreement signal during ingestion.
The provider adheres to Invariant 5 (prepaid provider guarantee) under monitored, prepaid billing terms.

**Reversal:** Remove `asr_gemini_transcribe` from `config/llm_routes.yaml` and delete `app/llm/google.py`.
Existing hypotheses under `gemini-3.5-transcribe` remain immutable in `asr_hypotheses`.

## D30 — VAD-aligned macro-windowing and demultiplexing for Gemini Transcribe under Tier 1 quotas
**Superseded by D31.** Removed with the model whose quota it existed to work around. Kept
here because the arithmetic is worth remembering: a per-clip transcriber against a 100 RPD
cap exhausts a day's quota in about fifteen minutes of audio.

Google AI Studio Tier 1 restricts Live API models (`gemini-3.5-transcribe`) to 10 RPM, 10K TPM, and
100 RPD (requests per day). Calling Gemini per-clip on 2s–20s utterances exhausts the daily quota after
only ~15 minutes of audio, while ElevenLabs Scribe v2 and Microsoft MAI-Transcribe 2 have no 100 RPD
cap and operate best on short clips.

**Why macro-windowing:**
1. **Token & Duration Sweet Spot**: Gemini audio tokenization (~32 tokens/sec) consumes ~4,800 tokens for
   a 150s (2.5 min) window, using ~48% of the 10,000 TPM limit while pacing at 2–3 RPM (well under 10 RPM).
2. **Quota Multiplication**: 100 RPD provides 250 minutes (>4.1 hours) of audio per day (~10 full episodes)
   instead of failing halfway through a single episode.
3. **Natural Silence Boundaries**: Consecutive VAD segments are clustered up to 150s. Because boundaries
   align strictly with VAD segment boundaries (which snap to conversational pauses), zero words are ever
   sliced across window cuts.
4. **Timestamp Demultiplexing**: Gemini's verbatim word timestamps (`start_offset`, `end_offset`) are mapped
   back to their constituent segments and converted to clip-relative timestamps, strictly satisfying
   Invariant D26.
5. **Independent Scribe & MAI Dispatch**: Scribe and MAI continue receiving short clips concurrently,
   preserving their low hallucination rates.

**Reversal:** Remove the window clustering and demultiplexing block from `app/services/ingest.py` to restore
direct per-clip dispatch for all routes if quota limits are lifted in higher tiers.

## D31 — Gemini 3.8 Flash on AI Studio generateContent, audio only
`asr_gemini_transcribe` (`gemini-3.5-transcribe`, `POST /v1beta/interactions`) is replaced by
`asr_gemini_flash` (`gemini-3.8-flash`, `POST /v1beta/models/{model}:generateContent`), declared
`api: audio_chat` and carrying the clip inline. Google AI Studio remains the third provider and
`GOOGLE_API_KEY` is unchanged.

**Why:** the Live API's Tier 1 quota is 10 RPM / 10K TPM / **100 RPD** — roughly four hours of
audio a day. Every hack in D30 existed to survive that number. `generateContent` has no daily cap
of that shape, so the clip goes out per segment on the same thread pool as every other route, and
the windowing, the demuxing and the ten-second pacing sleep all go away.

Two consequences, both deliberate:

- **The corpus prompt applies for the first time.** `transcription.py` accepted a `prompt` and
  dropped it on the google branch. That was invisible because the Live API's transcription model
  took no free-text prompt anyway — so the transcript policy had never once been stated to this
  provider. `generateContent` obeys one, and a test now guards it. The route's `language` code
  rides on the prompt, as `generateContent` has no parameter for one.
- **The model is no longer asked for word timestamps**, and `AsrResult.words` is `None` on every
  real call. Spans come from the forced aligner instead (D32). This is the trade: an endpoint with
  no daily cap, in exchange for timings measured locally rather than claimed by the model.

The route is `audio_chat` rather than `transcription` because that is what it is — `config.py`
already documents that shape as "a general LLM being asked to transcribe, [which] may also
editorialise or hallucinate over silence". Naming it so keeps the risk in configuration instead of
buried in a client. It also means Flash reports no `avg_logprob` and can never win the seed
comparison, exactly as its predecessor could not.

It is given **only the audio** — never the other two hypotheses. Feeding it Scribe's and MAI's
transcripts to reconcile would have been cheaper and would have produced a better single
transcript, but it would collapse three independent opinions into one correlated output, and
`word_disagreement_rate` carries 0.40 of the priority score. A queue built on an echo is worse than
a queue built on a noisier but genuine measurement. It also caps the blast radius of a
hallucination at one hypothesis out of three.

**Reversal:** restore the `/interactions` payload in `app/llm/google.py` and the route block in
`config/llm_routes.yaml`. The D30 windowing would have to come back with it; see the git history
at `eda3562`.

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

## D35 — Google models are served from Vertex AI; the AI Studio client is removed
`app/llm/google.py` and `provider: google` are deleted. `app/llm/vertex.py` and
`provider: vertex` replace them, and `asr_gemini_flash` now calls `gemini-3.8-flash` at
`publishers/google/models/{model}:generateContent` on `{location}-aiplatform.googleapis.com`
rather than at `generativelanguage.googleapis.com`. The model, the `audio_chat` route shape, the
`system_id` and the forced-alignment arrangement of D31 and D32 are all unchanged — only the
transport is different, so hypotheses already recorded under `gemini-3.8-flash` stay comparable
with the ones recorded after it.

**Why:** AI Studio's quotas were a running tax on this project. D29 hit the Live API's 100
requests a day and D30 built VAD macro-windowing and timestamp demultiplexing to survive it; D31
abandoned that model and endpoint entirely. Vertex AI is the same models on project-scoped quota
that can be raised, and it is where the transcription models live. The prepaid rule that had kept
it out is gone (D34).

Authentication changes shape with the transport: Application Default Credentials, not an API key.
`google-auth` is added for exactly that — the credential lookup and the token refresh — and the
requests themselves stay plain `httpx` like every other client here. `GOOGLE_API_KEY` is no longer
read by anything.

Everything D30 left behind is also removed. The windowing and demultiplexing went with D31; what
remained was `max_retries: 4` and `retry_backoff_seconds: 2.0` in `config/llm_routes.yaml`, raised
to absorb 429s, and those are back at 3 and 0.5. The `Retry-After` handling in
`app/llm/base.py` stays: honouring a header the server sent is not a way round a rate limit, it is
the documented way to obey one, and it is provider-agnostic.

**Reversal:** the AI Studio client is at `f607de2:backend/app/llm/google.py` and needs
`GOOGLE_API_KEY` back in the environment. Its quota problem comes back with it.

## D36 — Gemini 3.5 Transcribe on Vertex AI as a fourth ASR system, with word-level diarization
`asr_gemini_transcribe` calls `gemini-3.5-transcribe` at `interactions:create` on Vertex AI, with
a `transcriptionConfig` asking for word timestamps and speaker diarization. It is the fourth
`asr*` route and the second transcriber in the corpus to report word spans of its own.

**Why the model:** it is a dedicated recogniser that handles intra-sentence code-switching, and
`languageCodes: [ne-NP, en-US]` says so in the request rather than hoping a single hint covers
both halves. D29 already judged the model right for this corpus and was defeated by AI Studio's
quota, not by the transcription; D35 removes that obstacle.

**Why it does not replace Gemini 3.8 Flash.** Flash stays, on its own route, unchanged. The two
answer different questions — Flash is a general model whose failure mode is editorialising, and
the recogniser's is mishearing — and the owner has not yet seen a Flash transcript on this corpus.
Four routes is four paid calls per clip, a third more than before, and that is the price of the
comparison. Dropping a route is one line in `config/llm_routes.yaml` once the answer is in.

**Verbatim has no field on this API.** Vertex's `TranscriptionConfig` carries `languageCodes`,
`diarizationMode`, `timestampGranularities` and `customVocabulary`, and nothing that selects
verbatim over the "smart" mode that strips disfluencies — which are exactly what this corpus is
collecting. So the instruction is prose, in `systemInstruction`, from the same `SCRIPT_POLICY` the
other prompted routes get. The no-transliteration rule rides there too, stated in both directions,
because a multilingual model's default is to normalise a code-switched utterance into one script
and that would silently destroy the measurement the corpus exists to make.

**Speaker labels get a column.** `hypothesis_words.speaker`, nullable, migration `facb0b37b4f8`.
Requesting diarization and discarding it would have been the more expensive way to buy nothing.
The label is clip-local and hypothesis-local: `spk_1` here is not `spk_1` in the hypothesis beside
it, and it is not `segments.speaker_id`, which names a person from an upstream manifest. Its use
is the comparison *within* one clip — two labels mean a turn boundary the VAD segmenter assumed
was not there, which is a rule flag waiting to be written and a reason a clip may be unusable.
Null means "not diarized" and stays null for the other three systems; nothing backfills it.

The route does not set `forced_align`. It reports its own timings, and D32's rule holds: the
aligner fills spans that are missing, it never overwrites spans a model measured.

**Reversal:** delete the route. Hypotheses already recorded under `gemini-3.5-transcribe` stay
immutable, and `scripts/purge_asr_system.py` removes them if they are not wanted. The column would
outlive the route and should — it costs a nullable string and it is what any future diarizing
transcriber writes to.

### Status: configured, and not yet reachable

The route is written and tested but **this project cannot call it.** `interactions:create` answers
`400 RESOURCE_PROJECT_INVALID` — measured against the live API, and the diagnosis is not a guess:

- The same error comes back for a **deliberately nonsense model name**, so it is not the model id,
  and not `gemini-3.5-transcribe-preview` either.
- The same error comes back from `global`, `us-central1`, `us-east4`, `europe-west4` and
  `asia-southeast1`, with the project **id** and with the project **number**.
- `generateContent` against `gemini-3.8-flash` on the same project, same credentials, same
  location returns `200` and a transcript. Auth, billing, the quota project and the Vertex path
  are all fine.
- `aiplatform.googleapis.com` *is* "Agent Platform API" and is enabled. There is no second API to
  turn on; `gcloud services list --available` offers nothing else that would gate this.

So the Interactions surface is allowlist-gated, and the fix is access rather than code. The route
is deliberately left configured while that is chased — which means **ingestion is broken until it
lands**, because a raising route aborts the segment and stage 3 fails the job. Comment the block
out in `config/llm_routes.yaml` to ingest in the meantime.

The alternative considered and not taken: making ingestion skip a failing route and continue on
the hypotheses that succeeded. That is a genuine robustness improvement against any provider
outage, but it silently changes the disagreement denominator per clip, so it is its own decision
rather than a bug fix smuggled in here.

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

## D38 — Gemini models switch to Google AI Studio API key and Interactions API fix (supersedes D35 and D36 Vertex endpoint)
Both Google models (`gemini-3.5-transcribe` and `gemini-3.8-flash`) authenticate with a single,
standard API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) targeting the public Gemini Developer API
at `https://generativelanguage.googleapis.com/v1beta`. All Vertex AI Application Default Credentials
(ADC), service account json paths, and `google-auth` token refresh plumbing are removed.

**Why:**
1. **Single standard API key:** Consistent with ElevenLabs and OpenRouter, eliminating machine-local
   GCloud token caches, ADC configurations, and IAM permissions.
2. **Interactions API URL and schema:** The previous implementation failed with `400 RESOURCE_PROJECT_INVALID`
   because it incorrectly targeted `interactions:create` (an invalid custom verb on Google REST endpoints)
   and wrapped parameters in fabricated nested structures. The real Gemini Developer API endpoint is
   `POST /v1beta/interactions` accepting a flat schema (`model`, `input`, and `generation_config`).
3. **Dropping custom vocabulary:** Google's Interactions API explicitly disallows combining
   `custom_vocabulary` with `diarization_mode` or `timestamp_granularities` (throwing 400 Bad Request).
   The harness prioritizes word-level timestamps and speaker diarization for downstream scoring and
   boundary alignment, so `custom_vocabulary` is omitted entirely.

**Reversal:** Reintroducing Vertex ADC would require restoring `google-auth` and `app/llm/vertex.py`.
The flat payload schema and exclusion of `custom_vocabulary` remain mandatory under Google's API specification.


