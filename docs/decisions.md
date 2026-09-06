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



## D39 — Anti-aliased downsampling to 16 kHz with libsoxr (completes D25)
D25 chased audible clicking in ingested clips and fixed four real problems at the cut boundaries:
speech padding, a raised-cosine edge fade, energy-aware pause snapping, and two-pass linear
`loudnorm`. Those fixes hold — measured on shipped clips, every cut lands in silence, the first and
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

## D39 — Gemini runs on Vertex AI under one restricted API key (supersedes D38, restores D35/D36)
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

## D40 — Gemini 3.5 Transcribe is held out of the disagreement scores, not out of the corpus
`asr_gemini_transcribe` carries `exclude_from_disagreement: true`. Its hypothesis is still
requested, stored, exported and shown; it simply does not enter `word_disagreement_rate` or
`cer_between_hypotheses`.

**Why:** the route writes English words in Devanagari and cannot be told otherwise (D39).
`mean_pairwise_disagreement` is a raw `difflib` comparison over tokens with no script
normalisation, so this system disagrees with the other three on *every* English token without
anything having been misheard. `word_disagreement_rate` is the heaviest term in the priority score
at 0.40, so counting it would push the most heavily code-switched segments up the annotation queue
for a reason that is not difficulty — a bias aimed precisely at the phenomenon the corpus exists
to study. Measured on one segment: 0.0 with the hold-out, 0.3333 without, from orthography alone.

Stage 4's CMI and Devanagari/Latin ratio were never at risk: they read `hypotheses[0]`, which is
`asr_scribe_v2`.

**Why keep the hypothesis at all.** It is the only transcriber reporting speaker labels, and its
self-reported spans are the second timing reference the D33 boundary report compares. Beyond that,
a whole-corpus transcript that renders every English word phonetically in Devanagari, time-aligned
against three transcripts that keep Latin, is a parallel resource that cannot easily be bought:
see the uses recorded against this decision. Deleting it to tidy the score would throw that away.

**Implementation.** The flag lives on `LlmRoute`, and both places that compute disagreement --
`ingest.py` at transcribe time and `purge.py` when a purge changes a segment's hypothesis set --
read the hold-out set from `disagreement_excluded_system_ids()` in `app/llm/transcription.py`.
They must never name a system independently: a system excluded at ingest and counted at rescore
would silently rewrite every score a purge touched.

**Reversal:** drop the flag. If the disagreement metric ever becomes script-aware, the hold-out
stops being necessary and the route rejoins the comparison with no other change.

## D41 — Gemini Composite: the recogniser hears, a reasoning model spells (supersedes D40's hold-out)
`asr_gemini_transcribe` becomes `asr_gemini_composite`, `system_id: gemini-composite`. Gemini 3.5
Transcribe on Vertex still hears the clip and still supplies the text, the word spans and the
speaker labels. Its token list is then passed to Gemini 3.8 Flash on **OpenRouter**, which rewrites
each token into the script its own language uses. The result is recorded as one system, named so
the seam is visible; the paper discloses the two-model pipeline.

**Why:** D39 established that this recogniser accepts no steering of any kind, and therefore writes
English phonetically in Devanagari (`active` → `एक्टिभ`) with no lever to stop it. D40 dealt with
that by holding it out of the disagreement scores. Restoring the script instead fixes the cause
rather than the symptom, so `exclude_from_disagreement` is dropped and the corpus is back to
**four voting systems**. Measured on a 30 s clip: 93 tokens in, 93 out, 18 restored, 0 same-script
edits.

**One token in, one token out.** This is the whole design, and it is enforced in code, not asked
for in the prompt. Each restored word inherits the span the recogniser measured for it, so there is
no re-alignment and the forced aligner is not involved — `forced_align` stays false here (D33). A
rewrite returning a different token count is retried and then fails the segment: padding or
truncating would give every word after the first divergence someone else's timing, which is far
worse than a segment that fails loudly. `app/llm/script_restore.py` owns this.

**The rewrite runs on OpenRouter, not Vertex.** It is text inference, which is where OpenRouter
belongs in this harness, and Vertex answers Flash with a spurious `blockReason: SAFETY` often
enough to matter (below). Only the audio call needs Vertex, because only Vertex serves the
recogniser at all.

**A reasoning layer can lie, so its lying is measured.** A token that comes back in the *same*
script but different (`मिटिङ` → `बैठक`) is the model correcting the recogniser rather than
transliterating it. That is counted per hypothesis as `script_restore_same_script_edits` and
carried in `metadata_jsonb`, so a suspect segment can be found again. It is reported rather than
raised: one disputed token must not cost an episode. Standalone `asr_gemini_flash` is retained
partly as the control on this layer — it is the only route that hears audio and writes Latin
directly, so where it and the composite disagree on a script decision is the audit set.

**The raw Devanagari is kept as provenance, not as a hypothesis.** It rides in the hypothesis's
`metadata_jsonb` as `text_devanagari` and never reaches `text_raw`, the disagreement comparison,
the analysis or the queue. Its value is a word-aligned Devanagari/Latin parallel corpus that the
pipeline now produces for free.

### Two Gemini failure modes found while building this, both silent

**One language code, not two.** Sending two or more `languageCodes` makes the recogniser return
HTTP 200 with *no content* for any clip past roughly 15 seconds. Deterministic, three runs per
cell:

| `languageCodes` | 15 s | 18 s | 20 s | 25 s | 30 s |
|---|---|---|---|---|---|
| `[ne-NP, en-US]` | 45 w | EMPTY | EMPTY | EMPTY | EMPTY |
| `[ne-NP]` | 49 w | 59 w | 64 w | 77 w | 93 w |
| `[en-US]` | EMPTY | EMPTY | EMPTY | EMPTY | 2 w |
| `[ne-NP, en-US, hi-IN]` | 42 w | 1 w | EMPTY | EMPTY | 4 w |

`MAX_SEG_SECONDS = 20.0`, so a second code silently blanks the long end of every episode. This
supersedes D36's "send both codes" reasoning: that was right about the corpus and wrong about what
the API can do, and `en-US` was never buying script correctness anyway — the restore step is what
buys it. Scribe and MAI were swept at the same durations and are clean, so the cliff is Gemini's,
not the audio's.

**An empty 200 is not a success.** Both Gemini routes could return one — the recogniser past the
duration limit, and `audio_chat` on a spurious `blockReason: SAFETY` that clears on retry with the
request unchanged (observed on 4 of 7 clips in one sweep, then not reproducible on the same clip
minutes later; `OFF` is a valid threshold and the categories are correct, so it is not a
configuration error). `_send_with_retries` only ever sees a 200, so emptiness is now judged and
retried in `vertex.py`. An empty `audio_chat` answer with *no* block reason is still accepted:
`ASR_PROMPT` asks for an empty string when there is no intelligible speech.

**Reversal:** drop `restore_script_route` and the composite is the raw recogniser again — at which
point D40's hold-out has to come back with it, because the orthography artefact returns.

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

## D44 — The script-restore route runs with thinking off
`script_restore` sends OpenRouter's `reasoning: {enabled: false}`. Routes carry a
`reasoning_enabled` field for this; it is unset everywhere else, so every other route keeps
whatever the provider does by default.

**Why:** the first full ingest failed at `script restoration failed: unparseable rewrite: no JSON
array in the response`, and the rewrite was not unparseable — it was truncated. Gemini 3.8 Flash
thinks by default, and on a task that is a dictionary lookup it thought until the budget ran out:
24 of the run's 51 rewrites returned `finish_reason: length` with the JSON array stopping
mid-token, a mean 2,843 reasoning tokens against a 4,096 cap and 3,929 on the worst call. A
partial array cannot be aligned to the spans, so each of those failed its segment, and
`temperature: 0.0` made all three retry attempts the same draw.

**The thinking was not neutral, either.** The 3,929-token trace was spent circling one token,
`ट्युन`, over whether the speaker said "tune" or "tuned", and it concluded that it should restore
the *intended* orthography rather than the heard one — rule 2 of the instruction reasoned away in
the open, on a route whose entire premise is that it changes script and nothing else.
`count_same_script_edits` does not catch that class of drift: it only sees Devanagari that came
back as different Devanagari, not Devanagari that came back as the wrong Latin word. Turning
thinking off removes the surface the argument happened on.

**Cost, measured against that run:** ~469 prompt and ~206 completion tokens per segment, ~$0.0011
a call, against $3.31 per hour of audio with thinking on — about a sixteenth, with the retries
gone.

**What this does not fix.** `_extract_array` still reports a truncated response as "no JSON array",
which is what sent the diagnosis to the wrong place; `finish_reason` is never read. Raising
`max_tokens` was rejected as the fix — it buys headroom for the rambling rather than removing it —
but 4,096 now has a large margin over a 391-token worst case, so it stays.

**Reversal:** drop `reasoning_enabled: false` from the route; the field going unset restores the
provider default.

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

## D49 — Scribe is asked to diarize, because one source of speaker labels is not evidence

`asr_scribe_v2` sets `diarize: true`. Speaker labels now come from two systems instead of one,
and `LlmRoute.diarize` is the flag that says which.

**Why:** the composite's recogniser was the only transcriber reporting who said each word (D36),
and a label no other system can be checked against cannot be validated, only believed. Scribe
diarizes at no extra charge and its word spans are already the most trustworthy in the table
(D48), so it is the natural second opinion. Two sources make per-clip speaker agreement a
measurable quantity; one made it an assertion.

**The other two cannot supply a third.** MAI-Transcribe-2 returns no speaker field at all —
probed live against `/audio/transcriptions` with `diarize=true` and `response_format=verbose_json`,
the response carries `text`, `usage`, `language`, `duration`, `segments` and `words`, and neither
the word entries (`word`, `start`, `end`) nor the segment entries (`id`, `start`, `end`, `text`)
name a speaker. `asr_gemini_flash` returns no word timings of its own, so there is nothing to hang
a label on; its spans come from the CTC aligner afterwards, which knows about acoustics and not
about speakers.

**What these labels are, and what they are not.** They are clip-local. The pipeline cuts on speech
turns first and transcribes each clip independently, so `speaker_0` in one clip has no relation to
`speaker_0` in the next, and neither has any relation to `segments.speaker_id`. That supports one
question — do two systems agree about how many voices are in *this* clip, and where they change —
and not the question a sociolinguistic read actually asks, which is whether a given speaker's
code-switching rate differs from another's across a whole episode. Answering that needs *global*
speaker identity, which this pipeline cannot produce at all: it would take either a diarization
pass over the full episode before segmentation, with segments joining to it by time, or speaker
embeddings clustered across clips. Both are a new pipeline stage, not a flag. Nothing here should
be read as having delivered it.

**Not yet measured.** The one ingested episode is a single-presenter TV review — 1283 words
labelled `spk:0` against 6 labelled `spk:1` — so it contains no diarization signal to score
either system on. Agreement between Scribe and the composite is measurable only on multi-speaker
audio, and no such episode has been ingested.

**Reversal:** one line in `config/llm_routes.yaml`. The column already existed and null already
meant "not diarized", so no schema change and nothing already imported is affected.

## D50 — The composite is held out of the disagreement scores again, for a different defect

`asr_gemini_composite` carries `exclude_from_disagreement: true`. Its hypothesis is still
requested, stored, exported and shown; it does not enter `word_disagreement_rate` or
`cer_between_hypotheses`.

**This is not D40 returning.** D40 held the raw recogniser out because it wrote English in
Devanagari, so it disagreed with every other system on every English token without anything having
been misheard. D41 fixed that cause with the restore step and put all four systems back in the
comparison. **The restore step still works** — recogniser token count equals restored token count
on all 23 clips of the pilot episode, and the composite is not collapsed onto the model that does
the restoring (composite-vs-flash disagreement 0.273, mid-pack among the six pairs; the most
similar pair is mai-vs-flash at 0.211). D41's reasoning stands and is not superseded.

**The defect is upstream of the restore step.** Gemini 3.5 Transcribe omits speech it heard. On
8 of 23 clips it returns a transcript more than 10% shorter than the median of the other three —
worst cases −42% (seg 16), −35% (seg 20), −27% (seg 30). The omission is in the recogniser, not
the rewrite: seg 16's raw `text_devanagari` is already 36 tokens where the other systems have ~62.

**And the omission is not random.** Aligned against each of the other three systems in turn, the
tokens the composite lacks are 71–75% Latin, where the tokens it keeps are 32–34% Latin
(Flash 74.8%, Scribe 71.1%, MAI 75.2% — kept 32.7/32.5/34.3%). It is dropping the English-dominant
stretches: seg 16 loses `you will definitely appreciate this` and `But then this one is very close`
while keeping the Nepali between them. Of the 123 tokens missing against Flash, ~80 are absent by
raw token count and the remainder may be alignment artefacts, so the direction is firm and the
magnitude is approximate.

**It is deterministic, so it is not a retry problem.** Three repeats of the bare recogniser per
clip returned identical token counts every time — 36/36/36 and 35/35/35 on the truncated clips,
66/66/66 and 60/60/60 on clean ones. Not duration-driven (mean 19.44 s truncated against 19.08 s
not). The only remaining lever is `language_codes`, which D41 locked to `ne-NP` because two or more
codes blank long clips outright.

**Why that justifies a hold-out.** `word_disagreement_rate` carries 0.40 of the priority score. A
system that deletes English manufactures disagreement precisely where the corpus is most
interesting — the same class of bias D40 objected to, arriving by a different route.
Spearman(`code_switch_density`, `word_disagreement_rate`) over the pilot episode is **−0.616** with
this system held out and **−0.229** with it counted: held out, the measure carries a clean signal
(code-switch-dense segments are ones the systems agree on, English being acoustically distinct),
and counting the composite cancels half of it.

**Why keep the hypothesis.** It is one of only two systems that report speaker labels (D49), and
MAI cannot supply a third. Its self-reported spans are no longer a trustworthy timing reference —
see the triangulation in D48's follow-up — but the transcript remains a parallel resource and the
diarization is half the only cross-check there is.

**Evidence is one episode.** 23 clips, one presenter, consumer-tech Nepali, which is unusually
loanword-dense and therefore the condition most likely to expose an English-dropping failure. The
hold-out is cheap and reversible, so it is applied now rather than after replication; replicating
the drop rate across speakers and domains is still outstanding.

**Reversal:** drop the flag. Already-imported scores are unaffected until something recomputes
them — `purge.py` is the only path that does, and it reads the same hold-out set, so the two
computation sites cannot disagree.

## D51 — The Gemini composite is removed, not held out

`asr_gemini_composite` and the `script_restore` route it drove are gone from
`config/llm_routes.yaml`. Three ASR systems remain: Scribe, MAI and Flash. The hold-out set
`disagreement_excluded_system_ids()` returns is now empty, which is the honest state rather than a
forgotten flag — nothing else has ever needed holding out.

**D50 held it rather than removed it, and said why:** it was one of only two systems reporting
speaker labels, so it was "half the only cross-check there is". That argument is gone. Measured on
a two-speaker episode, 51 of 68 clips are single-speaker for *both* systems and agree trivially at
100%; the 17 contested clips agree at 94.8%. The headline 98.3% is an artefact of clips where
there is nothing to disagree about. Clip-local diarization cannot measure speaker agreement,
because the pipeline segments before it transcribes and a 20-second window of a podcast almost
always holds one speaker. More two-speaker episodes will not change that — the ceiling comes from
the segmentation order, not the corpus.

**What was left once that argument fell.** On leave-one-out consensus agreement over 68 clips,
script-blind: Scribe 0.623, Flash 0.620, MAI 0.610, composite **0.538**. The other three sit
within 0.013 of each other — noise — and the composite trails them by seven times that spread. It
was the least accurate of the four, and biased in the one direction that matters here.

**The defect replicated, on a better measure than D50 used.** D50's evidence was the Latin share
of tokens the composite "dropped" under alignment. On this episode that method reports 702–772
dropped tokens against an actual shortfall of 221–235, so roughly two-thirds of the "dropped" set
is lexical disagreement rather than omission — and English is exactly where two systems most often
disagree on spelling. That statistic is retired. The replacement never aligns anything:
Spearman(clip Latin share, composite token shortfall) is **+0.670** against Scribe, **+0.612**
against Flash and **+0.598** against MAI, n=68. Three references with different failure modes
agree. Clips more than 10% short: 24 of 68, against the pilot's 8 of 23 — 35% both times.

**Why removal rather than a disabled flag.** Leaving the route configured but unused would leave
`exclude_from_disagreement` as the only thing standing between its hypotheses and the score, and
AGENTS.md already warns that the two computation sites desynchronise silently when a system is
named in one place and not the other. A route that must never be used is better deleted than
remembered.

**The already-collected hypotheses are evidence and are dumped, not discarded.**
`scripts/purge_asr_system.py` writes every row to JSONL before deleting, and the composite's
output is the subject of the corpus's headline claim — that a Nepali-configured recogniser deletes
the English half. The dump is the record that claim rests on.

**`app/llm/script_restore.py` stays** even though nothing routes to it now. Any recogniser that
transliterates English into Devanagari and cannot be told not to needs exactly this repair, and
another one is being evaluated. Keeping it is a decision, not an oversight; delete it if that
evaluation ends without needing it.

**Evidence is two episodes and three speakers.** Both are pilot data.

**Reversal:** restore the two route blocks from git history and re-ingest. Hypotheses already
purged come back only from the JSONL dump or by re-transcribing, which costs money.

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

## D54 — The priority score is measured against the seed, on the clock

The queue formula becomes `0.60 * seed_outvoted + 0.25 * low_confidence + 0.15 * rule_flag_score`.
`word_disagreement_rate` and `code_switch_density` are dropped from ranking. `logprob_floor` moves
from −2.0 to −0.5.

**The formula it replaces ranked worse than shuffling.** Scored against the 22 labels available,
using realized WER from seed text to final human text as the target, the old `priority_score`
reached Spearman **−0.281**, and its top half captured 44.7% of all editing done against a 50%
baseline. Component by component the reason is unambiguous:

| Component | weight | ρ vs. realized edits | observed range |
|---|---|---|---|
| `word_disagreement_rate` | 0.40 | +0.123 | [0.20, 0.46] |
| `low_confidence` | 0.25 | +0.461 | [0.03, 0.10] |
| `code_switch_density` | 0.20 | **−0.768** | [0.09, 0.47] |
| `rule_flag_score` | 0.15 | 0.000 | [0.00, 0.00] |

A weak positive at full weight, a strong positive squashed to nothing, a strong **negative** at
full weight, and a term that was identically zero on every labelled segment. It summed to a
ranking that was slightly worse than random.

**Why code-switch density had to go.** It was the strongest signal in the formula and pointed the
wrong way. English is acoustically distinct, so code-switched speech is what the recognisers
*agree* on; boosting it spent annotator time on the easy segments. D50 saw the same correlation
(ρ(csd, wdr) = −0.616) and read it as the disagreement measure working. It was the code-mixing
term failing. Note this is a claim about *review order only* — code-switching is what the corpus
is for, it is still measured, stored and exported, and every segment is still labelled.

**Two of the four terms could never reach their stated weight.** `code_switch_density` is
`cmi/100`, and CMI is `100·(n − majority)/n` where the minority can never exceed half the tokens —
so it is structurally capped at 0.5 and its 0.20 weight bought at most 0.10. Real data: mean 0.201,
max exactly 0.500. `rule_flag_score` divides by seven flags, two of which are mutually exclusive.
`test_scoring.py` now asserts every term can reach its full weight; the old suite missed this
because it passed `code_switch_density=1.0` directly instead of going through the generator.

**Why the floor moved.** Scribe is the only system reporting an `avg_logprob`, and it spans about
−0.68 to −0.004. Against a −2.0 floor the term used the bottom third of 0–1, and on labelled data
only [0.03, 0.10] — so the one component that actually correlated was multiplied by ~0.065. −0.5
spans the observed range without fitting it exactly.

**What replaces them.** `seed_outvoted` — the share of slot time where every other system
contradicts the seed. Measured on its own it reached ρ **+0.558** and captured 80.5% of editing in
its top half (p = 0.006 against 20k random rankings). Through the live code path the assembled
formula reaches ρ **+0.577** and 77.8%. It is measured in *time* rather than words because the
systems disagree about how many words there are, so a word count cannot also be the unit.

**Fixing the old formula's bugs was not enough.** Dropping the anti-correlated term and
recalibrating the floor lifted it to ρ +0.238 / 61.3%, which is still statistically
indistinguishable from random (p = 0.188). The time-aligned signal is what moved it.

**The evidence is thin and the old score is kept because of it.** 22 labels, one episode, one
annotator, one seed system. That is enough to choose a direction and not enough to settle
parameters. The superseded formula is computed and written to `reason_jsonb.legacy` on every task,
so the first full run adjudicates on its own data. `queue.legacy_weights` exists only for that and
deliberately does not sum to 1.

**Reversal:** restore the old weights in `config/settings.yaml` and `QueueWeights`, and read the
legacy components back out of `reason_jsonb`. Nothing stored changes shape, and re-running the
queue builder rescores every active task, so it is a config edit plus one command.

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


## D63 — Two pots, filled to a duration; the tier records how hard a label was looked at

Supersedes D5. `episodes.pot` (`gold` / `train` / `unassigned`) replaces the hashed split as the
thing the corpus is organised by, and `segment_labels.verification_tier` (`verified` / `screened`)
records how much attention each decision actually got. `episodes.split` survives as a derived
label, tied to the pot by a CHECK: gold is always `test`, train is always `train` or `val`.

**D5's rule was right and its reason was wrong.** D5 forbade segment-level splits because "segments
from one episode share speaker, room and topic". The room half is already dead — D25's two-pass
loudnorm and D39's libsoxr resample to 16 kHz mono destroy most of what distinguishes one
professionally-produced podcast mic from another, so the pipeline had closed that leak before the
split ever ran. What actually justifies the rule is two things D5 barely mentions:

1. **Adjacency.** VAD cuts are contiguous and D25 *pads* speech at the edges, so consecutive clips
   share audio samples. Shuffling at clip level puts two halves of one sentence in train and test.
2. **Lexical clustering.** Within an episode the same names, jargon and rare words recur, and for a
   code-switching corpus the English switch points cluster by topic, so clip-level shuffling
   inflates every CMI number the corpus exists to report.

And D5 does not achieve what it claims. "Segments from one episode share speaker" — so do segments
from *every episode of the same show*. A host appearing across forty episodes is in the test set
however the episodes are sliced. The grouping key was never the episode; it is the show. Episode
granularity is kept because it is the unit ingestion produces, and the coverage-first selection
below is what actually spreads the benchmark across shows.

**A ratio cannot express what anyone wants from a corpus.** `assign_split()` hashed the episode id
into train/val/test by ratio. Three problems, in the order they hurt:

- The ratio was over episode **count**, not duration, and episodes run from minutes to the
  four-hour ingest ceiling. "Five hours of benchmark audio" was not expressible.
- A hash only hits its ratios in the limit. At the 30–60 episodes this corpus is heading for,
  `test: 0.1` is a coin flip that lands anywhere from two episodes to nine.
- Nothing stratified. A hash can hand back an all-one-show test set and no part of the system
  would object.

So `assign_pots` fills gold to an **hours** target, greedily, **coverage first**: of the episodes
that still fit, take the one adding the most unseen show / gender / age bracket / topic. Duration
alone would take the longest episodes, which is the fastest route to five hours and the most likely
route to five hours of one show.

**Three rules, each guarding a number that would otherwise look fine and be wrong.**

1. **The pot is assigned before any clip is seen.** Ingestion calls `assign_pots` between import and
   queue build. A pot chosen per clip *while looking at it* correlates with how hard the clip turned
   out to be — route the hard ones to gold and the benchmark reads pessimistic, route the quick ones
   and it reads optimistic — and nothing recorded afterwards can separate the two. Assigning ahead
   of time makes the routing independent of content by construction.
2. **Whole episodes.** The surviving half of D5, for the two reasons above.
3. **Gold is one-directional.** An episode never leaves gold, and by default never enters it from
   train: a recording that was trained on and later promoted to the benchmark turns it into a
   memorization test, silently. `allow_promote_from_train` exists for the window before anything has
   trained and names itself at every call site. The consequence is real and intended — once
   everything is placed, raising the gold target does nothing until new episodes arrive.

**The tier is the corpus's claim about itself.** The train pot is meant to be screened: accepted on
cross-ASR disagreement without listening, which is the only way 50 hours is affordable. But a
screened row written as `accepted_unchanged` by `annotator: owner` asserts a human verified it. That
is false, and it costs twice — a reviewer asking about the verification protocol gets a wrong
answer, and the disagreement gate can never be measured, because its decision was overwritten by a
confirmation nobody made. So the tier is a column, every export row carries it, and the manifest
reports the mix per split. The 5% audit sample (`queue.audit_sample_rate`) becomes an instrument:
re-verify a sample of screened clips and the gate's error rate falls out.

Screening a gold segment is refused at `record_decision` with a 409, and the `gold` export refuses
to write at all if a screened row reaches it — belt and braces, because the failure is silent and
the artefact outlives the session that made it.

**Deletion is not offered.** Bad audio is flagged `unusable_audio`, as before. Deleting would lose
"what fraction of real Nepanglish podcast audio is untranscribable", which is a publishable number,
and — worse — a delete key used casually in the train pot and carefully in gold is a biased filter
applied to one distribution and not the other, with nothing recording that it happened.

**Milestones** (`app/services/gamify.py`) are derived on read from existing tables and stored
nowhere. Two choices about what they reward: levels are measured in **audio cleared, not clips
decided**, so screening a thousand two-second clips does not outrank verifying an hour of hard ones;
and a verified second counts double a screened one, so the scoreboard does not pull against the
corpus's own quality claim.

**Reversal:** the migration has a working `downgrade`, and it backfilled rather than reassigned —
existing episodes took their pot from the split they already had, so nothing moved. Reversing now is
cheap because no real corpus is committed to a split yet; after a gold pot has been annotated and
exported, treat the pot assignment as permanent for the same reason D5 said to.

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
