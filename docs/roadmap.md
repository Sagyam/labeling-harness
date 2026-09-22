# Research roadmap: overlapped speech

What is next for the ASR work on the Nepanglish corpus, as replanned on 2026-09-22. What has been
measured so far is in [findings.md](findings.md). The earlier roadmap is retired (list at the end).

**Where this starts.** Single-speaker recognition is largely solved for this corpus. The Flex
fine-tune scores 6.5% folded WER on clean gold clips. Crosstalk is what remains: 17% at 5–15%
overlap and 29% above 15%, on 205 gold clips. The D96 sweep showed that mixing more synthetic
crosstalk into a model that writes one stream of text for one speaker does not move those numbers
(findings.md, *Synthetic crosstalk does not move real crosstalk*). The model is never told which
voice the label wants, so it hedges: fewer insertions, more deletions.

This chapter needs two things:
- references that say, per speaker, what was said in overlap;
- models that are told whom to follow (by diarization, enrolment, or an output that writes every
  speaker).

Depending on how strong the results are, it may become its own paper.

**Priority, set by the owner:**
1. **A.** Separation-assisted per-speaker labelling. *Pilot failed 2026-09-22; not built.*
2. **B.** Distilling the Flex fine-tune.
3. **C.** The augmentation pipeline.

D (models for overlapped speech) is scored on A's labels, so it waits for A. E is how all of it
is measured. Sections are lettered so they do not collide with the retired numbered items that code
comments still cite.

## A. Separation-assisted per-speaker labelling (priority 1; pilot failed 2026-09-22)

**Result: no-go, and A2 is not built.** The A1 listening pilot (findings.md, *Separation does not
give per-speaker labels*) found MossFormer2 good on light to moderate crosstalk and poor on heavy
crosstalk. It also found that a separated track sometimes holds two different people, so a track
cannot say who said what. Round 1 had 13 of 30 two-voice clips useless (43%, against a 20% bar).
Decoding in the model's 2 s training windows, in round 2, did not change the verdict. Target
speaker extraction is the obvious next separator, but published results on real conversational
overlap are poor (REAL-T). What replaces A as priority one is the owner's call. The rest of this
section is kept as the record of what was planned.

**The problem.** Gold labels in overlap are a single stream: the stronger voice, plus some of the
weaker one, with no speaker attribution (D95). Every multi-talker model is scored per speaker
(cpWER, tcpWER, ORC-WER). A model that correctly writes both voices is otherwise charged
insertions, which is the D96 trap again. Relabelling per speaker is the fix, but labelling
crosstalk means listening to two people at once. That is the owner's bottleneck, and the reason
the current labels drop part of the weaker voice.

**The idea (owner's proposal, 2026-09-22).** Split the mix into one track per voice with a speech
separation model (MossFormer2), transcribe each track as its own clip, and give the annotator the
original mix and both tracks side by side.

Separation is used here as a **listening aid, not a recogniser front end**. Its artifacts, which
hurt a recogniser trained on clean speech, matter little to a person who also hears the original.
The mix stays the ground truth; the tracks only make the second voice hearable. The two
transcripts are **never merged into one label**. Each voice keeps its own stream. The per-speaker
text is the research value, and a time-ordered merge can always be derived later from the word
timings.

The alternatives weighed and set aside:
- **Relabel per speaker by ear from the mix alone.** The labelling burden this idea removes.
- **Declare that gold transcribes only the main voice in overlap.** This keeps single-stream
  models fair but gives multi-talker models nothing to be scored on.

### A1. Pilot first, offline, with no harness changes

Owner's rule: nothing is built into the harness until a pilot on existing heavy-crosstalk clips has
passed. The pilot is a notebook built from `notebooks/src/` like the others. It loads data the way
`Finetune.ipynb` does: `ftkit.download_dataset`, the gold and analytics exports of
`Sagyam/nepanglish-asr`, and episode audio sliced by `ftkit.AudioStore`. It uploads its outputs to
`Sagyam/nepanglish-asr-flex-ft` under a pilot prefix. The results go into findings.md and the
experiment code is discarded afterwards.

- **Clips.** The 118 gold clips with `classes.overlap == ">15%"`, then the 87 at 5–15% if the
  first set passes.
  - `classes.speakers` gives the voice count. Clips with three or more voices are reported
    separately: a two-way split does not fit them.
  - `speaker_turns` (analytics export) gives the diarized turns, and `overlap_spans` the measured
    overlap.
- **Separation.** ClearerVoice-Studio's `MossFormer2_SS_16K`
  ([repo](https://github.com/modelscope/ClearerVoice-Studio)): 16 kHz, two outputs, matching
  invariant 7's format. Record which model and version was used. The license must be checked
  before anything is built on it.
- **Measurements:**
  1. **Separation by ear.** On about 30 clips, the owner rates each clip clean, bleeding but
     usable, or useless. The useless share decides whether anything is built.
  2. **Speaker mapping.** The tracks come out in arbitrary order. Map each to a diarized speaker by
     how well its energy envelope overlaps that speaker's turns. Report how often the mapping is
     unambiguous.
  3. **Recovered words.** Decode both tracks with the 2026-09-17 Flex fine-tune (in
     `Sagyam/nepanglish-asr-flex-ft`, `best/`). Against the current single-stream labels, report,
     per crosstalk bucket and with S/D/I:
     - the WER of the mix;
     - the WER of the track mapped to the main speaker;
     - the WER of both tracks' words taken together.

     Recovered weak-voice words show up as falling deletions. This doubles as the D3 experiment:
     separation in front of Flex.
  4. **Bleed.** Words that turn up on both tracks: how often, and how many.
  5. **Labelling time, second phase.** Only if 1–4 pass. The owner labels about 20 clips from the
     mix alone and about 20 with the mix and both tracks. Time comes from the client-reported
     `duration_ms`, the same quantity as the throughput baseline. This phase needs the editor to
     play the extra tracks: a throwaway prototype, or the first step of A2.
- **Go/no-go.** Fix the thresholds before the run, in the style of D96: a maximum useless share,
  and a minimum saving in time per clip. The owner sets them. A reasonable starting proposal is at
  most 20% useless and at least a 25% time saving.

### A2. What would be built if the pilot passes

This is a design change that needs a decision entry and an Alembic migration with a working
downgrade (invariant 1).

- **Separated tracks are child clips.** A new parent link on `segments` records each track's
  parent clip, track index and mapped diarized speaker.
  - Each child is a real 16 kHz mono FLAC holding one voice (invariant 7), with peaks precomputed
    at creation.
  - Each flows through transcription, fusion, hazards and labelling exactly like a single-speaker
    clip. After separation, a child is no different from a clip with no crosstalk.
  - No new status field (invariant 3).
- **Trigger at ingest.** A clip whose overlap share passes a threshold is sent for separation. The
  threshold is chosen from pilot data, not by feel.
  - **Where it runs.** Separation runs off the backend, on a Modal GPU like diarization (D79).
  - **Paid calls.** Each child's recogniser calls are routed and logged (invariant 6) and
    checkpointed after their `llm_requests` row (D93). `dry_run` is honoured.
  - **Cost.** Each flagged clip becomes two clips, each transcribed by three routes: six more paid
    calls per flagged clip. About 27% of gold clips are over 5% overlap.
- **Word timings per speaker.** Forced alignment on each child gives each speaker's word spans,
  which tcpWER needs. The D33 rule holds: routes that report their own timings are not
  overwritten.
- **Triage and editor.**
  - Triage already sorts by crosstalk, so a parent stands out.
  - Opening one shows the family: the original mix and both tracks, each playable, and each track's
    transcript editable.
  - The check is "every word is present, **and on the right track**". Bleed means a track's
    recognisers will sometimes write the other voice's words.
  - Every decision on a child writes a label, an event and an audit row in one transaction
    (invariant 8).
- **Children are always verified by ear.** A screened decision on a child is refused, as on gold
  (invariant 5).
- **Pots and splits follow the parent.** A child is in its parent's pot. `set_segment_pot` moves a
  family together, with an audit row for each clip (invariant 4). Split follows the episode.
- **What the tracks are for.**
  - **Training and scoring** happen on the *original mix*, with the per-speaker labels. The
    separated audio carries artifacts and is never treated as real audio.
  - **Donors.** Gold tracks are never training donors (D76).
  - **Export.** The export gains per-speaker streams for a parent (each child's text and word
    spans, keyed to the diarized speaker). Tracks are exported as audio only behind an explicit
    flag.
  - **Existing labels.** A parent's existing single-stream labels stay as history (invariant 2).
- **Open questions for the decision entry:**
  - how a clip with three or more voices is handled;
  - whether a failed or useless separation falls back to the current single-stream labelling;
  - whether the parent keeps its own recogniser hypotheses for single-stream comparisons.

## B. Distil the Flex fine-tune into other architectures (priority 2)

Flex is the only model that is good at Nepanglish, and it is closed in two ways: it decodes whole
utterances, and it reports no word timestamps. Many stronger designs were never trained on Nepali:
streaming transducers, models that report word timestamps, diarization-conditioned models like D1
and D4. Teaching one of them Nepali from Flex would open all of those. B does not wait on A,
because it is measured on single-speaker WER.

- **Teacher.** The current Flex fine-tune, with its greedy+retry decoder.
- **Students, in order of promise:**
  - **Whisper-large-v3-turbo.** It already knows Nepali's script, and it is DiCoW's backbone: a
    Nepali-strong Whisper feeds straight into D1.
  - **Parakeet / FastConformer (TDT or CTC).** Streaming, and word timestamps from the alignment.
    The English tokenizer has no Devanagari, so it gets a new SentencePiece tokenizer and a
    reinitialised decoder
    ([Hindi recipe](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/45)). The
    multitalker variant (D4) is built on the same encoder.
  - **AI4Bharat IndicConformer** ([repo](https://github.com/AI4Bharat/IndicConformerASR)). A NeMo
    hybrid CTC/RNN-T model for India's 22 scheduled languages, Nepali among them. It starts closer
    to our Nepali than either of the above. Check its checkpoint and licence first.
- **Method.**
  - **Pseudo-labels.** Decode unlabelled Nepanglish audio with the teacher and train the student
    on the output, plus our verified labels
    ([Distil-Whisper](https://arxiv.org/abs/2311.00430): 22k hours, WER-filtered).
  - **Filtering.** Keep only pseudo-labels the teacher is sure of. With no reference to compute
    WER against, use agreement between decoders or a
    [label-free filter](https://arxiv.org/abs/2407.01257). Loops caught by the retry are dropped.
  - **Distillation loss.** KL on the teacher's token distributions where tokenizers match,
    sequence-level (pseudo-label) distillation where they do not.
- **Open design questions.**
  - **Where the unlabelled audio comes from.** Ingest runs three paid ASR routes per clip, and
    episodes enter the harness only two ways (D86). A distillation corpus probably lives outside
    the harness database, decoded by the teacher alone. That needs a decision entry.
  - **How much audio.** The learning curve was flat on more of the same data (findings.md), but
    that measured a fine-tune, not a student learning a language from scratch.
- **Success bar.** The student comes within a stated margin of the teacher on gold and val, folded
  and raw, split into S/D/I, per clip class. Only then are its extras (streaming, timestamps,
  conditioning) worth their cost. Gold and val audio are never pseudo-labelled for training (D76).

## C. Augmentation pipeline (priority 3)

What the sweep taught: augmentation pays only when the model has a way to use it, either
conditioning that says whom to follow or an output format that writes both voices. The pipeline
below is built to feed D, not single-stream Flex. Gold and val audio are never a source of mixed-in
speech or noise (D76).

1. **Two-voice mixes from whole verified clips.** This lifts D95's blocker. The only text D95 had
   for a donor burst was an unverified recogniser's. If the second voice is a whole verified train
   clip, or a word-aligned stretch of one (`app/services/forced_align.py`), both transcripts are
   verified. That gives labels for both voices, as serialized output and target-speaker models
   need.
2. **Realistic conversation timing, overlap boosted.** Build conversations with a turn-taking
   model (hand-overs, interruptions, backchannels) instead of bursts. The candidate is
   [FastMSS](https://github.com/popcornell/FastMSS), open source, which takes utterances with word
   timestamps. What the literature measured for DiCoW
   ([Mind the Gap, 2026](https://arxiv.org/abs/2605.15442)):
   - Turn-taking realism and *boosted* overlap mattered most.
   - A diverse mix of sources beat an exact domain match.
   - Synthetic pre-training followed by real fine-tuning was best: 8.7% against 9.9% tcpWER for
     real data alone.

   [A timing study](https://arxiv.org/html/2607.08371) found that how often overlap happens
   matters more than how long each overlap lasts. Match the sustained talk-over of the >15%
   bucket, not the 0.3 s bursts of D95.
3. **Noise augmentation (owner's decision: part of the pipeline).** Background noise at a chosen
   SNR, drawn from the distribution Brouhaha measured on the corpus (D87), from a non-speech noise
   set such as MUSAN's music and noise subsets. Reverberation from room impulse responses is
   optional. Expect a modest gain for recognition: Mind the Gap measured −0.3 points of tcpWER from
   noise on Whisper-based DiCoW and none from reverberation, which matters for diarization instead.
   Our own clip classes found noise splits WER only through crosstalk. Measure it anyway, on the
   SNR buckets, with the clean-clip no-harm check.
4. **LLM-written conversations spoken by TTS (watch, not build).** 67 h of real plus 636 h of
   synthetic Hungarian beat a model trained on 2,700 h
   ([Conversations that Never Happened](https://arxiv.org/abs/2606.03957)). This needs a
   code-mixed Nepali TTS with many voices, which does not exist yet.

## D. Models for overlapped speech (scored on A's labels)

A's labels will not come from separation (A failed, 2026-09-22), so per-speaker references for
scoring D have to come from somewhere else first.

1. **DiCoW v3.3 and SE-DiCoW** (Brno University of Technology;
   [model](https://huggingface.co/BUT-FIT/DiCoW_v3_3),
   [SE-DiCoW](https://arxiv.org/html/2601.19194v1),
   [training code](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)).
   - **How it works.** Whisper-large-v3-turbo (0.9B) conditioned on diarization masks (silence,
     target, other, overlap) at every encoder layer. SE-DiCoW also enrols the target's clearest
     stretch, which roughly halves tcpWER against DiCoW.
   - **Why it fits.** The conditioning it needs already exists: the Modal pyannote turns
     (D78/D79), joined by time. It stays multilingual after English-only fine-tuning.
   - **Licence.** Weights CC-BY-4.0, code Apache-2.0.
   - **Risk.** Whisper is weaker on our Nepali than Flex (fine-tunes on the 2026-09-12 gold: 14.62
     against 11.44). B is what would close that gap.
   - **Order.** Zero-shot first, then fine-tune on C's data.
2. **SOT-DiCoW** ([paper](https://arxiv.org/abs/2510.03723)). A DiCoW encoder with one shared
   decoder that writes speaker-tagged text. On heavy synthetic overlap it beats DiCoW (17.2 against
   32.1 cpWER on 3-speaker mixtures); on real meetings it loses. A follow-up to D1, not a first
   step.
3. **Target-voice extraction, then Flex** (the old item 2). Separation in front of the existing
   Flex. A1 stopped before its third measurement, and its tracks sometimes held two voices
   (findings.md, 2026-09-22), so this is not promising with MossFormer2. The caveat: separation artifacts hurt a recogniser trained on
   clean speech ([2025](https://arxiv.org/abs/2503.17886)).
4. **The multitalker Parakeet method**
   ([NVIDIA, 0.6B](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)).
   - **How it works.** Speaker kernels are built from the diarizer's activity and injected into a
     FastConformer encoder, one model instance per speaker. It streams, and it is built to be
     fine-tuned from a strong single-speaker model.
   - **Limits.** English only, under NVIDIA's open model licence.
   - **What we would use.** The method, applied to Flex's conformer encoder or to a student
     from B.
5. **Commercial recognisers that diarize, as zero-shot baselines only.** No configured route
   diarizes (D52), and none should be switched on for this. A one-off cpWER baseline would still
   go through a named route and `llm_requests` (invariant 6).

## E. How any of this is measured

- **Per-speaker WER.** cpWER, tcpWER and ORC-WER (e.g. [MeetEval](https://github.com/fgnt/meeteval)),
  each speaker's stream folded by `app/services/fold.py`, split into S/D/I.
- **Buckets.** Every number by crosstalk bucket, with the clean bucket as the no-harm check.
- **Pairing.** Paired against a baseline and resampled by episode (`sweep.paired_bootstrap`).
- **A selection split that holds crosstalk.** Val has 4 clips over 15% overlap, so a winner chosen
  on val is chosen on clean speech.
- **A cost term** (latency, GPU, CPU inference, paid calls) in every decision rule. A small gain
  does not buy a large cost.

## Retired items

The roadmap before 2026-09-22 numbered its items 1–6. Code comments cite them by number.

- **Item 1 — clip classes.** In the standard pipeline since 2026-09-15 (D87). Open when retired:
  listen to the band-limited podcast clips, which score better within their episodes; check the
  voice links by ear.
- **Item 2 — target-voice extraction.** Now D3, measured inside A1.
- **Item 3 — synthetic augmentation for noise and crosstalk.** Crosstalk swept 2026-09-22 with no
  effect (D95, D96; findings.md). Noise and realistic mixing are now C.
- **Item 4 — newer architectures.** Now B and D.
- **Item 5 — held-out voices and microphones for gold.** Done 2026-09-16.
- **Item 6 — spelling convention.** Folded 2026-09-17 (D89, fold-v3). Open when retired: a
  listening check by native speakers; the references still mix both forms.
