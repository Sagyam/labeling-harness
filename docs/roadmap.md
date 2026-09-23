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
1. **A.** Per-speaker labelling as a multitrack editor.
2. **B.** Distilling the Flex fine-tune.
3. **C.** The augmentation pipeline.

D (models for overlapped speech) is scored on A's labels, so it waits for A. E is how all of it
is measured. Sections are lettered so they do not collide with the retired numbered items that code
comments still cite.

## A. Per-speaker labelling as a multitrack editor (priority 1)

**The problem.** Gold labels in overlap are a single stream: the stronger voice, plus some of the
weaker one, with no speaker attribution (D95). Every multi-talker model is scored per speaker
(cpWER, tcpWER, ORC-WER). A model that correctly writes both voices is otherwise charged
insertions, which is the D96 trap again. Relabelling per speaker is the fix, but labelling
crosstalk means listening to two people at once. That is the owner's bottleneck, and the reason
the current labels drop part of the weaker voice.

**The idea (owner's proposal, 2026-09-23).** Attribute words the way a DAW arranges a song from
tracks. The x axis is time and the y axis is speaker: each speaker is a lane, and each word is a
block on a lane, spanning the time it was spoken. Played together, the lanes are the seed
transcript, the "final mix".

It rests on an observation from labelling: the pipeline is usually right about *what* was said
and *when*. What is disputed is *who* said it, and that comes from the diarizer. So the annotator
does two things, and only two:
- **Move a word** to the lane of the person who said it. Every word starts on the lane of the
  diarized turn it falls in (D78, D79), joined by time as the editor's speaker colours already
  are, so most words need no move.
- **Add a word the seed does not have**, usually the weaker voice's in overlap: type it, drop it on
  a speaker's lane at the moment it was said, and drag its edges to give it a length. The
  karaoke animation then shows whether the block lines up with what is heard.

**Speakers are fixed.** Speaker identity is tracked globally (D78's episode-wide "Speaker 1, 2,
...", and D87's voices across episodes), so the editor cannot create or delete a speaker. Its lanes
are the episode's speakers; the annotator moves words between them and adds words to them.

**Time resolution: a 10 ms grid.** Blocks snap to 10 ms. That is finer than anything downstream
needs: the CTC aligner's frames are coarser, a listener cannot place a word boundary more
precisely than a few tens of milliseconds, and tcpWER's collar is measured in seconds. Storage cost
does not depend on the grid: a word is a start and an end whatever their precision, stored as the
seconds-with-millisecond floats that word spans already use. So 10 ms is a choice about how the
editor feels, not about storage.

**What this gives.** Every word carries a speaker and a span, so each clip yields per-speaker
streams with word timings: what cpWER and tcpWER need, and what D is scored on. The single-stream
text stays derivable (every lane, in time order). The words moved per clip are also a measurement
of their own: a word-level diarization error rate on Nepanglish conversation.

**What needs settling before it is built** (a decision entry replacing D95's convention, and a
migration with a working downgrade):
- **When the diarizer is wrong about speakers, not words.** With speakers fixed, two kinds of
  diarizer mistake cannot be fixed in this editor. A speaker it invented is only an empty lane,
  which is harmless. Two people it merged into one speaker cannot be split, because splitting
  means a new speaker. Such a clip needs a flag, and an episode-level fix outside this editor.
- **Which lanes a clip shows.** The speakers diarized in the clip, plus a way to bring in another
  of the episode's speakers when the diarizer missed that person entirely. This shows an existing
  speaker; it does not create one.
- **Candidates from the recognisers.** The three ASR routes sometimes hear a word in overlap that
  fusion dropped. Showing those as unplaced blocks to drag onto a lane would recover weak-voice
  words without typing. Only words the seed lacks would be offered, so nothing is scored twice.
- **Both voices saying the same word** (a shared "हो" in crosstalk) is one word in the seed but
  two in the truth. The editor needs a *copy to lane*, as a DAW duplicates a clip.
- **Resizing seed words.** Seed timings rarely need touching, but they are least reliable in
  overlap, where the aligner follows one stream through two voices. The same edge drag that sizes
  a new word should work on any word.
- **Storage.** Invariant 2 holds: an attributed label is a new `segment_labels` row, with a speaker
  and a span per word next to the text, written with its event and audit rows in one transaction
  (invariant 8). A separate label version keeps the existing single-stream labels current for
  single-stream scoring. Attribution is always verified by ear, never screened (invariant 5). No
  new status field (invariant 3). Hypotheses and diarization runs stay untouched; the correction
  lives only in the label.
- **Which clips.** Only clips with two or more diarized speakers or measured overlap need lanes:
  about 250 of 750 gold clips. A single-speaker clip is one lane and no work, unless the diarizer
  missed a second voice there, which the queue cannot surface.

**Pilot first.** Owner's rule, as for every A so far: a throwaway prototype on a handful of gold
clips before anything is built into the harness. It measures words moved, words added and time
per clip (the client-reported `duration_ms`, as for the throughput baseline).

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
   need. The measured mixer already exists (`notebooks/src/xtalk.py`, D96).

   The same mixes have a second use, as a **test bench for A**: who said what is known by
   construction, so they can tell whether the multitrack editor, its pre-assignment or a metric
   works at all before any ears are spent on real clips. They prove nothing about real overlap --
   D96 showed synthetic crosstalk does not transfer.
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

A's labels will not come from separation (a dead end, 2026-09-22/23), so per-speaker references for
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
3. **Target-voice extraction, then Flex** (the old item 2; dead end for now, like the separation
   pilot). Separation in front of the existing Flex. The pilot stopped before its third
   measurement, its tracks sometimes held two voices, and on 2026-09-23 three separators gave
   either bleed or broken words (findings.md). Separation artifacts also hurt a recogniser trained
   on clean speech ([2025](https://arxiv.org/abs/2503.17886)). Revisit with separation research
   after the ASR paper.
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
- **Item 2 — target-voice extraction.** Now D3; the separation pilot never reached it.
- **Item 3 — synthetic augmentation for noise and crosstalk.** Crosstalk swept 2026-09-22 with no
  effect (D95, D96; findings.md). Noise and realistic mixing are now C.
- **Item 4 — newer architectures.** Now B and D.
- **Item 5 — held-out voices and microphones for gold.** Done 2026-09-16.
- **Item 6 — spelling convention.** Folded 2026-09-17 (D89, fold-v3). Open when retired: a
  listening check by native speakers; the references still mix both forms.

The first version of A (2026-09-22) had three parts, which findings.md cites:

- **A1 — separation pilot; A2 — separated tracks as child clips.** A dead end: MossFormer2,
  TF-GridNet and TF-Locoformer gave either bleed from the other voice or broken words, on every
  heavy-crosstalk clip (findings.md, 2026-09-22 and 2026-09-23). A2 was never built. Separating
  voices in a real recording is worth pursuing as research after the ASR paper is published; for
  now it only adds complexity.
- **A3 — attribute words instead of separating audio.** Replaced by the multitrack editor, which
  is the same idea made concrete. Its second path, synthetic overlap as a test bench, moved to C1.
