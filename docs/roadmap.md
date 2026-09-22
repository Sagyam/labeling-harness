# Research roadmap: overlapped speech

What is next for the ASR work on the Nepanglish corpus, as replanned on 2026-09-22. What has been
measured so far is in [findings.md](findings.md). The earlier roadmap is retired (list at the end).

**Where this starts.** Single-speaker recognition is largely solved for this corpus. The Flex
fine-tune scores 6.5% folded WER on clean gold clips. Crosstalk is what remains: 17% at 5–15%
overlap and 29% above 15%, on 205 gold clips. The D96 sweep showed that mixing more synthetic
crosstalk into a model that writes one stream of text for one speaker does not move those numbers
(findings.md, *Synthetic crosstalk does not move real crosstalk*). The model is never told which
voice the label wants, so it hedges: fewer insertions, more deletions. This chapter is about
models that *are* told (by diarization, enrolment, or an output that writes every speaker), and
the data that trains them. Depending on how strong the results are, it may become its own paper.

Sections are lettered so they do not collide with the retired numbered items that code comments
still cite.

## A. The reference problem (prerequisite for everything below)

Gold labels in overlap are a single stream: the stronger voice, plus some of the weaker one, with
no speaker attribution (D95). Every multi-talker model is scored per speaker (cpWER, tcpWER,
ORC-WER), so this has to be settled before B or C can be measured. A model that correctly writes
both voices is otherwise charged insertions, which is the D96 trap again.

- **(a) Relabel the crosstalk gold per speaker.** About 205 clips over 5% overlap (261 with any
  overlap). Each voice gets its own verified line, keyed to the diarized turns (D78). This unlocks
  every model in C.
- **(b) Declare that gold transcribes only the main voice in overlap.** Single-stream Flex with
  target-voice extraction (C3) stays a fair contest. Multi-talker models lose their advantage.
- **(c) The owner's proposal**, still to be written down.

Invariants still hold: gold is chosen by hand, clip by clip (invariant 4, D71). A relabel is a new
`segment_labels` row, never an overwrite (invariant 2).

## B. Augmentation pipeline

What the sweep taught: augmentation pays only when the model has a way to use it, either
conditioning that says whom to follow or an output format that writes both voices. The pipeline
below is built to feed C, not single-stream Flex. Gold and val audio are never a source of mixed-in
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

## C. Models for overlapped speech

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
     against 11.44). D is what would close that gap.
   - **Order.** Zero-shot first, then fine-tune on B's data.
2. **SOT-DiCoW** ([paper](https://arxiv.org/abs/2510.03723)). A DiCoW encoder with one shared
   decoder that writes speaker-tagged text. On heavy synthetic overlap it beats DiCoW (17.2 against
   32.1 cpWER on 3-speaker mixtures); on real meetings it loses. A follow-up to C1, not a first
   step.
3. **Target-voice extraction, then Flex** (the old item 2).
   - **How it works.** Separate the target voice with MossFormer2
     ([ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)) or a
     target-speaker extractor told who the target is by an enrolment clip, then decode with the
     existing Flex.
   - **Cost.** No training: the cheapest test in this file.
   - **Caveats.** Separation artifacts hurt a recogniser trained on clean speech
     ([2025](https://arxiv.org/abs/2503.17886)), and two same-room voices within ±3 dB is the
     hardest case.
   - **Measure.** WER per crosstalk bucket, and on clean clips, where extraction must not hurt.
4. **The multitalker Parakeet method**
   ([NVIDIA, 0.6B](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)).
   - **How it works.** Speaker kernels are built from the diarizer's activity and injected into a
     FastConformer encoder, one model instance per speaker. It streams, and it is built to be
     fine-tuned from a strong single-speaker model.
   - **Limits.** English only, under NVIDIA's open model licence.
   - **What we would use.** The method, applied to Flex's conformer encoder or to a student
     from D.
5. **Commercial recognisers that diarize, as zero-shot baselines only.** No configured route
   diarizes (D52), and none should be switched on for this. A one-off cpWER baseline would still
   go through a named route and `llm_requests` (invariant 6).

## D. Distil the Flex fine-tune into other architectures

Flex is the only model that is good at Nepanglish, and it is closed in two ways: it decodes whole
utterances, and it reports no word timestamps. Many stronger designs were never trained on Nepali:
streaming transducers, models that report word timestamps, diarization-conditioned models like
C1 and C4. Teaching one of them Nepali from Flex would open all of those.

- **Teacher.** The current Flex fine-tune, with its greedy+retry decoder.
- **Students, in order of promise:**
  - **Whisper-large-v3-turbo.** It already knows Nepali's script, and it is DiCoW's backbone: a
    Nepali-strong Whisper feeds straight into C1.
  - **Parakeet / FastConformer (TDT or CTC).** Streaming, and word timestamps from the alignment.
    The English tokenizer has no Devanagari, so it gets a new SentencePiece tokenizer and a
    reinitialised decoder
    ([Hindi recipe](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/45)). The
    multitalker variant (C4) is built on the same encoder.
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

## E. How any of this is measured

- **Per-speaker WER.** cpWER, tcpWER and ORC-WER (e.g. [MeetEval](https://github.com/fgnt/meeteval)),
  each speaker's stream folded by `app/services/fold.py`, split into S/D/I.
- **Buckets.** Every number by crosstalk bucket, with the clean bucket as the no-harm check.
- **Pairing.** Paired against a baseline and resampled by episode (`sweep.paired_bootstrap`).
- **A selection split that holds crosstalk.** Val has 4 clips over 15% overlap, so a winner chosen
  on val is chosen on clean speech.
- **A cost term** (latency, GPU, CPU inference) in every decision rule. A small gain does not buy a
  large cost.

## Suggested order

1. Settle A.
2. Zero-shot, no training: C1 on our existing speaker turns, and C3 before Flex.
3. Start D in parallel. It does not wait on A, because it is measured on single-speaker WER.
4. If C1 shows promise: build B (two-voice mixes, timing, noise) and fine-tune, synthetic first,
   then real.

## Retired items

The roadmap before 2026-09-22 numbered its items 1–6. Code comments cite them by number.

- **Item 1 — clip classes.** In the standard pipeline since 2026-09-15 (D87). Open when retired:
  listen to the band-limited podcast clips, which score better within their episodes; check the
  voice links by ear.
- **Item 2 — target-voice extraction.** Now C3.
- **Item 3 — synthetic augmentation for noise and crosstalk.** Crosstalk swept 2026-09-22 with no
  effect (D95, D96; findings.md). Noise and realistic mixing are now B.
- **Item 4 — newer architectures.** Now C and D.
- **Item 5 — held-out voices and microphones for gold.** Done 2026-09-16.
- **Item 6 — spelling convention.** Folded 2026-09-17 (D89, fold-v3). Open when retired: a
  listening check by native speakers; the references still mix both forms.
