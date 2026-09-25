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

**Status, 2026-09-24 (D100).** A stopped. The diarizer's turns cannot be trusted, neither which
speaker said a word nor where a turn changes, so gold stays single-stream: everything said, in time
order, with overlap spans marked. The owner put overlap models on hold as out of scope for the
code-switching paper.

**Order, set by the owner (2026-09-24):**
1. **B.** Distil the Flex fine-tune.
2. **C.** The augmentation pipeline.
3. **F.** Fiddle with the diarizer, once B and C are mastered.
4. **Only if F works:** the custom architectures ([custom-arch.md](custom-arch.md)) and the overlap
   models of D. Both are conditioned on diarization, so both need a diarizer that can be trusted.

E is how all of it is measured. Sections are lettered so they do not collide with the retired
numbered items that code comments still cite, which is why F comes after E.

## A. Per-speaker labelling as a multitrack editor (stopped, D100)

Built on 2026-09-23 (D98) with voiceprint suggestions (D99), and piloted on gold crosstalk. On
2026-09-24 the owner listened to the diarizer's per-word attribution and ruled it unusable: turns
are hit or miss, and in crosstalk the per-word join decides by rule, not by audio (findings.md,
*Diarization cannot say who said a word*). The speakers queue is closed. The editor is kept on
purpose (the owner's call): if F yields turns that can be trusted, it is the tool that uses them.

## B. Distil the Flex fine-tune into other architectures (priority 1)

Flex is the only model that is good at Nepanglish, and it decodes whole utterances. Many stronger
designs were never trained on Nepali: streaming transducers, diarization-conditioned models like D1
and D4. Teaching one of them Nepali from Flex would open those. Word timestamps are not the reason:
the harness's CTC forced aligner (`app/services/forced_align.py`, MMS-300m, D32) already places any
transcript's words on its clip, Flex's included, and it is what times the fused seed. B does not wait on A,
because it is measured on single-speaker WER. Crosstalk is out of scope for B.

**Goal.** A student whose single-speaker WER matches Flex's is the target. A student that had
never heard Nepali and still matches it would be the best outcome.

### What distillation means here

- **Fine-tuning** is a pretrained student plus human labels. 04a and 04c were both fine-tunes.
- **Distillation** is a pretrained student plus labels written by a better model, the teacher
  (the current Flex fine-tune, with its greedy+retry decoder).

On the 30 h already labelled, distillation adds nothing: Flex's labels (about 6.5% WER) are worse
than the verified ones. It pays off only on audio nobody has labelled, so Flex becomes a label
factory and the student learns from far more audio than could be labelled by hand.

**Soft distillation is skipped.** Matching the teacher's per-token probabilities needs a shared
tokenizer and a shared decoding step, and no candidate shares either with Flex (AED against
transducer or CTC, different vocabularies). Sequence-level pseudo-labels are used instead. They
also did most of the work in [Distil-Whisper](https://arxiv.org/abs/2311.00430).

### Students

Checked on 2026-09-24:

| Student | Knows Nepali? | Writes our text as it is? | Notes |
|---|---|---|---|
| **Parakeet-TDT-0.6B-v2** (NVIDIA, CC-BY-4.0) | no | no: English only | FastConformer TDT trained on 120k h of English. Needs a new tokenizer and a reinitialised decoder ([Hindi recipe](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/45), which started from v2). Word timestamps from the transducer, no repeat loops, fast on CPU. A cache-aware streaming FastConformer takes the same recipe, and the multitalker variant (D4) uses the same encoder. v3 adds 24 European languages, none in a script we use, so it was not picked. |
| **IndicConformer** (AI4Bharat, MIT) | yes | no: no Latin at all | The `ne` checkpoint ([repo](https://github.com/AI4Bharat/IndicConformerASR), 523 MB `.nemo`) is a multilingual hybrid CTC/RNN-T Conformer-L (17 layers, d_model 512, 4x subsampling) with an aggregate tokenizer: 22 BPE vocabularies of 256 tokens. None of the 5,632 tokens is Latin, and the Nepali vocabulary has no `।` and no digits. So it needs a new tokenizer too. What it adds is an encoder that has heard Nepali. The README says it loads only with AI4Bharat's NeMo fork (`nemo-v2`). It does not need to: the notebook builds a stock NeMo hybrid model from the checkpoint's config with stock heads and copies only the encoder, which trains (2026-09-24). |
| **Whisper-large-v3-turbo** (OpenAI, MIT) | weakly (123% zero-shot, loops) | yes: byte-level BPE | 04a scored 14.62 against Flex's 11.44 on the 2026-09-12 gold. It is DiCoW's backbone, so a Nepali-strong Whisper feeds straight into D1. Costs: 800M parameters, every clip padded to 30 s, loops. Its decoder stops at 448 positions, and Devanagari costs several tokens a character: 9 train labels do not fit, and 5 gold clips cannot be written whole in one pass. |
| **Qwen3-ASR-0.6B** ([Alibaba](https://github.com/QwenLM/Qwen3-ASR), Apache-2.0) | no; Hindi among its 30 languages | yes: byte-level BPE | An audio encoder feeding a Qwen3 decoder, the 0.6B picked over the 1.7B for being Parakeet's size and smaller than Flex. Zero-shot it writes rough Nepanglish already. Streams through vLLM only; its forced aligner covers 11 languages, not Hindi or Nepali. Its prompt names the language, and `language None` means "no speech", so ours is fixed at `language Nepali`. `qwen-asr` pins transformers 4.57.6, so it runs in its own runtime. |

**Pick.** Parakeet-v2 is the main bet, with IndicConformer next to it at step 0 as the safety net.
Both need a new tokenizer, so they share one: a SentencePiece model trained on train-split labels
only, which is allowed to write `।` and Devanagari digits. With the same tokenizer, step 0
compares an English encoder with a Nepali one, not two vocabularies. Precedent for the English
encoder: Flex is built on `canary-1b-v2`, which had no Nepali and was taught it later. Qwen and
Whisper keep their own tokenizers: step 0 for them is a plain fine-tune, and they make the
comparison across architectures (transducer against encoder-decoder and decoder-only) the paper
reports. `notebooks/Distill.ipynb` trains the transducers, `notebooks/DistillHF.ipynb` the other two,
with the same scoring cells and the same Flex reference.

**Is 30 h enough?** For IndicConformer, possibly: Flex's learning curve was flat (4.6 h scored
about the same as 18.3 h), but Flex already knew Nepali. For Parakeet, probably not. That curve
says nothing about a model learning a new script and language, which usually takes hundreds of
hours. This is an estimate from general experience, not a measurement: step 0 measures it, and
step 4's curve says how much audio closes the gap.

### Workflow

0. **Plain fine-tune, no new data.** Fine-tune each student on the 30 h of verified labels. Score
   gold and val, folded and raw, split into S/D/I. This is the baseline distillation must beat,
   and it may end B early. Re-run Whisper-turbo on the current data too, since its 14.62 was
   measured on the old gold. **Done 2026-09-25** ([findings](findings.md)): no student meets
   the bar. Whisper-turbo is closest (gold 19.77 against Flex's 11.56, +8.21 [+6.46, +9.98]), and
   every student's gap is largest on pure Nepali, so step 1's audio must be Nepali speech from
   new voices.
1. **Collect unlabelled audio.**
   - Nepali podcasts and tech reviews from YouTube, in the corpus's genres.
   - Downloaded outside the harness, since D86 allows only two ways in. The corpus is files, never
     rows (D101): `scripts/prepare_distill_audio.py` cuts it like ingest,
     `scripts/screen_distill_audio.py` quarantines sources that sound like a gold voice, and
     `scripts/upload_distill_corpus.py` sends the cleared ones to a private HF dataset for
     `notebooks/Teacher.ipynb` (steps 2-3).
   - Cut into clips of 20 s or less with the same silero VAD.
   - Collected by the owner as whole playlists, audio only, with yt-dlp's info JSON (video id,
     channel, playlist) kept for the gold check and the provenance record. Prefer shows the corpus
     lacks: more episodes of the same speakers stopped helping on the learning curve.
   - **No channel or voice that appears in gold.** Gold is held out by speaker, and a scraped
     episode of the same show would leak it. Gold and val audio are never pseudo-labelled (D76).
     Gold holds Chill Pill clips, one Prime Television episode and 82 shorts/reels with no
     recorded source (2026-09-24). Skip the first two channels. Screen every scraped episode
     against gold's voices with the D99 voiceprints, on clean single-speaker stretches, where
     they are reliable.
   - First tranche: about 130 h of source for about 100 h of kept speech, collected while step 0
     runs. More only once step 0 says the gap is worth closing.
2. **Teacher decode.** Flex with the standard decoder (greedy and loop retry), keeping its
   per-token log-probs.
3. **Filter the labels, cheapest first.**
   - Drop clips the loop retry fired on.
   - Drop clips whose tokens per second fall outside the range seen in train.
   - Drop the least confident 10–20% by mean log-prob.
   - Add an agreement check with a second model, or a
     [label-free filter](https://arxiv.org/abs/2407.01257), only if step 4 shows noisy labels hurt.
   - Never send this audio through the paid routes.
4. **Train the student** on the pseudo-labelled audio plus the 30 h, with the human labels
   weighted up. Build a learning curve in hours of pseudo-labelled audio (100, then 300, then
   1000 h) and stop when it flattens.

**Success bar.** The student's gold and val WER falls within Flex's episode-bootstrap CI, per clip
class, folded and raw, split into S/D/I. Only then are its extras (streaming, timestamps,
conditioning) worth their cost.

**Raw-WER caveat.** Flex's tokenizer cannot write `।` or Devanagari digits, so its pseudo-labels
never hold them, while the human labels do. A student trained on both learns the two conventions
at once. Folded WER hides this; raw WER does not.

**Cost** (extrapolated from 04c, not measured). Teacher decode: Flex runs at a real-time factor of
about 0.009 on an A100, so 300 h is about 3 GPU-hours. Student training: a few hours per run for a
0.6B student on 300 h. The real cost is collecting the audio and checking it for gold overlap. A TPU is
not worth it: the decode is already about 9 A100-hours per 1000 h, Flex's autoregressive decode
with varying clip lengths and the loop retry recompiles constantly under XLA, the CPU (audio
decoding, VAD) is the likelier bottleneck, and NeMo students do not train on TPU.

## C. Augmentation pipeline (priority 2)

What the sweep taught: augmentation pays only when the model has a way to use it, either
conditioning that says whom to follow or an output format that writes both voices. The pipeline
below is built to feed D, not single-stream Flex. Gold and val audio are never a source of mixed-in
speech or noise (D76).

1. **Two-voice mixes from whole verified clips.** This lifts D95's blocker. The only text D95 had
   for a donor burst was an unverified recogniser's. If the second voice is a whole verified train
   clip, or a word-aligned stretch of one (`app/services/forced_align.py`), both transcripts are
   verified. That gives labels for both voices, as serialized output and target-speaker models
   need. The measured mixer already exists (`notebooks/src/xtalk.py`, D96).

   The same mixes are where D would be developed and selected, since who said what is known by
   construction (D100). They prove nothing about real overlap -- D96 showed synthetic crosstalk
   does not transfer -- so a real-crosstalk check comes last.
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

## D. Models for overlapped speech (on hold, D100; only if F works)

There will be no per-speaker references for real crosstalk: separation was a dead end
(2026-09-22/23), and so was attribution from diarization (2026-09-24). If D resumes, models are
selected on C's synthetic mixes. They are then checked on real crosstalk in two ways: recognition
against the single-stream gold, and attribution by grading the model's output by ear.

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

## F. Fiddling with the diarizer (priority 3, after B and C)

The owner wants to understand the diarizer before giving up on it. This is not a fix to build; it is
one bounded test, done as its own experiment.

- **How pyannote decides.** A segmentation model finds up to three local speakers in each 10 s
  window. Each local speaker gets a voice embedding, and all of them are clustered across the
  episode. Words never enter it: the harness joins them to turns by time. Clustering can only fix
  mistakes made at the clustering step. It cannot fix a turn boundary the segmentation model put
  in the wrong place, or a word said while both people talk.
- **Over-cluster, then merge by ear.** Force more clusters than people (6 for a two-person
  interview). If the host's excited delivery ends up as its own cluster, listening to a few samples
  per cluster and merging them fixes it in minutes per episode instead of word by word. At the
  default setting, the interview's 8 extra clusters were not pieces of the host (cosine ≤ 0.26 to
  both main voices), so this is unproven.
- **The exclusive track.** Pyannote also returns one speaker per moment, chosen from its own frame
  scores, for joining to ASR word timestamps. The harness never stored it. On the interview's 537
  words decided by the join's tie rule, it moved 282 to the other voice. That is either evidence
  from the audio or noise; only listening can tell.
- **The rule, fixed in advance.** On episode 205: one run with 6 clusters, 5 samples per cluster,
  and the exclusive track's picks on the tie words, all judged by ear. If every cluster is one
  person, cluster merging is viable and the multitrack editor comes back. If clusters mix people,
  word attribution from diarization is closed for good.

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
