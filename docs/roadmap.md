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
3. **G.** A text-only language model at decode time, and the low-risk fold rules (added
   2026-09-27, after C at the owner's request).
4. **F.** Fiddle with the diarizer, once B and C are mastered.
5. **Only if F works:** the custom architectures ([custom-arch.md](custom-arch.md)) and the overlap
   models of D. Both are conditioned on diarization, so both need a diarizer that can be trusted.

E is how all of it is measured. Sections are lettered so they do not collide with the retired
numbered items that code comments still cite, which is why F and G come after E.

**How a model is trained and evaluated (D105, 2026-09-30).** B and C are no longer run ad hoc:
the notebooks are numbered in the order they run, and that order is the protocol.

| Step | Notebook | Roadmap section |
|---|---|---|
| 1–2 | `01_EDA`, `02_Sociolinguistics` | the corpus |
| 3 | `03a_Flex_Train` to `03e_Flex_Ship` | Flex retrained, scored on the public sets, C's ablation, blended, one teacher frozen |
| 4 | `04_PreDistill` | B step 1, and each clip's overlap measured |
| 5 | `05_Teacher` | B steps 2–3 |
| 6 | `06a`–`06f_Student_*` | B step 4, one notebook per student |
| 7 | `07_Report` | E: every model in one table |

Every model goes through one evaluation (`notebooks/src/evalkit.py`), and every choice is made on
val by a rule fixed before the result. Each notebook has a smoke switch. **Status, 2026-10-04:**
03a–03e have run on the 2026-09-30 export. 03c kept no stage, so vanilla-s1 went into 03d, which
chose blend-075, and 03e froze it as the teacher with an int8 CPU export (findings.md, *The
teacher frozen*). 04's overlap pass and 05 have labelled the first tranche: 79.5 h kept of 102.1 h
at `MAX_OVERLAP_SHARE` 0.05 (findings.md, *The teacher labels the unlabelled corpus*). **Next:** the
students' 100 h point, 06a–06f.

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
  (the Flex fine-tune p00-s0, decoding greedily; a clip whose output loops is dropped, not
  retried, in step 3).

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
| **Omnilingual CTC** (Meta; 300M or 1B) | yes: 1,600+ languages | to check | Added for step 4 (2026-09-26), not in step 0. A self-supervised wav2vec 2.0 encoder with a CTC head: the one family the other students leave out, and the one least able to use the context a code-switch needs, since CTC predicts each token independently. The 1B reached ~16.6% on val in the 2026-09-14 bake-off before it was stopped at epoch 6, under a different protocol. |
| **Small Conformer from scratch** (NeMo) | no: no pretraining at all | own tokenizer | Added for step 4 (2026-09-26). The shared SentencePiece and heads of the transducers, a small config and random weights: it asks whether pretraining still matters once 145 h of pseudo and human labels exist. `06f_Student_Conformer` builds it from IndicConformer's config, resized, without the encoder copy (not yet run). |

**Pick, before step 0.** Parakeet-v2 was the main bet, with IndicConformer next to it as the
safety net. Step 0 overturned it (findings.md, 2026-09-25): Whisper-turbo came closest to Flex, and
Parakeet's English encoder did not generalise on 30 h. Step 4 carries every student forward
anyway, since the paper wants the curves, not a winner.
Both need a new tokenizer, so they share one: a SentencePiece model trained on train-split labels
only, which is allowed to write `।` and Devanagari digits. With the same tokenizer, step 0
compares an English encoder with a Nepali one, not two vocabularies. Precedent for the English
encoder: Flex is built on `canary-1b-v2`, which had no Nepali and was taught it later. Qwen and
Whisper keep their own tokenizers: step 0 for them is a plain fine-tune, and they make the
comparison across architectures (transducer against encoder-decoder and decoder-only) the paper
reports. Each student has its own notebook (`06a_Student_Whisper` to `06f_Student_Conformer`),
generated from one builder, with the same stages, the same scoring and the same frozen teacher
(D105).

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
     rows (D101). The owner uploads `distill.zip` to the dataset repo, and
     `notebooks/04_PreDistill.ipynb` cuts it in Colab with ingest's own code into `distill/`, which
     `notebooks/05_Teacher.ipynb` labels (steps 2-3). First tranche: about 100 h (2026-09-25).
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
2. **Teacher decode** (`notebooks/05_Teacher.ipynb`). The model `teacher.json` names (D105),
   greedy, keeping each clip's mean log-prob. No loop retry: step 3 drops a looping clip anyway,
   so a retry would be wasted GPU.
3. **Filter the labels, cheapest first.**
   - Drop clips overlapped beyond `MAX_OVERLAP_SHARE` (D105). Crosstalk is where the teacher is
     weakest. `04_PreDistill` measures each clip's overlapped share with the harness's detector and
     prices each threshold in hours per channel; the owner picks the threshold from that table.
     Round-table shows are the most overlapped and bring the most voices, so a strict one can leave
     a solo commentator carrying the corpus.
   - Drop clips whose output loops.
   - Drop clips whose tokens per second fall outside the range seen in train.
   - Drop the least confident 10–20% by mean log-prob.
   - Add an agreement check with a second model, or a
     [label-free filter](https://arxiv.org/abs/2407.01257), only if step 4 shows noisy labels hurt.
   - Never send this audio through the paid routes.
4. **Train the student** on the pseudo-labelled audio plus the 30 h, with the human labels
   weighted up. Build a learning curve in hours of pseudo-labelled audio (100, then 300, then
   1000 h) and stop when it flattens. Settled on 2026-09-25 and revised 2026-09-26, before any of
   it was built:
   - **Every student, for the paper** (owner, 2026-09-26): step 0's four plus Omnilingual CTC and
     the small Conformer from scratch. The findings are the point, not a deployable model; Flex
     already is one. Parakeet stays because it pairs with IndicConformer: same tokenizer, heads and
     recipe, and only the encoder differs (English against Nepali), so its curve says whether the
     13 points the English encoder cost at 30 h shrink with pseudo-labels.
   - **Each student continues its human-label fine-tune (D106, the owner, 2026-09-30).** Three
     stages: *human* (the pretrained weights on the verified labels), *distill* (those weights on
     the mixture of pseudo-labels and human labels) and *distill-aug* (distill again, from the
     human stage's weights, with the recipe that won Flex's ablation). This reverses the rule of
     2026-09-26, fresh weights for every point: the difference between the first two stages is now
     the pseudo-labels together with the extra training. The human labels take half of every
     epoch's draws, so a student does not end on the teacher's errors alone.
   - **Run the 100 h point for every student first**, and go further only for students whose
     curves are still rising. Six students at ~145 h each is several A100-days.
   - **Weight channels by the square root of their hours; cap none** (owner, 2026-09-26). Every
     clip stays in the pool, and a channel is drawn in proportion to √hours rather than hours, so
     the first tranche's 22 h political commentator weighs about 2× a 5 h channel, not 4.4×. A
     hard 5 h cap would have kept about 40 of the tranche's ~90 usable hours, on evidence that
     does not transfer: the 2026-09-13 curve that more of the same voices stops helping measured
     Flex, and Whisper-turbo is 8 points short of it. Adopted without a capped-against-weighted
     comparison, at the owner's word. The exponent is a training setting. On the cut tranche (2026-09-26,
     102 h of speech) it gives What_s_With 21% of draws for 34% of the hours, Sudheer Sharma 18%
     for 24%, and the five movies 19% for 6%. That misweights both ways, since What_s_With's round
     tables bring new guests every episode and Sudheer Sharma is the only solo channel. The owner
     kept it for the first run: if the students disappoint, oversampled data is cut or more is
     added then.
   - **Resume from HF per epoch.** At 100 h and more a run takes many hours, which a lost runtime
     or power cut must not cost again. Built 2026-09-30 (`ftkit.HubResume`): a stage saves its
     weights and counters to a scratch model repo after every epoch and continues from them.
   - **Order of the first run:** `04_PreDistill` (its overlap pass, on the corpus already cut),
     then `05_Teacher` with `SMOKE` (its log-prob masking and `output_scores` memory have never run
     on a GPU), then the whole corpus, then the students. Done up to the students, 2026-10-04.

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

**Built 2026-09-26: `notebooks/src/augment.py`**, one switch per stage, all off until a notebook
raises its `p`: SpecAugment, speed, reverb (synthetic or a bank), channel (microphone response),
noise (a `NoiseBank`), gain, codec (MP3, Opus, AAC, mu-law through ffmpeg) and crosstalk. Crosstalk
is D96's mixer (`xtalk.py`) made configurable through `CrosstalkConfig`: window length, level gap
and share as ranges instead of the measured bursts, and whole verified train clips as the second
voice (`donor="clip"`), named in the result so a label for both voices can be built. The second
voice can also get its own room and microphone before it is mixed (`donor_reverb`,
`donor_channel`), because in real crosstalk it reaches the target's microphone from further away.
The corpus's overlap is near-symmetric, though (both voices within 3 dB), so always processing the
donor would teach "ignore the wetter voice"; keep those below p = 1 unless that is the experiment.
The defaults reproduce D96 exactly. Since 2026-09-30 it is wired into `03c_Flex_Augment`, which
ablates one stage per run on Flex, and into every student's third stage, which switches on the
recipe that won there (D105). The strengths in `ABLATION` are first guesses.

**Status, 2026-10-03.** 03c ran speed, reverb, codec, gain, channel and crosstalk on Flex, and none
cleared D109's bar. Speed came closest, at −0.07 on val. Noise was not run. Flex is already robust
to the acoustic conditions, and what remains on gold is mostly vocabulary. The crosstalk stage's
label interleaved two sentences with no speaker marker, and most heavy-crosstalk gold labels
predate D100 (findings.md, *Augmentation on Flex: six stages, nothing kept, and why*). Noise is
worth running only as insurance, judged on the noisy no-crosstalk gold bucket with a threshold set
beforehand.

**What the label says** (`CrosstalkConfig.label`, 2026-09-26). D96 kept the clip's own text, which
teaches a model to leave the other voice out; since D100, gold writes everything said, so a
single-stream model trained that way is charged a deletion for every word it was taught to drop.
`label="everything"` merges the target's and each donor clip's words in time order, punctuation
kept, into `info["text"]`, which the collate function trains on. The spans are the training
export's `label_words` (the seed's aligned words, or a realignment for an edited label), so it needs
whole-clip donors and skips any clip without them. `label="target"` stays for target-speaker models
(D), which are told whom to follow.

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
     stretch. That halves tcpWER against the original DiCoW, but against DiCoW v3.3 it gains
     little on real meetings (NOTSOFAR-1 26.6 to 26.1, AMI SDM 18.6 to 18.5 with a real
     diarizer); its large gains are on 3-speaker synthetic mixes (Libri3Mix clean 38.6 to 35.6).
   - **Why it fits.** The conditioning it needs already exists: the Modal pyannote turns
     (D78/D79), joined by time. Its English-only fine-tuning keeps whatever languages Whisper
     had, and for our Nepali that is almost nothing: Whisper-turbo scored 123% WER on gold
     zero-shot, with 271 looping clips.
   - **Licence.** Weights CC-BY-4.0, code Apache-2.0.
   - **Risk.** Whisper is weaker on our Nepali than Flex (fine-tunes on the 2026-09-12 gold: 14.62
     against 11.44). B is what would close that gap.
   - **Enrolment.** Take it from a voice checked by the who pass in F. Never take it from a cluster
     heard mixing two people; that was 10 of 128 in the 2026-09-28 pilot. A recurring voice can be
     enrolled from another of its episodes.
   - **Order (owner, 2026-09-28).** No zero-shot run: it would measure Whisper's missing
     Nepali, not the conditioning. DiCoW waits for B's Nepali-strong Whisper student. Its
     conditioning is then fine-tuned onto that student: single-speaker clips with all-target
     masks first, as a check that Nepali survives, then C's mixes.
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
- **Where per-speaker WER is valid.** Every per-speaker metric needs a reference that says which
  words each speaker said, and ORC-WER still needs overlapping speech split into each speaker's
  utterances. C's synthetic mixes have that exactly. Real gold does not (D100), and knowing who
  each cluster is (F, 2026-09-28) does not supply it. On real gold:
  - recognition is scored by merging a model's streams in time order against the single-stream
    reference;
  - attribution is scored by ear on a fixed sample.
- **Buckets.** Every number by crosstalk bucket, with the clean bucket as the no-harm check.
- **Pairing.** Paired against a baseline and resampled by episode (`sweep.paired_bootstrap`).
- **A selection split that holds crosstalk.** Val has 4 clips over 15% overlap, so a winner chosen
  on val is chosen on clean speech.
- **A cost term** (latency, GPU, CPU inference, paid calls) in every decision rule. A small gain
  does not buy a large cost.
- **Error mining, not yet built** (D110):
  - *Near miss or different word*, for a substitution: did the model hear something close
    (`cache`/`cage`) or unrelated? The rows carry `similarity` and both sides romanized, but the
    threshold must be calibrated on pairs the owner judges by ear, and similarity alone mostly
    finds grammar (findings.md, 2026-10-01).
  - *Notes while reading*: marking a pair as convention or real error from the page needs its
    own storage, and a decision on what a mark may change.
  - *S/D/I by language* (Nepali, English, numbers) as a reported block; the rows carry script,
    so it is a query.
  - *Reference audit*: public-set clips where the models agree with each other and not with the
    reference, top of the list heard. FLEURS, OpenSLR 54 and Common Voice references are the
    prompt the speaker read, not what was said.
  - `07_Report` prints the crosstalk and number blocks only; the confusion table and the
    attribution card are on the Models page.

## F. Fiddling with the diarizer (priority 4, after B, C and G)

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
- **Who is who: answered 2026-09-28, corpus-wide** (findings.md, *Same voice or not*). This
  answers the clustering half of F differently from the plan above. Blind listening to 95 cluster
  pairs showed that voiceprint similarity decides most same-person calls:
  - at 0.72 or more, 91% were one person;
  - below 0.55, 90% were two people;
  - 52 of 456 corpus pairs fall between the two (132 if the band starts at 0.49), 12–30 minutes of
    listening once, then a minute or two per new episode.

  10 of 128 clusters mix two people; no merge fixes those. The episode 205 test above (over-cluster,
  exclusive track) was not run.
- **Parked idea: voice identity from similarity plus a short listening pass.** Nothing is built.
  1. Link or merge automatically above the upper cut-off, and keep clusters apart below the lower
     one.
  2. Queue the pairs in between for the owner's ear, with the same page and 13 s per pair.
  3. Mark clusters heard as mixing two people. They are never used as a voice's audio.
  4. Replace the linker's single 0.6 cut-off (D87). It is wrong on 7 of 44 cross-episode pairs and
     never merges within an episode.

  **What it would fix:** voice counts (sociolinguistics), the gold/train voice overlap, reel voice
  dedup, and two inputs of D1: SE-DiCoW's enrolment audio and the target/other masks. A speaker
  split into two clusters marks the target's own voice as "other".

  **What it would not fix:** which words each person said. Turn boundaries and crosstalk are the
  *when* problem D100 closed. Sized 2026-09-28, it is about 20,000 events in the clips (12.5k
  speaker switches, 7.8k overlap spans), so it is only viable as an algorithm checked by a fixed
  exam of ~150–200 events, never by labelling. Per-speaker references for real gold, and with them
  cpWER/tcpWER/ORC-WER on real gold, still do not exist (E).

## G. Vocabulary at decode time, and the low-risk fold rules (priority 3, after C)

Both come from the 2026-09-27 benchmark error anatomy (findings.md, *Public Nepali benchmarks, the
error anatomy, and weight blending*). The owner parked them for later.

### G1. Shallow fusion with a text-only language model

**Why.** Words our train labels never contain are the largest cause of error on the public sets:
3.23 of the fine-tuned model's 11.20 points, plus 1.09 for words seen 1–9 times. Blending recovers
the vocabulary fine-tuning erased; a text model adds vocabulary neither model ever had. Nepali text
(news, Wikipedia, books) is plentiful; Nepali speech is not.

**How.** A small model trained on text alone scores each candidate next piece while Flex decodes,
and the two scores are added: `Flex + λ · text model + a per-word bonus`. Nothing inside Flex
changes. It needs beam search (4–8 hypotheses), because greedy decoding leaves the text model
nothing to re-rank.

**Steps.**
1. **Build the text.** Nepali news and Wikipedia plus our own train labels, so the colloquial
   register and our spelling are represented. Remove every benchmark reference (FLEURS comes from
   Wikipedia; Common Voice, SLR54 and IndicVoices from public text) and every gold and val
   reference, checked by 8-gram overlap as on 2026-09-27. Without this the benchmark scores are
   contaminated.
2. **Cheapest test first: n-best rescoring.** Beam search writes its 8 best transcripts, and a word
   n-gram model (KenLM) re-ranks the finished ones. It needs no change to the decoding loop.
3. **Then full shallow fusion**, if rescoring shows a gain: the text model votes at every step, on
   Flex's own word-pieces.
4. **Tune λ and the word bonus on val**; report on gold and the five public sets, folded and plain,
   with S/D/I and the unseen-word error rate.

**Cost term.** Beam search was rejected on 2026-09-13 (−0.41 gold for about 4× decode time). The
decision has to weigh the fused gain against that slowdown, on the GPU and on the CPU playground.

**If it pays, bake it in** rather than ship two models: let the fused Flex be the distillation
teacher (B), so its labels carry the vocabulary into the student, or fine-tune Flex on sentences a
Nepali TTS reads aloud. Either way the deployed model stays one file with a greedy decoder.

**Risks.** Formal text pulls toward formal spelling (`गर्दछ`, English in Devanagari) and away from
our labelling convention: keep λ modest and our labels in the text. Scoring Flex's word-pieces from
a word-level model is the fiddly part.

### G2. Low-risk fold rules

**Why.** On the public sets about 0.65 of the fine-tuned model's 1.18 points of number errors are
the same number written another way (0.46 on our gold on 2026-09-27). It changes the measurement,
not the model. The owner's call: the long tail of number formats is not worth chasing for that, so
only the finite, low-risk rules are in scope, done when little else is left to optimise.

**In scope** (finite, and cannot join two different words):
- maths symbols and their words: `+`/`plus`, `=`/`equals`/`equal to`, `×`/`into`/`times`
  (0.20 on average, 0.93 on nepali_cs);
- spelling variants of number words the table lacks (`छप्पन`/`छपन्न`, `उनानब्बे`/`उनान्नब्बे`),
  found by listing the corpus and benchmark vocabulary against `_NUMBER_WORDS`;
- English number words written in Devanagari (`वान`, `टु`, `फोर्टी`, `हन्ड्रेड`, `थाउजन्ड`);
- fractions `डेढ`, `साढे`, `सवा`, `पौने`.

**Needs the owner's call first:** count words with a classifier (`2` = `दुईटा`, `1` = `एउटा`).
The speaker said the classifier and the reference dropped it.

**Out of scope:** Hindi forms (`सौ` for `सय`, `बारह`), which are real errors; Devanagari spelling
pairs (ण/न, श/ष/स, व/ब), which can join different words and would each need the vocabulary check
the colloquial table had.

**How.** A new fold version, tests first, a decision entry, and every table re-reported beside the
plain WER, for every system alike. Each new rule is a `RULEBOOK` entry with its examples and
counterexamples (D112). The Nepali ordinals (`-औँ`, `प्रथम`) landed in fold-v4; the
Devanagari spelling pairs are now tier-4 tags, still never folded.

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
