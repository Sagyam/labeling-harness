# Research roadmap: the last experiments

What is next for the ASR work on the Nepanglish corpus, replanned on 2026-09-22 and again on
2026-10-20 for the final phase. What has been measured so far is in [findings.md](findings.md).
Retired items are listed at the end.

**Where this stands.** Single-speaker recognition is largely solved for this corpus: the teacher
(blend-075) scores 10.85 folded WER on gold and 7.08 on val, and the students have run their 100 h
points. Crosstalk is what remains (17% at 5–15% overlap and 29% above 15% on the 2026-09-22 gold),
and it stays unsolved on purpose. The diarizer's turns cannot be trusted (D100), and on 2026-10-20
the owner retired both the models for overlapped speech and the diarizer experiment (D124). Gold
stays single-stream: everything said, in time order, with overlap spans marked.

**Order, set by the owner (2026-10-20):**
1. **H.** Data dosing on Flex: hours against voices, now that the corpus has grown.
2. **B step 6.** The council, convened once every student is in.
3. **I.** The commercial recognisers on the public sets.
4. **J.** WER at code-switch points.
5. **K.** A smaller Flex: the same encoder and a cut-down decoder, distilled with the teacher's
   per-token probabilities.

Still open from before: 06b (B), and the low-risk fold rules (G2). E is how all of it is
measured. Sections are lettered so they do not collide with the retired numbered items that code
comments still cite, which is why G to K come after E.

**How a model is trained and evaluated (D105, 2026-09-30).** B and C are no longer run ad hoc:
the notebooks are numbered in the order they run, and that order is the protocol.

| Step | Notebook | Roadmap section |
|---|---|---|
| 1–2 | `01_EDA`, `02_Sociolinguistics` | the corpus |
| 3 | `03a_Flex_Train` to `03e_Flex_Ship` | Flex retrained, scored on the public sets, C's ablation, blended, one teacher frozen |
| 4 | `04_PreDistill` | B step 1, and each clip's overlap measured |
| 5 | `05_Teacher` | B steps 2–3 |
| 6 | `06a`–`06f_Student_*` | B step 4, one notebook per ASR student |
| 7 | `07a`–`07b_Student_LLM_*` | B step 5, one notebook per speech-LLM student (D122) |
| 8 | `08_Judge_Headroom` | G1's pilot, step 0: the oracle headroom (D117) |
| 9 | `09_Report` | E: every model in one table, always last |

Every model goes through one evaluation (`notebooks/src/evalkit.py`), and every choice is made on
val by a rule fixed before the result. Each notebook has a smoke switch. **Status, 2026-10-04:**
03a–03e have run on the 2026-09-30 export. 03c kept no stage, so vanilla-s1 went into 03d, which
chose blend-075, and 03e froze it as the teacher with an int8 CPU export (score-card/flex-score-card.md, *The
teacher frozen*). 04's overlap pass and 05 have labelled the first tranche: 79.5 h kept of 102.1 h
at `MAX_OVERLAP_SHARE` 0.05 (score-card/flex-score-card.md, *The teacher labels the unlabelled corpus*). 06a has run
its 100 h point: Whisper's gold fell from 17.32 to 15.02 with the pseudo-labels (score-card/whisper-score-card.md,
*Whisper's 100 h point*). 06c followed: IndicConformer's gold fell from 18.42 to 15.64, a tie with
Whisper's 15.02 at a seventh of its size, its extra errors deletions in crosstalk (score-card/indicconformer-score-card.md,
*IndicConformer's 100 h point*). 06d then: Parakeet's gold fell from 28.63 to 21.53, its teacher
labels closing 42% of the gap Nepali pretraining gave IndicConformer, still 5.9 behind (score-card/parakeet-score-card.md,
*Parakeet's 100 h point*). 06f last: the Conformer from scratch fell from 43.70 to 27.08, its
teacher labels closing half its gap to the teacher, still 11 points behind the pretrained
students; its stage 3 was stopped after one epoch (score-card/conformer-score-card.md, *The Conformer from scratch*).
06e on 2026-10-09: Omnilingual CTC 1B's gold fell from 19.65 to 17.52, behind Whisper, Gemma 4 E2B
and IndicConformer, with the lowest gold CER of any student (score-card/omnilingual-score-card.md,
*The CTC student's 100 h point*).
07b on 2026-10-26 (trained 2026-10-20): Omnilingual LLM-ASR 1B's gold fell from 15.31 to 14.62,
the best student, tied with Whisper; its LLM decoder gained 1.44 points less from the teacher's
labels than 06e's CTC head (score-card/omni-llm-score-card.md, *Omnilingual LLM-ASR 1B's 100 h
point*; findings.md, *Teacher labels help a CTC head more than an LLM decoder*).
**Next:** the remaining student (06b) when there is GPU time, smoked first.

## A. Per-speaker labelling as a multitrack editor (stopped, D100)

Built on 2026-09-23 (D98) with voiceprint suggestions (D99), and piloted on gold crosstalk. On
2026-09-24 the owner listened to the diarizer's per-word attribution and ruled it unusable: turns
are hit or miss, and in crosstalk the per-word join decides by rule, not by audio (findings.md,
*Diarization cannot say who said a word*). The speakers queue is closed. The editor was kept for F,
which is retired (D124); its code stays until the owner decides otherwise, and nothing uses it.

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
| **Small Conformer from scratch** (NeMo) | no: no pretraining at all | own tokenizer | Added for step 4 (2026-09-26). The shared SentencePiece and heads of the transducers, a small config and random weights: it asks whether pretraining still matters once 145 h of pseudo and human labels exist. `06f_Student_Conformer` builds it from IndicConformer's config, resized, without the encoder copy. Ran 2026-10-05/06: pretraining is still worth ~11 points of gold at this data (score-card/conformer-score-card.md, *The Conformer from scratch*). |

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
     human stage's weights, with the recipe that won Flex's ablation; the Conformer from scratch
     runs 03c's six acoustic stages instead, D115). This reverses the rule of
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

5. **Speech-LLM students** (D121, D122). The bake-off ran on 2026-10-07 (findings.md,
   *Speech-LLMs as students*): Gemma 4 E2B and Omnilingual LLM-ASR 1B go on, as `07a` and `07b`,
   the smaller sibling of each kind; the rest are out. Each has the same three stages as the ASR
   students, held to D121's bars. The paper's family claim is the recipe's gain per family (stage
   2 minus stage 1), with 06e against 07b as the controlled pair, not which family wins. Both
   passed their smoke runs on 2026-10-07. 07a ran on 2026-10-07/08 (score-card/gemma-e2b-score-card.md, *Gemma 4 E2B's
   100 h point*): the best stage 1 of any student (gold 16.09), but stage 2 gained only 0.68, and
   it memorised the replayed human clips within two or three epochs. 07b ran on 2026-10-20 and
   was scored on 2026-10-26 (score-card/omni-llm-score-card.md): the best student (gold 14.62),
   and the controlled pair's answer is that its LLM decoder gained 1.44 [0.83, 2.04] points less
   on gold from the teacher's labels than 06e's CTC head (findings.md, *Teacher labels help a CTC
   head more than an LLM decoder*).

6. **At the very end, convene the council again** (owner, 2026-10-06). Once every student is
   trained, rerun the panel vote: the teacher's greedy decode and the students vote among the
   teacher's 8 beam candidates, and the candidate closest to them wins (findings.md, *The panel
   vote*). With four students it recovered 36–39% of the room between greedy and oracle@8 on val
   and gold (val 7.08 → 6.50, gold 10.85 → 10.16), the only picker that beat beam search alone.
   It is the promising recipe for the last bit of performance from a council of models.
   - **Cheap to rerun.** `judgekit.mbr_index(voters=...)` over transcripts already stored (08's
     candidates on the hub, each student's `harness/` and `benchmarks/` files): a CPU minute, no
     decoding. A new student joins by adding its transcripts.
   - **The owner hand-picks the council.** Questions left open on purpose, not worth a grid now:
     which students sit on it (only the Nepali-pretrained ones, only the best); the beam width
     (oracle@2/4/8 on val: 6.39, 5.88, 5.47); the students' own beams as candidates or voters.
     Already measured: without the Conformer from scratch (V1) the vote reads val 6.52 and gold
     10.20 against 6.50 and 10.16 with it, so its seat changes almost nothing.
   - **A vote is only as good as its voters.** It lost on Common Voice (7.85 → 8.09) and gained
     nothing on FLEURS or nepali_cs, where every student is weakest; it gained most on IndicVoices
     (12.29 → 11.47). Judge a council per set, not on val alone.
   - Only worth tuning if its output is used: relabelling a student's training data with the
     council's pick would make it a recipe question; a reported number does not.
   - **New seats since 2026-10-06** (added 2026-10-20): Gemma 4 E2B (07a, gold 15.41) and
     Omnilingual CTC 1B (06e, 17.52), Omnilingual LLM-ASR 1B (07b, 14.62), and 06b once it
     runs. Expect little: V1–V3 sat within 0.02 of each other on val, so a new voter may move the
     vote by a tenth of a point. It is run anyway because it costs a CPU minute.
   - **A word-level vote as a variant** (ROVER-style). The panel vote picks one whole candidate,
     so oracle@8 bounds it; a word-level vote aligns the voters' transcripts into slots and keeps
     each slot's majority, so it can join one candidate's fix to another's. Align and vote on
     folded tokens (`fold.py`, as every comparison), and write the winning token as its voter
     wrote it. The risk is a sentence no model wrote, with broken grammar at the joins: read a
     sample before trusting the number. It is chosen against the panel vote on val by the same
     rule, and reported per set.
   - **No commercial recogniser sits on the council on val or gold.** Those clips were seeded
     from the three recognisers' fusion (D74), so their votes would pull the pick towards the
     reference by construction. On the public sets, once I has their transcripts, a council with
     them is a fair variant; report it apart from the open-model council.

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

## C. Augmentation pipeline (done)

What the sweep taught: augmentation pays only when the model has a way to use it, either
conditioning that says whom to follow or an output format that writes both voices. Those models
were D, retired with F on 2026-10-20 (D124), so what is left of C is the switchboard below and
its run on Flex. Gold and val audio are never a source of mixed-in speech or noise (D76).

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

**Status, 2026-10-11.** 03c ran speed, reverb, codec, gain, channel and crosstalk on Flex, and none
cleared D109's bar. Speed came closest, at −0.07 on val. Noise ran last, on 2026-10-11, to observe
rather than to keep: it tied vanilla on val, gold and the noisy no-crosstalk bucket. Flex is already robust
to the acoustic conditions, and what remains on gold is mostly vocabulary. The crosstalk stage's
label interleaved two sentences with no speaker marker, and most heavy-crosstalk gold labels
predate D100 (score-card/flex-score-card.md, *Augmentation on Flex: seven stages, nothing kept, and
why*).

**What the label says** (`CrosstalkConfig.label`, 2026-09-26). D96 kept the clip's own text, which
teaches a model to leave the other voice out; since D100, gold writes everything said, so a
single-stream model trained that way is charged a deletion for every word it was taught to drop.
`label="everything"` merges the target's and each donor clip's words in time order, punctuation
kept, into `info["text"]`, which the collate function trains on. The spans are the training
export's `label_words` (the seed's aligned words, or a realignment for an edited label), so it needs
whole-clip donors and skips any clip without them. `label="target"` stays for target-speaker models
(D, retired), which are told whom to follow.

Items 1 and 2, two-voice mixes as a test bench and realistic conversation timing, existed to
develop D and are retired with it (list at the end). The `donor="clip"` mixer and
`label="everything"` stay in `augment.py`; 03c's crosstalk stage used them.

3. **Noise augmentation (owner's decision: part of the pipeline).** Background noise at a chosen
   SNR, drawn from the distribution Brouhaha measured on the corpus (D87), from a non-speech noise
   set such as MUSAN's music and noise subsets. Ran on Flex on 2026-10-11 and tied vanilla
   everywhere (status above).
4. **LLM-written conversations spoken by TTS (watch, not build).** 67 h of real plus 636 h of
   synthetic Hungarian beat a model trained on 2,700 h
   ([Conversations that Never Happened](https://arxiv.org/abs/2606.03957)). This needs a
   code-mixed Nepali TTS with many voices, which does not exist yet.

## E. How any of this is measured

- **Single-stream WER only.** Per-speaker metrics (cpWER, tcpWER, ORC-WER) need references that
  say which words each speaker said. Real gold has none (D100), and the models that would have
  needed them are retired (D124).
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
  - `09_Report` prints the crosstalk and number blocks only; the confusion table and the
    attribution card are on the Models page.

## G. Vocabulary at decode time, and the low-risk fold rules (G1 closed, G2 open)

Both come from the 2026-09-27 benchmark error anatomy (findings.md, *Public Nepali benchmarks, the
error anatomy, and weight blending*). The owner parked them for later.

### G1. Shallow fusion with a text-only language model

**Closed 2026-10-06 (D120).** Its pilot found an LLM judge a dead end: on val and gold both
judges lost to the panel vote, and the n-gram picker on our train labels gained nothing. By the
pilot rule below, fusion is not tried. What follows is the plan as it stood.

**Why.** Words our train labels never contain are the largest cause of error on the public sets:
3.23 of the fine-tuned model's 11.20 points, plus 1.09 for words seen 1–9 times. Blending recovers
the vocabulary fine-tuning erased; a text model adds vocabulary neither model ever had. Nepali text
(news, Wikipedia, books) is plentiful; Nepali speech is not.

**How.** A small model trained on text alone scores each candidate next piece while Flex decodes,
and the two scores are added: `Flex + λ · text model + a per-word bonus`. Nothing inside Flex
changes. It needs beam search (4–8 hypotheses), because greedy decoding leaves the text model
nothing to re-rank.

**Pilot first: can a small LLM judge pick a better transcript?** (owner, 2026-10-04.) Before
building any text model, ask the cheapest question: beam search writes 8 transcripts, and a local
LLM that fits in 40 GB picks one. It picks, it never rewrites, so it cannot invent words that
were not said. It runs once, after the students (B), on Flex and every student alike: the teacher
and its labels stay as they are, so no student is judged against a moved target. Fusion (step 3)
is tried only if the pilot shows promise.
- **Measure the headroom first** (`08_Judge_Headroom`, D117; ran 2026-10-06 on val, gold and the
  public sets: oracle@8 1.36 [1.09, 1.66] below the top candidate on val, so G1 continues; against
  greedy the ceiling is 1.6–1.8 points on our sets and 2.0–3.5 on the public ones, and MBR gets
  none of it; findings.md, *The judge's ceiling*). The panel vote (the teacher's greedy decode and
  four students voting among the beams) recovers a fifth to a quarter of the room above the top
  candidate: val 6.50, gold 10.16 (findings.md, *The panel vote*). **The pilot** (D118, its
  notebook since removed): Qwen3.8-27B reading the candidates and Gemma 4 12B hearing the clip,
  without thinking, against the panel vote; it stopped, and on every set the judges lost to the
  panel or tied it, FLEURS aside (findings.md, *An LLM judge is a dead end*). On
  val, the oracle WER of the 8 (always picking the candidate closest to the reference) bounds what
  any judge can gain. If the oracle is less than 1.0 point better than the top candidate, stop: no
  judge can help. No judge model is chosen yet: the owner wants a matrix of local models only
  (Qwen, Gemma, Granite, Nemotron and others), text-only and audio-aware (the audio judge hears
  the clip), against top@1, a random pick, KenLM and the oracle; up to 96 GB of VRAM if it shows
  promise. Gemma stays in and is scrutinised only if it wins by a wide margin.
- **Zero-shot first, no fine-tuning.** Shuffle the candidates for each clip (LLMs favour the first
  one shown), ask for the number only, and compare against two baselines: the top beam
  candidate, and a word n-gram model (KenLM) on the train labels picking from the same 8. An LLM
  that does not beat the n-gram model is not worth its cost.
- **Choose on val, report on gold and the public sets**, folded and plain, with S/D/I, the
  unseen-word error rate, decode time and judge time.
- **Fine-tune the judge only if zero-shot shows promise.** The hard part is data, not compute.
  A judge must learn from candidates the ASR wrote on audio it was not trained on, and Flex's
  beams on its own train clips are far too good to teach that. Val and gold cannot be used, so
  it needs held-out decodes, for example a model trained on half the episodes decoding the other
  half. A LoRA fine-tune of a 7–8B model on ~15k such clips fits one A100 in a few hours.
- **The synthetic Nepanglish text** (the twist below, set (c)) enters as text for the judge or the
  n-gram model, never as decoder training. Training an ASR's decoder on text with no audio teaches
  it to write without listening, the failure that shows up as hallucination on noisy and
  overlapped clips. That waits until the text has proved useful here.

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

**A twist, if the formal text disappoints.** Let step 2 compare two text sets: (a) our train labels
alone, and (b) those plus news and Wikipedia. If (b) gains little over (a), or drags towards formal
spelling, the likely cause is register: people do not talk like the news. Then try (c): the same
formal paragraphs rewritten by an LLM into colloquial Nepanglish, steered by what
`docs/sociolinguistics.md` measured, and rescored the same way. An LLM can also write such text
from a topic alone; a sample explaining a Conformer block read well (owner, 2026-10-04), lacking
only fillers, which the fold drops anyway, and with some word pairings a native speaker would
rarely use. Those pairings are the risk: a model trained on them prefers them, so keep λ tuned on
val.

- **One rewrite, not random swaps.** A word that switches is mostly an everyday noun whose formal
  Nepali nobody says (`दूरभाष` → *phone*); 68% of switches are one word; the English keeps its
  Nepali frame (`मा`, `को`, `हरू` after it, a form of `गर्नु` to make it a verb: `use गर्नुभयो`,
  never *used*); *so* opens 19% of sentences that open in English. A random word-for-word swap
  breaks all of these, so making the text colloquial and switching it are one step, given those
  rates and a few train labels (never gold or val) in our spelling.
- **The analytics check the output as well as steer it.** English share, single words against
  long runs, suffixed English stems, the `गर्नु` rate and sentence openers, measured on the
  synthetic text against the labels; reject a batch that drifts, and any Hindi (`है`, `में`,
  `नहीं`), which LLMs mix into Nepali.
- **The English comes from the source's topics**, not from the 7,059 English words the corpus
  already switches to: Flex's gap is rare English it has never heard, and a list it already has
  teaches it nothing.
- The rewrite is an inference call: a route and an `llm_requests` row if it runs in the harness,
  otherwise the prompt and model kept with the text so the set can be rebuilt. (c) goes through
  the same 8-gram decontamination as (a) and (b).

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

## H. Data dosing on Flex: hours against voices (added 2026-10-20)

**Why.** The only learning curve (findings.md, *Learning curve: more of the same data does not
help*, 2026-09-13) was one draw at 4.6, 9.3 and 18.3 h with 27 recurring voices. Val was flat and
gold fell about 0.25 per doubling, but part of that was gold episodes joining training, and the
hint that new speakers matter more than new hours (+0.62 [−0.16, +1.39]) had an interval
including zero. Since then train has grown to 51.27 h and 173 linked voices, and gold has been
heard to share no voice with train (findings.md, 2026-10-10), so the confound is gone. The
2026-09-30 corpus analysis projected 0.2–0.6 points on clean gold from the growth; this measures
it.

**Design.** Two curves that cross at one shared point, so the question "more hours or more
people?" is answered directly rather than read off one curve:
- **Hours, voices held.** A fixed set of voices, the ones with the most hours, at 25%, 50% and 100%
  of their hours.
- **Voices, hours held.** A fixed number of hours (the hours curve's smallest point that every
  voice point can reach), spread over 25%, 50% and 100% of train's voices.

The draw unit is the whole episode, nested within a draw as in 2026-09-13 (25 ⊂ 50 ⊂ 100).
Voices are the linker's (D87), which undercount distinct people (findings.md, 2026-10-10); the show
is the coarser check, and a point is reported with its show count too. The exact hours and voice
counts of every point come from the current export and are fixed, with the decision rule, before
any run.

**Recipe.** 03a's vanilla recipe, every point given the full run's optimizer steps as before, best
epoch by val. Seeds differ by 0.02–0.03 on val, so one seed per point, with the crossing point run
twice as the noise check.

**Reading it.** Report folded and raw WER with S/D/I on gold, clean gold (no crosstalk), val and
the five public sets, and the word-rarity block (D123): the 2026-09-30 analysis expects the gain
to come from unseen words. Only gold is held out by speaker, so the voices curve is read on gold.
That is a measurement, not a choice of model, so D105's rule (never choose on gold) is not broken;
no point becomes a teacher.

**Decision rule, fixed before the runs** (proposed; the owner fixes it): a gain of at least D109's
0.15 floor per doubling, interval clear of zero, means that axis still pays; less means it is
saturated. If voices pay and hours do not, the corpus advice is to hunt voices, not hours, and the
2026-09-13 reading becomes a measured result.

**Cost.** Five runs of 03a's length (two curves of three points, sharing one, plus the repeat).
A new Flex notebook generated from a builder like 03a–03e (D105), with a smoke switch.

## I. The commercial recognisers on the public sets (added 2026-10-20)

**Why.** Gold cannot compare Flex with Scribe v2, MAI Transcribe 2 and Gemini 3.8 Flash: its
reference is their fusion (D74), edited on 3 of 1,170 verified labels, so a recogniser's gold WER
is partly its distance from its own consensus (findings.md, *The reference is a consensus of the
systems it scores*). The five public sets are the only references none of them wrote. Any claim
in the paper that Flex matches or beats them comes from here.

**What runs.**
- The three `asr_*` routes on FLEURS (726 clips), Common Voice (287), IndicVoices (2,829),
  nepali_cs (1,763) and a sample of OpenSLR 54. Its 13,609 clips have a median of three words, so
  a sample drawn by speaker is enough; the teacher is scored on the same sample, paired.
- **Optionally the fusion** (`fuse_transcript`) over the three. That is the seed every gold label
  started from, scored for the first time against references it did not write: how good the
  starting point of the corpus was.

**How.**
- Through the named routes and their clients in `app/llm/`, each call an `llm_requests` row
  (invariant 6), from a script over those clients, never through ingest: public clips are not
  episodes (D86) and never become segments.
- `dry_run: true` on a handful of clips first. Price the full run from that before spending.
- **Coverage is reported, not hidden.** Gemini's safety blocks dropped about 16% of clips on the
  2026-09-16 gold refill even with filtering off. A blocked or empty answer is scored as all
  deletions in the headline row, with a covered-clips-only row beside it.
- FLEURS, OpenSLR 54 and Common Voice references are the prompt the speaker read, not what was
  said (E). Every system shares that bias, so the comparison stays fair, but the absolute numbers
  are not.

**Reported.** Folded and raw WER with `fold_version()`, S/D/I, the word-rarity block (D123), each
paired against the teacher's greedy decode with 95% intervals over each set's unit (speakers,
sentences or videos, as in 08). With them the cost term: dollars per audio hour from
`llm_requests`, and latency, beside the teacher's RTF on a GPU and on the CPU playground.

**Feeds.** B step 6's public-set council variant, and the paper's comparison table.

## J. WER at code-switch points (added 2026-10-20)

**The question.** Do recognisers err more at a language switch than away from one? It is the
result most specific to code-switching that nothing here has measured yet, and it ties the ASR
work to `docs/sociolinguistics.md` (a pause comes before 16% of switches into English against 7%
of Nepali-to-Nepali boundaries, in the median voice).

**Definitions.**
- A word's language is its script, as CMI reads it (`app/services/analysis.py`): the labels write
  English in Latin and Nepali in Devanagari. `fold.script` adds `mix` for a word that switches
  inside itself (`queriesहरू`), which is its own class, not a switch.
- A reference word is *at a switch* if its language differs from the reference word before or
  after it.
- Substitutions and deletions belong to their reference word. An insertion belongs to the gap it
  falls in, and that gap is at a switch if either neighbour is.

**What is compared.** S/D/I per 100 reference words, at a switch and away from one, for every
model on gold, val and nepali_cs, the only public set with much switching. The two regions sum to
the set's WER: a breakdown never changes a WER (D110).

**Confounds, controlled before reading it.**
- A switch word is often English and rare. Compare English words at a switch with English words
  inside an English run, and Nepali with Nepali, so being English is not read as being at a
  switch. Report it within the word-rarity buckets too (D123).
- Crosstalk: report it on the clean bucket as well.

**Cost.** No decoding: a block in `evalkit` beside the word-rarity one, backfilled from the stored
transcripts and labels (the error rows alone lack the correct words for a denominator), and shown
by `09_Report`.

## K. A smaller Flex: the same encoder, a cut-down decoder (added 2026-10-20)

From `custom-arch.md` §1 (the file was deleted on 2026-10-20; its text is in git history), the
one idea in it that did not depend on diarization.

**Why.** Flex's time goes to its decoder. On the CPU the encoder takes about 0.015 s per second of
audio, out of an RTF of 0.11 (findings.md, *CPU inference*): each of the ~8 tokens per second of
speech re-reads all 24 decoder layers. Every B student was a different architecture with its own
tokenizer, so none could learn from the teacher's per-token probabilities (B, *Soft distillation
is skipped*), and model size was never swept. A Flex with a smaller decoder does both: it shares
the teacher's tokenizer by construction, and the number of decoder layers is the size axis.
[Distil-Whisper](https://arxiv.org/abs/2311.00430) did this to Whisper (two decoder layers kept)
and ran about 6x faster within about a point of WER.

**How.**
- Keep the teacher's 32-layer encoder as it is; give the student a decoder of 2 or 4 layers,
  initialised from teacher layers spread across the 24 (Distil-Whisper kept the first and last).
- Train on the teacher's 79.5 h of filtered labels plus the human labels, the human labels
  taking half of each epoch's draws as in B (D106).
- The loss: cross-entropy on the labels plus KL to the teacher's per-token distribution. The
  KL needs the teacher's logits, either decoded again on the fly or stored per clip; a teacher
  forward pass is cheap beside the student's, so on the fly is the default. Run the pair with
  and without the KL term, since that is the comparison B could not make.
- Encoder frozen first; unfreeze only if the frozen run falls short.
- Built from the Flex notebooks' existing training pieces (teacher-forced cross-entropy, target
  encoding, SpecAugment), as a generated notebook with a smoke switch (D105).

**Reported.** Each decoder size against the teacher: gold and val, folded and raw, S/D/I, per clip
class, the five public sets, the word-rarity block (D123); B's success bar applies. Beside it the
cost term: ms per token and RTF on the GPU and on the CPU playground (int8, the owner's 7700X),
and model size. The result is a curve of decoder layers against WER and speed, not a winner; a
student that clears B's bar at a fraction of the decode time is the one worth importing.

## Retired items

**Retired 2026-10-20 (D124).** The owner closed overlapped speech and the diarizer for this
project. Code comments and other documents still cite these sections:

- **D — models for overlapped speech.** D1 DiCoW v3.3 and SE-DiCoW conditioned on diarization,
  D2 SOT-DiCoW, D3 target-voice extraction in front of Flex (the old item 2), D4 the multitalker
  Parakeet method transplanted into Flex's encoder, D5 commercial recognisers that diarize as
  zero-shot cpWER baselines. On hold since D100; all conditioned on diarization. Their design
  notes were in `docs/custom-arch.md`, deleted on 2026-10-20; the text is in git history.
- **F — fiddling with the diarizer.** The over-cluster-then-merge test and the exclusive-track
  check on episode 205 were never run. Its "who is who" half was answered corpus-wide on
  2026-09-28 (findings.md, *Same voice or not*); the parked idea of voice identity from similarity
  plus a short listening pass was not built. Turn boundaries and crosstalk (the *when* problem)
  stay closed by D100.
- **C1 — two-voice mixes from whole verified clips** as the bench where D would be developed;
  the mixer (`donor="clip"`, `label="everything"`) was built and used by 03c.
- **C2 — realistic conversation timing** (FastMSS, boosted overlap). Never built.

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
