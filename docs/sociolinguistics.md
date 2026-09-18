# Sociolinguistic value of the corpus

The corpus was collected to fine-tune an ASR model. This note records what it can and cannot
support as material for a sociolinguistic study of Nepali–English code-mixing, what to collect
next to make such a study robust, and what metadata is worth adding at ingest. The measurements
behind it are in `notebooks/Sociolinguistics.ipynb`, run on the export of 2026-09-18, and every
number below is printed by that notebook's verdict cell, so a rerun on a newer export revises it.
The Corpus page in the harness computes the voice counts live (D91) and should agree with the
notebook's; where they differ, the export is stale.

## What the corpus supports today

The unit of a sociolinguistic claim is a speaker, not a clip. On the 2026-09-18 export:

| | |
|---|---|
| voices with 300+ words attributable to one speaker | 59 |
| of those, gender resolved | 30 |
| of those, age bracket resolved | 24 |
| of those, host/guest role resolved | 22 |
| female voices aged 40–59 / 60–79 | 1 / 1 |
| voices in the tech-review register | 1 |
| fully resolved multi-voice episodes (both genders present) | 4 |
| recurring hosts with 5+ host–guest episodes | 2 |

What is recorded about a person is deliberately narrow: gender, a twenty-year age bracket and
a role, typed by the annotator from watching the episode (D58). Name, dialect and origin were
refused (D56). Everything else the notebook uses is derived from the reference transcript and
the diarizer's turns: English share, switch points, English run lengths (single-word insertion
versus multi-word alternation), English stems carrying Nepali morphology (*phonesहरूको*),
English function-word share, second-person address forms (*hajur / tapāī̃ / timī / tã*),
discourse markers in both languages, host–guest accommodation, and the gender composition of
the room.

**Supported.** A descriptive paper on the *form* of Nepali–English mixing in produced media
speech: how much of the English is lexical insertion and how much is alternation into English
syntax, which Nepali suffixes attach to English stems, which discourse markers are borrowed
(*so* is the most frequent English token in the corpus, ahead of *the*), how address forms
differ between host and guest. An accommodation study within the two recurring hosts, each of
whom tracks the guest's English share across episodes (Spearman 0.90 over five episodes each),
is the strongest finding and the most publishable one.

**Not supported.** A Labovian social-stratification study. There is no region, class,
education or first language by design; the age cells above 40 are one or three voices; and
gender is confounded with show, since most female speech comes from one podcast cluster. At
the observed between-speaker spread, the gender split can detect only a difference of about
20 points of English share, and a five-point difference would need around 200 voices per
group.

## What to collect next

Every item below adds *speakers who break a confound*. Hours of an existing speaker add
nothing to a paper's n.

1. **Female and older guests on the podcasts already in the corpus.** The only cells with more
   than four voices are men and women under 40. Interviews with senior professionals, women
   entrepreneurs, teachers, doctors and retired officials fill the cells that are one voice.
2. **Mixed-gender rooms.** Female guests on the male-hosted series, and a female-hosted podcast
   with male guests, separate gender from show for the first time.
3. **More tech reviewers, and more monologue vloggers in general.** The tech register is one
   person, so nothing about "review speech" can be claimed; two or three more reviewers turn an
   anecdote into a group. The reels are many voices but a minute each.
4. **More episodes per recurring host, and the same guest on more than one show.** Ten or more
   episodes per host, a third recurring host, and guests who appear with different hosts give
   the accommodation design where the guest is held constant and the host varies.
5. **Speech that is not produced media.** Vox pops, call-in shows, live streams, panel
   discussions and classroom recordings add register range; news interviews and press
   conferences anchor the Nepali-heavy end, since topic drives English share and the current
   topics skew to tech and business.
6. **Peer and intimate talk.** The low address form *tã* appears in nine voices. Friends
   chatting, comedy and gaming streams sample the *timī* and *tã* registers that interview
   formats never reach.

## Metadata worth adding at ingest

In order of value. Each is about the recording or is a closed, coarse fact; none reopens D56.

1. **Link declared speaker rows to diarized voices.** This is the largest gain and needs no new
   field. The forcing rules the notebook applies (one row meets one voice; a single host row
   meets a single recurring voice; the remaining rows agree) now live in the harness
   (`app/services/inventory/resolve.py`, D91), so the Corpus page shows the same 30 of 59
   usable voices with gender and reports the rest as unresolved rather than absent. What is
   still missing is the manual link: a ten-second listen per diarized speaker, with a click on
   the matching row, would cover nearly every remaining podcast voice. D78 stores the
   per-speaker embeddings for exactly this. This is a build item, not a reversal.
2. **Upload date, channel and view count.** The YouTube probe already fetches upload date and
   uploader and drops them before storage. Year of recording gives change over time; channel
   gives a clean show variable instead of parsing episode ids; a view-count bucket is an
   audience proxy. All three describe the recording, not the person.
3. **Interaction format and relationship, closed vocabularies.** Format: monologue, interview,
   panel, call-in. Relationship: first meeting, acquainted, unknown. Both are visible in the
   episode and both predict address form and accommodation.
4. **Facts the speaker states about themselves on air, with a provenance flag.** Medium of
   schooling and time lived abroad are the strongest known predictors of South Asian
   code-mixing, and neither can be seen. Guests often say them. A field that records only
   what was said in the episode, defaulting to unknown, is honest data and stays clear of the
   guessing D58 refused.
5. **Coarse professional domain of a guest** as the host introduces them: tech, business,
   medicine, arts, politics, education, other. A class proxy without being class. Keep it
   closed and coarse: with the voice and the topic it could become a re-identification key in
   a corpus this small.

Not to add: dialect, region or origin (D56), name (D56), anything inferred from audio (D58).

## A threat to validity to settle before writing

The mixing variable is the script of each reference word, and script is exactly what the three
recognisers disagree on: whether a word is written *phone* or *फोन* is partly a transcription
convention inherited from the fused seed (D74), not a fact about the speaker. Most of train was
screened on the disagreement signal without being played (D63). For a paper, define the mixing
measures on the verified tier, or hand-check script choice on a sample, and state the
convention explicitly. The gold pot is fully verified and is the natural sample for that check.
