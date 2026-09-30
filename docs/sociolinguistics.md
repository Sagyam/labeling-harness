# Sociolinguistic value of the corpus

The corpus was collected to fine-tune an ASR model. This note records what it can and cannot
support as material for a sociolinguistic study of Nepali–English code-mixing, what to collect
next to make such a study robust, and what metadata is worth adding at ingest. The measurements
behind it are in `notebooks/02_Sociolinguistics.ipynb`, run on the export of 2026-09-30, and every
number below is printed by that notebook, so a rerun on a newer export revises it. Who a voice is
(gender, age bracket, role) is resolved once, by the harness, and written on every turn of the
export (D107): the Voices page and the notebook read the same answer, and a difference between
them means the export is stale.

## What the corpus supports today

The unit of a sociolinguistic claim is a speaker, not a clip. On the 2026-09-30 export (342
recordings, 358 voices):

| | |
|---|---|
| voices with 300+ words attributable to one speaker | 171 |
| of those, gender known | 171 (35 women, 136 men) |
| of those, age bracket known | 171 |
| of those, host/guest role resolved | 105 |
| female voices aged 40–59 / 60–79, any talk time | 21 / 2 |
| genres with three or more such voices | 14 of 17 |
| multi-voice episodes with both genders present | 24 |
| recurring hosts with 5+ host–guest episodes | 3 |

What is recorded about a person is deliberately narrow: gender and a twenty-year age bracket,
typed by the annotator from watching the episode (D58) or set on the voice after listening to it
(D104), and a role. The owner listened to the voices and set the gender of 175 of the 358 by
ear, which is what took gender from 125 of those 171 voices to all of them; where a declared row
had already reached a voice, the two agreed on every gender and on all but four ages. Name, dialect and
origin were refused (D56). Everything else the notebook uses is derived from the reference
transcript and the diarizer's turns: English share, switch points, English run lengths
(single-word insertion versus multi-word alternation), English stems carrying Nepali morphology
(*phonesहरूको*), English function-word share, second-person address forms (*hajur / tapāī̃ /
timī / tã*), discourse markers in both languages, host–guest accommodation, and the gender
composition of the room.

**Supported.** A descriptive paper on the *form* of Nepali–English mixing in produced media
speech: how much of the English is lexical insertion and how much is alternation into English
syntax (37% of the English inside a mixed sentence is a single word, 18% sits in runs of six or
more), which Nepali suffixes attach to English stems, which discourse markers are borrowed (*so*
is the most frequent English token in the corpus, ahead of *the*), how address forms differ
between host and guest, and how all of these differ between the fourteen genres that hold enough
voices. Two findings are about people rather than formats:

- **Accommodation.** Over 34 host–guest episodes a host's English share tracks the guest's
  (Spearman 0.81). Within a host it holds for two of the three recurring ones (1.00 over six
  episodes, 0.90 over five); the third hosts an interview series whose guests barely mix at all,
  so there is nothing to track (0.14 over six).
- **Younger speakers switch more often, at the same English share.** Over the whole corpus
  English share falls with age (p = 0.0001), but that is genre: the older voices sit in the
  interviews, speeches and news, which are nearly all Nepali, and inside the podcasts the
  difference is gone (p = 0.95; medians 29%, 24% and 25% for 20–39, 40–59 and 60–79). The
  *switch rate* survives the same control: 25 switches per hundred words at 20–39 against 17 and
  12 in the two older brackets, inside the podcasts alone (p = 0.003, 27 / 17 / 4 voices). The
  young and the old use about as much English; the young move in and out of it more often.

**Not supported.** A Labovian social-stratification study. There is no region, class, education
or first language by design. Gender is confounded with genre: women hold between none and 74% of
the speech, depending on the genre, and only two of the 35 usable female voices are over 60.
With every voice's gender now known, men and women still do not differ on any measure the
notebook takes: English share p = 0.59 (medians 15% and 18%), and the one difference the
rule-resolved subset had shown, integrated stems at p = 0.03 on 20 women, is p = 0.59 on 35. At
the observed between-speaker spread the gender split can detect a difference of about 9 points
of English share (it was 12 before the by-ear values, and 21 on the 2026-09-18 export), and a
five-point difference would need about 165 voices per group.

## What to collect next

Every item below adds *speakers who break a confound*. Hours of an existing speaker add
nothing to a paper's n.

1. **Female and older guests on the podcasts already in the corpus.** Of the 171 usable voices
   35 are women and two are women over 60; most of the voices over 60 are interview guests.
   Senior professionals, women entrepreneurs, teachers, doctors and retired officials as podcast
   guests fill those cells inside a genre where age and gender can be compared at all.
2. **Mixed-gender rooms.** 24 episodes have both genders in the room against 64 that are men
   only and 4 that are women only. Female guests on the male-hosted series, and a female-hosted
   podcast with male guests, separate gender from show.
3. **More voices in the genres one person dominates.** Review has seven usable voices but one
   of them holds 72% of its words; commentary, audiobook, sketch and streaming are each over
   half one voice, and audiobook, street interview and FM radio have fewer than three usable
   voices. Two or three more voices in each turn an anecdote into a group.
4. **More episodes per recurring host, and the same guest on more than one show.** Ten or more
   episodes per host, a third recurring host, and guests who appear with different hosts give
   the accommodation design where the guest is held constant and the host varies.
5. **Speech that is not produced media.** Vox pops, call-in shows, live streams, panel
   discussions and classroom recordings add register range; news interviews and press
   conferences anchor the Nepali-heavy end, since topic drives English share and the current
   topics skew to tech and business.
6. **Peer and intimate talk.** The low address form *tã* appears in 38 of the 171 voices, a
   third of them in podcasts, and *timī* in 60. Friends chatting, comedy and gaming streams
   sample the *timī* and *tã* registers that interview formats never reach.

## Metadata worth adding at ingest

In order of value. Each is about the recording or is a closed, coarse fact; none reopens D56.

1. **Link declared speaker rows to diarized voices.** Done for gender and age. The forcing
   rules (one row meets one voice; a single host row meets a single recurring voice; the
   remaining rows agree) live in the harness (`app/services/inventory/resolve.py`, D91), the
   owner gave every voice the rules could not reach a gender and an age bracket by ear on the
   Voices page (D104), and the export writes the result on every turn (D107): 357 of 358 voices.
   What is still open is **role**: it comes from the declared rows alone, and reaches 105 of the
   171 usable voices.
2. **Upload date, channel and view count.** The harness now stores the first two for every
   episode (`episodes.published_at`, `show_id`) but the export does not carry them, so the
   notebook still reads a series off the episode id and can name only three. Writing both to
   the export gives change over time and a clean show variable; a view-count bucket, which is
   not collected, is an audience proxy. All three describe the recording, not the person.
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
