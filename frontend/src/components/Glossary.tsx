/**
 * The terms the harness uses, in plain words, for someone who did not build it: the help
 * dialog's Glossary tab, opened from every page's one-line intro. Each entry names the decision
 * behind it (docs/decisions.md) so a reader who wants the argument knows where to look.
 */

type Entry = [term: string, meaning: string]

const GROUPS: [title: string, entries: Entry[]][] = [
  [
    'The corpus',
    [
      [
        'Episode, show',
        'One recording (a podcast episode, a video), and the programme it belongs to. An episode is cut into clips of at most 20 s at pauses.',
      ],
      [
        'Genre, topic',
        'Genre is the recording format (podcast, interview, reel, ...: a closed list), topic is what it is about (D102).',
      ],
      [
        'Gold, train, val',
        'The two pots (D71). Gold is the held-out test set, chosen clip by clip by hand and never trained on; train is everything else, split into train and val per episode by a hash.',
      ],
      [
        'Voice',
        'An anonymous speaker id, followed across episodes. Gender, age and role are given by ear. Speaker turns come from automatic diarization and are not trusted for who said a given word (D100).',
      ],
      ['Seed', 'The transcript a clip starts from: three recognisers fused into one (D74). It is a draft, not a measurement.'],
      [
        'Verified, screened',
        'How hard a label was looked at: verified means played and read; screened means accepted on the recognisers agreeing, without listening. Gold is verified only.',
      ],
    ],
  ],
  [
    'Scoring',
    [
      [
        'WER',
        'Word error rate: substitutions, deletions and insertions per 100 reference words. Shown as S · D · I, which add up to it.',
      ],
      [
        'Folded, raw',
        'Folded WER forgives spelling conventions (टिम/team, गर्नुभयो/गर्नु भयो, 5/पाँच) through fold.py; raw WER does not. Every number names its fold version.',
      ],
      [
        'Interval [a, b]',
        '95% interval from resampling whole episodes (or a public set’s speakers), never single clips, since clips of one episode share voices and microphones.',
      ],
      [
        'Base, minus base',
        'The model a fine-tune is compared with, usually the released model; "minus base" is this model’s number minus that one’s on the same clips. Negative is better.',
      ],
      [
        'Public sets',
        'FLEURS, OpenSLR 54, Common Voice, IndicVoices and nepali-cs: outside Nepali test sets, scored the same way as gold.',
      ],
    ],
  ],
  [
    'Recording conditions',
    [
      [
        'Crosstalk',
        'Two voices at once, as a share of the clip, from a local overlap detector (D77). Buckets: none, 0-5%, 5-15%, over 15%.',
      ],
      [
        'SNR (noise)',
        'Speech-to-noise ratio in dB, measured by Brouhaha (D87). Low is noisy; 45 dB or more is clean.',
      ],
      ['C50 (room)', 'How much a room echoes, in dB: low is a reverberant room, high a dry one (D87).'],
      ['Bandwidth', 'The highest frequency the speech reaches: a phone line stops near 3.4 kHz, full band near 8 kHz.'],
      [
        'Code-mixing (CMI)',
        'How much a clip switches between Nepali and English, from 0 (one language) up.',
      ],
      [
        'Within episode, ×ratio',
        'A condition’s error rate over its baseline’s, comparing clips only with clips of the same episode (Mantel-Haenszel), so voices, mic and topic are held fixed. ×1.5 means half as many errors again.',
      ],
      [
        'Points of WER',
        'How much of a set’s WER a condition costs: the errors on its clips beyond what they would have made without it (D111). The rows of the card add up to the WER.',
      ],
    ],
  ],
  [
    'Errors',
    [
      [
        'Error file',
        'Every aligned word pair of an evaluation, matches included, kept as a Parquet file beside the model (D110). The Errors tab reads it.',
      ],
      [
        'Substituted, deleted, inserted',
        'A word written as another, a reference word left out, a word added. Folded and merged pairs are forgiven: the same word written another way, or split or joined.',
      ],
      ['Similarity', 'How alike two words are after romanization, from 0 to 1.'],
    ],
  ],
]

export function Glossary() {
  return (
    <div className="grid gap-6 sm:grid-cols-2">
      {GROUPS.map(([title, entries]) => (
        <section key={title} className="flex flex-col gap-1">
          <h4 className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">{title}</h4>
          <dl className="divide-y">
            {entries.map(([term, meaning]) => (
              <div key={term} className="py-1.5">
                <dt className="text-sm font-semibold">{term}</dt>
                <dd className="text-xs text-muted-foreground">{meaning}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  )
}
