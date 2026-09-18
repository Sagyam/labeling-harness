/**
 * Client-side cuts of the clip table (D91).
 *
 * The backend ships every clip as one compact row with a bucket index per category. Everything
 * the page draws under a filter -- a category card, the cross-tab, the voices' filtered talk time
 * -- is summed here from those rows, so clicking a bucket re-cuts the whole page without a round
 * trip. The arithmetic is the server's: seconds summed per bucket, voices counted per bucket, a
 * voice usable in a bucket once it has `MIN_VOICE_WORDS` reference words there.
 */

import type { ClipTable, CorpusInventory } from '@/types'

/** Category key -> the one bucket the page is filtered to. Filters AND across categories. */
export type Filters = Record<string, string>

/** Reference words a voice needs inside a bucket to count as one of its speakers. */
export const MIN_VOICE_WORDS = 300

export interface BucketAgg {
  bucket: string
  hours: number
  clips: number
  verifiedHours: number
  screenedHours: number
  unlabeledHours: number
  goldHours: number
  valHours: number
  trainHours: number
  /** Distinct linked voices with any clip in the bucket. */
  voices: number
  /** Voices with `MIN_VOICE_WORDS`+ words in the bucket. */
  usableVoices: number
}

interface Acc {
  seconds: number
  clips: number
  verified: number
  screened: number
  gold: number
  val: number
  train: number
  voices: Set<number>
  words: Map<number, number>
}

function acc(): Acc {
  return {
    seconds: 0,
    clips: 0,
    verified: 0,
    screened: 0,
    gold: 0,
    val: 0,
    train: 0,
    voices: new Set(),
    words: new Map(),
  }
}

function finish(bucket: string, a: Acc): BucketAgg {
  let usable = 0
  a.words.forEach((words) => {
    if (words >= MIN_VOICE_WORDS) usable += 1
  })
  const h = (s: number) => s / 3600
  return {
    bucket,
    hours: h(a.seconds),
    clips: a.clips,
    verifiedHours: h(a.verified),
    screenedHours: h(a.screened),
    unlabeledHours: h(Math.max(0, a.seconds - a.verified - a.screened)),
    goldHours: h(a.gold),
    valHours: h(a.val),
    trainHours: h(a.train),
    voices: a.voices.size,
    usableVoices: usable,
  }
}

/** Column positions and the filters resolved to bucket indexes, computed once per cut. */
export class Cut {
  readonly table: ClipTable
  readonly col: Record<string, number>
  readonly tierVerified: number
  readonly tierScreened: number
  readonly potGold: number
  readonly potVal: number
  readonly potTrain: number
  /** `[column, bucket index]` pairs every row must match. */
  private readonly wanted: Array<[number, number]>

  constructor(inventory: CorpusInventory, filters: Filters, exclude: string[] = []) {
    this.table = inventory.clips
    this.col = Object.fromEntries(this.table.columns.map((name, i) => [name, i]))
    this.tierVerified = this.table.tiers.indexOf('verified')
    this.tierScreened = this.table.tiers.indexOf('screened')
    this.potGold = this.table.pots.indexOf('gold')
    this.potVal = this.table.pots.indexOf('val')
    this.potTrain = this.table.pots.indexOf('train')
    this.wanted = []
    for (const [key, bucket] of Object.entries(filters)) {
      if (exclude.includes(key) || !(key in this.col)) continue
      const index = (this.table.buckets[key] ?? []).indexOf(bucket)
      if (index >= 0) this.wanted.push([this.col[key], index])
    }
  }

  matches(row: number[]): boolean {
    for (const [column, index] of this.wanted) if (row[column] !== index) return false
    return true
  }

  /** Add one row to an accumulator. */
  add(a: Acc, row: number[]): void {
    const seconds = row[this.col.seconds]
    a.seconds += seconds
    a.clips += 1
    const tier = row[this.col.tier]
    if (tier === this.tierVerified) a.verified += seconds
    else if (tier === this.tierScreened) a.screened += seconds
    const pot = row[this.col.pot]
    if (pot === this.potGold) a.gold += seconds
    else if (pot === this.potVal) a.val += seconds
    else if (pot === this.potTrain) a.train += seconds
    const voice = row[this.col.voice]
    if (voice >= 0) {
      a.voices.add(voice)
      const words = row[this.col.words]
      if (words > 0) a.words.set(voice, (a.words.get(voice) ?? 0) + words)
    }
  }
}

/**
 * One category's buckets under the filters, with the category's own filter left out so the
 * card keeps showing the whole distribution with the chosen bucket highlighted.
 */
export function aggregateCategory(
  inventory: CorpusInventory,
  key: string,
  filters: Filters
): BucketAgg[] {
  const cut = new Cut(inventory, filters, [key])
  const names = inventory.clips.buckets[key] ?? []
  const accs = names.map(() => acc())
  const column = cut.col[key]
  for (const row of inventory.clips.rows) {
    if (!cut.matches(row)) continue
    cut.add(accs[row[column]], row)
  }
  return names.map((name, i) => finish(name, accs[i]))
}

export function aggregateAll(inventory: CorpusInventory, filters: Filters): Record<string, BucketAgg[]> {
  return Object.fromEntries(
    inventory.categories.map((c) => [c.key, aggregateCategory(inventory, c.key, filters)])
  )
}

export interface CrossTab {
  rows: string[]
  cols: string[]
  cells: BucketAgg[][]
  peakHours: number
  peakVoices: number
}

/** Two categories against each other, with both their own filters left out. */
export function crossTab(
  inventory: CorpusInventory,
  rowKey: string,
  colKey: string,
  filters: Filters
): CrossTab {
  const cut = new Cut(inventory, filters, [rowKey, colKey])
  const rows = inventory.clips.buckets[rowKey] ?? []
  const cols = inventory.clips.buckets[colKey] ?? []
  const accs = rows.map(() => cols.map(() => acc()))
  const rc = cut.col[rowKey]
  const cc = cut.col[colKey]
  for (const row of inventory.clips.rows) {
    if (!cut.matches(row)) continue
    cut.add(accs[row[rc]][row[cc]], row)
  }
  const cells = accs.map((line, i) => line.map((a, j) => finish(`${rows[i]} × ${cols[j]}`, a)))
  return {
    rows,
    cols,
    cells,
    peakHours: Math.max(0, ...cells.flat().map((c) => c.hours)),
    peakVoices: Math.max(0, ...cells.flat().map((c) => c.usableVoices)),
  }
}

export interface Totals {
  hours: number
  clips: number
  episodes: number
  voices: number
  usableVoices: number
  verifiedHours: number
  goldHours: number
}

/** The whole filtered corpus in one line. */
export function totals(inventory: CorpusInventory, filters: Filters): Totals {
  const cut = new Cut(inventory, filters)
  const a = acc()
  const episodes = new Set<number>()
  for (const row of inventory.clips.rows) {
    if (!cut.matches(row)) continue
    cut.add(a, row)
    episodes.add(row[cut.col.episode])
  }
  const done = finish('all', a)
  return {
    hours: done.hours,
    clips: done.clips,
    episodes: episodes.size,
    voices: done.voices,
    usableVoices: done.usableVoices,
    verifiedHours: done.verifiedHours,
    goldHours: done.goldHours,
  }
}

export interface VoiceCut {
  seconds: number
  clips: number
  words: number
}

/** Each voice's clips under the filters, keyed by voice index into `inventory.voices`. */
export function voiceCuts(inventory: CorpusInventory, filters: Filters): Map<number, VoiceCut> {
  const cut = new Cut(inventory, filters, ['voice'])
  const out = new Map<number, VoiceCut>()
  for (const row of inventory.clips.rows) {
    if (!cut.matches(row)) continue
    const voice = row[cut.col.voice]
    if (voice < 0) continue
    const entry = out.get(voice) ?? { seconds: 0, clips: 0, words: 0 }
    entry.seconds += row[cut.col.seconds]
    entry.clips += 1
    entry.words += Math.max(0, row[cut.col.words])
    out.set(voice, entry)
  }
  return out
}

export function hasFilters(filters: Filters): boolean {
  return Object.keys(filters).length > 0
}
