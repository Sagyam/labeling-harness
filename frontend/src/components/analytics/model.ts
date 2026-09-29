/**
 * Client-side cuts of the clip table (D91, D104).
 *
 * The backend rates every bucket of every category per pot; what the page still computes itself is
 * the cross-tab, any category against any other, inside one pot. Gold and train/val are never
 * summed together: a benchmark and a training set want different things (D104).
 */

import type { CorpusInventory, CorpusPot } from '@/types'

export interface CellAgg {
  hours: number
  clips: number
  verifiedHours: number
  /** Distinct linked voices with any clip in the cell. */
  voices: number
  /** Voices with the pot's word floor inside the cell. */
  usableVoices: number
}

export interface CrossTab {
  rows: string[]
  cols: string[]
  cells: CellAgg[][]
  peakHours: number
  peakVoices: number
}

interface Acc {
  seconds: number
  clips: number
  verified: number
  voices: Set<number>
  words: Map<number, number>
}

const acc = (): Acc => ({ seconds: 0, clips: 0, verified: 0, voices: new Set(), words: new Map() })

/** Whether a clip's pot index belongs to a view: `gold`, or `train` for everything else. */
function potMatcher(inventory: CorpusInventory, pot: CorpusPot): (index: number) => boolean {
  const gold = inventory.clips.pots.indexOf('gold')
  return pot === 'gold' ? (i) => i === gold : (i) => i !== gold
}

/** Two categories against each other, inside one pot. */
export function crossTab(
  inventory: CorpusInventory,
  pot: CorpusPot,
  rowKey: string,
  colKey: string,
): CrossTab {
  const table = inventory.clips
  const col = Object.fromEntries(table.columns.map((name, i) => [name, i]))
  const inPot = potMatcher(inventory, pot)
  const verified = table.tiers.indexOf('verified')
  const voiceWords = inventory.pots[pot].floor.voice_words
  const rows = table.buckets[rowKey] ?? []
  const cols = table.buckets[colKey] ?? []
  const accs = rows.map(() => cols.map(() => acc()))
  for (const row of table.rows) {
    if (!inPot(row[col.pot])) continue
    const a = accs[row[col[rowKey]]][row[col[colKey]]]
    const seconds = row[col.seconds]
    a.seconds += seconds
    a.clips += 1
    if (row[col.tier] === verified) a.verified += seconds
    const voice = row[col.voice]
    if (voice >= 0) {
      a.voices.add(voice)
      const words = row[col.words]
      if (words > 0) a.words.set(voice, (a.words.get(voice) ?? 0) + words)
    }
  }
  const cells = accs.map((line) =>
    line.map((a) => {
      let usable = 0
      a.words.forEach((w) => {
        if (w >= voiceWords) usable += 1
      })
      return {
        hours: a.seconds / 3600,
        clips: a.clips,
        verifiedHours: a.verified / 3600,
        voices: a.voices.size,
        usableVoices: usable,
      }
    }),
  )
  return {
    rows,
    cols,
    cells,
    peakHours: Math.max(0, ...cells.flat().map((c) => c.hours)),
    peakVoices: Math.max(0, ...cells.flat().map((c) => c.usableVoices)),
  }
}
