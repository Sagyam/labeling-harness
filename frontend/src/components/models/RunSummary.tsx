/**
 * One run's headline numbers and where its errors live (D83).
 *
 * The ledger strip carries the numbers docs/findings.md reports -- folded WER with its episode
 * interval, raw WER beside it, CER, loops. The two breakdowns are bars of WER per slice captioned
 * with each slice's share of all errors, and clicking one filters the clip list to it: the point
 * of the page is to go from "podcasts with crosstalk are bad" to the clips themselves.
 */

import { Panel, PanelHeading, Stat, BarRow, humanize, percent } from '@/components/analytics/primitives'
import type { ModelBreakdown, ModelEvalRun, OverlapBucket } from '@/types'

const OVERLAP_ORDER: OverlapBucket[] = ['none', '0-5%', '5-15%', '>15%', 'unmeasured']
const OVERLAP_LABEL: Record<OverlapBucket, string> = {
  none: 'no overlap',
  '0-5%': '0–5% of the clip',
  '5-15%': '5–15% of the clip',
  '>15%': 'over 15% of the clip',
  unmeasured: 'never measured',
}

export function fmtWer(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? '--' : `${value.toFixed(digits)}%`
}

function Breakdown<K extends string>({
  title,
  note,
  entries,
  labels,
  active,
  onPick,
}: {
  title: string
  note: string
  entries: [K, ModelBreakdown][]
  labels?: Record<string, string>
  active: K | null
  onPick: (key: K | null) => void
}) {
  const peak = Math.max(1, ...entries.map(([, e]) => e.wer))
  return (
    <Panel className="min-w-0 flex-1">
      <PanelHeading title={title} note={note} />
      <div className="space-y-1.5">
        {entries.map(([key, entry]) => (
          <BarRow
            key={key}
            label={
              <span>
                {labels?.[key] ?? humanize(key)}{' '}
                <span className="text-[10px] text-muted-foreground">{entry.clips} clips</span>
              </span>
            }
            value={entry.wer}
            peak={peak}
            fill="bg-rose-500/70"
            caption={
              <span>
                {fmtWer(entry.wer, 1)}{' '}
                <span className="text-muted-foreground">· {percent(entry.share_of_errors)} of errors</span>
              </span>
            }
            active={active === key}
            onClick={() => onPick(active === key ? null : key)}
          />
        ))}
      </div>
    </Panel>
  )
}

export function RunSummary({
  run,
  genre,
  overlap,
  onPickGenre,
  onPickOverlap,
}: {
  run: ModelEvalRun
  genre: string | null
  overlap: OverlapBucket | null
  onPickGenre: (genre: string | null) => void
  onPickOverlap: (bucket: OverlapBucket | null) => void
}) {
  const m = run.metrics
  const skipped = (m.skipped?.not_in_split ?? 0) + (m.skipped?.no_reference ?? 0)
  const overlapEntries = OVERLAP_ORDER.filter((key) => m.by_overlap[key]).map(
    (key) => [key, m.by_overlap[key]] as [OverlapBucket, ModelBreakdown],
  )
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 divide-x rounded-lg border bg-card sm:grid-cols-3 lg:grid-cols-6">
        <Stat
          label="Folded WER"
          value={fmtWer(m.wer)}
          sub={m.wer_ci ? `95% CI ${m.wer_ci[0].toFixed(1)}–${m.wer_ci[1].toFixed(1)}` : 'one episode, no CI'}
          tone="text-rose-600 dark:text-rose-400"
          title="Errors pooled over reference words, script-folded (fold.py). Interval from resampling whole episodes."
        />
        <Stat label="Raw WER" value={fmtWer(m.raw_wer)} sub="no script folding" />
        <Stat label="CER" value={fmtWer(m.cer)} sub="folded tokens" />
        <Stat
          label="Loops"
          value={m.loops}
          sub="3-word run ×5"
          tone={m.loops > 0 ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat
          label="Clips"
          value={m.clips}
          sub={`${m.episodes} episodes · ${m.ref_words.toLocaleString()} words`}
        />
        <Stat
          label="Skipped"
          value={skipped}
          sub={
            skipped
              ? `${m.skipped?.not_in_split ?? 0} left ${run.split}, ${m.skipped?.no_reference ?? 0} no label`
              : 'every clip scored'
          }
          tone={skipped ? 'text-amber-600 dark:text-amber-400' : ''}
          title="Clips in the file that were not scored: moved out of the split since the notebook ran, or whose current label has no transcript."
        />
      </div>
      <div className="flex flex-col gap-3 lg:flex-row">
        <Breakdown
          title="By genre"
          note="click to filter the clips"
          entries={Object.entries(m.by_genre)}
          active={genre}
          onPick={onPickGenre}
        />
        <Breakdown
          title="By crosstalk"
          note="share of the clip with two voices at once (D77)"
          entries={overlapEntries}
          labels={OVERLAP_LABEL}
          active={overlap}
          onPick={onPickOverlap}
        />
      </div>
    </div>
  )
}
