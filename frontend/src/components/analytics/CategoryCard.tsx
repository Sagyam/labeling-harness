/**
 * One category: its buckets as stacked bars, cut by the page's filters (D91).
 *
 * Clicking a bucket filters the rest of the page to it; the card itself keeps showing the whole
 * distribution with the chosen bucket ringed, so the filter can be moved without clearing it.
 * The bar's split follows the page-wide measure -- labels (verified / screened / undecided),
 * pots (gold / val / train) or voices -- and every bar is direct-labelled with its own number.
 */

import { useState } from 'react'

import { GoalBadge, POT_COLOR, Panel, StackRow, type StackSegment } from '@/components/analytics/primitives'
import { bucketLabel, hoursOrMinutes } from '@/components/analytics/labels'
import type { BucketAgg } from '@/components/analytics/model'
import type { CategoryReport } from '@/types'

export type Measure = 'labels' | 'pots' | 'voices'

export const TIER_FILL = {
  verified: 'bg-indigo-500',
  screened: 'bg-indigo-500/40',
  unlabeled: 'bg-muted-foreground/25',
}

/** How many buckets an open category shows before it folds the rest. */
const FOLD_AT = 8

function segmentsFor(entry: BucketAgg, measure: Measure): StackSegment[] {
  if (measure === 'pots') {
    return [
      { value: entry.goldHours, fill: POT_COLOR.gold, title: `gold ${hoursOrMinutes(entry.goldHours)}` },
      { value: entry.valHours, fill: POT_COLOR.val, title: `val ${hoursOrMinutes(entry.valHours)}` },
      { value: entry.trainHours, fill: POT_COLOR.train, title: `train ${hoursOrMinutes(entry.trainHours)}` },
    ]
  }
  if (measure === 'voices') {
    return [
      { value: entry.usableVoices, fill: 'bg-indigo-500', title: `${entry.usableVoices} voices with 300+ words` },
      {
        value: entry.voices - entry.usableVoices,
        fill: 'bg-indigo-500/35',
        title: `${entry.voices - entry.usableVoices} voices with fewer words`,
      },
    ]
  }
  return [
    { value: entry.verifiedHours, fill: TIER_FILL.verified, title: `verified ${hoursOrMinutes(entry.verifiedHours)}` },
    { value: entry.screenedHours, fill: TIER_FILL.screened, title: `screened ${hoursOrMinutes(entry.screenedHours)}` },
    { value: entry.unlabeledHours, fill: TIER_FILL.unlabeled, title: `undecided ${hoursOrMinutes(entry.unlabeledHours)}` },
  ]
}

function captionFor(entry: BucketAgg, measure: Measure, unit: 'voices' | 'hours'): string {
  if (entry.clips === 0) return 'none'
  if (measure === 'voices') return `${entry.usableVoices} of ${entry.voices} voices`
  const people = unit === 'voices' ? ` · ${entry.usableVoices} v` : ''
  return `${hoursOrMinutes(entry.hours)}${people}`
}

export function CategoryCard({
  report,
  entries,
  measure,
  active,
  onToggle,
  minHours,
  minVoices,
  filtered,
}: {
  report: CategoryReport
  entries: BucketAgg[]
  measure: Measure
  /** The bucket the page is filtered to on this category, if any. */
  active: string | null
  onToggle: (bucket: string) => void
  minHours: number
  minVoices: number
  /** Whether other categories' filters are cutting these numbers. */
  filtered: boolean
}) {
  const [open, setOpen] = useState(false)
  const unknown = new Set(report.unknown)
  const defect = new Set(report.defect)
  const measured = entries.filter((e) => !unknown.has(e.bucket))
  const rest = entries.filter((e) => unknown.has(e.bucket))
  const peak = Math.max(
    0,
    ...entries.map((e) => (measure === 'voices' ? e.voices : e.hours))
  )
  const isOpen = report.buckets.length > FOLD_AT && !report.absent.length && report.key !== 'voice'
  const folded = isOpen && !open ? measured.slice(0, FOLD_AT) : measured
  const hidden = measured.length - folded.length
  const voiceCard = report.key === 'voice'
  const shown = voiceCard ? measured.slice(0, FOLD_AT) : folded

  const thin = (e: BucketAgg) =>
    e.clips > 0 &&
    !defect.has(e.bucket) &&
    (report.unit === 'voices' ? e.usableVoices < minVoices : e.hours < minHours)

  return (
    <Panel className="flex min-w-0 flex-col p-3">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <h3 className="truncate font-heading text-sm font-semibold" title={report.why}>
              {report.label}
            </h3>
            {report.goals.length === 2 ? (
              <GoalBadge goal="both" title="Matters for the recogniser and for the study" />
            ) : (
              <GoalBadge goal={report.goals[0]} />
            )}
          </div>
          <p className="line-clamp-2 text-[10px] leading-snug text-muted-foreground" title={report.why}>
            {report.why}
          </p>
        </div>
        <span
          className="shrink-0 font-mono text-[10px] text-muted-foreground"
          title={
            report.unit === 'voices'
              ? `A bucket needs ${minVoices} voices with 300+ words to count as a group`
              : `A bucket needs ${minHours} h to count as a condition`
          }
        >
          floor {report.unit === 'voices' ? `${minVoices} v` : `${minHours} h`}
        </span>
      </div>

      <div className="space-y-0.5">
        {shown.map((entry) => (
          <StackRow
            key={entry.bucket}
            label={
              <span className={entry.clips === 0 ? 'text-muted-foreground' : ''}>
                {bucketLabel(entry.bucket)}
                {report.buckets.find((b) => b.bucket === entry.bucket)?.off_vocabulary ? (
                  <span className="ml-1 text-[10px] text-amber-600 dark:text-amber-400" title="Typed outside the taxonomy (D57)">
                    off-taxonomy
                  </span>
                ) : null}
              </span>
            }
            segments={segmentsFor(entry, measure)}
            peak={peak}
            caption={captionFor(entry, measure, report.unit)}
            active={active === entry.bucket}
            onClick={() => onToggle(entry.bucket)}
            muted={entry.clips === 0}
            marker={
              thin(entry) ? (
                <span
                  className="inline-block size-1.5 shrink-0 rounded-full bg-amber-500"
                  title={report.unit === 'voices' ? `Under ${minVoices} usable voices` : `Under ${minHours} h`}
                />
              ) : entry.clips === 0 ? (
                <span className="inline-block size-1.5 shrink-0 rounded-full bg-rose-500" title="Absent" />
              ) : null
            }
          />
        ))}
        {voiceCard && measured.length > FOLD_AT ? (
          <p className="px-1 pt-1 text-[10px] text-muted-foreground">
            {measured.length - FOLD_AT} more voices in the table below.
          </p>
        ) : null}
        {!voiceCard && hidden > 0 ? (
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="px-1 pt-1 text-[10px] text-indigo-600 hover:underline dark:text-indigo-400"
          >
            {hidden} more
          </button>
        ) : null}
        {isOpen && open ? (
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="px-1 pt-1 text-[10px] text-muted-foreground hover:underline"
          >
            fewer
          </button>
        ) : null}
      </div>

      {rest.some((e) => e.clips > 0) ? (
        <div className="mt-2 space-y-0.5 border-t border-dashed pt-1.5">
          {rest
            .filter((e) => e.clips > 0)
            .map((entry) => (
              <StackRow
                key={entry.bucket}
                label={<span className="text-muted-foreground">{bucketLabel(entry.bucket)}</span>}
                segments={segmentsFor(entry, measure)}
                peak={peak}
                caption={captionFor(entry, measure, report.unit)}
                active={active === entry.bucket}
                onClick={() => onToggle(entry.bucket)}
                muted
              />
            ))}
        </div>
      ) : null}

      {filtered ? (
        <p className="mt-1.5 text-[10px] text-muted-foreground">cut by the active filters</p>
      ) : null}
    </Panel>
  )
}
