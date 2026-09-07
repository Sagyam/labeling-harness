/**
 * Pipeline health, compressed to one row.
 *
 * This is everything the page used to lead with. It is kept because the numbers are real and
 * occasionally decisive — an accept rate that collapses means the upstream ASR stage regressed —
 * but demoted below the corpus panels, because none of it tells you what to record next.
 */

import { RiPulseLine } from '@remixicon/react'

import { Panel, PanelHeading, hours, percent } from './primitives'
import type { AnalyticsReport } from '@/types'

/** Disposition colours are categorical and fixed: a disposition keeps its hue everywhere. */
const DISPOSITION = [
  { key: 'accepted_unchanged', label: 'Accepted', fill: 'bg-emerald-500', text: 'text-emerald-600 dark:text-emerald-400' },
  { key: 'edited', label: 'Edited', fill: 'bg-blue-500', text: 'text-blue-600 dark:text-blue-400' },
  { key: 'unusable_audio', label: 'Unusable', fill: 'bg-rose-500', text: 'text-rose-600 dark:text-rose-400' },
  { key: 'uncertain', label: 'Uncertain', fill: 'bg-amber-500', text: 'text-amber-600 dark:text-amber-400' },
] as const

/**
 * Accept rate per day as a column chart. Bars, not a line: the days are discrete working sessions
 * with gaps between them, and a line would draw a trend across days nobody worked.
 */
function AcceptTrend({ rows }: { rows: AnalyticsReport['accept_rate_by_day'] }) {
  if (rows.length === 0) return <p className="text-[11px] text-muted-foreground">No labels yet.</p>
  const recent = rows.slice(-30)
  return (
    <div>
      <div className="flex h-16 items-end gap-[2px]">
        {recent.map((row) => (
          <div
            key={row.day}
            title={`${row.day}: ${row.accepted}/${row.labeled} accepted unchanged (${percent(row.accept_rate)})`}
            className="flex-1 rounded-t-[3px] bg-emerald-500/70 hover:bg-emerald-500"
            style={{ height: `${Math.max(2, (row.accept_rate ?? 0) * 100)}%` }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>{recent[0]?.day}</span>
        <span>{recent[recent.length - 1]?.day}</span>
      </div>
    </div>
  )
}

export function PipelinePanel({ report }: { report: AnalyticsReport }) {
  const total = report.labels.total || 1
  const throughput = report.throughput

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Panel>
        <PanelHeading
          icon={<RiPulseLine className="size-4 text-primary" />}
          title="What annotators decided"
          note={`${report.labels.total.toLocaleString()} labels`}
        />
        <div className="mb-3 flex h-2.5 w-full gap-[2px] overflow-hidden rounded-sm bg-muted">
          {DISPOSITION.map(({ key, fill, label }) => (
            <div
              key={key}
              className={fill}
              style={{ width: `${((report.labels[key] || 0) / total) * 100}%` }}
              title={`${label}: ${report.labels[key] || 0}`}
            />
          ))}
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-4 lg:grid-cols-2">
          {DISPOSITION.map(({ key, label, fill, text }) => (
            <div key={key} className="flex items-baseline justify-between gap-2">
              <span className="flex items-center gap-1.5 truncate">
                <span className={`size-2 shrink-0 rounded-full ${fill}`} />
                <span className="truncate">{label}</span>
              </span>
              <span className={`font-mono tabular-nums ${text}`}>
                {(report.labels[key] || 0).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel>
        <PanelHeading
          icon={<RiPulseLine className="size-4 text-primary" />}
          title="Accept rate by day"
          note="if this collapses, the upstream ASR stage regressed"
        />
        <AcceptTrend rows={report.accept_rate_by_day} />
      </Panel>

      <Panel>
        <PanelHeading icon={<RiPulseLine className="size-4 text-primary" />} title="Queue and pace" />
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          {[
            ['Backlog', `${report.queue.backlog.toLocaleString()} clips`],
            [
              'Est. left',
              report.queue.projected_hours_to_finish
                ? `${hours(report.queue.projected_hours_to_finish)} h`
                : '--',
            ],
            [
              'Median / clip',
              throughput.median_seconds_per_segment ? `${throughput.median_seconds_per_segment}s` : '--',
            ],
            [
              'Clips / hour',
              throughput.segments_per_hour ? throughput.segments_per_hour.toFixed(0) : '--',
            ],
            ['Annotator hours', hours(throughput.annotator_hours)],
            ['Accept rate', percent(report.accept_rate, 1)],
            [
              'Word timings',
              `${percent(report.word_timestamp_coverage.fraction)} of ${report.word_timestamp_coverage.hypotheses_total}`,
            ],
            ['Mean disagreement', percent(report.scores.mean_word_disagreement_rate, 1)],
          ].map(([label, value]) => (
            <div key={label} className="flex items-baseline justify-between gap-2 border-b border-border/40 pb-1">
              <dt className="truncate text-muted-foreground">{label}</dt>
              <dd className="shrink-0 font-mono tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
      </Panel>
    </div>
  )
}
