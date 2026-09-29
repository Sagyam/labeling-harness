/**
 * What one pot is short of and what it has plenty of (D104).
 *
 * `GapSummary` is the one-look answer: every missing or thin bucket on the left, every plentiful
 * one on the right, grouped by category. `CategoryTable` is the evidence for one category: a row
 * per bucket with its status, its hours as a bar against the pot's floor, and its usable voices as
 * a number -- hours and voices are different quantities and never share a bar.
 */

import { useState } from 'react'

import { STATUS, StatusIcon, StatusTag } from '@/components/analytics/status'
import { bucketLabel, hoursOrMinutes } from '@/components/analytics/labels'
import { cn } from '@/lib/utils'
import type { BucketStats, CategoryReport, CoverageStatus, PotReport } from '@/types'

function Chip({ entry }: { entry: BucketStats }) {
  const style = STATUS[entry.status]
  return (
    <span
      className={cn('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px]', style.chip)}
      title={`${style.label}${entry.reason ? `: ${entry.reason}` : ''} · ${hoursOrMinutes(entry.hours)} · ${entry.usable_voices} usable voice${entry.usable_voices === 1 ? '' : 's'} · ${entry.episodes} episode${entry.episodes === 1 ? '' : 's'}`}
    >
      <StatusIcon status={entry.status} className="size-3" />
      <span>{bucketLabel(entry.bucket)}</span>
      {entry.status !== 'missing' ? (
        <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
          {entry.status === 'thin' && entry.reason.includes('voice')
            ? entry.reason.split(',')[0]
            : hoursOrMinutes(entry.hours)}
        </span>
      ) : null}
    </span>
  )
}

type Row = { report: CategoryReport; entries: BucketStats[] }

function Rows({ rows }: { rows: Row[] }) {
  return (
    <dl className="divide-y divide-border/60">
      {rows.map(({ report, entries }) => (
        <div key={report.key} className="grid grid-cols-[7.5rem_1fr] gap-2 py-1.5">
          <dt className="pt-0.5 text-xs text-muted-foreground">{report.label}</dt>
          <dd className="flex flex-wrap gap-1">
            {entries.map((e) => (
              <Chip key={e.bucket} entry={e} />
            ))}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function Column({
  title,
  note,
  rows,
  empty,
  quiet = [],
  quietLabel,
}: {
  title: React.ReactNode
  note: string
  rows: Row[]
  empty: string
  /** Rows folded away until asked for: the conditions, where plenty is the normal state. */
  quiet?: Row[]
  quietLabel?: string
}) {
  const [open, setOpen] = useState(false)
  const folded = quiet.reduce((n, r) => n + r.entries.length, 0)
  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-baseline gap-2">
        <h3 className="font-heading text-sm font-semibold">{title}</h3>
        <span className="text-[11px] text-muted-foreground">{note}</span>
      </div>
      {rows.length === 0 && folded === 0 ? (
        <p className="text-xs text-muted-foreground">{empty}</p>
      ) : (
        <>
          <Rows rows={open ? [...rows, ...quiet] : rows} />
          {folded > 0 ? (
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              className="mt-1 text-[11px] text-indigo-600 hover:underline dark:text-indigo-400"
            >
              {open ? 'Hide' : 'Show'} {folded} {quietLabel}
            </button>
          ) : null}
        </>
      )}
    </div>
  )
}

function pick(pot: PotReport, statuses: CoverageStatus[]): Row[] {
  return pot.categories
    .map((report) => ({
      report,
      entries: report.buckets
        .filter((b) => statuses.includes(b.status))
        .sort((a, b) => statuses.indexOf(a.status) - statuses.indexOf(b.status)),
    }))
    .filter((row) => row.entries.length > 0)
}

const count = (rows: Row[], status: CoverageStatus) =>
  rows.reduce((n, r) => n + r.entries.filter((e) => e.status === status).length, 0)

export function GapSummary({ pot }: { pot: PotReport }) {
  const short = pick(pot, ['missing', 'thin'])
  const much = pick(pot, ['overdone', 'plenty'])
  // Overdone always shows; plenty of a speech or acoustic condition is the normal state of a
  // large pot, so those rows fold away unless they are overdone.
  const isCondition = (r: Row) => r.report.group === 'speech' || r.report.group === 'acoustics'
  const loud = much
    .map((r) => (isCondition(r) ? { ...r, entries: r.entries.filter((e) => e.status === 'overdone') } : r))
    .filter((r) => r.entries.length > 0)
  const quiet = much
    .filter(isCondition)
    .map((r) => ({ ...r, entries: r.entries.filter((e) => e.status === 'plenty') }))
    .filter((r) => r.entries.length > 0)
  return (
    <section className="grid gap-6 rounded-lg border bg-card p-4 lg:grid-cols-2">
      <Column
        title={
          <span className="flex items-center gap-1.5">
            <StatusIcon status="thin" className="size-4" />
            Needs more
          </span>
        }
        note={`${count(short, 'missing')} missing · ${count(short, 'thin')} thin`}
        rows={short}
        empty="Nothing is missing or thin."
      />
      <Column
        title={
          <span className="flex items-center gap-1.5">
            <StatusIcon status="overdone" className="size-4" />
            Overdone and plenty
          </span>
        }
        note={`${count(much, 'overdone')} overdone · ${count(much, 'plenty')} plenty`}
        rows={loud}
        quiet={quiet}
        quietLabel="plentiful speech and acoustic conditions"
        empty="Nothing has reached plenty yet."
      />
    </section>
  )
}

const OPEN_LIMIT = 10

/** One category in one pot: a row per bucket. */
export function CategoryTable({ report, pot }: { report: CategoryReport; pot: PotReport }) {
  const [all, setAll] = useState(false)
  const floor = pot.floor.thin_hours
  const measured = report.buckets.filter((b) => !report.unknown.includes(b.bucket))
  const unknown = report.buckets.filter((b) => report.unknown.includes(b.bucket) && b.clips > 0)
  // Only an open list is cut short: a closed one's empty buckets are the point.
  const limited = !report.rated && measured.length > OPEN_LIMIT
  const shown = all || !limited ? measured : measured.slice(0, OPEN_LIMIT)
  const peak = Math.max(floor, ...report.buckets.map((b) => b.hours))
  const tick = report.rated ? (floor / peak) * 100 : null

  const row = (b: BucketStats, muted = false) => (
    <tr key={b.bucket} className={cn('border-b border-border/40 last:border-0', muted && 'text-muted-foreground')}>
      <td className="w-5 py-1 pr-1 align-middle">
        {b.status !== 'unrated' ? <StatusIcon status={b.status} /> : null}
      </td>
      <td className="max-w-40 truncate py-1 pr-2 text-xs" title={b.bucket}>
        {bucketLabel(b.bucket)}
      </td>
      <td className="w-full py-1 pr-2">
        <div className="flex items-center gap-2">
          <div className="relative h-2 flex-1 rounded-sm bg-muted">
            {b.hours > 0 ? (
              <div
                className={cn('absolute inset-y-0 left-0 rounded-sm', muted ? 'bg-muted-foreground/40' : 'bg-indigo-500')}
                style={{ width: `${Math.max(1, (b.hours / peak) * 100)}%` }}
              />
            ) : null}
            {tick !== null ? (
              <div
                className="absolute -inset-y-0.5 w-px bg-foreground/50"
                style={{ left: `${tick}%` }}
                title={`thin under ${hoursOrMinutes(floor)}`}
              />
            ) : null}
          </div>
          <span className="w-14 shrink-0 text-right font-mono text-[11px] tabular-nums">
            {b.hours > 0 ? hoursOrMinutes(b.hours) : '—'}
          </span>
        </div>
      </td>
      <td
        className="py-1 pr-2 text-right font-mono text-[11px] tabular-nums"
        title={`${b.usable_voices} of ${b.voices} voices have ${pot.floor.voice_words}+ words here`}
      >
        {b.clips > 0 ? b.usable_voices : '—'}
      </td>
      <td className={cn('whitespace-nowrap py-1 text-[11px]', STATUS[b.status].text)}>{b.reason}</td>
    </tr>
  )

  return (
    <section className="rounded-lg border bg-card p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="font-heading text-sm font-semibold" title={report.why}>
          {report.label}
        </h3>
        {report.rated ? (
          <div className="flex flex-wrap gap-x-2.5">
            {(['missing', 'thin', 'plenty', 'overdone'] as const).map((s) =>
              report.counts[s] ? (
                <span key={s} className="flex items-center gap-1">
                  <StatusTag status={s} />
                  <span className="font-mono text-[11px] text-muted-foreground">{report.counts[s]}</span>
                </span>
              ) : null,
            )}
          </div>
        ) : (
          <span className="text-[11px] text-muted-foreground">listed, not rated</span>
        )}
      </div>
      <p className="mb-2 text-[11px] leading-snug text-muted-foreground">{report.why}</p>
      <table className="w-full">
        <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
          <tr className="border-b">
            <th />
            <th className="pb-1 text-left font-semibold">Bucket</th>
            <th className="pb-1 text-left font-semibold">Hours</th>
            <th className="pb-1 pr-2 text-right font-semibold" title={`Voices with ${pot.floor.voice_words}+ attributed words in the bucket`}>
              Voices
            </th>
            <th />
          </tr>
        </thead>
        <tbody>
          {shown.map((b) => row(b))}
          {unknown.map((b) => row(b, true))}
        </tbody>
      </table>
      {limited ? (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-1 text-[11px] text-indigo-600 hover:underline dark:text-indigo-400"
        >
          {all ? `Show the top ${OPEN_LIMIT}` : `Show all ${measured.length}`}
        </button>
      ) : null}
    </section>
  )
}
