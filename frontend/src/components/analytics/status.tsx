/**
 * The coverage status of a bucket (D104), drawn the same way everywhere on the corpus page.
 *
 * Status is state, not identity, so it has its own reserved colours -- never a bar's fill -- and
 * always travels with an icon and its word, so it is never read from colour alone.
 */

import {
  RiArrowUpDoubleLine,
  RiCheckLine,
  RiCloseCircleLine,
  RiErrorWarningLine,
  RiQuestionLine,
  RiStackLine,
  RiSubtractLine,
} from '@remixicon/react'

import { cn } from '@/lib/utils'
import type { CoverageStatus } from '@/types'

interface StatusStyle {
  label: string
  title: string
  icon: React.ComponentType<{ className?: string }>
  /** Icon and word. */
  text: string
  /** A quiet background for a chip. */
  chip: string
}

export const STATUS: Record<CoverageStatus, StatusStyle> = {
  missing: {
    label: 'missing',
    title: 'A value of the closed list with no audio at all',
    icon: RiCloseCircleLine,
    text: 'text-rose-600 dark:text-rose-400',
    chip: 'bg-rose-500/10 ring-1 ring-rose-500/30',
  },
  thin: {
    label: 'thin',
    title: "Under the pot's hour floor, or too few voices for a people or content split",
    icon: RiErrorWarningLine,
    text: 'text-amber-600 dark:text-amber-400',
    chip: 'bg-amber-500/10 ring-1 ring-amber-500/30',
  },
  enough: {
    label: 'enough',
    title: 'Over the floor and under plenty',
    icon: RiCheckLine,
    text: 'text-emerald-600 dark:text-emerald-400',
    chip: 'bg-emerald-500/10 ring-1 ring-emerald-500/25',
  },
  plenty: {
    label: 'plenty',
    title: 'Three times the floor: more adds little',
    icon: RiArrowUpDoubleLine,
    text: 'text-sky-600 dark:text-sky-400',
    chip: 'bg-sky-500/10 ring-1 ring-sky-500/30',
  },
  overdone: {
    label: 'overdone',
    title: 'Over half its category: it crowds the other values out',
    icon: RiStackLine,
    text: 'text-violet-600 dark:text-violet-400',
    chip: 'bg-violet-500/10 ring-1 ring-violet-500/30',
  },
  unknown: {
    label: 'unknown',
    title: 'Not measured or not declared: paperwork, not a stratum',
    icon: RiQuestionLine,
    text: 'text-muted-foreground',
    chip: 'bg-muted',
  },
  defect: {
    label: 'defect',
    title: 'A clip the diarizer heard nobody in: a fault, not a condition to collect',
    icon: RiSubtractLine,
    text: 'text-muted-foreground',
    chip: 'bg-muted',
  },
  off_list: {
    label: 'off list',
    title: 'A value outside the closed vocabulary: fix the episode record',
    icon: RiErrorWarningLine,
    text: 'text-muted-foreground',
    chip: 'bg-muted',
  },
  unrated: {
    label: '',
    title: 'Listed, not rated: a show is where audio comes from, not a stratum',
    icon: RiSubtractLine,
    text: 'text-muted-foreground/60',
    chip: 'bg-muted',
  },
}

/** The strata statuses, worst first: the order of the summary and the legend. */
export const RATED: CoverageStatus[] = ['missing', 'thin', 'enough', 'plenty', 'overdone']

export function StatusIcon({ status, className }: { status: CoverageStatus; className?: string }) {
  const style = STATUS[status]
  const Icon = style.icon
  return <Icon className={cn('size-3.5 shrink-0', style.text, className)} aria-label={style.label} />
}

export function StatusTag({ status }: { status: CoverageStatus }) {
  const style = STATUS[status]
  return (
    <span className={cn('inline-flex items-center gap-1 text-[11px]', style.text)} title={style.title}>
      <StatusIcon status={status} />
      {style.label}
    </span>
  )
}

export function StatusLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      {RATED.map((s) => (
        <StatusTag key={s} status={s} />
      ))}
    </div>
  )
}
