/**
 * Small pieces the analytics panels are built from.
 *
 * No chart library. Everything here is a div or an inline SVG, which is enough for the four
 * forms this page needs (bar, heat cell, histogram, dot strip) and keeps the bundle honest.
 *
 * Colour follows two rules, and they are different rules:
 *
 * * **Identity is categorical and fixed.** A pot is always the same hue — gold amber, train
 *   emerald, val sky — so a filter that removes a series never repaints the survivors.
 * * **Magnitude is sequential**, one hue light to dark (`HEAT_STEPS`). Hours in a matrix cell are
 *   a quantity, not a category, and a rainbow over a quantity is unreadable.
 *
 * Every mark is direct-labelled with its own number, which is also what discharges the contrast
 * warning on the mid-tone fills: identity and value are never carried by colour alone.
 */

import type { ReactNode } from 'react'

/** Categorical: which pot. Stable per entity, never assigned by rank. */
export const POT_COLOR: Record<string, string> = {
  gold: 'bg-amber-500',
  train: 'bg-emerald-500',
  val: 'bg-sky-500',
  mixed: 'bg-orange-400',
  unassigned: 'bg-muted-foreground/40',
}

export const POT_TEXT: Record<string, string> = {
  gold: 'text-amber-600 dark:text-amber-400',
  train: 'text-emerald-600 dark:text-emerald-400',
  val: 'text-sky-600 dark:text-sky-400',
  mixed: 'text-orange-600 dark:text-orange-400',
  unassigned: 'text-muted-foreground',
}

/**
 * Sequential ramp for hours. Light to dark in one hue, so a darker cell is unambiguously more
 * audio. Index 0 is "none at all" and is deliberately not a pale tint of the ramp — an empty
 * stratum is a different kind of thing from a small one, and the page's whole job is to make that
 * difference visible.
 */
export const HEAT_STEPS = [
  'bg-indigo-500/15',
  'bg-indigo-500/35',
  'bg-indigo-500/55',
  'bg-indigo-500/75',
  'bg-indigo-500',
] as const

/** Which ramp step a value falls in, against the largest value present. */
export function heatStep(value: number, peak: number): number {
  if (value <= 0 || peak <= 0) return -1
  return Math.min(HEAT_STEPS.length - 1, Math.ceil((value / peak) * HEAT_STEPS.length) - 1)
}

/** Status ramp for how badly something is missing. Always shown beside the number itself. */
export function severityTone(priority: number): { dot: string; text: string; label: string } {
  if (priority >= 0.75) return { dot: 'bg-rose-500', text: 'text-rose-600 dark:text-rose-400', label: 'critical' }
  if (priority >= 0.5) return { dot: 'bg-amber-500', text: 'text-amber-600 dark:text-amber-400', label: 'serious' }
  return { dot: 'bg-sky-500', text: 'text-sky-600 dark:text-sky-400', label: 'worth doing' }
}

export function hours(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  if (value === 0) return '0'
  return value >= 10 ? value.toFixed(1) : value.toFixed(2)
}

export function percent(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined) return '--'
  return `${(fraction * 100).toFixed(digits)}%`
}

/** `age_bracket` -> `age bracket`, for anything that reaches a human. */
export function humanize(value: string): string {
  return value.replace(/_/g, ' ')
}

/**
 * One figure in the ledger strip. Deliberately not a card: nine of these need to fit on one row,
 * and a card's padding is most of what stops that.
 */
export function Stat({
  label,
  value,
  sub,
  tone = '',
  title,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: string
  title?: string
}) {
  return (
    <div className="min-w-0 px-3 py-2" title={title}>
      <div className="truncate font-heading text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className={`truncate font-mono text-lg font-bold leading-tight ${tone}`}>{value}</div>
      {sub ? <div className="truncate text-[10px] text-muted-foreground">{sub}</div> : null}
    </div>
  )
}

/**
 * A labelled horizontal bar. The label, the value and the bar all live on one line so a list of
 * these reads as a table that happens to be drawn.
 */
export function BarRow({
  label,
  value,
  peak,
  caption,
  fill = 'bg-indigo-500',
  secondary,
  secondaryFill = 'bg-indigo-500/35',
  onClick,
  active = false,
}: {
  label: ReactNode
  value: number
  peak: number
  caption?: ReactNode
  fill?: string
  /** A second, smaller quantity drawn inside the same track — labelled audio inside ingested. */
  secondary?: number
  secondaryFill?: string
  onClick?: () => void
  active?: boolean
}) {
  const width = peak > 0 ? Math.max(1.5, (value / peak) * 100) : 0
  const secondaryWidth = peak > 0 && secondary ? Math.max(0, (secondary / peak) * 100) : 0
  const Element = onClick ? 'button' : 'div'
  return (
    <Element
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      className={`group w-full space-y-1 rounded px-1 py-0.5 text-left ${
        onClick ? 'hover:bg-muted/60' : ''
      } ${active ? 'bg-muted' : ''}`}
    >
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="min-w-0 truncate">{label}</span>
        <span className="shrink-0 font-mono tabular-nums">{caption}</span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-sm bg-muted">
        <div className={`absolute inset-y-0 left-0 ${secondaryFill}`} style={{ width: `${width}%` }} />
        {secondary !== undefined ? (
          <div className={`absolute inset-y-0 left-0 ${fill}`} style={{ width: `${secondaryWidth}%` }} />
        ) : (
          <div className={`absolute inset-y-0 left-0 ${fill}`} style={{ width: `${width}%` }} />
        )}
      </div>
    </Element>
  )
}

/** A chip. Filled means present, dashed means known-about and absent. */
export function Chip({
  children,
  tone = 'present',
  title,
}: {
  children: ReactNode
  tone?: 'present' | 'absent' | 'dirty'
  title?: string
}) {
  const styles = {
    present: 'bg-indigo-500/15 text-indigo-700 dark:text-indigo-300',
    absent: 'border border-dashed border-muted-foreground/50 text-muted-foreground',
    dirty: 'border border-amber-500/60 text-amber-700 dark:text-amber-400',
  }[tone]
  return (
    <span title={title} className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${styles}`}>
      {children}
    </span>
  )
}

/** Section heading with an optional right-hand note. Keeps every panel's header identical. */
export function PanelHeading({
  icon,
  title,
  note,
  children,
}: {
  icon?: ReactNode
  title: string
  note?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="font-heading text-sm font-semibold">{title}</h2>
        {children}
      </div>
      {note ? <p className="text-[11px] text-muted-foreground">{note}</p> : null}
    </div>
  )
}

/** The panel shell. A plain bordered box — dense pages do not need card chrome. */
export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-lg border bg-card p-4 ${className}`}>{children}</section>
}

/** One segment of a stacked bar: a quantity and the fill that identifies it. */
export interface StackSegment {
  value: number
  fill: string
  title?: string
}

/**
 * A stacked horizontal bar in one line with its label and caption. Segments are drawn left to
 * right against a shared peak, so bars in one list compare by length and the split inside a
 * bar compares by colour. Identity lives in the fill and the caption, never in the fill alone.
 */
export function StackRow({
  label,
  segments,
  peak,
  caption,
  onClick,
  active = false,
  muted = false,
  marker,
}: {
  label: ReactNode
  segments: StackSegment[]
  peak: number
  caption?: ReactNode
  onClick?: () => void
  active?: boolean
  /** An unknown or absent bucket: drawn quieter so the measured ones carry the card. */
  muted?: boolean
  /** A small mark before the label -- the thin-stratum dot. */
  marker?: ReactNode
}) {
  const Element = onClick ? 'button' : 'div'
  const total = segments.reduce((sum, s) => sum + s.value, 0)
  return (
    <Element
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      aria-pressed={onClick ? active : undefined}
      className={`group w-full space-y-0.5 rounded px-1 py-0.5 text-left ${
        onClick ? 'hover:bg-muted/60' : ''
      } ${active ? 'bg-muted ring-1 ring-indigo-500/50' : ''} ${muted ? 'opacity-70' : ''}`}
    >
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="flex min-w-0 items-center gap-1.5 truncate">
          {marker}
          <span className="truncate">{label}</span>
        </span>
        <span className="shrink-0 font-mono text-[11px] tabular-nums">{caption}</span>
      </div>
      <div className="flex h-2 w-full overflow-hidden rounded-sm bg-muted">
        {peak > 0 &&
          segments.map((segment, i) =>
            segment.value > 0 ? (
              <div
                key={i}
                title={segment.title}
                className={`h-full ${segment.fill}`}
                style={{ width: `${Math.max(total > 0 ? 0.6 : 0, (segment.value / peak) * 100)}%` }}
              />
            ) : null
          )}
      </div>
    </Element>
  )
}

/** Which of the corpus's two purposes something serves. Same words everywhere on the page. */
export function GoalBadge({ goal, title }: { goal: 'asr' | 'paper' | 'both'; title?: string }) {
  const styles = {
    asr: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
    paper: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
    both: 'bg-muted text-foreground/80',
  }[goal]
  const text = { asr: 'ASR', paper: 'paper', both: 'both' }[goal]
  return (
    <span title={title} className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide ${styles}`}>
      {text}
    </span>
  )
}

/** A quiet key: swatch and word, for a stacked bar's fills. */
export function Legend({ items }: { items: Array<{ fill: string; label: string }> }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1">
          <span className={`inline-block size-2 rounded-sm ${item.fill}`} />
          {item.label}
        </span>
      ))}
    </div>
  )
}
