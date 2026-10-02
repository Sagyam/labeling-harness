/**
 * Where a set's WER comes from (D111): nine sources, each a row and a segment of one bar, adding
 * up to the WER. Two are recording conditions (crosstalk and noise), seven are kinds of error.
 *
 * Every clip sits in one cell -- its crosstalk bucket if any overlap was measured, else its SNR
 * bucket below 45 dB, else the baseline -- so no error is charged twice. A condition costs
 * `errors x (1 - 1/ratio)`, the ratio taken within episode: each clip against baseline clips of
 * its own episode (same voices, mic, room and topic), pooled with Mantel-Haenszel. Every error a
 * condition does not take is counted under its kind. The clean-floor figure (against the whole
 * set's baseline) is one sentence in the explanation: a second column made the card unreadable,
 * and one bucket for "everything else" hid where most of the WER is (owner, 2026-10-02).
 *
 * The colours are the dataviz reference palette's eight slots in stack order. Nine sources and
 * eight hues: the two Nepali-word kinds are one family and share slot 8, the similar one hatched.
 */

import { useState } from 'react'
import { RiArrowDownSLine, RiArrowRightSLine } from '@remixicon/react'

import { PanelHeading } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import type { Attribution, AttributionCondition, AttributionDiff, AttributionKind } from '@/types'

type Factor = 'crosstalk' | 'snr'

interface Source {
  /** A recording condition, or a kind of error. */
  key: Factor | AttributionKind
  label: string
  note: string
  fill: string
}

const SOURCES: Source[] = [
  {
    key: 'crosstalk',
    label: 'Crosstalk',
    note: 'Two voices at once, measured by the overlap detector (D77). Compared with clips of the same episode that have none.',
    fill: 'bg-[var(--viz-1)]',
  },
  {
    key: 'snr',
    label: 'Noise',
    note: 'Low speech-to-noise ratio, measured by Brouhaha (D87), on crosstalk-free clips. Compared with clips of the same episode at 45 dB or more.',
    fill: 'bg-[var(--viz-2)]',
  },
  {
    key: 'number',
    label: 'A number',
    note: 'Either side is a number (digits or a number word); counted before every other kind.',
    fill: 'bg-[var(--viz-3)]',
  },
  { key: 'deletion', label: 'A word left out', note: 'A reference word the model did not write.', fill: 'bg-[var(--viz-4)]' },
  {
    key: 'insertion',
    label: 'A word added',
    note: 'A word the model wrote that the reference does not have.',
    fill: 'bg-[var(--viz-5)]',
  },
  {
    key: 'script',
    label: 'English ↔ Devanagari',
    note: 'A substitution across scripts, or with a mixed word on either side, that fold.py did not match.',
    fill: 'bg-[var(--viz-6)]',
  },
  {
    key: 'english',
    label: 'A different English word',
    note: 'A substitution between two Latin-script words.',
    fill: 'bg-[var(--viz-7)]',
  },
  {
    key: 'nepali_similar',
    label: 'A similar Nepali word',
    note: 'Two Devanagari words at romanized similarity 0.75 or more: mostly a different suffix (रहेको/रहेका).',
    fill: 'viz-hatch-8',
  },
  {
    key: 'nepali_other',
    label: 'A different Nepali word',
    note: 'Any other substitution between two Devanagari words.',
    fill: 'bg-[var(--viz-8)]',
  },
]

const NOT_MEASURABLE =
  'No episode has clips both with and without this condition, so there is nothing to compare it with. Its errors are counted under their kinds.'

/** A segment narrower than this share of the bar carries no number; its tooltip, the legend
 * and the table do. */
const LABEL_SHARE = 0.07

function isFactor(key: Source['key']): key is Factor {
  return key === 'crosstalk' || key === 'snr'
}

function pts(value: number | null | undefined): string {
  return value === null || value === undefined ? '' : value.toFixed(2)
}

function Ci({ ci }: { ci: [number, number] | null | undefined }) {
  if (!ci) return null
  return (
    <span className="ml-1.5 text-[10px] font-normal text-muted-foreground">
      {ci[0].toFixed(2)} to {ci[1].toFixed(2)}
    </span>
  )
}

/** This model minus the comparison, coloured only when its interval clears zero: green where it
 * makes fewer errors, red where it makes more. */
function DiffCell({ d, strong = false }: { d: AttributionDiff | undefined; strong?: boolean }) {
  if (!d || d[0] === null) {
    return <td className="text-right text-[10px] text-muted-foreground">{d ? 'not measurable' : ''}</td>
  }
  const [v, lo, hi] = d
  const tone =
    lo !== null && lo > 0
      ? 'text-rose-600 dark:text-rose-400'
      : hi !== null && hi < 0
        ? 'text-emerald-600 dark:text-emerald-400'
        : 'text-muted-foreground'
  return (
    <td className={cn('pl-4 text-right font-mono tabular-nums', tone)}>
      <span className={cn(strong && 'font-semibold')}>
        {v > 0 ? '+' : ''}
        {v.toFixed(2)}
      </span>
      {lo !== null && hi !== null && (
        <span className="ml-1.5 text-[10px] font-normal">
          {lo.toFixed(2)} to {hi.toFixed(2)}
        </span>
      )}
    </td>
  )
}

function measurable(c: AttributionCondition): boolean {
  return c.within.points !== null && c.within.ratio !== null && c.within.ratio > 0
}

function bucketName(factor: Factor, bucket: string): string {
  return factor === 'crosstalk' ? `${bucket} of the clip overlapped` : `speech ${bucket} above the noise`
}

/** Each source's points and interval, within episode; `null` points for a condition no bucket
 * of which could be measured. */
function figures(a: Attribution, s: Source): { points: number | null; ci: [number, number] | null } {
  if (isFactor(s.key)) {
    const any = a.conditions.some((c) => c.factor === s.key && measurable(c))
    const total = a.factors[s.key]?.within
    return { points: any && total ? total.points : null, ci: total?.points_ci ?? null }
  }
  const kind = a.rest.within.kinds.find((k) => k.kind === s.key)
  return { points: kind?.points ?? 0, ci: kind?.points_ci ?? null }
}

/** The WER as one stacked bar, a segment per source labelled with the table's own figure. A
 * negative estimate (a condition's clips did better than their baseline) cannot be drawn; the
 * bar then says so, and its widths share what is drawn. */
function Bar({ a, shown }: { a: Attribution; shown: Source[] }) {
  const parts = shown.map((s) => ({ s, points: Math.max(0, figures(a, s).points ?? 0) }))
  const drawn = parts.reduce((sum, p) => sum + p.points, 0)
  const negative = shown.filter((s) => (figures(a, s).points ?? 0) < 0)
  return (
    <div className="space-y-1">
      <div
        className="flex h-7 w-full gap-[2px]"
        role="img"
        aria-label={parts.map((p) => `${p.s.label} ${pts(p.points)}`).join(', ') + ` points of ${pts(a.wer)}`}
      >
        {parts.map(
          ({ s, points }) =>
            points > 0 && (
              <div
                key={s.key}
                className={cn('flex items-center justify-center overflow-hidden rounded-[4px] text-[11px] text-white', s.fill)}
                style={{ width: `${(100 * points) / drawn}%` }}
                title={`${s.label}: ${pts(points)} of ${pts(a.wer)} points`}
              >
                {points / drawn >= LABEL_SHARE && (
                  <span className="truncate rounded-sm bg-black/25 px-1 font-mono tabular-nums">{pts(points)}</span>
                )}
              </div>
            ),
        )}
      </div>
      <Legend shown={shown} />
      {negative.map((s) => (
        <p key={s.key} className="text-[10px] text-muted-foreground">
          {s.label} is estimated at {pts(figures(a, s).points)} points: its clips did slightly better than their baseline,
          which cannot be drawn, so the bar's parts add up to a little more than the WER.
        </p>
      ))}
    </div>
  )
}

function Legend({ shown }: { shown: Source[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
      {shown.map((s) => (
        <span key={s.key} className="flex items-center gap-1">
          <span className={cn('inline-block size-2.5 rounded-sm', s.fill)} />
          {s.label}
        </span>
      ))}
    </div>
  )
}

function SourceRows({ a, s, vs }: { a: Attribution; s: Source; vs: Attribution['vs_base'] }) {
  const [open, setOpen] = useState(false)
  const { points, ci } = figures(a, s)
  const factor = isFactor(s.key) ? s.key : null
  const buckets = factor ? a.conditions.filter((c) => c.factor === factor) : []
  const Chevron = open ? RiArrowDownSLine : RiArrowRightSLine
  const diff = vs ? (factor ? vs.factors[factor] : vs.kinds[s.key as AttributionKind]) : undefined
  return (
    <>
      <tr
        className={cn('border-t', factor && 'cursor-pointer hover:bg-muted/40')}
        onClick={factor ? () => setOpen(!open) : undefined}
        aria-expanded={factor ? open : undefined}
      >
        <td className="py-1.5" title={s.note}>
          <span className="flex items-center gap-1.5">
            {factor ? <Chevron className="size-3.5 text-muted-foreground" /> : <span className="size-3.5" />}
            <span className={cn('inline-block size-2.5 shrink-0 rounded-sm', s.fill)} />
            <span className={cn(factor && 'font-semibold')}>{s.label}</span>
          </span>
        </td>
        <td className="text-right font-mono tabular-nums">
          {points === null ? (
            <span className="font-sans text-[11px] text-muted-foreground" title={NOT_MEASURABLE}>
              not measurable on this set
            </span>
          ) : (
            <>
              <span className="font-semibold">{pts(points)}</span>
              <Ci ci={ci} />
            </>
          )}
        </td>
        {vs && <DiffCell d={diff} strong />}
      </tr>
      {open &&
        factor &&
        buckets.map((c) => (
          <tr key={c.bucket} className="text-[11px] text-muted-foreground">
            <td className="py-0.5 pl-12">
              {bucketName(factor, c.bucket)}
              <span className="ml-2 text-[10px]">
                {c.clips} clips · WER {c.wer.toFixed(1)}
                {measurable(c) && ` · ×${c.within.ratio!.toFixed(2)} the error rate`}
              </span>
            </td>
            <td className="text-right font-mono tabular-nums">
              {measurable(c) ? (
                <>
                  <span className="text-foreground">{pts(c.within.points)}</span>
                  <Ci ci={c.within.points_ci} />
                </>
              ) : (
                <span className="font-sans" title={NOT_MEASURABLE}>
                  not measurable
                </span>
              )}
            </td>
            {vs && <DiffCell d={vs.conditions[`${factor}|${c.bucket}`]} />}
          </tr>
        ))}
    </>
  )
}

export function AttributionCard({
  a,
  setName,
  baseName,
}: {
  a: Attribution
  setName: string
  /** The comparison model's name, when one is picked and the card carries `vs_base`. */
  baseName?: string | null
}) {
  const vs = baseName ? a.vs_base : undefined
  // A condition the set has no clips in (FLEURS has almost no crosstalk) is left out.
  const shown = SOURCES.filter((s) => !isFactor(s.key) || a.factors[s.key])
  const floor = (factor: Factor) => a.factors[factor]?.floor.points
  return (
    <div className="min-w-0 max-w-3xl space-y-3">
      <PanelHeading
        title={`Where the ${pts(a.wer)} points of WER come from`}
        note={`${setName} · conditions compared within their own episode`}
      />
      <Bar a={a} shown={shown} />

      <table className="w-full text-xs">
        <thead className="text-[10px] text-muted-foreground">
          <tr>
            <th className="py-1 text-left font-normal">click a condition for its buckets</th>
            <th className="text-right font-normal" title="Points of the set's WER, with a 95% interval from resampling episodes">
              points of WER · 95% interval
            </th>
            {vs && (
              <th
                className="pl-4 text-right font-normal"
                title={`This model's points minus ${baseName}'s, on the ${vs.clips} clips both scored; green or red only when the interval clears zero`}
              >
                minus {baseName}
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {shown.map((s) => (
            <SourceRows key={s.key} a={a} s={s} vs={vs} />
          ))}
          <tr className="border-t">
            <td className="py-1.5 font-semibold">Total: the WER</td>
            <td className="text-right font-mono font-semibold tabular-nums">{pts(a.wer)}</td>
            {vs && <DiffCell d={vs.wer} strong />}
          </tr>
        </tbody>
      </table>

      <p className="text-[10px] text-muted-foreground">
        An error crosstalk or noise does not take is counted under its kind, assuming a condition adds errors in the same
        mix its clips already show.
        {a.unmeasured.clips > 0 &&
          ` ${a.unmeasured.clips} clip(s) were never measured on a condition; their ${a.unmeasured.errors} error(s) are counted under their kinds.`}
      </p>

      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">How this is computed (D111)</summary>
        <div className="mt-2 space-y-1.5 text-muted-foreground">
          <p>
            Every clip sits in one place: its crosstalk bucket if any overlap was measured, else its noise bucket if its
            speech is under 45 dB above the noise, else the clean baseline. No error is counted twice.
          </p>
          <p>
            Each clip is compared only with clean clips from <b>its own episode</b> (a public set's speaker or video):
            same voices, microphone, room and topic, so what differs is mostly the condition. The episodes are pooled
            with the Mantel-Haenszel estimator, as By class does. A condition's cost is the errors on its clips beyond
            what the same clips would have made without it, errors × (1 − 1/ratio), per 100 words of the whole set.
            Intervals come from 1000 resamples of episodes. A negative cost means those clips did better than their
            baseline: noise in the estimate, not a benefit.
          </p>
          <p>
            Every other error is counted under its kind, in the order listed: a number first, then a word left out or
            added, then a substitution by script and similarity. A condition's clips keep 1/ratio of their errors here.
          </p>
          <p>
            "Not measurable" means no episode has clips both with and without the condition. It is an honest gap, not
            a zero: the set cannot answer the question, and those clips' errors are counted under their kinds.
          </p>
          {floor('crosstalk') !== undefined && (
            <p>
              Compared with the whole set's clean clips instead, crosstalk would appear to cost{' '}
              {pts(floor('crosstalk'))} points
              {floor('snr') !== undefined && ` and noise ${pts(floor('snr'))}`}. That comparison also charges the
              condition for the shows it comes in: on gold, crosstalk lives in podcasts and talk shows, which are
              harder even where one voice speaks. The difference from the figures above is that.
            </p>
          )}
        </div>
      </details>
    </div>
  )
}
