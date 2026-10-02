/**
 * Where a set's WER comes from (D111): the points each recording condition costs, and the rest
 * split by kind of error. The rows add up to the WER.
 *
 * Every clip sits in one cell -- its crosstalk bucket if any overlap was measured, else its SNR
 * bucket below 45 dB, else the baseline -- so no error is charged twice. A condition costs
 * `errors x (1 - 1/ratio)`, the ratio taken within episode: each clip against baseline clips of
 * its own episode (same voices, mic, room and topic), pooled with Mantel-Haenszel. The card shows
 * that estimate alone; the clean-floor figure (against the whole set's baseline) is one sentence
 * in the explanation, because a second column of numbers made the card unreadable (owner,
 * 2026-10-02).
 */

import { useState } from 'react'
import { RiArrowDownSLine, RiArrowRightSLine } from '@remixicon/react'

import { Legend, PanelHeading } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import type { Attribution, AttributionCondition, AttributionDiff, AttributionKind } from '@/types'

type Factor = 'crosstalk' | 'snr'

const FACTOR: Record<Factor, { label: string; fill: string; note: string }> = {
  crosstalk: {
    label: 'Crosstalk',
    fill: 'bg-[var(--viz-crosstalk)]',
    note: 'Two voices at once, measured by the overlap detector (D77). Compared with clips of the same episode that have none.',
  },
  snr: {
    label: 'Noise',
    fill: 'bg-[var(--viz-snr)]',
    note: 'Low speech-to-noise ratio, measured by Brouhaha (D87), on crosstalk-free clips. Compared with clips of the same episode at 45 dB or more.',
  },
}

const KIND: Record<AttributionKind, [string, string]> = {
  number: ['a number', 'Either side is a number (digits or a number word); counted first.'],
  deletion: ['a word left out', 'A reference word the model did not write.'],
  insertion: ['a word added', 'A word the model wrote that the reference does not have.'],
  script: [
    'English ↔ Devanagari',
    'A substitution across scripts, or with a mixed word on either side, that fold.py did not match.',
  ],
  english: ['a different English word', 'A substitution between two Latin-script words.'],
  nepali_similar: [
    'a similar Nepali word',
    'Two Devanagari words at romanized similarity 0.75 or more: mostly a different suffix (रहेको/रहेका).',
  ],
  nepali_other: ['a different Nepali word', 'Any other substitution between two Devanagari words.'],
}

const NOT_MEASURABLE =
  'No episode has clips both with and without this condition, so there is nothing to compare it with. Its errors stay in "everything else".'

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

/** The WER as one stacked bar: crosstalk, noise, everything else, each labelled with the
 * table's own figure. A negative estimate (the condition's clips did better than their
 * baseline) cannot be drawn; the bar then says so, and its widths share what is drawn. */
function Bar({ a }: { a: Attribution }) {
  const parts = (['crosstalk', 'snr'] as const).map((factor) => ({
    factor,
    points: Math.max(0, a.factors[factor]?.within.points ?? 0),
  }))
  const rest = Math.max(0, a.rest.within.points)
  const drawn = parts.reduce((s, p) => s + p.points, 0) + rest
  const width = (v: number) => (drawn > 0 ? `${(100 * v) / drawn}%` : '0%')
  const negative = (['crosstalk', 'snr'] as const).filter((f) => (a.factors[f]?.within.points ?? 0) < 0)
  return (
    <div className="space-y-1">
      <BarFills parts={parts} rest={rest} wer={a.wer} width={width} />
      {negative.map((f) => (
        <p key={f} className="text-[10px] text-muted-foreground">
          {FACTOR[f].label} is estimated at {pts(a.factors[f]?.within.points)} points: its clips did slightly better
          than their baseline, which cannot be drawn, so the bar's parts add up to a little more than the WER.
        </p>
      ))}
    </div>
  )
}

function BarFills({
  parts,
  rest,
  wer,
  width,
}: {
  parts: { factor: Factor; points: number }[]
  rest: number
  wer: number
  width: (v: number) => string
}) {
  return (
    <div
      className="flex h-6 w-full gap-[2px]"
      role="img"
      aria-label={`${parts.map((p) => `${FACTOR[p.factor].label} ${pts(p.points)}`).join(', ')}, everything else ${pts(rest)} points of ${pts(wer)}`}
    >
      {parts.map(
        (p) =>
          p.points > 0 && (
            <div
              key={p.factor}
              className={cn('flex items-center overflow-hidden rounded-[4px] px-1.5 text-[11px] text-white', FACTOR[p.factor].fill)}
              style={{ width: width(p.points) }}
              title={`${FACTOR[p.factor].label}: ${pts(p.points)} of ${pts(wer)} points`}
            >
              <span className="truncate font-mono tabular-nums">{pts(p.points)}</span>
            </div>
          ),
      )}
      <div
        className="flex items-center overflow-hidden rounded-[4px] bg-muted px-1.5 text-[11px] text-muted-foreground"
        style={{ width: width(rest) }}
        title={`Everything else: ${pts(rest)} of ${pts(wer)} points`}
      >
        <span className="truncate font-mono tabular-nums">{pts(rest)} everything else</span>
      </div>
    </div>
  )
}

function FactorRows({ a, factor }: { a: Attribution; factor: Factor }) {
  const [open, setOpen] = useState(false)
  const buckets = a.conditions.filter((c) => c.factor === factor)
  const shown = buckets.filter(measurable)
  const total = a.factors[factor]?.within
  const Chevron = open ? RiArrowDownSLine : RiArrowRightSLine
  return (
    <>
      <tr className="cursor-pointer border-t hover:bg-muted/40" onClick={() => setOpen(!open)} aria-expanded={open}>
        <td className="py-1.5" title={FACTOR[factor].note}>
          <span className="flex items-center gap-1.5 font-semibold">
            <Chevron className="size-3.5 text-muted-foreground" />
            <span className={cn('inline-block size-2.5 rounded-sm', FACTOR[factor].fill)} />
            {FACTOR[factor].label}
          </span>
        </td>
        <td className="text-right font-mono tabular-nums">
          {shown.length === 0 ? (
            <span className="font-sans text-[11px] text-muted-foreground" title={NOT_MEASURABLE}>
              not measurable on this set
            </span>
          ) : (
            <>
              <span className="font-semibold">{pts(total?.points)}</span>
              <Ci ci={total?.points_ci} />
            </>
          )}
        </td>
        {a.vs_base && <DiffCell d={a.vs_base.factors[factor]} strong />}
      </tr>
      {open &&
        buckets.map((c) => (
          <tr key={c.bucket} className="text-[11px] text-muted-foreground">
            <td className="py-0.5 pl-9">
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
            {a.vs_base && <DiffCell d={a.vs_base.conditions[`${factor}|${c.bucket}`]} />}
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
  const factors = (['crosstalk', 'snr'] as const).filter((f) => a.factors[f])
  const rest = a.rest.within
  const floor = (factor: Factor) => a.factors[factor]?.floor.points
  return (
    <div className="min-w-0 max-w-3xl space-y-3">
      <PanelHeading
        title={`Where the ${pts(a.wer)} points of WER come from`}
        note={`${setName} · each condition compared within its own episode`}
      />
      <Bar a={a} />
      <Legend
        items={[
          { fill: FACTOR.crosstalk.fill, label: FACTOR.crosstalk.label },
          { fill: FACTOR.snr.fill, label: FACTOR.snr.label },
          { fill: 'bg-muted', label: 'everything else' },
        ]}
      />

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
          {factors.map((factor) => (
            <FactorRows key={factor} a={vs ? a : { ...a, vs_base: undefined }} factor={factor} />
          ))}
          <tr className="border-t">
            <td className="py-1.5">
              <span className="flex items-center gap-1.5 font-semibold">
                <span className="size-3.5" />
                <span className="inline-block size-2.5 rounded-sm bg-muted" />
                Everything else, by kind of error
              </span>
            </td>
            <td className="text-right font-mono tabular-nums">
              <span className="font-semibold">{pts(rest.points)}</span>
              <Ci ci={rest.points_ci} />
            </td>
            {vs && <DiffCell d={vs.rest} strong />}
          </tr>
          {rest.kinds.map((k) => (
            <tr key={k.kind} className="text-[11px]">
              <td className="py-0.5 pl-9 text-muted-foreground" title={KIND[k.kind][1]}>
                {KIND[k.kind][0]}
              </td>
              <td className="text-right font-mono tabular-nums">
                {pts(k.points)}
                <Ci ci={k.points_ci} />
              </td>
              {vs && <DiffCell d={vs.kinds[k.kind]} />}
            </tr>
          ))}
          <tr className="border-t">
            <td className="py-1.5 font-semibold">Total: the WER</td>
            <td className="text-right font-mono font-semibold tabular-nums">{pts(a.wer)}</td>
            {vs && <DiffCell d={vs.wer} strong />}
          </tr>
        </tbody>
      </table>

      <p className="text-[10px] text-muted-foreground">
        The split by kind of error assumes each condition adds errors in the same mix its clips already show.
        {a.unmeasured.clips > 0 &&
          ` ${a.unmeasured.clips} clip(s) were never measured on a condition; their ${a.unmeasured.errors} error(s) are in everything else.`}
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
            "Not measurable" means no episode has clips both with and without the condition. It is an honest gap, not
            a zero: the set cannot answer the question.
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
