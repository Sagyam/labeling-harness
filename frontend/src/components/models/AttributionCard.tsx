/**
 * Where a set's WER comes from (D111): the points each recording condition costs, and the rest
 * split by kind of error. The rows add up to the WER.
 *
 * Every clip sits in one cell -- its crosstalk bucket if any overlap was measured, else its SNR
 * bucket below 45 dB, else the baseline -- so no error is charged twice. A condition costs
 * `errors x (1 - 1/ratio)`, with the ratio taken two ways, shown side by side because the gap
 * between them is itself a finding:
 *
 *   within  each clip against baseline clips of its own episode (same voices, mic, room, topic);
 *   floor   each clip against the whole set's baseline, so it also pays for the shows it is in.
 */

import { Fragment } from 'react'

import { Legend, PanelHeading } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import type { Attribution, AttributionEstimate, AttributionKind, AttributionMethod } from '@/types'

const METHODS: [AttributionMethod, string, string][] = [
  [
    'within',
    'within episode',
    'Each clip is compared only with clean clips of its own episode (Mantel-Haenszel, as By class does): what the condition itself costs.',
  ],
  [
    'floor',
    'against the clean floor',
    "Each clip is compared with the whole set's clean clips: also charges the condition for the shows it comes in.",
  ],
]

const FACTOR: Record<'crosstalk' | 'snr', { label: string; fill: string; note: string }> = {
  crosstalk: {
    label: 'Crosstalk',
    fill: 'bg-[var(--viz-crosstalk)]',
    note: 'Two voices at once, measured by the overlap detector (D77); compared with clips that have none.',
  },
  snr: {
    label: 'Noise (low SNR)',
    fill: 'bg-[var(--viz-snr)]',
    note: "Speech-to-noise ratio, measured by Brouhaha (D87), on crosstalk-free clips; compared with clips at 45 dB or more.",
  },
}

const KIND: Record<AttributionKind, [string, string]> = {
  number: ['a number', 'Either side is a number (digits or a number word); counted first.'],
  deletion: ['a word left out', 'A reference word the model did not write.'],
  insertion: ['a word added', 'A word the model wrote that the reference does not have.'],
  script: [
    'English ↔ Devanagari',
    'A substitution across scripts, or with a mixed word on either side; fold.py did not match it.',
  ],
  english: ['a different English word', 'A substitution between two Latin-script words.'],
  nepali_similar: [
    'a similar Nepali word',
    'Two Devanagari words at romanized similarity 0.75 or more: mostly a different suffix (रहेको/रहेका).',
  ],
  nepali_other: ['a different Nepali word', 'Any other substitution between two Devanagari words.'],
}

function pts(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toFixed(2)
}

function Ci({ ci, digits = 2 }: { ci: [number, number] | null | undefined; digits?: number }) {
  if (!ci) return null
  return (
    <span className="text-[10px] font-normal text-muted-foreground">
      {' '}
      [{ci[0].toFixed(digits)}, {ci[1].toFixed(digits)}]
    </span>
  )
}

const NOT_ESTIMABLE =
  'Not estimated: no episode has clips both with and without this condition, so nothing can be compared. Its errors stay in "everything else".'

function Ratio({ est }: { est: AttributionEstimate }) {
  if (est.ratio === null || est.ratio <= 0) {
    return (
      <span className="text-muted-foreground" title={NOT_ESTIMABLE}>
        —
      </span>
    )
  }
  const groups = est.groups !== undefined ? ` · compared in ${est.groups} episode(s) or groups` : ''
  return (
    <span title={`error rate ${est.ratio.toFixed(2)} times the baseline's${groups}`}>
      ×{est.ratio.toFixed(2)}
      <Ci ci={est.ratio_ci} />
    </span>
  )
}

function Points({ est }: { est: AttributionEstimate }) {
  if (est.points === null) {
    return (
      <span className="text-muted-foreground" title={NOT_ESTIMABLE}>
        —
      </span>
    )
  }
  return (
    <span className="font-semibold">
      {pts(est.points)}
      <Ci ci={est.points_ci} />
    </span>
  )
}

/** One method's WER as a stacked bar: crosstalk, noise, everything else. A negative estimate
 * (the condition's clips did better than its baseline) is drawn as nothing; the table has it. */
function Bar({ a, method, label, note }: { a: Attribution; method: AttributionMethod; label: string; note: string }) {
  const parts = (['crosstalk', 'snr'] as const).map((factor) => ({
    factor,
    points: Math.max(0, a.factors[factor]?.[method].points ?? 0),
  }))
  const rest = Math.max(0, a.wer - parts.reduce((s, p) => s + p.points, 0))
  const width = (v: number) => (a.wer > 0 ? `${(100 * v) / a.wer}%` : '0%')
  return (
    <div className="space-y-0.5">
      <div className="text-[11px] text-muted-foreground" title={note}>
        {label}
      </div>
      <div className="flex h-5 w-full gap-[2px]" role="img" aria-label={`${label}: ${parts.map((p) => `${FACTOR[p.factor].label} ${pts(p.points)}`).join(', ')}, everything else ${pts(rest)} points`}>
        {parts.map(
          (p) =>
            p.points > 0 && (
              <div
                key={p.factor}
                className={cn('flex items-center overflow-hidden rounded-[4px] px-1 text-[10px] text-white', FACTOR[p.factor].fill)}
                style={{ width: width(p.points) }}
                title={`${FACTOR[p.factor].label}: ${pts(p.points)} of ${pts(a.wer)} points`}
              >
                <span className="truncate font-mono tabular-nums">{pts(p.points)}</span>
              </div>
            ),
        )}
        <div
          className="flex items-center overflow-hidden rounded-[4px] bg-muted px-1 text-[10px] text-muted-foreground"
          style={{ width: width(rest) }}
          title={`Everything else: ${pts(rest)} of ${pts(a.wer)} points`}
        >
          <span className="truncate font-mono tabular-nums">{pts(rest)} everything else</span>
        </div>
      </div>
    </div>
  )
}

function bucketName(factor: 'crosstalk' | 'snr', bucket: string): string {
  return factor === 'crosstalk' ? `${bucket} of the clip overlapped` : `SNR ${bucket}`
}

export function AttributionCard({ a, setName }: { a: Attribution; setName: string }) {
  const factors = (['crosstalk', 'snr'] as const).filter((f) => a.factors[f])
  const sum = (method: AttributionMethod) =>
    a.conditions.reduce((s, c) => s + (c[method].points ?? 0), 0) + a.rest[method].points
  return (
    <div className="min-w-0 space-y-3">
      <PanelHeading
        title={`Where the ${pts(a.wer)} points come from`}
        note={`${setName} · points of WER per recording condition, the rest by kind of error · the rows add up to the WER`}
      />
      <div className="grid gap-2 md:grid-cols-2">
        {METHODS.map(([method, label, note]) => (
          <Bar key={method} a={a} method={method} label={label} note={note} />
        ))}
      </div>
      <Legend
        items={[
          { fill: FACTOR.crosstalk.fill, label: FACTOR.crosstalk.label },
          { fill: FACTOR.snr.fill, label: FACTOR.snr.label },
          { fill: 'bg-muted', label: 'everything else' },
        ]}
      />

      <div className="overflow-x-auto">
        <table className="w-full text-xs [&_td]:pr-3 [&_td]:whitespace-nowrap [&_th]:pr-3">
          <thead className="text-[10px] text-muted-foreground">
            <tr className="text-left">
              <th className="py-1 font-normal" />
              <th className="font-normal" title="Clips in this cell">
                clips
              </th>
              <th className="font-normal" title="WER on these clips alone">
                WER here
              </th>
              {METHODS.map(([method, label, note]) => (
                <th key={method} colSpan={2} className="font-normal" title={note}>
                  {label}: ratio · points
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {factors.map((factor) => (
              <Fragment key={factor}>
                <tr className="border-t font-sans">
                  <td className="py-1 font-semibold" title={FACTOR[factor].note}>
                    <span className={cn('mr-1.5 inline-block size-2 rounded-sm', FACTOR[factor].fill)} />
                    {FACTOR[factor].label}
                  </td>
                  <td />
                  <td />
                  {METHODS.map(([method]) => (
                    <Fragment key={method}>
                      <td />
                      <td className="font-mono font-semibold tabular-nums">
                        {pts(a.factors[factor]?.[method].points)}
                        <Ci ci={a.factors[factor]?.[method].points_ci} />
                      </td>
                    </Fragment>
                  ))}
                </tr>
                {a.conditions
                  .filter((c) => c.factor === factor)
                  .map((c) => (
                    <tr key={c.bucket} className="text-[11px]">
                      <td className="py-0.5 pl-4 font-sans text-muted-foreground">{bucketName(factor, c.bucket)}</td>
                      <td>{c.clips}</td>
                      <td>{c.wer.toFixed(2)}</td>
                      {METHODS.map(([method]) => (
                        <Fragment key={method}>
                          <td>
                            <Ratio est={c[method]} />
                          </td>
                          <td>
                            <Points est={c[method]} />
                          </td>
                        </Fragment>
                      ))}
                    </tr>
                  ))}
              </Fragment>
            ))}
            <tr className="border-t font-sans">
              <td className="py-1 font-semibold" title="Every error no condition took: the baseline's, what each condition's clips would have made anyway, and unmeasured clips'.">
                <span className="mr-1.5 inline-block size-2 rounded-sm bg-muted" />
                Everything else, by kind of error
              </td>
              <td />
              <td />
              {METHODS.map(([method]) => (
                <Fragment key={method}>
                  <td />
                  <td className="font-mono font-semibold tabular-nums">
                    {pts(a.rest[method].points)}
                    <Ci ci={a.rest[method].points_ci} />
                  </td>
                </Fragment>
              ))}
            </tr>
            {a.rest.within.kinds.map((k, i) => (
              <tr key={k.kind} className="text-[11px]">
                <td className="py-0.5 pl-4 font-sans text-muted-foreground" title={KIND[k.kind][1]}>
                  {KIND[k.kind][0]}
                </td>
                <td />
                <td />
                {METHODS.map(([method]) => {
                  const entry = a.rest[method].kinds[i]
                  return (
                    <Fragment key={method}>
                      <td />
                      <td>
                        {pts(entry.points)}
                        <Ci ci={entry.points_ci} />
                      </td>
                    </Fragment>
                  )
                })}
              </tr>
            ))}
            <tr className="border-t font-sans">
              <td className="py-1 font-semibold">Total: the WER</td>
              <td />
              <td className="font-mono tabular-nums">{pts(a.wer)}</td>
              {METHODS.map(([method]) => (
                <Fragment key={method}>
                  <td />
                  <td className="font-mono font-semibold tabular-nums">{pts(sum(method))}</td>
                </Fragment>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {a.unmeasured.clips > 0 && (
        <p className="text-[10px] text-muted-foreground">
          {a.unmeasured.clips} clip(s), {a.unmeasured.ref_words.toLocaleString()} words, were never measured on a
          condition; their {a.unmeasured.errors} error(s) stay in everything else.
        </p>
      )}

      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">How this is computed (D111)</summary>
        <div className="mt-2 max-w-3xl space-y-1.5 text-muted-foreground">
          <p>
            Every clip sits in one cell: its crosstalk bucket if any overlap was measured, else its SNR bucket if it is
            under 45 dB, else the baseline (no crosstalk, 45 dB or more). No error is charged twice.
          </p>
          <p>
            A condition's <b>ratio</b> is its clips' error rate over its baseline's. Its cost in <b>points</b> is the
            errors on its clips beyond what the same clips would have made without it, errors × (1 − 1/ratio), per 100
            reference words of the whole set.
          </p>
          <p>
            <b>Within episode</b> compares each clip only with baseline clips of its own episode (a public set's speaker or
            video), pooled with the Mantel-Haenszel estimator: same voices, microphone, room and topic. It is the
            estimate of what the condition itself costs. A bucket no episode can compare shows —.
          </p>
          <p>
            <b>Against the clean floor</b> compares with the whole set's baseline. It also charges the condition for
            what comes with it: on gold, crosstalk lives in podcasts and talk shows, which are harder even where one
            voice speaks. The gap between the two columns is that difference.
          </p>
          <p>
            <b>Everything else</b> is split by kind of error, each clip's errors weighted 1/ratio, which assumes a
            condition adds errors in the mix its clips already show. Intervals are 95%, from 1000 resamples of
            episodes. A negative cost means the condition's clips did better than its baseline: noise, not a benefit.
          </p>
        </div>
      </details>
    </div>
  )
}
