/**
 * The shopping list: what to go and record next, highest priority first.
 *
 * This sits at the top of the page because it is the only panel that answers a question with an
 * action. Everything below it is evidence for these rows.
 *
 * Each row carries the measurement that produced it. That is not decoration — a recommendation
 * whose number you cannot see is an opinion, and the ranking is only worth following if you can
 * check what it was computed from.
 */

import { RiCompass3Line, RiSearchLine } from '@remixicon/react'

import { Panel, PanelHeading, hours, percent, severityTone } from './primitives'
import type { Recommendation } from '@/types'

/** What each kind of gap is about, for the grouping label. */
const KIND_LABEL: Record<Recommendation['kind'], string> = {
  gender: 'Speaker gender',
  age_bracket: 'Speaker age',
  register: 'Code-switching',
  register_spread: 'Code-switching',
  show_concentration: 'Show breadth',
  gold_coverage: 'Benchmark span',
  episode_length: 'Episode length',
  topic: 'Topic',
}

function Row({ recommendation }: { recommendation: Recommendation }) {
  const tone = severityTone(recommendation.priority)
  return (
    <li className="flex items-start gap-3 border-b border-border/60 py-2.5 last:border-0">
      <span className={`mt-1.5 size-2 shrink-0 rounded-full ${tone.dot}`} aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-[13px] font-medium leading-snug">{recommendation.target}</span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {KIND_LABEL[recommendation.kind] ?? recommendation.kind}
          </span>
        </div>
        <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
          {recommendation.reason}
        </p>
      </div>
      <div className="shrink-0 text-right">
        <div className={`font-mono text-xs font-bold tabular-nums ${tone.text}`}>
          {percent(recommendation.priority)}
        </div>
        {recommendation.hours_needed > 0 ? (
          <div className="font-mono text-[10px] text-muted-foreground">
            +{hours(recommendation.hours_needed)} h
          </div>
        ) : null}
      </div>
    </li>
  )
}

export function SourcingPanel({
  recommendations,
  minStratumHours,
}: {
  recommendations: Recommendation[]
  minStratumHours: number
}) {
  if (recommendations.length === 0) {
    return (
      <Panel>
        <PanelHeading
          icon={<RiCompass3Line className="size-4 text-primary" />}
          title="What to record next"
        />
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <RiSearchLine className="size-4" />
          Nothing is absent, thin or dominated. Every stratum clears the {minStratumHours} h floor
          and no single show or speaker group holds the corpus.
        </p>
      </Panel>
    )
  }

  const critical = recommendations.filter((r) => r.priority >= 0.75).length

  return (
    <Panel>
      <PanelHeading
        icon={<RiCompass3Line className="size-4 text-primary" />}
        title="What to record next"
        note={
          <>
            Ranked by how much the gap matters times how bad it is. A stratum counts as thin under{' '}
            <span className="font-mono">{minStratumHours} h</span>.
          </>
        }
      >
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {recommendations.length} gaps · {critical} critical
        </span>
      </PanelHeading>
      <ol className="grid grid-cols-1 gap-x-8 lg:grid-cols-2">
        {recommendations.map((recommendation) => (
          <Row key={`${recommendation.kind}:${recommendation.target}`} recommendation={recommendation} />
        ))}
      </ol>
    </Panel>
  )
}
