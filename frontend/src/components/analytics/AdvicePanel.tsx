/**
 * What to record next, one category at a time (D91).
 *
 * Every row is a gap the backend measured, tagged with the category it belongs to and the purpose
 * it hurts. The chips filter by category; the page's goal switch filters by purpose. Clicking a
 * row that names a bucket filters the page to that bucket, so the evidence for the advice is one
 * click away. The list is computed on the whole corpus, never on the filtered view.
 */

import { useMemo, useState } from 'react'
import { RiCompass3Line } from '@remixicon/react'

import { GoalBadge, Panel, PanelHeading, severityTone } from '@/components/analytics/primitives'
import { GOAL_TITLE, KIND_LABEL, KIND_TITLE, bucketLabel } from '@/components/analytics/labels'
import { cn } from '@/lib/utils'
import type { CategoryReport, Goal, Recommendation } from '@/types'

const SHOW_AT_FIRST = 12

export function AdvicePanel({
  recommendations,
  categories,
  goal,
  category,
  onPickCategory,
  onPickBucket,
}: {
  recommendations: Recommendation[]
  categories: CategoryReport[]
  goal: Goal | 'both'
  /** The category chip that is on, or null for all. */
  category: string | null
  onPickCategory: (key: string | null) => void
  onPickBucket: (category: string, bucket: string) => void
}) {
  const [all, setAll] = useState(false)
  const labels = useMemo(() => Object.fromEntries(categories.map((c) => [c.key, c.label])), [categories])

  const byGoal = recommendations.filter((r) => goal === 'both' || r.goal === 'both' || r.goal === goal)
  const counts = new Map<string, number>()
  for (const r of byGoal) counts.set(r.category, (counts.get(r.category) ?? 0) + 1)
  const rows = category ? byGoal.filter((r) => r.category === category) : byGoal
  const shown = all ? rows : rows.slice(0, SHOW_AT_FIRST)

  return (
    <Panel>
      <PanelHeading
        icon={<RiCompass3Line className="size-4 text-primary" />}
        title="What to record next"
        note={`${rows.length} of ${recommendations.length} rows${goal !== 'both' ? ` for the ${goal === 'asr' ? 'recogniser' : 'paper'}` : ''}; measured on the whole corpus`}
      />

      <div className="mb-3 flex flex-wrap gap-1">
        <Chip active={category === null} onClick={() => onPickCategory(null)}>
          all <span className="opacity-60">{byGoal.length}</span>
        </Chip>
        {categories
          .filter((c) => counts.get(c.key))
          .map((c) => (
            <Chip key={c.key} active={category === c.key} onClick={() => onPickCategory(category === c.key ? null : c.key)}>
              {c.label} <span className="opacity-60">{counts.get(c.key)}</span>
            </Chip>
          ))}
      </div>

      {rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">Nothing to ask for here.</p>
      ) : (
        <ol className="divide-y">
          {shown.map((r, i) => {
            const tone = severityTone(r.priority)
            const clickable = r.bucket !== null && r.kind !== 'unmeasured'
            return (
              <li key={`${r.category}-${r.kind}-${r.bucket}-${i}`}>
                <button
                  type="button"
                  disabled={!clickable}
                  onClick={() => r.bucket && onPickBucket(r.category, r.bucket)}
                  className={cn(
                    'flex w-full items-start gap-3 px-1 py-2 text-left',
                    clickable ? 'hover:bg-muted/60' : 'cursor-default'
                  )}
                  title={clickable ? `Filter the page to ${labels[r.category]}: ${bucketLabel(r.bucket!)}` : undefined}
                >
                  <span className="mt-1 flex w-10 shrink-0 items-center gap-1.5">
                    <span className={`inline-block size-2 rounded-full ${tone.dot}`} />
                    <span className={`font-mono text-[11px] tabular-nums ${tone.text}`}>{r.priority.toFixed(2)}</span>
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm leading-snug">{r.target}</span>
                    <span className="block text-[11px] leading-snug text-muted-foreground">{r.reason}</span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1 pt-0.5">
                    <span className="flex items-center gap-1">
                      <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">{labels[r.category] ?? r.category}</span>
                      <GoalBadge goal={r.goal} title={GOAL_TITLE[r.goal]} />
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground" title={KIND_TITLE[r.kind]}>
                      {KIND_LABEL[r.kind]}
                      {r.voices_needed > 0 ? ` · +${r.voices_needed} voices` : ''}
                      {r.hours_needed > 0 && r.voices_needed === 0 ? ` · +${r.hours_needed.toFixed(2)} h` : ''}
                    </span>
                  </span>
                </button>
              </li>
            )
          })}
        </ol>
      )}
      {rows.length > SHOW_AT_FIRST ? (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-2 text-[11px] text-indigo-600 hover:underline dark:text-indigo-400"
        >
          {all ? 'Show the top rows only' : `Show all ${rows.length}`}
        </button>
      ) : null}
    </Panel>
  )
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-full border px-2 py-0.5 text-[11px]',
        active ? 'border-indigo-500 bg-indigo-500/15 text-indigo-700 dark:text-indigo-300' : 'hover:bg-muted'
      )}
    >
      {children}
    </button>
  )
}
