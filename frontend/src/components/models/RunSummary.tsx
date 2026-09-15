/**
 * One run's headline numbers and where its errors live (D83, D87).
 *
 * The ledger strip carries the numbers docs/findings.md reports -- folded WER with its episode
 * interval, raw WER beside it, CER, loops. Below it, the run split two ways: by genre, and by
 * every class of roadmap item 1 (crosstalk, speakers, pauses, bandwidth, voices...). Each class
 * bucket shows its WER and share of errors, and -- against the axis's baseline -- its
 * within-episode rate ratio: the ratio of error rates inside each episode, pooled. A ratio whose
 * 95% interval holds 1 is a condition ruled out as a source of errors, which is half of what the
 * panel is for. Clicking a bar filters the clip list to it.
 */

import { Panel, PanelHeading, Stat, BarRow, humanize, percent } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import type { ClassAxis, ModelBreakdown, ModelEvalRun } from '@/types'

/** Bucket names that read badly on their own. Everything else is humanized. */
const BUCKET_LABEL: Record<string, string> = {
  none: 'none',
  '0-5%': '0–5% of the clip',
  '5-15%': '5–15% of the clip',
  '>15%': 'over 15% of the clip',
  unmeasured: 'never measured',
  undiarized: 'never diarized',
  unlinked: 'no linked voice',
  undeclared: 'not declared',
  mixed: 'speakers differ',
  unseen: 'not in train',
}

export function fmtWer(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? '--' : `${value.toFixed(digits)}%`
}

export function bucketLabel(bucket: string): string {
  return BUCKET_LABEL[bucket] ?? humanize(bucket)
}

type Verdict = 'splits' | 'ruled out' | 'no ratio' | 'descriptive'

/** Whether an interval clears 1, and which way. */
function effect(entry: ModelBreakdown): 'up' | 'down' | 'none' | null {
  const ci = entry.rate_ratio_ci
  if (!ci) return null
  if (ci[0] > 1) return 'up'
  if (ci[1] < 1) return 'down'
  return 'none'
}

export function verdict(axis: ClassAxis, buckets: Record<string, ModelBreakdown> | undefined): Verdict {
  if (axis.descriptive) return 'descriptive'
  const effects = Object.values(buckets ?? {}).map(effect).filter((e) => e !== null)
  if (effects.some((e) => e === 'up' || e === 'down')) return 'splits'
  return effects.length ? 'ruled out' : 'no ratio'
}

const VERDICT_STYLE: Record<Verdict, string> = {
  splits: 'bg-rose-500/15 text-rose-700 dark:text-rose-300',
  'ruled out': 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  'no ratio': 'border border-dashed border-muted-foreground/50 text-muted-foreground',
  descriptive: 'bg-muted text-muted-foreground',
}

const VERDICT_TITLE: Record<Verdict, string> = {
  splits: 'At least one bucket changes the error rate: its 95% interval excludes 1.',
  'ruled out': "Every bucket's 95% interval holds 1: no effect on errors at this sample size.",
  'no ratio': 'Nothing to compare within an episode: no baseline, or no episode holds both.',
  descriptive: 'Kept to describe the corpus, not to explain errors.',
}

function RatioCaption({ entry }: { entry: ModelBreakdown }) {
  if (entry.rate_ratio === undefined) return null
  if (entry.rate_ratio === null) {
    return <span className="text-muted-foreground" title="No episode has clips here and in the baseline."> · ×–</span>
  }
  const direction = effect(entry)
  const tone =
    direction === 'up'
      ? 'text-rose-600 dark:text-rose-400'
      : direction === 'down'
        ? 'text-emerald-600 dark:text-emerald-400'
        : 'text-muted-foreground'
  const ci = entry.rate_ratio_ci
  return (
    <span
      className={tone}
      title={`Within-episode error-rate ratio against the baseline, over ${entry.rate_ratio_episodes ?? 0} episodes with both.`}
    >
      {' '}
      · ×{entry.rate_ratio.toFixed(2)}
      {ci ? ` [${ci[0].toFixed(2)}–${ci[1].toFixed(2)}]` : ''}
    </span>
  )
}

function Breakdown({
  title,
  note,
  entries,
  baseline,
  active,
  onPick,
  className,
}: {
  title?: string
  note?: string
  entries: [string, ModelBreakdown][]
  baseline?: string | null
  active: string | null
  onPick: (key: string | null) => void
  className?: string
}) {
  const peak = Math.max(1, ...entries.map(([, e]) => e.wer))
  return (
    <div className={cn('min-w-0 space-y-1.5', className)}>
      {title && <PanelHeading title={title} note={note} />}
      {entries.map(([key, entry]) => (
        <BarRow
          key={key}
          label={
            <span>
              {bucketLabel(key)}{' '}
              <span className="text-[10px] text-muted-foreground">
                {entry.clips} clips{key === baseline ? ' · baseline' : ''}
              </span>
            </span>
          }
          value={entry.wer}
          peak={peak}
          fill="bg-rose-500/70"
          caption={
            <span>
              {fmtWer(entry.wer, 1)}
              {entry.cer !== undefined && <span className="text-muted-foreground"> · CER {fmtWer(entry.cer, 1)}</span>}
              <span className="text-muted-foreground"> · {percent(entry.share_of_errors)} of errors</span>
              <RatioCaption entry={entry} />
            </span>
          }
          active={active === key}
          onClick={() => onPick(active === key ? null : key)}
        />
      ))}
    </div>
  )
}

/** The buckets of one axis in display order: the axis's own, or by share of errors for voices. */
function ordered(axis: ClassAxis, buckets: Record<string, ModelBreakdown>): [string, ModelBreakdown][] {
  if (axis.buckets.length) {
    return axis.buckets.filter((b) => buckets[b]).map((b) => [b, buckets[b]])
  }
  return Object.entries(buckets).sort(
    ([a, x], [b, y]) => Number(a === axis.unmeasured) - Number(b === axis.unmeasured) || y.share_of_errors - x.share_of_errors,
  )
}

function ClassPanel({
  run,
  axes,
  axis,
  bucket,
  onPickAxis,
  onPickBucket,
}: {
  run: ModelEvalRun
  axes: ClassAxis[]
  axis: string
  bucket: string | null
  onPickAxis: (axis: string) => void
  onPickBucket: (bucket: string | null) => void
}) {
  const byClass = run.metrics.by_class
  if (!byClass) {
    return (
      <Panel className="min-w-0 flex-[2]">
        <PanelHeading title="By class" />
        <p className="text-xs text-muted-foreground">
          This run was scored before clips had classes. Run{' '}
          <code className="font-mono">scripts/reclassify_runs.py</code> to split it.
        </p>
      </Panel>
    )
  }
  const current = axes.find((a) => a.name === axis) ?? axes[0]
  return (
    <Panel className="min-w-0 flex-[2]">
      <PanelHeading
        title="By class"
        note="×: within-episode error-rate ratio against the baseline bucket, 95% interval over episodes"
      />
      <div className="grid gap-4 md:grid-cols-[13rem_1fr]">
        <div className="flex flex-col gap-0.5">
          {axes
            .filter((a) => byClass[a.name])
            .map((a) => {
              const v = verdict(a, byClass[a.name])
              return (
                <button
                  key={a.name}
                  type="button"
                  onClick={() => onPickAxis(a.name)}
                  className={cn(
                    'flex items-center justify-between gap-2 rounded px-2 py-1 text-left text-xs',
                    a.name === current?.name ? 'bg-muted font-semibold' : 'hover:bg-muted/60',
                  )}
                >
                  <span className="truncate">{a.label}</span>
                  <span
                    title={VERDICT_TITLE[v]}
                    className={cn('shrink-0 rounded-full px-1.5 py-px font-mono text-[9px]', VERDICT_STYLE[v])}
                  >
                    {v}
                  </span>
                </button>
              )
            })}
        </div>
        {current && byClass[current.name] && (
          <Breakdown
            entries={ordered(current, byClass[current.name])}
            baseline={current.baseline}
            active={bucket}
            onPick={onPickBucket}
          />
        )}
      </div>
    </Panel>
  )
}

export function RunSummary({
  run,
  axes,
  genre,
  classAxis,
  classBucket,
  onPickGenre,
  onPickAxis,
  onPickBucket,
}: {
  run: ModelEvalRun
  axes: ClassAxis[]
  genre: string | null
  classAxis: string
  classBucket: string | null
  onPickGenre: (genre: string | null) => void
  onPickAxis: (axis: string) => void
  onPickBucket: (bucket: string | null) => void
}) {
  const m = run.metrics
  const skipped = (m.skipped?.not_in_split ?? 0) + (m.skipped?.no_reference ?? 0)
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 divide-x rounded-lg border bg-card sm:grid-cols-3 lg:grid-cols-6">
        <Stat
          label="Folded WER"
          value={fmtWer(m.wer)}
          sub={m.wer_ci ? `95% CI ${m.wer_ci[0].toFixed(1)}–${m.wer_ci[1].toFixed(1)}` : 'one episode, no CI'}
          tone="text-rose-600 dark:text-rose-400"
          title="Errors pooled over reference words, script-folded (fold.py). Interval from resampling whole episodes."
        />
        <Stat label="Raw WER" value={fmtWer(m.raw_wer)} sub="no script folding" />
        <Stat label="CER" value={fmtWer(m.cer)} sub="folded tokens" />
        <Stat
          label="Loops"
          value={m.loops}
          sub="3-word run ×5"
          tone={m.loops > 0 ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat
          label="Clips"
          value={m.clips}
          sub={`${m.episodes} episodes · ${m.ref_words.toLocaleString()} words`}
        />
        <Stat
          label="Skipped"
          value={skipped}
          sub={
            skipped
              ? `${m.skipped?.not_in_split ?? 0} left ${run.split}, ${m.skipped?.no_reference ?? 0} no label`
              : 'every clip scored'
          }
          tone={skipped ? 'text-amber-600 dark:text-amber-400' : ''}
          title="Clips in the file that were not scored: moved out of the split since the notebook ran, or whose current label has no transcript."
        />
      </div>
      <div className="flex flex-col gap-3 lg:flex-row">
        <Panel className="min-w-0 flex-1">
          <Breakdown
            title="By genre"
            note="click to filter the clips"
            entries={Object.entries(m.by_genre)}
            active={genre}
            onPick={onPickGenre}
          />
        </Panel>
        <ClassPanel
          run={run}
          axes={axes}
          axis={classAxis}
          bucket={classBucket}
          onPickAxis={onPickAxis}
          onPickBucket={onPickBucket}
        />
      </div>
    </div>
  )
}
