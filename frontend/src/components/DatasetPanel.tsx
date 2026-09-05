import {
  RiCheckboxCircleFill,
  RiEyeOffLine,
  RiFlashlightLine,
  RiLockLine,
  RiMedalLine,
  RiSoundModuleLine,
  RiTrophyLine,
} from '@remixicon/react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { BucketName, Milestone, PotPanel, ProgressPanel, VerificationPanel } from '@/types'

/** Colour per bucket, used by both the pot bars and the coverage chips so they read as one system. */
const BUCKET_STYLE: Record<BucketName, { bar: string; text: string; dot: string; label: string }> = {
  gold: {
    bar: 'bg-amber-500',
    text: 'text-amber-600 dark:text-amber-400',
    dot: 'bg-amber-500',
    label: 'Gold — benchmark',
  },
  train: {
    bar: 'bg-emerald-500',
    text: 'text-emerald-600 dark:text-emerald-400',
    dot: 'bg-emerald-500',
    label: 'Train',
  },
  val: {
    bar: 'bg-sky-500',
    text: 'text-sky-600 dark:text-sky-400',
    dot: 'bg-sky-500',
    label: 'Val',
  },
  unassigned: {
    bar: 'bg-muted-foreground/40',
    text: 'text-muted-foreground',
    dot: 'bg-muted-foreground/40',
    label: 'Unplaced',
  },
}

const COVERAGE_LABEL: Record<string, string> = {
  show_id: 'Shows',
  gender: 'Gender',
  age_bracket: 'Age',
  topic: 'Topic',
}

const GROUP_LABEL: Record<string, string> = {
  volume: 'Volume',
  quality: 'Quality',
  habit: 'Habit',
  corpus: 'Corpus',
}

function hours(value: number) {
  return value >= 10 ? value.toFixed(1) : value.toFixed(2)
}

/** A ring that fills clockwise. Pure SVG so it needs no chart library. */
function Ring({
  fraction,
  label,
  sub,
  className = 'text-primary',
}: {
  fraction: number
  label: string
  sub: string
  className?: string
}) {
  const radius = 34
  const circumference = 2 * Math.PI * radius
  const filled = Math.max(0, Math.min(1, fraction)) * circumference
  return (
    <div className="relative flex size-24 shrink-0 items-center justify-center">
      <svg viewBox="0 0 80 80" className="size-24 -rotate-90">
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="none"
          strokeWidth="6"
          className="stroke-muted"
        />
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference - filled}`}
          className={`${className} stroke-current transition-all duration-500`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-xl font-bold leading-none">{label}</span>
        <span className="mt-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
          {sub}
        </span>
      </div>
    </div>
  )
}

/** One pot's progress toward its duration target, with labelled audio shaded inside it. */
function PotBar({
  bucket,
  entry,
  target,
  label,
}: {
  bucket: BucketName
  entry: { episodes: number; hours: number; labeled_hours: number }
  target?: number
  label?: string
}) {
  const style = BUCKET_STYLE[bucket]
  const denominator = target ?? Math.max(entry.hours, 0.001)
  const ingestedPct = Math.min(100, (entry.hours / denominator) * 100)
  const labeledPct = Math.min(100, (entry.labeled_hours / denominator) * 100)

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className={`size-2 rounded-full ${style.dot}`} />
          <span className="text-xs font-medium">{label ?? style.label}</span>
          <span className="text-[10px] text-muted-foreground">{entry.episodes} ep</span>
        </div>
        <div className="font-mono text-xs">
          <span className={`font-bold ${style.text}`}>{hours(entry.hours)}h</span>
          {target ? (
            <span className="text-muted-foreground"> / {hours(target)}h</span>
          ) : null}
        </div>
      </div>
      <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-muted">
        <div className={`absolute inset-y-0 left-0 ${style.bar} opacity-30`} style={{ width: `${ingestedPct}%` }} />
        <div className={`absolute inset-y-0 left-0 ${style.bar}`} style={{ width: `${labeledPct}%` }} />
      </div>
      <div className="text-[10px] text-muted-foreground">
        {hours(entry.labeled_hours)}h labelled of {hours(entry.hours)}h ingested
      </div>
    </div>
  )
}

function MilestoneCard({ milestone }: { milestone: Milestone }) {
  const pct = milestone.target ? Math.min(100, (milestone.progress / milestone.target) * 100) : 0
  return (
    <div
      className={`rounded-md border p-3 transition-colors ${
        milestone.unlocked ? 'border-amber-500/40 bg-amber-500/5' : 'bg-card/40'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5">
          {milestone.unlocked ? (
            <RiTrophyLine className="size-3.5 shrink-0 text-amber-500" />
          ) : (
            <RiLockLine className="size-3.5 shrink-0 text-muted-foreground" />
          )}
          <span
            className={`text-xs font-semibold ${
              milestone.unlocked ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground'
            }`}
          >
            {milestone.name}
          </span>
        </div>
        <span className="font-mono text-[10px] text-muted-foreground">
          {milestone.progress >= 1 || milestone.progress === 0
            ? Math.round(milestone.progress)
            : milestone.progress.toFixed(1)}
          /{milestone.target >= 1 ? Math.round(milestone.target) : milestone.target}
        </span>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-muted-foreground">{milestone.description}</p>
      <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={milestone.unlocked ? 'h-full bg-amber-500' : 'h-full bg-primary/50'}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

/** Twelve weeks of daily activity, one square per day, densest week last. */
function ActivityStrip({ activity }: { activity: Array<{ day: string; segments: number }> }) {
  if (activity.length === 0) return null
  const byDay = new Map(activity.map((row) => [row.day, row.segments]))
  const peak = Math.max(...activity.map((row) => row.segments), 1)

  const days: Array<{ day: string; segments: number }> = []
  const cursor = new Date()
  for (let offset = 83; offset >= 0; offset -= 1) {
    const date = new Date(cursor)
    date.setDate(cursor.getDate() - offset)
    const key = date.toISOString().slice(0, 10)
    days.push({ day: key, segments: byDay.get(key) ?? 0 })
  }

  return (
    <div className="flex flex-wrap gap-[3px]">
      {days.map(({ day, segments }) => {
        const intensity = segments === 0 ? 0 : Math.ceil((segments / peak) * 4)
        const tone = [
          'bg-muted',
          'bg-emerald-500/25',
          'bg-emerald-500/50',
          'bg-emerald-500/75',
          'bg-emerald-500',
        ][intensity]
        return (
          <div
            key={day}
            title={`${day}: ${segments} segment${segments === 1 ? '' : 's'}`}
            className={`size-3 rounded-[2px] ${tone}`}
          />
        )
      })}
    </div>
  )
}

export function DatasetPanel({
  pots,
  verification,
  progress,
}: {
  pots: PotPanel
  verification: VerificationPanel
  progress: ProgressPanel
}) {
  // The train pot is train + val: they hold the same standard, and the target is set over the
  // pot rather than over either half, because the train/val line is redrawable.
  const trainPot = {
    episodes: pots.buckets.train.episodes + pots.buckets.val.episodes,
    hours: pots.buckets.train.hours + pots.buckets.val.hours,
    labeled_hours: pots.buckets.train.labeled_hours + pots.buckets.val.labeled_hours,
  }
  const verifiedPct = verification.total
    ? Math.round((verification.verified / verification.total) * 100)
    : 0

  const groups: Array<Milestone['group']> = ['volume', 'quality', 'habit', 'corpus']

  return (
    <>
      {/* Progress hero: level, streak, daily goal, gold pot */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="bg-card/50">
          <CardContent className="flex items-center gap-4 pt-6">
            <Ring
              fraction={progress.level_fraction}
              label={`${progress.level}`}
              sub="Level"
              className="text-primary"
            />
            <div className="min-w-0">
              <div className="font-mono text-lg font-bold">
                {progress.level_minutes.toFixed(0)}
                <span className="ml-1 text-xs font-normal text-muted-foreground">min cleared</span>
              </div>
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                {(progress.minutes_per_level - progress.minutes_into_level).toFixed(0)} min to
                level {progress.level + 1}. Measured in audio, not clicks.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50">
          <CardContent className="flex items-center gap-4 pt-6">
            <Ring
              fraction={progress.daily_goal_fraction}
              label={`${progress.today_segments}`}
              sub="Today"
              className="text-emerald-500"
            />
            <div className="min-w-0">
              <div className="font-mono text-lg font-bold">
                {Math.round(progress.daily_goal_fraction * 100)}%
                <span className="ml-1 text-xs font-normal text-muted-foreground">of goal</span>
              </div>
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                Daily goal {progress.daily_goal_segments}. Best day{' '}
                {progress.best_day_segments || '--'}
                {progress.best_day ? ` on ${progress.best_day}` : ''}.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Streak
            </CardTitle>
            <RiFlashlightLine className="size-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="font-mono text-2xl font-bold text-orange-600 dark:text-orange-400">
              {progress.current_streak_days}
              <span className="ml-1 text-xs font-normal text-muted-foreground">
                day{progress.current_streak_days === 1 ? '' : 's'}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Longest {progress.longest_streak_days} &middot; {progress.active_days} active days
              {progress.current_streak_days > 0 && !progress.streak_active_today
                ? ' · nothing logged today yet'
                : ''}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-card/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Milestones
            </CardTitle>
            <RiMedalLine className="size-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="font-mono text-2xl font-bold">
              {progress.unlocked_count}
              <span className="ml-1 text-xs font-normal text-muted-foreground">
                / {progress.achievements.length}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {progress.achievements.length - progress.unlocked_count} still locked.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* The two pots */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <RiSoundModuleLine className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">The Two Pots</CardTitle>
              {pots.buckets.gold.hours >= pots.gold_target_hours ? (
                <Badge variant="default" className="text-amber-600 dark:text-amber-400">
                  Gold target met
                </Badge>
              ) : pots.gold_capped_by_corpus_size ? (
                <Badge variant="secondary">
                  Gold held at {hours(pots.gold_effective_target_hours)}h — corpus too small
                </Badge>
              ) : (
                <Badge variant="secondary">
                  {hours(Math.max(0, pots.gold_target_hours - pots.buckets.gold.hours))}h of gold
                  to go
                </Badge>
              )}
            </div>
            <CardDescription className="text-xs">
              Gold is the benchmark: every clip listened to, and an episode never leaves it. Train
              and val hold the same standard as each other and may be redrawn. Solid bar is
              labelled; faded is ingested but not yet decided.
              {pots.gold_capped_by_corpus_size ? (
                <>
                  {' '}
                  Gold is capped at a share of the corpus so a small one is not swallowed whole —
                  it grows toward its {hours(pots.gold_target_hours)}h target as you ingest more.
                </>
              ) : null}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <PotBar bucket="gold" entry={pots.buckets.gold} target={pots.gold_target_hours} />
            <PotBar
              bucket="train"
              entry={trainPot}
              target={pots.train_target_hours}
              label="Train pot"
            />
            <div className="grid grid-cols-2 gap-4 border-l-2 pl-4">
              <PotBar bucket="train" entry={pots.buckets.train} />
              <PotBar bucket="val" entry={pots.buckets.val} />
            </div>
            {pots.buckets.unassigned.episodes > 0 ? (
              <PotBar bucket="unassigned" entry={pots.buckets.unassigned} />
            ) : null}
          </CardContent>
        </Card>

        {/* Verification mix */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <RiEyeOffLine className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">How It Was Checked</CardTitle>
            </div>
            <CardDescription className="text-xs">
              The claim the corpus makes about itself. Verified means you played the clip; screened
              means it was accepted on model disagreement alone.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
              <div
                style={{ width: `${verifiedPct}%` }}
                className="bg-violet-500"
                title={`Verified: ${verifiedPct}%`}
              />
              <div
                style={{ width: `${100 - verifiedPct}%` }}
                className="bg-slate-400 dark:bg-slate-600"
                title={`Screened: ${100 - verifiedPct}%`}
              />
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="space-y-1 rounded-md border p-2.5">
                <div className="flex items-center gap-1.5 font-medium text-violet-600 dark:text-violet-400">
                  <span className="size-2 rounded-full bg-violet-500" />
                  Verified
                </div>
                <div className="font-mono text-lg font-bold">
                  {verification.verified.toLocaleString()}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {hours(verification.hours.verified)}h heard
                </div>
              </div>
              <div className="space-y-1 rounded-md border p-2.5">
                <div className="flex items-center gap-1.5 font-medium text-muted-foreground">
                  <span className="size-2 rounded-full bg-slate-400 dark:bg-slate-600" />
                  Screened
                </div>
                <div className="font-mono text-lg font-bold">
                  {verification.screened.toLocaleString()}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {hours(verification.hours.screened)}h waved through
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* What the benchmark spans */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <RiCheckboxCircleFill
              className={`size-4 ${pots.coverage_complete ? 'text-emerald-500' : 'text-amber-500'}`}
            />
            <CardTitle className="text-sm font-semibold">What the Benchmark Spans</CardTitle>
            {pots.coverage_complete ? (
              <Badge variant="default" className="text-emerald-600 dark:text-emerald-400">
                Full coverage
              </Badge>
            ) : (
              <Badge variant="destructive">Gaps</Badge>
            )}
          </div>
          <CardDescription className="text-xs">
            A five-hour benchmark drawn from one show measures that show. Filled chips are covered
            by the gold pot; outlined ones exist in the corpus but not in gold.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(pots.corpus_coverage).map(([key, values]) => {
              const covered = pots.gold_coverage[key] ?? {}
              const names = Object.keys(values).sort()
              return (
                <div key={key} className="space-y-2 rounded-lg border p-3">
                  <div className="flex items-baseline justify-between">
                    <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {COVERAGE_LABEL[key] ?? key}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {Object.keys(covered).length}/{names.length}
                    </span>
                  </div>
                  {names.length === 0 ? (
                    <p className="text-[11px] text-muted-foreground">Not recorded yet.</p>
                  ) : (
                    <div className="flex flex-wrap gap-1">
                      {names.map((name) => {
                        const inGold = name in covered
                        return (
                          <span
                            key={name}
                            title={
                              inGold
                                ? `${covered[name]} gold episode(s)`
                                : 'in the corpus but not in the gold pot'
                            }
                            className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${
                              inGold
                                ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
                                : 'border border-dashed border-muted-foreground/40 text-muted-foreground'
                            }`}
                          >
                            {name}
                          </span>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Milestones */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <RiTrophyLine className="size-4 text-amber-500" />
            <CardTitle className="text-sm font-semibold">Milestones</CardTitle>
            <Badge variant="secondary" className="font-mono">
              {progress.unlocked_count}/{progress.achievements.length}
            </Badge>
          </div>
          <CardDescription className="text-xs">
            Nothing here changes the corpus. It is a scoreboard for a job that otherwise only ever
            counts down.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {groups.map((group) => {
            const items = progress.achievements.filter((a) => a.group === group)
            if (items.length === 0) return null
            return (
              <div key={group} className="space-y-2">
                <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {GROUP_LABEL[group] ?? group}
                </span>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  {items.map((milestone) => (
                    <MilestoneCard key={milestone.id} milestone={milestone} />
                  ))}
                </div>
              </div>
            )
          })}
        </CardContent>
      </Card>

      {/* Activity */}
      {progress.activity.length > 0 ? (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <RiFlashlightLine className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">Last Twelve Weeks</CardTitle>
            </div>
            <CardDescription className="text-xs">
              One square per day, darker is more decided. Hover for the count.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ActivityStrip activity={progress.activity} />
          </CardContent>
        </Card>
      ) : null}
    </>
  )
}
