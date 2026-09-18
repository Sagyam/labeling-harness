/**
 * The corpus page (D91): the dataset subdivided every way it can be, the people in it followed
 * across episodes, and advice on what to record next, one category at a time.
 *
 * The page is one cross-filter. Every clip arrives with a bucket on each of sixteen categories;
 * clicking a bucket anywhere -- a card, the cross-tab, a recommendation, a voice -- filters every
 * other card to the clips in it, and the arithmetic is done here from the clip table, so it is
 * instant. The goal switch reads the page for one of the corpus's two purposes at a time: the
 * recogniser cares about conditions and unseen voices, the paper cares about people and what
 * confounds a comparison between them.
 *
 * Order: the ledger, then the advice (the only section that ends in an action), then the
 * evidence for it -- people, content, speech, acoustics -- then the cross-tab, the voices and
 * the paperwork.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  RiChat3Line,
  RiCloseLine,
  RiDatabase2Line,
  RiErrorWarningLine,
  RiGroupLine,
  RiRefreshLine,
  RiSoundModuleLine,
  RiVolumeUpLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { AdvicePanel } from '@/components/analytics/AdvicePanel'
import { CategoryCard, TIER_FILL, type Measure } from '@/components/analytics/CategoryCard'
import { CrossTabPanel } from '@/components/analytics/CrossTabPanel'
import { RecordsPanel } from '@/components/analytics/RecordsPanel'
import { VoicesPanel } from '@/components/analytics/VoicesPanel'
import { GROUP_NOTE, bucketLabel, hoursOrMinutes } from '@/components/analytics/labels'
import { aggregateAll, hasFilters, totals, type Filters } from '@/components/analytics/model'
import { Legend, POT_COLOR, Stat, hours } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { CategoryGroup, CorpusInventory, Goal } from '@/types'

const GROUP_ICON: Record<CategoryGroup, React.ComponentType<{ className?: string }>> = {
  people: RiGroupLine,
  content: RiChat3Line,
  speech: RiVolumeUpLine,
  acoustics: RiSoundModuleLine,
}

const SEGMENT =
  'rounded px-2 py-0.5 text-[11px] data-[on=true]:bg-background data-[on=true]:shadow-sm data-[on=true]:text-foreground text-muted-foreground'

function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: Array<[T, string, string?]>
  onChange: (next: T) => void
  label: string
}) {
  return (
    <div className="flex items-center gap-1 rounded-md bg-muted p-0.5" role="group" aria-label={label}>
      {options.map(([key, text, title]) => (
        <button key={key} type="button" data-on={value === key} onClick={() => onChange(key)} className={SEGMENT} title={title}>
          {text}
        </button>
      ))}
    </div>
  )
}

export function AnalyticsView() {
  const [inventory, setInventory] = useState<CorpusInventory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<Filters>({})
  const [goal, setGoal] = useState<Goal | 'both'>('both')
  const [measure, setMeasure] = useState<Measure>('labels')
  const [adviceCategory, setAdviceCategory] = useState<string | null>(null)
  const [cross, setCross] = useState<{ rows: string; cols: string; unit: 'hours' | 'voices' }>({
    rows: 'gender',
    cols: 'age_bracket',
    unit: 'hours',
  })

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      setInventory(await api.getInventory())
    } catch (err: any) {
      setError(err.message || 'Failed to load the corpus')
      toast.error('Failed to load the corpus page')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const aggregates = useMemo(() => (inventory ? aggregateAll(inventory, filters) : {}), [inventory, filters])
  const cut = useMemo(() => (inventory ? totals(inventory, filters) : null), [inventory, filters])

  const toggle = (key: string, bucket: string) =>
    setFilters((prev) => {
      const next = { ...prev }
      if (next[key] === bucket) delete next[key]
      else next[key] = bucket
      return next
    })

  if (loading && !inventory) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-sm text-muted-foreground">
        <Spinner className="size-6 text-primary" />
        <span>Cutting the corpus…</span>
      </div>
    )
  }

  if (error && !inventory) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <Alert variant="destructive">
          <RiErrorWarningLine className="size-5" />
          <AlertTitle>Could not load the corpus</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button onClick={load} className="mt-4 gap-2">
          <RiRefreshLine className="size-4" /> Retry
        </Button>
      </div>
    )
  }

  if (!inventory || !cut) return null

  const { totals: t, voice_summary: voices } = inventory
  const labels = Object.fromEntries(inventory.categories.map((c) => [c.key, c.label]))
  // The voice category is filterable and cross-tabulated, but its card is the Voices table below.
  const visible = inventory.categories.filter(
    (c) => c.key !== 'voice' && (goal === 'both' || c.goals.includes(goal))
  )
  const filtered = hasFilters(filters)

  return (
    <div className="scrollbar-thin flex-1 space-y-4 overflow-y-auto bg-background p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex items-center gap-2">
          <RiDatabase2Line className="size-5 text-primary" />
          <h1 className="font-heading text-lg font-bold tracking-tight">Corpus</h1>
          <p className="text-xs text-muted-foreground">every clip, cut by who, what, how and where; every voice followed</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Segmented
            label="Goal"
            value={goal}
            onChange={setGoal}
            options={[
              ['both', 'Both goals'],
              ['asr', 'ASR', 'Conditions a recogniser trips on, and the voices a benchmark should hold out'],
              ['paper', 'Paper', 'People, and what confounds a comparison between them'],
            ]}
          />
          <Segmented
            label="Measure"
            value={measure}
            onChange={setMeasure}
            options={[
              ['labels', 'Labels', 'Verified, screened and undecided hours'],
              ['pots', 'Pots', 'Gold, val and train hours'],
              ['voices', 'Voices', 'Voices per bucket, usable ones solid'],
            ]}
          />
          <span className="font-mono text-[11px] text-muted-foreground">{new Date(inventory.generated_at).toLocaleTimeString()}</span>
          <Button variant="outline" size="sm" onClick={load} className="h-8 gap-1.5">
            <RiRefreshLine className="size-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      {/* The ledger: size on the left, people on the right, because people are the constraint. */}
      <div className="grid grid-cols-2 divide-x divide-y rounded-lg border bg-card sm:grid-cols-4 xl:grid-cols-8 xl:divide-y-0">
        <Stat label="Audio" value={`${hours(t.hours)} h`} sub={`${t.clips.toLocaleString()} clips · ${t.episodes} episodes`} />
        <Stat label="Speech" value={`${hours(t.speech_hours)} h`} sub={`${t.words.toLocaleString()} reference words`} title="VAD speech inside the clips" />
        <Stat
          label="Heard"
          value={`${hours(t.verified_hours)} h`}
          sub={`${hours(t.screened_hours)} h screened`}
          title="Verified: played and read. Screened: accepted on the disagreement signal without listening (D63)."
        />
        <Stat
          label="Gold"
          value={`${hours(t.gold_hours)} h`}
          sub={`of ${t.gold_target_hours} h · val ${hours(t.val_hours)} h`}
          tone={t.gold_hours < t.gold_target_hours ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat label="Shows" value={t.shows} sub="series recorded from" />
        <Stat label="Voices" value={voices.voices} sub={`${voices.recurring} recur across episodes`} />
        <Stat
          label="Usable"
          value={voices.usable}
          sub="voices with 300+ words"
          title="The paper's n: a voice needs enough words attributed to one person before anything can be said about them"
        />
        <Stat
          label="Known"
          value={`${voices.gender_resolved}/${voices.age_resolved}`}
          sub="gender / age resolved"
          tone={voices.gender_resolved < voices.usable ? 'text-amber-600 dark:text-amber-400' : ''}
          title="Voices whose declared row the episode forces; the rest are declared but unmatched, or in episodes with no rows"
        />
      </div>

      {/* The active cut. Every card below is summed over these clips. */}
      <div
        className={cn(
          'flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-xs',
          filtered ? 'border-indigo-500/40 bg-indigo-500/5' : 'border-dashed text-muted-foreground'
        )}
      >
        {filtered ? (
          <>
            <span className="font-semibold">Filtered to</span>
            {Object.entries(filters).map(([key, bucket]) => (
              <button
                key={key}
                type="button"
                onClick={() => toggle(key, bucket)}
                className="flex items-center gap-1 rounded-full bg-indigo-500/15 px-2 py-0.5 text-[11px] text-indigo-700 hover:bg-indigo-500/25 dark:text-indigo-300"
                title="Remove this filter"
              >
                {labels[key] ?? key}: {bucketLabel(bucket)}
                <RiCloseLine className="size-3" />
              </button>
            ))}
            <span className="ml-auto font-mono text-[11px] tabular-nums">
              {hoursOrMinutes(cut.hours)} · {cut.clips.toLocaleString()} clips · {cut.episodes} episodes · {cut.usableVoices} of {cut.voices} voices usable ·
              verified {hoursOrMinutes(cut.verifiedHours)} · gold {hoursOrMinutes(cut.goldHours)}
            </span>
            <button type="button" onClick={() => setFilters({})} className="text-[11px] underline hover:text-foreground">
              clear
            </button>
          </>
        ) : (
          <>
            <span>Click any bucket, cell or voice to cut every other card to it. Filters combine across categories.</span>
            <span className="ml-auto">
              <Legend
                items={
                  measure === 'pots'
                    ? [
                        { fill: POT_COLOR.gold, label: 'gold' },
                        { fill: POT_COLOR.val, label: 'val' },
                        { fill: POT_COLOR.train, label: 'train' },
                      ]
                    : measure === 'voices'
                      ? [
                          { fill: 'bg-indigo-500', label: 'usable voices' },
                          { fill: 'bg-indigo-500/35', label: 'under 300 words' },
                        ]
                      : [
                          { fill: TIER_FILL.verified, label: 'verified' },
                          { fill: TIER_FILL.screened, label: 'screened' },
                          { fill: TIER_FILL.unlabeled, label: 'undecided' },
                        ]
                }
              />
            </span>
          </>
        )}
      </div>

      <AdvicePanel
        recommendations={inventory.recommendations}
        categories={inventory.categories}
        goal={goal}
        category={adviceCategory}
        onPickCategory={setAdviceCategory}
        onPickBucket={(category, bucket) => setFilters((prev) => ({ ...prev, [category]: bucket }))}
      />

      {inventory.groups.map((group) => {
        const cards = visible.filter((c) => c.group === group.key)
        if (!cards.length) return null
        const Icon = GROUP_ICON[group.key]
        return (
          <section key={group.key} className="space-y-2">
            <div className="flex flex-wrap items-baseline gap-2 px-1">
              <h2 className="flex items-center gap-1.5 font-heading text-sm font-semibold">
                <Icon className="size-4 text-primary" />
                {group.label}
              </h2>
              <p className="text-[11px] text-muted-foreground">{GROUP_NOTE[group.key]}</p>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {cards.map((report) => (
                <CategoryCard
                  key={report.key}
                  report={report}
                  entries={aggregates[report.key] ?? []}
                  measure={measure}
                  active={filters[report.key] ?? null}
                  onToggle={(bucket) => toggle(report.key, bucket)}
                  minHours={t.min_stratum_hours}
                  minVoices={t.min_stratum_voices}
                  filtered={Object.keys(filters).some((k) => k !== report.key)}
                />
              ))}
            </div>
          </section>
        )
      })}

      <CrossTabPanel
        inventory={inventory}
        filters={filters}
        rows={cross.rows}
        cols={cross.cols}
        unit={cross.unit}
        onChange={(next) => setCross((prev) => ({ ...prev, ...next }))}
        onPick={(rowsKey, rowBucket, colsKey, colBucket) =>
          setFilters((prev) => ({ ...prev, [rowsKey]: rowBucket, [colsKey]: colBucket }))
        }
      />

      <VoicesPanel
        inventory={inventory}
        filters={filters}
        activeVoice={filters.voice ?? null}
        onPickVoice={(voice) =>
          setFilters((prev) => {
            const next = { ...prev }
            if (voice === null) delete next.voice
            else next.voice = voice
            return next
          })
        }
      />

      <RecordsPanel records={inventory.records} />
    </div>
  )
}
