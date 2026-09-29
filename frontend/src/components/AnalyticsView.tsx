/**
 * The corpus page (D91, D104): what one pot is short of and what it has plenty of.
 *
 * Gold and train/val are separate views, never drawn on one axis: a benchmark wants minutes per
 * stratum and many different voices, a training set wants hours. Each is rated against its own
 * floor (`dataset.coverage`), and the page leads with the answer -- the missing and thin buckets
 * beside the plentiful ones -- then a table per category as the evidence, the cross-tab, and the
 * paperwork. People are followed on the Voices page.
 */

import { useEffect, useState } from 'react'
import {
  RiChat3Line,
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
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { CategoryTable, GapSummary } from '@/components/analytics/Coverage'
import { CrossTabPanel } from '@/components/analytics/CrossTabPanel'
import { RecordsPanel } from '@/components/analytics/RecordsPanel'
import { GROUP_NOTE, hoursOrMinutes } from '@/components/analytics/labels'
import { Stat, percent } from '@/components/analytics/primitives'
import { StatusLegend } from '@/components/analytics/status'
import { api } from '@/services/api'
import type { CategoryGroup, CorpusInventory, CorpusPot } from '@/types'

const GROUP_ICON: Record<CategoryGroup, React.ComponentType<{ className?: string }>> = {
  people: RiGroupLine,
  content: RiChat3Line,
  speech: RiVolumeUpLine,
  acoustics: RiSoundModuleLine,
}

const POT_KEY = 'corpus-pot'

/** A threshold, rounded the way a person would say it: "10 min", "1 h", "3 h". */
const floorText = (hours: number) =>
  hours < 1 ? `${Math.round(hours * 60)} min` : `${Number(hours.toFixed(1))} h`

function storedPot(): CorpusPot {
  try {
    return window.localStorage.getItem(POT_KEY) === 'gold' ? 'gold' : 'train'
  } catch {
    return 'train'
  }
}

export function AnalyticsView() {
  const [inventory, setInventory] = useState<CorpusInventory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [pot, setPotState] = useState<CorpusPot>(storedPot)
  const [cross, setCross] = useState<{ rows: string; cols: string; unit: 'hours' | 'voices' }>({
    rows: 'genre',
    cols: 'topic',
    unit: 'hours',
  })

  const setPot = (next: CorpusPot) => {
    setPotState(next)
    try {
      window.localStorage.setItem(POT_KEY, next)
    } catch {
      // a remembered tab is a convenience
    }
  }

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

  if (!inventory) return null

  const report = inventory.pots[pot]
  const t = report.totals
  const f = report.floor
  const heard = t.hours > 0 ? t.verified_hours / t.hours : 0

  return (
    <div className="scrollbar-thin flex-1 space-y-4 overflow-y-auto bg-background p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex items-center gap-3">
          <RiDatabase2Line className="size-5 text-primary" />
          <h1 className="font-heading text-lg font-bold tracking-tight">Corpus</h1>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            spacing={0}
            value={pot}
            onValueChange={(v) => v && setPot(v as CorpusPot)}
            aria-label="Pot"
          >
            {(['train', 'gold'] as const).map((key) => (
              <ToggleGroupItem key={key} value={key} className="gap-1.5 px-3">
                {inventory.pots[key].label}
                <span className="font-mono text-[10px] text-muted-foreground">
                  {hoursOrMinutes(inventory.pots[key].totals.hours)}
                </span>
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-muted-foreground">
            {new Date(inventory.generated_at).toLocaleTimeString()}
          </span>
          <Button variant="outline" size="sm" onClick={load} className="h-8 gap-1.5">
            <RiRefreshLine className="size-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 divide-x divide-y rounded-lg border bg-card sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
        <Stat
          label="Audio"
          value={hoursOrMinutes(t.hours)}
          sub={t.target_hours ? `of ${t.target_hours} h target` : undefined}
        />
        <Stat label="Clips" value={t.clips.toLocaleString()} sub={`${t.words.toLocaleString()} words`} />
        <Stat label="Episodes" value={t.episodes} sub={`${t.shows} shows`} />
        <Stat label="Voices" value={t.voices} sub="lead at least one clip" />
        <Stat
          label="Verified"
          value={percent(heard)}
          sub={`${hoursOrMinutes(t.screened_hours)} screened`}
          title="Verified: played and read. Screened: accepted on the disagreement signal without listening."
        />
        {pot === 'train' ? (
          <Stat label="Of which val" value={hoursOrMinutes(t.val_hours)} sub="redrawn freely" />
        ) : (
          <Stat label="Undecided" value={hoursOrMinutes(t.unlabeled_hours)} sub="no label yet" />
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-[11px] text-muted-foreground">
        <StatusLegend />
        <span>
          Thin under {floorText(f.thin_hours)}, or under {f.thin_voices} voices with {f.voice_words}+ words for
          people and content · plenty from {floorText(f.thin_hours * f.plenty_factor)} · overdone over{' '}
          {percent(f.dominant_share)} of a category · the tick on each bar is the floor
        </span>
      </div>

      <GapSummary pot={report} />

      {inventory.groups.map((group) => {
        const tables = report.categories.filter((c) => c.group === group.key)
        if (!tables.length) return null
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
            <div className="grid items-start gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {tables.map((c) => (
                <CategoryTable key={c.key} report={c} pot={report} />
              ))}
            </div>
          </section>
        )
      })}

      <CrossTabPanel
        inventory={inventory}
        pot={pot}
        rows={cross.rows}
        cols={cross.cols}
        unit={cross.unit}
        onChange={(next) => setCross((prev) => ({ ...prev, ...next }))}
      />

      <RecordsPanel records={inventory.records} />
    </div>
  )
}
