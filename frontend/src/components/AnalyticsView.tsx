/**
 * The analytics page: what is in the corpus, what is missing, and what to record next (D69).
 *
 * The order of the page is an argument. It opens with the ledger — nine numbers that say how big
 * the corpus is and how many different people are in it — then goes straight to the shopping
 * list, because that is the only section that ends in an action. Everything below is the evidence
 * for those rows, arranged so each one can be checked: who is in it, how they speak, what the
 * pots hold, what the records are missing, and finally the raw episode inventory.
 *
 * Pipeline health sits at the bottom. It used to lead the page, and it is real, but it answers
 * how the work is going rather than whether the corpus is any good.
 */

import { useEffect, useState } from 'react'
import { RiDatabase2Line, RiErrorWarningLine, RiRefreshLine } from '@remixicon/react'
import { toast } from 'sonner'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { CoveragePanel } from '@/components/analytics/CoveragePanel'
import { EpisodesTable, MetadataPanel, ShowsTable } from '@/components/analytics/InventoryTables'
import { PipelinePanel } from '@/components/analytics/PipelinePanel'
import { PotsPanel } from '@/components/analytics/PotsPanel'
import { RegisterSection } from '@/components/analytics/RegisterPanel'
import { SourcingPanel } from '@/components/analytics/SourcingPanel'
import { Stat, hours, percent } from '@/components/analytics/primitives'
import { api } from '@/services/api'
import type { AnalyticsReport, CorpusInventory } from '@/types'

export function AnalyticsView() {
  const [inventory, setInventory] = useState<CorpusInventory | null>(null)
  const [report, setReport] = useState<AnalyticsReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      // Two endpoints, one round trip: the inventory answers what the corpus is, the report
      // answers how the work on it is going, and they are separate services for that reason.
      const [nextInventory, nextReport] = await Promise.all([api.getInventory(), api.getReport()])
      setInventory(nextInventory)
      setReport(nextReport)
    } catch (err: any) {
      setError(err.message || 'Failed to load the corpus inventory')
      toast.error('Failed to load analytics')
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
        <span>Reading the corpus…</span>
      </div>
    )
  }

  if (error && !inventory) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <Alert variant="destructive">
          <RiErrorWarningLine className="size-5" />
          <AlertTitle>Could not load the corpus inventory</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <Button onClick={load} className="mt-4 gap-2">
          <RiRefreshLine className="size-4" /> Retry
        </Button>
      </div>
    )
  }

  if (!inventory) return null

  const { totals, dimensions, register, length_profile: length } = inventory
  const genderVocabulary = totals.gender_values + (dimensions.gender?.absent.length ?? 0)
  const ageVocabulary = totals.age_values + (dimensions.age_bracket?.absent.length ?? 0)

  return (
    <div className="scrollbar-thin flex-1 space-y-4 overflow-y-auto bg-background p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex items-center gap-2">
          <RiDatabase2Line className="size-5 text-primary" />
          <h1 className="font-heading text-lg font-bold tracking-tight">Corpus</h1>
          <p className="text-xs text-muted-foreground">
            what is in it, what is missing, and what to record next
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] text-muted-foreground">
            {new Date(inventory.generated_at).toLocaleTimeString()}
          </span>
          <Button variant="outline" size="sm" onClick={load} className="h-8 gap-1.5">
            <RiRefreshLine className="size-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      {/* The ledger. Nine figures, one row: size on the left, breadth on the right, because
          breadth is the constraint the corpus is actually under. */}
      <div className="grid grid-cols-2 divide-x divide-y rounded-lg border bg-card sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-9 xl:divide-y-0">
        <Stat
          label="Audio"
          value={`${hours(totals.hours)} h`}
          sub={`${totals.segments.toLocaleString()} clips`}
        />
        <Stat
          label="Labelled"
          value={percent(totals.labeled_fraction)}
          sub={`${hours(totals.labeled_hours)} h decided`}
        />
        <Stat
          label="Heard"
          value={percent(totals.verified_fraction)}
          sub={`${hours(totals.verified_hours)} h of it played`}
          title="Share of labelled audio someone actually listened to, rather than screening on the disagreement signal"
        />
        <Stat label="Episodes" value={totals.episodes} sub={`median ${length.median_minutes ?? '--'} min`} />
        <Stat
          label="Shows"
          value={totals.shows}
          sub={`top holds ${percent(dimensions.show_id?.top_share ?? 0)}`}
          tone={(dimensions.show_id?.top_share ?? 0) > 0.5 ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat
          label="Speakers"
          value={`≥ ${totals.speaker_profiles}`}
          sub="distinct profiles"
          title="Distinct (show, role, gender, age) combinations. Two guests of one show in the same bracket collapse into one, so this is a floor."
        />
        <Stat
          label="Gender"
          value={`${totals.gender_values}/${genderVocabulary}`}
          sub="values carried"
          tone={totals.gender_values < genderVocabulary ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat
          label="Age"
          value={`${totals.age_values}/${ageVocabulary}`}
          sub="brackets carried"
          tone={totals.age_values < ageVocabulary ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat
          label="English"
          value={percent(register.mean, 1)}
          sub={
            register.show_mean_spread !== null
              ? `${percent(register.show_mean_spread)} spread across shows`
              : 'one show only'
          }
        />
      </div>

      <SourcingPanel
        recommendations={inventory.recommendations}
        minStratumHours={totals.min_stratum_hours}
      />

      <CoveragePanel
        matrix={inventory.speaker_matrix}
        dimensions={dimensions}
        corpusHours={totals.hours}
        speakerProfiles={totals.speaker_profiles}
        lengthProfile={length}
      />

      <RegisterSection data={register} minStratumHours={totals.min_stratum_hours} />

      {report ? <PotsPanel pots={report.pots} verification={report.verification} /> : null}

      <MetadataPanel rows={inventory.metadata_completeness} />

      <ShowsTable shows={inventory.shows} />

      <EpisodesTable episodes={inventory.episodes} />

      {report ? <PipelinePanel report={report} /> : null}
    </div>
  )
}
