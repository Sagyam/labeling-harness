/**
 * The Models page: every fine-tuned model, its scores on gold and val, and the clips it got
 * wrong, worst first (D83).
 *
 * Nothing here runs a model. The notebook transcribes gold and val on a GPU, the owner copies the
 * model's folder to `data/models/asr/<slug>/`, and Rescan scores it against the current labels.
 * The page then reads top to bottom as the error hunt goes: which model, how good, where the
 * errors live (click a bar to filter), and the clips themselves -- listen, read the diff, `j`/`k`
 * to the next one.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { RiCpuLine, RiErrorWarningLine, RiFolderOpenLine, RiRefreshLine } from '@remixicon/react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { ClipPanel } from '@/components/models/ClipPanel'
import { ClipTable, type ClipFilters } from '@/components/models/ClipTable'
import { RunSummary, fmtWer } from '@/components/models/RunSummary'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { AsrModel, ModelClip, ModelClipPage, ModelEvalRun } from '@/types'

const PAGE_SIZE = 50
const CARD_SHOWN = new Set(['name', 'description', 'architecture', 'created_at', 'decoder'])
const DEFAULT_FILTERS: ClipFilters = {
  sort: 'errors',
  order: 'desc',
  genre: null,
  overlap: null,
  loopsOnly: false,
  minErrors: 0,
}

function fmtDate(value: string | null): string {
  if (!value) return 'undated'
  return new Date(value).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

/** The run a model opens on: its newest gold run, else its newest run of any split. */
function defaultRun(model: AsrModel): ModelEvalRun | null {
  const gold = model.runs.filter((r) => r.split === 'gold')
  const pool = gold.length ? gold : model.runs
  return pool.length ? pool[pool.length - 1] : null
}

function ModelList({
  models,
  selected,
  onSelect,
}: {
  models: AsrModel[]
  selected: string | null
  onSelect: (slug: string) => void
}) {
  return (
    <div className="space-y-1.5">
      {models.map((model) => {
        const run = defaultRun(model)
        return (
          <button
            key={model.slug}
            type="button"
            onClick={() => onSelect(model.slug)}
            className={cn(
              'w-full rounded-md border px-3 py-2 text-left transition-colors',
              selected === model.slug ? 'border-primary/60 bg-muted' : 'hover:bg-muted/50',
            )}
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate text-sm font-semibold">{model.name}</span>
              <span className="shrink-0 font-mono text-xs font-bold text-rose-600 tabular-nums dark:text-rose-400">
                {run ? fmtWer(run.metrics.wer) : '--'}
              </span>
            </div>
            <div className="flex items-baseline justify-between gap-2 text-[10px] text-muted-foreground">
              <span className="truncate">{fmtDate(model.trained_at)}</span>
              <span className="shrink-0">{run ? `${run.split} WER` : 'no runs'}</span>
            </div>
            {model.architecture && (
              <div className="truncate text-[10px] text-muted-foreground">{model.architecture}</div>
            )}
          </button>
        )
      })}
    </div>
  )
}

function ModelHeader({ model }: { model: AsrModel }) {
  const extras = Object.entries(model.card).filter(([key]) => !CARD_SHOWN.has(key))
  const decoder = model.card.decoder
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="font-heading text-lg font-bold">{model.name}</h1>
        <span className="font-mono text-[11px] text-muted-foreground">{model.slug}</span>
        <span className="text-[11px] text-muted-foreground">trained {fmtDate(model.trained_at)}</span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
        {model.architecture && <span>{model.architecture}</span>}
        {typeof decoder === 'string' && <span>decoder: {decoder}</span>}
      </div>
      {model.description && <p className="max-w-3xl text-sm">{model.description}</p>}
      {extras.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">Model card ({extras.length} more fields)</summary>
          <dl className="mt-1 grid max-w-3xl grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
            {extras.map(([key, value]) => (
              <div key={key} className="contents">
                <dt className="font-mono text-muted-foreground">{key}</dt>
                <dd className="break-all font-mono">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </div>
  )
}

function EmptyState({ onRescan, busy }: { onRescan: () => void; busy: boolean }) {
  return (
    <div className="mx-auto max-w-xl space-y-3 p-8 text-sm">
      <div className="flex items-center gap-2 font-heading text-base font-semibold">
        <RiFolderOpenLine className="size-5" /> No models yet
      </div>
      <p className="text-muted-foreground">
        Copy a fine-tuned model's folder from Drive into{' '}
        <code className="rounded bg-muted px-1 font-mono text-xs">data/models/asr/&lt;slug&gt;/</code>{' '}
        with its <code className="rounded bg-muted px-1 font-mono text-xs">model_card.json</code> and{' '}
        <code className="rounded bg-muted px-1 font-mono text-xs">gold.jsonl</code> (and{' '}
        <code className="rounded bg-muted px-1 font-mono text-xs">val.jsonl</code> if it has one), then
        rescan. No weights are needed: the notebook already transcribed the clips.
      </p>
      <Button onClick={onRescan} disabled={busy} className="gap-2">
        {busy ? <Spinner className="size-4" /> : <RiRefreshLine className="size-4" />} Rescan models
      </Button>
    </div>
  )
}

export function ModelsView() {
  const [models, setModels] = useState<AsrModel[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [rescanning, setRescanning] = useState(false)
  const [slug, setSlug] = useState<string | null>(null)
  const [runId, setRunId] = useState<number | null>(null)
  const [filters, setFilters] = useState<ClipFilters>(DEFAULT_FILTERS)
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<ModelClipPage | null>(null)
  const [loadingClips, setLoadingClips] = useState(false)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const model = useMemo(() => models?.find((m) => m.slug === slug) ?? null, [models, slug])
  const run = useMemo(() => model?.runs.find((r) => r.id === runId) ?? null, [model, runId])

  const loadModels = useCallback(async () => {
    try {
      const next = await api.getModels()
      setModels(next)
      setError(null)
      setSlug((current) => (current && next.some((m) => m.slug === current) ? current : next[0]?.slug ?? null))
    } catch (err: any) {
      setError(err.detail || err.message || 'Failed to load models')
    }
  }, [])

  useEffect(() => {
    loadModels()
  }, [loadModels])

  // A different model opens on its default run, with the filters cleared.
  useEffect(() => {
    setRunId(model ? defaultRun(model)?.id ?? null : null)
    setFilters(DEFAULT_FILTERS)
  }, [model?.slug])

  useEffect(() => {
    setOffset(0)
    setSelectedId(null)
  }, [runId, filters])

  useEffect(() => {
    if (runId === null) {
      setPage(null)
      return
    }
    let cancelled = false
    setLoadingClips(true)
    api
      .getRunClips(runId, {
        sort: filters.sort,
        order: filters.order,
        genre: filters.genre ?? undefined,
        overlap: filters.overlap ?? undefined,
        loops_only: filters.loopsOnly || undefined,
        min_errors: filters.minErrors || undefined,
        offset,
        limit: PAGE_SIZE,
      })
      .then((next) => {
        if (cancelled) return
        setPage(next)
        setSelectedId((current) =>
          current !== null && next.rows.some((r) => r.segment_id === current)
            ? current
            : next.rows[0]?.segment_id ?? null,
        )
      })
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load clips'))
      .finally(() => !cancelled && setLoadingClips(false))
    return () => {
      cancelled = true
    }
  }, [runId, filters, offset])

  // j / k walk the list; the clip panel owns Space and r.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return
      if (!page || page.rows.length === 0 || (e.key !== 'j' && e.key !== 'k')) return
      e.preventDefault()
      const index = page.rows.findIndex((r) => r.segment_id === selectedId)
      const next = e.key === 'j' ? Math.min(page.rows.length - 1, index + 1) : Math.max(0, index - 1)
      setSelectedId(page.rows[next].segment_id)
      document.querySelector('[data-selected="true"]')?.scrollIntoView({ block: 'nearest' })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [page, selectedId])

  const rescan = async () => {
    setRescanning(true)
    try {
      const report = await api.rescanModels()
      const skipped = report.skipped ? `, ${report.skipped} clips skipped` : ''
      toast.success(
        `${report.models.length} model(s): ${report.runs_created} new run(s), ${report.runs_unchanged} unchanged${skipped}`,
      )
      await loadModels()
    } catch (err: any) {
      toast.error(err.detail || 'Rescan failed', { duration: 10000 })
    } finally {
      setRescanning(false)
    }
  }

  const updateFilters = (next: Partial<ClipFilters>) => setFilters((current) => ({ ...current, ...next }))

  if (models === null) {
    return error ? (
      <div className="flex flex-1 items-center justify-center gap-2 p-8 text-sm text-destructive">
        <RiErrorWarningLine className="size-4" /> {error}
      </div>
    ) : (
      <div className="flex flex-1 items-center justify-center p-8">
        <Spinner className="size-6 text-primary" />
      </div>
    )
  }

  if (models.length === 0) {
    return <EmptyState onRescan={rescan} busy={rescanning} />
  }

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-72 shrink-0 flex-col gap-3 overflow-y-auto border-r p-3">
        <div className="flex items-center justify-between">
          <h2 className="flex items-center gap-1.5 font-heading text-sm font-semibold">
            <RiCpuLine className="size-4" /> Models
          </h2>
          <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" onClick={rescan} disabled={rescanning}>
            {rescanning ? <Spinner className="size-3" /> : <RiRefreshLine className="size-3.5" />} Rescan
          </Button>
        </div>
        <ModelList models={models} selected={slug} onSelect={setSlug} />
      </aside>

      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4">
        {model && <ModelHeader model={model} />}

        {model && model.runs.length > 1 && (
          <div className="flex flex-wrap gap-1 rounded-lg border bg-muted/40 p-1">
            {model.runs.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => setRunId(r.id)}
                className={cn(
                  'rounded-md px-3 py-1 text-xs',
                  r.id === runId ? 'bg-background font-semibold shadow-xs' : 'text-muted-foreground hover:bg-background/50',
                )}
                title={`${r.source ?? ''} · imported ${fmtDate(r.created_at)} · fold ${r.fold_version}`}
              >
                {r.split} · {fmtWer(r.metrics.wer)} · {fmtDate(r.created_at)}
              </button>
            ))}
          </div>
        )}

        {model && !run && (
          <p className="text-sm text-muted-foreground">
            This model has a card but no transcripts yet. Add <code className="font-mono">gold.jsonl</code> to
            its folder and rescan.
          </p>
        )}

        {run && (
          <>
            <RunSummary
              run={run}
              genre={filters.genre}
              overlap={filters.overlap}
              onPickGenre={(genre) => updateFilters({ genre })}
              onPickOverlap={(overlap) => updateFilters({ overlap })}
            />
            <div className="grid gap-3 xl:grid-cols-[minmax(22rem,28rem)_1fr]">
              <div className="h-[70vh] min-h-0">
                <ClipTable
                  page={page}
                  loading={loadingClips}
                  filters={filters}
                  onChange={updateFilters}
                  selectedId={selectedId}
                  onSelect={(clip: ModelClip) => setSelectedId(clip.segment_id)}
                  onPage={setOffset}
                />
              </div>
              <div className="min-w-0 xl:sticky xl:top-0 xl:self-start">
                {selectedId !== null ? (
                  <ClipPanel runId={run.id} segmentId={selectedId} />
                ) : (
                  <p className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">
                    Pick a clip to hear it and see what the model wrote.
                  </p>
                )}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
