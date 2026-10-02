/**
 * The folding rulebook (fold-v4): every rule `fold.py` forgives by and every tag it only
 * describes, with the evidence for each from a model's error files. Read-only on purpose: a rule
 * changes by fold version, commit and decision entry, because every WER moves when it does.
 */

import { useEffect, useMemo, useState } from 'react'
import { RiAlertLine, RiSearchLine } from '@remixicon/react'
import { toast } from 'sonner'

import { Panel, PanelHeading } from '@/components/analytics/primitives'
import { OccurrenceRow, setLabel } from '@/components/models/ErrorsSection'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { Occurrence, Rulebook, RulebookEvidence, RulebookRule } from '@/types'

/** error_mining.SETS, in its order. */
const SETS = ['gold', 'val', 'fleurs', 'slr54', 'common_voice', 'indicvoices', 'nepali_cs']
const SAMPLE = 20

function fmtRate(value: number | undefined): string {
  if (value === undefined || value === 0) return '—'
  return value < 0.01 ? '<0.01' : value.toFixed(2)
}

function Pair({ a, b, same }: { a: string; b: string; same: boolean }) {
  return (
    <span
      className={cn(
        'inline-flex items-baseline gap-1 rounded border px-1.5 py-0.5 text-[13px]',
        same
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-rose-500/30 bg-rose-500/5',
      )}
    >
      <span>{a}</span>
      <span className="text-[10px] text-muted-foreground">{same ? '=' : '≠'}</span>
      <span>{b}</span>
    </span>
  )
}

function Occurrences({ model, set, rule }: { model: string; set: string; rule: RulebookRule }) {
  const [rows, setRows] = useState<Occurrence[] | null>(null)
  const [total, setTotal] = useState(0)
  useEffect(() => {
    let cancelled = false
    setRows(null)
    const filter = rule.action === 'tag' ? { variant: rule.id } : { fold_rule: rule.id }
    api
      .getErrorPairs(model, set, { ...filter, sample: SAMPLE, seed: 0 })
      .then((page) => {
        if (cancelled) return
        setRows(page.rows)
        setTotal(page.total)
      })
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load where the rule fired'))
    return () => {
      cancelled = true
    }
  }, [model, set, rule])
  if (rows === null) return <Spinner className="size-4" />
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">Not once on {setLabel(set)}.</p>
  return (
    <div className="space-y-1.5">
      <p className="text-[11px] text-muted-foreground">
        {rows.length} of {total} on {setLabel(set)}, drawn at random (seed 0), each in its clip.
      </p>
      {rows.map((occ) => (
        <OccurrenceRow key={`${occ.clip_id}-${occ.pos}`} occ={occ} active={false} onOpen={() => {}} />
      ))}
    </div>
  )
}

function RuleCard({
  rule,
  evidence,
  set,
  onPickSet,
  model,
}: {
  rule: RulebookRule
  evidence: RulebookEvidence[]
  set: string
  onPickSet: (set: string) => void
  model: string | null
}) {
  const [open, setOpen] = useState(false)
  const bySet = new Map(evidence.map((e) => [e.set, e.rules[rule.id]]))
  const here = bySet.get(set)
  const isTag = rule.action === 'tag'
  return (
    <article className="border-t py-3 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <h3 className="text-sm font-semibold">{rule.title}</h3>
        <code className="text-[11px] text-muted-foreground">{rule.id}</code>
        <span
          className={cn(
            'rounded-full px-1.5 text-[10px] font-medium',
            isTag
              ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
              : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
          )}
          title={isTag ? 'Describes a charged error; never forgives it' : 'Forgives the pair: one word written two ways'}
        >
          {isTag ? 'tag · still an error' : 'folds'}
        </span>
      </div>
      <p className="mt-1 max-w-3xl text-[13px] text-muted-foreground">{rule.description}</p>
      {(rule.examples.length > 0 || rule.counterexamples.length > 0) && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {rule.examples.map(([a, b]) => (
            <Pair key={`e-${a}-${b}`} a={a} b={b} same={!isTag} />
          ))}
          {rule.counterexamples.map(([a, b]) => (
            <Pair key={`c-${a}-${b}`} a={a} b={b} same={false} />
          ))}
        </div>
      )}
      {evidence.length > 0 && (
        <div className="mt-2 flex flex-wrap items-baseline gap-1 text-[11px]">
          <span className="mr-1 text-muted-foreground" title="Pairs this rule forgave, or this tag tagged, per 100 reference words">
            per 100 words
          </span>
          {SETS.filter((s) => bySet.has(s)).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onPickSet(s)}
              className={cn(
                'rounded border px-1.5 py-0.5 tabular-nums hover:bg-muted',
                s === set && 'border-primary/60 bg-muted',
              )}
              title={`${bySet.get(s)?.pairs ?? 0} pairs on ${setLabel(s)}`}
            >
              {setLabel(s)} <span className="font-mono">{fmtRate(bySet.get(s)?.per_100)}</span>
            </button>
          ))}
        </div>
      )}
      {here && here.top.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[13px]">
          <span className="text-[11px] text-muted-foreground">most often on {setLabel(set)}:</span>
          {here.top.map((t) => (
            <span key={`${t.ref}|${t.hyp}`}>
              {t.ref} <span className="text-muted-foreground">→</span> {t.hyp}{' '}
              <span className="font-mono text-[11px] text-muted-foreground">×{t.count}</span>
            </span>
          ))}
          {model && (
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              className="text-[11px] text-primary underline-offset-2 hover:underline"
            >
              {open ? 'Hide where it fired' : 'Where it fired'}
            </button>
          )}
        </div>
      )}
      {open && model && (
        <div className="mt-2">
          <Occurrences model={model} set={set} rule={rule} />
        </div>
      )}
    </article>
  )
}

export function RulebookView() {
  const [book, setBook] = useState<Rulebook | null>(null)
  const [model, setModel] = useState<string | null>(null)
  const [set, setSet] = useState('gold')
  const [search, setSearch] = useState('')

  useEffect(() => {
    let cancelled = false
    api
      .getRulebook()
      .then((next) => {
        if (cancelled) return
        setBook(next)
        const current = next.evidence.find((e) => e.current) ?? next.evidence[0]
        if (current) setModel(current.model)
      })
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load the rulebook'))
    return () => {
      cancelled = true
    }
  }, [])

  const models = useMemo(() => {
    const seen = new Map<string, string>()
    for (const e of book?.evidence ?? []) seen.set(e.model, e.model_name)
    return [...seen.entries()]
  }, [book])
  const evidence = useMemo(() => (book?.evidence ?? []).filter((e) => e.model === model), [book, model])
  const stale = evidence.filter((e) => !e.current)

  useEffect(() => {
    if (evidence.length > 0 && !evidence.some((e) => e.set === set)) setSet(evidence[0].set)
  }, [evidence, set])

  if (!book) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner className="size-5" />
      </div>
    )
  }

  const needle = search.trim().toLowerCase()
  const matches = (r: RulebookRule) =>
    !needle ||
    [r.id, r.title, r.description, ...r.examples.flat(), ...r.counterexamples.flat()].some((t) =>
      t.toLowerCase().includes(needle),
    )
  const folds = book.rules.filter((r) => r.action === 'fold').length

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-4 p-4 sm:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="font-heading text-lg font-semibold">Folding rulebook</h1>
            <p className="text-xs text-muted-foreground">
              <code>{book.fold_version}</code> · {folds} rules that forgive a pair ·{' '}
              {book.rules.length - folds} tags that only describe an error. Rules change by fold version and
              commit, never here.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs">
              <RiSearchLine className="size-3.5 text-muted-foreground" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Find a rule or a word"
                className="w-40 bg-transparent outline-none"
              />
            </label>
            {models.length > 0 && (
              <select
                value={model ?? ''}
                onChange={(e) => setModel(e.target.value)}
                className="rounded-md border bg-background px-2 py-1 text-xs"
                aria-label="Evidence from model"
              >
                {models.map(([slug, name]) => (
                  <option key={slug} value={slug}>
                    {name}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {models.length === 0 && (
          <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
            No model has error files yet, so the rules show without evidence. Derive them with{' '}
            <code>scripts/mine_errors.py</code> or upload them on the Models page.
          </p>
        )}
        {stale.length > 0 && (
          <p className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
            <RiAlertLine className="mt-0.5 size-4 shrink-0 text-amber-600" />
            <span>
              {stale.map((e) => setLabel(e.set)).join(', ')} {stale.length === 1 ? 'was' : 'were'} derived under{' '}
              <code>{stale[0].fold_version}</code>, not <code>{book.fold_version}</code>: those counts are what an
              older rulebook did. Derive the files again to see this one.
            </span>
          </p>
        )}

        {book.tiers.map((tier) => {
          const rules = book.rules.filter((r) => r.tier === tier.tier && matches(r))
          if (rules.length === 0) return null
          return (
            <Panel key={tier.tier}>
              <PanelHeading title={`Tier ${tier.tier}`} note={tier.title} />
              {rules.map((rule) => (
                <RuleCard
                  key={rule.id}
                  rule={rule}
                  evidence={evidence}
                  set={set}
                  onPickSet={setSet}
                  model={model}
                />
              ))}
            </Panel>
          )
        })}
      </div>
    </main>
  )
}
