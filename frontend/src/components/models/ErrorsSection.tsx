/**
 * Errors: every aligned word pair of a model's evaluations, classified (docs/WER-Breakdown.md).
 *
 * The clip table above answers "what went wrong here". This section answers it across clips, from
 * the model's error files (`errors/<set>.parquet`, written by the notebook or by
 * `scripts/mine_errors.py`): one set at a time, its WER beside where the errors sit by crosstalk
 * and numbers, then the confusion table -- which word became which, how often -- and the
 * occurrences behind any row, with context. Reading is done by sampling inside a class: Sample 50
 * draws at random with a visible seed, so a second look sees the same 50.
 *
 * A breakdown tags errors, it never forgives them: nothing here changes a WER. Every text on the
 * panel comes from a model, a label or a public dataset, and React escapes all of it.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { RiAlertLine, RiCloseLine, RiDiceLine, RiUploadLine } from '@remixicon/react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { Panel, PanelHeading, humanize, percent } from '@/components/analytics/primitives'
import { AlignedDiff, DiffLegend } from '@/components/models/AlignedDiff'
import { ClipPanel } from '@/components/models/ClipPanel'
import { bucketLabel, fmtWer } from '@/components/models/RunSummary'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type {
  AsrModel,
  ConfusionPage,
  ConfusionRow,
  ErrorBreakdown,
  ErrorClip,
  ErrorFiles,
  ErrorKind,
  ErrorQuery,
  Forgiven,
  Occurrence,
  OccurrencePage,
  OverlapBucket,
  ScriptName,
  VsBase,
} from '@/types'

const SELECT =
  'h-7 rounded-md border border-input bg-background/50 px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring'
const ERRORS: ErrorKind[] = ['sub', 'del', 'ins']
const KINDS: [ErrorKind, string][] = [
  ['sub', 'substituted'],
  ['del', 'deleted'],
  ['ins', 'inserted'],
  ['fold', 'folded'],
  ['merge', 'merged'],
  ['match', 'matched'],
]
const FORGIVEN: Forgiven[] = ['spelling', 'script', 'number', 'merge']
const SCRIPTS: [ScriptName, string][] = [
  ['dev', 'Devanagari'],
  ['lat', 'Latin'],
  ['mix', 'both in one word'],
  ['none', 'digits only'],
]
const SET_LABEL: Record<string, string> = {
  gold: 'gold',
  val: 'val',
  fleurs: 'FLEURS',
  slr54: 'OpenSLR 54',
  common_voice: 'Common Voice',
  indicvoices: 'IndicVoices',
  nepali_cs: 'nepali-cs',
}
const CONFUSION_PAGE = 25
const SAMPLE = 50

function setLabel(set: string): string {
  return SET_LABEL[set] ?? set
}

function fmtPts(value: number | null | undefined): string {
  if (value === null || value === undefined) return '--'
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}`
}

/** `+0.42 [−0.10, +0.95]`, coloured when the interval clears zero. */
function Diff({ vs }: { vs: VsBase | null | undefined }) {
  if (!vs) return <span className="text-muted-foreground">--</span>
  const [d, lo, hi] = vs.wer
  const tone =
    lo !== null && lo > 0
      ? 'text-rose-600 dark:text-rose-400'
      : hi !== null && hi < 0
        ? 'text-emerald-600 dark:text-emerald-400'
        : 'text-muted-foreground'
  return (
    <span className={cn('font-mono tabular-nums', tone)} title={`on ${vs.clips} clips both scored`}>
      {fmtPts(d)}
      {lo !== null && hi !== null && (
        <span className="text-[10px]">
          {' '}
          [{fmtPts(lo)}, {fmtPts(hi)}]
        </span>
      )}
    </span>
  )
}

function Sid({ sub, del, ins }: { sub: number; del: number; ins: number }) {
  return (
    <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
      S {sub.toFixed(2)} · D {del.toFixed(2)} · I {ins.toFixed(2)}
    </span>
  )
}

function Chip({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <button
      type="button"
      onClick={onClear}
      className="flex items-center gap-1 rounded-full bg-indigo-500/15 px-2 py-0.5 text-[11px] text-indigo-700 hover:bg-indigo-500/25 dark:text-indigo-300"
      title="Remove this filter"
    >
      {label}
      <RiCloseLine className="size-3" />
    </button>
  )
}

function Toggle({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={cn(
        'rounded-md px-2 py-0.5 text-[11px]',
        on ? 'bg-background font-semibold shadow-xs' : 'text-muted-foreground hover:bg-background/50',
      )}
    >
      {children}
    </button>
  )
}

// --- the files -----------------------------------------------------------------------------------

function FileList({ files, onImport, busy }: { files: ErrorFiles; onImport: (f: File[]) => void; busy: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={input}
          type="file"
          accept=".parquet"
          multiple
          className="hidden"
          onChange={(e) => {
            const picked = Array.from(e.target.files ?? [])
            e.target.value = ''
            if (picked.length) onImport(picked)
          }}
        />
        <Button variant="outline" size="sm" className="h-7 gap-1.5 text-xs" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? <Spinner className="size-3" /> : <RiUploadLine className="size-3.5" />} Import .parquet
        </Button>
        {files.files.map((f) => (
          <span
            key={f.set}
            className="flex items-baseline gap-1.5 rounded-md border px-2 py-0.5 text-[11px]"
            title={`${f.run} · ${f.fold_version} · ${f.miner_version} · written ${f.created_at}`}
          >
            <span className="font-semibold">{setLabel(f.set)}</span>
            <span className="font-mono tabular-nums">{fmtWer(f.wer)}</span>
            <span className="text-muted-foreground">{f.run}</span>
            {!f.fold_current && (
              <span className="flex items-center gap-0.5 text-amber-600 dark:text-amber-400" title="Written under older fold rules: its WER is not today's. Derive it again.">
                <RiAlertLine className="size-3" /> {f.fold_version}
              </span>
            )}
          </span>
        ))}
      </div>
      {files.refused.map((why) => (
        <p key={why} className="flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400">
          <RiAlertLine className="size-3 shrink-0" /> Cannot read {why}
        </p>
      ))}
    </div>
  )
}

// --- blocks 1 and 2 ------------------------------------------------------------------------------

function Headline({ b }: { b: ErrorBreakdown }) {
  const drift =
    b.imported_wer !== null && Math.abs(b.imported_wer - b.wer) >= 0.005 ? (
      <p className="text-[11px] text-amber-700 dark:text-amber-400">
        This file says {fmtWer(b.wer)}; the imported run says {fmtWer(b.imported_wer)}. The file was scored against the
        export's labels and the run against today's: the labels moved since. Neither is wrong.
      </p>
    ) : null
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="font-mono text-xl font-bold text-rose-600 tabular-nums dark:text-rose-400">{fmtWer(b.wer)}</span>
        {b.wer_ci && (
          <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
            [{b.wer_ci[0].toFixed(2)}, {b.wer_ci[1].toFixed(2)}]
          </span>
        )}
        <Sid sub={b.sub} del={b.del} ins={b.ins} />
        <span className="text-[11px] text-muted-foreground">
          {b.clips} clips · {b.ref_words.toLocaleString()} words · {b.errors.toLocaleString()} errors
        </span>
        {b.base && (
          <span className="text-xs">
            minus {b.base}: <Diff vs={b.vs_base} />
          </span>
        )}
      </div>
      {drift}
    </div>
  )
}

function OverlapBlock({
  b,
  active,
  onPick,
}: {
  b: ErrorBreakdown
  active: OverlapBucket | undefined
  onPick: (bucket: OverlapBucket | undefined) => void
}) {
  return (
    <div className="min-w-0 overflow-x-auto">
      <PanelHeading title="Crosstalk" note="measured on every clip · click a bucket to filter below" />
      <table className="w-full text-xs [&_td]:pr-3 [&_td]:whitespace-nowrap [&_th]:pr-3">
        <thead className="text-[10px] text-muted-foreground">
          <tr className="text-left">
            <th className="py-1 font-normal">overlap</th>
            <th className="font-normal">clips</th>
            <th className="font-normal">words</th>
            <th className="font-normal">errors</th>
            <th className="font-normal">WER</th>
            <th className="font-normal">S · D · I</th>
            {b.base && <th className="font-normal">minus base</th>}
          </tr>
        </thead>
        <tbody>
          {b.overlap.map((row) => (
            <tr
              key={row.bucket}
              onClick={() => onPick(active === row.bucket ? undefined : row.bucket)}
              className={cn('cursor-pointer border-t hover:bg-muted/50', active === row.bucket && 'bg-indigo-500/10')}
            >
              <td className="py-1">{bucketLabel(row.bucket)}</td>
              <td className="font-mono tabular-nums">{row.clips}</td>
              <td className="font-mono tabular-nums">{percent(row.share_of_words)}</td>
              <td className="font-mono tabular-nums">{percent(row.share_of_errors)}</td>
              <td className="font-mono font-semibold tabular-nums">
                {fmtWer(row.wer)}
                {row.wer_ci && (
                  <span className="text-[10px] font-normal text-muted-foreground">
                    {' '}
                    [{row.wer_ci[0].toFixed(1)}, {row.wer_ci[1].toFixed(1)}]
                  </span>
                )}
              </td>
              <td>
                <Sid sub={row.sub} del={row.del} ins={row.ins} />
              </td>
              {b.base && (
                <td className="text-[11px]">
                  <Diff vs={row.vs_base} />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {b.overlap.some((r) => r.bucket === 'unmeasured') && (
        <p className="mt-1 text-[10px] text-muted-foreground">
          Never measured: scripts/measure_benchmark_overlap.py, then derive the file again.
        </p>
      )}
    </div>
  )
}

function NumbersBlock({ b, active, onPick }: { b: ErrorBreakdown; active: boolean; onPick: () => void }) {
  const n = b.numbers
  const vs = n.vs_base
  return (
    <div className="min-w-0 space-y-1.5 text-xs">
      <PanelHeading title="Numbers" note="an error is a number error when either side is a number" />
      <button
        type="button"
        onClick={onPick}
        className={cn('w-full rounded-md border px-2 py-1 text-left hover:bg-muted/50', active && 'bg-indigo-500/10')}
        title="Filter the table below to number errors"
      >
        <span className="font-mono font-semibold tabular-nums">{n.errors.toLocaleString()}</span> number errors (S {n.sub} · D{' '}
        {n.del} · I {n.ins}), <span className="font-mono tabular-nums">{percent(n.share_of_errors, 1)}</span> of all
        {vs && <span className="text-muted-foreground"> · {vs.errors > 0 ? '+' : ''}{vs.errors} against base</span>}
      </button>
      <div className="grid grid-cols-[auto_auto_1fr] gap-x-3 gap-y-0.5">
        <span className="text-muted-foreground">WER</span>
        <span className="font-mono tabular-nums">{fmtWer(b.wer)}</span>
        <span>{b.base && <Diff vs={b.vs_base} />}</span>
        <span className="text-muted-foreground" title="Every pair that involves a number left out, its words with it">
          without numbers
        </span>
        <span className="font-mono tabular-nums">{fmtWer(n.wer_without)}</span>
        <span>{vs && <Diff vs={vs.wer_without_paired} />}</span>
        <span className="text-muted-foreground">clips with a number said</span>
        <span className="font-mono tabular-nums">{fmtWer(n.ref_clip_wer)}</span>
        <span className="flex items-baseline gap-2">
          <span className="text-[10px] text-muted-foreground">
            {n.ref_clips} clips · {n.ref_clip_words.toLocaleString()} words
          </span>
          {vs && <Diff vs={vs.ref_clip_wer} />}
        </span>
      </div>
    </div>
  )
}

// --- block 3 and the occurrences -----------------------------------------------------------------

function Filters({
  query,
  byValues,
  bothWays,
  sort,
  hasBase,
  onQuery,
  onBothWays,
  onSort,
}: {
  query: ErrorQuery
  byValues: string[]
  bothWays: boolean
  sort: 'count' | 'change'
  hasBase: boolean
  onQuery: (next: Partial<ErrorQuery>) => void
  onBothWays: (on: boolean) => void
  onSort: (sort: 'count' | 'change') => void
}) {
  const kinds = query.kind ?? []
  const flip = (kind: ErrorKind) =>
    onQuery({ kind: kinds.includes(kind) ? kinds.filter((k) => k !== kind) : [...kinds, kind] })
  const chips: [string, () => void][] = []
  if (query.overlap_bucket) chips.push([`overlap: ${bucketLabel(query.overlap_bucket)}`, () => onQuery({ overlap_bucket: undefined })])
  if (query.number !== undefined) chips.push([query.number ? 'numbers only' : 'no numbers', () => onQuery({ number: undefined })])
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap gap-0.5 rounded-lg border bg-muted/40 p-0.5">
          {KINDS.map(([kind, label]) => (
            <Toggle key={kind} on={kinds.includes(kind)} onClick={() => flip(kind)}>
              {label}
            </Toggle>
          ))}
        </div>
        <select
          aria-label="How the pair was forgiven"
          className={SELECT}
          value={query.forgiven ?? ''}
          onChange={(e) => onQuery({ forgiven: (e.target.value || undefined) as Forgiven | undefined })}
        >
          <option value="">forgiven: any</option>
          {FORGIVEN.map((f) => (
            <option key={f} value={f}>
              forgiven: {f}
            </option>
          ))}
        </select>
        {(['ref_script', 'hyp_script'] as const).map((side) => (
          <select
            key={side}
            aria-label={side === 'ref_script' ? 'Reference script' : 'Model script'}
            className={SELECT}
            value={query[side] ?? ''}
            onChange={(e) => onQuery({ [side]: (e.target.value || undefined) as ScriptName | undefined })}
          >
            <option value="">{side === 'ref_script' ? 'reference' : 'model'}: any script</option>
            {SCRIPTS.map(([value, label]) => (
              <option key={value} value={value}>
                {side === 'ref_script' ? 'reference' : 'model'}: {label}
              </option>
            ))}
          </select>
        ))}
        <select
          aria-label="Numbers"
          className={SELECT}
          value={query.number === undefined ? '' : String(query.number)}
          onChange={(e) => onQuery({ number: e.target.value === '' ? undefined : e.target.value === 'true' })}
        >
          <option value="">numbers: either</option>
          <option value="true">numbers only</option>
          <option value="false">no numbers</option>
        </select>
        {byValues.length > 0 && (
          <select
            aria-label="The set's own split"
            className={SELECT}
            value={query.by ?? ''}
            onChange={(e) => onQuery({ by: e.target.value || undefined })}
          >
            <option value="">split: all</option>
            {byValues.map((v) => (
              <option key={v} value={v}>
                {humanize(v)}
              </option>
            ))}
          </select>
        )}
        <label className="flex items-center gap-1 text-[11px] text-muted-foreground" title="Romanized similarity of a substitution, 0-1">
          alike
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            placeholder="0"
            aria-label="Similarity at least"
            className={cn(SELECT, 'w-16')}
            value={query.similarity_min ?? ''}
            onChange={(e) => onQuery({ similarity_min: e.target.value === '' ? undefined : Number(e.target.value) })}
          />
          –
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            placeholder="1"
            aria-label="Similarity at most"
            className={cn(SELECT, 'w-16')}
            value={query.similarity_max ?? ''}
            onChange={(e) => onQuery({ similarity_max: e.target.value === '' ? undefined : Number(e.target.value) })}
          />
        </label>
        <label className="flex items-center gap-1 text-[11px]">
          <input type="checkbox" checked={bothWays} onChange={(e) => onBothWays(e.target.checked)} />
          A→B and B→A as one
        </label>
        {hasBase && (
          <select aria-label="Sort" className={SELECT} value={sort} onChange={(e) => onSort(e.target.value as 'count' | 'change')}>
            <option value="count">most frequent</option>
            <option value="change">grew most against base</option>
          </select>
        )}
      </div>
      {chips.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {chips.map(([label, clear]) => (
            <Chip key={label} label={label} onClear={clear} />
          ))}
        </div>
      )}
    </div>
  )
}

function Words({ text, tone }: { text: string; tone: string }) {
  return text ? <span className={tone}>{text}</span> : <span className="text-muted-foreground">∅</span>
}

function ConfusionTable({
  page,
  offset,
  hasBase,
  bothWays,
  selected,
  onSelect,
  onPage,
}: {
  page: ConfusionPage | null
  offset: number
  hasBase: boolean
  bothWays: boolean
  selected: ConfusionRow | null
  onSelect: (row: ConfusionRow | null) => void
  onPage: (offset: number) => void
}) {
  if (!page) return <Spinner className="size-4" />
  if (page.total === 0) return <p className="text-xs text-muted-foreground">No pair matches these filters.</p>
  const same = (r: ConfusionRow) => selected && r.kind === selected.kind && r.ref === selected.ref && r.hyp === selected.hyp
  return (
    <div className="space-y-1">
      <table className="w-full text-xs [&_td]:pr-3 [&_td]:whitespace-nowrap [&_th]:pr-3">
        <thead className="text-[10px] text-muted-foreground">
          <tr className="text-left">
            <th className="py-1 font-normal">kind</th>
            <th className="font-normal">reference</th>
            <th className="font-normal">model</th>
            <th className="font-normal">count</th>
            <th className="font-normal">of its kind</th>
            {bothWays && <th className="font-normal">→ · ←</th>}
            {hasBase && <th className="font-normal">base</th>}
            {hasBase && <th className="font-normal">change</th>}
          </tr>
        </thead>
        <tbody>
          {page.rows.map((r) => (
            <tr
              key={`${r.kind}|${r.ref}|${r.hyp}`}
              onClick={() => onSelect(same(r) ? null : r)}
              className={cn('cursor-pointer border-t hover:bg-muted/50', same(r) && 'bg-indigo-500/10')}
            >
              <td className="py-1 text-muted-foreground">{r.kind}</td>
              <td className="max-w-56 truncate text-[13px]">
                <Words text={r.ref} tone="text-rose-700 dark:text-rose-300" />
              </td>
              <td className="max-w-56 truncate text-[13px]">
                <Words text={r.hyp} tone="text-amber-700 dark:text-amber-300" />
              </td>
              <td className="font-mono font-semibold tabular-nums">{r.count}</td>
              <td className="font-mono tabular-nums">{percent(r.share, 1)}</td>
              {bothWays && (
                <td className="font-mono text-muted-foreground tabular-nums">
                  {r.forward} · {r.backward}
                </td>
              )}
              {hasBase && <td className="font-mono text-muted-foreground tabular-nums">{r.base_count}</td>}
              {hasBase && (
                <td
                  className={cn(
                    'font-mono tabular-nums',
                    (r.change ?? 0) > 0 ? 'text-rose-600 dark:text-rose-400' : (r.change ?? 0) < 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-muted-foreground',
                  )}
                >
                  {fmtPts(r.change).replace('.00', '')}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>
          {offset + 1}–{offset + page.rows.length} of {page.total.toLocaleString()} distinct pairs
        </span>
        <span className="flex gap-1">
          <Button variant="ghost" size="sm" className="h-6 text-xs" disabled={offset === 0} onClick={() => onPage(Math.max(0, offset - CONFUSION_PAGE))}>
            previous
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-xs"
            disabled={offset + page.rows.length >= page.total}
            onClick={() => onPage(offset + CONFUSION_PAGE)}
          >
            next
          </Button>
        </span>
      </div>
    </div>
  )
}

function OccurrenceRow({ occ, active, onOpen }: { occ: Occurrence; active: boolean; onOpen: () => void }) {
  return (
    <div className={cn('rounded-md border px-2 py-1.5', active && 'border-primary/60 bg-muted')}>
      <div className="mb-1 flex flex-wrap items-baseline gap-x-2 text-[10px] text-muted-foreground">
        <button type="button" onClick={onOpen} className="font-mono text-foreground underline-offset-2 hover:underline">
          {occ.clip_id}
        </button>
        <span>{bucketLabel(occ.overlap_bucket)}</span>
        {occ.by && <span>{humanize(occ.by)}</span>}
        <span>group {occ.group || '—'}</span>
      </div>
      <div className="flex flex-wrap items-baseline gap-x-1.5">
        {occ.before.length > 0 && <AlignedDiff ops={occ.before} />}
        <span className="rounded ring-2 ring-indigo-500/50">
          <AlignedDiff ops={[occ]} />
        </span>
        {occ.after.length > 0 && <AlignedDiff ops={occ.after} />}
      </div>
    </div>
  )
}

function ClipView({ slug, set, clipId }: { slug: string; set: string; clipId: string }) {
  const [clip, setClip] = useState<ErrorClip | null>(null)
  useEffect(() => {
    let cancelled = false
    setClip(null)
    api
      .getErrorClip(slug, set, clipId)
      .then((next) => !cancelled && setClip(next))
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load the clip'))
    return () => {
      cancelled = true
    }
  }, [slug, set, clipId])
  if (!clip) return <Spinner className="size-4" />
  if (clip.run_id !== null && clip.segment_id !== null) {
    return <ClipPanel runId={clip.run_id} segmentId={clip.segment_id} />
  }
  return (
    <div className="space-y-2 rounded-lg border bg-card p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-mono text-xs">{clipId}</span>
        <DiffLegend />
      </div>
      <AlignedDiff ops={clip.ops} />
      <p className="text-[10px] text-muted-foreground">
        {set === 'gold' || set === 'val'
          ? 'This clip is not in the imported run, so its audio cannot be opened here.'
          : 'A public set: the harness does not hold its audio, only the alignment.'}
      </p>
    </div>
  )
}

// --- the section ---------------------------------------------------------------------------------

export function ErrorsSection({ model, models }: { model: AsrModel; models: AsrModel[] }) {
  const slug = model.slug
  const [files, setFiles] = useState<ErrorFiles | null>(null)
  const [uploading, setUploading] = useState(false)
  const [set, setSet] = useState<string | null>(null)
  const [base, setBase] = useState<string | null>(null)
  const [others, setOthers] = useState<Record<string, string[]>>({})
  const [breakdown, setBreakdown] = useState<ErrorBreakdown | null>(null)
  const [query, setQuery] = useState<ErrorQuery>({ kind: ERRORS })
  const [bothWays, setBothWays] = useState(false)
  const [sort, setSort] = useState<'count' | 'change'>('count')
  const [confOffset, setConfOffset] = useState(0)
  const [confusion, setConfusion] = useState<ConfusionPage | null>(null)
  const [row, setRow] = useState<ConfusionRow | null>(null)
  const [seed, setSeed] = useState<number | null>(null)
  const [occurrences, setOccurrences] = useState<OccurrencePage | null>(null)
  // The set a clip was opened in, so switching sets never asks the new set for the old clip.
  const [opened, setOpened] = useState<{ set: string; clipId: string } | null>(null)
  const clipId = opened && opened.set === set ? opened.clipId : null

  const loadFiles = useCallback(async () => {
    try {
      const next = await api.getModelErrors(slug)
      setFiles(next)
      setSet((current) => (current && next.files.some((f) => f.set === current) ? current : next.files[0]?.set ?? null))
    } catch (err: any) {
      toast.error(err.detail || 'Failed to load the error files')
    }
  }, [slug])

  useEffect(() => {
    setFiles(null)
    setSet(null)
    setBase(null)
    loadFiles()
  }, [loadFiles])

  // Which other model has a file for which set: the base picker offers only those.
  useEffect(() => {
    let cancelled = false
    const rest = models.filter((m) => m.slug !== slug)
    Promise.allSettled(rest.map((m) => api.getModelErrors(m.slug))).then((results) => {
      if (cancelled) return
      const next: Record<string, string[]> = {}
      results.forEach((r, i) => {
        if (r.status === 'fulfilled') next[rest[i].slug] = r.value.files.map((f) => f.set)
      })
      setOthers(next)
    })
    return () => {
      cancelled = true
    }
  }, [models, slug])

  const bases = set ? Object.keys(others).filter((s) => others[s].includes(set)) : []
  useEffect(() => {
    if (base && !bases.includes(base)) setBase(null)
  }, [base, bases.join('|')])

  useEffect(() => {
    setQuery({ kind: ERRORS })
    setRow(null)
    setOpened(null)
  }, [slug, set])

  useEffect(() => {
    if (!set) return setBreakdown(null)
    let cancelled = false
    setBreakdown(null)
    api
      .getErrorBreakdown(slug, set, base)
      .then((next) => !cancelled && setBreakdown(next))
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load the breakdown'))
    return () => {
      cancelled = true
    }
  }, [slug, set, base])

  useEffect(() => setConfOffset(0), [slug, set, base, query, bothWays, sort])
  useEffect(() => setRow(null), [query, bothWays])

  useEffect(() => {
    if (!set) return setConfusion(null)
    let cancelled = false
    api
      .getErrorConfusion(slug, set, { ...query, both_ways: bothWays || undefined, base, sort, offset: confOffset, limit: CONFUSION_PAGE })
      .then((next) => !cancelled && setConfusion(next))
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load the confusion table'))
    return () => {
      cancelled = true
    }
  }, [slug, set, base, query, bothWays, sort, confOffset])

  useEffect(() => {
    if (!set) return setOccurrences(null)
    let cancelled = false
    // A row folded both ways stands for two directions; its occurrences are the forward ones.
    const pick = row ? { kind: [row.kind], ref: row.ref, hyp: row.hyp } : {}
    api
      .getErrorPairs(slug, set, { ...query, ...pick, ...(seed === null ? { limit: SAMPLE } : { sample: SAMPLE, seed }) })
      .then((next) => !cancelled && setOccurrences(next))
      .catch((err) => !cancelled && toast.error(err.detail || 'Failed to load the occurrences'))
    return () => {
      cancelled = true
    }
  }, [slug, set, query, row, seed])

  const importFiles = async (picked: File[]) => {
    setUploading(true)
    try {
      const stored = await api.uploadModelErrors(slug, picked)
      toast.success(`Stored ${stored.map((f) => setLabel(f.set)).join(', ')} (${stored[0]?.run})`)
      await loadFiles()
    } catch (err: any) {
      toast.error(err.detail || 'Import failed', { duration: 10000 })
    } finally {
      setUploading(false)
    }
  }

  const updateQuery = (next: Partial<ErrorQuery>) => setQuery((current) => ({ ...current, ...next }))

  if (!files) {
    return (
      <Panel>
        <PanelHeading title="Errors" />
        <Spinner className="size-4" />
      </Panel>
    )
  }

  return (
    <Panel className="space-y-4">
      <PanelHeading
        title="Errors"
        note="every aligned word pair, classified · a breakdown tags errors, it never forgives them"
      />
      <FileList files={files} onImport={importFiles} busy={uploading} />
      {files.files.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No error files yet. Import the notebook's <code className="font-mono">errors/*.parquet</code>, copy them into{' '}
          <code className="font-mono">data/models/asr/{slug}/errors/</code> and rescan, or derive them from this model's runs
          with <code className="font-mono">scripts/mine_errors.py {slug}</code>.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex flex-wrap gap-0.5 rounded-lg border bg-muted/40 p-0.5">
              {files.files.map((f) => (
                <Toggle key={f.set} on={f.set === set} onClick={() => setSet(f.set)}>
                  {setLabel(f.set)}
                </Toggle>
              ))}
            </div>
            <select
              aria-label="Base model"
              className={SELECT}
              value={base ?? ''}
              onChange={(e) => setBase(e.target.value || null)}
              disabled={bases.length === 0}
              title={bases.length ? 'Compare against another model on the same set' : 'No other model has a file for this set'}
            >
              <option value="">{bases.length ? 'against: nothing' : 'no base for this set'}</option>
              {bases.map((s) => (
                <option key={s} value={s}>
                  against: {models.find((m) => m.slug === s)?.name ?? s} ({s})
                </option>
              ))}
            </select>
          </div>

          {breakdown ? (
            <>
              <Headline b={breakdown} />
              <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
                <OverlapBlock
                  b={breakdown}
                  active={query.overlap_bucket}
                  onPick={(bucket) => updateQuery({ overlap_bucket: bucket })}
                />
                <NumbersBlock
                  b={breakdown}
                  active={query.number === true}
                  onPick={() => updateQuery({ number: query.number === true ? undefined : true })}
                />
              </div>
            </>
          ) : (
            <Spinner className="size-4" />
          )}

          <div className="space-y-2">
            <PanelHeading title="Confusion table" note="which reference text the model wrote as what · click a row for its occurrences" />
            <Filters
              query={query}
              byValues={breakdown?.by_values ?? []}
              bothWays={bothWays}
              sort={sort}
              hasBase={!!base}
              onQuery={updateQuery}
              onBothWays={setBothWays}
              onSort={setSort}
            />
            <ConfusionTable
              page={confusion}
              offset={confOffset}
              hasBase={!!base}
              bothWays={bothWays}
              selected={row}
              onSelect={setRow}
              onPage={setConfOffset}
            />
          </div>

          <div className="grid gap-3 xl:grid-cols-[minmax(22rem,1fr)_1fr]">
            <div className="min-w-0 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-heading text-sm font-semibold">
                  Occurrences{' '}
                  <span className="text-[11px] font-normal text-muted-foreground">
                    {row ? `of ${row.ref || '∅'} → ${row.hyp || '∅'}` : 'under these filters'}
                    {occurrences ? ` · ${occurrences.total.toLocaleString()} in all` : ''}
                    {seed !== null ? ` · ${SAMPLE} drawn with seed ${seed}` : ` · first ${SAMPLE} in clip order`}
                  </span>
                </h3>
                <span className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1 text-xs"
                    onClick={() => setSeed(Math.floor(Math.random() * 100000))}
                    title="Draw 50 at random; the seed is shown so the same 50 can be drawn again"
                  >
                    <RiDiceLine className="size-3.5" /> Sample {SAMPLE}
                  </Button>
                  {seed !== null && (
                    <>
                      <input
                        aria-label="Seed"
                        type="number"
                        className={cn(SELECT, 'w-20')}
                        value={seed}
                        onChange={(e) => setSeed(Number(e.target.value) || 0)}
                      />
                      <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setSeed(null)}>
                        in order
                      </Button>
                    </>
                  )}
                </span>
              </div>
              <div className="max-h-[70vh] space-y-1.5 overflow-y-auto pr-1">
                {occurrences ? (
                  occurrences.rows.map((occ) => (
                    <OccurrenceRow
                      key={`${occ.clip_id}|${occ.pos}`}
                      occ={occ}
                      active={occ.clip_id === clipId}
                      onOpen={() => set && setOpened({ set, clipId: occ.clip_id })}
                    />
                  ))
                ) : (
                  <Spinner className="size-4" />
                )}
              </div>
            </div>
            <div className="min-w-0 xl:sticky xl:top-0 xl:self-start">
              {clipId && set ? (
                <ClipView slug={slug} set={set} clipId={clipId} />
              ) : (
                <p className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">
                  Click a clip id to open the clip{set === 'gold' || set === 'val' ? ' with its audio' : ''}.
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </Panel>
  )
}
