/**
 * The voices page (D91, D104): every anonymous voice in the corpus, followed across episodes.
 *
 * Pick a voice on the left; on the right, play its sample, give it a gender and an age bracket by
 * ear, see where it speaks, and listen to its clips -- its stretches alone, which can be confirmed
 * into its print (D99), and, on request, every clip it shares with someone else. A voice is an id
 * and nothing more (D56): what it can be given is exactly what a declared speaker row may carry.
 * "Tag by ear" swaps the two panes for `VoiceTagger`, which walks the voices still missing either.
 */

import { Fragment, useEffect, useMemo, useState } from 'react'
import {
  RiHeadphoneLine,
  RiErrorWarningLine,
  RiFileList3Line,
  RiPauseFill,
  RiPlayFill,
  RiRefreshLine,
  RiSearchLine,
  RiUserVoiceLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { VoiceClips, useVoiceSample } from '@/components/VoiceDialog'
import { VoiceTagger } from '@/components/VoiceTagger'
import { bucketLabel, minutes } from '@/components/analytics/labels'
import { POT_TEXT, Stat, percent } from '@/components/analytics/primitives'
import { cn } from '@/lib/utils'
import { api } from '@/services/api'
import type { VoiceList, VoiceProfile } from '@/types'

const GENDERS = ['female', 'male'] as const
const AGES = ['under_20', '20_39', '40_59', '60_79', '80_plus'] as const
const NONE = '__none__'
/** A picked value must read as picked at a glance: the default toggle tint is too faint. */
const ON = 'data-[state=on]:bg-primary data-[state=on]:text-primary-foreground'

type PotFilter = 'all' | 'train' | 'gold'
type SortKey = 'talk' | 'episodes' | 'voice'

const RESOLVED_BY: Record<string, string> = {
  only_pair: 'one declared row, one voice',
  recurring_host: 'the one recurring voice is the declared host',
  rows_agree: 'every remaining declared row agrees',
}

const trainMinutes = (v: VoiceProfile) => (v.hours.train + v.hours.val) * 60
const goldMinutes = (v: VoiceProfile) => v.hours.gold * 60

/** Where a field's value came from: by ear, from the declared rows, or nowhere. */
function source(voice: VoiceProfile, field: 'gender' | 'age_bracket'): 'ear' | 'rows' | null {
  if (voice.manual.includes(field)) return 'ear'
  return voice[field] ? 'rows' : null
}

function PersonCell({ voice }: { voice: VoiceProfile }) {
  if (voice.identity === 'conflict' && !voice.gender && !voice.age_bracket) {
    return <span className="text-amber-600 dark:text-amber-400">conflict</span>
  }
  if (!voice.gender && !voice.age_bracket) return <span className="text-muted-foreground">—</span>
  const byEar = voice.manual.length > 0
  return (
    <span className="inline-flex items-center gap-1" title={byEar ? 'set by ear' : 'from the declared rows'}>
      {byEar ? <RiHeadphoneLine className="size-3 text-muted-foreground" /> : <RiFileList3Line className="size-3 text-muted-foreground" />}
      {voice.gender ?? '?'} · {voice.age_bracket ? bucketLabel(voice.age_bracket) : '?'}
    </span>
  )
}

function Field({
  label,
  voice,
  field,
  options,
  saving,
  onSet,
}: {
  label: string
  voice: VoiceProfile
  field: 'gender' | 'age_bracket'
  options: readonly string[]
  saving: boolean
  onSet: (value: string | null) => void
}) {
  const from = source(voice, field)
  return (
    <div className="space-y-1">
      <div className="flex items-baseline gap-2 text-xs">
        <span className="font-semibold">{label}</span>
        <span className="text-[11px] text-muted-foreground">
          {from === 'ear' ? 'set by ear' : from === 'rows' ? 'from the declared rows' : 'not assigned'}
        </span>
        {voice.disagrees.includes(field) ? (
          <span className="flex items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400">
            <RiErrorWarningLine className="size-3" /> the episode rows say otherwise
          </span>
        ) : null}
      </div>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        spacing={0}
        disabled={saving}
        value={voice[field] ?? NONE}
        onValueChange={(v) => {
          if (!v) return
          onSet(v === NONE ? null : v)
        }}
        aria-label={label}
      >
        {options.map((o) => (
          <ToggleGroupItem key={o} value={o} className={cn('px-2.5', ON)}>
            {bucketLabel(o)}
          </ToggleGroupItem>
        ))}
        <ToggleGroupItem value={NONE} className="px-2.5 text-muted-foreground" title="Leave unassigned by ear">
          —
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  )
}

function VoiceDetail({
  voice,
  onChange,
  onPick,
  playVoice,
  playingVoice,
}: {
  voice: VoiceProfile
  onChange: (next: VoiceProfile) => void
  onPick: (voice: string) => void
  playVoice: (voice: string) => void
  playingVoice: string | null
}) {
  const [saving, setSaving] = useState(false)

  const set = async (field: 'gender' | 'age_bracket', value: string | null) => {
    // Send both by-ear fields: a field left to the declared rows stays null here.
    const next = {
      gender: voice.manual.includes('gender') ? voice.gender : null,
      age_bracket: voice.manual.includes('age_bracket') ? voice.age_bracket : null,
      [field]: value,
    }
    setSaving(true)
    try {
      await api.setVoiceAttributes(voice.voice, next)
      const manual = (['gender', 'age_bracket'] as const).filter((k) => next[k] !== null)
      onChange({
        ...voice,
        [field]: value,
        manual,
        disagrees: voice.disagrees.filter((k) => k !== field),
        identity: value || voice.gender || voice.age_bracket ? 'resolved' : voice.identity,
      })
      if (value === null) toast.info('Cleared by ear; refresh to see what the declared rows say')
    } catch (err: any) {
      toast.error(err.detail || 'Could not save')
    } finally {
      setSaving(false)
    }
  }

  const coVoices = Object.entries(voice.co_voices).slice(0, 12)
  return (
    <div className="flex min-h-0 flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          size="icon"
          variant={playingVoice === voice.voice ? 'default' : 'outline'}
          onClick={() => playVoice(voice.voice)}
          aria-label="Play a sample of this voice"
          title="Play its best stretch alone, confirmed first"
        >
          {playingVoice === voice.voice ? <RiPauseFill /> : <RiPlayFill />}
        </Button>
        <h2 className="font-mono text-xl font-bold">{voice.voice}</h2>
        <span className="text-xs text-muted-foreground">
          {minutes(voice.talk_minutes)} of talk · {voice.episode_count} episode{voice.episode_count === 1 ? '' : 's'} ·{' '}
          {voice.shows.length} show{voice.shows.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="grid gap-3 rounded-lg border bg-card p-3 sm:grid-cols-2">
        <Field label="Gender" voice={voice} field="gender" options={GENDERS} saving={saving} onSet={(v) => set('gender', v)} />
        <Field label="Age" voice={voice} field="age_bracket" options={AGES} saving={saving} onSet={(v) => set('age_bracket', v)} />
        {Object.keys(voice.resolved_by).length ? (
          <p className="text-[11px] text-muted-foreground sm:col-span-2">
            Declared rows reach it in{' '}
            {Object.entries(voice.resolved_by)
              .map(([k, n]) => `${n} episode${n === 1 ? '' : 's'} (${RESOLVED_BY[k] ?? k})`)
              .join('; ')}
            .
          </p>
        ) : null}
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">Leads</dt>
          <dd className="font-mono">
            {voice.clips} clips · {voice.words.toLocaleString()} words
          </dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">Pots</dt>
          <dd className="font-mono">
            <span className={POT_TEXT.train}>train {minutes(trainMinutes(voice))}</span> ·{' '}
            <span className={POT_TEXT.gold}>gold {minutes(goldMinutes(voice))}</span>
          </dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">Roles</dt>
          <dd>
            {Object.entries(voice.roles)
              .map(([r, n]) => (n > 1 ? `${r} ×${n}` : r))
              .join(', ') || '—'}
          </dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">Genres</dt>
          <dd className="truncate" title={voice.genres.join(', ')}>
            {voice.genres.map(bucketLabel).join(', ') || '—'}
          </dd>
        </div>
      </dl>

      <div>
        <h3 className="mb-1 font-heading text-xs font-semibold">Episodes</h3>
        <div className="max-h-40 space-y-0.5 overflow-y-auto text-[11px]">
          {voice.episodes.map((e) => (
            <div key={e.external_id} className="flex min-w-0 items-baseline gap-2">
              <span className="min-w-0 flex-1 truncate" title={e.title ?? e.external_id}>
                {e.title ?? e.external_id}
              </span>
              <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
                {minutes(e.minutes)} · {percent(e.share)}
                {e.role ? ` · ${e.role}` : ''}
                {e.with_voices.length ? (
                  <>
                    {' · with '}
                    {e.with_voices.slice(0, 4).map((v, i) => (
                      <Fragment key={v}>
                        {i > 0 ? ', ' : ''}
                        <button type="button" className="underline decoration-dotted hover:text-foreground" onClick={() => onPick(v)}>
                          {v}
                        </button>
                      </Fragment>
                    ))}
                    {e.with_voices.length > 4 ? ` +${e.with_voices.length - 4}` : ''}
                  </>
                ) : (
                  ' · alone'
                )}
              </span>
            </div>
          ))}
        </div>
        {coVoices.length ? (
          <p className="mt-1 text-[11px] text-muted-foreground">
            Shares rooms with{' '}
            {coVoices.map(([v, n], i) => (
              <Fragment key={v}>
                {i > 0 ? ', ' : ''}
                <button type="button" className="font-mono underline decoration-dotted hover:text-foreground" onClick={() => onPick(v)}>
                  {v}
                </button>
                {n > 1 ? ` ×${n}` : ''}
              </Fragment>
            ))}
          </p>
        ) : null}
      </div>

      <div className="flex min-h-[24rem] flex-1 flex-col">
        <h3 className="mb-1 font-heading text-xs font-semibold">Clips</h3>
        <VoiceClips key={voice.voice} voice={voice.voice} allowShared globalKeys className="flex-1" />
      </div>
    </div>
  )
}

export function VoicesView() {
  const [data, setData] = useState<VoiceList | null>(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [pot, setPot] = useState<PotFilter>('all')
  const [needsPerson, setNeedsPerson] = useState(false)
  const [sort, setSort] = useState<SortKey>('talk')
  const [selected, setSelected] = useState<string | null>(null)
  const [tagging, setTagging] = useState(false)
  const { playVoice, playingVoice, stop } = useVoiceSample()

  const load = async () => {
    setLoading(true)
    try {
      const next = await api.getVoices()
      setData(next)
      setSelected((s) => s ?? next.voices[0]?.voice ?? null)
    } catch (err: any) {
      toast.error(err.detail || 'Failed to load the voices')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const rows = useMemo(() => {
    if (!data) return []
    const q = query.trim().toLowerCase()
    const out = data.voices.filter((v) => {
      if (q && !v.voice.includes(q) && !v.episodes.some((e) => (e.title ?? e.external_id).toLowerCase().includes(q)))
        return false
      if (pot === 'gold' && goldMinutes(v) === 0) return false
      if (pot === 'train' && trainMinutes(v) === 0) return false
      if (needsPerson && v.gender && v.age_bracket) return false
      return true
    })
    const key = {
      talk: (v: VoiceProfile) => -v.talk_minutes,
      episodes: (v: VoiceProfile) => -v.episode_count,
      voice: (v: VoiceProfile) => Number(v.voice.slice(1)),
    }[sort]
    return out.sort((a, b) => key(a) - key(b))
  }, [data, query, pot, needsPerson, sort])

  const current = data?.voices.find((v) => v.voice === selected) ?? null

  const replace = (next: VoiceProfile) =>
    setData((d) => {
      if (!d) return d
      const voices = d.voices.map((v) => (v.voice === next.voice ? next : v))
      return {
        ...d,
        voices,
        summary: {
          ...d.summary,
          gender_resolved: voices.filter((v) => v.gender).length,
          age_resolved: voices.filter((v) => v.age_bracket).length,
        },
      }
    })

  if (loading && !data) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-sm text-muted-foreground">
        <Spinner className="size-6 text-primary" />
        <span>Following every voice…</span>
      </div>
    )
  }
  if (!data) return null

  const s = data.summary
  const inGold = data.voices.filter((v) => goldMinutes(v) > 0).length
  const inTrain = data.voices.filter((v) => trainMinutes(v) > 0).length
  const untagged = data.voices.filter((v) => !v.gender || !v.age_bracket).length

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden bg-background p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex items-center gap-2">
          <RiUserVoiceLine className="size-5 text-primary" />
          <h1 className="font-heading text-lg font-bold tracking-tight">Voices</h1>
          <p className="text-xs text-muted-foreground">
            anonymous ids linked across episodes by the diarizer's embeddings; gender and age are declared or set by ear
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={tagging ? 'default' : 'outline'}
            size="sm"
            onClick={() => {
              stop()
              setTagging((t) => !t)
            }}
            className="h-8 gap-1.5"
            title="Hear each voice without a gender or an age and set both with one key each"
          >
            <RiHeadphoneLine className="size-3.5" />
            Tag by ear · {untagged}
          </Button>
          <Button variant="outline" size="sm" onClick={load} className="h-8 gap-1.5">
            <RiRefreshLine className="size-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-3 divide-x divide-y rounded-lg border bg-card sm:grid-cols-6 sm:divide-y-0">
        <Stat label="Voices" value={s.voices} sub={`${s.talk_hours.toFixed(1)} h of talk`} />
        <Stat label="Gender known" value={s.gender_resolved} sub={`${s.voices - s.gender_resolved} to assign`} />
        <Stat label="Age known" value={s.age_resolved} sub={`${s.voices - s.age_resolved} to assign`} />
        <Stat label="Recurring" value={s.recurring} sub={`${s.single_episode} in one episode`} />
        <Stat label="In train" value={inTrain} sub="lead a train/val clip" />
        <Stat label="In gold" value={inGold} sub="lead a gold clip" />
      </div>

      {tagging ? (
        <VoiceTagger voices={data.voices} initialPot={pot} onChange={replace} onClose={() => setTagging(false)} />
      ) : (
        <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div className="flex min-h-0 flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <div className="relative">
                <RiSearchLine className="absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="voice or episode"
                  className="h-8 w-44 pl-7 text-xs"
                />
              </div>
              <ToggleGroup type="single" variant="outline" size="sm" spacing={0} value={pot} onValueChange={(v) => v && setPot(v as PotFilter)}>
                <ToggleGroupItem value="all">All</ToggleGroupItem>
                <ToggleGroupItem value="train">Train + val</ToggleGroupItem>
                <ToggleGroupItem value="gold">Gold</ToggleGroupItem>
              </ToggleGroup>
              <label className="flex items-center gap-1.5">
                <Switch checked={needsPerson} onCheckedChange={setNeedsPerson} />
                needs gender or age
              </label>
              <select
                value={sort}
                onChange={(e) => setSort(e.target.value as SortKey)}
                className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                aria-label="Sort"
              >
                <option value="talk">most talk</option>
                <option value="episodes">most episodes</option>
                <option value="voice">id</option>
              </select>
              <span className="ml-auto text-muted-foreground">{rows.length} shown</span>
            </div>
            <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto rounded-lg border bg-card">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-card text-[10px] uppercase tracking-wider text-muted-foreground">
                  <tr className="border-b text-left">
                    <th className="w-8" />
                    <th className="py-1.5 pr-2 font-semibold">Voice</th>
                    <th className="py-1.5 pr-2 font-semibold">Person</th>
                    <th className="py-1.5 pr-2 text-right font-semibold">Talk</th>
                    <th className="py-1.5 pr-2 text-right font-semibold">Eps</th>
                    <th className="py-1.5 pr-2 font-semibold">Pots</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((v) => (
                    <tr
                      key={v.voice}
                      onClick={() => setSelected(v.voice)}
                      className={cn(
                        'cursor-pointer border-b border-border/50 hover:bg-muted/40',
                        selected === v.voice && 'bg-indigo-500/10',
                      )}
                    >
                      <td className="py-1 pl-1">
                        <Button
                          size="icon-sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation()
                            playVoice(v.voice)
                          }}
                          aria-label={`Play ${v.voice}`}
                        >
                          {playingVoice === v.voice ? <RiPauseFill /> : <RiPlayFill />}
                        </Button>
                      </td>
                      <td className="py-1 pr-2 font-mono">{v.voice}</td>
                      <td className="py-1 pr-2">
                        <PersonCell voice={v} />
                      </td>
                      <td className="py-1 pr-2 text-right font-mono tabular-nums">{minutes(v.talk_minutes)}</td>
                      <td className="py-1 pr-2 text-right font-mono tabular-nums">{v.episode_count}</td>
                      <td className="py-1 pr-2 font-mono text-[10px]">
                        {trainMinutes(v) > 0 ? <span className={POT_TEXT.train}>train </span> : null}
                        {goldMinutes(v) > 0 ? <span className={POT_TEXT.gold}>gold</span> : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="scrollbar-thin min-h-0 overflow-y-auto pr-1">
            {current ? (
              <VoiceDetail
                voice={current}
                onChange={replace}
                onPick={setSelected}
                playVoice={playVoice}
                playingVoice={playingVoice}
              />
            ) : (
              <p className="p-8 text-center text-sm text-muted-foreground">Pick a voice.</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
