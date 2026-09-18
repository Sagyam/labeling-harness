/**
 * The voices: one anonymous person per row, followed across every clip and episode (D91).
 *
 * A voice is an id and its measurements (D56). The table is sorted by talk time under the page's
 * filters; clicking a voice filters the whole page to it, and opening a row shows the episodes
 * it appears in as a strip -- one block per episode, sized by minutes, coloured by show -- with
 * the people it shared each room with. What the corpus knows about the person (gender, age) is
 * shown with the rule that resolved it, and never guessed from the audio.
 */

import { Fragment, useMemo, useState } from 'react'
import { RiArrowDownSLine, RiArrowRightSLine, RiUserVoiceLine } from '@remixicon/react'

import { GoalBadge, POT_TEXT, Panel, PanelHeading, Stat, percent } from '@/components/analytics/primitives'
import { bucketLabel, minutes } from '@/components/analytics/labels'
import { type Filters, type VoiceCut, voiceCuts } from '@/components/analytics/model'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import type { CorpusInventory, VoiceProfile } from '@/types'

const SHOW_AT_FIRST = 25

/** Categorical, one per show, assigned in order of first appearance and stable for the page. */
const SHOW_FILLS = [
  'bg-indigo-500',
  'bg-emerald-500',
  'bg-amber-500',
  'bg-sky-500',
  'bg-rose-500',
  'bg-violet-500',
  'bg-teal-500',
  'bg-orange-500',
  'bg-lime-600',
  'bg-fuchsia-500',
  'bg-cyan-600',
]

const RESOLVED_BY: Record<string, string> = {
  only_pair: 'one declared row, one voice',
  recurring_host: 'the one recurring voice is the declared host',
  rows_agree: 'every remaining declared row agrees',
}

function Identity({ voice }: { voice: VoiceProfile }) {
  const how = Object.keys(voice.resolved_by)
    .map((k) => RESOLVED_BY[k] ?? k)
    .join('; ')
  if (voice.identity === 'conflict') {
    return (
      <span className="text-amber-600 dark:text-amber-400" title="Two episodes declare this voice differently">
        conflict
      </span>
    )
  }
  if (!voice.gender && !voice.age_bracket) {
    return <span className="text-muted-foreground">unresolved</span>
  }
  return (
    <span title={how}>
      {voice.gender ?? '?'} · {voice.age_bracket ? bucketLabel(voice.age_bracket) : '?'}
    </span>
  )
}

function Roles({ voice }: { voice: VoiceProfile }) {
  const entries = Object.entries(voice.roles)
  if (!entries.length) return <span className="text-muted-foreground">–</span>
  return (
    <span>
      {entries.map(([role, n]) => (
        <span key={role} className="mr-1">
          {role}
          {n > 1 ? <span className="text-muted-foreground"> ×{n}</span> : null}
        </span>
      ))}
    </span>
  )
}

function EpisodeStrip({
  voice,
  fills,
  onPickVoice,
}: {
  voice: VoiceProfile
  fills: Map<string, string>
  onPickVoice: (voice: string) => void
}) {
  const total = voice.episodes.reduce((s, e) => s + e.minutes, 0) || 1
  return (
    <div className="space-y-2 px-2 py-2">
      <div className="flex h-5 w-full gap-px overflow-hidden rounded">
        {voice.episodes.map((e) => (
          <div
            key={e.external_id}
            className={cn('h-full min-w-[3px]', fills.get(e.show_id ?? '') ?? 'bg-muted-foreground/40')}
            style={{ width: `${(e.minutes / total) * 100}%` }}
            title={`${e.title ?? e.external_id}\n${e.show_id ?? 'no show'} · ${e.published_at ?? 'undated'} · ${e.split}\n${minutes(e.minutes)} in ${e.clips} clips, ${percent(e.share)} of the episode's talk${e.role ? ` · ${e.role}` : ''}${e.with_voices.length ? `\nwith ${e.with_voices.join(', ')}` : '\nalone'}`}
          />
        ))}
      </div>
      <div className="grid gap-x-4 gap-y-1 text-[11px] sm:grid-cols-2 lg:grid-cols-3">
        {voice.episodes.map((e) => (
          <div key={e.external_id} className="flex min-w-0 items-baseline gap-1.5">
            <span className={cn('mt-1 inline-block size-2 shrink-0 rounded-sm', fills.get(e.show_id ?? '') ?? 'bg-muted-foreground/40')} />
            <span className="min-w-0 truncate" title={e.title ?? e.external_id}>
              {e.title ?? e.external_id}
            </span>
            <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
              {minutes(e.minutes)} · {percent(e.share)}
              {e.role ? ` · ${e.role}` : ''}
              {e.with_voices.length ? (
                <>
                  {' · with '}
                  {e.with_voices.map((v, i) => (
                    <Fragment key={v}>
                      {i > 0 ? ', ' : ''}
                      <button type="button" className="underline decoration-dotted hover:text-foreground" onClick={() => onPickVoice(v)}>
                        {v}
                      </button>
                    </Fragment>
                  ))}
                </>
              ) : (
                ' · alone'
              )}
            </span>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span>
          {voice.words.toLocaleString()} attributed words
          {voice.usable ? '' : ' (under 300: not a usable voice)'}
        </span>
        <span>verified {minutes(voice.verified_minutes)} · screened {minutes(voice.screened_minutes)}</span>
        <span>
          <span className={POT_TEXT.gold}>gold {minutes(voice.hours.gold * 60)}</span> ·{' '}
          <span className={POT_TEXT.val}>val {minutes(voice.hours.val * 60)}</span> ·{' '}
          <span className={POT_TEXT.train}>train {minutes(voice.hours.train * 60)}</span>
        </span>
        <span>seen in train: {bucketLabel(voice.exposure)}</span>
        {voice.first_seen ? (
          <span>
            {voice.first_seen}
            {voice.last_seen && voice.last_seen !== voice.first_seen ? ` → ${voice.last_seen}` : ''}
          </span>
        ) : null}
        {Object.keys(voice.resolved_by).length ? (
          <span>
            resolved by{' '}
            {Object.entries(voice.resolved_by)
              .map(([k, n]) => `${RESOLVED_BY[k] ?? k} (${n})`)
              .join('; ')}
          </span>
        ) : null}
      </div>
    </div>
  )
}

export function VoicesPanel({
  inventory,
  filters,
  activeVoice,
  onPickVoice,
}: {
  inventory: CorpusInventory
  filters: Filters
  activeVoice: string | null
  onPickVoice: (voice: string | null) => void
}) {
  const [recurringOnly, setRecurringOnly] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const [all, setAll] = useState(false)
  const summary = inventory.voice_summary
  const filtered = Object.keys(filters).some((k) => k !== 'voice')
  const cuts = useMemo(() => voiceCuts(inventory, filters), [inventory, filters])

  const fills = useMemo(() => {
    const map = new Map<string, string>()
    const shows = inventory.episodes.map((e) => e.show_id ?? '').filter((s) => s)
    for (const show of shows) if (!map.has(show)) map.set(show, SHOW_FILLS[map.size % SHOW_FILLS.length])
    return map
  }, [inventory])

  const rows = useMemo(() => {
    const out: Array<{ voice: VoiceProfile; cut: VoiceCut }> = []
    inventory.voices.forEach((voice, index) => {
      const cut = cuts.get(index)
      if (!cut) return
      if (recurringOnly && voice.episode_count < 2) return
      out.push({ voice, cut })
    })
    out.sort((a, b) => b.cut.seconds - a.cut.seconds)
    return out
  }, [inventory, cuts, recurringOnly])

  const shown = all || activeVoice ? rows : rows.slice(0, SHOW_AT_FIRST)

  return (
    <Panel>
      <PanelHeading
        icon={<RiUserVoiceLine className="size-4 text-primary" />}
        title="Voices"
        note="anonymous ids linked across episodes by the diarizer's embeddings (D87); nothing about a person is inferred from audio"
      >
        <GoalBadge goal="both" title="The unit of a sociolinguistic claim, and what a benchmark should hold out" />
      </PanelHeading>

      <div className="mb-3 grid grid-cols-2 divide-x divide-y rounded-lg border sm:grid-cols-4 lg:grid-cols-8 lg:divide-y-0">
        <Stat label="Voices" value={summary.voices} sub={`${summary.talk_hours} h of linked talk`} />
        <Stat
          label="Usable"
          value={summary.usable}
          sub="300+ attributed words"
          title="The paper's n: voices with enough words attributed to one person to measure anything about them"
        />
        <Stat label="Recurring" value={summary.recurring} sub={`${summary.single_episode} in one episode only`} />
        <Stat
          label="Hosts"
          value={summary.recurring_hosts}
          sub="5+ host–guest episodes"
          tone={summary.recurring_hosts < 3 ? 'text-amber-600 dark:text-amber-400' : ''}
          title={summary.recurring_host_voices.join(', ') || 'none'}
        />
        <Stat
          label="Cross-show guests"
          value={summary.guests_on_two_shows}
          sub="a guest with two hosts"
          tone={summary.guests_on_two_shows === 0 ? 'text-amber-600 dark:text-amber-400' : ''}
        />
        <Stat label="Gender known" value={`${summary.gender_resolved}/${summary.voices}`} sub="voices resolved to a row" />
        <Stat label="Age known" value={`${summary.age_resolved}/${summary.voices}`} sub="voices resolved to a row" />
        <Stat
          label="Largest voice"
          value={percent(summary.top_voice_share)}
          sub={summary.top_voice ?? '–'}
          tone={summary.top_voice_share > 0.25 ? 'text-amber-600 dark:text-amber-400' : ''}
        />
      </div>

      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5">
            <Switch checked={recurringOnly} onCheckedChange={setRecurringOnly} />
            recurring only
          </label>
          {activeVoice ? (
            <button
              type="button"
              className="rounded-full bg-indigo-500/15 px-2 py-0.5 text-[11px] text-indigo-700 hover:bg-indigo-500/25 dark:text-indigo-300"
              onClick={() => onPickVoice(null)}
            >
              page filtered to {activeVoice} · clear
            </button>
          ) : null}
        </div>
        <span className="text-muted-foreground">
          {rows.length} voices{filtered ? ' in the filtered clips' : ''}; talk is time in the corpus's clips
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
            <tr className="border-b text-left">
              <th className="w-6" />
              <th className="py-1 pr-2 font-semibold">Voice</th>
              <th className="py-1 pr-2 font-semibold">Person</th>
              <th className="py-1 pr-2 font-semibold">Role</th>
              <th className="py-1 pr-2 text-right font-semibold" title="Length of the clips this voice leads, under the page's filters">
                Leads
              </th>
              <th className="py-1 pr-2 text-right font-semibold">Clips</th>
              <th className="py-1 pr-2 text-right font-semibold">Episodes</th>
              <th className="py-1 pr-2 text-right font-semibold">Shows</th>
              <th className="py-1 pr-2 text-right font-semibold" title="Hours-weighted code-mixing index of the clips this voice leads">
                CMI
              </th>
              <th className="py-1 pr-2 text-right font-semibold" title="Reference words per second of speech">
                w/s
              </th>
              <th className="py-1 pr-2 text-right font-semibold" title="Share of this voice's clips someone listened to">
                verified
              </th>
              <th className="py-1 font-semibold">Pots</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(({ voice, cut }) => {
              const isOpen = open === voice.voice
              const led = voice.clips ? voice.verified_minutes / (voice.verified_minutes + voice.screened_minutes || 1) : 0
              return (
                <Fragment key={voice.voice}>
                  <tr
                    className={cn(
                      'border-b border-border/60 hover:bg-muted/40',
                      activeVoice === voice.voice && 'bg-indigo-500/10',
                      !voice.usable && 'text-muted-foreground'
                    )}
                  >
                    <td className="py-1">
                      <button
                        type="button"
                        onClick={() => setOpen(isOpen ? null : voice.voice)}
                        className="rounded p-0.5 hover:bg-muted"
                        aria-label={isOpen ? 'Collapse' : 'Expand'}
                      >
                        {isOpen ? <RiArrowDownSLine className="size-3.5" /> : <RiArrowRightSLine className="size-3.5" />}
                      </button>
                    </td>
                    <td className="py-1 pr-2 font-mono">
                      <button
                        type="button"
                        className={cn('hover:underline', activeVoice === voice.voice && 'font-bold')}
                        onClick={() => onPickVoice(activeVoice === voice.voice ? null : voice.voice)}
                        title="Filter the page to this voice"
                      >
                        {voice.voice}
                      </button>
                      {summary.recurring_host_voices.includes(voice.voice) ? (
                        <span className="ml-1 rounded bg-muted px-1 text-[9px] uppercase tracking-wide">host</span>
                      ) : null}
                    </td>
                    <td className="py-1 pr-2">
                      <Identity voice={voice} />
                    </td>
                    <td className="py-1 pr-2">
                      <Roles voice={voice} />
                    </td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums">{minutes(cut.seconds / 60)}</td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums">{cut.clips}</td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums">{voice.episode_count}</td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums" title={voice.shows.join(', ')}>
                      {voice.shows.length}
                    </td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums">
                      {voice.mean_cmi === null ? '–' : voice.mean_cmi.toFixed(0)}
                    </td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums">
                      {voice.words_per_second === null ? '–' : voice.words_per_second.toFixed(1)}
                    </td>
                    <td className="py-1 pr-2 text-right font-mono tabular-nums">{percent(led)}</td>
                    <td className="py-1 font-mono text-[10px]">
                      {voice.hours.gold > 0 ? <span className={POT_TEXT.gold}>gold </span> : null}
                      {voice.hours.val > 0 ? <span className={POT_TEXT.val}>val </span> : null}
                      {voice.hours.train > 0 ? <span className={POT_TEXT.train}>train</span> : null}
                    </td>
                  </tr>
                  {isOpen ? (
                    <tr className="border-b bg-muted/20">
                      <td />
                      <td colSpan={11}>
                        <EpisodeStrip voice={voice} fills={fills} onPickVoice={onPickVoice} />
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
      {rows.length > SHOW_AT_FIRST && !activeVoice ? (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-2 text-[11px] text-indigo-600 hover:underline dark:text-indigo-400"
        >
          {all ? `Show the top ${SHOW_AT_FIRST}` : `Show all ${rows.length} voices`}
        </button>
      ) : null}
    </Panel>
  )
}
