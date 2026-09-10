/**
 * The two pots, what was actually heard, and what the benchmark does not span (D63, D71).
 *
 * Condensed from the panel this replaces: the pots are a constraint on the corpus rather than the
 * subject of the page, so they get one row instead of a section. What survives is what qualifies
 * every other number here — how much of the corpus somebody listened to, and whether the
 * benchmark spans the corpus it is supposed to measure.
 */

import { RiEyeOffLine, RiSoundModuleLine } from '@remixicon/react'

import { Chip, POT_COLOR, POT_TEXT, Panel, PanelHeading, hours, humanize, percent } from './primitives'
import type { PotPanel as PotData, VerificationPanel } from '@/types'

const COVERAGE_LABEL: Record<string, string> = {
  show_id: 'Shows',
  gender: 'Gender',
  age_bracket: 'Age',
  topic: 'Topic',
}

/** One pot against its duration target, with labelled audio shaded inside ingested. */
function PotBar({
  bucket,
  label,
  entry,
  target,
}: {
  bucket: string
  label: string
  entry: { episodes: number; hours: number; labeled_hours: number }
  target?: number
}) {
  const denominator = target ?? Math.max(entry.hours, 0.001)
  const ingested = Math.min(100, (entry.hours / denominator) * 100)
  const labeled = Math.min(100, (entry.labeled_hours / denominator) * 100)
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="flex items-center gap-1.5">
          <span className={`size-2 rounded-full ${POT_COLOR[bucket]}`} />
          <span className="font-medium">{label}</span>
          <span className="text-[10px] text-muted-foreground">{entry.episodes} ep</span>
        </span>
        <span className="font-mono tabular-nums">
          <span className={`font-bold ${POT_TEXT[bucket]}`}>{hours(entry.hours)} h</span>
          {target ? <span className="text-muted-foreground"> / {hours(target)}</span> : null}
        </span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-sm bg-muted">
        <div className={`absolute inset-y-0 left-0 opacity-30 ${POT_COLOR[bucket]}`} style={{ width: `${ingested}%` }} />
        <div className={`absolute inset-y-0 left-0 ${POT_COLOR[bucket]}`} style={{ width: `${labeled}%` }} />
      </div>
      <div className="text-[10px] text-muted-foreground">
        {hours(entry.labeled_hours)} h labelled of {hours(entry.hours)} h ingested
      </div>
    </div>
  )
}

export function PotsPanel({
  pots,
  verification,
}: {
  pots: PotData
  verification: VerificationPanel
}) {
  const trainPot = {
    episodes: pots.buckets.train.episodes + pots.buckets.val.episodes,
    hours: pots.buckets.train.hours + pots.buckets.val.hours,
    labeled_hours: pots.buckets.train.labeled_hours + pots.buckets.val.labeled_hours,
  }
  const verifiedPct = verification.total ? verification.verified / verification.total : 0

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Panel>
        <PanelHeading
          icon={<RiSoundModuleLine className="size-4 text-primary" />}
          title="The two pots"
          note={
            pots.gold_episodes_spanning_pots > 0
              ? `${pots.gold_segments_in_spanning_episodes} gold clips share an episode with train`
              : undefined
          }
        />
        <div className="space-y-3">
          <PotBar
            bucket="gold"
            label={`Gold — ${pots.buckets.gold.segments} clips`}
            entry={pots.buckets.gold}
            target={pots.gold_target_hours}
          />
          <PotBar bucket="train" label="Train pot" entry={trainPot} target={pots.train_target_hours} />
          {pots.buckets.unassigned.episodes > 0 ? (
            <PotBar bucket="unassigned" label="Unplaced" entry={pots.buckets.unassigned} />
          ) : null}
        </div>
      </Panel>

      <Panel>
        <PanelHeading
          icon={<RiEyeOffLine className="size-4 text-primary" />}
          title="How it was checked"
          note="the claim the corpus makes about itself"
        />
        <div className="space-y-3">
          <div className="flex h-2.5 w-full overflow-hidden rounded-sm bg-muted">
            <div className="bg-violet-500" style={{ width: `${verifiedPct * 100}%` }} />
            <div className="bg-slate-400 dark:bg-slate-600" style={{ width: `${(1 - verifiedPct) * 100}%` }} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-md border p-2">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-violet-600 dark:text-violet-400">
                <span className="size-2 rounded-full bg-violet-500" />
                Verified — played
              </div>
              <div className="font-mono text-base font-bold">{verification.verified.toLocaleString()}</div>
              <div className="text-[10px] text-muted-foreground">
                {hours(verification.hours.verified)} h heard · {percent(verifiedPct)}
              </div>
            </div>
            <div className="rounded-md border p-2">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                <span className="size-2 rounded-full bg-slate-400 dark:bg-slate-600" />
                Screened — not
              </div>
              <div className="font-mono text-base font-bold">{verification.screened.toLocaleString()}</div>
              <div className="text-[10px] text-muted-foreground">
                {hours(verification.hours.screened)} h waved through
              </div>
            </div>
          </div>
        </div>
      </Panel>

      <Panel>
        <PanelHeading
          icon={<RiSoundModuleLine className="size-4 text-primary" />}
          title="What the benchmark spans"
          note={pots.coverage_complete ? 'full coverage' : 'gaps'}
        />
        <div className="space-y-2.5">
          {Object.entries(pots.corpus_coverage).map(([key, values]) => {
            const covered = pots.gold_coverage[key] ?? {}
            const names = Object.keys(values).sort()
            return (
              <div key={key} className="space-y-1">
                <div className="flex items-baseline justify-between">
                  <span className="font-heading text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {COVERAGE_LABEL[key] ?? key}
                  </span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {Object.keys(covered).length}/{names.length} in gold
                  </span>
                </div>
                {names.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground">Not recorded yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {names.map((name) => (
                      <Chip
                        key={name}
                        tone={name in covered ? 'present' : 'absent'}
                        title={
                          name in covered
                            ? `${covered[name]} gold episode(s)`
                            : 'in the corpus but not in the gold pot'
                        }
                      >
                        {humanize(name)}
                      </Chip>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </Panel>
    </div>
  )
}
