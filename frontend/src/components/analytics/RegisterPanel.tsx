/**
 * How much English gets mixed in, and — the part that matters for sourcing — whether the shows
 * differ from each other about it.
 *
 * Two views of one variable, answering two different questions:
 *
 * * The **histogram** is over clips, weighted by audio. It shows the shape of the corpus.
 * * The **show strip** is over show means. It shows the shape of the *sample of speakers*.
 *
 * Only the second one is about what to record next. Clip density spans nearly the whole range
 * inside any single show — one sentence is all Nepali, the next is half English — so a wide
 * histogram is perfectly compatible with every speaker in the corpus being the same kind of
 * speaker. A code-switching study needs variance in the speaker, and that is the strip.
 */

import { RiTranslate2 } from '@remixicon/react'

import { HEAT_STEPS, Panel, PanelHeading, heatStep, hours, percent } from './primitives'
import type { RegisterData } from '@/types'

/**
 * Audio by code-switch density, in ten bins. Sequential fill keyed to the bin's own position, so
 * the colour encodes the same variable the x-axis does rather than a second, invented one.
 */
function Histogram({ data }: { data: RegisterData }) {
  const peak = Math.max(...data.histogram.map((b) => b.hours), 0.0001)
  return (
    <div>
      <div className="flex h-32 items-end gap-[3px]">
        {data.histogram.map((bin, index) => {
          const height = (bin.hours / peak) * 100
          return (
            <div key={bin.lower} className="relative flex h-full flex-1 flex-col justify-end">
              <div
                title={`${percent(bin.lower)}–${percent(bin.upper)} English: ${hours(bin.hours)} h across ${bin.segments} clips`}
                className={`w-full rounded-t-[4px] ${HEAT_STEPS[Math.min(4, Math.floor((index / data.histogram.length) * 5))]} ${
                  bin.hours === 0 ? 'opacity-30' : ''
                }`}
                style={{ height: `${Math.max(bin.hours > 0 ? 3 : 1, height)}%` }}
              />
            </div>
          )
        })}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>0% English</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

/**
 * Every show as a dot on the density axis, with the band they all fall inside shaded.
 *
 * The shaded span is the finding: if it is narrow, every speaker recorded so far says roughly the
 * same thing about how much English gets mixed in, and the dependent variable of the study has
 * almost no variance left to explain.
 */
function ShowStrip({ data }: { data: RegisterData }) {
  if (data.show_means.length === 0) return null
  const min = data.show_mean_min ?? 0
  const max = data.show_mean_max ?? 1
  const axisMax = 0.6 // shows never approach 1.0; a full axis would squash every dot together
  const position = (value: number) => Math.min(100, (value / axisMax) * 100)

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Spread across shows
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {percent(min)} – {percent(max)}
          {data.show_mean_spread !== null ? ` · ${percent(data.show_mean_spread)} wide` : ''}
        </span>
      </div>

      <div className="relative h-14">
        {/* The band every show already sits inside. */}
        <div
          className="absolute inset-y-3 rounded-sm bg-muted"
          style={{ left: `${position(min)}%`, width: `${Math.max(1, position(max) - position(min))}%` }}
        />
        <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
        {data.show_means.map((show) => (
          <div
            key={show.show_id}
            title={`${show.show_id}: ${percent(show.mean_cmi, 1)} English over ${hours(show.hours)} h`}
            className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${position(show.mean_cmi)}%` }}
          >
            <div
              className="rounded-full bg-indigo-500 ring-2 ring-card"
              style={{
                width: `${Math.max(8, Math.min(20, 8 + show.hours * 1.5))}px`,
                height: `${Math.max(8, Math.min(20, 8 + show.hours * 1.5))}px`,
              }}
            />
          </div>
        ))}
        <div className="absolute inset-x-0 bottom-0 flex justify-between font-mono text-[10px] text-muted-foreground">
          <span>0%</span>
          <span>30%</span>
          <span>60%</span>
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground">
        One dot per show, sized by hours. A source outside the shaded band adds variance to the
        variable the corpus exists to measure; another one inside it adds hours.
      </p>
    </div>
  )
}

export function RegisterSection({
  data,
  minStratumHours,
}: {
  data: RegisterData
  minStratumHours: number
}) {
  const bandPeak = Math.max(...data.bands.map((b) => b.hours), 0.0001)

  return (
    <Panel>
      <PanelHeading
        icon={<RiTranslate2 className="size-4 text-primary" />}
        title="How much English is mixed in"
        note={
          <>
            Mean <span className="font-mono">{percent(data.mean, 1)}</span> over{' '}
            <span className="font-mono">{hours(data.measured_hours)} h</span> of scored audio
          </>
        }
      />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Audio by clip density
          </span>
          <div className="mt-2">
            <Histogram data={data} />
          </div>
        </div>

        <div className="space-y-3">
          <span className="font-heading text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            The three poles
          </span>
          <div className="space-y-2">
            {data.bands.map((band) => {
              const step = heatStep(band.hours, bandPeak)
              const thin = band.hours < minStratumHours
              return (
                <div key={band.name} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-2 text-xs">
                    <span className="font-medium capitalize">
                      {band.name}
                      <span className="ml-1 font-mono text-[10px] text-muted-foreground">
                        {percent(band.lower)}–{percent(band.upper)}
                      </span>
                    </span>
                    <span
                      className={`font-mono tabular-nums ${
                        thin ? 'text-rose-600 dark:text-rose-400' : ''
                      }`}
                    >
                      {hours(band.hours)} h
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-sm bg-muted">
                    <div
                      className={`h-full ${step < 0 ? '' : HEAT_STEPS[step]}`}
                      style={{ width: `${(band.hours / bandPeak) * 100}%` }}
                    />
                  </div>
                  <p className="text-[10px] leading-snug text-muted-foreground">
                    {band.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="mt-5 border-t pt-4">
        <ShowStrip data={data} />
      </div>
    </Panel>
  )
}
