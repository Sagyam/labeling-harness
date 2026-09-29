import { Checkbox } from '@/components/ui/checkbox'
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field'
import { Input } from '@/components/ui/input'

interface ClipFieldProps {
  enabled: boolean
  /** As typed, so a half-edited box is not snapped back to a number. */
  minutes: string
  onEnabledChange: (enabled: boolean) => void
  onMinutesChange: (minutes: string) => void
  /** What the settings suggest for the chosen genre, shown so an edit can be judged against it. */
  suggestedMinutes: number
  genreLabel: string | null
  /** The source's length, when a probe knows it; an upload's is only known at stage 1. */
  sourceSeconds: number | null
}

/**
 * The opt-in clip to a recording's first N minutes (D103). Off by default: ticking it prefills
 * the length the settings suggest for the genre, which the annotator may change.
 */
export function ClipField({
  enabled,
  minutes,
  onEnabledChange,
  onMinutesChange,
  suggestedMinutes,
  genreLabel,
  sourceSeconds,
}: ClipFieldProps) {
  const kept = Number(minutes)
  const sourceMinutes = sourceSeconds === null ? null : sourceSeconds / 60
  const cutsNothing = enabled && sourceMinutes !== null && kept >= sourceMinutes

  return (
    <div className="flex flex-col gap-2 rounded-lg border bg-card p-3">
      <label className="flex cursor-pointer items-center gap-2 text-sm font-medium">
        <Checkbox
          id="ingest-clip"
          checked={enabled}
          onCheckedChange={(checked) => onEnabledChange(checked === true)}
        />
        Keep only the opening of a long recording
      </label>

      {enabled && (
        <Field className="pl-6">
          <FieldLabel htmlFor="ingest-clip-minutes">First N minutes</FieldLabel>
          {/* Wrapped: Field stretches its direct children to full width. */}
          <div>
            <Input
              id="ingest-clip-minutes"
              type="number"
              inputMode="numeric"
              min={1}
              step={1}
              className="w-28 font-mono"
              value={minutes}
              aria-invalid={!Number.isInteger(kept) || kept < 1}
              onChange={(e) => onMinutesChange(e.target.value)}
            />
          </div>
          <FieldDescription className="text-xs">
            Suggested for {genreLabel ?? 'this genre'}: {suggestedMinutes} min.
            {sourceMinutes !== null &&
              (cutsNothing
                ? ` The source is ${sourceMinutes.toFixed(0)} min, so nothing is cut.`
                : ` Keeps ${kept || 0} of ${sourceMinutes.toFixed(0)} min; the rest is never transcribed.`)}
          </FieldDescription>
        </Field>
      )}
    </div>
  )
}
