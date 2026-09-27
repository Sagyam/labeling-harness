import { RiAddLine, RiDeleteBin6Line } from '@remixicon/react'

import { Button } from '@/components/ui/button'

import { NativeSelect } from './NativeSelect'
import {
  AGE_BRACKET_OPTIONS,
  GENDER_OPTIONS,
  MAX_SPEAKERS,
  ROLE_OPTIONS,
  type SpeakerDraft,
  emptySpeaker,
} from './speakers'

interface SpeakerRowsProps {
  speakers: SpeakerDraft[]
  onChange: (speakers: SpeakerDraft[]) => void
  /** Rows that cannot be removed: the ingest form always declares at least one. */
  minRows: number
  /** A line under the heading, e.g. what the diarizer does with the count. */
  hint?: string
}

/** Declared speakers: role, gender and age bracket per person (D56, D58). */
export function SpeakerRows({ speakers, onChange, minRows, hint }: SpeakerRowsProps) {
  const update = (index: number, patch: Partial<SpeakerDraft>) =>
    onChange(speakers.map((s, i) => (i === index ? { ...s, ...patch } : s)))

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-foreground">
          Speakers ({speakers.length} of {MAX_SPEAKERS})
          {hint && <span className="ml-1.5 font-normal text-muted-foreground">· {hint}</span>}
        </span>
        <Button
          type="button"
          variant="outline"
          size="xs"
          disabled={speakers.length >= MAX_SPEAKERS}
          onClick={() => onChange([...speakers, emptySpeaker(speakers.length)])}
        >
          <RiAddLine className="size-3.5" />
          Add speaker
        </Button>
      </div>

      {speakers.map((speaker, index) => (
        <div key={index} className="flex items-center gap-2 rounded border bg-muted/20 p-2">
          <span className="w-5 shrink-0 text-center font-mono text-xs text-muted-foreground">
            {index}
          </span>
          <div className="grid flex-1 gap-2 sm:grid-cols-3">
            <NativeSelect
              aria-label={`Speaker ${index} role`}
              value={speaker.role}
              onChange={(e) => update(index, { role: e.target.value })}
            >
              {ROLE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </NativeSelect>
            <NativeSelect
              aria-label={`Speaker ${index} gender`}
              value={speaker.gender}
              onChange={(e) => update(index, { gender: e.target.value })}
            >
              <option value="">Gender (optional)</option>
              {GENDER_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </NativeSelect>
            <NativeSelect
              aria-label={`Speaker ${index} age bracket`}
              value={speaker.ageBracket}
              onChange={(e) => update(index, { ageBracket: e.target.value })}
            >
              <option value="">Age (optional)</option>
              {AGE_BRACKET_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </NativeSelect>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="text-muted-foreground hover:text-destructive"
            disabled={speakers.length <= minRows}
            onClick={() => onChange(speakers.filter((_, i) => i !== index))}
            aria-label={`Remove speaker ${index}`}
          >
            <RiDeleteBin6Line className="size-3.5" />
          </Button>
        </div>
      ))}
    </div>
  )
}
