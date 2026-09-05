import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { RiPlayFill } from '@remixicon/react'

import { Chip } from '@/components/Chip'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { Dispute } from '@/types'

interface DisputePopoverProps {
  dispute: Dispute
  /** The word span that was clicked; the panel is positioned under it. */
  anchor: HTMLElement
  /** Replace the disputed word in the transcript with this one. */
  onReplace: (replacement: string) => void
  /** Play just the disputed moment. */
  onPlayMoment: (start: number, end: number) => void
  onClose: () => void
}

/**
 * What the other systems heard at one disputed moment, and a one-click swap.
 *
 * This is the same 2:1 signal a ROVER composite would vote on, handed to the annotator instead
 * of resolved behind their back. The transcript they are editing stays one system's hypothesis
 * with a recorded human edit on top, so nothing invents a transcript no model produced.
 */
export function DisputePopover({
  dispute,
  anchor,
  onReplace,
  onPlayMoment,
  onClose,
}: DisputePopoverProps) {
  const panelRef = useRef<HTMLDivElement | null>(null)
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null)

  // Positioned against the viewport after layout, so the panel can be measured and kept on
  // screen for a word near the right edge.
  useLayoutEffect(() => {
    const rect = anchor.getBoundingClientRect()
    const width = panelRef.current?.offsetWidth ?? 240
    const left = Math.min(Math.max(8, rect.left), window.innerWidth - width - 8)
    setPosition({ top: rect.bottom + 6, left })
  }, [anchor, dispute])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    const onPointer = (e: PointerEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose()
    }
    // Capture, so Escape closes the panel rather than exiting the editor to triage.
    document.addEventListener('keydown', onKey, true)
    document.addEventListener('pointerdown', onPointer)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      document.removeEventListener('pointerdown', onPointer)
    }
  }, [onClose])

  // The same word from several systems is one option, not three rows.
  const options = new Map<string, string[]>()
  for (const alt of dispute.alternatives) {
    options.set(alt.word, [...(options.get(alt.word) ?? []), alt.system_id])
  }

  return (
    <div
      ref={panelRef}
      role="dialog"
      aria-label={`Alternatives for ${dispute.seed_word}`}
      style={{ top: position?.top ?? -9999, left: position?.left ?? -9999 }}
      className={cn(
        'fixed z-50 min-w-56 max-w-80 bg-popover p-1 text-popover-foreground shadow-md',
        'ring-1 ring-foreground/10',
        position ? 'opacity-100' : 'opacity-0',
      )}
    >
      <div className="flex items-center justify-between gap-2 px-2 py-1.5">
        <span className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Others heard
        </span>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Play this moment"
          onClick={() => onPlayMoment(dispute.start_time, dispute.end_time)}
        >
          <RiPlayFill />
        </Button>
      </div>

      <div className="px-2 pb-1.5 font-devanagari text-sm text-muted-foreground">
        seed: <span className="text-foreground">{dispute.seed_word}</span>
      </div>

      <div className="flex flex-col">
        {[...options].map(([word, systems]) => (
          <button
            key={word}
            type="button"
            onClick={() => onReplace(word)}
            className={cn(
              'flex items-center justify-between gap-3 px-2 py-1.5 text-left',
              'hover:bg-accent focus-visible:bg-accent focus-visible:outline-none',
            )}
          >
            <span className="font-devanagari text-base">{word}</span>
            <span className="flex shrink-0 flex-wrap justify-end gap-1">
              {systems.map((s) => (
                <Chip key={s} className="bg-foreground/10 text-foreground">
                  {s}
                </Chip>
              ))}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
