/**
 * One line under the header saying what the current page answers, with a way into the glossary:
 * a newcomer otherwise meets eight pages and their jargon with nothing to start from. The Models
 * page states its question per tab instead, and the editor is a workspace, so neither has one.
 */

import { RiBookOpenLine } from '@remixicon/react'

import type { HeaderMode } from '@/components/Header'

const INTRO: Partial<Record<HeaderMode, string>> = {
  triage:
    'Clips waiting for a decision. Accept a seed transcript that is right, open the rest in the editor, flag audio that is unusable.',
  episodes:
    'Every recording ingested: its clips, its genre, topic and show. Open one to label its clips or to correct what it is.',
  analytics:
    'What the labelled corpus covers, one pot at a time: who speaks, about what, in what conditions, and what is missing or overdone.',
  voices:
    'Every anonymous voice, followed across episodes: hear it, give it a gender and an age by ear, and see where it speaks.',
  export: 'Build the dataset models are trained and scored on (train, val, and gold as the test set), and check what it holds.',
  costs: 'What every paid transcription call cost, by vendor and model, from the log every inference call writes.',
  ingest:
    'Add a recording, from a YouTube link or an audio file: it is cut into clips, transcribed by three recognisers and fused into a seed.',
}

export function PageIntro({ mode, onOpenGlossary }: { mode: HeaderMode; onOpenGlossary: () => void }) {
  const text = INTRO[mode]
  if (!text) return null
  return (
    <div className="flex shrink-0 items-center justify-between gap-3 border-b bg-muted/20 px-4 py-1.5 text-xs text-muted-foreground sm:px-6">
      <span>{text}</span>
      <button
        type="button"
        onClick={onOpenGlossary}
        className="flex shrink-0 items-center gap-1 hover:text-foreground"
        title="What the harness's terms mean"
      >
        <RiBookOpenLine className="size-3.5" /> Glossary
      </button>
    </div>
  )
}
