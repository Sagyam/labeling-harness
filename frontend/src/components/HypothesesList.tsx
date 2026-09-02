import { Chip } from '@/components/Chip'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemTitle,
} from '@/components/ui/item'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import type { Hypothesis } from '@/types'

interface HypothesesListProps {
  hypotheses: Hypothesis[]
  seedHypothesisId: number | null
  onSelectHypothesis: (text: string) => void
  onSeek?: (time: number) => void
}

export function HypothesesList({
  hypotheses,
  seedHypothesisId,
  onSelectHypothesis,
  onSeek,
}: HypothesesListProps) {
  return (
    <div className="flex min-h-0 flex-col bg-card ring-1 ring-foreground/5">
      <div className="flex h-10 shrink-0 items-center justify-between gap-2 border-b px-3">
        <span className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Upstream ASR hypotheses ({hypotheses.length})
        </span>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="hidden items-center gap-1.5 sm:inline-flex">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning" />
            <span>Click word to seek</span>
          </span>
          <span className="flex items-center gap-1.5">
            <Kbd>Alt + 1…5</Kbd> to load
          </span>
        </div>
      </div>

      <ItemGroup className="scrollbar-thin min-h-0 flex-1 divide-y overflow-y-auto">
        {hypotheses.map((hyp, index) => {
          const isSeed = hyp.id === seedHypothesisId
          const hasWords = Boolean(hyp.words && hyp.words.length > 0)
          const lowConfCount =
            hyp.words?.filter((w) => w.confidence !== null && w.confidence < 0.7).length ?? 0

          return (
            <Item
              key={hyp.id}
              className={cn('items-start gap-4 px-3 py-2.5', isSeed && 'bg-accent/50')}
            >
              <ItemContent className="gap-1.5">
                <ItemTitle className="flex w-full flex-wrap items-center gap-1.5 line-clamp-none">
                  <Chip className="bg-foreground/10 text-foreground">{hyp.system_id}</Chip>
                  {hyp.model_id && <Chip>{hyp.model_id}</Chip>}
                  {isSeed && <Chip className="bg-info/15 text-info">seed</Chip>}
                  {hyp.avg_logprob !== null && (
                    <Chip title="Average log probability">
                      logprob {hyp.avg_logprob.toFixed(2)}
                    </Chip>
                  )}
                  <Chip>{hyp.word_count} words</Chip>
                  {lowConfCount > 0 && (
                    <Chip className="bg-warning/15 text-warning" title={`${lowConfCount} word(s) with confidence < 70%`}>
                      {lowConfCount} low-conf
                    </Chip>
                  )}
                </ItemTitle>

                {hasWords ? (
                  <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-1 font-devanagari text-sm leading-relaxed text-foreground">
                    {hyp.words!.map((w, wIdx) => {
                      const isLowConf = w.confidence !== null && w.confidence < 0.7
                      const hasTiming = w.start_time !== null

                      return (
                        <Tooltip key={wIdx}>
                          <TooltipTrigger asChild>
                            <span
                              role={hasTiming ? 'button' : undefined}
                              tabIndex={hasTiming ? 0 : undefined}
                              onClick={() => hasTiming && onSeek?.(w.start_time!)}
                              onKeyDown={(e) => {
                                if (hasTiming && (e.key === 'Enter' || e.key === ' ')) {
                                  e.preventDefault()
                                  onSeek?.(w.start_time!)
                                }
                              }}
                              className={cn(
                                'inline-block rounded px-1 py-0.5 transition-colors select-text',
                                hasTiming &&
                                  'cursor-pointer hover:bg-accent hover:text-accent-foreground',
                                isLowConf &&
                                  'border-b-2 border-warning/70 bg-warning/15 font-medium text-warning-foreground',
                              )}
                            >
                              {w.word}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent className="font-mono text-xs">
                            {hasTiming && (
                              <div>
                                {w.start_time?.toFixed(2)}s – {w.end_time?.toFixed(2)}s
                              </div>
                            )}
                            {w.confidence !== null && (
                              <div>
                                confidence: {(w.confidence * 100).toFixed(0)}%
                                {isLowConf && ' (low)'}
                              </div>
                            )}
                            {hasTiming && (
                              <div className="mt-0.5 text-[10px] text-muted-foreground">
                                Click to play from here
                              </div>
                            )}
                          </TooltipContent>
                        </Tooltip>
                      )
                    })}
                  </div>
                ) : (
                  <ItemDescription className="font-devanagari text-sm leading-relaxed text-foreground line-clamp-none">
                    {hyp.text}
                  </ItemDescription>
                )}
              </ItemContent>

              <ItemActions>
                <Button variant="outline" size="xs" onClick={() => onSelectHypothesis(hyp.text)}>
                  Load
                  <Kbd className="ml-1">Alt+{index + 1}</Kbd>
                </Button>
              </ItemActions>
            </Item>
          )
        })}
      </ItemGroup>
    </div>
  )
}
