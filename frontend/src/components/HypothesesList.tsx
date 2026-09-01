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
import { cn } from '@/lib/utils'
import type { Hypothesis } from '@/types'

interface HypothesesListProps {
  hypotheses: Hypothesis[]
  seedHypothesisId: number | null
  onSelectHypothesis: (text: string) => void
}

export function HypothesesList({
  hypotheses,
  seedHypothesisId,
  onSelectHypothesis,
}: HypothesesListProps) {
  return (
    <div className="flex min-h-0 flex-col bg-card ring-1 ring-foreground/5">
      <div className="flex h-10 shrink-0 items-center justify-between gap-2 border-b px-3">
        <span className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Upstream ASR hypotheses ({hypotheses.length})
        </span>
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Kbd>Alt + 1…5</Kbd> to load
        </span>
      </div>

      <ItemGroup className="scrollbar-thin min-h-0 flex-1 overflow-y-auto divide-y">
        {hypotheses.map((hyp, index) => {
          const isSeed = hyp.id === seedHypothesisId
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
                </ItemTitle>
                <ItemDescription className="font-devanagari text-sm leading-relaxed text-foreground line-clamp-none">
                  {hyp.text}
                </ItemDescription>
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
