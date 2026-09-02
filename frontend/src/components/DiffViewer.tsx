import { cn } from '@/lib/utils'

interface DiffViewerProps {
  seedText: string
  currentText: string
}

interface DiffSegment {
  type: 'same' | 'added' | 'removed'
  value: string
}

// Simple word-level diff algorithm
function computeWordDiff(original: string, modified: string): DiffSegment[] {
  const origWords = original.trim().split(/\s+/).filter(Boolean)
  const modWords = modified.trim().split(/\s+/).filter(Boolean)

  if (original.trim() === modified.trim()) {
    return [{ type: 'same', value: modified }]
  }

  // Find longest common subsequence (LCS) matrix
  const m = origWords.length
  const n = modWords.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (origWords[i - 1] === modWords[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  // Backtrack to build diff segments
  let i = m
  let j = n
  const reversedDiff: DiffSegment[] = []

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && origWords[i - 1] === modWords[j - 1]) {
      reversedDiff.push({ type: 'same', value: origWords[i - 1] })
      i--
      j--
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      reversedDiff.push({ type: 'added', value: modWords[j - 1] })
      j--
    } else if (i > 0 && (j === 0 || dp[i][j - 1] < dp[i - 1][j])) {
      reversedDiff.push({ type: 'removed', value: origWords[i - 1] })
      i--
    }
  }

  return reversedDiff.reverse()
}

export function DiffViewer({ seedText, currentText }: DiffViewerProps) {
  const isIdentical = seedText.trim() === currentText.trim()
  const diffs = computeWordDiff(seedText, currentText)

  return (
    <div className="flex min-h-0 flex-col bg-card ring-1 ring-foreground/5">
      <div className="flex h-10 shrink-0 items-center justify-between gap-2 border-b px-3">
        <span className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Diff vs seed
        </span>
        <span
          className={cn(
            'font-mono text-[11px]',
            isIdentical ? 'text-muted-foreground' : 'text-warning',
          )}
        >
          {isIdentical ? '● unchanged' : '● edited'}
        </span>
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-3 font-devanagari leading-8 break-words">
        {diffs.map((part, index) => {
          if (part.type === 'removed') {
            return (
              <span key={index} className="mx-0.5 bg-destructive/15 px-1 text-destructive line-through">
                {part.value}
              </span>
            )
          }
          if (part.type === 'added') {
            return (
              <span key={index} className="mx-0.5 bg-success/15 px-1 text-success">
                {part.value}
              </span>
            )
          }
          return <span key={index}> {part.value} </span>
        })}
      </div>
    </div>
  )
}
