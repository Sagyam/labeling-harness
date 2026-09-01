import React from 'react'

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

export const DiffViewer: React.FC<DiffViewerProps> = ({ seedText, currentText }) => {
  const isIdentical = seedText.trim() === currentText.trim()
  const diffs = computeWordDiff(seedText, currentText)

  return (
    <div className="diff-viewer">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
          Diff vs Seed Hypothesis
        </span>
        <span
          style={{
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            color: isIdentical ? 'var(--emerald)' : 'var(--amber)',
          }}
        >
          {isIdentical ? '● Unchanged' : '● Edited'}
        </span>
      </div>

      <div style={{ wordBreak: 'break-word', lineHeight: 1.7 }}>
        {diffs.map((part, index) => {
          if (part.type === 'removed') {
            return (
              <span key={index} className="diff-del">
                {part.value}
              </span>
            )
          }
          if (part.type === 'added') {
            return (
              <span key={index} className="diff-ins">
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
