import React from 'react'
import { Hypothesis } from '../types'

interface HypothesesListProps {
  hypotheses: Hypothesis[]
  seedHypothesisId: number | null
  onSelectHypothesis: (text: string) => void
}

export const HypothesesList: React.FC<HypothesesListProps> = ({
  hypotheses,
  seedHypothesisId,
  onSelectHypothesis,
}) => {
  return (
    <div className="hypotheses-panel">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
          All Upstream ASR Hypotheses ({hypotheses.length})
        </span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-faint)' }}>
          Press <kbd>Alt+1..5</kbd> to load directly
        </span>
      </div>

      <div className="hypotheses-list">
        {hypotheses.map((hyp, index) => {
          const isSeed = hyp.id === seedHypothesisId
          return (
            <div
              key={hyp.id}
              className={`hypothesis-card ${isSeed ? 'is-seed' : ''}`}
            >
              <div style={{ flex: 1 }}>
                <div className="hyp-header">
                  <span className="system-chip">{hyp.system_id}</span>
                  {hyp.model_id && (
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-faint)' }}>
                      ({hyp.model_id})
                    </span>
                  )}
                  {isSeed && (
                    <span
                      style={{
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        color: 'var(--primary-light)',
                        background: 'rgba(99, 102, 241, 0.2)',
                        padding: '0.1rem 0.4rem',
                        borderRadius: '3px',
                      }}
                    >
                      SEED
                    </span>
                  )}
                  {hyp.avg_logprob !== null && (
                    <span
                      style={{
                        fontSize: '0.68rem',
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--text-faint)',
                      }}
                      title="Average log probability"
                    >
                      logprob: {hyp.avg_logprob.toFixed(2)}
                    </span>
                  )}
                  <span
                    style={{
                      fontSize: '0.68rem',
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--text-faint)',
                    }}
                  >
                    words: {hyp.word_count}
                  </span>
                </div>

                <div className="hyp-text">{hyp.text}</div>
              </div>

              <div>
                <button
                  className="hyp-load-btn"
                  onClick={() => onSelectHypothesis(hyp.text)}
                  title={`Load into editor (Alt+${index + 1})`}
                >
                  <kbd style={{ fontSize: '0.65rem' }}>Alt+{index + 1}</kbd>
                  <span>Load</span>
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
