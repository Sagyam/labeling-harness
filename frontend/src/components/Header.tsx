import React from 'react'
import { HealthResponse, StatsResponse } from '../types'

interface HeaderProps {
  stats: StatsResponse | null
  activeQueue: string
  onChangeQueue: (queue: string) => void
  activeMode: 'triage' | 'editor'
  onChangeMode: (mode: 'triage' | 'editor') => void
  onResume: () => void
  onOpenHelp: () => void
  health: HealthResponse | null
}

export const Header: React.FC<HeaderProps> = ({
  stats,
  activeQueue,
  onChangeQueue,
  activeMode,
  onChangeMode,
  onResume,
  onOpenHelp,
  health,
}) => {
  const isHealthy = health?.status === 'ok'
  const queueStats = (stats?.queues as Record<string, number>) || {}
  const throughput = stats?.throughput
  const sessionStats = stats?.session

  // Format projected time
  const formatProjectedTime = (seconds?: number | null) => {
    if (seconds === null || seconds === undefined) return '--'
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    const hours = Math.floor(seconds / 3600)
    const mins = Math.round((seconds % 3600) / 60)
    return `${hours}h ${mins}m`
  }

  // Format accept rate
  const acceptRateFormatted =
    stats?.accept_rate !== null && stats?.accept_rate !== undefined
      ? `${(stats.accept_rate * 100).toFixed(1)}%`
      : '--'

  const medianSecFormatted =
    sessionStats?.median_seconds_per_segment !== null &&
    sessionStats?.median_seconds_per_segment !== undefined
      ? `${sessionStats.median_seconds_per_segment.toFixed(1)}s`
      : throughput?.median_seconds_per_segment !== null &&
        throughput?.median_seconds_per_segment !== undefined
      ? `${throughput.median_seconds_per_segment.toFixed(1)}s`
      : '--'

  return (
    <header className="app-header">
      <div className="header-left">
        <div className="brand">
          <span
            className="brand-dot"
            style={{
              backgroundColor: isHealthy ? 'var(--emerald)' : 'var(--rose)',
              boxShadow: `0 0 10px ${isHealthy ? 'var(--emerald)' : 'var(--rose)'}`,
            }}
            title={isHealthy ? 'Connected to backend' : 'Backend unreachable'}
          />
          <span>Nepanglish</span>
          <span className="brand-badge">Harness</span>
        </div>

        {/* Queue Selector Tabs */}
        <div className="queue-tabs" role="tablist">
          {['review', 'audit', 'error'].map((q) => (
            <button
              key={q}
              className={`queue-tab ${activeQueue === q ? 'active' : ''}`}
              onClick={() => onChangeQueue(q)}
              role="tab"
              aria-selected={activeQueue === q}
            >
              <span style={{ textTransform: 'capitalize' }}>{q}</span>
              <span className="queue-count">{queueStats[q] ?? 0}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Progress & Throughput Display (Phase 6.3) */}
      <div className="header-stats">
        <div className="stat-item" title="Completed segments out of total">
          <span className="stat-label">Progress</span>
          <span className="stat-value">
            {stats?.tasks?.done ?? 0} / {stats?.tasks?.total ?? 0}
          </span>
        </div>

        <div className="stat-item" title="Fraction of labels accepted unchanged">
          <span className="stat-label">Accept Rate</span>
          <span className="stat-value highlight-emerald">{acceptRateFormatted}</span>
        </div>

        <div className="stat-item" title="Completed during this session">
          <span className="stat-label">Session Done</span>
          <span className="stat-value highlight-cyan">{sessionStats?.completed ?? 0}</span>
        </div>

        <div className="stat-item" title="Median seconds spent per segment">
          <span className="stat-label">Median / Seg</span>
          <span className="stat-value highlight-amber">{medianSecFormatted}</span>
        </div>

        <div className="stat-item" title="Projected finish time for remaining queue">
          <span className="stat-label">Est. Remaining</span>
          <span className="stat-value">
            {formatProjectedTime(throughput?.projected_seconds_to_finish)}
          </span>
        </div>
      </div>

      {/* Header Right Controls */}
      <div className="header-right">
        <button
          className="btn-resume"
          onClick={onResume}
          title="Jump directly to active / highest-priority pending segment"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
          </svg>
          Resume
        </button>

        <div className="mode-toggle-group">
          <button
            className={`mode-btn ${activeMode === 'triage' ? 'active' : ''}`}
            onClick={() => onChangeMode('triage')}
            title="Triage mode (dense keyboard list)"
          >
            Triage
          </button>
          <button
            className={`mode-btn ${activeMode === 'editor' ? 'active' : ''}`}
            onClick={() => onChangeMode('editor')}
            title="Editor mode (deep correction)"
          >
            Editor
          </button>
        </div>

        <button className="btn-icon" onClick={onOpenHelp} title="Keyboard shortcuts (?)">
          <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>?</span>
        </button>
      </div>
    </header>
  )
}
