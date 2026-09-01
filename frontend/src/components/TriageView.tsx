import React, { useEffect, useRef, useState } from 'react'
import { resolveUrl } from '../services/api'
import { QueueRow } from '../types'

interface TriageViewProps {
  rows: QueueRow[]
  focusedIndex: number
  onSetFocusedIndex: (index: number) => void
  selectedIds: Set<number>
  onToggleSelect: (taskId: number) => void
  onSelectAll: (all: boolean) => void
  onAcceptRow: (taskId: number, durationMs: number) => Promise<void>
  onOpenEditor: (taskId: number) => void
  onFlagRow: (taskId: number, disposition: 'unusable_audio' | 'uncertain', durationMs: number) => Promise<void>
  onBulkAccept: (taskIds: number[]) => Promise<void>
}

export const TriageView: React.FC<TriageViewProps> = ({
  rows,
  focusedIndex,
  onSetFocusedIndex,
  selectedIds,
  onToggleSelect,
  onSelectAll,
  onAcceptRow,
  onOpenEditor,
  onFlagRow,
  onBulkAccept,
}) => {
  const [playingTaskId, setPlayingTaskId] = useState<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rowRefs = useRef<(HTMLTableRowElement | null)[]>([])
  const focusedRowOpenedAtRef = useRef<number>(Date.now())

  // Scroll focused row into view smoothly
  useEffect(() => {
    focusedRowOpenedAtRef.current = Date.now()
    const el = rowRefs.current[focusedIndex]
    if (el) {
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [focusedIndex])

  const getFocusedDurationMs = () => Math.max(0, Date.now() - focusedRowOpenedAtRef.current)

  // Audio Playback
  const togglePlay = (row: QueueRow) => {
    if (playingTaskId === row.task_id) {
      if (audioRef.current) {
        audioRef.current.pause()
        setPlayingTaskId(null)
      }
    } else {
      if (audioRef.current) {
        audioRef.current.pause()
      }
      const audio = new Audio(resolveUrl(row.audio_url))
      audioRef.current = audio
      setPlayingTaskId(row.task_id)
      audio.play().catch((err) => console.error('Audio play error:', err))
      audio.onended = () => setPlayingTaskId(null)
      audio.onerror = () => setPlayingTaskId(null)
    }
  }

  // Keyboard navigation for Triage mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (targetTag === 'input' || targetTag === 'textarea') return

      if (rows.length === 0) return
      const focusedRow = rows[focusedIndex]

      // j / ArrowDown: Move focus down
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault()
        onSetFocusedIndex(Math.min(rows.length - 1, focusedIndex + 1))
        return
      }

      // k / ArrowUp: Move focus up
      if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault()
        onSetFocusedIndex(Math.max(0, focusedIndex - 1))
        return
      }

      // Space: Toggle play/pause focused row
      if (e.code === 'Space') {
        e.preventDefault()
        if (focusedRow) togglePlay(focusedRow)
        return
      }

      // Shift+Enter: Bulk accept selected rows
      if (e.shiftKey && e.key === 'Enter') {
        e.preventDefault()
        if (selectedIds.size > 0) {
          onBulkAccept(Array.from(selectedIds))
        } else if (focusedRow) {
          onBulkAccept([focusedRow.task_id])
        }
        return
      }

      // Enter: Accept focused row unchanged
      if (!e.shiftKey && !e.ctrlKey && e.key === 'Enter') {
        e.preventDefault()
        if (focusedRow) {
          onAcceptRow(focusedRow.task_id, getFocusedDurationMs())
        }
        return
      }

      // e: Open editor
      if (e.key === 'e' || e.key === 'E') {
        e.preventDefault()
        if (focusedRow) {
          onOpenEditor(focusedRow.task_id)
        }
        return
      }

      // f: Flag unusable audio
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault()
        if (focusedRow) {
          onFlagRow(focusedRow.task_id, 'unusable_audio', getFocusedDurationMs())
        }
        return
      }

      // u: Mark uncertain
      if (e.key === 'u' || e.key === 'U') {
        e.preventDefault()
        if (focusedRow) {
          onFlagRow(focusedRow.task_id, 'uncertain', getFocusedDurationMs())
        }
        return
      }

      // x: Toggle checkbox
      if (e.key === 'x' || e.key === 'X') {
        e.preventDefault()
        if (focusedRow) {
          onToggleSelect(focusedRow.task_id)
        }
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [rows, focusedIndex, selectedIds, playingTaskId])

  // Stop audio when unmounting
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
      }
    }
  }, [])

  const allSelected = rows.length > 0 && rows.every((r) => selectedIds.has(r.task_id))

  return (
    <div className="triage-container">
      {/* Triage Toolbar */}
      <div className="triage-toolbar">
        <div className="triage-actions-left">
          <button
            className="btn-bulk-accept"
            disabled={selectedIds.size === 0}
            onClick={() => onBulkAccept(Array.from(selectedIds))}
            title="Accept all checked rows (Shift+Enter)"
          >
            <span>Accept Selected ({selectedIds.size})</span>
            <kbd style={{ fontSize: '0.65rem' }}>Shift+↵</kbd>
          </button>
        </div>

        <div className="triage-actions-right">
          <span>{rows.length} queued segments</span>
        </div>
      </div>

      {/* Table Wrapper */}
      <div className="triage-table-wrapper">
        <table className="triage-table">
          <thead>
            <tr>
              <th className="cell-select">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={(e) => onSelectAll(e.target.checked)}
                  title="Select all"
                />
              </th>
              <th className="cell-priority">Priority</th>
              <th className="cell-audio">Audio</th>
              <th className="cell-segment-id">Segment</th>
              <th>Seed Hypothesis</th>
              <th className="cell-flags">Flags & Reason</th>
              <th className="cell-actions">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const isFocused = index === focusedIndex
              const isPlaying = playingTaskId === row.task_id
              const isChecked = selectedIds.has(row.task_id)

              // Priority style
              const score = row.priority_score
              const priClass =
                score >= 0.45 ? 'priority-high' : score >= 0.25 ? 'priority-med' : 'priority-low'

              return (
                <tr
                  key={row.task_id}
                  ref={(el) => {
                    rowRefs.current[index] = el
                  }}
                  className={`triage-row ${isFocused ? 'focused' : ''} ${isPlaying ? 'playing' : ''}`}
                  onClick={() => onSetFocusedIndex(index)}
                  onDoubleClick={() => onOpenEditor(row.task_id)}
                >
                  <td className="cell-select" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => onToggleSelect(row.task_id)}
                    />
                  </td>

                  <td className="cell-priority">
                    <span
                      className={`priority-chip ${priClass}`}
                      title={`Score: ${score.toFixed(3)}\nDisagreement: ${
                        row.reason?.components?.word_disagreement_rate?.toFixed(2) ?? '0'
                      }\nCode-switch: ${
                        row.reason?.components?.code_switch_density?.toFixed(2) ?? '0'
                      }`}
                    >
                      {score.toFixed(2)}
                    </span>
                  </td>

                  <td className="cell-audio" onClick={(e) => e.stopPropagation()}>
                    <button
                      className={`btn-play-mini ${isPlaying ? 'is-playing' : ''}`}
                      onClick={() => togglePlay(row)}
                      title="Play/Pause (Space)"
                    >
                      <span>{isPlaying ? '⏸' : '▶'}</span>
                      <span>{row.duration_seconds.toFixed(1)}s</span>
                    </button>
                  </td>

                  <td className="cell-segment-id">
                    <span title={row.segment_external_id}>{row.segment_external_id}</span>
                  </td>

                  <td className="cell-text">
                    <span title={row.seed_text || ''}>{row.seed_text || '—'}</span>
                  </td>

                  <td className="cell-flags">
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
                      {row.seed_system_id && (
                        <span className="system-chip">{row.seed_system_id}</span>
                      )}
                      {row.flags.map((flag) => (
                        <span key={flag} className="flag-chip" title={flag}>
                          {flag}
                        </span>
                      ))}
                    </div>
                  </td>

                  <td className="cell-actions" onClick={(e) => e.stopPropagation()}>
                    <div className="action-btn-group">
                      <button
                        className="btn-action-sm btn-accept-sm"
                        onClick={() => onAcceptRow(row.task_id, getFocusedDurationMs())}
                        title="Accept unchanged (Enter)"
                      >
                        Accept
                      </button>
                      <button
                        className="btn-action-sm btn-edit-sm"
                        onClick={() => onOpenEditor(row.task_id)}
                        title="Open editor (e)"
                      >
                        Edit
                      </button>
                      <button
                        className="btn-action-sm btn-flag-sm"
                        onClick={() => onFlagRow(row.task_id, 'unusable_audio', getFocusedDurationMs())}
                        title="Flag unusable audio (f)"
                      >
                        Flag
                      </button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Footer Quick Shortcut Bar */}
      <footer className="footer-shortcut-bar">
        <div className="footer-keys">
          <div className="footer-key-item">
            <kbd>j</kbd>/<kbd>k</kbd> <span>Navigate</span>
          </div>
          <div className="footer-key-item">
            <kbd>Space</kbd> <span>Play</span>
          </div>
          <div className="footer-key-item">
            <kbd>Enter</kbd> <span>Accept</span>
          </div>
          <div className="footer-key-item">
            <kbd>e</kbd> <span>Editor</span>
          </div>
          <div className="footer-key-item">
            <kbd>f</kbd> <span>Unusable</span>
          </div>
          <div className="footer-key-item">
            <kbd>u</kbd> <span>Uncertain</span>
          </div>
          <div className="footer-key-item">
            <kbd>x</kbd> <span>Select</span>
          </div>
          <div className="footer-key-item">
            <kbd>Shift+Enter</kbd> <span>Bulk Accept</span>
          </div>
        </div>

        <div>
          <span>Press <kbd>?</kbd> for all shortcuts</span>
        </div>
      </footer>
    </div>
  )
}
