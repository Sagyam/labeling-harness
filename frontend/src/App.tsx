import { useCallback, useEffect, useState } from 'react'
import { Header } from './components/Header'
import { IngestModal } from './components/IngestModal'
import { EpisodeManagerModal } from './components/EpisodeManagerModal'
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal'
import { TriageView } from './components/TriageView'
import { EditorView } from './components/EditorView'
import { ToastContainer, ToastMessage } from './components/Toast'
import { api } from './services/api'
import { HealthResponse, QueueRow, StatsResponse, Task } from './types'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [activeQueue, setActiveQueue] = useState<string>('review')
  const [activeMode, setActiveMode] = useState<'triage' | 'editor'>('triage')

  // Ingestion & Episode modal state
  const [isIngestOpen, setIsIngestOpen] = useState<boolean>(false)
  const [isEpisodesOpen, setIsEpisodesOpen] = useState<boolean>(false)

  // Triage state
  const [queueRows, setQueueRows] = useState<QueueRow[]>([])
  const [focusedIndex, setFocusedIndex] = useState<number>(0)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

  // Editor state
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false)
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  // Toast feedback helper
  const showToast = useCallback(
    (text: string, type: 'success' | 'info' | 'warning' | 'error' = 'info') => {
      const id = String(Date.now() + Math.random())
      setToasts((prev) => [...prev, { id, text, type }])
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
      }, 3500)
    },
    []
  )

  const dismissToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }

  // Load stats & health
  const refreshStats = useCallback(async () => {
    try {
      const s = await api.getStats()
      setStats(s)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => setHealth({ status: 'unreachable' }))
    refreshStats()

    const interval = setInterval(refreshStats, 15000)
    return () => clearInterval(interval)
  }, [refreshStats])

  // Load queue rows
  const loadQueue = useCallback(
    async (qName: string) => {
      try {
        const rows = await api.getQueue({ queue: qName, limit: 100 })
        setQueueRows(rows)
        setSelectedIds(new Set())
        setFocusedIndex((prev) => (prev >= rows.length ? Math.max(0, rows.length - 1) : prev))
      } catch (err) {
        console.error('Failed to load queue:', err)
        showToast('Failed to load queue', 'error')
      }
    },
    [showToast]
  )

  useEffect(() => {
    loadQueue(activeQueue)
  }, [activeQueue, loadQueue])

  // Resume active task
  const handleResume = async () => {
    try {
      const task = await api.getNextTask({ queue: activeQueue })
      setCurrentTask(task)
      setActiveMode('editor')
      showToast(`Resumed ${task.segment.external_id}`, 'info')
    } catch (err: any) {
      if (err.status === 404) {
        showToast('No pending tasks in queue', 'warning')
      } else {
        showToast('Failed to resume task', 'error')
      }
    }
  }

  // Open task in editor
  const handleOpenEditor = async (taskId: number) => {
    try {
      const task = await api.getTask(taskId)
      setCurrentTask(task)
      setActiveMode('editor')
    } catch (err) {
      showToast(`Failed to load task #${taskId}`, 'error')
    }
  }

  // Triage: Accept row
  const handleAcceptRow = async (taskId: number, durationMs: number) => {
    const targetRow = queueRows.find((r) => r.task_id === taskId)
    try {
      await api.acceptTask(taskId, {
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      showToast(
        `Accepted ${targetRow?.segment_external_id || `#${taskId}`} (${(durationMs / 1000).toFixed(1)}s)`,
        'success'
      )
      setQueueRows((prev) => prev.filter((r) => r.task_id !== taskId))
      setSelectedIds((prev) => {
        const next = new Set(prev)
        next.delete(taskId)
        return next
      })
      refreshStats()
    } catch (err: any) {
      showToast(err.detail || 'Failed to accept task', 'error')
    }
  }

  // Triage: Flag row
  const handleFlagRow = async (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number
  ) => {
    const targetRow = queueRows.find((r) => r.task_id === taskId)
    try {
      await api.flagTask(taskId, {
        disposition,
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      showToast(
        `Flagged ${targetRow?.segment_external_id || `#${taskId}`} as ${disposition}`,
        'warning'
      )
      setQueueRows((prev) => prev.filter((r) => r.task_id !== taskId))
      setSelectedIds((prev) => {
        const next = new Set(prev)
        next.delete(taskId)
        return next
      })
      refreshStats()
    } catch (err: any) {
      showToast(err.detail || 'Failed to flag task', 'error')
    }
  }

  // Triage: Bulk accept
  const handleBulkAccept = async (taskIds: number[]) => {
    if (taskIds.length === 0) return
    try {
      await api.bulkAccept({ task_ids: taskIds })
      showToast(`Bulk accepted ${taskIds.length} segments`, 'success')
      const idSet = new Set(taskIds)
      setQueueRows((prev) => prev.filter((r) => !idSet.has(r.task_id)))
      setSelectedIds(new Set())
      refreshStats()
    } catch (err: any) {
      showToast(err.detail || 'Bulk accept failed', 'error')
    }
  }

  // Triage: Selection toggles
  const handleToggleSelect = (taskId: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(taskId)) next.delete(taskId)
      else next.add(taskId)
      return next
    })
  }

  const handleSelectAll = (selectAll: boolean) => {
    if (selectAll) {
      setSelectedIds(new Set(queueRows.map((r) => r.task_id)))
    } else {
      setSelectedIds(new Set())
    }
  }

  // Editor: Save and Next
  const handleSaveAndNext = async (taskId: number, finalText: string, durationMs: number) => {
    if (!currentTask) return
    const seedHyp = currentTask.segment.hypotheses.find(
      (h) => h.id === currentTask.seed_hypothesis_id
    )
    const isUnchanged = finalText.trim() === (seedHyp?.text || '').trim()

    try {
      if (isUnchanged) {
        await api.acceptTask(taskId, {
          duration_ms: durationMs,
          opened_at: new Date(Date.now() - durationMs).toISOString(),
        })
        showToast(`Accepted ${currentTask.segment.external_id} unchanged`, 'success')
      } else {
        await api.labelTask(taskId, {
          final_text: finalText,
          duration_ms: durationMs,
          opened_at: new Date(Date.now() - durationMs).toISOString(),
        })
        showToast(`Saved edits for ${currentTask.segment.external_id}`, 'success')
      }

      refreshStats()

      // Advance to next task in queue
      try {
        const next = await api.getNextTask({ queue: activeQueue })
        setCurrentTask(next)
      } catch (err: any) {
        if (err.status === 404) {
          showToast('Queue complete! Returning to triage', 'info')
          setActiveMode('triage')
          loadQueue(activeQueue)
        } else {
          showToast('Error loading next task', 'error')
        }
      }
    } catch (err: any) {
      showToast(err.detail || 'Save failed', 'error')
      throw err
    }
  }

  // Editor: Save and Stay
  const handleSaveAndStay = async (taskId: number, finalText: string, durationMs: number) => {
    if (!currentTask) return
    const seedHyp = currentTask.segment.hypotheses.find(
      (h) => h.id === currentTask.seed_hypothesis_id
    )
    const isUnchanged = finalText.trim() === (seedHyp?.text || '').trim()

    try {
      if (isUnchanged) {
        await api.acceptTask(taskId, {
          duration_ms: durationMs,
          opened_at: new Date(Date.now() - durationMs).toISOString(),
        })
        showToast(`Saved unchanged`, 'success')
      } else {
        await api.labelTask(taskId, {
          final_text: finalText,
          duration_ms: durationMs,
          opened_at: new Date(Date.now() - durationMs).toISOString(),
        })
        showToast(`Saved edits`, 'success')
      }
      refreshStats()
    } catch (err: any) {
      showToast(err.detail || 'Save failed', 'error')
      throw err
    }
  }

  // Editor: Flag
  const handleEditorFlag = async (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number
  ) => {
    try {
      await api.flagTask(taskId, {
        disposition,
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      showToast(`Flagged as ${disposition}`, 'warning')
      refreshStats()

      // Advance
      try {
        const next = await api.getNextTask({ queue: activeQueue })
        setCurrentTask(next)
      } catch {
        setActiveMode('triage')
        loadQueue(activeQueue)
      }
    } catch (err: any) {
      showToast(err.detail || 'Flag failed', 'error')
    }
  }

  // Editor: Skip
  const handleEditorSkip = async (taskId: number, durationMs: number) => {
    try {
      await api.skipTask(taskId, {
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      showToast(`Deferred task`, 'info')
      refreshStats()

      // Advance
      try {
        const next = await api.getNextTask({ queue: activeQueue })
        setCurrentTask(next)
      } catch {
        setActiveMode('triage')
        loadQueue(activeQueue)
      }
    } catch (err: any) {
      showToast(err.detail || 'Skip failed', 'error')
    }
  }

  // Global Keybindings (? for help)
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (targetTag === 'input' || targetTag === 'textarea') return

      if (e.key === '?') {
        e.preventDefault()
        setIsHelpOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => window.removeEventListener('keydown', handleGlobalKeyDown)
  }, [])

  return (
    <div className="app-container">
      <Header
        stats={stats}
        activeQueue={activeQueue}
        onChangeQueue={(q) => {
          setActiveQueue(q)
          setActiveMode('triage')
        }}
        activeMode={activeMode}
        onChangeMode={(m) => {
          if (m === 'editor' && !currentTask) {
            handleResume()
          } else {
            setActiveMode(m)
          }
        }}
        onResume={handleResume}
        onOpenHelp={() => setIsHelpOpen(true)}
        onOpenIngest={() => setIsIngestOpen(true)}
        onOpenEpisodes={() => setIsEpisodesOpen(true)}
        health={health}
      />

      {activeMode === 'triage' || !currentTask ? (
        <TriageView
          rows={queueRows}
          focusedIndex={focusedIndex}
          onSetFocusedIndex={setFocusedIndex}
          selectedIds={selectedIds}
          onToggleSelect={handleToggleSelect}
          onSelectAll={handleSelectAll}
          onAcceptRow={handleAcceptRow}
          onOpenEditor={handleOpenEditor}
          onFlagRow={handleFlagRow}
          onBulkAccept={handleBulkAccept}
        />
      ) : (
        <EditorView
          task={currentTask}
          onSaveAndNext={handleSaveAndNext}
          onSaveAndStay={handleSaveAndStay}
          onFlag={handleEditorFlag}
          onSkip={handleEditorSkip}
          onExitToTriage={() => {
            setActiveMode('triage')
            loadQueue(activeQueue)
          }}
          onToast={showToast}
        />
      )}

      {/* Episode Management Modal */}
      <EpisodeManagerModal
        isOpen={isEpisodesOpen}
        onClose={() => setIsEpisodesOpen(false)}
        onDataChanged={() => {
          refreshStats()
          loadQueue(activeQueue)
        }}
      />

      {/* Ingest Episode Modal */}
      <IngestModal
        isOpen={isIngestOpen}
        onClose={() => setIsIngestOpen(false)}
        onComplete={(episodeId) => {
          refreshStats()
          loadQueue(activeQueue)
          setActiveMode('triage')
          showToast(`Episode '${episodeId}' ingested! Priority queue updated.`, 'success')
        }}
        onToast={showToast}
      />

      {/* Shortcuts Help Modal */}
      <KeyboardShortcutsModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />

      {/* Floating Notifications */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}
