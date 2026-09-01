import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'

import { EditorView } from '@/components/EditorView'
import { EpisodeManagerModal } from '@/components/EpisodeManagerModal'
import { Header } from '@/components/Header'
import { IngestModal } from '@/components/IngestModal'
import { KeyboardShortcutsModal } from '@/components/KeyboardShortcutsModal'
import { TriageView } from '@/components/TriageView'
import { api } from '@/services/api'
import type { HealthResponse, QueueRow, StatsResponse, Task } from '@/types'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [activeQueue, setActiveQueue] = useState<string>('review')
  const [activeMode, setActiveMode] = useState<'triage' | 'editor'>('triage')

  // Ingestion & episode modal state
  const [isIngestOpen, setIsIngestOpen] = useState<boolean>(false)
  const [isEpisodesOpen, setIsEpisodesOpen] = useState<boolean>(false)

  // Triage state
  const [queueRows, setQueueRows] = useState<QueueRow[]>([])
  const [focusedIndex, setFocusedIndex] = useState<number>(0)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

  // Editor state
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false)

  // Load stats & health
  const refreshStats = useCallback(async () => {
    try {
      setStats(await api.getStats())
    } catch {
      // stats are advisory; a failed refresh should not interrupt annotation
    }
  }, [])

  useEffect(() => {
    api
      .getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable' }))
    refreshStats()

    const interval = setInterval(refreshStats, 15000)
    return () => clearInterval(interval)
  }, [refreshStats])

  // Load queue rows
  const loadQueue = useCallback(async (qName: string) => {
    try {
      const rows = await api.getQueue({ queue: qName, limit: 100 })
      setQueueRows(rows)
      setSelectedIds(new Set())
      setFocusedIndex((prev) => (prev >= rows.length ? Math.max(0, rows.length - 1) : prev))
    } catch (err) {
      console.error('Failed to load queue:', err)
      toast.error('Failed to load queue')
    }
  }, [])

  useEffect(() => {
    loadQueue(activeQueue)
  }, [activeQueue, loadQueue])

  // Resume active task
  const handleResume = async () => {
    try {
      const task = await api.getNextTask({ queue: activeQueue })
      setCurrentTask(task)
      setActiveMode('editor')
      toast.info(`Resumed ${task.segment.external_id}`)
    } catch (err: any) {
      if (err.status === 404) {
        toast.warning('No pending tasks in queue')
      } else {
        toast.error('Failed to resume task')
      }
    }
  }

  // Open task in editor
  const handleOpenEditor = async (taskId: number) => {
    try {
      setCurrentTask(await api.getTask(taskId))
      setActiveMode('editor')
    } catch {
      toast.error(`Failed to load task #${taskId}`)
    }
  }

  const dropRow = (taskId: number) => {
    setQueueRows((prev) => prev.filter((r) => r.task_id !== taskId))
    setSelectedIds((prev) => {
      const next = new Set(prev)
      next.delete(taskId)
      return next
    })
  }

  // Triage: Accept row
  const handleAcceptRow = async (taskId: number, durationMs: number) => {
    const targetRow = queueRows.find((r) => r.task_id === taskId)
    try {
      await api.acceptTask(taskId, {
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      toast.success(
        `Accepted ${targetRow?.segment_external_id || `#${taskId}`}`,
        { description: `${(durationMs / 1000).toFixed(1)}s` },
      )
      dropRow(taskId)
      refreshStats()
    } catch (err: any) {
      toast.error(err.detail || 'Failed to accept task')
    }
  }

  // Triage: Flag row
  const handleFlagRow = async (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number,
  ) => {
    const targetRow = queueRows.find((r) => r.task_id === taskId)
    try {
      await api.flagTask(taskId, {
        disposition,
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      toast.warning(
        `Flagged ${targetRow?.segment_external_id || `#${taskId}`} as ${disposition}`,
      )
      dropRow(taskId)
      refreshStats()
    } catch (err: any) {
      toast.error(err.detail || 'Failed to flag task')
    }
  }

  // Triage: Bulk accept
  const handleBulkAccept = async (taskIds: number[]) => {
    if (taskIds.length === 0) return
    try {
      await api.bulkAccept({ task_ids: taskIds })
      toast.success(`Bulk accepted ${taskIds.length} segments`)
      const idSet = new Set(taskIds)
      setQueueRows((prev) => prev.filter((r) => !idSet.has(r.task_id)))
      setSelectedIds(new Set())
      refreshStats()
    } catch (err: any) {
      toast.error(err.detail || 'Bulk accept failed')
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
    setSelectedIds(selectAll ? new Set(queueRows.map((r) => r.task_id)) : new Set())
  }

  /** Accept-unchanged or label depending on whether the seed text was edited. */
  const persistDecision = async (taskId: number, finalText: string, durationMs: number) => {
    if (!currentTask) return { edited: false }
    const seedHyp = currentTask.segment.hypotheses.find(
      (h) => h.id === currentTask.seed_hypothesis_id,
    )
    const isUnchanged = finalText.trim() === (seedHyp?.text || '').trim()
    const opened_at = new Date(Date.now() - durationMs).toISOString()

    if (isUnchanged) {
      await api.acceptTask(taskId, { duration_ms: durationMs, opened_at })
    } else {
      await api.labelTask(taskId, { final_text: finalText, duration_ms: durationMs, opened_at })
    }
    return { edited: !isUnchanged }
  }

  /** Fetch the next task, falling back to triage when the queue is exhausted. */
  const advanceToNextTask = async () => {
    try {
      setCurrentTask(await api.getNextTask({ queue: activeQueue }))
    } catch (err: any) {
      if (err.status === 404) {
        toast.info('Queue complete — returning to triage')
      } else {
        toast.error('Error loading next task')
      }
      setActiveMode('triage')
      loadQueue(activeQueue)
    }
  }

  // Editor: Save and next
  const handleSaveAndNext = async (taskId: number, finalText: string, durationMs: number) => {
    if (!currentTask) return
    const externalId = currentTask.segment.external_id
    try {
      const { edited } = await persistDecision(taskId, finalText, durationMs)
      toast.success(edited ? `Saved edits for ${externalId}` : `Accepted ${externalId} unchanged`)
      refreshStats()
      await advanceToNextTask()
    } catch (err: any) {
      toast.error(err.detail || 'Save failed')
      throw err
    }
  }

  // Editor: Save and stay
  const handleSaveAndStay = async (taskId: number, finalText: string, durationMs: number) => {
    if (!currentTask) return
    try {
      const { edited } = await persistDecision(taskId, finalText, durationMs)
      toast.success(edited ? 'Saved edits' : 'Saved unchanged')
      refreshStats()
    } catch (err: any) {
      toast.error(err.detail || 'Save failed')
      throw err
    }
  }

  // Editor: Flag
  const handleEditorFlag = async (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number,
  ) => {
    try {
      await api.flagTask(taskId, {
        disposition,
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      toast.warning(`Flagged as ${disposition}`)
      refreshStats()
      await advanceToNextTask()
    } catch (err: any) {
      toast.error(err.detail || 'Flag failed')
    }
  }

  // Editor: Skip
  const handleEditorSkip = async (taskId: number, durationMs: number) => {
    try {
      await api.skipTask(taskId, {
        duration_ms: durationMs,
        opened_at: new Date(Date.now() - durationMs).toISOString(),
      })
      toast.info('Deferred task')
      refreshStats()
      await advanceToNextTask()
    } catch (err: any) {
      toast.error(err.detail || 'Skip failed')
    }
  }

  // Global keybinding: ? toggles the shortcut reference
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
    <div className="flex h-full flex-col bg-background">
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
        />
      )}

      <EpisodeManagerModal
        isOpen={isEpisodesOpen}
        onClose={() => setIsEpisodesOpen(false)}
        onDataChanged={() => {
          refreshStats()
          loadQueue(activeQueue)
        }}
      />

      <IngestModal
        isOpen={isIngestOpen}
        onClose={() => setIsIngestOpen(false)}
        onComplete={(episodeId) => {
          refreshStats()
          loadQueue(activeQueue)
          setActiveMode('triage')
          toast.success(`Episode '${episodeId}' ingested`, {
            description: 'The priority queue has been updated.',
          })
        }}
      />

      <KeyboardShortcutsModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  )
}
