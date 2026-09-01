import React, { useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import { IngestEvent, IngestJobStatus, IngestLogEntry } from '../types'

interface IngestModalProps {
  isOpen: boolean
  onClose: () => void
  onComplete: (episodeId: string) => void
  onToast: (text: string, type?: 'success' | 'info' | 'warning' | 'error') => void
}

const STAGES = [
  { key: 'normalizing', label: '1. Normalize Audio', desc: 'FFmpeg loudnorm, 16 kHz mono FLAC' },
  { key: 'segmenting', label: '2. Silero VAD', desc: 'CPU speech turn detection (2s-20s)' },
  { key: 'transcribing', label: '3. Cloud ASR', desc: 'OpenRouter speech-to-text inference' },
  { key: 'analyzing', label: '4. Token Analysis', desc: 'Devanagari/Latin tagging & CMI' },
  { key: 'importing', label: '5. Direct Import', desc: 'Database records & queue building' },
]

export const IngestModal: React.FC<IngestModalProps> = ({
  isOpen,
  onClose,
  onComplete,
  onToast,
}) => {
  // Form State
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [showId, setShowId] = useState<string>('nepanglish')
  const [episodeTitle, setEpisodeTitle] = useState<string>('')
  const [episodeId, setEpisodeId] = useState<string>('')
  const [isManualEpisodeId, setIsManualEpisodeId] = useState<boolean>(false)
  const [isDragging, setIsDragging] = useState<boolean>(false)

  // Execution State
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<IngestJobStatus | null>(null)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [logs, setLogs] = useState<IngestLogEntry[]>([])
  const [completedSummary, setCompletedSummary] = useState<Record<string, any> | null>(null)

  const logContainerRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Auto-slugify episode title into episode ID
  const slugify = (text: string) => {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .trim()
      .replace(/[-\s]+/g, '_')
      .slice(0, 40)
  }

  const handleTitleChange = (val: string) => {
    setEpisodeTitle(val)
    if (!isManualEpisodeId) {
      setEpisodeId(slugify(val))
    }
  }

  // Handle Drag & Drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0])
    }
  }

  const handleFileSelected = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase() || ''
    const allowed = ['mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg']
    if (!allowed.includes(ext)) {
      onToast(`Unsupported format .${ext}. Please use MP3, M4A, WAV, or FLAC.`, 'error')
      return
    }
    setSelectedFile(file)
    if (!episodeTitle) {
      const baseName = file.name.replace(/\.[^/.]+$/, '')
      setEpisodeTitle(baseName)
      if (!isManualEpisodeId) {
        setEpisodeId(slugify(baseName))
      }
    }
  }

  // Submit Ingestion Job
  const handleStartIngestion = async () => {
    if (!selectedFile) {
      onToast('Please select an audio file', 'warning')
      return
    }
    if (!episodeTitle.trim()) {
      onToast('Please enter an episode title', 'warning')
      return
    }

    setIsSubmitting(true)
    setLogs([])
    setCompletedSummary(null)

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('episode_title', episodeTitle.trim())
    formData.append('show_id', showId.trim() || 'podcast')
    formData.append('episode_id', episodeId.trim())

    try {
      const res = await api.startIngest(formData)
      setJobId(res.job_id)
      onToast(`Ingestion started for '${res.title}'`, 'info')
    } catch (err: any) {
      setIsSubmitting(false)
      onToast(err.message || 'Failed to start ingestion', 'error')
    }
  }

  // Subscribe to SSE Log & Progress Stream
  useEffect(() => {
    if (!jobId) return

    const unsubscribe = api.subscribeIngestEvents(
      jobId,
      (evt: IngestEvent) => {
        if (evt.type === 'log') {
          setLogs((prev) => [
            ...prev,
            { timestamp: evt.timestamp, level: evt.level, message: evt.message },
          ])
        } else if (evt.type === 'progress') {
          setJobStatus((prev) =>
            prev
              ? {
                  ...prev,
                  stage: evt.stage as any,
                  progress: evt.progress,
                  active_segments: evt.active_segments,
                  total_segments: evt.total_segments,
                }
              : null
          )
        } else if (evt.type === 'complete') {
          setCompletedSummary(evt.summary)
          setJobStatus((prev) =>
            prev
              ? {
                  ...prev,
                  status: 'completed',
                  stage: 'complete',
                  progress: 100.0,
                }
              : null
          )
          setIsSubmitting(false)
          onToast('Ingestion completed! Episode is ready for annotation.', 'success')
        } else if (evt.type === 'error') {
          setJobStatus((prev) =>
            prev
              ? {
                  ...prev,
                  status: 'failed',
                  stage: 'failed',
                  error: evt.error,
                }
              : null
          )
          setIsSubmitting(false)
          onToast(`Ingestion error: ${evt.error}`, 'error')
        }
      },
      () => {
        // SSE connection closed or error
      }
    )

    // Initial snapshot fetch
    api.getIngestStatus(jobId).then((status) => {
      setJobStatus(status)
      if (status.logs && status.logs.length > 0) {
        setLogs(status.logs)
      }
      if (status.status === 'completed') {
        setIsSubmitting(false)
      }
    })

    return () => unsubscribe()
  }, [jobId, onToast])

  // Auto-scroll terminal log console
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  // Reset state on close
  const handleClose = () => {
    if (isSubmitting) {
      if (!window.confirm('Ingestion is actively running in the background. Close modal?')) {
        return
      }
    }
    setJobId(null)
    setJobStatus(null)
    setIsSubmitting(false)
    setSelectedFile(null)
    setEpisodeTitle('')
    setEpisodeId('')
    setLogs([])
    setCompletedSummary(null)
    onClose()
  }

  // Finish and open triage
  const handleStartAnnotating = () => {
    const finalId = jobStatus?.episode_id || episodeId || 'new'
    onComplete(finalId)
    handleClose()
  }

  // Copy terminal logs
  const handleCopyLogs = () => {
    const text = logs.map((l) => `[${l.timestamp}] [${l.level.toUpperCase()}] ${l.message}`).join('\n')
    navigator.clipboard.writeText(text)
    onToast('Logs copied to clipboard', 'info')
  }

  if (!isOpen) return null

  // Determine stage status
  const currentStage = jobStatus?.stage || 'upload'
  const isComplete = jobStatus?.status === 'completed' || currentStage === 'complete'
  const isFailed = jobStatus?.status === 'failed' || currentStage === 'failed'

  const getStageState = (stageKey: string) => {
    if (isComplete) return 'done'
    const stageOrder = ['normalizing', 'segmenting', 'transcribing', 'analyzing', 'importing']
    const currentIndex = stageOrder.indexOf(currentStage)
    const thisIndex = stageOrder.indexOf(stageKey)

    if (currentIndex === -1) return 'pending'
    if (thisIndex < currentIndex) return 'done'
    if (thisIndex === currentIndex) return isFailed ? 'failed' : 'active'
    return 'pending'
  }

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div
        className="modal-container ingest-modal-dialog"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="ingest-modal-title"
      >
        {/* Modal Header */}
        <div className="modal-header">
          <div className="ingest-modal-title-group">
            <div className="ingest-icon-badge">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <div>
              <h2 id="ingest-modal-title" className="modal-title">
                Ingest Podcast Episode
              </h2>
              <span className="modal-subtitle">
                Automated loudnorm, Silero VAD segmentation & Cloud ASR
              </span>
            </div>
          </div>
          <button className="modal-close" onClick={handleClose} aria-label="Close modal">
            &times;
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body ingest-modal-body">
          {!jobId ? (
            /* Upload & Configuration Form */
            <div className="ingest-form-flow">
              {/* File Dropzone */}
              <div
                className={`ingest-dropzone ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept=".mp3,.m4a,.wav,.flac,.aac,.ogg"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileSelected(e.target.files[0])
                    }
                  }}
                />

                {selectedFile ? (
                  <div className="dropzone-file-info">
                    <div className="file-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2">
                        <path d="M9 18V5l12-2v13" />
                        <circle cx="6" cy="18" r="3" />
                        <circle cx="18" cy="16" r="3" />
                      </svg>
                    </div>
                    <div className="file-details">
                      <span className="file-name">{selectedFile.name}</span>
                      <span className="file-meta">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB &bull;{' '}
                        {selectedFile.name.split('.').pop()?.toUpperCase()} Audio
                      </span>
                    </div>
                    <button
                      type="button"
                      className="btn-change-file"
                      onClick={(e) => {
                        e.stopPropagation()
                        fileInputRef.current?.click()
                      }}
                    >
                      Change
                    </button>
                  </div>
                ) : (
                  <div className="dropzone-placeholder">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                      <path d="M12 12v9" />
                      <path d="m8 16 4-4 4 4" />
                    </svg>
                    <p className="dropzone-prompt">
                      <strong>Click to upload</strong> or drag and drop podcast audio
                    </p>
                    <div className="format-badges">
                      <span>MP3</span>
                      <span>M4A</span>
                      <span>WAV</span>
                      <span>FLAC</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Metadata Inputs */}
              <div className="ingest-inputs-grid">
                <div className="input-group">
                  <label htmlFor="episode-title">Episode Title *</label>
                  <input
                    id="episode-title"
                    type="text"
                    className="modal-input"
                    placeholder="e.g. Episode 42: AI in Kathmandu"
                    value={episodeTitle}
                    onChange={(e) => handleTitleChange(e.target.value)}
                  />
                </div>

                <div className="input-row-split">
                  <div className="input-group">
                    <label htmlFor="show-id">Show ID</label>
                    <input
                      id="show-id"
                      type="text"
                      className="modal-input"
                      placeholder="podcast"
                      value={showId}
                      onChange={(e) => setShowId(e.target.value)}
                    />
                  </div>

                  <div className="input-group">
                    <div className="label-with-toggle">
                      <label htmlFor="episode-id">Episode ID (Slug)</label>
                      <button
                        type="button"
                        className="btn-text-action"
                        onClick={() => {
                          setIsManualEpisodeId(!isManualEpisodeId)
                          if (isManualEpisodeId) setEpisodeId(slugify(episodeTitle))
                        }}
                      >
                        {isManualEpisodeId ? 'Auto-generate' : 'Custom'}
                      </button>
                    </div>
                    <input
                      id="episode-id"
                      type="text"
                      className="modal-input"
                      placeholder="ep_42_ai_in_kathmandu"
                      value={episodeId}
                      readOnly={!isManualEpisodeId}
                      onChange={(e) => setEpisodeId(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Pipeline Overview Card */}
              <div className="pipeline-overview-card">
                <span className="overview-title">Pipeline Workflow</span>
                <div className="overview-steps">
                  <div className="ov-step">
                    <span className="ov-dot" />
                    <span>Loudnorm 16kHz</span>
                  </div>
                  <span className="ov-arrow">&rarr;</span>
                  <div className="ov-step">
                    <span className="ov-dot" />
                    <span>Silero VAD (2s-20s)</span>
                  </div>
                  <span className="ov-arrow">&rarr;</span>
                  <div className="ov-step">
                    <span className="ov-dot" />
                    <span>Cloud ASR</span>
                  </div>
                  <span className="ov-arrow">&rarr;</span>
                  <div className="ov-step">
                    <span className="ov-dot" />
                    <span>CMI & Queue</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Active Live Progress & Streaming Terminal Console */
            <div className="ingest-progress-flow">
              {/* Stepper */}
              <div className="ingest-stepper">
                {STAGES.map((s) => {
                  const state = getStageState(s.key)
                  return (
                    <div key={s.key} className={`stepper-item ${state}`}>
                      <div className="stepper-indicator">
                        {state === 'done' ? (
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        ) : state === 'active' ? (
                          <span className="spinner-dot" />
                        ) : state === 'failed' ? (
                          <span>!</span>
                        ) : (
                          <span className="hollow-dot" />
                        )}
                      </div>
                      <div className="stepper-info">
                        <span className="stepper-label">{s.label}</span>
                        <span className="stepper-desc">{s.desc}</span>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Live Progress Bar */}
              <div className="ingest-progress-card">
                <div className="progress-header">
                  <span className="progress-stage-title">
                    {isComplete
                      ? 'Ingestion Complete!'
                      : isFailed
                      ? 'Pipeline Failed'
                      : `Processing: ${currentStage.toUpperCase()}...`}
                  </span>
                  <div className="progress-metrics">
                    {jobStatus?.active_segments ? (
                      <span className="segment-badge">
                        {jobStatus.active_segments} segments
                      </span>
                    ) : null}
                    <span className="percentage-badge">
                      {Math.round(jobStatus?.progress || 0)}%
                    </span>
                  </div>
                </div>
                <div className="progress-track">
                  <div
                    className={`progress-fill ${isFailed ? 'failed' : isComplete ? 'complete' : ''}`}
                    style={{ width: `${Math.min(100, jobStatus?.progress || 0)}%` }}
                  />
                </div>
              </div>

              {/* Terminal Log Console */}
              <div className="terminal-console-card">
                <div className="terminal-header">
                  <div className="terminal-dots">
                    <span className="dot red" />
                    <span className="dot yellow" />
                    <span className="dot green" />
                  </div>
                  <span className="terminal-title">Live Pipeline Log Stream (SSE)</span>
                  <button
                    type="button"
                    className="btn-terminal-copy"
                    onClick={handleCopyLogs}
                    title="Copy logs"
                  >
                    Copy
                  </button>
                </div>

                <div className="terminal-body" ref={logContainerRef}>
                  {logs.length === 0 ? (
                    <div className="terminal-empty">Connecting to event stream...</div>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} className={`log-line ${log.level}`}>
                        <span className="log-time">{log.timestamp}</span>
                        <span className={`log-badge ${log.level}`}>
                          [{log.level.toUpperCase()}]
                        </span>
                        <span className="log-msg">{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Completion Banner */}
              {isComplete && completedSummary && (
                <div className="ingest-success-card">
                  <div className="success-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="2.5">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                  </div>
                  <div className="success-content">
                    <span className="success-title">Episode ready for review</span>
                    <span className="success-desc">
                      {completedSummary.segments} segments extracted and {completedSummary.tasks_created} tasks added to the prioritization queue.
                    </span>
                  </div>
                </div>
              )}

              {/* Error Banner */}
              {isFailed && (
                <div className="ingest-error-card">
                  <div className="error-icon">!</div>
                  <div className="error-content">
                    <span className="error-title">Ingestion Error</span>
                    <span className="error-desc">{jobStatus?.error || 'An unknown error occurred.'}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer ingest-modal-footer">
          {!jobId ? (
            <>
              <button type="button" className="btn-cancel" onClick={handleClose}>
                Cancel
              </button>
              <button
                type="button"
                className="btn-primary btn-start-ingest"
                disabled={!selectedFile || !episodeTitle.trim() || isSubmitting}
                onClick={handleStartIngestion}
              >
                {isSubmitting ? 'Starting...' : 'Start Ingestion'}
              </button>
            </>
          ) : isComplete ? (
            <>
              <button type="button" className="btn-cancel" onClick={handleClose}>
                Close
              </button>
              <button
                type="button"
                className="btn-primary btn-start-annotating"
                onClick={handleStartAnnotating}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                </svg>
                Start Annotating Now
              </button>
            </>
          ) : isFailed ? (
            <>
              <button type="button" className="btn-cancel" onClick={handleClose}>
                Close
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={() => setJobId(null)}
              >
                Try Again
              </button>
            </>
          ) : (
            <div className="ingest-processing-status">
              <span className="pulse-indicator" />
              <span>Job #{jobId.slice(0, 8)} running in background...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
