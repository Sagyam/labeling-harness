import React, { useEffect, useRef, useState } from 'react'
import { api, resolveUrl } from '../services/api'
import { PeaksPayload, Task } from '../types'
import { DiffViewer } from './DiffViewer'
import { HypothesesList } from './HypothesesList'
import { TranslitEditor } from './TranslitEditor'
import { Waveform } from './Waveform'

interface EditorViewProps {
  task: Task
  onSaveAndNext: (taskId: number, finalText: string, durationMs: number) => Promise<void>
  onSaveAndStay: (taskId: number, finalText: string, durationMs: number) => Promise<void>
  onFlag: (taskId: number, disposition: 'unusable_audio' | 'uncertain', durationMs: number) => Promise<void>
  onSkip: (taskId: number, durationMs: number) => Promise<void>
  onExitToTriage: () => void
  onToast: (msg: string, type?: 'success' | 'info' | 'warning' | 'error') => void
}

export const EditorView: React.FC<EditorViewProps> = ({
  task,
  onSaveAndNext,
  onSaveAndStay,
  onFlag,
  onSkip,
  onExitToTriage,
}) => {
  const segment = task.segment
  const seedHypothesis = segment.hypotheses.find((h) => h.id === task.seed_hypothesis_id)
  const seedText = seedHypothesis?.text || segment.hypotheses[0]?.text || ''

  const [text, setText] = useState<string>(seedText)
  const [peaks, setPeaks] = useState<PeaksPayload | null>(null)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [duration, setDuration] = useState<number>(segment.duration_seconds || 0)
  const [playbackRate, setPlaybackRate] = useState<number>(1.0)
  const [isLooping, setIsLooping] = useState<boolean>(false)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [isPopupOpen, setIsPopupOpen] = useState<boolean>(false)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const openedAtRef = useRef<number>(Date.now())

  // Reset text and timers when task changes
  useEffect(() => {
    setText(seedText)
    openedAtRef.current = Date.now()
    setCurrentTime(0)
    setIsPlaying(false)

    // Fetch peaks
    if (segment.peaks_url) {
      api
        .getPeaks(segment.peaks_url)
        .then(setPeaks)
        .catch((err) => console.error('Failed to load peaks:', err))
    }
  }, [task.id, seedText, segment.peaks_url])

  // Audio event listeners
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime)
    const handleLoadedMetadata = () => {
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration)
      }
    }
    const handleEnded = () => {
      if (isLooping) {
        audio.currentTime = 0
        audio.play().catch(() => {})
      } else {
        setIsPlaying(false)
      }
    }
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)

    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('loadedmetadata', handleLoadedMetadata)
    audio.addEventListener('ended', handleEnded)
    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePause)

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata)
      audio.removeEventListener('ended', handleEnded)
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePause)
    }
  }, [isLooping])

  // Playback speed sync
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackRate
    }
  }, [playbackRate])

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      audio.play().catch((err) => console.error('Audio play error:', err))
    } else {
      audio.pause()
    }
  }

  const seek = (time: number) => {
    const audio = audioRef.current
    if (!audio) return
    const target = Math.max(0, Math.min(duration, time))
    audio.currentTime = target
    setCurrentTime(target)
  }

  const seekDelta = (delta: number) => {
    if (audioRef.current) {
      seek(audioRef.current.currentTime + delta)
    }
  }

  const getDurationMs = () => Math.max(0, Date.now() - openedAtRef.current)

  // Actions
  const handleSave = async (andNext: boolean) => {
    if (isSubmitting) return
    setIsSubmitting(true)
    try {
      const dur = getDurationMs()
      if (andNext) {
        await onSaveAndNext(task.id, text, dur)
      } else {
        await onSaveAndStay(task.id, text, dur)
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  // Global Editor Shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      const isTextarea = targetTag === 'textarea' || targetTag === 'input'

      // Ctrl+Space: Play/Pause (always, even when textarea is focused)
      if (e.ctrlKey && e.code === 'Space') {
        e.preventDefault()
        togglePlay()
        return
      }

      // Space: Play/Pause when textarea is NOT focused
      if (!isTextarea && e.code === 'Space') {
        e.preventDefault()
        togglePlay()
        return
      }

      // Ctrl+Shift+Enter: Save and stay
      if (e.ctrlKey && e.shiftKey && e.key === 'Enter') {
        e.preventDefault()
        handleSave(false)
        return
      }

      // Ctrl+Enter: Save and next
      if (e.ctrlKey && !e.shiftKey && e.key === 'Enter') {
        e.preventDefault()
        handleSave(true)
        return
      }

      // Ctrl+L: Toggle loop
      if (e.ctrlKey && (e.key === 'l' || e.key === 'L')) {
        e.preventDefault()
        setIsLooping((prev) => !prev)
        return
      }

      // Alt+Left: Seek -2s
      if (e.altKey && e.key === 'ArrowLeft') {
        e.preventDefault()
        seekDelta(-2)
        return
      }

      // Alt+Right: Seek +2s
      if (e.altKey && e.key === 'ArrowRight') {
        e.preventDefault()
        seekDelta(2)
        return
      }

      // Alt+1..5: Load hypothesis N
      if (e.altKey && /^[1-5]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1
        if (idx < segment.hypotheses.length) {
          e.preventDefault()
          setText(segment.hypotheses[idx].text)
          return
        }
      }

      // Esc: Tiered exit
      if (e.key === 'Escape') {
        if (isPopupOpen) {
          // Let the TranslitEditor handle closing its popup
          return
        }
        if (isTextarea) {
          // Unfocus textarea
          ;(e.target as HTMLElement).blur()
          return
        }
        // Exit to triage
        onExitToTriage()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [segment.hypotheses, text, isPopupOpen, isSubmitting])

  const audioUrl = resolveUrl(segment.audio_url)

  return (
    <div className="editor-container">
      {/* Hidden audio element */}
      <audio ref={audioRef} src={audioUrl} preload="auto" />

      {/* Editor Header Bar */}
      <div className="editor-header-bar">
        <div className="editor-segment-meta">
          <button
            className="btn-icon"
            onClick={onExitToTriage}
            title="Return to triage list (Esc)"
          >
            ←
          </button>
          <span className="editor-title">{segment.external_id}</span>
          <div className="meta-tags">
            <span className={`badge-split badge-${segment.split}`}>{segment.split}</span>
            <span className="system-chip">{segment.episode_external_id}</span>
            {segment.speaker_id && <span className="system-chip">{segment.speaker_id}</span>}
            {segment.lid && <span className="flag-chip">{segment.lid}</span>}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <button
            className="btn-action-sm btn-flag-sm"
            onClick={() => onFlag(task.id, 'unusable_audio', getDurationMs())}
            title="Mark audio unusable (f)"
          >
            Unusable Audio
          </button>
          <button
            className="btn-action-sm"
            style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', borderColor: 'rgba(245, 158, 11, 0.3)' }}
            onClick={() => onFlag(task.id, 'uncertain', getDurationMs())}
            title="Mark uncertain (u)"
          >
            Uncertain
          </button>
          <button
            className="btn-action-sm"
            style={{ background: 'var(--bg-card-hover)', color: 'var(--text-muted)' }}
            onClick={() => onSkip(task.id, getDurationMs())}
            title="Defer task without judging"
          >
            Skip
          </button>
        </div>
      </div>

      {/* Waveform Panel */}
      <div className="waveform-panel">
        <Waveform
          peaks={peaks}
          currentTime={currentTime}
          duration={duration}
          onSeek={seek}
          isPlaying={isPlaying}
        />

        {/* Playback Controls */}
        <div className="playback-controls-bar">
          <div className="playback-left">
            <button
              className="btn-play-large"
              onClick={togglePlay}
              title="Play/Pause (Ctrl+Space)"
            >
              {isPlaying ? '⏸' : '▶'}
            </button>

            <div className="time-display">
              {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
            </div>

            <button
              className="btn-icon"
              style={{ width: 'auto', padding: '0 0.5rem', fontSize: '0.72rem' }}
              onClick={() => seekDelta(-2)}
              title="Seek back 2 seconds (Alt+Left)"
            >
              -2s
            </button>
            <button
              className="btn-icon"
              style={{ width: 'auto', padding: '0 0.5rem', fontSize: '0.72rem' }}
              onClick={() => seekDelta(2)}
              title="Seek forward 2 seconds (Alt+Right)"
            >
              +2s
            </button>

            <button
              className={`btn-loop ${isLooping ? 'active' : ''}`}
              onClick={() => setIsLooping((prev) => !prev)}
              title="Toggle segment loop (Ctrl+L)"
            >
              Loop {isLooping ? 'ON' : 'OFF'}
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-faint)' }}>Speed:</span>
            <div className="speed-selector">
              {[0.75, 1.0, 1.25].map((rate) => (
                <button
                  key={rate}
                  className={`speed-btn ${playbackRate === rate ? 'active' : ''}`}
                  onClick={() => setPlaybackRate(rate)}
                >
                  {rate}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid: Transliteration Editor + Diff */}
      <div className="editor-main-grid">
        <TranslitEditor
          value={text}
          onChange={setText}
          textareaRef={textareaRef}
          onPopupOpenChange={setIsPopupOpen}
        />

        <DiffViewer seedText={seedText} currentText={text} />
      </div>

      {/* Upstream Hypotheses Comparison */}
      <HypothesesList
        hypotheses={segment.hypotheses}
        seedHypothesisId={task.seed_hypothesis_id}
        onSelectHypothesis={(selectedText) => setText(selectedText)}
      />

      {/* Bottom Sticky Action Bar */}
      <div className="editor-bottom-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)' }}>
            Priority Score: <strong>{task.priority_score.toFixed(3)}</strong>
          </span>
          {task.reason?.flags?.map((f) => (
            <span key={f} className="flag-chip">
              {f}
            </span>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            className="btn-primary-action btn-save-stay"
            onClick={() => handleSave(false)}
            disabled={isSubmitting}
            title="Save changes and stay on this segment (Ctrl+Shift+Enter)"
          >
            <span>Save & Stay</span>
            <kbd style={{ fontSize: '0.65rem' }}>Ctrl+Shift+↵</kbd>
          </button>

          <button
            className="btn-primary-action btn-save-next"
            onClick={() => handleSave(true)}
            disabled={isSubmitting}
            title="Save changes and advance to next segment (Ctrl+Enter)"
          >
            <span>Save & Next</span>
            <kbd style={{ fontSize: '0.65rem' }}>Ctrl+↵</kbd>
          </button>
        </div>
      </div>
    </div>
  )
}
