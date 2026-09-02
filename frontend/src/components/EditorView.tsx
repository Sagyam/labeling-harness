import { useEffect, useRef, useState } from 'react'
import {
  RiArrowLeftLine,
  RiDeleteBin6Line,
  RiPauseFill,
  RiPlayFill,
  RiRepeat2Line,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Chip } from '@/components/Chip'
import { DiffViewer } from '@/components/DiffViewer'
import { HypothesesList } from '@/components/HypothesesList'
import { TranslitEditor } from '@/components/TranslitEditor'
import { Waveform } from '@/components/Waveform'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import { Kbd } from '@/components/ui/kbd'
import { Separator } from '@/components/ui/separator'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { api, resolveUrl } from '@/services/api'
import { cn } from '@/lib/utils'
import type { PeaksPayload, Task } from '@/types'

interface EditorViewProps {
  task: Task
  onSaveAndNext: (taskId: number, finalText: string, durationMs: number) => Promise<void>
  onSaveAndStay: (taskId: number, finalText: string, durationMs: number) => Promise<void>
  onFlag: (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number,
  ) => Promise<void>
  onSkip: (taskId: number, durationMs: number) => Promise<void>
  onExitToTriage: () => void
}

const SPEEDS = ['0.75', '1', '1.25'] as const

export function EditorView({
  task,
  onSaveAndNext,
  onSaveAndStay,
  onFlag,
  onSkip,
  onExitToTriage,
}: EditorViewProps) {
  const segment = task.segment
  const seedHypothesis = segment.hypotheses.find((h) => h.id === task.seed_hypothesis_id)
  const seedText = seedHypothesis?.text || segment.hypotheses[0]?.text || ''

  const [text, setText] = useState<string>(seedText)
  const [peaks, setPeaks] = useState<PeaksPayload | null>(null)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [duration, setDuration] = useState<number>(segment.duration_seconds || 0)
  const [playbackRate, setPlaybackRate] = useState<string>('1')
  const [isLooping, setIsLooping] = useState<boolean>(false)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [isPopupOpen, setIsPopupOpen] = useState<boolean>(false)
  const [isDeleteOpen, setIsDeleteOpen] = useState<boolean>(false)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const openedAtRef = useRef<number>(Date.now())

  // Reset text and timers when task changes
  useEffect(() => {
    setText(seedText)
    openedAtRef.current = Date.now()
    setCurrentTime(0)
    setIsPlaying(false)

    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }

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
      if (audio.duration && !isNaN(audio.duration)) setDuration(audio.duration)
    }
    const handleEnded = () => {
      if (isLooping) {
        audio.pause()
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
    if (audioRef.current) audioRef.current.playbackRate = Number(playbackRate)
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
    if (audioRef.current) seek(audioRef.current.currentTime + delta)
  }

  const getDurationMs = () => Math.max(0, Date.now() - openedAtRef.current)

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

  // Global editor shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      const isTextarea = targetTag === 'textarea' || targetTag === 'input'

      // Ctrl+Space: Play/Pause (always, even when the textarea is focused)
      if (e.ctrlKey && e.code === 'Space') {
        e.preventDefault()
        togglePlay()
        return
      }

      // Space: Play/Pause when the textarea is NOT focused
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

      // Alt+Left / Alt+Right: Seek ∓2s
      if (e.altKey && e.key === 'ArrowLeft') {
        e.preventDefault()
        seekDelta(-2)
        return
      }

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
        if (isPopupOpen || isDeleteOpen) {
          // Let the transliteration popup / dialog handle closing itself
          return
        }
        if (isTextarea) {
          ;(e.target as HTMLElement).blur()
          return
        }
        onExitToTriage()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [segment.hypotheses, text, isPopupOpen, isDeleteOpen, isSubmitting])

  const handleDeleteSegment = async () => {
    try {
      await api.deleteSegment(segment.id)
      toast.info(`Segment ${segment.external_id} deleted`)
      setIsDeleteOpen(false)
      onExitToTriage()
    } catch (err: any) {
      toast.error(`Failed to delete segment: ${err.message}`)
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <audio ref={audioRef} src={resolveUrl(segment.audio_url)} preload="auto" />

      {/* Segment header */}
      <div className="flex h-12 shrink-0 flex-wrap items-center gap-3 border-b px-4">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon-sm" onClick={onExitToTriage} aria-label="Back">
              <RiArrowLeftLine />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Return to triage (Esc)</TooltipContent>
        </Tooltip>

        <span className="font-mono text-sm">{segment.external_id}</span>

        <div className="flex flex-wrap items-center gap-1">
          <Chip className="bg-info/15 text-info">{segment.split}</Chip>
          <Chip>{segment.episode_external_id}</Chip>
          {segment.speaker_id && <Chip>{segment.speaker_id}</Chip>}
          {segment.lid && <Chip>{segment.lid}</Chip>}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => onFlag(task.id, 'unusable_audio', getDurationMs())}
          >
            Unusable audio
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-warning hover:bg-warning/10 hover:text-warning"
            onClick={() => onFlag(task.id, 'uncertain', getDurationMs())}
          >
            Uncertain
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onSkip(task.id, getDurationMs())}>
            Skip
          </Button>
          <Separator orientation="vertical" className="h-6" />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                onClick={() => setIsDeleteOpen(true)}
                aria-label="Delete segment"
              >
                <RiDeleteBin6Line />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Permanently delete this segment</TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="scrollbar-thin flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
        {/* Waveform + transport */}
        <div className="bg-card ring-1 ring-foreground/5">
          <Waveform peaks={peaks} currentTime={currentTime} duration={duration} onSeek={seek} />

          <div className="flex flex-wrap items-center justify-between gap-3 border-t px-3 py-2">
            <div className="flex items-center gap-3">
              <ButtonGroup>
                <Button
                  variant="outline"
                  size="sm"
                  className="font-mono normal-case"
                  onClick={() => seekDelta(-2)}
                  title="Seek back 2 seconds (Alt+←)"
                >
                  -2s
                </Button>
                <Button size="icon-sm" onClick={togglePlay} aria-label={isPlaying ? 'Pause' : 'Play'}>
                  {isPlaying ? <RiPauseFill /> : <RiPlayFill />}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="font-mono normal-case"
                  onClick={() => seekDelta(2)}
                  title="Seek forward 2 seconds (Alt+→)"
                >
                  +2s
                </Button>
              </ButtonGroup>

              <span className="font-mono text-xs text-muted-foreground tabular-nums">
                {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
              </span>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={isLooping ? 'secondary' : 'outline'}
                    size="sm"
                    onClick={() => setIsLooping((prev) => !prev)}
                    className={cn(isLooping && 'text-info')}
                  >
                    <RiRepeat2Line data-icon="inline-start" />
                    Loop {isLooping ? 'on' : 'off'}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Toggle segment loop (Ctrl+L)</TooltipContent>
              </Tooltip>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Speed</span>
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                spacing={0}
                value={playbackRate}
                onValueChange={(value) => value && setPlaybackRate(value)}
              >
                {SPEEDS.map((rate) => (
                  <ToggleGroupItem key={rate} value={rate} className="font-mono normal-case">
                    {rate}x
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>
          </div>
        </div>

        {/* Transcript + diff */}
        <div className="grid min-h-56 gap-4 lg:grid-cols-2">
          <TranslitEditor
            value={text}
            onChange={setText}
            textareaRef={textareaRef}
            onPopupOpenChange={setIsPopupOpen}
          />
          <DiffViewer seedText={seedText} currentText={text} />
        </div>

        <HypothesesList
          hypotheses={segment.hypotheses}
          seedHypothesisId={task.seed_hypothesis_id}
          onSelectHypothesis={setText}
        />
      </div>

      {/* Action bar */}
      <div className="flex h-14 shrink-0 flex-wrap items-center justify-between gap-3 border-t bg-card/40 px-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">
            Priority{' '}
            <span className="font-mono text-foreground tabular-nums">
              {task.priority_score.toFixed(3)}
            </span>
          </span>
          {task.reason?.flags?.map((flag) => (
            <Chip key={flag} className="bg-warning/15 text-warning">
              {flag}
            </Chip>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled={isSubmitting} onClick={() => handleSave(false)}>
            Save &amp; stay
            <Kbd className="ml-1">Ctrl+⇧+↵</Kbd>
          </Button>
          <Button size="sm" disabled={isSubmitting} onClick={() => handleSave(true)}>
            Save &amp; next
            <Kbd className="ml-1 bg-primary-foreground/15 text-primary-foreground">Ctrl+↵</Kbd>
          </Button>
        </div>
      </div>

      <AlertDialog open={isDeleteOpen} onOpenChange={setIsDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this segment?</AlertDialogTitle>
            <AlertDialogDescription>
              <span className="font-mono">{segment.external_id}</span> and its audio clip will be
              removed from the corpus. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteSegment}>Delete segment</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
