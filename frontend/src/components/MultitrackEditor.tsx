import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  RiAddLine,
  RiArrowGoBackLine,
  RiArrowGoForwardLine,
  RiArrowLeftLine,
  RiPauseFill,
  RiUserVoiceLine,
  RiPlayFill,
  RiRepeat2Line,
  RiZoomInLine,
  RiZoomOutLine,
} from '@remixicon/react'
import { toast } from 'sonner'

import { Chip } from '@/components/Chip'
import { KaraokeTranscript } from '@/components/KaraokeTranscript'
import { VoiceDialog, useVoiceSample } from '@/components/VoiceDialog'
import { Waveform } from '@/components/Waveform'
import { Button } from '@/components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Kbd } from '@/components/ui/kbd'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  type Block,
  type CandidateBlock,
  MIN_BLOCK,
  TYPED_LENGTH,
  clamp,
  hasSuggestion,
  inTimeOrder,
  initialBlocks,
  initialCandidates,
  laneText,
  newId,
  snap,
  speakerSlot,
  summarize,
  toSubmission,
} from '@/lib/lanes'
import { SPEEDS, usePlaybackRate } from '@/lib/playback'
import { cn } from '@/lib/utils'
import { api, resolveUrl } from '@/services/api'
import type { HypothesisWord, LaneWord, PeaksPayload, SpeakerLanes, Task } from '@/types'

interface MultitrackEditorProps {
  task: Task & { lanes: SpeakerLanes }
  /** Store the lanes and move on to the next clip in the speakers queue. */
  onSave: (taskId: number, runId: number, words: LaneWord[], durationMs: number) => Promise<void>
  onFlag: (
    taskId: number,
    disposition: 'unusable_audio' | 'uncertain',
    durationMs: number,
    notes?: string,
  ) => Promise<void>
  onSkip: (taskId: number, durationMs: number) => Promise<void>
  onExitToTriage: () => void
}

/** Height of one lane, in px. */
const LANE_H = 48
/** Width of the lane names on the left, in px. */
const GUTTER = 176
const MIN_PPS = 40
const MAX_PPS = 600

// Full class names, so Tailwind sees every one of them.
const BLOCK_CLASS: Record<number, string> = {
  1: 'border-speaker-1 bg-speaker-1/15 data-[active]:bg-speaker-1/45',
  2: 'border-speaker-2 bg-speaker-2/15 data-[active]:bg-speaker-2/45',
  3: 'border-speaker-3 bg-speaker-3/20 data-[active]:bg-speaker-3/50',
  4: 'border-speaker-4 bg-speaker-4/20 data-[active]:bg-speaker-4/50',
}
const TURN_CLASS: Record<number, string> = {
  1: 'bg-speaker-1/8',
  2: 'bg-speaker-2/8',
  3: 'bg-speaker-3/10',
  4: 'bg-speaker-4/10',
}
const SWATCH_CLASS: Record<number, string> = {
  1: 'bg-speaker-1',
  2: 'bg-speaker-2',
  3: 'bg-speaker-3',
  4: 'bg-speaker-4',
}

type Row =
  | { kind: 'speaker'; key: string; speaker: number }
  | { kind: 'unplaced'; key: string }
  | { kind: 'candidates'; key: string }

type Drag =
  | {
      kind: 'move'
      id: string
      x: number
      y: number
      start: number
      end: number
      speaker: number | null
      axis: 'x' | 'y' | null
      before: Block[]
    }
  | { kind: 'resize'; id: string; edge: 'start' | 'end'; x: number; start: number; end: number; before: Block[] }
  | { kind: 'candidate'; id: string; x: number; y: number; moved: boolean }

const isLatin = (text: string) => /^[A-Za-z']+$/.test(text)

/**
 * The multitrack editor: one lane per speaker, one block per word, on a 10 ms grid (D98).
 *
 * The annotator does two things. They **move a word** to the lane of the person who said it --
 * drag it up or down, or select it and press that speaker's number -- and they **add a word** the
 * transcript lacks, by typing it at the playhead, double-clicking a lane where it was said, or
 * taking one the recognisers heard from the bottom row. Edges drag to size a block. Speakers are
 * the episode's and cannot be created or deleted here; a clip whose diarization merged two people
 * is flagged instead.
 */
export function MultitrackEditor({
  task,
  onSave,
  onFlag,
  onSkip,
  onExitToTriage,
}: MultitrackEditorProps) {
  const segment = task.segment
  const lanes = task.lanes
  const duration = Math.max(segment.duration_seconds || 0, 0.1)

  const [blocks, setBlocksState] = useState<Block[]>(() => initialBlocks(lanes, duration))
  const [candidates, setCandidates] = useState<CandidateBlock[]>(() => initialCandidates(lanes))
  const [history, setHistory] = useState<Block[][]>([])
  const [future, setFuture] = useState<Block[][]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [extraLanes, setExtraLanes] = useState<number[]>([])
  const [dropLane, setDropLane] = useState<string | null>(null)
  const [voiceOpen, setVoiceOpen] = useState<string | null>(null)
  const voiceOpenRef = useRef<string | null>(null)
  voiceOpenRef.current = voiceOpen
  const { playVoice, playingVoice, stop: stopVoice } = useVoiceSample()

  const [peaks, setPeaks] = useState<PeaksPayload | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [playbackRate, setPlaybackRate] = usePlaybackRate()
  const [isLooping, setIsLooping] = useState(false)
  const [pps, setPps] = useState(120)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const [draft, setDraft] = useState('')
  const [draftAt, setDraftAt] = useState<{ speaker: number; time: number } | null>(null)
  const [devanagari, setDevanagari] = useState(true)
  const [choices, setChoices] = useState<string[] | null>(null)
  const [choiceIndex, setChoiceIndex] = useState(0)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const playheadRef = useRef<HTMLDivElement | null>(null)
  const blockRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const inputRef = useRef<HTMLInputElement | null>(null)
  const dragRef = useRef<Drag | null>(null)
  const blocksRef = useRef<Block[]>(blocks)
  blocksRef.current = blocks
  const openedAtRef = useRef<number>(Date.now())
  const momentTimerRef = useRef<number | null>(null)

  const speakerNumbers = useMemo(() => lanes.speakers.map((s) => s.number), [lanes.speakers])
  const speakerByNumber = useMemo(
    () => new Map(lanes.speakers.map((s) => [s.number, s])),
    [lanes.speakers],
  )

  // A new clip: start again from what the server served.
  useEffect(() => {
    setBlocksState(initialBlocks(lanes, duration))
    setCandidates(initialCandidates(lanes))
    setHistory([])
    setFuture([])
    setSelectedId(null)
    setExtraLanes([])
    setDraft('')
    setDraftAt(null)
    setChoices(null)
    openedAtRef.current = Date.now()
    setCurrentTime(0)
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
    // Fit the whole clip in view when it fits, but never so tight that a short word cannot be
    // read or grabbed: 200 px a second gives a 0.2 s word 40 px; longer clips scroll.
    const width = (scrollRef.current?.clientWidth ?? 1200) - GUTTER - 24
    setPps(clamp(Math.floor(width / duration), 200, MAX_PPS))
  }, [task.id])

  // --- edits, with undo -----------------------------------------------------------------------

  const commit = useCallback((next: Block[], before: Block[] = blocksRef.current) => {
    setHistory((h) => [...h.slice(-199), before])
    setFuture([])
    setBlocksState(next)
  }, [])

  // Read through refs: the keyboard handler is bound once per render of its own dependencies,
  // and a stale history there would undo to the wrong state.
  const historyRef = useRef(history)
  historyRef.current = history
  const futureRef = useRef(future)
  futureRef.current = future

  const undo = () => {
    const past = historyRef.current
    if (past.length === 0) return
    setHistory(past.slice(0, -1))
    setFuture([blocksRef.current, ...futureRef.current])
    setBlocksState(past[past.length - 1])
  }

  const redo = () => {
    const [next, ...rest] = futureRef.current
    if (!next) return
    setFuture(rest)
    setHistory([...historyRef.current, blocksRef.current])
    setBlocksState(next)
  }

  const showLane = (speaker: number) =>
    setExtraLanes((lanesShown) => (lanesShown.includes(speaker) ? lanesShown : [...lanesShown, speaker]))

  const moveTo = (id: string, speaker: number) => {
    const current = blocksRef.current
    const block = current.find((b) => b.id === id)
    if (!block || block.speaker === speaker) return
    showLane(speaker)
    commit(current.map((b) => (b.id === id ? { ...b, speaker } : b)))
  }

  const copyTo = (id: string, speaker: number) => {
    const current = blocksRef.current
    const block = current.find((b) => b.id === id)
    if (!block) return
    if (block.speaker === speaker) {
      toast.info('Already on that lane')
      return
    }
    showLane(speaker)
    const copy: Block = {
      ...block,
      id: newId(),
      speaker,
      proposed_speaker: null,
      source: 'copy',
      suggested_speaker: null,
    }
    commit([...current, copy])
    setSelectedId(copy.id)
  }

  const adopt = (candidateId: string, speaker: number) => {
    const candidate = candidates.find((c) => c.id === candidateId)
    if (!candidate) return
    showLane(speaker)
    const block: Block = {
      id: newId(),
      word: candidate.word,
      start: candidate.start,
      end: candidate.end,
      speaker,
      proposed_speaker: null,
      source: 'recogniser',
      estimated: false,
    }
    commit([...blocksRef.current, block])
    setCandidates((cs) => cs.filter((c) => c.id !== candidateId))
    setSelectedId(block.id)
  }

  const removeBlock = (id: string) => {
    const block = blocksRef.current.find((b) => b.id === id)
    if (!block) return
    if (block.source === 'label') {
      toast.warning('A word of the verified text is moved, not deleted', {
        description: 'If it was never said, fix the transcript in the text editor, or flag this clip.',
      })
      return
    }
    commit(blocksRef.current.filter((b) => b.id !== id))
    setSelectedId(null)
  }

  /** Put blocks on the lane their voiceprint suggests (D99): one, or every one in the clip. */
  const takeSuggestions = (ids: string[] | null) => {
    const current = blocksRef.current
    const chosen = current.filter((b) => hasSuggestion(b) && (ids === null || ids.includes(b.id)))
    if (chosen.length === 0) {
      toast.info(ids === null ? 'No voiceprint suggestions left' : 'No suggestion on this word')
      return
    }
    const ids_ = new Set(chosen.map((b) => b.id))
    chosen.forEach((b) => showLane(b.suggested_speaker as number))
    commit(
      current.map((b) => (ids_.has(b.id) ? { ...b, speaker: b.suggested_speaker as number } : b)),
    )
    if (ids === null) toast.success(`Moved ${chosen.length} words where the voiceprint hears them`)
  }

  const addWord = (word: string) => {
    const token = word.trim()
    if (!token) return
    if (/\s/.test(token)) {
      toast.error('One word at a time: each word is its own block')
      return
    }
    const selected = blocksRef.current.find((b) => b.id === selectedId)
    const speaker =
      draftAt?.speaker ??
      selected?.speaker ??
      lanes.clip_speakers[0] ??
      speakerNumbers[0]
    if (speaker === undefined || speaker === null) return
    const at = draftAt?.time ?? audioRef.current?.currentTime ?? 0
    const start = clamp(snap(at), 0, Math.max(0, duration - MIN_BLOCK))
    const end = clamp(snap(start + TYPED_LENGTH), start + MIN_BLOCK, duration)
    const block: Block = {
      id: newId(),
      word: token,
      start,
      end,
      speaker,
      proposed_speaker: null,
      source: 'typed',
      estimated: false,
    }
    showLane(speaker)
    commit([...blocksRef.current, block])
    setSelectedId(block.id)
    setDraft('')
    setDraftAt(null)
    setChoices(null)
    inputRef.current?.blur()
  }

  /** Enter in the word box: Latin text is offered in Devanagari first, as in the text editor. */
  const submitDraft = async () => {
    if (choices) {
      const picked = choices[choiceIndex] ?? draft
      if (picked !== draft) api.translitChoice(draft, picked).catch(() => {})
      addWord(picked)
      return
    }
    const token = draft.trim()
    if (!token) return
    if (devanagari && isLatin(token)) {
      try {
        const res = await api.translit(token, 5)
        const options = [...res.candidates.filter((c) => c !== token), token]
        setChoices(options)
        setChoiceIndex(0)
        return
      } catch {
        // No transliteration service: the word goes in as typed.
      }
    }
    addWord(token)
  }

  // --- audio ----------------------------------------------------------------------------------

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime = () => setCurrentTime(audio.currentTime)
    const onEnded = () => {
      if (isLooping) {
        audio.currentTime = 0
        audio.play().catch(() => {})
      } else {
        setIsPlaying(false)
      }
    }
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('ended', onEnded)
    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)
    return () => {
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('ended', onEnded)
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('pause', onPause)
    }
  }, [isLooping])

  // A new source resets an element's rate to its default, so both are set, again per clip.
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    audio.defaultPlaybackRate = Number(playbackRate)
    audio.playbackRate = Number(playbackRate)
  }, [playbackRate, task.id])

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (momentTimerRef.current !== null) {
      window.clearTimeout(momentTimerRef.current)
      momentTimerRef.current = null
    }
    if (audio.paused) audio.play().catch((err) => console.error('Audio play error:', err))
    else audio.pause()
  }

  const seek = (time: number, andPlay = false) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = clamp(time, 0, duration)
    setCurrentTime(audio.currentTime)
    if (andPlay && audio.paused) audio.play().catch(() => {})
  }

  /** Play one block, padded so it does not start or end clipped. Stopped on a timer, as in the
   * text editor: `timeupdate` is too coarse to stop a word that lasts a third of a second. */
  const playSpan = (start: number, end: number) => {
    const audio = audioRef.current
    if (!audio) return
    if (momentTimerRef.current !== null) window.clearTimeout(momentTimerRef.current)
    const from = Math.max(0, start - 0.05)
    const to = Math.min(duration, end + 0.05)
    seek(from, true)
    momentTimerRef.current = window.setTimeout(
      () => {
        audio.pause()
        momentTimerRef.current = null
      },
      ((to - from) / Number(playbackRate)) * 1000,
    )
  }

  // Playhead and lit blocks, at frame rate. Every block the playhead is inside lights up, so in
  // crosstalk both voices' words are lit at once -- which is the point of lanes.
  useEffect(() => {
    let frame = 0
    const tick = () => {
      const audio = audioRef.current
      const t = audio?.currentTime ?? 0
      if (playheadRef.current) playheadRef.current.style.transform = `translateX(${t * pps}px)`
      for (const block of blocksRef.current) {
        const el = blockRefs.current.get(block.id)
        if (!el || block.start === null || block.end === null) continue
        const on = block.start <= t && t < block.end
        if (on !== el.hasAttribute('data-active')) {
          if (on) el.setAttribute('data-active', '')
          else el.removeAttribute('data-active')
        }
      }
      // Keep the playhead in view while playing.
      const scroller = scrollRef.current
      if (audio && !audio.paused && scroller) {
        const x = GUTTER + t * pps
        const left = scroller.scrollLeft
        const right = left + scroller.clientWidth
        if (x < left + GUTTER || x > right - 40) scroller.scrollLeft = x - GUTTER - 60
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [pps])

  // --- rows -----------------------------------------------------------------------------------

  const rows: Row[] = useMemo(() => {
    const shown = new Set<number>(
      lanes.clip_speakers.length > 0 ? lanes.clip_speakers : speakerNumbers,
    )
    extraLanes.forEach((n) => shown.add(n))
    blocks.forEach((b) => b.speaker !== null && shown.add(b.speaker))
    const out: Row[] = [...shown]
      .sort((a, b) => a - b)
      .map((speaker) => ({ kind: 'speaker' as const, key: `s${speaker}`, speaker }))
    if (blocks.some((b) => b.speaker === null)) out.push({ kind: 'unplaced', key: 'unplaced' })
    if (candidates.length > 0) out.push({ kind: 'candidates', key: 'candidates' })
    return out
  }, [lanes.clip_speakers, speakerNumbers, extraLanes, blocks, candidates.length])

  const hiddenSpeakers = speakerNumbers.filter(
    (n) => !rows.some((r) => r.kind === 'speaker' && r.speaker === n),
  )

  /** The row under a screen y, by the rows' own rectangles. */
  const rowAt = (clientY: number): Row | null => {
    for (const row of rows) {
      const el = rowRefs.current.get(row.key)
      if (!el) continue
      const rect = el.getBoundingClientRect()
      if (clientY >= rect.top && clientY < rect.bottom) return row
    }
    return null
  }

  // --- pointer interactions -------------------------------------------------------------------

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return
      const dx = e.clientX - drag.x
      if (drag.kind === 'resize') {
        const delta = dx / pps
        setBlocksState((current) =>
          current.map((b) => {
            if (b.id !== drag.id) return b
            if (drag.edge === 'start') {
              const start = clamp(snap(drag.start + delta), 0, drag.end - MIN_BLOCK)
              return { ...b, start, estimated: false }
            }
            const end = clamp(snap(drag.end + delta), drag.start + MIN_BLOCK, duration)
            return { ...b, end, estimated: false }
          }),
        )
        return
      }
      const dy = e.clientY - drag.y
      if (drag.kind === 'candidate') {
        if (!drag.moved && Math.hypot(dx, dy) < 4) return
        drag.moved = true
        const row = rowAt(e.clientY)
        setDropLane(row?.kind === 'speaker' ? row.key : null)
        return
      }
      if (drag.axis === null) {
        if (Math.hypot(dx, dy) < 4) return
        drag.axis = Math.abs(dx) > Math.abs(dy) ? 'x' : 'y'
      }
      if (drag.axis === 'x') {
        const length = drag.end - drag.start
        const start = clamp(snap(drag.start + dx / pps), 0, Math.max(0, duration - length))
        setBlocksState((current) =>
          current.map((b) =>
            b.id === drag.id ? { ...b, start, end: snap(start + length), estimated: false } : b,
          ),
        )
      } else {
        const row = rowAt(e.clientY)
        const speaker =
          row?.kind === 'speaker' ? row.speaker : row?.kind === 'unplaced' ? null : undefined
        if (speaker === undefined) return
        setDropLane(row?.key ?? null)
        setBlocksState((current) =>
          current.map((b) => (b.id === drag.id && b.speaker !== speaker ? { ...b, speaker } : b)),
        )
      }
    }
    const onUp = (e: PointerEvent) => {
      const drag = dragRef.current
      dragRef.current = null
      setDropLane(null)
      if (!drag) return
      if (drag.kind === 'candidate') {
        const row = rowAt(e.clientY)
        if (drag.moved && row?.kind === 'speaker') adopt(drag.id, row.speaker)
        else if (!drag.moved) {
          const candidate = candidates.find((c) => c.id === drag.id)
          if (candidate) seek(candidate.start)
        }
        return
      }
      if (drag.kind === 'move' && drag.axis === null) {
        // A click, not a drag: the block is selected and the playhead goes to it.
        seek(drag.start)
        return
      }
      const after = blocksRef.current
      if (after !== drag.before) commit(after, drag.before)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [pps, rows, candidates, duration])

  const startMove = (e: React.PointerEvent, block: Block) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    setSelectedId(block.id)
    dragRef.current = {
      kind: 'move',
      id: block.id,
      x: e.clientX,
      y: e.clientY,
      start: block.start as number,
      end: block.end as number,
      speaker: block.speaker,
      axis: null,
      before: blocksRef.current,
    }
  }

  const startResize = (e: React.PointerEvent, block: Block, edge: 'start' | 'end') => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    setSelectedId(block.id)
    dragRef.current = {
      kind: 'resize',
      id: block.id,
      edge,
      x: e.clientX,
      start: block.start as number,
      end: block.end as number,
      before: blocksRef.current,
    }
  }

  const startCandidate = (e: React.PointerEvent, candidate: CandidateBlock) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    setSelectedId(candidate.id)
    dragRef.current = { kind: 'candidate', id: candidate.id, x: e.clientX, y: e.clientY, moved: false }
  }

  const timeAt = (e: React.MouseEvent, track: HTMLElement) =>
    clamp((e.clientX - track.getBoundingClientRect().left) / pps, 0, duration)

  // --- keyboard -------------------------------------------------------------------------------

  const ordered = useMemo(() => inTimeOrder(blocks), [blocks])
  const summary = useMemo(() => summarize(blocks), [blocks])

  const getDurationMs = () => Math.max(0, Date.now() - openedAtRef.current)

  const handleSave = async () => {
    if (isSubmitting) return
    if (summary.unplaced > 0) {
      const first = ordered.find((b) => b.speaker === null)
      if (first) setSelectedId(first.id)
      toast.error(
        `${summary.unplaced} word${summary.unplaced === 1 ? ' is' : 's are'} on no lane yet`,
        { description: 'Put each one on the lane of the person who said it.' },
      )
      return
    }
    setIsSubmitting(true)
    try {
      await onSave(task.id, lanes.diarization_run_id, toSubmission(blocks), getDurationMs())
    } finally {
      setIsSubmitting(false)
    }
  }

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // The voice page owns the keyboard while it is open.
      if (voiceOpenRef.current) return
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      const typing = tag === 'input' || tag === 'textarea'
      const handled = () => {
        e.preventDefault()
        // Captured before the app's own shortcuts: here a digit means a lane, not a view.
        e.stopPropagation()
      }
      if (e.ctrlKey && e.key === 'Enter') {
        handled()
        handleSave()
        return
      }
      if (e.ctrlKey && e.code === 'Space') {
        handled()
        togglePlay()
        return
      }
      if (typing) return

      const selectedBlock = blocks.find((b) => b.id === selectedId)
      const selectedCandidate = candidates.find((c) => c.id === selectedId)
      const digit = /^Digit([1-9])$/.exec(e.code)

      if (e.code === 'Space') {
        handled()
        togglePlay()
      } else if (digit && !e.ctrlKey && !e.altKey && !e.metaKey) {
        const speaker = Number(digit[1])
        handled()
        if (!speakerByNumber.has(speaker)) {
          toast.warning(`This episode has no speaker ${speaker}`)
          return
        }
        if (selectedCandidate) adopt(selectedCandidate.id, speaker)
        else if (selectedBlock && e.shiftKey) copyTo(selectedBlock.id, speaker)
        else if (selectedBlock) moveTo(selectedBlock.id, speaker)
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        handled()
        if (e.altKey) {
          seek((audioRef.current?.currentTime ?? 0) + (e.key === 'ArrowRight' ? 2 : -2))
          return
        }
        const index = ordered.findIndex((b) => b.id === selectedId)
        const next =
          index < 0
            ? e.key === 'ArrowRight'
              ? 0
              : ordered.length - 1
            : clamp(index + (e.key === 'ArrowRight' ? 1 : -1), 0, ordered.length - 1)
        const block = ordered[next]
        if (block) {
          setSelectedId(block.id)
          seek(block.start as number)
          const el = blockRefs.current.get(block.id)
          el?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
        }
      } else if (e.key === 'Enter') {
        handled()
        if (selectedBlock) playSpan(selectedBlock.start as number, selectedBlock.end as number)
        else if (selectedCandidate) playSpan(selectedCandidate.start, selectedCandidate.end)
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        handled()
        if (selectedCandidate) {
          setCandidates((cs) => cs.filter((c) => c.id !== selectedCandidate.id))
          setSelectedId(null)
        } else if (selectedBlock) removeBlock(selectedBlock.id)
      } else if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
        handled()
        if (e.shiftKey) redo()
        else undo()
      } else if (e.ctrlKey && (e.key === 'y' || e.key === 'Y')) {
        handled()
        redo()
      } else if (e.ctrlKey && (e.key === 'l' || e.key === 'L')) {
        handled()
        setIsLooping((v) => !v)
      } else if ((e.key === 'v' || e.key === 'V') && !e.ctrlKey && !e.metaKey) {
        handled()
        if (e.shiftKey) takeSuggestions(null)
        else if (selectedBlock) takeSuggestions([selectedBlock.id])
      } else if (e.key === 'a' || e.key === 'A') {
        handled()
        inputRef.current?.focus()
      } else if (e.key === '=' || e.key === '+') {
        handled()
        setPps((v) => clamp(Math.round(v * 1.4), MIN_PPS, MAX_PPS))
      } else if (e.key === '-') {
        handled()
        setPps((v) => clamp(Math.round(v / 1.4), MIN_PPS, MAX_PPS))
      } else if (e.key === 'Escape') {
        handled()
        if (selectedId) setSelectedId(null)
        else onExitToTriage()
      }
    }
    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [blocks, candidates, selectedId, ordered, isSubmitting, summary, playbackRate])

  // --- rendering ------------------------------------------------------------------------------

  const width = Math.ceil(duration * pps)
  const tickStep = pps >= 250 ? 0.1 : pps >= 80 ? 0.5 : 1
  const ticks = useMemo(() => {
    const out: number[] = []
    for (let t = 0; t <= duration + 1e-9; t += tickStep) out.push(Math.round(t * 100) / 100)
    return out
  }, [duration, tickStep])

  const turnsBySpeaker = useMemo(() => {
    const map = new Map<number, { start: number; end: number }[]>()
    for (const turn of segment.speaker_turns ?? []) {
      map.set(turn.speaker, [...(map.get(turn.speaker) ?? []), turn])
    }
    return map
  }, [segment.speaker_turns])

  const mixWords: HypothesisWord[] = useMemo(
    () =>
      ordered
        .filter((b) => b.start !== null)
        .map((b, index) => ({
          position: index,
          word: b.word,
          start_time: b.start,
          end_time: b.end,
          confidence: null,
        })),
    [ordered],
  )
  const mixVoices = useMemo(
    () =>
      ordered
        .filter((b) => b.start !== null)
        .map((b) => ({ speaker: b.speaker, overlap: false })),
    [ordered],
  )

  const speakerName = (n: number) => {
    const s = speakerByNumber.get(n)
    return s?.voice ? `S${n} · ${s.voice}` : `S${n}`
  }

  const renderBlock = (block: Block) => {
    const start = block.start as number
    const end = block.end as number
    const selected = block.id === selectedId
    const slot = block.speaker !== null ? speakerSlot(block.speaker) : null
    const moved = block.source === 'label' && block.speaker !== null && block.speaker !== block.proposed_speaker
    const suggested = hasSuggestion(block)
    return (
      <div
        key={block.id}
        ref={(el) => {
          if (el) blockRefs.current.set(block.id, el)
          else blockRefs.current.delete(block.id)
        }}
        role="button"
        tabIndex={-1}
        onPointerDown={(e) => startMove(e, block)}
        onDoubleClick={(e) => {
          e.stopPropagation()
          playSpan(start, end)
        }}
        title={`${block.word}  ${start.toFixed(2)}–${end.toFixed(2)} s${
          block.source !== 'label' ? ` · ${block.source}` : ''
        }${moved ? ` · moved from S${block.proposed_speaker ?? '?'}` : ''}${
          suggested
            ? ` · the voiceprint hears S${block.suggested_speaker} (margin ${block.suggestion_margin?.toFixed(2) ?? '?'}); V moves it`
            : ''
        }${
          block.estimated ? ' · position estimated, check it' : ''
        }`}
        className={cn(
          'group absolute top-1.5 bottom-1.5 flex cursor-grab touch-none items-center overflow-hidden rounded-sm border px-1 select-none',
          'font-devanagari text-[13px] leading-none whitespace-nowrap text-foreground/90 transition-colors',
          'data-[active]:font-semibold data-[active]:text-foreground',
          slot !== null ? BLOCK_CLASS[slot] : 'border-muted-foreground/60 bg-muted',
          block.estimated && 'border-dashed',
          suggested && 'outline-1 outline-offset-1 outline-info outline-dashed',
          block.source !== 'label' && 'border-2',
          selected && 'z-10 ring-2 ring-info ring-offset-1 ring-offset-background',
        )}
        style={{ left: start * pps, width: Math.max(4, (end - start) * pps) }}
      >
        <span
          onPointerDown={(e) => startResize(e, block, 'start')}
          className="absolute inset-y-0 left-0 w-1.5 cursor-ew-resize bg-foreground/0 hover:bg-foreground/25"
        />
        <span className="pointer-events-none mx-auto truncate px-0.5">{block.word}</span>
        {moved && !suggested && (
          <span className="pointer-events-none absolute top-0 right-1 font-mono text-[8px] text-muted-foreground">
            ↕
          </span>
        )}
        {suggested && (
          <span className="pointer-events-none absolute -top-px right-0 rounded-bl-sm bg-info px-0.5 font-mono text-[8px] leading-3 text-info-foreground">
            →S{block.suggested_speaker}
          </span>
        )}
        <span
          onPointerDown={(e) => startResize(e, block, 'end')}
          className="absolute inset-y-0 right-0 w-1.5 cursor-ew-resize bg-foreground/0 hover:bg-foreground/25"
        />
      </div>
    )
  }

  const laneLabel = (row: Row) => {
    if (row.kind === 'speaker') {
      const slot = speakerSlot(row.speaker)
      const count = blocks.filter((b) => b.speaker === row.speaker).length
      const speaker = speakerByNumber.get(row.speaker)
      const voice = speaker?.voice ?? null
      return (
        <div className="flex w-full items-center gap-2">
          <span className={cn('size-2.5 shrink-0 rounded-full', SWATCH_CLASS[slot])} />
          <div className="min-w-0">
            <div className="flex items-center gap-1 font-mono text-xs">
              S{row.speaker}
              {voice && (
                <button
                  type="button"
                  onClick={() => {
                    stopVoice()
                    audioRef.current?.pause()
                    setVoiceOpen(voice)
                  }}
                  className="rounded-sm px-0.5 text-info underline-offset-2 hover:underline"
                  title={`${voice}: open this voice's page, its solo clips across every episode`}
                >
                  · {voice}
                </button>
              )}
            </div>
            <div
              className="truncate text-[10px] text-muted-foreground"
              title={
                speaker?.print_source === 'confirmed'
                  ? `Voiceprint from ${speaker.print_clips} clip(s) you confirmed`
                  : speaker?.print_source === 'diarizer'
                    ? "Voiceprint is the diarizer's centroid; confirm clips on the voice page to replace it"
                    : 'No voiceprint'
              }
            >
              {count} word{count === 1 ? '' : 's'}
              {!lanes.clip_speakers.includes(row.speaker) && ' · not here'}
              {speaker?.print_source === 'confirmed' && ` · ✓${speaker.print_clips}`}
            </div>
          </div>
          <div className="ml-auto flex items-center gap-1">
            {voice && (
              <button
                type="button"
                onClick={() => {
                  audioRef.current?.pause()
                  playVoice(voice, segment.episode_external_id)
                }}
                className="flex size-5 items-center justify-center rounded-sm text-muted-foreground hover:bg-muted hover:text-foreground"
                aria-label={`Hear ${voice}`}
                title={`Hear ${voice} alone`}
              >
                {playingVoice === voice ? (
                  <RiPauseFill className="size-3.5" />
                ) : (
                  <RiUserVoiceLine className="size-3.5" />
                )}
              </button>
            )}
            <Kbd>{row.speaker}</Kbd>
          </div>
        </div>
      )
    }
    if (row.kind === 'unplaced') {
      return (
        <div>
          <div className="text-xs font-medium text-warning">On no lane</div>
          <div className="text-[10px] text-muted-foreground">place before saving</div>
        </div>
      )
    }
    return (
      <div>
        <div className="text-xs font-medium text-muted-foreground">Recognisers heard</div>
        <div className="text-[10px] text-muted-foreground">drag onto a lane</div>
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <audio ref={audioRef} src={resolveUrl(segment.audio_url)} preload="auto" />

      {/* Header */}
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
          <Chip className="bg-info/15 text-info">speakers</Chip>
          {segment.pot === 'gold' && (
            <Chip className="bg-amber-500/15 text-amber-700 dark:text-amber-300">gold</Chip>
          )}
          <Chip>{segment.episode_external_id}</Chip>
          <Chip>
            {lanes.base === 'speakers'
              ? 'reopened'
              : lanes.base === 'label'
                ? 'from verified text'
                : 'from seed'}
          </Chip>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="text-warning hover:bg-warning/10 hover:text-warning"
                onClick={() =>
                  onFlag(task.id, 'uncertain', getDurationMs(), 'diarizer merged two voices')
                }
              >
                Voices merged
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Two people share one lane. Speakers are fixed here, so this is flagged for an
              episode-level fix.
            </TooltipContent>
          </Tooltip>
          <Button
            variant="ghost"
            size="sm"
            className="text-warning hover:bg-warning/10 hover:text-warning"
            onClick={() => onFlag(task.id, 'uncertain', getDurationMs())}
          >
            Uncertain
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => onFlag(task.id, 'unusable_audio', getDurationMs())}
          >
            Unusable audio
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onSkip(task.id, getDurationMs())}>
            Skip
          </Button>
        </div>
      </div>

      <div className="scrollbar-thin flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
        {/* Transport */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <ButtonGroup>
              <Button
                variant="outline"
                size="sm"
                className="font-mono normal-case"
                onClick={() => seek((audioRef.current?.currentTime ?? 0) - 2)}
                title="Back 2 s (Alt+←)"
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
                onClick={() => seek((audioRef.current?.currentTime ?? 0) + 2)}
                title="Forward 2 s (Alt+→)"
              >
                +2s
              </Button>
            </ButtonGroup>
            <span className="font-mono text-xs text-muted-foreground tabular-nums">
              {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
            </span>
            <Button
              variant={isLooping ? 'secondary' : 'outline'}
              size="sm"
              onClick={() => setIsLooping((v) => !v)}
              className={cn(isLooping && 'text-info')}
              title="Loop the clip (Ctrl+L)"
            >
              <RiRepeat2Line data-icon="inline-start" />
              Loop {isLooping ? 'on' : 'off'}
            </Button>
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

          <div className="flex items-center gap-2">
            <ButtonGroup>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={undo}
                disabled={history.length === 0}
                aria-label="Undo"
                title="Undo (Ctrl+Z)"
              >
                <RiArrowGoBackLine />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={redo}
                disabled={future.length === 0}
                aria-label="Redo"
                title="Redo (Ctrl+Shift+Z)"
              >
                <RiArrowGoForwardLine />
              </Button>
            </ButtonGroup>
            <ButtonGroup>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={() => setPps((v) => clamp(Math.round(v / 1.4), MIN_PPS, MAX_PPS))}
                aria-label="Zoom out"
                title="Zoom out (-)"
              >
                <RiZoomOutLine />
              </Button>
              <Button
                variant="outline"
                size="icon-sm"
                onClick={() => setPps((v) => clamp(Math.round(v * 1.4), MIN_PPS, MAX_PPS))}
                aria-label="Zoom in"
                title="Zoom in (+)"
              >
                <RiZoomInLine />
              </Button>
            </ButtonGroup>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" disabled={hiddenSpeakers.length === 0}>
                  <RiAddLine data-icon="inline-start" />
                  Show speaker
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>The episode's other speakers</DropdownMenuLabel>
                {hiddenSpeakers.map((n) => (
                  <DropdownMenuItem key={n} onSelect={() => showLane(n)}>
                    <span className={cn('size-2 rounded-full', SWATCH_CLASS[speakerSlot(n)])} />
                    {speakerName(n)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Timeline */}
        <div className="bg-card ring-1 ring-foreground/5">
          <div ref={scrollRef} className="scrollbar-thin relative overflow-x-auto">
            <div className="relative" style={{ width: GUTTER + width + 24 }}>
              {/* Ruler */}
              <div className="flex h-6 border-b">
                <div className="sticky left-0 z-20 shrink-0 border-r bg-card" style={{ width: GUTTER }} />
                <div
                  className="relative cursor-pointer"
                  style={{ width }}
                  onPointerDown={(e) => seek(timeAt(e, e.currentTarget))}
                >
                  {ticks.map((t) => {
                    const major = Math.abs(t - Math.round(t)) < 1e-6
                    return (
                      <div
                        key={t}
                        className={cn(
                          'absolute bottom-0 w-px',
                          major ? 'h-3 bg-muted-foreground/70' : 'h-1.5 bg-muted-foreground/40',
                        )}
                        style={{ left: t * pps }}
                      >
                        {major && (
                          <span className="absolute -top-0.5 left-1 font-mono text-[10px] text-muted-foreground">
                            {t.toFixed(0)}s
                          </span>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Waveform */}
              <div className="flex border-b">
                <div
                  className="sticky left-0 z-20 flex shrink-0 items-center border-r bg-card px-3 text-[10px] tracking-widest text-muted-foreground uppercase"
                  style={{ width: GUTTER }}
                >
                  Mix
                </div>
                <div style={{ width }}>
                  <Waveform
                    peaks={peaks}
                    currentTime={currentTime}
                    duration={duration}
                    onSeek={seek}
                    className="h-16"
                  />
                </div>
              </div>

              {/* Lanes */}
              {rows.map((row) => (
                <div key={row.key} className="flex border-b last:border-b-0">
                  <div
                    className={cn(
                      'sticky left-0 z-20 flex shrink-0 items-center border-r bg-card px-3',
                      row.kind === 'candidates' && 'bg-muted/60',
                    )}
                    style={{ width: GUTTER, height: LANE_H }}
                  >
                    {laneLabel(row)}
                  </div>
                  <div
                    ref={(el) => {
                      if (el) rowRefs.current.set(row.key, el)
                      else rowRefs.current.delete(row.key)
                    }}
                    className={cn(
                      'relative',
                      row.kind === 'candidates' && 'bg-muted/30',
                      row.kind === 'unplaced' && 'bg-warning/5',
                      dropLane === row.key && 'bg-info/10',
                    )}
                    style={{ width, height: LANE_H }}
                    onPointerDown={(e) => {
                      if (e.target !== e.currentTarget) return
                      setSelectedId(null)
                      seek(timeAt(e, e.currentTarget))
                    }}
                    onDoubleClick={(e) => {
                      if (row.kind !== 'speaker' || e.target !== e.currentTarget) return
                      setDraftAt({ speaker: row.speaker, time: timeAt(e, e.currentTarget) })
                      inputRef.current?.focus()
                    }}
                  >
                    {/* Where the diarizer heard this speaker. */}
                    {row.kind === 'speaker' &&
                      (turnsBySpeaker.get(row.speaker) ?? []).map((turn, i) => (
                        <div
                          key={i}
                          className={cn(
                            'pointer-events-none absolute inset-y-0',
                            TURN_CLASS[speakerSlot(row.speaker)],
                          )}
                          style={{ left: turn.start * pps, width: (turn.end - turn.start) * pps }}
                        />
                      ))}
                    {/* Where the overlap detector heard two voices at once (D77). */}
                    {row.kind !== 'candidates' &&
                      (segment.overlap_spans ?? []).map(([a, b], i) => (
                        <div
                          key={`o${i}`}
                          className="pointer-events-none absolute inset-y-0 bg-[repeating-linear-gradient(135deg,transparent_0_5px,var(--color-warning)_5px_6px)] opacity-25"
                          style={{ left: a * pps, width: (b - a) * pps }}
                        />
                      ))}
                    {draftAt && row.kind === 'speaker' && draftAt.speaker === row.speaker && (
                      <div
                        className="pointer-events-none absolute inset-y-1 w-0.5 bg-info"
                        style={{ left: draftAt.time * pps }}
                      />
                    )}
                    {row.kind === 'speaker' &&
                      blocks.filter((b) => b.speaker === row.speaker).map(renderBlock)}
                    {row.kind === 'unplaced' &&
                      blocks.filter((b) => b.speaker === null).map(renderBlock)}
                    {row.kind === 'candidates' &&
                      candidates.map((c) => (
                        <div
                          key={c.id}
                          role="button"
                          tabIndex={-1}
                          onPointerDown={(e) => startCandidate(e, c)}
                          onDoubleClick={() => playSpan(c.start, c.end)}
                          title={`${c.word}  ${c.start.toFixed(2)}–${c.end.toFixed(2)} s · heard by ${c.systems.join(', ')} · drag onto a lane, or select and press its number`}
                          className={cn(
                            'absolute top-1.5 bottom-1.5 flex cursor-grab touch-none items-center overflow-hidden rounded-sm border border-dashed border-muted-foreground/60 bg-background/60 px-1 select-none',
                            'font-devanagari text-[13px] leading-none whitespace-nowrap text-muted-foreground',
                            c.id === selectedId && 'z-10 ring-2 ring-info ring-offset-1 ring-offset-background',
                          )}
                          style={{ left: c.start * pps, width: Math.max(4, (c.end - c.start) * pps) }}
                        >
                          <span className="pointer-events-none mx-auto truncate">{c.word}</span>
                        </div>
                      ))}
                  </div>
                </div>
              ))}

              {/* Playhead, across ruler, waveform and every lane. */}
              <div
                ref={playheadRef}
                className="pointer-events-none absolute top-0 bottom-0 z-10 w-px bg-info"
                style={{ left: GUTTER }}
              />
            </div>
          </div>

          {/* Add a word */}
          <div className="flex flex-wrap items-center gap-2 border-t px-3 py-2">
            <div className="relative">
              <Input
                ref={inputRef}
                value={draft}
                onChange={(e) => {
                  setDraft(e.target.value)
                  setChoices(null)
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.ctrlKey) {
                    e.preventDefault()
                    submitDraft()
                  } else if (e.key === 'Escape') {
                    e.preventDefault()
                    e.stopPropagation()
                    if (choices) setChoices(null)
                    else {
                      setDraftAt(null)
                      ;(e.target as HTMLInputElement).blur()
                    }
                  } else if (choices && (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'Tab')) {
                    e.preventDefault()
                    setChoiceIndex((i) => (i + 1) % choices.length)
                  } else if (choices && (e.key === 'ArrowLeft' || e.key === 'ArrowUp')) {
                    e.preventDefault()
                    setChoiceIndex((i) => (i - 1 + choices.length) % choices.length)
                  }
                }}
                placeholder={
                  draftAt
                    ? `Word for S${draftAt.speaker} at ${draftAt.time.toFixed(2)}s`
                    : 'Add a word at the playhead (A)'
                }
                className="h-8 w-64 font-devanagari"
              />
            </div>
            {choices && (
              <div className="flex flex-wrap items-center gap-1">
                {choices.map((choice, i) => (
                  <button
                    key={choice}
                    type="button"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => {
                      if (choice !== draft) api.translitChoice(draft, choice).catch(() => {})
                      addWord(choice)
                    }}
                    className={cn(
                      'rounded-sm border px-2 py-0.5 font-devanagari text-sm',
                      i === choiceIndex ? 'border-info bg-info/15' : 'border-border',
                    )}
                  >
                    {choice}
                  </button>
                ))}
              </div>
            )}
            <Button size="sm" variant="outline" onClick={submitDraft} disabled={!draft.trim()}>
              Add
            </Button>
            <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Switch checked={devanagari} onCheckedChange={setDevanagari} size="sm" />
              Devanagari
            </label>
            <span className="ml-auto text-[11px] text-muted-foreground">
              <Kbd>1</Kbd>–<Kbd>9</Kbd> move to lane · <Kbd>⇧</Kbd>+number copy · <Kbd>←</Kbd>
              <Kbd>→</Kbd> next word · <Kbd>↵</Kbd> play word · <Kbd>Del</Kbd> remove added · <Kbd>V</Kbd> take suggestion ·
              double-click a lane to add there
            </span>
          </div>
        </div>

        {/* Per speaker */}
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="bg-card ring-1 ring-foreground/5">
            <div className="flex h-9 items-center border-b px-3 font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
              Per speaker
            </div>
            <div className="flex flex-col gap-2 p-3">
              {rows
                .filter((r): r is Extract<Row, { kind: 'speaker' }> => r.kind === 'speaker')
                .map((row) => (
                  <div key={row.key} className="flex gap-3">
                    <span className="flex w-16 shrink-0 items-center gap-1.5 font-mono text-xs text-muted-foreground">
                      <span className={cn('size-2 rounded-full', SWATCH_CLASS[speakerSlot(row.speaker)])} />
                      S{row.speaker}
                    </span>
                    <span className="font-devanagari text-sm leading-7">
                      {laneText(blocks, row.speaker) || (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </span>
                  </div>
                ))}
            </div>
          </div>
          <div className="bg-card ring-1 ring-foreground/5">
            <div className="flex h-9 items-center border-b px-3 font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
              Mix, in time order
            </div>
            <KaraokeTranscript
              words={mixWords}
              voices={mixVoices}
              resetKey={task.id}
              audioRef={audioRef}
              onSeekWord={(time) => seek(time, true)}
            />
          </div>
        </div>
      </div>

      {/* Action bar */}
      <div className="flex h-14 shrink-0 flex-wrap items-center justify-between gap-3 border-t bg-card/40 px-4">
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span>
            <span className="font-mono text-foreground tabular-nums">{blocks.length}</span> words
          </span>
          <Separator orientation="vertical" className="h-4" />
          <span>
            moved <span className="font-mono text-foreground tabular-nums">{summary.moved}</span>
          </span>
          <span>
            added <span className="font-mono text-foreground tabular-nums">{summary.added}</span>
          </span>
          {summary.unplaced > 0 && (
            <span className="text-warning">
              on no lane <span className="font-mono tabular-nums">{summary.unplaced}</span>
            </span>
          )}
          {lanes.voiceprint && (
            <button
              type="button"
              className="hover:text-foreground"
              onClick={() => takeSuggestions(null)}
              title="Move every word to the lane its voiceprint suggests (Shift+V)"
            >
              voiceprint suggests{' '}
              <span className="font-mono text-info tabular-nums">{summary.suggested}</span>
            </button>
          )}
          <span>
            recogniser words left{' '}
            <span className="font-mono text-foreground tabular-nums">{candidates.length}</span>
          </span>
        </div>
        <Button size="sm" disabled={isSubmitting} onClick={handleSave}>
          Save &amp; next
          <Kbd className="ml-1 bg-primary-foreground/15 text-primary-foreground">Ctrl+↵</Kbd>
        </Button>
      </div>
      <VoiceDialog
        voice={voiceOpen}
        episode={segment.episode_external_id}
        onClose={() => setVoiceOpen(null)}
      />
    </div>
  )
}
