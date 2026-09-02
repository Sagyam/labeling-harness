import React, { useEffect, useRef, useState } from 'react'

import { Kbd } from '@/components/ui/kbd'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { api } from '@/services/api'
import { cn } from '@/lib/utils'

interface TranslitEditorProps {
  value: string
  onChange: (value: string) => void
  textareaRef?: React.RefObject<HTMLTextAreaElement | null>
  disabled?: boolean
  onPopupOpenChange?: (isOpen: boolean) => void
}

interface PopupState {
  isOpen: boolean
  token: string
  startPos: number
  endPos: number
  candidates: string[]
  selectedIndex: number
  left: number
  top: number
}

// Simple token regex: consecutive Latin characters
const LATIN_TOKEN_REGEX = /^[a-zA-Z]+$/

export function TranslitEditor({
  value,
  onChange,
  textareaRef: externalRef,
  disabled = false,
  onPopupOpenChange,
}: TranslitEditorProps) {
  const internalRef = useRef<HTMLTextAreaElement | null>(null)
  const textarea = externalRef || internalRef

  const [translitEnabled, setTranslitEnabled] = useState(true)
  const [popup, setPopup] = useState<PopupState | null>(null)

  useEffect(() => {
    onPopupOpenChange?.(!!popup?.isOpen)
  }, [popup?.isOpen, onPopupOpenChange])

  // Measure pixel coordinate of cursor in textarea
  const getCaretCoordinates = (el: HTMLTextAreaElement, position: number) => {
    const div = document.createElement('div')
    const style = window.getComputedStyle(el)
    for (const prop of Array.from(style)) {
      div.style.setProperty(prop, style.getPropertyValue(prop))
    }
    div.style.position = 'absolute'
    div.style.visibility = 'hidden'
    div.style.whiteSpace = 'pre-wrap'
    div.style.wordWrap = 'break-word'
    div.style.top = '0px'
    div.style.left = '0px'
    div.style.width = `${el.clientWidth}px`
    div.style.height = 'auto'

    div.textContent = el.value.substring(0, position)
    const span = document.createElement('span')
    span.textContent = el.value.substring(position) || '.'
    div.appendChild(span)

    document.body.appendChild(div)
    const spanRect = span.getBoundingClientRect()
    const divRect = div.getBoundingClientRect()
    document.body.removeChild(div)

    const left = Math.min(el.clientWidth - 200, Math.max(10, spanRect.left - divRect.left))
    const top = spanRect.top - divRect.top + 26
    return { left, top }
  }

  // Check word before cursor and trigger translit popup
  const checkAndTriggerTranslit = async (text: string, caretPos: number) => {
    if (!translitEnabled || disabled) return false

    // Find token boundary ending at caretPos
    let start = caretPos - 1
    while (start >= 0 && /[a-zA-Z]/.test(text[start])) {
      start--
    }
    start++ // first Latin char
    const token = text.substring(start, caretPos)

    if (token.length > 0 && LATIN_TOKEN_REGEX.test(token)) {
      try {
        const res = await api.translit(token, 5)
        if (res.candidates && res.candidates.length > 0 && textarea.current) {
          const coords = getCaretCoordinates(textarea.current, caretPos)
          setPopup({
            isOpen: true,
            token,
            startPos: start,
            endPos: caretPos,
            candidates: res.candidates,
            selectedIndex: 0,
            left: coords.left,
            top: coords.top,
          })
          return true
        }
      } catch (err) {
        console.error('Translit lookup error:', err)
      }
    }
    return false
  }

  // Apply chosen candidate
  const applyCandidate = (candidate: string) => {
    if (!popup || !textarea.current) return

    const { token, startPos, endPos } = popup
    const before = value.substring(0, startPos)
    const after = value.substring(endPos)
    const replacement = candidate + ' '
    onChange(before + replacement + after)

    // Train correction memory
    api.translitChoice(token, candidate).catch(() => {})

    setPopup(null)

    // Position cursor after inserted candidate + space
    setTimeout(() => {
      if (textarea.current) {
        const newCursorPos = startPos + replacement.length
        textarea.current.focus()
        textarea.current.setSelectionRange(newCursorPos, newCursorPos)
      }
    }, 10)
  }

  // Dismiss popup and keep Latin as typed (crucial for English words)
  const dismissPopup = (insertSpace: boolean = false) => {
    if (!popup || !textarea.current) return
    const caret = popup.endPos
    setPopup(null)

    if (insertSpace) {
      const before = value.substring(0, caret)
      const after = value.substring(caret)
      onChange(before + ' ' + after)
      setTimeout(() => {
        if (textarea.current) {
          textarea.current.focus()
          textarea.current.setSelectionRange(caret + 1, caret + 1)
        }
      }, 10)
    } else {
      textarea.current.focus()
    }
  }

  // Keyboard navigation inside textarea & popup
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Quick toggle transliteration: Ctrl+T
    if (e.ctrlKey && (e.key === 't' || e.key === 'T')) {
      e.preventDefault()
      setTranslitEnabled((prev) => !prev)
      if (popup?.isOpen) setPopup(null)
      return
    }

    // When popup is open: handle candidate selection keys
    if (popup?.isOpen) {
      // Numbers 1-5 select corresponding candidate
      if (/^[1-5]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1
        if (idx < popup.candidates.length) {
          e.preventDefault()
          applyCandidate(popup.candidates[idx])
          return
        }
      }

      // Enter selects highlighted or first candidate
      if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault()
        applyCandidate(popup.candidates[popup.selectedIndex || 0])
        return
      }

      // Space also accepts the primary candidate and advances
      if (e.key === ' ') {
        e.preventDefault()
        applyCandidate(popup.candidates[popup.selectedIndex || 0])
        return
      }

      // Arrow navigation
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setPopup((prev) =>
          prev
            ? { ...prev, selectedIndex: (prev.selectedIndex + 1) % prev.candidates.length }
            : null,
        )
        return
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setPopup((prev) =>
          prev
            ? {
                ...prev,
                selectedIndex:
                  (prev.selectedIndex - 1 + prev.candidates.length) % prev.candidates.length,
              }
            : null,
        )
        return
      }

      // Escape keeps the Latin as typed!
      if (e.key === 'Escape') {
        e.preventDefault()
        e.stopPropagation()
        dismissPopup(false)
        return
      }

      // Tab key
      if (e.key === 'Tab') {
        e.preventDefault()
        applyCandidate(popup.candidates[popup.selectedIndex || 0])
        return
      }
    }

    // If popup is not open, check if user pressed Space after a Latin token
    if (e.key === ' ' && !e.ctrlKey && !e.altKey && translitEnabled && textarea.current) {
      const caretPos = textarea.current.selectionStart
      let start = caretPos - 1
      while (start >= 0 && /[a-zA-Z]/.test(value[start])) {
        start--
      }
      start++
      const token = value.substring(start, caretPos)

      if (token.length > 0 && LATIN_TOKEN_REGEX.test(token)) {
        e.preventDefault()
        checkAndTriggerTranslit(value, caretPos).then((opened) => {
          if (!opened && textarea.current) {
            // No candidates, just insert a regular space
            const before = value.substring(0, caretPos)
            const after = value.substring(caretPos)
            onChange(before + ' ' + after)
            setTimeout(() => {
              textarea.current?.setSelectionRange(caretPos + 1, caretPos + 1)
            }, 5)
          }
        })
      }
    }
  }

  return (
    <div className="relative flex min-h-0 flex-col bg-card ring-1 ring-foreground/5">
      <div className="flex h-10 shrink-0 items-center justify-between gap-2 border-b px-3">
        <span className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Active transcript
        </span>

        <div className="flex items-center gap-2">
          <Switch
            id="translit-toggle"
            checked={translitEnabled}
            onCheckedChange={(checked) => {
              setTranslitEnabled(checked)
              if (!checked) setPopup(null)
            }}
          />
          <Label htmlFor="translit-toggle" className="text-xs font-normal text-muted-foreground">
            Devanagari
          </Label>
          <Kbd>Ctrl+T</Kbd>
        </div>
      </div>

      <Textarea
        ref={textarea}
        className="min-h-0 flex-1 resize-none rounded-none border-0 bg-transparent p-3 font-devanagari text-base leading-8 shadow-none focus-visible:ring-0"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Transcript in Devanagari / English…"
        disabled={disabled}
      />

      {/* Floating Devanagari candidate popup */}
      {popup?.isOpen && (
        <div
          className="absolute z-50 w-56 bg-popover text-popover-foreground shadow-md ring-1 ring-foreground/10"
          style={{ left: `${popup.left}px`, top: `${popup.top + 40}px` }}
        >
          <div className="flex items-center justify-between gap-2 border-b px-2 py-1.5 text-[10px] tracking-wider text-muted-foreground uppercase">
            <span className="font-mono normal-case">{popup.token} →</span>
            <span>Esc keeps English</span>
          </div>

          <div className="py-1">
            {popup.candidates.map((cand, idx) => (
              <button
                key={cand}
                type="button"
                className={cn(
                  'flex w-full items-center justify-between px-2 py-1.5 text-left font-devanagari text-sm',
                  idx === popup.selectedIndex ? 'bg-accent text-accent-foreground' : 'hover:bg-muted',
                )}
                onClick={() => applyCandidate(cand)}
              >
                <span>{cand}</span>
                <Kbd>{idx + 1}</Kbd>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
