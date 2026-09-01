import React, { useEffect } from 'react'

interface KeyboardShortcutsModalProps {
  isOpen: boolean
  onClose: () => void
}

export const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({
  isOpen,
  onClose,
}) => {
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
            Keyboard Shortcuts Reference
          </h3>
          <button className="btn-icon" onClick={onClose} title="Close (Esc)">
            ✕
          </button>
        </div>

        <div className="modal-body">
          {/* Triage Mode */}
          <div className="shortcut-group">
            <h4>Triage Mode (List)</h4>
            <div className="shortcut-row">
              <span>Move between rows</span>
              <div>
                <kbd>j</kbd> / <kbd>k</kbd>
              </div>
            </div>
            <div className="shortcut-row">
              <span>Play / Pause row audio</span>
              <kbd>Space</kbd>
            </div>
            <div className="shortcut-row">
              <span>Accept unchanged & advance</span>
              <kbd>Enter</kbd>
            </div>
            <div className="shortcut-row">
              <span>Open in Editor</span>
              <kbd>e</kbd>
            </div>
            <div className="shortcut-row">
              <span>Flag unusable audio</span>
              <kbd>f</kbd>
            </div>
            <div className="shortcut-row">
              <span>Mark uncertain</span>
              <kbd>u</kbd>
            </div>
            <div className="shortcut-row">
              <span>Toggle row selection</span>
              <kbd>x</kbd>
            </div>
            <div className="shortcut-row">
              <span>Accept selected rows</span>
              <kbd>Shift+Enter</kbd>
            </div>
          </div>

          {/* Editor Mode */}
          <div className="shortcut-group">
            <h4>Editor Mode</h4>
            <div className="shortcut-row">
              <span>Play / Pause audio (always)</span>
              <kbd>Ctrl+Space</kbd>
            </div>
            <div className="shortcut-row">
              <span>Play / Pause (textarea unfocused)</span>
              <kbd>Space</kbd>
            </div>
            <div className="shortcut-row">
              <span>Save & advance to next</span>
              <kbd>Ctrl+Enter</kbd>
            </div>
            <div className="shortcut-row">
              <span>Save & stay on segment</span>
              <kbd>Ctrl+Shift+Enter</kbd>
            </div>
            <div className="shortcut-row">
              <span>Load hypothesis 1..5</span>
              <kbd>Alt+1..5</kbd>
            </div>
            <div className="shortcut-row">
              <span>Seek audio -2s / +2s</span>
              <div>
                <kbd>Alt+←</kbd> / <kbd>Alt+→</kbd>
              </div>
            </div>
            <div className="shortcut-row">
              <span>Toggle loop playback</span>
              <kbd>Ctrl+L</kbd>
            </div>
            <div className="shortcut-row">
              <span>Toggle Transliteration</span>
              <kbd>Ctrl+T</kbd>
            </div>
            <div className="shortcut-row">
              <span>Exit editor to triage</span>
              <kbd>Esc</kbd>
            </div>
          </div>

          {/* Transliteration Helper */}
          <div className="shortcut-group" style={{ gridColumn: 'span 2' }}>
            <h4>Devanagari Transliteration Helper (Phase 5)</h4>
            <div className="shortcut-row">
              <span>Trigger transliteration popup</span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Type Latin token and press <kbd>Space</kbd>
              </span>
            </div>
            <div className="shortcut-row">
              <span>Select candidate 1 through 5</span>
              <kbd>1</kbd> .. <kbd>5</kbd>
            </div>
            <div className="shortcut-row">
              <span>Select primary candidate</span>
              <div>
                <kbd>Enter</kbd> or <kbd>Space</kbd>
              </div>
            </div>
            <div className="shortcut-row">
              <span>Dismiss popup & keep Latin as typed (English)</span>
              <kbd>Esc</kbd>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
