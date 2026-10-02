import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Kbd, KbdGroup } from '@/components/ui/kbd'
import { Glossary } from '@/components/Glossary'
import { cn } from '@/lib/utils'

export type HelpTab = 'glossary' | 'shortcuts'

interface KeyboardShortcutsModalProps {
  isOpen: boolean
  onClose: () => void
  /** Which tab is showing; the dialog is opened on one and can switch. */
  tab: HelpTab
  onTab: (tab: HelpTab) => void
}

type Shortcut = [description: string, keys: string[]]

const TRIAGE: Shortcut[] = [
  ['Move between rows', ['j', 'k']],
  ['Play / pause row audio', ['Space']],
  ['Accept unchanged, verified & advance', ['Enter']],
  ['Screen — accept without listening', ['s']],
  ['Add to gold / remove from gold', ['g']],
  ['Open in editor', ['e']],
  ['Flag unusable audio', ['f']],
  ['Mark uncertain', ['u']],
  ['Toggle row selection', ['x']],
  ['Accept selected rows, verified', ['Shift+Enter']],
  ['Screen selected rows', ['Shift+S']],
]

const EDITOR: Shortcut[] = [
  ['Play / pause audio (always)', ['Ctrl+Space']],
  ['Play / pause (textarea unfocused)', ['Space']],
  ['Save & advance to next', ['Ctrl+Enter']],
  ['Save & stay on segment', ['Ctrl+Shift+Enter']],
  ['Load hypothesis 1…5', ['Alt+1…5']],
  ['Seek audio -2s / +2s', ['Alt+Left', 'Alt+Right']],
  ['Toggle loop playback', ['Ctrl+L']],
  ['Toggle transliteration', ['Ctrl+T']],
  ['Exit editor to triage', ['Esc']],
]

const TRANSLIT: Shortcut[] = [
  ['Trigger popup after a Latin token', ['Space']],
  ['Select candidate 1 through 5', ['1', '…', '5']],
  ['Select primary candidate', ['Enter', 'Space']],
  ['Dismiss popup, keep Latin as typed', ['Esc']],
]

const NAVIGATION: Shortcut[] = [
  ['Switch to Triage view', ['1']],
  ['Switch to Editor view', ['2']],
  ['Switch to Episodes & Segments', ['3']],
  ['Switch to Analytics dashboard', ['4']],
  ['Switch to Dataset Export', ['5']],
  ['Switch to Cost Tracker', ['6']],
  ['Switch to Ingest', ['7']],
  ['Switch to Models', ['8']],
  ['Models: next / previous clip', ['j', 'k']],
  ['Models: play or pause / replay the clip', ['Space', 'r']],
  ['Toggle shortcuts reference', ['?']],
]

function ShortcutGroup({ title, shortcuts }: { title: string; shortcuts: Shortcut[] }) {
  return (
    <section className="flex flex-col gap-1">
      <h4 className="font-heading text-xs font-semibold tracking-widest text-muted-foreground uppercase">
        {title}
      </h4>
      <dl className="divide-y">
        {shortcuts.map(([description, keys]) => (
          <div key={description} className="flex items-center justify-between gap-4 py-1.5">
            <dt className="text-sm">{description}</dt>
            <dd>
              <KbdGroup>
                {keys.map((key) => (
                  <Kbd key={key}>{key}</Kbd>
                ))}
              </KbdGroup>
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

export function KeyboardShortcutsModal({ isOpen, onClose, tab, onTab }: KeyboardShortcutsModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[85vh] gap-4 overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Help</DialogTitle>
          <DialogDescription>
            What the harness's terms mean, and the keys every decision and view switch is bound to.
          </DialogDescription>
        </DialogHeader>

        <div className="flex gap-1 border-b" role="tablist">
          {(
            [
              ['glossary', 'Glossary'],
              ['shortcuts', 'Keyboard shortcuts'],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              onClick={() => onTab(id)}
              className={cn(
                '-mb-px border-b-2 px-3 py-1.5 text-sm',
                tab === id
                  ? 'border-primary font-semibold text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              )}
            >
              {label}
            </button>
          ))}
        </div>

        {tab === 'glossary' ? (
          <Glossary />
        ) : (
          <div className="grid gap-6 sm:grid-cols-2">
            <ShortcutGroup title="Global navigation" shortcuts={NAVIGATION} />
            <ShortcutGroup title="Triage mode" shortcuts={TRIAGE} />
            <ShortcutGroup title="Editor mode" shortcuts={EDITOR} />
            <ShortcutGroup title="Devanagari transliteration" shortcuts={TRANSLIT} />
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
