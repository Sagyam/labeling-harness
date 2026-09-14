/**
 * Reference against hypothesis, drawn from the folded alignment the backend counted (D83).
 *
 * Not the editor's `DiffViewer`: that one is a raw word LCS, and would mark `टिम`/`team` as an
 * error that the WER never charged. Here every coloured mark is exactly one counted error, so the
 * page and the number cannot disagree. A `fold` (one word, two scripts) and a `merge` (one word
 * split in two) are matches; they get a dotted underline so they can still be spotted.
 */

import type { AlignOp } from '@/types'

const MARK = {
  del: 'rounded-sm bg-rose-500/15 px-0.5 text-rose-700 line-through decoration-rose-500/70 dark:text-rose-300',
  ins: 'rounded-sm bg-sky-500/15 px-0.5 text-sky-700 dark:text-sky-300',
  subRef: 'text-rose-700/80 line-through decoration-rose-500/60 dark:text-rose-300/80',
  subHyp: 'text-amber-700 dark:text-amber-300',
  folded: 'underline decoration-dotted decoration-muted-foreground/70 underline-offset-4',
}

function Op({ op }: { op: AlignOp }) {
  const ref = op.ref.join(' ')
  const hyp = op.hyp.join(' ')
  switch (op.kind) {
    case 'match':
      return <span>{ref}</span>
    case 'fold':
    case 'merge':
      return (
        <span
          className={MARK.folded}
          title={`reference: ${ref} — ${op.kind === 'fold' ? 'same word, other script' : 'same word, split differently'} (not an error)`}
        >
          {hyp}
        </span>
      )
    case 'del':
      return (
        <span className={MARK.del} title="deleted: in the reference, missing from the model">
          {ref}
        </span>
      )
    case 'ins':
      return (
        <span className={MARK.ins} title="inserted: the model wrote a word nobody said">
          {hyp}
        </span>
      )
    case 'sub':
      return (
        <span
          className="rounded-sm bg-amber-500/10 px-0.5"
          title={`substituted — sounds ${Math.round(op.similarity * 100)}% alike`}
        >
          <span className={MARK.subRef}>{ref}</span>{' '}
          <span className={MARK.subHyp}>{hyp}</span>
        </span>
      )
  }
}

export function AlignedDiff({ ops }: { ops: AlignOp[] }) {
  if (ops.length === 0) {
    return <p className="text-sm text-muted-foreground">Both texts are empty.</p>
  }
  return (
    <p className="flex flex-wrap gap-x-1.5 gap-y-1 text-[15px] leading-relaxed">
      {ops.map((op, index) => (
        <Op key={index} op={op} />
      ))}
    </p>
  )
}

export function DiffLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
      <span className={MARK.del}>deleted</span>
      <span className={MARK.ins}>inserted</span>
      <span className="rounded-sm bg-amber-500/10 px-0.5">
        <span className={MARK.subRef}>ref</span> <span className={MARK.subHyp}>hyp</span>
      </span>
      <span className={MARK.folded}>folded (not an error)</span>
    </div>
  )
}
