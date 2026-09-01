import * as React from 'react'

import { cn } from '@/lib/utils'

/** Dense, squared metadata chip used for system ids, flags and split labels. */
export function Chip({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 bg-muted px-1.5 py-0.5 font-mono text-[10px] leading-4 tracking-wide text-muted-foreground whitespace-nowrap',
        className,
      )}
      {...props}
    />
  )
}
