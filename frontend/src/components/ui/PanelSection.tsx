import type { ReactNode } from 'react'

interface Props {
  title: string
  /** Rendered as "TITLE (count)" - spec 5.8, e.g. "OBJECTS (40)". */
  count?: number
  /** Compact ghost-button slot, right-aligned in the header row. */
  action?: ReactNode
  children?: ReactNode
  className?: string
}

/** Section header used inside the right-hand inspector panel (OBJECTS, COMMANDS, ...) -
 * spec 5.8. Caps + mono is a functional layer boundary in a dense panel, not decoration
 * (spec 4.3) - don't reuse this styling anywhere outside the inspector.
 *
 * The header is exactly one 32px row tall and carries the panel's own horizontal
 * padding, which is what makes "the section headers in the two panels sit on the same
 * baseline" a property of the primitive rather than something each screen re-establishes
 * by hand. The old top border is gone: --section-gap separates sections now, and a rule
 * plus a gap was saying the same thing twice. */
export default function PanelSection({ title, count, action, children, className = '' }: Props) {
  return (
    <div className={className}>
      {/* h-row and the panel's own padding: the header is one 32px row, so the first
         header in the left panel and the first in the right land on the same baseline
         without either knowing about the other. h-8 would be 30px here - Tailwind's rem
         spacing against this app's 15px root. */}
      <div className="flex h-row items-center justify-between gap-s2 px-s4">
        <span className="ui-monocaps truncate text-fg">
          {title}
          {count !== undefined && <span> ({count})</span>}
        </span>
        {action}
      </div>
      {children}
    </div>
  )
}
