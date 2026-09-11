import type { ReactNode } from 'react'

export type BadgeVariant = 'unreachable' | 'live' | 'neutral'

interface Props {
  variant?: BadgeVariant
  children: ReactNode
  className?: string
  /** Declared and forwarded explicitly - TSX accepts an unknown `data-*` prop on a
   * component without complaint, so without this it is silently dropped (the same trap
   * Checkbox hit). */
  'data-testid'?: string
}

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  unreachable: 'bg-data-unreachable/15 text-data-unreachable',
  live: 'bg-fg/12 text-fg',
  neutral: 'bg-fg/8 text-muted',
}

/** Short status label next to an entity - spec 5.9. Color is never the only carrier of
 * meaning: the text itself names the state (e.g. "unreachable"), the color reinforces it. */
export default function Badge({
  variant = 'neutral',
  children,
  className = '',
  'data-testid': testId,
}: Props) {
  return (
    <span
      data-testid={testId}
      className={`inline-flex h-[18px] items-center rounded-[3px] px-1.5 font-mono text-[10px] font-medium uppercase ${VARIANT_CLASSES[variant]} ${className}`}
    >
      {children}
    </span>
  )
}
