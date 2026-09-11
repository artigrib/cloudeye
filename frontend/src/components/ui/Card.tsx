import { forwardRef } from 'react'
import type { HTMLAttributes } from 'react'

interface Props extends HTMLAttributes<HTMLDivElement> {
  /** Border brightens and surface lifts on hover - used for anything clickable
   * (project cards, video rows). Off for static cards. */
  hoverable?: boolean
}

/** The class string alone, for callers that need the Card look on a non-div element
 * (e.g. react-router's <Link> used as a whole card - spec 5.7). */
export function cardClassName(hoverable = false): string {
  return `rounded-md border-hair border-border-strong bg-surface p-4 transition-colors duration-100 ${
    hoverable ? 'hover:border-fg/30 hover:bg-surface-raised' : ''
  }`
}

const Card = forwardRef<HTMLDivElement, Props>(function Card({ hoverable, className = '', ...rest }, ref) {
  return <div ref={ref} className={`${cardClassName(hoverable)} ${className}`} {...rest} />
})

export default Card
