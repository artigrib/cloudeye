import { cloneElement, isValidElement, useRef, useState } from 'react'
import type { ReactElement, ReactNode } from 'react'

interface Props {
  content: ReactNode
  children: ReactElement
  disabled?: boolean
  /** Let the bubble wrap at this width instead of running on one line. A one-line
   * tooltip is right for a two-word hint and wrong for a sentence: centred on a trigger
   * near the left edge of the screen, a 500px nowrap bubble is drawn off the side of the
   * viewport where nobody can read it. */
  maxWidthPx?: number
}

const SHOW_DELAY_MS = 300

/** Replaces the native title="" attribute wherever a tooltip is load-bearing (spec:
 * hover hints, especially in the scene overlays). Positioned in fixed coordinates from
 * the trigger's own bounding box, so it escapes any overflow:hidden ancestor. */
export default function Tooltip({ content, children, disabled, maxWidthPx }: Props) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrapperRef = useRef<HTMLElement>(null)

  function show() {
    if (disabled) return
    timer.current = setTimeout(() => {
      const rect = wrapperRef.current?.getBoundingClientRect()
      if (rect) setPos({ x: rect.left + rect.width / 2, y: rect.top })
    }, SHOW_DELAY_MS)
  }

  function hide() {
    if (timer.current) clearTimeout(timer.current)
    setPos(null)
  }

  if (!isValidElement(children)) return children

  const trigger = cloneElement(children as ReactElement<Record<string, unknown>>, {
    ref: wrapperRef,
    onMouseEnter: show,
    onMouseLeave: hide,
    onFocus: show,
    onBlur: hide,
  })

  return (
    <>
      {trigger}
      {pos && (
        <div
          role="tooltip"
          className={`pointer-events-none fixed z-50 -translate-x-1/2 -translate-y-[calc(100%+6px)] rounded-sm border-hair border-border bg-surface-raised px-[7px] py-[3px] text-xs text-fg shadow-lg ${
            maxWidthPx ? '' : 'whitespace-nowrap'
          }`}
          style={{ left: pos.x, top: pos.y, maxWidth: maxWidthPx }}
        >
          {content}
        </div>
      )}
    </>
  )
}
