import { useEffect, useRef, useState } from 'react'

const DURATION_MS = 400

function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3)
}

/** Animates a displayed integer toward `target` over 400ms ease-out whenever `target`
 * changes - the reachability count's "the number visibly changed" cue (spec 6.5), the
 * one unprovoked animation in the app besides the object-list stagger it pairs with.
 * `null` passes through immediately (nothing to count from/to yet). */
export function useCountUp(target: number | null): number | null {
  const [display, setDisplay] = useState(target)
  const fromRef = useRef(target)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (target === null) {
      setDisplay(null)
      fromRef.current = null
      return
    }
    const from = fromRef.current ?? target
    if (from === target) {
      setDisplay(target)
      return
    }
    const start = performance.now()
    function tick(now: number) {
      const t = Math.min(1, (now - start) / DURATION_MS)
      setDisplay(Math.round(from + (target! - from) * easeOut(t)))
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = target
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
    // Intentionally keyed only on `target`: `from` is read live off fromRef rather than
    // captured as a dependency, so a rapid second change restarts the tween from
    // wherever the first one currently is instead of snapping back to the old start.
  }, [target])

  return display
}
