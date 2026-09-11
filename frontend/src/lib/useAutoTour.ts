import { useCallback, useEffect, useRef, useState } from 'react'
import type { SceneObjectResponse } from '../api/types'
import { TOUR_MAX_STOPS, selectTourStops } from './autoTour'

export interface UseAutoTourOptions {
  objects: SceneObjectResponse[]
  /** Posts the command for one stop and resolves once the robot has arrived there (or
   * the attempt was cut short) - see ScenePage's visitTourStop, which reuses the same
   * goto + animation path a double-click already takes. Never rejects. */
  visitStop: (stop: SceneObjectResponse) => Promise<void>
  /** A change while the tour is running stops it outright (see module doc on `stop`). */
  selectedPlatformId: string | null
  /** Fires exactly once when the tour is cut short - a manual stop or a platform
   * switch - never on natural completion, since the robot is already exactly where it
   * should be in that case. Typically used to halt the robot's on-screen position in
   * place (see ScenePage's reset(anim.robotPosition)). */
  onInterrupt: () => void
  maxStops?: number
}

export interface AutoTour {
  running: boolean
  /** No-op if a tour is already running, reachability/position aren't known yet, or
   * nothing reachable qualifies for a stop. */
  start: (reachableObjectIds: Set<string> | null, robotPosition: { x: number; z: number } | null) => void
  /** No-op if no tour is running. */
  stop: () => void
}

/** Drives the "Auto tour" button: picks stops via selectTourStops, then visits them one
 * at a time strictly in sequence (never in parallel, and never queuing the next stop
 * before the current one resolves) so each visitStop always plans "from" the robot's
 * true current position. Deliberately does no pathfinding/animation of its own - see
 * `visitStop`. */
export function useAutoTour({
  objects,
  visitStop,
  selectedPlatformId,
  onInterrupt,
  maxStops = TOUR_MAX_STOPS,
}: UseAutoTourOptions): AutoTour {
  const [running, setRunning] = useState(false)
  const runningRef = useRef(false)
  const cancelledRef = useRef(false)

  const visitStopRef = useRef(visitStop)
  visitStopRef.current = visitStop
  const onInterruptRef = useRef(onInterrupt)
  onInterruptRef.current = onInterrupt

  const stop = useCallback(() => {
    if (!runningRef.current) return
    cancelledRef.current = true
    runningRef.current = false
    setRunning(false)
    onInterruptRef.current()
  }, [])

  const start = useCallback(
    (reachableObjectIds: Set<string> | null, robotPosition: { x: number; z: number } | null) => {
      if (runningRef.current || !reachableObjectIds || !robotPosition) return
      const stops = selectTourStops(objects, reachableObjectIds, robotPosition, maxStops)
      if (stops.length === 0) return

      cancelledRef.current = false
      runningRef.current = true
      setRunning(true)

      void (async () => {
        for (const target of stops) {
          if (cancelledRef.current) break
          await visitStopRef.current(target)
          if (cancelledRef.current) break
        }
        // Natural completion (the loop ran out of stops without being cancelled) -
        // `stop()` already handled the cancelled case's own flip back to idle.
        if (runningRef.current) {
          runningRef.current = false
          setRunning(false)
        }
      })()
    },
    [objects, maxStops],
  )

  // Switching platform mid-tour stops it outright rather than recomputing a route for a
  // different footprint on the fly (see module doc) - a no-op via stop()'s own guard
  // whenever no tour is running, including the very first time selectedPlatformId
  // resolves from null to the default platform on load.
  const prevPlatformIdRef = useRef(selectedPlatformId)
  useEffect(() => {
    if (prevPlatformIdRef.current !== selectedPlatformId) {
      prevPlatformIdRef.current = selectedPlatformId
      stop()
    }
  }, [selectedPlatformId, stop])

  return { running, start, stop }
}
