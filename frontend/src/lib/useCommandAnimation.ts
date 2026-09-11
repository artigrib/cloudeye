import { useCallback, useEffect, useRef, useState } from 'react'
import { buildPathTable, pointAtFraction } from './path-interp'
import type { CommandResponse, SceneObjectResponse } from '../api/types'

export interface AnimatedState {
  robotPosition: { x: number; z: number } | null
  activeObjectId: string | null
  pickedIds: Set<string>
  label: string | null
  playing: boolean
}

function distance3(
  o: SceneObjectResponse,
  p: [number, number, number],
): number {
  return (o.pos_x - p[0]) ** 2 + (o.pos_y - p[1]) ** 2 + (o.pos_z - p[2]) ** 2
}

function resolveObjectId(
  objects: SceneObjectResponse[],
  name: string | null,
  position: [number, number, number] | null,
): string | null {
  if (!name) return null
  const named = objects.filter((o) => o.name.toLowerCase() === name.toLowerCase())
  const pool = named.length ? named : objects.filter((o) => !o.is_fragment)
  if (!pool.length) return null
  if (!position) return pool[0].id
  return pool.reduce((best, o) => (distance3(o, position) < distance3(best, position) ? o : best)).id
}

/** Best-effort guess at which object this whole command is "about", so the map can
 * highlight it as soon as the robot starts moving - not only once it picks it up. */
function resolveCommandTarget(command: CommandResponse, objects: SceneObjectResponse[]): string | null {
  for (const step of command.steps) {
    if (step.object) return resolveObjectId(objects, step.object, step.position)
  }
  // `target` before `object`: a goto plans a single `move` step, which carries no
  // `object` at all, and its parsed action names the destination under `target`
  // ("action":"goto","target":"bed"). Reading only `object` therefore returned null for
  // EVERY goto - the one command shape the direct-goto path produces - so
  // `activeObjectId` was never set by a double-click or a Go press, and neither the
  // list's accent dot nor the live-target highlight had anything to key on. `object` is
  // kept for pick/place actions, which do use it.
  const parsed = command.parsed_action
  const parsedTarget = parsed?.target ?? parsed?.object
  if (typeof parsedTarget === 'string') return resolveObjectId(objects, parsedTarget, null)
  return null
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

export function useCommandAnimation(
  sceneKey: string | undefined,
  initialRobotPosition: { x: number; z: number } | null,
) {
  const [state, setState] = useState<AnimatedState>({
    robotPosition: initialRobotPosition,
    activeObjectId: null,
    pickedIds: new Set(),
    label: null,
    playing: false,
  })
  const generation = useRef(0)
  const rafId = useRef<number | null>(null)

  // On mount, `sceneKey`/`initialRobotPosition` are both still unknown (the scene
  // fetch is async) - useState's initializer above only runs once, so it locks in
  // `robotPosition: null` before the real coordinates ever exist, and the robot never
  // becomes visible without this. Also re-seeds if the user navigates to a different
  // scene without a full page reload (ScenePage isn't remounted on sceneId change),
  // but never overwrites an in-progress command/reset for the SAME scene - only fires
  // once per distinct sceneKey.
  const seededForRef = useRef<string | undefined>(undefined)
  useEffect(() => {
    if (initialRobotPosition && seededForRef.current !== sceneKey) {
      seededForRef.current = sceneKey
      setState((s) => ({ ...s, robotPosition: initialRobotPosition }))
    }
  }, [sceneKey, initialRobotPosition])

  const play = useCallback(
    async (command: CommandResponse, objects: SceneObjectResponse[]) => {
      const gen = ++generation.current
      if (rafId.current !== null) cancelAnimationFrame(rafId.current)
      const isCurrent = () => gen === generation.current

      if (command.status !== 'done') {
        // The failure itself is shown in the chat history, not here - showing it in both
        // places was a duplicate error message (see ChatPanel's failed-command bubble).
        setState((s) => ({ ...s, label: null, playing: false }))
        return
      }

      const targetId = resolveCommandTarget(command, objects)
      setState((s) => ({ ...s, activeObjectId: targetId, label: null, playing: true }))

      for (const step of command.steps) {
        if (!isCurrent()) return

        if (step.type === 'move' && step.path && step.path.length > 0) {
          const table = buildPathTable(step.path)
          const durationMs = Math.max(step.duration_sec, 0.001) * 1000
          const start = performance.now()
          setState((s) => ({ ...s, label: 'Moving…' }))
          await new Promise<void>((resolve) => {
            const tick = (now: number) => {
              if (!isCurrent()) return resolve()
              const fraction = (now - start) / durationMs
              const [x, z] = pointAtFraction(step.path!, table, fraction)
              setState((s) => ({ ...s, robotPosition: { x, z } }))
              if (fraction >= 1) {
                resolve()
              } else {
                rafId.current = requestAnimationFrame(tick)
              }
            }
            rafId.current = requestAnimationFrame(tick)
          })
        } else if (step.type === 'pick') {
          setState((s) => ({ ...s, label: step.object ? `Picking up ${step.object}…` : 'Picking up…' }))
          await delay(step.duration_sec * 1000)
          if (!isCurrent()) return
          const pickedId = resolveObjectId(objects, step.object, step.position)
          setState((s) => {
            const pickedIds = new Set(s.pickedIds)
            if (pickedId) pickedIds.add(pickedId)
            return { ...s, pickedIds }
          })
        } else if (step.type === 'place') {
          setState((s) => ({ ...s, label: step.at ? `Placing at ${step.at}…` : 'Placing…' }))
          await delay(step.duration_sec * 1000)
        }
      }

      // No "Done." The status badge exists to say what the robot is doing WHILE it is
      // doing it; a permanent "Done." after the last one is a label with no expiry that
      // sits in the left column for the rest of the session and is wrong the moment
      // anything else happens. `label: null` retires the badge instead, which is what the
      // reset path (and the pre-move state) already do.
      if (isCurrent()) setState((s) => ({ ...s, label: null, playing: false }))
    },
    [],
  )

  /** Snaps the robot straight to `pos` (no animation) and cancels any in-flight
   * command animation - backs the "return robot to start" control. */
  const reset = useCallback((pos: { x: number; z: number } | null) => {
    generation.current += 1
    if (rafId.current !== null) cancelAnimationFrame(rafId.current)
    setState({ robotPosition: pos, activeObjectId: null, pickedIds: new Set(), label: null, playing: false })
  }, [])

  return { state, play, reset }
}
