// Pure selection/ordering logic for the "Auto tour" button - kept separate from the
// orchestration hook (useAutoTour) so the stop-selection rules are unit-testable without
// mounting anything. See useAutoTour.ts for the sequencing/cancellation half.

import type { SceneObjectResponse } from '../api/types'

/** Cap on how many stops a tour ever visits, even when more objects qualify - keeps a
 * demo recording to roughly 30-40s. See selectTourStops. */
export const TOUR_MAX_STOPS = 8

interface Point {
  x: number
  z: number
}

function dist2(a: Point, b: Point): number {
  const dx = a.x - b.x
  const dz = a.z - b.z
  return dx * dx + dz * dz
}

function toPoint(o: SceneObjectResponse): Point {
  return { x: o.pos_x, z: o.pos_z }
}

/** Greedy farthest-point sampling: picks `count` items out of `candidates` that spread
 * across the room, rather than clustering. Starts from whichever candidate sits farthest
 * from the group's centroid, then repeatedly adds whichever remaining candidate is
 * farthest from the nearest already-picked one. */
function pickSpread(candidates: SceneObjectResponse[], count: number): SceneObjectResponse[] {
  if (candidates.length <= count) return candidates.slice()

  const centroid = candidates.reduce(
    (acc, o) => ({ x: acc.x + o.pos_x, z: acc.z + o.pos_z }),
    { x: 0, z: 0 },
  )
  centroid.x /= candidates.length
  centroid.z /= candidates.length

  const remaining = candidates.slice()
  let seedIdx = 0
  let seedDist = -Infinity
  remaining.forEach((o, i) => {
    const d = dist2(toPoint(o), centroid)
    if (d > seedDist) {
      seedDist = d
      seedIdx = i
    }
  })
  const selected = [remaining.splice(seedIdx, 1)[0]]

  while (selected.length < count && remaining.length > 0) {
    let bestIdx = 0
    let bestMinDist = -Infinity
    remaining.forEach((o, i) => {
      const p = toPoint(o)
      let minDist = Infinity
      for (const s of selected) minDist = Math.min(minDist, dist2(p, toPoint(s)))
      if (minDist > bestMinDist) {
        bestMinDist = minDist
        bestIdx = i
      }
    })
    selected.push(remaining.splice(bestIdx, 1)[0])
  }

  return selected
}

/** Orders `items` into a reasonable visiting sequence via greedy nearest-neighbour,
 * starting from `from` - purely cosmetic (the spec doesn't mandate a visiting order),
 * just avoids an obviously zig-zaggy tour. */
function orderByNearestNeighbor(items: SceneObjectResponse[], from: Point): SceneObjectResponse[] {
  const remaining = items.slice()
  const ordered: SceneObjectResponse[] = []
  let cur = from
  while (remaining.length > 0) {
    let bestIdx = 0
    let bestDist = Infinity
    remaining.forEach((o, i) => {
      const d = dist2(toPoint(o), cur)
      if (d < bestDist) {
        bestDist = d
        bestIdx = i
      }
    })
    const next = remaining.splice(bestIdx, 1)[0]
    ordered.push(next)
    cur = toPoint(next)
  }
  return ordered
}

/**
 * Picks and orders the stops for an auto tour.
 *
 * Mandatory rules (see the demo spec this backs):
 * - Only objects currently marked reachable for the selected platform, never a fragment.
 * - Capped at `maxStops` even when more qualify - if more do, prefer a spread across the
 *   room over just the first N in `objects`' order.
 */
export function selectTourStops(
  objects: SceneObjectResponse[],
  reachableObjectIds: Set<string>,
  from: Point,
  maxStops: number = TOUR_MAX_STOPS,
): SceneObjectResponse[] {
  const candidates = objects.filter((o) => !o.is_fragment && reachableObjectIds.has(o.id))
  const picked = pickSpread(candidates, maxStops)
  return orderByNearestNeighbor(picked, from)
}
