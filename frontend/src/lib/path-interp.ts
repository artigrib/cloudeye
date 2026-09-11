// Pure geometry for animating a robot along a CommandStep's path - kept separate from
// the React hook so the interpolation math is unit-testable without fake timers.

export type PathPoint = [number, number]

export interface PathTable {
  totalLength: number
  /** cumulative length at each vertex, same length as the input path */
  cumulative: number[]
}

export function buildPathTable(path: PathPoint[]): PathTable {
  const cumulative = [0]
  for (let i = 1; i < path.length; i++) {
    const [x0, z0] = path[i - 1]
    const [x1, z1] = path[i]
    const segLen = Math.hypot(x1 - x0, z1 - z0)
    cumulative.push(cumulative[i - 1] + segLen)
  }
  return { totalLength: cumulative[cumulative.length - 1] ?? 0, cumulative }
}

/** Point on `path` at arc-length `distance` (clamped to [0, totalLength]). */
export function pointAtDistance(path: PathPoint[], table: PathTable, distance: number): PathPoint {
  if (path.length === 0) return [0, 0]
  if (path.length === 1) return path[0]

  const d = Math.min(Math.max(distance, 0), table.totalLength)
  // Find the segment whose cumulative range contains d.
  let i = 1
  while (i < table.cumulative.length - 1 && table.cumulative[i] < d) i++

  const segStart = table.cumulative[i - 1]
  const segEnd = table.cumulative[i]
  const segLen = segEnd - segStart
  const t = segLen > 0 ? (d - segStart) / segLen : 0

  const [x0, z0] = path[i - 1]
  const [x1, z1] = path[i]
  return [x0 + (x1 - x0) * t, z0 + (z1 - z0) * t]
}

/** Point on `path` at fraction (0..1) of its total duration/length. */
export function pointAtFraction(path: PathPoint[], table: PathTable, fraction: number): PathPoint {
  return pointAtDistance(path, table, table.totalLength * Math.min(Math.max(fraction, 0), 1))
}
