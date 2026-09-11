// Uniform-index decimation for point-cloud rendering. Picking the first N points would
// silently cut off whatever part of the room happened to be scanned last (or first),
// so instead we spread the kept indices evenly across the whole range.

/** Returns the indices to keep: `min(pointCount, maxPoints)` of them, evenly spaced
 * across `[0, pointCount)`. A no-op (identity 0..pointCount-1) when already under
 * budget. */
export function decimateIndices(pointCount: number, maxPoints: number): Uint32Array {
  if (pointCount <= 0) return new Uint32Array(0)
  if (maxPoints <= 0 || pointCount <= maxPoints) {
    const identity = new Uint32Array(pointCount)
    for (let i = 0; i < pointCount; i++) identity[i] = i
    return identity
  }

  const indices = new Uint32Array(maxPoints)
  const step = pointCount / maxPoints
  for (let i = 0; i < maxPoints; i++) {
    indices[i] = Math.min(pointCount - 1, Math.floor(i * step))
  }
  return indices
}
