// Renderer spec section 4: a continuous "distance to nearest obstacle" field over the
// occupancy grid, computed once per scene and re-colored (never recomputed) whenever the
// selected robot's radius changes - see the shader/material that consumes this for the
// cheap part; this file is only the expensive one-time computation.

const INF = 1e20

/** Felzenszwalb & Huttenlocher's exact 1D squared-distance transform ("Distance
 * Transforms of Sampled Functions", 2012) - the standard two-pass building block for a
 * separable 2D Euclidean distance transform. `f[i]` is the squared "seed" value at
 * index i (0 at an obstacle, INF at free space); returns the squared distance from each
 * index to the nearest 0. O(n), not O(n^2) - the naive per-cell nearest-obstacle search
 * this replaces would be far too slow for a 200x150 grid recomputed on every scene load. */
export function distanceTransform1D(f: Float64Array): Float64Array {
  const n = f.length
  const d = new Float64Array(n)
  if (n === 0) return d
  const v = new Int32Array(n)
  const z = new Float64Array(n + 1)
  let k = 0
  v[0] = 0
  z[0] = -Infinity
  z[1] = Infinity
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    while (s <= z[k]) {
      k--
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    }
    k++
    v[k] = q
    z[k] = s
    z[k + 1] = Infinity
  }
  k = 0
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]
  }
  return d
}

/** cells[ix][iz] - axis 0 = X, axis 1 = Z, matching GET /api/scenes/{id}/map (spec 9 in
 * the app doc, and app/schemas.py's OccupancyGridResponse). 0 = free, 1 = obstacle,
 * 2 = unknown. */
export type OccupancyCells = number[][]

/** Distance, in meters, from every cell to the nearest obstacle - `unknown` cells count
 * as obstacles too (conservative: a robot shouldn't trust unmapped space, spec 4.1
 * point 1). Flattened `[ix * height + iz]`, same axis convention as the input. Computed
 * once per scene; the robot-radius selector only changes how this field gets colored,
 * never recomputed itself (spec: "поле не зависит от робота"). */
export function computeClearanceField(
  cells: OccupancyCells,
  width: number,
  height: number,
  resolutionM: number,
): Float32Array {
  const isObstacle = (v: number | undefined) => v === 1 || v === 2

  // Seed: 0 at obstacles, INF at free - the 1D transform then gives each free cell's
  // squared distance to the nearest 0.
  const seed = new Float64Array(width * height)
  for (let ix = 0; ix < width; ix++) {
    const column = cells[ix]
    for (let iz = 0; iz < height; iz++) {
      seed[ix * height + iz] = isObstacle(column?.[iz]) ? 0 : INF
    }
  }

  // Pass 1: transform each column (fixed ix, varying iz).
  const pass1 = new Float64Array(width * height)
  const colBuf = new Float64Array(height)
  for (let ix = 0; ix < width; ix++) {
    for (let iz = 0; iz < height; iz++) colBuf[iz] = seed[ix * height + iz]
    const d = distanceTransform1D(colBuf)
    for (let iz = 0; iz < height; iz++) pass1[ix * height + iz] = d[iz]
  }

  // Pass 2: transform each row (fixed iz, varying ix) of pass1's result - the standard
  // separable 2D EDT: transforming along one axis and then the other over the first
  // pass's squared distances gives the true 2D squared Euclidean distance.
  const out = new Float32Array(width * height)
  const rowBuf = new Float64Array(width)
  for (let iz = 0; iz < height; iz++) {
    for (let ix = 0; ix < width; ix++) rowBuf[ix] = pass1[ix * height + iz]
    const d = distanceTransform1D(rowBuf)
    for (let ix = 0; ix < width; ix++) out[ix * height + iz] = Math.sqrt(d[ix]) * resolutionM
  }

  return out
}
