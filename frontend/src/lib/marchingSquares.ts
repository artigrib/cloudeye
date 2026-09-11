// Renderer spec section 5: vector obstacle contours drawn over the clearance field, so
// the border reads as a chart isoline rather than the raw 5cm cells ("one cell = 5
// screen pixels" at any real zoom level). Contours are vector, so they stay sharp at any
// zoom - the problem this replaces.

export type Point = [number, number]

interface Segment {
  a: Point
  b: Point
}

/** Binary mask, `mask[ix][iz]` - 1 = inside the contour (spec: the obstacle mask at
 * threshold 0.5, i.e. `cells[ix][iz] === 1` specifically, NOT clearanceField's
 * conservative "unknown counts as obstacle" - contours trace confirmed walls). */
export type BinaryMask = (0 | 1)[][]

/** One grid cell's contribution to the contour, derived from its four corner values
 * rather than a 16-case lookup table: an edge is "crossed" exactly where its two
 * corners differ (a basic topological fact - a cell boundary always has an even count
 * of sign changes around its perimeter), and for a binary 0/1 field the crossing point
 * is always the edge's exact midpoint (linear interpolation to the 0.5 threshold
 * between a 0 and a 1 corner is always t=0.5). Two crossed edges join with one segment;
 * four crossed edges is the ambiguous "saddle" case (diagonal corners agree, adjacent
 * corners don't), resolved by keeping the two same-valued diagonal corners each in
 * their own small triangle - a standard, deterministic (if imperfect) tie-break. */
function cellSegments(tl: number, tr: number, br: number, bl: number, ix: number, iz: number): Segment[] {
  const T: Point = [ix + 0.5, iz]
  const R: Point = [ix + 1, iz + 0.5]
  const B: Point = [ix + 0.5, iz + 1]
  const L: Point = [ix, iz + 0.5]

  const edgeT = tl !== tr
  const edgeR = tr !== br
  const edgeB = bl !== br
  const edgeL = tl !== bl
  const activeCount = (edgeT ? 1 : 0) + (edgeR ? 1 : 0) + (edgeB ? 1 : 0) + (edgeL ? 1 : 0)

  if (activeCount === 0) return []

  if (activeCount === 4) {
    return tl === br ? [{ a: L, b: T }, { a: R, b: B }] : [{ a: T, b: R }, { a: L, b: B }]
  }

  // Exactly 2 active edges (activeCount === 4 is handled above; an odd count is
  // topologically impossible) - connect them directly, order doesn't matter for a line.
  const pts: Point[] = []
  if (edgeT) pts.push(T)
  if (edgeR) pts.push(R)
  if (edgeB) pts.push(B)
  if (edgeL) pts.push(L)
  return [{ a: pts[0], b: pts[1] }]
}

function keyOf(p: Point): string {
  return `${p[0].toFixed(4)},${p[1].toFixed(4)}`
}

/** Stitches unordered edge-midpoint segments into continuous polylines by chaining
 * shared endpoints - greedy nearest-unused-neighbor, which is exact for the simple
 * (non-self-crossing) contours a binary occupancy mask produces. Closed loops end where
 * they started; polylines that hit the grid boundary end there instead. */
export function linkSegmentsIntoPolylines(segments: Segment[]): Point[][] {
  interface Entry {
    seg: number
    end: 'a' | 'b'
  }
  const adjacency = new Map<string, Entry[]>()
  function addAdjacency(key: string, entry: Entry) {
    const list = adjacency.get(key)
    if (list) list.push(entry)
    else adjacency.set(key, [entry])
  }
  segments.forEach((s, i) => {
    addAdjacency(keyOf(s.a), { seg: i, end: 'a' })
    addAdjacency(keyOf(s.b), { seg: i, end: 'b' })
  })

  const used = new Array<boolean>(segments.length).fill(false)
  const polylines: Point[][] = []

  for (let i = 0; i < segments.length; i++) {
    if (used[i]) continue
    used[i] = true
    const seg = segments[i]
    const line: Point[] = [seg.a, seg.b]

    for (let extending = true; extending; ) {
      extending = false
      const candidates = adjacency.get(keyOf(line[line.length - 1])) ?? []
      for (const c of candidates) {
        if (used[c.seg]) continue
        used[c.seg] = true
        const other = segments[c.seg]
        line.push(c.end === 'a' ? other.b : other.a)
        extending = true
        break
      }
    }

    for (let extending = true; extending; ) {
      extending = false
      const candidates = adjacency.get(keyOf(line[0])) ?? []
      for (const c of candidates) {
        if (used[c.seg]) continue
        used[c.seg] = true
        const other = segments[c.seg]
        line.unshift(c.end === 'a' ? other.b : other.a)
        extending = true
        break
      }
    }

    polylines.push(line)
  }

  return polylines
}

function perpendicularDistance(p: Point, a: Point, b: Point): number {
  const [x, y] = p
  const [x1, y1] = a
  const [x2, y2] = b
  const dx = x2 - x1
  const dy = y2 - y1
  const lenSq = dx * dx + dy * dy
  if (lenSq === 0) return Math.hypot(x - x1, y - y1)
  const t = ((x - x1) * dx + (y - y1) * dy) / lenSq
  return Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
}

/** Ramer-Douglas-Peucker: drops points that stay within `tolerance` of the line between
 * their neighbors - removes the "staircase" a binary grid's contour would otherwise
 * show at any zoom, without moving the line beyond what the grid's own resolution
 * justifies (spec: tolerance = half a cell). */
export function simplifyPolyline(points: Point[], tolerance: number): Point[] {
  if (points.length < 3) return points
  const end = points.length - 1
  let maxDist = 0
  let index = 0
  for (let i = 1; i < end; i++) {
    const d = perpendicularDistance(points[i], points[0], points[end])
    if (d > maxDist) {
      maxDist = d
      index = i
    }
  }
  if (maxDist > tolerance) {
    const left = simplifyPolyline(points.slice(0, index + 1), tolerance)
    const right = simplifyPolyline(points.slice(index), tolerance)
    return left.slice(0, -1).concat(right)
  }
  return [points[0], points[end]]
}

export interface ContourOptions {
  /** Simplification tolerance, in grid-cell units (spec: half a cell). */
  simplifyTolerance?: number
}

/** Full pipeline: binary mask -> per-cell segments -> linked polylines -> simplified.
 * Coordinates are in grid-cell units (not meters) - the caller scales by the grid's
 * resolution and offsets by its origin, same as every other consumer of `coords.ts`'s
 * cell<->world convention. */
export function computeContours(mask: BinaryMask, width: number, height: number, opts: ContourOptions = {}): Point[][] {
  const segments: Segment[] = []
  for (let ix = 0; ix < width - 1; ix++) {
    for (let iz = 0; iz < height - 1; iz++) {
      const tl = mask[ix]?.[iz] ?? 0
      const tr = mask[ix + 1]?.[iz] ?? 0
      const br = mask[ix + 1]?.[iz + 1] ?? 0
      const bl = mask[ix]?.[iz + 1] ?? 0
      segments.push(...cellSegments(tl, tr, br, bl, ix, iz))
    }
  }
  const tolerance = opts.simplifyTolerance ?? 0.5
  return linkSegmentsIntoPolylines(segments).map((line) => simplifyPolyline(line, tolerance))
}
