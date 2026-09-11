import { describe, expect, it } from 'vitest'
import {
  computeContours,
  linkSegmentsIntoPolylines,
  simplifyPolyline,
  type BinaryMask,
  type Point,
} from './marchingSquares'

function buildMask(width: number, height: number, inside: [number, number][]): BinaryMask {
  const mask: BinaryMask = Array.from({ length: width }, () => Array(height).fill(0) as (0 | 1)[])
  for (const [ix, iz] of inside) mask[ix][iz] = 1
  return mask
}

describe('computeContours', () => {
  it('produces no contour for an all-free or all-obstacle grid', () => {
    expect(computeContours(buildMask(4, 4, []), 4, 4)).toEqual([])
    const all = buildMask(4, 4, [])
    for (let ix = 0; ix < 4; ix++) for (let iz = 0; iz < 4; iz++) all[ix][iz] = 1
    expect(computeContours(all, 4, 4)).toEqual([])
  })

  it('traces a single-cell obstacle as one closed loop of 4 segments', () => {
    // A 1x1 obstacle in the middle of a 3x3 grid - exactly 4 cells have a nonzero case
    // (each corner-adjacent to the obstacle), producing one diamond-shaped loop.
    const mask = buildMask(3, 3, [[1, 1]])
    const contours = computeContours(mask, 3, 3, { simplifyTolerance: 0 })
    expect(contours).toHaveLength(1)
    const loop = contours[0]
    // Closed: first and last point coincide.
    expect(loop[0]).toEqual(loop[loop.length - 1])
    // The diamond's corners sit at the obstacle cell's edge midpoints: (1,0.5), (1.5,1),
    // (1,1.5), (0.5,1) in some rotation/direction.
    const expectedCorners: Point[] = [
      [1, 0.5],
      [1.5, 1],
      [1, 1.5],
      [0.5, 1],
    ]
    for (const corner of expectedCorners) {
      expect(loop.some(([x, y]) => Math.abs(x - corner[0]) < 1e-6 && Math.abs(y - corner[1]) < 1e-6)).toBe(true)
    }
  })

  it('traces a straight wall as a single simplified segment, not a staircase', () => {
    // A horizontal wall spanning ix=0..3 at iz=1, in a 4x3 grid - free above, obstacle
    // row, free below. The contour is a horizontal line at iz=0.5, simplifiable to its
    // two endpoints.
    const mask = buildMask(4, 3, [
      [0, 1],
      [1, 1],
      [2, 1],
      [3, 1],
    ])
    const contours = computeContours(mask, 4, 3, { simplifyTolerance: 0.5 })
    // Top edge of the wall and bottom edge of the wall are two separate open polylines
    // (the wall spans the full grid width, so neither closes into a loop).
    expect(contours.length).toBeGreaterThan(0)
    for (const line of contours) {
      // Every point on a given contour shares the same iz - simplification must not
      // have introduced any zig-zag.
      const ys = line.map(([, y]) => y)
      expect(new Set(ys).size).toBe(1)
      expect(line.length).toBe(2) // straight run -> collapses to just the two ends
    }
  })
})

describe('linkSegmentsIntoPolylines', () => {
  it('chains segments sharing endpoints into one polyline', () => {
    const segments = [
      { a: [0, 0] as Point, b: [1, 0] as Point },
      { a: [1, 0] as Point, b: [2, 0] as Point },
    ]
    const lines = linkSegmentsIntoPolylines(segments)
    expect(lines).toHaveLength(1)
    expect(lines[0]).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
    ])
  })

  it('keeps disjoint segments as separate polylines', () => {
    const segments = [
      { a: [0, 0] as Point, b: [1, 0] as Point },
      { a: [5, 5] as Point, b: [6, 5] as Point },
    ]
    expect(linkSegmentsIntoPolylines(segments)).toHaveLength(2)
  })
})

describe('simplifyPolyline', () => {
  it('drops collinear points', () => {
    const line: Point[] = [
      [0, 0],
      [1, 0],
      [2, 0],
    ]
    expect(simplifyPolyline(line, 0.01)).toEqual([
      [0, 0],
      [2, 0],
    ])
  })

  it('keeps a point that deviates beyond tolerance', () => {
    const line: Point[] = [
      [0, 0],
      [1, 1],
      [2, 0],
    ]
    expect(simplifyPolyline(line, 0.1)).toEqual(line)
  })

  it('drops a point that deviates within tolerance', () => {
    const line: Point[] = [
      [0, 0],
      [1, 0.01],
      [2, 0],
    ]
    expect(simplifyPolyline(line, 0.1)).toEqual([
      [0, 0],
      [2, 0],
    ])
  })

  it('leaves short polylines untouched', () => {
    const line: Point[] = [
      [0, 0],
      [1, 1],
    ]
    expect(simplifyPolyline(line, 100)).toEqual(line)
  })
})
