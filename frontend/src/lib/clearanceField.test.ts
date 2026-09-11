import { describe, expect, it } from 'vitest'
import { computeClearanceField, distanceTransform1D, type OccupancyCells } from './clearanceField'

const INF = 1e20

function buildGrid(width: number, height: number, obstacles: [number, number][], unknowns: [number, number][] = []): OccupancyCells {
  const cells: OccupancyCells = Array.from({ length: width }, () => Array(height).fill(0))
  for (const [ix, iz] of obstacles) cells[ix][iz] = 1
  for (const [ix, iz] of unknowns) cells[ix][iz] = 2
  return cells
}

describe('distanceTransform1D', () => {
  it('gives squared distance-to-nearest-zero along a 1D line', () => {
    const f = new Float64Array([INF, INF, 0, INF, INF])
    const d = distanceTransform1D(f)
    expect(Array.from(d)).toEqual([4, 1, 0, 1, 4])
  })

  it('handles a single element', () => {
    expect(Array.from(distanceTransform1D(new Float64Array([0])))).toEqual([0])
  })

  it('handles multiple zero seeds - each point sees its nearest one', () => {
    const f = new Float64Array([0, INF, INF, INF, 0])
    const d = distanceTransform1D(f)
    expect(Array.from(d)).toEqual([0, 1, 4, 1, 0])
  })
})

describe('computeClearanceField', () => {
  it('is zero at the obstacle and grows with Euclidean distance from it', () => {
    const cells = buildGrid(3, 3, [[1, 1]])
    const field = computeClearanceField(cells, 3, 3, 1)

    const at = (ix: number, iz: number) => field[ix * 3 + iz]
    expect(at(1, 1)).toBeCloseTo(0)
    expect(at(0, 1)).toBeCloseTo(1)
    expect(at(2, 1)).toBeCloseTo(1)
    expect(at(1, 0)).toBeCloseTo(1)
    expect(at(1, 2)).toBeCloseTo(1)
    expect(at(0, 0)).toBeCloseTo(Math.SQRT2)
    expect(at(2, 2)).toBeCloseTo(Math.SQRT2)
  })

  it('scales by the grid resolution', () => {
    const cells = buildGrid(3, 1, [[0, 0]])
    const field = computeClearanceField(cells, 3, 1, 0.05)
    expect(field[2]).toBeCloseTo(2 * 0.05, 5) // cell (2,0), 2 cells from the obstacle
  })

  it('treats unknown cells as obstacles - conservative by design', () => {
    const cells = buildGrid(3, 1, [], [[1, 0]])
    const field = computeClearanceField(cells, 3, 1, 1)
    expect(field[1]).toBeCloseTo(0) // (ix=1, iz=0) -> ix*height+iz = 1*1+0
    expect(field[0]).toBeCloseTo(1) // (ix=0, iz=0) -> ix*height+iz = 0*1+0
  })

  it('respects the [ix][iz] axis convention on a non-square grid', () => {
    // Obstacle far along the X axis (ix=4) but at iz=0 on a grid taller than it is wide
    // in the other axis - a swapped-axis bug would put the near-zero distances in the
    // wrong place entirely, not just off by a constant.
    const cells = buildGrid(5, 2, [[4, 0]])
    const field = computeClearanceField(cells, 5, 2, 1)
    const at = (ix: number, iz: number) => field[ix * 2 + iz]
    expect(at(4, 0)).toBeCloseTo(0)
    expect(at(0, 0)).toBeCloseTo(4)
    expect(at(4, 1)).toBeCloseTo(1)
  })
})
