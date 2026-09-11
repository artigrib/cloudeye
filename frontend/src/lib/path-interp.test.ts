import { describe, expect, it } from 'vitest'
import { buildPathTable, pointAtDistance, pointAtFraction, type PathPoint } from './path-interp'

describe('buildPathTable', () => {
  it('sums straight-line segment lengths cumulatively', () => {
    const path: PathPoint[] = [
      [0, 0],
      [3, 0], // +3
      [3, 4], // +4 (3-4-5 triangle)
    ]
    const table = buildPathTable(path)
    expect(table.cumulative).toEqual([0, 3, 7])
    expect(table.totalLength).toBe(7)
  })

  it('is zero-length for a single-point path', () => {
    expect(buildPathTable([[1, 1]]).totalLength).toBe(0)
  })
})

describe('pointAtDistance', () => {
  const path: PathPoint[] = [
    [0, 0],
    [10, 0],
    [10, 10],
  ]
  const table = buildPathTable(path)

  it('returns the start point at distance 0', () => {
    expect(pointAtDistance(path, table, 0)).toEqual([0, 0])
  })

  it('returns the end point at the total length', () => {
    expect(pointAtDistance(path, table, table.totalLength)).toEqual([10, 10])
  })

  it('interpolates linearly within the first segment', () => {
    const [x, z] = pointAtDistance(path, table, 5)
    expect(x).toBeCloseTo(5)
    expect(z).toBeCloseTo(0)
  })

  it('crosses into the second segment correctly', () => {
    const [x, z] = pointAtDistance(path, table, 15)
    expect(x).toBeCloseTo(10)
    expect(z).toBeCloseTo(5)
  })

  it('clamps distances beyond the path length', () => {
    expect(pointAtDistance(path, table, 9999)).toEqual([10, 10])
  })

  it('clamps negative distances to the start', () => {
    expect(pointAtDistance(path, table, -5)).toEqual([0, 0])
  })
})

describe('pointAtFraction', () => {
  it('maps fraction 0.5 to the halfway arc-length point', () => {
    const path: PathPoint[] = [
      [0, 0],
      [10, 0],
    ]
    const table = buildPathTable(path)
    const [x, z] = pointAtFraction(path, table, 0.5)
    expect(x).toBeCloseTo(5)
    expect(z).toBeCloseTo(0)
  })
})
