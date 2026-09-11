import { describe, expect, it } from 'vitest'
import type { OccupancyGridResponse, SceneObjectResponse } from '../api/types'
import { applyMovesToGrid, applyMovesToObjects, moveParams } from './objectMoves'

function object(id: string, x: number, z: number, half = 0.05): SceneObjectResponse {
  return {
    id,
    scene_id: 's',
    name: id,
    description: null,
    pos_x: x,
    pos_y: 0,
    pos_z: z,
    bbox_min_x: x - half,
    bbox_min_y: 0,
    bbox_min_z: z - half,
    bbox_max_x: x + half,
    bbox_max_y: 1,
    bbox_max_z: z + half,
    num_views: 1,
    num_points: 10,
    is_fragment: false,
    mesh_path: null,
    created_at: '2026-01-01T00:00:00Z',
  }
}

/** A 4x1 grid of 0.1 m cells whose centres are at x = 0.05, 0.15, 0.25, 0.35. */
function grid(values: number[]): OccupancyGridResponse {
  return {
    resolution: 0.1,
    origin_x: 0,
    origin_z: 0,
    width: values.length,
    height: 1,
    legend: { '0': 'free', '1': 'obstacle', '2': 'unknown' },
    cells: values.map((v) => [v]),
    // This fixture is about moving objects on the grid, not about what was observed -
    // every cell measured, and a layer that exists.
    unobserved: values.map(() => [false]),
    layer_available: true,
    layer_unavailable_reason: null,
  }
}

describe('moveParams', () => {
  it('formats one parameter per move, as the backend parses them', () => {
    expect(moveParams([{ objectId: 'abc', dx: 0.5, dz: -1.25 }])).toEqual(['abc:0.5:-1.25'])
  })

  it('is empty for no moves, so the request is the unmoved one', () => {
    expect(moveParams([])).toEqual([])
  })
})

describe('applyMovesToObjects', () => {
  const objects = [object('desk', 1, 2), object('bed', 3, 4)]

  it('slides position and bbox together', () => {
    const [desk] = applyMovesToObjects(objects, [{ objectId: 'desk', dx: 0.5, dz: -0.25 }])
    expect(desk.pos_x).toBeCloseTo(1.5)
    expect(desk.pos_z).toBeCloseTo(1.75)
    expect(desk.bbox_min_x).toBeCloseTo(1.45)
    expect(desk.bbox_max_z).toBeCloseTo(1.8)
    // The box keeps its size: a moved object is never a bigger obstacle.
    expect(desk.bbox_max_x - desk.bbox_min_x).toBeCloseTo(0.1)
  })

  it('leaves untouched objects identical, by identity', () => {
    const moved = applyMovesToObjects(objects, [{ objectId: 'desk', dx: 1, dz: 0 }])
    expect(moved[1]).toBe(objects[1])
  })

  it('returns the very same array for no moves, not a copy', () => {
    // Identity, not equality: SceneMap3D rebuilds every object marker when this prop
    // changes reference, so a fresh copy per render would recreate 17 meshes per frame.
    expect(applyMovesToObjects(objects, [])).toBe(objects)
  })
})

describe('applyMovesToGrid', () => {
  // The object sits over the cell centred at x = 0.15.
  const objects = [object('box', 0.15, 0.05, 0.04)]

  it('cuts the old cells free and pastes the new ones as obstacle', () => {
    const before = grid([0, 1, 0, 0])
    const after = applyMovesToGrid(before, objects, [{ objectId: 'box', dx: 0.2, dz: 0 }])
    expect(after.cells.map((c) => c[0])).toEqual([0, 0, 0, 1])
  })

  it('leaves the input grid untouched', () => {
    const before = grid([0, 1, 0, 0])
    applyMovesToGrid(before, objects, [{ objectId: 'box', dx: 0.2, dz: 0 }])
    expect(before.cells.map((c) => c[0])).toEqual([0, 1, 0, 0])
  })

  it('pastes after cutting, so an overlapping move does not erase itself', () => {
    // 0.1 m is one cell: the source and target rectangles are adjacent, and a
    // paste-then-cut order would clear the cell it had just drawn.
    const after = applyMovesToGrid(grid([0, 1, 0, 0]), objects, [{ objectId: 'box', dx: 0.1, dz: 0 }])
    expect(after.cells.map((c) => c[0])).toEqual([0, 0, 1, 0])
  })

  it('returns the same grid object for no moves', () => {
    const before = grid([0, 1, 0, 0])
    expect(applyMovesToGrid(before, objects, [])).toBe(before)
  })

  it('ignores a move naming an object this scene does not have', () => {
    const before = grid([0, 1, 0, 0])
    expect(applyMovesToGrid(before, objects, [{ objectId: 'nope', dx: 1, dz: 0 }])).toBe(before)
  })
})
