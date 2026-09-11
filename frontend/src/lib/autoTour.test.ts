import { describe, expect, it } from 'vitest'
import { TOUR_MAX_STOPS, selectTourStops } from './autoTour'
import type { SceneObjectResponse } from '../api/types'

let seq = 0
function obj(overrides: Partial<SceneObjectResponse> & { pos_x: number; pos_z: number }): SceneObjectResponse {
  seq += 1
  return {
    id: overrides.id ?? `obj-${seq}`,
    scene_id: 'scene-1',
    name: overrides.name ?? `object-${seq}`,
    description: null,
    pos_x: overrides.pos_x,
    pos_y: 0,
    pos_z: overrides.pos_z,
    bbox_min_x: overrides.pos_x - 0.1,
    bbox_min_y: -0.1,
    bbox_min_z: overrides.pos_z - 0.1,
    bbox_max_x: overrides.pos_x + 0.1,
    bbox_max_y: 0.1,
    bbox_max_z: overrides.pos_z + 0.1,
    num_views: 1,
    num_points: 100,
    is_fragment: overrides.is_fragment ?? false,
    mesh_path: null,
    created_at: '2026-01-01T00:00:00Z',
  }
}

const ORIGIN = { x: 0, z: 0 }

describe('selectTourStops', () => {
  it('never queues an object that is not marked reachable', () => {
    const reachable = obj({ id: 'a', pos_x: 1, pos_z: 0 })
    const unreachable = obj({ id: 'b', pos_x: 2, pos_z: 0 })
    const stops = selectTourStops([reachable, unreachable], new Set(['a']), ORIGIN)
    expect(stops.map((o) => o.id)).toEqual(['a'])
  })

  it('excludes fragments even when marked reachable', () => {
    const whole = obj({ id: 'a', pos_x: 1, pos_z: 0 })
    const fragment = obj({ id: 'b', pos_x: 2, pos_z: 0, is_fragment: true })
    const stops = selectTourStops([whole, fragment], new Set(['a', 'b']), ORIGIN)
    expect(stops.map((o) => o.id)).toEqual(['a'])
  })

  it('returns every reachable object when at or under the cap', () => {
    const objects = Array.from({ length: TOUR_MAX_STOPS }, (_, i) => obj({ id: `o${i}`, pos_x: i, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const stops = selectTourStops(objects, reachable, ORIGIN)
    expect(stops).toHaveLength(TOUR_MAX_STOPS)
    expect(new Set(stops.map((o) => o.id))).toEqual(reachable)
  })

  it('caps the tour even when more objects are reachable', () => {
    const objects = Array.from({ length: TOUR_MAX_STOPS + 12 }, (_, i) => obj({ id: `o${i}`, pos_x: i, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const stops = selectTourStops(objects, reachable, ORIGIN)
    expect(stops.length).toBeLessThanOrEqual(TOUR_MAX_STOPS)
    expect(stops.length).toBeGreaterThanOrEqual(6)
  })

  it('prefers a spread across the room over the first N in list order', () => {
    // A tight cluster of 10 objects right at the origin, plus two outliers far away at
    // opposite ends of the room. "First N in list order" (a naive slice) would only ever
    // see the cluster and miss both outliers; a spread-preferring selection should pick
    // up at least one of them.
    const cluster = Array.from({ length: 10 }, (_, i) => obj({ id: `cluster-${i}`, pos_x: i * 0.01, pos_z: 0 }))
    const outlierA = obj({ id: 'outlier-a', pos_x: 20, pos_z: 0 })
    const outlierB = obj({ id: 'outlier-b', pos_x: -20, pos_z: 20 })
    const objects = [...cluster, outlierA, outlierB]
    const reachable = new Set(objects.map((o) => o.id))

    const stops = selectTourStops(objects, reachable, ORIGIN, 4)

    expect(stops).toHaveLength(4)
    const ids = new Set(stops.map((o) => o.id))
    expect(ids.has('outlier-a') || ids.has('outlier-b')).toBe(true)
  })

  it('never exceeds an explicit maxStops override', () => {
    const objects = Array.from({ length: 20 }, (_, i) => obj({ id: `o${i}`, pos_x: i, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const stops = selectTourStops(objects, reachable, ORIGIN, 3)
    expect(stops).toHaveLength(3)
  })

  it('returns an empty tour when nothing is reachable', () => {
    const objects = [obj({ id: 'a', pos_x: 1, pos_z: 0 })]
    expect(selectTourStops(objects, new Set(), ORIGIN)).toEqual([])
  })
})
