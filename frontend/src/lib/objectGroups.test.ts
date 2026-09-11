import { describe, expect, it } from 'vitest'
import type { SceneObjectResponse } from '../api/types'
import { groupObjects, visibleObjects } from './objectGroups'

function obj(partial: Partial<SceneObjectResponse> & { id: string; name: string }): SceneObjectResponse {
  return {
    scene_id: 'scene-1',
    description: null,
    pos_x: 0,
    pos_y: 0,
    pos_z: 0,
    bbox_min_x: 0,
    bbox_min_y: 0,
    bbox_min_z: 0,
    bbox_max_x: 1,
    bbox_max_y: 1,
    bbox_max_z: 1,
    num_views: 1,
    num_points: 100,
    is_fragment: false,
    mesh_path: null,
    created_at: '2026-01-01T00:00:00Z',
    ...partial,
  }
}

describe('groupObjects', () => {
  it('groups by lowercased name and keeps first-appearance order', () => {
    const groups = groupObjects([
      obj({ id: 'a', name: 'Lamp' }),
      obj({ id: 'b', name: 'desk' }),
      obj({ id: 'c', name: 'lamp' }),
    ])
    expect(groups.map((g) => g.key)).toEqual(['lamp', 'desk'])
    expect(groups[0].items.map((o) => o.id).sort()).toEqual(['a', 'c'])
    expect(groups[0].name).toBe('Lamp')
  })

  it('orders instances inside a group by num_views, best-observed first', () => {
    const groups = groupObjects([
      obj({ id: 'a', name: 'lamp', num_views: 3 }),
      obj({ id: 'b', name: 'lamp', num_views: 11 }),
      obj({ id: 'c', name: 'lamp', num_views: 7 }),
    ])
    expect(groups[0].items.map((o) => o.id)).toEqual(['b', 'c', 'a'])
  })

  it('returns no groups for no objects', () => {
    expect(groupObjects([])).toEqual([])
  })
})

describe('visibleObjects', () => {
  const objects = [obj({ id: 'a', name: 'desk' }), obj({ id: 'f', name: 'blob', is_fragment: true })]

  it('drops fragments by default', () => {
    expect(visibleObjects(objects, false).map((o) => o.id)).toEqual(['a'])
  })

  it('keeps them when fragments are shown', () => {
    expect(visibleObjects(objects, true).map((o) => o.id)).toEqual(['a', 'f'])
  })
})
