import { describe, expect, it } from 'vitest'
import { classifyNode } from './MsaSceneViewer'

// Node names below match what export_glb.py's GLB *actually* contains after
// trimesh's exporter strips every "/" from its `node_name`/`parent_node_name`
// strings (see classifyNode's docstring) - e.g. Python-side
// `f"{obj_id}/visual/part_{j}"` becomes the glTF node name `bed_0visualpart_0`.
describe('classifyNode', () => {
  it('classifies wall and floor nodes as mesh', () => {
    expect(classifyNode('wall_0')).toBe('mesh')
    expect(classifyNode('wall_12')).toBe('mesh')
    expect(classifyNode('floor')).toBe('mesh')
  })

  it('classifies object visual/collision children by substring', () => {
    expect(classifyNode('bed_0visualpart_0')).toBe('mesh')
    expect(classifyNode('bed_0visual')).toBe('mesh')
    expect(classifyNode('bed_0collisionhull')).toBe('collision')
    expect(classifyNode('bed_0collision')).toBe('collision')
  })

  it('classifies Plan nodes', () => {
    expect(classifyNode('Plan')).toBe('plan')
    expect(classifyNode('Planwall_0')).toBe('plan')
    expect(classifyNode('Planfloor')).toBe('plan')
  })

  it('classifies cloud nodes', () => {
    expect(classifyNode('cloud')).toBe('cloud')
    expect(classifyNode('cloudpoints')).toBe('cloud')
    expect(classifyNode('PointCloud')).toBe('cloud')
  })

  it('does not classify Path/Target - they are always-visible, not layer-gated', () => {
    expect(classifyNode('Path')).toBeNull()
    expect(classifyNode('Target')).toBeNull()
  })

  it('does not misclassify an object id that merely contains "wall" or "floor"', () => {
    expect(classifyNode('hallway_0visual')).toBe('mesh') // matched via "visual", not the name
    expect(classifyNode('wallpaper_0')).toBeNull() // not /^wall_\d+$/
  })
})
