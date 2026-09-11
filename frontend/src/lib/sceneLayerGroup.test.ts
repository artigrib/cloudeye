import { describe, expect, it } from 'vitest'
import * as THREE from 'three/webgpu'
import { MESH_LAYER_NAME, attachLayerGroup, layerGroupNames } from './sceneLayerGroup'

/** The acceptance for the four layer toggles: a layer that is switched off is not in the
 * scene graph. It used to be switched off by setting `.visible = false` on the nodes
 * inside a group that stayed in the graph, which is why "turn the mesh off" did not turn
 * the mesh off. */
describe('attachLayerGroup', () => {
  it('leaves nothing in the scene after a layer is toggled on and off twice', () => {
    const scene = new THREE.Scene()
    const payload = new THREE.Object3D()
    payload.name = 'gltf-root'

    const before = scene.children.length
    for (let i = 0; i < 2; i++) {
      const detach = attachLayerGroup(scene, payload, MESH_LAYER_NAME)
      expect(layerGroupNames(scene)).toContain(MESH_LAYER_NAME)
      detach()
      expect(layerGroupNames(scene)).not.toContain(MESH_LAYER_NAME)
    }
    expect(scene.children.length).toBe(before)
    // The payload itself survives - it is a cached GLTF parse shared across toggles, so
    // detaching must not orphan or dispose it.
    expect(payload.parent).toBeNull()
  })

  it('does not touch a later attach when an earlier cleanup runs twice', () => {
    const scene = new THREE.Scene()
    const payload = new THREE.Object3D()
    const detachFirst = attachLayerGroup(scene, payload, MESH_LAYER_NAME)
    detachFirst()
    const detachSecond = attachLayerGroup(scene, payload, MESH_LAYER_NAME)
    detachFirst() // idempotent - must not remove the second group
    expect(layerGroupNames(scene)).toContain(MESH_LAYER_NAME)
    detachSecond()
    expect(layerGroupNames(scene)).not.toContain(MESH_LAYER_NAME)
  })

  it('keeps layers independent - removing one leaves the others', () => {
    const scene = new THREE.Scene()
    const detachMesh = attachLayerGroup(scene, new THREE.Object3D(), MESH_LAYER_NAME)
    attachLayerGroup(scene, new THREE.Object3D(), 'msa-overlay')
    detachMesh()
    expect(layerGroupNames(scene)).toEqual(['msa-overlay'])
  })
})
