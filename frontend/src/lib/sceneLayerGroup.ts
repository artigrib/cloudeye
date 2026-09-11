import * as THREE from 'three/webgpu'

/** The scene-graph name the Mesh layer's group carries. A probe reading
 * `window.__cloudeyeThree.scene` looks for exactly this. */
export const MESH_LAYER_NAME = 'mesh-layer'
/** The same, for the MSA Layout overlay, which builds its group inline. */
export const MSA_LAYER_NAME = 'msa-overlay'

/** Put one layer's object into the scene under a named group, and hand back the exact
 * undo for it.
 *
 * This is the whole contract behind "a layer that is off is not in the scene graph": the
 * effect that switches a layer on calls this, and returns what it hands back as its
 * cleanup. It exists as a function rather than as six lines inside an effect so the
 * invariant can be tested against a real THREE.Scene without a renderer - see
 * sceneLayerGroup.test.ts, which switches a layer on and off twice and asserts the scene
 * is left exactly as it was found.
 *
 * Nothing is disposed. The objects passed here come from `useSceneGlb`, which parses a
 * GLB once and caches it across remounts (lib/glbCache.ts); disposing on detach would
 * leave the next switch-on rendering freed buffers. */
export function attachLayerGroup(scene: THREE.Scene, object: THREE.Object3D, name: string): () => void {
  const group = new THREE.Group()
  group.name = name
  group.add(object)
  scene.add(group)
  let detached = false
  return () => {
    // Idempotent: React can run a cleanup once, and a StrictMode double-invoke runs the
    // pair twice - neither may remove something a later attach has since added.
    if (detached) return
    detached = true
    scene.remove(group)
    group.remove(object)
  }
}

/** Every group `attachLayerGroup` has put in this scene, by name. The check a probe (and
 * the test) makes after toggling: nothing of a switched-off layer is left. */
export function layerGroupNames(scene: THREE.Scene): string[] {
  return scene.children.filter((c) => c.name).map((c) => c.name)
}
