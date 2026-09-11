import * as THREE from 'three'
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js'

// A single bundled static asset - not glbCache.ts's scene-id-keyed, abort-on-navigate cache
// (that one is built for a 130-150MB per-scene download that must be cancellable when the
// user navigates away mid-load). Each robot mesh here is a few hundred KB, loads once per
// session per platform, and should never be cancelled: once loaded, a model is shared across
// every scene and every 2D<->3D remount for the rest of the tab's lifetime.
//
// Keyed by URL (not a single slot) since the robots-integration pass added eight selectable
// platforms (see app/robots.py on the backend) - the viewer may load more than one mesh in a
// session as the user switches platforms, and each one is cached independently so switching
// back to a previously-selected platform is instant.

export const DEFAULT_MODEL_URL = '/models/turtlebot3_burger.glb'

// Expected extent in scene metres (Y-up), used only for the dev-mode sanity check below.
// The largest registered platform (husky.glb) is ~0.985m at its longest dimension, so this
// stays generous rather than tuned to any one mesh - it exists to catch a units mistake
// (mm vs m), not to assert a tight per-platform bound.
const MAX_EXPECTED_DIMENSION_M = 2

const pendingByUrl = new Map<string, Promise<GLTF>>()
const loadedByUrl = new Map<string, GLTF>()

/** Loads (or reuses the in-flight/cached load of) a robot model, `DEFAULT_MODEL_URL`
 * (TurtleBot3 Burger) unless a different platform's mesh URL is passed. No cancel function,
 * deliberately - these are static assets the app always wants, not a per-scene resource a
 * navigation might abandon mid-flight. A failed load clears the pending promise for that URL
 * so the next mount's call retries instead of replaying the same rejection forever. */
export function loadRobotModel(modelUrl: string = DEFAULT_MODEL_URL): Promise<GLTF> {
  const loaded = loadedByUrl.get(modelUrl)
  if (loaded) return Promise.resolve(loaded)
  const pending = pendingByUrl.get(modelUrl)
  if (pending) return pending

  const next = new GLTFLoader().loadAsync(modelUrl).then((gltf) => {
    loadedByUrl.set(modelUrl, gltf)
    pendingByUrl.delete(modelUrl)
    if (import.meta.env.DEV) {
      const box = new THREE.Box3().setFromObject(gltf.scene)
      const size = box.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z)
      if (maxDim > MAX_EXPECTED_DIMENSION_M) {
        console.warn(
          `${modelUrl} is ${maxDim.toFixed(2)}m at its largest dimension - expected well ` +
            `under ${MAX_EXPECTED_DIMENSION_M}m. Likely a unit-scale mistake (mm vs m) in a ` +
            `regenerated model.`,
        )
      }
    }
    return gltf
  })
  next.catch(() => {
    pendingByUrl.delete(modelUrl)
  })
  pendingByUrl.set(modelUrl, next)
  return next
}

/** Synchronous cache peek, so a component can attach the already-loaded model on the very
 * first paint after a 2D<->3D remount (or a re-select of a previously-loaded platform)
 * instead of a load flicker. */
export function peekRobotModel(modelUrl: string = DEFAULT_MODEL_URL): GLTF | null {
  return loadedByUrl.get(modelUrl) ?? null
}

export interface RobotModelInstance {
  object: THREE.Group
  /** Exactly the materials this call cloned - the only things the caller may dispose. The
   * source geometry and the cached GLTF itself are shared and must never be disposed. */
  materials: THREE.Material[]
}

/** Clones a fresh, independently-mutable instance of the cached model. Geometry is shared
 * (three's Object3D.clone() reuses BufferGeometry by reference - cheap, and correct since
 * nothing here ever edits vertex data); materials are cloned per-instance, mirroring the
 * discipline in scene3dPoints.ts, so tweaking one viewer's material never corrupts the
 * shared cache entry other viewers/remounts read from. */
export function instantiateRobotModel(gltf: GLTF): RobotModelInstance {
  const object = gltf.scene.clone(true) as THREE.Group
  const materials: THREE.Material[] = []
  object.traverse((child) => {
    const mesh = child as THREE.Mesh
    if (!mesh.isMesh) return
    const material = (mesh.material as THREE.MeshStandardMaterial).clone()
    // Re-asserting what each platform's build-*-glb.mjs script already baked in:
    // belt-and-braces against a regenerated model accidentally picking up a
    // metallic/glossy material.
    material.roughness = 0.9
    material.metalness = 0
    material.envMapIntensity = 0
    mesh.material = material
    mesh.castShadow = false
    mesh.receiveShadow = false
    materials.push(material)
  })
  return { object, materials }
}
