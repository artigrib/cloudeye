import * as THREE from 'three'
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type { CloudWorkerRequest, CloudWorkerResponse } from './cloudBinWorker'
import CloudBinWorker from './cloudBinWorker?worker'

export interface GlbLoadProgress {
  loaded: number
  total: number
}

/** What the viewer actually needs off a loaded cloud: a scene graph holding one
 * THREE.Points whose position/color attributes `scene3dPoints.buildDecimatedPointCloud`
 * reads. A GLTF satisfies this structurally, and so does the object the binary path
 * builds - so both sources share this cache, this cancel path and this consumer. */
export interface SceneCloudSource {
  scene: THREE.Object3D
}

interface CacheEntry {
  promise: Promise<SceneCloudSource>
  /** Only the GLTFLoader path can be aborted mid-flight; the binary path sets this to
   * null and is cancelled by terminating its worker instead. */
  manager: THREE.LoadingManager | null
  worker: Worker | null
  gltf: SceneCloudSource | null
}

const cache = new Map<string, CacheEntry>()

/** Builds the scene-graph wrapper around the worker's buffers. O(1) - the arrays arrive
 * already parsed and already rotated into the viewer's frame, and are adopted by
 * reference, never copied. uint8 colours are attached `normalized`, exactly as
 * GLTFLoader attaches the GLB's own uint8 COLOR_0. */
function pointsFromBuffers(positions: Float32Array, colors: Uint8Array | null): THREE.Object3D {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true))
  geometry.computeBoundingBox()
  const group = new THREE.Group()
  group.add(new THREE.Points(geometry, new THREE.PointsMaterial({ vertexColors: !!colors })))
  return group
}

/** Resolves with the binary cloud, or with null when this scene has none (404) - in which
 * case the caller falls back to the GLB. Rejects only on a real failure. */
function loadBinaryCloud(cloudUrl: string, entry: CacheEntry): Promise<SceneCloudSource | null> {
  return new Promise((resolve, reject) => {
    const worker: Worker = new CloudBinWorker()
    entry.worker = worker
    worker.onmessage = (event: MessageEvent<CloudWorkerResponse>) => {
      const msg = event.data
      worker.terminate()
      entry.worker = null
      if (msg.ok) resolve({ scene: pointsFromBuffers(msg.positions, msg.colors) })
      else if (msg.notFound) resolve(null)
      else reject(new Error(msg.error))
    }
    worker.onerror = (err) => {
      worker.terminate()
      entry.worker = null
      reject(new Error(err.message || 'cloud worker failed'))
    }
    worker.postMessage({ url: cloudUrl } satisfies CloudWorkerRequest)
  })
}

/** Loads (or reuses the in-flight/cached load of) a scene's point cloud, keyed by scene
 * id - so switching 2D<->3D, or between scenes and back, doesn't refetch the file. Cache
 * lives only for this tab's session (module-scope, gone on reload); a failed load is
 * evicted so a transient network error doesn't stick around.
 *
 * Two sources, one cache. `cloudUrl` (CEPC binary, parsed in a worker) is tried first and
 * wins when the scene has one; a 404 there means an ordinary pipeline scene, which only
 * has the points-primitive GLB, and the GLTFLoader path runs instead. Progress events come
 * from the GLB path only - the binary path does its whole download inside the worker, and
 * an indeterminate spinner is what the caller already renders when `progress` is null. */
export function loadSceneGlb(
  sceneId: string,
  url: string,
  onProgress?: (p: GlbLoadProgress) => void,
  cloudUrl?: string,
): Promise<SceneCloudSource> {
  const cached = cache.get(sceneId)
  if (cached) return cached.promise

  const manager = new THREE.LoadingManager()
  const loader = new GLTFLoader(manager)
  const entry: CacheEntry = {
    manager, worker: null, gltf: null,
    promise: undefined as unknown as Promise<SceneCloudSource>,
  }
  const loadGlb = () => new Promise<SceneCloudSource>((resolve, reject) => {
    loader.load(
      url,
      (gltf: GLTF) => {
        entry.gltf = gltf
        resolve(gltf)
      },
      (event) => onProgress?.({ loaded: event.loaded, total: event.total }),
      reject,
    )
  })
  entry.promise = cloudUrl
    ? loadBinaryCloud(cloudUrl, entry).then((bin) => {
        if (!bin) return loadGlb()
        entry.gltf = bin
        return bin
      })
    : loadGlb()
  entry.promise.catch(() => cache.delete(sceneId))
  cache.set(sceneId, entry)
  return entry.promise
}

/** Synchronous cache peek, so a component can render the already-loaded model on the
 * very first paint after remounting instead of flashing a loading state. */
export function peekSceneGlb(sceneId: string): SceneCloudSource | null {
  return cache.get(sceneId)?.gltf ?? null
}

/** Cancels sceneId's in-flight load (no-op if it already finished, or was never
 * started) - call when navigating away from the scene before its GLB finished loading.
 *
 * Evicts the cache entry synchronously (not just on the promise's eventual rejection):
 * React StrictMode double-invokes effects in dev, mounting - cleaning up - remounting
 * within the same tick. If the aborted entry stayed cached that long, the remount's
 * `loadSceneGlb` call would hand back the same now-doomed promise instead of starting a
 * fresh load. */
export function cancelSceneGlbLoad(sceneId: string): void {
  const entry = cache.get(sceneId)
  // Already resolved - keep it cached for the rest of the session, nothing to cancel.
  if (!entry || entry.gltf) return
  entry.worker?.terminate()
  entry.manager?.abort()
  cache.delete(sceneId)
}
