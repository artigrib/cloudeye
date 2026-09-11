import * as THREE from 'three'
import { isometricCameraPose, type BoundingBox3 } from './scene3dCamera'
import { buildDecimatedPointCloud } from './scene3dPoints'
import { createPointCloudMaterial } from './classicPointCloudMaterial'
import { CANVAS, toHex } from './tokens'

// Project preview thumbnails (spec 11): captured once, in-browser, the first time a
// scene's point cloud has been loaded (see useScenePreviewCapture, called from
// SceneMap3D once `glb.gltf` resolves) - there's no server-side render of this, so a
// scene that's never been opened in 3D has no thumbnail yet (HomePage falls back to a
// skeleton for that case). Stored in IndexedDB, not localStorage: a 640x360 webp is
// tens of KB, comfortably past localStorage's ~5MB *string* quota once a project list
// gets past a handful of scenes.

const DB_NAME = 'cloudeye-scene-previews'
const STORE = 'previews'
const PREVIEW_WIDTH = 640
const PREVIEW_HEIGHT = 360
const PREVIEW_MAX_POINTS = 150_000
// Chunkier than SceneMap3D's DEFAULT_POINT_SIZE (0.015) - at card-thumbnail size the
// room's sparser wall points all but disappear at the live view's default, and there's
// no slider here to compensate.
const PREVIEW_POINT_SIZE = 0.03
// isometricCameraPose's own framing (shared with the live 3D view's iso preset) leaves
// headroom that reads as "mostly empty" once a 640x360 render gets scaled down into a
// ~280x168 card - pull the camera in rather than reuse the live framing verbatim.
const PREVIEW_ZOOM = 0.7

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE)
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

/** Null if the scene has no cached preview yet (never opened in 3D, or capture failed). */
export async function getScenePreviewBlob(sceneId: string): Promise<Blob | null> {
  const db = await openDb()
  try {
    return await new Promise<Blob | null>((resolve, reject) => {
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(sceneId)
      req.onsuccess = () => resolve((req.result as Blob | undefined) ?? null)
      req.onerror = () => reject(req.error)
    })
  } finally {
    db.close()
  }
}

async function putScenePreviewBlob(sceneId: string, blob: Blob): Promise<void> {
  const db = await openDb()
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).put(blob, sceneId)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } finally {
    db.close()
  }
}

/** Renders an isometric, height-ramp-colored snapshot of `root` (a loaded scene GLB's
 * root object) to a detached offscreen canvas and caches it in IndexedDB under
 * `sceneId`. No-ops if a preview is already cached. Never throws - a failed capture
 * just leaves HomePage's skeleton fallback in place, which is an acceptable outcome,
 * not an error worth surfacing. Independent of the live 3D view's own decimated
 * geometry/renderer (does its own smaller-cap decimation into its own throwaway
 * WebGLRenderer) so it can't be affected by that view's cleanup/dispose timing. */
export async function captureScenePreview(sceneId: string, root: THREE.Object3D): Promise<void> {
  try {
    if (await getScenePreviewBlob(sceneId)) return

    const built = buildDecimatedPointCloud(root, PREVIEW_MAX_POINTS)
    const bb = built?.geometry.boundingBox
    if (!built || !bb) return

    const canvas = document.createElement('canvas')
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true })
    renderer.setSize(PREVIEW_WIDTH, PREVIEW_HEIGHT, false)
    renderer.setClearColor(toHex(CANVAS), 1)

    const material = createPointCloudMaterial(built.hasColor, built.colorItemSize)
    material.uniforms.uSize.value = PREVIEW_POINT_SIZE
    material.uniforms.uScale.value = PREVIEW_HEIGHT * 0.5
    material.uniforms.uColorMode.value = 1 // height ramp, regardless of the live view's mode
    material.uniforms.uHeightMin.value = bb.min.y
    material.uniforms.uHeightMax.value = bb.max.y
    material.uniforms.uCeilingMode.value = 0

    const box: BoundingBox3 = { min: [bb.min.x, bb.min.y, bb.min.z], max: [bb.max.x, bb.max.y, bb.max.z] }
    const roomDiagonal = Math.hypot(box.max[0] - box.min[0], box.max[2] - box.min[2])
    material.uniforms.uFadeStart.value = roomDiagonal * 0.9
    material.uniforms.uFadeEnd.value = roomDiagonal * 1.6

    const scene = new THREE.Scene()
    scene.add(new THREE.Points(built.geometry, material))

    const camera = new THREE.PerspectiveCamera(55, PREVIEW_WIDTH / PREVIEW_HEIGHT, 0.05, 500)
    const pose = isometricCameraPose(box)
    camera.position.set(
      pose.target[0] + (pose.position[0] - pose.target[0]) * PREVIEW_ZOOM,
      pose.target[1] + (pose.position[1] - pose.target[1]) * PREVIEW_ZOOM,
      pose.target[2] + (pose.position[2] - pose.target[2]) * PREVIEW_ZOOM,
    )
    camera.lookAt(...pose.target)

    renderer.render(scene, camera)
    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/webp', 0.85)
    })

    built.geometry.dispose()
    material.dispose()
    renderer.dispose()
    renderer.forceContextLoss()

    if (blob) {
      await putScenePreviewBlob(sceneId, blob)
    } else {
      console.error('Scene preview capture produced no blob (canvas.toBlob returned null)')
    }
  } catch (err) {
    // Capture is best-effort - see the docstring above - but still worth a console
    // trace, the same way the (also display-only, also best-effort) camera-track fetch
    // in ScenePage logs its own failures instead of swallowing them silently.
    console.error('Scene preview capture failed', err)
  }
}
