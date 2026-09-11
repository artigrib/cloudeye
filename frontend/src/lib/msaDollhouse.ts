import * as THREE from 'three'

/** "Dollhouse" presentation helpers for the MSA scene viewer (T15c).
 *
 * export_glb.py extrudes every wall footprint into a closed, outward-facing box
 * (trimesh `extrude_polygon`, winding verified consistent, volume > 0). A closed box
 * can't be made see-through-from-outside with `material.side` alone: with FrontSide
 * the near wall's *outer* face (normal toward the camera) still occludes the room,
 * with BackSide its *inner* face does. So `dollhouseWallGeometry` keeps only the
 * faces whose normal points INTO the room (plus the top cap, so the wall still reads
 * as having thickness from above) and the viewer renders them FrontSide: a wall
 * between the camera and the room faces away from the camera and is culled, a wall
 * behind the room faces the camera and renders opaque. The classic dollhouse trick,
 * done on the mesh rather than by hand-authoring single-sided quads in the exporter
 * (whose walls also feed the collision layer / Isaac and must stay closed). */

// `wall_<i>` = a per-fragment wall (pre-T15g exports); `wall_outline` = T15g's single
// closed wall band extruded from the regularized room outline.
export const WALL_NAME_RE = /^wall_(\d+|outline)$/

/** Faces are kept when they face `roomCenter` (in the geometry's own space, so pass
 * `mesh.worldToLocal(center)` for a transformed node) or point up. Bottom caps and
 * outward faces are dropped. Returns a new non-indexed geometry with flat normals
 * (and the source's UVs, if any). */
export function dollhouseWallGeometry(geometry: THREE.BufferGeometry, roomCenter: THREE.Vector3): THREE.BufferGeometry {
  const src = geometry.index ? geometry.toNonIndexed() : geometry
  const pos = src.getAttribute('position')
  const uv = src.getAttribute('uv')
  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const c = new THREE.Vector3()
  const ab = new THREE.Vector3()
  const ac = new THREE.Vector3()
  const n = new THREE.Vector3()
  for (let i = 0; i + 2 < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i)
    b.fromBufferAttribute(pos, i + 1)
    c.fromBufferAttribute(pos, i + 2)
    ab.subVectors(b, a)
    ac.subVectors(c, a)
    n.crossVectors(ab, ac)
    if (n.lengthSq() < 1e-18) continue
    n.normalize()
    const cx = (a.x + b.x + c.x) / 3
    const cz = (a.z + b.z + c.z) / 3
    const facesRoom = (roomCenter.x - cx) * n.x + (roomCenter.z - cz) * n.z > 0
    const keep = n.y > 0.5 || (n.y > -0.5 && facesRoom)
    if (!keep) continue
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)
    normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z)
    if (uv) {
      uvs.push(uv.getX(i), uv.getY(i), uv.getX(i + 1), uv.getY(i + 1), uv.getX(i + 2), uv.getY(i + 2))
    }
  }
  const out = new THREE.BufferGeometry()
  out.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  out.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  if (uv) out.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  out.computeBoundingBox()
  out.computeBoundingSphere()
  return out
}

export interface ViewPose {
  position: THREE.Vector3
  target: THREE.Vector3
}

export type ViewPresetKey = 'threeQuarter' | 'top'

/** Camera poses framing a scene bounding box: a 3/4 view from outside/above (the
 * dollhouse look - two far walls opaque, two near walls culled) and a straight
 * top-down view. `radius` is half the box's longest horizontal side. */
export function viewPresets(box: THREE.Box3): Record<ViewPresetKey, ViewPose> {
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  const radius = Math.max(size.x, size.z, 1) / 2
  const target = new THREE.Vector3(center.x, box.min.y + Math.min(size.y, 1.2) * 0.5, center.z)
  return {
    threeQuarter: {
      position: new THREE.Vector3(center.x + radius * 1.15, box.min.y + radius * 0.85 + size.y, center.z + radius * 1.15),
      target,
    },
    top: {
      // Tiny Z offset keeps OrbitControls' up-vector well-defined looking straight down.
      position: new THREE.Vector3(center.x, box.max.y + radius * 1.7, center.z + 0.001),
      target: new THREE.Vector3(center.x, box.min.y, center.z),
    },
  }
}

/** Points a shadow-casting key light at the box from high on the +X/+Z side and
 * sizes its orthographic shadow frustum to cover the whole scene. */
export function fitKeyLight(light: THREE.DirectionalLight, box: THREE.Box3): void {
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  const radius = Math.max(size.length() / 2, 1)
  light.position.set(center.x + radius * 0.9, center.y + radius * 1.8, center.z + radius * 0.6)
  light.target.position.copy(center)
  light.target.updateMatrixWorld()
  const cam = light.shadow.camera
  cam.left = -radius * 1.2
  cam.right = radius * 1.2
  cam.top = radius * 1.2
  cam.bottom = -radius * 1.2
  cam.near = 0.1
  cam.far = radius * 5
  cam.updateProjectionMatrix()
}
