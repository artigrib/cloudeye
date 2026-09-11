// MSA (Measured Scene Assembly, var/scratch/msa/SPEC.md) glTF layer classification -
// shared between MsaSceneViewer.tsx (the standalone `/_msa-viewer` dev route) and
// SceneMap3D.tsx (the main scene canvas, which overlays these same layers on top of
// the point-cloud view - see docs/DECISIONS.md's "MSA layers land in the main viewer"
// entry). Factored out here rather than duplicated so both consumers classify glTF
// nodes identically.

export type LayerKey = 'cloud' | 'mesh' | 'collision' | 'plan' | 'gaps'
export const LAYER_KEYS: LayerKey[] = ['cloud', 'mesh', 'collision', 'plan', 'gaps']

/** Classifies a glTF node produced by export_glb.py into one of the SPEC §8 layers.
 *
 * export_glb.py's trimesh scene graph uses `/`-delimited node names to express
 * hierarchy at the Python level (e.g. `node_name=f"{obj_id}/visual/part_{j}"`,
 * `parent_node_name=f"{obj_id}/visual"` - see that file). But trimesh's glTF
 * exporter builds REAL nested glTF nodes from `parent_node_name`, and each node's
 * own `name` field written to the file is the same original string with every `/`
 * simply deleted (not replaced, not split) - confirmed by inspecting the actual
 * exported file: `bed_0/visual/part_0` -> glTF node named `bed_0visualpart_0`
 * (nested under a node named `bed_0visual`, itself nested under `bed_0`), and
 * `Plan/wall_0` -> `Planwall_0`. So a node's own name already contains every
 * ancestor's (slash-stripped) name as a prefix, and a plain substring match below
 * is sufficient - no need to walk `.parent`. This is a real quirk of the current
 * exporter (a hierarchy-safe separator, e.g. `_`, would avoid it) worth revisiting
 * in `scripts/msa/export_glb.py` - not fixed here since Stage E0 is already merged
 * and this only affects how a *frontend* reads names, not the exported file's
 * correctness for Isaac/USD (whose export path is entirely separate, see
 * `export_usd.py`). `Path`/`Target` are intentionally NOT gated behind a layer
 * toggle - SPEC §8 lists exactly five toggleable layers and doesn't include them. */
export function classifyNode(name: string): LayerKey | null {
  if (name === 'cloud' || name.startsWith('cloud') || name === 'PointCloud') return 'cloud'
  if (name.includes('collision')) return 'collision'
  if (name.includes('visual')) return 'mesh'
  if (name.startsWith('Plan')) return 'plan'
  if (name === 'floor' || /^wall_(\d+|outline)$/.test(name)) return 'mesh'
  return null
}

/** T15i: object classes whose placeholder renders at TRANSLUCENT_OPACITY in every
 * viewer - this one, `scripts/msa/render_perspective.py`'s still and the presentation
 * USD's bound UsdPreviewSurface. Mirrors `scripts/msa/visibility.py`
 * TRANSLUCENT_OBJECT_CLASSES / TRANSLUCENT_OPACITY; keep the two in sync. */
export const TRANSLUCENT_OBJECT_CLASSES: ReadonlySet<string> = new Set(['curtain', 'drape', 'blind', 'mirror'])
export const TRANSLUCENT_OPACITY = 0.3

/** The object class of an MSA glTF node: `curtain_0visualpart_0` -> `curtain`,
 * `coffee maker_1collisionhull` -> `coffee maker`, `curtain_0` -> `curtain`. Same rule as
 * `render_perspective._object_class_from_id` (strip one trailing `_<index>`), applied to
 * the object-id prefix that survives trimesh's slash-stripping (see classifyNode). Null
 * for walls/floor/Plan/cloud nodes and anything without an `_<index>`. */
export function objectClassFromNodeName(name: string): string | null {
  if (name === 'floor' || /^wall_(\d+|outline)$/.test(name) || name.startsWith('Plan') || name.startsWith('cloud')) return null
  const m = /^(.+?)_\d+(?:visual|collision|$)/.exec(name)
  return m ? m[1] : null
}

export function isTranslucentObjectNode(name: string): boolean {
  const cls = objectClassFromNodeName(name)
  return cls !== null && TRANSLUCENT_OBJECT_CLASSES.has(cls.toLowerCase())
}

/** The yaw-normalization subset of bootstrap's `scene_meta.json`, as served by
 * `GET /api/scenes/{id}/msa-meta` (app.routers.scenes.get_scene_msa_meta - see
 * scene_service.MSA_META_FIELDS for the exact list; every key is optional on the wire
 * because a pre-yaw export simply has none of the `yaw_*` keys). */
export interface MsaSceneMeta {
  yaw_applied?: boolean
  yaw_correction_rad?: number
  yaw_correction_deg?: number
  /** Pivot of the yaw rotation, an (x, z) floor-plane point in the Y-up world frame
   * (the `_xy` suffix is bootstrap's - same naming quirk as gaps.json's
   * `measurement_point_xy`, see msaGaps.ts). */
  yaw_rotation_center_xy?: [number, number]
  yaw_method?: string
  floor_y?: number
  ceiling_y?: number
  room_polygon?: [number, number][]
}

/** The rigid transform that puts a yaw-normalized MSA export back onto the UNROTATED
 * point cloud: a rotation about the vertical axis through the bootstrap pivot. Plain
 * numbers rather than a Matrix4 so this module stays three-free (its consumers use two
 * different three builds - `three` in MsaSceneViewer.tsx, `three/webgpu` in
 * SceneMap3D.tsx; see docs/DECISIONS.md's "MSA layers land in the main viewer"). */
export interface MsaOverlayTransform {
  /** Object3D.rotation.y to apply (three.js convention, see msaOverlayTransform). */
  rotationY: number
  /** World (x, z) the rotation pivots about - bootstrap's `yaw_rotation_center_xy`. */
  pivotX: number
  pivotZ: number
}

/** Inverse of scripts/msa/bootstrap.py's yaw normalization, for overlaying the MSA
 * GLB / gaps on the point cloud this app serves (which bootstrap never rotates).
 *
 * Bootstrap rotates every XZ point by `yaw_correction_rad` about
 * `yaw_rotation_center_xy` with `geometry.rotate_point_xz` - standard CCW in the (x, z)
 * plane: `x' = cx + dx cos a - dz sin a`, `z' = cz + dx sin a + dz cos a` (equivalently
 * the matrix `[[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]` bootstrap's own
 * `_rotate_camera_poses` spells out). three.js's `Object3D.rotation.y = t` applies
 * `[[cos t, 0, sin t], [0, 1, 0], [-sin t, 0, cos t]]` - the same family with the
 * OPPOSITE sign convention, so bootstrap's rotation by `a` is three's `rotation.y = -a`,
 * and the inverse we want here (rotate by `-a` in bootstrap's convention) is three's
 * `rotation.y = +a`. The round-trip test in msaLayers.test.ts pins this down
 * numerically against a real three.js Object3D rather than relying on the derivation.
 *
 * Returns null when the export was not yaw-rotated (`yaw_applied` false/absent, or a
 * zero angle) - callers leave the group at identity. */
export function msaOverlayTransform(meta: MsaSceneMeta | null | undefined): MsaOverlayTransform | null {
  if (!meta || !meta.yaw_applied) return null
  const yaw = meta.yaw_correction_rad
  if (typeof yaw !== 'number' || !Number.isFinite(yaw) || yaw === 0) return null
  const pivot = meta.yaw_rotation_center_xy ?? [0, 0]
  return { rotationY: yaw, pivotX: pivot[0], pivotZ: pivot[1] }
}

/** Minimal structural view of a three.js Object3D (either `three` or `three/webgpu`'s -
 * they're distinct classes at the same revision, see msaOverlayTransform's doc). */
export interface OverlayTarget {
  position: { set(x: number, y: number, z: number): unknown }
  rotation: { set(x: number, y: number, z: number): unknown }
}

/** Applies `t` (or identity for null) to a group so that `world = T(pivot) · R_y · T(-pivot) · local`,
 * expressed as three's `rotation.y` plus a compensating `position`: with R the rotation,
 * `world = R · local + (pivot - R · pivot)`. Only ever touches the group's own
 * position/rotation - never the GLB children, never the point cloud. */
export function applyMsaOverlayTransform(group: OverlayTarget, t: MsaOverlayTransform | null): void {
  if (!t) {
    group.rotation.set(0, 0, 0)
    group.position.set(0, 0, 0)
    return
  }
  const c = Math.cos(t.rotationY)
  const s = Math.sin(t.rotationY)
  // R_y(rotationY) applied to the pivot (px, 0, pz), three.js convention.
  const rx = t.pivotX * c + t.pivotZ * s
  const rz = -t.pivotX * s + t.pivotZ * c
  group.rotation.set(0, t.rotationY, 0)
  group.position.set(t.pivotX - rx, 0, t.pivotZ - rz)
}

/** Same rigid transform as `applyMsaOverlayTransform`, expressed as a pure point map
 * (bootstrap's yaw-rotated frame -> the point cloud's own unrotated frame) instead of
 * an Object3D mutation - used to bring `room_polygon` (always in the rotated frame,
 * see MsaSceneMeta) into the cloud's frame so cloud points (in their native
 * coordinates) can be tested against it directly. Identity when `t` is null. */
export function overlayLocalToCloudFrame(x: number, z: number, t: MsaOverlayTransform | null): [number, number] {
  if (!t) return [x, z]
  const c = Math.cos(t.rotationY)
  const s = Math.sin(t.rotationY)
  const rx = t.pivotX * c + t.pivotZ * s
  const rz = -t.pivotX * s + t.pivotZ * c
  return [x * c + z * s + (t.pivotX - rx), -x * s + z * c + (t.pivotZ - rz)]
}

/** `room_polygon`, transformed into the point cloud's own frame with
 * `overlayLocalToCloudFrame`. Null when there's no polygon to transform. */
export function roomPolygonToCloudFrame(
  polygon: readonly (readonly [number, number])[] | undefined,
  t: MsaOverlayTransform | null,
): [number, number][] | null {
  if (!polygon || polygon.length < 3) return null
  return polygon.map(([x, z]) => overlayLocalToCloudFrame(x, z, t))
}

/** Standard ray-casting point-in-polygon test on the XZ plane (Y ignored) - assumes a
 * simple, non-self-intersecting polygon, true of every exported room outline. */
function pointInPolygonXZ(x: number, z: number, polygon: readonly (readonly [number, number])[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, zi] = polygon[i]
    const [xj, zj] = polygon[j]
    const crosses = zi > z !== zj > z && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi
    if (crosses) inside = !inside
  }
  return inside
}

function distanceToSegment(px: number, pz: number, ax: number, az: number, bx: number, bz: number): number {
  const dx = bx - ax
  const dz = bz - az
  const lenSq = dx * dx + dz * dz
  const t = lenSq > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / lenSq)) : 0
  return Math.hypot(px - (ax + t * dx), pz - (az + t * dz))
}

function distanceToPolygonBoundaryXZ(x: number, z: number, polygon: readonly (readonly [number, number])[]): number {
  let best = Infinity
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    best = Math.min(best, distanceToSegment(x, z, polygon[j][0], polygon[j][1], polygon[i][0], polygon[i][1]))
  }
  return best
}

/** True if (x, z) is inside `polygon`, or within `marginM` of its boundary - the
 * "room_polygon + margin" test that clips the point cloud in the MSA-overlaid viewer
 * (see SceneMap3D.tsx's point-cloud build effect). Distance-to-boundary rather than a
 * true polygon offset (Minkowski sum): simpler and still correct for a concave outline
 * like the doorway spur, where naively growing each vertex outward would distort the
 * notch. `polygon` must already be in the same frame as (x, z) - see
 * `roomPolygonToCloudFrame`. No polygon (or a degenerate one) means "don't clip". */
export function isNearRoomPolygonXZ(
  x: number,
  z: number,
  polygon: readonly (readonly [number, number])[] | null,
  marginM: number,
): boolean {
  if (!polygon || polygon.length < 3) return true
  if (pointInPolygonXZ(x, z, polygon)) return true
  return distanceToPolygonBoundaryXZ(x, z, polygon) <= marginM
}
