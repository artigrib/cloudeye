import { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three/webgpu'
import { Line2NodeMaterial } from 'three/webgpu'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { Line2 } from 'three/addons/lines/webgpu/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js'
import { IconEye, IconRotate360 } from '@tabler/icons-react'
import type {
  CameraPoseResponse,
  OccupancyGridResponse,
  SceneLayersResponse,
  SceneObjectResponse,
} from '../api/types'
import Button from './ui/Button'
import Checkbox from './ui/Checkbox'
import SegmentedControl from './ui/SegmentedControl'
import Select from './ui/Select'
import Slider from './ui/Slider'
import Tooltip from './ui/Tooltip'
import {
  cameraClipPlanes,
  easeInOutCubic,
  frameObjectPose,
  frontCameraPose,
  initialCameraPose,
  isometricCameraPose,
  lerpPose,
  topCameraPose,
  type BoundingBox3,
  type CameraPose,
} from '../lib/scene3dCamera'
import {
  buildDecimatedPointCloud,
  clipPointCloudToPolygonXZ,
  createPointCloudMaterials,
  createPointCloudSprite,
  setPointCloudFadeRange,
  type PointCloudUniforms,
} from '../lib/scene3dPoints'
import { createEdlPass, type EdlPass } from '../lib/scene3dEdl'
import { probeDevice } from '../lib/deviceProbe'
import {
  createFrameBudget,
  distanceLodBudget,
  FALLBACK_POINT_BUDGET,
  initialPointBudget,
  loadPointBudget,
  observeFrame,
  pendingBudget,
  savePointBudget,
  type FrameBudgetState,
} from '../lib/pointBudget'
import { loadPointCloudViewSettings, savePointCloudViewSettings, type CeilingMode } from '../lib/pointcloudViewSettings'
import type { RenderMode } from '../lib/renderMode'
import { captureScenePreview } from '../lib/scenePreview'
import { MESH_LAYER_NAME, attachLayerGroup } from '../lib/sceneLayerGroup'
import { useSceneGlb } from '../lib/useSceneGlb'
import {
  applyMsaOverlayTransform,
  classifyNode,
  isNearRoomPolygonXZ,
  msaOverlayTransform,
  roomPolygonToCloudFrame,
  type LayerKey,
  type MsaSceneMeta,
} from '../lib/msaLayers'
import { fetchSceneMsaMeta } from '../api/scenes'
import { dampAngle, headingYaw, shortestAngleDelta } from '../lib/robot-heading'
import { DEFAULT_MODEL_URL, instantiateRobotModel, loadRobotModel, peekRobotModel } from '../lib/robotModelCache'
import { isClickCandidate, isClickGesture, type PointerSample } from '../lib/pointerGesture'
import { classifyWheel } from '../lib/scene3dWheel'
import {
  isNavKey,
  isTypingTarget,
  keyStateFromCodes,
  KEY_PAN_SPEED_PX_PER_SEC,
  KEY_SHIFT_MULTIPLIER,
  panAxesForKeys,
  pixelsToWorldUnits,
  stepDistancePx,
} from '../lib/scene3dKeys'
import {
  ACCENT,
  BORDER,
  BORDER_STRONG,
  CANVAS,
  CLEARANCE_TIGHT,
  DATA_OBJECT,
  DATA_ROBOT,
  DATA_TRAJECTORY,
  FAINT,
  FG,
  OCC_OBSTACLE,
  oklchToSrgb,
  toHex,
  toRgba,
} from '../lib/tokens'
import { computeClearanceField } from '../lib/clearanceField'
import { buildClearancePixels } from '../lib/clearanceRender'
import {
  FLOOR_SUPERSAMPLE,
  buildOccupancyLayerPixels,
  upscaleCellPixels,
} from '../lib/occupancyLayers'
import { computeContours, type Point } from '../lib/marchingSquares'
import { buildFloorGeometry, buildFloorTexture, updateFloorTexture } from '../lib/scene3dFloor'
import { buildContours, disposeContours, type ContourLayer } from '../lib/scene3dContours'

// World metres. 0.03 is the owner's requested default (was 0.015) - a denser, more solid
// cloud at the framings the demo uses. Note what it costs: splat area is quadratic in
// this, so doubling it is up to 4x the fragments per point wherever the [1, 32]
// device-pixel clamp isn't already binding, and it is the single biggest lever on frame
// rate with the camera inside the cloud (see scene3dPoints.ts's pointShapeAlpha).
const DEFAULT_POINT_SIZE = 0.03
const MIN_POINT_SIZE = 0.003
const MAX_POINT_SIZE = 0.06
const ROBOT_FALLBACK_RADIUS = 0.2
// The footprint indicator (disc + ring + pin), replacing what used to be an opaque
// cylinder tall enough to hide the robot model underneath it. Y-offsets keep it clear of
// the GridHelper at y=0: depth precision at this camera's near/far (0.05/500) and a
// typical ~5m view distance is roughly z^2*(1/near - 1/far)/2^24 ~= 3e-5m, so 4-6mm is
// ~130x that noise floor. The bigger reason these don't z-fight, though, is `depthWrite:
// false` + being in the transparent pass (see the indicator effect) - that's the part
// that still holds up at the near-horizon camera angles (maxPolarAngle 89 deg) where a
// few mm of offset alone stops being reliable.
const ROBOT_DISC_Y = 0.004
const ROBOT_RING_Y = 0.006
const ROBOT_RING_WIDTH = 0.012
// The platform's start marker - a thin ring on the floor, drawn under the robot
// indicator's own disc so the two don't z-fight while the robot is still standing on it.
const START_MARKER_Y = 0.003
const START_MARKER_RADIUS_M = 0.14
const START_MARKER_WIDTH_M = 0.02
const ROBOT_PIN_HEIGHT = 0.35 // taller than the real robot (0.191m) so it reads at a glance
const ROBOT_PIN_RADIUS = 0.005
const PATH_Y_OFFSET = 0.03
// Just under PATH_Y_OFFSET so the two never z-fight where a command's planned-path
// preview and the accumulated trail cover the same segment.
const TRAIL_Y_OFFSET = 0.025
// Screen-space pixels (Line2/LineMaterial's default, non-worldUnits mode) - thick enough
// to read clearly in a screen recording, unlike a 1px LineBasicMaterial hairline.
const TRAIL_LINE_WIDTH_PX = 4
// Robot-POV camera height when the selected platform's own dimensions_m.height_m isn't
// known yet (e.g. before the registry loads) - close to the bundled TurtleBot3 Burger's
// real 0.192m, see ROBOT_FALLBACK_RADIUS for the equivalent fallback on the radius side.
const POV_EYE_HEIGHT_FALLBACK_M = 0.15
const POV_LOOK_DISTANCE_M = 2 // how far ahead the POV camera's target point sits
// Just under TRAIL_Y_OFFSET so none of the path/trail/camera-path lines z-fight where
// they happen to cover the same ground.
const CAMERA_PATH_Y_OFFSET = 0.02
const CAMERA_PATH_LINE_WIDTH_PX = 3
const CAMERA_POSE_RADIUS = 0.025 // small enough not to obscure the point cloud (spec)
const CAMERA_POSE_HIGHLIGHT_RADIUS = 0.06
const DAMPING_TAIL_MS = 500
const YAW_SMOOTH_MS = 120
const YAW_SETTLE_EPS = 0.008 // ~0.5 degree; below this the settle loop stops ticking
const YAW_DT_CLAMP_MS = 100 // a backgrounded tab resuming shouldn't snap the heading in one step
const MIN_CAMERA_Y = 0.1 // Q (down) never flies the perspective camera under the floor
const FRAME_TWEEN_MS = 400 // shift+double-click "frame this object" camera tween (3D only)
// 1 maps one degree of finger twist to one degree of camera orbit - a direct 1:1 feel,
// like turning the scene with your fingers. Adjust this single constant if it ever
// feels too fast/slow rather than changing the sign or math below.
const GESTURE_ROTATE_SPEED = 1

// Safari's non-standard two-finger trackpad rotate (and pinch) gesture events - not in
// the TS DOM lib, hence this minimal manually-declared shape. Chrome and Firefox never
// fire these at all (there is no cross-browser signal for a trackpad twist, unlike
// pinch-zoom's ctrlKey+wheel), so the rotate-with-two-fingers feature this drives is
// Safari-only by nature, not by choice - left-drag still orbits in every browser.
interface SafariGestureEvent extends Event {
  rotation: number
}

// Spec 9: one color means one thing, everywhere - object markers are always
// --data-object (amber), never recolored for hover/selection (those get an outline +
// scale bump instead, see the marker-color effect below); trajectory/trail/camera-path
// were three different colors (indigo/magenta/violet) and are now all --data-trajectory;
// the robot model and its footprint disc are the neutral --data-robot, not indigo.
const COLOR_DEFAULT = toHex(DATA_OBJECT)
const COLOR_ACTIVE = toHex(ACCENT)
const COLOR_UNREACHABLE = toHex(FAINT)
const COLOR_ROBOT = toHex(DATA_ROBOT)
const COLOR_PATH = toHex(DATA_TRAJECTORY)
const COLOR_TRAIL = toHex(DATA_TRAJECTORY)
const COLOR_CAMERA_PATH = toHex(DATA_TRAJECTORY)
// Selection outline (spec 8: "выбор ... bounding box в изометрии", using --fg, not a
// second accent color).
const COLOR_BBOX = toHex(FG)
// Metric grid (spec renderer 3): 1m lines --border, 5m/center lines --border-strong -
// GridHelper only has the two-tier center/regular distinction, not a real 1m/5m split
// (that needs the custom LineSegments grid the scene merge introduces).
const COLOR_GRID_CENTER = toHex(BORDER_STRONG)
const COLOR_GRID = toHex(BORDER)

// Floor (spec renderer 4) - sits just under the grid/robot indicator, whose own
// z-fight-avoidance offsets (see ROBOT_DISC_Y etc. above) already assume a y=0 floor
// beneath them.
const FLOOR_Y = -0.002
// Vector obstacle contours (spec renderer 5) - between the floor and the robot
// indicator's disc (ROBOT_DISC_Y=0.004), same z-fight-avoidance logic.
const CONTOUR_Y = 0.001
const CONTOUR_LINE_WIDTH_PX = 1.5
const CONTOUR_FILL_OPACITY = 0.1
// Half a grid cell (spec 5, point 2) - simplification tolerance in cell units.
const CONTOUR_SIMPLIFY_TOLERANCE = 0.5
const COLOR_CONTOUR = toHex(OCC_OBSTACLE)
// Matches app/config.py's ROBOT_RADIUS_M (TurtleBot3 Burger) - used only as a
// placeholder coloring before the scene's own reachability response has seeded the
// real radius, same as OccupancyMap's old DEFAULT_RADIUS_M.
const DEFAULT_RADIUS_M = 0.1
// CLEARANCE_TIGHT isn't a --occ-*/--data-* :root token (see tokens.ts) - only the
// clearance shader/legend need its actual color, so it's resolved to CSS once here
// rather than adding a token that nothing else in the design system references.
const CLEARANCE_TIGHT_RGB = oklchToSrgb(CLEARANCE_TIGHT).map((c) => Math.round(c * 255))
const CLEARANCE_TIGHT_CSS = `rgb(${CLEARANCE_TIGHT_RGB.join(' ')})`

// 2D (orthographic) camera - min/max `camera.zoom`, applied on top of a base frustum
// sized to the room (see fitOrthoToBox). Chosen empirically to roughly bracket
// OccupancyMap's old MIN_SCALE(3)/MAX_SCALE(24) cell-pixel range across typical room
// sizes, not derived from them exactly (the two aren't the same unit).
const MIN_ORTHO_ZOOM = 0.2
const MAX_ORTHO_ZOOM = 40
// Margin multiplier so a freshly fit 2D view doesn't crop the room right at its walls.
const ORTHO_FIT_MARGIN = 1.15

const CEILING_MODE_UNIFORM: Record<CeilingMode, number> = { visible: 0, hidden: 1, fade: 2 }

/** Numeric uCeilingMode for the point shader - forced to `visible` (0) when the scene's
 * ceiling height isn't known yet, regardless of the selected mode, since there's nothing
 * to hide/fade against. */
function ceilingUniformMode(mode: CeilingMode, ceilingY: number | null): number {
  return ceilingY == null ? 0 : CEILING_MODE_UNIFORM[mode]
}

// Selected-object hull highlight: a real convex hull of the object's own points when
// there are enough of them, else its bbox corners (ConvexGeometry of a box's 8 corners
// IS the box) - see build_convex_hull's identical rule server-side, app/services/usd_export.py.
const MAX_HULL_SAMPLE_POINTS = 4000 // keeps ConvexGeometry's QuickHull cheap even off a large decimated cloud

// PM review of the take-1 dry run (2026-09-07): the point cloud spilled outside the
// measured room outline on the west side once the MSA overlay's room_polygon was
// available (msa-meta serving 200 - see docs/DECISIONS.md's "morning-2" entry) -
// captured-but-unmeasured clutter (a hallway rug near the entry threshold, per an
// earlier DECISIONS.md note) sitting past the wall. Clip to `room_polygon` + this
// margin in THIS viewer only (it's the one that overlays MSA data on the cloud); a
// scene with no MSA export has no room_polygon and stays fully unclipped, same as
// every other cloud-only view in the app (scenePreview.ts thumbnails included).
const ROOM_POLYGON_CLOUD_MARGIN_M = 0.3
const MIN_HULL_POINTS = 8 // fewer real points than this reads as noise, not a shape - fall back to the bbox

/** Every point in `positions` whose world coordinates land inside `box`, thinned to at
 * most `cap` by keeping every Nth one (not just the first `cap`) so the sample isn't
 * biased by however the source buffer happens to be ordered. */
function pointsInBox(positions: THREE.BufferAttribute, box: BoundingBox3, cap: number): THREE.Vector3[] {
  const found: THREE.Vector3[] = []
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i)
    const y = positions.getY(i)
    const z = positions.getZ(i)
    if (x < box.min[0] || x > box.max[0]) continue
    if (y < box.min[1] || y > box.max[1]) continue
    if (z < box.min[2] || z > box.max[2]) continue
    found.push(new THREE.Vector3(x, y, z))
  }
  if (found.length <= cap) return found
  const stride = Math.ceil(found.length / cap)
  return found.filter((_, i) => i % stride === 0)
}

function boxCorners(box: BoundingBox3): THREE.Vector3[] {
  const [x0, y0, z0] = box.min
  const [x1, y1, z1] = box.max
  return [
    new THREE.Vector3(x0, y0, z0),
    new THREE.Vector3(x1, y0, z0),
    new THREE.Vector3(x0, y1, z0),
    new THREE.Vector3(x1, y1, z0),
    new THREE.Vector3(x0, y0, z1),
    new THREE.Vector3(x1, y0, z1),
    new THREE.Vector3(x0, y1, z1),
    new THREE.Vector3(x1, y1, z1),
  ]
}

/** The selected object's highlight geometry: a real convex hull of its own captured
 * points sampled from the loaded (decimated) point cloud, falling back to its bbox
 * corners when too few land inside it (a sparse object, or the cloud not loaded yet) -
 * and further to a plain box if even that degenerates (a near-flat bbox QuickHull can't
 * build a solid from). `positions` is null before the point cloud has finished loading. */
function buildSelectionHullGeometry(box: BoundingBox3, positions: THREE.BufferAttribute | null): THREE.BufferGeometry {
  const sampled = positions ? pointsInBox(positions, box, MAX_HULL_SAMPLE_POINTS) : []
  const points = sampled.length >= MIN_HULL_POINTS ? sampled : boxCorners(box)
  try {
    return new ConvexGeometry(points)
  } catch {
    const size = box.max.map((v, i) => Math.max(v - box.min[i], 0.01)) as [number, number, number]
    const geom = new THREE.BoxGeometry(...size)
    geom.translate(...(boxCenter3(box) as [number, number, number]))
    return geom
  }
}

function boxCenter3(box: BoundingBox3): [number, number, number] {
  return [
    (box.min[0] + box.max[0]) / 2,
    (box.min[1] + box.max[1]) / 2,
    (box.min[2] + box.max[2]) / 2,
  ]
}

/** Rebuilds the ortho camera's frustum from a fixed world half-height (the "zoom=1"
 * reference, set once per room by fitOrthoToBox) and the current viewport aspect -
 * `camera.zoom` (driven by orthoControls' wheel handling) scales on top of this,
 * exactly like `camera.aspect` does for the perspective camera's own frustum. */
function applyOrthoFrustum(camera: THREE.OrthographicCamera, aspect: number, baseHalfHeight: number): void {
  camera.top = baseHalfHeight
  camera.bottom = -baseHalfHeight
  camera.left = -baseHalfHeight * aspect
  camera.right = baseHalfHeight * aspect
  camera.updateProjectionMatrix()
}

const BG_RGB = toRgba(CANVAS)
// Matches OccupancyMap's old reachability-overlay canvas (`ctx.fillStyle =
// 'rgba(5, 6, 10, 0.6)'`) - kept as the same literal rather than a token since it was
// never one to begin with, just a fixed darkening tint.
const UNREACHABLE_OVERLAY: readonly [number, number, number, number] = [5, 6, 10, 0.6]

/** `occupancy` shows the raw grid, one cell one pixel; every other render mode shows
 * the clearance field (2D/floor has no per-pixel height or photo data of its own -
 * spec: "не применим -> показывает Clearance"), same rule OccupancyMap's canvas effect
 * used to follow. */
function computeFloorPixels(
  gridResp: OccupancyGridResponse,
  clearanceField: Float32Array,
  colorMode: RenderMode,
  radiusM: number,
  reachableCells: boolean[][] | null,
  layers: SceneLayersResponse | null,
): Uint8ClampedArray<ArrayBuffer> {
  // "Occupancy" is the honest raw view, so it is the one that draws what the
  // reconstruction actually observed: the band mask's obstacle cells, the nvblox pack's
  // measured distance field over them, and a hatch on every cell nobody ever saw. Every
  // other mode renders one pixel per cell as before and is block-upscaled to the same
  // texture size, so switching modes repaints the texture instead of rebuilding it.
  //
  // The hatch comes from `gridResp.unobserved` - the scene's OWN navigation layer, the
  // same file the routes were planned on - and not from the pack, which is matched to a
  // scene by its grid and is absent for most of them. One question, one source.
  const base =
    colorMode === 'occupancy'
      ? buildOccupancyLayerPixels({
          cells: gridResp.cells,
          grid: gridResp,
          layers: layers ? { esdfM: layers.esdf_m, unobserved: layers.unobserved } : null,
          unobserved: gridResp.unobserved ?? null,
          robotRadiusM: radiusM,
        })
      : upscaleCellPixels(
          buildClearancePixels(clearanceField, gridResp.cells, gridResp.width, gridResp.height, radiusM),
          gridResp.width,
          gridResp.height,
        )
  return compositeFloorPixels(base, gridResp.width, gridResp.height, reachableCells)
}

/** Flattens the (possibly semi-transparent, per clearanceColorAt's zone alphas) base
 * pixels onto the app's `--canvas` background - the same backdrop the old stacked
 * Canvas2D elements always sat on (OccupancyMap's viewport div was `bg-canvas`) - then
 * darkens cells outside the robot's actually-reachable area on top, same blend
 * OccupancyMap's separate overlay canvas produced. Baking both into one fully-opaque
 * buffer means the floor's THREE.Material never needs `transparent: true`, so it can't
 * accidentally sort/blend oddly against the (genuinely transparent) point cloud. */
function compositeFloorPixels(
  base: Uint8ClampedArray<ArrayBuffer>,
  width: number,
  height: number,
  reachableCells: boolean[][] | null,
): Uint8ClampedArray<ArrayBuffer> {
  // `base` is FLOOR_SUPERSAMPLE pixels per cell in each axis; the reachable mask is per
  // CELL, so the lookup goes through cellForPixel rather than assuming the two are the
  // same index (which they were, until the hatch needed room inside a cell).
  const ss = FLOOR_SUPERSAMPLE
  const pw = width * ss
  const ph = height * ss
  const out = new Uint8ClampedArray(new ArrayBuffer(pw * ph * 4))
  const [bgR, bgG, bgB] = BG_RGB
  const [dR, dG, dB, dA] = UNREACHABLE_OVERLAY
  for (let px = 0; px < pw; px++) {
    const column = reachableCells?.[Math.floor(px / ss)]
    for (let py = 0; py < ph; py++) {
      const iz = Math.floor(py / ss)
      const off = (py * pw + px) * 4
      const a = base[off + 3] / 255
      let r = base[off] * a + bgR * (1 - a)
      let g = base[off + 1] * a + bgG * (1 - a)
      let b = base[off + 2] * a + bgB * (1 - a)
      if (reachableCells && !column?.[iz]) {
        r = r * (1 - dA) + dR * dA
        g = g * (1 - dA) + dG * dA
        b = b * (1 - dA) + dB * dA
      }
      out[off] = r
      out[off + 1] = g
      out[off + 2] = b
      out[off + 3] = 255
    }
  }
  return out
}

interface Props {
  sceneId: string
  meshUrl: string
  /** THE POINTS LAYER'S ONE SOURCE, resolved by the caller from what the endpoints
   * actually contain (lib/layerAvailability.resolveSceneLayers): exactly one of these is
   * set, and the other is null.
   *
   * `meshUrl`/`cloudUrl` below are what this used to be - both passed unconditionally,
   * with the loader falling back from /cloud to /mesh on a 404. That fallback is what
   * made "the same file drawn twice" possible: on a scene whose /cloud 404s, Points was
   * already reading /mesh, and the Mesh toggle offered the same file again. Points now
   * falls back to /mesh only when /mesh is a POINTS primitive, and the Mesh toggle is
   * enabled only when it is a triangulated one - which are mutually exclusive answers
   * about the same file. */
  pointsCloudUrl?: string | null
  pointsMeshUrl?: string | null
  /** Binary point cloud URL (api/scenes.sceneCloudUrl). Tried before meshUrl; a 404 there
   * is the ordinary case and falls back to the GLB - see glbCache.loadSceneGlb. */
  cloudUrl?: string
  /** Scene's ceiling height, in meters - the height-slice slider's max (and default:
   * the slider starts at the ceiling, meaning nothing is hidden). null while not yet
   * known, in which case the slider is disabled. */
  ceilingY: number | null
  objects: readonly SceneObjectResponse[]
  selectedObjectId: string | null
  onSelectObject: (id: string | null) => void
  hoveredObjectId: string | null
  onHoverObject: (id: string | null) => void
  activeObjectId?: string | null
  /** null while reachability hasn't loaded yet, in which case nothing is greyed out -
   * same convention as OccupancyMap. */
  reachableObjectIds: Set<string> | null
  robotPosition: { x: number; z: number } | null
  /** The SELECTED PLATFORM's start - the nearest cell to the first camera pose that this
   * platform actually fits in, recomputed per radius by the backend
   * (`pathfinding.resolve_start_for_radius`, surfaced as ReachabilityResponse.start_x/z).
   * Marked on the floor in both 2D and 3D so it stays visible after the robot has driven
   * away from it. null when the platform fits nowhere in this scene, or before the first
   * reachability response for the current radius. */
  startPosition: { x: number; z: number } | null
  robotRadius: number | null
  /** The selected platform's real length_m/width_m (RobotPlatformResponse.dimensions_m) -
   * draws the footprint indicator as a rectangle (length along world +x, width along
   * world +z - this component has no robot-heading data, so the rectangle is always
   * axis-aligned, not rotated to match the platform model's actual facing) instead of
   * the radius-only circle. Either undefined falls back to a `robotRadius`-derived
   * square (2*radius x 2*radius), same conservative "still show SOMETHING usable"
   * convention `app.robots.resolve_footprint_m` uses on the backend. */
  robotLengthM?: number | null
  robotWidthM?: number | null
  /** Mesh URL of the currently selected robot platform (e.g. "/models/limo.glb"), relative
   * to the site root - see app/robots.py's `mesh_path` on the backend. Defaults to the
   * bundled TurtleBot3 Burger (`DEFAULT_MODEL_URL`) when omitted/undefined, e.g. before the
   * platform registry has loaded. */
  robotMeshUrl?: string
  /** The selected platform's own height, in meters (RobotPlatformResponse.dimensions_m.height_m)
   * - used as the robot-POV camera's eye height so a taller platform's POV sits higher.
   * Undefined/null falls back to POV_EYE_HEIGHT_FALLBACK_M. */
  robotHeightM?: number | null
  currentPath: [number, number][] | null
  /** Every point the robot has actually travelled across, accumulated command by
   * command - unlike `currentPath` (the latest command's own planned route, replaced
   * each time), this only ever grows and is cleared solely by "Reset robot to start"
   * (see ScenePage). Rendered as the "show trail" toggle's line. */
  trailPoints: [number, number][]
  /** The capture-order camera trajectory - "where the person walked while filming."
   * null hides it entirely (the "camera path" toggle lives in ScenePage, shared with the
   * 2D view - passing null here IS the toggle being off). */
  cameraTrack: CameraPoseResponse[] | null
  /** Index into `cameraTrack` to draw larger/highlighted - the pose nearest the video
   * player's current playback time. null highlights nothing. */
  highlightedPoseIndex: number | null
  /** Double-click a marker: send the robot straight there, no chat round-trip. */
  onGoto: (id: string, name: string) => void
  onUnreachableClick: (name: string) => void
  /** The shared render-mode selector (spec 6) - owned by ScenePage, not this component,
   * since it's cross-view state. Only `height` changes point-cloud coloring today;
   * `clearance`/`occupancy` fall back to `photo` for the *points* (the "points colored
   * by floor clearance" part of the spec's mode table would need per-point clearance
   * lookup, not built). The *floor* (see the floor-texture effect) does support all
   * four modes, same as OccupancyMap did. */
  colorMode: RenderMode
  /** Which camera is live - perspective+orbit ("3d") or the top-down orthographic pan/
   * zoom camera ("2d"). One scene, two cameras (redesign plan section 6): every other
   * prop/object here is shared between both, only the active camera/controls differ. */
  viewMode: '2d' | '3d'
  /** Backs the floor texture and vector contours (both 2D-visual-language concerns that
   * now live on the shared floor rather than a separate Canvas2D/DOM renderer) - null
   * while still loading, in which case neither renders yet. */
  grid: OccupancyGridResponse | null
  /** reachableCells[ix][iz] at the current robot radius - darkens the floor outside the
   * robot's actually-reachable area (distinct from the clearance field's own radius
   * coloring, which is purely geometric and ignores connectivity/obstacles in between).
   * null while not yet loaded, in which case nothing is darkened - same convention the
   * old OccupancyMap used. */
  reachableCells: boolean[][] | null
  /** This scene's nvblox layer pack, or null when it has none (GET /layers 404s) - in
   * which case the Occupancy mode draws the occupancy grid alone and the legend says so.
   * Only ever served on this scene's own grid, so it indexes exactly like `grid`. */
  layers: SceneLayersResponse | null
  /** Whether the MSA mesh layer is shown. Owned by the toolbar now, not by Shift+2 - a
   * hidden chord is not a control anyone finds, and this is the one MSA layer a
   * non-developer has a reason to switch on. The first time it goes true the 42 MB
   * export starts downloading, exactly as the first Shift+2 press used to. Shift+3/4
   * still toggle the collision and plan layers, which remain developer tools. */
  msaMeshVisible: boolean
  /** The four independent layers the 3D toolbar switches (lib/sceneLayers.ts).
   *
   * Each one gates the EFFECT that adds its objects, not their `.visible` flag, so an
   * "off" layer is not in the scene graph at all. That is the fix for the reported
   * "mesh stays drawn": the only switch that existed flipped `.visible` on the nodes
   * inside a group that never left the graph, so anything that re-read the group - or
   * any node the classifier had not matched - kept drawing.
   *
   * `showMap` is ignored in 2D: there, the floor plane IS the map, and a screen with it
   * switched off would have nothing to show. */
  showMap: boolean
  showPoints: boolean
  showMesh: boolean
  /** The GLB served by /mesh. NOTE, because the name is misleading and the endpoint's
   * own docstring says so: this is a POINTS-primitive glTF (gpu/stage_export_glb.py),
   * not a triangulated surface - no endpoint on this deployment serves the nvblox
   * triangle mesh (it exists only under nvblox_v2/results, unserved). On a scene whose
   * /cloud 404s - hero-74 is one - the Points layer has already fallen back to this same
   * file, so Points+Mesh draws it twice. */
  meshLayerUrl?: string | null
  /** Object id -> the backend's own word for why it is not reachable. Shown under the
   * object's name in the hover label, verbatim - the 2D map greys an unreachable marker
   * and this is what says why, without needing the object list open beside it. */
  unreachableReasons: Record<string, string>
  /** Which object a click in the Objects list (not a marker click - see the shift+
   * double-click gesture below for that path) most recently asked to be framed. Paired
   * with `focusNonce` (bumped on every request, even a re-click of the same object) so
   * the framing effect can tell "clicked again" from "no new request" - the id alone
   * can't, since re-selecting an already-focused object wouldn't otherwise change. 3D
   * only, same as shift+double-click: 2D's orthographic camera frames by zoom/pan, not
   * position distance, so tweening its position the same way wouldn't reframe anything. */
  focusObjectId?: string | null
  focusNonce?: number
  /** URL of this scene's optional MSA (Measured Scene Assembly) glTF export (see
   * api/scenes.ts's sceneMsaGlbUrl) - loaded into THIS SAME scene graph, alongside the
   * point cloud, rather than a separate viewer/camera (spec: MSA layers must open in
   * the same viewer, same camera, as the point-cloud view). Undefined/null skips
   * loading entirely (most scenes have no MSA export yet); a 404 is treated the same
   * as "no export" (see useSceneGlb's error handling) - nothing renders, Shift+2..4 are
   * just no-ops. */
  msaGlbUrl?: string | null
  /** URL of this scene's optional MSA scene_meta.json (see api/scenes.ts's
   * sceneMsaMetaUrl). bootstrap yaw-normalizes the GLB's geometry (rotates it by
   * `yaw_correction_rad` about the XZ pivot `yaw_rotation_center_xy`) while the cloud
   * this app is never rotated, so the MSA layer group gets the INVERSE rotation from
   * these fields (lib/msaLayers.msaOverlayTransform) to land on the cloud's walls. A
   * missing/404/malformed meta just leaves the group at identity - same
   * fail-soft contract as msaGlbUrl. */
  msaMetaUrl?: string | null
  /** Keyboard shortcut hook (physical "C" key, 3D view only - see the mount effect's
   * handleKeyDown): cycles the point cloud between height-gradient and true-RGB
   * coloring. Owned by the caller (SceneView), which also owns the render-mode
   * <select> this toggles the same underlying state as - see SceneView.tsx. */
  onToggleColorMode?: () => void
}

/** The four MSA glTF layers this component overlays on the point cloud (SPEC §8's
 * "cloud" layer excluded - see the mount effect's Shift+1 handling doc comment for why
 * Shift+1 toggles the REAL point cloud instead of an always-empty MSA cloud node). */
type MsaMeshLayerKey = Extract<LayerKey, 'mesh' | 'collision' | 'plan'>
const MSA_MESH_LAYER_KEYS: MsaMeshLayerKey[] = ['mesh', 'collision', 'plan']

/** Shift+2/3/4 -> MSA layer, matching MsaSceneViewer.tsx's LAYER_KEYS order (cloud,
 * mesh, collision, plan, gaps) with "cloud" reassigned to Shift+1's real-point-cloud
 * toggle - see handleKeyDown's doc comment for why Shift is required (bare "2"/"3" are
 * already ScenePage.tsx's 2D/3D view-switch hotkeys). */
const DIGIT_TO_MSA_LAYER: Partial<Record<string, MsaMeshLayerKey>> = {
  // Digit2 is deliberately absent: the mesh layer is a toolbar control now (see the
  // msaMeshVisible prop). Leaving the chord in as a second way to reach the same state
  // would let the two disagree - the toolbar showing "off" while the layer is on.
  Digit3: 'collision',
  Digit4: 'plan',
}

/** One 2D map object box: its outline div, the object it draws, and its label with the
 * label's measured width (measured once, on the first frame it is visible - the text
 * never changes, and `offsetWidth` on every label on every frame of a pan would not be
 * free). */
interface ObjectBoxEntry {
  el: HTMLDivElement
  object: SceneObjectResponse
  label: HTMLSpanElement
  labelW: number
}

/** Label geometry for the de-overlap pass. `LABEL_BASE_DY` is the -14px the label's old
 * `-top-3.5` class hard-coded; `LABEL_ROW_PX` is one row's worth of movement away from
 * the box, and `LABEL_H_PX` the 9px type's line box. */
const LABEL_BASE_DY = -14
const LABEL_ROW_PX = 13
const LABEL_H_PX = 12
/** Candidate positions a label tries before it gives up and takes the last one. They
 * alternate up and down from the natural spot, so seven candidates reach +/-39px: past
 * that a label is far enough from its own box to be misleading about which object it
 * names, which is worse than being hard to read. */
const MAX_LABEL_ROWS = 7
/** Capacity of the scratch rect array. Non-fragment objects per scene, with headroom -
 * the hero has 17. Labels past this are drawn without being de-overlapped rather than
 * dropped. */
const MAX_LABELS = 256

export default function SceneMap3D({
  sceneId,
  meshUrl,
  cloudUrl,
  pointsCloudUrl,
  pointsMeshUrl,
  ceilingY,
  objects,
  selectedObjectId,
  onSelectObject,
  hoveredObjectId,
  onHoverObject,
  activeObjectId,
  reachableObjectIds,
  robotPosition,
  startPosition,
  robotRadius,
  robotLengthM,
  robotWidthM,
  robotMeshUrl,
  robotHeightM,
  currentPath,
  trailPoints,
  cameraTrack,
  highlightedPoseIndex,
  onGoto,
  onUnreachableClick,
  colorMode,
  viewMode,
  grid,
  reachableCells,
  layers,
  msaMeshVisible,
  showMap,
  showPoints,
  showMesh,
  meshLayerUrl,
  unreachableReasons,
  focusObjectId,
  focusNonce,
  msaGlbUrl,
  msaMetaUrl,
  onToggleColorMode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const labelRef = useRef<HTMLDivElement>(null)
  const [pointSize, setPointSize] = useState(DEFAULT_POINT_SIZE)
  // Eye-Dome Lighting toggle - view-local like pointSize above, resets on a 2D<->3D
  // remount. On by default; forced off (and the checkbox disabled) if createEdlPass
  // couldn't allocate a half-float render target on this device - see edlSupported below.
  const [edlEnabled, setEdlEnabled] = useState(true)
  const [edlSupported, setEdlSupported] = useState(true)
  // Per-device render budget (see pointBudget.ts) - seeded once in the mount effect
  // (device probe or a valid cache hit) and only ever revised downward, by measured
  // frame time, applied once the current interaction settles (see maybeStopLoop).
  const [pointBudget, setPointBudget] = useState(FALLBACK_POINT_BUDGET)
  // View-local display preference, like pointSize above - resets on a 2D<->3D remount,
  // not persisted (see frontend/README's rationale for pointSize following the same rule).
  const [showRobotModel, setShowRobotModel] = useState(true)
  const [robotModelFailed, setRobotModelFailed] = useState(false)
  // Same view-local, non-persisted rule as showRobotModel above. Default on, per spec.
  const [showTrail, setShowTrail] = useState(true)
  // The Shift+1 point-cloud toggle's state, as a user PREFERENCE rather than as the
  // object's `.visible` flag directly. What actually gets drawn is this AND 3D mode (see
  // the cloud-visibility effect): the cloud is a 3D layer and has no business over the
  // 2D map, but a 2D visit must not silently clear a preference the user set in 3D.
  const [showPointCloud, setShowPointCloud] = useState(true)
  // Robot-POV ("first-person") camera toggle - view-local like showRobotModel/showTrail,
  // resets on a 2D<->3D remount. Off by default: OrbitControls' free camera is still the
  // primary way to look around the scene.
  const [povActive, setPovActive] = useState(false)
  // Navigation-controls help panel, behind the "?" corner button - view-local like the
  // toggles above, closed by default so it never costs screen space unasked. 3D-only,
  // like the whole keyboard/trackpad navigation feature it documents.
  const [showNavHelp, setShowNavHelp] = useState(false)
  // Purely cosmetic: which view-preset segment reads as "pressed" in the SegmentedControl.
  // The camera is free to orbit away from a preset immediately after (OrbitControls has no
  // notion of "current preset"), so this just tracks the last preset clicked, not live state.
  const [viewPreset, setViewPreset] = useState<'top' | 'front' | 'iso'>('iso')

  // Ceiling handling - unlike pointSize/showRobotModel above, this persists per scene id
  // (see pointcloudViewSettings) rather than resetting on every 2D<->3D toggle. Seeded
  // from storage lazily so the very first render already shows the saved value instead
  // of flashing the default for one frame; re-seeded whenever `sceneId` changes since
  // this component doesn't remount across a scene switch. Colour mode is no longer
  // owned here - it's the shared render mode selector, see the `colorMode` prop.
  const [ceilingMode, setCeilingMode] = useState<CeilingMode>(() => loadPointCloudViewSettings(sceneId).ceilingMode)
  // Continuous "Height cutoff" slider (reinstated alongside the Ceiling mode select -
  // see pointcloudViewSettings.ts's doc comment): null means "no explicit cutoff", i.e.
  // fall back to the scene's own ceiling_y wherever this is read below.
  const [heightCutoff, setHeightCutoff] = useState<number | null>(
    () => loadPointCloudViewSettings(sceneId).heightCutoff,
  )
  const loadedSceneIdRef = useRef(sceneId)
  useEffect(() => {
    if (loadedSceneIdRef.current === sceneId) return
    loadedSceneIdRef.current = sceneId
    const settings = loadPointCloudViewSettings(sceneId)
    setCeilingMode(settings.ceilingMode)
    setHeightCutoff(settings.heightCutoff)
  }, [sceneId])
  useEffect(() => {
    savePointCloudViewSettings(sceneId, { ceilingMode, heightCutoff })
  }, [sceneId, ceilingMode, heightCutoff])
  // The threshold actually fed to the shader's uCeilingY uniform: the slider's value
  // when the user has touched it, else the scene's own ceiling height.
  const effectiveCeilingCutoff = heightCutoff ?? ceilingY ?? Infinity

  // The point cloud GLB is 130-150MB and shouldn't download until the user actually
  // asks for 3D - this component is now always mounted (it owns both cameras), so that
  // laziness has to live here instead of in a parent's mount-on-first-3D gate the way
  // it used to (see SceneView.tsx's history). Sticky once true: switching back to 2D
  // shouldn't cancel/redo the download.
  const [hasEverBeen3d, setHasEverBeen3d] = useState(viewMode === '3d')
  useEffect(() => {
    if (viewMode === '3d') setHasEverBeen3d(true)
  }, [viewMode])
  // Which effects take this, and which deliberately do not.
  //
  // TAKE IT: every effect whose dependencies cannot be relied on to change again after
  // mount. `three.current` is assigned inside an async IIFE after `await renderer.init()`,
  // so such an effect runs exactly once, against a null, and never again - the object
  // markers were invisible for the whole session for precisely this reason. Cheap to
  // re-run, so an extra post-init pass costs nothing.
  //
  // DO NOT: the three effects keyed on a LOADED ASSET - the point cloud
  // (`glb.gltf`), the MSA export (`msaGlb.gltf`) and the robot model (`robotMeshUrl`).
  // Their dependency is null until a fetch resolves, which is always well after init, so
  // they already get their post-init run; adding this flag only makes them run a SECOND
  // time and rebuild the asset they just built. Measured on the 15 fps scene open,
  // median of three runs: 4.37 s with them ungated, 5.98 s with all of them gated, over
  // the probe's own 5 s bar. Gating everything uniformly was the more obviously
  // consistent change and the wrong one.
  // True once `three.current` exists. `three.current` is assigned inside an async IIFE,
  // after `await renderer.init()`, so EVERY effect that starts `const t = three.current;
  // if (!t) return` runs at least once against a null and silently does nothing. That is
  // harmless for effects whose deps change again later, and permanent for those whose
  // don't - which is exactly what left the default 2D view rendering through the
  // PERSPECTIVE camera (the view-mode effect below never re-ran, because `viewMode`
  // never changed) and left the ortho camera unframed whenever `grid` happened to
  // arrive before init finished. This flag is a dependency those effects can take to
  // get their one guaranteed post-init run.
  // Bumped once per SUCCESSFUL `await renderer.init()`. Every effect that starts
  // `const t = three.current; if (!t) return` runs at least once against a null, because
  // `three.current` is assigned inside an async IIFE after that await - harmless for
  // effects whose deps change again later, permanent for those whose don't. That is what
  // left the default 2D view rendering through the PERSPECTIVE camera (the view-mode
  // effect's deps never changed) and left the ortho camera unframed whenever `grid`
  // arrived before init finished. This is the dependency those effects take to get their
  // one guaranteed post-init run.
  //
  // A COUNTER, not the boolean it used to be, and that is load-bearing for the WebGPU ->
  // WebGL2 guard below. The boolean was set false in the teardown and true again after
  // the rebuild's init - but `WebGLBackend.init()` resolves in a microtask, before React
  // flushes the re-render the teardown's update scheduled, so React coalesced false and
  // true into one no-op update, bailed out of re-rendering, and NO effect's deps ever
  // changed. Measured: after a fallback the scene graph held 4 objects where it had held
  // 98 - lights, the grid helper and the point cloud (which had its own dep) and nothing
  // else. A value that only ever increases cannot be coalesced away.
  const [rendererGeneration, setRendererGeneration] = useState(0)
  // The one-shot WebGPU -> WebGL2 guard. `THREE.WebGPURenderer` picks its backend once,
  // at construction; there is no way to switch a live one. So the guard bumps this epoch,
  // the mount effect below takes it as a dependency and rebuilds the whole renderer, and
  // `forceWebGL` is read off `webglFallbackRef` on the way through.
  //
  // The <canvas> is keyed on this epoch too, and that is not cosmetic: a canvas that has
  // already handed out a 'webgpu' context can never return a 'webgl2' one, so React has
  // to mount a fresh element or the rebuilt renderer has nothing to draw into.
  const [rendererEpoch, setRendererEpoch] = useState(0)
  const webglFallbackRef = useRef(false)
  // Camera state does not survive the effect that owns the cameras. Captured in the
  // teardown, re-applied after the rebuild's init, so a fallback is invisible except for
  // one console line.
  const poseSnapshotRef = useRef<{
    persp: [number, number, number]
    target: [number, number, number]
    orthoZoom: number
  } | null>(null)

  // Exactly one source, decided before the fetch. `pointsMeshUrl ?? pointsCloudUrl` for
  // the GLB url and `pointsCloudUrl` for the binary: when the source is the binary those
  // are the same URL and loadSceneGlb never reaches its fallback; when it is /mesh there
  // is no cloud url to try first. The legacy meshUrl/cloudUrl pair is the fallback for a
  // caller that has not resolved a source - only the tests do that now.
  const sourceResolved = pointsCloudUrl !== undefined || pointsMeshUrl !== undefined
  const pointsUrl = sourceResolved ? (pointsMeshUrl ?? pointsCloudUrl ?? '') : meshUrl
  const pointsBinaryUrl = sourceResolved ? (pointsCloudUrl ?? undefined) : cloudUrl
  const glb = useSceneGlb(sceneId, pointsUrl, hasEverBeen3d && Boolean(pointsUrl), pointsBinaryUrl)
  // Small (~100KB), unlike the point-cloud GLB above - loaded whenever a URL is given,
  // not gated on hasEverBeen3d. Cache key is sceneId + a suffix (not just sceneId) so
  // this doesn't collide with the point-cloud GLB's own cache entry in glbCache.ts,
  // which is keyed purely by scene id.
  // The MSA export is NOT requested on scene open - it is requested by the first
  // Shift+2/3/4 press and never before. Between 2cd9fbc and this change it was fetched
  // eagerly here, and on the hero scene that meant 44,195,740 bytes (23 embedded
  // textures) parsed by GLTFLoader on the main thread with no Worker and no Draco/
  // Meshopt decoder (lib/glbCache.ts). Measured on scene open: 19,933 ms of longtask
  // over the first 30 s, longest single task 3,193 ms, and the point cloud's own /mesh
  // could not even start inside that window. The three MSA layers all live in this one
  // file, so any of their keys triggers the load - see handleKeyDown.
  // THE MESH LAYER - /mesh, on its own cache key and its own scene-graph group, fetched
  // only while the toggle is on. Separate from `glb` above even though that loader falls
  // back to the same URL when /cloud 404s: these are two independent toggles and each has
  // to be able to be the only one on.
  const wantMeshLayer = Boolean(meshLayerUrl) && showMesh
  const meshLayerGlb = useSceneGlb(
    wantMeshLayer ? `${sceneId}:meshlayer` : null,
    meshLayerUrl ?? '',
    wantMeshLayer,
  )

  const [msaGlbRequested, setMsaGlbRequested] = useState(false)
  const msaGlbRequestedRef = useRef(false)
  const wantMsaGlb = Boolean(msaGlbUrl) && msaGlbRequested
  const msaGlb = useSceneGlb(wantMsaGlb ? `${sceneId}:msa` : null, msaGlbUrl ?? '', wantMsaGlb)
  // The GLB's yaw meta (see Props.msaMetaUrl) - null until fetched, and stays null for
  // "no export"/"not rotated"/any fetch failure (fetchSceneMsaMeta's fail-soft contract)
  // - the overlay effect below then just leaves the MSA group at identity.
  const [msaMeta, setMsaMeta] = useState<MsaSceneMeta | null>(null)
  useEffect(() => {
    setMsaMeta(null)
    if (!msaMetaUrl) return
    const ctrl = new AbortController()
    fetchSceneMsaMeta(msaMetaUrl, ctrl.signal)
      .then((meta) => {
        if (!ctrl.signal.aborted) setMsaMeta(meta)
      })
      .catch(() => {
        /* aborted or network failure - leave the overlay at identity */
      })
    return () => ctrl.abort()
  }, [rendererGeneration, msaMetaUrl])
  // The single THREE.Group every MSA layer object lives under (the GLB root today;
  // gaps markers would go here too) so the inverse-yaw overlay transform is one
  // position/rotation on one node - see the MSA overlay effect below.
  const msaGroupRef = useRef<THREE.Group | null>(null)

  // Latest callback props, readable from the native pointer handlers set up once in the
  // mount effect below - avoids re-attaching DOM listeners on every render.
  const onSelectObjectRef = useRef(onSelectObject)
  onSelectObjectRef.current = onSelectObject
  const onHoverObjectRef = useRef(onHoverObject)
  onHoverObjectRef.current = onHoverObject
  const selectedObjectIdRef = useRef(selectedObjectId)
  selectedObjectIdRef.current = selectedObjectId
  const onGotoRef = useRef(onGoto)
  onGotoRef.current = onGoto
  const onUnreachableClickRef = useRef(onUnreachableClick)
  onUnreachableClickRef.current = onUnreachableClick
  const onToggleColorModeRef = useRef(onToggleColorMode)
  onToggleColorModeRef.current = onToggleColorMode
  const objectsRef = useRef(objects)
  objectsRef.current = objects
  const reachableObjectIdsRef = useRef(reachableObjectIds)
  reachableObjectIdsRef.current = reachableObjectIds
  const robotPositionRef = useRef(robotPosition)
  robotPositionRef.current = robotPosition
  const showRobotModelRef = useRef(showRobotModel)
  showRobotModelRef.current = showRobotModel
  const showTrailRef = useRef(showTrail)
  showTrailRef.current = showTrail
  const showPointCloudRef = useRef(showPointCloud)
  showPointCloudRef.current = showPointCloud
  const povActiveRef = useRef(povActive)
  povActiveRef.current = povActive
  // Gates the whole keyboard/trackpad-gesture/shift-frame navigation feature below to
  // the 3D (perspective) camera - none of it has an orthographic equivalent (no FOV,
  // no orbit), and the 2D camera already has its own OrbitControls-driven pan/zoom.
  // Read from window-level listeners set up once in the mount effect, so it has to be
  // a ref, same idiom as povActiveRef above.
  const viewModeRef = useRef(viewMode)
  viewModeRef.current = viewMode
  const robotHeightMRef = useRef(robotHeightM)
  robotHeightMRef.current = robotHeightM
  const edlEnabledRef = useRef(edlEnabled)
  edlEnabledRef.current = edlEnabled

  // All actual Three.js objects live in refs and are mutated imperatively - there is no
  // per-frame render loop (see requestRenderOnce/startLoop), so React state isn't the
  // right place for any of this.
  //
  // `camera`/`controls` are the *active* pair - swapped between the persp/ortho ones
  // below whenever `viewMode` changes (see the view-mode effect) - every existing
  // call site (requestRenderOnce, pickMarker, applyCameraPose, updateLabelPosition,
  // the resize handler's Line2 stuff) reads through these two fields generically and
  // doesn't need to know which camera is actually live.
  const three = useRef<{
    renderer: THREE.WebGPURenderer
    scene: THREE.Scene
    perspCamera: THREE.PerspectiveCamera
    orthoCamera: THREE.OrthographicCamera
    perspControls: OrbitControls
    orthoControls: OrbitControls
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera
    controls: OrbitControls
  } | null>(null)

  // GPU-instanced Sprite, not a THREE.Points object - WebGPU/TSL hardcodes Points
  // primitives to 1px, so variable point size only works via instanced sprite
  // billboarding - see scene3dPoints.ts's createPointCloudSprite/createPointCloudMaterials.
  const pointsRef = useRef<THREE.Sprite | null>(null)
  // PointsNodeMaterial has no `.uniforms` bag the way a classic ShaderMaterial did - the
  // live-mutable uniform() nodes are tracked separately here instead, set alongside
  // pointsRef in the build effect below.
  const pointUniformsRef = useRef<PointCloudUniforms | null>(null)
  // The decimated point cloud's own world-space position attribute, kept around (not
  // just read locally in the build effect) so the selected-object hull effect below can
  // filter it down to one object's own points without holding a second copy in memory -
  // same "share the buffer, don't copy" rule buildDecimatedPointCloud already follows.
  const pointPositionsRef = useRef<THREE.BufferAttribute | null>(null)
  // The loaded MSA GLB's nodes, classified into layers (see lib/msaLayers.classifyNode)
  // and grouped for the Shift+2/3/4 visibility toggles below - rebuilt whenever msaGlb
  // finishes loading (see the MSA layer effect). "cloud" is deliberately not tracked
  // here: MSA's own cloud layer is always empty for a bootstrap-only export (see
  // docs/DECISIONS.md's "MSA Stage E1" entry), so Shift+1 toggles the real point cloud
  // (pointsRef) instead - see MSA_MESH_LAYER_KEYS/handleKeyDown.
  const msaLayerObjectsRef = useRef<Record<MsaMeshLayerKey, THREE.Object3D[]>>({
    mesh: [],
    collision: [],
    plan: [],
  })
  // Current on/off state per layer, mutated by the Shift+2/3/4 handlers and applied to
  // every object in msaLayerObjectsRef[key] both immediately (on toggle) and once more
  // when a new GLB load repopulates that array (see the MSA layer effect) - so toggling
  // a layer before the GLB has finished loading still "sticks" once it does. Mesh
  // defaults to visible so that the Shift+2 that STARTS the download (the export is not
  // fetched on scene open - see msaGlbRequested) shows the mesh as soon as it arrives;
  // collision/plan start hidden, same as MsaSceneViewer's own defaults, and their own
  // first press sets them visible explicitly in handleKeyDown.
  const msaLayerVisibleRef = useRef<Record<MsaMeshLayerKey, boolean>>({
    mesh: true,
    collision: false,
    plan: false,
  })
  const edlRef = useRef<EdlPass | null>(null)
  const pointBudgetStateRef = useRef<FrameBudgetState>(createFrameBudget(FALLBACK_POINT_BUDGET))
  // Set by the render loop's tick when observeFrame produces a fresh downshift; consumed
  // (and cleared) by maybeStopLoop once the interaction actually settles - never applied
  // mid-drag, since re-decimating the cloud is itself a visible stutter.
  const pointBudgetPendingRef = useRef<number | null>(null)
  // The device-tier + frame-time corrected budget BEFORE any distance-LOD scaling -
  // distanceLodBudget (pointBudget.ts) always scales off this, never off the `pointBudget`
  // state itself, so repeated zoom-out/zoom-in cycles can't compound a shrink into
  // nothing. Updated on the initial seed and whenever a genuine frame-time downshift
  // lands (see maybeStopLoop); read (not written) by the distance-LOD calc there.
  const basePointBudgetRef = useRef(FALLBACK_POINT_BUDGET)
  // True whenever the camera has been at rest for DAMPING_TAIL_MS (see maybeStopLoop) -
  // i.e. the render loop isn't ticking. Gates EDL in requestRenderOnce: the eye-dome pass
  // is a second, more expensive render+composite of the whole point cloud, so it only
  // runs on the settled/idle frame, never on every one of the ~60 loop ticks a drag
  // produces (see requestRenderOnce below and docs/DECISIONS.md's viewer-fps entry).
  // Starts true - the camera hasn't moved yet at mount, so the very first frame is
  // already "idle".
  const cameraIdleRef = useRef(true)
  // Which sceneId the camera was last auto-posed for - guards the point-cloud build
  // effect's initialCameraPose/cameraClipPlanes call so a budget-driven rebuild (same
  // scene, fewer points) never yanks the camera back to the initial framing the way a
  // real scene change should.
  const posedSceneRef = useRef<string | null>(null)
  const bboxRef = useRef<BoundingBox3 | null>(null)
  const gridRef = useRef<THREE.GridHelper | null>(null)
  // World half-height (at zoom=1) of the ortho camera's base frustum, fit to the room
  // once its bbox is known (see fitOrthoToBox) - the resize handler re-derives
  // left/right/top/bottom from this and the current aspect on every resize, and
  // `camera.zoom` (driven by orthoControls' own wheel handling) scales on top of it.
  const orthoBaseHalfHeightRef = useRef(5)
  const floorMeshRef = useRef<THREE.Mesh | null>(null)
  const floorTextureRef = useRef<THREE.CanvasTexture | null>(null)
  // Computed once per grid (spec 4.1: "поле не зависит от робота") by the floor-geometry
  // effect, then reused by the recolor effect below whenever colorMode/radius/reachable
  // change - never recomputed for those, only the color mapping is.
  const clearanceFieldRef = useRef<Float32Array | null>(null)
  const contoursRef = useRef<ContourLayer | null>(null)
  const markerMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map())
  // Two separate scene objects rather than one root with children: the indicator rebuilds
  // on every radius-slider tick, the model must never react to the radius at all (it's
  // physical, not a scaled abstraction - see the model effect below). Keeping them as
  // separate objects with separate effects makes that decoupling structural, not just a
  // rule someone has to remember.
  const robotIndicatorRef = useRef<THREE.Group | null>(null)
  const robotModelRef = useRef<THREE.Group | null>(null)
  const robotYawRef = useRef(0)
  const robotYawTargetRef = useRef(0)
  const prevRobotPosRef = useRef<{ x: number; z: number } | null>(null)
  const lastYawTickRef = useRef<number | null>(null)
  const yawRafRef = useRef<number | null>(null)
  const pathLineRef = useRef<THREE.Line | null>(null)
  const trailLineRef = useRef<Line2 | null>(null)
  const trailMaterialRef = useRef<InstanceType<typeof Line2NodeMaterial> | null>(null)
  const cameraPathLineRef = useRef<Line2 | null>(null)
  const cameraPathMaterialRef = useRef<InstanceType<typeof Line2NodeMaterial> | null>(null)
  // One small sphere mesh per camera pose, each with its own material so the
  // highlight effect can recolor a single one without touching the rest.
  const cameraPoseMeshesRef = useRef<THREE.Mesh[]>([])
  // The selected object's convex-hull highlight (fill + outline), rebuilt whenever the
  // selection changes - see the hull-building effect below. Falls back to a hull built
  // from the object's own bbox corners (visually indistinguishable from a box) when too
  // few of the loaded point cloud's own points land inside it, the same "hull unless
  // unusable, then bbox" rule the USD export's build_convex_hull already follows server-
  // side (app/services/usd_export.py).
  const hullMeshRef = useRef<THREE.Mesh | null>(null)
  const hullEdgesRef = useRef<THREE.LineSegments | null>(null)
  const labelObjectRef = useRef<SceneObjectResponse | null>(null)
  // The 2D map's labelled object boxes: one absolutely-positioned <div> per object,
  // built when the object list or the reachability set changes and only re-POSITIONED
  // per frame. DOM rather than three.js geometry because half of each box is text, and
  // a top-down orthographic camera projects an axis-aligned bbox to an axis-aligned
  // rectangle exactly - so a div is not an approximation of the box here, it is the box.
  const objectBoxesRef = useRef<HTMLDivElement>(null)
  const objectBoxEntriesRef = useRef<ObjectBoxEntry[]>([])
  // Scratch for the label de-overlap pass in updateObjectBoxes: placed label rects as
  // runs of four (left, top, right, bottom). One array for the life of the component,
  // because that pass runs on every frame of a pan or a zoom.
  const placedLabelsRef = useRef<Float64Array>(new Float64Array(MAX_LABELS * 4))
  // Read inside updateLabelPosition, which runs on every render tick and must not
  // re-create itself per prop change - same "latest ref" idiom as onGotoRef below.
  const unreachableReasonsRef = useRef(unreachableReasons)
  unreachableReasonsRef.current = unreachableReasons
  // The free-camera pose active right before switching into robot-POV, so toggling POV
  // off restores it instead of leaving the user at wherever the robot last was.
  const prePovPoseRef = useRef<CameraPose | null>(null)
  // Latest applyPovPose, readable from startYawSettle below despite applyPovPose being
  // declared later in this file (applyCameraPose, which it depends on, sits right before
  // pickMarker) - same "latest ref" idiom as onGotoRef etc. above, just for a callback
  // instead of a prop.
  const applyPovPoseRef = useRef<(() => void) | null>(null)

  const loopRunningRef = useRef(false)
  const rafRef = useRef<number | null>(null)
  //: The deferred post-resize render (see the ResizeObserver). Held so it can be
  //: cancelled on unmount - a render scheduled for a frame that arrives after
  //: `renderer.dispose()` would submit against a destroyed device.
  const resizeRafRef = useRef<number | null>(null)
  // Replaces the old mount-effect-local `stopTimer` closure variable - holdLoop/
  // releaseLoop below need it from outside that effect too (applyCameraPose,
  // startFrameTween, toggleRobotPov), not just OrbitControls' own start/end events.
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastFrameTickRef = useRef<number | null>(null)
  // Latest per-frame hook, called from the render loop's tick before controls.update()
  // - same "latest ref" idiom as applyPovPoseRef above. Assigned once handleFrame (the
  // keyboard stepper + framing-tween driver) exists further down.
  const onFrameRef = useRef<((dtMs: number) => void) | null>(null)
  // Currently-held navigation key codes (see scene3dKeys.ts) and whether Shift is
  // currently down - Shift accelerates only while held, never toggled. 3D-only.
  const keysHeldRef = useRef<Set<string>>(new Set())
  const shiftHeldRef = useRef(false)
  // The in-flight shift+double-click "frame this object" tween, if any. 3D-only.
  const tweenRef = useRef<{ from: CameraPose; to: CameraPose; startMs: number } | null>(null)

  const updateLabelPosition = useCallback(() => {
    const t = three.current
    const labelEl = labelRef.current
    const container = containerRef.current
    const obj = labelObjectRef.current
    if (!t || !labelEl || !container || !obj) {
      if (labelEl) labelEl.style.display = 'none'
      return
    }
    const vec = new THREE.Vector3(obj.pos_x, obj.pos_y, obj.pos_z).project(t.camera)
    if (vec.z > 1) {
      labelEl.style.display = 'none'
      return
    }
    const rect = container.getBoundingClientRect()
    const x = (vec.x * 0.5 + 0.5) * rect.width
    const y = (-vec.y * 0.5 + 0.5) * rect.height
    // Name, and - for an object this robot cannot reach - the backend's own reason
    // under it, printed exactly as the API sent it (see lib/unreachableReason.ts). The
    // 2D map greys an unreachable marker; without this, "why" needed the object list.
    //
    // Rebuilt only when the TEXT changes, not on every render tick: this runs from
    // requestRenderOnce, i.e. once per frame while the camera is moving, and the label
    // is a fixed two-line structure that only ever changes when the pointer moves to a
    // different object.
    const reason = unreachableReasonsRef.current[obj.id]
    const key = `${obj.id}|${reason ?? ''}`
    if (labelEl.dataset.labelKey !== key) {
      labelEl.dataset.labelKey = key
      labelEl.replaceChildren()
      const nameEl = document.createElement('div')
      nameEl.textContent = obj.name
      labelEl.appendChild(nameEl)
      if (reason) {
        const reasonEl = document.createElement('div')
        reasonEl.textContent = reason
        reasonEl.className = 'font-mono text-[10px] text-data-unreachable'
        reasonEl.dataset.hoverReason = reason
        labelEl.appendChild(reasonEl)
      }
    }
    labelEl.style.display = 'block'
    labelEl.style.transform = `translate(${x + 10}px, ${y - 8}px)`
  }, [])

  /** Re-projects every labelled object box. Runs in the same per-frame pass as the
   * hover label, from requestRenderOnce, so the boxes track pan and zoom exactly. 2D
   * only: in a perspective view an axis-aligned bbox does NOT project to an
   * axis-aligned rectangle, and a div would start lying about where the object is. */
  const updateObjectBoxes = useCallback(() => {
    const t = three.current
    const container = containerRef.current
    const host = objectBoxesRef.current
    if (!t || !container || !host) return
    if (viewModeRef.current !== '2d') {
      host.style.display = 'none'
      return
    }
    host.style.display = 'block'
    const rect = container.getBoundingClientRect()
    const project = (x: number, z: number) => {
      const v = new THREE.Vector3(x, 0, z).project(t.camera)
      return [(v.x * 0.5 + 0.5) * rect.width, (-v.y * 0.5 + 0.5) * rect.height] as const
    }
    // Placed labels, as [left, top, right, bottom] runs of 4 in one reused array - this
    // runs on every frame of a pan or a zoom, so it allocates nothing per frame.
    const placed = placedLabelsRef.current
    let placedCount = 0

    for (const entry of objectBoxEntriesRef.current) {
      const { el, object, label } = entry
      const [x0, y0] = project(object.bbox_min_x, object.bbox_min_z)
      const [x1, y1] = project(object.bbox_max_x, object.bbox_max_z)
      const left = Math.min(x0, x1)
      const top = Math.min(y0, y1)
      el.style.transform = `translate(${left}px, ${top}px)`
      el.style.width = `${Math.max(Math.abs(x1 - x0), 2)}px`
      el.style.height = `${Math.max(Math.abs(y1 - y0), 2)}px`

      // Objects in one room put their bounding boxes' top-left corners within a few
      // pixels of each other, and every label was pinned to its own corner with
      // `whitespace-nowrap` and no z-order, so overlapping names printed on top of one
      // another and neither could be read. Measured on the hero scene through
      // demo/probe_occupancy.py: 1 overlapping pair of 17 labels before this
      // ('television' over 'curtain', 279 px2), 0 after.
      //
      // The label keeps its object's x and moves in whole 13px rows: up first, then down,
      // then further up, then further down. Up first because that is away from the box
      // and cannot land on its own outline; down as the alternative because a box near
      // the top of the canvas has no room above it, and the host clips - a label pushed
      // off the top is simply gone. After MAX_LABEL_ROWS it takes the last candidate
      // rather than vanishing: a label that is hard to read still names something.
      if (entry.labelW === 0) entry.labelW = label.offsetWidth
      const w = entry.labelW || 1
      let dy = 0
      for (let row = 0; row < MAX_LABEL_ROWS; row++) {
        // 0, -13, +13, -26, +26, ...
        dy = ((row + 1) >> 1) * LABEL_ROW_PX * (row % 2 === 1 ? -1 : 1)
        const lt = top + LABEL_BASE_DY + dy
        // Above the canvas is not a placement: the host is overflow-hidden.
        if (lt < 0) continue
        const lb = lt + LABEL_H_PX
        let clear = true
        for (let i = 0; i < placedCount * 4; i += 4) {
          if (left < placed[i + 2] && left + w > placed[i] && lt < placed[i + 3] && lb > placed[i + 1]) {
            clear = false
            break
          }
        }
        if (clear) break
      }
      const lt = top + LABEL_BASE_DY + dy
      if (placedCount * 4 < placed.length) {
        placed[placedCount * 4] = left
        placed[placedCount * 4 + 1] = lt
        placed[placedCount * 4 + 2] = left + w
        placed[placedCount * 4 + 3] = lt + LABEL_H_PX
        placedCount++
      }
      // `top-0` plus a transform, rather than the old `-top-3.5`: one place decides where
      // a label sits, and it is this loop.
      label.style.transform = `translateY(${LABEL_BASE_DY + dy}px)`
    }
  }, [])

  const requestRenderOnce = useCallback(() => {
    const t = three.current
    if (!t) return
    const edl = edlRef.current
    const points = pointsRef.current
    // Idle-gated (cameraIdleRef, see its doc comment): EDL is a second render+composite
    // pass of the whole point cloud, so it only runs once the camera has settled, never
    // on every tick of an active drag/pan/zoom.
    const edlOn = edlEnabledRef.current && edl !== null && points !== null && cameraIdleRef.current
    if (edlOn) {
      edl!.renderPrepass(t.renderer, t.scene, t.camera, points!, [
        trailLineRef.current,
        cameraPathLineRef.current,
      ])
    }
    // Unchanged: same canvas, same MSAA, same colors - EDL never touches this render.
    t.renderer.setRenderTarget(null)
    t.renderer.render(t.scene, t.camera)
    if (edlOn) edl!.composite(t.renderer)
    updateLabelPosition()
    updateObjectBoxes()
  }, [updateLabelPosition])

  // Rendering on demand still needs a short-lived per-frame loop while OrbitControls'
  // damping inertia is settling - without it, `enableDamping` has nothing driving its
  // `.update()` calls and just looks frozen until the next unrelated render. Keyboard
  // movement and the framing tween below (both 3D-only) reuse this same loop via
  // onFrameRef rather than running their own rAF.
  const startLoop = useCallback(() => {
    if (loopRunningRef.current) return
    loopRunningRef.current = true
    cameraIdleRef.current = false
    lastFrameTickRef.current = null
    const tick = () => {
      if (!loopRunningRef.current) return
      const now = performance.now()
      const dtMs = lastFrameTickRef.current === null ? 0 : now - lastFrameTickRef.current
      lastFrameTickRef.current = now
      onFrameRef.current?.(dtMs)
      three.current?.controls.update()
      requestRenderOnce()
      // Frame-time correction for the point budget - see pointBudget.ts. Consecutive
      // frames only exist while this loop is running (on-demand rendering otherwise), so
      // an active drag/pan/tween is the only time worth sampling, and exactly when a
      // too-high budget would actually be felt.
      const prevBudgetState = pointBudgetStateRef.current
      const nextBudgetState = observeFrame(prevBudgetState, dtMs)
      pointBudgetStateRef.current = nextBudgetState
      const pending = pendingBudget(prevBudgetState, nextBudgetState)
      if (pending !== null) pointBudgetPendingRef.current = pending
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [requestRenderOnce])

  const stopLoop = useCallback(() => {
    loopRunningRef.current = false
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  // Refuses to stop while a key is held or the framing tween is running, so the
  // DAMPING_TAIL_MS timer below can't cut off a longer key-hold or the 400ms tween.
  const maybeStopLoop = useCallback(() => {
    if (keysHeldRef.current.size > 0 || tweenRef.current) return
    const t = three.current
    // Apply a measured downshift now, not mid-drag - re-decimating the point cloud is
    // itself a visible stutter, so it only ever happens once the interaction is over.
    const pending = pointBudgetPendingRef.current
    if (pending !== null) {
      pointBudgetPendingRef.current = null
      basePointBudgetRef.current = pending
      if (t) {
        const s = t.renderer.getDrawingBufferSize(new THREE.Vector2())
        savePointBudget(pending, s.x * s.y)
      }
    }
    // Distance LOD (pointBudget.ts's distanceLodBudget): re-evaluated on every settle,
    // not only when a frame-time downshift just happened, so zooming back in after a
    // zoomed-out reduction restores full detail too. Always scales off basePointBudgetRef
    // (the frame-time-corrected value above), never off the possibly-already-LOD-reduced
    // `pointBudget` state, so repeated zoom cycles can't compound a shrink. 3D-only - the
    // 2D top-down view has no meaningful "camera distance from the room" to react to.
    const box = bboxRef.current
    let nextBudget = basePointBudgetRef.current
    if (t && box && viewModeRef.current === '3d') {
      const center = new THREE.Vector3(
        (box.min[0] + box.max[0]) / 2,
        (box.min[1] + box.max[1]) / 2,
        (box.min[2] + box.max[2]) / 2,
      )
      const distance = t.camera.position.distanceTo(center)
      const roomDiagonal = Math.hypot(box.max[0] - box.min[0], box.max[2] - box.min[2])
      nextBudget = distanceLodBudget(basePointBudgetRef.current, distance, roomDiagonal)
    }
    setPointBudget((prev) => (prev === nextBudget ? prev : nextBudget))
    cameraIdleRef.current = true
    stopLoop()
    // One more render now that the camera (and cameraIdleRef) has actually settled - the
    // last frame the loop drew was rendered while still moving, i.e. with EDL gated off;
    // without this, nothing would repaint the now-idle, EDL-eligible frame until some
    // unrelated prop change happened to trigger one.
    requestRenderOnce()
  }, [stopLoop, requestRenderOnce])

  // Starts (or keeps alive) the render loop, cancelling any pending stop - shared by
  // OrbitControls' own 'start' event, a keydown, and the start of a wheel-pan/tween.
  const holdLoop = useCallback(() => {
    if (stopTimerRef.current) {
      clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
    startLoop()
  }, [startLoop])

  // Schedules the loop to stop after DAMPING_TAIL_MS, same as the old inline stopTimer
  // - but via maybeStopLoop, so it backs off if something else is still driving frames.
  const releaseLoop = useCallback(() => {
    if (stopTimerRef.current) clearTimeout(stopTimerRef.current)
    stopTimerRef.current = setTimeout(() => {
      stopTimerRef.current = null
      maybeStopLoop()
    }, DAMPING_TAIL_MS)
  }, [maybeStopLoop])

  const cancelYawSettle = useCallback(() => {
    if (yawRafRef.current !== null) {
      cancelAnimationFrame(yawRafRef.current)
      yawRafRef.current = null
    }
  }, [])

  // A short self-cancelling rAF that keeps damping the robot's yaw toward its target after
  // robotPosition stops changing (a move ends, or the driving useCommandAnimation loop's
  // last tick already landed close but not exactly on target). Without this the yaw effect
  // below - which only runs when robotPosition changes - would leave the model frozen a
  // few degrees short forever. Same shape as startLoop/stopLoop above (a temporary rAF
  // settling an inertial value after its driving input stops); shares lastYawTickRef with
  // the position effect, which cancels this loop on every tick so the two never integrate
  // dt in the same frame.
  const startYawSettle = useCallback(() => {
    if (yawRafRef.current !== null) return
    const step = () => {
      yawRafRef.current = null
      const now = performance.now()
      const dtMs = Math.min(now - (lastYawTickRef.current ?? now), YAW_DT_CLAMP_MS)
      lastYawTickRef.current = now
      robotYawRef.current = dampAngle(robotYawRef.current, robotYawTargetRef.current, dtMs, YAW_SMOOTH_MS)
      const model = robotModelRef.current
      if (model) model.rotation.y = robotYawRef.current
      if (povActiveRef.current) applyPovPoseRef.current?.()
      requestRenderOnce()
      if (Math.abs(shortestAngleDelta(robotYawRef.current, robotYawTargetRef.current)) > YAW_SETTLE_EPS) {
        yawRafRef.current = requestAnimationFrame(step)
      }
    }
    yawRafRef.current = requestAnimationFrame(step)
  }, [requestRenderOnce])

  const applyCameraPose = useCallback(
    (pose: CameraPose) => {
      const t = three.current
      if (!t) return
      // A preset button or POV toggle mid-tween means "go here now" - let the tween's
      // own damping restore (below) still happen, just cancel the tween itself.
      if (tweenRef.current) {
        tweenRef.current = null
        t.controls.enableDamping = true
      }
      t.camera.position.set(...pose.position)
      t.controls.target.set(...pose.target)
      t.camera.updateProjectionMatrix()
      t.controls.update()
      requestRenderOnce()
    },
    [requestRenderOnce],
  )

  // Frames the ortho camera on the occupancy grid's own footprint - deliberately NOT
  // derived from the point cloud's bbox (bboxRef, set only once the 130-150MB GLB has
  // loaded): 2D mode has to work for a user who never switches to 3D at all, and the
  // (lightweight, always-fetched) grid already carries the room's real XZ extent.
  const fitOrthoToGrid = useCallback(() => {
    const t = three.current
    if (!t || !grid) return
    const sizeX = grid.width * grid.resolution
    const sizeZ = grid.height * grid.resolution
    const centerX = grid.origin_x + sizeX / 2
    const centerZ = grid.origin_z + sizeZ / 2
    orthoBaseHalfHeightRef.current = (Math.max(sizeX, sizeZ) / 2) * ORTHO_FIT_MARGIN
    const rect = containerRef.current?.getBoundingClientRect()
    const aspect = rect && rect.height > 0 ? rect.width / rect.height : 1
    applyOrthoFrustum(t.orthoCamera, aspect, orthoBaseHalfHeightRef.current)
    t.orthoCamera.zoom = 1
    t.orthoCamera.position.set(centerX, Math.max(sizeX, sizeZ, 10), centerZ)
    t.orthoCamera.updateProjectionMatrix()
    t.orthoControls.target.set(centerX, 0, centerZ)
    t.orthoControls.update()
    requestRenderOnce()
  }, [grid, requestRenderOnce])

  // `rendererGeneration` (not just `fitOrthoToGrid`): whenever the grid resolved before
  // `renderer.init()` did, the callback's one run found `three.current` null and the
  // ortho camera kept its construction-time unit frustum.
  useEffect(() => {
    fitOrthoToGrid()
  }, [fitOrthoToGrid, rendererGeneration])

  // Snaps the camera to the robot's current position/heading, looking POV_LOOK_DISTANCE_M
  // ahead along its facing direction. No-op while the robot's position isn't known yet.
  // Forward-direction convention pinned by robot-heading.ts's module doc: a yaw theta about
  // scene +Y sends local forward (1,0,0) to (cos theta, 0, -sin theta).
  const applyPovPose = useCallback(() => {
    const pos = robotPositionRef.current
    if (!pos) return
    const yaw = robotYawRef.current
    const eyeY = robotHeightMRef.current ?? POV_EYE_HEIGHT_FALLBACK_M
    const forwardX = Math.cos(yaw)
    const forwardZ = -Math.sin(yaw)
    applyCameraPose({
      position: [pos.x, eyeY, pos.z],
      target: [pos.x + forwardX * POV_LOOK_DISTANCE_M, eyeY, pos.z + forwardZ * POV_LOOK_DISTANCE_M],
    })
  }, [applyCameraPose])
  applyPovPoseRef.current = applyPovPose

  // Starts (or restarts) the shift+double-click "frame this object" tween (3D-only): an
  // easeInOutCubic dolly from the current pose to `to`, driven by handleFrame below.
  // Damping is switched off for the tween's duration - `update()` zeroes
  // `_sphericalDelta`/`_panOffset` in one frame while it's off (see OrbitControls'
  // `update()`), which flushes any leftover drag/wheel inertia so it can't fight the
  // interpolated pose we write each frame - and restored once the tween completes (or
  // is cancelled: see applyCameraPose and handleStart in the mount effect below).
  const startFrameTween = useCallback(
    (to: CameraPose) => {
      const t = three.current
      if (!t) return
      const from: CameraPose = {
        position: [t.camera.position.x, t.camera.position.y, t.camera.position.z],
        target: [t.controls.target.x, t.controls.target.y, t.controls.target.z],
      }
      t.controls.enableDamping = false
      tweenRef.current = { from, to, startMs: performance.now() }
      holdLoop()
    },
    [holdLoop],
  )

  // Per-frame driver for keyboard movement and the framing tween (both 3D-only),
  // called from the render loop's tick (see startLoop's onFrameRef hook) before
  // controls.update() runs. A tween owns the camera outright for its duration -
  // keyboard input is ignored until it finishes, rather than the two fighting over the
  // same frame.
  const handleFrame = useCallback(
    (dtMs: number) => {
      const t = three.current
      if (!t) return

      const tween = tweenRef.current
      if (tween) {
        const k = Math.min(1, (performance.now() - tween.startMs) / FRAME_TWEEN_MS)
        const pose = lerpPose(tween.from, tween.to, easeInOutCubic(k))
        t.camera.position.set(...pose.position)
        t.controls.target.set(...pose.target)
        if (k >= 1) {
          tweenRef.current = null
          t.controls.enableDamping = true
          releaseLoop()
        }
        return
      }

      if (viewModeRef.current !== '3d') return
      if (povActiveRef.current) return
      if (keysHeldRef.current.size === 0) return
      // Pausing (not clearing) held keys while a text field has focus covers the case
      // where a key was already held down before the user clicked into e.g. the
      // command box without releasing it - the isTypingTarget guard on keydown alone
      // only stops a *new* key from arming movement, not one already held.
      if (isTypingTarget(document.activeElement)) return

      const axes = panAxesForKeys(keyStateFromCodes(keysHeldRef.current))
      const speed = KEY_PAN_SPEED_PX_PER_SEC * (shiftHeldRef.current ? KEY_SHIFT_MULTIPLIER : 1)
      const dist = stepDistancePx(speed, dtMs)

      if (axes.right !== 0 || axes.forward !== 0) {
        // controls.pan(deltaX, deltaY) moves the camera opposite deltaX (drag-right ->
        // camera-left) and along the floor-plane view direction for positive deltaY
        // (screenSpacePanning=false) - negating right and keeping forward as-is here
        // reproduces "D moves right" / "W moves the way the camera looks" out of
        // three's own pan math, no custom camera vectors needed.
        t.controls.pan(-axes.right * dist, axes.forward * dist)
      }

      if (axes.up !== 0) {
        const clientHeight = canvasRef.current?.clientHeight ?? 0
        const targetDistance = t.camera.position.distanceTo(t.controls.target)
        const fov = t.camera instanceof THREE.PerspectiveCamera ? t.camera.fov : 50
        let dy = pixelsToWorldUnits(axes.up * dist, fov, targetDistance, clientHeight)
        // Moving position and target by the identical delta leaves (position - target)
        // unchanged, so the next controls.update() recomputes the same spherical
        // radius/phi it already had - Q/E rises and falls without fighting
        // maxPolarAngle or drifting the orbit angle.
        if (t.camera.position.y + dy < MIN_CAMERA_Y) dy = MIN_CAMERA_Y - t.camera.position.y
        t.camera.position.y += dy
        t.controls.target.y += dy
      }
    },
    [releaseLoop],
  )
  onFrameRef.current = handleFrame

  function pickMarker(clientX: number, clientY: number): string | null {
    const t = three.current
    const canvas = canvasRef.current
    if (!t || !canvas) return null
    const rect = canvas.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    )
    const raycaster = new THREE.Raycaster()
    raycaster.setFromCamera(ndc, t.camera)
    // Only marker meshes are raycast targets - intersecting the point cloud itself
    // would be both meaningless (hitting one point in a room-scale cloud) and slow
    // across hundreds of thousands of points.
    const hits = raycaster.intersectObjects(Array.from(markerMeshesRef.current.values()), false)
    return hits.length > 0 ? ((hits[0].object.userData.objectId as string) ?? null) : null
  }

  // Mount once: renderer/scene/camera/controls, resize handling, pointer picking.
  //
  // WebGPURenderer.render() throws if called before `await renderer.init()` resolves -
  // everything below up to (and including) the ResizeObserver/event-listener setup is
  // safe to run pre-init (setSize/setPixelRatio/getDrawingBufferSize all operate on
  // internal state constructed before init; nothing here calls .render()). Only
  // `three.current` assignment and the initial requestRenderOnce() - the two things that
  // let `t.renderer.render(...)` actually run - wait on init, in the async block near the
  // end of this effect. `cancelled` handles React 19 StrictMode's synchronous dev-mode
  // mount->cleanup->remount: if this effect is torn down before init resolves, the
  // cleanup below still fully disposes the renderer (documented-safe pre-init) and the
  // async continuation's `if (cancelled) return` guard skips touching three.current or
  // rendering once init eventually does resolve.
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    let cancelled = false
    // `forceWebGL` is false on the first build and true for the life of the tab once the
    // guard has fired - see webglFallbackRef above.
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      forceWebGL: webglFallbackRef.current,
    })
    // Capped at 1.5, not the device's real DPR (commonly 2-3 on retina/high-DPI
    // displays): this is a fill-rate-dominated point-cloud scene (hundreds of thousands
    // of overlapping alpha-blended sprites), so every extra fraction of DPR costs a
    // roughly quadratic number of extra shaded fragments for a resolution bump that's
    // barely visible on point-sprite edges. 1.5x is still enough headroom over 1x to
    // avoid visibly blocky sprite edges, at a fraction of 2x's fill-rate cost - see
    // docs/DECISIONS.md's viewer-fps entry for the measured FPS delta.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
    renderer.setClearColor(toHex(CANVAS), 1)

    const scene = new THREE.Scene()
    const perspCamera = new THREE.PerspectiveCamera(55, 1, 0.05, 500)
    perspCamera.position.set(3, 3, 3)

    const perspControls = new OrbitControls(perspCamera, renderer.domElement)
    perspControls.enableDamping = true
    perspControls.dampingFactor = 0.08
    perspControls.maxPolarAngle = (89 * Math.PI) / 180
    // Dolly toward the pointer rather than the orbit target - makes navigating into a
    // corner feel natural instead of the room sliding sideways as you zoom.
    perspControls.zoomToCursor = true
    // minDistance keeps zoomToCursor from letting the orbit radius collapse toward 0
    // (which would make rotation direction ill-defined right next to a surface) - the
    // near plane is 0.05, so 0.15 stays comfortably clear of it.
    perspControls.minDistance = 0.15
    // false (rather than the default true) is what makes vertical pan - both drag-pan
    // and WASD/arrow "forward" below - walk the floor plane at a constant height
    // instead of flying up/down with the camera's current tilt. This is also what
    // changes right-drag pan from rising/falling with the view to walking the floor, a
    // deliberate behaviour change alongside the keyboard/trackpad navigation work.
    perspControls.screenSpacePanning = false
    // Trading middle-drag's default DOLLY for PAN matches common CAD/DCC convention
    // (wheel/pinch already dolly) - middle and right both pan, left orbits, and
    // ctrl/meta/shift+left-drag panning is already stock OrbitControls behaviour.
    perspControls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN }

    edlRef.current = createEdlPass(renderer)
    setEdlSupported(edlRef.current !== null)

    // 2D view (spec renderer 6): a top-down orthographic camera, panned/zoomed via a
    // SECOND OrbitControls instance bound to it rather than hand-rolled pan/zoom math -
    // three's own OrbitControls already handles ortho-camera dolly (scales
    // camera.zoom, not distance) and screen-space panning correctly, including
    // zoom-toward-cursor (zoomToCursor below), so there's no reason to re-derive that.
    // Rotation is disabled outright: this camera only ever looks straight down.
    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.05, 500)
    const orthoControls = new OrbitControls(orthoCamera, renderer.domElement)
    orthoControls.enableRotate = false
    orthoControls.screenSpacePanning = true
    orthoControls.zoomToCursor = true
    orthoControls.enableDamping = true
    orthoControls.dampingFactor = 0.08
    orthoControls.minZoom = MIN_ORTHO_ZOOM
    orthoControls.maxZoom = MAX_ORTHO_ZOOM
    // OrbitControls' default mouseButtons.LEFT is ROTATE - with rotate disabled above,
    // a left-drag would otherwise do nothing at all. Map left-drag (and one-finger
    // touch) to PAN instead, matching OccupancyMap's old "drag to pan" convention.
    orthoControls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }
    orthoControls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN }
    // Both controls attach listeners to the same canvas unconditionally - only the
    // active pair's `.enabled` is true at any given time (see the view-mode effect),
    // same mechanism the POV toggle already used to disable perspControls outright.
    orthoControls.enabled = false

    // The app's only lights. Everything that existed before this - the point cloud
    // (PointsMaterial), the object markers/bbox helper/robot indicator (MeshBasicMaterial),
    // the path line (LineBasicMaterial) - is unlit and completely unaffected:
    // ShaderLib.basic/points/line declare `lights: false`, so WebGLPrograms never injects
    // light uniforms into those programs. Only the robot model's MeshStandardMaterial
    // samples these.
    //
    // If the model looks too hot or too flat, the ONLY two knobs to touch are
    // keyLight.intensity and ambient.intensity. Do NOT set renderer.outputColorSpace,
    // renderer.toneMapping, or THREE.ColorManagement.enabled - three already runs this
    // renderer fully sRGB-managed by default (verified: WebGLRenderer's constructor sets
    // outputColorSpace = SRGBColorSpace, and ColorManagement.enabled is true out of the
    // box), so touching any of those would re-encode every existing unlit color in the
    // scene: the marker palette would desaturate, the point cloud's per-vertex COLOR_0
    // would darken hard through the midtones, and the dark grid lines (COLOR_GRID) would
    // linearize to near-black and vanish against the clear color.
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2)
    keyLight.position.set(2.5, 6, 3.5) // direction only - DirectionalLight has no falloff,
    scene.add(keyLight) // so it never needs to follow the robot around the room
    const ambient = new THREE.AmbientLight(0xffffff, 0.55) // lifts the shadow side off the clear color
    scene.add(ambient)

    const rect = container.getBoundingClientRect()
    renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false)
    const initialAspect = Math.max(1, rect.width) / Math.max(1, rect.height)
    perspCamera.aspect = initialAspect
    perspCamera.updateProjectionMatrix()
    applyOrthoFrustum(orthoCamera, initialAspect, orthoBaseHalfHeightRef.current)

    const handleStart = () => {
      // A user drag/wheel taking over mid-tween should cancel it outright rather than
      // fight it next frame - restore damping the same way applyCameraPose does.
      if (tweenRef.current) {
        tweenRef.current = null
        perspControls.enableDamping = true
      }
      holdLoop()
    }
    const handleEnd = () => {
      releaseLoop()
    }
    const handleChange = () => {
      if (!loopRunningRef.current) requestRenderOnce()
    }
    perspControls.addEventListener('start', handleStart)
    perspControls.addEventListener('end', handleEnd)
    perspControls.addEventListener('change', handleChange)
    orthoControls.addEventListener('start', handleStart)
    orthoControls.addEventListener('end', handleEnd)
    orthoControls.addEventListener('change', handleChange)

    const ro = new ResizeObserver(() => {
      const r = container.getBoundingClientRect()
      const w = Math.max(1, r.width)
      const h = Math.max(1, r.height)
      renderer.setSize(w, h, false)
      const aspect = w / h
      perspCamera.aspect = aspect
      perspCamera.updateProjectionMatrix()
      applyOrthoFrustum(orthoCamera, aspect, orthoBaseHalfHeightRef.current)
      // Unlike the old raw-GLSL point material, PointsNodeMaterial's sizeAttenuation
      // resyncs its own perspective-scale uniform every frame via onFrameUpdate (see
      // scene3dPoints.ts's PointCloudUniforms doc comment) - no manual uScale sync
      // needed here anymore. Line2NodeMaterial similarly reads the renderer's real
      // viewport internally (LineSegments2.onBeforeRender) rather than a `resolution`
      // property the app has to keep in sync - no manual resolution.set() either.
      edlRef.current?.setSize(renderer)
      // Contours (scene3dContours.ts) are Line2NodeMaterial too now, so like the
      // trail/camera-path lines above they read the renderer's viewport themselves and
      // need no manual resolution sync here. (They used the classic LineMaterial, which
      // WebGPURenderer's NodeBuilder rejects outright - see scene3dContours.ts.)
      //
      // NOT `requestRenderOnce()` here, synchronously. `renderer.setSize` destroys and
      // reallocates the backend's canvas-sized targets; rendering in the same callback
      // submits a command buffer that can still reference the texture that was just
      // destroyed. Under WebGL that is invisible - the driver is forgiving - and under
      // WebGPU it is a validation error that kills the frame:
      //
      //   THREE.WebGPURenderer: Uncaptured WebGPU GPUValidationError: Destroyed texture
      //   [Texture (unlabeled 740x666 px, TextureFormat::RGBA16Float)] used in a submit.
      //
      // reported ~40x from a real Chrome on one 2D->3D->2D round trip, the size
      // alternating 740x666 / 740x693 - which is exactly this canvas's height changing
      // as the 2D legend bar enters and leaves the flow. Deferring one frame lets the
      // backend finish reallocating before anything is submitted against it.
      if (resizeRafRef.current !== null) cancelAnimationFrame(resizeRafRef.current)
      resizeRafRef.current = requestAnimationFrame(() => {
        resizeRafRef.current = null
        requestRenderOnce()
      })
    })
    ro.observe(container)

    let pointerDown: PointerSample | null = null
    let lastHoverId: string | null = null

    const handlePointerDown = (e: PointerEvent) => {
      // A middle/right-button press (both used for panning, in either camera) must not
      // arm the click detector - and must clear any stale candidate, since a left press
      // followed by a second button going down mid-drag is a pan, never a click.
      if (!isClickCandidate(e)) {
        pointerDown = null
        return
      }
      pointerDown = { pointerId: e.pointerId, button: e.button, x: e.clientX, y: e.clientY, timeMs: performance.now() }
    }
    const handlePointerUp = (e: PointerEvent) => {
      const down = pointerDown
      pointerDown = null
      // Shift is reserved for the 3D shift+double-click frame gesture below - a
      // shift-click must never toggle selection.
      if (e.shiftKey || !down) return
      const up: PointerSample = { pointerId: e.pointerId, button: e.button, x: e.clientX, y: e.clientY, timeMs: performance.now() }
      if (!isClickGesture(down, up)) return
      const id = pickMarker(e.clientX, e.clientY)
      if (id) onSelectObjectRef.current(id === selectedObjectIdRef.current ? null : id)
    }
    const handlePointerCancel = () => {
      pointerDown = null
    }
    const handlePointerMove = (e: PointerEvent) => {
      const id = pickMarker(e.clientX, e.clientY)
      if (id !== lastHoverId) {
        lastHoverId = id
        onHoverObjectRef.current(id)
      }
      canvas.style.cursor = id ? 'pointer' : 'auto'
    }
    const handleDoubleClick = (e: MouseEvent) => {
      const id = pickMarker(e.clientX, e.clientY)
      if (!id) return
      const obj = objectsRef.current.find((o) => o.id === id)
      if (!obj) return
      if (e.shiftKey && viewModeRef.current === '3d') {
        // Shift+double-click recentres/frames the camera on the object (3D only - no
        // orthographic equivalent) - the plain double-click below is already the "send
        // the robot here" command, so this branches out before touching the robot at all.
        onSelectObjectRef.current(id)
        const box: BoundingBox3 = {
          min: [obj.bbox_min_x, obj.bbox_min_y, obj.bbox_min_z],
          max: [obj.bbox_max_x, obj.bbox_max_y, obj.bbox_max_z],
        }
        const cam = three.current?.camera.position
        const from: [number, number, number] = cam ? [cam.x, cam.y, cam.z] : [0, 0, 0]
        startFrameTween(frameObjectPose(box, from))
        return
      }
      const reachable = reachableObjectIdsRef.current
      if (reachable && !reachable.has(id)) {
        onUnreachableClickRef.current(obj.name)
      } else {
        // Select, don't toggle. The two `click`s the browser fires before this one have
        // already run the pointerup handler's toggle twice, so the object is back to
        // whatever it was; a double-click that plans a route to an object must leave
        // that object picked. No camera move here either - framing is shift+double-click,
        // handled above and 3D-only.
        onSelectObjectRef.current(id)
        onGotoRef.current(id, obj.name)
      }
    }

    // Two-finger trackpad scroll and pinch both arrive as `wheel` events - ctrlKey is
    // the only thing telling them apart (the OS synthesizes it for a pinch). Registered
    // on the CONTAINER, not the canvas, with capture:true: the canvas is a *descendant*
    // of the container, so this listener's capture phase is guaranteed to run before
    // OrbitControls' own canvas-level 'wheel' listener, regardless of registration
    // order - that's DOM event-path ordering, not a race (see classifyWheel). 3D-only:
    // the 2D camera's own OrbitControls already does scroll-to-zoom, unmodified.
    const handleContainerWheel = (e: WheelEvent) => {
      if (e.target !== canvas) return // overlay sliders/selects scroll normally
      if (viewModeRef.current !== '3d') return
      // Always prevent default: blocks macOS page-zoom on pinch AND two-finger
      // horizontal overscroll -> browser back/forward navigation on pan. This also
      // closes a stock OrbitControls gap where its own preventDefault sits behind an
      // enabled/state check, so pinching during POV or mid-drag zooms the whole page.
      e.preventDefault()
      if (povActiveRef.current) {
        e.stopPropagation()
        return
      }
      if (e.ctrlKey) return // pinch -> fall through to OrbitControls' own dolly handler
      e.stopPropagation() // two-finger scroll -> ours, OrbitControls never sees it
      const classified = classifyWheel(e)
      if (classified.kind === 'pan') three.current?.controls.pan(classified.panX, classified.panY)
    }
    // OrbitControls already suppresses the context menu on the canvas, but only while
    // `controls.enabled` - which is false during robot-POV - so right-click during POV
    // falls through to the OS menu. This closes that one gap; applies in both camera
    // modes since the right button is mapped to pan in both. Everywhere else on the
    // page (including this view's own buttons/sliders) keeps its normal menu.
    const handleContainerContextMenu = (e: MouseEvent) => {
      if (e.target !== canvas) return
      e.preventDefault()
    }
    container.addEventListener('wheel', handleContainerWheel, { capture: true, passive: false })
    container.addEventListener('contextmenu', handleContainerContextMenu)

    // Two-finger trackpad rotate ("twist") orbits the (3D) camera around the target,
    // the same as a left-drag - Safari-only (see SafariGestureEvent), and unlike the
    // wheel-based pan/zoom above there is no non-Safari signal to fall back to for this
    // specific gesture, so it's simply absent elsewhere; left-drag orbit still works
    // everywhere including Safari. Doesn't go through OrbitControls' own pointer
    // machinery at all, so the render loop is held/released by hand rather than via its
    // 'start'/'end' events. No orthographic equivalent - it only ever orbits.
    let lastGestureRotationDeg = 0
    const handleGestureStart = (e: Event) => {
      if (e.target !== canvas) return
      e.preventDefault()
      if (viewModeRef.current !== '3d' || povActiveRef.current) return
      if (tweenRef.current) {
        tweenRef.current = null
        perspControls.enableDamping = true
      }
      lastGestureRotationDeg = (e as SafariGestureEvent).rotation ?? 0
      holdLoop()
    }
    const handleGestureChange = (e: Event) => {
      if (e.target !== canvas) return
      e.preventDefault()
      if (viewModeRef.current !== '3d' || povActiveRef.current) return
      const rotationDeg = (e as SafariGestureEvent).rotation ?? 0
      const deltaDeg = rotationDeg - lastGestureRotationDeg
      lastGestureRotationDeg = rotationDeg
      // Positive `rotation` is a clockwise twist (WebKit convention); rotateLeft with a
      // positive angle is the same direction OrbitControls' own rotateLeft takes for a
      // rightward mouse drag - so a clockwise twist orbits the same way a right-drag
      // would, keeping this consistent with the existing drag gesture.
      three.current?.controls.rotateLeft(deltaDeg * (Math.PI / 180) * GESTURE_ROTATE_SPEED)
    }
    const handleGestureEnd = (e: Event) => {
      if (e.target !== canvas) return
      e.preventDefault()
      if (viewModeRef.current !== '3d' || povActiveRef.current) return
      releaseLoop()
    }
    container.addEventListener('gesturestart', handleGestureStart)
    container.addEventListener('gesturechange', handleGestureChange)
    container.addEventListener('gestureend', handleGestureEnd)

    // Keyboard nav (3D-only, see handleFrame) lives on window, not the canvas: the
    // canvas isn't focusable. isTypingTarget keeps WASD/arrows/QE out of the command
    // box, this view's own selects, and the video player's seek controls.
    const handleKeyDown = (e: KeyboardEvent) => {
      shiftHeldRef.current = e.shiftKey // update even for a bare Shift press, and even while typing
      if (e.metaKey || e.ctrlKey || e.altKey) return // never swallow browser/OS shortcuts
      if (viewModeRef.current !== '3d' || povActiveRef.current) return
      if (isTypingTarget(document.activeElement)) return

      // Point cloud color mode toggle (spec: height-gradient <-> true-RGB, both driven
      // by the same uColorMode uniform - see scene3dPoints.ts). Owned by the caller
      // (SceneView), which also owns the <select> this toggles the same state as.
      if (e.code === 'KeyC') {
        e.preventDefault()
        onToggleColorModeRef.current?.()
        return
      }

      // MSA layer toggles (spec: 1=cloud, 2=mesh, 3=collision, 4=plan - the fixed order
      // MsaSceneViewer.tsx's own LAYER_KEYS already establishes) - SHIFT+1..4, not bare
      // digits: ScenePage.tsx's own "Spec 9 hotkeys" already bind bare "2"/"3" (checked
      // via e.key, not e.code) to switching the whole view between 2D and 3D - a real,
      // pre-existing global `window` keydown listener with no viewMode guard of its own,
      // confirmed by testing (bare Digit2 here silently also flipped the app into the 2D
      // orthographic camera - exactly the "separate camera" the MSA-layers spec forbids).
      // Holding Shift changes `e.key` for these to "!"/"@"/"#"/"$" (US layout), which
      // ScenePage's `e.key === '2'/'3'` checks don't match - no other global hotkey
      // claims Shift+digit today (checked: ScenePage, ProjectPage, VideoPlayer). Digit1
      // toggles the REAL point cloud rather than MSA's own (always-empty, for a
      // bootstrap-only export - see docs/DECISIONS.md) cloud node - see
      // msaLayerObjectsRef's doc comment above.
      if (e.shiftKey && e.code === 'Digit1') {
        e.preventDefault()
        // Flips the preference, not the object - the cloud-visibility effect below owns
        // `.visible`. Pressing this in 2D therefore changes what 3D will show rather
        // than un-hiding the cloud over the map. (This is also why the key looked dead
        // in 2D: it flipped `.visible` on an object 2D should not have been drawing at
        // all - see HANDOFF risk 3c, take 1 shots 05-06 differing by 1.00%.)
        setShowPointCloud((v) => !v)
        return
      }
      const msaLayer = e.shiftKey ? DIGIT_TO_MSA_LAYER[e.code] : undefined
      if (msaLayer) {
        e.preventDefault()
        // First press starts the download (see msaGlbRequested) and means "show me this
        // layer" - toggling here instead would set the layer to hidden and load 42 MB
        // for nothing. Later presses toggle as they always did.
        if (!msaGlbRequestedRef.current) {
          msaGlbRequestedRef.current = true
          setMsaGlbRequested(true)
          msaLayerVisibleRef.current[msaLayer] = true
          for (const obj of msaLayerObjectsRef.current[msaLayer]) obj.visible = true
          requestRenderOnce()
          return
        }
        const nextVisible = !msaLayerVisibleRef.current[msaLayer]
        msaLayerVisibleRef.current[msaLayer] = nextVisible
        for (const obj of msaLayerObjectsRef.current[msaLayer]) obj.visible = nextVisible
        requestRenderOnce()
        return
      }

      if (!isNavKey(e.code)) return
      e.preventDefault() // only for keys we actually consume, so nothing else is stolen
      const wasEmpty = keysHeldRef.current.size === 0
      keysHeldRef.current.add(e.code)
      if (wasEmpty) holdLoop()
    }
    const handleKeyUp = (e: KeyboardEvent) => {
      shiftHeldRef.current = e.shiftKey
      keysHeldRef.current.delete(e.code)
      if (keysHeldRef.current.size === 0) releaseLoop()
    }
    // A held key whose keyup never arrives - losing focus to another window/tab via
    // Cmd-Tab, or the tab itself backgrounding - would otherwise drift the camera
    // forever. Belt-and-suspenders with handleFrame's own isTypingTarget check above.
    const clearHeldKeys = () => {
      if (keysHeldRef.current.size === 0) return
      keysHeldRef.current.clear()
      releaseLoop()
    }
    const handleWindowBlur = () => clearHeldKeys()
    const handleVisibilityChange = () => {
      if (document.hidden) clearHeldKeys()
    }
    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', handleWindowBlur)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    canvas.addEventListener('pointerdown', handlePointerDown)
    canvas.addEventListener('pointerup', handlePointerUp)
    canvas.addEventListener('pointercancel', handlePointerCancel)
    canvas.addEventListener('pointermove', handlePointerMove)
    canvas.addEventListener('dblclick', handleDoubleClick)

    void (async () => {
      await renderer.init()
      if (cancelled) return

      three.current = {
        renderer,
        scene,
        perspCamera,
        orthoCamera,
        perspControls,
        orthoControls,
        camera: perspCamera,
        controls: perspControls,
      }

      // Dev-only inspection handle. The viewer has no data-testid and no other way to
      // ask "what is actually in the scene graph", so a Playwright probe cannot tell a
      // misoriented mesh from a correctly oriented one, or a hidden layer from a
      // removed one - it only sees pixels. Guarded by import.meta.env.DEV, so it exists
      // under `vite dev` (what demo/ probes drive) and is dropped from a production
      // build entirely. Read-only by convention: probes traverse it, never mutate it.
      if (import.meta.env.DEV) {
        ;(window as unknown as { __cloudeyeThree?: unknown }).__cloudeyeThree = three.current
      }
      setRendererGeneration((g) => g + 1)

      // Arm the one-shot guard. An uncaptured WebGPU validation error does NOT raise -
      // three logs it and the frame is silently lost, which is exactly how "3D is black"
      // reached the owner with a clean `pageerror` count. Nothing here tries to identify
      // WHICH error it is: any uncaptured device error means this backend has already
      // dropped at least one frame on the floor, and the honest response is to stop using
      // it rather than to keep guessing which ones are survivable.
      const device = (renderer.backend as { device?: GPUDevice } | undefined)?.device
      if (device && !webglFallbackRef.current) {
        const fallBack = (why: string) => {
          if (webglFallbackRef.current) return
          webglFallbackRef.current = true
          // Logged exactly once, deliberately: forty of these in one round trip is what
          // the original report looked like, and forty log lines would be no better.
          console.warn(
            `THREE.WebGPURenderer reported ${why}; falling back to WebGL2 for the rest of ` +
              'this session. The view is rebuilt in place and the camera is preserved.',
          )
          setRendererEpoch((e) => e + 1)
        }
        device.addEventListener('uncapturederror', (event) => {
          const err = (event as GPUUncapturedErrorEvent).error
          fallBack(`an uncaptured ${err?.constructor?.name ?? 'device error'}`)
        })
        // A lost device cannot be rendered with at all; `reason: 'destroyed'` is our own
        // dispose() and must not trigger anything.
        void device.lost.then((info) => {
          if (info.reason !== 'destroyed') fallBack(`a lost device (${info.reason})`)
        })
      }

      // Re-apply the camera the previous renderer was showing. Only ever set by this
      // component's own teardown, so it can only be a rebuild - a fresh mount has a null
      // ref and falls through to the normal framing.
      const pose = poseSnapshotRef.current
      if (pose) {
        poseSnapshotRef.current = null
        perspCamera.position.set(...pose.persp)
        perspControls.target.set(...pose.target)
        perspControls.update()
        orthoCamera.zoom = pose.orthoZoom
        orthoCamera.updateProjectionMatrix()
      }

      // Seed the point budget: a valid cache hit skips the frame-time measurement below
      // entirely (device tier doesn't change between sessions), a miss falls back to the
      // static device-signal prior and lets the render loop's tick correct it against
      // reality - see pointBudget.ts's module doc comment. Runs post-init so
      // deviceProbe's renderer.getContext() call (WebGL-only machinery, defensively
      // guarded, but never even attempted pre-init) has a fully-configured backend to
      // read from.
      const bufSize = renderer.getDrawingBufferSize(new THREE.Vector2())
      const drawingBufferPixels = bufSize.x * bufSize.y
      const cachedBudget = loadPointBudget(drawingBufferPixels)
      const seededBudget = cachedBudget ?? initialPointBudget(probeDevice(renderer))
      pointBudgetStateRef.current =
        cachedBudget !== null ? { ...createFrameBudget(cachedBudget), settled: true } : createFrameBudget(seededBudget)
      basePointBudgetRef.current = seededBudget
      setPointBudget(seededBudget)

      requestRenderOnce()
    })()

    const keysHeld = keysHeldRef.current // stable Set instance for this ref's lifetime

    return () => {
      cancelled = true
      if (stopTimerRef.current) clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
      keysHeld.clear()
      tweenRef.current = null
      stopLoop()
      if (resizeRafRef.current !== null) {
        cancelAnimationFrame(resizeRafRef.current)
        resizeRafRef.current = null
      }
      cancelYawSettle()
      ro.disconnect()
      canvas.removeEventListener('pointerdown', handlePointerDown)
      canvas.removeEventListener('pointerup', handlePointerUp)
      canvas.removeEventListener('pointercancel', handlePointerCancel)
      canvas.removeEventListener('dblclick', handleDoubleClick)
      canvas.removeEventListener('pointermove', handlePointerMove)
      container.removeEventListener('wheel', handleContainerWheel, { capture: true })
      container.removeEventListener('contextmenu', handleContainerContextMenu)
      container.removeEventListener('gesturestart', handleGestureStart)
      container.removeEventListener('gesturechange', handleGestureChange)
      container.removeEventListener('gestureend', handleGestureEnd)
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleWindowBlur)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      perspControls.removeEventListener('start', handleStart)
      perspControls.removeEventListener('end', handleEnd)
      perspControls.removeEventListener('change', handleChange)
      perspControls.dispose()
      orthoControls.removeEventListener('start', handleStart)
      orthoControls.removeEventListener('end', handleEnd)
      orthoControls.removeEventListener('change', handleChange)
      orthoControls.dispose()
      scene.remove(keyLight)
      keyLight.dispose()
      scene.remove(ambient)
      ambient.dispose()
      edlRef.current?.dispose()
      edlRef.current = null
      // Dispose after the current frame, not inside this teardown. `stopLoop()` above
      // cancels the NEXT frame, but a frame already submitted this tick is still in the
      // queue, and destroying its textures out from under it is the same validation error
      // the ResizeObserver hit. Everything else is torn down synchronously; only the
      // GPU-side teardown waits, and it holds its own `renderer` reference so it does not
      // depend on any of the state cleared below.
      // Captured before anything is cleared, so a rebuild (the WebGPU -> WebGL2 guard)
      // comes back looking at what the user was looking at. Harmless on a real unmount:
      // the refs die with the component.
      poseSnapshotRef.current = {
        persp: perspCamera.position.toArray() as [number, number, number],
        target: perspControls.target.toArray() as [number, number, number],
        orthoZoom: orthoCamera.zoom,
      }
      three.current = null
      requestAnimationFrame(() => renderer.dispose())
      if (import.meta.env.DEV) {
        delete (window as unknown as { __cloudeyeThree?: unknown }).__cloudeyeThree
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererEpoch, requestRenderOnce, startLoop, stopLoop, cancelYawSettle, holdLoop, releaseLoop, startFrameTween])

  // Point cloud: (re)built whenever a new GLB finishes loading, or the point budget
  // changes (device-tier seed on mount, or a measured downshift - see pointBudget.ts).
  // Never holds the un-decimated arrays past this effect.
  useEffect(() => {
    const t = three.current
    // showPoints gates the BUILD, not the sprite's visibility: switching the layer off
    // runs this effect's cleanup, which removes the sprite from the scene graph and
    // disposes its buffers. `.visible = false` (what the viewMode gate below still does,
    // for the 2D/3D swap) leaves a million-instance draw call in the graph.
    if (!t || !glb.gltf || !showPoints) return

    let built = buildDecimatedPointCloud(glb.gltf.scene, pointBudget)
    if (!built) return

    // Clip to the measured room outline (+ROOM_POLYGON_CLOUD_MARGIN_M) whenever this
    // scene has MSA yaw metadata - see the constant's doc comment above. room_polygon
    // is always in bootstrap's yaw-rotated frame; roomPolygonToCloudFrame brings it
    // into the cloud's own (unrotated) frame with the same transform the MSA overlay
    // group itself uses, so the two stay visually consistent.
    const roomPolygonCloudFrame = roomPolygonToCloudFrame(msaMeta?.room_polygon, msaOverlayTransform(msaMeta))
    if (roomPolygonCloudFrame) {
      built = clipPointCloudToPolygonXZ(built, (x, z) =>
        isNearRoomPolygonXZ(x, z, roomPolygonCloudFrame, ROOM_POLYGON_CLOUD_MARGIN_M),
      )
    }

    const posAttr = built.geometry.getAttribute('position') as THREE.BufferAttribute
    const colorAttr = (built.geometry.getAttribute('color') as THREE.BufferAttribute | undefined) ?? null
    pointPositionsRef.current = posAttr
    const materials = createPointCloudMaterials(posAttr, colorAttr)
    const pixelRatio = t.renderer.getPixelRatio()
    // uSize is in WORLD METRES - the raw slider value, no pre-multiplication. DPR and
    // the [1, 32] device-px clamp are both applied inside scene3dPoints.ts's sizeNode now
    // (PointsNodeMaterial applies screenDPR to sizeNode itself, unconditionally -
    // multiplying by pixelRatio here too double-counted it; see docs/DECISIONS.md).
    // `pixelRatio` stays declared - still needed below for the EDL radius, a genuinely
    // device-pixel quantity that edlRadiusPx computes from the metres value itself.
    materials.uniforms.uSize.value = pointSize
    materials.uniforms.uCeilingMode.value = ceilingUniformMode(ceilingMode, ceilingY)
    materials.uniforms.uCeilingY.value = effectiveCeilingCutoff
    materials.uniforms.uColorMode.value = colorMode === 'height' ? 1 : 0
    const mesh = createPointCloudSprite(materials.beauty, built.keptCount, built.geometry.boundingBox)
    // Seeded here as well as in the effect below, through refs, so a rebuild that lands
    // while the user is in 2D (a point-budget downshift, a new decimation) never flashes
    // the cloud over the map for a frame.
    mesh.visible = viewModeRef.current === '3d' && showPointCloudRef.current
    t.scene.add(mesh)
    pointsRef.current = mesh
    pointUniformsRef.current = materials.uniforms
    edlRef.current?.setPointMaterial(materials.depth)
    edlRef.current?.setPointSize(pointSize, DEFAULT_POINT_SIZE, pixelRatio)

    let grid: THREE.GridHelper | null = null
    const bb = built.geometry.boundingBox
    if (bb) {
      const box: BoundingBox3 = { min: [bb.min.x, bb.min.y, bb.min.z], max: [bb.max.x, bb.max.y, bb.max.z] }
      bboxRef.current = box
      // Only re-pose the camera (and re-derive clip planes) on an actual scene change -
      // a budget-driven rebuild of the SAME scene must never yank the camera back to the
      // initial framing mid-session. See posedSceneRef's doc comment.
      if (posedSceneRef.current !== sceneId) {
        posedSceneRef.current = sceneId
        const clip = cameraClipPlanes(box)
        t.camera.near = clip.near
        t.camera.far = clip.far
        applyCameraPose(initialCameraPose(box))
      }
      // Height coloring is normalised to the scene's own floor/ceiling, never a
      // hardcoded range - see createPointCloudMaterials's uHeightMin/uHeightMax.
      materials.uniforms.uHeightMin.value = bb.min.y
      materials.uniforms.uHeightMax.value = bb.max.y

      const sizeX = box.max[0] - box.min[0]
      const sizeZ = box.max[2] - box.min[2]
      // Far depth-fade (spec 7, point 3): a fraction of the room's own footprint
      // diagonal, so a large room doesn't fade out points that are still well inside
      // it, and a small room doesn't need to render past its own walls.
      const roomDiagonal = Math.hypot(sizeX, sizeZ)
      setPointCloudFadeRange(materials, roomDiagonal)
      const gridSize = Math.max(2, Math.ceil(Math.max(sizeX, sizeZ)) + 2)
      grid = new THREE.GridHelper(gridSize, gridSize, COLOR_GRID_CENTER, COLOR_GRID)
      grid.position.set((box.min[0] + box.max[0]) / 2, 0, (box.min[2] + box.max[2]) / 2)
      t.scene.add(grid)
      gridRef.current = grid
    }

    requestRenderOnce()

    return () => {
      t.scene.remove(mesh)
      // mesh.geometry is the private per-instance quad createPointCloudSprite built
      // (never the shared Sprite-singleton default) - disposing it is what actually
      // frees the instanced position/color GPU buffers, since node-derived attributes
      // are tracked against whatever geometry/RenderObject actually got rendered, not
      // against built.geometry (which was only ever read from, never itself rendered).
      // See scene3dPoints.ts's createPointSpriteGeometry doc comment.
      mesh.geometry.dispose()
      built.geometry.dispose()
      materials.beauty.dispose()
      materials.depth.dispose()
      edlRef.current?.setPointMaterial(null)
      if (pointsRef.current === mesh) {
        pointsRef.current = null
        pointUniformsRef.current = null
        pointPositionsRef.current = null
      }
      if (grid) {
        t.scene.remove(grid)
        grid.dispose()
        if (gridRef.current === grid) gridRef.current = null
      }
    }
    // msaMeta is a genuine dependency now (the room-polygon clip above reads it) - it
    // resolves from its own async fetch effect, sometimes after this one's first run,
    // so a rebuild must fire once it lands rather than leaving an unclipped cloud
    // until the next unrelated rebuild (a budget change).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererGeneration, glb.gltf, pointBudget, msaMeta, showPoints])

  // The Mesh layer's own group. Added when the toggle is on, REMOVED when it is off -
  // the acceptance for this row of toggles is that switching one off leaves nothing of
  // it behind in the scene graph, which `.visible = false` does not satisfy.
  //
  // Nothing is disposed on the way out, deliberately: `useSceneGlb` hands back a GLTF
  // parsed once and cached across remounts (lib/glbCache.ts), so disposing its geometry
  // here would leave the next switch-on rendering freed buffers. The MSA group above
  // does dispose, and gets away with it only because it is rebuilt whenever its gltf
  // identity changes; this one is switched on and off against one cached parse.
  useEffect(() => {
    const t = three.current
    if (!t || !showMesh || !meshLayerGlb.gltf) return

    const detach = attachLayerGroup(t.scene, meshLayerGlb.gltf.scene, MESH_LAYER_NAME)
    requestRenderOnce()

    return () => {
      detach()
      requestRenderOnce()
    }
  }, [rendererGeneration, showMesh, meshLayerGlb.gltf, requestRenderOnce])

  // MSA (Measured Scene Assembly) layers: added into THIS SAME scene graph as the
  // point cloud above, not a separate viewer/camera (spec - see msaGlbUrl's doc
  // comment on Props). Classic-`three` GLTFLoader output (glbCache.ts) added directly
  // into the `three/webgpu`-backed scene: cross-package Object3D/Mesh instances, but
  // this codebase already does exactly this for the point-cloud GLB above
  // (buildDecimatedPointCloud(glb.gltf.scene, ...) takes a classic-three Object3D where
  // scene3dPoints.ts's own types say `three/webgpu`) - both packages ship the same
  // duck-typed (`.isObject3D`/`.isMesh`/`.isMeshStandardMaterial`) shape at the same
  // three.js revision, and WebGPURenderer's node system auto-converts recognized
  // classic materials (MeshStandardMaterial included) the same way it does for the
  // rest of the scene. Geometry/materials are disposed on cleanup, same as the point
  // cloud's own build effect.
  useEffect(() => {
    const t = three.current
    // msaMeshVisible is the Layout toggle. It gates the GROUP, not the nodes inside it:
    // this effect used to run unconditionally and a separate effect flipped `.visible`
    // on the nodes `classifyNode` recognised, which is why switching it off left
    // geometry on screen - anything the classifier did not match was never touched.
    if (!t || !msaGlb.gltf || !msaMeshVisible) return

    const root = msaGlb.gltf.scene
    // morning-2: one group around the GLB root (not the root itself - it's the cached
    // GLTFLoader scene, shared across remounts via glbCache, so its own transform is
    // left pristine) carrying the inverse of bootstrap's yaw normalization, applied by
    // the MSA overlay effect below from `msaMeta`. The cloud (pointsRef) is NOT under it.
    const group = new THREE.Group()
    group.name = 'msa-overlay'
    group.add(root)
    applyMsaOverlayTransform(group, msaOverlayTransform(msaMetaRef.current))
    t.scene.add(group)
    msaGroupRef.current = group

    const layerObjects: Record<MsaMeshLayerKey, THREE.Object3D[]> = { mesh: [], collision: [], plan: [] }
    root.traverse((obj) => {
      const layer = classifyNode(obj.name)
      if (layer === 'mesh' || layer === 'collision' || layer === 'plan') layerObjects[layer].push(obj)
    })
    msaLayerObjectsRef.current = layerObjects
    for (const key of MSA_MESH_LAYER_KEYS) {
      const visible = msaLayerVisibleRef.current[key]
      for (const obj of layerObjects[key]) obj.visible = visible
    }
    requestRenderOnce()

    return () => {
      t.scene.remove(group)
      group.remove(root)
      if (msaGroupRef.current === group) msaGroupRef.current = null
      root.traverse((obj) => {
        const mesh = obj as unknown as { isMesh?: boolean; geometry?: THREE.BufferGeometry; material?: THREE.Material | THREE.Material[] }
        if (!mesh.isMesh) return
        mesh.geometry?.dispose()
        const mat = mesh.material
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
        else mat?.dispose()
      })
      if (msaLayerObjectsRef.current === layerObjects) {
        msaLayerObjectsRef.current = { mesh: [], collision: [], plan: [] }
      }
    }
  }, [rendererGeneration, msaGlb.gltf, msaMeshVisible, requestRenderOnce])

  // MSA overlay transform (morning-2): the inverse of bootstrap's yaw normalization on
  // the group above, so the yaw-rotated GLB lands on the UNROTATED point cloud. Kept as
  // its own effect (not a dep of the layer effect) so a meta arriving after the GLB -
  // or vice versa - just re-applies the transform without re-adding/disposing the GLB.
  // In three.js terms: rotation.y = +yaw_correction_rad about (pivotX, 0, pivotZ) -
  // see lib/msaLayers.msaOverlayTransform for why the sign is + and not -.
  const msaMetaRef = useRef<MsaSceneMeta | null>(null)
  msaMetaRef.current = msaMeta
  useEffect(() => {
    const group = msaGroupRef.current
    if (!group) return
    applyMsaOverlayTransform(group, msaOverlayTransform(msaMeta))
    requestRenderOnce()
  }, [msaMeta, requestRenderOnce])

  // Floor + contours (spec renderer 4-5) - the 2D visual language, now real scene
  // geometry instead of OccupancyMap's separate Canvas2D/DOM/SVG renderer. Independent
  // of the point cloud/GLB entirely (unlike the effect above, this only needs `grid`,
  // which loads fast and doesn't require ever switching into 3D) - geometry rebuilds
  // only on a genuine scene change; the color mapping (radius/mode/reachability) is a
  // separate effect below that repaints the SAME texture in place.
  useEffect(() => {
    const t = three.current
    // In 2D the floor plane IS the map - the toggle is a 3D control and cannot take the
    // 2D view's only content away. In 3D, off means removed: the cleanup below runs and
    // the plane, its texture and the contour lines all leave the graph.
    if (!t || !grid || !(showMap || viewMode === '2d')) return

    const geometry = buildFloorGeometry(grid)
    const clearanceField = computeClearanceField(grid.cells, grid.width, grid.height, grid.resolution)
    clearanceFieldRef.current = clearanceField
    const pixels = computeFloorPixels(grid, clearanceField, colorMode, robotRadius ?? DEFAULT_RADIUS_M, reachableCells, layers)
    const texture = buildFloorTexture(pixels, grid.width * FLOOR_SUPERSAMPLE, grid.height * FLOOR_SUPERSAMPLE)
    const material = new THREE.MeshBasicMaterial({ map: texture })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.position.y = FLOOR_Y
    t.scene.add(mesh)
    floorMeshRef.current = mesh
    floorTextureRef.current = texture

    // Vector obstacle contours - the confirmed-obstacle mask specifically (value === 1),
    // distinct from the clearance field's conservative "unknown counts as obstacle too"
    // treatment above.
    const mask = grid.cells.map((column) => column.map((v): 0 | 1 => (v === 1 ? 1 : 0)))
    const polylines: Point[][] = computeContours(mask, grid.width, grid.height, {
      simplifyTolerance: CONTOUR_SIMPLIFY_TOLERANCE,
    })
    const contourLayer = buildContours(polylines, grid, {
      yOffset: CONTOUR_Y,
      color: COLOR_CONTOUR,
      lineWidthPx: CONTOUR_LINE_WIDTH_PX,
      fillOpacity: CONTOUR_FILL_OPACITY,
    })
    // No resolution seeding: Line2NodeMaterial reads the renderer's own viewport in
    // LineSegments2.onBeforeRender - see scene3dContours.ts and the ResizeObserver above.
    t.scene.add(contourLayer.group)
    contoursRef.current = contourLayer

    requestRenderOnce()

    return () => {
      t.scene.remove(mesh)
      geometry.dispose()
      material.dispose()
      texture.dispose()
      if (floorMeshRef.current === mesh) floorMeshRef.current = null
      if (floorTextureRef.current === texture) floorTextureRef.current = null
      clearanceFieldRef.current = null
      t.scene.remove(contourLayer.group)
      disposeContours(contourLayer)
      if (contoursRef.current === contourLayer) contoursRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererGeneration, grid, showMap, viewMode])

  // Floor recolor - mutates the existing texture rather than rebuilding geometry, same
  // rule as the point cloud's uniforms/marker colors. Radius comes from the robot-size
  // slider/platform selection; render mode from the shared 2D/3D selector.
  useEffect(() => {
    const texture = floorTextureRef.current
    const field = clearanceFieldRef.current
    if (!texture || !field || !grid) return
    const pixels = computeFloorPixels(grid, field, colorMode, robotRadius ?? DEFAULT_RADIUS_M, reachableCells, layers)
    updateFloorTexture(texture, pixels, grid.width * FLOOR_SUPERSAMPLE, grid.height * FLOOR_SUPERSAMPLE)
    requestRenderOnce()
  }, [grid, colorMode, robotRadius, reachableCells, layers, requestRenderOnce])

  // Project preview thumbnail (spec 11) - fire-and-forget, does its own independent
  // decimation/render into a throwaway offscreen canvas, so it can't race the point
  // cloud effect above disposing its geometry. No-ops if a preview is already cached.
  useEffect(() => {
    if (!glb.gltf) return
    void captureScenePreview(sceneId, glb.gltf.scene)
  }, [glb.gltf, sceneId])

  // Point-size slider - mutates the existing material's uniform rather than rebuilding
  // geometry. uSize is world metres - see the build effect's comment; DPR/clamp live
  // entirely in scene3dPoints.ts's sizeNode now.
  useEffect(() => {
    const t = three.current
    const uniforms = pointUniformsRef.current
    if (uniforms && t) {
      const pixelRatio = t.renderer.getPixelRatio()
      uniforms.uSize.value = pointSize
      // Keeps the EDL neighbor-tap radius in sync with the on-screen splat size - see
      // edlSettings.edlRadiusPx's doc comment for why this must track point size.
      edlRef.current?.setPointSize(pointSize, DEFAULT_POINT_SIZE, pixelRatio)
      requestRenderOnce()
    }
  }, [rendererGeneration, pointSize, requestRenderOnce])

  // EDL toggle - no uniform to mutate, requestRenderOnce reads edlEnabledRef directly.
  useEffect(() => {
    requestRenderOnce()
  }, [edlEnabled, requestRenderOnce])

  // Ceiling select + height-cutoff slider - same rule as pointSize above: mutate the
  // live material's uniforms, never rebuild geometry. Re-runs when `ceilingY` becomes
  // known after the point cloud was already built, same as the old height-cutoff
  // slider did.
  useEffect(() => {
    const uniforms = pointUniformsRef.current
    if (!uniforms) return
    uniforms.uCeilingMode.value = ceilingUniformMode(ceilingMode, ceilingY)
    uniforms.uCeilingY.value = effectiveCeilingCutoff
    requestRenderOnce()
  }, [ceilingMode, ceilingY, effectiveCeilingCutoff, requestRenderOnce])

  // Colour-mode switch (photo <-> height) - same rule as the effects above: mutate the
  // live material's uniform, never rebuild geometry.
  useEffect(() => {
    const uniforms = pointUniformsRef.current
    if (!uniforms) return
    uniforms.uColorMode.value = colorMode === 'height' ? 1 : 0
    requestRenderOnce()
  }, [colorMode, requestRenderOnce])

  // Object markers - rebuilt when the object list changes (in practice: once per scene
  // load, since `objects` is a stable reference after that).
  //
  // `rendererGeneration` is load-bearing, and its absence was a real defect: `objects` is
  // already its final value when SceneMap3D mounts and never changes again, so this
  // effect ran exactly once, against a `three.current` that `await renderer.init()` had
  // not assigned yet, and returned - and no object marker was drawn on either view for
  // the rest of the session. Measured on the hero scene through window.__cloudeyeThree:
  // the scene graph held no 'object-markers' group at all, so 17 objects rendered as 0
  // markers, nothing could be hovered or double-clicked on the map, and the object list
  // was the only place an object existed. Exactly the failure mode `rendererGeneration`'s own
  // comment describes ("permanent for those whose [deps] don't [change]").
  useEffect(() => {
    const t = three.current
    if (!t) return
    const group = new THREE.Group()
    // Named for demo/ probes: the dev handle (window.__cloudeyeThree) is the only way
    // to ask "is this layer actually drawn", and an unnamed Group is indistinguishable
    // from any other in a traversal.
    group.name = 'object-markers'
    const sphereGeom = new THREE.SphereGeometry(0.06, 12, 8)
    const map = new Map<string, THREE.Mesh>()
    for (const o of objects) {
      if (o.is_fragment) continue
      const material = new THREE.MeshBasicMaterial({ color: COLOR_DEFAULT })
      const mesh = new THREE.Mesh(sphereGeom, material)
      mesh.position.set(o.pos_x, o.pos_y, o.pos_z)
      mesh.userData.objectId = o.id
      group.add(mesh)
      map.set(o.id, mesh)
    }
    t.scene.add(group)
    markerMeshesRef.current = map
    requestRenderOnce()

    return () => {
      t.scene.remove(group)
      sphereGeom.dispose()
      for (const mesh of map.values()) (mesh.material as THREE.Material).dispose()
      markerMeshesRef.current = new Map()
    }
  }, [objects, rendererGeneration, requestRenderOnce])

  // Marker color/scale, the hovered-or-selected label, and the selected object's bbox
  // helper - all cheap enough to just recompute in full on any of these changes.
  useEffect(() => {
    const t = three.current
    if (!t) return

    for (const [id, mesh] of markerMeshesRef.current) {
      const isActive = id === activeObjectId
      const isSelected = id === selectedObjectId
      const isHovered = id === hoveredObjectId
      const isUnreachable = reachableObjectIds ? !reachableObjectIds.has(id) : false
      // Spec 9: a marker's own color never changes for hover/selection - only --data-object
      // (or --accent/--faint for active/unreachable, which are real state, not focus).
      // Selection is conveyed by the scale bump below plus the bbox outline instead.
      const color = isActive ? COLOR_ACTIVE : isUnreachable ? COLOR_UNREACHABLE : COLOR_DEFAULT
      ;(mesh.material as THREE.MeshBasicMaterial).color.setHex(color)
      mesh.scale.setScalar(isSelected || isHovered ? 1.5 : 1)
    }

    const labelId = hoveredObjectId ?? selectedObjectId
    labelObjectRef.current = labelId ? (objects.find((o) => o.id === labelId) ?? null) : null

    if (hullMeshRef.current) {
      t.scene.remove(hullMeshRef.current)
      hullMeshRef.current.geometry.dispose()
      ;(hullMeshRef.current.material as THREE.Material).dispose()
      hullMeshRef.current = null
    }
    if (hullEdgesRef.current) {
      t.scene.remove(hullEdgesRef.current)
      hullEdgesRef.current.geometry.dispose()
      ;(hullEdgesRef.current.material as THREE.Material).dispose()
      hullEdgesRef.current = null
    }
    const selected = selectedObjectId ? objects.find((o) => o.id === selectedObjectId) : null
    if (selected) {
      const box: BoundingBox3 = {
        min: [selected.bbox_min_x, selected.bbox_min_y, selected.bbox_min_z],
        max: [selected.bbox_max_x, selected.bbox_max_y, selected.bbox_max_z],
      }
      // A real convex-hull highlight (spec change: replaces the plain bbox outline this
      // used to draw) so the highlight actually reads as "this object", not just "the box
      // it happens to sit in" - see buildSelectionHullGeometry's doc comment for the
      // fallback chain when the point cloud can't supply enough of the object's own points.
      const hullGeom = buildSelectionHullGeometry(box, pointPositionsRef.current)
      const hullMesh = new THREE.Mesh(
        hullGeom,
        new THREE.MeshBasicMaterial({
          color: COLOR_BBOX,
          transparent: true,
          opacity: 0.18,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      )
      t.scene.add(hullMesh)
      hullMeshRef.current = hullMesh

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(hullGeom),
        new THREE.LineBasicMaterial({ color: COLOR_BBOX }),
      )
      t.scene.add(edges)
      hullEdgesRef.current = edges
    }

    requestRenderOnce()
    // `glb.gltf` is read (via pointPositionsRef, populated by the build effect above)
    // only to re-run this once the point cloud finishes loading - selecting an object
    // before then falls back to its bbox corners, and would otherwise never upgrade to
    // its real hull once the cloud arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // `rendererGeneration` for the same reason the marker-building effect above takes it:
    // this styles the meshes that effect creates, so it has to run again once they
    // exist. Without it, whether a marker was ever greyed depended on the order the
    // renderer's init and the reachability response happened to finish in.
  }, [objects, selectedObjectId, hoveredObjectId, activeObjectId, reachableObjectIds, rendererGeneration, requestRenderOnce, glb.gltf])

  // Build (not position - see updateObjectBoxes) the 2D map's labelled object boxes.
  // Rebuilt only when the objects, the reachable set or the view mode change; an
  // unreachable object's box and label are drawn in the muted/unreachable colours, the
  // same distinction the markers and the object list already make.
  useEffect(() => {
    const host = objectBoxesRef.current
    if (!host) return
    host.replaceChildren()
    const entries: ObjectBoxEntry[] = []
    if (viewMode === '2d') {
      for (const object of objects) {
        if (object.is_fragment) continue
        const unreachable = reachableObjectIds !== null && !reachableObjectIds.has(object.id)
        const el = document.createElement('div')
        el.className = `absolute left-0 top-0 rounded-[1px] border ${
          unreachable ? 'border-faint/60' : 'border-data-object/70'
        }`
        el.dataset.objectBox = object.id
        if (unreachable) el.dataset.objectBoxUnreachable = 'true'
        const label = document.createElement('span')
        label.textContent = object.name
        // No `-top-3.5`: updateObjectBoxes owns the vertical offset now, so it can move a
        // label out of another one's way.
        label.className = `pointer-events-none absolute left-0 top-0 whitespace-nowrap font-mono text-[9px] ${
          unreachable ? 'text-faint' : 'text-data-object'
        }`
        el.appendChild(label)
        host.appendChild(el)
        entries.push({ el, object, label, labelW: 0 })
      }
    }
    objectBoxEntriesRef.current = entries
    updateObjectBoxes()
    requestRenderOnce()
    return () => {
      objectBoxEntriesRef.current = []
      host.replaceChildren()
    }
  }, [objects, reachableObjectIds, viewMode, updateObjectBoxes, requestRenderOnce])

  // The MSA mesh layer follows the toolbar's switch. First time it goes on, this is
  // also what starts the (large) export download - the same "asking for it is asking for
  // it" rule the Shift chords use for the layers they still own.
  useEffect(() => {
    if (msaMeshVisible && !msaGlbRequestedRef.current) {
      msaGlbRequestedRef.current = true
      setMsaGlbRequested(true)
    }
    msaLayerVisibleRef.current.mesh = msaMeshVisible
    for (const obj of msaLayerObjectsRef.current.mesh) obj.visible = msaMeshVisible
    requestRenderOnce()
  }, [msaMeshVisible, msaGlb.gltf, requestRenderOnce])

  // Objects-list "focus" request (a click on a row, not a marker - see ObjectList's
  // onFocusObject): frames the 3D camera on that object, reusing the exact same
  // frameObjectPose + startFrameTween tween as the shift+double-click gesture above -
  // just triggered from the list instead of the canvas. Keyed on `focusNonce` alone (not
  // `focusObjectId`) so re-clicking the SAME already-selected object still re-triggers
  // the tween - the id wouldn't change in that case, but the nonce always does. 3D-only,
  // same reasoning as shift+double-click: the orthographic camera frames via zoom/pan,
  // not position distance, so tweening its position wouldn't reframe anything in 2D.
  useEffect(() => {
    if (!focusNonce || viewModeRef.current !== '3d') return
    const obj = objects.find((o) => o.id === focusObjectId)
    if (!obj) return
    const box: BoundingBox3 = {
      min: [obj.bbox_min_x, obj.bbox_min_y, obj.bbox_min_z],
      max: [obj.bbox_max_x, obj.bbox_max_y, obj.bbox_max_z],
    }
    const cam = three.current?.camera.position
    const from: [number, number, number] = cam ? [cam.x, cam.y, cam.z] : [0, 0, 0]
    startFrameTween(frameObjectPose(box, from))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusNonce])

  // Footprint indicator (rectangle fill + rectangle outline "ring" + pin) - only
  // rebuilt when the footprint size changes (the robot-size slider, or picking a
  // platform), which is rare; position updates happen far more often (every animation
  // frame during a move) in a later effect and shouldn't pay for a geometry rebuild.
  // Deliberately NEVER reacts to the robot model or its load state - this is the
  // planner's own footprint, shown honestly even when it disagrees with the physical
  // model sitting on top of it (see the model effect below for why that disagreement
  // is kept, not hidden).
  //
  // Drawn as a rectangle (length_m x width_m, world +x/+z respectively - this
  // component has no robot-heading data, so it's always axis-aligned, never rotated
  // to match the model's actual facing) instead of a circle: `robotLengthM`/
  // `robotWidthM` (the selected platform's real dimensions_m) when known, else a
  // radius-derived square, same conservative fallback `app.robots.resolve_footprint_m`
  // uses on the backend - so the indicator always shows SOME usable footprint, not just
  // for platforms with measured dimensions.
  //
  // Rebuilding geometry per slider tick (rather than building once at unit size and
  // scaling the group) is deliberate: scaling would also scale the outline's
  // ROBOT_RING_WIDTH stroke, so the indicator's line weight would visibly change while
  // dragging. A handful of triangles is cheaper than the cylinder this originally
  // replaced anyway.
  useEffect(() => {
    const t = three.current
    if (!t) return
    const radius = robotRadius ?? ROBOT_FALLBACK_RADIUS
    const lengthM = robotLengthM ?? 2 * radius
    const widthM = robotWidthM ?? 2 * radius
    const halfL = lengthM / 2
    const halfW = widthM / 2

    const group = new THREE.Group()

    const outerShape = new THREE.Shape()
    outerShape.moveTo(-halfL, -halfW)
    outerShape.lineTo(halfL, -halfW)
    outerShape.lineTo(halfL, halfW)
    outerShape.lineTo(-halfL, halfW)
    outerShape.closePath()

    const fillGeom = new THREE.ShapeGeometry(outerShape)
    const fillMat = new THREE.MeshBasicMaterial({
      color: COLOR_ROBOT,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const fill = new THREE.Mesh(fillGeom, fillMat)
    fill.rotation.x = -Math.PI / 2
    fill.position.y = ROBOT_DISC_Y
    fill.renderOrder = 1
    group.add(fill)

    const innerHalfL = Math.max(0, halfL - ROBOT_RING_WIDTH)
    const innerHalfW = Math.max(0, halfW - ROBOT_RING_WIDTH)
    const outlineShape = new THREE.Shape()
    outlineShape.moveTo(-halfL, -halfW)
    outlineShape.lineTo(halfL, -halfW)
    outlineShape.lineTo(halfL, halfW)
    outlineShape.lineTo(-halfL, halfW)
    outlineShape.closePath()
    const hole = new THREE.Path()
    hole.moveTo(-innerHalfL, -innerHalfW)
    hole.lineTo(-innerHalfL, innerHalfW)
    hole.lineTo(innerHalfL, innerHalfW)
    hole.lineTo(innerHalfL, -innerHalfW)
    hole.closePath()
    outlineShape.holes.push(hole)

    const outlineGeom = new THREE.ShapeGeometry(outlineShape)
    const outlineMat = new THREE.MeshBasicMaterial({
      color: COLOR_ROBOT,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const outline = new THREE.Mesh(outlineGeom, outlineMat)
    outline.rotation.x = -Math.PI / 2
    outline.position.y = ROBOT_RING_Y
    outline.renderOrder = 2
    group.add(outline)

    const pinGeom = new THREE.CylinderGeometry(ROBOT_PIN_RADIUS, ROBOT_PIN_RADIUS, ROBOT_PIN_HEIGHT, 8)
    const pinMat = new THREE.MeshBasicMaterial({ color: COLOR_ROBOT })
    const pin = new THREE.Mesh(pinGeom, pinMat)
    pin.position.y = ROBOT_PIN_HEIGHT / 2
    group.add(pin)

    const pos = robotPositionRef.current
    group.visible = !!pos
    if (pos) group.position.set(pos.x, 0, pos.z)
    t.scene.add(group)
    robotIndicatorRef.current = group
    requestRenderOnce()

    return () => {
      t.scene.remove(group)
      fillGeom.dispose()
      fillMat.dispose()
      outlineGeom.dispose()
      outlineMat.dispose()
      pinGeom.dispose()
      pinMat.dispose()
      if (robotIndicatorRef.current === group) robotIndicatorRef.current = null
    }
  }, [rendererGeneration, robotRadius, robotLengthM, robotWidthM, requestRenderOnce])

  // The selected platform's start marker: a thin ring on the floor at `startPosition`.
  // Rebuilt only when the position actually changes (rare - a platform switch, or a
  // radius that resolves to a different cell), and present in BOTH view modes, since it
  // is the one thing that says where "reset" goes and where a route was planned from.
  // Deliberately independent of the robot indicator: once the robot drives away, the
  // start is no longer where the robot is, and that is exactly when this matters.
  useEffect(() => {
    const t = three.current
    if (!t || !startPosition) return
    const geometry = new THREE.RingGeometry(
      START_MARKER_RADIUS_M - START_MARKER_WIDTH_M,
      START_MARKER_RADIUS_M,
      24,
    )
    const material = new THREE.MeshBasicMaterial({
      color: COLOR_ROBOT,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const ring = new THREE.Mesh(geometry, material)
    ring.name = 'start-marker'
    // RingGeometry is built in the XY plane, like ShapeGeometry above - lay it flat.
    ring.rotation.x = -Math.PI / 2
    ring.position.set(startPosition.x, START_MARKER_Y, startPosition.z)
    t.scene.add(ring)
    requestRenderOnce()
    return () => {
      t.scene.remove(ring)
      geometry.dispose()
      material.dispose()
    }
  }, [startPosition, rendererGeneration, requestRenderOnce])

  // Robot model load/attach/dispose - reruns whenever the selected platform's mesh URL
  // changes (in addition to once per mount), tearing down the previous platform's model and
  // swapping in the new one via the same cached-GLB path (see lib/robotModelCache.ts).
  // Deliberately excludes robotRadius from its deps: the model is the selected platform's
  // physical mesh, never scaled to match the radius slider (see e.g.
  // scripts/build-turtlebot-glb.mjs and the indicator effect above) - when the slider moves
  // away from that platform's own footprint, the ring above visibly disagrees with the
  // chassis instead of the model silently stretching to match.
  //
  // Also note: each platform's mesh origin is its URDF's base_footprint - on the floor, at
  // the wheel-axle/body center used by that platform's own build script. The chassis is not
  // necessarily centered on it (e.g. TurtleBot3 Burger's is 100.8mm back, 36.8mm forward of
  // it), so the chassis can poke past the ring on one side while sitting inside it on the
  // other. That's the real, physically-correct pivot point per platform - do not shift a
  // model to visually center it in the ring, that would make it pivot around the chassis
  // rather than the real axle/body center and would be wrong the moment anything ever adds a
  // rotate-in-place step.
  useEffect(() => {
    const t = three.current
    if (!t) return
    const modelUrl = robotMeshUrl ?? DEFAULT_MODEL_URL
    let cancelled = false
    let instance: ReturnType<typeof instantiateRobotModel> | null = null
    setRobotModelFailed(false) // clear a previous platform's load failure on (re)try

    const attach = (gltf: GLTF) => {
      if (cancelled) return
      instance = instantiateRobotModel(gltf)
      const pos = robotPositionRef.current // may already be set - this resolves async
      instance.object.visible = !!pos && showRobotModelRef.current
      if (pos) instance.object.position.set(pos.x, 0, pos.z)
      instance.object.rotation.y = robotYawRef.current
      t.scene.add(instance.object)
      robotModelRef.current = instance.object
      requestRenderOnce() // required - nothing else repaints after this async resolve
    }

    const cached = peekRobotModel(modelUrl)
    if (cached) {
      attach(cached)
    } else {
      loadRobotModel(modelUrl)
        .then(attach)
        .catch((err) => {
          console.error(`Failed to load ${modelUrl}`, err)
          if (!cancelled) setRobotModelFailed(true)
        })
    }

    return () => {
      cancelled = true
      if (!instance) return
      t.scene.remove(instance.object)
      // Geometry (and the cached GLTF itself) is shared across every instance and must
      // NEVER be disposed here - that would blank the robot on every subsequent mount or
      // re-select of the same platform. Only the materials instantiateRobotModel() cloned
      // belong to this instance.
      for (const m of instance.materials) m.dispose()
      if (robotModelRef.current === instance.object) robotModelRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererGeneration, requestRenderOnce, robotMeshUrl])

  // Robot model visibility toggle. Kept out of the load effect's deps deliberately - if
  // showRobotModel were a dep there, toggling it would tear down and re-clone the whole
  // model instead of just flipping a flag.
  useEffect(() => {
    const model = robotModelRef.current
    if (!model) return // load hasn't resolved yet; the load effect applies the current flag then
    model.visible = showRobotModel && !!robotPositionRef.current
    requestRenderOnce()
  }, [showRobotModel, requestRenderOnce])

  // Robot position + heading - high frequency during command animation (once per rAF tick,
  // driven by useCommandAnimation's move loop). Heading is derived from consecutive
  // position deltas rather than a real path tangent (see robot-heading.ts) and damped
  // in-place against wall-clock time, riding the same ~60Hz cadence this effect already
  // gets during a move - no separate rAF loop is needed while moving, only the short
  // settle tail below once motion stops.
  useEffect(() => {
    cancelYawSettle() // the move's own cadence drives dt for as long as it's firing
    const indicator = robotIndicatorRef.current
    const model = robotModelRef.current

    if (!robotPosition) {
      if (indicator) indicator.visible = false
      if (model) model.visible = false
      prevRobotPosRef.current = null // next appearance must not be read as a teleport
      lastYawTickRef.current = null
      requestRenderOnce()
      return
    }

    const now = performance.now()
    const dtMs = lastYawTickRef.current === null ? 0 : Math.min(now - lastYawTickRef.current, YAW_DT_CLAMP_MS)
    lastYawTickRef.current = now

    const prev = prevRobotPosRef.current
    if (prev) {
      const yaw = headingYaw(robotPosition.x - prev.x, robotPosition.z - prev.z)
      if (yaw !== null) robotYawTargetRef.current = yaw // null => noise or teleport: hold heading
    }
    prevRobotPosRef.current = { x: robotPosition.x, z: robotPosition.z }

    robotYawRef.current = dampAngle(robotYawRef.current, robotYawTargetRef.current, dtMs, YAW_SMOOTH_MS)

    if (indicator) {
      indicator.visible = true
      indicator.position.set(robotPosition.x, 0, robotPosition.z)
    }
    if (model) {
      model.visible = showRobotModelRef.current
      model.position.set(robotPosition.x, 0, robotPosition.z)
      model.rotation.y = robotYawRef.current
    }
    if (povActiveRef.current) applyPovPose()
    requestRenderOnce()

    if (Math.abs(shortestAngleDelta(robotYawRef.current, robotYawTargetRef.current)) > YAW_SETTLE_EPS) {
      startYawSettle()
    }
  }, [robotPosition, requestRenderOnce, cancelYawSettle, startYawSettle, applyPovPose])

  // Robot path - a flat polyline lifted slightly above the floor so it doesn't z-fight
  // with the point cloud.
  useEffect(() => {
    const t = three.current
    if (!t) return
    if (pathLineRef.current) {
      t.scene.remove(pathLineRef.current)
      pathLineRef.current.geometry.dispose()
      ;(pathLineRef.current.material as THREE.Material).dispose()
      pathLineRef.current = null
    }
    if (currentPath && currentPath.length > 1) {
      const points = currentPath.map(([x, z]) => new THREE.Vector3(x, PATH_Y_OFFSET, z))
      const geometry = new THREE.BufferGeometry().setFromPoints(points)
      const material = new THREE.LineBasicMaterial({ color: COLOR_PATH })
      const line = new THREE.Line(geometry, material)
      line.name = 'current-path'
      t.scene.add(line)
      pathLineRef.current = line
    }
    requestRenderOnce()
  }, [rendererGeneration, currentPath, requestRenderOnce])

  // Movement trail - every point the robot has actually travelled, accumulated across
  // commands (see the `trailPoints` prop doc). Uses Line2/LineMaterial rather than a
  // plain THREE.Line purely for a legible screen-space width (LineBasicMaterial's
  // `linewidth` is a no-op on most GPUs/ANGLE backends) - see COLOR_TRAIL and
  // TRAIL_LINE_WIDTH_PX. Rebuilt whenever the accumulated trail grows; visibility is
  // toggled separately below so flipping "show trail" doesn't pay for a rebuild.
  useEffect(() => {
    const t = three.current
    if (!t) return
    if (trailLineRef.current) {
      t.scene.remove(trailLineRef.current)
      trailLineRef.current.geometry.dispose()
      trailMaterialRef.current?.dispose()
      trailLineRef.current = null
      trailMaterialRef.current = null
    }
    if (trailPoints.length > 1) {
      const positions: number[] = []
      for (const [x, z] of trailPoints) positions.push(x, TRAIL_Y_OFFSET, z)
      const geometry = new LineGeometry()
      geometry.setPositions(positions)
      // Line2NodeMaterial forces blending to NoBlending (no transparency support yet -
      // see three's own source comment on the class) - `transparent`/`opacity` are kept
      // for when that lands upstream, but the trail renders fully opaque under WebGPU
      // for now. No `.resolution` to set - see the ResizeObserver's comment above.
      const material = new Line2NodeMaterial({ color: COLOR_TRAIL, linewidth: TRAIL_LINE_WIDTH_PX, transparent: true })
      const line = new Line2(geometry, material)
      line.computeLineDistances()
      line.visible = showTrailRef.current
      t.scene.add(line)
      trailLineRef.current = line
      trailMaterialRef.current = material
    }
    requestRenderOnce()
  }, [rendererGeneration, trailPoints, requestRenderOnce])

  // "show trail" toggle - kept out of the build effect's deps deliberately, same reason
  // as the robot-model visibility toggle above: flip a flag, don't rebuild geometry.
  useEffect(() => {
    if (trailLineRef.current) trailLineRef.current.visible = showTrail
    requestRenderOnce()
  }, [showTrail, requestRenderOnce])

  // The point cloud is a 3D layer: hidden whenever the 2D map is on screen, regardless
  // of the Shift+1 preference. Before this, switching 2D -> 3D -> 2D left the cloud
  // drawn on top of the occupancy map (the ortho camera looks straight down at it), so
  // the map was read through a layer of points. Same "flip a flag, don't rebuild"
  // discipline as the trail and robot-model toggles above - the geometry, the materials
  // and glbCache's entry all stay exactly as they were, so coming back to 3D is instant.
  useEffect(() => {
    if (pointsRef.current) pointsRef.current.visible = viewMode === '3d' && showPointCloud
    requestRenderOnce()
  }, [viewMode, showPointCloud, requestRenderOnce])

  // Camera path: a polyline through the aligned keyframe positions (capture order),
  // plus a small marker sphere at each one - see the `cameraTrack` prop doc. `null`
  // (the "camera path" toggle off, or not yet loaded) tears everything down. No
  // per-pose view-direction frustums: cameras_aligned.json only ever carries positions,
  // never orientation (confirmed against real scene data, not assumed), so drawing one
  // would be fabricating data this display-only feature is meant to show, not invent.
  useEffect(() => {
    const t = three.current
    if (!t) return

    if (cameraPathLineRef.current) {
      t.scene.remove(cameraPathLineRef.current)
      cameraPathLineRef.current.geometry.dispose()
      cameraPathMaterialRef.current?.dispose()
      cameraPathLineRef.current = null
      cameraPathMaterialRef.current = null
    }
    if (cameraPoseMeshesRef.current.length) {
      for (const mesh of cameraPoseMeshesRef.current) {
        t.scene.remove(mesh)
        mesh.geometry.dispose()
        ;(mesh.material as THREE.Material).dispose()
      }
      cameraPoseMeshesRef.current = []
    }

    if (cameraTrack && cameraTrack.length > 1) {
      const positions: number[] = []
      for (const p of cameraTrack) positions.push(p.x, p.y + CAMERA_PATH_Y_OFFSET, p.z)
      const geometry = new LineGeometry()
      geometry.setPositions(positions)
      // See the trail line's comment above: Line2NodeMaterial has no transparency
      // support yet (opacity 0.85 has no visual effect under WebGPU today) and no
      // `.resolution` to set.
      const material = new Line2NodeMaterial({
        color: COLOR_CAMERA_PATH,
        linewidth: CAMERA_PATH_LINE_WIDTH_PX,
        transparent: true,
        opacity: 0.85,
      })
      const line = new Line2(geometry, material)
      line.computeLineDistances()
      t.scene.add(line)
      cameraPathLineRef.current = line
      cameraPathMaterialRef.current = material

      const sphereGeom = new THREE.SphereGeometry(CAMERA_POSE_RADIUS, 10, 8)
      const meshes = cameraTrack.map((p, i) => {
        const isHighlighted = i === highlightedPoseIndex
        const material = new THREE.MeshBasicMaterial({
          color: isHighlighted ? COLOR_ACTIVE : COLOR_CAMERA_PATH,
        })
        const mesh = new THREE.Mesh(sphereGeom, material)
        mesh.position.set(p.x, p.y + CAMERA_PATH_Y_OFFSET, p.z)
        if (isHighlighted) mesh.scale.setScalar(CAMERA_POSE_HIGHLIGHT_RADIUS / CAMERA_POSE_RADIUS)
        t.scene.add(mesh)
        return mesh
      })
      cameraPoseMeshesRef.current = meshes
    }

    requestRenderOnce()
    // highlightedPoseIndex deliberately excluded - only read here to seed the initial
    // build's colors/scale; the effect below handles it changing afterwards without a
    // full rebuild.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererGeneration, cameraTrack, requestRenderOnce])

  // Camera-path highlight - recolors/rescales just the one pose mesh nearest the video
  // player's current time, without rebuilding the path (same "mutate in place" rule as
  // the object-marker color/scale effect above).
  useEffect(() => {
    for (let i = 0; i < cameraPoseMeshesRef.current.length; i++) {
      const mesh = cameraPoseMeshesRef.current[i]
      const isHighlighted = i === highlightedPoseIndex
      ;(mesh.material as THREE.MeshBasicMaterial).color.setHex(isHighlighted ? COLOR_ACTIVE : COLOR_CAMERA_PATH)
      mesh.scale.setScalar(isHighlighted ? CAMERA_POSE_HIGHLIGHT_RADIUS / CAMERA_POSE_RADIUS : 1)
    }
    requestRenderOnce()
  }, [highlightedPoseIndex, requestRenderOnce])

  // One scene, two cameras (redesign plan section 6): swaps which camera/controls pair
  // is "active" (see the `three.current` shape's own doc comment) whenever the 2D/3D
  // toggle changes. Also covers robot-POV mode, which locks the (perspective-only)
  // camera to the robot and makes manual orbiting meaningless - OrbitControls is
  // disabled for as long as POV is active, same as before this merge.
  useEffect(() => {
    const t = three.current
    if (!t) return
    if (viewMode === '2d') {
      t.camera = t.orthoCamera
      t.controls = t.orthoControls
      t.perspControls.enabled = false
      t.orthoControls.enabled = true
    } else {
      t.camera = t.perspCamera
      t.controls = t.perspControls
      t.perspControls.enabled = !povActive
      t.orthoControls.enabled = false
    }
    requestRenderOnce()
    // `rendererGeneration` is in the deps so this runs once `three.current` actually exists:
    // on a fresh scene open neither `viewMode` nor `povActive` changes again, so without
    // it the pair chosen at construction (perspective) stayed active and 2D rendered
    // tilted, through the wrong camera, until the user toggled 3D and back.
  }, [viewMode, povActive, rendererGeneration, requestRenderOnce])

  function handlePreset(kind: 'top' | 'front' | 'iso' | 'reset') {
    if (povActive) setPovActive(false) // a preset click means "give me the free camera back"
    if (kind !== 'reset') setViewPreset(kind)
    const box = bboxRef.current
    if (!box) return
    const pose =
      kind === 'top'
        ? topCameraPose(box)
        : kind === 'front'
          ? frontCameraPose(box)
          : kind === 'iso'
            ? isometricCameraPose(box)
            : initialCameraPose(box)
    applyCameraPose(pose)
  }

  // Toggles robot-POV: saves the current free-camera pose so turning it back off can
  // restore it, rather than leaving the user wherever the robot last stood.
  function toggleRobotPov() {
    const t = three.current
    if (povActive) {
      setPovActive(false)
      if (prePovPoseRef.current) applyCameraPose(prePovPoseRef.current)
    } else if (t) {
      prePovPoseRef.current = {
        position: [t.camera.position.x, t.camera.position.y, t.camera.position.z],
        target: [t.controls.target.x, t.controls.target.y, t.controls.target.z],
      }
      setPovActive(true)
      applyPovPose()
    }
  }

  const effectiveRadius = robotRadius ?? DEFAULT_RADIUS_M

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={containerRef} className="relative min-h-0 flex-1 w-full overscroll-none bg-canvas">
        <canvas key={rendererEpoch} ref={canvasRef} className="absolute inset-0 block h-full w-full touch-none" />
        {/* Labelled object boxes (2D only) - see updateObjectBoxes. pointer-events-none
           throughout: the boxes are an annotation over the map, never something that
           swallows a click meant for a marker underneath them. */}
        <div ref={objectBoxesRef} className="pointer-events-none absolute inset-0 overflow-hidden" data-testid="object-boxes" />
        <div
          ref={labelRef}
          style={{ display: 'none' }}
          className="pointer-events-none absolute left-0 top-0 whitespace-nowrap rounded-sm border-hair border-border bg-surface-raised px-[7px] py-[3px] text-xs text-fg shadow-lg"
        />

        {viewMode === '3d' ? (
          <div className="absolute left-2 top-2 z-10 flex items-center gap-1 rounded-md border-hair border-border bg-surface/85 p-1 backdrop-blur-md">
            <SegmentedControl
              value={viewPreset}
              onChange={(v) => handlePreset(v)}
              aria-label="Camera preset"
              bare
              segments={[
                { value: 'top', label: 'top' },
                { value: 'front', label: 'front' },
                { value: 'iso', label: 'isometric' },
              ]}
            />
            <Tooltip content="Reset view">
              <Button
                variant="ghost"
                size="compact"
                aria-label="Reset view"
                onClick={() => handlePreset('reset')}
                icon={<IconRotate360 size={16} />}
              />
            </Tooltip>
            <Tooltip content={robotPosition ? 'Toggle first-person view from the robot' : 'Robot has no position yet'}>
              <Button
                variant={povActive ? 'secondary' : 'ghost'}
                size="compact"
                aria-label="Robot view"
                onClick={toggleRobotPov}
                disabled={!robotPosition}
                icon={<IconEye size={16} />}
              />
            </Tooltip>
            <Tooltip content="Navigation controls">
              <Button
                variant={showNavHelp ? 'secondary' : 'ghost'}
                size="compact"
                aria-label="Navigation controls"
                onClick={() => setShowNavHelp((v) => !v)}
              >
                ?
              </Button>
            </Tooltip>
            {showNavHelp && <NavHelpPanel />}
          </div>
        ) : (
          <div className="absolute left-2 top-2 z-10 flex items-center gap-1 rounded-md border-hair border-border bg-surface/85 p-1 backdrop-blur-md">
            <Tooltip content="Fit map to view">
              <Button
                variant="ghost"
                size="compact"
                aria-label="Fit map to view"
                onClick={fitOrthoToGrid}
                icon={<IconRotate360 size={16} />}
              />
            </Tooltip>
          </div>
        )}

        {/* Bottom-right, not bottom-center. The original reason - a video docked
           bottom-left over the canvas, expanding to half its width - is gone: the video
           has been in the left column since 008e6d4, and the expand it never used was
           deleted in the picture-in-picture pass. The anchor stays where it is because
           these are 3D-only controls and the 2D legend now overlays the SAME canvas along
           its bottom edge; a centred cluster would sit on top of it in 2D, and a
           bottom-right one cannot, because the two views never draw at once. */}
        {viewMode === '3d' && (
          <div className="absolute bottom-2 right-2 z-10 flex flex-col items-end gap-1.5">
            <div className="flex items-center gap-4 rounded-md border-hair border-border bg-surface/85 px-2.5 py-1.5 backdrop-blur-md">
              <Slider
                value={pointSize}
                onChange={setPointSize}
                min={MIN_POINT_SIZE}
                max={MAX_POINT_SIZE}
                step={0.001}
                label="Point size"
                formatValue={(v) => v.toFixed(3)}
                aria-label="Point size"
              />
              <Checkbox
                checked={showRobotModel}
                onChange={setShowRobotModel}
                disabled={robotModelFailed}
                label="Robot model"
                className="shrink-0"
              />
              <Checkbox checked={showTrail} onChange={setShowTrail} label="Show trail" className="shrink-0" />
              <Tooltip
                content={
                  edlSupported ? 'Eye-Dome Lighting: darkens depth edges for a 3D depth cue' : 'Not supported on this device'
                }
              >
                <Checkbox
                  checked={edlEnabled && edlSupported}
                  onChange={setEdlEnabled}
                  disabled={!edlSupported}
                  label="Depth shading"
                  className="shrink-0"
                />
              </Tooltip>
            </div>

            <div className="flex items-center gap-4 rounded-md border-hair border-border bg-surface/85 px-2.5 py-1.5 backdrop-blur-md">
              <span className="flex items-center gap-2 text-xs text-muted">
                <span className="min-w-[84px] shrink-0">Ceiling</span>
                <Select
                  value={ceilingMode}
                  onChange={(v) => setCeilingMode(v as CeilingMode)}
                  disabled={ceilingY == null}
                  aria-label="Ceiling"
                  className="w-28 shrink-0"
                  options={[
                    { value: 'visible', label: 'Visible' },
                    { value: 'hidden', label: 'Hidden' },
                    { value: 'fade', label: 'Fade' },
                  ]}
                />
              </span>
              {/* Enabled whenever the scene knows its ceiling height. It used to be
                  disabled while Ceiling was "Visible" too - which is the DEFAULT, so on
                  every fresh scene the control the user reaches for first was dead, with
                  nothing on screen saying why, and the way to wake it up was to find a
                  different control first. Dragging it now means "cut here", so it moves
                  the mode to Hidden itself; Fade is left alone, since a cutoff is exactly
                  what Fade fades around. The only remaining disabled case is a scene with
                  no ceiling_y at all, and that one says so on hover rather than being
                  silently dead - the readout is an em dash there (formatMeasure's rule). */}
              <Tooltip
                content={
                  ceilingY == null
                    ? 'This scene has no measured ceiling height, so there is nothing to cut against.'
                    : 'Hides everything above this height. Dragging switches Ceiling to Hidden.'
                }
              >
                <span className="flex flex-1 items-center">
                  <Slider
                    value={heightCutoff ?? ceilingY ?? 0}
                    onChange={(v) => {
                      setHeightCutoff(v)
                      if (ceilingMode === 'visible') setCeilingMode('hidden')
                    }}
                    min={0}
                    max={ceilingY ?? 0}
                    step={0.01}
                    disabled={ceilingY == null}
                    label="Height cutoff"
                    formatValue={(v) => (ceilingY == null ? '—' : `${v.toFixed(2)} m`)}
                    aria-label="Height cutoff"
                  />
                </span>
              </Tooltip>
            </div>
          </div>
        )}

        {glb.loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-canvas/60">
            <div className="text-center">
              <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-accent/30 border-t-accent" />
              <p className="text-sm text-subtle">Loading point cloud…</p>
              {glb.progress && glb.progress.total > 0 && (
                <p className="mt-1 text-xs text-muted">
                  {(glb.progress.loaded / 1e6).toFixed(0)} of {(glb.progress.total / 1e6).toFixed(0)} MB
                </p>
              )}
            </div>
          </div>
        )}

        {glb.error && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-canvas/60">
            <p className="text-sm text-data-unreachable">{glb.error}</p>
          </div>
        )}
        {/* The 2D legend OVERLAYS the canvas; it must never take flow height. As a
          * sibling of the canvas host in the column flex it cost the host ~27px on every
          * entry into 2D, and that height change is what fired the ResizeObserver above -
          * `setSize` reallocating the backend's canvas-sized targets mid-flight is the
          * `Destroyed texture ... used in a submit` fault (HANDOFF, 2026-09-09). Absolute
          * positioning deletes the trigger. `pointer-events-none` keeps drag/zoom on the
          * map underneath. */}
        {viewMode === '2d' && grid && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-wrap items-center gap-4 border-t-hair border-border bg-surface/85 px-3 py-1.5 text-xs text-muted backdrop-blur-md">
            {colorMode === 'occupancy' ? (
              <>
                <Legend color="var(--occ-free)" label="free" />
                <Legend color="var(--occ-obstacle)" label="obstacle" />
                <Legend color="var(--occ-unknown)" label="unknown" />
              </>
            ) : (
              <>
                <Legend color="var(--data-unreachable)" label={`< ${effectiveRadius.toFixed(2)} m · won't fit`} />
                <Legend
                  color={CLEARANCE_TIGHT_CSS}
                  label={`${effectiveRadius.toFixed(2)}–${(effectiveRadius * 2).toFixed(2)} m · tight`}
                />
                <Legend color="var(--occ-free)" label="free" />
                <Legend color="var(--occ-unknown)" label="unknown" />
              </>
            )}
            {cameraTrack && cameraTrack.length > 1 && <Legend color="var(--data-trajectory)" label="camera path" />}
            <span className="text-faint">·</span>
            <span>scroll to zoom, drag to pan</span>
          </div>
        )}
      </div>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
      {label}
    </span>
  )
}

// Rows for the "?" navigation-help panel (3D only) - a gesture/key on the left, what
// it does on the right. Kept as data rather than inline JSX so NavHelpPanel stays a
// plain table.
const NAV_HELP_ROWS: [string, string][] = [
  ['Left drag', 'Orbit'],
  ['Right / middle drag', 'Pan'],
  ['Shift + left drag', 'Pan'],
  ['Two-finger scroll', 'Pan'],
  ['Two-finger rotate (Safari)', 'Orbit'],
  ['Pinch / Ctrl + scroll', 'Zoom'],
  ['W A S D / arrow keys', 'Move'],
  ['Q / E', 'Down / up'],
  ['Shift (held)', 'Move faster'],
  ['Double-click object', 'Send robot here'],
  ['Shift + double-click object', 'Frame object'],
]

function NavHelpPanel() {
  return (
    <div className="w-64 rounded-md border-hair border-border bg-surface-raised p-2.5 text-[11px] shadow-lg">
      <p className="mb-1.5 font-medium text-fg">Navigation</p>
      <table className="w-full border-collapse">
        <tbody>
          {NAV_HELP_ROWS.map(([keys, action]) => (
            <tr key={keys}>
              <td className="py-0.5 pr-2 text-muted">{keys}</td>
              <td className="py-0.5 text-right text-subtle">{action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
