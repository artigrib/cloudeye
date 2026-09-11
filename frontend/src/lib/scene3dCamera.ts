// Pure geometry for the 3D view's camera framing - kept separate from the Three.js
// component so it's testable without a WebGL context.

export interface BoundingBox3 {
  min: [number, number, number]
  max: [number, number, number]
}

export interface CameraPose {
  position: [number, number, number]
  target: [number, number, number]
}

function boxCenter(box: BoundingBox3): [number, number, number] {
  return [
    (box.min[0] + box.max[0]) / 2,
    (box.min[1] + box.max[1]) / 2,
    (box.min[2] + box.max[2]) / 2,
  ]
}

function boxSize(box: BoundingBox3): [number, number, number] {
  return [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]]
}

/** Distance back from the bbox center that comfortably frames it, regardless of room
 * size - the ikport room (3.6x4.3m) and a much larger one both end up with the whole
 * footprint in view. */
function framingDistance(box: BoundingBox3): number {
  const [sx, sy, sz] = boxSize(box)
  const horizontal = Math.hypot(sx, sz)
  return Math.max(horizontal, sy, 0.5) * 1.3
}

/** Camera near/far planes derived from the point cloud's own bounding box diagonal,
 * replacing the fixed 0.05/500 the camera was constructed with. NOT a depth-precision
 * fix at room scale - with 24-bit depth, 0.05/500 already resolves to roughly 11
 * micrometers at a typical 3m viewing distance, about 500x tighter than the 4-6mm
 * offsets this file's PATH_Y_OFFSET/TRAIL_Y_OFFSET already rely on to avoid z-fighting.
 * This is forward-looking instead: a scan an order of magnitude larger than one room (a
 * whole floor, a warehouse) shouldn't inherit clip planes tuned for a 4m space, where
 * near=0.05 would leave a 200m scan with only ~12mm precision at 100m and the same
 * ground-offset lines would start visibly z-fighting. */
export function cameraClipPlanes(box: BoundingBox3): { near: number; far: number } {
  const [sx, sy, sz] = boxSize(box)
  const diag = Math.hypot(sx, sy, sz)
  return {
    near: Math.min(0.1, Math.max(0.02, diag * 0.01)),
    far: Math.max(50, diag * 8),
  }
}

/** Initial camera pose for a freshly loaded point cloud: above the room, looking down
 * at roughly 45 degrees, target at the bbox center. */
export function initialCameraPose(box: BoundingBox3): CameraPose {
  const target = boxCenter(box)
  const distance = framingDistance(box)
  const elevation = Math.PI / 4 // 45 degrees above the horizon
  return {
    target,
    position: [target[0], target[1] + distance * Math.sin(elevation), target[2] + distance * Math.cos(elevation)],
  }
}

/** Straight down, floor-plan style. A tiny Z offset keeps the camera off the singular
 * "looking straight down the up-axis" pose, which makes OrbitControls' orbit direction
 * ill-defined right after a reset. */
export function topCameraPose(box: BoundingBox3): CameraPose {
  const target = boxCenter(box)
  const distance = framingDistance(box)
  return { target, position: [target[0], target[1] + distance, target[2] + distance * 0.001] }
}

/** Roughly eye-level, looking at the room from outside one wall. */
export function frontCameraPose(box: BoundingBox3): CameraPose {
  const target = boxCenter(box)
  const distance = framingDistance(box)
  return { target, position: [target[0], target[1] + distance * 0.15, target[2] + distance] }
}

/** A corner/isometric-ish view - same elevation as the initial pose, offset to a
 * diagonal so two walls are visible at once instead of one face-on. */
export function isometricCameraPose(box: BoundingBox3): CameraPose {
  const target = boxCenter(box)
  const distance = framingDistance(box)
  const elevation = Math.PI / 5
  const azimuth = Math.PI / 4
  return {
    target,
    position: [
      target[0] + distance * Math.cos(elevation) * Math.sin(azimuth),
      target[1] + distance * Math.sin(elevation),
      target[2] + distance * Math.cos(elevation) * Math.cos(azimuth),
    ],
  }
}

/** Camera pose that recentres the orbit target on an object's bbox and frames it,
 * for the "shift+double-click an object to look at it" gesture. Unlike the preset
 * poses above, this keeps the *current* viewing direction (the unit vector from the
 * old target to the camera) rather than picking a fixed elevation/azimuth: dollying
 * straight in along the existing direction reads as "zoom to this thing", not a
 * camera cut, and - since that direction came from OrbitControls, which already
 * clamps polar angle - guarantees the result pose is still within maxPolarAngle, so
 * nothing snaps on the next controls.update(). Falls back to the standard 45-degree
 * elevation when the camera is (near enough) sitting on the old target already, since
 * there's no direction to preserve. */
export function frameObjectPose(box: BoundingBox3, from: [number, number, number]): CameraPose {
  const target = boxCenter(box)
  const distance = framingDistance(box)
  const dir: [number, number, number] = [from[0] - target[0], from[1] - target[1], from[2] - target[2]]
  const len = Math.hypot(dir[0], dir[1], dir[2])
  if (len < 1e-6) return initialCameraPose(box)
  return {
    target,
    position: [
      target[0] + (dir[0] / len) * distance,
      target[1] + (dir[1] / len) * distance,
      target[2] + (dir[2] / len) * distance,
    ],
  }
}

/** Cubic ease-in-out, t in [0, 1] -> [0, 1]. Used to tween the camera between poses
 * (see frameObjectPose) without the linear "start/stop instantly" feel of a plain lerp. */
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2
}

function lerp3(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/** Interpolates position and target independently between two camera poses. Valid for
 * a straight dolly between two poses that share a viewing direction (see
 * frameObjectPose) - it is not a spherical/orbit interpolation, so it would visibly
 * cut corners between two arbitrarily different poses. */
export function lerpPose(from: CameraPose, to: CameraPose, t: number): CameraPose {
  return { position: lerp3(from.position, to.position, t), target: lerp3(from.target, to.target, t) }
}
