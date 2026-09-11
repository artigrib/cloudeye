// Pure angle math for the robot's yaw in the 3D scene - kept separate from SceneMap3D so it's
// unit-testable without a WebGL context, and separate from path-interp.ts because that module
// is about arc-length parameterisation of a polyline (consumed by useCommandAnimation); this is
// about angles and temporal damping (consumed only by SceneMap3D).
//
// Convention this module assumes and pins with tests: the robot GLB is baked (see
// scripts/build-turtlebot-glb.mjs) with a single rotateX(-PI/2), which maps ROS/URDF axes
// (x,y,z) to scene axes (x, z, -y). ROS +X (forward) -> scene +X, so the model's local forward
// is scene +X, and a yaw theta about scene +Y sends (1,0,0) to (cos theta, 0, -sin theta).
// That's why headingYaw below returns atan2(-dz, dx), not atan2(dz, dx) - get this backwards
// and the robot faces exactly opposite its direction of travel.

const TAU = Math.PI * 2

/** Normalizes an angle to [-PI, PI). Note the half-open interval: normalizeAngle(PI) === -PI. */
export function normalizeAngle(a: number): number {
  // Double mod because JS `%` keeps the sign of its left operand (e.g. -1 % TAU === -1, not
  // TAU - 1), so a single mod can leave the result outside [0, TAU) before the final shift.
  return ((((a + Math.PI) % TAU) + TAU) % TAU) - Math.PI
}

/** Signed smallest rotation from `a` to `b`, in [-PI, PI). Correct across the +-PI seam. */
export function shortestAngleDelta(a: number, b: number): number {
  return normalizeAngle(b - a)
}

export interface HeadingYawOptions {
  /** Below this move length (metres), treat it as sensor/interpolation noise and hold heading. */
  minMove?: number
  /** Above this move length (metres), treat it as a teleport (e.g. a new command's start pose
   * snapping in), not motion - hold heading rather than pointing at a discontinuity. */
  maxMove?: number
}

const DEFAULT_MIN_MOVE = 1e-4 // 0.1mm - at 0.3 m/s a 16ms frame covers ~5mm, 50x this
const DEFAULT_MAX_MOVE = 0.5 // half a metre in one frame is a command boundary, not a move

/**
 * Yaw about scene +Y that points the model's local forward (+X) along the ground-plane
 * displacement (dx, dz). Returns null when the displacement is too small to infer a direction
 * from, or too large to be real motion - callers should hold the previous heading rather than
 * snapping to whatever this returns.
 */
export function headingYaw(dx: number, dz: number, opts: HeadingYawOptions = {}): number | null {
  const minMove = opts.minMove ?? DEFAULT_MIN_MOVE
  const maxMove = opts.maxMove ?? DEFAULT_MAX_MOVE
  const len = Math.hypot(dx, dz)
  if (len < minMove || len > maxMove) return null
  return Math.atan2(-dz, dx)
}

/**
 * Frame-rate-independent exponential approach of `current` toward `target`, wrapping correctly
 * across the +-PI seam. `smoothTimeMs` is the time to close ~63% of the remaining angular gap
 * (1 - 1/e); repeated short steps and one long step of the same total duration agree exactly,
 * because this is a true exponential decay rather than a fixed per-frame lerp fraction - this
 * app's render cadence is irregular (setState-driven during a move, rAF-driven while
 * OrbitControls damps), so a fixed factor would turn the robot at different rates depending on
 * frame rate.
 */
export function dampAngle(current: number, target: number, dtMs: number, smoothTimeMs: number): number {
  if (dtMs <= 0) return normalizeAngle(current)
  if (smoothTimeMs <= 0) return normalizeAngle(target)
  const alpha = 1 - Math.exp(-dtMs / smoothTimeMs)
  return normalizeAngle(current + shortestAngleDelta(current, target) * alpha)
}
