// Pure keyboard-navigation logic for the 3D view - kept separate from SceneMap3D so
// it's testable without a WebGL context (same rationale as scene3dCamera.ts).
//
// Bindings are keyed on KeyboardEvent.code (physical key position), not .key, so WASD
// stays under the same three keys on AZERTY/Dvorak layouts the way game controls
// conventionally do; the arrow keys are always available as a layout-independent
// fallback.

/** Every key code this module consumes - used both to decide whether to
 * preventDefault() a keydown and to track which keys are currently held. */
export const NAV_KEYS = new Set([
  'KeyW',
  'KeyA',
  'KeyS',
  'KeyD',
  'KeyQ',
  'KeyE',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
])

export function isNavKey(code: string): boolean {
  return NAV_KEYS.has(code)
}

const TYPING_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT', 'VIDEO'])

/** True while keyboard input should be treated as typing/interacting with a control,
 * not as a camera command - the command box (ChatPanel's "Command the robot" input),
 * this view's own fade/colour <select>s, any contenteditable, and a focused <video>
 * (whose native controls use arrow keys/space for seeking). Nothing else in the app
 * currently claims WASD/arrows/Q/E, so anywhere else is fair game. */
export function isTypingTarget(el: Element | null): boolean {
  if (!el) return false
  // `isContentEditable` also catches an element that only *inherits* editability from
  // an ancestor, which a plain attribute check on `el` itself would miss - but jsdom
  // (this file's test environment) doesn't implement the getter at all, hence the
  // attribute fallback below.
  if (el instanceof HTMLElement && el.isContentEditable) return true
  const editable = el.getAttribute('contenteditable')
  if (editable === '' || editable === 'true') return true
  if (TYPING_TAGS.has(el.tagName)) return true
  return el.getAttribute('role') === 'textbox'
}

export interface KeyAxes {
  /** -1 (left) .. 1 (right), floor-plane, relative to the camera's current view. */
  right: number
  /** -1 (back) .. 1 (forward, the way the camera looks projected onto the floor). */
  forward: number
  /** -1 (down, Q) .. 1 (up, E). Independent of forward/right - vertical flight isn't
   * normalised together with the floor-plane axes, so holding W+E moves forward at
   * full speed AND rises at full speed, rather than splitting speed between them. */
  up: number
}

export interface KeyState {
  forward: boolean
  back: boolean
  left: boolean
  right: boolean
  down: boolean
  up: boolean
}

/** Derives the held-key state from a set of currently-pressed `KeyboardEvent.code`s. */
export function keyStateFromCodes(codes: ReadonlySet<string>): KeyState {
  return {
    forward: codes.has('KeyW') || codes.has('ArrowUp'),
    back: codes.has('KeyS') || codes.has('ArrowDown'),
    left: codes.has('KeyA') || codes.has('ArrowLeft'),
    right: codes.has('KeyD') || codes.has('ArrowRight'),
    down: codes.has('KeyQ'),
    up: codes.has('KeyE'),
  }
}

/** Turns a key state into a movement vector. The floor-plane component (right,
 * forward) is normalised so a diagonal (e.g. W+A) travels at the same speed as a
 * cardinal direction, not sqrt(2) times faster. */
export function panAxesForKeys(keys: KeyState): KeyAxes {
  let right = (keys.right ? 1 : 0) - (keys.left ? 1 : 0)
  let forward = (keys.forward ? 1 : 0) - (keys.back ? 1 : 0)
  const len = Math.hypot(right, forward)
  if (len > 1) {
    right /= len
    forward /= len
  }
  const up = (keys.up ? 1 : 0) - (keys.down ? 1 : 0)
  return { right, forward, up }
}

/** Base keyboard-pan speed, in screen pixels/second, fed into OrbitControls' own
 * pan() - matching drag-pan's units means keyboard motion at this speed feels like
 * dragging at the same rate, and is automatically scale-relative (see
 * pixelsToWorldUnits below): a small room pans slowly, a large one quickly, exactly
 * like a mouse drag would. */
export const KEY_PAN_SPEED_PX_PER_SEC = 900

/** Shift multiplies speed while held, per spec ("Hold to accelerate, not toggle"). */
export const KEY_SHIFT_MULTIPLIER = 3

/** A backgrounded tab resuming after a long gap must not teleport the camera in one
 * jump - same clamp precedent as robot-heading.ts's YAW_DT_CLAMP_MS. */
export const MAX_KEY_DT_MS = 100

/** Pixel distance to travel this frame, given a px/sec speed and elapsed time. */
export function stepDistancePx(speedPxPerSec: number, dtMs: number): number {
  return speedPxPerSec * (Math.min(dtMs, MAX_KEY_DT_MS) / 1000)
}

/** Converts a screen-pixel distance into world units, at the given vertical FOV and
 * viewport height - the exact formula OrbitControls' own `_pan` uses internally
 * (`2 * distancePx * targetDistance * tan(fov/2) / clientHeight`), reproduced here so
 * Q/E's vertical motion (which can't go through `controls.pan()` once
 * screenSpacePanning is false) still matches the horizontal pan speed the eye already
 * calibrated to when dragging or using WASD. */
export function pixelsToWorldUnits(
  distancePx: number,
  fovDegrees: number,
  targetDistance: number,
  clientHeightPx: number,
): number {
  if (clientHeightPx <= 0) return 0
  const scaled = targetDistance * Math.tan((fovDegrees / 2) * (Math.PI / 180))
  return (2 * distancePx * scaled) / clientHeightPx
}
