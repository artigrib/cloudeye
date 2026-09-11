// Pure click-vs-drag classification for the 3D view's canvas - kept separate from
// SceneMap3D so it's testable without a WebGL context (same rationale as
// scene3dCamera.ts). The canvas has clickable object markers, so a camera drag (orbit,
// pan) must never be misread as a click on one of them, and vice versa.

/** A single pointerdown/pointerup sample worth comparing. */
export interface PointerSample {
  pointerId: number
  /** MouseEvent.button - 0 is the primary (left) button. */
  button: number
  x: number
  y: number
  timeMs: number
}

/** Beyond this many pixels of movement between down and up, it's a drag. */
export const CLICK_MOVE_THRESHOLD_PX = 5

/** Beyond this many milliseconds between down and up, it's a drag (or a long-press),
 * not a click - this is what stops "press, orbit around, release back near the start"
 * from registering as a click on whatever marker happens to be under the pointer. */
export const CLICK_MAX_DURATION_MS = 500

/** Whether a pointerdown should even be considered as the start of a possible click.
 * Only the primary button, from the primary pointer, counts - a middle/right-button
 * press (both used for panning) must not arm the click detector at all. */
export function isClickCandidate(e: { button: number; isPrimary: boolean }): boolean {
  return e.isPrimary && e.button === 0
}

/** Whether a down/up pair, taken together, constitutes a click rather than a drag.
 * Requires the same pointer and button throughout, small enough movement, and a short
 * enough press. Thresholds are inclusive - exactly 5px or exactly 500ms still counts. */
export function isClickGesture(down: PointerSample, up: PointerSample): boolean {
  if (down.pointerId !== up.pointerId) return false
  if (up.button !== 0) return false
  const dist = Math.hypot(up.x - down.x, up.y - down.y)
  if (dist > CLICK_MOVE_THRESHOLD_PX) return false
  const duration = up.timeMs - down.timeMs
  if (duration > CLICK_MAX_DURATION_MS) return false
  return true
}
