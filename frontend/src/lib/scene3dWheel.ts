// Pure classification of a `wheel` event for the 3D view - kept separate from
// SceneMap3D so it's testable without a WebGL context (same rationale as
// scene3dCamera.ts and pointerGesture.ts).
//
// On macOS trackpads, two-finger scroll and pinch-to-zoom both arrive as `wheel`
// events - the only thing that tells them apart is `ctrlKey`: the OS synthesizes
// ctrlKey=true for a pinch (and for an actual Ctrl+scroll, which OrbitControls also
// already treats as zoom). A real mouse wheel has no pinch gesture at all, so its
// unmodified scroll falls into the same "not ctrl -> pan" bucket as trackpad scroll,
// which matches how every other pan-on-scroll app (Figma, Google Maps, etc.) behaves.
//
// `deltaMode` needs normalising because trackpads report DOM_DELTA_PIXEL (0) while a
// physical mouse wheel typically reports DOM_DELTA_LINE (1) - without correcting for
// this, the same physical "one wheel click" travels a wildly different distance than a
// trackpad's pixel-accurate scroll. Three's own OrbitControls._customWheelEvent does
// the same correction, but only for deltaY (its dolly only ever reads one axis); this
// module has to cover deltaX too since pan uses both.

export interface WheelInput {
  deltaX: number
  deltaY: number
  /** WheelEvent.deltaMode: 0 = DOM_DELTA_PIXEL, 1 = DOM_DELTA_LINE, 2 = DOM_DELTA_PAGE. */
  deltaMode: number
  ctrlKey: boolean
}

export type WheelClassification = { kind: 'dolly' } | { kind: 'pan'; panX: number; panY: number }

// Same constants as OrbitControls._customWheelEvent (OrbitControls.js), so a "one wheel
// click" scroll and a "one wheel click" zoom feel like the same physical gesture size.
const LINE_HEIGHT_PX = 16
const PAGE_HEIGHT_PX = 100

/** Multiplier from normalised wheel pixels to the pixel amount handed to
 * `controls.pan()`. 1 keeps two-finger scroll 1:1 with the pixels the fingers moved. */
export const WHEEL_PAN_SPEED = 1

/** Classifies a wheel event as a pinch-zoom (ctrlKey) or a pan (everything else),
 * normalising deltaMode and inverting sign so panning tracks the fingers: content under
 * the fingers follows them, the same convention as a native scrollable container. */
export function classifyWheel(e: WheelInput): WheelClassification {
  if (e.ctrlKey) return { kind: 'dolly' }
  const scale = e.deltaMode === 1 ? LINE_HEIGHT_PX : e.deltaMode === 2 ? PAGE_HEIGHT_PX : 1
  return {
    kind: 'pan',
    panX: -e.deltaX * scale * WHEEL_PAN_SPEED,
    panY: -e.deltaY * scale * WHEEL_PAN_SPEED,
  }
}
