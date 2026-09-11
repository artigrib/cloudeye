// Adaptive point-render budget - pure logic, no WebGL/DOM (see deviceProbe.ts for the
// impure boundary that gathers the inputs this module consumes). Replaces the old fixed
// `MAX_POINTS = 400_000` (see pointcloudDecimate.ts) with a per-device budget: a static
// prior from signals that actually correlate with rendering cost, corrected by measured
// frame time during the first real interaction. Kept pure and colocated with a test per
// this repo's convention - every other frontend test is pure logic with no WebGL context.

/** Hard floor - below this a scene reads as obviously sparse regardless of device. */
export const MIN_POINT_BUDGET = 50_000
/** Hard ceiling on what the viewer will ever draw, whatever the device prior says and
 * however many points the source has. 1,000,000, down from the 2,000,000 that mirrored
 * gpu/stage_align.py's pre-SOR working cap: that number described what the PIPELINE
 * keeps, not what a browser should rasterise, and the 15 fps hero ships 3,263,666 points
 * (cloud.bin, 48,955,030 bytes), so on a capable client the prior could ask for twice
 * what any viewer benefits from. Points over the ceiling are dropped by
 * pointcloudDecimate.decimateIndices, which spreads the kept indices evenly across the
 * whole cloud rather than truncating - taking the first N would silently cut off
 * whichever part of the room was scanned last. */
export const MAX_POINT_BUDGET = 1_000_000
/** What MAX_POINTS used to be unconditionally - now only the value used when a device
 * probe genuinely can't produce anything better (e.g. in a test/SSR context). Also
 * doubles as the GLB's own export budget (see gpu/stage_export_glb.py via
 * app.config.Settings.glb_max_points) - after Phase 0, "already under budget" is the
 * common case for a device landing exactly here. */
export const FALLBACK_POINT_BUDGET = 400_000

const STORAGE_KEY = 'cloudeye:pointBudget:v1'

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

function roundToNearest10k(v: number): number {
  return Math.round(v / 10_000) * 10_000
}

/** Everything the static prior needs - gathered by deviceProbe.ts's `probeDevice`, kept
 * as a plain data interface here so the scoring logic is testable without touching
 * `navigator`/`matchMedia`/WebGL. */
export interface DeviceBudgetInput {
  /** matchMedia('(pointer: coarse)') - a proxy for high dpr, tight memory budget, and
   * thermal throttling under sustained rendering, not literally "is this a phone". */
  coarsePointer: boolean
  /** navigator.hardwareConcurrency, or null where unavailable/untrustworthy. */
  hardwareConcurrency: number | null
  /** navigator.deviceMemory (Chromium-only, quantised to {0.25,0.5,1,2,4,8}), or null
   * on browsers that don't expose it (Safari, Firefox). */
  deviceMemoryGb: number | null
  /** renderer.getDrawingBufferSize() width*height - point-cloud cost is fill-rate
   * dominated, and this is measured, not guessed, unlike the other signals here. */
  drawingBufferPixels: number
  /** WEBGL_debug_renderer_info matched against a software-rasterizer pattern
   * (SwiftShader/llvmpipe/Basic Render Driver) - a hard floor, not a tier: on these,
   * even a modest point count is catastrophic, and no other signal here reliably
   * predicts it. */
  softwareRenderer: boolean
  /** navigator.connection?.saveData - a real user-expressed preference, not a capability
   * signal, but the same "give me less" intent applies to point count as to network. */
  saveData: boolean
}

/** A conservative device prior BEFORE the first frame has ever rendered. None of these
 * signals are strong individually (see deviceProbe.ts's doc comments for why
 * navigator.gpu and maxTextureSize were rejected outright as tier signals) - this is a
 * starting point that observeFrame below then corrects against reality. */
export function initialPointBudget(input: DeviceBudgetInput): number {
  if (input.softwareRenderer) return roundToNearest10k(clamp(60_000, MIN_POINT_BUDGET, MAX_POINT_BUDGET))

  let budget = input.coarsePointer ? 350_000 : 1_200_000

  if (input.deviceMemoryGb === null) budget *= 0.85
  else if (input.deviceMemoryGb <= 2) budget *= 0.4
  else if (input.deviceMemoryGb <= 4) budget *= 0.7

  if (input.hardwareConcurrency === null) budget *= 0.85
  else if (input.hardwareConcurrency <= 4) budget *= 0.6
  else if (input.hardwareConcurrency <= 8) budget *= 0.85

  // ~1920x1080 is the reference the base budgets above were picked against; a denser
  // drawing buffer (retina, 4K) pays more per point in fill rate for the same count.
  const REFERENCE_PIXELS = 1920 * 1080
  if (input.drawingBufferPixels > 0) {
    budget *= clamp(REFERENCE_PIXELS / input.drawingBufferPixels, 0.5, 1.0)
  }

  if (input.saveData) budget *= 0.5

  return roundToNearest10k(clamp(budget, MIN_POINT_BUDGET, MAX_POINT_BUDGET))
}

/** Frame-time-based correction, applied on top of the static prior. Deliberately
 * one-directional: on-demand rendering means consecutive frames only exist during real
 * interaction (a drag, a pan), which is exactly when a too-high budget is felt - and
 * exactly the moment worth sampling. */
export interface FrameBudgetState {
  budget: number
  samples: number[]
  downshifts: number
  settled: boolean
}

const WARMUP_SAMPLES_TO_DROP = 3
const SAMPLES_TO_COLLECT = 20
const SLOW_FRAME_MS = 33 // ~30fps floor; absolute, not relative to a 16.7ms/60fps assumption
const MAX_DOWNSHIFTS = 2

export function createFrameBudget(initial: number): FrameBudgetState {
  return { budget: clamp(initial, MIN_POINT_BUDGET, MAX_POINT_BUDGET), samples: [], downshifts: 0, settled: false }
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

/** Feeds one frame's duration into the sampler. Returns a new state - callers should
 * replace their stored state with the result, then check `pendingBudget` to see whether
 * a downshift is ready to apply. Never mutates `state`. */
export function observeFrame(state: FrameBudgetState, frameMs: number): FrameBudgetState {
  if (state.settled || state.downshifts >= MAX_DOWNSHIFTS) return { ...state, settled: true }

  const samples = [...state.samples, frameMs]
  // Drop warmup samples (shader compile, first buffer upload, GC) by simply not
  // counting toward SAMPLES_TO_COLLECT until enough have passed - simpler than a
  // separate counter, and self-correcting if a slow warmup frame sneaks past 3.
  if (samples.length <= WARMUP_SAMPLES_TO_DROP) return { ...state, samples }

  const collected = samples.slice(WARMUP_SAMPLES_TO_DROP)
  if (collected.length < SAMPLES_TO_COLLECT) return { ...state, samples }

  const m = median(collected)
  if (m > SLOW_FRAME_MS) {
    const nextBudget = roundToNearest10k(clamp(state.budget * 0.5, MIN_POINT_BUDGET, MAX_POINT_BUDGET))
    return { budget: nextBudget, samples: [], downshifts: state.downshifts + 1, settled: false }
  }
  // Never upshift - see the module doc comment. A budget that turned out generous
  // enough just stops sampling.
  return { ...state, samples: [], settled: true }
}

/** Non-null exactly when observeFrame just produced a fresh downshift this call - i.e.
 * "a rebuild is due". Callers should apply it once the current interaction settles
 * (never mid-drag - a point-cloud rebuild while the user is dragging is the one thing
 * worse than the slow frames that triggered it), then keep going with the returned
 * state either way. */
export function pendingBudget(prev: FrameBudgetState, next: FrameBudgetState): number | null {
  return next.downshifts > prev.downshifts ? next.budget : null
}

/** Below this many room-diagonals of camera-to-room-center distance, the camera is still
 * "in" or close to the room - full budget, no LOD reduction. Zooming INTO a room must
 * never cost detail, only pulling back from it. */
const DISTANCE_LOD_NEAR_RATIO = 1.5
/** Beyond this many room-diagonals away, the floor fraction below applies and stays
 * flat - once the room already reads as a small object in frame, shrinking the budget
 * further buys nothing (there's no benefit rendering to a sub-pixel point density that
 * fine when the whole room only covers a few hundred screen pixels). */
const DISTANCE_LOD_FAR_RATIO = 5
/** Floor fraction of `baseBudget` kept once fully zoomed out (see DISTANCE_LOD_FAR_RATIO)
 * - never goes to zero, so the room doesn't visibly thin out to nothing at a distance. */
const DISTANCE_LOD_FLOOR_FRACTION = 0.35

/** Distance-based level-of-detail on top of `baseBudget` (the device-tier + frame-time
 * corrected budget from initialPointBudget/observeFrame above): scales the point count
 * down as the camera pulls back from the room, linearly between DISTANCE_LOD_NEAR_RATIO
 * and DISTANCE_LOD_FAR_RATIO room-diagonals of distance, flooring at
 * DISTANCE_LOD_FLOOR_FRACTION of `baseBudget` beyond that - a gradual ramp, not a single
 * pop threshold, since this feeds a geometry rebuild (see SceneMap3D's maybeStopLoop)
 * that's only ever applied once the camera settles, not continuously mid-drag.
 *
 * A no-op (returns `baseBudget` unchanged) whenever `roomDiagonal` isn't known yet (<=0,
 * before the point cloud's own bounding box has been computed) or `cameraDistance` isn't
 * a finite number. */
export function distanceLodBudget(baseBudget: number, cameraDistance: number, roomDiagonal: number): number {
  if (roomDiagonal <= 0 || !Number.isFinite(cameraDistance)) return baseBudget
  const ratio = cameraDistance / roomDiagonal
  if (ratio <= DISTANCE_LOD_NEAR_RATIO) return baseBudget
  const t = clamp((ratio - DISTANCE_LOD_NEAR_RATIO) / (DISTANCE_LOD_FAR_RATIO - DISTANCE_LOD_NEAR_RATIO), 0, 1)
  const fraction = 1 - t * (1 - DISTANCE_LOD_FLOOR_FRACTION)
  return roundToNearest10k(clamp(baseBudget * fraction, MIN_POINT_BUDGET, baseBudget))
}

interface StoredPointBudget {
  budget: number
  drawingBufferPixels: number
}

/** Cache is a device property, not a per-scene one - keyed globally, not by sceneId
 * (contrast pointcloudViewSettings.ts, which is deliberately per-scene). Invalidated
 * when the drawing buffer moved enough to matter (a window dragged to another monitor,
 * a browser zoom change) rather than on every pixel change. */
export function pointBudgetCacheIsValid(cached: StoredPointBudget, currentDrawingBufferPixels: number): boolean {
  if (currentDrawingBufferPixels <= 0 || cached.drawingBufferPixels <= 0) return false
  const ratio = currentDrawingBufferPixels / cached.drawingBufferPixels
  return ratio >= 1 / 1.5 && ratio <= 1.5
}

export function loadPointBudget(currentDrawingBufferPixels: number): number | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<StoredPointBudget>
    if (typeof parsed.budget !== 'number' || typeof parsed.drawingBufferPixels !== 'number') return null
    if (!pointBudgetCacheIsValid(parsed as StoredPointBudget, currentDrawingBufferPixels)) return null
    return clamp(parsed.budget, MIN_POINT_BUDGET, MAX_POINT_BUDGET)
  } catch {
    return null
  }
}

export function savePointBudget(budget: number, drawingBufferPixels: number): void {
  try {
    const payload: StoredPointBudget = { budget, drawingBufferPixels }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  } catch {
    // localStorage unavailable (private mode, disabled) - non-fatal, just re-measures
    // next session instead of once ever.
  }
}
