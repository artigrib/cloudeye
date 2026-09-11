// Point-cloud splat sizing - pure math, no three/WebGPU. The TSL node graph in
// scene3dPoints.ts's createPointCloudMaterials can't be unit-tested directly (it only
// compiles inside a live renderer - no test in this repo imports 'three/tsl' or mocks a
// WebGPU context), so the INTENDED sizing math lives here as plain JS, is pinned by
// pointSizing.test.ts, and the TSL expression is written to mirror it term for term.
// Same convention as edlSettings.ts (see its module comment) - kept in its own file (not
// inside scene3dPoints.ts) so this stays a pure-logic test like every other test under
// frontend/src/lib/, since scene3dPoints.ts imports 'three/webgpu' and does a
// module-load-time monkeypatch.
//
// The contract, in full - this is the formula BOTH renderers must produce. It's three's
// own classic WebGLRenderer points formula (WebGLMaterials.refreshUniformsPoints), which
// classicPointCloudMaterial.ts's vertex shader implements directly:
//
//   sizeDevicePx = clamp( sizeMeters * pixelRatio * halfCanvasHeightLogicalPx / -viewZ,
//                         MIN, MAX )
//
// The clamp is applied to a DEVICE-pixel quantity, i.e. AFTER pixelRatio - a floor of 1
// device px, not 1 CSS px (a bug fixed by this file: the previous WebGPU/TSL code
// clamped the pre-attenuation, pre-DPR METERS value to [1, 32], which is always >= 1 for
// every point size on the slider (0.003-0.06m) - pinning every point at a constant 1.0
// and disabling the slider entirely. See docs/DECISIONS.md).

/** On-screen splat floor, in DEVICE pixels. Below ~1 device px a splat shrinks into a
 * subpixel and effectively disappears, which reads as holes in the cloud rather than as
 * smaller points - see docs/DECISIONS.md's sparsity investigation. */
export const POINT_SIZE_MIN_DEVICE_PX = 1.0
/** On-screen splat ceiling, in DEVICE pixels - stops a near-camera point from becoming a
 * screen-filling disc. */
export const POINT_SIZE_MAX_DEVICE_PX = 32.0

/** Perspective-attenuated splat size in device pixels, BEFORE the [MIN, MAX] clamp.
 * `sizeMeters` is the user-facing slider value (world metres - SceneMap3D's
 * MIN/MAX_POINT_SIZE = 0.003..0.06). `viewZ` is view-space z (negative in front of the
 * camera - the raw `positionView.z` node value). `halfCanvasHeightLogicalPx` is
 * `renderer.getSize().y * 0.5`, in CSS/logical px, NOT device px - the pixelRatio factor
 * is applied separately and exactly once here, which is what makes a double-DPR
 * regression visible as an exact 2x in the unclamped value at DPR=2 vs DPR=1. */
export function pointSizeDevicePxUnclamped(
  sizeMeters: number,
  viewZ: number,
  halfCanvasHeightLogicalPx: number,
  pixelRatio: number,
): number {
  const depth = Math.max(-viewZ, 1e-6)
  return (sizeMeters * pixelRatio * halfCanvasHeightLogicalPx) / depth
}

/** The final on-screen splat size in device pixels - what both the classic GLSL material
 * (classicPointCloudMaterial.ts) and the WebGPU/TSL material (scene3dPoints.ts) must
 * produce for the same inputs, at every device-pixel-ratio. */
export function pointSizeDevicePx(
  sizeMeters: number,
  viewZ: number,
  halfCanvasHeightLogicalPx: number,
  pixelRatio: number,
): number {
  const raw = pointSizeDevicePxUnclamped(sizeMeters, viewZ, halfCanvasHeightLogicalPx, pixelRatio)
  return Math.min(POINT_SIZE_MAX_DEVICE_PX, Math.max(POINT_SIZE_MIN_DEVICE_PX, raw))
}

/** What `PointsNodeMaterial.sizeNode` must evaluate to, given that the material itself
 * applies `pointSize.mul(screenDPR)` UNCONDITIONALLY (three@0.185.1,
 * PointsNodeMaterial.js's setupVertexSprite, L109) - that line is not gated by
 * `sizeAttenuation`, so it runs even with attenuation computed ourselves and
 * `sizeAttenuation = false`. Pre-dividing here is the only way to get an exact
 * device-pixel clamp out the far side: three offers no supported hook into its own
 * post-attenuation arithmetic (the `scale` uniform it uses is module-private and
 * unexported), so the whole formula is computed in our own sizeNode expression and this
 * division cancels the material's own multiply. The invariant this file (and
 * pointSizing.test.ts) pins: `pointSizeSizeNode(...) * pixelRatio ===
 * pointSizeDevicePx(...)`. */
export function pointSizeSizeNode(
  sizeMeters: number,
  viewZ: number,
  halfCanvasHeightLogicalPx: number,
  pixelRatio: number,
): number {
  return pointSizeDevicePx(sizeMeters, viewZ, halfCanvasHeightLogicalPx, pixelRatio) / pixelRatio
}
