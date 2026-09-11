// Eye-Dome Lighting tuning - pure math, no WebGL. See scene3dEdl.ts for the pass that
// actually renders this. Kept separate and pure so the falloff curve and radius formula
// are pinned by a test instead of by a comment, matching this repo's convention (every
// other frontend test is pure logic with no WebGL context).

/** log2-depth neighbor-response falloff coefficient. Derived (not guessed) from two
 * target cases against the 8-tap mean response computed in the composite shader:
 * a 2m/2.6m silhouette step should read at roughly shade=0.35, a ~0.3%-per-pixel flat
 * wall should read at roughly shade=0.99 (i.e. untouched). exp(-k * 0.19) = 0.35 gives
 * k ~= 5.5; 6.0 rounds that up slightly and leaves the flat-wall case at exp(-0.011),
 * still ~0.99. */
export const DEFAULT_EDL_STRENGTH = 6.0
/** Blends the computed shade against "no shading" (1.0) - the one knob a UI slider
 * should drive. 1 = full effect. */
export const DEFAULT_EDL_OPACITY = 1.0

/** Neighbor-tap radius in *device* pixels at the reference point size, before scaling by
 * pixelRatio and the current/reference point-size ratio - see edlRadiusPx. */
export const EDL_RADIUS_BASE_CSS_PX = 1.4
export const EDL_RADIUS_MIN_PX = 1.0
export const EDL_RADIUS_MAX_PX = 8.0

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

/** Neighbor-tap radius, in device pixels, for the EDL composite pass. Must scale with
 * the current point size: if the radius is smaller than the on-screen splat, every tap
 * lands inside the same splat and the response is always 0 - EDL silently does nothing.
 * Also scales with pixelRatio so the effect reads the same apparent thickness on a
 * retina display as on a standard one (uSize is world metres, not device pixels - this
 * function's own pixelRatio multiplication is what converts it, see scene3dPoints.ts's
 * sizeNode for the point-cloud material's equivalent conversion). `referencePointSize`
 * is the point size the base radius was tuned
 * against (SceneMap3D's DEFAULT_POINT_SIZE). */
export function edlRadiusPx(pointSizeWorld: number, referencePointSize: number, pixelRatio: number): number {
  if (referencePointSize <= 0) return EDL_RADIUS_MIN_PX
  const scaled = EDL_RADIUS_BASE_CSS_PX * pixelRatio * (pointSizeWorld / referencePointSize)
  return clamp(scaled, EDL_RADIUS_MIN_PX, EDL_RADIUS_MAX_PX)
}

/** JS mirror of the composite fragment shader's final blend
 * (`mix(1.0, exp(-strength * response), opacity)`) - lets the tuning in
 * DEFAULT_EDL_STRENGTH's doc comment be pinned by a test rather than eyeballed in the
 * browser. `response` is the mean log2-depth step from the 8-tap neighbor comparison;
 * 0 means no depth discontinuity (nothing to shade). */
export function edlShade(response: number, strength: number, opacity: number): number {
  const shaded = Math.exp(-strength * Math.max(0, response))
  return 1 - opacity * (1 - shaded)
}
