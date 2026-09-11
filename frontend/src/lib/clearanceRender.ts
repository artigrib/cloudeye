// Renderer spec section 4.2: maps the clearance field (meters to nearest obstacle) to
// pixels, colored relative to the selected robot's radius - the field itself never
// changes with the robot; only this mapping does, which is why a platform switch
// recolors instantly instead of recomputing anything.

import { CLEARANCE_TIGHT, DATA_UNREACHABLE, OCC_FREE, OCC_UNKNOWN, toRgba, type OklchColor } from './tokens'
import type { OccupancyCells } from './clearanceField'

/** Half-width of the smoothing band around each threshold, as a fraction of the robot
 * radius - spec: "переходы плавные по 0.15·r", read as the band's half-width so the
 * full transition spans 0.3·r. */
const TRANSITION_FRACTION = 0.15
const UNREACHABLE_ALPHA = 0.22
const TIGHT_ALPHA = 0.14

type Rgba = readonly [number, number, number, number]

function withAlpha(token: OklchColor, a: number): Rgba {
  return toRgba({ ...token, a })
}

function lerpRgba(a: Rgba, b: Rgba, t: number): Rgba {
  const clamped = Math.min(1, Math.max(0, t))
  return [
    Math.round(a[0] + (b[0] - a[0]) * clamped),
    Math.round(a[1] + (b[1] - a[1]) * clamped),
    Math.round(a[2] + (b[2] - a[2]) * clamped),
    Math.round(a[3] + (b[3] - a[3]) * clamped),
  ]
}

/** d < r: --data-unreachable (won't fit). r <= d < 2r: a tight-fit amber warning tint.
 * d >= 2r: --occ-free. Smoothly blended across each threshold rather than a hard step,
 * per spec 4.2. */
export function clearanceColorAt(distanceM: number, radiusM: number): Rgba {
  const unreachable = withAlpha(DATA_UNREACHABLE, UNREACHABLE_ALPHA)
  const tight = withAlpha(CLEARANCE_TIGHT, TIGHT_ALPHA)
  const free = withAlpha(OCC_FREE, 1)
  const margin = Math.max(radiusM * TRANSITION_FRACTION, 1e-6)

  if (distanceM < radiusM - margin) return unreachable
  if (distanceM < radiusM + margin) return lerpRgba(unreachable, tight, (distanceM - (radiusM - margin)) / (2 * margin))
  if (distanceM < 2 * radiusM - margin) return tight
  if (distanceM < 2 * radiusM + margin) {
    return lerpRgba(tight, free, (distanceM - (2 * radiusM - margin)) / (2 * margin))
  }
  return free
}

/** `clearance` is the flattened `[ix * height + iz]` field from computeClearanceField -
 * same axis convention as buildGridPixels, output in the same row-major (iz, then ix)
 * layout ready for `new ImageData(buf, width, height)`. `cells` (the raw grid this
 * scene's clearance field was computed from) is used only to give `unknown` cells their
 * own flat color instead of a clearance value (spec 4.3: absence of data should look
 * like absence of data, not a third kind of surface) - a screen-space diagonal-hatch
 * texture over --canvas is the fuller spec treatment, not yet implemented; this flat
 * fill is the interim, still-correct-in-substance version. */
export function buildClearancePixels(
  clearance: Float32Array,
  cells: OccupancyCells,
  width: number,
  height: number,
  radiusM: number,
): Uint8ClampedArray<ArrayBuffer> {
  const pixels = new Uint8ClampedArray(new ArrayBuffer(width * height * 4))
  const unknown = toRgba(OCC_UNKNOWN)
  for (let ix = 0; ix < width; ix++) {
    const column = cells[ix]
    for (let iz = 0; iz < height; iz++) {
      const rgba = column?.[iz] === 2 ? unknown : clearanceColorAt(clearance[ix * height + iz], radiusM)
      const offset = (iz * width + ix) * 4
      pixels[offset] = rgba[0]
      pixels[offset + 1] = rgba[1]
      pixels[offset + 2] = rgba[2]
      pixels[offset + 3] = rgba[3]
    }
  }
  return pixels
}
