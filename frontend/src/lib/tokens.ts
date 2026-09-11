// Single source of truth for the data-semantic colors that can't be expressed as
// Tailwind classes: three.js material colors and the occupancy-grid rasterizer
// (lib/grid-render.ts). Both used to hardcode their own hex/0x values, independent of
// index.css and of each other. The numeric triples here MUST match index.css's :root
// OKLCH values - tokens.test.ts parses index.css and asserts that.
//
// three's THREE.Color.setStyle() doesn't parse oklch() (as of r185), and round-tripping
// through a <canvas> to resolve a CSS color string doesn't work in the jsdom tests this
// module needs to run under - so colors are declared here as OKLCH triples and converted
// in JS, not read from the DOM.

export interface OklchColor {
  /** Lightness, 0-1 */
  l: number
  /** Chroma, roughly 0-0.4 */
  c: number
  /** Hue, degrees */
  h: number
  /** Alpha, 0-1. Defaults to 1 when omitted. */
  a?: number
}

function srgbTransfer(linear: number): number {
  const clamped = Math.min(1, Math.max(0, linear))
  return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055
}

/** OKLCH -> OKLab -> linear sRGB -> gamma-encoded sRGB. Reference: Björn Ottosson's
 * "A perceptual color space for image processing" (the OKLab/OKLCH original writeup). */
export function oklchToSrgb({ l, c, h }: OklchColor): [number, number, number] {
  const hRad = (h * Math.PI) / 180
  const a = c * Math.cos(hRad)
  const b = c * Math.sin(hRad)

  const l_ = l + 0.3963377774 * a + 0.2158037573 * b
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b
  const s_ = l - 0.0894841775 * a - 1.291485548 * b

  const l3 = l_ * l_ * l_
  const m3 = m_ * m_ * m_
  const s3 = s_ * s_ * s_

  const rLin = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
  const gLin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
  const bLin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3

  return [srgbTransfer(rLin), srgbTransfer(gLin), srgbTransfer(bLin)]
}

/** [r, g, b, a], each 0-255, ready for a Uint8Array/ImageData buffer. */
export function toRgba(token: OklchColor): [number, number, number, number] {
  const [r, g, b] = oklchToSrgb(token)
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), Math.round((token.a ?? 1) * 255)]
}

/** Packed 0xRRGGBB, the form THREE.Color's numeric constructor and material color
 * setters expect. */
export function toHex(token: OklchColor): number {
  const [r, g, b] = oklchToSrgb(token)
  return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255)
}

// Data-semantic tokens - keep these triples byte-for-byte identical to index.css's
// :root block (see tokens.test.ts).
export const DATA_OBJECT: OklchColor = { l: 0.8, c: 0.16, h: 85 }
export const DATA_TRAJECTORY: OklchColor = { l: 0.68, c: 0.25, h: 340 }
export const DATA_ROBOT: OklchColor = { l: 0.8, c: 0, h: 0 }
export const DATA_UNREACHABLE: OklchColor = { l: 0.64, c: 0.19, h: 25 }

export const H_0: OklchColor = { l: 0.32, c: 0.16, h: 300 }
export const H_1: OklchColor = { l: 0.45, c: 0.17, h: 288 }
export const H_2: OklchColor = { l: 0.58, c: 0.13, h: 240 }
export const H_3: OklchColor = { l: 0.72, c: 0.13, h: 195 }
export const H_4: OklchColor = { l: 0.85, c: 0.16, h: 155 }
export const HEIGHT_RAMP: readonly OklchColor[] = [H_0, H_1, H_2, H_3, H_4]

export const OCC_FREE: OklchColor = { l: 0.24, c: 0, h: 0 }
export const OCC_OBSTACLE: OklchColor = { l: 0.62, c: 0, h: 0 }
export const OCC_UNKNOWN: OklchColor = { l: 0.16, c: 0, h: 0 }

/** The clearance field's "tight fit" zone color (spec: r <= d < 2r), the one data color
 * that isn't a :root token - it's a one-off warning tint local to the clearance shader. */
export const CLEARANCE_TIGHT: OklchColor = { l: 0.7, c: 0.1, h: 60 }

export const ACCENT: OklchColor = { l: 0.7, c: 0.16, h: 162 }

// Neutral surface/text/border ramp - needed wherever a three.js material has to match a
// UI-chrome color exactly (e.g. the metric grid, a selection outline) rather than a
// data-semantic one.
export const BG: OklchColor = { l: 0.13, c: 0, h: 0 }
export const SURFACE: OklchColor = { l: 0.17, c: 0, h: 0 }
export const SURFACE_RAISED: OklchColor = { l: 0.21, c: 0, h: 0 }
export const CANVAS: OklchColor = { l: 0.09, c: 0, h: 0 }
export const FG: OklchColor = { l: 0.97, c: 0, h: 0 }
export const SUBTLE: OklchColor = { l: 0.76, c: 0, h: 0 }
export const MUTED: OklchColor = { l: 0.56, c: 0, h: 0 }
export const FAINT: OklchColor = { l: 0.42, c: 0, h: 0 }
export const BORDER: OklchColor = { l: 0.26, c: 0, h: 0 }
export const BORDER_STRONG: OklchColor = { l: 0.36, c: 0, h: 0 }
