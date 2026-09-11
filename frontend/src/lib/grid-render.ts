// Occupancy-grid cell values -> RGBA pixel buffer for the raw "Occupancy" render mode
// (spec 6: one cell one pixel, no smoothing - deliberately, an honest raw view for
// anyone who wants to see the data without interpretation). Kept separate from any DOM
// canvas so the color mapping is unit-testable without a jsdom canvas mock.

import { OCC_FREE, OCC_OBSTACLE, OCC_UNKNOWN, toRgba } from './tokens'

export const FREE_RGBA = toRgba(OCC_FREE)
export const OBSTACLE_RGBA = toRgba(OCC_OBSTACLE)
export const UNKNOWN_RGBA = toRgba(OCC_UNKNOWN)

/**
 * cells is [ix][iz] (axis 0 = X, axis 1 = Z, per GET /map). Produces a width*height*4
 * buffer in row-major (iz, then ix) order, i.e. ready for `new ImageData(buf, width, height)`.
 */
export function buildGridPixels(
  cells: number[][],
  width: number,
  height: number,
): Uint8ClampedArray<ArrayBuffer> {
  const pixels = new Uint8ClampedArray(new ArrayBuffer(width * height * 4))
  for (let ix = 0; ix < width; ix++) {
    const column = cells[ix] ?? []
    for (let iz = 0; iz < height; iz++) {
      const value = column[iz]
      const rgba = value === 1 ? OBSTACLE_RGBA : value === 2 ? UNKNOWN_RGBA : FREE_RGBA
      const offset = (iz * width + ix) * 4
      pixels[offset] = rgba[0]
      pixels[offset + 1] = rgba[1]
      pixels[offset + 2] = rgba[2]
      pixels[offset + 3] = rgba[3]
    }
  }
  return pixels
}
