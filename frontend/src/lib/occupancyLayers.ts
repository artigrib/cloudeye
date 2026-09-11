// The 2D map's "Occupancy" render pass: what the reconstruction actually observed, drawn
// on the scene's own grid.
//
// Four layers, in this order, each from a named source:
//
//   floor fill        the scene's occupancy grid   free / unknown
//   obstacle cells    the scene's occupancy grid   the 0.1-0.5 m BAND mask (cells === 1)
//   ESDF colormap     the nvblox layer pack        signed distance to the nearest obstacle
//   unobserved hatch  THE SCENE'S OWN NAV LAYER    cells never observed at all
//
// The hatch used to come from the nvblox pack too, and that was a second answer to "was
// this observed" living beside the first. The pack is matched to a scene BY ITS GRID, so
// most scenes have none and got no hatch at all; the nav layer ships `unobserved_mask.npy`
// for every scene that has a layer, and `GET /map` now serves it alongside the cells. The
// pack is down to one job here - the ESDF tint - which is the only thing it is the sole
// source of.
//
// The obstacle cells come from the scene's own band mask and NOT from the pack, and that
// is deliberate. The pack ships one horizontal ESDF slice at 0.3 m; measured on the hero
// grid, `esdf <= 0` marks 1753 cells where the band marks 2857, and the slice's own ESDF
// saturates at 0.342 m - a robot wider than that scores no free cell on it anywhere. The
// slice is drawn AS a slice, over the band, so the two can be compared; neither is
// presented as the other. (HANDOFF section 6, item 10.)
//
// The ESDF is coloured with `clearanceColorAt`, the same function the Clearance mode
// uses, because it is the same quantity in the same units - metres to the nearest
// obstacle - and it should mean the same thing to the eye. No new color is introduced by
// this module; every value comes from lib/tokens.ts.

import { buildClearancePixels } from './clearanceRender'
import { clearanceColorAt } from './clearanceRender'
import { OCC_FREE, OCC_OBSTACLE, OCC_UNKNOWN, toRgba } from './tokens'

/** Pixels per grid cell along each axis.
 *
 * The floor texture used to be exactly one pixel per cell, which cannot express anything
 * INSIDE a cell - and "unobserved" has to read as hatched rather than as another flat
 * colour, or it competes with the data it is qualifying. Three is the smallest factor
 * that draws a diagonal line through a cell and still leaves two pixels of the cell's own
 * colour showing. Every mode renders at this scale so the texture's dimensions never
 * change with the mode; NearestFilter keeps the blocky look the 2D view is designed
 * around (see scene3dFloor.buildFloorTexture). */
export const FLOOR_SUPERSAMPLE = 3

const FREE_RGBA = toRgba(OCC_FREE)
const OBSTACLE_RGBA = toRgba(OCC_OBSTACLE)
const UNKNOWN_RGBA = toRgba(OCC_UNKNOWN)
/** The hatch stroke: the "never observed" colour, drawn at full strength over whatever
 * the cell was painted, so a hatched cell is unmistakably qualified rather than tinted. */
const HATCH_RGBA = toRgba({ ...OCC_UNKNOWN, l: OCC_UNKNOWN.l + 0.1 })

/** The grid every layer here is sampled on - the scene's own occupancy grid, which is
 * also the grid the layer pack is only ever served on (see the backend's
 * scene_layers.find_pack_for_grid). */
export interface LayerGridMeta {
  width: number
  height: number
  resolution: number
  origin_x: number
  origin_z: number
}

/** The pack, as GET /scenes/{id}/layers returns it. `esdfM[ix][iz]` is null where the
 * cell was never observed. */
export interface LayerPackData {
  esdfM: (number | null)[][]
  unobserved: boolean[][]
}

// --- coordinates ---------------------------------------------------------------------
//
// One convention, stated once. The floor texture's UVs put pixel (0, 0) at world
// (origin_x, origin_z) with +U along +X and +V along +Z (scene3dFloor.buildFloorGeometry),
// and the buffer is row-major in (py, px). At FLOOR_SUPERSAMPLE = S, cell [ix][iz] owns
// the SxS block of pixels whose top-left is (ix*S, iz*S).

/** Byte offset of pixel (px, py) in a width*S x height*S RGBA buffer. */
export function pixelOffset(px: number, py: number, gridWidth: number, ss = FLOOR_SUPERSAMPLE): number {
  return (py * gridWidth * ss + px) * 4
}

/** Byte offset of the TOP-LEFT pixel of cell [ix][iz]. */
export function cellPixelOffset(ix: number, iz: number, gridWidth: number, ss = FLOOR_SUPERSAMPLE): number {
  return pixelOffset(ix * ss, iz * ss, gridWidth, ss)
}

/** Which cell a pixel belongs to. */
export function cellForPixel(px: number, py: number, ss = FLOOR_SUPERSAMPLE): [number, number] {
  return [Math.floor(px / ss), Math.floor(py / ss)]
}

/** World (x, z) at the centre of a grid cell - the same rule the backend's
 * `cell_to_world` uses, so a cell means the same place on both sides. */
export function cellCentreWorld(ix: number, iz: number, grid: LayerGridMeta): [number, number] {
  return [
    grid.origin_x + (ix + 0.5) * grid.resolution,
    grid.origin_z + (iz + 0.5) * grid.resolution,
  ]
}

/** World (x, z) at the centre of a texture PIXEL - a third of a cell at S = 3. */
export function pixelCentreWorld(
  px: number,
  py: number,
  grid: LayerGridMeta,
  ss = FLOOR_SUPERSAMPLE,
): [number, number] {
  const step = grid.resolution / ss
  return [grid.origin_x + (px + 0.5) * step, grid.origin_z + (py + 0.5) * step]
}

/** Which cell a world point falls in. Out-of-grid points return out-of-range indices
 * rather than being clamped - a clamp would silently draw an off-map thing on the edge. */
export function worldToCell(x: number, z: number, grid: LayerGridMeta): [number, number] {
  return [
    Math.floor((x - grid.origin_x) / grid.resolution),
    Math.floor((z - grid.origin_z) / grid.resolution),
  ]
}

// --- the render pass -------------------------------------------------------------------

function paintCell(
  out: Uint8ClampedArray,
  ix: number,
  iz: number,
  gridWidth: number,
  ss: number,
  rgba: readonly [number, number, number, number],
): void {
  for (let dy = 0; dy < ss; dy++) {
    for (let dx = 0; dx < ss; dx++) {
      const offset = pixelOffset(ix * ss + dx, iz * ss + dy, gridWidth, ss)
      out[offset] = rgba[0]
      out[offset + 1] = rgba[1]
      out[offset + 2] = rgba[2]
      out[offset + 3] = rgba[3]
    }
  }
}

/** Alpha-composite `src` over whatever is already at that pixel. */
function blendPixel(out: Uint8ClampedArray, offset: number, src: readonly [number, number, number, number]): void {
  const a = src[3] / 255
  if (a <= 0) return
  out[offset] = Math.round(out[offset] * (1 - a) + src[0] * a)
  out[offset + 1] = Math.round(out[offset + 1] * (1 - a) + src[1] * a)
  out[offset + 2] = Math.round(out[offset + 2] * (1 - a) + src[2] * a)
  out[offset + 3] = 255
}

export interface OccupancyLayerOptions {
  /** `cells[ix][iz]`: 0 free, 1 obstacle (the band mask), 2 unknown. */
  cells: number[][]
  grid: LayerGridMeta
  /** The nvblox pack, or null when this scene has none - then the ESDF tint is skipped,
   * and the caller says so rather than the map pretending to a completeness it does not
   * have. It is no longer the hatch's source; see `unobserved`. */
  layers: LayerPackData | null
  /** `unobserved[ix][iz]` from the scene's OWN navigation layer (GET /map), on the same
   * grid as `cells`. The hatch source. Optional so a caller with an older grid response
   * simply draws no hatch rather than throwing. */
  unobserved?: boolean[][] | null
  /** The selected robot's radius, for the ESDF colormap's fits / tight / blocked bands. */
  robotRadiusM: number
  ss?: number
}

/** The composed "Occupancy" pass, as a (width*S)x(height*S) RGBA buffer in row-major
 * (py, px) order - ready for `new ImageData(buf, width*S, height*S)`. */
export function buildOccupancyLayerPixels({
  cells,
  grid,
  layers,
  unobserved = null,
  robotRadiusM,
  ss = FLOOR_SUPERSAMPLE,
}: OccupancyLayerOptions): Uint8ClampedArray<ArrayBuffer> {
  const w = grid.width * ss
  const h = grid.height * ss
  const out = new Uint8ClampedArray(new ArrayBuffer(w * h * 4))

  for (let ix = 0; ix < grid.width; ix++) {
    const column = cells[ix] ?? []
    for (let iz = 0; iz < grid.height; iz++) {
      // 1 + 2: floor fill, and the band mask's obstacle cells over it.
      const value = column[iz]
      paintCell(out, ix, iz, grid.width, ss,
        value === 1 ? OBSTACLE_RGBA : value === 2 ? UNKNOWN_RGBA : FREE_RGBA)

      // 3: the pack's measured distance field, over the cells it has a value for.
      const esdf = layers?.esdfM[ix]?.[iz]
      if (esdf !== null && esdf !== undefined && Number.isFinite(esdf)) {
        const tint = clearanceColorAt(esdf, robotRadiusM)
        for (let dy = 0; dy < ss; dy++) {
          for (let dx = 0; dx < ss; dx++) {
            blendPixel(out, pixelOffset(ix * ss + dx, iz * ss + dy, grid.width, ss), tint)
          }
        }
      }

      // 4: hatch every cell THE SCENE'S OWN LAYER calls unobserved. The stripe runs along
      // the anti-diagonal, so adjacent hatched cells form one continuous 45-degree pattern
      // rather than a grid of disconnected ticks.
      if (unobserved?.[ix]?.[iz]) {
        for (let dy = 0; dy < ss; dy++) {
          for (let dx = 0; dx < ss; dx++) {
            const px = ix * ss + dx
            const py = iz * ss + dy
            if ((px + py) % ss !== 0) continue
            const offset = pixelOffset(px, py, grid.width, ss)
            out[offset] = HATCH_RGBA[0]
            out[offset + 1] = HATCH_RGBA[1]
            out[offset + 2] = HATCH_RGBA[2]
            out[offset + 3] = 255
          }
        }
      }
    }
  }
  return out
}

/** Any other render mode, at the same supersampled scale, by block-replicating the
 * one-pixel-per-cell buffer that mode already produces. Keeping every mode on one
 * texture size means switching modes repaints the texture instead of rebuilding it. */
export function upscaleCellPixels(
  base: Uint8ClampedArray,
  width: number,
  height: number,
  ss = FLOOR_SUPERSAMPLE,
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(new ArrayBuffer(width * ss * height * ss * 4))
  for (let iz = 0; iz < height; iz++) {
    for (let ix = 0; ix < width; ix++) {
      const src = (iz * width + ix) * 4
      paintCell(out, ix, iz, width, ss, [base[src], base[src + 1], base[src + 2], base[src + 3]])
    }
  }
  return out
}

/** Re-exported so SceneMap3D has one import for the floor's pixel sources. */
export { buildClearancePixels }
