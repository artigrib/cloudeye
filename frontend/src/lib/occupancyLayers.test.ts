import { describe, expect, it } from 'vitest'
import {
  FLOOR_SUPERSAMPLE,
  buildOccupancyLayerPixels,
  cellCentreWorld,
  cellForPixel,
  cellPixelOffset,
  pixelCentreWorld,
  pixelOffset,
  upscaleCellPixels,
  worldToCell,
  type LayerGridMeta,
} from './occupancyLayers'
import { OCC_FREE, OCC_OBSTACLE, OCC_UNKNOWN, toRgba } from './tokens'

const S = FLOOR_SUPERSAMPLE

// Deliberately not square, and with a non-zero, non-round origin in BOTH axes: a
// width/height mix-up, an ix/iz swap or a forgotten origin all survive a square grid at
// the origin, and all three are the mistakes this file exists to catch. These are the
// hero scene's own numbers, scaled down.
const grid: LayerGridMeta = {
  width: 4,
  height: 3,
  resolution: 0.05,
  origin_x: -1.4456723332432926,
  origin_z: -6.07137199104707,
}

function freeCells(): number[][] {
  return Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => 0))
}

function pixelAt(buf: Uint8ClampedArray, px: number, py: number): number[] {
  const o = pixelOffset(px, py, grid.width)
  return [buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]
}

describe('render coordinates against grid_meta', () => {
  it('puts a known cell at the pixels the UV mapping says it occupies', () => {
    // One obstacle cell, at an asymmetric position so ix and iz cannot be confused.
    const cells = freeCells()
    const IX = 2
    const IZ = 1
    cells[IX][IZ] = 1

    const buf = buildOccupancyLayerPixels({ cells, grid, layers: null, robotRadiusM: 0.1 })
    expect(buf.length).toBe(grid.width * S * grid.height * S * 4)

    const obstacle = [...toRgba(OCC_OBSTACLE)]
    const free = [...toRgba(OCC_FREE)]

    // Every pixel of that cell's own SxS block is the obstacle colour...
    for (let dy = 0; dy < S; dy++) {
      for (let dx = 0; dx < S; dx++) {
        expect(pixelAt(buf, IX * S + dx, IZ * S + dy)).toEqual(obstacle)
      }
    }
    // ...and no pixel outside it is. Checked exhaustively, not by sampling: an off-by-one
    // in either axis would still land inside a neighbouring block.
    for (let py = 0; py < grid.height * S; py++) {
      for (let px = 0; px < grid.width * S; px++) {
        const [ix, iz] = cellForPixel(px, py)
        expect(pixelAt(buf, px, py)).toEqual(ix === IX && iz === IZ ? obstacle : free)
      }
    }
  })

  it('agrees with cellPixelOffset about where a cell starts', () => {
    const cells = freeCells()
    cells[3][2] = 1
    const buf = buildOccupancyLayerPixels({ cells, grid, layers: null, robotRadiusM: 0.1 })
    const o = cellPixelOffset(3, 2, grid.width)
    expect([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]).toEqual([...toRgba(OCC_OBSTACLE)])
  })

  it('places cell centres at origin + (i + 0.5) * resolution, per grid_meta', () => {
    expect(cellCentreWorld(0, 0, grid)).toEqual([
      grid.origin_x + 0.5 * grid.resolution,
      grid.origin_z + 0.5 * grid.resolution,
    ])
    const [x, z] = cellCentreWorld(2, 1, grid)
    expect(x).toBeCloseTo(-1.4456723332432926 + 2.5 * 0.05, 12)
    expect(z).toBeCloseTo(-6.07137199104707 + 1.5 * 0.05, 12)
  })

  it('round-trips a cell centre back to its own cell', () => {
    for (let ix = 0; ix < grid.width; ix++) {
      for (let iz = 0; iz < grid.height; iz++) {
        const [x, z] = cellCentreWorld(ix, iz, grid)
        expect(worldToCell(x, z, grid)).toEqual([ix, iz])
      }
    }
  })

  it("puts a pixel's world centre inside its own cell", () => {
    for (let py = 0; py < grid.height * S; py++) {
      for (let px = 0; px < grid.width * S; px++) {
        const [x, z] = pixelCentreWorld(px, py, grid)
        expect(worldToCell(x, z, grid)).toEqual(cellForPixel(px, py))
      }
    }
  })

  it('does not clamp a point outside the grid onto its edge', () => {
    expect(worldToCell(grid.origin_x - 1, grid.origin_z, grid)[0]).toBeLessThan(0)
    expect(worldToCell(grid.origin_x, grid.origin_z + 99, grid)[1]).toBeGreaterThanOrEqual(grid.height)
  })
})

describe('the layers, one at a time', () => {
  it('draws unknown cells distinctly from free and from obstacle', () => {
    const cells = freeCells()
    cells[0][0] = 2
    const buf = buildOccupancyLayerPixels({ cells, grid, layers: null, robotRadiusM: 0.1 })
    expect(pixelAt(buf, 0, 0)).toEqual([...toRgba(OCC_UNKNOWN)])
    expect(pixelAt(buf, S, 0)).toEqual([...toRgba(OCC_FREE)])
  })

  it('ignores the PACK\'s unobserved mask - the hatch is the nav layer\'s', () => {
    // Two answers to "was this observed" used to exist side by side: the pack's, which is
    // matched to a scene by its grid and absent for most scenes, and the scene's own
    // layer. The pack lost. If it ever silently wins again, this fails.
    const cells = freeCells()
    const esdfM = Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => null as number | null))
    const packSaysUnobserved = Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => true))
    const buf = buildOccupancyLayerPixels({
      cells,
      grid,
      layers: { esdfM, unobserved: packSaysUnobserved },
      unobserved: null,
      robotRadiusM: 0.1,
    })
    for (let dy = 0; dy < S; dy++) {
      for (let dx = 0; dx < S; dx++) {
        expect(pixelAt(buf, dx, dy)).toEqual([...toRgba(OCC_FREE)])
      }
    }
  })

  it('tints a cell the pack measured, and leaves one it did not alone', () => {
    const cells = freeCells()
    const esdfM = Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => null as number | null))
    esdfM[1][1] = 0.02 // well under the robot's radius: reads as "will not fit"
    const buf = buildOccupancyLayerPixels({
      cells,
      grid,
      layers: { esdfM, unobserved: Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => false)) },
      robotRadiusM: 0.25,
    })
    expect(pixelAt(buf, 1 * S, 1 * S)).not.toEqual([...toRgba(OCC_FREE)])
    expect(pixelAt(buf, 0, 0)).toEqual([...toRgba(OCC_FREE)])
  })

  it('hatches an unobserved cell without filling it, so the cell below still reads', () => {
    const cells = freeCells()
    const unobserved = Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => false))
    unobserved[0][0] = true
    const esdfM = Array.from({ length: grid.width }, () => Array.from({ length: grid.height }, () => null as number | null))
    // The mask comes from the SCENE'S OWN nav layer (GET /map), not from the pack.
    const buf = buildOccupancyLayerPixels({
      cells,
      grid,
      layers: { esdfM, unobserved: [] },
      unobserved,
      robotRadiusM: 0.1,
    })

    const free = [...toRgba(OCC_FREE)]
    let stroke = 0
    let untouched = 0
    for (let dy = 0; dy < S; dy++) {
      for (let dx = 0; dx < S; dx++) {
        if (pixelAt(buf, dx, dy).join() === free.join()) untouched++
        else stroke++
      }
    }
    // A hatch, not a fill: some of the cell is struck through and some is not.
    expect(stroke).toBeGreaterThan(0)
    expect(untouched).toBeGreaterThan(0)
    expect(stroke + untouched).toBe(S * S)
  })

  it('draws only the first two layers when the scene has no pack', () => {
    const cells = freeCells()
    cells[1][2] = 1
    const withPack = buildOccupancyLayerPixels({ cells, grid, layers: null, robotRadiusM: 0.1 })
    expect(pixelAt(withPack, 1 * S, 2 * S)).toEqual([...toRgba(OCC_OBSTACLE)])
    expect(pixelAt(withPack, 0, 0)).toEqual([...toRgba(OCC_FREE)])
  })
})

describe('upscaleCellPixels', () => {
  it('block-replicates one pixel per cell into the same SxS blocks', () => {
    // A 2x1 grid: cell 0 red, cell 1 green, in the row-major (iz, ix) order every
    // one-pixel-per-cell renderer in this app produces.
    const base = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255])
    const out = upscaleCellPixels(base, 2, 1)
    expect(out.length).toBe(2 * S * 1 * S * 4)
    for (let dy = 0; dy < S; dy++) {
      for (let dx = 0; dx < S; dx++) {
        const left = pixelOffset(dx, dy, 2)
        const right = pixelOffset(S + dx, dy, 2)
        expect([out[left], out[left + 1], out[left + 2]]).toEqual([255, 0, 0])
        expect([out[right], out[right + 1], out[right + 2]]).toEqual([0, 255, 0])
      }
    }
  })
})
