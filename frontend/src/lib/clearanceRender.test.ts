import { describe, expect, it } from 'vitest'
import type { OccupancyCells } from './clearanceField'
import { buildClearancePixels, clearanceColorAt } from './clearanceRender'
import { CLEARANCE_TIGHT, DATA_UNREACHABLE, OCC_FREE, OCC_UNKNOWN, toRgba } from './tokens'

function emptyCells(width: number, height: number): OccupancyCells {
  return Array.from({ length: width }, () => Array(height).fill(0))
}

describe('clearanceColorAt', () => {
  const r = 0.25

  it('is the unreachable color well inside the radius', () => {
    expect(clearanceColorAt(0, r)).toEqual(toRgba({ ...DATA_UNREACHABLE, a: 0.22 }))
  })

  it('is the tight-fit color between r and 2r', () => {
    expect(clearanceColorAt(r * 1.5, r)).toEqual(toRgba({ ...CLEARANCE_TIGHT, a: 0.14 }))
  })

  it('is free space well beyond 2r', () => {
    expect(clearanceColorAt(r * 5, r)).toEqual(toRgba(OCC_FREE))
  })

  it('blends smoothly across the r threshold rather than stepping', () => {
    const justBelow = clearanceColorAt(r - r * 0.15, r)
    const justAbove = clearanceColorAt(r + r * 0.15, r)
    const atThreshold = clearanceColorAt(r, r)
    // Midpoint of the transition band should sit roughly between the two endpoint colors,
    // not equal to either one outright.
    expect(atThreshold).not.toEqual(justBelow)
    expect(atThreshold).not.toEqual(justAbove)
  })

  it('scales the transition band with the radius', () => {
    const smallR = clearanceColorAt(0.1, 0.1)
    const bigR = clearanceColorAt(0.1, 0.5)
    // 0.1m at r=0.1 sits right at the threshold (mid-transition); at r=0.5 it's deep
    // inside the unreachable zone - the two must differ.
    expect(smallR).not.toEqual(bigR)
  })
})

describe('buildClearancePixels', () => {
  it('produces a width*height*4 buffer in row-major (iz, then ix) order', () => {
    const width = 2
    const height = 3
    const clearance = new Float32Array(width * height).fill(10) // far from everything
    const pixels = buildClearancePixels(clearance, emptyCells(width, height), width, height, 0.25)
    expect(pixels.length).toBe(width * height * 4)
    const free = toRgba(OCC_FREE)
    for (let i = 0; i < width * height; i++) {
      expect(pixels[i * 4]).toBe(free[0])
      expect(pixels[i * 4 + 1]).toBe(free[1])
      expect(pixels[i * 4 + 2]).toBe(free[2])
      expect(pixels[i * 4 + 3]).toBe(free[3])
    }
  })

  it('places a near-zero clearance cell at the right pixel offset', () => {
    const width = 3
    const height = 2
    const clearance = new Float32Array(width * height).fill(10)
    // Cell ix=2, iz=1 -> flattened index ix*height+iz = 2*2+1 = 5.
    clearance[2 * height + 1] = 0
    const pixels = buildClearancePixels(clearance, emptyCells(width, height), width, height, 0.25)
    // Row-major (iz, then ix): pixel offset for (ix=2,iz=1) is (iz*width+ix)*4 = (1*3+2)*4 = 12.
    const offset = (1 * width + 2) * 4
    expect([pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]]).toEqual(
      Array.from(toRgba({ ...DATA_UNREACHABLE, a: 0.22 })),
    )
  })

  it('gives unknown cells the flat occ-unknown color instead of a clearance value', () => {
    const width = 2
    const height = 2
    const cells = emptyCells(width, height)
    cells[1][0] = 2 // unknown
    const clearance = new Float32Array(width * height).fill(0) // would otherwise be "unreachable"
    const pixels = buildClearancePixels(clearance, cells, width, height, 0.25)
    const offset = 1 * 4 // (ix=1, iz=0) -> (iz*width+ix)*4 = (0*width+1)*4
    expect([pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]]).toEqual(
      Array.from(toRgba(OCC_UNKNOWN)),
    )
  })
})
