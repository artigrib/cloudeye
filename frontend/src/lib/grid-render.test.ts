import { describe, expect, it } from 'vitest'
import { buildGridPixels, FREE_RGBA, OBSTACLE_RGBA, UNKNOWN_RGBA } from './grid-render'

describe('buildGridPixels', () => {
  // 2 wide (ix) x 3 tall (iz): free, obstacle, unknown down column 0; free down column 1.
  const cells = [
    [0, 1, 2],
    [0, 0, 0],
  ]

  it('writes each column of cells into the matching image row (row-major by iz)', () => {
    const pixels = buildGridPixels(cells, 2, 3)

    const at = (ix: number, iz: number) => {
      const o = (iz * 2 + ix) * 4
      return [pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]]
    }

    expect(at(0, 0)).toEqual([...FREE_RGBA])
    expect(at(0, 1)).toEqual([...OBSTACLE_RGBA])
    expect(at(0, 2)).toEqual([...UNKNOWN_RGBA])
    expect(at(1, 0)).toEqual([...FREE_RGBA])
    expect(at(1, 1)).toEqual([...FREE_RGBA])
  })

  it('produces a buffer of exactly width * height * 4 bytes', () => {
    expect(buildGridPixels(cells, 2, 3).length).toBe(2 * 3 * 4)
  })

  it('treats any value other than 1 or 2 as free (defensive default)', () => {
    const pixels = buildGridPixels([[99]], 1, 1)
    expect([pixels[0], pixels[1], pixels[2], pixels[3]]).toEqual([...FREE_RGBA])
  })
})
