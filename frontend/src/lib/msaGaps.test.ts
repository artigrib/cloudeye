import { describe, expect, it } from 'vitest'
import { gapPasses, type MsaGap } from './msaGaps'

function gap(overrides: Partial<MsaGap> = {}): MsaGap {
  return {
    a: 'wall_0',
    a_kind: 'wall',
    b: 'bed_0',
    b_kind: 'object',
    width_m: 0.5,
    measurement_point_xy: [0, 0],
    platforms: [],
    ...overrides,
  }
}

describe('gapPasses', () => {
  it('is false when no platform fits', () => {
    expect(
      gapPasses(
        gap({
          platforms: [
            { platform_id: 'burger', fits: false, required_clearance_m: 0.2 },
            { platform_id: 'husky', fits: false, required_clearance_m: 1.1 },
          ],
        }),
      ),
    ).toBe(false)
  })

  it('is true when at least one platform fits', () => {
    expect(
      gapPasses(
        gap({
          platforms: [
            { platform_id: 'burger', fits: true, required_clearance_m: 0.2 },
            { platform_id: 'husky', fits: false, required_clearance_m: 1.1 },
          ],
        }),
      ),
    ).toBe(true)
  })

  it('is false with no platforms registered at all', () => {
    expect(gapPasses(gap({ platforms: [] }))).toBe(false)
  })
})
