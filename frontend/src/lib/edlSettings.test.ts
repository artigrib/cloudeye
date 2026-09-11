import { describe, expect, it } from 'vitest'
import {
  DEFAULT_EDL_STRENGTH,
  EDL_RADIUS_MAX_PX,
  EDL_RADIUS_MIN_PX,
  edlRadiusPx,
  edlShade,
} from './edlSettings'

describe('edlRadiusPx', () => {
  it('scales with pixelRatio at the reference point size', () => {
    const at1x = edlRadiusPx(0.015, 0.015, 1)
    const at2x = edlRadiusPx(0.015, 0.015, 2)
    expect(at2x).toBeCloseTo(at1x * 2, 5)
  })

  it('scales with point size relative to the reference', () => {
    const base = edlRadiusPx(0.015, 0.015, 1)
    const doubled = edlRadiusPx(0.03, 0.015, 1)
    expect(doubled).toBeCloseTo(Math.min(EDL_RADIUS_MAX_PX, base * 2), 5)
  })

  it('never returns a radius smaller than the current splat would need, clamped to the min', () => {
    const tiny = edlRadiusPx(0.001, 0.015, 1)
    expect(tiny).toBeGreaterThanOrEqual(EDL_RADIUS_MIN_PX)
  })

  it('clamps at the max for a very large point size', () => {
    const huge = edlRadiusPx(0.06, 0.015, 2)
    expect(huge).toBe(EDL_RADIUS_MAX_PX)
  })

  it('falls back to the minimum radius for a degenerate zero reference size', () => {
    expect(edlRadiusPx(0.015, 0, 1)).toBe(EDL_RADIUS_MIN_PX)
  })
})

describe('edlShade', () => {
  it('is fully unshaded (1.0) when there is no depth discontinuity', () => {
    expect(edlShade(0, DEFAULT_EDL_STRENGTH, 1)).toBe(1)
  })

  it('darkens as the response grows, monotonically', () => {
    const shadeLow = edlShade(0.05, DEFAULT_EDL_STRENGTH, 1)
    const shadeHigh = edlShade(0.3, DEFAULT_EDL_STRENGTH, 1)
    expect(shadeHigh).toBeLessThan(shadeLow)
    expect(shadeLow).toBeLessThan(1)
  })

  it('reads roughly shade=0.35 for the silhouette case the default strength was tuned against', () => {
    // 2m foreground vs 2.6m background: log2(2.6/2) with half the 8-tap ring landing on
    // the far side -> response ~0.19 (see DEFAULT_EDL_STRENGTH's doc comment).
    const shade = edlShade(0.19, DEFAULT_EDL_STRENGTH, 1)
    expect(shade).toBeGreaterThan(0.25)
    expect(shade).toBeLessThan(0.45)
  })

  it('leaves a flat-wall-scale response essentially untouched', () => {
    const shade = edlShade(0.002, DEFAULT_EDL_STRENGTH, 1)
    expect(shade).toBeGreaterThan(0.98)
  })

  it('opacity=0 disables the effect regardless of response', () => {
    expect(edlShade(0.5, DEFAULT_EDL_STRENGTH, 0)).toBe(1)
  })

  it('opacity interpolates linearly between unshaded and fully shaded', () => {
    const full = edlShade(0.3, DEFAULT_EDL_STRENGTH, 1)
    const half = edlShade(0.3, DEFAULT_EDL_STRENGTH, 0.5)
    expect(half).toBeCloseTo((1 + full) / 2, 10)
  })
})
