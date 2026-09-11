import { describe, expect, it } from 'vitest'
import { formatMetres, formatNumber, formatSeconds, formatSigned, NO_VALUE } from './formatMeasure'

describe('formatMeasure', () => {
  it('renders an em dash instead of calling toFixed on a missing value', () => {
    expect(formatMetres(null)).toBe(NO_VALUE)
    expect(formatMetres(undefined)).toBe(NO_VALUE)
    expect(formatMetres(Number.NaN)).toBe(NO_VALUE)
    expect(formatSeconds(null)).toBe(NO_VALUE)
    expect(formatNumber(null)).toBe(NO_VALUE)
  })

  it('formats real measurements with their unit', () => {
    // Husky A200's own radius_m from app/robots.py - the platform whose null corridor
    // width crashed the panel.
    expect(formatMetres(0.5528)).toBe('0.55 m')
    expect(formatMetres(0)).toBe('0.00 m')
    expect(formatSeconds(4.68)).toBe('4.7s')
    expect(formatNumber(2.4029)).toBe('2.40')
  })

  it('keeps an infinite measurement rather than hiding it', () => {
    expect(formatMetres(Number.POSITIVE_INFINITY)).toBe('Infinity m')
  })
})

// A sign that only survives rounding. `(-0.001).toFixed(2)` is "-0.00", and so is
// `(-0).toFixed(2)`: both print a minus in front of a zero, which reads as a measured
// negative. The scene's start position showed "(0.00, -0.00)" for a point at the origin -
// one coordinate apparently signed, the other not, for the same zero.
describe('a number that rounds to zero has no sign', () => {
  it('formatNumber', () => {
    expect(formatNumber(-0)).toBe('0.00')
    expect(formatNumber(-0.001)).toBe('0.00')
    expect(formatNumber(-0.006)).toBe('-0.01')   // rounds to a real negative, keeps it
    expect(formatNumber(0.001)).toBe('0.00')
  })

  it('formatMetres, at any precision', () => {
    expect(formatMetres(-0.001)).toBe('0.00 m')
    expect(formatMetres(-0.04, 1)).toBe('0.0 m')
    expect(formatMetres(-0.06, 1)).toBe('-0.1 m')
  })

  it('formatSeconds', () => {
    expect(formatSeconds(-0.04)).toBe('0.0s')
  })

  it('formatSigned marks direction only where there is one', () => {
    expect(formatSigned(0.8)).toBe('+0.80')
    expect(formatSigned(-0.25)).toBe('-0.25')
    expect(formatSigned(-0.001)).toBe('0.00')
    expect(formatSigned(0)).toBe('0.00')
    expect(formatSigned(null)).toBe(NO_VALUE)
  })
})
