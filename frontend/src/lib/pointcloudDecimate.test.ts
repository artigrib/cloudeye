import { describe, expect, it } from 'vitest'
import { decimateIndices } from './pointcloudDecimate'

describe('decimateIndices', () => {
  it('keeps every index untouched when already under budget', () => {
    const result = decimateIndices(100, 400_000)
    expect(result.length).toBe(100)
    expect(Array.from(result)).toEqual(Array.from({ length: 100 }, (_, i) => i))
  })

  it('caps the output at maxPoints when over budget', () => {
    const result = decimateIndices(1_000_000, 400_000)
    expect(result.length).toBe(400_000)
  })

  it('samples uniformly across the whole range, not just the first N', () => {
    const pointCount = 1_000_000
    const maxPoints = 400_000
    const result = decimateIndices(pointCount, maxPoints)

    // Strictly increasing (no duplicates, no going backwards) - a real uniform sample.
    // (Single assertion after the scan, not one `expect()` per element - 400k
    // individual assertions is slow enough to be a self-inflicted test timeout.)
    let strictlyIncreasing = true
    for (let i = 1; i < result.length; i++) {
      if (result[i] <= result[i - 1]) {
        strictlyIncreasing = false
        break
      }
    }
    expect(strictlyIncreasing).toBe(true)

    // Covers the full index range, not a truncated prefix.
    expect(result[0]).toBe(0)
    expect(result[result.length - 1]).toBeGreaterThan(pointCount * 0.99)

    // Every decile of the original range contributed at least one kept index.
    const decileHit = new Array(10).fill(false)
    for (const idx of result) {
      const decile = Math.min(9, Math.floor((idx / pointCount) * 10))
      decileHit[decile] = true
    }
    expect(decileHit).toEqual(new Array(10).fill(true))
  })

  it('returns an empty array for an empty cloud', () => {
    expect(decimateIndices(0, 400_000).length).toBe(0)
  })

  it('treats a non-positive budget as "no decimation" rather than dropping everything', () => {
    expect(decimateIndices(50, 0).length).toBe(50)
  })
})
