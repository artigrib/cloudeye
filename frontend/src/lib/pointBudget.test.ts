import { describe, expect, it } from 'vitest'
import {
  createFrameBudget,
  distanceLodBudget,
  FALLBACK_POINT_BUDGET,
  initialPointBudget,
  loadPointBudget,
  MAX_POINT_BUDGET,
  MIN_POINT_BUDGET,
  observeFrame,
  pendingBudget,
  pointBudgetCacheIsValid,
  savePointBudget,
  type DeviceBudgetInput,
} from './pointBudget'

const desktopInput: DeviceBudgetInput = {
  coarsePointer: false,
  hardwareConcurrency: 16,
  deviceMemoryGb: 8,
  drawingBufferPixels: 1920 * 1080,
  softwareRenderer: false,
  saveData: false,
}

describe('initialPointBudget', () => {
  it('gives a capable desktop the largest budget', () => {
    const budget = initialPointBudget(desktopInput)
    expect(budget).toBeGreaterThan(FALLBACK_POINT_BUDGET)
  })

  it('gives a coarse-pointer (touch) device a smaller budget than desktop', () => {
    const touch = initialPointBudget({ ...desktopInput, coarsePointer: true })
    const desktop = initialPointBudget(desktopInput)
    expect(touch).toBeLessThan(desktop)
  })

  it('reduces the budget for low device memory', () => {
    const low = initialPointBudget({ ...desktopInput, deviceMemoryGb: 2 })
    const high = initialPointBudget({ ...desktopInput, deviceMemoryGb: 8 })
    expect(low).toBeLessThan(high)
  })

  it('reduces the budget for few CPU cores', () => {
    const low = initialPointBudget({ ...desktopInput, hardwareConcurrency: 2 })
    const high = initialPointBudget({ ...desktopInput, hardwareConcurrency: 16 })
    expect(low).toBeLessThan(high)
  })

  it('reduces the budget for a denser drawing buffer at the same everything else', () => {
    const retina = initialPointBudget({ ...desktopInput, drawingBufferPixels: 3840 * 2160 })
    const standard = initialPointBudget(desktopInput)
    expect(retina).toBeLessThan(standard)
  })

  it('treats null memory/cores as mildly conservative, not as "unknown = max"', () => {
    const unknown = initialPointBudget({ ...desktopInput, deviceMemoryGb: null, hardwareConcurrency: null })
    const known = initialPointBudget(desktopInput)
    expect(unknown).toBeLessThan(known)
    expect(unknown).toBeGreaterThan(MIN_POINT_BUDGET)
  })

  it('forces a hard floor for a software rasterizer regardless of other signals', () => {
    const budget = initialPointBudget({ ...desktopInput, softwareRenderer: true })
    expect(budget).toBeLessThanOrEqual(100_000)
  })

  it('halves the budget when the user has requested reduced data usage', () => {
    // Deliberately NOT desktopInput: its prior is 1,200,000, above MAX_POINT_BUDGET
    // (1,000,000), so the un-halved side is clamped and the halving stops being visible
    // in the result - the ratio would read 0.6 rather than 0.5 through no fault of the
    // saveData factor. 4 GB drops the prior to 840,000, which is under the ceiling AND
    // halves to a whole multiple of 10,000, so roundToNearest10k doesn't blur the
    // comparison either (350,000 would: it halves to 175,000 and rounds to 180,000).
    const base: DeviceBudgetInput = { ...desktopInput, deviceMemoryGb: 4 }
    const withSaveData = initialPointBudget({ ...base, saveData: true })
    const without = initialPointBudget(base)
    expect(without).toBeLessThan(MAX_POINT_BUDGET)
    expect(withSaveData).toBeCloseTo(without / 2, -3)
  })

  it('never returns outside [MIN_POINT_BUDGET, MAX_POINT_BUDGET]', () => {
    const extreme = initialPointBudget({
      coarsePointer: true,
      hardwareConcurrency: 1,
      deviceMemoryGb: 0.25,
      drawingBufferPixels: 7680 * 4320,
      softwareRenderer: false,
      saveData: true,
    })
    expect(extreme).toBeGreaterThanOrEqual(MIN_POINT_BUDGET)
    expect(extreme).toBeLessThanOrEqual(MAX_POINT_BUDGET)
  })
})

describe('observeFrame / pendingBudget', () => {
  it('does not downshift while still collecting samples', () => {
    let state = createFrameBudget(400_000)
    for (let i = 0; i < 10; i++) state = observeFrame(state, 40)
    expect(state.downshifts).toBe(0)
    expect(state.budget).toBe(400_000)
  })

  it('downshifts once enough consistently-slow frames land, dropping warmup samples', () => {
    let state = createFrameBudget(400_000)
    // 3 warmup frames (a slow shader-compile spike) should not count toward the median.
    for (let i = 0; i < 3; i++) state = observeFrame(state, 500)
    for (let i = 0; i < 20; i++) {
      const prev = state
      state = observeFrame(state, 40) // consistently slow (> SLOW_FRAME_MS)
      const pending = pendingBudget(prev, state)
      if (pending !== null) {
        expect(pending).toBe(200_000)
        expect(state.downshifts).toBe(1)
        return
      }
    }
    throw new Error('expected a downshift to have occurred')
  })

  it('settles without downshifting when frames are consistently fast', () => {
    let state = createFrameBudget(400_000)
    for (let i = 0; i < 25; i++) state = observeFrame(state, 10)
    expect(state.downshifts).toBe(0)
    expect(state.budget).toBe(400_000)
    expect(state.settled).toBe(true)
  })

  it('never upshifts after settling fast', () => {
    let state = createFrameBudget(100_000)
    for (let i = 0; i < 25; i++) state = observeFrame(state, 5)
    expect(state.budget).toBe(100_000)
  })

  it('caps downshifts at MAX_DOWNSHIFTS (stays at a floor rather than spiraling to zero)', () => {
    let state = createFrameBudget(200_000)
    for (let round = 0; round < 5; round++) {
      for (let i = 0; i < 25; i++) state = observeFrame(state, 100)
    }
    expect(state.downshifts).toBeLessThanOrEqual(2)
    expect(state.budget).toBeGreaterThanOrEqual(MIN_POINT_BUDGET)
  })

  it('stops producing new pendingBudget values once settled', () => {
    let state = createFrameBudget(400_000)
    for (let i = 0; i < 25; i++) state = observeFrame(state, 10)
    const prev = state
    const next = observeFrame(state, 100)
    expect(pendingBudget(prev, next)).toBeNull()
  })
})

describe('distanceLodBudget', () => {
  it('keeps the full budget while the camera is still near the room', () => {
    expect(distanceLodBudget(400_000, 5, 5)).toBe(400_000) // 1x diagonal away
    expect(distanceLodBudget(400_000, 7, 5)).toBe(400_000) // 1.4x, still under the 1.5x near ratio
  })

  it('reduces the budget once the camera pulls back past the near ratio', () => {
    const near = distanceLodBudget(400_000, 7.5, 5) // 1.5x
    const far = distanceLodBudget(400_000, 15, 5) // 3x
    expect(far).toBeLessThan(near)
    expect(near).toBe(400_000)
  })

  it('floors at a fraction of the base budget once fully zoomed out, never zero', () => {
    const veryFar = distanceLodBudget(400_000, 100, 5) // 20x diagonal away
    expect(veryFar).toBeGreaterThanOrEqual(MIN_POINT_BUDGET)
    expect(veryFar).toBeGreaterThan(400_000 * 0.3)
    expect(veryFar).toBeLessThan(400_000)
  })

  it('never exceeds baseBudget regardless of distance', () => {
    expect(distanceLodBudget(400_000, 0, 5)).toBeLessThanOrEqual(400_000)
    expect(distanceLodBudget(400_000, 1000, 5)).toBeLessThanOrEqual(400_000)
  })

  it('is a no-op when roomDiagonal is not known yet', () => {
    expect(distanceLodBudget(400_000, 50, 0)).toBe(400_000)
    expect(distanceLodBudget(400_000, 50, -1)).toBe(400_000)
  })

  it('is a no-op for a non-finite camera distance', () => {
    expect(distanceLodBudget(400_000, NaN, 5)).toBe(400_000)
    expect(distanceLodBudget(400_000, Infinity, 5)).toBe(400_000)
  })
})

describe('point budget cache', () => {
  it('is valid when the drawing buffer size is unchanged', () => {
    expect(pointBudgetCacheIsValid({ budget: 400_000, drawingBufferPixels: 1000 }, 1000)).toBe(true)
  })

  it('is valid within a 1.5x drift in either direction', () => {
    expect(pointBudgetCacheIsValid({ budget: 400_000, drawingBufferPixels: 1000 }, 1400)).toBe(true)
    expect(pointBudgetCacheIsValid({ budget: 400_000, drawingBufferPixels: 1000 }, 700)).toBe(true)
  })

  it('is invalid past a 1.5x drift (e.g. moved to a much higher-res display)', () => {
    expect(pointBudgetCacheIsValid({ budget: 400_000, drawingBufferPixels: 1000 }, 3000)).toBe(false)
  })

  it('round-trips through localStorage', () => {
    localStorage.clear()
    savePointBudget(250_000, 2_073_600)
    expect(loadPointBudget(2_073_600)).toBe(250_000)
  })

  it('returns null when nothing is stored', () => {
    localStorage.clear()
    expect(loadPointBudget(2_073_600)).toBeNull()
  })

  it('returns null when the stored value is for a very different drawing buffer size', () => {
    localStorage.clear()
    savePointBudget(250_000, 2_073_600)
    expect(loadPointBudget(100)).toBeNull()
  })
})
