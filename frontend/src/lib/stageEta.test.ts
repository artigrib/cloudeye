import { describe, expect, it } from 'vitest'
import type { StageTimingResponse } from '../api/types'
import { MEAN_TOTAL_DURATION_SEC, STAGE_FRACTIONS, estimateEta, formatDurationCompact, stageLabel } from './stageEta'

function timing(stage: string, started_at: string, duration_sec: number | null = null): StageTimingResponse {
  return { stage, started_at, duration_sec }
}

describe('STAGE_FRACTIONS', () => {
  it('sums to 1 across the measured stages (finalize excluded - no measurable duration)', () => {
    const total = Object.values(STAGE_FRACTIONS).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 2)
  })
})

describe('estimateEta', () => {
  it('returns null when there are no stage timings yet', () => {
    expect(estimateEta([])).toBeNull()
  })

  it('falls back to the historical mean total duration while still in the first stage', () => {
    const now = new Date('2026-09-05T12:00:30Z')
    const timings = [timing('keyframes', '2026-09-05T12:00:00Z', null)]
    const eta = estimateEta(timings, now)
    expect(eta).not.toBeNull()
    expect(eta!.isRough).toBe(true)
    expect(eta!.elapsedSec).toBeCloseTo(30, 0)
    expect(eta!.remainingSec).toBeCloseTo(MEAN_TOTAL_DURATION_SEC - 30, 0)
    expect(eta!.completedFraction).toBe(0)
  })

  it('extrapolates from completed stage fractions once at least one stage has finished', () => {
    // keyframes measured at 16s, its historical share is 4.49% of total - so a job
    // whose own keyframes stage also took 16s should extrapolate to roughly the same
    // ~356s total the historical data averages to.
    const now = new Date('2026-09-05T12:00:16Z')
    const timings = [
      timing('keyframes', '2026-09-05T12:00:00Z', 16),
      timing('vocab', '2026-09-05T12:00:16Z', null),
    ]
    const eta = estimateEta(timings, now)
    expect(eta).not.toBeNull()
    expect(eta!.isRough).toBe(false)
    expect(eta!.completedFraction).toBeCloseTo(STAGE_FRACTIONS.keyframes, 4)
    const estimatedTotal = eta!.elapsedSec / eta!.completedFraction
    expect(eta!.remainingSec).toBeCloseTo(estimatedTotal - eta!.elapsedSec, 3)
    expect(estimatedTotal).toBeGreaterThan(300)
    expect(estimatedTotal).toBeLessThan(420)
  })

  it('never returns a negative remaining time even if elapsed already exceeds the estimate', () => {
    const now = new Date('2026-09-05T13:00:00Z') // an hour later - way past any real stage
    const timings = [timing('keyframes', '2026-09-05T12:00:00Z', 16), timing('vocab', '2026-09-05T12:00:16Z', null)]
    const eta = estimateEta(timings, now)
    expect(eta!.remainingSec).toBeGreaterThanOrEqual(0)
  })

  it('reports the last (still-running) stage as currentStage', () => {
    const now = new Date('2026-09-05T12:03:00Z')
    const timings = [
      timing('keyframes', '2026-09-05T12:00:00Z', 16),
      timing('infer', '2026-09-05T12:00:16Z', null),
    ]
    expect(estimateEta(timings, now)!.currentStage).toBe('infer')
  })
})

describe('formatDurationCompact', () => {
  it('formats sub-minute durations as seconds only', () => {
    expect(formatDurationCompact(45)).toBe('45s')
    expect(formatDurationCompact(0)).toBe('0s')
  })

  it('formats minute-plus durations as "Xm Ys"', () => {
    expect(formatDurationCompact(150)).toBe('2m 30s')
  })

  it('clamps negative input to zero', () => {
    expect(formatDurationCompact(-5)).toBe('0s')
  })
})

describe('stageLabel', () => {
  it('returns a human label for known stages', () => {
    expect(stageLabel('infer')).toBe('Reconstructing geometry')
  })

  it('falls back to the raw stage name for an unrecognized stage', () => {
    expect(stageLabel('some_future_stage')).toBe('some_future_stage')
  })
})
