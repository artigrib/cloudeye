import { describe, expect, it } from 'vitest'
import type { ProcessingStageResponse, ProcessingStatusResponse } from '../api/types'
import {
  SUBSTAGES,
  attemptLine,
  attentionBanners,
  currentStage,
  elapsedSince,
  formatElapsed,
  parseStamp,
  substageLine,
} from './progressRows'

function row(over: Partial<ProcessingStageResponse> = {}): ProcessingStageResponse {
  return {
    stage: 'GPU_UP_NV',
    state: 'running',
    started_at: '2026-09-09T13:46:39+02:00',
    finished_at: null,
    duration_sec: null,
    error: null,
    substage: null,
    detail: null,
    since: null,
    attempt: null,
    next_retry: null,
    ...over,
  }
}

function status(over: Partial<ProcessingStatusResponse> = {}): ProcessingStatusResponse {
  return {
    workspace_id: 'w',
    workspace_name: 'ws',
    scene_id: null,
    source: 'status.json',
    state: 'GPU_UP_NV',
    error: null,
    stages: [],
    seconds_since_last_record: 1,
    worker_connected: true,
    is_done: false,
    is_failed: false,
    fallback_notice: null,
    requested_provider: null,
    provider_used: null,
    fallback_reason: null,
    ...over,
  }
}

describe('SUBSTAGES', () => {
  it("is pipeline-v1's closed vocabulary, in its order", () => {
    expect(SUBSTAGES).toEqual([
      'searching',
      'creating',
      'provisioning',
      'waiting_capacity',
      'transfer_gate',
      'refused',
      'waiting_manual',
      'skipped',
    ])
  })
})

describe('parseStamp', () => {
  it('parses local ISO-8601 with an offset, which is what pipeline-v1 writes', () => {
    expect(parseStamp('2026-09-09T13:46:39+02:00')).toBe(Date.parse('2026-09-09T11:46:39Z'))
  })

  it('is null for absent or unparseable, never NaN', () => {
    expect(parseStamp(null)).toBeNull()
    expect(parseStamp(undefined)).toBeNull()
    expect(parseStamp('')).toBeNull()
    expect(parseStamp('later on')).toBeNull()
  })
})

describe('formatElapsed', () => {
  it('counts whole seconds under a minute', () => {
    expect(formatElapsed(0)).toBe('0s')
    expect(formatElapsed(4_900)).toBe('4s')
    expect(formatElapsed(59_999)).toBe('59s')
  })

  it('pads the seconds so the number stops jumping about', () => {
    expect(formatElapsed(60_000)).toBe('1m 00s')
    expect(formatElapsed(221_000)).toBe('3m 41s')
  })

  it('drops to hours and minutes past an hour', () => {
    expect(formatElapsed(3_600_000)).toBe('1h 00m')
    expect(formatElapsed(3_600_000 + 5 * 60_000)).toBe('1h 05m')
  })
})

describe('elapsedSince', () => {
  const now = Date.parse('2026-09-09T11:50:20Z')

  it("counts from the SUBSTAGE's clock, not the step's", () => {
    // started 13:46:39+02:00 = 11:46:39Z, since 13:50:10+02:00 = 11:50:10Z.
    expect(elapsedSince('2026-09-09T13:50:10+02:00', now)).toBe('10s')
  })

  it('is null when the writer recorded no `since`', () => {
    expect(elapsedSince(null, now)).toBeNull()
    expect(elapsedSince(undefined, now)).toBeNull()
  })

  it('is null for a start in the future rather than a negative count', () => {
    // A clock that has not reached its own start is a fact about the clock; "-3s" would
    // be the screen guessing which of the two is wrong.
    expect(elapsedSince('2026-09-09T13:59:00+02:00', now)).toBeNull()
  })
})

describe('substageLine', () => {
  it('joins substage and detail into one line', () => {
    expect(substageLine(row({ substage: 'waiting_capacity', detail: 'no box on attempt 4' })))
      .toBe('waiting_capacity · no box on attempt 4')
  })

  it('shows either half alone', () => {
    expect(substageLine(row({ substage: 'searching' }))).toBe('searching')
    expect(substageLine(row({ detail: 'transfer gate closed' }))).toBe('transfer gate closed')
  })

  it('is null when there is neither, so nothing is drawn', () => {
    expect(substageLine(row())).toBeNull()
    expect(substageLine(row({ substage: '', detail: '   ' }))).toBeNull()
  })

  it('shows a substage the client does not know, as itself', () => {
    expect(substageLine(row({ substage: 'teleporting' }))).toBe('teleporting')
  })
})

describe('attemptLine', () => {
  it('names the attempt, and the retry time when there is one', () => {
    const line = attemptLine(row({ attempt: 4, next_retry: '2026-09-09T13:51:10+02:00' }))
    expect(line).toMatch(/^attempt 4 · retry at /)
  })

  it('names the attempt alone when no retry is scheduled', () => {
    expect(attemptLine(row({ attempt: 4 }))).toBe('attempt 4')
  })

  it('is null for attempt 0 - the value Status.enter writes before any retry loop', () => {
    expect(attemptLine(row({ attempt: 0 }))).toBeNull()
    expect(attemptLine(row({ attempt: 0, next_retry: '2026-09-09T13:51:10+02:00' }))).toBeNull()
  })

  it('is null when the field is absent', () => {
    expect(attemptLine(row())).toBeNull()
  })

  it('ignores an unparseable retry time rather than printing it raw', () => {
    expect(attemptLine(row({ attempt: 2, next_retry: 'soon' }))).toBe('attempt 2')
  })
})

describe('attentionBanners', () => {
  it('banners a refusal, with the writer’s own detail', () => {
    const banners = attentionBanners(
      status({ stages: [row({ stage: 'GPU_UP_NV', substage: 'refused', detail: 'account not armed' })] }),
    )
    expect(banners).toEqual([
      { kind: 'refused', title: 'Refused', detail: 'account not armed', stage: 'GPU_UP_NV' },
    ])
  })

  it('banners a run waiting on a person', () => {
    const banners = attentionBanners(
      status({ stages: [row({ stage: 'MAPANYTHING', substage: 'waiting_manual', detail: 'upload the frames' })] }),
    )
    expect(banners[0].kind).toBe('waiting_manual')
    expect(banners[0].title).toBe('Waiting for you')
  })

  it('carries a null detail rather than inventing one', () => {
    const banners = attentionBanners(status({ stages: [row({ substage: 'refused' })] }))
    expect(banners[0].detail).toBeNull()
  })

  it('banners a provider that is not the one that was asked for', () => {
    const banners = attentionBanners(
      status({
        requested_provider: 'openrouter-glm',
        provider_used: 'vertex-gemma',
        fallback_reason: 'openrouter 429, 3 retries',
      }),
    )
    expect(banners).toEqual([
      {
        kind: 'fallback',
        title: 'Ran on vertex-gemma, not openrouter-glm',
        detail: 'openrouter 429, 3 retries',
      },
    ])
  })

  it('says nothing when the provider matches, or when none was reported', () => {
    expect(attentionBanners(status({ requested_provider: 'a', provider_used: 'a' }))).toEqual([])
    // Every run today: the worker writes no provider at all (JOB_SPEC section 2).
    expect(attentionBanners(status({ requested_provider: 'a', provider_used: null }))).toEqual([])
  })

  it('says nothing at all for an ordinary run', () => {
    expect(attentionBanners(status({ stages: [row({ substage: 'provisioning' }), row()] }))).toEqual([])
  })
})

describe('currentStage', () => {
  it('is the top-level state, when a row matches it', () => {
    expect(currentStage(status({ state: 'NVBLOX', stages: [row({ stage: 'NVBLOX' })] }))).toBe('NVBLOX')
  })

  it('is null on a terminal document - DONE and FAILED are not stages', () => {
    expect(currentStage(status({ state: 'DONE', is_done: true, stages: [row({ stage: 'INGEST' })] }))).toBeNull()
    expect(currentStage(status({ state: 'FAILED', is_failed: true, stages: [row()] }))).toBeNull()
  })

  it('is null when the state names a stage the feed has no row for', () => {
    expect(currentStage(status({ state: 'SEMANTICS', stages: [row({ stage: 'NVBLOX' })] }))).toBeNull()
  })
})
