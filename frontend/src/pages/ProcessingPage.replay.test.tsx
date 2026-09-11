import { render, screen, act } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ProcessingStageResponse, ProcessingStatusResponse } from '../api/types'
import replay from '../fixtures/progressReplay.json'
import refused from '../fixtures/progressRefused.json'

const getProcessingStatus = vi.fn()
vi.mock('../api/workspaces', () => ({ getProcessingStatus: (id: string) => getProcessingStatus(id) }))

const POLL_MS = 5000

interface RawStep {
  state?: string | null
  started?: string | null
  finished?: string | null
  duration_s?: number | null
  error?: string | null
  substage?: string | null
  detail?: string | null
  since?: string | null
  attempt?: number | null
  next_retry?: string | null
}
interface RawDoc {
  state?: string | null
  error?: string | null
  steps: Record<string, RawStep>
  provider_used?: string | null
  fallback_reason?: string | null
}

/** status.json -> the response shape, mirroring app/services/processing_status.py's
 * `_rows_from_status` field for field. Deliberately a plain mapping and not a clever one:
 * the point of this test is the SCREEN, and a mapper that differed from the server's
 * would be testing the wrong thing quietly. */
function toResponse(doc: RawDoc): ProcessingStatusResponse {
  const stages: ProcessingStageResponse[] = Object.entries(doc.steps).map(([stage, s]) => ({
    stage,
    state: s.state ?? 'unknown',
    started_at: s.started ?? null,
    finished_at: s.finished ?? null,
    duration_sec: s.duration_s ?? null,
    error: s.error ?? null,
    substage: s.substage || null,
    detail: s.detail || null,
    since: s.since || null,
    attempt: typeof s.attempt === 'number' ? s.attempt : null,
    next_retry: s.next_retry || null,
  }))
  const isDone = doc.state === 'DONE'
  return {
    workspace_id: 'w-1',
    workspace_name: 'own_0901_161054',
    // null so the DONE snapshot does not navigate away mid-assertion.
    scene_id: null,
    source: 'status.json',
    state: doc.state ?? null,
    error: doc.error ?? null,
    stages,
    seconds_since_last_record: 1,
    worker_connected: true,
    is_done: isDone,
    is_failed: doc.state === 'FAILED',
    fallback_notice: null,
    requested_provider: 'openrouter-glm',
    provider_used: doc.provider_used ?? null,
    fallback_reason: doc.fallback_reason ?? null,
  }
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/workspaces/w-1/processing']}>
      <Routes>
        <Route path="/workspaces/:workspaceId/processing" element={<ProcessingPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

// Imported after the mock is registered.
const { default: ProcessingPage } = await import('./ProcessingPage')

beforeEach(() => {
  vi.useFakeTimers()
  // Inside the recorded run's own window, so the elapsed clocks have something real to
  // count against. Set five seconds EARLY: `tick()` advances the fake clock by one poll
  // interval, so the assertions below run at 13:50:20+02:00.
  vi.setSystemTime(new Date('2026-09-09T11:50:15Z'))
  getProcessingStatus.mockReset()
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

/** Advance one poll and let the component settle. */
async function tick() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(POLL_MS)
  })
}

const snapshots = (replay as { snapshots: RawDoc[] }).snapshots

describe('replaying the recorded own_0901_161054 run', () => {
  it('was reconstructed from the real file and still says so', () => {
    const meta = replay as unknown as Record<string, string>
    expect(meta._source).toBe('var/scratch/pipeline_live/scene161054/status.json')
    expect(meta._scene).toBe('own_0901_161054__optimal_step2')
    expect(snapshots).toHaveLength(16)
  })

  it('shows each stage as the run enters it, and highlights only that one', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(snapshots[0]))
    renderPage()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    // Step by step through the whole recorded run.
    for (let i = 0; i < snapshots.length; i++) {
      getProcessingStatus.mockImplementation(async () => toResponse(snapshots[i]))
      await tick()

      const snapshot = snapshots[i]
      const names = Object.keys(snapshot.steps)

      // Every stage entered so far is on screen, and nothing that has not been.
      const rows = document.querySelectorAll('[data-stage-row]')
      expect([...rows].map((r) => r.getAttribute('data-stage-row'))).toEqual(names)

      // Exactly one row is the current one - and none is, once the run is terminal.
      const current = [...document.querySelectorAll('[data-stage-current="true"]')]
      if (snapshot.state === 'DONE') {
        expect(current).toHaveLength(0)
      } else {
        expect(current).toHaveLength(1)
        expect(current[0].getAttribute('data-stage-row')).toBe(snapshot.state)
      }
    }
  })

  it('prints each step\'s recorded state verbatim, including the two nobody enumerated', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(snapshots[snapshots.length - 1]))
    renderPage()
    await tick()

    const states = [...document.querySelectorAll('[data-stage-row]')].map((r) =>
      r.getAttribute('data-stage-state'),
    )
    expect(new Set(states)).toEqual(new Set(['done', 'skipped', 'skipped_precomputed', 'stubbed']))
    // The real numbers off the recorded file, on the rows they belong to.
    expect(screen.getByText('NVBLOX').closest('li')!.textContent).toContain('2m 28s')
    expect(screen.getByText('GPU_UP_NV').closest('li')!.textContent).toContain('32.2s')
  })

  it('omits every substage row, because this run recorded none of those fields', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(snapshots[8]))
    renderPage()
    await tick()

    expect(document.querySelectorAll('[data-stage-substage]')).toHaveLength(0)
    expect(document.querySelectorAll('[data-stage-attempt]')).toHaveLength(0)
    expect(document.querySelectorAll('[data-testid^="banner-"]')).toHaveLength(0)
    // No placeholder crept in where a field was absent.
    expect(screen.queryByText(/attempt/i)).toBeNull()
    expect(screen.queryByText(/retry at/i)).toBeNull()
  })
})

describe('the synthetic refused / fallback file', () => {
  it('is labelled synthetic, because nothing recorded on this box reaches these', () => {
    expect((refused as unknown as Record<string, string>)._source).toMatch(/^SYNTHETIC/)
  })

  it('banners the refusal, the manual wait and the provider fallback, with their own words', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(refused as unknown as RawDoc))
    renderPage()
    await tick()

    expect(screen.getByTestId('banner-refused').textContent).toContain('vast account not armed for this key')
    expect(screen.getByTestId('banner-refused').textContent).toContain('NVBLOX')
    expect(screen.getByTestId('banner-waiting_manual').textContent).toContain(
      'no per-view npz on the box; upload them or re-run FRAMES',
    )
    expect(screen.getByTestId('banner-fallback').textContent).toContain('Ran on vertex-gemma, not openrouter-glm')
    expect(screen.getByTestId('banner-fallback').textContent).toContain('openrouter 429, 3 retries')
  })

  it('shows the waiting step\'s substage, its own clock, and the attempt with a retry', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(refused as unknown as RawDoc))
    renderPage()
    await tick()

    const row = screen.getByText('GPU_UP_NV').closest('li')!
    expect(row.textContent).toContain('waiting_capacity · no box available on attempt 4')
    // `since` is 13:50:10+02:00 and the clock is 13:50:20+02:00 - ten seconds into the
    // SUBSTAGE, not the three and a half minutes since the step started at 13:46:39.
    expect(row.textContent).toContain('10s')
    expect(row.textContent).not.toContain('3m 41s')
    expect(row.textContent).toMatch(/attempt 4 · retry at /)
  })

  it('draws no attempt line for a step sitting at the initial attempt 0', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(refused as unknown as RawDoc))
    renderPage()
    await tick()

    const queued = screen.getByText('QUEUED').closest('li')!
    expect(queued.querySelector('[data-stage-attempt]')).toBeNull()
    // ...and no substage line either: QUEUED recorded none.
    expect(queued.querySelector('[data-stage-substage]')).toBeNull()
  })

  it('shows a substage that has no detail, and a detail with no attempt', async () => {
    getProcessingStatus.mockImplementation(async () => toResponse(refused as unknown as RawDoc))
    renderPage()
    await tick()

    const frames = screen.getByText('FRAMES').closest('li')!
    expect(frames.textContent).toContain('skipped · frames already on disk')
    expect(frames.querySelector('[data-stage-attempt]')).toBeNull()
  })
})
