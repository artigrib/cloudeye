import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { useAutoTour } from './useAutoTour'
import type { SceneObjectResponse } from '../api/types'

let seq = 0
function obj(overrides: Partial<SceneObjectResponse> & { pos_x: number; pos_z: number }): SceneObjectResponse {
  seq += 1
  return {
    id: overrides.id ?? `obj-${seq}`,
    scene_id: 'scene-1',
    name: overrides.name ?? `object-${seq}`,
    description: null,
    pos_x: overrides.pos_x,
    pos_y: 0,
    pos_z: overrides.pos_z,
    bbox_min_x: overrides.pos_x - 0.1,
    bbox_min_y: -0.1,
    bbox_min_z: overrides.pos_z - 0.1,
    bbox_max_x: overrides.pos_x + 0.1,
    bbox_max_y: 0.1,
    bbox_max_z: overrides.pos_z + 0.1,
    num_views: 1,
    num_points: 100,
    is_fragment: overrides.is_fragment ?? false,
    mesh_path: null,
    created_at: '2026-01-01T00:00:00Z',
  }
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void
  const promise = new Promise<void>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

// Named explicitly, same reason as useCommandAnimation.test.ts's AnimHookProps: a
// literal `null` in `initialProps` would otherwise narrow renderHook's inferred Props
// type down to exactly `null` instead of this wider, correct shape.
type PlatformProps = { platformId: string | null }

/** Drains pending microtasks - enough for the hook's internal await chain to settle
 * between assertions, without relying on fake timers. */
async function flush(times = 20) {
  for (let i = 0; i < times; i++) {
    await Promise.resolve()
  }
}

describe('useAutoTour', () => {
  it('never calls visitStop for an object outside reachableObjectIds', async () => {
    const reachableObj = obj({ id: 'a', pos_x: 1, pos_z: 0 })
    const unreachableObj = obj({ id: 'b', pos_x: 2, pos_z: 0 })
    const visitStop = vi.fn().mockResolvedValue(undefined)
    const { result } = renderHook(() =>
      useAutoTour({ objects: [reachableObj, unreachableObj], visitStop, selectedPlatformId: 'p1', onInterrupt: vi.fn() }),
    )

    act(() => {
      result.current.start(new Set(['a']), { x: 0, z: 0 })
    })
    await act(async () => { await flush() })

    expect(visitStop).toHaveBeenCalledTimes(1)
    expect(visitStop).toHaveBeenCalledWith(expect.objectContaining({ id: 'a' }))
    expect(result.current.running).toBe(false)
  })

  it('caps the tour at maxStops even when more objects are reachable, and completes without interrupting', async () => {
    const objects = Array.from({ length: 12 }, (_, i) => obj({ id: `o${i}`, pos_x: i, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const visitStop = vi.fn().mockResolvedValue(undefined)
    const onInterrupt = vi.fn()
    const { result } = renderHook(() =>
      useAutoTour({ objects, visitStop, selectedPlatformId: 'p1', onInterrupt }),
    )

    act(() => {
      result.current.start(reachable, { x: 0, z: 0 })
    })
    await act(async () => { await flush() })

    expect(visitStop.mock.calls.length).toBeLessThanOrEqual(8)
    expect(visitStop.mock.calls.length).toBeGreaterThanOrEqual(6)
    expect(result.current.running).toBe(false)
    expect(onInterrupt).not.toHaveBeenCalled()
  })

  it('visits stops strictly one at a time, never queuing the next before the current resolves', async () => {
    const objects = Array.from({ length: 3 }, (_, i) => obj({ id: `o${i}`, pos_x: i + 1, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    // A fresh deferred per call (rather than one shared gate) - resolving call N's must
    // not also resolve call N+1's, which hasn't even been created yet at that point.
    const gates: ReturnType<typeof deferred>[] = []
    const visitStop = vi.fn().mockImplementation(() => {
      const d = deferred()
      gates.push(d)
      return d.promise
    })
    const { result } = renderHook(() =>
      useAutoTour({ objects, visitStop, selectedPlatformId: 'p1', onInterrupt: vi.fn() }),
    )

    act(() => {
      result.current.start(reachable, { x: 0, z: 0 })
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(1)

    act(() => {
      gates[0].resolve()
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(2)

    act(() => {
      gates[1].resolve()
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(3)

    act(() => {
      gates[2].resolve()
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(3)
    expect(result.current.running).toBe(false)
  })

  it('stop() halts the tour immediately and does not resume once the in-flight stop resolves', async () => {
    const objects = Array.from({ length: 3 }, (_, i) => obj({ id: `o${i}`, pos_x: i + 1, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const gate = deferred()
    const visitStop = vi.fn().mockImplementation(() => gate.promise)
    const onInterrupt = vi.fn()
    const { result } = renderHook(() =>
      useAutoTour({ objects, visitStop, selectedPlatformId: 'p1', onInterrupt }),
    )

    act(() => {
      result.current.start(reachable, { x: 0, z: 0 })
    })
    await act(async () => { await flush() })
    expect(result.current.running).toBe(true)

    act(() => {
      result.current.stop()
    })
    expect(result.current.running).toBe(false)
    expect(onInterrupt).toHaveBeenCalledTimes(1)

    act(() => {
      gate.resolve()
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(1)
  })

  it('a platform switch mid-tour stops it and does not queue the next stop', async () => {
    const objects = Array.from({ length: 3 }, (_, i) => obj({ id: `o${i}`, pos_x: i + 1, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const gate = deferred()
    const visitStop = vi.fn().mockImplementation(() => gate.promise)
    const onInterrupt = vi.fn()

    const { result, rerender } = renderHook<ReturnType<typeof useAutoTour>, PlatformProps>(
      ({ platformId }) => useAutoTour({ objects, visitStop, selectedPlatformId: platformId, onInterrupt }),
      { initialProps: { platformId: 'p1' } },
    )

    act(() => {
      result.current.start(reachable, { x: 0, z: 0 })
    })
    await act(async () => { await flush() })
    expect(result.current.running).toBe(true)
    expect(visitStop).toHaveBeenCalledTimes(1)

    rerender({ platformId: 'p2' })

    expect(result.current.running).toBe(false)
    expect(onInterrupt).toHaveBeenCalledTimes(1)

    act(() => {
      gate.resolve()
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(1)
  })

  it('does not treat the initial platform id resolving from null as an interrupt', () => {
    const objects = [obj({ id: 'a', pos_x: 1, pos_z: 0 })]
    const onInterrupt = vi.fn()
    const { rerender } = renderHook<ReturnType<typeof useAutoTour>, PlatformProps>(
      ({ platformId }) =>
        useAutoTour({
          objects,
          visitStop: vi.fn().mockResolvedValue(undefined),
          selectedPlatformId: platformId,
          onInterrupt,
        }),
      { initialProps: { platformId: null } },
    )

    rerender({ platformId: 'default-robot' })

    expect(onInterrupt).not.toHaveBeenCalled()
  })

  it('start() is a no-op while a tour is already running', async () => {
    const objects = Array.from({ length: 3 }, (_, i) => obj({ id: `o${i}`, pos_x: i + 1, pos_z: 0 }))
    const reachable = new Set(objects.map((o) => o.id))
    const gate = deferred()
    const visitStop = vi.fn().mockImplementation(() => gate.promise)
    const { result } = renderHook(() =>
      useAutoTour({ objects, visitStop, selectedPlatformId: 'p1', onInterrupt: vi.fn() }),
    )

    act(() => {
      result.current.start(reachable, { x: 0, z: 0 })
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(1)

    act(() => {
      result.current.start(reachable, { x: 0, z: 0 })
    })
    await act(async () => { await flush() })
    expect(visitStop).toHaveBeenCalledTimes(1)
  })

  it('start() does nothing without a known reachability set or robot position', () => {
    const objects = [obj({ id: 'a', pos_x: 1, pos_z: 0 })]
    const visitStop = vi.fn().mockResolvedValue(undefined)
    const { result } = renderHook(() =>
      useAutoTour({ objects, visitStop, selectedPlatformId: 'p1', onInterrupt: vi.fn() }),
    )

    act(() => {
      result.current.start(null, { x: 0, z: 0 })
    })
    expect(result.current.running).toBe(false)

    act(() => {
      result.current.start(new Set(['a']), null)
    })
    expect(result.current.running).toBe(false)
    expect(visitStop).not.toHaveBeenCalled()
  })

  it('stop() is a no-op when no tour is running', () => {
    const onInterrupt = vi.fn()
    const { result } = renderHook(() =>
      useAutoTour({
        objects: [],
        visitStop: vi.fn().mockResolvedValue(undefined),
        selectedPlatformId: 'p1',
        onInterrupt,
      }),
    )

    act(() => {
      result.current.stop()
    })
    expect(onInterrupt).not.toHaveBeenCalled()
  })
})
