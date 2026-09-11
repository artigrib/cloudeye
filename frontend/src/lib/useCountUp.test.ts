import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useCountUp } from './useCountUp'

describe('useCountUp', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('starts at the initial target with no tween', () => {
    const { result } = renderHook(() => useCountUp(16))
    expect(result.current).toBe(16)
  })

  it('passes null through immediately', () => {
    const { result, rerender } = renderHook(({ v }) => useCountUp(v), { initialProps: { v: 16 as number | null } })
    rerender({ v: null })
    expect(result.current).toBeNull()
  })

  it('animates toward a new target and settles exactly on it', () => {
    const { result, rerender } = renderHook(({ v }) => useCountUp(v), { initialProps: { v: 40 } })
    expect(result.current).toBe(40)

    rerender({ v: 16 })
    // Mid-tween: somewhere between the old and new value, not necessarily settled yet.
    act(() => {
      vi.advanceTimersByTime(200)
    })
    expect(result.current).toBeLessThanOrEqual(40)
    expect(result.current).toBeGreaterThanOrEqual(16)

    act(() => {
      vi.advanceTimersByTime(400)
    })
    expect(result.current).toBe(16)
  })
})
