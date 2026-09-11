import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DOUBLE_CLICK_GRACE_MS, createDeferredClick } from './deferredClick'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

describe('createDeferredClick', () => {
  it('runs a lone click once the grace period passes', () => {
    const run = vi.fn()
    const d = createDeferredClick()
    d.click(run)
    expect(run).not.toHaveBeenCalled()
    vi.advanceTimersByTime(DOUBLE_CLICK_GRACE_MS)
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('never runs the click a double-click cancelled - the camera dive case', () => {
    const focus = vi.fn()
    const d = createDeferredClick()
    // What the DOM actually delivers for one double-click: click, click, dblclick.
    d.click(focus)
    d.click(focus)
    d.cancel()
    vi.advanceTimersByTime(10_000)
    expect(focus).not.toHaveBeenCalled()
  })

  it('keeps only the latest scheduled action', () => {
    const first = vi.fn()
    const second = vi.fn()
    const d = createDeferredClick()
    d.click(first)
    d.click(second)
    vi.advanceTimersByTime(DOUBLE_CLICK_GRACE_MS)
    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledTimes(1)
  })

  it('reports whether anything is scheduled, and cancel is idempotent', () => {
    const d = createDeferredClick()
    expect(d.pending()).toBe(false)
    d.click(() => {})
    expect(d.pending()).toBe(true)
    d.cancel()
    d.cancel()
    expect(d.pending()).toBe(false)
  })

  it('honours a custom delay', () => {
    const run = vi.fn()
    const d = createDeferredClick(50)
    d.click(run)
    vi.advanceTimersByTime(49)
    expect(run).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(run).toHaveBeenCalledTimes(1)
  })
})
