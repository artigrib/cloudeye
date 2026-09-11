import { describe, expect, it } from 'vitest'
import { CLICK_MAX_DURATION_MS, CLICK_MOVE_THRESHOLD_PX, isClickCandidate, isClickGesture } from './pointerGesture'

describe('isClickCandidate', () => {
  it('accepts the primary pointer, left button', () => {
    expect(isClickCandidate({ button: 0, isPrimary: true })).toBe(true)
  })

  it('rejects the middle button', () => {
    expect(isClickCandidate({ button: 1, isPrimary: true })).toBe(false)
  })

  it('rejects the right button', () => {
    expect(isClickCandidate({ button: 2, isPrimary: true })).toBe(false)
  })

  it('rejects a non-primary pointer even on the left button', () => {
    expect(isClickCandidate({ button: 0, isPrimary: false })).toBe(false)
  })
})

describe('isClickGesture', () => {
  const down = { pointerId: 1, button: 0, x: 100, y: 100, timeMs: 1000 }

  it('is a click when the pointer barely moved and released quickly', () => {
    expect(isClickGesture(down, { ...down, x: 101, y: 100, timeMs: 1050 })).toBe(true)
  })

  it('is still a click at exactly the move threshold', () => {
    const up = { ...down, x: down.x + CLICK_MOVE_THRESHOLD_PX, timeMs: down.timeMs + 10 }
    expect(isClickGesture(down, up)).toBe(true)
  })

  it('is a drag just past the move threshold', () => {
    const up = { ...down, x: down.x + CLICK_MOVE_THRESHOLD_PX + 1, timeMs: down.timeMs + 10 }
    expect(isClickGesture(down, up)).toBe(false)
  })

  it('is still a click at exactly the duration threshold', () => {
    const up = { ...down, timeMs: down.timeMs + CLICK_MAX_DURATION_MS }
    expect(isClickGesture(down, up)).toBe(true)
  })

  it('is a drag (long press) just past the duration threshold', () => {
    const up = { ...down, timeMs: down.timeMs + CLICK_MAX_DURATION_MS + 1 }
    expect(isClickGesture(down, up)).toBe(false)
  })

  it('rejects a mismatched pointerId (e.g. a second finger/pointer)', () => {
    const up = { ...down, pointerId: 2, timeMs: down.timeMs + 10 }
    expect(isClickGesture(down, up)).toBe(false)
  })

  it('rejects when the release button is not the primary button', () => {
    const up = { ...down, button: 2, timeMs: down.timeMs + 10 }
    expect(isClickGesture(down, up)).toBe(false)
  })

  it('rejects a slow return to nearly the same spot (orbit-and-back)', () => {
    const up = { ...down, x: down.x + 1, timeMs: down.timeMs + 2000 }
    expect(isClickGesture(down, up)).toBe(false)
  })
})
