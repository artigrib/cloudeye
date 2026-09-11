import { describe, expect, it } from 'vitest'
import {
  isNavKey,
  isTypingTarget,
  keyStateFromCodes,
  MAX_KEY_DT_MS,
  panAxesForKeys,
  pixelsToWorldUnits,
  stepDistancePx,
} from './scene3dKeys'

describe('isNavKey', () => {
  it('accepts WASD, QE and the arrow keys', () => {
    for (const code of ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) {
      expect(isNavKey(code)).toBe(true)
    }
  })

  it('rejects everything else, including letters that are not navigation keys', () => {
    expect(isNavKey('KeyF')).toBe(false)
    expect(isNavKey('Space')).toBe(false)
    expect(isNavKey('ShiftLeft')).toBe(false)
  })
})

describe('isTypingTarget', () => {
  it('is false for null (nothing focused)', () => {
    expect(isTypingTarget(null)).toBe(false)
  })

  it('is true for an input, textarea, select and video', () => {
    for (const tag of ['input', 'textarea', 'select', 'video']) {
      expect(isTypingTarget(document.createElement(tag))).toBe(true)
    }
  })

  it('is true for a contenteditable element', () => {
    // jsdom doesn't implement the `contentEditable` property setter/`isContentEditable`
    // getter at all - set the attribute directly, the way it actually appears in
    // rendered HTML (and the way real browsers reflect the property too).
    const div = document.createElement('div')
    div.setAttribute('contenteditable', 'true')
    expect(isTypingTarget(div)).toBe(true)
  })

  it('is true for role="textbox"', () => {
    const div = document.createElement('div')
    div.setAttribute('role', 'textbox')
    expect(isTypingTarget(div)).toBe(true)
  })

  it('is false for the canvas or a plain div', () => {
    expect(isTypingTarget(document.createElement('canvas'))).toBe(false)
    expect(isTypingTarget(document.createElement('div'))).toBe(false)
  })
})

describe('panAxesForKeys', () => {
  const none = { forward: false, back: false, left: false, right: false, down: false, up: false }

  it('is zero when nothing is held', () => {
    expect(panAxesForKeys(keyStateFromCodes(new Set()))).toEqual({ right: 0, forward: 0, up: 0 })
  })

  it('moves forward at full speed for W alone', () => {
    const axes = panAxesForKeys({ ...none, forward: true })
    expect(axes).toEqual({ right: 0, forward: 1, up: 0 })
  })

  it('normalises a diagonal (W+A) to unit speed, not sqrt(2)', () => {
    const axes = panAxesForKeys({ ...none, forward: true, left: true })
    expect(Math.hypot(axes.right, axes.forward)).toBeCloseTo(1, 10)
  })

  it('cancels opposing keys (W+S) to zero', () => {
    const axes = panAxesForKeys({ ...none, forward: true, back: true })
    expect(axes.forward).toBe(0)
  })

  it('keeps up/down independent of the floor-plane axes (W+E is full speed both ways)', () => {
    const axes = panAxesForKeys({ ...none, forward: true, up: true })
    expect(axes.forward).toBe(1)
    expect(axes.up).toBe(1)
  })

  it('E is up, Q is down', () => {
    expect(panAxesForKeys({ ...none, up: true }).up).toBe(1)
    expect(panAxesForKeys({ ...none, down: true }).up).toBe(-1)
  })
})

describe('keyStateFromCodes', () => {
  it('treats arrow keys as aliases for WASD', () => {
    expect(keyStateFromCodes(new Set(['ArrowUp']))).toMatchObject({ forward: true })
    expect(keyStateFromCodes(new Set(['ArrowLeft']))).toMatchObject({ left: true })
  })
})

describe('stepDistancePx', () => {
  it('scales linearly with elapsed time, below the dt clamp', () => {
    expect(stepDistancePx(900, MAX_KEY_DT_MS)).toBe(90)
    expect(stepDistancePx(900, MAX_KEY_DT_MS / 2)).toBe(45)
  })

  it('clamps a large dt (e.g. a backgrounded tab resuming) to MAX_KEY_DT_MS', () => {
    expect(stepDistancePx(900, 1000)).toBe(stepDistancePx(900, MAX_KEY_DT_MS))
    expect(stepDistancePx(900, 5000)).toBe(stepDistancePx(900, MAX_KEY_DT_MS))
  })
})

describe('pixelsToWorldUnits', () => {
  it('matches OrbitControls._pan\'s own formula for a hand-computed case', () => {
    // 2 * distancePx * targetDistance * tan(fov/2) / clientHeight
    const distancePx = 100
    const fovDegrees = 55
    const targetDistance = 5
    const clientHeightPx = 800
    const expected = (2 * distancePx * targetDistance * Math.tan((fovDegrees / 2) * (Math.PI / 180))) / clientHeightPx
    expect(pixelsToWorldUnits(distancePx, fovDegrees, targetDistance, clientHeightPx)).toBeCloseTo(expected, 10)
  })

  it('scales linearly with targetDistance (further away -> faster in world units)', () => {
    const near = pixelsToWorldUnits(100, 55, 2, 800)
    const far = pixelsToWorldUnits(100, 55, 4, 800)
    expect(far).toBeCloseTo(near * 2, 10)
  })

  it('is zero for a zero-height viewport instead of dividing by zero', () => {
    expect(pixelsToWorldUnits(100, 55, 5, 0)).toBe(0)
  })
})
