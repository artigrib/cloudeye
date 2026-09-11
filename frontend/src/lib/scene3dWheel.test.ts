import { describe, expect, it } from 'vitest'
import { classifyWheel } from './scene3dWheel'

describe('classifyWheel', () => {
  it('treats a ctrlKey wheel (trackpad pinch, or a real Ctrl+scroll) as dolly regardless of deltas', () => {
    expect(classifyWheel({ deltaX: 500, deltaY: -500, deltaMode: 0, ctrlKey: true })).toEqual({ kind: 'dolly' })
  })

  it('treats a plain wheel (mouse or two-finger trackpad scroll) as pan', () => {
    const result = classifyWheel({ deltaX: 0, deltaY: 10, deltaMode: 0, ctrlKey: false })
    expect(result.kind).toBe('pan')
  })

  it('passes deltaMode 0 (pixels) through unscaled', () => {
    const result = classifyWheel({ deltaX: 12, deltaY: -34, deltaMode: 0, ctrlKey: false })
    expect(result).toMatchObject({ panX: -12, panY: 34 })
  })

  it('scales both axes by 16 for deltaMode 1 (lines) - a mouse wheel', () => {
    const result = classifyWheel({ deltaX: 1, deltaY: 2, deltaMode: 1, ctrlKey: false })
    expect(result).toMatchObject({ panX: -16, panY: -32 })
  })

  it('scales both axes by 100 for deltaMode 2 (pages)', () => {
    const result = classifyWheel({ deltaX: 1, deltaY: -1, deltaMode: 2, ctrlKey: false })
    expect(result).toMatchObject({ panX: -100, panY: 100 })
  })

  it('inverts sign so content follows the fingers/wheel direction', () => {
    const scrollDown = classifyWheel({ deltaX: 0, deltaY: 20, deltaMode: 0, ctrlKey: false })
    if (scrollDown.kind !== 'pan') throw new Error('expected pan')
    expect(scrollDown.panY).toBeLessThan(0)
  })
})
