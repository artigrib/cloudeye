import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { normalizeAngle, shortestAngleDelta, headingYaw, dampAngle } from './robot-heading'

describe('normalizeAngle', () => {
  it('maps 3PI/2 into the negative half', () => {
    expect(normalizeAngle((3 * Math.PI) / 2)).toBeCloseTo(-Math.PI / 2)
  })

  it('maps -3PI to -PI', () => {
    expect(normalizeAngle(-3 * Math.PI)).toBeCloseTo(-Math.PI)
  })

  it('pins the half-open interval: PI normalizes to -PI', () => {
    expect(normalizeAngle(Math.PI)).toBeCloseTo(-Math.PI)
  })

  it('leaves a large multiple of TAU in range', () => {
    expect(normalizeAngle(0.3 + 10 * Math.PI * 2)).toBeCloseTo(0.3)
  })
})

describe('shortestAngleDelta', () => {
  it('takes the short way across the +-PI wrap', () => {
    const a = Math.PI - 0.1
    const b = -Math.PI + 0.1
    expect(shortestAngleDelta(a, b)).toBeCloseTo(0.2)
  })

  it('is negative when going the other way across the wrap', () => {
    expect(shortestAngleDelta(0.1, -0.1)).toBeCloseTo(-0.2)
  })

  it('always returns a result in [-PI, PI)', () => {
    for (let a = -10; a <= 10; a += 1.3) {
      for (let b = -10; b <= 10; b += 1.7) {
        const d = shortestAngleDelta(a, b)
        expect(d).toBeGreaterThanOrEqual(-Math.PI)
        expect(d).toBeLessThan(Math.PI)
      }
    }
  })
})

describe('headingYaw', () => {
  it.each([
    [0.2, 0, 0],
    [0, -0.2, Math.PI / 2],
    [0, 0.2, -Math.PI / 2],
    // atan2(-dz, dx) at dz=0, dx<0 is atan2(-0, -0.2) === -PI (IEEE 754 signed zero), the same
    // angle as +PI but on the canonical (normalizeAngle-consistent) side of the +-PI seam.
    [-0.2, 0, -Math.PI],
  ])('(dx=%d, dz=%d) -> yaw %d', (dx, dz, expected) => {
    expect(headingYaw(dx, dz)).toBeCloseTo(expected)
  })

  it('returns null below the noise floor', () => {
    expect(headingYaw(1e-6, 0)).toBeNull()
  })

  it('returns null above the teleport threshold', () => {
    expect(headingYaw(1, 1)).toBeNull() // len ~1.41 > default maxMove 0.5
  })

  it('respects custom minMove/maxMove', () => {
    expect(headingYaw(0.05, 0, { minMove: 0.1 })).toBeNull()
    expect(headingYaw(2, 0, { maxMove: 5 })).toBeCloseTo(0)
  })

  // The convention-pinning test: catches a sign error at its source rather than by squinting
  // at the canvas. For a sampling of displacements, the yaw headingYaw() returns must rotate
  // the model's local forward (+X) to point the same way (parallel, same sign) as the
  // scene-space displacement (dx, 0, dz).
  it('round-trips through THREE.Euler to point along the displacement', () => {
    const samples: [number, number][] = [
      [1, 0],
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
      [0.3, 0.05],
    ]
    for (const [dx, dz] of samples) {
      const yaw = headingYaw(dx, dz, { maxMove: 10 })
      expect(yaw).not.toBeNull()
      const forward = new THREE.Vector3(1, 0, 0).applyEuler(new THREE.Euler(0, yaw as number, 0))
      const disp = new THREE.Vector3(dx, 0, dz).normalize()
      expect(forward.dot(disp)).toBeCloseTo(1, 5)
    }
  })
})

describe('dampAngle', () => {
  it('does nothing for dtMs <= 0', () => {
    expect(dampAngle(0.5, 2.0, 0, 100)).toBeCloseTo(0.5)
    expect(dampAngle(0.5, 2.0, -10, 100)).toBeCloseTo(0.5)
  })

  it('jumps straight to target when smoothTimeMs <= 0', () => {
    expect(dampAngle(0.5, 2.0, 16, 0)).toBeCloseTo(normalizeAngle(2.0))
  })

  it('monotonically shrinks the remaining gap without overshoot', () => {
    let current = 0
    const target = 2.5
    let prevGap = Math.abs(shortestAngleDelta(current, target))
    for (let i = 0; i < 20; i++) {
      current = dampAngle(current, target, 16, 120)
      const gap = Math.abs(shortestAngleDelta(current, target))
      expect(gap).toBeLessThan(prevGap)
      prevGap = gap
    }
  })

  it('wraps the short way across +-PI rather than sweeping through zero', () => {
    const current = 3.0
    const target = -3.0 // short way is +0.283, the long way is ~-6.0
    const result = dampAngle(current, target, 16, 120)
    // Taking the short way means we've moved *away* from 0 (further positive, or wrapped
    // negative past PI) - never landed just above zero, which is what sweeping through the
    // long way for one small step would look like.
    expect(Math.abs(shortestAngleDelta(result, target))).toBeLessThan(0.3)
  })

  it('is frame-rate independent: ten 16ms steps ~= one 160ms step', () => {
    const smoothTimeMs = 120
    let stepped = 0.2
    for (let i = 0; i < 10; i++) {
      stepped = dampAngle(stepped, 3.0, 16, smoothTimeMs)
    }
    const single = dampAngle(0.2, 3.0, 160, smoothTimeMs)
    expect(stepped).toBeCloseTo(single, 6)
  })

  it('never returns NaN for a huge dt and lands within eps of target', () => {
    const result = dampAngle(0.1, 2.9, 1e9, 120)
    expect(Number.isNaN(result)).toBe(false)
    expect(Math.abs(shortestAngleDelta(result, 2.9))).toBeLessThan(1e-6)
  })
})
