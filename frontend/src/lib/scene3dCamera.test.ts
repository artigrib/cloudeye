import { describe, expect, it } from 'vitest'
import {
  cameraClipPlanes,
  easeInOutCubic,
  frameObjectPose,
  initialCameraPose,
  lerpPose,
  type BoundingBox3,
} from './scene3dCamera'

describe('cameraClipPlanes', () => {
  it('matches the ikport-room-scale defaults closely (near ~0.05-0.1, far ~50)', () => {
    const box: BoundingBox3 = { min: [-1.8, 0, -2.15], max: [1.8, 2.59, 2.15] }
    const { near, far } = cameraClipPlanes(box)
    expect(near).toBeGreaterThanOrEqual(0.02)
    expect(near).toBeLessThanOrEqual(0.1)
    expect(far).toBe(50)
  })

  it('scales both planes up for a much larger scan', () => {
    const room: BoundingBox3 = { min: [-1.8, 0, -2.15], max: [1.8, 2.59, 2.15] }
    const warehouse: BoundingBox3 = { min: [-50, 0, -50], max: [50, 8, 50] }
    const roomPlanes = cameraClipPlanes(room)
    const warehousePlanes = cameraClipPlanes(warehouse)
    expect(warehousePlanes.near).toBeGreaterThan(roomPlanes.near)
    expect(warehousePlanes.far).toBeGreaterThan(roomPlanes.far)
  })

  it('clamps near to a sane range even for a degenerate (point-sized) box', () => {
    const point: BoundingBox3 = { min: [0, 0, 0], max: [0, 0, 0] }
    const { near, far } = cameraClipPlanes(point)
    expect(near).toBeGreaterThanOrEqual(0.02)
    expect(far).toBeGreaterThanOrEqual(50)
  })

  it('never lets near exceed far', () => {
    const huge: BoundingBox3 = { min: [-1000, 0, -1000], max: [1000, 50, 1000] }
    const { near, far } = cameraClipPlanes(huge)
    expect(near).toBeLessThan(far)
  })
})

describe('initialCameraPose', () => {
  // The ikport room: ~3.6 x 4.3m footprint, 2.59m ceiling, floor at Y=0.
  const box: BoundingBox3 = { min: [-1.8, 0, -2.15], max: [1.8, 2.59, 2.15] }

  it('targets the bbox center', () => {
    const { target } = initialCameraPose(box)
    expect(target).toEqual([0, 1.295, 0])
  })

  it('places the camera above the target (looking down)', () => {
    const { position, target } = initialCameraPose(box)
    expect(position[1]).toBeGreaterThan(target[1])
  })

  it('places the camera outside the footprint on the horizontal plane', () => {
    const { position, target } = initialCameraPose(box)
    const horizontalOffset = Math.hypot(position[0] - target[0], position[2] - target[2])
    expect(horizontalOffset).toBeGreaterThan(0)
  })

  it('sits at roughly a 45 degree elevation above the target', () => {
    const { position, target } = initialCameraPose(box)
    const rise = position[1] - target[1]
    const run = Math.hypot(position[0] - target[0], position[2] - target[2])
    expect(rise / run).toBeCloseTo(1, 1) // tan(45deg) === 1
  })

  it('scales framing distance with room size instead of a fixed distance', () => {
    const small = initialCameraPose(box)
    const big: BoundingBox3 = { min: [-10, 0, -10], max: [10, 3, 10] }
    const bigPose = initialCameraPose(big)
    const smallDist = Math.hypot(small.position[0] - small.target[0], small.position[2] - small.target[2])
    const bigDist = Math.hypot(bigPose.position[0] - bigPose.target[0], bigPose.position[2] - bigPose.target[2])
    expect(bigDist).toBeGreaterThan(smallDist)
  })

  it('never collapses to a zero framing distance for a degenerate (point-sized) box', () => {
    const point: BoundingBox3 = { min: [0, 0, 0], max: [0, 0, 0] }
    const { position, target } = initialCameraPose(point)
    expect(Math.hypot(position[0] - target[0], position[1] - target[1], position[2] - target[2])).toBeGreaterThan(0)
  })
})

describe('frameObjectPose', () => {
  // A small object (a mug, roughly) sitting away from the world origin.
  const box: BoundingBox3 = { min: [1.9, 0.7, -0.35], max: [2.1, 0.85, -0.15] }

  it('targets the bbox center', () => {
    const { target } = frameObjectPose(box, [0, 3, 5])
    expect(target[0]).toBeCloseTo(2, 10)
    expect(target[1]).toBeCloseTo(0.775, 10)
    expect(target[2]).toBeCloseTo(-0.25, 10)
  })

  it('preserves the viewing direction from the given camera position', () => {
    const from: [number, number, number] = [2, 3, 3]
    const { position, target } = frameObjectPose(box, from)
    const oldDir = normalize([from[0] - target[0], from[1] - target[1], from[2] - target[2]])
    const newDir = normalize([position[0] - target[0], position[1] - target[1], position[2] - target[2]])
    // Same direction: cross product ~0, dot product > 0 (not preserved-but-flipped).
    const cross = [
      oldDir[1] * newDir[2] - oldDir[2] * newDir[1],
      oldDir[2] * newDir[0] - oldDir[0] * newDir[2],
      oldDir[0] * newDir[1] - oldDir[1] * newDir[0],
    ]
    expect(Math.hypot(...cross)).toBeCloseTo(0, 6)
    expect(oldDir[0] * newDir[0] + oldDir[1] * newDir[1] + oldDir[2] * newDir[2]).toBeGreaterThan(0)
  })

  it('keeps a safe distance (clear of the 0.05 near plane) even for a tiny box', () => {
    const tiny: BoundingBox3 = { min: [0, 0, 0], max: [0.02, 0.02, 0.02] }
    const { position, target } = frameObjectPose(tiny, [0, 1, 1])
    const distance = Math.hypot(position[0] - target[0], position[1] - target[1], position[2] - target[2])
    expect(distance).toBeGreaterThanOrEqual(0.65)
  })

  it('falls back to a standard elevated pose when the camera sits on the target already', () => {
    const { position, target } = frameObjectPose(box, [2, 0.775, -0.25])
    expect(Number.isFinite(position[0])).toBe(true)
    expect(Number.isFinite(position[1])).toBe(true)
    expect(Number.isFinite(position[2])).toBe(true)
    expect(position[1]).toBeGreaterThan(target[1]) // still looking down, not undefined/NaN
  })
})

describe('easeInOutCubic', () => {
  it('starts at 0 and ends at 1', () => {
    expect(easeInOutCubic(0)).toBe(0)
    expect(easeInOutCubic(1)).toBe(1)
  })

  it('is exactly the midpoint at t=0.5', () => {
    expect(easeInOutCubic(0.5)).toBeCloseTo(0.5, 10)
  })

  it('is monotonically increasing', () => {
    let prev = -Infinity
    for (let t = 0; t <= 1; t += 0.1) {
      const v = easeInOutCubic(t)
      expect(v).toBeGreaterThanOrEqual(prev)
      prev = v
    }
  })
})

describe('lerpPose', () => {
  const from = { position: [0, 0, 0] as [number, number, number], target: [0, 0, 0] as [number, number, number] }
  const to = { position: [10, 20, 30] as [number, number, number], target: [1, 2, 3] as [number, number, number] }

  it('matches the start pose at t=0 and the end pose at t=1', () => {
    expect(lerpPose(from, to, 0)).toEqual(from)
    expect(lerpPose(from, to, 1)).toEqual(to)
  })

  it('is the midpoint at t=0.5', () => {
    expect(lerpPose(from, to, 0.5)).toEqual({ position: [5, 10, 15], target: [0.5, 1, 1.5] })
  })
})

function normalize(v: [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2])
  return [v[0] / len, v[1] / len, v[2] / len]
}
