import { describe, expect, it } from 'vitest'
import {
  POINT_SIZE_MAX_DEVICE_PX,
  POINT_SIZE_MIN_DEVICE_PX,
  pointSizeDevicePx,
  pointSizeDevicePxUnclamped,
  pointSizeSizeNode,
} from './pointSizing'

// SceneMap3D's DEFAULT_POINT_SIZE and a representative 800px-tall canvas (the
// /_pointtest harness's viewport) -> half-height 400 logical px.
const DEFAULT_SIZE_M = 0.015
const HALF_H = 400

describe('pointSizeDevicePx', () => {
  it('is exactly 2x at DPR=2 for the same metres/depth/canvas, pre-clamp', () => {
    // THE double-DPR regression guard. The bug this pins was
    // `uSize.value = pointSize * renderer.getPixelRatio()` in SceneMap3D on top of
    // PointsNodeMaterial's own unconditional `.mul(screenDPR)` - which produced 4x here,
    // not 2x. Deliberately asserted pre-clamp so the clamp can't mask the ratio.
    const at1x = pointSizeDevicePxUnclamped(DEFAULT_SIZE_M, -4, HALF_H, 1)
    const at2x = pointSizeDevicePxUnclamped(DEFAULT_SIZE_M, -4, HALF_H, 2)
    expect(at2x).toBeCloseTo(at1x * 2, 10)
  })

  it('matches the classic WebGL material formula for the default slider position', () => {
    // classicPointCloudMaterial.ts: uSize(=m*DPR) * uScale(=halfHeightLogical) / -viewZ.
    // 0.015 * 1 * 400 / 4 = 1.5 device px at DPR=1, 3.0 at DPR=2.
    expect(pointSizeDevicePx(DEFAULT_SIZE_M, -4, HALF_H, 1)).toBeCloseTo(1.5, 10)
    expect(pointSizeDevicePx(DEFAULT_SIZE_M, -4, HALF_H, 2)).toBeCloseTo(3.0, 10)
  })

  it('scales linearly with the slider value', () => {
    const base = pointSizeDevicePxUnclamped(DEFAULT_SIZE_M, -4, HALF_H, 1)
    expect(pointSizeDevicePxUnclamped(DEFAULT_SIZE_M * 2, -4, HALF_H, 1)).toBeCloseTo(base * 2, 10)
  })

  it('attenuates with distance - a point twice as far is half the size', () => {
    const near = pointSizeDevicePxUnclamped(DEFAULT_SIZE_M, -4, HALF_H, 1)
    const far = pointSizeDevicePxUnclamped(DEFAULT_SIZE_M, -8, HALF_H, 1)
    expect(far).toBeCloseTo(near / 2, 10)
  })

  it('clamps to the floor for a far, tiny point rather than letting it vanish', () => {
    // 0.003m at 40m, DPR=1 -> 0.03 device px raw.
    expect(pointSizeDevicePxUnclamped(0.003, -40, HALF_H, 1)).toBeLessThan(POINT_SIZE_MIN_DEVICE_PX)
    expect(pointSizeDevicePx(0.003, -40, HALF_H, 1)).toBe(POINT_SIZE_MIN_DEVICE_PX)
  })

  it('clamps to the ceiling for a large, very near point', () => {
    // 0.06m at 0.2m, DPR=2 -> 240 device px raw.
    expect(pointSizeDevicePx(0.06, -0.2, HALF_H, 2)).toBe(POINT_SIZE_MAX_DEVICE_PX)
  })

  it('clamps in DEVICE pixels, not logical - the floor/ceiling are DPR-independent', () => {
    // The guard against "clamp before DPR" (the actual bug this file fixes), which would
    // give a 2-device-px floor and a 64-device-px ceiling at DPR=2, silently disagreeing
    // with the classic renderer's fixed [1, 32] device-px range at every DPR.
    expect(pointSizeDevicePx(0.003, -40, HALF_H, 2)).toBe(POINT_SIZE_MIN_DEVICE_PX)
    expect(pointSizeDevicePx(0.06, -0.05, HALF_H, 2)).toBe(POINT_SIZE_MAX_DEVICE_PX)
  })

  it('never returns NaN for a degenerate point exactly on the camera plane', () => {
    expect(Number.isFinite(pointSizeDevicePx(DEFAULT_SIZE_M, 0, HALF_H, 1))).toBe(true)
  })
})

describe('pointSizeSizeNode', () => {
  it("cancels PointsNodeMaterial's own unconditional screenDPR multiply exactly", () => {
    for (const dpr of [1, 1.5, 2, 3]) {
      const node = pointSizeSizeNode(DEFAULT_SIZE_M, -4, HALF_H, dpr)
      expect(node * dpr).toBeCloseTo(pointSizeDevicePx(DEFAULT_SIZE_M, -4, HALF_H, dpr), 10)
    }
  })
})
