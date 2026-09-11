import { describe, expect, it } from 'vitest'
import { CLOUD_BIN_HEADER_BYTES, CloudBinError, parseCloudBin, zUpToYUp } from './cloudBin'

/** Builds a CEPC v1 buffer the same way scripts/ingest/write_cloud_bin.py does. */
function makeCloudBin(
  points: number[][],
  colors: number[][] | null,
  opts: { magic?: string; version?: number; count?: number } = {},
): ArrayBuffer {
  const count = opts.count ?? points.length
  const hasRgb = colors !== null
  const bytes = CLOUD_BIN_HEADER_BYTES + points.length * (hasRgb ? 15 : 12)
  const buf = new ArrayBuffer(bytes)
  const view = new DataView(buf)
  const magic = opts.magic ?? 'CEPC'
  for (let i = 0; i < 4; i++) view.setUint8(i, magic.charCodeAt(i))
  view.setUint32(4, opts.version ?? 1, true)
  view.setUint32(8, count, true)
  view.setUint32(12, hasRgb ? 1 : 0, true)
  const flat = points.flat()
  const bbox = [0, 1, 2].map((c) => Math.min(...points.map((p) => p[c])))
    .concat([0, 1, 2].map((c) => Math.max(...points.map((p) => p[c]))))
  bbox.forEach((v, i) => view.setFloat32(16 + i * 4, v, true))
  flat.forEach((v, i) => view.setFloat32(CLOUD_BIN_HEADER_BYTES + i * 4, v, true))
  if (colors) {
    const off = CLOUD_BIN_HEADER_BYTES + points.length * 12
    colors.flat().forEach((v, i) => view.setUint8(off + i, v))
  }
  return buf
}

describe('zUpToYUp', () => {
  it('maps (u, v, h) to (u, h, -v)', () => {
    expect(Array.from(zUpToYUp(new Float32Array([1, 2, 3])))).toEqual([1, 3, -2])
  })

  it('is a proper rotation - it preserves handedness, it does not mirror', () => {
    // cross(e_u, e_v) = e_h in the source frame; after the map the images must still
    // satisfy cross(x, y) = z, which a mirror (+v instead of -v) would break.
    const u = Array.from(zUpToYUp(new Float32Array([1, 0, 0])))
    const v = Array.from(zUpToYUp(new Float32Array([0, 1, 0])))
    const h = Array.from(zUpToYUp(new Float32Array([0, 0, 1])))
    // `+ 0` normalises -0 to 0: the components are exact, but a signed zero would fail a
    // structural compare against an unsigned one for no mathematical reason.
    const cross = (a: number[], b: number[]) => [
      a[1] * b[2] - a[2] * b[1] + 0,
      a[2] * b[0] - a[0] * b[2] + 0,
      a[0] * b[1] - a[1] * b[0] + 0,
    ]
    expect(cross(u, v)).toEqual(h.map((n) => n + 0))
  })

  it('preserves lengths', () => {
    const p = zUpToYUp(new Float32Array([3, -4, 12]))
    expect(Math.hypot(p[0], p[1], p[2])).toBeCloseTo(13, 5)
  })
})

describe('parseCloudBin', () => {
  it('reads points and colours, rotated into the viewer frame', () => {
    const buf = makeCloudBin([[1, 2, 3], [-1, -2, -3]], [[10, 20, 30], [40, 50, 60]])
    const cloud = parseCloudBin(buf)
    expect(cloud.count).toBe(2)
    expect(Array.from(cloud.positions)).toEqual([1, 3, -2, -1, -3, 2])
    expect(Array.from(cloud.colors!)).toEqual([10, 20, 30, 40, 50, 60])
    // bbox is reported as stored, in the Z-up frame
    expect(Array.from(cloud.bboxZUp)).toEqual([-1, -2, -3, 1, 2, 3])
  })

  it('accepts a cloud with no colours', () => {
    const cloud = parseCloudBin(makeCloudBin([[0, 0, 0]], null))
    expect(cloud.colors).toBeNull()
    expect(cloud.count).toBe(1)
  })

  it('hands back arrays whose buffers are detachable independently', () => {
    // The worker transfers positions and colours as separate transferables; if either were
    // a view onto the response buffer rather than a copy, transferring one would detach
    // the other. byteOffset 0 and an exactly-sized buffer is what makes that safe.
    const cloud = parseCloudBin(makeCloudBin([[1, 2, 3]], [[1, 2, 3]]))
    expect(cloud.positions.byteOffset).toBe(0)
    expect(cloud.positions.buffer.byteLength).toBe(12)
    expect(cloud.colors!.byteOffset).toBe(0)
    expect(cloud.colors!.buffer.byteLength).toBe(3)
  })

  it('rejects a file that is not CEPC', () => {
    expect(() => parseCloudBin(makeCloudBin([[0, 0, 0]], null, { magic: 'GLTF' })))
      .toThrow(CloudBinError)
  })

  it('rejects an unsupported version', () => {
    expect(() => parseCloudBin(makeCloudBin([[0, 0, 0]], null, { version: 2 })))
      .toThrow(/version 2/)
  })

  it('rejects a truncated file rather than reading past the end', () => {
    // header says 2 points, body holds 1
    expect(() => parseCloudBin(makeCloudBin([[0, 0, 0]], null, { count: 2 })))
      .toThrow(/does not match/)
  })

  it('rejects a buffer shorter than the header', () => {
    expect(() => parseCloudBin(new ArrayBuffer(8))).toThrow(/too short/)
  })
})
