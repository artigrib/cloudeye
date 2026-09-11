/** CEPC v1 - the binary point cloud `GET /api/scenes/{id}/cloud` serves.
 *
 * Why a second cloud format at all, next to the points-primitive GLB: the GLB path exists
 * because the pipeline's own exporter writes glTF, and it costs a full GLTFLoader parse
 * (JSON header, accessor indirection, three's scene graph) on the MAIN thread. This format
 * is two contiguous typed-array slices, so a worker can parse it with zero per-point work
 * and hand the buffers over as transferables - the main thread only ever receives them.
 *
 * Layout, little-endian:
 *     0   char[4]    magic "CEPC"
 *     4   uint32     version = 1
 *     8   uint32     count
 *    12   uint32     flags        bit0 = rgb present
 *    16   float32[6] bbox  xmin, ymin, zmin, xmax, ymax, zmax
 *    40   float32[3*count]  xyz
 *        uint8[3*count]     rgb, immediately after xyz (12*count is a multiple of 4,
 *                           so the Float32Array view is always aligned)
 *
 * Coordinates in the file are the nvblox slice frame - (u, v, height above the fitted
 * floor), Z-up with the floor at 0, the same convention as
 * nvblox_v2/isaac_export/<scene>/transform.json. `zUpToYUp` below maps that onto the frame the
 * rest of the viewer uses (Y up, floor y = 0) as a PROPER rotation - (x, y, z) = (u, h, -v),
 * det +1. Using +v instead would mirror the scene, which is exactly what
 * `alignment.json.is_reflection` exists to catch.
 */

export const CLOUD_BIN_MAGIC = 0x43_45_50_43 // "CEPC" read big-endian as one uint32
export const CLOUD_BIN_HEADER_BYTES = 40

export interface CloudBin {
  /** Y-up, floor at 0, ready for THREE.BufferAttribute('position', …, 3). */
  positions: Float32Array
  /** uint8 RGB, to be attached with `normalized = true`; null when flags bit0 is clear. */
  colors: Uint8Array | null
  count: number
  /** bbox as stored (Z-up), min then max. */
  bboxZUp: Float32Array
}

export class CloudBinError extends Error {}

/** Rotates a Z-up (u, v, h) triple onto the viewer's Y-up frame, in place. */
export function zUpToYUp(xyz: Float32Array): Float32Array {
  for (let i = 0; i < xyz.length; i += 3) {
    const v = xyz[i + 1]
    xyz[i + 1] = xyz[i + 2]
    xyz[i + 2] = -v
  }
  return xyz
}

export function parseCloudBin(buffer: ArrayBuffer): CloudBin {
  if (buffer.byteLength < CLOUD_BIN_HEADER_BYTES) {
    throw new CloudBinError(`too short: ${buffer.byteLength} bytes, header alone is ${CLOUD_BIN_HEADER_BYTES}`)
  }
  const view = new DataView(buffer)
  if (view.getUint32(0, false) !== CLOUD_BIN_MAGIC) {
    throw new CloudBinError('not a CEPC file (bad magic)')
  }
  const version = view.getUint32(4, true)
  if (version !== 1) throw new CloudBinError(`unsupported CEPC version ${version}`)

  const count = view.getUint32(8, true)
  const hasRgb = (view.getUint32(12, true) & 1) === 1
  const expected = CLOUD_BIN_HEADER_BYTES + count * (hasRgb ? 15 : 12)
  if (buffer.byteLength !== expected) {
    throw new CloudBinError(`length ${buffer.byteLength} does not match ${count} points (expected ${expected})`)
  }

  const bboxZUp = new Float32Array(buffer.slice(16, 40))
  // Copied, not viewed: the positions are rewritten in place by zUpToYUp, and a view onto
  // the response buffer would leave the untouched tail (rgb) sharing it, so the two
  // transferables could not be detached independently.
  const positions = new Float32Array(buffer.slice(CLOUD_BIN_HEADER_BYTES, CLOUD_BIN_HEADER_BYTES + count * 12))
  const colors = hasRgb
    ? new Uint8Array(buffer.slice(CLOUD_BIN_HEADER_BYTES + count * 12))
    : null

  return { positions: zUpToYUp(positions), colors, count, bboxZUp }
}
