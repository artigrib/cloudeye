import * as THREE from 'three'

export interface FloorGridMeta {
  width: number
  height: number
  resolution: number
  origin_x: number
  origin_z: number
}

/** World-space quad exactly covering the occupancy grid's footprint, lying flat on the
 * floor (y=0). UVs are chosen so texture pixel (0,0) - the [ix=0,iz=0] cell, the
 * top-left corner of the row-major buffer buildGridPixels/buildClearancePixels produce
 * (see clearanceRender.ts's `(iz * width + ix) * 4` indexing) - lands at world
 * (origin_x, origin_z), with +U following +X and +V following +Z. Pairs with
 * `texture.flipY = false` in buildFloorTexture below - without both halves of this,
 * three's default flipY would mirror the floor vertically relative to the point cloud
 * and markers, which already sit at their real, unflipped (x, y, z). */
export function buildFloorGeometry(grid: FloorGridMeta): THREE.BufferGeometry {
  const x0 = grid.origin_x
  const z0 = grid.origin_z
  const x1 = grid.origin_x + grid.width * grid.resolution
  const z1 = grid.origin_z + grid.height * grid.resolution

  const positions = new Float32Array([
    x0, 0, z0,
    x1, 0, z0,
    x0, 0, z1,
    x1, 0, z1,
  ])
  const uvs = new Float32Array([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
  ])
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  geometry.setIndex([0, 2, 1, 1, 2, 3])
  geometry.computeVertexNormals()
  return geometry
}

/** Wraps a `buildGridPixels`/`buildClearancePixels` RGBA buffer in a CanvasTexture -
 * `NEAREST` filtering both ways to keep the same blocky, one-pixel-per-cell look the
 * old Canvas2D + `image-rendering: pixelated` 2D view had (this is a deliberate part of
 * the clearance field's design, not a missing smoothing pass - see clearanceRender.ts's
 * own doc comment). `SRGBColorSpace` because these pixels are already gamma-encoded
 * sRGB bytes (via lib/tokens.ts's toRgba), the same as any ordinary color texture -
 * without it three would sample them as linear data and the floor would render too dark. */
export function buildFloorTexture(pixels: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (ctx) ctx.putImageData(new ImageData(pixels, width, height), 0, 0)

  const texture = new THREE.CanvasTexture(canvas)
  texture.flipY = false
  texture.minFilter = THREE.NearestFilter
  texture.magFilter = THREE.NearestFilter
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true
  return texture
}

/** Repaints an existing floor texture in place (e.g. a robot-radius or render-mode
 * change) without touching the geometry - same "mutate, don't rebuild" rule the point
 * cloud's uniforms and the marker colors already follow. */
export function updateFloorTexture(texture: THREE.CanvasTexture, pixels: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): void {
  const canvas = texture.image as HTMLCanvasElement
  const ctx = canvas.getContext('2d')
  if (ctx) ctx.putImageData(new ImageData(pixels, width, height), 0, 0)
  texture.needsUpdate = true
}
