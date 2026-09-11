import * as THREE from 'three/webgpu'
import {
  Discard,
  Fn,
  clamp,
  cos,
  float,
  length,
  log2,
  max,
  mix,
  oneMinus,
  positionView,
  positionWorld,
  pow,
  radians,
  screenDPR,
  select,
  sin,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from 'three/tsl'
import { decimateIndices } from './pointcloudDecimate'
import { POINT_SIZE_MAX_DEVICE_PX, POINT_SIZE_MIN_DEVICE_PX } from './pointSizing'
import { HEIGHT_RAMP } from './tokens'

// WORKAROUND for two compounding three@0.185.1 (r185) upstream bugs that make TSL's own
// documented `instancedBufferAttribute()` helper silently non-functional for a plain
// vec3/vec4 attribute (our case - position/color) under the WebGL2 fallback backend.
// Both verified directly against node_modules/three source (see docs/DECISIONS.md for
// the full writeup); remove this whole block once fixed upstream.
//
// Bug 1 - the `instanced` flag never reaches the node for a non-mat3/mat4 attribute.
// `instancedBufferAttribute()` calls `createBufferAttribute(array, type, stride, offset,
// StaticDrawUsage, true)` (BufferAttributeNode.js). That function's mat3/mat4 branches
// correctly chain `.setInstanced(instanced)`, but its COMMON fallback branch - the one
// taken for every plain vec2/vec3/vec4 attribute, i.e. every point-cloud position/color
// use case - does not:
//   return new BufferAttributeNode( array, type, stride, offset ).setUsage( usage );
// `instanced` is silently dropped; the resulting node's `this.instanced` stays `false`
// no matter which TSL function you called. Confirmed empirically: logging
// `this.instanced` inside `setup()` for our position/color nodes printed `false`.
// Fix: don't call `instancedBufferAttribute()` at all - construct `BufferAttributeNode`
// directly and call `.setInstanced(true)` ourselves (see `myInstancedBufferAttribute`
// below), the same way `createBufferAttribute`'s mat3/mat4 branches do internally.
//
// Bug 2 - even with `instanced` correctly `true`, the WebGL2 fallback backend still
// never advances the buffer per-instance. `BufferAttributeNode.prototype.setup()` wraps
// every attribute (instanced or not) in a plain `THREE.InterleavedBuffer`, and only ever
// sets `this.attribute.isInstancedBufferAttribute = this.instanced` on the ATTRIBUTE -
// never `isInstancedInterleavedBuffer`/`meshPerAttribute` on the underlying BUFFER,
// literally flagged by three's own `// @TODO: Add a possible:
// InstancedInterleavedBufferAttribute` comment at that exact line. WebGLBackend's
// vertex-attribute setup only calls `gl.vertexAttribDivisor()` via one of two branches:
// `isInstancedBufferAttribute && !isInterleavedBufferAttribute` (never true - our
// attribute IS interleaved), or `isInterleavedBufferAttribute &&
// data.isInstancedInterleavedBuffer` (never true either, since that flag is never set).
// Neither fires, divisor is never set, every vertex reads instance 0's data.
// Fix: monkey-patch `BufferAttributeNode.prototype.setup` to set the two missing flags
// on the underlying buffer ourselves afterward - exactly what a real
// `THREE.InstancedInterleavedBuffer` would have (compare
// src/core/InstancedInterleavedBuffer.js). Idempotent (setup() itself early-returns once
// `this.attribute` is set, so this just re-applies harmlessly on any later call).
{
  const BufferAttributeNodeCtor = (THREE as unknown as { BufferAttributeNode: new (...args: TslNode[]) => TslNode })
    .BufferAttributeNode
  const proto = BufferAttributeNodeCtor.prototype
  const originalSetup = proto.setup
  proto.setup = function (this: TslNode, builder: TslNode) {
    const result = originalSetup.call(this, builder)
    if (this.instanced && this.attribute && this.attribute.data) {
      this.attribute.data.isInstancedInterleavedBuffer = true
      if (this.attribute.data.meshPerAttribute === undefined) this.attribute.data.meshPerAttribute = 1
    }
    return result
  }
}

/** Fixed replacement for TSL's `instancedBufferAttribute()` - see the workaround block
 * above for why the real one silently produces a non-instanced node for a plain
 * vec3/vec4 attribute. Two things this must get right, both confirmed empirically
 * against the actual installed build/three.webgpu.js (NOT just node_modules/three/src,
 * which turned out to describe a different code path than what's actually bundled):
 *  1. Call `.setInstanced(true)` ourselves (bug 1 above).
 *  2. Pass the raw TYPED ARRAY, never the source THREE.BufferAttribute object. The
 *     bundled BufferAttributeNode constructor has a fast-path node_modules/three/src
 *     doesn't show: `if (value.isBufferAttribute && value.itemSize <= 4) { this.attribute
 *     = value; this.instanced = value.isInstancedBufferAttribute; }` - i.e. passing an
 *     existing BufferAttribute makes the node adopt it AS-IS and SKIP setup() entirely
 *     (setup() early-returns once `this.attribute` is non-null), so the interleaved-
 *     buffer flags the setup() patch above sets never get a chance to run, and
 *     `.setInstanced(true)` never reaches the actual attribute object the WebGL backend
 *     inspects. Passing `array.array` (not `array`) skips that fast path entirely and
 *     forces the real setup() body to run, which the setup() patch above does fix. */
function myInstancedBufferAttribute(array: THREE.BufferAttribute): TslNode {
  const BufferAttributeNodeCtor = (THREE as unknown as { BufferAttributeNode: new (...args: TslNode[]) => TslNode })
    .BufferAttributeNode
  const type = array.itemSize >= 4 ? 'vec4' : 'vec3'
  return new BufferAttributeNodeCtor(toFloat32AttributeArray(array), type).setInstanced(true)
}

// Bug 3 (real WebGPU backend only - not exercised by the WebGL2 fallback this workaround
// was originally verified against): BufferAttributeNode.setup()'s raw-typed-array branch
// builds `new InterleavedBufferAttribute(buffer, itemSize, offset)` with no `normalized`
// argument (three.webgpu.js, BufferAttributeNode class) - it's unconditionally `false`
// for this path, regardless of what `array.normalized` said. Our color attribute is a
// glTF COLOR_0 accessor - Uint8Array, normalized:true (gpu/stage_export_glb.py writes
// uint8 RGBA) - so WebGPU infers an integer vertex format for it while the TSL shader
// declares it a `vec4` (float), producing "Attribute base type (Uint ...) does not match
// the shader's base type (Float)" and an invalid render pipeline (point cloud never
// draws). Converting to a pre-divided Float32Array ourselves sidesteps the missing-flag
// path entirely - a Float32Array needs no normalization concept, so it always infers a
// matching float vertex format. Positions are already Float32Array/unnormalized and are
// returned by reference (no copy).
function maxValueForNormalizedType(array: ArrayLike<number>): number {
  if (array instanceof Int8Array) return 127
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray) return 255
  if (array instanceof Int16Array) return 32767
  if (array instanceof Uint16Array) return 65535
  if (array instanceof Int32Array) return 2147483647
  if (array instanceof Uint32Array) return 4294967295
  return 1
}

function toFloat32AttributeArray(attr: THREE.BufferAttribute): Float32Array {
  const src = attr.array
  if (src instanceof Float32Array && !attr.normalized) return src
  const divisor = attr.normalized ? maxValueForNormalizedType(src) : 1
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = src[i] / divisor
  return out
}

// TSL's node types are an elaborate overload-resolution system tuned for authoring
// shaders, not for being spelled out across function boundaries - this project doesn't
// run in `strict` mode (see tsconfig), and correctness of a node graph is a runtime
// property (does it compile/render right) far more than a compile-time one. `TslNode`
// is a deliberate escape hatch for internal plumbing; the public API below (BufferAttribute
// params, PointCloudUniforms, etc.) stays strongly typed.
type TslNode = any

export interface DecimatedPointCloud {
  geometry: THREE.BufferGeometry
  /** Whether the source GLB carried a per-vertex COLOR_0 attribute - if not, the point
   * cloud shader falls back to solid white rather than declaring an attribute that
   * doesn't exist. */
  hasColor: boolean
  /** 3 (rgb) or 4 (rgba) - matches whatever the GLB's COLOR_0 accessor used. Kept for
   * introspection; TSL's instancedBufferAttribute() infers vec3/vec4 from the
   * attribute's own itemSize, so nothing downstream needs to branch on this anymore. */
  colorItemSize: number
  originalCount: number
  keptCount: number
  /** false when originalCount was already at or under budget and the source position/
   * color BufferAttributes were reused by reference instead of copied - see
   * buildDecimatedPointCloud's short-circuit. Informational only: disposing the
   * returned geometry is safe either way (three's attribute-GPU-buffer cache is keyed by
   * attribute uuid, not by owning geometry, so freeing a shared attribute's GPU buffer
   * just means it gets re-uploaded, harmlessly, if something else renders it later). */
  decimated: boolean
}

function findPointsObject(root: THREE.Object3D): THREE.Points | null {
  let found: THREE.Points | null = null
  root.traverse((child) => {
    if (!found && (child as THREE.Points).isPoints) found = child as THREE.Points
  })
  return found
}

// This project's GLB is always a points-primitive glTF (see gpu/stage_export_glb.py) -
// position + per-vertex color, no faces. If a triangulated mesh ever gets exported
// instead, `scene.is_reflected` (det(transform) === -1) means normals point inward and
// vertex winding is inverted - that would need handling here (flip normals / winding)
// before rendering. Points have no orientation, so it's a no-op today.
//
// This still reads the GLB's own THREE.Points node purely for its position/color
// BufferAttributes - see createPointCloudSprite below for why the actual render object
// is no longer a THREE.Points (WebGPU/TSL point-size limitation, see docs/DECISIONS.md).
export function buildDecimatedPointCloud(root: THREE.Object3D, maxPoints: number): DecimatedPointCloud | null {
  const points = findPointsObject(root)
  if (!points) return null

  const srcGeom = points.geometry
  const posAttr = srcGeom.getAttribute('position') as THREE.BufferAttribute
  const colorAttr = srcGeom.getAttribute('color') as THREE.BufferAttribute | undefined
  const originalCount = posAttr.count

  const geometry = new THREE.BufferGeometry()

  // After Phase 0 (the GLB itself exports at roughly the client's render budget), this
  // is the common case - a copy of every position/color element into fresh typed arrays
  // would be pure waste, and doubles peak memory right when a large cloud is most
  // memory-constrained. Sharing the source attributes by reference is safe to dispose
  // later - see DecimatedPointCloud.decimated's doc comment.
  if (originalCount <= maxPoints) {
    geometry.setAttribute('position', posAttr)
    if (colorAttr) geometry.setAttribute('color', colorAttr)
    geometry.computeBoundingBox()
    return {
      geometry,
      hasColor: !!colorAttr,
      colorItemSize: colorAttr?.itemSize ?? 3,
      originalCount,
      keptCount: originalCount,
      decimated: false,
    }
  }

  const indices = decimateIndices(originalCount, maxPoints)

  const positions = new Float32Array(indices.length * 3)
  for (let i = 0; i < indices.length; i++) {
    const src = indices[i]
    positions[i * 3] = posAttr.getX(src)
    positions[i * 3 + 1] = posAttr.getY(src)
    positions[i * 3 + 2] = posAttr.getZ(src)
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

  if (colorAttr) {
    const itemSize = colorAttr.itemSize
    const srcArray = colorAttr.array
    const colors = new (srcArray.constructor as new (n: number) => typeof srcArray)(indices.length * itemSize)
    for (let i = 0; i < indices.length; i++) {
      const src = indices[i]
      for (let c = 0; c < itemSize; c++) {
        colors[i * itemSize + c] = srcArray[src * itemSize + c]
      }
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, itemSize, colorAttr.normalized))
  }

  geometry.computeBoundingBox()
  return {
    geometry,
    hasColor: !!colorAttr,
    colorItemSize: colorAttr?.itemSize ?? 3,
    originalCount,
    keptCount: indices.length,
    decimated: true,
  }
}

/** Clips an already-built `DecimatedPointCloud`'s position/color buffers to
 * `isNear(x, z)` (XZ plane, Y ignored) - used by SceneMap3D.tsx to keep the cloud from
 * spilling outside the measured room outline (bootstrap's `room_polygon` + a margin,
 * see msaLayers.isNearRoomPolygonXZ/roomPolygonToCloudFrame) in the MSA-overlaid
 * viewer, without touching what a plain cloud view (no MSA export, no room_polygon)
 * shows - callers simply skip calling this when there's nothing to clip to. Always
 * copies into fresh typed arrays (never shares the source attributes by reference),
 * since the kept point count changes; returns the SAME object (no copy) when nothing
 * was actually clipped, so an unclipped rebuild costs nothing extra. */
export function clipPointCloudToPolygonXZ(
  cloud: DecimatedPointCloud,
  isNear: (x: number, z: number) => boolean,
): DecimatedPointCloud {
  const posAttr = cloud.geometry.getAttribute('position') as THREE.BufferAttribute
  const colorAttr = (cloud.geometry.getAttribute('color') as THREE.BufferAttribute | undefined) ?? null

  const keep: number[] = []
  for (let i = 0; i < posAttr.count; i++) {
    if (isNear(posAttr.getX(i), posAttr.getZ(i))) keep.push(i)
  }
  if (keep.length === posAttr.count) return cloud

  const positions = new Float32Array(keep.length * 3)
  for (let i = 0; i < keep.length; i++) {
    const src = keep[i]
    positions[i * 3] = posAttr.getX(src)
    positions[i * 3 + 1] = posAttr.getY(src)
    positions[i * 3 + 2] = posAttr.getZ(src)
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

  if (colorAttr) {
    const itemSize = colorAttr.itemSize
    const srcArray = colorAttr.array
    const colors = new (srcArray.constructor as new (n: number) => typeof srcArray)(keep.length * itemSize)
    for (let i = 0; i < keep.length; i++) {
      const src = keep[i]
      for (let c = 0; c < itemSize; c++) colors[i * itemSize + c] = srcArray[src * itemSize + c]
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, itemSize, colorAttr.normalized))
  }
  geometry.computeBoundingBox()

  return { ...cloud, geometry, keptCount: keep.length, decimated: true }
}

/** Builds the private 4-vertex unit quad every point-cloud Sprite instance is stamped
 * from. Deliberately NOT `new THREE.Sprite()`'s default geometry: that's a shared
 * module-level singleton (three/src/objects/Sprite.js's `_geometry`), and disposing a
 * shared object on every rebuild would be a correctness bug for any other Sprite in the
 * app. A private geometry per point-cloud Sprite means `.dispose()` on rebuild/teardown
 * correctly frees exactly (and only) this cloud's instanced position/color GPU buffers -
 * see Geometries.js's 'dispose'-event-driven attribute cleanup, which is the only code
 * path that actually frees them. */
function createPointSpriteGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry()
  // prettier-ignore
  const positions = new Float32Array([
    -0.5, -0.5, 0,   0.5, -0.5, 0,   0.5, 0.5, 0,
    -0.5, -0.5, 0,   0.5, 0.5, 0,   -0.5, 0.5, 0,
  ])
  // prettier-ignore
  const uvs = new Float32Array([
    0, 0,   1, 0,   1, 1,
    0, 0,   1, 1,   0, 1,
  ])
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  return geometry
}

const CEILING_FADE_BAND_M = 0.3
const CEILING_FADE_ALPHA_FLOOR = 0.15

// Scratch target for the per-frame renderer.getSize() read in createPointCloudMaterials's
// uHalfCanvasHeightPx uniform - module-level so the uniform's onFrameUpdate callback never
// allocates. Same trick three's own (unexported) `scale` uniform uses internally.
const canvasSizeScratch = new THREE.Vector2()

/** Uniforms both materials created below share by reference (never `.clone()`'d) - see
 * createPointCloudMaterials's doc comment for why that's load-bearing. TSL `uniform()`
 * nodes carry no material/graph binding at construction, so the same node instance can
 * sit in two different materials' node graphs; mutating `.value` updates both.
 *
 * Merge note (main's frontend redesign x this branch's WebGPU/TSL migration, see
 * docs/DECISIONS.md): main replaced the old free-form uHeightCutoff numeric slider and
 * the manual near/far uFadeEnabled toggle with spec 7.3's explicit CeilingMode selector
 * and an always-on far depth-fade, both ported here onto TSL nodes instead of main's raw
 * GLSL - see pointVisibilityAlpha below. */
export interface PointCloudUniforms {
  /** Point size in WORLD METRES - the raw slider value (SceneMap3D's MIN/MAX_POINT_SIZE,
   * 0.003..0.06), NOT pre-multiplied by pixelRatio and NOT pre-clamped. Both the DPR
   * factor and the [1, 32] device-pixel clamp are applied inside `sizeNode` (see
   * createPointCloudMaterials) - PointsNodeMaterial.setupVertexSprite applies
   * `screenDPR` to `sizeNode` UNCONDITIONALLY (three@0.185.1, PointsNodeMaterial.js
   * L109), so multiplying by pixelRatio here too double-counts it, and clamping a
   * metres-scale value to [1, 32] is a no-op that pins every point at a constant 1.0m -
   * both were the actual bug (see docs/DECISIONS.md). Mirrors
   * classicPointCloudMaterial.ts's `uSize`/`uScale` pair; lib/pointSizing.ts holds the
   * shared formula both renderers must produce, pinned by pointSizing.test.ts. */
  uSize: TslNode
  /** 0 = visible (nothing hidden), 1 = hidden (discard above uCeilingY), 2 = fade
   * (gradually dims to CEILING_FADE_ALPHA_FLOOR over CEILING_FADE_BAND_M) - spec 7.3. */
  uCeilingMode: TslNode
  uCeilingY: TslNode
  /** Far depth-fade (spec 7, point 3) - always active, not user-facing: points beyond
   * uFadeEnd fade to invisible, so the cloud dissolves rather than clipping hard at the
   * far plane. Set from the room's own footprint via setPointCloudFadeRange, never a
   * fixed constant. */
  uFadeStart: TslNode
  uFadeEnd: TslNode
  /** 0 = photo (per-vertex colour), 1 = height (5-stop OKLCH ramp from lib/tokens.ts's
   * HEIGHT_RAMP, over uHeightMin..uHeightMax) - see heightColor() below. */
  uColorMode: TslNode
  uHeightMin: TslNode
  uHeightMax: TslNode
  uH0: TslNode
  uH1: TslNode
  uH2: TslNode
  uH3: TslNode
  uH4: TslNode
}

/** [l, c, h] triples for the shader's uH0..uH4 uniforms - h in degrees, matching
 * tokens.ts's OklchColor shape (oklchToSrgb takes the same values apart from the scalar
 * vs. TSL-vec3 packing). */
function heightRampUniforms(): THREE.Vector3[] {
  return HEIGHT_RAMP.map(({ l, c, h }) => new THREE.Vector3(l, c, h))
}

function createPointCloudUniforms(): PointCloudUniforms {
  const [h0, h1, h2, h3, h4] = heightRampUniforms()
  return {
    uSize: uniform(1),
    uCeilingMode: uniform(0),
    uCeilingY: uniform(Infinity),
    uFadeStart: uniform(Infinity),
    uFadeEnd: uniform(Infinity),
    uColorMode: uniform(0),
    uHeightMin: uniform(0),
    uHeightMax: uniform(1),
    uH0: uniform(h0),
    uH1: uniform(h1),
    uH2: uniform(h2),
    uH3: uniform(h3),
    uH4: uniform(h4),
  }
}

/** Coverage mask for one splat. SQUARE - the whole quad is covered, so this is a
 * constant 1 with no per-fragment work at all.
 *
 * It used to carve a circle out of the quad: `smoothstep(0.45, 0.5, length(uv - 0.5))`
 * plus a `Discard` for the corners. That is three texture-free ALU ops and a discard on
 * every fragment of every splat, and the discard is the expensive half - a shader that
 * may discard cannot use early depth rejection, so every fragment runs the full shader
 * before the depth test. With the camera INSIDE the cloud that is the dominant cost:
 * near points hit the [1, 32] device-pixel clamp (see pointSizing.ts), so a single splat
 * can cover 32x32 = 1024 fragments and the whole cloud covers the viewport many times
 * over. Measured on the 15 fps hero, camera at room centre at 1.2 m, orbit drag on this
 * (SwiftShader, GPU-less) stand: 0.94 fps with round splats.
 *
 * Square splats are also what the owner asked for, and at these sizes the round mask was
 * buying very little: below ~4 device px the smoothstep band is most of the splat and
 * the circle reads as a blurry square anyway.
 *
 * Kept as a function, exported, and still called from both the beauty and the depth
 * (EDL) color nodes: the two must agree on which fragments exist, and routing that
 * through one place is what guarantees it. The EDL depth material relies on being able
 * to call this for its mask channel.
 */
export function pointShapeAlpha(): TslNode {
  return float(1)
}

/** Ceiling-mode discard/fade + far depth-fade, shared between the beauty and depth (EDL)
 * color nodes below. If these ever drifted apart, a point hidden by the ceiling mode (or
 * faded to invisible) could still write into the EDL depth/mask target, darkening the
 * frame under geometry the beauty pass never drew. Must be called from inside a Fn()
 * (Discard() defers via .toStack(), which needs an active node-builder stack). Returns
 * the alpha node.
 *
 * Ported from main's raw-GLSL uCeilingMode/uCeilingY if/else and unconditional far-fade
 * multiply (see docs/DECISIONS.md's merge note) onto this branch's TSL pipeline -
 * replaces the old free-form uHeightCutoff + manual near/far uFadeEnabled toggle this
 * function used to implement. */
export function pointVisibilityAlpha(u: PointCloudUniforms): TslNode {
  // Ceiling (spec 7.3): hidden (mode 1) discards outright; fade (mode 2) dims to
  // CEILING_FADE_ALPHA_FLOOR over CEILING_FADE_BAND_M instead of disappearing; visible
  // (mode 0) is a no-op on both branches below.
  Discard(
    positionWorld.y.greaterThan(u.uCeilingY).and(u.uCeilingMode.greaterThan(0.5)).and(u.uCeilingMode.lessThan(1.5)),
  )
  const ceilingFadeT = clamp(positionWorld.y.sub(u.uCeilingY).div(CEILING_FADE_BAND_M), 0, 1)
  const ceilingAlpha = select(
    u.uCeilingMode.greaterThan(1.5),
    mix(1, CEILING_FADE_ALPHA_FLOOR, ceilingFadeT),
    1,
  )

  // Far depth-fade (spec 7, point 3) - always on, not a user control (see
  // setPointCloudFadeRange).
  const distFade = oneMinus(smoothstep(u.uFadeStart, u.uFadeEnd, length(positionView)))

  const alpha = ceilingAlpha.mul(distFade)
  Discard(alpha.lessThanEqual(0.001))
  return alpha
}

// OKLCH -> linear sRGB -> sRGB, mirroring lib/tokens.ts's oklchToSrgb exactly, so the
// height ramp's colors match the design tokens bit for bit rather than approximating
// them (replaces this branch's old viridis() polynomial fit - main's redesign moved the
// height ramp onto the shared design-token palette). Plain functions (not TSL Fn()s) -
// they only compose nodes, don't need their own deferred shader-stack context, and are
// called from within the colorNode Fn() below regardless.
function srgbChannel(c: TslNode): TslNode {
  const cc = clamp(c, 0, 1)
  return select(cc.lessThanEqual(0.0031308), cc.mul(12.92), pow(cc, 1 / 2.4).mul(1.055).sub(0.055))
}

function linearToSrgb(c: TslNode): TslNode {
  return vec3(srgbChannel(c.x), srgbChannel(c.y), srgbChannel(c.z))
}

function oklchToLinearSrgb(oklch: TslNode): TslNode {
  const L = oklch.x
  const C = oklch.y
  const h = radians(oklch.z)
  const a = C.mul(cos(h))
  const b = C.mul(sin(h))
  const l_ = L.add(a.mul(0.3963377774)).add(b.mul(0.2158037573))
  const m_ = L.sub(a.mul(0.1055613458)).sub(b.mul(0.0638541728))
  const s_ = L.sub(a.mul(0.0894841775)).sub(b.mul(1.291485548))
  const l3 = l_.mul(l_).mul(l_)
  const m3 = m_.mul(m_).mul(m_)
  const s3 = s_.mul(s_).mul(s_)
  return vec3(
    l3.mul(4.0767416621).sub(m3.mul(3.3077115913)).add(s3.mul(0.2309699292)),
    l3.mul(-1.2684380046).add(m3.mul(2.6097574011)).sub(s3.mul(0.3413193965)),
    l3.mul(-0.0041960863).sub(m3.mul(0.7034186147)).add(s3.mul(1.707614701)),
  )
}

// Interpolates the 5-stop ramp in OKLCH space (lerping L/C/hue directly) via nested
// select()s - TSL has no if/else statement inside a Fn(), so this is the branchless
// equivalent of main's `t < 1.0 ? ... : t < 2.0 ? ... : ...` chain - then converts once
// at the end; mixing already-gamma-corrected RGB endpoints instead would produce the
// muddy banding OKLCH interpolation is specifically meant to avoid.
function heightColor(t: TslNode, u: PointCloudUniforms): TslNode {
  const t4 = clamp(t, 0, 1).mul(4)
  const seg0 = mix(u.uH0, u.uH1, t4)
  const seg1 = mix(u.uH1, u.uH2, t4.sub(1))
  const seg2 = mix(u.uH2, u.uH3, t4.sub(2))
  const seg3 = mix(u.uH3, u.uH4, t4.sub(3))
  const oklch = select(t4.lessThan(1), seg0, select(t4.lessThan(2), seg1, select(t4.lessThan(3), seg2, seg3)))
  return linearToSrgb(oklchToLinearSrgb(oklch))
}

export interface PointCloudMaterials {
  /** What's actually drawn to the canvas - per-vertex/height color, round sprites,
   * height-cutoff + fade. */
  beauty: InstanceType<typeof THREE.PointsNodeMaterial>
  /** Depth-only variant for the EDL prepass (see scene3dEdl.ts): same point size/shape
   * and the same height-cutoff/fade discards as `beauty` (by sharing the same discard
   * functions and the `uniforms` object, not by convention), writing view-space log-
   * depth instead of color. Never added to the scene directly - scene3dEdl swaps it onto
   * the point-cloud Sprite for one sub-pass, then swaps back. */
  depth: InstanceType<typeof THREE.PointsNodeMaterial>
  uniforms: PointCloudUniforms
}

/** Builds the point cloud's beauty material and a depth-only twin for EDL, both as
 * `PointsNodeMaterial`s meant to be used on a `THREE.Sprite` with `sprite.count` set for
 * GPU instancing - see createPointCloudSprite. NOT usable on a `THREE.Points` object:
 * WebGPU hardcodes point primitives to 1px, and PointsNodeMaterial's own sizing logic
 * (`setupVertexSprite`) only activates for `isPoints !== true` objects - see
 * docs/DECISIONS.md for the citations. `positionNode` is set to the SAME
 * instancedBufferAttribute node on both materials (one underlying GPU buffer, read
 * identically by both draws). `beauty` and `depth` share one `uniforms` object of TSL
 * `uniform()` nodes BY REFERENCE - this is what lets every existing point-size/height-
 * slice/fade effect in SceneMap3D drive both materials with zero extra wiring.
 *
 * `sizeAttenuation` is deliberately FALSE on both materials below - perspective
 * attenuation, the device-pixel-ratio factor, and the [1, 32] device-px clamp are all
 * computed in `sizeNode` here instead, because `PointsNodeMaterial` offers no hook into
 * its own internal sizing to clamp AFTER attenuation+DPR (see the comment on `sizeNode`
 * below) - and letting it attenuate too, on top of that, would attenuate twice. */
export function createPointCloudMaterials(
  posAttr: THREE.BufferAttribute,
  colorAttr: THREE.BufferAttribute | null,
): PointCloudMaterials {
  const uniforms = createPointCloudUniforms()
  const positionNode = myInstancedBufferAttribute(posAttr)
  const hasColor = colorAttr !== null

  // Half the canvas height in LOGICAL (CSS) pixels, refreshed once per frame - our OWN
  // copy of the exact uniform PointsNodeMaterial keeps privately (`scale` in
  // PointsNodeMaterial.js, unexported) for its own sizeAttenuation math, needed here
  // because sizeAttenuation is off and we replicate that math ourselves below. Not
  // derived from TSL's `screenSize` node: `screenSize` swaps to the bound render
  // target's dimensions while a render target is active (ScreenNode.update), which
  // would silently diverge from the beauty pass once scene3dEdl.ts's depth prepass
  // binds a target - and even without a target, `getDrawingBufferSize()` floors
  // `logical*pixelRatio` while SceneMap3D passes fractional CSS sizes to
  // `renderer.setSize()`, so a screenSize-derived value isn't bit-exact with this one.
  // Module-scope `canvasSizeScratch` avoids allocating a Vector2 every frame, same trick
  // three's own `scale` uniform uses internally.
  const uHalfCanvasHeightPx = uniform(1)
  uHalfCanvasHeightPx.onFrameUpdate((frame) => {
    const renderer = frame.renderer as THREE.WebGPURenderer | null
    if (renderer) uHalfCanvasHeightPx.value = 0.5 * renderer.getSize(canvasSizeScratch).y
  })

  // Point size, in full. See lib/pointSizing.ts for this exact formula in plain JS
  // (pinned by pointSizing.test.ts) and classicPointCloudMaterial.ts's vertex shader for
  // the WebGL path this must match pixel-for-pixel at every DPR.
  //
  // PointsNodeMaterial.setupVertexSprite (three@0.185.1, L105-119) does, regardless of
  // `sizeAttenuation`:
  //   pointSize = vec2(sizeNode).mul(screenDPR)     // L107+L109, UNCONDITIONAL
  //   if (sizeAttenuation) pointSize = pointSize.mul(scale.div(-positionView.z))  // L113-117, gated
  // With sizeAttenuation off, only the unconditional `.mul(screenDPR)` still runs on
  // whatever we return - so this expression computes the full clamped device-pixel
  // target ourselves, then divides by screenDPR once so the material's own multiply
  // cancels it back out exactly. (The previous, buggy version fed a raw, unattenuated,
  // un-DPR'd metres value straight into `clamp(uSize, 1, 32)` - always >= 1 for every
  // point size on the slider, pinning every point at a constant 1m. See
  // docs/DECISIONS.md.) Shared by reference between beauty/depth for the same reason
  // `uniforms` is.
  const sizeNode = clamp(
    uniforms.uSize.mul(screenDPR).mul(uHalfCanvasHeightPx).div(max(positionView.z.negate(), 1e-6)),
    float(POINT_SIZE_MIN_DEVICE_PX),
    float(POINT_SIZE_MAX_DEVICE_PX),
  ).div(screenDPR)

  const beauty = new THREE.PointsNodeMaterial()
  beauty.positionNode = positionNode
  beauty.sizeNode = sizeNode
  beauty.sizeAttenuation = false
  // Always transparent (unlike this branch's old fade-off-means-opaque optimization):
  // the ceiling fade and far depth-fade in pointVisibilityAlpha are both unconditional
  // now (main's redesign, spec 7.3/7.3), so alpha is no longer reliably 1.0 for every
  // surviving fragment the way it was when fade was a manual, off-by-default toggle.
  beauty.transparent = true
  beauty.depthWrite = true
  beauty.depthTest = true
  beauty.colorNode = Fn(() => {
    const shapeAlpha = pointShapeAlpha()
    const alpha = pointVisibilityAlpha(uniforms).mul(shapeAlpha)
    const base: TslNode = hasColor ? vec3(myInstancedBufferAttribute(colorAttr!) as TslNode) : vec3(1)
    const t = clamp(
      positionWorld.y.sub(uniforms.uHeightMin).div(max(uniforms.uHeightMax.sub(uniforms.uHeightMin), 1e-6)),
      0,
      1,
    )
    // uColorMode is always exactly 0 or 1 (see SceneMap3D's colorMode effect), so a
    // straight mix() is equivalent to the old `if (uColorMode > 0.5)` branch.
    const color = mix(base, heightColor(t, uniforms), uniforms.uColorMode)
    return vec4(color, alpha)
  })()

  const depth = new THREE.PointsNodeMaterial()
  depth.positionNode = positionNode
  depth.sizeNode = sizeNode
  depth.sizeAttenuation = false
  depth.transparent = false
  depth.depthWrite = true
  depth.depthTest = true
  // R = log2(view-space depth), G = 1 (this pixel is a point-cloud fragment - lets the
  // EDL composite skip taps that land on background or on an opaque occluder rendered
  // into the same target, rather than trying to disambiguate via a magic depth value).
  depth.colorNode = Fn(() => {
    pointShapeAlpha()
    pointVisibilityAlpha(uniforms)
    return vec4(log2(max(positionView.z.negate(), 1e-4)), 1, 0, 1)
  })()

  return { beauty, depth, uniforms }
}

/** Sets the automatic far depth-fade range (spec 7, point 3) from the room's own XZ
 * footprint diagonal - a large room doesn't fade out points still well inside it, a
 * small room doesn't need to render past its own walls. Always on (no enable flag,
 * unlike the manual near/far toggle this replaced) - see PointCloudUniforms's
 * uFadeStart/uFadeEnd doc comment. The uniforms only need writing once since
 * `beauty`/`depth` share the object by reference. */
export function setPointCloudFadeRange(materials: Pick<PointCloudMaterials, 'uniforms'>, roomDiagonal: number): void {
  materials.uniforms.uFadeStart.value = roomDiagonal * 0.75
  materials.uniforms.uFadeEnd.value = roomDiagonal * 1.4
}

/** Builds the point-cloud Sprite: a private unit-quad geometry (see
 * createPointSpriteGeometry) instanced `keptCount` times via `sprite.count`.
 *
 * **`frustumCulled` is false, permanently, and it is not an oversight.** three culls a
 * Sprite with `Frustum.intersectsSprite`, which is (three 0.185.1, Frustum.js):
 *
 *     _sphere.center.set(0, 0, 0)
 *     _sphere.radius = 0.7071067811865476 + defaultSpriteCenter.distanceTo(sprite.center)
 *     _sphere.applyMatrix4(sprite.matrixWorld)
 *     return this.intersectsSphere(_sphere)
 *
 * It never reads `geometry.boundingSphere`. So the bounds this function used to assign
 * were inert, and `frustumCulled = true` was testing a fixed 0.707 m sphere sitting at
 * the sprite's own world origin - which is not where the instances are. Zoom in far
 * enough that that tiny sphere leaves the frustum and the ENTIRE instanced draw is
 * skipped: the cloud vanishes while the floor, the contours and the markers (real
 * objects, with real bounds) keep drawing. Reported on own_0902_140657 in Photo mode -
 * zoom in, cloud gone; zoom out, cloud back - and reproduced by
 * demo/probe_viewer_bugs.py's `incloud`.
 *
 * Recomputing the bounding sphere after every positions update - the obvious other fix -
 * cannot help, because nothing reads it. It would leave a plausible-looking line of code
 * standing over a live bug.
 *
 * What this costs: one instanced draw call issued when the cloud is genuinely off-screen,
 * which the GPU then clips per-primitive for free. Getting that saving back means testing
 * the cloud's real Box3 against the camera frustum ourselves and setting `.visible` - a
 * separate change, and worth doing only against a measurement that shows it matters. */
export function createPointCloudSprite(
  material: InstanceType<typeof THREE.PointsNodeMaterial>,
  keptCount: number,
  cloudBounds?: THREE.Box3 | null,
) {
  const sprite = new THREE.Sprite(material)
  sprite.geometry = createPointSpriteGeometry()
  // Kept on the geometry for anything that wants to know where the cloud is (probes read
  // it, and a future explicit cull would need it). NOT read by three's own culling.
  if (cloudBounds && !cloudBounds.isEmpty()) {
    sprite.geometry.boundingBox = cloudBounds.clone()
    sprite.geometry.boundingSphere = cloudBounds.getBoundingSphere(new THREE.Sphere())
    // The splats stick out past the instance positions by up to half a point size.
    sprite.geometry.boundingSphere.radius *= 1.05
  }
  sprite.frustumCulled = false
  sprite.count = keptCount
  return sprite
}
