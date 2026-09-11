import * as THREE from 'three'
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js'
import { DEFAULT_EDL_OPACITY, DEFAULT_EDL_STRENGTH, edlRadiusPx } from './edlSettings'

/** Eye-Dome Lighting: a screen-space depth-discontinuity darkening pass for the point
 * cloud, standing in for SSAO (which needs normals a point cloud doesn't have - see
 * docs/DECISIONS.md / the plan this implements for why SSAO/GTAO were rejected).
 *
 * Deliberately NOT an EffectComposer/RenderPass/ShaderPass pipeline: routing the beauty
 * render through a render target hits the constraint documented at SceneMap3D.tsx's
 * mount effect (never touch renderer.outputColorSpace/toneMapping/ColorManagement) -
 * inside an RT, three encodes to the *working* color space regardless of
 * renderer.outputColorSpace, so the point shader's raw sRGB bytes and every built-in
 * material's <colorspace_fragment> chunk would land in the buffer in two different
 * color spaces with no way to reconcile them at the final blit.
 *
 * Instead: render the frame to the canvas exactly as today (untouched), then multiply-
 * blend a fullscreen EDL quad on top. EDL is purely a darkening term, so a multiply is
 * all it needs, and the canvas pixels it multiplies against are never touched.
 */
// createEdlPass always returns null now (see its doc comment) - the methods below never
// actually execute, but SceneMap3D.tsx's call sites still need to type-check against
// whatever it currently has in hand (a THREE.WebGPURenderer, a THREE.Sprite point cloud,
// a PointsNodeMaterial depth material - none of which existed when this pass was
// written for classic WebGLRenderer). `unknown` here - cast back to the real WebGL-era
// types inside the implementation below - documents that these signatures are
// provisional pending the TSL rewrite, rather than pretending precision across a
// boundary that's currently unreachable.
export interface EdlPass {
  /** Renders occluders (everything in `scene` except `points` and anything in
   * `excluded`) as opaque black via `scene.overrideMaterial`, then the point cloud via
   * its depth material (set through `setPointMaterial`) - into this pass's own
   * off-screen target. Call once per frame, BEFORE the normal beauty render. No-op if
   * `setPointMaterial(null)` (no point cloud built yet). */
  renderPrepass(
    renderer: unknown,
    scene: THREE.Scene,
    camera: THREE.Camera,
    points: unknown,
    excluded: ReadonlyArray<THREE.Object3D | null>,
  ): void
  /** Draws the multiply-blended EDL quad on top of whatever is currently in the default
   * framebuffer (i.e. call this AFTER the normal beauty render). */
  composite(renderer: unknown): void
  /** Resizes the internal render target to the renderer's current drawing-buffer size
   * and re-syncs the composite shader's texel size. Call from the same ResizeObserver
   * that resizes the main renderer. */
  setSize(renderer: unknown): void
  /** Keeps the neighbor-tap radius in sync with the point-size slider - see
   * edlSettings.edlRadiusPx for why this must track point size or EDL silently stops
   * doing anything at large sizes. */
  setPointSize(pointSizeWorld: number, referencePointSize: number, pixelRatio: number): void
  /** The point cloud's depth-only material (PointCloudMaterials.depth from
   * scene3dPoints.ts), or null while no point cloud exists yet (before the GLB loads, or
   * after its cleanup runs). */
  setPointMaterial(material: unknown): void
  setStrength(k: number): void
  setOpacity(a: number): void
  dispose(): void
}

const compositeVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`

// 8-tap log-depth neighbor comparison (Potree/CloudCompare's EDL, with two departures:
// a mask channel instead of a magic "depth==0 means background" sentinel - log2(1m)==0
// would otherwise collide with the single most common depth in a room-scale scene - and
// no background halo term, since the mask already tells taps landing on background or
// on an opaque occluder to contribute nothing). See edlSettings.ts for the strength/
// radius tuning this depends on.
const compositeFragmentShader = `
  uniform sampler2D uEdlMap;
  uniform vec2 uTexel;
  uniform float uRadius;
  uniform float uStrength;
  uniform float uOpacity;
  varying vec2 vUv;

  float tap(vec2 dir, float d) {
    vec2 s = texture2D(uEdlMap, vUv + uRadius * uTexel * dir).xy;
    return s.y > 0.5 ? max(0.0, d - s.x) : 0.0;
  }

  void main() {
    vec2 c = texture2D(uEdlMap, vUv).xy;
    if (c.y < 0.5) {
      gl_FragColor = vec4(1.0);
      return;
    }
    float d = c.x;
    const float K = 0.70710678;
    float response = (
        tap(vec2( 1.0,  0.0), d) + tap(vec2(   K,    K), d)
      + tap(vec2( 0.0,  1.0), d) + tap(vec2(  -K,    K), d)
      + tap(vec2(-1.0,  0.0), d) + tap(vec2(  -K,   -K), d)
      + tap(vec2( 0.0, -1.0), d) + tap(vec2(   K,   -K), d)
    ) * 0.125;

    float shade = mix(1.0, exp(-uStrength * response), uOpacity);
    gl_FragColor = vec4(vec3(shade), 1.0);
  }
`

/** Builds an EdlPass, or returns null if either this isn't a classic WebGLRenderer (see
 * below) or the device can't render to a half-float target
 * (EXT_color_buffer_half_float / EXT_color_buffer_float) - the latter is universal on
 * WebGL2 in practice, but the app already runs on a bare WebGL2 context (see
 * SceneMap3D's old LineMaterial/Line2 usage) so it was a real, not theoretical, guard.
 * Callers should disable the EDL toggle in the UI when this returns null.
 *
 * TODO(webgpu-migration): this whole pass is raw GLSL (a THREE.ShaderMaterial composite
 * quad, scene.overrideMaterial occluders, THREE.WebGLRenderTarget) and only ever worked
 * under a classic WebGLRenderer. Since the app now exclusively constructs
 * THREE.WebGPURenderer (see SceneMap3D's mount effect), this always returns null -
 * EDL is disabled pending a TSL rewrite (THREE.RenderPipeline, not the deprecated
 * PostProcessing), deferred as its own pass alongside GPU compute decimation and GLB
 * quantization - see docs/DECISIONS.md. Accepting `unknown` rather than
 * THREE.WebGLRenderer so a WebGPURenderer caller doesn't need an unsafe cast. */
export function createEdlPass(renderer: unknown): EdlPass | null {
  if (typeof renderer !== 'object' || renderer === null || (renderer as { isWebGPURenderer?: boolean }).isWebGPURenderer === true) {
    return null
  }
  const glRenderer = renderer as THREE.WebGLRenderer

  const gl = glRenderer.getContext()
  const hasHalfFloat =
    gl.getExtension('EXT_color_buffer_half_float') !== null || gl.getExtension('EXT_color_buffer_float') !== null
  if (!hasHalfFloat) return null

  const size = glRenderer.getDrawingBufferSize(new THREE.Vector2())
  const rt = new THREE.WebGLRenderTarget(Math.max(1, size.x), Math.max(1, size.y), {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    generateMipmaps: false,
    depthBuffer: true,
    stencilBuffer: false,
  })

  // Depth-only occluder pass: replaces every non-point material in the scene so the
  // prepass gets real depth from the robot/markers/grid without caring what their real
  // materials are. Color is irrelevant except that it must read as mask=0 (rgb all
  // zero) - see the module doc comment.
  const occluderMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 })

  const compositeUniforms = {
    uEdlMap: { value: rt.texture },
    uTexel: { value: new THREE.Vector2(1 / rt.width, 1 / rt.height) },
    uRadius: { value: 1.4 },
    uStrength: { value: DEFAULT_EDL_STRENGTH },
    uOpacity: { value: DEFAULT_EDL_OPACITY },
  }
  const compositeMaterial = new THREE.ShaderMaterial({
    uniforms: compositeUniforms,
    vertexShader: compositeVertexShader,
    fragmentShader: compositeFragmentShader,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    // A multiply blend using factors legal since WebGL1 (GL_SRC_COLOR is only legal as
    // a *source* factor pre-WebGL2, so DstColorFactor-as-src + ZeroFactor-as-dst is used
    // rather than the equivalent SrcColorFactor-as-dst).
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.DstColorFactor,
    blendDst: THREE.ZeroFactor,
  })
  const quad = new FullScreenQuad(compositeMaterial)

  let pointMaterial: THREE.ShaderMaterial | null = null
  let pointSizeWorld = 0
  let referencePointSize = 1
  let pixelRatio = 1

  const tmpClearColor = new THREE.Color()

  return {
    renderPrepass(rendererArg, scene, camera, pointsArg, excluded) {
      if (!pointMaterial) return
      const renderer = rendererArg as THREE.WebGLRenderer
      const points = pointsArg as THREE.Points

      const prevClearColor = renderer.getClearColor(tmpClearColor)
      const prevClearAlpha = renderer.getClearAlpha()
      const prevAutoClear = renderer.autoClear
      const prevOverrideMaterial = scene.overrideMaterial
      const prevPointMaterial = points.material
      const prevPointVisible = points.visible
      const prevExcludedVisible = excluded.map((o) => o?.visible ?? null)

      try {
        renderer.setRenderTarget(rt)
        renderer.setClearColor(0x000000, 0)
        renderer.autoClear = true

        points.visible = false
        for (const o of excluded) if (o) o.visible = false
        scene.overrideMaterial = occluderMaterial
        renderer.render(scene, camera)

        scene.overrideMaterial = null
        renderer.autoClear = false
        points.visible = true
        points.material = pointMaterial
        renderer.render(points, camera)
      } finally {
        points.material = prevPointMaterial
        points.visible = prevPointVisible
        for (let i = 0; i < excluded.length; i++) {
          const o = excluded[i]
          if (o && prevExcludedVisible[i] !== null) o.visible = prevExcludedVisible[i]!
        }
        scene.overrideMaterial = prevOverrideMaterial
        renderer.autoClear = prevAutoClear
        renderer.setClearColor(prevClearColor, prevClearAlpha)
        renderer.setRenderTarget(null)
      }
    },

    composite(rendererArg) {
      const renderer = rendererArg as THREE.WebGLRenderer
      // renderer.render() clears the target whenever autoClear is true, regardless of
      // what's rendering (WebGLBackground.render() clears unconditionally on autoClear,
      // and a parentless quad's "scene" isn't a THREE.Scene so it never opts out via a
      // background). Without this guard, drawing the EDL quad would wipe out the beauty
      // frame requestRenderOnce just rendered to the same canvas.
      const prevAutoClear = renderer.autoClear
      renderer.autoClear = false
      quad.render(renderer)
      renderer.autoClear = prevAutoClear
    },

    setSize(rendererArg) {
      const renderer = rendererArg as THREE.WebGLRenderer
      const s = renderer.getDrawingBufferSize(new THREE.Vector2())
      const w = Math.max(1, Math.round(s.x))
      const h = Math.max(1, Math.round(s.y))
      rt.setSize(w, h)
      compositeUniforms.uTexel.value.set(1 / w, 1 / h)
      pixelRatio = renderer.getPixelRatio()
      compositeUniforms.uRadius.value = edlRadiusPx(pointSizeWorld, referencePointSize, pixelRatio)
    },

    setPointSize(nextPointSizeWorld, nextReferencePointSize, nextPixelRatio) {
      pointSizeWorld = nextPointSizeWorld
      referencePointSize = nextReferencePointSize
      pixelRatio = nextPixelRatio
      compositeUniforms.uRadius.value = edlRadiusPx(pointSizeWorld, referencePointSize, pixelRatio)
    },

    setPointMaterial(material) {
      pointMaterial = material as THREE.ShaderMaterial | null
    },

    setStrength(k) {
      compositeUniforms.uStrength.value = k
    },

    setOpacity(a) {
      compositeUniforms.uOpacity.value = a
    },

    dispose() {
      rt.dispose()
      occluderMaterial.dispose()
      compositeMaterial.dispose()
      quad.dispose()
    },
  }
}
