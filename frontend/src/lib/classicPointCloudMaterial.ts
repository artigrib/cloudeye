import * as THREE from 'three'
import { HEIGHT_RAMP } from './tokens'

/** Standalone classic-GLSL point-cloud material for the two lightweight, WebGLRenderer-only
 * consumers that deliberately never adopted the WebGPU/TSL pipeline in scene3dPoints.ts:
 * LandingHeroScene.tsx (a decorative marketing-page widget that shouldn't require WebGPU
 * support) and scenePreview.ts (a synchronous throwaway-canvas thumbnail capture, which
 * WebGPURenderer's async render/compile model doesn't fit). Split out from scene3dPoints.ts
 * when that module moved entirely to TSL/PointsNodeMaterial (WebGPU-only) - see
 * docs/DECISIONS.md. Kept verbatim (same shader math/uniform names) so behavior is unchanged
 * from before that split. */


/** Uniforms the point-cloud material exposes for live tweaking (point size, ceiling
 * handling, colour mode) without ever rebuilding the geometry - see SceneMap3D's
 * point-size/ceiling/colour effects, which just mutate these in place. */
export interface PointCloudUniforms {
  [key: string]: THREE.IUniform
  uSize: THREE.IUniform<number>
  uScale: THREE.IUniform<number>
  /** 0 = visible (nothing hidden), 1 = hidden (discard above uCeilingY), 2 = fade
   * (gradually dims to CEILING_FADE_ALPHA_FLOOR over CEILING_FADE_BAND_M) - spec 7.3. */
  uCeilingMode: THREE.IUniform<number>
  uCeilingY: THREE.IUniform<number>
  /** Far depth-fade (spec 7, point 3) - always active, not user-facing: points beyond
   * uFadeEnd fade to invisible, so the cloud dissolves rather than clipping hard at the
   * far plane. Distinct from the removed near-camera fade this replaced in the UI. */
  uFadeStart: THREE.IUniform<number>
  uFadeEnd: THREE.IUniform<number>
  /** 0 = photo (per-vertex colour), 1 = height (--h-0..--h-4 gradient over uHeightMin..uHeightMax). */
  uColorMode: THREE.IUniform<number>
  uHeightMin: THREE.IUniform<number>
  uHeightMax: THREE.IUniform<number>
}

const CEILING_FADE_BAND_M = 0.3
const CEILING_FADE_ALPHA_FLOOR = 0.15

/** [l, c, h] triples for the shader's uH0..uH4 uniforms - h in degrees, matching
 * tokens.ts's OklchColor shape (oklchToSrgb takes the same values apart from the scalar
 * vs. GLSL-vec3 packing). */
function heightRampUniforms(): [number, number, number][] {
  return HEIGHT_RAMP.map(({ l, c, h }) => [l, c, h])
}

/** A custom points shader replacing three's built-in PointsMaterial: same per-vertex
 * color + perspective size-attenuation it already had (see WebGLMaterials.js's
 * refreshUniformsPoints - `uScale`/`uSize` are meant to be kept in sync with that same
 * `height * 0.5` / `size * pixelRatio` formula by the caller, since a plain
 * ShaderMaterial doesn't get those refreshed automatically like PointsMaterial does),
 * plus the ceiling and far depth-fade effects below. All read live uniforms, so
 * dragging a slider or flipping the ceiling select never touches the geometry. */
export function createPointCloudMaterial(hasColor: boolean, colorItemSize: number): THREE.ShaderMaterial {
  const colorType = colorItemSize >= 4 ? 'vec4' : 'vec3'

  const vertexShader = `
    ${hasColor ? `attribute ${colorType} color;` : ''}
    uniform float uSize;
    uniform float uScale;
    varying vec3 vColor;
    varying float vDist;
    varying float vWorldY;

    bool isPerspective(mat4 m) {
      return m[2][3] == -1.0;
    }

    void main() {
      vColor = ${hasColor ? 'color.rgb' : 'vec3(1.0)'};
      vec4 worldPosition = modelMatrix * vec4(position, 1.0);
      vWorldY = worldPosition.y;
      vec4 mvPosition = viewMatrix * worldPosition;
      vDist = length(mvPosition.xyz);
      gl_Position = projectionMatrix * mvPosition;
      gl_PointSize = uSize;
      if (isPerspective(projectionMatrix)) {
        gl_PointSize *= (uScale / -mvPosition.z);
      }
      // Spec 7, point 2: uSize already carries the device pixel ratio (set by the
      // caller, see SceneMap3D's uSize.value = pointSize * renderer.getPixelRatio()) -
      // this just keeps the attenuated result inside a sane on-screen pixel range.
      gl_PointSize = clamp(gl_PointSize, 1.0, 32.0);
    }
  `

  const fragmentShader = `
    uniform float uCeilingMode;
    uniform float uCeilingY;
    uniform float uFadeStart;
    uniform float uFadeEnd;
    uniform float uColorMode;
    uniform float uHeightMin;
    uniform float uHeightMax;
    uniform vec3 uH0;
    uniform vec3 uH1;
    uniform vec3 uH2;
    uniform vec3 uH3;
    uniform vec3 uH4;
    varying vec3 vColor;
    varying float vDist;
    varying float vWorldY;

    // OKLCH -> linear sRGB -> sRGB, mirroring lib/tokens.ts's oklchToSrgb exactly, so
    // the height ramp's colors match the design tokens bit for bit rather than
    // approximating them. Kept here (not sampled from a baked LUT texture) because the
    // ramp is only 5 stops and this runs per-fragment on already-clipped, decimated
    // points - cheap enough not to need one.
    vec3 oklchToLinearSrgb(vec3 oklch) {
      float L = oklch.x;
      float C = oklch.y;
      float h = radians(oklch.z);
      float a = C * cos(h);
      float b = C * sin(h);
      float l_ = L + 0.3963377774 * a + 0.2158037573 * b;
      float m_ = L - 0.1055613458 * a - 0.0638541728 * b;
      float s_ = L - 0.0894841775 * a - 1.291485548 * b;
      float l3 = l_ * l_ * l_;
      float m3 = m_ * m_ * m_;
      float s3 = s_ * s_ * s_;
      return vec3(
        4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
        -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
        -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3
      );
    }

    float srgbChannel(float c) {
      c = clamp(c, 0.0, 1.0);
      return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }

    vec3 linearToSrgb(vec3 c) {
      return vec3(srgbChannel(c.r), srgbChannel(c.g), srgbChannel(c.b));
    }

    // Interpolates the 5-stop ramp in OKLCH space (lerping L/C/hue directly), then
    // converts once at the end - mixing already-gamma-corrected RGB endpoints instead
    // would produce the muddy banding OKLCH interpolation is specifically meant to avoid.
    vec3 heightColor(float t) {
      t = clamp(t, 0.0, 1.0) * 4.0;
      vec3 oklch;
      if (t < 1.0) oklch = mix(uH0, uH1, t);
      else if (t < 2.0) oklch = mix(uH1, uH2, t - 1.0);
      else if (t < 3.0) oklch = mix(uH2, uH3, t - 2.0);
      else oklch = mix(uH3, uH4, t - 3.0);
      return linearToSrgb(oklchToLinearSrgb(oklch));
    }

    void main() {
      // Round points instead of three's default square gl_PointCoord quad (spec 7,
      // point 1) - a soft ~1px edge via smoothstep rather than a hard circle mask.
      vec2 c = gl_PointCoord - vec2(0.5);
      float d = length(c);
      float roundMask = 1.0 - smoothstep(0.45, 0.5, d);
      if (roundMask < 0.01) discard;

      float alpha = roundMask;

      // Far depth-fade (spec 7, point 3) - always on, not a user control.
      alpha *= 1.0 - smoothstep(uFadeStart, uFadeEnd, vDist);

      // Ceiling (spec 7, point 5): hidden discards outright, fade dims to a floor
      // rather than disappearing, visible is a no-op.
      if (uCeilingMode > 1.5) {
        float t = clamp((vWorldY - uCeilingY) / ${CEILING_FADE_BAND_M.toFixed(6)}, 0.0, 1.0);
        alpha *= mix(1.0, ${CEILING_FADE_ALPHA_FLOOR.toFixed(6)}, t);
      } else if (uCeilingMode > 0.5) {
        if (vWorldY > uCeilingY) discard;
      }

      if (alpha <= 0.001) discard;

      vec3 color = vColor;
      if (uColorMode > 0.5) {
        float t = clamp((vWorldY - uHeightMin) / max(uHeightMax - uHeightMin, 1e-6), 0.0, 1.0);
        color = heightColor(t);
      }
      gl_FragColor = vec4(color, alpha);
    }
  `

  const [h0, h1, h2, h3, h4] = heightRampUniforms()
  const uniforms: PointCloudUniforms = {
    uSize: { value: 1 },
    uScale: { value: 1 },
    uCeilingMode: { value: 0 },
    uCeilingY: { value: Infinity },
    uFadeStart: { value: Infinity },
    uFadeEnd: { value: Infinity },
    uColorMode: { value: 0 },
    uHeightMin: { value: 0 },
    uHeightMax: { value: 1 },
    uH0: { value: h0 },
    uH1: { value: h1 },
    uH2: { value: h2 },
    uH3: { value: h3 },
    uH4: { value: h4 },
  }

  return new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader, transparent: true })
}
