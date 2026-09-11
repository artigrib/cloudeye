import * as THREE from 'three/webgpu'
import type { DeviceBudgetInput } from './pointBudget'

// The impure boundary for pointBudget.ts's static prior - all the navigator/matchMedia/
// WebGL calls live here, untested by convention (matches scene3dPoints.ts/
// robotModelCache.ts: WebGL-touching code in this app doesn't get a jsdom test, since
// jsdom has no real WebGL context to probe).
//
// navigator.gpu and renderer.capabilities.maxTextureSize were both considered and
// rejected as device-tier signals (see docs/DECISIONS.md / the plan this implements):
// navigator.gpu tracks browser version more than GPU capability, and maxTextureSize is
// 16384 on essentially every device shipped since ~2015 - neither has real variance to
// read a tier from.
//
// renderer.getContext() returns `unknown` under the common Renderer base (WebGPURenderer
// included - see docs/DECISIONS.md) since its actual shape is backend-dependent: a real
// WebGPU backend returns a GPUCanvasContext-ish object with no `.getExtension`, so this
// whole probe degrades to "can't tell, assume hardware" there rather than throwing -
// acceptable, since the softwareRenderer signal was always a hard-floor guard, not load-
// bearing for the common case.

const SOFTWARE_RENDERER_PATTERN = /swiftshader|llvmpipe|software|basic render|microsoft basic/i

function detectSoftwareRenderer(gl: unknown): boolean {
  try {
    const ctx = gl as WebGLRenderingContext | WebGL2RenderingContext
    const ext = ctx.getExtension('WEBGL_debug_renderer_info')
    if (!ext) return false
    const renderer = ctx.getParameter(ext.UNMASKED_RENDERER_WEBGL) as string | null
    return typeof renderer === 'string' && SOFTWARE_RENDERER_PATTERN.test(renderer)
  } catch {
    // Not a WebGL context at all (WebGPU backend) - or Firefox's resistFingerprinting
    // making the extension/parameter throw. Either way, "can't tell" means "assume
    // hardware".
    return false
  }
}

/** Gathers the raw signals initialPointBudget scores. Call once per mount, before the
 * first point cloud builds - see SceneMap3D's mount effect. */
export function probeDevice(renderer: THREE.WebGPURenderer): DeviceBudgetInput {
  let gl: unknown = null
  try {
    gl = renderer.getContext()
  } catch {
    gl = null
  }
  const size = renderer.getDrawingBufferSize(new THREE.Vector2())

  let coarsePointer = false
  let saveData = false
  if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
    try {
      coarsePointer = window.matchMedia('(pointer: coarse)').matches
    } catch {
      coarsePointer = false
    }
  }

  const nav = typeof navigator !== 'undefined' ? navigator : undefined
  const connection = (nav as unknown as { connection?: { saveData?: boolean } } | undefined)?.connection
  saveData = connection?.saveData === true

  const hardwareConcurrency =
    typeof nav?.hardwareConcurrency === 'number' && nav.hardwareConcurrency > 0 ? nav.hardwareConcurrency : null
  const deviceMemoryGb =
    typeof (nav as unknown as { deviceMemory?: number })?.deviceMemory === 'number'
      ? (nav as unknown as { deviceMemory?: number }).deviceMemory!
      : null

  return {
    coarsePointer,
    hardwareConcurrency,
    deviceMemoryGb,
    drawingBufferPixels: size.x * size.y,
    softwareRenderer: detectSoftwareRenderer(gl),
    saveData,
  }
}
