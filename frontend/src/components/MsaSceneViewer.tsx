import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import Checkbox from './ui/Checkbox'
import SegmentedControl from './ui/SegmentedControl'
import Slider from './ui/Slider'
import { ACCENT, BG, DATA_UNREACHABLE, toHex } from '../lib/tokens'
import { gapPasses, loadMsaGaps, type MsaGap } from '../lib/msaGaps'
import { classifyNode, isTranslucentObjectNode, LAYER_KEYS, TRANSLUCENT_OPACITY, type LayerKey } from '../lib/msaLayers'
import { dollhouseWallGeometry, fitKeyLight, viewPresets, WALL_NAME_RE, type ViewPresetKey } from '../lib/msaDollhouse'

// classifyNode/LayerKey/LAYER_KEYS now live in lib/msaLayers.ts (shared with
// SceneMap3D.tsx, which overlays the same MSA layers on the main scene canvas - see
// docs/DECISIONS.md). Re-exported here so MsaSceneViewer.test.ts's existing import
// path keeps working unchanged.
export { classifyNode, LAYER_KEYS, type LayerKey }

export interface MsaSceneViewerProps {
  /** URL of an MSA bootstrap/refined-stage GLB (scripts/msa/export_glb.py output). */
  glbUrl: string
  /** URL of the matching gaps.json (scripts/msa/gaps.py output) - omit to hide the
   * gaps layer's toggle entirely rather than show it disabled. */
  gapsUrl?: string
  className?: string
}

type ViewKey = ViewPresetKey | 'free'

const PASS_COLOR = toHex(ACCENT)
const FAIL_COLOR = toHex(DATA_UNREACHABLE)

function makeGapLabel(gap: MsaGap): HTMLDivElement {
  const el = document.createElement('div')
  const pass = gapPasses(gap)
  el.className = 'msa-gap-label'
  el.style.cssText = [
    'font:11px/1.3 ui-monospace,monospace',
    'padding:1px 5px',
    'border-radius:3px',
    'white-space:nowrap',
    'pointer-events:none',
    'transform:translate(-50%,-100%)',
    `background:${pass ? 'rgba(80,200,150,0.85)' : 'rgba(220,90,90,0.85)'}`,
    'color:#0a0a0a',
  ].join(';')
  el.textContent = `${Math.round(gap.width_m * 100)}cm ${pass ? '✓' : '✗'}`
  return el
}

/** Dollhouse material/shadow setup for one loaded GLB (T15c). Walls lose their
 * outward faces (see `dollhouseWallGeometry`) and render single-sided so near walls
 * cull and far walls stay opaque; floor receives shadows; object meshes cast and
 * receive. `roomCenter` is world-space (the floor's bbox centre).
 *
 * T15i: curtain/drape/blind/mirror placeholders (`isTranslucentObjectNode`) get their
 * own cloned material at TRANSLUCENT_OPACITY, `transparent`, `depthWrite=false`, no
 * shadow casting - the 2.7 m near-white curtain boxes otherwise hide the bed (T15e
 * run2). The material is flagged in `userData.msaTranslucent` so the per-frame
 * cloud<->mesh blend (`applyFrameState`) scales, rather than overwrites, its opacity. */
export function applyDollhouseLook(root: THREE.Object3D, roomCenter: THREE.Vector3): void {
  root.updateMatrixWorld(true)
  root.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return
    const layer = classifyNode(obj.name)
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material]
    if (WALL_NAME_RE.test(obj.name)) {
      const localCenter = obj.worldToLocal(roomCenter.clone())
      obj.geometry = dollhouseWallGeometry(obj.geometry, localCenter)
      for (const mat of materials) mat.side = THREE.FrontSide
      obj.castShadow = true
      obj.receiveShadow = true
    } else if (obj.name === 'floor') {
      obj.receiveShadow = true
    } else if (layer === 'mesh' && isTranslucentObjectNode(obj.name)) {
      // Clone: glTF materials are shared between primitives, and only this object
      // may go translucent.
      const cloned = materials.map((mat) => {
        const m = mat.clone()
        m.transparent = true
        m.opacity = TRANSLUCENT_OPACITY
        m.depthWrite = false
        m.userData.msaTranslucent = true
        return m
      })
      obj.material = Array.isArray(obj.material) ? cloned : cloned[0]
      obj.castShadow = false
      obj.receiveShadow = true
    } else if (layer === 'mesh') {
      obj.castShadow = true
      obj.receiveShadow = true
    }
  })
}

/** MSA scene viewer (SPEC.md §8): loads an export_glb.py GLB and renders it with five
 * independently toggleable layers (cloud / mesh / collision / plan / gaps), a
 * cloud<->mesh opacity blend, and gap-width labels via CSS2DRenderer. Deliberately a
 * small standalone viewer rather than an extension of SceneMap3D.tsx - that component
 * is coupled to the main (non-MSA) pipeline's data model (point-cloud GLB, DB-backed
 * scene state, robot path animation); MSA scenes are file-based script output with a
 * much smaller, different node schema, and forcing them through the same 2000+ line
 * component would mean threading MSA-specific branches through code with nothing to
 * do with MSA.
 *
 * Presentation (T15c "dollhouse"): hemisphere + ambient fill, a shadow-casting key
 * light fitted to the scene, ACES tone mapping, sRGB output; walls rendered as
 * room-facing single-sided faces (see `applyDollhouseLook`), textured floor/walls
 * when the GLB carries the baked PBR materials from `scripts/msa/textures.py`. */
export default function MsaSceneViewer({ glbUrl, gapsUrl, className = '' }: MsaSceneViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    cloud: true,
    mesh: true,
    collision: false,
    plan: false,
    gaps: false,
  })
  const [cloudMeshBlend, setCloudMeshBlend] = useState(1) // 0 = cloud, 1 = mesh
  const [view, setView] = useState<ViewKey>('threeQuarter')
  const [error, setError] = useState<string | null>(null)
  const [hasCloud, setHasCloud] = useState(false)

  // Read inside the render-loop effect via refs so toggling a layer/slider doesn't
  // need to re-run GLB loading - only the latest values, applied on every rAF tick.
  const layersRef = useRef(layers)
  const blendRef = useRef(cloudMeshBlend)
  useEffect(() => {
    layersRef.current = layers
    blendRef.current = cloudMeshBlend
  }, [layers, cloudMeshBlend])

  // Set by the render effect once the camera/controls exist; the view control calls it.
  const applyViewRef = useRef<(key: ViewPresetKey) => void>(() => {})

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let disposed = false
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(toHex(BG))

    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200)
    camera.position.set(6, 6, 6)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap // PCFSoftShadowMap is deprecated in three r185
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    renderer.outputColorSpace = THREE.SRGBColorSpace
    container.appendChild(renderer.domElement)

    const labelRenderer = new CSS2DRenderer()
    labelRenderer.domElement.style.position = 'absolute'
    labelRenderer.domElement.style.top = '0'
    labelRenderer.domElement.style.left = '0'
    labelRenderer.domElement.style.pointerEvents = 'none'
    container.appendChild(labelRenderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 1, 0)
    controls.update()
    const onUserOrbit = () => setView('free')
    controls.addEventListener('start', onUserOrbit)

    // Dollhouse lighting: sky/ground hemisphere fill so vertical walls and the floor
    // read differently, a low ambient floor, a warm shadow-casting key from high on
    // the +X/+Z side (fitted to the scene once the GLB is in), and a cool fill from
    // the opposite side so the culled-wall side of objects isn't pitch black.
    scene.add(new THREE.HemisphereLight(0xe6edf8, 0x6b6157, 1.0))
    scene.add(new THREE.AmbientLight(0xffffff, 0.18))
    const key = new THREE.DirectionalLight(0xfff1dc, 2.6)
    key.position.set(5, 10, 5)
    key.castShadow = true
    key.shadow.mapSize.set(2048, 2048)
    key.shadow.bias = -0.0004
    key.shadow.normalBias = 0.02
    scene.add(key)
    scene.add(key.target)
    const fill = new THREE.DirectionalLight(0xc9d9ff, 0.7)
    fill.position.set(-6, 5, -4)
    scene.add(fill)

    // An arrow function assigned to a const (not a hoisted function declaration) so
    // TS's control-flow narrowing of `container` (guarded non-null above) actually
    // carries into this closure - `tsc -b`'s stricter project-reference build (unlike
    // plain `tsc --noEmit`) does not narrow `container` inside a `function resize() {}`
    // declaration, since a hoisted declaration could in principle be invoked before the
    // guard runs.
    const resize = () => {
      const w = container.clientWidth || 1
      const h = container.clientHeight || 1
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      labelRenderer.setSize(w, h)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    const layerNodes: Record<LayerKey, THREE.Object3D[]> = {
      cloud: [],
      mesh: [],
      collision: [],
      plan: [],
      gaps: [],
    }

    const gapGroup = new THREE.Group()
    gapGroup.name = 'gaps-layer'
    scene.add(gapGroup)
    layerNodes.gaps.push(gapGroup)

    let presets: ReturnType<typeof viewPresets> | null = null
    applyViewRef.current = (presetKey) => {
      if (!presets) return
      const pose = presets[presetKey]
      camera.position.copy(pose.position)
      controls.target.copy(pose.target)
      controls.update()
    }

    new GLTFLoader().load(
      glbUrl,
      (loaded) => {
        if (disposed) return
        scene.add(loaded.scene)
        loaded.scene.traverse((obj) => {
          const layer = classifyNode(obj.name)
          if (layer) layerNodes[layer].push(obj)
        })
        setHasCloud(layerNodes.cloud.length > 0)

        // Frame + light the room from the mesh layer's extent (walls, floor, objects).
        const box = new THREE.Box3()
        for (const obj of layerNodes.mesh) box.expandByObject(obj)
        if (box.isEmpty()) box.setFromObject(loaded.scene)
        const floor = loaded.scene.getObjectByName('floor')
        const roomCenter = floor ? new THREE.Box3().setFromObject(floor).getCenter(new THREE.Vector3()) : box.getCenter(new THREE.Vector3())
        applyDollhouseLook(loaded.scene, roomCenter)
        fitKeyLight(key, box)
        presets = viewPresets(box)
        applyViewRef.current('threeQuarter')
        setView('threeQuarter')
      },
      undefined,
      (err) => {
        if (!disposed) setError(err instanceof Error ? err.message : 'Failed to load GLB')
      },
    )

    if (gapsUrl) {
      loadMsaGaps(gapsUrl)
        .then((gaps) => {
          if (disposed) return
          for (const gap of gaps) {
            const [x, z] = gap.measurement_point_xy
            const marker = new THREE.Mesh(
              new THREE.SphereGeometry(0.03, 8, 8),
              new THREE.MeshBasicMaterial({ color: gapPasses(gap) ? PASS_COLOR : FAIL_COLOR, toneMapped: false }),
            )
            marker.position.set(x, 0.15, z)
            const label = new CSS2DObject(makeGapLabel(gap))
            label.position.set(0, 0.15, 0)
            marker.add(label)
            gapGroup.add(marker)
          }
        })
        .catch(() => {
          /* gaps layer just stays empty - not fatal to the rest of the viewer */
        })
    }

    function applyFrameState() {
      const currentLayers = layersRef.current
      for (const key of LAYER_KEYS) {
        const visible = currentLayers[key]
        for (const obj of layerNodes[key]) obj.visible = visible
      }
      const blend = layerNodes.cloud.length > 0 ? blendRef.current : 1
      // Only go through the transparent pass while actually blending toward the cloud -
      // opaque PBR walls/floor/objects must stay in the opaque pass for correct depth
      // sorting and shadow reception.
      const transparent = blend < 1
      for (const obj of layerNodes.mesh) {
        obj.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            const mats = Array.isArray(child.material) ? child.material : [child.material]
            for (const mat of mats) {
              if (mat.userData.msaTranslucent) {
                // T15i: a translucent placeholder stays translucent; the blend only scales it.
                mat.transparent = true
                mat.opacity = blend * TRANSLUCENT_OPACITY
                continue
              }
              mat.transparent = transparent
              mat.opacity = blend
            }
          }
        })
      }
    }

    let raf = 0
    function animate() {
      raf = requestAnimationFrame(animate)
      controls.update()
      applyFrameState()
      renderer.render(scene, camera)
      labelRenderer.render(scene, camera)
    }
    animate()

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.removeEventListener('start', onUserOrbit)
      controls.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
      container.removeChild(labelRenderer.domElement)
      applyViewRef.current = () => {}
    }
  }, [glbUrl, gapsUrl])

  const layerToggles = useMemo(
    () =>
      (
        [
          ['cloud', 'Point cloud'],
          ['mesh', 'Mesh'],
          ['collision', 'Collision'],
          ['plan', '2D plan'],
          ['gaps', 'Gaps'],
        ] as [LayerKey, string][]
      ).map(([key, label]) => (
        <Checkbox
          key={key}
          checked={layers[key]}
          onChange={(v) => setLayers((prev) => ({ ...prev, [key]: v }))}
          label={label}
          disabled={key === 'cloud' && !hasCloud}
        />
      )),
    [layers, hasCloud],
  )

  return (
    <div className={`relative flex h-full w-full flex-col gap-3 ${className}`}>
      <div className="flex flex-wrap items-center gap-4 border-b-hair border-border pb-2">
        {layerToggles}
        <Slider
          label="Cloud ↔ Mesh"
          value={cloudMeshBlend}
          onChange={setCloudMeshBlend}
          min={0}
          max={1}
          step={0.01}
          disabled={!hasCloud}
        />
        <SegmentedControl<ViewKey>
          aria-label="View"
          value={view}
          onChange={(v) => {
            setView(v)
            if (v !== 'free') applyViewRef.current(v)
          }}
          segments={[
            { value: 'threeQuarter', label: '3/4', title: 'Dollhouse 3/4 view from outside' },
            { value: 'top', label: 'Top', title: 'Straight top-down view' },
            { value: 'free', label: 'Free', title: 'Wherever you orbited to' },
          ]}
        />
      </div>
      <div ref={containerRef} className="relative min-h-0 flex-1" />
      {error && <div className="absolute bottom-2 left-2 text-[12px] text-red-400">{error}</div>}
    </div>
  )
}
