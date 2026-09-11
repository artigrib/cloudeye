import { useEffect, useRef, useState } from 'react'
import type { ComponentProps } from 'react'
import type { RenderMode } from '../lib/renderMode'
import {
  SCENE_LAYER_KEYS,
  SCENE_LAYER_LABELS,
  SCENE_LAYER_TOOLTIPS,
  SCENE_LAYER_UNAVAILABLE,
  type SceneLayerKey,
  type SceneLayerState,
} from '../lib/sceneLayers'
import { type MapViewMode } from './MapViewToggle'
import SceneMap3D from './SceneMap3D'
import Checkbox from './ui/Checkbox'
import Select from './ui/Select'
import Tooltip from './ui/Tooltip'

type SceneMap3DProps = Omit<ComponentProps<typeof SceneMap3D>, 'colorMode' | 'viewMode' | 'onToggleColorMode'>

interface Props {
  viewMode: MapViewMode
  /** The shared render-mode selector (spec 6) - one Select for both views, injected into
   * the shared canvas's own color/render prop below rather than threaded twice by the
   * caller. */
  renderMode: RenderMode
  onChangeRenderMode: (mode: RenderMode) => void
  /** The two switches that change what the canvas draws but are not layers - they live
   * in the overflow menu now, because the four layer toggles are the row's subject and
   * two checkboxes beside them read as two more layers. null hides the camera-path
   * switch for a scene with no track, rather than showing a dead one. */
  showCameraPath: boolean | null
  onChangeShowCameraPath: (value: boolean) => void
  showFragments: boolean
  onChangeShowFragments: (value: boolean) => void
  /** The four layers, and which of them this scene actually has. */
  layers: SceneLayerState
  onToggleLayer: (key: SceneLayerKey, value: boolean) => void
  layersAvailable: SceneLayerState
  sceneProps: SceneMap3DProps
}

/** Hosts the merged 2D/3D scene canvas (redesign plan section 6: "one scene, two
 * cameras") - a single SceneMap3D owns both an orthographic top-down camera and the
 * existing perspective one, sharing every layer (floor, contours, point cloud,
 * markers, robot, trail/camera-track) between them. Switching `viewMode` swaps the
 * active camera/controls in place; nothing here unmounts or remounts on toggle, so
 * orbit state, 2D pan/zoom, and the (lazily loaded) point cloud all survive switching
 * back and forth. */
export default function SceneView({
  viewMode,
  renderMode,
  onChangeRenderMode,
  showCameraPath,
  onChangeShowCameraPath,
  showFragments,
  onChangeShowFragments,
  layers,
  onToggleLayer,
  layersAvailable,
  sceneProps,
}: Props) {
  // Keyboard shortcut hook (physical "C" key - see SceneMap3D's mount effect): cycles
  // the point cloud between height-gradient and true-RGB coloring, i.e. the same two
  // states the render-mode <select> below can reach via "Height"/"Photo" - "clearance"
  // and "occupancy" both already render points the same way "photo" does (see
  // RenderMode's own doc comment), so toggling always lands on one of exactly two
  // meaningfully-different point colorings regardless of which of the four modes was
  // active beforehand.
  function toggleColorMode() {
    onChangeRenderMode(renderMode === 'height' ? 'photo' : 'height')
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* The canvas toolbar: 40px, the same height as the first row of each side panel,
         so the three columns start on one line. Everything on it changes what the canvas
         draws and nothing on it changes anything else.
         `flex` never wraps by default and every child is `shrink-0`, so this row is one
         line at 1440x900 (measured: the four toggles, the mode select and the overflow
         button come to ~520px in an ~810px column). */}
      <div
        className="flex h-toolbar shrink-0 items-center gap-s2 border-b-hair border-border bg-surface px-s3"
        data-testid="canvas-toolbar"
      >
        {/* THE FOUR LAYERS, in the order they stack: the floor plane, the points on it,
           the reconstructed surface, the simplified room. 3D only - in 2D the map IS the
           floor plane and there is nothing to compose. */}
        {viewMode === '3d' &&
          SCENE_LAYER_KEYS.map((key) => (
            <LayerToggle
              key={key}
              layerKey={key}
              on={layers[key]}
              available={layersAvailable[key]}
              onToggle={(value) => onToggleLayer(key, value)}
            />
          ))}

        {/* Kept beside the toggles, and disabled when Points is off: it is the point
           cloud's colouring, and a live control over a layer that is not drawn is a
           control that does nothing. */}
        <Select
          value={renderMode}
          onChange={onChangeRenderMode}
          aria-label="Render mode"
          className="w-32 shrink-0"
          disabled={viewMode === '3d' && !layers.points}
          options={[
            { value: 'clearance', label: 'Clearance' },
            { value: 'height', label: 'Height' },
            { value: 'photo', label: 'Photo' },
            { value: 'occupancy', label: 'Occupancy' },
          ]}
        />

        <div className="flex-1" />

        <OverflowMenu>
          {showCameraPath !== null && (
            <Checkbox
              checked={showCameraPath}
              onChange={onChangeShowCameraPath}
              label="camera path"
              className="shrink-0"
              data-testid="toggle-camera-path"
            />
          )}
          <Checkbox
            checked={showFragments}
            onChange={onChangeShowFragments}
            label="show fragments"
            className="shrink-0"
            data-testid="toggle-fragments"
          />
        </OverflowMenu>
      </div>

      <div className="relative min-h-0 flex-1">
        <div className="absolute inset-0">
          <SceneMap3D {...sceneProps} viewMode={viewMode} colorMode={renderMode} onToggleColorMode={toggleColorMode} />
        </div>

      </div>
    </div>
  )
}

/** One layer, on or off. `aria-pressed` rather than a checkbox: these are four
 * independent switches over what is drawn, not four members of a set, and the toolbar
 * reads as a row of states rather than a form. Unavailable is DISABLED, never hidden -
 * "this scene has no layout" is an answer, and a toggle that vanishes per scene makes
 * the toolbar a different shape on every scene. */
function LayerToggle({
  layerKey,
  on,
  available,
  onToggle,
}: {
  layerKey: SceneLayerKey
  on: boolean
  available: boolean
  onToggle: (value: boolean) => void
}) {
  return (
    <Tooltip content={available ? SCENE_LAYER_TOOLTIPS[layerKey] : SCENE_LAYER_UNAVAILABLE[layerKey]}>
      <button
        type="button"
        onClick={() => onToggle(!on)}
        disabled={!available}
        aria-pressed={on && available}
        data-testid={`layer-${layerKey}`}
        className={`hit-target shrink-0 rounded-sm border-hair px-s2 text-bodyux transition-colors duration-100 disabled:cursor-not-allowed disabled:opacity-40 ${
          on && available
            ? 'border-accent/40 bg-accent-dim text-fg'
            : 'border-border text-subtle hover:bg-fg/10 hover:text-fg'
        }`}
      >
        {SCENE_LAYER_LABELS[layerKey]}
      </button>
    </Tooltip>
  )
}

/** The "..." menu. Everything on this toolbar that is not a layer lives in here, so the
 * row's subject stays "what is drawn" and the toolbar cannot wrap as switches are added.
 * Closes on outside pointerdown and on Escape; `aria-expanded` is what
 * demo/probe_layout_inventory.py opens it by. */
function OverflowMenu({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onDown(e: PointerEvent) {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={ref} className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="More view options"
        data-testid="canvas-more"
        className="hit-target flex w-8 shrink-0 items-center justify-center rounded-sm border-hair border-border text-subtle transition-colors duration-100 hover:bg-fg/10 hover:text-fg"
      >
        …
      </button>
      {open && (
        <div
          role="menu"
          data-testid="canvas-more-menu"
          className="absolute right-0 top-full z-30 mt-1 flex flex-col gap-s2 rounded-md border-hair border-border bg-surface-raised p-s2 shadow-xl"
        >
          {children}
        </div>
      )}
    </div>
  )
}
