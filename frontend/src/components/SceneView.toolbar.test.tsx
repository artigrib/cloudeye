import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import SceneView from './SceneView'
import { DEFAULT_SCENE_LAYERS } from '../lib/sceneLayers'

// The canvas itself is a WebGPU renderer and cannot mount in jsdom; the toolbar is the
// subject here, so SceneMap3D is stubbed out.
vi.mock('./SceneMap3D', () => ({ default: () => <div data-testid="scene-map-3d" /> }))

const AVAILABLE = { map: true, points: true, mesh: true, layout: true }

function setup(over: Partial<React.ComponentProps<typeof SceneView>> = {}) {
  const onToggleLayer = vi.fn()
  const props = {
    viewMode: '3d' as const,
    renderMode: 'clearance' as const,
    onChangeRenderMode: vi.fn(),
    showCameraPath: false,
    onChangeShowCameraPath: vi.fn(),
    showFragments: false,
    onChangeShowFragments: vi.fn(),
    layers: DEFAULT_SCENE_LAYERS,
    onToggleLayer,
    layersAvailable: AVAILABLE,
    sceneProps: {} as never,
    ...over,
  }
  render(<SceneView {...props} />)
  return { onToggleLayer, props }
}

describe('the 3D layer toolbar', () => {
  it('offers exactly Map, Points, Mesh and Layout', () => {
    setup()
    expect(screen.getByTestId('layer-map')).toHaveTextContent('Map')
    expect(screen.getByTestId('layer-points')).toHaveTextContent('Points')
    expect(screen.getByTestId('layer-mesh')).toHaveTextContent('Mesh')
    expect(screen.getByTestId('layer-layout')).toHaveTextContent('Layout')
  })

  it('defaults to Map + Points on, Mesh + Layout off', () => {
    setup()
    expect(screen.getByTestId('layer-map')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('layer-points')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('layer-mesh')).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByTestId('layer-layout')).toHaveAttribute('aria-pressed', 'false')
  })

  it('toggles each layer independently', () => {
    const { onToggleLayer } = setup()
    fireEvent.click(screen.getByTestId('layer-mesh'))
    expect(onToggleLayer).toHaveBeenCalledWith('mesh', true)
    fireEvent.click(screen.getByTestId('layer-map'))
    expect(onToggleLayer).toHaveBeenCalledWith('map', false)
  })

  it('disables a layer this scene has no endpoint for', () => {
    setup({ layersAvailable: { ...AVAILABLE, layout: false } })
    expect(screen.getByTestId('layer-layout')).toBeDisabled()
    expect(screen.getByTestId('layer-mesh')).toBeEnabled()
  })

  it('disables the render-mode dropdown when Points is off - it colours Points only', () => {
    setup({ layers: { ...DEFAULT_SCENE_LAYERS, points: false } })
    expect(screen.getByLabelText('Render mode')).toBeDisabled()
  })

  it('hides the layer toggles in 2D, where the floor plane is the map', () => {
    setup({ viewMode: '2d' })
    expect(screen.queryByTestId('layer-map')).toBeNull()
    expect(screen.getByLabelText('Render mode')).toBeEnabled()
  })

  it('keeps camera path and show fragments behind the overflow menu', () => {
    setup()
    expect(screen.queryByTestId('toggle-camera-path')).toBeNull()
    expect(screen.queryByTestId('toggle-fragments')).toBeNull()
    fireEvent.click(screen.getByTestId('canvas-more'))
    expect(screen.getByTestId('toggle-camera-path')).toBeInTheDocument()
    expect(screen.getByTestId('toggle-fragments')).toBeInTheDocument()
  })
})
