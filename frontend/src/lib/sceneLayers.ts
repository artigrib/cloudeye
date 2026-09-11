/** The four things the 3D view can draw, each from one source, each independently
 * switchable.
 *
 * They were checkboxes ("mesh") plus two chords (Shift+3/4) plus nothing at all for the
 * floor and the points, which is why "turn the mesh off" did not turn the mesh off: the
 * only switch that existed flipped `.visible` on the nodes inside a group that stayed in
 * the scene graph. A layer that is off is REMOVED from the graph now - see SceneMap3D's
 * three gated effects - so "off" and "never loaded" are the same picture.
 */
export type SceneLayerKey = 'map' | 'points' | 'mesh' | 'layout'

export const SCENE_LAYER_KEYS: readonly SceneLayerKey[] = ['map', 'points', 'mesh', 'layout']

export type SceneLayerState = Record<SceneLayerKey, boolean>

/** Map + Points on, Mesh + Layout off. The two that are on are the two every scene has
 * and the two the screen's question (can this robot get there) is actually about; the
 * other two are heavy files that should not be fetched until asked for. */
export const DEFAULT_SCENE_LAYERS: SceneLayerState = {
  map: true,
  points: true,
  mesh: false,
  layout: false,
}

/** Exact labels - the toolbar shows these and nothing else. */
export const SCENE_LAYER_LABELS: Record<SceneLayerKey, string> = {
  map: 'Map',
  points: 'Points',
  mesh: 'Mesh',
  layout: 'Layout',
}

/** What each toggle draws, and where it comes from. Shown as the enabled tooltip, so the
 * source is one hover away rather than something you have to read the code for. */
export const SCENE_LAYER_TOOLTIPS: Record<SceneLayerKey, string> = {
  map: 'clearance layer plane',
  points: 'point cloud',
  mesh: 'nvblox TSDF mesh',
  layout: 'MSA layout',
}

/** The disabled tooltip, for a scene whose endpoint answers 404. Only Mesh and Layout
 * can be in that state: Map is derived from the grid every done scene has, and Points
 * falls back from /cloud to /mesh (which is why a 404 on /cloud alone does not disable
 * it - hero-74 is exactly that scene). */
export const SCENE_LAYER_UNAVAILABLE: Record<SceneLayerKey, string> = {
  map: 'no map for this scene',
  points: 'no points for this scene',
  mesh: 'no triangle mesh for this scene',
  layout: 'no layout for this scene',
}
