// Renderer spec section 6: the one mode selector shared by both the 2D and 3D views -
// unlike ceiling handling (lib/pointcloudViewSettings.ts), which is 3D-only, this is
// genuinely cross-view state, so it's owned by ScenePage and threaded down into the
// merged scene canvas (SceneMap3D) rather than living inside it.

/** - `clearance` (default): the distance-to-nearest-obstacle field, colored relative to
 *    the selected robot's radius. 2D: the whole map. 3D: the point cloud keeps its own
 *    colour (photo/height only actually change point colour); the floor decal is this
 *    field either way (see SceneView's floor-decal wiring).
 * - `height`: 2D has no per-pixel height data of its own, so it falls back to
 *   `clearance` (spec: "не применим -> показывает Clearance"); 3D colours the point
 *   cloud by world-space Y.
 * - `photo`: 2D falls back to `clearance` too (no orthophoto exists in this pipeline -
 *   spec's "ортофото сверху, если есть" always resolves to "if available", and it
 *   never is here); 3D uses the point cloud's own per-vertex colour.
 * - `occupancy`: the raw free/obstacle/unknown grid, one cell one pixel, no smoothing -
 *   in both views (2D: the whole map; 3D: the floor decal). */
export type RenderMode = 'clearance' | 'height' | 'photo' | 'occupancy'

export const DEFAULT_RENDER_MODE: RenderMode = 'clearance'

function isRenderMode(v: unknown): v is RenderMode {
  return v === 'clearance' || v === 'height' || v === 'photo' || v === 'occupancy'
}

function storageKey(sceneId: string): string {
  return `cloudeye:renderMode:${sceneId}`
}

export function loadRenderMode(sceneId: string): RenderMode {
  try {
    const raw = localStorage.getItem(storageKey(sceneId))
    return isRenderMode(raw) ? raw : DEFAULT_RENDER_MODE
  } catch {
    return DEFAULT_RENDER_MODE
  }
}

export function saveRenderMode(sceneId: string, mode: RenderMode): void {
  try {
    localStorage.setItem(storageKey(sceneId), mode)
  } catch {
    // localStorage unavailable (private mode, disabled) - non-fatal, setting just won't persist.
  }
}
