// Per-scene, localStorage-persisted ceiling-handling preference for SceneMap3D - 3D-only
// (2D has no ceiling), unlike the shared render mode (lib/renderMode.ts). Keyed by scene
// id, so switching to a *different* scene starts fresh at the default (ceiling visible)
// - a saved ceiling treatment from one room wouldn't mean anything in another - while
// returning to the *same* scene later restores what was set.

/** Spec 7.3 replaced the old (numeric height-cutoff slider + "hide ceiling" button +
 * separate near-camera fade select) with one explicit `ceilingMode` choice:
 * - `visible`: nothing hidden (uCeilingMode = 0, the threshold below is ignored).
 * - `hidden`: points at or above the cutoff height are discarded outright.
 * - `fade`: same threshold, but points fade to alpha 0.15 over a 0.3m band instead of
 *   disappearing - "the ceiling is there, it's just not in the way".
 * The near-camera fade this replaced is still gone from the UI; the far depth-fade
 * scene3dPoints.ts applies (spec 7, point 3) is automatic, not user-facing.
 *
 * `heightCutoff` brings back the continuous slider that used to set this threshold
 * directly: null means "no explicit cutoff", i.e. the threshold defaults to the
 * scene's own ceiling_y (see SceneMap3D's `heightCutoff ?? ceilingY` reads) - moving the
 * slider lets hidden/fade discard everything above some height *below* the ceiling too,
 * not just the ceiling itself. */
export type CeilingMode = 'visible' | 'hidden' | 'fade'

export interface PointCloudViewSettings {
  ceilingMode: CeilingMode
  /** World-space Y above which points are hidden/faded (when ceilingMode isn't
   * `visible`). null = no cutoff, i.e. use the scene's own ceiling_y. */
  heightCutoff: number | null
}

export const DEFAULT_POINT_CLOUD_VIEW_SETTINGS: PointCloudViewSettings = {
  ceilingMode: 'visible',
  heightCutoff: null,
}

function storageKey(sceneId: string): string {
  return `cloudeye:pointcloudView:${sceneId}`
}

function isCeilingMode(v: unknown): v is CeilingMode {
  return v === 'visible' || v === 'hidden' || v === 'fade'
}

export function loadPointCloudViewSettings(sceneId: string): PointCloudViewSettings {
  try {
    const raw = localStorage.getItem(storageKey(sceneId))
    if (!raw) return DEFAULT_POINT_CLOUD_VIEW_SETTINGS
    const parsed = JSON.parse(raw) as Partial<PointCloudViewSettings>
    const ceilingMode: CeilingMode = isCeilingMode(parsed.ceilingMode) ? parsed.ceilingMode : 'visible'
    const heightCutoff = typeof parsed.heightCutoff === 'number' ? parsed.heightCutoff : null
    return { ceilingMode, heightCutoff }
  } catch {
    return DEFAULT_POINT_CLOUD_VIEW_SETTINGS
  }
}

export function savePointCloudViewSettings(sceneId: string, settings: PointCloudViewSettings): void {
  try {
    localStorage.setItem(storageKey(sceneId), JSON.stringify(settings))
  } catch {
    // localStorage unavailable (private mode, disabled) - non-fatal, setting just won't persist.
  }
}
