import type { MsaSceneMeta } from '../lib/msaLayers'
import { apiDelete, apiGet, apiPost } from './client'
import type {
  CameraTrackResponse,
  CommandListResponse,
  CommandResponse,
  CriticReportResponse,
  FitProbResponse,
  MsaObjectsResponse,
  OccupancyGridResponse,
  ReachabilityResponse,
  SceneLayersResponse,
  SceneObjectListResponse,
  SceneResponse,
  SceneStatusResponse,
} from './types'

export function getScene(id: string): Promise<SceneResponse> {
  return apiGet(`/scenes/${id}`)
}

export function getSceneStatus(id: string): Promise<SceneStatusResponse> {
  return apiGet(`/scenes/${id}/status`)
}

/** 409 (SceneNotReadyError) until the scene's status is "done". */
export function getSceneObjects(id: string): Promise<SceneObjectListResponse> {
  return apiGet(`/scenes/${id}/objects`)
}

/** 409 (SceneNotReadyError) until the scene's status is "done".
 *
 * `robotId` is not optional in practice, even though the parameter is: the grid is DERIVED
 * from the layer's obstacle-height map at that platform's own roof, so omitting it draws
 * the DEFAULT platform's map. A 0.192 m TurtleBot drives under a 0.694 m bed top that
 * walls off a 0.40 m Go2, and a user looking at the short robot's picture while planning
 * the tall one's routes is exactly the "two answers to one question" this layer removed. */
export function getSceneMap(id: string, robotId?: string | null): Promise<OccupancyGridResponse> {
  const query = robotId ? `?robot_id=${encodeURIComponent(robotId)}` : ''
  return apiGet(`/scenes/${id}/map${query}`)
}

export function sceneMapPreviewUrl(id: string): string {
  return `/api/scenes/${id}/map/preview`
}

export function sceneMeshUrl(id: string): string {
  return `/api/scenes/${id}/mesh`
}

/** Binary point cloud (CEPC v1, see lib/cloudBin.ts) for scenes that have one. 404s for
 * every scene the ordinary GPU pipeline produced - those only have the points-primitive
 * GLB at sceneMeshUrl - and the loader falls back to that, so this is safe to pass
 * unconditionally, same as sceneMsaGlbUrl. */
export function sceneCloudUrl(id: string): string {
  return `/api/scenes/${id}/cloud`
}

/** Optional MSA (Measured Scene Assembly) glTF export for this scene - see
 * app.routers.scenes.get_scene_msa_glb's docstring. 404s for the (currently common)
 * case of a scene with no MSA export placed at its fixed `<scene_dir>/msa/scene.glb`
 * convention; SceneMap3D's loader treats that the same as any other load failure
 * (silently leaves the MSA layers empty - see useSceneGlb). Safe to pass to every
 * scene unconditionally, same as sceneMeshUrl. */
export function sceneMsaGlbUrl(id: string): string {
  return `/api/scenes/${id}/msa-glb`
}

/** M7b addition (c): the canonical, de-cluttered MSA object list (id/label only) -
 * see app.routers.scenes.get_scene_msa_objects's docstring. 404s the same way
 * sceneMsaGlbUrl's target does for a scene with no MSA export; callers should treat
 * that as "fall back to scene.objects' raw per-class counts", not an error. */
export function getSceneMsaObjects(id: string): Promise<MsaObjectsResponse> {
  return apiGet(`/scenes/${id}/msa-objects`)
}

/** Optional scene-critic report (scripts/audit/critic_rules.py + critic_vlm.py +
 * critic_merge.py, var/scratch/QUEUE.md M14) - see
 * app.routers.scenes.get_scene_critic's docstring. Report-only: reading this never
 * triggers a critic run and never mutates scene data, it only serves a report
 * that's already on disk. 404s (NotFoundError, see api/client.ts) for the (default)
 * case of a scene no critic pass has been run against - callers should treat that
 * the same as "no report available" and hide the badge, not surface an error. */
export function getSceneCritic(id: string): Promise<CriticReportResponse> {
  return apiGet(`/scenes/${id}/critic`)
}

/** Sibling of sceneMsaGlbUrl: the bootstrap scene_meta.json next to the served MSA GLB
 * (app.routers.scenes.get_scene_msa_meta). Carries the yaw normalization the GLB was
 * rotated by, which SceneMap3D inverts to overlay the GLB on the unrotated point cloud
 * (lib/msaLayers.msaOverlayTransform). 404 when there is no export/meta - same
 * "safe to pass unconditionally" contract as sceneMsaGlbUrl. */
export function sceneMsaMetaUrl(id: string): string {
  return `/api/scenes/${id}/msa-meta`
}

/** Fetches `url` (a sceneMsaMetaUrl) and returns null for a 404 (no MSA export or a
 * pre-scene_meta.json export) - the "not rotated" case for the overlay - and for any
 * other non-2xx status: a broken meta must never take the MSA layers or the cloud down
 * with it, the layers just render untransformed as they did before morning-2. */
export async function fetchSceneMsaMeta(url: string, signal?: AbortSignal): Promise<MsaSceneMeta | null> {
  const res = await fetch(url, { signal })
  if (!res.ok) return null
  return (await res.json()) as MsaSceneMeta
}

/** OpenUSD export for Isaac Sim - collision boxes for floor/walls built from the
 * occupancy grid, and per-object collision meshes that are the convex hull of each
 * object's own captured point cloud (falling back to a bbox when unavailable/unusable)
 * - never the raw point cloud itself (see the backend's usd_export.py docstring for
 * why). Generated lazily server-side on first request per (includePointcloud,
 * includeFragments) and cached, so this URL is safe to hit directly from a download
 * link/button. */
export function sceneUsdUrl(
  id: string,
  { includePointcloud = true, includeFragments = false } = {},
): string {
  const params = new URLSearchParams({
    include_pointcloud: String(includePointcloud),
    include_fragments: String(includeFragments),
  })
  return `/api/scenes/${id}/usd?${params.toString()}`
}

export interface PostCommandOptions {
  /** Natural-language command text, parsed via the VLM. Omit when `targetObjectId` is
   * given instead. */
  text?: string
  /** Skips the VLM parse entirely and plans a direct "goto" - used by double-click. */
  targetObjectId?: string
  /** Overrides the server's configured robot radius for this one command - pass the
   * same value the radius slider is currently showing, so what gets planned matches
   * what the map/object list said was reachable. */
  radius?: number
  /** The selected platform. Radius alone stopped being enough on 2026-09-10: it still
   * decides inflation, but the GRID is derived at the platform's HEIGHT, and a command
   * without this is planned on the DEFAULT platform's map. Measured on hero-74, Husky to
   * the desk: 2.216 m with it, 2.198 m without - so the object list and the command panel
   * printed two different lengths for one route, which is the exact thing
   * `path_length_m`'s contract exists to prevent. */
  robotId?: string | null
  /** World-meter [x, z] to plan from instead of the scene's fixed robot start - pass
   * the robot's current on-screen position so it continues from there instead of
   * teleporting back to the start on every command. */
  from?: [number, number]
}

/** Resolves with status "failed" (HTTP 200) for a normal planning failure; only
 * rejects (NotFoundError) when the mentioned object doesn't exist in the scene. */
export function postCommand(sceneId: string, opts: PostCommandOptions): Promise<CommandResponse> {
  return apiPost(`/scenes/${sceneId}/command`, {
    text: opts.text,
    target_object_id: opts.targetObjectId,
    radius: opts.radius,
    robot_id: opts.robotId,
    from: opts.from,
  })
}

/** Which objects are reachable at a given robot radius, plus the grid's connectivity
 * numbers there. Omit `radius` to get the server's configured default. */
/** `moves` are "what if that were somewhere else" offsets, one per moved object, as
 * `<object id>:<dx>:<dz>` in metres - see the backend's `_parse_moves`. An empty list
 * (the normal case) sends no parameter at all, so the request is byte-identical to the
 * one this function has always made and hits the same server-side cache. */
export function getReachability(
  sceneId: string,
  radius?: number,
  moves: readonly string[] = [],
  robotId?: string | null,
): Promise<ReachabilityResponse> {
  const params = new URLSearchParams()
  if (radius !== undefined) params.set('radius', String(radius))
  // `radius` alone is no longer the whole platform. It still decides inflation, but the
  // GRID is derived at the platform's HEIGHT, and with no robot_id the server falls back
  // to the default platform's - so a Go2 answer would be computed on a TurtleBot's map.
  if (robotId) params.set('robot_id', robotId)
  for (const move of moves) params.append('move', move)
  const query = params.size > 0 ? `?${params}` : ''
  return apiGet(`/scenes/${sceneId}/reachability${query}`)
}

/** The Monte-Carlo fit-probability audit (`scripts.audit.fit_prob`) for this scene,
 * keyed by platform id - 404s (NotFoundError) for the (currently common) case of a
 * scene with no audit on disk at its fixed `<scene_dir>/msa/fit_prob.json`
 * convention; ScenePage treats that the same as "no data yet" and falls back to
 * showing only the existing reachable-count display - see ReachabilityCard. */
export function getFitProb(sceneId: string): Promise<FitProbResponse> {
  return apiGet(`/scenes/${sceneId}/fit-prob`)
}

/** This scene's nvblox layer pack (ESDF slice + unobserved mask) on its own grid.
 * 404s (NotFoundError) for a scene with no pack sampled on that grid - the common case,
 * treated as "no ESDF layer to draw", never as an error. */
export function getSceneLayers(sceneId: string): Promise<SceneLayersResponse> {
  return apiGet(`/scenes/${sceneId}/layers`)
}

/** The aligned camera trajectory from the capture video - display only, backs the
 * (default-off) "camera path" toggle in both the 2D and 3D views. */
export function getCameraTrack(sceneId: string): Promise<CameraTrackResponse> {
  return apiGet(`/scenes/${sceneId}/camera-track`)
}

export function listCommands(
  sceneId: string,
  skip = 0,
  limit = 50,
): Promise<CommandListResponse> {
  return apiGet(`/scenes/${sceneId}/commands?skip=${skip}&limit=${limit}`)
}

export function deleteScene(id: string): Promise<void> {
  return apiDelete(`/scenes/${id}`)
}
