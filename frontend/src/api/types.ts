// Mirrors app/schemas.py (Pydantic v2 models) on the CloudEye backend.

export type SceneStatus = 'queued' | 'processing' | 'done' | 'failed'

export interface ModelOptionResponse {
  /** Selector value for an editable stage's option; null for a fixed stage's single option. */
  id: string | null
  provider: string
  model: string
}

export interface ModelStageResponse {
  stage: string
  editable: boolean
  options: ModelOptionResponse[]
  /** Matches an option's `id`; only meaningful when `editable` is true. */
  default: string | null
}

export interface ProviderHealthResponse {
  available: boolean
  latency_ms: number | null
  checked_at: string
  detail: string | null
}

export interface VideoResponse {
  id: string
  project_id: string | null
  filename: string
  size_bytes: number
  format: string
  duration_sec: number | null
  resolution: string | null
  status: string
  /** Per-upload override of the keyframe-extraction blur filter. Null = stage default (100.0). */
  blur_threshold: number | null
  /** Per-upload override of the keyframe-extraction similarity filter. Null = stage default (0.98). */
  similarity_threshold: number | null
  /** Fixed vocabulary skipping the scene-vocabulary provider call entirely. Null = normal detection. */
  vocab_override: unknown[] | null
  /** Which scene-vocabulary provider to use ("openrouter"/"vertex"). Null = default (openrouter). */
  vocab_provider: string | null
  created_at: string
}

export interface VideoListResponse {
  total: number
  skip: number
  limit: number
  items: VideoResponse[]
}

export interface ProjectResponse {
  id: string
  name: string
  description: string | null
  primary_scene_id: string | null
  created_at: string
  updated_at: string
  archived: boolean
}

export interface ProjectListResponse {
  total: number
  skip: number
  limit: number
  items: ProjectResponse[]
}

export interface SceneSummary {
  id: string
  video_id: string
  label: string | null
  status: SceneStatus
  created_at: string
  completed_at: string | null
}

export interface ProjectDetailResponse {
  project: ProjectResponse
  scenes: SceneSummary[]
  videos: VideoResponse[]
}

export interface SceneObjectResponse {
  id: string
  scene_id: string
  name: string
  description: string | null
  pos_x: number
  pos_y: number
  pos_z: number
  bbox_min_x: number
  bbox_min_y: number
  bbox_min_z: number
  bbox_max_x: number
  bbox_max_y: number
  bbox_max_z: number
  num_views: number
  num_points: number
  is_fragment: boolean
  mesh_path: string | null
  created_at: string
}

/** M7b addition (c): one object from the canonical de-cluttered MSA object list -
 * see app.schemas.MsaObjectSummary. */
export interface MsaObjectSummary {
  id: string
  label: string
}

export interface MsaObjectsResponse {
  objects: MsaObjectSummary[]
  variant: string
}

export interface SceneObjectListResponse {
  scene_id: string
  total: number
  items: SceneObjectResponse[]
}

export interface SceneResponse {
  id: string
  video_id: string
  project_id: string
  label: string | null
  status: SceneStatus
  stage: string | null
  stage_timings: StageTimingResponse[]
  error_message: string | null
  mesh_path: string | null
  pointcloud_path: string | null
  occupancy_path: string | null
  map_preview_path: string | null
  transform_path: string | null
  usd_path: string | null
  grid_resolution: number | null
  grid_origin_x: number | null
  grid_origin_z: number | null
  grid_width: number | null
  grid_height: number | null
  floor_y: number | null
  ceiling_y: number | null
  is_reflected: boolean
  /** Non-fatal object_max_extent flag(s), e.g. "bed (2.92m)" - null if none. Doesn't
   * block the scene; a real object this size is very likely a merged-cluster artifact. */
  oversized_object_warning: string | null
  /** vocab.json's "source" - "openrouter"/"vertex"/"override"/"default_no_key"/etc. */
  vocab_source: string | null
  /** The exact model string actually used (e.g. "google/gemma-4-31b-it"), if any. */
  vocab_model: string | null
  robot_start_x: number | null
  robot_start_z: number | null
  created_at: string
  completed_at: string | null
  objects: SceneObjectResponse[]
}

/** One pipeline stage transition, as observed in this scene's own `logs/worker.log`
 * (see app/services/stage_timings.py). `duration_sec` is null for the most recently
 * started stage while the job is still `processing` - no next transition yet to
 * measure against. */
export interface StageTimingResponse {
  stage: string
  started_at: string
  duration_sec: number | null
}

export interface SceneStatusResponse {
  id: string
  status: SceneStatus
  stage: string | null
  stage_timings: StageTimingResponse[]
  error_message: string | null
  created_at: string
  completed_at: string | null
}

export interface OccupancyGridResponse {
  resolution: number
  origin_x: number
  origin_z: number
  width: number
  height: number
  legend: Record<string, string>
  cells: number[][] // cells[ix][iz]
  /** cells[ix][iz], true where the scan never covered that cell. NOT derivable from
   * `cells`: `cells` is the selected robot's tri-state view, and a cell holding only a
   * high surface reads FREE for a short robot while still being ground nobody observed.
   * This is the hatch source for the Occupancy render - see lib/occupancyLayers.ts. */
  unobserved: boolean[][]
  /** False when this scene has no navigation layer at all. `cells`/`unobserved` are then
   * empty and width/height are 0. The endpoint is 200 either way: "no layer" is a state
   * of the scene, not a failed request, so there is no error branch to write. */
  layer_available: boolean
  /** The operator-facing sentence, naming the command that produces a layer. */
  layer_unavailable_reason: string | null
}

export interface ReachabilityResponse {
  scene_id: string
  radius: number
  component_count: number
  largest_component_size: number
  start_component_size: number
  reachable_object_ids: string[]
  /** Object id -> the backend's own word for why it is NOT reachable: one of
   * "outside_grid", "disconnected", "no_free_cell_within_2m",
   * "no_visible_cell_within_2m", "robot_does_not_fit" (app/schemas.py). Shown verbatim -
   * see lib/unreachableReason.ts. Only ids absent from `reachable_object_ids` appear. */
  unreachable_reasons: Record<string, string>
  /** Object id -> the length in metres of the route from this platform's start to it,
   * keyed on exactly `reachable_object_ids`. Planned by the same astar + smoothing a
   * real `goto` uses, so the object list and the command panel never print two
   * different lengths for one route. An unreachable object is absent, never 0. */
  path_length_m: Record<string, number>
  reachable_cells: boolean[][] // reachable_cells[ix][iz]
  /** Where the robot actually stands AT THIS RADIUS - re-derived per request from the
   * first camera pose, not read from the scene's persisted robot_start_x/z (which was
   * decided once, at whatever radius was default then). See the backend schema.
   *  - 'original': the anchor fits this platform; start_x/z echo it, start_moved_m 0.
   *  - 'moved':    it does not; start_x/z is the nearest cell that does.
   *  - 'none':     no cell in the grid has clearance >= this radius, so this platform
   *                has nowhere to stand in this scene; start_x/z are null. This is what
   *                a truthful "0 reachable" looks like, and the panel must say it in
   *                those words rather than showing 0/N. */
  start_status: 'original' | 'moved' | 'none'
  start_x: number | null
  start_z: number | null
  start_moved_m: number
  /** How many DISTINCT cells the routes behind `reachable_object_ids` cross that nobody
   * ever observed. Unknown is traversable at 1.5x rather than blocked - blocking it
   * measures 0 of 17 reachable for every robot on the hero scene - so a reachable count
   * can lean on ground the scan never covered, and the verdict bar says by how much. */
  unknown_cells_on_route: number
  /** Always 0. The gate's number, not the UI's: routes are planned on the inflated cost
   * grid and this counts them against the grid `/map` serves, so a non-zero here means
   * the picture and the plan have come apart. See demo/probe_route_obstacles.py. */
  obstacle_cells_on_route: number
  /** False when this scene has no navigation layer. Every count above is then zero and
   * `start_status` is 'none' - which is ALSO what "this robot fits nowhere in this room"
   * looks like, so this flag is the only thing separating them. Rendering the second as
   * "0 of 17 reachable" would be lying with arithmetic. */
  layer_available: boolean
  layer_unavailable_reason: string | null
}

/** One platform's row from `scripts.audit.fit_prob`'s output - see that module's
 * docstring for the Monte-Carlo methodology (N jittered-geometry A* trials). */
export interface FitProbPlatformResult {
  platform_id: string
  display_name: string
  radius_m: number
  trials: number
  fits_count: number
  /** null when `fits_count` is 0: there is no successful trial to take a percentile
   * over, so scripts/audit/fit_prob.py writes null rather than 0. On the hero scene
   * Husky A200 (radius 0.5528 m) is exactly that - 0/100 fits, all three null. */
  corridor_width_p5_m: number | null
  corridor_width_p50_m: number | null
  corridor_width_p95_m: number | null
  first_block_point: { x: number; z: number; label: string; count: number; blocked_trials: number } | null
}

/** GET /scenes/{id}/fit-prob - `<scene_dir>/msa/fit_prob.json` verbatim, keyed by
 * platform id (app/robots.py). 404s for the (currently common) case of a scene with
 * no fit-probability audit on disk - see api/scenes.ts's sceneFitProbUrl/getFitProb. */
export interface FitProbResponse {
  schema: string
  seed: number
  trials: number
  platforms: Record<string, FitProbPlatformResult>
}

/** GET /scenes/{id}/layers - this scene's nvblox layer pack, on the scene's own
 * occupancy grid. 404s (NotFoundError) when no pack is sampled on that grid, which is
 * the common case and not an error; the 2D map then draws the first two layers only.
 * Same [ix][iz] indexing, origin and resolution as OccupancyGridResponse - the backend
 * serves a pack only when the two grids match exactly, so these arrays need no
 * resampling and introduce no second coordinate convention. */
export interface SceneLayersResponse {
  scene_id: string
  pack: string
  variant: string
  resolution: number
  origin_x: number
  origin_z: number
  width: number
  height: number
  /** Height above the floor the ESDF was sliced at. A SLICE, not a band - see the
   * backend's scene_layers docstring for the measured difference. */
  slice_height_m: number | null
  /** Signed distance to the nearest obstacle, metres, [ix][iz]; null = never observed. */
  esdf_m: (number | null)[][]
  unobserved: boolean[][]
  observed_cells: number
  obstacle_cells: number
  esdf_max_m: number | null
  /** The pack's own verdict on this variant, verbatim. */
  quality: string | null
}

export interface CameraPoseResponse {
  x: number
  y: number
  z: number
  /** Video-playback time this pose corresponds to, seconds - null when it couldn't be
   * reconstructed for this scene (see the backend's scene_ingest.read_keyframe_timestamps). */
  timestamp_sec: number | null
}

/** The aligned camera trajectory from the capture video, in capture order - "where the
 * person walked while filming." Display only. */
export interface CameraTrackResponse {
  scene_id: string
  poses: CameraPoseResponse[]
}

export type CommandAction = 'take' | 'goto' | 'look'
export type CommandStatus = 'pending' | 'done' | 'failed'

export interface CommandStep {
  type: 'move' | 'pick' | 'place'
  path: [number, number][] | null
  object: string | null
  position: [number, number, number] | null
  at: string | null
  duration_sec: number
  /** Route length in metres for a "move" step (the runtime planner's own
   * app.services.pathfinding.PathResult.length_m) - null for "pick"/"place". */
  length_m: number | null
}

export interface CommandResponse {
  command_id: string
  scene_id: string
  user_text: string
  action: CommandAction | null
  parsed_action: Record<string, unknown> | null
  steps: CommandStep[]
  total_duration_sec: number | null
  /** Sum of every "move" step's length_m - the runtime planner's own total route
   * length in metres, alongside total_duration_sec. */
  total_length_m: number | null
  /** Which chat LLM parsed this command's text - one of the job spec's `chat_llm`
   * values. null when no model was asked: a direct goto, or a command recorded before
   * this field existed. */
  provider_used: string | null
  /** "chat_llm_not_connected" when the configured chat LLM has no credentials on this
   * deployment; null for every other outcome. See the backend schema for why this is a
   * code and not a pattern over `error_message`. */
  error_code: string | null
  status: CommandStatus
  error_message: string | null
  created_at: string
}

export interface CommandListResponse {
  scene_id: string
  total: number
  skip: number
  limit: number
  items: CommandResponse[]
}

// Mirrors app/robots.py's registry - see app/routers/robots.py and app/schemas.py's
// RobotPlatformResponse. Closed enums on the backend (real Python `enum.Enum`s, not free
// strings - see app/robots.py's module docstring), kept as literal unions here for the
// same reason: a `switch` over `radius_source` should fail to typecheck if a ninth value
// is ever added without updating this file too.
export type Kinematics =
  | 'differential_drive'
  | 'skid_steer'
  | 'ackermann'
  | 'legged'
  | 'mobile_manipulator'

export type RadiusSource = 'vendor_nav2' | 'mesh_measured' | 'gait_approximate'

export interface RobotDimensionsResponse {
  length_m: number
  width_m: number
  height_m: number
}

export interface RobotPlatformResponse {
  id: string
  display_name: string
  vendor: string
  kinematics: Kinematics
  /** Relative to frontend/public/, e.g. "models/limo.glb" - prefix with "/" for a usable URL. */
  mesh_path: string
  license_path: string
  dimensions_m: RobotDimensionsResponse | null
  radius_m: number | null
  radius_source: RadiusSource
  reach_m: number | null
  notes: string
}

export interface RobotListResponse {
  default_robot_id: string
  total: number
  items: RobotPlatformResponse[]
}

/** One scripts/audit/critic_rules.py finding - see that module's docstring for the
 * exact check list (yaw_vs_wall, bbox_vs_class_limit, centroid_outside_room,
 * asset_centroid_vs_hull, missing_support_reason). Always `source: 'rule'`,
 * `unverified: false` - deterministic, threshold-checked, gating (PM decision,
 * 2026-09-07: "the rules engine is the gate; the VLM is advisory only"). */
export interface CriticRuleFinding {
  check: string
  source: 'rule'
  unverified: false
  object_id: string
  severity: 'high' | 'medium' | 'info' | string
  measured: number | null
  threshold: number | null
  detail: string
}

/** One scripts/audit/critic_vlm.py finding - the model's own free-text discrepancy
 * report for one real-frame-vs-render pair (see scripts/audit/prompts/critic_vlm.md).
 * Always `source: 'vlm'`, `unverified: true` - advisory/corroborating only, NEVER
 * gating (same PM decision as CriticRuleFinding above). CriticBadge.tsx's count and
 * colour must never be derived from these. */
export interface CriticVlmFinding {
  object_id_or_region: string
  issue: string
  severity: 'high' | 'medium' | 'low' | string
  view: string
  source: 'vlm'
  unverified: true
}

/** GET /api/scenes/{id}/critic - the merged report scripts/audit/critic_merge.py
 * writes to `<scene_dir>/msa/critic.json` (mirrors msa-glb: optional, 404 when no
 * critic pass has been run for this scene yet). Report-only - nothing in this shape
 * or the route that serves it ever mutates scene data. */
export interface CriticReportResponse {
  scene: string
  rules: {
    findings: CriticRuleFinding[]
    n_findings: number
    notes: string[]
  }
  vlm: {
    findings: CriticVlmFinding[]
    n_findings: number
    model: string
  } | null
  /** The rules-gate/VLM-advisory statement in machine-readable form - `gates` is
   * always false (the VLM never gates); `corroboration_sentence` is the PM's own
   * wording, e.g. "23 VLM findings, 1 corroborated". */
  vlm_advisory: {
    gates: false
    note: string
    corroboration_sentence: string
  }
  summary: {
    n_rule_findings: number
    n_vlm_findings: number
    n_overlap: number
  }
}

/** One row of the processing screen's stage feed. `state` is whatever pipeline-v1 wrote
 * ("running" | "ok" | "failed" | ...), passed through rather than mapped, so an
 * unfamiliar value shows as itself instead of silently becoming "unknown". */
export interface ProcessingStageResponse {
  stage: string
  state: string
  started_at: string | null
  finished_at: string | null
  duration_sec: number | null
  error: string | null
  /** What the step is doing INSIDE its state - one of pipeline-v1's
   * `run_pipeline.SUBSTAGES`, passed through verbatim like `state` so an unfamiliar
   * value shows as itself. null when the writer recorded none, which is every field
   * below for a run written before they existed; the screen then omits the row rather
   * than printing a placeholder. */
  substage: string | null
  detail: string | null
  /** When the CURRENT SUBSTAGE began - not when the step did. The two differ exactly
   * when a step is waiting, which is the case worth showing. */
  since: string | null
  /** 0 until a retry loop starts counting; `Status.enter` writes the zero. The server
   * passes it through deliberately, and lib/progressRows decides that 0 is not an
   * attempt. */
  attempt: number | null
  next_retry: string | null
}

export interface ProcessingStatusResponse {
  workspace_id: string
  workspace_name: string
  /** null until the pipeline has created a scene row for this workspace. */
  scene_id: string | null
  /** "status.json" | "worker.log" | "none" - which source the stages came from. */
  source: string
  state: string | null
  error: string | null
  stages: ProcessingStageResponse[]
  seconds_since_last_record: number | null
  worker_connected: boolean
  is_done: boolean
  is_failed: boolean
  /** Ready-made sentence for the provider-fallback case, or null when none happened -
   * which is every run today (docs/JOB_SPEC.md section 2). */
  fallback_notice: string | null
  requested_provider: string | null
  provider_used: string | null
  /** The raw reason, beside the ready-made `fallback_notice` sentence: the banner shows
   * the reason itself, and a sentence that has already folded it in cannot sit next to
   * it without saying it twice. */
  fallback_reason: string | null
}
