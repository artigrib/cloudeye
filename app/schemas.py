"""Pydantic v2 request/response schemas."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.robots import Kinematics, RadiusSource


class ModelOptionResponse(BaseModel):
    id: str | None
    provider: str
    model: str


class ModelStageResponse(BaseModel):
    stage: str
    editable: bool
    options: list[ModelOptionResponse]
    default: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    db: Literal["ok", "error"]
    redis: Literal["ok", "error"]


class ProviderHealthResponse(BaseModel):
    available: bool
    latency_ms: float | None
    checked_at: datetime
    detail: str | None = None


class VideoResponse(BaseModel):
    """Metadata describing a stored video, returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID | None
    filename: str
    size_bytes: int
    format: str
    duration_sec: float | None
    resolution: str | None
    status: str
    blur_threshold: float | None
    similarity_threshold: float | None
    vocab_override: list | None
    vocab_provider: str | None
    created_at: datetime


class VideoListResponse(BaseModel):
    """A page of video metadata."""

    total: int
    skip: int
    limit: int
    items: list[VideoResponse]


class RerunResponse(BaseModel):
    """Response for POST /api/videos/{id}/rerun - the job is enqueued asynchronously
    (same as upload), so this doesn't carry the rerun's outcome, just confirmation it
    was queued. `job_id` is null (and `enqueued` False) only in the same "already in
    flight" case app.queue.enqueue_process_video documents - practically unreachable
    for a rerun, since its job id is always freshly generated, but kept for parity
    with that function's return contract."""

    video_id: uuid.UUID
    scene_id: uuid.UUID
    job_id: str | None
    enqueued: bool


# --- Projects ---------------------------------------------------------------


class ProjectCreate(BaseModel):
    """Body for creating a project explicitly."""

    name: str
    description: str | None = None


class ProjectUpdate(BaseModel):
    """Body for renaming a project, changing its primary scene, or archiving it. All
    fields optional; omitting `archived` leaves it unchanged (unlike primary_scene_id,
    it has no separate "clear" state to distinguish from "not provided")."""

    name: str | None = None
    description: str | None = None
    primary_scene_id: uuid.UUID | None = None
    archived: bool | None = None


class ProjectResponse(BaseModel):
    """A project (room), without its child scenes/videos."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    primary_scene_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    archived: bool


class ProjectListResponse(BaseModel):
    """A page of projects, for a sidebar-style listing."""

    total: int
    skip: int
    limit: int
    items: list[ProjectResponse]


class ProjectDetailResponse(BaseModel):
    """A single project plus its scenes and videos, for a project detail view."""

    project: ProjectResponse
    scenes: list["SceneSummary"]
    videos: list[VideoResponse]


# --- Scenes -------------------------------------------------------------------


class SceneSummary(BaseModel):
    """The lightweight scene shape embedded in a project detail response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    video_id: uuid.UUID
    label: str | None
    status: str
    created_at: datetime
    completed_at: datetime | None


class MsaObjectSummary(BaseModel):
    """One object from an MSA (Measured Scene Assembly) `objects.json`-shaped export -
    the canonical, de-cluttered object list (merged duplicates, class caps, support
    rule already applied - see scripts/msa/bootstrap.py), as opposed to
    `SceneObjectResponse`'s raw per-detection DB rows. Deliberately minimal (just
    what the frontend's object-count/grouping UI needs) - read-only, mirrors the file
    on disk, nothing persisted."""

    id: str
    label: str


class MsaObjectsResponse(BaseModel):
    """Response for GET /{scene_id}/msa-objects. `variant` names which candidate file
    under `<scene_dir>/msa/` was actually served - see
    scene_service.MSA_OBJECTS_CANDIDATES for the fallback order."""

    objects: list[MsaObjectSummary]
    variant: str


class SceneObjectResponse(BaseModel):
    """A single detected object in a scene, with its 3D position and bounding box."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    scene_id: uuid.UUID
    name: str
    description: str | None
    pos_x: float
    pos_y: float
    pos_z: float
    bbox_min_x: float
    bbox_min_y: float
    bbox_min_z: float
    bbox_max_x: float
    bbox_max_y: float
    bbox_max_z: float
    num_views: int
    num_points: int
    is_fragment: bool
    mesh_path: str | None
    created_at: datetime


class SceneObjectListResponse(BaseModel):
    """All objects detected in one scene."""

    scene_id: uuid.UUID
    total: int
    items: list[SceneObjectResponse]


class StageTimingResponse(BaseModel):
    """One pipeline stage transition, as observed in this scene's own
    `logs/worker.log` (see app.services.stage_timings). `duration_sec` is null for the
    most recently started stage while a job is still `processing` (no next transition
    to measure against yet)."""

    stage: str
    started_at: datetime
    duration_sec: float | None = None


class SceneResponse(BaseModel):
    """Full scene metadata plus its detected objects. `stage`/`stage_timings` are
    populated the same way as SceneStatusResponse's (see
    app.services.scene_service.apply_stage_timings) - useful while a scene the
    frontend already has open (ScenePage) is still `processing`, without a second
    polling loop against the lighter `/status` endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    video_id: uuid.UUID
    project_id: uuid.UUID
    label: str | None
    status: str
    stage: str | None = None
    stage_timings: list[StageTimingResponse] = Field(default_factory=list)
    error_message: str | None
    mesh_path: str | None
    pointcloud_path: str | None
    occupancy_path: str | None
    map_preview_path: str | None
    transform_path: str | None
    usd_path: str | None
    grid_resolution: float | None
    grid_origin_x: float | None
    grid_origin_z: float | None
    grid_width: int | None
    grid_height: int | None
    floor_y: float | None
    ceiling_y: float | None
    is_reflected: bool
    points_below_floor_frac: float | None
    reflective_floor_suspected: bool
    oversized_object_warning: str | None
    vocab_source: str | None
    vocab_model: str | None
    vocab_entries: list | None
    robot_start_x: float | None
    robot_start_z: float | None
    created_at: datetime
    completed_at: datetime | None
    objects: list[SceneObjectResponse] = Field(default_factory=list)


class SceneStatusResponse(BaseModel):
    """Status shape for frontend polling. No DB joins; does read this scene's small
    local `logs/worker.log` while `status == "processing"` to populate `stage`/
    `stage_timings` for the frontend's stage-by-stage ETA display (see
    app.routers.scenes.get_scene_status) - a deliberate, small trade-off against this
    endpoint's original "no file I/O" invariant, logged in docs/DECISIONS.md."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    stage: str | None = None
    stage_timings: list[StageTimingResponse] = Field(default_factory=list)
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None


class OccupancyGridResponse(BaseModel):
    """The 2D occupancy grid, in world-meter metadata plus a cell-value matrix."""

    resolution: float
    origin_x: float
    origin_z: float
    width: int
    height: int
    legend: dict[str, str] = Field(
        default_factory=lambda: {"0": "free", "1": "obstacle", "2": "unknown"}
    )
    cells: list[list[int]]
    #: False when this scene has no navigation layer - `cells`/`unobserved` are then empty
    #: and `width`/`height` are 0. Same contract as ReachabilityResponse: always 200, two
    #: shapes, no error branch.
    layer_available: bool = True
    layer_unavailable_reason: str | None = None
    #: `[ix][iz]`, True where the scan never covered that cell. Not derivable from
    #: `cells`: `cells` is this robot's tri-state view, and a cell holding only a high
    #: surface reads FREE for a short robot while still being ground nobody observed.
    unobserved: list[list[bool]] = Field(default_factory=list)


# --- Commands -----------------------------------------------------------------


class CommandCreate(BaseModel):
    """Body for issuing a robot command against a scene: either a natural-language
    `text` (parsed via the VLM) or a `target_object_id` (a direct "goto", no VLM call -
    used by the frontend's double-click-to-go interaction, where the object is already
    known)."""

    model_config = ConfigDict(populate_by_name=True)

    text: str | None = None
    target_object_id: uuid.UUID | None = None
    # Which robot platform (app/robots.py registry id, e.g. "burger") to plan with -
    # sets the radius the platform selector shows by default. Unset/unknown falls back
    # to the default platform, same as before platforms existed (see
    # app.robots.resolve_radius_m).
    robot_id: str | None = None
    # Optional per-command override of the effective radius - lets the frontend's radius
    # slider (see GET .../reachability) plan with the same radius it's showing the user,
    # without changing the configured default for every other command. Always wins over
    # `robot_id`'s platform radius (see app.robots.resolve_radius_m) - this is how the
    # slider keeps overriding the platform's starting value. Bounds match the slider's
    # own range, with a little headroom on each side.
    radius: float | None = Field(default=None, ge=0.02, le=0.6)
    # World-meter (x, z) to plan from, instead of the scene's fixed `robot_start_x/z` -
    # lets the frontend plan from wherever the robot visually is (mid-animation or
    # stopped at the end of its last path) rather than teleporting it back to the start
    # on every command. Session-only state on the frontend - never persisted server-side.
    from_: list[float] | None = Field(default=None, alias="from", min_length=2, max_length=2)

    @model_validator(mode="after")
    def _check_text_or_target(self) -> "CommandCreate":
        if not self.text and self.target_object_id is None:
            raise ValueError("either 'text' or 'target_object_id' is required")
        return self


class CommandStep(BaseModel):
    """A single step in a planned robot command (a move, pick, or place)."""

    type: Literal["move", "pick", "place"]
    path: list[list[float]] | None = None
    object: str | None = None
    position: list[float] | None = None
    at: str | None = None
    duration_sec: float
    # Route length in metres for a "move" step (app.services.pathfinding.PathResult.
    # length_m) - None for "pick"/"place" (no travel). Added alongside duration_sec so
    # the panel can show the runtime planner's own route length, not just its time -
    # see CommandResponse.total_length_m for the command-level total.
    length_m: float | None = None


class CommandResponse(BaseModel):
    """The parsed action and planned path/steps for one command."""

    model_config = ConfigDict(from_attributes=True)

    command_id: uuid.UUID
    scene_id: uuid.UUID
    user_text: str
    action: str | None = None
    parsed_action: dict[str, Any] | None = None
    steps: list[CommandStep] = Field(default_factory=list)
    total_duration_sec: float | None = None
    # Sum of every "move" step's length_m (None if there are none, e.g. a failed
    # command with no steps at all) - the runtime planner's own total route length,
    # in metres, alongside total_duration_sec.
    total_length_m: float | None = None
    # Which chat LLM parsed this command's text - one of vlm_client.CHAT_PROVIDERS, taken
    # from the workspace's job spec (`JobSpec.chat_llm`). None means no model was asked:
    # a direct `target_object_id` goto, or a command from before this field existed.
    # Recorded so an answer can be traced to the model that produced it - "which model
    # said that" is otherwise unanswerable after the fact, and the spec allows two.
    provider_used: str | None = None
    # A machine-readable tag for the ONE failure that is about the deployment rather than
    # about the command: "chat_llm_not_connected", when the configured chat LLM has no
    # credentials here. None for every other outcome, including every ordinary failure
    # ("no path found", "no such object"), which belong to the command that caused them.
    # A code rather than a regex over `error_message`: the message names whichever
    # credential is missing (OPENROUTER_API_KEY, GCP_PROJECT_ID, ADC), and a client that
    # has to keep a pattern in sync with that list will eventually miss one.
    error_code: str | None = None
    status: str
    error_message: str | None = None
    created_at: datetime


class ReachabilityResponse(BaseModel):
    """Which of a scene's objects the robot can actually path to at a given radius, plus
    the grid-connectivity numbers behind that (see pathfinding.compute_reachability).

    ALWAYS 200. A scene either has a verdict or says, in this body, that it has no
    navigation layer to give one from - it never answers 5xx and never answers 409, so a
    client has exactly two shapes to render and no error branch to guess at. See
    `layer_available`."""

    #: False when this scene has no navigation layer at all - a scene reconstructed before
    #: the band layer existed and never backfilled. Everything below is then empty rather
    #: than absent: no objects reachable, no reasons, no cells, `start_status` "none".
    #: That is NOT the same answer as "this robot fits nowhere", which is also empty and
    #: also a 200 - hence the flag rather than leaving a client to infer one from the
    #: other. `layer_unavailable_reason` carries the operator-facing sentence, including
    #: the command that produces the layer.
    layer_available: bool = True
    layer_unavailable_reason: str | None = None

    scene_id: uuid.UUID
    radius: float
    # Which registry platform (app/robots.py) `radius` was resolved for, if any -
    # None when the caller passed an explicit `radius` without a `robot_id`, or when
    # `robot_id` wasn't a registered platform. Purely informational (echoes the
    # request back); defaults to None so constructing this response without it (e.g.
    # older call sites) stays valid.
    robot_id: str | None = None
    component_count: int
    largest_component_size: int
    start_component_size: int
    reachable_object_ids: list[uuid.UUID]
    # Object id -> why it's NOT reachable: "outside_grid" (its centroid is off the
    # occupancy grid entirely - bad data for that object), "disconnected" (resolved to
    # a valid cell, just not connected to the robot's start component), or
    # "robot_does_not_fit" (every object, when the robot's own start point has no free
    # cell within 0.5m at this radius - see pathfinding.compute_reachability). Defaults
    # to {} so constructing this response without it (e.g. older call sites) stays valid.
    unreachable_reasons: dict[uuid.UUID, str] = {}
    # Object id -> the length in metres of the route from this platform's start to that
    # object. Keyed on exactly `reachable_object_ids` and nothing else: an unreachable
    # object has no route, and a 0.0 would read as "it is already there". Planned by the
    # same astar + smoothing a real `goto` uses over the same cost grid, so the object
    # list and the command panel never print two different lengths for one route - see
    # pathfinding.ReachabilityResult.path_length_m. Defaults to {} so constructing this
    # response without it (older call sites, tests) stays valid.
    path_length_m: dict[uuid.UUID, float] = {}
    # Which objects this answer was computed with MOVED, from the request's `move`
    # parameters (see routers/scenes._parse_moves). Empty for every ordinary request.
    # Present so a client can never mistake a what-if answer for the room's real numbers:
    # a non-empty list here is the signal to label the answer approximate.
    moved_object_ids: list[uuid.UUID] = []
    # [ix][iz], same shape/convention as OccupancyGridResponse.cells - True where that
    # cell is in the robot's start component at this radius.
    reachable_cells: list[list[bool]]
    #: How many distinct cells the routes behind `reachable_object_ids` cross that nobody
    #: ever observed. UNKNOWN is traversable at 1.5x rather than blocked (blocking it
    #: measures 0/17 for every robot - see pathfinding.build_cost_grid), so a reachable
    #: count on a sparsely-scanned scene can lean on ground the scan never covered. This
    #: is the number that says how much.
    unknown_cells_on_route: int = 0
    #: How many cells those routes cross that the layer calls OBSTACLE. Always 0 - it is
    #: exposed so the gate can assert on a number instead of on a promise. Not
    #: tautological: routes are planned on the inflated cost grid and this counts against
    #: the tri-state grid `GET /map?robot_id=...` serves, so it is a cross-check between
    #: the grid a route was planned on and the grid the user is looking at. That is the
    #: pair that came apart under the old 0.1-0.5 m histogram, which could not see a table
    #: top and let routes be drawn through furniture.
    obstacle_cells_on_route: int = 0
    # Where the robot actually stands AT THIS RADIUS, re-derived per request by
    # pathfinding.resolve_start_for_radius rather than read from the scene's persisted
    # robot_start_x/z (which was decided once, at whatever radius was default then).
    # "original" - the persisted start fits this platform, start_x/z echo it.
    # "moved"    - it does not; start_x/z is the nearest cell that does, start_moved_m
    #              away. The panel says so instead of quietly planning from elsewhere.
    # "none"     - no cell in the whole grid has clearance >= this radius, so this
    #              platform has nowhere to stand in this scene at all; start_x/z are
    #              null and every object is "robot_does_not_fit". That is a real answer
    #              about the scene, not a failure, and reads very differently from 0/N.
    start_status: str = "none"
    start_x: float | None = None
    start_z: float | None = None
    start_moved_m: float = 0.0


class SceneLayersResponse(BaseModel):
    """One scene's nvblox layer pack, on that scene's own occupancy grid.

    Same [ix][iz] indexing and the same origin/resolution as OccupancyGridResponse - the
    pack is only served when its grid matches the scene's exactly (see
    app/services/scene_layers.find_pack_for_grid), so a client can draw these arrays on
    the same cells it draws the occupancy grid on, with no resampling and no second
    coordinate convention to get wrong."""

    scene_id: uuid.UUID
    #: The pack directory's name, and which of its grid blocks matched.
    pack: str
    variant: str
    #: Echoed from the scene's own grid so a client can check what it is drawing against
    #: the geometry it already has, rather than assuming the two agree.
    resolution: float
    origin_x: float
    origin_z: float
    width: int
    height: int
    #: Height above the floor the ESDF slice was cut at, metres. This is a SLICE, not a
    #: band: see scene_layers' module docstring for the measured difference and why the
    #: viewer must not present one as the other.
    slice_height_m: float | None = None
    #: Signed distance to the nearest obstacle in metres, [ix][iz]; null where the cell
    #: was never observed (NaN in the source array, which JSON cannot carry).
    esdf_m: list[list[float | None]]
    #: True where the cell was never observed, [ix][iz].
    unobserved: list[list[bool]]
    #: How many cells carry a value at all, and how many of those the slice calls
    #: obstacle (esdf <= 0) - stated so a client can show the coverage rather than draw a
    #: mostly-empty layer as if it were complete.
    observed_cells: int
    obstacle_cells: int
    #: The largest ESDF value present, metres. The slice saturates well below a real
    #: clearance field's range, which is exactly why it is not used as one.
    esdf_max_m: float | None = None
    #: The pack's own verdict on this variant, verbatim, when it records one.
    quality: str | None = None


class CameraPoseResponse(BaseModel):
    """One keyframe's aligned camera position, in the scene's world frame - same axes as
    every other 3D position this API returns (objects, robot_start_x/z, ...)."""

    x: float
    y: float
    z: float
    # Video-playback time this pose corresponds to, seconds - null when it couldn't be
    # reconstructed for this scene (see scene_ingest.read_keyframe_timestamps).
    timestamp_sec: float | None


class CameraTrackResponse(BaseModel):
    """The aligned camera trajectory from the capture video, in capture order - "where
    the person walked while filming." Display only (see app/services/scene_ingest.py's
    read_camera_track/read_keyframe_timestamps): never recomputed, never touches
    pathfinding or the alignment stage."""

    scene_id: uuid.UUID
    poses: list[CameraPoseResponse] = Field(default_factory=list)


class CommandListResponse(BaseModel):
    """A page of a scene's command history."""

    scene_id: uuid.UUID
    total: int
    skip: int
    limit: int
    items: list[CommandResponse]


# --- Robots ---------------------------------------------------------------------


class RobotDimensionsResponse(BaseModel):
    """Overall footprint of a robot platform, meters."""

    model_config = ConfigDict(from_attributes=True)

    length_m: float
    width_m: float
    height_m: float


class RobotPlatformResponse(BaseModel):
    """One entry from the robot platform registry (app/robots.py) - backs the
    frontend's platform selector. `radius_m`/`dimensions_m` are null for a platform
    whose mesh hasn't been measured yet; see `notes`."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    vendor: str
    kinematics: Kinematics
    mesh_path: str
    license_path: str
    dimensions_m: RobotDimensionsResponse | None
    radius_m: float | None
    radius_source: RadiusSource
    reach_m: float | None = None
    notes: str = ""


class RobotListResponse(BaseModel):
    """Every registered robot platform, for a selector dropdown."""

    default_robot_id: str
    total: int
    items: list[RobotPlatformResponse]


class ProcessingStageResponse(BaseModel):
    """One row of the processing screen's feed - a stage pipeline-v1 entered.

    `state` is passed through verbatim from status.json rather than mapped onto an enum,
    so a value this app has not seen before shows up as itself instead of silently
    becoming "unknown". See docs/JOB_SPEC.md section 2.
    """

    stage: str
    state: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    #: What the step is doing inside its state - one of run_pipeline.SUBSTAGES, passed
    #: through verbatim like `state`. None when the writer recorded none, which is every
    #: field below for a run written before these existed; the screen omits the row
    #: rather than printing a placeholder.
    substage: str | None = None
    detail: str | None = None
    #: When the CURRENT SUBSTAGE began, not when the step did.
    since: str | None = None
    #: Present and > 0 only once a retry loop is counting. 0 is the initial value and is
    #: not an attempt; it is sent as-is and filtered by the client.
    attempt: int | None = None
    next_retry: str | None = None


class ProcessingStatusResponse(BaseModel):
    """Everything GET /api/workspaces/{id}/processing tells the screen. The screen keeps
    no state of its own, so this is the whole of what it renders."""

    workspace_id: uuid.UUID
    workspace_name: str
    #: None until the pipeline has created a scene row for this workspace.
    scene_id: uuid.UUID | None = None
    #: "status.json" | "worker.log" | "none" - which source the stages came from, shown so
    #: a degraded feed is never mistaken for the authoritative one.
    source: str
    state: str | None = None
    error: str | None = None
    stages: list[ProcessingStageResponse] = []
    #: Seconds since the newest timestamp in either source, or None if there has never
    #: been one. The screen says "worker not connected" rather than spinning once this
    #: passes processing_status.WORKER_SILENT_AFTER_SEC, or when it is None.
    seconds_since_last_record: float | None = None
    worker_connected: bool = False
    is_done: bool = False
    is_failed: bool = False
    #: Ready-made sentence for the provider-fallback case, or None when no fallback
    #: happened - which is every run today, since pipeline-v1 writes neither
    #: provider_used nor fallback_reason (docs/JOB_SPEC.md section 2).
    fallback_notice: str | None = None
    #: The raw reason, alongside the ready-made `fallback_notice` sentence: the screen
    #: banners the reason itself, and a sentence that has already folded it in cannot be
    #: shown next to it without saying it twice.
    fallback_reason: str | None = None
    requested_provider: str | None = None
    provider_used: str | None = None
