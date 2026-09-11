"""Application configuration loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the video backend service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://robot:robot@127.0.0.1:5432/robotdb"
    upload_dir: str = "./var/uploads"
    max_upload_bytes: int = 2_147_483_648  # 2 GiB

    # --- Queue ---
    redis_url: str = "redis://127.0.0.1:6379/0"

    # --- GPU orchestration ---
    # gpu_ssh_host defaults to the "gpu" alias already defined in ~/.ssh/config (direct
    # connection). Leaving port/key unset lets ssh fall through to that config file entirely.
    gpu_ssh_host: str = "gpu"
    gpu_ssh_port: int | None = None
    gpu_ssh_key: str | None = None
    gpu_workspace_dir: str = "/workspace"
    job_timeout_sec: int = 3600
    gpu_poll_interval_sec: int = 15
    keyframe_fps: float = 2.0
    max_keyframes: int = 200
    # Points kept in scene_points.glb. Matches the frontend's render budget
    # (frontend/src/lib/pointcloudDecimate.ts's MAX_POINTS) rather than the pipeline's
    # internal 1.5M working resolution (gpu/stage_align.py) - the GLB has exactly one
    # consumer (the 3D viewer), which previously downloaded 1.5M points just to
    # decimate 73% of them away client-side. See docs/DECISIONS.md.
    glb_max_points: int = 400_000
    # Opt-in: retain each frame's per-view artifacts (per_view/*.npz - full extrinsics +
    # intrinsics + raw depth/points - and per_view_png/ preview images) in the LOCAL
    # fetched scene directory instead of discarding them (gpu_client.fetch_results
    # excludes them by default). The remote copy is always deleted regardless (the
    # orchestrator's own `finally`-block `cleanup()` rm -rf's the whole remote job dir -
    # unrelated to this flag, left as-is). Off by default: this is a real disk-cost
    # increase per scene (up to max_keyframes=200 frames of PNG+npz each) that the
    # pose/intrinsics-retention spike hasn't yet validated is worth paying for every
    # scene going forward - see docs/SPRINT-poisson-meshing.md, Workstream C.
    retain_per_view_artifacts: bool = False
    # Opt-in: run gpu/plane_regularize.py's bounded RANSAC plane-position projection
    # on the aligned room point cloud (gpu/stage_align.py, after the SOR block). This
    # is the one pipeline step that edits observed point *positions* - each
    # plane-inlier point is moved only along that plane's own fitted normal, only by
    # its own signed distance to the plane, capped to <= gpu/plane_regularize.py's
    # DEFAULT_DIST_THRESH_M (4cm), never sideways, never extrapolated past the
    # inlier set's own observed footprint (see that module's docstring for the full
    # guardrail). Off by default: it's only been validated on a small spike scene, not
    # yet run at production scale - and it's a bigger intervention than it may sound,
    # since a full-scale dry run during design measured 77% of a real scene's points
    # claimed as planar. See docs/SPRINT-poisson-meshing.md, Workstream A, for the
    # write-up (including why `mode=project` was needed - normal-only override alone
    # did not fix the room-scale "rippled cave wall" artifact).
    regularize_planes: bool = False

    # --- OpenRouter (VLM calls: object vocabulary + command parsing) ---
    # Two separate models: vision (GPU-side object vocabulary, needs image input) and
    # command (backend-side text -> {action, target, destination} parsing). See
    # app/services/vlm_client.py and gpu/stage_vocab.py for where each is used.
    openrouter_api_key: str = ""
    openrouter_model_command: str = "nvidia/nemotron-3-nano-30b-a3b"
    openrouter_model_vision: str = "z-ai/glm-5.3-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Vertex AI / Agent Platform (primary scene-vocabulary provider as of
    # 2026-09-07, docs/DECISIONS.md; OpenRouter is the automatic fallback) ---
    # No API key: auth is via Application Default Credentials (gcloud auth
    # application-default login) - see app/services/vertex_auth.py. Personal identity,
    # not a service account (org policy forbids service-account keys) - acceptable for
    # this prototype, not for a real multi-tenant deployment. Required for the default
    # vocabulary path (video.vocab_provider unset or "vertex"); unset here degrades that
    # path to OpenRouter (pipeline_orchestrator.py logs it, doesn't fail the job).
    gcp_project_id: str | None = None

    # Where the nvblox layer packs live - one directory per reconstructed scene, each
    # holding that scene's ESDF slice and unobserved mask (see
    # app/services/scene_layers.py). Read-only, and OUTSIDE the scene tree: these are the
    # geometry side's own output, published for the viewer rather than produced by this
    # service's pipeline. A scene is matched to a pack by its GRID, never by its name -
    # see scene_layers.find_pack_for_grid.
    frontend_layers_dir: str = "./var/frontend_layers"

    # --- Robot / pathfinding ---
    # 0.10m is what ROBOTIS ships in its own Nav2 config for the TurtleBot3 Burger
    # (turtlebot3_navigation2/param/burger.yaml, robot_radius: 0.1, identical on
    # humble/jazzy/master). The assembled turtlebot3_description mesh measures 0.1045m at
    # its widest about the base origin - under half a grid cell (0.05m) from this value, so
    # the footprint the planner inflates and the model the 3D view renders agree.
    #
    # Do not "correct" this to 0.1045/0.105: inflate() takes ceil(r / 0.05), so anything
    # above 0.10 jumps from 2 cells to 3 - the same inflation as 0.15 - and drops the real
    # "ikport" scene (see GET /scenes/{id}/reachability) off its reachability plateau.
    # Measured there: 23/23 objects reachable at radius=0, 12/23 across the whole 0.01-0.10
    # range (a handful of real passages are exactly one grid cell wide), 7/23 at 0.25m
    # where the grid fragments into 8 disconnected islands. 0.10m is the largest radius
    # still on that 12/23 plateau.
    robot_radius_m: float = 0.10
    robot_speed_mps: float = 0.5


settings = Settings()
