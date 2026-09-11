#!/usr/bin/env python3
"""Record a reproducible Isaac Sim demo clip: two robot platforms (TurtleBot3 Burger,
then Husky A200 - app/robots.py registry) each follow the SAME planned route
(/World/Path, baked into the export by app/services/usd_export.py from a real
app.services.pathfinding.plan_to_object call) toward a target object (/World/Target,
default "table"). Burger's small radius clears the route; Husky's much larger radius
is expected to physically get blocked wherever the route runs through a gap too narrow
for it - the point of running the same path twice, not two separately-planned ones.

Clip structure (fixed order, each phase's duration is a CLI flag):
    establish (wide shot, nothing moving yet) -> Burger drives the path -> Husky drives
    the same path -> a small sphere is dropped over the route START (quick, independent
    collision sanity check; T15h - never over the target, whose column may hold an
    object hull) -> a closing static hold. 1920x1080 @ 60fps by default.
    spec.result.json carries `burger.dist_to_target_m` / `reached_target` against
    /World/Target (= the route goal for MSA presentation exports) and
    `sphere_drop_point` (T15h).

Frames are assembled into an mp4 with ffmpeg: a caption, a live progress counter
counter, and a color-to-class legend (from usd_export's own per-object-class color
function, so the legend always matches what's actually in frame) are burned in via
drawtext filters.

WHY THIS FILE IS SPLIT IN TWO, AND WHY IT DOESN'T IMPORT app.* AT MODULE LEVEL:
Isaac Sim ships its own bundled Python interpreter (isaac-sim.sh / python.sh) with no
access to this repo's backend venv (no pydantic-settings, no sqlalchemy, ...) - the
same constraint tools/isaac_validate.py documents and works around. So this script has
two phases that normally run as two processes:

  Phase 1 ("driver", this repo's own venv - `uv run python demo/record_isaac.py ...`):
      parses CLI args, resolves both platforms from app.robots (real registry import,
      not a guess), resolves the target object and plans a route to it
      (app.services.pathfinding.plan_to_object) from a data-driven start point (the
      longest robot-radius-inflated free run in the occupancy grid), exports a
      demo-specific USD with that route baked in as /World/Path + /World/Target (NOT
      the generic cached export the /usd endpoint serves - that has no concept of a
      goal), writes a small JSON "job spec", then - IF --isaac-python-exe is given -
      re-invokes THIS SAME FILE under Isaac's interpreter for phase 2.

  Phase 2 ("worker", Isaac Sim's own interpreter, invoked as
      `<isaac-python> record_isaac.py --_isaac-worker-spec spec.json`):
      stdlib + lazily-imported pxr/isaacsim/omni only, no app.* imports at all - this
      is what actually opens the stage, runs physics, and renders. It only needs the
      driver-authored spec.json, the input USD, and ffmpeg on PATH.

If Isaac Sim isn't installed alongside this repo's checkout (true today - the backend
runs on one host, Isaac Sim 6.0.1 on a separate throwaway GPU instance with no network
share between them), run phase 1 without --isaac-python-exe to just get spec.json,
copy {this file, spec.json, the input USD} to the Isaac host yourself, and run phase 2
there directly - see docs/DECISIONS.md / DIAG.md for concrete examples from real runs.

Usage (phase 1, normal/eventual single-host case, auto-runs phase 2 too):
    uv run python demo/record_isaac.py <scene_id> out.mp4 --target table \
        --isaac-python-exe /path/to/isaac/python.sh

Usage (phase 1 only, split-host case):
    uv run python demo/record_isaac.py <scene_id> out.mp4 --target table \
        --spec-out /tmp/spec.json
    # then, on the Isaac host (T15i: copy scripts/msa/visibility.py - stdlib-only -
    # next to record_isaac.py as well; it holds the camera visibility rule):
    OMNI_KIT_ALLOW_ROOT=1 <isaac-python> record_isaac.py --_isaac-worker-spec /tmp/spec.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "cloudeye.record_isaac/5"  # /5 (morning-4): + path_visibility_min_fraction / baked worst case; worker re-solves with robot-along-route boxes + 0.3 m eye clearance


def _load_visibility_module():
    """T15i: `scripts/msa/visibility.py` (stdlib-only) holds the camera visibility rule
    shared with export_presentation. Phase 1 imports it from the repo; the Isaac-host
    worker is a copied file, so it also accepts a `visibility.py` sitting next to this
    script (copy it along with record_isaac.py, spec.json and the USD)."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    if (root / "scripts" / "msa" / "visibility.py").is_file() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts.msa import visibility as module  # type: ignore

        return module
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for candidate in (here / "visibility.py", here / "scripts" / "msa" / "visibility.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("cloudeye_msa_visibility", candidate)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        "scripts/msa/visibility.py not importable - on the Isaac host copy it next to record_isaac.py "
        "(it is stdlib-only) together with spec.json and the USD"
    )


visibility = _load_visibility_module()

# Sphere-drop / closing-static camera height above the floor. Same bug this fixed
# originally still applies: too low and the void below y=0 (nothing rendered there -
# no environment/skybox) fills the bottom of frame as a black band.
STATIC_CAMERA_HEIGHT_ABOVE_FLOOR_M = 1.2

# T15d: the wide/establishing shot (establish + both drive passes) follows the SAME
# rule as the web 3/4 render - scripts.msa.render_perspective.compute_camera (anchor =
# bed footprint centroid else room centroid, direction through the room polygon's
# farthest vertex continued outward past it, distance max(0.7*diagonal, 3m) from the
# room centroid, 2.6m above the floor, look-at = room centroid at floor height, pitch
# solved so the farthest floor point sits at 85% of frame height). Phase 1 (this
# repo's venv) either reads that camera straight from the USD (a presentation export
# bakes it at PRESENTATION_CAMERA_PATH, scripts/msa/export_presentation.py) or
# computes it by importing compute_camera itself - Phase 2 (Isaac's interpreter, no
# numpy/trimesh/shapely guaranteed) only ever reads the numbers from spec.json and
# applies the framing back-off below. The previous top-down tilt-shift wide shot
# (WIDE_SHOT_TILT_DEG=75, per-scene aperture solve) is gone.
PRESENTATION_CAMERA_PATH = "/World/PresentationCamera"
WIDE_SHOT_FOCAL_MM = 24.0
# Framing guarantee (T15d): the robot proxy at its start, every /World/Path point and
# the clearance ring must project inside the frame with >= this margin on every side;
# if not, the eye backs off along its own view axis (look-at, pitch, roll unchanged)
# in CAMERA_BACKOFF_STEP_M steps until they do - deterministic, and logged to
# spec.result.json as `camera_backoff_m`.
FRAMING_MARGIN_FRAC = 0.05
CAMERA_BACKOFF_STEP_M = 0.05
CAMERA_BACKOFF_MAX_M = 40.0
FRAMING_NEAR_M = 0.05
CLEARANCE_RING_CHECK_SEGMENTS = 24

# T15d: DomeLight intensity x3 relative to the 1200 usd_export.py/this script used
# before (the dark-recolored dome contributes ~5% of its nominal intensity, see
# BACKDROP_COLOR below - 3600 brings the ambient fill back up without touching the
# recolor rule). Only used for the fallback dome when the stage has none; a stage
# that already carries its own DomeLight/DistantLight (usd_export.py's, or the
# presentation export's) is never double-lit - see run_isaac_worker.
DOME_LIGHT_FALLBACK_INTENSITY = 3600.0
KEY_LIGHT_FALLBACK_INTENSITY = 3000.0


# --- framing check (pure stdlib math - shared by Phase 2 and the local still) --------


def _v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_scale(a, s: float):
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_norm(a):
    n = math.sqrt(_v_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def clearance_ring_points(center, radius_m: float, *, ground_axes=(0, 1), n: int = CLEARANCE_RING_CHECK_SEGMENTS) -> list:
    """`n` points on the clearance ring of `radius_m` around `center` (3-tuple), in the
    ground plane spanned by `ground_axes` - (0, 1) = XY for Isaac's Z-up frame (this
    script's own frame), (0, 2) = XZ for the Y-up frame the MSA modules/local still
    use. The third coordinate is `center`'s own."""
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        p = list(center)
        p[ground_axes[0]] = center[ground_axes[0]] + radius_m * math.cos(theta)
        p[ground_axes[1]] = center[ground_axes[1]] + radius_m * math.sin(theta)
        pts.append(tuple(p))
    return pts


def framing_check_points(
    start_xyz, path_points_xyz: list, ring_radius_m: float, robot_height_m: float, *, ground_axes=(0, 1), up_axis: int = 2
) -> list:
    """Everything that must be in frame (T15d): the robot proxy at its start (its
    footprint = the ring's own circle, plus the same circle lifted by the robot's
    height so a tall platform's top isn't cropped), every planned path point, and the
    clearance ring itself."""
    pts = list(path_points_xyz)
    pts.append(tuple(start_xyz))
    ring = clearance_ring_points(start_xyz, ring_radius_m, ground_axes=ground_axes)
    pts += ring
    lift = [0.0, 0.0, 0.0]
    lift[up_axis] = robot_height_m
    pts += [_v_add(p, lift) for p in ring]
    return pts


def points_in_frame(
    points, eye, forward, right, up, fov_v_deg: float, aspect: float, *, margin_frac: float = FRAMING_MARGIN_FRAC, near_m: float = FRAMING_NEAR_M
) -> bool:
    """True iff every point projects inside a pinhole frame (vertical FOV `fov_v_deg`,
    width/height `aspect`) with at least `margin_frac` of the frame's width/height
    left on each side, and sits in front of the near plane. `forward`/`right`/`up`
    must be an orthonormal camera basis (same convention as
    scripts.msa.render_perspective: +right = image right, +up = image up).
    T15i: the one implementation lives in scripts/msa/visibility.py."""
    return visibility.points_in_frame(points, eye, forward, right, up, fov_v_deg, aspect, margin_frac=margin_frac, near_m=near_m)


def camera_backoff_for_framing(
    eye, forward, right, up, fov_v_deg: float, aspect: float, points, *,
    margin_frac: float = FRAMING_MARGIN_FRAC, step_m: float = CAMERA_BACKOFF_STEP_M, max_backoff_m: float = CAMERA_BACKOFF_MAX_M,
) -> tuple[float, tuple]:
    """Smallest back-off (a multiple of `step_m`, scanned from 0 - deterministic) along
    the camera's own view axis (-forward) that puts every point of `points` in frame
    per `points_in_frame`. Returns `(backoff_m, new_eye)`; if even `max_backoff_m`
    can't do it (a point behind the camera, or a hopeless geometry), returns
    `(max_backoff_m, eye_at_max)` - the caller logs it, the shot is still rendered."""
    forward = _v_norm(forward)
    n_steps = int(math.ceil(max_backoff_m / step_m))
    for i in range(n_steps + 1):
        backoff = i * step_m
        candidate = _v_sub(eye, _v_scale(forward, backoff))
        if points_in_frame(points, candidate, forward, right, up, fov_v_deg, aspect, margin_frac=margin_frac):
            return backoff, candidate
    return max_backoff_m, _v_sub(eye, _v_scale(forward, max_backoff_m))

DEFAULT_TARGET_NAME = "table"
# app.services.pathfinding.plan_to_object needs a nominal speed for its own
# duration_sec estimate (informational only - actual drive speed in Phase 2 is
# derived from the route length and the requested pass duration, see run_isaac_worker).
PATH_PLANNING_SPEED_MPS = 0.5

# --- platform registry fallback -----------------------------------------------------
# Mirrors app/robots.py's ROBOTS list (id, display_name, radius_m, height_m, length_m,
# width_m) as of the feat/footprint-planner pass (2026-09-05). Only used when
# app.robots can't be imported (Phase 2, running under Isaac Sim's own interpreter -
# see module docstring). Keep in sync with app/robots.py if either changes; app.robots
# is always the source of truth when it's importable (Phase 1).
_FALLBACK_HEIGHT_M = 0.2  # used only for a platform whose dimensions_m/height_m is unmeasured (None)
_FALLBACK_REGISTRY: dict[str, tuple[str, float, float, float, float]] = {
    # id: (display_name, radius_m, height_m, length_m, width_m)
    "burger": ("TurtleBot3 Burger", 0.10, 0.192, 0.138, 0.178),
    "limo": ("AgileX LIMO", 0.1940, 0.2514, 0.3215, 0.2173),
    "waffle_pi": ("TurtleBot3 Waffle Pi", 0.15, 0.1410, 0.2739, 0.3062),
    "turtlebot4": ("TurtleBot 4", 0.175, 0.3467, 0.3415, 0.3382),
    "jackal": ("Clearpath Jackal", 0.334, 0.249, 0.511, 0.430),
    "go2": ("Unitree Go2", 0.2496, 0.1847, 0.4600, 0.1940),
    "husky": ("Husky A200", 0.5528, 0.3963, 0.985, 0.6693),
    "rosbot_xl_arm": ("Husarion ROSbot XL + Arm", 0.218561, 0.131989, 0.332054, 0.284280),
}

# The two platforms this demo always compares, in pass order - not CLI-configurable
# (the whole point is this specific pair: a small differential-drive robot vs. a much
# wider skid-steer one, run over the identical route).
DEMO_PLATFORM_IDS = ("burger", "husky")


@dataclass(frozen=True)
class PlatformSpec:
    id: str
    display_name: str
    radius_m: float
    height_m: float
    height_source: str  # "measured" | "fallback_default" | "fallback_registry_snapshot"
    # Real L x W (RobotDimensions.length_m/width_m) used for the rectangle RigidBody
    # proxy _make_robot_proxy builds (feat/footprint-planner) - falls back to a
    # radius-derived square (2*radius_m each side), same circle-derived-square
    # convention app.robots.resolve_footprint_m uses, when a platform's own
    # dimensions_m isn't available (fallback-registry snapshot path, or a future
    # placeholder platform with dimensions_m=None).
    length_m: float = 0.0
    width_m: float = 0.0

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.width_m <= 0:
            object.__setattr__(self, "length_m", 2 * self.radius_m)
            object.__setattr__(self, "width_m", 2 * self.radius_m)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "target"


def resolve_platform(platform_id: str) -> PlatformSpec:
    """app.robots is the real source of truth and is tried first; the embedded
    snapshot above is only a fallback for when app.* isn't importable (Phase 2)."""
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from app.robots import get_robot, resolve_footprint_m, resolve_radius_m  # type: ignore
    except ImportError:
        entry = _FALLBACK_REGISTRY.get(platform_id)
        if entry is None:
            raise SystemExit(
                f"unknown platform id {platform_id!r}, and app.robots isn't importable here "
                f"(no backend venv) to consult the full registry - known fallback ids: "
                f"{sorted(_FALLBACK_REGISTRY)}"
            )
        display_name, radius_m, height_m, length_m, width_m = entry
        return PlatformSpec(
            platform_id, display_name, radius_m, height_m, "fallback_registry_snapshot", length_m, width_m
        )

    platform = get_robot(platform_id)
    if platform is None:
        raise SystemExit(f"unknown platform id {platform_id!r} - see app/robots.py's ROBOTS")
    radius_m = resolve_radius_m(platform_id, None)
    length_m, width_m = resolve_footprint_m(platform_id, None, None)
    if platform.dimensions_m is not None:
        height_m, height_source = platform.dimensions_m.height_m, "measured"
    else:
        height_m, height_source = _FALLBACK_HEIGHT_M, "fallback_default (platform.dimensions_m is None)"
    return PlatformSpec(platform.id, platform.display_name, radius_m, height_m, height_source, length_m, width_m)


def _longest_free_run_start(grid_cells, meta, robot_radius_m: float) -> dict | None:
    """Find the longest contiguous run of FREE occupancy-grid cells (pathfinding.FREE -
    not UNKNOWN; a run must be actually mapped-and-clear, not merely unexplored) along
    the grid's iz axis (this project's Y-up world Z), scanning every ix column. Ties
    broken toward the column nearest the grid's horizontal center (a better-framed
    shot than an edge column).

    IMPORTANT: `grid_cells` must already be inflated by the robot's radius (see
    `_build_demo_export_async`, which calls `pathfinding.inflate` before this). A raw,
    uninflated grid only guarantees the run's own centerline column is clear - first
    found the hard way against a real scene (feat/isaac-demo-look, 02_modular_home): a
    run picked from the raw grid put a 0.1m-radius robot's swept path directly through
    furniture one column over from dead center, and it stalled a few tenths of a meter
    into what was reported as a 3.4m-long "clear" run.

    Returns world (x, z) in this project's Y-up frame (Scene.robot_start_x/z's own
    convention) - just a good, data-driven place to START the planned route from, not
    a drive direction (the route itself, from `plan_to_object`, determines that) - or
    None if the grid has no FREE run at all.
    """
    from app.services.pathfinding import FREE, OBSTACLE

    width, height = grid_cells.shape
    best: tuple[int, int, int] | None = None  # (run_len, ix, iz_end_exclusive)
    center_ix = (width - 1) / 2.0

    for ix in range(width):
        iz = 0
        while iz < height:
            if grid_cells[ix, iz] != FREE:
                iz += 1
                continue
            iz0 = iz
            while iz < height and grid_cells[ix, iz] == FREE:
                iz += 1
            run_len = iz - iz0
            if best is None or run_len > best[0] or (
                run_len == best[0] and abs(ix - center_ix) < abs(best[1] - center_ix)
            ):
                best = (run_len, ix, iz)

    if best is None:
        return None

    run_len, ix, iz_end = best
    x = meta.origin_x + (ix + 0.5) * meta.resolution
    wall_z = meta.origin_z + iz_end * meta.resolution  # boundary at the run's high-iz end
    inset = min(robot_radius_m + 0.1, run_len * meta.resolution * 0.5)
    z = wall_z - inset
    wall_confirmed = iz_end < height and grid_cells[ix, iz_end] == OBSTACLE
    return {
        "x": x, "z": z,
        "run_len_m": run_len * meta.resolution,
        "wall_confirmed": bool(wall_confirmed),
    }


def _rasterize_object_obstacles(cells, meta, objects) -> None:
    """Mark each non-fragment object's XZ bbox footprint OBSTACLE, in place.

    occupancy.npy (gpu/stage_occupancy.py) is a top-down point-DENSITY grid over a
    fixed height band (occupancy_band_min_m/max_m) of the aligned point cloud - not a
    per-object rasterization. A scene object's exported collider, though, is the convex
    hull of that object's OWN full point cloud (usd_export.py's build_convex_hull),
    unfiltered by that band. A chair whose seat sits above the band (only its thin legs
    fall inside it) shows up in the occupancy grid as a few near-empty cells, while its
    actual collider is seat-width at the robot's collision height - a run the grid
    calls clear can still be blocked by real Object geometry the grid never saw. Only
    non-fragment objects: usd_export's Isaac path always exports with
    include_fragments=False, so fragments have no collider in the actual scene."""
    from app.services.pathfinding import OBSTACLE

    width, height = cells.shape
    for obj in objects:
        if obj.is_fragment:
            continue
        ix0 = int((obj.bbox_min_x - meta.origin_x) / meta.resolution)
        ix1 = int(math.ceil((obj.bbox_max_x - meta.origin_x) / meta.resolution))
        iz0 = int((obj.bbox_min_z - meta.origin_z) / meta.resolution)
        iz1 = int(math.ceil((obj.bbox_max_z - meta.origin_z) / meta.resolution))
        ix0, ix1 = max(0, ix0), min(width, ix1 + 1)
        iz0, iz1 = max(0, iz0), min(height, iz1 + 1)
        if ix0 < ix1 and iz0 < iz1:
            cells[ix0:ix1, iz0:iz1] = OBSTACLE


async def _build_demo_export_async(scene_id: str, burger_radius_m: float, target_name: str) -> dict:
    """Builds a demo-specific USD export with /World/Path (a real planned route to
    `target_name`, from app.services.pathfinding.plan_to_object) and /World/Target
    baked in - NOT the generic cached export the /usd endpoint serves (that has no
    concept of a goal, so it can't have a path). Always regenerates, no caching - this
    is a "reproducible recording script" per the module docstring, a stale route would
    defeat that (and export is fast - seconds, not the minutes a full GPU pipeline run
    costs).

    Raises SystemExit on anything that would leave the demo with nothing sensible to
    record (scene not found/not ingested, target object not found, no path exists at
    all) - unlike the old grid-only robot-start lookup this replaces, there's no
    reasonable silent fallback for a two-robot path-following demo without an actual
    route."""
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.database import async_session_maker  # type: ignore
    from app.services import command_service, scene_ingest, scene_service, usd_export  # type: ignore
    from app.services.command_service import ObjectNotFoundError  # type: ignore
    from app.services.pathfinding import (  # type: ignore
        NoPathError, ObjectFootprint, inflate, load_grid, plan_to_object,
    )

    scene_uuid = uuid.UUID(scene_id)
    async with async_session_maker() as session:
        scene = await scene_service.get_scene(session, scene_uuid)
        if scene is None:
            raise SystemExit(f"scene {scene_id} not found")
        if not scene.occupancy_path:
            raise SystemExit(f"scene {scene_id} has no occupancy grid yet")
        if scene.ceiling_y is None:
            raise SystemExit(f"scene {scene_id} has no ceiling_y recorded")
        objects = await scene_service.list_scene_objects(session, scene_uuid)
        occupancy_path = scene.occupancy_path
        pointcloud_path = scene.pointcloud_path
        ceiling_y = scene.ceiling_y
        is_reflected = scene.is_reflected
        db_robot_start = (
            (scene.robot_start_x, scene.robot_start_z)
            if scene.robot_start_x is not None and scene.robot_start_z is not None
            else None
        )

    grid_meta = scene_ingest.read_grid_metadata(scene_service.scene_dir(scene_uuid))
    grid = load_grid(scene_uuid, occupancy_path, grid_meta)

    try:
        target_obj = command_service.resolve_object(objects, target_name)
    except ObjectNotFoundError as exc:
        raise SystemExit(
            f"--target {target_name!r} not found in scene {scene_id} - known object "
            f"names: {sorted({o.name for o in objects})}"
        ) from exc
    target_index = next(i for i, o in enumerate(objects) if o.id == target_obj.id)

    obstacle_cells = grid.cells.copy()
    _rasterize_object_obstacles(obstacle_cells, grid_meta, objects)
    inflated_cells = inflate(obstacle_cells, grid_meta, burger_radius_m)
    start_found = _longest_free_run_start(inflated_cells, grid_meta, burger_radius_m)
    if start_found is not None:
        start_world = (start_found["x"], start_found["z"])
        start_source = "grid_longest_free_run"
    elif db_robot_start is not None:
        start_world = db_robot_start
        start_source = "scene.robot_start_x/z (grid search found no free run)"
    else:
        raise SystemExit(
            f"scene {scene_id}: no usable robot start point (grid search found no "
            f"free run, and scene.robot_start_x/z isn't set)"
        )

    footprint = ObjectFootprint(
        x=target_obj.pos_x, z=target_obj.pos_z,
        bbox_min_x=target_obj.bbox_min_x, bbox_min_z=target_obj.bbox_min_z,
        bbox_max_x=target_obj.bbox_max_x, bbox_max_z=target_obj.bbox_max_z,
    )
    try:
        path_result = plan_to_object(
            grid, start_world, footprint,
            robot_radius_m=burger_radius_m, speed_mps=PATH_PLANNING_SPEED_MPS,
        )
    except NoPathError as exc:
        raise SystemExit(
            f"no path from {start_world} to {target_name!r} ({target_obj.name}) in "
            f"scene {scene_id}: {exc}"
        ) from exc

    export_objects = [
        usd_export.UsdExportObject(
            name=o.name,
            bbox_min=(o.bbox_min_x, o.bbox_min_y, o.bbox_min_z),
            bbox_max=(o.bbox_max_x, o.bbox_max_y, o.bbox_max_z),
            num_views=o.num_views,
            num_points=o.num_points,
            is_fragment=o.is_fragment,
            mesh_path=o.mesh_path,
        )
        for o in objects
    ]
    export_input = usd_export.UsdExportInput(
        grid_cells=grid.cells,
        grid_meta=grid_meta,
        ceiling_y=ceiling_y,
        robot_start=start_world,
        is_reflected=is_reflected,
        robot_radius_m=burger_radius_m,
        objects=export_objects,
        pointcloud_path=pointcloud_path,
        include_fragments=False,
        path_points=path_result.points,
        target_point=(target_obj.pos_x, target_obj.pos_z),
        target_object_index=target_index,
    )
    out_path = (
        scene_service.scene_dir(scene_uuid) / "usd"
        / f"demo_v{usd_export.EXPORT_SCHEMA_VERSION}_{_slugify(target_obj.name)}.usd"
    )
    usd_export.export_scene_usd(export_input, out_path)

    # One color per class, straight from usd_export's own function - guarantees the
    # mp4's burned-in legend (see _build_legend_filters) always matches what's
    # actually in frame, rather than a hand-maintained palette drifting out of sync.
    class_colors = {
        o.name: [round(c * 255) for c in usd_export._object_class_color(o.name)]
        for o in objects if not o.is_fragment
    }

    return {
        "usd_path": str(out_path),
        "start_world": list(start_world),
        "start_source": start_source,
        "target_name": target_obj.name,
        "path_points_world": [list(p) for p in path_result.points],
        "path_length_m": path_result.length_m,
        "class_colors": class_colors,
    }


def build_demo_export(scene_id: str, burger_radius_m: float, target_name: str) -> dict:
    return asyncio.run(_build_demo_export_async(scene_id, burger_radius_m, target_name))


def _isaac_to_yup(p) -> tuple[float, float, float]:
    """Inverse of app.services.usd_export.to_isaac ((x, y, z) -> (x, -z, y))."""
    return (float(p[0]), float(p[2]), -float(p[1]))


def _yup_to_isaac(p) -> tuple[float, float, float]:
    return (float(p[0]), -float(p[2]), float(p[1]))


PRESENTATION_CAMERA_HUSKY_PATH = "/World/PresentationCameraHusky"  # export_usd.PRESENTATION_CAMERA_HUSKY_PATH (two-pose mode)


def resolve_wide_camera(usd_path: Path, *, width: int, height: int, camera_path: str = PRESENTATION_CAMERA_PATH,
                        required: bool = True) -> dict | None:
    """Phase 1 only (needs pxr; may import scripts.msa.render_perspective). The wide
    shot's camera in ISAAC's Z-up frame, from one of two sources (`camera_path` selects
    the prim - `PRESENTATION_CAMERA_HUSKY_PATH` for the two-pose Husky-pass camera;
    with `required=False` a missing prim returns None instead of the computed fallback):

    1. `PRESENTATION_CAMERA_PATH` already in the stage (scripts/msa/export_presentation.py
       bakes the web 3/4 rule's camera there, framing back-off included) - read as-is,
       so the recorded clip and the local still share one camera by construction.
    2. Otherwise computed here by importing `scripts.msa.render_perspective.compute_camera`
       (the rule itself - never re-implemented) from the stage's own room outline
       (`/World/Plan/floor` if the export has it, else `/World/Floor`'s bbox corners)
       and bed prims (`/World/Objects/bed*`, footprint centroid = mean of their bbox
       midpoints).

    Returns a JSON-able dict (`eye`/`forward`/`right`/`up` Isaac-frame, `fov_v_deg`,
    `source`, plus the rule's own numbers for the report) - Phase 2 only reads it."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise SystemExit(f"could not open {usd_path} to resolve the wide camera")
    bbcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    floor_prim = stage.GetPrimAtPath("/World/Floor")
    if not floor_prim.IsValid():
        raise SystemExit(f"{usd_path}: /World/Floor missing - can't derive the room outline for the wide shot")
    floor_rng = bbcache.ComputeWorldBound(floor_prim).ComputeAlignedRange()
    fmin, fmax = floor_rng.GetMin(), floor_rng.GetMax()
    floor_y = float(fmax[2])  # Isaac z-up top of the floor == Y-up floor_y

    # Room outline (both branches need it - T15i: the worker frames the whole dollhouse).
    plan_floor = stage.GetPrimAtPath("/World/Plan/floor")
    room_source = "/World/Plan/floor"
    room_xz = []
    if plan_floor.IsValid():
        pts = UsdGeom.BasisCurves(plan_floor).GetPointsAttr().Get() or []
        room_xz = [(_isaac_to_yup(p)[0], _isaac_to_yup(p)[2]) for p in pts]
    if len(room_xz) < 3:
        room_source = "/World/Floor bbox corners"
        corners = [(fmin[0], fmin[1]), (fmax[0], fmin[1]), (fmax[0], fmax[1]), (fmin[0], fmax[1])]
        room_xz = [(float(x), -float(y)) for x, y in corners]
    if room_xz and room_xz[0] == room_xz[-1]:
        room_xz = room_xz[:-1]
    room_outline_isaac = [[float(x), float(-z)] for x, z in room_xz]

    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid() and not required:
        return None
    if cam_prim.IsValid():
        m = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = tuple(m.Transform(Gf.Vec3d(0, 0, 0)))
        forward = _v_norm(tuple(m.TransformDir(Gf.Vec3d(0, 0, -1))))
        up = _v_norm(tuple(m.TransformDir(Gf.Vec3d(0, 1, 0))))
        right = _v_norm(tuple(m.TransformDir(Gf.Vec3d(1, 0, 0))))
        cam = UsdGeom.Camera(cam_prim)
        fov_attr = cam_prim.GetAttribute("cloudeye:cameraFovVDeg")
        if fov_attr and fov_attr.HasValue():
            fov_v_deg = float(fov_attr.Get())
        else:
            # Isaac re-derives verticalAperture = horizontalAperture * h / w (see the
            # 2026-09-05 v3 DECISIONS entry) - so that's the vertical FOV it will render.
            h_ap = float(cam.GetHorizontalApertureAttr().Get())
            focal = float(cam.GetFocalLengthAttr().Get())
            fov_v_deg = math.degrees(2.0 * math.atan((h_ap * height / width) / (2.0 * focal)))
        extras = {}
        for attr_name, key in (
            ("cloudeye:cameraPitchDeg", "pitch_deg"), ("cloudeye:cameraDistanceM", "distance_m"),
            ("cloudeye:cameraBackoffM", "baked_backoff_m"), ("cloudeye:cameraRule", "rule"),
            ("cloudeye:cameraHeightRaiseM", "baked_height_raise_m"), ("cloudeye:roomFrameCoverage", "baked_room_frame_coverage"),
            ("cloudeye:cameraVisibilityOk", "baked_visibility_ok"),
            ("cloudeye:pathVisibilityWorstCase", "baked_path_visibility_worst_case"),
            ("cloudeye:pathVisibilityMinFraction", "path_visibility_min_fraction"),
            ("cloudeye:cameraMinClearanceM", "baked_min_eye_clearance_m"),
        ):
            attr = cam_prim.GetAttribute(attr_name)
            if attr and attr.HasValue():
                extras[key] = attr.Get()
        # T15i: the visibility rule's inputs baked by export_presentation (Isaac frame).
        aim_attr = cam_prim.GetAttribute("cloudeye:cameraAimPoint")
        if aim_attr and aim_attr.HasValue():
            aim = [float(v) for v in aim_attr.Get()]
        else:
            aim = list(visibility.aim_point_on_floor(eye, forward, floor_y, up_axis=2))
        boxes_attr = cam_prim.GetAttribute("cloudeye:occluderBoxes")
        ids_attr = cam_prim.GetAttribute("cloudeye:occluderIds")
        if boxes_attr and boxes_attr.HasValue():
            ids = list(ids_attr.Get()) if ids_attr and ids_attr.HasValue() else None
            occluders = visibility.boxes_from_flat(list(boxes_attr.Get()), ids)
            occluder_source = "usd:cloudeye:occluderBoxes"
        else:
            occluders = _collect_occluder_boxes_from_stage(stage, bbcache)
            occluder_source = "computed:visible Structure/Objects visual AABBs"
        culled_attr = cam_prim.GetAttribute("cloudeye:culledStubEdges")
        culled = [int(i) for i in culled_attr.Get()] if culled_attr and culled_attr.HasValue() else []
        return {
            "source": f"usd:{camera_path}", "eye": list(eye), "forward": list(forward),
            "right": list(right), "up": list(up), "fov_v_deg": fov_v_deg, "aim": aim, "floor_z": floor_y,
            "room_outline_isaac": room_outline_isaac, "room_outline_source": room_source,
            "occluders": occluders, "occluder_source": occluder_source, "n_occluders": len(occluders),
            "culled_stub_edges": culled, **extras,
        }

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import numpy as np  # type: ignore
    from scripts.msa.render_perspective import FOV_DEG, compute_camera  # type: ignore

    bed_mids = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Objects/") or path.count("/") != 3:
            continue
        if not path.rsplit("/", 1)[-1].lower().startswith("bed"):
            continue
        rng = bbcache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mid = rng.GetMidpoint()
        bed_mids.append((float(mid[0]), -float(mid[1])))
    bed_centroid = (
        (sum(p[0] for p in bed_mids) / len(bed_mids), sum(p[1] for p in bed_mids) / len(bed_mids)) if bed_mids else None
    )

    pos, fwd, right, up = compute_camera(np.asarray(room_xz, dtype=float), floor_y, bed_centroid_xz=bed_centroid, image_height_px=height)
    cx = float(np.mean([p[0] for p in room_xz])); cz = float(np.mean([p[1] for p in room_xz]))
    eye_isaac = _yup_to_isaac(pos)
    fwd_isaac = _yup_to_isaac(fwd)
    occluders = _collect_occluder_boxes_from_stage(stage, bbcache)
    return {
        "source": "computed:scripts.msa.render_perspective.compute_camera",
        "room_outline_source": room_source,
        "bed_centroid_xz": list(bed_centroid) if bed_centroid else None,
        "eye": list(eye_isaac), "forward": list(fwd_isaac),
        "right": list(_yup_to_isaac(right)), "up": list(_yup_to_isaac(up)),
        "fov_v_deg": float(FOV_DEG),
        "position_yup": [float(v) for v in pos],
        "pitch_deg": math.degrees(math.asin(float(fwd[1]))),
        "distance_m": math.hypot(float(pos[0]) - cx, float(pos[2]) - cz),
        "height_above_floor_m": float(pos[1]) - floor_y,
        # T15i: the worker's visibility re-solve needs these too.
        "aim": list(visibility.aim_point_on_floor(eye_isaac, fwd_isaac, floor_y, up_axis=2)),
        "floor_z": floor_y,
        "room_outline_isaac": room_outline_isaac,
        "occluders": occluders, "occluder_source": "computed:visible Structure/Objects visual AABBs", "n_occluders": len(occluders),
        "culled_stub_edges": [],
    }


def _collect_occluder_boxes_from_stage(stage, bbcache) -> list:
    """Phase 1 fallback (a USD without `cloudeye:occluderBoxes`): the world AABB of
    every VISIBLE Mesh/Cube under /World/Structure and /World/Objects that is not a
    `/collision/` prim, whose object class is not translucent and whose top is more
    than `visibility.LOW_OCCLUDER_MAX_TOP_M` above the floor (rugs are not occluders) -
    in Isaac's frame, as `scripts.msa.visibility` boxes. Conservative (an AABB
    contains the mesh)."""
    from pxr import UsdGeom

    floor_prim = stage.GetPrimAtPath("/World/Floor")
    floor_top = float(bbcache.ComputeWorldBound(floor_prim).ComputeAlignedRange().GetMax()[2]) if floor_prim.IsValid() else None
    boxes = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not (path.startswith("/World/Structure/") or path.startswith("/World/Objects/")):
            continue
        if "/collision/" in path or prim.GetTypeName() not in ("Mesh", "Cube"):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if path.startswith("/World/Objects/"):
            obj_id = path.split("/")[3]
            if visibility.is_translucent_class(obj_id):
                continue
        rng = bbcache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        if floor_top is not None and float(hi[2]) - floor_top <= visibility.LOW_OCCLUDER_MAX_TOP_M:
            continue
        boxes.append(visibility.aabb_box(path, (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])))
    return boxes


# --- CLI -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene_id", nargs="?", help="scene UUID (required unless --_isaac-worker-spec is given)")
    p.add_argument("output_mp4", nargs="?", type=Path, help="output .mp4 path (Phase 1 host)")
    p.add_argument("--target", default=DEFAULT_TARGET_NAME, help="object name to plan a route to (default: table)")
    p.add_argument("--usd-path-override", type=Path, default=None,
                   help="skip DB/pathfinding/export entirely, use this USD file directly - "
                        "it must already have /World/Path and /World/Target (e.g. copied "
                        "from a prior normal Phase 1 run)")
    p.add_argument("--isaac-python-exe", type=Path, default=None, help="if given, Phase 1 auto-runs Phase 2 locally via this interpreter")
    p.add_argument("--spec-out", type=Path, default=None, help="where to write the Phase 2 job spec JSON (default: alongside output_mp4)")
    p.add_argument("--frames-dir", type=Path, default=None, help="scratch dir for offscreen PNG frames (default: alongside output_mp4)")

    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--establish-s", type=float, default=3.0, help="wide establishing shot, nothing moving yet")
    p.add_argument("--pass-s", type=float, default=4.0, help="duration of EACH robot's drive (Burger, then Husky)")
    p.add_argument("--drop-s", type=float, default=1.0, help="sphere-drop collision sanity check near the target")
    p.add_argument("--static-s", type=float, default=2.0, help="closing static hold")

    p.add_argument("--_isaac-worker-spec", dest="isaac_worker_spec", type=Path, default=None, help=argparse.SUPPRESS)

    args = p.parse_args(argv)
    if args.isaac_worker_spec is None:
        if not args.scene_id or not args.output_mp4:
            p.error("scene_id and output_mp4 are required unless --_isaac-worker-spec is given")
        for name in ("establish_s", "pass_s", "drop_s", "static_s"):
            if getattr(args, name) <= 0:
                p.error(f"--{name.replace('_', '-')} must be positive")
    return args


# --- Phase 1: driver -------------------------------------------------------------------


def build_spec(args: argparse.Namespace) -> dict:
    platforms = [resolve_platform(pid) for pid in DEMO_PLATFORM_IDS]
    burger = platforms[0]

    if args.usd_path_override is not None:
        usd_path = args.usd_path_override
        usd_source = "override"
        demo_export = None
    else:
        demo_export = build_demo_export(args.scene_id, burger.radius_m, args.target)
        usd_path = Path(demo_export["usd_path"])
        usd_source = "planned_export"
        print(
            f"[record_isaac] planned route to {demo_export['target_name']!r}: "
            f"{demo_export['path_length_m']:.2f}m, {len(demo_export['path_points_world'])} waypoints "
            f"(start: {demo_export['start_source']})"
        )
    if not usd_path.is_file():
        raise SystemExit(f"USD not found at {usd_path} (source: {usd_source})")

    output_mp4 = args.output_mp4.resolve()
    frames_dir = (args.frames_dir or output_mp4.parent / f"{output_mp4.stem}_frames").resolve()

    wide_camera = resolve_wide_camera(usd_path, width=args.width, height=args.height)
    print(f"[record_isaac] wide camera: {wide_camera['source']} eye={[round(v, 3) for v in wide_camera['eye']]} "
          f"fov_v={wide_camera['fov_v_deg']:.1f}deg occluders={wide_camera.get('n_occluders')} "
          f"({wide_camera.get('occluder_source')}) culled_stub_edges={wide_camera.get('culled_stub_edges')}")
    # Two-pose mode (orchestrator 2026-09-07): a second baked camera for Husky's pass.
    wide_camera_husky = resolve_wide_camera(usd_path, width=args.width, height=args.height,
                                            camera_path=PRESENTATION_CAMERA_HUSKY_PATH, required=False)
    if wide_camera_husky is not None:
        print(f"[record_isaac] husky-pass camera: {wide_camera_husky['source']} eye={[round(v, 3) for v in wide_camera_husky['eye']]} "
              f"baked path visibility={wide_camera_husky.get('baked_path_visibility_worst_case')}")

    return {
        "schema": SCHEMA,
        "scene_id": args.scene_id,
        "usd_path": str(usd_path),
        "usd_source": usd_source,
        "wide_camera": wide_camera,
        "wide_camera_husky": wide_camera_husky,
        "output_mp4": str(output_mp4),
        "frames_dir": str(frames_dir),
        "platforms": [asdict(p) for p in platforms],
        "target_name": args.target,
        "class_colors": (demo_export or {}).get("class_colors", {}),
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "establish_s": args.establish_s,
        "pass_s": args.pass_s,
        "drop_s": args.drop_s,
        "static_s": args.static_s,
        "caption": "Isaac Sim, headless · kinematic drive",
    }


def run_driver(args: argparse.Namespace) -> int:
    spec = build_spec(args)
    spec_out = args.spec_out or Path(spec["output_mp4"]).with_suffix(".spec.json")
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps(spec, indent=2))
    print(f"[record_isaac] wrote job spec: {spec_out}")
    for platform in spec["platforms"]:
        print(f"[record_isaac] platform: {platform['display_name']} "
              f"(footprint={platform['length_m']}x{platform['width_m']}m, height={platform['height_m']}m, "
              f"source={platform['height_source']})")
    print(f"[record_isaac] input USD: {spec['usd_path']} (source: {spec['usd_source']})")

    if args.isaac_python_exe is None:
        print("[record_isaac] no --isaac-python-exe given - stopping after Phase 1.")
        print(f"[record_isaac] run Phase 2 with: OMNI_KIT_ALLOW_ROOT=1 <isaac-python> "
              f"{Path(__file__).name} --_isaac-worker-spec {spec_out}")
        return 0

    cmd = [str(args.isaac_python_exe), str(Path(__file__).resolve()), "--_isaac-worker-spec", str(spec_out)]
    print(f"[record_isaac] running Phase 2: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# --- Phase 2: Isaac Sim worker -------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smootherstep(t: float) -> float:
    """Ken Perlin's 'smootherstep': like the classic cubic smoothstep but zero first
    AND second derivative at both endpoints, so a transition has no velocity or
    acceleration discontinuity where it starts/ends - used for the one camera move
    this clip has (wide establishing/drive shot -> sphere-drop close shot)."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _find_font() -> str | None:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


# Isaac demo-look: dark background instead of the DomeLight's default white/gray.
# Matches the frontend's own dark canvas (tokens.ts's CANVAS token is close; #0b0d12
# is the literal spec value). Two things this ISN'T, both eliminated on a real GPU
# run rather than assumed: (1) a large enclosing sphere with an emissive dark
# material as a separate manual backdrop - fully OPAQUE, it blocked the DomeLight's
# own illumination (and the DistantLight key light, entering from "outside" the same
# shell) from ever reaching the room, rendering every frame solid black; (2) boosting
# the DomeLight's intensity to compensate for its color now being dark (so total
# light contribution matched the un-recolored baseline) - washed the whole scene out
# overexposed instead, and the background itself STAYED light gray regardless, so
# background brightness apparently isn't driven by intensity the way exposure math
# would suggest. Just recoloring the DomeLight (intensity untouched) is what's
# actually being tested now - see docs/DECISIONS.md for whichever of these ends up
# being the one that worked.
BACKDROP_COLOR = (0x0B / 255, 0x0D / 255, 0x12 / 255)

# Robot proxy look: bright, contrasting orange (spec: "контрастный оранжевый") - used
# for both platforms' cylinders, their clearance rings, and their trails.
ROBOT_COLOR = (1.0, 0.45, 0.05)
RING_THICKNESS_M = 0.02
RING_SEGMENTS = 48
RING_HEIGHT_ABOVE_FLOOR_M = 0.01
TRAIL_WIDTH_M = 0.03
TRAIL_HEIGHT_ABOVE_FLOOR_M = 0.03
TRAIL_SAMPLE_EVERY_N_FRAMES = 3

WAYPOINT_ARRIVE_EPS_M = 0.15
STALL_EPS_M = 0.002  # per-frame displacement below this counts as "not moving"
ROBOT_SETTLE_S = 0.3  # brief pause after (re)spawning before driving commands begin

# Same numbers tools/isaac_validate.py's own sphere_drop check uses by default -
# directly comparable, and this demo's drop is meant to read as "the same collision
# safety check the validator runs," not a new one.
SPHERE_DROP_RADIUS_M = 0.1
SPHERE_DROP_MASS_KG = 1.0
SPHERE_DROP_HEIGHT_M = 2.0
# T15h: the sphere is dropped over the route START, never over the goal/target object -
# on the hero (T15e GPU run) the target sat inside the desk's and a lamp's hulls, so the
# sphere spawned inside a collider and tunneled. The drop column (sphere radius around
# the point, floor to spawn height) must be free of every object hull's world extent;
# if the start is not, the point walks along the path in SPHERE_DROP_SHIFT_STEP_M steps
# until it is (choose_sphere_drop_point) - logged as result["sphere_drop_point"].
SPHERE_DROP_SHIFT_STEP_M = 0.05
# T15i: the drop point is chosen AT DROP TIME and must clear both robots' current
# footprints (half-diagonal + this margin) - see choose_sphere_drop_point.
SPHERE_DROP_ROBOT_MARGIN_M = 0.15
# T15h acceptance: "the Burger finished within this distance of /World/Target" - the
# target is the route GOAL since scripts/msa/export_presentation.py PATH_SCHEMA /2 (the
# desk centroid is /World/TargetObject), so the number is measurable.
TARGET_REACH_TOLERANCE_M = 0.3

CAMERA_TRANSITION_S = 0.75  # wide shot -> sphere-drop close shot, at the drop phase's start


def _polyline_samples(path_xy: list, step_m: float) -> list:
    """Points along a polyline every `step_m` (start included, every vertex included,
    end included) - pure stdlib, shared by Phase 2 (Isaac's interpreter)."""
    if not path_xy:
        return []
    out = [(float(path_xy[0][0]), float(path_xy[0][1]), 0.0)]
    travelled = 0.0
    for (ax, ay), (bx, by) in zip(path_xy, path_xy[1:]):
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        n = max(1, int(math.ceil(seg / step_m)))
        for k in range(1, n + 1):
            t = k / n
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t, travelled + seg * t))
        travelled += seg
    return out


def hulls_blocking_column(xy, hull_ranges: list, radius_m: float) -> list:
    """Prim paths of every hull whose world XY extent, grown by `radius_m`, contains
    `xy` - i.e. whose collider could sit in a sphere's drop column there. Extents are
    axis-aligned world bounds (`{"prim", "min": (x, y, z), "max": (x, y, z)}`), so this
    is conservative for a rotated hull."""
    x, y = float(xy[0]), float(xy[1])
    return [
        h["prim"] for h in hull_ranges
        if h["min"][0] - radius_m <= x <= h["max"][0] + radius_m and h["min"][1] - radius_m <= y <= h["max"][1] + radius_m
    ]


def robots_blocking_point(xy, robot_footprints: list, margin_m: float = SPHERE_DROP_ROBOT_MARGIN_M) -> list:
    """T15i: names of the robots whose CURRENT footprint (a circle of half-diagonal +
    `margin_m` around the body centre) contains `xy`. `robot_footprints` =
    `[{"name", "xy", "half_diagonal_m"}, ...]`."""
    x, y = float(xy[0]), float(xy[1])
    return [
        r["name"] for r in robot_footprints
        if math.hypot(x - float(r["xy"][0]), y - float(r["xy"][1])) < float(r["half_diagonal_m"]) + margin_m
    ]


def point_in_polygon_xy(xy, polygon_xy: list) -> bool:
    """Even-odd ray-casting test (stdlib) - the room outline for the worker."""
    x, y = float(xy[0]), float(xy[1])
    inside = False
    n = len(polygon_xy)
    for i in range(n):
        x0, y0 = float(polygon_xy[i][0]), float(polygon_xy[i][1])
        x1, y1 = float(polygon_xy[(i + 1) % n][0]), float(polygon_xy[(i + 1) % n][1])
        if (y0 > y) != (y1 > y):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < x_cross:
                inside = not inside
    return inside


def distance_to_polygon_boundary_xy(xy, polygon_xy: list) -> float:
    x, y = float(xy[0]), float(xy[1])
    best = float("inf")
    n = len(polygon_xy)
    for i in range(n):
        x0, y0 = float(polygon_xy[i][0]), float(polygon_xy[i][1])
        x1, y1 = float(polygon_xy[(i + 1) % n][0]), float(polygon_xy[(i + 1) % n][1])
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 < 1e-18 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / seg2))
        best = min(best, math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)))
    return best


# Morning-5 off-route drop-point fallback: the nearest FREE floor cell to the goal - the
# room outline eroded by this much, clear of every hull column and both robots' footprints.
SPHERE_DROP_ROOM_EROSION_M = 0.2
SPHERE_DROP_FREE_CELL_M = 0.1


def free_floor_cell_nearest(goal_xy, room_outline_xy: list, hull_ranges: list, radius_m: float, robot_footprints: list, *,
                            erosion_m: float = SPHERE_DROP_ROOM_EROSION_M, cell_m: float = SPHERE_DROP_FREE_CELL_M) -> dict | None:
    """Morning-5: grid the room outline's bbox at `cell_m`, keep the cell centres that
    are inside the outline by >= `erosion_m`, clear of every hull column (grown by
    `radius_m`) and of every robot footprint (half-diagonal + margin), and return the
    one nearest `goal_xy` (`{"xy", "distance_to_goal_m", "n_free_cells"}`) or None."""
    if not room_outline_xy or len(room_outline_xy) < 3:
        return None
    xs = [float(p[0]) for p in room_outline_xy]
    ys = [float(p[1]) for p in room_outline_xy]
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    best = None
    n_free = 0
    nx = int(math.ceil((max(xs) - min(xs)) / cell_m))
    ny = int(math.ceil((max(ys) - min(ys)) / cell_m))
    for i in range(nx):
        x = min(xs) + (i + 0.5) * cell_m
        for j in range(ny):
            y = min(ys) + (j + 0.5) * cell_m
            if not point_in_polygon_xy((x, y), room_outline_xy) or distance_to_polygon_boundary_xy((x, y), room_outline_xy) < erosion_m:
                continue
            if hulls_blocking_column((x, y), hull_ranges, radius_m) or robots_blocking_point((x, y), robot_footprints):
                continue
            n_free += 1
            d = math.hypot(x - gx, y - gy)
            if best is None or d < best[0]:
                best = (d, x, y)
    if best is None:
        return None
    return {"xy": [best[1], best[2]], "distance_to_goal_m": best[0], "n_free_cells": n_free, "erosion_m": erosion_m, "cell_m": cell_m}


def choose_sphere_drop_point(path_xy: list, hull_ranges: list, radius_m: float = SPHERE_DROP_RADIUS_M,
                             step_m: float = SPHERE_DROP_SHIFT_STEP_M, robot_footprints: list | None = None,
                             room_outline_xy: list | None = None) -> dict:
    """T15h: where the sanity sphere is dropped. The route START if its drop column is
    free of every object hull extent (`hulls_blocking_column`), else the first point
    along the path (sampled every `step_m`) that is; if no point on the path is free
    the start is used anyway and `source` says so. Never the goal/target.

    T15i (`robot_footprints` given - the worker calls this AT DROP TIME with both
    robots' current poses): the point must also clear every robot's footprint by
    half-diagonal + SPHERE_DROP_ROBOT_MARGIN_M (`robots_blocking_point`) - T15e run2's
    sphere fell 0.045 m behind blocked Husky's rear face, clipped it and rolled 1.17 m.
    The nearest such route point from the start wins; `source` is then
    `route_point_clear_of_robots` (with `shift_along_path_m`, 0 when the start itself
    is clear) and `blocking_robots_at_start` lists what was in the way.

    Morning-5 (`room_outline_xy` given as well): when NO route point is clear (T15e
    run3 - blocked Husky's 0.745 m exclusion covered the whole 0.89 m route) the drop
    point goes OFF-ROUTE to the nearest free floor cell to the goal
    (`free_floor_cell_nearest`: outline eroded by SPHERE_DROP_ROOM_EROSION_M, clear of
    every hull column and both robots), `source = off_route_free_cell_nearest_goal`.
    The `route_start_unverified...` fallback only remains for a room with no free
    cell at all."""
    robots = list(robot_footprints or [])
    start = (float(path_xy[0][0]), float(path_xy[0][1]))
    blocking_at_start = hulls_blocking_column(start, hull_ranges, radius_m)
    robots_at_start = robots_blocking_point(start, robots)
    if robot_footprints is not None:
        samples = _polyline_samples(path_xy, step_m)
        for i, (x, y, along) in enumerate(samples, start=1):
            if hulls_blocking_column((x, y), hull_ranges, radius_m) or robots_blocking_point((x, y), robots):
                continue
            return {
                "xy": [x, y], "source": "route_point_clear_of_robots", "shift_along_path_m": along,
                "blocking_hulls_at_start": blocking_at_start, "blocking_robots_at_start": robots_at_start,
                "robots_checked": [r["name"] for r in robots], "robot_margin_m": SPHERE_DROP_ROBOT_MARGIN_M, "checked_points": i,
            }
        common = {
            "blocking_hulls_at_start": blocking_at_start, "blocking_robots_at_start": robots_at_start,
            "robots_checked": [r["name"] for r in robots], "robot_margin_m": SPHERE_DROP_ROBOT_MARGIN_M, "checked_points": len(samples),
        }
        goal = (float(path_xy[-1][0]), float(path_xy[-1][1]))
        cell = free_floor_cell_nearest(goal, room_outline_xy or [], hull_ranges, radius_m, robots) if room_outline_xy else None
        if cell is not None:
            return {
                "xy": cell["xy"], "source": "off_route_free_cell_nearest_goal", "shift_along_path_m": None,
                "off_route": {k: v for k, v in cell.items() if k != "xy"}, "distance_to_route_start_m": math.hypot(cell["xy"][0] - start[0], cell["xy"][1] - start[1]),
                **common,
            }
        return {
            "xy": list(start), "source": "route_start_unverified_no_point_clear_of_hulls_and_robots", "shift_along_path_m": 0.0,
            "off_route_searched": bool(room_outline_xy), **common,
        }
    if not blocking_at_start:
        return {"xy": list(start), "source": "route_start", "shift_along_path_m": 0.0, "blocking_hulls_at_start": [], "checked_points": 1}
    samples = _polyline_samples(path_xy, step_m)
    for i, (x, y, along) in enumerate(samples[1:], start=2):
        if not hulls_blocking_column((x, y), hull_ranges, radius_m):
            return {
                "xy": [x, y], "source": "shifted_along_path", "shift_along_path_m": along,
                "blocking_hulls_at_start": blocking_at_start, "checked_points": i,
            }
    return {
        "xy": list(start), "source": "route_start_unverified_no_free_point_on_path", "shift_along_path_m": 0.0,
        "blocking_hulls_at_start": blocking_at_start, "checked_points": len(samples),
    }


# Morning-5 close shot: a ring of candidate eyes around the drop point - radii 1.5..3 m,
# heights 1.2..2.6 m above the floor, every 15 deg - the closest one that is not inside a
# robot footprint (half-diagonal + margin) or a hull column and that SEES the drop point
# and the sphere's expected rest area (drop point +- CLOSE_SHOT_REST_AREA_M).
CLOSE_SHOT_RADII_M = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
CLOSE_SHOT_HEIGHTS_M = (1.2, 1.5, 1.8, 2.1, 2.4, 2.6)
CLOSE_SHOT_AZIMUTH_STEP_DEG = 15.0
CLOSE_SHOT_REST_AREA_M = 0.5
CLOSE_SHOT_HULL_MARGIN_M = 0.15
CLOSE_SHOT_LOOK_AT_HEIGHT_M = 0.3


def robot_footprint_boxes(robot_footprints: list, floor_z: float, *, margin_m: float = SPHERE_DROP_ROBOT_MARGIN_M) -> list:
    """Conservative occluder boxes for the robots at their CURRENT poses (heading unknown
    here): an axis-aligned square of half-side half-diagonal + margin, floor -> body top
    (`height_m`, default 0.5 m)."""
    boxes = []
    for r in robot_footprints:
        h = float(r.get("height_m") or 0.5)
        half = float(r["half_diagonal_m"]) + margin_m
        boxes.append(visibility.make_box(
            f"robot:{r['name']}", (float(r["xy"][0]), float(r["xy"][1]), floor_z + h / 2.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (half, half, h / 2.0),
        ))
    return boxes


def hull_column_boxes(hull_ranges: list, margin_m: float = 0.0) -> list:
    """The object hull world AABBs (`_collect_object_hull_ranges`) as occluder boxes."""
    boxes = []
    for h in hull_ranges:
        lo = (h["min"][0] - margin_m, h["min"][1] - margin_m, h["min"][2])
        hi = (h["max"][0] + margin_m, h["max"][1] + margin_m, h["max"][2])
        boxes.append(visibility.aabb_box(f"hull:{h['prim']}", lo, hi))
    return boxes


def robot_footprint_ranges(robot_footprints: list, floor_z: float) -> list:
    """The robots' current poses as world AABBs (`{"prim", "min", "max"}`) for the
    camera-clearance check - conservative: half-diagonal square, floor -> body top."""
    out = []
    for r in robot_footprints:
        h = float(r.get("height_m") or 0.5)
        half = float(r["half_diagonal_m"])
        x, y = float(r["xy"][0]), float(r["xy"][1])
        out.append({"prim": f"robot:{r['name']}", "min": (x - half, y - half, floor_z), "max": (x + half, y + half, floor_z + h)})
    return out


class CameraClearanceViolation(RuntimeError):
    """Addendum 2026-09-07: a camera eye inside, or within CAMERA_MIN_CLEARANCE_M of, a
    prim's world bounding box on some frame - the run FAILS on the first one."""

    def __init__(self, violation: dict):
        super().__init__(f"camera clearance violation on frame {violation.get('frame')}: eye {violation.get('camera')} "
                         f"is {violation.get('distance_m'):.3f} m from {violation.get('prim')} (< {violation.get('min_clearance_m')} m)")
        self.violation = violation


def choose_close_shot_eye(drop_xy, floor_z: float, *, occluders: list, hull_ranges: list, robot_footprints: list,
                          room_outline_xy: list | None = None, radii_m=CLOSE_SHOT_RADII_M, heights_m=CLOSE_SHOT_HEIGHTS_M,
                          azimuth_step_deg: float = CLOSE_SHOT_AZIMUTH_STEP_DEG, rest_area_m: float = CLOSE_SHOT_REST_AREA_M,
                          clearance_ranges: list | None = None, min_clearance_m: float = visibility.CAMERA_MIN_CLEARANCE_M,
                          transition_from=None) -> dict:
    """Morning-5: the sphere-drop/static close shot's eye, chosen with the wide shot's
    occlusion machinery (`scripts/msa/visibility.py`). T15e run3 ended with the close
    shot half-filled by Husky's proxy: the eye was `drop point - back_off` along Y
    regardless of what stood there. Now every candidate on the ring (radius x height
    x azimuth) must (a) not lie inside any robot footprint (half-diagonal +
    SPHERE_DROP_ROBOT_MARGIN_M) or hull column (+ CLOSE_SHOT_HULL_MARGIN_M), (b) see
    the drop point (at floor + sphere radius) and the 8 points of the sphere's expected
    rest area (drop point +- `rest_area_m`, on the floor) past the wide shot's
    occluders (kept stub edges + placeholder parts), the hull columns and the robots'
    current boxes; the CLOSEST valid eye wins (then the lowest, then inside the room
    before outside). If none is valid the candidate with the fewest occluded points is
    returned with `ok = False`. Returns `{"eye", "look_at", "ok", "radius_m",
    "height_m", "azimuth_deg", "n_candidates", "n_valid", "occluded", ...}`.

    Addendum (2026-09-07): with `clearance_ranges` (every prim's world AABB) the eye
    must also keep >= `min_clearance_m` from every prim AND from the robots' current
    boxes (`camera_clearance_violation`), and, with `transition_from` (the wide eye),
    the straight camera move from there must keep the same clearance at every
    interpolated point (`segment_clearance_violation`) - the worker fails the run on
    the first frame that violates it, so the solver must never pick such a pose."""
    dx, dy = float(drop_xy[0]), float(drop_xy[1])
    look_at = (dx, dy, floor_z + CLOSE_SHOT_LOOK_AT_HEIGHT_M)
    robots = list(robot_footprints or [])
    boxes = list(occluders or []) + hull_column_boxes(hull_ranges) + robot_footprint_boxes(robots, floor_z)
    ranges = list(clearance_ranges or []) + robot_footprint_ranges(robots, floor_z)
    points = [("drop_point", (dx, dy, floor_z + SPHERE_DROP_RADIUS_M))]
    for k in range(8):
        a = 2 * math.pi * k / 8
        points.append((f"rest_{k}", (dx + rest_area_m * math.cos(a), dy + rest_area_m * math.sin(a), floor_z + SPHERE_DROP_RADIUS_M)))
    # A rest-area point that lies INSIDE a robot's (margin-grown) box or a hull column
    # cannot be seen from anywhere (the sphere cannot rest there either); it is
    # reported and left out of the check, like visibility.solve_visible_camera's
    # `inside` points. The drop point itself is always checked.
    inside = [label for label, p in points[1:] if any(visibility.point_inside_box(p, b) for b in boxes)]
    points = [(label, p) for label, p in points if label not in inside]
    n_az = max(1, int(round(360.0 / azimuth_step_deg)))
    best = None
    n_valid = 0
    n_candidates = 0
    for radius in radii_m:
        for height in heights_m:
            for k in range(n_az):
                az = k * azimuth_step_deg
                ex, ey = dx + radius * math.cos(math.radians(az)), dy + radius * math.sin(math.radians(az))
                eye = (ex, ey, floor_z + height)
                n_candidates += 1
                blocked_by = robots_blocking_point((ex, ey), robots) + hulls_blocking_column((ex, ey), hull_ranges, CLOSE_SHOT_HULL_MARGIN_M)
                clearance = visibility.camera_clearance_violation(eye, ranges, min_clearance_m) if ranges else None
                transition = None
                if clearance is None and transition_from is not None and ranges:
                    transition = visibility.segment_clearance_violation(tuple(transition_from), eye, ranges, min_clearance_m)
                occluded = [{"label": label, "by": by} for label, p in points for by in [visibility.first_occluder(eye, p, boxes)] if by is not None]
                inside_room = point_in_polygon_xy((ex, ey), room_outline_xy) if room_outline_xy else True
                ok = not blocked_by and not occluded and clearance is None and transition is None
                n_valid += int(ok)
                score = (0 if ok else 1, 0 if clearance is None else 1, 0 if transition is None else 1, len(blocked_by), len(occluded),
                         0 if inside_room else 1, radius, height, az)
                if best is None or score < best[0]:
                    best = (score, {"eye": list(eye), "look_at": list(look_at), "ok": ok, "radius_m": radius, "height_m": height,
                                    "azimuth_deg": az, "eye_blocked_by": blocked_by, "occluded": occluded, "inside_room": inside_room,
                                    "clearance_violation": clearance, "transition_violation": transition})
    out = dict(best[1])
    out.update({"n_candidates": n_candidates, "n_valid": n_valid, "n_check_points": len(points), "rest_points_inside_boxes": inside,
                "rest_area_m": rest_area_m,
                "n_occluder_boxes": len(boxes), "n_clearance_ranges": len(ranges), "min_clearance_m": min_clearance_m,
                "robots_checked": [r["name"] for r in robots],
                "rule": "morning-5: ring of eyes (radius x height x azimuth) around the drop point; not inside a robot footprint / hull column; "
                        ">= min_clearance_m from every prim AABB and robot box (eye and the transition from the wide eye); "
                        "drop point + rest area (8 points at +-0.5 m) unoccluded by stubs, parts, hulls and robots; closest valid eye wins"})
    return out


def _collect_clearance_ranges(stage) -> list:
    """Phase 2 (pxr): the world AABB of EVERY geometric prim the camera must keep
    clear of - all Mesh/Cube/Sphere/BasisCurves prims under /World/Structure (stubs
    AND the invisible full-height colliders), /World/Floor and /World/Objects (visual
    parts and hulls), visibility ignored. Robots and the drop sphere are dynamic and
    added per frame (`robot_footprint_ranges`, the sphere's own box)."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"], useExtentsHint=False, ignoreVisibility=True)
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not (path.startswith("/World/Structure/") or path == "/World/Floor" or path.startswith("/World/Objects/")):
            continue
        if prim.GetTypeName() not in ("Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule"):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        out.append({"prim": path, "min": (float(lo[0]), float(lo[1]), float(lo[2])), "max": (float(hi[0]), float(hi[1]), float(hi[2]))})
    return out


def proxy_visibility_schedule(frame: int, burger_start_frame: int, husky_start_frame: int) -> dict:
    """T15i: which robot proxy (body + clearance ring) is shown on `frame`. Burger -
    placed at the route start - is visible with its ring from frame 0 (the establishing
    shot shows the robot that is about to drive) until Husky's pass starts, then it is
    parked 20 m away and hidden; Husky is pinned off-room and hidden until its pass
    starts, then visible (with its ring) for the rest of the clip - including the drop
    and static phases, where it stands held at the pose it finished/blocked in. A ring
    is only ever visible for a robot that is in the room, and it is repositioned to
    that robot's pose on the same frame it becomes visible (T15e run2: Husky's ring was
    made visible at f420 but only moved at f438 - its translate op had never been set -
    so for 18 frames it sat at the world origin, 1.24 m outside the hero room: the
    "yellow ring bottom-right" of frame f0420). Trails are handled separately: hidden
    until they have >= 2 real points (a fresh trail is a degenerate 2-point curve at
    the origin - the stray "yellow dot" inside that ring and in Burger's frames)."""
    burger_visible = frame < husky_start_frame
    husky_visible = frame >= husky_start_frame
    return {
        "burger": {"body": burger_visible, "ring": burger_visible},
        "husky": {"body": husky_visible, "ring": husky_visible},
    }


def _collect_object_hull_ranges(stage, bbcache) -> list:
    """Phase 2 (pxr): world-space extents of every object collider - MSA's
    `/World/Objects/<id>/collision/hull` meshes, or the main pipeline's hull
    Mesh/Cube prims directly under `/World/Objects/` - as `choose_sphere_drop_point`
    wants them. `/visual/` parts (never colliders, SPEC §5) are skipped."""
    ranges = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Objects/") or "/visual/" in path or prim.GetTypeName() not in ("Mesh", "Cube"):
            continue
        rng = bbcache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        ranges.append({"prim": path, "min": (float(lo[0]), float(lo[1]), float(lo[2])), "max": (float(hi[0]), float(hi[1]), float(hi[2]))})
    return ranges


def _make_ring_mesh(stage, path: str, radius_m: float, color: tuple[float, float, float]):
    """A flat annulus (outward radius `radius_m`, RING_THICKNESS_M wide) lying in the
    prim's local XY plane - Isaac's ground plane is XY (Z-up), so no rotation is
    needed, just a translate op each frame to follow the robot (see _update_ring)."""
    from pxr import Gf, UsdGeom, Vt

    outer = radius_m
    inner = max(0.001, radius_m - RING_THICKNESS_M)
    n = RING_SEGMENTS
    points = []
    for ring_radius in (outer, inner):
        for i in range(n):
            theta = 2 * math.pi * i / n
            points.append(Gf.Vec3f(ring_radius * math.cos(theta), ring_radius * math.sin(theta), 0.0))

    face_vertex_counts = []
    face_vertex_indices = []
    for i in range(n):
        o0, o1 = i, (i + 1) % n
        i0, i1 = n + i, n + (i + 1) % n
        face_vertex_counts.append(4)
        face_vertex_indices += [o0, o1, i1, i0]

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreateDoubleSidedAttr(True)
    mesh.AddTranslateOp()
    return mesh


def _update_ring(ring_mesh, center_isaac) -> None:
    ring_mesh.GetOrderedXformOps()[0].Set(center_isaac)


def _make_trail_curve(stage, path: str, color: tuple[float, float, float]):
    from pxr import Gf, UsdGeom, Vt

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    # start with a degenerate 2-point curve at the origin - real points land on the
    # first _update_trail call. An empty curve (0 points) is invalid USD.
    curves.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 0)]))
    curves.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
    curves.CreateWidthsAttr(Vt.FloatArray([TRAIL_WIDTH_M, TRAIL_WIDTH_M]))
    return curves


def _update_trail(curves, points_isaac: list) -> None:
    """T15i: a trail stays invisible until it has >= 2 real points (the placeholder
    2-point curve at the origin was the stray dot in T15e run2's wide frames)."""
    from pxr import UsdGeom, Vt

    if len(points_isaac) < 2:
        return
    curves.GetPointsAttr().Set(Vt.Vec3fArray(points_isaac))
    curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(points_isaac)]))
    curves.GetWidthsAttr().Set(Vt.FloatArray([TRAIL_WIDTH_M] * len(points_isaac)))
    UsdGeom.Imageable(curves.GetPrim()).MakeVisible()


def _set_visible(prim, visible: bool) -> None:
    from pxr import UsdGeom

    if visible:
        UsdGeom.Imageable(prim).MakeVisible()
    else:
        UsdGeom.Imageable(prim).MakeInvisible()


def _make_robot_proxy(stage, group_path: str, length_m: float, width_m: float, height_m: float):
    """A box - RigidBody + CollisionAPI on a `length_m x width_m x height_m`
    UsdGeom.Cube, replacing the same-radius-in-every-direction cylinder this used to
    be (feat/footprint-planner) - plus its clearance ring and its trail, all grouped
    under one Xform so activating/deactivating the group shows/hides all three
    together (used to remove Burger from the scene once Husky's pass begins - see
    run_isaac_worker)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    UsdGeom.Xform.Define(stage, group_path)
    cube = UsdGeom.Cube.Define(stage, f"{group_path}/Body")
    cube.CreateSizeAttr(1.0)  # unit cube [-0.5, 0.5]^3, so the scale op == full extents -
    # same convention app.services.usd_export._add_cube uses for its own USD boxes.
    cube.CreateDisplayColorAttr([Gf.Vec3f(*ROBOT_COLOR)])
    # Translate op MUST be added before the scale op. xformOpOrder lists ops in the
    # order applied to a point, so [translate, scale] gives the intended
    # world = (local * scale) + center; adding scale first would apply translate to
    # the point BEFORE scale, silently multiplying this prim's own position by its
    # size - the exact bug _add_cube's own comment documents as the root cause of a
    # real ~9m Floor/Structure-vs-Objects offset on this project. Double precision to
    # match _add_cube too: run_isaac_worker repositions this op every frame (via
    # GetOrderedXformOps()[0].Set(...) below and RigidPrim world-pose calls in the
    # main loop), and float32 rounding was measurably enough to break exact-agreement
    # numerics at comparable scale values there.
    cube.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    cube.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(length_m, width_m, height_m))
    body_prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    UsdPhysics.CollisionAPI.Apply(body_prim)
    # Scaled off the burger reference platform's own footprint AREA (was radius-ratio
    # under the cylinder proxy, which had no notion of length vs width) - a wide-but-
    # thin platform (e.g. Jackal) now gets a mass proportional to how much floor it
    # actually covers, not to a single half-diagonal-derived circle radius.
    burger_area_m2 = 0.138 * 0.178
    mass_kg = max(1.0, 4.0 * ((length_m * width_m) / burger_area_m2))
    UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(mass_kg)

    # The clearance ring is still drawn as a circle (a rectangle "clearance zone"
    # indicator wasn't in scope for this pass - see the rectangle indicator in
    # frontend/src/components/SceneMap3D.tsx for the 3D-viewer equivalent instead) -
    # its radius is now the L x W footprint's own half-diagonal, the same
    # mesh_measured convention app/robots.py's RadiusSource.MESH_MEASURED uses, rather
    # than a separately-passed radius_m.
    ring_radius_m = math.hypot(length_m, width_m) / 2.0
    ring = _make_ring_mesh(stage, f"{group_path}/Ring", ring_radius_m, ROBOT_COLOR)
    trail = _make_trail_curve(stage, f"{group_path}/Trail", ROBOT_COLOR)
    # T15i: nothing of a proxy is visible until the main loop says so (the ring is
    # positioned on the frame it is shown; the trail once it has real points).
    UsdGeom.Imageable(ring.GetPrim()).MakeInvisible()
    UsdGeom.Imageable(trail.GetPrim()).MakeInvisible()
    return body_prim, ring, trail, mass_kg


def _drive_toward_waypoints(robot_rigid, path_xy: list[tuple[float, float]], state: dict, speed_mps: float) -> None:
    """Advances `state` (in place: wp_idx, stall_frames, prev_xy, blocked, finished)
    by one physics step, commanding velocity toward the current waypoint. A "stalled"
    robot (no progress for STALL_FRAMES_THRESHOLD frames while a waypoint remains) is
    marked `blocked` and given zero velocity - this is the demo's whole point for
    Husky: real geometry, not a scripted failure, stops it."""
    if state["finished"] or state["blocked"]:
        return
    pos, _ = robot_rigid.get_world_poses()
    cur_xy = (float(pos[0][0]), float(pos[0][1]))

    if state["prev_xy"] is not None:
        moved = math.hypot(cur_xy[0] - state["prev_xy"][0], cur_xy[1] - state["prev_xy"][1])
        state["stall_frames"] = 0 if moved >= STALL_EPS_M else state["stall_frames"] + 1
    state["prev_xy"] = cur_xy

    if state["stall_frames"] >= state["stall_threshold_frames"]:
        robot_rigid.set_linear_velocities([[0.0, 0.0, 0.0]])
        state["blocked"] = True
        return

    target_xy = path_xy[state["wp_idx"]]
    dx, dy = target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1]
    dist = math.hypot(dx, dy)
    if dist < WAYPOINT_ARRIVE_EPS_M:
        state["wp_idx"] += 1
        state["stall_frames"] = 0
        if state["wp_idx"] >= len(path_xy):
            robot_rigid.set_linear_velocities([[0.0, 0.0, 0.0]])
            state["finished"] = True
            return
        target_xy = path_xy[state["wp_idx"]]
        dx, dy = target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1]
        dist = math.hypot(dx, dy)

    vx, vy = (dx / dist) * speed_mps, (dy / dist) * speed_mps
    robot_rigid.set_linear_velocities([[vx, vy, 0.0]])


def run_isaac_worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    result: dict = {"stage": "start"}
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdLux
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim

        usd_context = omni.usd.get_context()
        usd_context.open_stage(spec["usd_path"])
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(f"failed to open stage {spec['usd_path']}")
        result["stage_opened"] = True

        physics_path = "/World/PhysicsScene"
        if not stage.GetPrimAtPath(physics_path):
            physics_scene = UsdPhysics.Scene.Define(stage, physics_path)
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
            physics_scene.CreateGravityMagnitudeAttr(9.81)

        # usd_export.py EXPORT_SCHEMA_VERSION 4 (feat/isaac-demo-look) made Objects
        # static collision only (no RigidBodyAPI) - no more kinematic-freeze
        # workaround needed here for the dynamics-instability finding that used to
        # require one (see docs/DECISIONS.md).

        # Scenes bake in both a DomeLight AND a DistantLight key light (usd_export.py
        # since schema 3; the MSA presentation export authors /World/DomeLight +
        # /World/KeyLight itself, T15d) - these fallbacks only fire for a stage that
        # has no UsdLux prims at all, or is dome-only. Never double-added: a stage
        # that already carries a light of the given type keeps exactly what it has.
        dome_prims = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        distant_prims = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)]
        lights_added = []
        if not dome_prims:
            dome = UsdLux.DomeLight.Define(stage, "/World/DemoDomeLight")
            dome.CreateIntensityAttr(DOME_LIGHT_FALLBACK_INTENSITY)
            lights_added.append("/World/DemoDomeLight")
        if not distant_prims:
            fill = UsdLux.DistantLight.Define(stage, "/World/DemoFillLight")
            fill.CreateIntensityAttr(KEY_LIGHT_FALLBACK_INTENSITY)
            UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3d(-55, 25, 0))
            lights_added.append("/World/DemoFillLight")
        result["lights_added"] = lights_added
        result["lights_found"] = {"dome": dome_prims, "distant": distant_prims}

        # Dark background: recolor whichever DomeLight is actually in the stage (the
        # one usd_export.py bakes in, or the DemoDomeLight fallback just above) - see
        # BACKDROP_COLOR's comment for why this replaced a separate backdrop mesh.
        for dome_prim in stage.Traverse():
            if not dome_prim.IsA(UsdLux.DomeLight):
                continue
            dome_light = UsdLux.DomeLight(dome_prim)
            dome_light.CreateColorAttr(Gf.Vec3f(*BACKDROP_COLOR))

        bbcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"])

        floor_prim = stage.GetPrimAtPath("/World/Floor")
        if not floor_prim.IsValid():
            raise RuntimeError("/World/Floor missing")
        floor_rng = bbcache.ComputeWorldBound(floor_prim).ComputeAlignedRange()
        floor_min, floor_max = floor_rng.GetMin(), floor_rng.GetMax()
        floor_top_z = float(floor_max[2])

        path_prim = stage.GetPrimAtPath("/World/Path")
        if not path_prim.IsValid():
            raise RuntimeError(
                "/World/Path missing - this worker needs a planned-route export "
                "(Phase 1 without --usd-path-override, or a USD copied from one)"
            )
        path_points_isaac = [tuple(pt) for pt in UsdGeom.BasisCurves(path_prim).GetPointsAttr().Get()]
        path_xy = [(p[0], p[1]) for p in path_points_isaac]
        result["path_points_count"] = len(path_xy)

        # T15h: /World/Target is the route GOAL (export_presentation PATH_SCHEMA /2) -
        # the acceptance distance below is measured against it; /World/TargetObject
        # (the target object's footprint centroid) is reported when present.
        target_prim = stage.GetPrimAtPath("/World/Target")
        if target_prim.IsValid():
            target_isaac = tuple(UsdGeom.Xformable(target_prim).GetOrderedXformOps()[0].Get())
        else:
            target_isaac = path_points_isaac[-1]
        result["target_isaac"] = [float(v) for v in target_isaac]
        result["target_to_last_path_point_m"] = math.hypot(
            float(target_isaac[0]) - float(path_points_isaac[-1][0]), float(target_isaac[1]) - float(path_points_isaac[-1][1])
        )
        target_object_prim = stage.GetPrimAtPath("/World/TargetObject")
        if target_object_prim.IsValid():
            to = UsdGeom.Xformable(target_object_prim).GetOrderedXformOps()[0].Get()
            result["target_object_isaac"] = [float(v) for v in to]
            id_attr = target_object_prim.GetAttribute("cloudeye:targetObjectId")
            result["target_object_id"] = id_attr.Get() if id_attr and id_attr.HasValue() else None

        # T15h: sphere-drop point = a route point verified free of every object hull
        # extent - T15i: chosen AT DROP TIME (see the drop_start_frame branch below) so
        # both robots' CURRENT footprints are avoided too; this planning-time preview is
        # only logged.
        hull_ranges = _collect_object_hull_ranges(stage, bbcache)
        drop_preview = choose_sphere_drop_point(path_xy, hull_ranges, SPHERE_DROP_RADIUS_M)
        result["sphere_drop_point_preview_hulls_only"] = {**drop_preview, "hulls_checked": len(hull_ranges)}
        drop_xy = (float(drop_preview["xy"][0]), float(drop_preview["xy"][1]))

        fps = int(spec["fps"])
        physics_dt = 1.0 / fps

        establish_frames = int(round(spec["establish_s"] * fps))
        pass_frames = int(round(spec["pass_s"] * fps))
        drop_frames = int(round(spec["drop_s"] * fps))
        static_frames = int(round(spec["static_s"] * fps))
        total_frames = establish_frames + 2 * pass_frames + drop_frames + static_frames
        burger_start_frame = establish_frames
        husky_start_frame = establish_frames + pass_frames
        drop_start_frame = establish_frames + 2 * pass_frames
        transition_frames = max(1, int(round(CAMERA_TRANSITION_S * fps)))
        settle_frames = max(1, int(round(ROBOT_SETTLE_S * fps)))
        stall_threshold_frames = int(round(0.5 * fps))

        # --- wide shot (establish + both drive passes): the web 3/4 rule's camera,
        # resolved by Phase 1 into spec["wide_camera"] (see resolve_wide_camera /
        # PRESENTATION_CAMERA_PATH) - this interpreter only applies it. Fixed for the
        # whole window. The previous tilt-shift/top-down wide shot and its per-scene
        # aperture solve are gone (T15d); what survives from the 2026-09-05 v3
        # finding is the aperture handling: Isaac's Hydra pipeline re-derives
        # verticalAperture = horizontalAperture * h / w regardless of what's
        # authored, so the vertical FOV the rule was solved for is pinned by authoring
        # horizontalAperture = verticalAperture * (w / h) at a fixed focal length -
        # the value Isaac derives back is then exactly the one we wanted.
        WIDE_REFERENCE_FOCAL_MM = WIDE_SHOT_FOCAL_MM
        CLOSE_H_APERTURE_MM = 36.0
        CLOSE_V_APERTURE_MM = 24.0
        CLOSE_FOCAL_MM = 24.0

        wide_spec = spec.get("wide_camera")
        if not wide_spec:
            raise RuntimeError(
                "spec has no 'wide_camera' - re-run Phase 1 (this script's driver mode) with "
                "this version; the wide shot is resolved there (record_isaac.SCHEMA /3)"
            )
        render_w, render_h = int(spec["width"]), int(spec["height"])
        aspect = render_w / render_h
        fov_v_deg = float(wide_spec["fov_v_deg"])
        wide_forward = _v_norm(tuple(wide_spec["forward"]))
        wide_right_vec = _v_norm(tuple(wide_spec["right"]))
        wide_up_vec = _v_norm(tuple(wide_spec["up"]))
        wide_eye_spec = tuple(float(v) for v in wide_spec["eye"])

        # Framing + visibility guarantee (T15i, scripts/msa/visibility.py - the same
        # rule export_presentation baked the camera with, re-run here with the LARGER
        # platform's ring and height since both robots ride this camera): the robot
        # proxy start (footprint + top), every path point, the clearance ring and the
        # whole room outline (floor + stub top) must sit inside the frame with
        # FRAMING_MARGIN_FRAC on every side, AND the route start, every path point, the
        # goal, the robot tops at start/goal and >= half of each ring must be
        # unoccluded by the opaque boxes in spec["wide_camera"]["occluders"] (the
        # kept stub edges + non-translucent placeholder parts). If the baked pose
        # fails, the eye is raised in 0.25 m steps to 4.5 m above the floor (re-aimed
        # at the baked floor aim point), then backed off along the aim->eye direction
        # in 0.1 m steps - deterministic; everything is logged.
        start_x0, start_y0 = path_xy[0]
        ring_check_radius = max(math.hypot(p["length_m"], p["width_m"]) / 2.0 for p in spec["platforms"])
        robot_check_height = max(float(p["height_m"]) for p in spec["platforms"])
        path_pts_f = [tuple(float(v) for v in p) for p in path_points_isaac]
        frame_points = framing_check_points(
            (start_x0, start_y0, floor_top_z), path_pts_f, ring_check_radius, robot_check_height, ground_axes=(0, 1), up_axis=2,
        )
        stub_top_z = floor_top_z + float(stage.GetRootLayer().customLayerData.get("cloudeye:presentationWallHeightM", 1.2))
        room_outline = [tuple(float(v) for v in p) for p in (wide_spec.get("room_outline_isaac") or [])]
        outline_points = [(x, y, floor_top_z) for x, y in room_outline] + [(x, y, stub_top_z) for x, y in room_outline]
        vis_points = visibility.visibility_check_points(
            (start_x0, start_y0, floor_top_z + 0.03), path_pts_f, path_pts_f[-1], ring_check_radius, robot_check_height,
            ground_axes=(0, 1), up_axis=2,
        )
        occluders = list(wide_spec.get("occluders") or [])
        aim = tuple(float(v) for v in wide_spec["aim"]) if wide_spec.get("aim") else visibility.aim_point_on_floor(
            wide_eye_spec, wide_forward, floor_top_z, up_axis=2,
        )
        # Morning-4: the path RIBBON must stay >= PATH_VISIBILITY_MIN_FRACTION visible with
        # a robot (either platform, oriented along the route) standing at any point of
        # the route, sampled every ROUTE_ROBOT_SAMPLE_STEP_M - worst case over samples.
        # Addendum: the eye keeps >= CAMERA_MIN_CLEARANCE_M from every prim's AABB.
        path_vis_min = float(wide_spec.get("path_visibility_min_fraction") or visibility.PATH_VISIBILITY_MIN_FRACTION)
        ribbon_points = visibility.ribbon_sample_points(path_pts_f, floor_level=floor_top_z, up_axis=2)
        clearance_ranges = _collect_clearance_ranges(stage)
        result["camera_clearance"] = {"min_clearance_m": visibility.CAMERA_MIN_CLEARANCE_M, "n_static_ranges": len(clearance_ranges),
                                      "rule": "every frame: eye not inside / within min_clearance_m of any prim AABB (static prims, robots at their current pose, the sphere)"}
        # Two-pose mode (orchestrator 2026-09-07): with spec["wide_camera_husky"] the
        # baked 3/4 pose serves establish + Burger's pass (checked with Burger's box on
        # the route) and the steeper pose serves Husky's pass (Husky's box); the move
        # between them is a smootherstep of <= 1 s whose every frame passes the 0.3 m
        # clearance (checked per frame below, and the straight segment here).
        husky_spec = spec.get("wide_camera_husky")
        platforms_by_id = {p["id"]: p for p in spec["platforms"]}

        def solve_pose(cam_spec, platform_ids, *, transition_from=None):
            eye_spec = tuple(float(v) for v in cam_spec["eye"])
            fwd_spec = _v_norm(tuple(cam_spec["forward"]))
            cam_aim = tuple(float(v) for v in cam_spec["aim"]) if cam_spec.get("aim") else visibility.aim_point_on_floor(
                eye_spec, fwd_spec, floor_top_z, up_axis=2,
            )
            cam_occluders = list(cam_spec.get("occluders") or [])
            box_sets = visibility.route_robot_box_sets(path_pts_f, [platforms_by_id[p] for p in platform_ids], floor_level=floor_top_z, up_axis=2)
            s = visibility.solve_visible_camera(
                eye_spec, cam_aim, fov_v_deg, aspect, frame_points, vis_points, cam_occluders, up_axis=2, floor_level=floor_top_z,
                outline_points=outline_points, ribbon_points=ribbon_points, robot_box_sets=box_sets,
                path_visibility_min_fraction=path_vis_min, clearance_ranges=clearance_ranges, transition_from=transition_from,
            )
            rec = {
                **{k: v for k, v in cam_spec.items() if k != "occluders"},
                "eye_after_backoff": list(s["eye"]), "eye_after_solve": list(s["eye"]),
                "forward_after_solve": list(s["forward"]), "aim": list(cam_aim), "aspect": aspect,
                "framing_margin_frac": FRAMING_MARGIN_FRAC, "framing_points_checked": len(frame_points),
                "outline_margin_frac": visibility.OUTLINE_MARGIN_FRAC, "outline_points_checked": len(outline_points),
                "framing_ring_radius_m": ring_check_radius,
                "framing_ok": bool(s["framing_ok"]),
                "visibility_ok": bool(s["visibility"]["ok"]),
                "visibility_points_checked": len(vis_points),
                "visibility": s["visibility"],
                "path_visibility_worst_case": s["path_visibility"],
                "path_visibility_ok": bool(s["path_visibility"]["ok"]) if s.get("path_visibility") else None,
                "path_visibility_min_fraction": path_vis_min,
                "route_robot_platforms": list(platform_ids),
                "n_ribbon_points": len(ribbon_points), "n_route_robot_placements": len(box_sets),
                "eye_clearance": s["eye_clearance"],
                "n_occluders": len(cam_occluders),
                "height_above_floor_m": s["height_above_floor_m"],
                "height_raise_m": s["height_raise_m"],
                "backoff_m": s["backoff_m"],
                "pitch_deg_after_solve": s["pitch_deg"],
                "n_candidates_tried": s["n_candidates"],
                "camera_ok": bool(s["ok"]),
            }
            return s, rec, cam_aim, cam_occluders

        wide_platforms = [p["id"] for p in spec["platforms"]] if husky_spec is None else [spec["platforms"][0]["id"]]
        solved, result["wide_camera"], aim, occluders = solve_pose(wide_spec, wide_platforms)
        wide_eye_t, wide_forward, wide_right_vec, wide_up_vec = solved["eye"], solved["forward"], solved["right"], solved["up"]
        camera_backoff_m = solved["backoff_m"]
        result["camera_backoff_m"] = camera_backoff_m
        result["camera_height_raise_m"] = solved["height_raise_m"]
        result["wide_shot_pitch_deg"] = math.degrees(math.asin(max(-1.0, min(1.0, wide_forward[2]))))
        husky_eye_t, husky_aim = None, None
        if husky_spec is not None:
            solved_h, result["wide_camera_husky"], husky_aim, _ = solve_pose(husky_spec, [spec["platforms"][1]["id"]], transition_from=wide_eye_t)
            husky_eye_t = solved_h["eye"]
            result["wide_camera_husky"]["transition_from_eye"] = list(wide_eye_t)
            result["wide_camera_husky"]["transition_clearance"] = visibility.segment_clearance_violation(tuple(wide_eye_t), tuple(husky_eye_t), clearance_ranges)
            result["wide_camera_husky"]["pitch_deg_after_solve"] = solved_h["pitch_deg"]
        else:
            result["wide_camera_husky"] = None
        result["camera_ok"] = bool(result["wide_camera"]["camera_ok"] and (husky_spec is None or result["wide_camera_husky"]["camera_ok"]))

        wide_eye = Gf.Vec3d(*wide_eye_t)
        wide_center = Gf.Vec3d(*aim)
        husky_eye = Gf.Vec3d(*husky_eye_t) if husky_eye_t is not None else None
        husky_center = Gf.Vec3d(*husky_aim) if husky_aim is not None else None
        # the wide pose the drop transition starts from (the Husky pose in two-pose mode)
        last_wide_eye = husky_eye if husky_eye is not None else wide_eye
        last_wide_center = husky_center if husky_center is not None else wide_center
        last_wide_eye_t = tuple(husky_eye_t) if husky_eye_t is not None else tuple(wide_eye_t)
        up = Gf.Vec3d(0, 0, 1)
        wide_v_aperture = 2.0 * WIDE_REFERENCE_FOCAL_MM * math.tan(math.radians(fov_v_deg) / 2.0)
        wide_h_aperture = wide_v_aperture * aspect
        wide_v_aperture_offset = 0.0

        # --- sphere-drop / closing static shot: fixed, centered on the DROP POINT (T15h:
        # never the target; T15i: the point is chosen when the drop phase starts, so the
        # close shot is aimed then too). Morning-5: the eye comes from
        # `choose_close_shot_eye` - the wide shot's occluders + hull columns + both
        # robots' current boxes; T15e run3's close shot was half Husky proxy because the
        # eye was placed by a fixed back-off with no check of what stood there.
        def place_close_shot(xy, robots: list):
            chosen = choose_close_shot_eye(
                xy, floor_top_z, occluders=occluders, hull_ranges=hull_ranges, robot_footprints=robots, room_outline_xy=room_outline,
                clearance_ranges=clearance_ranges, transition_from=last_wide_eye_t,
            )
            result["close_shot"] = chosen
            result["close_shot_center_isaac"] = list(chosen["look_at"])
            return Gf.Vec3d(*chosen["eye"]), Gf.Vec3d(*chosen["look_at"])

        drop_eye, drop_target = place_close_shot(drop_xy, [])
        result["close_shot_preview_no_robots"] = result["close_shot"]

        cam_path = "/World/DemoCamera"
        UsdGeom.Camera.Define(stage, cam_path)
        cam_prim = stage.GetPrimAtPath(cam_path)
        cam_geom = UsdGeom.Camera(cam_prim)
        h_aperture_attr = cam_geom.CreateHorizontalApertureAttr(wide_h_aperture)
        v_aperture_attr = cam_geom.CreateVerticalApertureAttr(wide_v_aperture)
        v_aperture_offset_attr = cam_geom.CreateVerticalApertureOffsetAttr(wide_v_aperture_offset)
        focal_length_attr = cam_geom.CreateFocalLengthAttr(WIDE_REFERENCE_FOCAL_MM)
        cam_xf = UsdGeom.Xformable(cam_prim)
        cam_xf.ClearXformOpOrder()
        cam_xform_op = cam_xf.AddTransformOp()

        def set_camera(eye: Gf.Vec3d, target: Gf.Vec3d) -> None:
            view = Gf.Matrix4d(1).SetLookAt(eye, target, up)
            cam_xform_op.Set(view.GetInverse())

        rep.orchestrator.set_capture_on_play(False)
        render_product = rep.create.render_product(cam_path, (int(spec["width"]), int(spec["height"])))
        writer = rep.WriterRegistry.get("BasicWriter")
        frames_dir = Path(spec["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        writer.initialize(output_dir=str(frames_dir), rgb=True)
        writer.attach([render_product])

        # --- robot proxies: BOTH created active from frame 0 (Husky parked well off
        # to the side, invisible, until its pass) - deliberately NOT using
        # Usd.Prim.SetActive() to show/hide either one mid-simulation: doing that to
        # a prim already tracked by a live RigidPrim tensor view was confirmed (a real
        # GPU run, feat/isaac-demo-look v2) to invalidate PhysX's WHOLE
        # simulationView, breaking every other RigidPrim query for the rest of the
        # run ("Failed to get rigid body velocities from backend"). Visibility
        # (UsdGeom.Imageable) and position (RigidPrim.set_world_poses) are ordinary
        # runtime operations that don't touch the tensor view - but a parked body is
        # still fully dynamic, so gravity keeps accelerating it: a one-time
        # "teleport and zero velocity" isn't enough (also confirmed on a real run -
        # a robot parked 100m up free-fell back down, tunneled through the floor at
        # ~44m/s, and ended up 140m BELOW it). Re-pinned every frame it's parked
        # instead (see the main loop) - cheap, and reuses only already-proven calls.
        PARK_OFFSET_M = 20.0  # well outside any real room this demo targets

        burger_spec, husky_spec = spec["platforms"]
        start_x, start_y = path_xy[0]
        parked_pose = (start_x + PARK_OFFSET_M, start_y + PARK_OFFSET_M, floor_top_z + 2.0)

        def spawn_pose(height_m: float, xyz) -> Gf.Vec3d:
            x, y, z = xyz
            return Gf.Vec3d(x, y, z + height_m / 2.0 + 0.05)

        burger_body, burger_ring, burger_trail, _ = _make_robot_proxy(
            stage, "/World/BurgerProxy", burger_spec["length_m"], burger_spec["width_m"], burger_spec["height_m"]
        )
        UsdGeom.Xformable(burger_body).GetOrderedXformOps()[0].Set(
            spawn_pose(burger_spec["height_m"], (start_x, start_y, floor_top_z))
        )
        _update_ring(burger_ring, Gf.Vec3d(start_x, start_y, floor_top_z + RING_HEIGHT_ABOVE_FLOOR_M))
        UsdGeom.Imageable(burger_ring.GetPrim()).MakeVisible()  # frame 0: Burger + its ring at the start

        husky_body, husky_ring, husky_trail, _ = _make_robot_proxy(
            stage, "/World/HuskyProxy", husky_spec["length_m"], husky_spec["width_m"], husky_spec["height_m"]
        )
        UsdGeom.Xformable(husky_body).GetOrderedXformOps()[0].Set(
            spawn_pose(husky_spec["height_m"], parked_pose)
        )
        UsdGeom.Imageable(husky_body).MakeInvisible()
        # Husky's ring stays invisible (and unpositioned) until its pass - see
        # proxy_visibility_schedule; it is moved to Husky's pose on that same frame.
        ring_visibility_log: list = []  # T15i: (frame, burger_ring_visible, husky_ring_visible) at every switch

        fps_dt = physics_dt
        world = World(physics_dt=fps_dt, rendering_dt=fps_dt)
        world.reset()

        burger_rigid = RigidPrim("/World/BurgerProxy/Body")
        husky_rigid = RigidPrim("/World/HuskyProxy/Body")

        def pin_parked(rigid, height_m: float) -> None:
            rigid.set_world_poses(positions=[[parked_pose[0], parked_pose[1], parked_pose[2] + height_m / 2.0]])
            rigid.set_linear_velocities([[0.0, 0.0, 0.0]])
            rigid.set_angular_velocities([[0.0, 0.0, 0.0]])

        def hold_in_place(rigid, xyz) -> None:
            """Re-pin a rigid body at an arbitrary (already-reached) world position,
            every frame, the same way `pin_parked` holds a body at its fixed parked
            pose - a dynamic rigid body left alone still has gravity acting on it, so
            a one-time teleport+zero-velocity isn't enough (see pin_parked's own
            docstring/comment above). Used for a robot that finished or got blocked
            mid-drive: it must stay exactly where it stopped, not fall through the
            floor for the remaining frames (the bug this fixes - Husky previously had
            no re-pin after its own `state["finished"]`/`state["blocked"]` fired,
            unlike Burger, which was already re-parked every frame once Husky's pass
            started)."""
            rigid.set_world_poses(positions=[list(xyz)])
            rigid.set_linear_velocities([[0.0, 0.0, 0.0]])
            rigid.set_angular_velocities([[0.0, 0.0, 0.0]])

        # Real speed: cover the planned route length within one pass's duration, minus
        # the settle window each pass spends not yet driving.
        route_length_m = sum(
            math.dist(path_xy[i], path_xy[i + 1]) for i in range(len(path_xy) - 1)
        )
        available_s = max(0.5, spec["pass_s"] - ROBOT_SETTLE_S)
        # 1.3x margin: waypoint-following isn't a straight line at constant max
        # speed the whole way (deceleration/redirection at each waypoint), so the
        # naive route_length_m/available_s undershoots how fast it actually needs to
        # go to finish in time.
        drive_speed_mps = max(0.1, min(2.5, (route_length_m / available_s) * 1.3))
        result["route_length_m"] = route_length_m
        result["drive_speed_m_s"] = drive_speed_mps

        def new_drive_state() -> dict:
            return {
                "wp_idx": 0, "stall_frames": 0, "prev_xy": None,
                "blocked": False, "finished": False,
                "stall_threshold_frames": stall_threshold_frames,
            }

        burger_state = new_drive_state()
        husky_state = new_drive_state()
        burger_trail_points: list = []
        husky_trail_points: list = []
        active = None  # "burger" | "husky" | None
        husky_hold_pose: list | None = None  # set once husky finishes/is blocked; see hold_in_place
        burger_stop_pose: list | None = None  # T15h: Burger's pose when its pass ends (before parking)

        sphere_rigid = None

        # Per-frame trajectory log (both robots) - lightweight (frame, x, y, z) samples
        # for offline diagnosis/plotting. Not part of the pre-existing output contract
        # (spec.result.json), written alongside it as its own JSON file so existing
        # consumers of spec.result.json are unaffected.
        burger_trajectory: list[list[float]] = []
        husky_trajectory: list[list[float]] = []

        for frame in range(total_frames):
            if frame == burger_start_frame:
                active = "burger"
            elif frame == husky_start_frame:
                # Burger's job is done - park it (see pin_parked, applied every frame
                # below) and hide it (body AND ring - proxy_visibility_schedule), so
                # Husky isn't blocked by Burger's own resting body instead of the real
                # geometry. Its trail is a separate (non-physics) prim, left visible as
                # context for what Husky is about to attempt.
                sched = proxy_visibility_schedule(frame, burger_start_frame, husky_start_frame)
                _set_visible(burger_body, sched["burger"]["body"])
                _set_visible(burger_ring.GetPrim(), sched["burger"]["ring"])
                _bstop, _ = burger_rigid.get_world_poses()  # T15h: where Burger actually ended its pass
                burger_stop_pose = [float(_bstop[0][0]), float(_bstop[0][1]), float(_bstop[0][2])]
                pin_parked(burger_rigid, burger_spec["height_m"])

                husky_rigid.set_world_poses(
                    positions=[[start_x, start_y, floor_top_z + husky_spec["height_m"] / 2.0 + 0.05]],
                )
                husky_rigid.set_linear_velocities([[0.0, 0.0, 0.0]])
                husky_rigid.set_angular_velocities([[0.0, 0.0, 0.0]])
                # T15i: position Husky's ring at Husky's spawn pose BEFORE showing it -
                # the ring's translate op had never been set (world origin) in run2.
                _update_ring(husky_ring, Gf.Vec3d(start_x, start_y, floor_top_z + RING_HEIGHT_ABOVE_FLOOR_M))
                _set_visible(husky_body, sched["husky"]["body"])
                _set_visible(husky_ring.GetPrim(), sched["husky"]["ring"])
                ring_visibility_log.append({"frame": frame, "burger_ring": sched["burger"]["ring"], "husky_ring": sched["husky"]["ring"]})
                active = "husky"
            elif frame == drop_start_frame:
                active = None
                # T15i: the drop point is chosen NOW, against both robots' current
                # footprints (Burger parked 20 m off, Husky held where it stopped) and
                # every object hull - run2's sphere landed on blocked Husky's rear edge.
                _bnow, _ = burger_rigid.get_world_poses()
                _hnow, _ = husky_rigid.get_world_poses()
                robot_footprints = [
                    {"name": "burger", "xy": [float(_bnow[0][0]), float(_bnow[0][1])],
                     "half_diagonal_m": math.hypot(burger_spec["length_m"], burger_spec["width_m"]) / 2.0,
                     "height_m": float(burger_spec["height_m"])},
                    {"name": "husky", "xy": [float(_hnow[0][0]), float(_hnow[0][1])],
                     "half_diagonal_m": math.hypot(husky_spec["length_m"], husky_spec["width_m"]) / 2.0,
                     "height_m": float(husky_spec["height_m"])},
                ]
                # Morning-5: off-route fallback (nearest free floor cell to the goal) when
                # every route point is inside a robot's exclusion - run3's case.
                drop = choose_sphere_drop_point(
                    path_xy, hull_ranges, SPHERE_DROP_RADIUS_M, robot_footprints=robot_footprints, room_outline_xy=room_outline,
                )
                drop_xy = (float(drop["xy"][0]), float(drop["xy"][1]))
                result["sphere_drop_point"] = {
                    **drop, "hulls_checked": len(hull_ranges), "spawn_z": floor_top_z + SPHERE_DROP_HEIGHT_M,
                    "chosen_at_frame": frame, "robot_footprints_at_drop": robot_footprints,
                }
                drop_eye, drop_target = place_close_shot(drop_xy, robot_footprints)
                sphere_path = "/World/DropSphere"
                sphere_geom = UsdGeom.Sphere.Define(stage, sphere_path)
                sphere_geom.CreateRadiusAttr(SPHERE_DROP_RADIUS_M)
                sphere_geom.CreateDisplayColorAttr([Gf.Vec3f(*ROBOT_COLOR)])
                sphere_xf = UsdGeom.Xformable(sphere_geom)
                sphere_xf.AddTranslateOp().Set(Gf.Vec3d(drop_xy[0], drop_xy[1], floor_top_z + SPHERE_DROP_HEIGHT_M))
                sphere_prim = sphere_geom.GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
                UsdPhysics.CollisionAPI.Apply(sphere_prim)
                UsdPhysics.MassAPI.Apply(sphere_prim).CreateMassAttr(SPHERE_DROP_MASS_KG)
                sphere_rigid = RigidPrim(sphere_path)
                result["drop_sphere_spawned_frame"] = frame

            # --- keep the not-yet-active / already-done robot pinned at its parked
            # pose every frame - gravity is still simulating it otherwise, and a
            # one-time teleport+zero-velocity isn't enough to hold it there (see the
            # comment above pin_parked's definition). Burger's pinning and Husky's
            # hold-in-place are independent conditions, not mutually exclusive elif
            # branches - Burger must stay parked for the rest of the run once Husky's
            # pass starts, AND (separately) Husky must stay held once IT finishes/is
            # blocked, which can happen on any later frame - both need to keep firing
            # every frame after their own trigger point.
            if frame < husky_start_frame:
                pin_parked(husky_rigid, husky_spec["height_m"])
            else:
                pin_parked(burger_rigid, burger_spec["height_m"])
                if husky_hold_pose is not None:
                    hold_in_place(husky_rigid, husky_hold_pose)

            # --- driving (only the active robot, and only past its settle window) ---
            if active == "burger" and frame >= burger_start_frame + settle_frames:
                _drive_toward_waypoints(burger_rigid, path_xy, burger_state, drive_speed_mps)
                pos, _ = burger_rigid.get_world_poses()
                if frame % TRAIL_SAMPLE_EVERY_N_FRAMES == 0:
                    burger_trail_points.append(
                        Gf.Vec3f(float(pos[0][0]), float(pos[0][1]), floor_top_z + TRAIL_HEIGHT_ABOVE_FLOOR_M)
                    )
                    _update_trail(burger_trail, burger_trail_points)
            elif active == "husky" and frame >= husky_start_frame + settle_frames:
                _drive_toward_waypoints(husky_rigid, path_xy, husky_state, drive_speed_mps)
                pos, _ = husky_rigid.get_world_poses()
                if (husky_state["finished"] or husky_state["blocked"]) and husky_hold_pose is None:
                    # Capture the pose at the exact instant it stops - before it can
                    # fall even one frame - and hold it there from the NEXT frame's
                    # pin block onward (see hold_in_place above).
                    husky_hold_pose = [float(pos[0][0]), float(pos[0][1]), float(pos[0][2])]
                if frame % TRAIL_SAMPLE_EVERY_N_FRAMES == 0:
                    husky_trail_points.append(
                        Gf.Vec3f(float(pos[0][0]), float(pos[0][1]), floor_top_z + TRAIL_HEIGHT_ABOVE_FLOOR_M)
                    )
                    _update_trail(husky_trail, husky_trail_points)

            # --- clearance ring: follows the VISIBLE robot every frame (settle window
            # and held/finished frames included), never the parked/pinned one (T15i,
            # proxy_visibility_schedule). ---
            sched = proxy_visibility_schedule(frame, burger_start_frame, husky_start_frame)
            if sched["burger"]["ring"]:
                _bp, _ = burger_rigid.get_world_poses()
                _update_ring(burger_ring, Gf.Vec3d(float(_bp[0][0]), float(_bp[0][1]), floor_top_z + RING_HEIGHT_ABOVE_FLOOR_M))
            if sched["husky"]["ring"]:
                _hp, _ = husky_rigid.get_world_poses()
                _update_ring(husky_ring, Gf.Vec3d(float(_hp[0][0]), float(_hp[0][1]), floor_top_z + RING_HEIGHT_ABOVE_FLOOR_M))

            # --- trajectory log (both robots, every frame - cheap, small floats) ---
            _bpos, _ = burger_rigid.get_world_poses()
            _hpos, _ = husky_rigid.get_world_poses()
            burger_trajectory.append([frame, float(_bpos[0][0]), float(_bpos[0][1]), float(_bpos[0][2])])
            husky_trajectory.append([frame, float(_hpos[0][0]), float(_hpos[0][1]), float(_hpos[0][2])])

            # --- camera: wide shot through both drive passes, then a smootherstep
            # transition into the fixed sphere-drop/static shot. The wide shot's
            # asymmetric aperture (see above) is specific to that shot - blended back
            # to a plain centered 36x24 sensor over the same transition as everything
            # else, not just snapped, so the "sensor size" doesn't visibly pop.
            if frame < drop_start_frame:
                if husky_eye is None or frame < husky_start_frame - transition_frames:
                    eye, target = wide_eye, wide_center
                elif frame < husky_start_frame:
                    # two-pose mode: smootherstep from the 3/4 pose to the Husky-pass pose over
                    # the last CAMERA_TRANSITION_S of Burger's pass
                    t = _smootherstep((frame - (husky_start_frame - transition_frames)) / transition_frames)
                    eye = Gf.Vec3d(*(_lerp(wide_eye[i], husky_eye[i], t) for i in range(3)))
                    target = Gf.Vec3d(*(_lerp(wide_center[i], husky_center[i], t) for i in range(3)))
                else:
                    eye, target = husky_eye, husky_center
                set_camera(eye, target)
                focal_length_attr.Set(WIDE_REFERENCE_FOCAL_MM)
                h_aperture_attr.Set(wide_h_aperture)
                v_aperture_attr.Set(wide_v_aperture)
                v_aperture_offset_attr.Set(wide_v_aperture_offset)
            else:
                t = _smootherstep((frame - drop_start_frame) / transition_frames)
                eye = Gf.Vec3d(*(_lerp(last_wide_eye[i], drop_eye[i], t) for i in range(3)))
                target = Gf.Vec3d(*(_lerp(last_wide_center[i], drop_target[i], t) for i in range(3)))
                set_camera(eye, target)
                focal_length_attr.Set(_lerp(WIDE_REFERENCE_FOCAL_MM, CLOSE_FOCAL_MM, t))
                h_aperture_attr.Set(_lerp(wide_h_aperture, CLOSE_H_APERTURE_MM, t))
                v_aperture_attr.Set(_lerp(wide_v_aperture, CLOSE_V_APERTURE_MM, t))
                v_aperture_offset_attr.Set(_lerp(wide_v_aperture_offset, 0.0, t))

            # --- addendum: camera clearance on EVERY frame - static prims + both robot
            # proxies at their current pose + the sphere. First violation fails the run.
            dynamic_ranges = robot_footprint_ranges([
                {"name": "burger", "xy": [float(_bpos[0][0]), float(_bpos[0][1])], "half_diagonal_m": math.hypot(burger_spec["length_m"], burger_spec["width_m"]) / 2.0,
                 "height_m": float(burger_spec["height_m"])},
                {"name": "husky", "xy": [float(_hpos[0][0]), float(_hpos[0][1])], "half_diagonal_m": math.hypot(husky_spec["length_m"], husky_spec["width_m"]) / 2.0,
                 "height_m": float(husky_spec["height_m"])},
            ], floor_top_z)
            # the robots' AABB z from their actual body centre (a parked robot sits 20 m away and 2 m up)
            dynamic_ranges[0]["min"] = (dynamic_ranges[0]["min"][0], dynamic_ranges[0]["min"][1], float(_bpos[0][2]) - burger_spec["height_m"] / 2.0)
            dynamic_ranges[0]["max"] = (dynamic_ranges[0]["max"][0], dynamic_ranges[0]["max"][1], float(_bpos[0][2]) + burger_spec["height_m"] / 2.0)
            dynamic_ranges[1]["min"] = (dynamic_ranges[1]["min"][0], dynamic_ranges[1]["min"][1], float(_hpos[0][2]) - husky_spec["height_m"] / 2.0)
            dynamic_ranges[1]["max"] = (dynamic_ranges[1]["max"][0], dynamic_ranges[1]["max"][1], float(_hpos[0][2]) + husky_spec["height_m"] / 2.0)
            if sphere_rigid is not None:
                _spos, _ = sphere_rigid.get_world_poses()
                sx_, sy_, sz_ = float(_spos[0][0]), float(_spos[0][1]), float(_spos[0][2])
                dynamic_ranges.append({"prim": "/World/DropSphere", "min": (sx_ - SPHERE_DROP_RADIUS_M, sy_ - SPHERE_DROP_RADIUS_M, sz_ - SPHERE_DROP_RADIUS_M),
                                       "max": (sx_ + SPHERE_DROP_RADIUS_M, sy_ + SPHERE_DROP_RADIUS_M, sz_ + SPHERE_DROP_RADIUS_M)})
            eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
            violation = visibility.camera_clearance_violation(eye_t, clearance_ranges + dynamic_ranges, visibility.CAMERA_MIN_CLEARANCE_M)
            if violation is not None:
                phase = "wide" if frame < drop_start_frame else ("transition" if frame < drop_start_frame + transition_frames else ("drop" if frame < drop_start_frame + drop_frames else "static"))
                raise CameraClearanceViolation({**violation, "frame": frame, "phase": phase, "camera": list(eye_t), "look_at": [float(v) for v in target]})

            world.step(render=True)
            rep.orchestrator.step(rt_subframes=1)

        rep.orchestrator.wait_until_complete()

        burger_final_pos, _ = burger_rigid.get_world_poses()
        husky_final_pos, _ = husky_rigid.get_world_poses()
        def _robot_result(state: dict, final_pos) -> dict:
            # T15h: distance to /World/Target (= the route goal) measured at the pose
            # the robot was left in - for Burger that is where it stopped before being
            # parked, captured below at its park time - so the acceptance "finished
            # within TARGET_REACH_TOLERANCE_M of the target" is a number in this file.
            fx, fy = float(final_pos[0]), float(final_pos[1])
            dist = math.hypot(fx - float(target_isaac[0]), fy - float(target_isaac[1]))
            return {
                "finished": state["finished"], "blocked": state["blocked"], "waypoints_reached": state["wp_idx"],
                "final_pos": [float(x) for x in final_pos], "dist_to_target_m": dist,
                "reached_target": bool(state["finished"] and dist <= TARGET_REACH_TOLERANCE_M),
                "target_reach_tolerance_m": TARGET_REACH_TOLERANCE_M,
            }

        result["burger"] = _robot_result(burger_state, burger_stop_pose if burger_stop_pose is not None else burger_final_pos[0])
        result["burger"]["parked_final_pos"] = [float(x) for x in burger_final_pos[0]]
        result["husky"] = _robot_result(husky_state, husky_final_pos[0])
        if sphere_rigid is not None:
            sphere_final_pos, _ = sphere_rigid.get_world_poses()
            final_z = float(sphere_final_pos[0][2])
            result["sphere_drop"] = {
                "final_pos": [float(x) for x in sphere_final_pos[0]],
                "drop_xy": list(drop_xy),
                "rest_height_above_floor_m": final_z - floor_top_z,
                # same rest test as tools/isaac_validate.py's sphere_drop: |z - (floor + r)| < 0.02
                "settled_on_floor": abs(final_z - (floor_top_z + SPHERE_DROP_RADIUS_M)) < 0.02,
                "below_floor": final_z < floor_top_z - SPHERE_DROP_RADIUS_M,
            }
        result["frames_rendered"] = total_frames
        result["frames_dir"] = str(frames_dir)
        result["ring_visibility"] = {
            "rule": "proxy_visibility_schedule: Burger body+ring f0..husky_start-1, Husky body+ring husky_start..end; trails hidden until >= 2 points",
            "switches": [{"frame": 0, "burger_ring": True, "husky_ring": False}] + ring_visibility_log,
        }

        trajectory_path = spec_path.with_name(spec_path.stem + ".trajectory.json")
        trajectory_path.write_text(json.dumps(
            {
                "columns": ["frame", "x", "y", "z"],
                "floor_top_z": floor_top_z,
                "burger": burger_trajectory,
                "husky": husky_trajectory,
            },
            default=str,
        ))
        result["trajectory_path"] = str(trajectory_path)

        t_ffmpeg_start = time.time()
        _assemble_mp4(
            frames_dir, Path(spec["output_mp4"]), fps, spec["caption"],
            class_colors=spec.get("class_colors") or {}, total_frames=total_frames,
        )
        result["ffmpeg_s"] = time.time() - t_ffmpeg_start
        result["status"] = "ok"

    except CameraClearanceViolation as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
        result["camera_clearance_violation"] = exc.violation
    except Exception as exc:
        import traceback

        result["status"] = "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result_path = spec_path.with_suffix(".result.json")
        result_path.write_text(json.dumps(result, indent=2, default=str))
        try:
            sim_app.close()
        except Exception:
            pass

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "ok" else 1


# Legend rows are capped - a 1080p corner can't usefully show more than this many, and
# past this point it stops being a quick-reference legend.
MAX_LEGEND_ROWS = 10


def _build_legend_filters(class_colors: dict, base_y: int, font: str) -> list[str]:
    items = sorted(class_colors.items())[:MAX_LEGEND_ROWS]
    filters = []
    for i, (name, rgb) in enumerate(items):
        r, g, b = (int(c) for c in rgb)
        hexcolor = f"0x{r:02x}{g:02x}{b:02x}"
        safe_name = name.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:")
        y = base_y + i * 22
        filters.append(
            f"drawtext=fontfile={font}:text='{safe_name}':fontsize=18:fontcolor={hexcolor}:"
            f"box=1:boxcolor=black@0.45:boxborderw=4:x=20:y={y}"
        )
    return filters


def _assemble_mp4(
    frames_dir: Path, output_mp4: Path, fps: int, caption: str, *, class_colors: dict, total_frames: int
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf_filters = []
    font = _find_font()
    caption_escaped = caption.replace("\\", "\\\\").replace("'", "’")
    if font:
        vf_filters.append(
            f"drawtext=fontfile={font}:text='{caption_escaped}':fontsize=28:fontcolor=white:"
            f"box=1:boxcolor=black@0.55:boxborderw=10:x=(w-text_w)/2:y=h-70"
        )
        # %{n}: ffmpeg drawtext's own live output-frame-number expansion - a real
        # per-frame counter, not a static string, despite this filter chain being
        # built once ahead of time.
        vf_filters.append(
            f"drawtext=fontfile={font}:text='Isaac Sim, headless · kinematic drive':"
            f"fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=20:y=20"
        )
        vf_filters += _build_legend_filters(class_colors, base_y=56, font=font)
    else:
        print("[record_isaac] no bundled font found for ffmpeg drawtext - shipping without burned-in text")

    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "rgb_%04d.png")]
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(output_mp4)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.isaac_worker_spec is not None:
        return run_isaac_worker(args.isaac_worker_spec)
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
