"""Monte-Carlo fit-probability audit for a bootstrapped scene (fit-probability
feature, see the ops task that added this module).

For every requested robot platform (`app/robots.py`), runs `--trials` (default
100) independent A* attempts from the scene's fixed robot start to its route
goal, each against a FRESH, randomly jittered version of the scene's geometry:

  - every object footprint is translated by `N(0, OBJECT_TRANSLATE_SIGMA_M)`
    (independent draws for x and z) and scaled by
    `U(OBJECT_SCALE_LOW, OBJECT_SCALE_HIGH)` about its own centre - one scale
    factor per object per trial, applied to both `size_uv` axes (isotropic:
    the spec says "scaled by U(...)", singular, and this is the simplest
    reading of "measured run-to-run object [size] variance" - an object's
    overall size estimate wobbling trial to trial, not its aspect ratio);
  - every wall polygon vertex is perturbed independently by
    `N(0, WALL_VERTEX_SIGMA_M)` (again independent x/z draws per vertex, not
    a single per-wall shift) - see `_jitter_wall_vertices`.

GOAL RESOLUTION - per platform, per trial, not a fixed point: the goal for a
`"goto <object>"`-style command is resolved by the SHARED
`app.services.goal_resolution.resolve_goal_cell` rule (see that module's own
docstring) - the nearest traversable cell to the target object's own footprint
EDGE (not its centroid) whose clearance - distance to every other object, this
trial's own (jittered) target object, every wall, and the room boundary alike
- is at least `radius_m + goal_resolution.CLEARANCE_MARGIN_M`, searched within
`goal_resolution.SEARCH_RADIUS_M` of the target footprint. That module is also
what `app.services.pathfinding.resolve_goal_cell` (the runtime `/reachability`
and `POST /command` planner) now uses - see ITS docstring for the exact three-
way diff against what it used to do (centroid anchor, 1.0m ring, no margin) -
so this is, as of this fix, the SAME rule both places use; this module is no
longer the odd one out.

FIT CRITERIA - a trial counts as `fits=True` only if BOTH hold (this is
stricter than "A* found a path", which used to be the whole criterion):

  (a) the shared `app.services.goal_resolution.resolve_goal_cell` rule finds a
      clearance-respecting cell (its own default `CLEARANCE_MARGIN_M`, 0.10m -
      unchanged, the same bar the runtime planner uses) searching around the
      ENTIRE target footprint - `target_dist_m` is a true whole-polygon
      distance, shapely's own point-to-polygon `.distance()` over every
      edge/vertex, not a single anchor point - within
      `goal_resolution.SEARCH_RADIUS_M` (2.0m); AND that resolved cell (the
      nearest clearance-respecting one to the footprint, whichever it is) is
      itself within `radius_m + GOAL_PRIMARY_PROXIMITY_MARGIN_M` (0.15m) of
      the footprint's own edge to count as "reached". If a clearance-
      respecting cell was found but it's farther than that (a room where the
      only clear spot near the object is out past 0.15m), this module accepts
      it anyway PROVIDED it's within the looser
      `radius_m + GOAL_FALLBACK_PROXIMITY_MARGIN_M` (0.30m) - and tags the
      trial `used_fallback_goal_margin=True` when it does (still counts
      toward `fits_count` if criterion (b) also holds, but the fallback rate
      is reported per platform - see `PlatformFitResult.fallback_trials` - so
      a platform leaning on the loosened bar to hit its numbers is visible,
      not hidden inside a single pass/fail count). `REASON_GOAL_UNRESOLVED`
      covers both failure shapes: no clearance-respecting cell anywhere
      within the search radius, or one exists but even the nearest is farther
      than the fallback proximity bound;
  (b) A* finds a path from start to that goal AND the minimum corridor width
      along the ACTUAL PATH TAKEN (not a relaxed/best-effort one) is at least
      the platform's full WIDTH (`2 * radius_m`, the diameter - not the
      radius): `app.services.pathfinding.inflate`'s cell-quantized dilation
      (`ceil(radius_m / resolution)`) can occasionally let A* through a gap
      that, measured continuously via this module's own `scipy.ndimage`
      distance transform, is a hair under the platform's true diameter - see
      `_evaluate_trial`'s "path_too_narrow" branch. This reconciles the
      discrete inflate()-based A* result against a continuous verification.

`fits_count` is how many of the `trials` attempts satisfy both. The p5/p50/p95
CORRIDOR WIDTH DISTRIBUTION is computed ONLY from the min-corridor-width of
trials that actually fit (not the pseudo-widths of blocked/rejected trials -
see the next paragraph for why that used to be wrong), so it describes the
paths that actually succeeded, per platform, as reported.

BUG FIXED (p5=0.00 with fits>0): earlier versions of this module computed
p5/p50/p95 over EVERY trial's min-corridor-width, including blocked ones -
whose "width" came from a relaxed best-effort path that deliberately crosses
obstacles (see `_blocked_outcome`) and can be ~0. Mixing those into the same
population as real successful-path widths meant a platform with even a modest
blocked fraction could show `p5=0.00` right next to a large `fits_count` -
correct as "the 5th percentile of ALL 100 attempts, blocked ones included" but
misleading as a description of what a *successful* run looks like, which is
what a reader wants from "the p5 corridor width". Fixed by restricting the
percentile population to `fits=True` trials only (`None` when there are none).

For every trial (whether it fits or not) this also measures the MINIMUM
physical corridor width encountered along whatever path/attempt was actually
used to decide the outcome: `2 * distance to the nearest obstacle cell`, from
a `scipy.ndimage` distance transform over that trial's own (jittered, but NOT
robot-radius-inflated) obstacle mask - a property of the physical world, not
of the robot. Each of the 100 jittered worlds is generated ONCE per run,
shared across every platform, not regenerated per platform (see
`_generate_trial_worlds`) - that's what makes cross-platform comparison on the
same seed meaningful, and keeps the RNG stream length independent of
which/how many platforms were requested.

COORDINATE FRAME - read this before changing anything here: `--scene-dir`'s
`occupancy.npy`/`occupancy_meta.json` are in the RAW (pre yaw-correction)
capture frame (see `gpu/stage_occupancy.py`). `--bootstrap-out`'s
`objects.json`/`objects_audited.json`/`scene_meta.json`/`path.json` are in
the YAW-CORRECTED frame `scripts/msa/bootstrap.run_bootstrap` rotates walls,
room_polygon, objects, and the recorded start/goal into (see
`scene_meta.json`'s `yaw_correction_rad`/`yaw_applied`). Those two frames are
NOT the same coordinate system once a real yaw correction has been applied
(the hero scene's is ~50.5 degrees) - reusing the raw grid's `origin_x`/
`origin_z`/`width`/`height` directly against the rotated geometry would
silently place every object and wall in the wrong spot (exactly the class of
coordinate-frame bug `app/services/pathfinding.py`'s own module docstring
warns this project has hit repeatedly). So this module takes ONLY
`resolution` from `--scene-dir`'s `occupancy_meta.json` (a legitimate "grid
convention" to reuse - the cell size) and derives its own `origin_x`/
`origin_z`/`width`/`height` from the bootstrapped (rotated) room_polygon's
own bounding box (`_derive_grid_meta`) - a grid that actually matches the
frame every other input here (objects, walls, start, goal) is expressed in.

PLANNER GRID (default, `--grid-source planner`) - this module now runs on the
PLANNER's own grid, not a synthetic one built from the room-polygon bbox: the
scene-dir's real `occupancy.npy` (raw, pre-yaw-correction frame - see COORDINATE
FRAME above), rotated into the exported frame via `scene_meta.json`'s
`yaw_correction_rad`/`yaw_rotation_center_xy` - the SAME transform
`scripts/msa/textures.py` applies to the point cloud (verified: same CCW
`rotate_point_xz` convention, checked directly against that module) - with every
object's own bbox then layered on top as an extra OBSTACLE source via the SAME
`app.services.pathfinding.rasterize_footprints_as_obstacles` the runtime planner
itself now uses (see `_generate_planner_trial_worlds`). Rotating a RASTER grid is
done by inverse-warp nearest-neighbor resampling (`_build_planner_base_grid`):
iterate over the OUTPUT (exported-frame) grid's cells, inverse-rotate each one's
centre back into the raw frame, and sample the nearest raw cell - the standard,
hole-free way to rotate a raster (a forward warp - iterating the input and
scattering to the output - leaves gaps). Output cells whose inverse-rotated centre
falls outside the raw grid's own scanned bounds are `UNKNOWN` (never `OBSTACLE`
by default here - genuinely unscanned floor, not "outside the room"), so
`grid_extent_covers_room_polygon_pct` is now a REAL measurement of how much of
room_polygon's own area the raw sensor scan actually reached, not the old
synthetic-grid mode's 100%-by-construction figure.

Per-object footprints use each object's `hull_xz`'s own axis-aligned bounding box
(`_bbox_from_hull`), not the oriented-rectangle `center_xy`/`size_uv`/`angle_rad`
synthetic-mode uses - matching `app.services.pathfinding.rasterize_footprints_as_
obstacles`'s own bbox convention (and, in turn, `SceneObject.bbox_min_x/z`/
`bbox_max_x/z`'s own real-point-cloud-derived axis-aligned extent) exactly, per
platform-grid mode's whole point: run the SAME shapes and the SAME rules the
runtime planner uses. The jitter itself (object translate/scale) is unchanged;
`WALL_VERTEX_SIGMA_M` jitter does NOT apply in planner-grid mode - there is no
separate "wall polygon" abstraction to perturb here; the room boundary is fixed
per trial (see `_clip_to_room_polygon` immediately below).

ROOM-POLYGON CLIP (`_clip_to_room_polygon`) - the raw occupancy grid's own OBSTACLE
cells are NOT, by themselves, a complete obstacle set: checked directly at the
hero's west doorway-spur notch, the rotated raw grid has no OBSTACLE cell there at
all, only a FREE-to-UNKNOWN transition - the sensor's mapped free space simply ENDS
there, it never classified that boundary as an obstacle. `UNKNOWN` is traversable-
at-a-penalty by design (real unscanned floor - see `build_cost_grid`), which is
right for genuinely unscanned floor INSIDE the room but wrong for space outside it
entirely, where a wall physically exists (and is a real, solid collider in Isaac's
simulated geometry) whether or not the sensor's OBSTACLE classifier happened to
fire on it. So: every cell outside `room_polygon` is forced OBSTACLE - matching
`scripts/msa/export_presentation.py`'s own `clip_grid_to_room` (T15h) - regardless
of what the raw grid sampled there; cells INSIDE `room_polygon` keep the raw grid's
own FREE/OBSTACLE/UNKNOWN classification unchanged (UNKNOWN still traversable-at-
a-penalty there).

THREE NOTIONS OF TRAVERSABILITY (a real, standalone architectural finding, not
just this module's own bug) - the SAME question "is this cell traversable" now
gets three DIFFERENT answers depending which consumer you ask, because each reads
a different obstacle set: (1) raw-grid-OBSTACLE-cells-only - what
`app.services.pathfinding.py` (the LIVE `/reachability`/`POST /command` planner)
still uses today: no room_polygon clip at all, so it is exposed to exactly the
same doorway/boundary blind spot this module had before this fix; (2) polygon-
clipped - `scripts/msa/export_presentation.py`'s own `clip_grid_to_room` (T15h,
the demo/presentation route) and now this module's `--grid-source planner`
(default) too; (3) collider geometry - Isaac Sim's own simulated walls/objects
(extruded from the room's wall band), the actual physical arbiter two real GPU
runs (T15e run4/run5) already ruled by: Husky physically stopped by exported
geometry, matching (2)'s clipped model, not (1)'s raw-obstacles-only one. This
module now agrees with (2)/(3); the live runtime planner (1) is still exposed to
the gap this task's report documents - flagged here, not fixed (out of this
task's scope - `app/services/pathfinding.py` already has one authorized change
this task chain made to it; extending the room-polygon clip there is a separate
decision).

`--grid-source synthetic` keeps the OLD room-polygon-bbox-and-rasterized-vector-
geometry grid (walls AND objects rasterized fresh, jittered, from
objects.json/objects_audited.json, `grid_source: "synthetic_from_footprints"`,
100%-by-construction coverage) available - fast and hermetic for this module's own
unit tests (no real occupancy.npy fixture needed), but no longer the default and
not what a production run should use.

CLI:
    python -m scripts.audit.fit_prob --bootstrap-out <dir> --scene-dir <raw scene dir> \\
        [--grid-source planner|synthetic] [--platforms all] [--trials 100] [--seed 0] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage
from shapely.geometry import Point, Polygon, box

from app.robots import RobotPlatform, list_robots, resolve_radius_m
from app.services import goal_resolution
from app.services import pathfinding as pf
from app.services.pathfinding import (
    FREE,
    OBSTACLE,
    UNKNOWN,
    NoPathError,
    ObjectFootprint,
    astar,
    build_cost_grid,
    cell_to_world,
    inflate,
    snap_if_blocked,
    world_to_cell,
)
from app.services.scene_ingest import GridMeta
from scripts.msa.bootstrap import rasterize_object_footprint
from scripts.msa.objects_io import BootstrapObjects, WallPolygon, load_objects_json

logger = logging.getLogger(__name__)

# --- Jitter parameters (spec-fixed - see module docstring; NOT CLI-tunable on
# purpose, so a bad acceptance number can never be fixed by quietly narrowing these). ---
OBJECT_TRANSLATE_SIGMA_M = 0.05
OBJECT_SCALE_LOW = 0.93
OBJECT_SCALE_HIGH = 1.21
WALL_VERTEX_SIGMA_M = 0.08

# Margin added around the room_polygon/object bounding box when deriving this
# module's own grid frame (see the module docstring's COORDINATE FRAME section).
GRID_MARGIN_M = 0.5

# Start/goal cells are snapped to the nearest traversable cell within this radius
# before A* runs (same mechanism as app.services.pathfinding.plan_path, just a much
# smaller radius: this is meant to absorb grid-quantization slop right at a
# provided world point, not to rescue a start/goal a jittered trial has genuinely
# swallowed - that should count as blocked).
SNAP_RADIUS_M = 0.15

# A "blocked" trial still gets *a* route (see _blocked_outcome) by letting A* cross
# obstacle cells at a huge but finite cost - this is that cost. Large enough that
# any real path is always preferred, small enough to stay well under float64 inf
# arithmetic edge cases.
RELAXED_OBSTACLE_COST = 1.0e6

# _label_block_point thresholds: how close (metres) a block point must be to the
# scene's start / an object's footprint to be attributed to it, before falling back
# to the generic "room boundary" label.
LABEL_START_RADIUS_M = 0.3
LABEL_OBJECT_RADIUS_M = 0.6

# Goal resolution MECHANISM (whole-footprint search, clearance bar, search radius)
# lives entirely in app.services.goal_resolution, called here at its own default
# CLEARANCE_MARGIN_M (0.10m - the same bar the runtime planner uses, unchanged) - see
# the module docstring's GOAL RESOLUTION section. This module supplies its OWN,
# ADDITIONAL, separate PROXIMITY policy on top of the single cell that mechanism
# resolves (see FIT CRITERIA in the module docstring): is the resolved cell close
# enough to the target's own edge to count as "reached"? Try the tighter PRIMARY bar
# first; if the resolved cell is farther than that (but still within
# goal_resolution.SEARCH_RADIUS_M, i.e. a clearance-respecting cell does exist,
# just not a close one), accept it anyway at the looser FALLBACK bar, tagging the
# trial so the fallback rate is reported, not hidden - see
# PlatformFitResult.fallback_trials.
GOAL_PRIMARY_PROXIMITY_MARGIN_M = 0.15
GOAL_FALLBACK_PROXIMITY_MARGIN_M = 0.30


# --- Grid frame + geometry rasterization ---------------------------------------------


def _cell_center_points(meta: GridMeta) -> np.ndarray:
    """(width*height, 2) array of every cell's world-space centre, in the same
    [ix, iz] row-major order as `cells.reshape((meta.width, meta.height))` - see
    `_rasterize_polygon_mask`."""
    xs = meta.origin_x + (np.arange(meta.width) + 0.5) * meta.resolution
    zs = meta.origin_z + (np.arange(meta.height) + 0.5) * meta.resolution
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")  # (width, height) each
    return np.column_stack([grid_x.ravel(), grid_z.ravel()])


def _rasterize_polygon_mask(
    vertices_xz: list[tuple[float, float]], cell_points: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Boolean mask (shape `(width, height)`, `[ix, iz]`) of every cell centre inside
    `vertices_xz` (auto-closed by `matplotlib.path.Path`). Used for walls (arbitrary
    jittered polygons - `rasterize_object_footprint` only handles oriented
    rectangles) and the room polygon; object footprints use
    `scripts.msa.bootstrap.rasterize_object_footprint` instead (vectorized and
    already exactly the convention `gaps.json`/room-polygon extraction use)."""
    if len(vertices_xz) < 3:
        return np.zeros(shape, dtype=bool)
    inside = MplPath(vertices_xz).contains_points(cell_points)
    return inside.reshape(shape)


def _derive_grid_meta(
    room_polygon: list[tuple[float, float]], objects: list[dict], resolution: float, margin_m: float = GRID_MARGIN_M
) -> GridMeta:
    """This module's own grid frame - see the module docstring's COORDINATE FRAME
    section for why it is NOT the raw scene-dir's occupancy grid frame. Sized to the
    bounding box of the (already yaw-corrected) room_polygon, padded to also cover
    every object's jitter-scaled worst case (its footprint's own half-diagonal, a
    generous over-estimate since the actual jitter/scale is much smaller) plus
    `margin_m`, rounded outward to whole cells."""
    xs = [x for x, _ in room_polygon]
    zs = [z for _, z in room_polygon]
    for obj in objects:
        cx, cz = obj["center_xy"]
        length_u, width_v = obj["size_uv"]
        half_diag = math.hypot(length_u, width_v) / 2.0
        xs += [cx - half_diag, cx + half_diag]
        zs += [cz - half_diag, cz + half_diag]

    min_x, max_x = min(xs) - margin_m, max(xs) + margin_m
    min_z, max_z = min(zs) - margin_m, max(zs) + margin_m
    width = max(1, int(math.ceil((max_x - min_x) / resolution)))
    height = max(1, int(math.ceil((max_z - min_z) / resolution)))
    return GridMeta(resolution=resolution, origin_x=min_x, origin_z=min_z, width=width, height=height)


# --- Per-trial jitter + grid construction ---------------------------------------------


def _jitter_object(obj: dict, rng: np.random.Generator) -> dict:
    """One trial's jittered footprint for `obj` - translate the centre (independent
    x/z draws) and scale `size_uv` isotropically about that (already-translated)
    centre, per the module docstring. Returns only the three fields
    `rasterize_object_footprint` needs."""
    cx, cz = obj["center_xy"]
    dx = rng.normal(0.0, OBJECT_TRANSLATE_SIGMA_M)
    dz = rng.normal(0.0, OBJECT_TRANSLATE_SIGMA_M)
    scale = rng.uniform(OBJECT_SCALE_LOW, OBJECT_SCALE_HIGH)
    length_u, width_v = obj["size_uv"]
    return {
        "center_xy": (cx + dx, cz + dz),
        "size_uv": (length_u * scale, width_v * scale),
        "angle_rad": obj["angle_rad"],
    }


def _jitter_wall_vertices(wall: WallPolygon, rng: np.random.Generator) -> list[tuple[float, float]]:
    """One trial's jittered ring for `wall` - every vertex perturbed independently
    (its own x/z draws), not a single shift applied to the whole wall."""
    return [(x + rng.normal(0.0, WALL_VERTEX_SIGMA_M), z + rng.normal(0.0, WALL_VERTEX_SIGMA_M)) for x, z in wall.vertices]


def _build_trial_cells(
    meta: GridMeta,
    room_mask: np.ndarray,
    objects: list[dict],
    walls: list[WallPolygon],
    cell_points: np.ndarray,
    rng: np.random.Generator,
    *,
    target_id: str,
) -> tuple[np.ndarray, dict]:
    """One trial's tri-state occupancy grid (`FREE`/`OBSTACLE`/`UNKNOWN`, same
    convention `app.services.pathfinding` expects): `room_mask` (fixed - the room
    polygon itself is not jittered, only what's inside it) as the FREE base, then
    this trial's jittered walls and objects stamped in as OBSTACLE on top
    (walls/objects always win over FREE, matching how a real occupancy grid would
    classify a cell actually covered by furniture or a wall).

    Outside `room_mask` defaults to OBSTACLE, not UNKNOWN: `UNKNOWN` in
    `app.services.pathfinding.build_cost_grid` is traversable at a penalty (it means
    "genuinely unscanned floor" in a real occupancy grid), which is the wrong
    semantics here - `_derive_grid_meta`'s margin outside `room_polygon` is not
    unscanned floor, it is space known NOT to be part of the room, and treating it
    as passable would let a trial's A* silently route AROUND a wall or object that
    happens to sit near the room boundary by cutting through that margin instead of
    actually threading the gap - defeating the entire point of a fit-probability
    audit. (Caught by this module's own narrow-corridor test: a wall block that
    doesn't quite reach the room's mapped edge was "passable" via the margin until
    this was OBSTACLE-by-default.)

    Draw order (fixed, so a given seed reproduces byte-identical jitter regardless
    of caller): all objects in `objects`' own order, then all walls in `walls`' own
    order, each wall's vertices in order - see the module docstring.

    Also returns this trial's own jittered footprint (`center_xy`/`size_uv`/
    `angle_rad`) of the object matching `target_id` - goal resolution
    (`_resolve_goal_cell`) needs THIS trial's own realization of the target's
    footprint, not the nominal/unjittered one, to stay consistent with the rest of
    the trial's geometry (see the module docstring's GOAL RESOLUTION section).
    """
    shape = (meta.width, meta.height)
    cells = np.full(shape, OBSTACLE, dtype=np.uint8)
    cells[room_mask] = FREE

    target_jittered: dict | None = None
    for obj in objects:
        jittered = _jitter_object(obj, rng)
        mask = rasterize_object_footprint(jittered, shape, meta.resolution, meta.origin_x, meta.origin_z)
        cells[mask] = OBSTACLE
        if obj["id"] == target_id:
            target_jittered = jittered

    for wall in walls:
        verts = _jitter_wall_vertices(wall, rng)
        mask = _rasterize_polygon_mask(verts, cell_points, shape)
        cells[mask] = OBSTACLE

    if target_jittered is None:
        # run_fit_prob validates target_id against the object list up front - this
        # would mean that validation was bypassed by a direct caller, not bad scene
        # data (which is reported earlier, once, not per-trial).
        raise ValueError(f"target object id {target_id!r} not found among objects")

    return cells, target_jittered


def _distance_to_obstacle_m(cells: np.ndarray, resolution: float) -> np.ndarray:
    """For every cell, straight-line distance (metres) to the nearest OBSTACLE cell
    in `cells` - 0 at an obstacle cell itself. `2 *` this is this module's corridor-
    width proxy (the diameter of the widest obstacle-free disc centred there) -
    a property of the jittered geometry alone, independent of any robot radius."""
    obstacle = cells == OBSTACLE
    return ndimage.distance_transform_edt(~obstacle) * resolution


@dataclass
class TrialWorld:
    cells: np.ndarray
    dist_m: np.ndarray
    # This trial's own jittered target footprint (see _build_trial_cells) - kept (not
    # just its distance grid below) because app.services.goal_resolution.resolve_goal_cell
    # needs the polygon itself for its own out-of-bounds check even when a precomputed
    # target_dist_m is supplied.
    target_polygon: Polygon
    # Distance (metres) from every cell centre to `target_polygon` - shape
    # (width, height), same [ix, iz] convention as cells. Passed into
    # goal_resolution.resolve_goal_cell as `target_dist_m` so it isn't recomputed once
    # per platform (it doesn't depend on robot_radius_m) - see that function's own
    # docstring.
    target_dist_m: np.ndarray


def _generate_trial_worlds(
    meta: GridMeta,
    room_polygon: list[tuple[float, float]],
    walls: list[WallPolygon],
    objects: list[dict],
    target_id: str,
    n_trials: int,
    seed: int,
) -> list[TrialWorld]:
    """The `n_trials` jittered worlds for this run, generated ONCE (see the module
    docstring) from a single `np.random.default_rng(seed)` stream - reproducible for
    a fixed seed, and identical across every platform evaluated in the same run."""
    rng = np.random.default_rng(seed)
    cell_points = _cell_center_points(meta)
    shape = (meta.width, meta.height)
    room_mask = _rasterize_polygon_mask(room_polygon, cell_points, shape)

    worlds = []
    for _ in range(n_trials):
        cells, target_jittered = _build_trial_cells(meta, room_mask, objects, walls, cell_points, rng, target_id=target_id)
        target_poly = _footprint_polygon_shapely(target_jittered)
        target_dist_m = goal_resolution.target_distance_grid(meta, target_poly)
        worlds.append(
            TrialWorld(
                cells=cells,
                dist_m=_distance_to_obstacle_m(cells, meta.resolution),
                target_polygon=target_poly,
                target_dist_m=target_dist_m,
            )
        )
    return worlds


# --- Per-trial / per-platform evaluation ------------------------------------------


# The three (mutually exclusive) reasons a trial can end up fits=False - see the
# module docstring's FIT CRITERIA section.
REASON_GOAL_UNRESOLVED = "goal_unresolved"  # no cell anywhere around the WHOLE footprint met even the fallback bar
REASON_NO_PATH = "no_path"  # a close-enough goal existed, but A* found no route to it
REASON_PATH_TOO_NARROW = "path_too_narrow"  # A* found a route, but its min width < the platform's full diameter


@dataclass
class TrialOutcome:
    fits: bool
    min_corridor_width_m: float
    block_point: tuple[float, float] | None
    block_cell: tuple[int, int] | None
    reason: str | None  # one of the REASON_* constants above, or None iff fits
    # True iff the resolved goal cell needed GOAL_FALLBACK_PROXIMITY_MARGIN_M (0.30m)
    # because it was farther than the primary 0.15m bar from the target's own edge -
    # set regardless of whether the trial goes on to fit or fails criterion (b)
    # afterward (see PlatformFitResult.fallback_trials, which counts every trial
    # where this is True).
    used_fallback_goal_margin: bool


def _blocked_outcome(
    world: TrialWorld,
    meta: GridMeta,
    cost_grid: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    reason: str,
    used_fallback_goal_margin: bool,
) -> TrialOutcome:
    """The trial found no real path (REASON_GOAL_UNRESOLVED or REASON_NO_PATH - the
    two reasons that mean "no route to aim at was ever confirmed usable", unlike
    REASON_PATH_TOO_NARROW, which has its own real path and doesn't need this). Still
    produce a "best-effort" route toward `goal_cell` by relaxing every impassable
    (inf-cost) cell to `RELAXED_OBSTACLE_COST` instead - A* on that grid is fully
    connected (every cell finite-cost), so it always finds *some* route, biased away
    from real+inflated obstacles wherever possible but forced through them where it
    must be. The pinch point along THAT route - measured against `world.dist_m`, the
    physical (non-inflated) corridor width, not the relaxed cost - is this trial's
    block point and its min-corridor-width contribution."""
    relaxed = cost_grid.copy()
    relaxed[~np.isfinite(relaxed)] = RELAXED_OBSTACLE_COST
    try:
        path_cells = astar(relaxed, start_cell, goal_cell)
    except NoPathError:
        # Every relaxed cell is finite-cost and the grid is fully 8-connected, so
        # this should be unreachable - but one pathological trial (e.g. a start/goal
        # right at the grid edge) must not abort the whole Monte-Carlo run.
        logger.warning("relaxed A* still found no route at %s -> %s; using start cell as the block point", start_cell, goal_cell)
        path_cells = [start_cell]

    widths = [2.0 * float(world.dist_m[c]) for c in path_cells]
    min_idx = int(np.argmin(widths))
    block_cell = path_cells[min_idx]
    block_point = cell_to_world(*block_cell, meta)
    return TrialOutcome(
        fits=False,
        min_corridor_width_m=widths[min_idx],
        block_point=block_point,
        block_cell=block_cell,
        reason=reason,
        used_fallback_goal_margin=used_fallback_goal_margin,
    )


def _fallback_goal_cell(world: TrialWorld) -> tuple[int, int]:
    """Used only when goal_resolution finds nothing at either clearance bar (no cell
    anywhere around the whole footprint meets even the loosened FALLBACK margin this
    trial) - the cell nearest the target footprint regardless of clearance, purely so
    `_blocked_outcome` has somewhere to aim its relaxed "best-effort" A* attempt.
    Never used to decide fits/blocked itself - that trial is already blocked by
    construction (goal resolution itself failed)."""
    ix, iz = np.unravel_index(int(np.argmin(world.target_dist_m)), world.target_dist_m.shape)
    return int(ix), int(iz)


def _resolve_goal_with_fallback(
    world: TrialWorld, meta: GridMeta, radius_m: float
) -> tuple[tuple[int, int], bool] | None:
    """Criterion (a) - see the module docstring's FIT CRITERIA section. ONE call to
    the shared `app.services.goal_resolution.resolve_goal_cell` (its own default
    `CLEARANCE_MARGIN_M`, 0.10m - unchanged) finds the single nearest clearance-
    respecting cell to the ENTIRE target footprint (whole-polygon distance, not a
    single anchor point) within `goal_resolution.SEARCH_RADIUS_M` (2.0m) - there is
    no need to re-search at a different clearance bar, since that resolved cell IS
    the closest one that exists. This module then checks how far THAT cell is from
    the footprint's own edge: within `radius_m + GOAL_PRIMARY_PROXIMITY_MARGIN_M`
    (0.15m) -> accepted outright; farther but still within
    `radius_m + GOAL_FALLBACK_PROXIMITY_MARGIN_M` (0.30m) -> accepted, tagged as
    fallback; farther than that (or no clearance-respecting cell existed at all) ->
    `None` (`REASON_GOAL_UNRESOLVED`).

    Returns `(goal_cell, used_fallback)`, or `None`."""
    try:
        goal_cell, target_dist_m = goal_resolution.resolve_goal_cell(
            world.dist_m, meta, world.target_polygon, radius_m, target_dist_m=world.target_dist_m
        )
    except (goal_resolution.TargetOutOfBoundsError, goal_resolution.GoalUnreachableError):
        return None

    distance_to_edge_m = float(target_dist_m[goal_cell])
    if distance_to_edge_m <= radius_m + GOAL_PRIMARY_PROXIMITY_MARGIN_M + 1e-9:
        return goal_cell, False
    if distance_to_edge_m <= radius_m + GOAL_FALLBACK_PROXIMITY_MARGIN_M + 1e-9:
        return goal_cell, True
    return None


def _evaluate_trial(
    world: TrialWorld,
    meta: GridMeta,
    start_world: tuple[float, float],
    radius_m: float,
    snap_radius_cells: int,
) -> TrialOutcome:
    """Reuses `app.services.pathfinding.inflate`/`build_cost_grid`/`astar` directly
    (the same inflation + A* production `/reachability`/`goto` run on) against this
    trial's jittered grid, at `radius_m`, from `start_world` to a goal resolved via
    `_resolve_goal_with_fallback` (criterion a). See the module docstring's FIT
    CRITERIA section for the three ways this can end up `fits=False` (`REASON_*`
    constants) versus the one way it ends up `fits=True`."""
    inflated = inflate(world.cells, meta, radius_m)
    cost_grid = build_cost_grid(inflated)
    start_cell = world_to_cell(*start_world, meta)

    resolved = _resolve_goal_with_fallback(world, meta, radius_m)
    if resolved is None:
        return _blocked_outcome(
            world, meta, cost_grid, start_cell, _fallback_goal_cell(world), REASON_GOAL_UNRESOLVED, used_fallback_goal_margin=False
        )
    goal_cell, used_fallback_goal_margin = resolved

    try:
        s_cell, _ = snap_if_blocked(cost_grid, start_cell, snap_radius_cells)
        path_cells = astar(cost_grid, s_cell, goal_cell)
    except NoPathError:
        return _blocked_outcome(world, meta, cost_grid, start_cell, goal_cell, REASON_NO_PATH, used_fallback_goal_margin)

    widths = [2.0 * float(world.dist_m[c]) for c in path_cells]
    min_width = min(widths)

    # Criterion (b): the ACTUAL path's minimum corridor width must be at least the
    # platform's full diameter, not just "A* under inflate()'s cell-quantized radius
    # happened to connect" - see the module docstring for why these can disagree.
    diameter_m = 2.0 * radius_m
    if min_width < diameter_m - 1e-9:
        min_idx = int(np.argmin(widths))
        block_cell = path_cells[min_idx]
        block_point = cell_to_world(*block_cell, meta)
        return TrialOutcome(
            fits=False,
            min_corridor_width_m=min_width,
            block_point=block_point,
            block_cell=block_cell,
            reason=REASON_PATH_TOO_NARROW,
            used_fallback_goal_margin=used_fallback_goal_margin,
        )

    return TrialOutcome(
        fits=True,
        min_corridor_width_m=min_width,
        block_point=None,
        block_cell=None,
        reason=None,
        used_fallback_goal_margin=used_fallback_goal_margin,
    )


def _footprint_polygon_shapely(obj: dict) -> Polygon:
    """The object's oriented-rectangle footprint as a shapely polygon - same corner
    construction as `scripts.msa.bootstrap._footprint_polygon` (reused pattern, not
    imported: that helper is private to bootstrap.py and this module only needs the
    four corners, not bootstrap's own shapely import placement)."""
    cx, cz = obj["center_xy"]
    length_u, width_v = obj["size_uv"]
    angle = obj["angle_rad"]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u, v = length_u / 2, width_v / 2
    local_corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
    world_corners = [(cx + lx * cos_a - lz * sin_a, cz + lx * sin_a + lz * cos_a) for lx, lz in local_corners]
    return Polygon(world_corners)


def _label_block_point(point_xz: tuple[float, float], start_world: tuple[float, float], objects: list[dict]) -> str:
    """A short human label for a block point, per the task spec: "start" if it's
    close to the scene's own start point, else the nearest object's label (e.g.
    "desk corridor") if one is close enough, else the generic "room boundary" -
    nearest-wins, start checked first since a robot that cannot move AT ALL from its
    start (e.g. Husky sealed off immediately) should read as "start", not as
    whatever object happens to be nearby."""
    if math.hypot(point_xz[0] - start_world[0], point_xz[1] - start_world[1]) <= LABEL_START_RADIUS_M:
        return "start"

    point = Point(point_xz)
    best_label: str | None = None
    best_dist = math.inf
    for obj in objects:
        dist = _footprint_polygon_shapely(obj).distance(point)
        if dist < best_dist:
            best_dist, best_label = dist, str(obj["label"])

    if best_label is not None and best_dist <= LABEL_OBJECT_RADIUS_M:
        return f"{best_label} corridor"
    return "room boundary"


@dataclass
class PlatformFitResult:
    platform_id: str
    display_name: str
    radius_m: float
    trials: int
    fits_count: int
    # p5/p50/p95 of min-corridor-width ALONG THE ACTUAL PATH, computed ONLY over
    # fits=True trials (see the module docstring's "BUG FIXED" section) - None when
    # fits_count == 0 (no successful path to describe), never a misleading 0.0.
    corridor_width_p5_m: float | None
    corridor_width_p50_m: float | None
    corridor_width_p95_m: float | None
    first_block_point: dict | None  # {"x", "z", "label", "count", "blocked_trials"} or None if fits_count == trials
    # Breakdown of the (trials - fits_count) blocked trials by REASON_* - see the
    # module docstring's FIT CRITERIA section. Always sum to (trials - fits_count).
    goal_unresolved_trials: int
    no_path_trials: int
    path_too_narrow_trials: int
    # Of ALL `trials` (fit or not), how many needed GOAL_FALLBACK_PROXIMITY_MARGIN_M
    # (0.30m) because the nearest clearance-respecting cell was farther than the
    # primary 0.15m bar from the target's own edge - the fallback rate the task asked
    # to make visible per platform, not folded silently into fits_count.
    fallback_trials: int


_REASON_FIELD = {
    REASON_GOAL_UNRESOLVED: "goal_unresolved_trials",
    REASON_NO_PATH: "no_path_trials",
    REASON_PATH_TOO_NARROW: "path_too_narrow_trials",
}


def _evaluate_platform(
    platform: RobotPlatform,
    radius_m: float,
    worlds: list[TrialWorld],
    meta: GridMeta,
    start_world: tuple[float, float],
    objects: list[dict],
) -> PlatformFitResult:
    snap_radius_cells = max(1, int(math.ceil(SNAP_RADIUS_M / meta.resolution)))

    fit_widths: list[float] = []
    fits_count = 0
    fallback_trials = 0
    reason_counts: dict[str, int] = dict.fromkeys(_REASON_FIELD, 0)
    block_counts: Counter[tuple[int, int]] = Counter()
    block_points: dict[tuple[int, int], tuple[float, float]] = {}

    for world in worlds:
        outcome = _evaluate_trial(world, meta, start_world, radius_m, snap_radius_cells)
        if outcome.used_fallback_goal_margin:
            fallback_trials += 1
        if outcome.fits:
            fits_count += 1
            fit_widths.append(outcome.min_corridor_width_m)
        else:
            assert outcome.block_cell is not None and outcome.block_point is not None and outcome.reason is not None
            reason_counts[outcome.reason] += 1
            block_counts[outcome.block_cell] += 1
            block_points[outcome.block_cell] = outcome.block_point

    trials = len(worlds)
    if fit_widths:
        p5, p50, p95 = (float(v) for v in np.percentile(fit_widths, [5, 50, 95]))
    else:
        p5 = p50 = p95 = None

    first_block_point = None
    if block_counts:
        mode_cell, mode_count = block_counts.most_common(1)[0]
        mode_point = block_points[mode_cell]
        first_block_point = {
            "x": mode_point[0],
            "z": mode_point[1],
            "label": _label_block_point(mode_point, start_world, objects),
            "count": mode_count,
            "blocked_trials": trials - fits_count,
        }

    return PlatformFitResult(
        platform_id=platform.id,
        display_name=platform.display_name,
        radius_m=radius_m,
        trials=trials,
        fits_count=fits_count,
        corridor_width_p5_m=p5,
        corridor_width_p50_m=p50,
        corridor_width_p95_m=p95,
        first_block_point=first_block_point,
        goal_unresolved_trials=reason_counts[REASON_GOAL_UNRESOLVED],
        no_path_trials=reason_counts[REASON_NO_PATH],
        path_too_narrow_trials=reason_counts[REASON_PATH_TOO_NARROW],
        fallback_trials=fallback_trials,
    )


# --- Loading bootstrap-out / scene-dir inputs ---------------------------------------


def _resolve_objects_json_path(bootstrap_out: Path) -> Path:
    """Prefer the audited object list (`objects_audited.json` - the M7 de-clutter
    pass's final output, when present) over the plain bootstrap `objects.json`, same
    "most recent/most final wins" precedent as `scene_service.MSA_GLB_CANDIDATES`."""
    for name in ("objects_audited.json", "objects.json"):
        candidate = bootstrap_out / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{bootstrap_out}: neither objects_audited.json nor objects.json found")


def _load_bootstrap_objects(bootstrap_out: Path) -> tuple[BootstrapObjects, Path]:
    path = _resolve_objects_json_path(bootstrap_out)
    return load_objects_json(path), path


def _load_start(bootstrap_out: Path) -> tuple[tuple[float, float], float]:
    scene_meta = json.loads((bootstrap_out / "scene_meta.json").read_text())
    if "start_xy" not in scene_meta:
        raise ValueError(f"{bootstrap_out}/scene_meta.json has no start_xy")
    x, z = scene_meta["start_xy"]
    return (float(x), float(z)), float(scene_meta.get("start_yaw_rad", 0.0))


def _load_target_id(bootstrap_out: Path) -> str:
    """Which object this run routes to - `path.json`'s `target_id` (e.g. `"desk_1"`).
    Only the object identity is read from `path.json`; the actual goal POINT is no
    longer taken from its `target_point` (that was resolved by
    `scripts.msa.export_presentation.plan_route` for one specific platform at
    bootstrap time) - this module now resolves its own goal per platform, per trial,
    against each trial's own jittered geometry (see the module docstring's GOAL
    RESOLUTION section / `_resolve_goal_cell`)."""
    path_data = json.loads((bootstrap_out / "path.json").read_text())
    target_id = path_data.get("target_id")
    if not target_id:
        raise ValueError(f"{bootstrap_out}/path.json has no target_id")
    return str(target_id)


def _load_resolution(scene_dir: Path) -> float:
    meta = json.loads((scene_dir / "occupancy_meta.json").read_text())
    return float(meta["resolution"])


def _grid_covers_room_polygon_pct(room_polygon: list[tuple[float, float]], meta: GridMeta) -> float:
    """What fraction of `room_polygon`'s own area falls inside this module's grid
    bounding box. Computed, not assumed: `_derive_grid_meta` builds its bbox FROM
    `room_polygon` plus a positive margin, so this is 100.0 by construction every
    time - the point of computing it (rather than hardcoding 100.0) is that the
    output JSON should say so as a measured fact next to `grid_source`, not as an
    unverified claim (see run_fit_prob's output "grid" block)."""
    room = Polygon(room_polygon)
    if not room.is_valid or room.area <= 0:
        return 0.0
    grid_box = box(
        meta.origin_x, meta.origin_z, meta.origin_x + meta.width * meta.resolution, meta.origin_z + meta.height * meta.resolution
    )
    return 100.0 * room.intersection(grid_box).area / room.area


# --- Planner grid (default - see the module docstring's PLANNER GRID section) -----


def _load_raw_grid(scene_dir: Path) -> tuple[np.ndarray, GridMeta]:
    """The scene-dir's raw (pre-yaw-correction) occupancy.npy/occupancy_meta.json -
    see the module docstring's COORDINATE FRAME section."""
    cells = np.load(scene_dir / "occupancy.npy")
    meta_json = json.loads((scene_dir / "occupancy_meta.json").read_text())
    meta = GridMeta(
        resolution=float(meta_json["resolution"]),
        origin_x=float(meta_json["origin_x"]),
        origin_z=float(meta_json["origin_z"]),
        width=int(meta_json["width"]),
        height=int(meta_json["height"]),
    )
    return cells, meta


def _load_yaw_transform(bootstrap_out: Path) -> tuple[float, tuple[float, float]]:
    """`yaw_correction_rad`/`yaw_rotation_center_xy` from bootstrap-out's
    scene_meta.json - the SAME transform scripts/msa/textures.py applies to the
    point cloud when baking it into the exported (rotated) frame. `(0.0, (0.0,
    0.0))` (the identity) when the scene recorded no yaw correction - matching
    `scripts.msa.bootstrap.run_bootstrap`'s own "no correction needed" case."""
    scene_meta = json.loads((bootstrap_out / "scene_meta.json").read_text())
    angle = float(scene_meta.get("yaw_correction_rad") or 0.0)
    center_raw = scene_meta.get("yaw_rotation_center_xy") or (0.0, 0.0)
    return angle, (float(center_raw[0]), float(center_raw[1]))


def _build_planner_base_grid(
    raw_cells: np.ndarray, raw_meta: GridMeta, out_meta: GridMeta, angle_rad: float, center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Resample `raw_cells` onto `out_meta`'s grid via inverse-warp nearest-neighbor
    sampling - see the module docstring's PLANNER GRID section for why (rotating a
    raster this way, not a forward warp, avoids holes). Returns `(cells,
    covered_mask)`: `covered_mask` is True wherever the inverse-rotated centre
    landed inside the raw grid's own bounds (real sampled data); `cells` is
    `UNKNOWN` everywhere `covered_mask` is False."""
    xs = out_meta.origin_x + (np.arange(out_meta.width) + 0.5) * out_meta.resolution
    zs = out_meta.origin_z + (np.arange(out_meta.height) + 0.5) * out_meta.resolution
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    cx, cz = center
    cos_a, sin_a = math.cos(-angle_rad), math.sin(-angle_rad)
    dx, dz = grid_x - cx, grid_z - cz
    raw_x = cx + dx * cos_a - dz * sin_a
    raw_z = cz + dx * sin_a + dz * cos_a
    raw_ix = np.floor((raw_x - raw_meta.origin_x) / raw_meta.resolution).astype(np.int64)
    raw_iz = np.floor((raw_z - raw_meta.origin_z) / raw_meta.resolution).astype(np.int64)
    covered = (raw_ix >= 0) & (raw_ix < raw_meta.width) & (raw_iz >= 0) & (raw_iz < raw_meta.height)

    safe_ix = np.clip(raw_ix, 0, raw_meta.width - 1)
    safe_iz = np.clip(raw_iz, 0, raw_meta.height - 1)
    sampled = raw_cells[safe_ix, safe_iz]

    cells = np.full((out_meta.width, out_meta.height), UNKNOWN, dtype=np.uint8)
    cells[covered] = sampled[covered]
    return cells, covered


def _clip_to_room_polygon(cells: np.ndarray, meta: GridMeta, room_polygon: list[tuple[float, float]]) -> np.ndarray:
    """Cells outside `room_polygon` become OBSTACLE - THE FIX for the finding this
    module's PLANNER GRID mode surfaced: the raw occupancy grid's own OBSTACLE cells
    alone are NOT a complete obstacle set. At the hero's west doorway-spur notch,
    checked directly, the rotated raw grid has no OBSTACLE cell at all - only a
    FREE-to-UNKNOWN transition - because that boundary is where the sensor's mapped
    free space simply ENDS, not a wall it detected as an obstacle. UNKNOWN is
    traversable-at-a-penalty by design (real unscanned floor - see
    `app.services.pathfinding.build_cost_grid`), which is the right semantics for
    genuinely unscanned floor INSIDE a room but the wrong one for space outside it
    entirely: a wall physically exists there whether or not the sensor's OBSTACLE
    classifier fired on it, and it is a solid collider in Isaac's simulated geometry
    (extruded from the wall band around `room_polygon`) regardless. This matches
    `scripts/msa/export_presentation.py`'s own `clip_grid_to_room` (T15h) - same
    "outside the room polygon is not traversable" rule, applied here to the raw
    (unjittered per-trial - see the module docstring) base grid once, not per
    export-frame goal search. Cells INSIDE room_polygon keep whatever the raw grid
    said (FREE/OBSTACLE/UNKNOWN, UNKNOWN still traversable-at-a-penalty there) -
    unchanged from before this fix.

    See the module docstring's PLANNER GRID section for the broader "three notions
    of traversability" finding this is one leg of (raw-grid-obstacles-only /
    polygon-clipped / Isaac's own collider geometry)."""
    cell_points = _cell_center_points(meta)
    shape = (meta.width, meta.height)
    room_mask = _rasterize_polygon_mask(room_polygon, cell_points, shape)
    out = cells.copy()
    out[~room_mask] = OBSTACLE
    return out


def _grid_covers_room_polygon_pct_from_mask(
    room_polygon: list[tuple[float, float]], meta: GridMeta, covered_mask: np.ndarray
) -> float:
    """Planner-grid mode's REAL version of `_grid_covers_room_polygon_pct`: fraction
    of room_polygon's own area whose cells were actually sampled from the raw
    occupancy grid's own scanned extent (`covered_mask`), not defaulted to UNKNOWN
    because the inverse-rotated point fell outside the raw grid's bounds entirely -
    the real sensor-coverage gap a scene can have (see e.g.
    progress/13e_sofa_check.md's ~92.7% finding on an earlier build of this same
    hero scene's grid), unlike synthetic mode's 100%-by-construction figure."""
    cell_points = _cell_center_points(meta)
    shape = (meta.width, meta.height)
    room_mask = _rasterize_polygon_mask(room_polygon, cell_points, shape)
    room_cell_count = int(room_mask.sum())
    if room_cell_count == 0:
        return 0.0
    return 100.0 * int((room_mask & covered_mask).sum()) / room_cell_count


def _bbox_from_hull(obj: dict) -> tuple[float, float, float, float]:
    """`(bbox_min_x, bbox_min_z, bbox_max_x, bbox_max_z)` from `obj["hull_xz"]` - the
    real point-cloud convex hull's own axis-aligned extent, the MSA-side equivalent
    of `SceneObject.bbox_min_x/z`/`bbox_max_x/z` (both ultimately derived from the
    same raw point cloud) - used instead of the oriented-rectangle `center_xy`/
    `size_uv`/`angle_rad` footprint so obstacle rasterization matches
    `app.services.pathfinding.rasterize_footprints_as_obstacles`'s own bbox
    convention exactly - see the module docstring's PLANNER GRID section."""
    xs = [p[0] for p in obj["hull_xz"]]
    zs = [p[1] for p in obj["hull_xz"]]
    return min(xs), min(zs), max(xs), max(zs)


def _jitter_bbox(bbox: tuple[float, float, float, float], rng: np.random.Generator) -> tuple[float, float, float, float]:
    """One trial's jittered bbox - the same translate/scale jitter `_jitter_object`
    applies to the synthetic mode's oriented rectangle, applied here to a bbox
    instead: translate the centre (independent x/z draws) and scale both extents
    isotropically about that (already-translated) centre."""
    bmin_x, bmin_z, bmax_x, bmax_z = bbox
    cx, cz = (bmin_x + bmax_x) / 2.0, (bmin_z + bmax_z) / 2.0
    half_w, half_h = (bmax_x - bmin_x) / 2.0, (bmax_z - bmin_z) / 2.0
    dx = rng.normal(0.0, OBJECT_TRANSLATE_SIGMA_M)
    dz = rng.normal(0.0, OBJECT_TRANSLATE_SIGMA_M)
    scale = rng.uniform(OBJECT_SCALE_LOW, OBJECT_SCALE_HIGH)
    ncx, ncz = cx + dx, cz + dz
    nhw, nhh = half_w * scale, half_h * scale
    return ncx - nhw, ncz - nhh, ncx + nhw, ncz + nhh


def _generate_planner_trial_worlds(
    base_cells: np.ndarray, out_meta: GridMeta, objects: list[dict], target_id: str, n_trials: int, seed: int
) -> list[TrialWorld]:
    """The `n_trials` jittered worlds for planner-grid mode. `base_cells` (this
    scene's real occupancy grid, already rotated into the exported frame - see
    `_build_planner_base_grid`) stays FIXED across every trial - it is real sensor
    data, not a jitterable vector abstraction (no `WALL_VERTEX_SIGMA_M` jitter here -
    see the module docstring). Only each object's own bbox is jittered per trial and
    rasterized as an ADDITIONAL obstacle layer on top, via the SAME
    `app.services.pathfinding.rasterize_footprints_as_obstacles` the runtime planner
    itself now uses. Reuses the SAME `TrialWorld` dataclass synthetic mode does -
    every downstream function (`_evaluate_trial`, `_evaluate_platform`, the fit
    criteria) is generic over `TrialWorld` and needs no changes for this mode."""
    rng = np.random.default_rng(seed)
    worlds = []
    for _ in range(n_trials):
        footprints: list[ObjectFootprint] = []
        target_bbox: tuple[float, float, float, float] | None = None
        for obj in objects:
            jmin_x, jmin_z, jmax_x, jmax_z = _jitter_bbox(_bbox_from_hull(obj), rng)
            footprints.append(
                ObjectFootprint(
                    x=(jmin_x + jmax_x) / 2.0,
                    z=(jmin_z + jmax_z) / 2.0,
                    bbox_min_x=jmin_x,
                    bbox_min_z=jmin_z,
                    bbox_max_x=jmax_x,
                    bbox_max_z=jmax_z,
                )
            )
            if obj["id"] == target_id:
                target_bbox = (jmin_x, jmin_z, jmax_x, jmax_z)

        if target_bbox is None:
            # run_fit_prob validates target_id against the object list up front -
            # this would mean that validation was bypassed by a direct caller.
            raise ValueError(f"target object id {target_id!r} not found among objects")

        cells = pf.rasterize_footprints_as_obstacles(base_cells, out_meta, footprints)
        dist_m = _distance_to_obstacle_m(cells, out_meta.resolution)
        target_polygon = box(*target_bbox)
        target_dist_m = goal_resolution.target_distance_grid(out_meta, target_polygon)
        worlds.append(TrialWorld(cells=cells, dist_m=dist_m, target_polygon=target_polygon, target_dist_m=target_dist_m))
    return worlds


# --- Top-level run + CLI -----------------------------------------------------------


def run_fit_prob(
    bootstrap_out: Path,
    scene_dir: Path,
    platform_ids: list[str] | None,
    n_trials: int,
    seed: int,
    grid_source: str = "planner",
) -> dict:
    """The full audit: load inputs, generate `n_trials` jittered worlds once, then
    evaluate every requested platform (`platform_ids`, or every registered platform
    when None) against the SAME worlds. Returns the JSON-serializable result dict.

    `grid_source`: `"planner"` (default) runs on the planner's own grid - the real
    occupancy grid, rotated into the exported frame, plus every object's own bbox
    layered on top exactly as `app.services.pathfinding` now does - see the module
    docstring's PLANNER GRID section. `"synthetic"` keeps the old room-polygon-bbox
    vector-geometry grid available for this module's own fast, hermetic unit tests
    (no real occupancy.npy fixture needed) - see the module docstring's COORDINATE
    FRAME section - but is no longer the default and should not be used for a
    production run."""
    if grid_source not in ("planner", "synthetic"):
        raise ValueError(f"grid_source must be 'planner' or 'synthetic', got {grid_source!r}")
    bootstrap_out = Path(bootstrap_out)
    scene_dir = Path(scene_dir)

    bootstrap_objects, objects_json_path = _load_bootstrap_objects(bootstrap_out)
    if not bootstrap_objects.room_polygon or len(bootstrap_objects.room_polygon) < 3:
        raise ValueError(f"{objects_json_path}: no usable room_polygon")

    start_world, start_yaw_rad = _load_start(bootstrap_out)
    target_id = _load_target_id(bootstrap_out)
    if target_id not in {obj["id"] for obj in bootstrap_objects.objects}:
        raise ValueError(f"{objects_json_path}: target object {target_id!r} (from {bootstrap_out}/path.json) not found")

    if grid_source == "planner":
        raw_cells, raw_meta = _load_raw_grid(scene_dir)
        angle_rad, center = _load_yaw_transform(bootstrap_out)
        meta = _derive_grid_meta(bootstrap_objects.room_polygon, bootstrap_objects.objects, raw_meta.resolution)
        base_cells, covered_mask = _build_planner_base_grid(raw_cells, raw_meta, meta, angle_rad, center)
        # Coverage (below) is measured against the UNCLIPPED sample - it's a fact
        # about the raw sensor scan's own extent, not about this clip. THE clip
        # itself: outside room_polygon becomes OBSTACLE - see _clip_to_room_polygon's
        # own docstring for why the raw grid's OBSTACLE cells alone are not enough.
        base_cells = _clip_to_room_polygon(base_cells, meta, bootstrap_objects.room_polygon)
        worlds = _generate_planner_trial_worlds(base_cells, meta, bootstrap_objects.objects, target_id, n_trials, seed)
        grid_block = {
            "resolution": meta.resolution,
            "origin_x": meta.origin_x,
            "origin_z": meta.origin_z,
            "width": meta.width,
            "height": meta.height,
            "grid_source": "planner_grid_rotated_occupancy_clipped_to_room_plus_hulls",
            "grid_source_note": (
                "The real scene-dir occupancy.npy (raw, pre-yaw-correction frame), resampled into the "
                "exported frame via inverse-warp nearest-neighbor sampling using scene_meta.json's own "
                "yaw_correction_rad/yaw_rotation_center_xy (the same transform scripts/msa/textures.py "
                "applies to the point cloud), then CLIPPED to room_polygon (cells outside become "
                "OBSTACLE - see _clip_to_room_polygon's own docstring for why the raw grid's OBSTACLE "
                "cells alone are not a complete obstacle set, and the module docstring's THREE NOTIONS "
                "OF TRAVERSABILITY section), with every object's own bbox (from hull_xz's axis-aligned "
                "extent) then layered on top as an extra OBSTACLE source per Monte-Carlo trial via "
                "app.services.pathfinding.rasterize_footprints_as_obstacles - the SAME function the "
                "runtime planner itself now uses (note: the runtime planner does NOT clip to "
                "room_polygon - see the module docstring). grid_extent_covers_room_polygon_pct below "
                "is measured against the UNCLIPPED raw sample (a fact about the sensor scan's own "
                "coverage, unaffected by this clip) - a REAL measurement, not 100% by construction "
                "(see --grid-source synthetic for the old vector-geometry grid this replaces as the "
                "default)."
            ),
            "grid_extent_covers_room_polygon_pct": _grid_covers_room_polygon_pct_from_mask(
                bootstrap_objects.room_polygon, meta, covered_mask
            ),
            "yaw_correction_rad": angle_rad,
            "yaw_rotation_center_xy": list(center),
            "raw_occupancy_meta_json": str(scene_dir / "occupancy_meta.json"),
        }
    else:
        resolution = _load_resolution(scene_dir)
        meta = _derive_grid_meta(bootstrap_objects.room_polygon, bootstrap_objects.objects, resolution)
        worlds = _generate_trial_worlds(
            meta, bootstrap_objects.room_polygon, bootstrap_objects.walls, bootstrap_objects.objects, target_id, n_trials, seed
        )
        grid_block = {
            "resolution": meta.resolution,
            "origin_x": meta.origin_x,
            "origin_z": meta.origin_z,
            "width": meta.width,
            "height": meta.height,
            "grid_source": "synthetic_from_footprints",
            "grid_source_note": (
                "LEGACY/TEST-ONLY mode (--grid-source synthetic), not what a production run should use "
                "(see --grid-source planner, the default). This grid's obstacles come entirely from "
                "objects.json/objects_audited.json's object footprints and wall polygons (rasterized "
                "fresh per Monte-Carlo trial, see _build_trial_cells), not from the scene-dir's raw "
                "occupancy.npy - only that file's own `resolution` value is reused (see the module "
                "docstring's COORDINATE FRAME section). grid_extent_covers_room_polygon_pct below is "
                "therefore a fact about how this synthetic grid was constructed (its bbox is built FROM "
                "room_polygon, so this is 100% by construction - see _derive_grid_meta), NOT a "
                "measurement of the raw sensor grid's own coverage."
            ),
            "grid_extent_covers_room_polygon_pct": _grid_covers_room_polygon_pct(bootstrap_objects.room_polygon, meta),
        }

    all_platforms = list_robots()
    if platform_ids is not None:
        wanted = set(platform_ids)
        platforms = [p for p in all_platforms if p.id in wanted]
        missing = wanted - {p.id for p in platforms}
        if missing:
            raise ValueError(f"unknown platform id(s): {sorted(missing)}")
    else:
        platforms = all_platforms

    results: dict[str, PlatformFitResult] = {}
    for platform in platforms:
        radius_m = resolve_radius_m(platform.id, None)
        results[platform.id] = _evaluate_platform(platform, radius_m, worlds, meta, start_world, bootstrap_objects.objects)

    return {
        "schema": "cloudeye.audit.fit_prob/1",
        "seed": seed,
        "trials": n_trials,
        "grid": grid_block,
        "start_xy": list(start_world),
        "start_yaw_rad": start_yaw_rad,
        "goal_target_id": target_id,
        "goal_rule": {
            "description": (
                "Goal resolution mechanism: ONE call to app.services.goal_resolution.resolve_goal_cell "
                "(SHARED with app.services.pathfinding.resolve_goal_cell, the runtime /reachability "
                "and POST /command planner, and originally scripts.msa.export_presentation.plan_route's "
                "own rule) at its own default clearance_margin_m (0.10m, unchanged) - the nearest "
                "clearance-respecting cell to the target's own footprint, searched around the ENTIRE "
                "footprint (shapely's whole-polygon point-to-polygon distance - every edge/vertex, not "
                "a single anchor), within search_radius_m. This module then applies its OWN, separate "
                "PROXIMITY policy to that one resolved cell: within "
                "radius_m + primary_proximity_margin_m (0.15m) of the edge -> accepted; farther but "
                "still within radius_m + fallback_proximity_margin_m (0.30m) -> accepted, tagged "
                "(see PlatformFitResult.fallback_trials); farther than that (or no clearance-respecting "
                "cell existed at all) -> goal_unresolved. Separately, criterion (b): the ACTUAL A* "
                "path's minimum corridor width must be >= the platform's full diameter (2*radius_m), "
                "not merely 'A* under cell-quantized inflation connected'. See "
                "scripts/audit/fit_prob.py's module docstring FIT CRITERIA section."
            ),
            "clearance_margin_m": goal_resolution.CLEARANCE_MARGIN_M,
            "search_radius_m": goal_resolution.SEARCH_RADIUS_M,
            "primary_proximity_margin_m": GOAL_PRIMARY_PROXIMITY_MARGIN_M,
            "fallback_proximity_margin_m": GOAL_FALLBACK_PROXIMITY_MARGIN_M,
            "shared_module": "app.services.goal_resolution",
        },
        "source_paths": {
            "bootstrap_out": str(bootstrap_out),
            "scene_dir": str(scene_dir),
            "objects_json": str(objects_json_path),
            "scene_meta_json": str(bootstrap_out / "scene_meta.json"),
            "path_json": str(bootstrap_out / "path.json"),
            "occupancy_meta_json": str(scene_dir / "occupancy_meta.json"),
        },
        "platforms": {pid: asdict(result) for pid, result in results.items()},
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte-Carlo fit-probability audit: N jittered-geometry A* trials per robot platform."
    )
    parser.add_argument("--bootstrap-out", type=Path, required=True, help="scripts.msa.bootstrap output dir")
    parser.add_argument("--scene-dir", type=Path, required=True, help="raw scene dir (occupancy.npy/occupancy_meta.json)")
    parser.add_argument(
        "--platforms", default="all", help='Comma-separated app/robots.py platform ids, or "all" (default).'
    )
    parser.add_argument("--trials", type=int, default=100, help="A* trials per platform (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for jitter, recorded in the output (default 0)")
    parser.add_argument(
        "--grid-source",
        choices=("planner", "synthetic"),
        default="planner",
        help="'planner' (default): the real occupancy grid, rotated + object hulls layered on top, "
        "exactly as app.services.pathfinding now does. 'synthetic': the old room-polygon-bbox vector-"
        "geometry grid - test-only, not for production runs. See the module docstring's PLANNER GRID section.",
    )
    parser.add_argument("--out", type=Path, default=None, help="write JSON here instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    platform_ids = None if args.platforms == "all" else [s.strip() for s in args.platforms.split(",") if s.strip()]

    result = run_fit_prob(args.bootstrap_out, args.scene_dir, platform_ids, args.trials, args.seed, grid_source=args.grid_source)
    text = json.dumps(result, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
