"""T15d: MSA presentation variant of a bootstrap output - `scene_presentation.usd`
(1.2 m wall stubs for the eye, full-height invisible colliders for physics, a
DomeLight x3 + warm key light, the web 3/4 rule's camera baked in, and a planned
route `/World/Path` + `/World/Target` so `demo/record_isaac.py --usd-path-override`
can drive it), plus `path.json` and two local still renders (no Isaac Sim):
`07_presentation_34.png` (the 3/4 shot) and `07_presentation_top.png`.

    uv run python -m scripts.msa.export_presentation \
        --bootstrap-out <bootstrap out dir> --scene-dir <raw scene dir> \
        --target desk --platform burger

Inputs: the bootstrap OUTPUT dir (`scene.glb` for walls/floor/object placeholders
+ measured hulls, `scene_meta.json` for floor_y/ceiling_y/room_polygon/yaw) and
the RAW scene dir (`occupancy.npy` + `occupancy_meta.json` for planning). The
bootstrap itself is never re-run and `scripts.msa.bootstrap`/`geometry` are only
imported for `rotate_point_xz` (owned by other in-flight tasks).

Frames: the occupancy grid is in the RAW scene frame, `[ix, iz]`, while every
bootstrap output (GLB, USD, scene_meta.room_polygon) is yaw-rotated by
`yaw_correction_rad` about `yaw_rotation_center_xy` (`bootstrap._rotate_objects`
-> `geometry.rotate_point_xz`). So: object hulls are un-rotated (-yaw) into the
raw frame to rasterize/target them on the grid, the route is planned there with
`app.services.pathfinding`'s grid-level A* (no DB: `inflate` -> `build_cost_grid`
-> `astar` -> `smooth_path`), and the resulting points are rotated (+yaw) back into
the exported frame before they're written.

T15h (after the T15e GPU check): the traversable grid is clipped to the room polygon
eroded by the robot radius, the goal must keep radius + 0.10 m to every hull and to
the room boundary, `/World/Target` is the route GOAL (not the desk centroid - that is
now `/World/TargetObject`, a visual marker), and the wall band's collider is a set
of per-edge boxes (`export_usd.add_wall_edge_box_colliders`) instead of a convexHull
of the ring.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely import contains_xy
from shapely.geometry import Point, Polygon

from app.services.pathfinding import (
    FREE,
    OBSTACLE,
    NoPathError,
    astar,
    build_cost_grid,
    cell_to_world,
    inflate,
    path_length_m,
    smooth_path,
    world_to_cell,
)
from app.services.scene_ingest import GridMeta
from scripts.msa.record_isaac import (
    DEMO_PLATFORM_IDS,
    FRAMING_MARGIN_FRAC,
    PlatformSpec,
    _longest_free_run_start,
    camera_backoff_for_framing,
    framing_check_points,
    points_in_frame,
    resolve_platform,
)
from scripts.msa.export_usd import (
    PRESENTATION_WALL_HEIGHT_M,
    culled_stub_edge_ids,
    export_presentation_usd,
    iter_scene_world_meshes,
    stub_edge_boxes,
    stub_edge_occluder_box,
    stub_edge_trimesh,
    wall_stub_mesh,
)
from scripts.msa.export_glb import export_glb, path_ribbon_mesh
from scripts.msa.geometry import WALL_THICKNESS_M, convex_hull_2d, rotate_point_xz
from scripts.msa.render_perspective import (
    CAMERA_HEIGHT_M,
    DEFAULT_IMAGE_SIZE,
    FOV_DEG,
    _bed_centroid_xz,
    _object_class_from_id,
    _render_scene,
    _room_centroid,
    compute_camera,
)
from scripts.msa.visibility import (
    CAMERA_MIN_CLEARANCE_M,
    LOW_OCCLUDER_MAX_TOP_M,
    MAX_EYE_HEIGHT_ABOVE_FLOOR_M,
    OUTLINE_MARGIN_FRAC,
    PATH_VISIBILITY_MIN_FRACTION,
    aabb_box,
    aim_point_on_floor,
    centre_points_in_frame,
    is_translucent_class,
    make_box,
    projected_bbox,
    ribbon_sample_points,
    route_robot_box_sets,
    solve_visible_camera,
    visibility_check_points,
)

PATH_SCHEMA = "cloudeye.msa.presentation_path/4"  # /4 (morning-3/4): doorway-spur start, corridor report, path-ribbon visibility + eye clearance in presentation_camera
DEFAULT_TARGET = "desk"
DEFAULT_PLATFORM = "burger"
PATH_HEIGHT_ABOVE_FLOOR_M = 0.03
TARGET_HEIGHT_ABOVE_FLOOR_M = 0.05
GOAL_SEARCH_RADIUS_M = 2.0  # traversable cells this close to the target footprint are goal candidates
GOAL_CANDIDATES_MAX = 25  # tried nearest-first until A* succeeds
# T15h: the goal cell must keep this much clearance beyond the robot radius to every
# object hull AND the room boundary. T15e's goal sat 0.098 m from the desk hull (= the
# radius, by construction) and 0.128 m from the floor edge - a 0.23 m strip in which the
# validator's 0.2 m sphere wedged between the desk and the floor and fell through.
GOAL_CLEARANCE_MARGIN_M = 0.10
PATH_CLEARANCE_SAMPLE_M = 0.02  # spacing of the samples used for the min-clearance report
STILL_34_NAME = "07_presentation_34.png"
STILL_TOP_NAME = "07_presentation_top.png"
STILL_HUSKY_NAME = "07_presentation_husky.png"  # the steeper Husky-pass pose (two-pose mode only)
WORST_CASE_ROBOT_RGB = (140, 60, 10)  # the worst-case robot box in the stills (darker than the live robot)
STILL_WALL_OPACITY = 0.92  # pre-T15i translucent preview; the still now uses `opaque_walls=True`
# T15i: the still is rendered at the Isaac frame (1920x1080, 16:9) with opaque stubs so
# it IS the preview of the recording - the 4:3 translucent one hid T15e run2's problem.
STILL_IMAGE_SIZE = (1920, 1080)
TOP_DOWN_MARGIN = 1.15
ROBOT_RGB = (255, 115, 13)  # demo/record_isaac.py ROBOT_COLOR
RING_RGB = (255, 214, 0)  # the clearance ring reads yellow in the Isaac frames (T15e run2) - drawn yellow here too
PATH_RGB = (64, 217, 255)
TARGET_RGB = (255, 77, 51)
TARGET_OBJECT_RGB = (255, 191, 51)
RING_THICKNESS_M = 0.02
# T15i camera rule (see presentation_camera): the room's projected bbox must span this
# much of the frame width, the eye stays this far outside the room polygon, distance
# is adjusted in these steps before the visibility solve.
ROOM_FRAME_COVERAGE_MIN = 0.60
ROOM_FRAME_COVERAGE_MAX = 0.90
CAMERA_DISTANCE_STEP_M = 0.10
CAMERA_MIN_OUTSIDE_ROOM_M = 0.5
CAMERA_CULL_ITERATIONS = 4
# Morning-4: the presentation solve backs off at most this far (the user's "+3 m") and,
# when no height/back-off on the bed-anchored azimuth keeps the ribbon >= 95 % visible
# with a robot on the route, tries these azimuth offsets (deg, about the room centroid).
PRESENTATION_MAX_BACKOFF_M = 3.0
CAMERA_AZIMUTH_STEPS_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0)
# Orchestrator 2026-09-07: the wide-shot search is widened - eye up to WIDE_MAX_EYE_HEIGHT_M,
# back-off up to +4 m, and the eye may move IN toward the aim (near top-down pitch); the room
# must still span >= ROOM_FRAME_COVERAGE_MIN of the width, relaxed to
# ROOM_FRAME_COVERAGE_RELAXED_MIN only when nothing passes. If no single pose passes,
# TWO wide poses: the 3/4 one for establish + Burger's pass, a steeper one for Husky's.
# M7 (orchestrator decision): 7.5 m is the default cap - the height at which the hero's
# Husky pose reaches the 0.95 ribbon-visibility bar (morning-4b: 6.0 m gave 0.933,
# 7.5 m gives 0.963); `--max-eye-height` overrides it.
WIDE_MAX_EYE_HEIGHT_M = 7.5
WIDE_MAX_BACKOFF_M = 4.0
WIDE_MAX_MOVE_IN_M = 8.0
ROOM_FRAME_COVERAGE_RELAXED_MIN = 0.50


@dataclass
class BootstrapOutput:
    out_dir: Path
    scene: trimesh.Scene
    floor_y: float
    ceiling_y: float
    room_polygon: list[tuple[float, float]]
    yaw_rad: float
    yaw_center_xz: tuple[float, float]
    meta: dict


ASSETS_GLB_NAME = "scene_assets.glb"
ASSETS_PLACED_NAME = "assets_placed.json"


def resolve_assets(bootstrap_out: Path, assets="auto", assets_placed: Path | None = None) -> tuple[Path | None, list[dict] | None]:
    """M7: `(assets GLB, assets_placed.json records)` for the presentation.
    `assets="auto"` picks `<bootstrap_out>/scene_assets.glb` + `assets_placed.json`
    when both exist (the `assemble_with_generated --assets` output); a Path is an
    explicit GLB (with `assets_placed` next to it unless given); None/False keeps
    the placeholder `scene.glb`."""
    bootstrap_out = Path(bootstrap_out)
    if assets is None or assets is False:
        return None, None
    if assets == "auto":
        glb = bootstrap_out / ASSETS_GLB_NAME
        placed = assets_placed or bootstrap_out / ASSETS_PLACED_NAME
        if not (glb.exists() and placed.exists()):
            return None, None
    else:
        glb = Path(assets)
        placed = assets_placed or glb.parent / ASSETS_PLACED_NAME
        if not glb.exists():
            raise FileNotFoundError(f"assets GLB not found: {glb}")
        if not placed.exists():
            raise FileNotFoundError(f"assets_placed.json not found next to {glb}: {placed}")
    from scripts.msa.usd_assets import load_assets_placed

    return glb, load_assets_placed(placed)


def load_bootstrap_output(bootstrap_out: Path, *, scene_glb: Path | None = None) -> BootstrapOutput:
    """`scene_glb` (M7): read the object visuals/hulls from this GLB instead of
    `scene.glb` - the assembled `scene_assets.glb`, whose visuals are the real
    furniture assets. Walls, floor and hulls are identical between the two."""
    bootstrap_out = Path(bootstrap_out)
    meta = json.loads((bootstrap_out / "scene_meta.json").read_text())
    scene = trimesh.load(str(scene_glb or (bootstrap_out / "scene.glb")), force="scene")
    yaw = float(meta.get("yaw_correction_rad") or 0.0) if meta.get("yaw_applied", True) else 0.0
    center = tuple(meta.get("yaw_rotation_center_xy") or (0.0, 0.0))
    room_polygon = [tuple(p) for p in (meta.get("room_polygon") or meta.get("floor_polygon") or [])]
    if len(room_polygon) < 3:
        from scripts.msa.textures import floor_polygon_from_scene

        room_polygon = [tuple(p) for p in (floor_polygon_from_scene(scene) or [])]
    return BootstrapOutput(
        out_dir=bootstrap_out, scene=scene, floor_y=float(meta["floor_y"]), ceiling_y=float(meta["ceiling_y"]),
        room_polygon=room_polygon, yaw_rad=yaw, yaw_center_xz=(float(center[0]), float(center[1])), meta=meta,
    )


def load_grid(scene_dir: Path) -> tuple[np.ndarray, GridMeta]:
    """`occupancy.npy` ([ix, iz], uint8 FREE/OBSTACLE/UNKNOWN - gpu/stage_occupancy.py)
    + `occupancy_meta.json` -> (cells, GridMeta). No DB, no scene_ingest artifact
    parsing beyond the two files the raw scene dir always has."""
    scene_dir = Path(scene_dir)
    cells = np.load(scene_dir / "occupancy.npy")
    m = json.loads((scene_dir / "occupancy_meta.json").read_text())
    meta = GridMeta(
        resolution=float(m["resolution"]), origin_x=float(m["origin_x"]), origin_z=float(m["origin_z"]),
        width=int(m["width"]), height=int(m["height"]),
    )
    if cells.shape != (meta.width, meta.height):
        raise SystemExit(
            f"{scene_dir}/occupancy.npy is {cells.shape} but occupancy_meta.json says width x height = "
            f"{meta.width} x {meta.height} - the grid must be [ix, iz]"
        )
    return cells, meta


# --- objects / target ---------------------------------------------------------------


def object_hulls_xz(scene: trimesh.Scene) -> dict[str, np.ndarray]:
    """`{obj_id: hull ring (Nx2, world XZ, exported frame)}` from every
    `<id>/collision/hull` node - the bottom-cap vertices' 2D convex hull (the
    hull was extruded from a convex `hull_xz`, so this recovers it exactly)."""
    hulls: dict[str, np.ndarray] = {}
    for node_name, mesh in iter_scene_world_meshes(scene):
        if not node_name.endswith("/collision/hull"):
            continue
        verts = np.asarray(mesh.vertices, dtype=float)
        bottom = verts[np.isclose(verts[:, 1], verts[:, 1].min(), atol=1e-6)][:, [0, 2]]
        ring = convex_hull_2d(bottom)
        if len(ring) >= 3:
            hulls[node_name.split("/", 1)[0]] = np.asarray(ring, dtype=float)
    return hulls


def select_target(hulls: dict[str, np.ndarray], target: str) -> str:
    """`target` is either an exact object id (`desk_1`) or a class (`desk`); a class
    with several instances resolves to the one with the largest footprint
    (deterministic, and a bigger desk is the more obvious demo target)."""
    if target in hulls:
        return target
    candidates = [(Polygon(h).area, oid) for oid, h in hulls.items() if _object_class_from_id(oid).lower() == target.lower()]
    if not candidates:
        classes = sorted({_object_class_from_id(oid) for oid in hulls})
        raise SystemExit(f"--target {target!r}: no object of that id/class has a collision hull - classes present: {classes}")
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][1]


def to_raw_frame(points_xz, bo: BootstrapOutput) -> np.ndarray:
    return np.asarray([rotate_point_xz(float(x), float(z), -bo.yaw_rad, bo.yaw_center_xz) for x, z in points_xz], dtype=float)


def to_exported_frame(points_xz, bo: BootstrapOutput) -> np.ndarray:
    return np.asarray([rotate_point_xz(float(x), float(z), bo.yaw_rad, bo.yaw_center_xz) for x, z in points_xz], dtype=float)


def rasterize_hulls(cells: np.ndarray, meta: GridMeta, hulls_raw: dict[str, np.ndarray]) -> int:
    """Mark every cell whose center lies inside an object hull OBSTACLE, in place
    (the exact-polygon analogue of demo/record_isaac._rasterize_object_obstacles's
    bbox stamp - occupancy.npy is a height-band density grid, not object-aware, so
    a desk whose top is above the band can read as clear; see the 2026-09-05 v1
    DECISIONS entry, finding 3). Returns the number of cells newly marked."""
    ix, iz = np.meshgrid(np.arange(meta.width), np.arange(meta.height), indexing="ij")
    xs = meta.origin_x + (ix + 0.5) * meta.resolution
    zs = meta.origin_z + (iz + 0.5) * meta.resolution
    before = int((cells == OBSTACLE).sum())
    for ring in hulls_raw.values():
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        inside = contains_xy(poly, xs.ravel(), zs.ravel()).reshape(cells.shape)
        cells[inside] = OBSTACLE
    return int((cells == OBSTACLE).sum()) - before


# --- planning -----------------------------------------------------------------------


def _valid_polygon(ring) -> Polygon | None:
    if ring is None or len(ring) < 3:
        return None
    poly = Polygon([(float(x), float(z)) for x, z in ring])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def _cell_centres(meta: GridMeta) -> tuple[np.ndarray, np.ndarray]:
    ix, iz = np.meshgrid(np.arange(meta.width), np.arange(meta.height), indexing="ij")
    return meta.origin_x + (ix + 0.5) * meta.resolution, meta.origin_z + (iz + 0.5) * meta.resolution


def clip_grid_to_room(inflated: np.ndarray, meta: GridMeta, room_poly: Polygon, robot_radius_m: float) -> int:
    """T15h: mark every cell whose centre lies outside `room_poly` eroded by the robot
    radius OBSTACLE, in place, and return how many were newly blocked. The
    occupancy grid's FREE region is a density map that runs past the room polygon
    (T15e: path point 2 lay 0.28 m outside the authored floor mesh), so the
    traversable grid is the intersection of "clear in the grid" and "inside the
    floor by >= the robot radius". Applied AFTER `pathfinding.inflate` so the
    boundary margin is exactly the radius, not twice it."""
    eroded = room_poly.buffer(-robot_radius_m)
    if eroded.is_empty:
        raise SystemExit(f"room polygon eroded by the robot radius {robot_radius_m} m is empty - the room is narrower than the robot")
    xs, zs = _cell_centres(meta)
    inside = contains_xy(eroded, xs.ravel(), zs.ravel()).reshape(inflated.shape)
    newly = int(((~inside) & (inflated != OBSTACLE)).sum())
    inflated[~inside] = OBSTACLE
    return newly


def clearance_m(points_xz, hull_polys: list[Polygon], room_poly: Polygon | None) -> np.ndarray:
    """Per point: the distance to the nearest object hull or room boundary (whichever
    is closer); a point inside a hull or outside the room gets 0. Vectorized
    (shapely 2) - called for every goal candidate and every path sample."""
    import shapely

    pts_arr = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    pts = shapely.points(pts_arr[:, 0], pts_arr[:, 1])
    best = np.full(len(pts_arr), np.inf)
    for poly in hull_polys:
        best = np.minimum(best, shapely.distance(poly, pts))
    if room_poly is not None:
        d_boundary = shapely.distance(room_poly.boundary, pts)
        d_boundary[~contains_xy(room_poly, pts_arr[:, 0], pts_arr[:, 1])] = 0.0
        best = np.minimum(best, d_boundary)
    return best


def sample_polyline(points_xz, spacing_m: float) -> np.ndarray:
    pts = np.asarray(points_xz, dtype=float)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / spacing_m)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def room_core_rectangle(room_polygon_xz) -> Polygon | None:
    """Morning-3: the room's "main rectangle" = the largest-area axis-aligned
    rectangle CONTAINED in the (Manhattan, exported-frame) room polygon, searched
    over the polygon's own vertex coordinates (for a rectilinear polygon the maximal
    inscribed axis-aligned rectangle has its sides on vertex coordinates). The
    polygon's min-area bounding rectangle cannot play this role - it contains the
    polygon, so nothing would ever lie "outside" it."""
    room = _valid_polygon(room_polygon_xz)
    if room is None:
        return None
    xs = sorted({round(float(x), 6) for x, _ in room.exterior.coords})
    zs = sorted({round(float(z), 6) for _, z in room.exterior.coords})
    if len(xs) < 2 or len(zs) < 2:
        return None
    # inside[i, k]: the grid cell between xs[i]..xs[i+1] x zs[k]..zs[k+1] lies in the room
    cx = np.asarray([(a + b) / 2.0 for a, b in zip(xs, xs[1:])])
    cz = np.asarray([(a + b) / 2.0 for a, b in zip(zs, zs[1:])])
    gx, gz = np.meshgrid(cx, cz, indexing="ij")
    inside = contains_xy(room, gx.ravel(), gz.ravel()).reshape(gx.shape)
    # prefix sums so any index rectangle's "all inside" test is O(1)
    ps = np.zeros((inside.shape[0] + 1, inside.shape[1] + 1), dtype=int)
    ps[1:, 1:] = np.cumsum(np.cumsum(inside.astype(int), axis=0), axis=1)
    best, best_area = None, 0.0
    n, m = inside.shape
    for i0 in range(n):
        for i1 in range(i0 + 1, n + 1):
            w = xs[i1] - xs[i0]
            for k0 in range(m):
                for k1 in range(k0 + 1, m + 1):
                    if ps[i1, k1] - ps[i0, k1] - ps[i1, k0] + ps[i0, k0] != (i1 - i0) * (k1 - k0):
                        break  # a taller rectangle at this k0 cannot be inside either
                    area = w * (zs[k1] - zs[k0])
                    if area > best_area:
                        best_area, best = area, (xs[i0], zs[k0], xs[i1], zs[k1])
    if best is None:
        return None
    from shapely.geometry import box

    return box(*best)


SPUR_MIN_AREA_M2 = 0.5  # a remainder piece smaller than this is a rasterization sliver, not a doorway appendix


def doorway_spur_polygon(room_polygon_xz) -> tuple[Polygon | None, dict]:
    """Morning-3 rule for "near the doorway": the DOORWAY SPUR is the piece of the
    Manhattan room outline that lies outside its core rectangle
    (`room_core_rectangle`) with the greatest DEPTH (largest distance of any of its
    vertices from the core rectangle), among pieces >= SPUR_MIN_AREA_M2. T15g keeps
    the doorway appendix (the corridor nub the T15b opening leaves) as exactly such
    a piece; a shallow strip along a wall (the hero's 0.47 m deep strip where the
    desk stands) is not a doorway. Returns `(spur polygon | None, info)` with the
    core rectangle and every candidate piece in `info`."""
    core = room_core_rectangle(room_polygon_xz)
    room = _valid_polygon(room_polygon_xz)
    info: dict = {"rule": "polygon minus largest inscribed axis-aligned rectangle; deepest piece >= SPUR_MIN_AREA_M2",
                  "core_rect_bounds": list(core.bounds) if core is not None else None, "pieces": [], "spur_min_area_m2": SPUR_MIN_AREA_M2}
    if core is None or room is None:
        return None, info
    rest = room.difference(core.buffer(1e-6))
    pieces = [rest] if rest.geom_type == "Polygon" else [g for g in getattr(rest, "geoms", []) if g.geom_type == "Polygon"]
    best, best_depth = None, -1.0
    for piece in pieces:
        if piece.is_empty or piece.area < SPUR_MIN_AREA_M2:
            continue
        depth = max(core.exterior.distance(Point(c)) for c in piece.exterior.coords)
        info["pieces"].append({"bounds": list(piece.bounds), "area_m2": float(piece.area), "depth_m": float(depth)})
        if depth > best_depth:
            best, best_depth = piece, depth
    info["spur_bounds"] = list(best.bounds) if best is not None else None
    info["spur_depth_m"] = float(best_depth) if best is not None else None
    return best, info


# Morning-3 demo-route acceptance: the route must be at least this long and must pass
# through the corridor between the target and the room boundary (T15e: 0.414 m on the
# hero, which blocks Husky) - `corridor_report`.
MIN_ROUTE_LENGTH_M = 2.5
CORRIDOR_SEARCH_M = 0.6  # a path sample within this of BOTH the target hull and the room boundary is "in the corridor"
START_CANDIDATES_MAX = 25


def corridor_report(samples_xz, target_poly: Polygon, other_hulls: list[Polygon], room_poly: Polygon | None,
                    *, search_m: float = CORRIDOR_SEARCH_M) -> dict:
    """Where the path runs between the target hull and the room boundary: for every
    path sample the corridor width there is `d(target) + d(boundary)`; the corridor
    crossed is the narrowest such sample among those within `search_m` of both
    (and closer to the target than to any other hull). Also reports the narrowest
    passage of any kind along the path (`d(nearest obstacle) x 2`)."""
    import shapely

    pts_arr = np.asarray(samples_xz, dtype=float).reshape(-1, 2)
    pts = shapely.points(pts_arr[:, 0], pts_arr[:, 1])
    d_t = shapely.distance(target_poly, pts)
    d_b = np.full(len(pts_arr), np.inf) if room_poly is None else shapely.distance(room_poly.boundary, pts)
    d_o = np.full(len(pts_arr), np.inf)
    for h in other_hulls:
        d_o = np.minimum(d_o, shapely.distance(h, pts))
    in_corridor = (d_t <= search_m) & (d_b <= search_m) & (d_t <= d_o)
    out = {"crossed": bool(in_corridor.any()), "search_m": search_m, "n_samples_in_corridor": int(in_corridor.sum()),
           "closest_approach_target_m": float(d_t.min()), "closest_approach_boundary_m": float(d_b.min()) if np.isfinite(d_b).any() else None}
    if in_corridor.any():
        width = np.where(in_corridor, d_t + d_b, np.inf)
        k = int(width.argmin())
        out.update({"width_m": float(width[k]), "at_xz": [float(v) for v in pts_arr[k]], "d_target_m": float(d_t[k]), "d_boundary_m": float(d_b[k])})
    return out


def plan_route(
    cells: np.ndarray, meta: GridMeta, hulls_raw: dict[str, np.ndarray], target_id: str, robot_radius_m: float,
    room_polygon_raw=None, *, goal_clearance_margin_m: float = GOAL_CLEARANCE_MARGIN_M,
    start_region_raw=None, start_rule: str = "longest_free_run", fixed_start_raw=None,
) -> dict:
    """Grid-level A* in the RAW frame: obstacles = grid OBSTACLE cells + every hull
    rasterized, inflated by the robot radius (`pathfinding.inflate`), then (T15h)
    clipped to `room_polygon_raw` eroded by the radius (`clip_grid_to_room`).

    Start (`start_rule`):
      - `"longest_free_run"` (pre-morning-3 default, kept for the fixture tests):
        `record_isaac._longest_free_run_start` on the clipped grid.
      - `"farthest_from_target"` (morning-3): the traversable cell whose centre lies
        inside `start_region_raw` (the doorway spur, raw frame - see
        `doorway_spur_polygon`; None = the whole traversable grid) that is FARTHEST
        from the target footprint (ties: more clearance), trying the next-farthest
        (up to START_CANDIDATES_MAX) when A* cannot connect it.
      - `"fixed"` (run5 prep): a single caller-supplied raw-frame point
        (`fixed_start_raw`, e.g. `scene_meta.json`'s `start_xy` rotated to the raw
        frame) - snapped to the nearest traversable cell if the exact cell is not
        finite in the inflated/clipped cost grid. No spur/farthest-from-target
        search; the only candidate tried.
    Goal = the nearest traversable (preferably FREE) cell to the target footprint
    whose clearance to EVERY hull and to the room boundary is >= radius +
    `goal_clearance_margin_m` (`clearance_m`), trying up to `GOAL_CANDIDATES_MAX`
    nearest-first until A* connects. Returns cells + raw-frame world points (cell
    centres, line-of-sight smoothed) plus the T15h quality report: goal clearance,
    min clearance along the path (sampled every `PATH_CLEARANCE_SAMPLE_M`), the
    smallest distance from a path point to the room boundary (asserted >= the
    radius), and (morning-3) the `corridor_report`."""
    room_poly = _valid_polygon(room_polygon_raw)
    obstacle_cells = cells.copy()
    n_rasterized = rasterize_hulls(obstacle_cells, meta, hulls_raw)
    inflated = inflate(obstacle_cells, meta, robot_radius_m)
    n_clipped = clip_grid_to_room(inflated, meta, room_poly, robot_radius_m) if room_poly is not None else 0
    cost = build_cost_grid(inflated)

    hull_polys = {oid: _valid_polygon(ring) for oid, ring in hulls_raw.items()}
    hull_polys = {oid: p for oid, p in hull_polys.items() if p is not None}
    target_poly = hull_polys[target_id]
    all_hulls = list(hull_polys.values())
    required_clearance = robot_radius_m + goal_clearance_margin_m
    finite_ix, finite_iz = np.nonzero(np.isfinite(cost))
    cand_xz = np.column_stack([meta.origin_x + (finite_ix + 0.5) * meta.resolution, meta.origin_z + (finite_iz + 0.5) * meta.resolution])
    import shapely

    d_target = shapely.distance(target_poly, shapely.points(cand_xz[:, 0], cand_xz[:, 1]))

    # --- start candidates ------------------------------------------------------------
    start_info: dict = {"rule": start_rule}
    if start_rule == "longest_free_run":
        start = _longest_free_run_start(inflated, meta, robot_radius_m)
        if start is None:
            raise SystemExit("no FREE run in the inflated occupancy grid - nowhere to start the robot")
        start_cell = world_to_cell(start["x"], start["z"], meta)
        if not np.isfinite(cost[start_cell]):
            raise SystemExit(f"start cell {start_cell} is blocked after inflation (unexpected for a FREE run)")
        start_candidates = [(start_cell, (float(start["x"]), float(start["z"])))]
        start_info["run_len_m"] = float(start["run_len_m"])
    elif start_rule == "farthest_from_target":
        region = _valid_polygon(start_region_raw) if start_region_raw is not None else None
        in_region = contains_xy(region, cand_xz[:, 0], cand_xz[:, 1]) if region is not None else np.ones(len(cand_xz), dtype=bool)
        start_info["region_applied"] = region is not None
        start_info["n_traversable_cells_in_region"] = int(in_region.sum())
        if not in_region.any():
            raise SystemExit("morning-3 start rule: no traversable cell lies inside the doorway-spur region - the spur is narrower than the robot")
        region_clear = clearance_m(cand_xz[in_region], all_hulls, room_poly)
        order = sorted(
            zip((-d_target[in_region]).tolist(), (-region_clear).tolist(), finite_ix[in_region].tolist(), finite_iz[in_region].tolist())
        )
        start_candidates = [((cix, ciz), (float(meta.origin_x + (cix + 0.5) * meta.resolution), float(meta.origin_z + (ciz + 0.5) * meta.resolution)))
                            for _, _, cix, ciz in order[:START_CANDIDATES_MAX]]
        start_info["farthest_distance_to_target_m"] = float(-order[0][0])
    elif start_rule == "fixed":
        if fixed_start_raw is None:
            raise ValueError("start_rule 'fixed' requires fixed_start_raw")
        fx, fz = float(fixed_start_raw[0]), float(fixed_start_raw[1])
        s_cell = world_to_cell(fx, fz, meta)
        if 0 <= s_cell[0] < cost.shape[0] and 0 <= s_cell[1] < cost.shape[1] and np.isfinite(cost[s_cell]):
            s_xz = (fx, fz)
            start_info["snapped_m"] = 0.0
        else:
            d2 = (cand_xz[:, 0] - fx) ** 2 + (cand_xz[:, 1] - fz) ** 2
            k = int(np.argmin(d2))
            s_cell = (int(finite_ix[k]), int(finite_iz[k]))
            s_xz = (float(cand_xz[k, 0]), float(cand_xz[k, 1]))
            start_info["snapped_m"] = float(np.sqrt(d2[k]))
        start_candidates = [(s_cell, s_xz)]
        start_info["fixed_xz"] = (fx, fz)
    else:
        raise ValueError(f"unknown start_rule {start_rule!r}")

    # --- goal candidates -------------------------------------------------------------
    near = d_target <= GOAL_SEARCH_RADIUS_M
    if not near.any():
        raise SystemExit(f"no traversable cell within {GOAL_SEARCH_RADIUS_M} m of target {target_id!r}'s footprint")
    clear = clearance_m(cand_xz[near], all_hulls, room_poly)
    ok = clear >= required_clearance - 1e-9
    n_near = int(near.sum())
    if not ok.any():
        raise SystemExit(
            f"none of the {n_near} traversable cells within {GOAL_SEARCH_RADIUS_M} m of {target_id!r} keeps "
            f">= {required_clearance:.3f} m to every hull and the room boundary (best {clear.max():.3f} m)"
        )
    near_ix, near_iz = finite_ix[near][ok], finite_iz[near][ok]
    candidates = sorted(
        (int(inflated[cix, ciz] != FREE), float(d), int(cix), int(ciz))
        for cix, ciz, d in zip(near_ix.tolist(), near_iz.tolist(), d_target[near][ok].tolist())
    )

    path_cells = None
    goal_cell = None
    start_cell = None
    start_xz = None
    errors = []
    n_starts_tried = 0
    for s_cell, s_xz in start_candidates:
        n_starts_tried += 1
        for _, _, cix, ciz in candidates[:GOAL_CANDIDATES_MAX]:
            try:
                path_cells = astar(cost, s_cell, (cix, ciz))
                goal_cell = (cix, ciz)
                start_cell, start_xz = s_cell, s_xz
                break
            except NoPathError as exc:
                errors.append(str(exc))
        if path_cells is not None:
            break
    if path_cells is None:
        raise SystemExit(
            f"no path from any of {n_starts_tried} start candidates ({start_rule}) to any of the "
            f"{min(len(candidates), GOAL_CANDIDATES_MAX)} nearest traversable cells around {target_id!r}: {errors[-1] if errors else '?'}"
        )
    smoothed = smooth_path(path_cells, cost)
    points_raw = np.asarray([cell_to_world(ix, iz, meta) for ix, iz in smoothed], dtype=float)
    goal_xz = points_raw[-1]

    # --- T15h quality report -------------------------------------------------------
    samples = sample_polyline(points_raw, PATH_CLEARANCE_SAMPLE_M)
    sample_clearance = clearance_m(samples, all_hulls, room_poly)
    point_clearance = clearance_m(points_raw, all_hulls, room_poly)
    goal_clearance = float(clearance_m([goal_xz], all_hulls, room_poly)[0])
    inside_by = None
    if room_poly is not None:
        d_boundary = np.asarray(clearance_m(points_raw, [], room_poly), dtype=float)
        inside_by = float(d_boundary.min())
        if inside_by < robot_radius_m - 1e-6:
            worst = points_raw[int(d_boundary.argmin())]
            raise SystemExit(
                f"path point ({worst[0]:.3f}, {worst[1]:.3f}) is only {inside_by:.3f} m inside the room polygon "
                f"(< robot radius {robot_radius_m} m) - the room clipping did not hold"
            )
    other_hulls = [p for oid, p in hull_polys.items() if oid != target_id]
    corridor = corridor_report(samples, target_poly, other_hulls, room_poly)
    return {
        "start_raw_xz": (float(start_xz[0]), float(start_xz[1])),
        "start_cell": list(start_cell),
        "start_rule": start_rule,
        "start_info": start_info,
        "start_candidates_tried": n_starts_tried,
        "start_run_len_m": start_info.get("run_len_m"),
        "start_clearance_m": float(clearance_m([[start_xz[0], start_xz[1]]], all_hulls, room_poly)[0]),
        "start_distance_to_target_m": float(target_poly.distance(Point(*start_xz))),
        "goal_cell": list(goal_cell),
        "goal_raw_xz": [float(v) for v in goal_xz],
        "goal_distance_to_footprint_m": float(target_poly.distance(Point(*goal_xz))),
        "goal_distance_to_room_boundary_m": float(room_poly.boundary.distance(Point(*goal_xz))) if room_poly is not None else None,
        "goal_clearance_m": goal_clearance,
        "goal_required_clearance_m": float(required_clearance),
        "goal_candidates_near_target": n_near,
        "goal_candidates_with_clearance": int(ok.sum()),
        "goal_candidates_tried": len(errors) + 1,
        "path_cells": [list(c) for c in smoothed],
        "n_astar_cells": len(path_cells),
        "points_raw_xz": [[float(x), float(z)] for x, z in points_raw],
        "path_length_m": float(path_length_m([(float(x), float(z)) for x, z in points_raw])),
        "path_min_clearance_m": float(sample_clearance.min()),
        "path_points_min_clearance_m": float(point_clearance.min()),
        "path_points_inside_room_min_m": inside_by,
        "path_clearance_sample_m": PATH_CLEARANCE_SAMPLE_M,
        "corridor": corridor,
        "n_hull_cells_rasterized": n_rasterized,
        "n_cells_clipped_outside_room": n_clipped,
        "room_clip_applied": room_poly is not None,
    }


def plan_presentation_path(bo: BootstrapOutput, scene_dir: Path, target: str, platform: PlatformSpec, *,
                           min_route_length_m: float = MIN_ROUTE_LENGTH_M, blocked_platform_id: str = "husky",
                           require_corridor: bool = True) -> dict:
    """The `path.json` payload: route + target in the EXPORTED (yaw-rotated, Y-up)
    frame the GLB/USD live in, with the raw-frame planning record alongside.

    Morning-3 demo route: start = the traversable cell inside the doorway spur
    (`doorway_spur_polygon` of the exported Manhattan outline, rotated into the raw
    frame) farthest from the target footprint - or, without a spur, the traversable
    cell farthest from the target anywhere in the room; goal as T15h. The route
    must be >= `min_route_length_m` long and must pass through the corridor between
    the target and the room boundary (`corridor_report`) - both are asserted here
    (SystemExit with the numbers), never silently re-planned. The corridor width is
    compared with `blocked_platform_id`'s body width: the demo expects it to be
    blocked there."""
    cells, meta = load_grid(scene_dir)
    hulls = object_hulls_xz(bo.scene)
    if not hulls:
        raise SystemExit(f"{bo.out_dir}/scene.glb has no <id>/collision/hull nodes - nothing to target")
    target_id = select_target(hulls, target)
    hulls_raw = {oid: to_raw_frame(ring, bo) for oid, ring in hulls.items()}
    room_raw = to_raw_frame(bo.room_polygon, bo) if len(bo.room_polygon) >= 3 else None
    spur, spur_info = doorway_spur_polygon(bo.room_polygon) if len(bo.room_polygon) >= 3 else (None, {"rule": "no room polygon"})
    spur_raw = to_raw_frame(list(spur.exterior.coords)[:-1], bo) if spur is not None else None
    fixed_start_xy = bo.meta.get("start_xy")
    if fixed_start_xy is not None:
        fixed_start_raw = to_raw_frame([tuple(fixed_start_xy)], bo)[0]
        route = plan_route(cells, meta, hulls_raw, target_id, platform.radius_m, room_raw, start_rule="fixed", fixed_start_raw=fixed_start_raw)
        start_source = "scene_meta_start_xy (run5 prep: shared Burger/Go2/Husky start, >= 0.5 m free radius, written into scene_meta.json)"
    else:
        route = plan_route(cells, meta, hulls_raw, target_id, platform.radius_m, room_raw, start_region_raw=spur_raw, start_rule="farthest_from_target")
        start_source = (
            "doorway_spur_farthest_from_target (morning-3: traversable cell inside the doorway spur farthest from the target footprint)"
            if spur is not None else
            "farthest_from_target_no_spur (morning-3: no doorway spur in the outline - traversable cell farthest from the target)"
        )

    blocked = resolve_platform(blocked_platform_id)
    corridor = dict(route["corridor"])
    corridor["blocked_platform_id"] = blocked.id
    corridor["blocked_platform_width_m"] = float(blocked.width_m)
    corridor["blocks_platform"] = bool(corridor.get("crossed") and corridor["width_m"] < float(blocked.width_m))
    corridor["required"] = bool(require_corridor)
    if require_corridor and not corridor.get("crossed"):
        raise SystemExit(
            f"morning-3: the planned route does not pass through the corridor between {target_id!r} and the room boundary "
            f"(no path sample within {CORRIDOR_SEARCH_M} m of both; closest approach to the target {corridor['closest_approach_target_m']:.3f} m, "
            f"to the boundary {corridor['closest_approach_boundary_m']}) - route start {route['start_raw_xz']}, goal {route['goal_raw_xz']}, "
            f"length {route['path_length_m']:.2f} m"
        )
    if route["path_length_m"] < min_route_length_m - 1e-9:
        raise SystemExit(
            f"morning-3: the planned route is {route['path_length_m']:.2f} m long, < the {min_route_length_m} m demo minimum - "
            f"start rule {route['start_rule']} ({start_source.split(' (')[0]}), farthest traversable cell in the start region is "
            f"{route['start_distance_to_target_m']:.2f} m from {target_id!r} (spur: {spur_info.get('spur_bounds')}); "
            f"the goal is the T15h nearest clear cell ({route['goal_raw_xz']}). Pass --min-route-length to accept a shorter demo route."
        )

    points_xz = to_exported_frame(route["points_raw_xz"], bo)
    y = bo.floor_y + PATH_HEIGHT_ABOVE_FLOOR_M
    path_points_world = [[float(x), float(y), float(z)] for x, z in points_xz]
    start_xz = to_exported_frame([route["start_raw_xz"]], bo)[0]
    # T15h: /World/Target IS the route goal (the last path point) - the acceptance
    # "robot within 0.3 m of the target" is measurable there; the target object's
    # footprint centroid (T15d's target, 1.11 m from the goal and inside the desk hull
    # on the hero) is kept as the separate visual marker /World/TargetObject.
    goal_xz = points_xz[-1]
    target_point = [float(goal_xz[0]), bo.floor_y + TARGET_HEIGHT_ABOVE_FLOOR_M, float(goal_xz[1])]
    target_centroid = Polygon(hulls[target_id]).centroid
    target_object_point = [float(target_centroid.x), bo.floor_y + TARGET_HEIGHT_ABOVE_FLOOR_M, float(target_centroid.y)]
    return {
        "schema": PATH_SCHEMA,
        "frame": "bootstrap exported frame: Y-up, yaw-rotated by yaw_correction_rad about yaw_rotation_center_xy",
        "platform": {
            "id": platform.id, "display_name": platform.display_name, "radius_m": platform.radius_m,
            "height_m": platform.height_m, "length_m": platform.length_m, "width_m": platform.width_m,
        },
        "target_id": target_id,
        "target_class": _object_class_from_id(target_id),
        "target_point": target_point,
        "target_point_semantics": "route goal = last path point (/World/Target); the object's centroid is target_object_point (/World/TargetObject)",
        "target_object_point": target_object_point,
        "goal_to_target_object_centroid_m": float(math.hypot(goal_xz[0] - target_centroid.x, goal_xz[1] - target_centroid.y)),
        "target_hull_xz": [[float(x), float(z)] for x, z in hulls[target_id]],
        "start_xy": [float(start_xz[0]), float(start_xz[1])],
        "start_source": start_source,
        "doorway_spur": {**spur_info, "spur_polygon_xz": [[float(x), float(z)] for x, z in spur.exterior.coords] if spur is not None else None},
        "path_points_world": path_points_world,
        "n_waypoints": len(path_points_world),
        "path_length_m": float(path_length_m([(p[0], p[2]) for p in path_points_world])),
        "route_quality": {
            "goal_clearance_m": route["goal_clearance_m"],
            "goal_required_clearance_m": route["goal_required_clearance_m"],
            "goal_distance_to_footprint_m": route["goal_distance_to_footprint_m"],
            "goal_distance_to_room_boundary_m": route["goal_distance_to_room_boundary_m"],
            "path_min_clearance_m": route["path_min_clearance_m"],
            "path_points_inside_room_min_m": route["path_points_inside_room_min_m"],
            "start_clearance_m": route["start_clearance_m"],
            "start_distance_to_target_m": route["start_distance_to_target_m"],
            "start_rule": route["start_rule"],
            "start_candidates_tried": route["start_candidates_tried"],
            "room_clip_applied": route["room_clip_applied"],
            "n_cells_clipped_outside_room": route["n_cells_clipped_outside_room"],
            "min_route_length_m": float(min_route_length_m),
            "corridor": {**corridor, "at_xz": [float(v) for v in to_exported_frame([corridor["at_xz"]], bo)[0]] if corridor.get("at_xz") else None,
                         "at_raw_xz": corridor.get("at_xz")},
        },
        "yaw_correction_rad": bo.yaw_rad,
        "yaw_rotation_center_xy": list(bo.yaw_center_xz),
        "grid": {
            "scene_dir": str(scene_dir), "resolution": meta.resolution, "origin_x": meta.origin_x,
            "origin_z": meta.origin_z, "width": meta.width, "height": meta.height, "layout": "[ix, iz]",
        },
        "planner": "app.services.pathfinding: inflate -> build_cost_grid -> astar -> smooth_path (grid-level, no DB)",
        "raw_frame": {k: v for k, v in route.items()},
    }


# --- local stills -------------------------------------------------------------------


STILL_MAX_TRIANGLE_EDGE_M = 0.5


def _subdivided(mesh: trimesh.Trimesh, max_edge_m: float = STILL_MAX_TRIANGLE_EDGE_M) -> trimesh.Trimesh:
    """render_perspective's painter's algorithm sorts per TRIANGLE by mean depth, so a
    4 m wall or floor triangle can be drawn over a small robot that is actually in
    front of part of it. Splitting the big structural triangles to <= `max_edge_m`
    keeps the sort honest (objects/placeholders are already small)."""
    from trimesh.remesh import subdivide_to_size

    verts, faces = subdivide_to_size(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces), max_edge=max_edge_m)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return out


def _colored(mesh: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile([*rgb, 255], (len(mesh.vertices), 1)))
    return mesh


def _segment_box(a_xz, b_xz, y: float, width: float, height: float, rgb) -> trimesh.Trimesh | None:
    dx, dz = b_xz[0] - a_xz[0], b_xz[1] - a_xz[1]
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return None
    box = trimesh.creation.box(extents=(length, height, width))
    box.apply_transform(trimesh.transformations.rotation_matrix(math.atan2(-dz, dx), [0, 1, 0]))
    box.apply_translation(((a_xz[0] + b_xz[0]) / 2.0, y, (a_xz[1] + b_xz[1]) / 2.0))
    return _colored(box, rgb)


def wall_thickness_of(bo: BootstrapOutput) -> float:
    return float(bo.meta.get("wall_band_thickness_m") or WALL_THICKNESS_M)


def presentation_stub_records(bo: BootstrapOutput, *, wall_height_m: float = PRESENTATION_WALL_HEIGHT_M) -> list[dict]:
    """The per-edge stub boxes of the T15g outline (empty for a pre-T15g bootstrap)."""
    has_outline = any(node_name == "wall_outline" for node_name, _ in iter_scene_world_meshes(bo.scene))
    if not has_outline or len(bo.room_polygon) < 3:
        return []
    return stub_edge_boxes(bo.room_polygon, bo.floor_y, wall_height_m, wall_thickness_of(bo))


def object_part_occluder_boxes(scene: trimesh.Scene, floor_y: float | None = None) -> list[dict]:
    """T15i: one Y-up oriented box per `<id>/visual/part_j` mesh of a non-translucent
    object - the opaque placeholder geometry a camera ray must not cross. The box is
    the part's minimum-area oriented rectangle in XZ (`trimesh.bounds.oriented_bounds_2D`,
    exact for the box parts `assets.build_placeholder` emits, tight for cylinders) x its
    Y extent. NOT the AABB: on the 02_modular_home fixture `stool_1` sits at 38 deg and
    its AABB contained the route goal (0.2 m clear of the real hull), so no camera
    could ever "see" it. Parts whose top is within `visibility.LOW_OCCLUDER_MAX_TOP_M`
    of `floor_y` (rugs) are skipped - see that constant."""
    from trimesh.bounds import oriented_bounds_2D

    boxes = []
    for node_name, mesh in iter_scene_world_meshes(scene):
        if "/visual/part_" not in node_name:
            continue
        obj_id = node_name.split("/", 1)[0]
        if is_translucent_class(obj_id):
            continue
        verts = np.asarray(mesh.vertices, dtype=float)
        lo, hi = mesh.bounds
        if floor_y is not None and float(hi[1]) - float(floor_y) <= LOW_OCCLUDER_MAX_TOP_M:
            continue
        xz = verts[:, [0, 2]]
        try:
            T, extents = oriented_bounds_2D(xz)
        except Exception:  # degenerate (collinear) footprint - fall back to the AABB
            boxes.append(aabb_box(f"object:{node_name}", lo, hi))
            continue
        R = T[:2, :2]
        t = T[:2, 2]
        center_xz = -R.T @ t
        u, w = R[0], R[1]  # rows: the box's local axes in world XZ
        boxes.append(make_box(
            f"object:{node_name}",
            (float(center_xz[0]), float((lo[1] + hi[1]) / 2.0), float(center_xz[1])),
            ((float(u[0]), 0.0, float(u[1])), (0.0, 1.0, 0.0), (float(w[0]), 0.0, float(w[1]))),
            (float(extents[0]) / 2.0, float(hi[1] - lo[1]) / 2.0, float(extents[1]) / 2.0),
        ))
    return boxes


def scene_clearance_ranges(scene: trimesh.Scene) -> list[dict]:
    """Addendum: every mesh node's world AABB (Y-up) as `{"prim", "min", "max"}` -
    walls (the band = the stubs' and colliders' footprint), floor, every visual part
    and collision hull - the camera eye must keep >= CAMERA_MIN_CLEARANCE_M from all."""
    out = []
    for node_name, mesh in iter_scene_world_meshes(scene):
        lo, hi = mesh.bounds
        out.append({"prim": node_name, "min": tuple(float(v) for v in lo), "max": tuple(float(v) for v in hi)})
    return out


def build_presentation_scene(
    bo: BootstrapOutput, path_json: dict, *, wall_height_m: float = PRESENTATION_WALL_HEIGHT_M, culled_stub_edges=None,
    worst_case_box: dict | None = None,
) -> trimesh.Scene:
    """The still's geometry: 1.2 m wall stubs, floor, object placeholders (visuals
    only, colours preserved), the path as thin boxes per segment + waypoint cubes,
    the robot as an L x W x H box at the start heading down the first segment, the
    clearance ring as a flat annulus (radius = platform radius), and the target
    sphere. Node names follow `build_scene`'s `<id>/visual/part` convention so
    `render_perspective._iter_scene_node_faces` colours them as objects.

    T15i: the T15g `wall_outline` band becomes one box per outline edge
    (`wall_outline_edge_i`, `export_usd.stub_edge_boxes` - the same boxes the USD
    authors) and the edges in `culled_stub_edges` are left out, so the still shows
    exactly what Isaac renders of the presentation USD."""
    scene = trimesh.Scene()
    culled = {int(i) for i in (culled_stub_edges or [])}
    for node_name, mesh in iter_scene_world_meshes(bo.scene):
        if node_name == "wall_outline" and len(bo.room_polygon) >= 3:
            for rec in stub_edge_boxes(bo.room_polygon, bo.floor_y, wall_height_m, wall_thickness_of(bo)):
                if rec["edge"] in culled:
                    continue
                scene.add_geometry(_subdivided(stub_edge_trimesh(rec)), node_name=f"wall_outline_edge_{rec['edge']}")
        elif node_name.startswith("wall_"):
            scene.add_geometry(_subdivided(wall_stub_mesh(mesh, bo.floor_y, wall_height_m)), node_name=node_name)
        elif node_name == "floor":
            scene.add_geometry(_subdivided(mesh), node_name=node_name)
        elif "/visual/part_" in node_name:
            scene.add_geometry(mesh, node_name=node_name)

    pts = [(p[0], p[2]) for p in path_json["path_points_world"]]
    # Morning-4: the SAME 3 cm ribbon at floor + 0.03 m as the USD/GLB (`path_ribbon_mesh`)
    # - what the >= 95 % visibility criterion is about - drawn as a thin slab so the
    # painter's renderer shows it from any elevation.
    ribbon = path_ribbon_mesh(path_json["path_points_world"], bo.floor_y, rgb=PATH_RGB)
    if ribbon is not None:
        slab = ribbon.copy()
        slab.apply_translation((0.0, -0.01, 0.0))
        scene.add_geometry(_colored(ribbon, PATH_RGB), node_name="path/visual/ribbon")
        scene.add_geometry(_colored(slab, PATH_RGB), node_name="path/visual/ribbon_under")
    for i, (x, z) in enumerate(pts):
        cube = trimesh.creation.box(extents=(0.07, 0.03, 0.07))
        cube.apply_translation((x, bo.floor_y + 0.017, z))
        scene.add_geometry(_colored(cube, PATH_RGB), node_name=f"path/visual/wp_{i}")

    platform = path_json["platform"]
    sx, sz = path_json["start_xy"]
    heading = math.atan2(-(pts[1][1] - pts[0][1]), pts[1][0] - pts[0][0]) if len(pts) >= 2 else 0.0
    body = trimesh.creation.box(extents=(platform["length_m"], platform["height_m"], platform["width_m"]))
    body.apply_transform(trimesh.transformations.rotation_matrix(heading, [0, 1, 0]))
    body.apply_translation((sx, bo.floor_y + platform["height_m"] / 2.0 + 0.01, sz))
    scene.add_geometry(_colored(body, ROBOT_RGB), node_name="robot/visual/body")

    # T15i: the ring radius is the footprint's half-diagonal, exactly what
    # record_isaac._make_robot_proxy draws in Isaac (the platform radius alone hides
    # under Burger's own 0.138 x 0.178 body).
    r = math.hypot(float(platform["length_m"]), float(platform["width_m"])) / 2.0
    ring = trimesh.creation.annulus(r_min=max(0.005, r - RING_THICKNESS_M), r_max=r, height=0.008)
    ring.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    ring.apply_translation((sx, bo.floor_y + 0.006, sz))
    scene.add_geometry(_colored(ring, RING_RGB), node_name="ring/visual/annulus")

    tx, ty, tz = path_json["target_point"]
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    sphere.apply_translation((tx, ty + 0.1, tz))
    scene.add_geometry(_colored(sphere, TARGET_RGB), node_name="target/visual/sphere")
    if worst_case_box is not None:
        # the robot box at the pose's worst-case placement (visibility box: axes rows t, side, up)
        t, side, up_ax = (np.asarray(a, dtype=float) for a in worst_case_box["axes"])
        hl, hw, hh = worst_case_box["half"]
        box = trimesh.creation.box(extents=(2 * hl, 2 * hw, 2 * hh))
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2] = t, side, up_ax
        m[:3, 3] = np.asarray(worst_case_box["center"], dtype=float)
        box.apply_transform(m)
        scene.add_geometry(_colored(box, WORST_CASE_ROBOT_RGB), node_name="robot_worst/visual/body")
    if path_json.get("target_object_point") is not None:
        ox, oy, oz = path_json["target_object_point"]
        marker = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        marker.apply_translation((ox, oy + 0.05, oz))
        scene.add_geometry(_colored(marker, TARGET_OBJECT_RGB), node_name="target_object/visual/sphere")
    return scene


def presentation_check_points(bo: BootstrapOutput, path_json: dict, *, wall_height_m: float = PRESENTATION_WALL_HEIGHT_M,
                              ring_radius_m: float | None = None, robot_height_m: float | None = None) -> tuple[list, list, list]:
    """`(frame_points, labelled_visibility_points, room_box_points)` (Y-up) for the
    camera solve: frame points = T15d's `framing_check_points` (robot start footprint +
    top, every path point, the ring; 5 % margin); visibility points =
    `visibility.visibility_check_points` (start, path, goal, robot tops, 8 points on
    the ring at start and goal); room box points = the polygon at floor and stub
    height - what "the room fills the frame" is measured on AND what must be in frame
    with the looser `visibility.OUTLINE_MARGIN_FRAC` (the whole dollhouse is shown)."""
    platform = path_json["platform"]
    r = float(ring_radius_m if ring_radius_m is not None else platform["radius_m"])
    h = float(robot_height_m if robot_height_m is not None else platform["height_m"])
    sx, sz = path_json["start_xy"]
    path_pts = [tuple(float(v) for v in p) for p in path_json["path_points_world"]]
    start = (float(sx), bo.floor_y + PATH_HEIGHT_ABOVE_FLOOR_M, float(sz))
    goal = path_pts[-1]
    room_box = [(float(x), bo.floor_y, float(z)) for x, z in bo.room_polygon]
    room_box += [(float(x), bo.floor_y + wall_height_m, float(z)) for x, z in bo.room_polygon]
    frame_points = framing_check_points(start, path_pts, r, h, ground_axes=(0, 2), up_axis=1)
    vis_points = visibility_check_points(start, path_pts, goal, r, h, ground_axes=(0, 2), up_axis=1)
    return frame_points, vis_points, room_box


def presentation_camera(
    bo: BootstrapOutput, path_json: dict, *, image_size: tuple[int, int] = STILL_IMAGE_SIZE, fov_deg: float = FOV_DEG,
    wall_height_m: float = PRESENTATION_WALL_HEIGHT_M, max_eye_height_m: float | None = None,
) -> dict:
    """The T15i camera rule (replaces T15d's frustum-only framing + far-floor pitch):

    1. Position: `render_perspective.compute_camera`'s anchor/direction rule unchanged -
       the eye sits on the ray from the bed centroid (else room centroid) through the
       room polygon's farthest vertex, outside the polygon, at the rule's distance
       `max(0.7 * diagonal, 3 m)` from the room centroid, `CAMERA_HEIGHT_M` up.
    2. Aim: yaw AND pitch are solved so the projected bounding box of the room (its
       polygon at floor level and at the 1.2 m stub top) is CENTRED in the frame
       (`visibility.centre_points_in_frame`). The T15d far-floor-at-85 % pitch rule
       saturated at -1 deg on the hero (the horizon at frame centre, the room in the
       lower 40 %); centring the room box is what makes the room fill the picture.
    3. Coverage: the bbox must span >= ROOM_FRAME_COVERAGE_MIN (60 %) of the frame
       width and <= ROOM_FRAME_COVERAGE_MAX (90 %): the distance is stepped closer
       (never within CAMERA_MIN_OUTSIDE_ROOM_M of the polygon) or farther in
       CAMERA_DISTANCE_STEP_M steps until it does.
    4. Dollhouse cull: the outline edges whose line separates the eye from the room
       centroid are culled (`export_usd.culled_stub_edge_ids` - the web viewer's
       rule; for a convex room these are the edges facing the eye, for the hero's
       doorway notch also the notch wall the eye looks across); the OTHER edges' stub
       boxes plus every non-translucent object placeholder part's oriented box are
       the occluders.
    5. Visibility (`visibility.solve_visible_camera`, shared with record_isaac's
       worker): every check point (route start, path points, goal, robot tops at
       start/goal, 8 ring points at each) must be unoccluded AND every frame point
       (T15d's set + the room polygon at floor and stub height) inside the frame with
       FRAMING_MARGIN_FRAC per side; otherwise the eye is raised in 0.25 m steps to
       4.5 m (re-aimed at the same floor point), then backed off along the ray in
       0.1 m steps and the heights retried. Steps 4-5 iterate until the cull list is
       stable for the solved eye (the cull depends on the eye's XZ).
    The frame is the Isaac one (16:9, `image_size`); the returned dict carries every
    number for the summary, the USD camera attrs and record_isaac."""
    room_pts = np.asarray(bo.room_polygon, dtype=float)
    room_poly = Polygon(room_pts).buffer(0)
    bed = _bed_centroid_xz(bo.scene)
    pos0, _fwd0, _r0, _u0 = compute_camera(room_pts, bo.floor_y, bed_centroid_xz=bed, fov_deg=fov_deg, image_height_px=image_size[1])
    cx, cz = _room_centroid(room_pts)
    direction = np.array([pos0[0] - cx, pos0[2] - cz], dtype=float)
    d_rule = float(np.hypot(*direction))
    direction = direction / d_rule if d_rule > 1e-9 else np.array([1.0, 0.0])
    aspect = image_size[0] / image_size[1]
    eye_height = float(pos0[1] - bo.floor_y)  # CAMERA_HEIGHT_M unless compute_camera changes

    frame_points, vis_points, room_box = presentation_check_points(bo, path_json, wall_height_m=wall_height_m)
    stub_records = presentation_stub_records(bo, wall_height_m=wall_height_m)
    object_boxes = object_part_occluder_boxes(bo.scene, bo.floor_y)
    # Morning-4: the ribbon's points and a robot box (every demo platform, oriented
    # along the route) at every 0.25 m of the route - worst case must keep >= 95 %.
    path_pts = [tuple(float(v) for v in p) for p in path_json["path_points_world"]]
    ribbon_points = ribbon_sample_points(path_pts, floor_level=bo.floor_y, up_axis=1)
    platforms = [asdict(resolve_platform(pid)) for pid in DEMO_PLATFORM_IDS]
    box_sets_by_platform = {p["id"]: route_robot_box_sets(path_pts, [p], floor_level=bo.floor_y, up_axis=1) for p in platforms}
    robot_box_sets = [b for p in platforms for b in box_sets_by_platform[p["id"]]]
    # Addendum: the eye keeps >= CAMERA_MIN_CLEARANCE_M from every prim's AABB (walls,
    # floor, every object part and hull) - the worker fails a run that violates it.
    clearance_ranges = scene_clearance_ranges(bo.scene)

    direction_rule = direction.copy()

    def pose_at(d: float, direction):
        eye = (float(cx + direction[0] * d), float(bo.floor_y + eye_height), float(cz + direction[1] * d))
        forward, bbox = centre_points_in_frame(eye, room_box, fov_deg, aspect, up_axis=1)
        coverage = (bbox[2] - bbox[0]) / 2.0 if bbox is not None else 0.0
        return eye, forward, coverage

    route_mid = np.asarray(ribbon_points, dtype=float)[:, [0, 2]].mean(axis=0) if ribbon_points else np.array([cx, cz])

    def solve_for_direction(direction, *, box_sets, max_height_m, max_backoff_m, move_in_max_m, min_width_frac, fixed_culled=None,
                            transition_from=None, aim_at_route: bool = False):
        # --- 3. distance for the coverage band ---------------------------------------
        d = d_rule
        eye, forward, coverage = pose_at(d, direction)
        coverage_at_rule = coverage
        if coverage < ROOM_FRAME_COVERAGE_MIN:
            while coverage < ROOM_FRAME_COVERAGE_MIN:
                d_try = d - CAMERA_DISTANCE_STEP_M
                eye_try, forward_try, cov_try = pose_at(d_try, direction)
                if d_try <= 0 or room_poly.distance(Point(eye_try[0], eye_try[2])) < CAMERA_MIN_OUTSIDE_ROOM_M:
                    break
                d, eye, forward, coverage = d_try, eye_try, forward_try, cov_try
        elif coverage > ROOM_FRAME_COVERAGE_MAX:
            while coverage > ROOM_FRAME_COVERAGE_MAX and d < d_rule + 30.0:
                d += CAMERA_DISTANCE_STEP_M
                eye, forward, coverage = pose_at(d, direction)
        aim = aim_point_on_floor(eye, forward, bo.floor_y, up_axis=1, fallback_distance_m=d)
        if aim_at_route:
            # the Husky-pass pose: aim at the route's midpoint, eye on the same azimuth ray
            # from THAT point (the solver's move-in then brings the eye above the route);
            # the whole outline need not be in frame, only the room-width rule holds.
            aim = (float(route_mid[0]), float(bo.floor_y), float(route_mid[1]))
            eye = (float(route_mid[0] + direction[0] * d), float(bo.floor_y + eye_height), float(route_mid[1] + direction[1] * d))

        # --- 4 + 5. cull for the eye, solve, re-cull for the solved eye --------------
        culled = list(fixed_culled) if fixed_culled is not None else culled_stub_edge_ids(stub_records, (eye[0], eye[2]), (cx, cz))
        solved = None
        cull_iterations = 0
        boxes = []
        for _ in range(CAMERA_CULL_ITERATIONS):
            cull_iterations += 1
            boxes = [stub_edge_occluder_box(rec, f"stub_edge_{rec['edge']}") for rec in stub_records if rec["edge"] not in set(culled)]
            boxes += object_boxes
            solved = solve_visible_camera(
                eye, aim, fov_deg, aspect, frame_points, vis_points, boxes, up_axis=1, floor_level=bo.floor_y, outline_points=room_box,
                ribbon_points=ribbon_points, robot_box_sets=box_sets, path_visibility_min_fraction=PATH_VISIBILITY_MIN_FRACTION,
                clearance_ranges=clearance_ranges, min_eye_clearance_m=CAMERA_MIN_CLEARANCE_M, max_backoff_m=max_backoff_m,
                max_height_above_floor_m=max_height_m, move_in_max_m=move_in_max_m, min_outline_width_frac=min_width_frac,
                transition_from=transition_from, outline_frame_required=not aim_at_route,
                min_aim_distance_m=0.0 if aim_at_route else 0.5,  # the Husky pose may sit directly above the route midpoint
            )
            if fixed_culled is not None:
                break
            new_culled = culled_stub_edge_ids(stub_records, (solved["eye"][0], solved["eye"][2]), (cx, cz))
            if new_culled == culled:
                break
            culled = new_culled
        assert solved is not None
        return {"solved": solved, "eye0": eye, "aim": aim, "d": d, "coverage_at_rule": coverage_at_rule, "culled": culled,
                "cull_iterations": cull_iterations, "boxes": boxes}

    def sweep_azimuths(**kw):
        # Morning-4: the bed-anchored azimuth first; if no height/back-off there passes
        # (a camera looking ALONG the route sees the ribbon behind the robot's body
        # hidden), rotate the eye about the room centroid in CAMERA_AZIMUTH_STEPS_DEG and
        # take the first azimuth that passes (else the best path-visibility fraction).
        attempts = []
        chosen = None
        for az_off in CAMERA_AZIMUTH_STEPS_DEG:
            a = math.radians(az_off)
            direction = np.array([direction_rule[0] * math.cos(a) - direction_rule[1] * math.sin(a),
                                  direction_rule[0] * math.sin(a) + direction_rule[1] * math.cos(a)])
            att = solve_for_direction(direction, **kw)
            pv = att["solved"].get("path_visibility") or {}
            attempts.append({"azimuth_offset_deg": az_off, "ok": bool(att["solved"]["ok"]), "path_visibility": pv.get("fraction"),
                             "framing_ok": bool(att["solved"]["framing_ok"]), "visibility_ok": bool(att["solved"]["visibility"]["ok"]),
                             "height_above_floor_m": att["solved"]["height_above_floor_m"], "backoff_m": att["solved"]["backoff_m"],
                             "pitch_deg": att["solved"]["pitch_deg"], "n_candidates": att["solved"]["n_candidates"]})
            att["azimuth_offset_deg"] = az_off
            if chosen is None:
                chosen = att
            elif not chosen["solved"]["ok"]:
                key_new = (att["solved"]["ok"], att["solved"]["framing_ok"], att["solved"]["visibility"]["ok"], pv.get("fraction") or 0.0)
                pvc = chosen["solved"].get("path_visibility") or {}
                key_old = (chosen["solved"]["ok"], chosen["solved"]["framing_ok"], chosen["solved"]["visibility"]["ok"], pvc.get("fraction") or 0.0)
                if key_new > key_old:
                    chosen = att
            if chosen["solved"]["ok"]:
                break
        chosen["attempts"] = attempts
        return chosen

    def sweep_with_coverage_relax(**kw):
        # the room must span >= ROOM_FRAME_COVERAGE_MIN of the frame width; only if no
        # pose passes at all is it relaxed to ROOM_FRAME_COVERAGE_RELAXED_MIN
        levels = []
        for min_width in (ROOM_FRAME_COVERAGE_MIN, ROOM_FRAME_COVERAGE_RELAXED_MIN):
            res = sweep_azimuths(min_width_frac=min_width, **kw)
            res["min_width_frac"] = min_width
            levels.append(res)
            if res["solved"]["ok"]:
                return res
        return max(levels, key=lambda r: ((r["solved"].get("path_visibility") or {}).get("fraction") or 0.0))

    # --- pose A: ONE widened pose for the whole wide window (both platforms) ----------
    wide_limits = dict(max_height_m=float(max_eye_height_m if max_eye_height_m is not None else WIDE_MAX_EYE_HEIGHT_M),
                       max_backoff_m=WIDE_MAX_BACKOFF_M, move_in_max_m=WIDE_MAX_MOVE_IN_M)
    single = sweep_with_coverage_relax(box_sets=robot_box_sets, **wide_limits)
    two_pose = not single["solved"]["ok"]
    husky_pose = None
    if two_pose:
        # --- pose B: the 3/4 pose for establish + Burger's pass (Burger's box only), and
        # a steeper pose for Husky's pass (Husky's box only, the SAME kept stubs as the
        # USD renders, the straight move from the Burger eye clearance-checked).
        burger_id, husky_id = DEMO_PLATFORM_IDS[0], DEMO_PLATFORM_IDS[1]
        chosen = sweep_with_coverage_relax(
            box_sets=box_sets_by_platform[burger_id], max_height_m=MAX_EYE_HEIGHT_ABOVE_FLOOR_M, max_backoff_m=PRESENTATION_MAX_BACKOFF_M,
            move_in_max_m=0.0,
        )
        husky_pose = sweep_with_coverage_relax(
            box_sets=box_sets_by_platform[husky_id], fixed_culled=chosen["culled"], transition_from=chosen["solved"]["eye"],
            aim_at_route=True, **wide_limits,
        )
    else:
        chosen = single
    solved, eye, aim, d, coverage_at_rule, culled, cull_iterations, boxes = (
        chosen["solved"], chosen["eye0"], chosen["aim"], chosen["d"], chosen["coverage_at_rule"], chosen["culled"],
        chosen["cull_iterations"], chosen["boxes"],
    )
    attempts = chosen["attempts"]

    def pose_record(att, *, role: str, platforms_used, limits: dict) -> dict:
        s = att["solved"]
        f, r, u = s["forward"], s["right"], s["up"]
        pos_after = np.asarray(s["eye"], dtype=float)
        bbox = projected_bbox(room_box, s["eye"], f, r, u, fov_deg, aspect)
        final_coverage = (bbox[2] - bbox[0]) / 2.0 if bbox is not None else 0.0
        bbox_centre = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0] if bbox is not None else None
        return {
            "rule": (
                "T15i: compute_camera anchor/direction ray; yaw+pitch centre the room bbox; distance for 60-90 % frame width; "
                "dollhouse-culled stub edges + opaque object parts as occluders; visibility.solve_visible_camera (heights in 0.25 m "
                "steps, back-off / move-in in 0.1 m steps, azimuth sweep) over start/path/goal/robot tops/rings + the path ribbon "
                "with a robot box at every 0.25 m of the route (covered points excluded) + 0.3 m eye clearance"
            ),
            "pose_role": role,
            "pos": [float(v) for v in pos_after],
            "pos_before_backoff": [float(v) for v in att["eye0"]],
            "forward": [float(v) for v in f], "right": [float(v) for v in r], "up": [float(v) for v in u],
            "aim": [float(v) for v in att["aim"]],
            "fov_v_deg": float(fov_deg),
            "aspect": float(aspect),
            "pitch_deg": float(s["pitch_deg"]),
            "distance_m": float(math.hypot(pos_after[0] - cx, pos_after[2] - cz)),
            "distance_rule_m": d_rule,
            "distance_before_backoff_m": float(att["d"]),
            "coverage_distance_adjust_m": float(att["d"] - d_rule),
            "height_above_floor_m": float(s["height_above_floor_m"]),
            "height_raise_m": float(s["height_raise_m"]),
            "backoff_m": float(s["backoff_m"]),
            "framing_margin_frac": FRAMING_MARGIN_FRAC,
            "outline_margin_frac": OUTLINE_MARGIN_FRAC,
            "framing_points_checked": len(frame_points),
            "outline_points_checked": len(room_box),
            "framing_ok": bool(s["framing_ok"]),
            "visibility_points_checked": len(vis_points),
            "visibility_ok": bool(s["visibility"]["ok"]),
            "visibility": s["visibility"],
            "path_visibility_worst_case": s["path_visibility"],
            "path_visibility_ok": bool(s["path_visibility"]["ok"]) if s.get("path_visibility") else None,
            "path_visibility_min_fraction": PATH_VISIBILITY_MIN_FRACTION,
            "n_ribbon_points": len(ribbon_points),
            "n_route_robot_placements": sum(len(box_sets_by_platform[p]) for p in platforms_used),
            "route_robot_platforms": list(platforms_used),
            "eye_clearance": s["eye_clearance"],
            "min_eye_clearance_m": CAMERA_MIN_CLEARANCE_M,
            "camera_ok": bool(s["ok"]),
            "azimuth_offset_deg": float(att["azimuth_offset_deg"]),
            "azimuth_attempts": att["attempts"],
            "search_limits": {**limits, "min_width_frac": att["min_width_frac"]},
            "n_candidates_tried": int(s["n_candidates"]),
            "culled_stub_edges": [int(i) for i in att["culled"]],
            "cull_iterations": att["cull_iterations"],
            "n_stub_edges": len(stub_records),
            "occluders": att["boxes"],
            "n_occluders": len(att["boxes"]),
            "room_frame_coverage": float(final_coverage),
            "room_frame_coverage_at_rule_distance": float(att["coverage_at_rule"]),
            "room_frame_coverage_ok": bool(final_coverage >= att["min_width_frac"] - 1e-9),
            "room_bbox_ndc": [float(v) for v in bbox] if bbox is not None else None,
            "room_bbox_centre_ndc": bbox_centre,
            "bed_centroid_xz": [float(v) for v in bed] if bed is not None else None,
            "room_centroid_xz": [float(cx), float(cz)],
            "camera_outside_room_polygon": bool(not room_poly.contains(Point(pos_after[0], pos_after[2]))),
        }

    main_platforms = [p["id"] for p in platforms] if not two_pose else [DEMO_PLATFORM_IDS[0]]
    main_limits = wide_limits if not two_pose else dict(max_height_m=MAX_EYE_HEIGHT_ABOVE_FLOOR_M, max_backoff_m=PRESENTATION_MAX_BACKOFF_M, move_in_max_m=0.0)
    out = pose_record(chosen, role="single_wide" if not two_pose else "establish_and_burger", platforms_used=main_platforms, limits=main_limits)
    out["two_pose"] = two_pose
    out["single_pose_attempt"] = None if not two_pose else {
        "path_visibility": (single["solved"].get("path_visibility") or {}).get("fraction"), "azimuth_offset_deg": single["azimuth_offset_deg"],
        "height_above_floor_m": single["solved"]["height_above_floor_m"], "pitch_deg": single["solved"]["pitch_deg"],
        "min_width_frac": single["min_width_frac"], "attempts": single["attempts"],
    }
    if husky_pose is not None:
        hp = pose_record(husky_pose, role="husky_pass", platforms_used=[DEMO_PLATFORM_IDS[1]], limits=wide_limits)
        hp["transition_from_pos"] = out["pos"]
        hp["transition_clearance_ok"] = hp["eye_clearance"]["ok"]
        out["camera_husky"] = hp
    else:
        out["camera_husky"] = None
    return out


TOP_DOWN_FOV_DEG = 30.0  # T15i: narrower than the 3/4 shot's 55 deg -> the eye sits ~2x higher and the
# 1.2 m stubs' inner faces no longer smear into big slanted panels around the plan (they did at 55 deg / 5 m)


def top_down_camera(bo: BootstrapOutput, path_json: dict, *, image_size=STILL_IMAGE_SIZE, fov_deg: float = TOP_DOWN_FOV_DEG) -> dict:
    pts = np.vstack([np.asarray(bo.room_polygon, dtype=float), [(p[0], p[2]) for p in path_json["path_points_world"]]])
    cx, cz = _room_centroid(np.asarray(bo.room_polygon, dtype=float))
    half_x = max(abs(pts[:, 0] - cx).max(), 0.5)
    half_z = max(abs(pts[:, 1] - cz).max(), 0.5)
    tan_v = math.tan(math.radians(fov_deg) / 2.0)
    aspect = image_size[0] / image_size[1]
    height = TOP_DOWN_MARGIN * max(half_z / tan_v, half_x / (tan_v * aspect))
    pos = np.array([cx, bo.floor_y + height, cz])
    fwd, right, up = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    sx, sz = path_json["start_xy"]
    platform = path_json["platform"]
    points = framing_check_points(
        (sx, bo.floor_y, sz), [tuple(p) for p in path_json["path_points_world"]], float(platform["radius_m"]),
        float(platform["height_m"]), ground_axes=(0, 2), up_axis=1,
    )
    backoff, pos_after = camera_backoff_for_framing(tuple(pos), tuple(fwd), tuple(right), tuple(up), fov_deg, aspect, points)
    return {"pos": [float(v) for v in pos_after], "forward": fwd.tolist(), "right": right.tolist(), "up": up.tolist(),
            "height_above_floor_m": float(pos_after[1] - bo.floor_y), "backoff_m": float(backoff), "fov_v_deg": float(fov_deg)}


def render_presentation_stills(bo: BootstrapOutput, path_json: dict, camera: dict, out_dir: Path, *,
                               wall_height_m: float = PRESENTATION_WALL_HEIGHT_M, image_size=STILL_IMAGE_SIZE) -> dict:
    """T15i faithful preview: the 3/4 still draws the SAME geometry Isaac gets (kept
    stub edges only, opaque; translucent curtains at 30 %) at the Isaac frame size;
    the top-down still keeps every edge (nothing to occlude from above)."""
    room_pts = np.asarray(bo.room_polygon, dtype=float)
    # each wide still draws the robot box at that pose's WORST-CASE placement on the route
    scene = build_presentation_scene(bo, path_json, wall_height_m=wall_height_m, culled_stub_edges=camera.get("culled_stub_edges"),
                                     worst_case_box=worst_case_robot_box(bo, path_json, camera))
    out_34 = Path(out_dir) / STILL_34_NAME
    _render_scene(
        scene, room_pts, bo.floor_y, out_34, image_size=image_size, fov_deg=camera.get("fov_v_deg", FOV_DEG),
        camera=(camera["pos"], camera["forward"], camera["right"], camera["up"]), opaque_walls=True,
    )
    out_husky = None
    husky = camera.get("camera_husky")
    if husky is not None:
        husky_scene = build_presentation_scene(bo, path_json, wall_height_m=wall_height_m, culled_stub_edges=husky.get("culled_stub_edges"),
                                               worst_case_box=worst_case_robot_box(bo, path_json, husky))
        out_husky = Path(out_dir) / STILL_HUSKY_NAME
        _render_scene(
            husky_scene, room_pts, bo.floor_y, out_husky, image_size=image_size, fov_deg=husky.get("fov_v_deg", FOV_DEG),
            camera=(husky["pos"], husky["forward"], husky["right"], husky["up"]), opaque_walls=True,
        )
    top_scene = build_presentation_scene(bo, path_json, wall_height_m=wall_height_m)
    top = top_down_camera(bo, path_json, image_size=image_size)
    out_top = Path(out_dir) / STILL_TOP_NAME
    _render_scene(
        top_scene, room_pts, bo.floor_y, out_top, image_size=image_size, fov_deg=top["fov_v_deg"],
        camera=(top["pos"], top["forward"], top["right"], top["up"]), opaque_walls=True,
    )
    return {"still_34": str(out_34), "still_husky": str(out_husky) if out_husky else None, "still_top": str(out_top),
            "top_camera": top, "image_size": list(image_size)}


def worst_case_robot_box(bo: BootstrapOutput, path_json: dict, camera: dict) -> dict | None:
    """The oriented robot box (visibility box, Y-up) at the pose's worst-case placement
    (`path_visibility_worst_case.worst`): that platform standing on the route at that
    sample, heading along the route - drawn in the still so the hidden ribbon (if any)
    is what the eye sees."""
    pv = camera.get("path_visibility_worst_case") or {}
    worst = pv.get("worst")
    if not worst:
        return None
    plat = asdict(resolve_platform(worst["platform"]))
    path_pts = [tuple(float(v) for v in p) for p in path_json["path_points_world"]]
    sets = route_robot_box_sets(path_pts, [plat], floor_level=bo.floor_y, up_axis=1)
    for rec in sets:
        if rec["sample"] == worst["sample"]:
            return rec["box"]
    return None


# --- orchestration / CLI ------------------------------------------------------------


def run(bootstrap_out: Path, scene_dir: Path, *, target: str = DEFAULT_TARGET, platform_id: str = DEFAULT_PLATFORM,
        out_dir: Path | None = None, wall_height_m: float = PRESENTATION_WALL_HEIGHT_M, image_size=STILL_IMAGE_SIZE,
        textures_json: Path | None = None, min_route_length_m: float = MIN_ROUTE_LENGTH_M, require_corridor: bool = True,
        max_eye_height_m: float | None = None, assets="auto", assets_placed: Path | None = None) -> dict:
    # M7: the presentation shows the SAME assets as the web GLB - the still and
    # scene_presentation.glb draw scene_assets.glb's visuals, the USD references
    # the converted assets with the same placement (export_presentation_usd).
    assets_glb, asset_placements = resolve_assets(bootstrap_out, assets, assets_placed)
    if assets_glb is None:
        print("[export_presentation] no scene_assets.glb + assets_placed.json in the bootstrap dir - objects stay placeholder boxes "
              "(run scripts.msa.assemble_with_generated --assets first)")
    bo = load_bootstrap_output(bootstrap_out, scene_glb=assets_glb)
    out_dir = Path(out_dir or bootstrap_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    platform = resolve_platform(platform_id)
    textures = None
    if textures_json is not None:
        # T15c-prep: a `textures.json` from `python -m scripts.msa.textures` - the
        # baked PNGs get bound to /World/Floor + the wall stubs in the USD below.
        from scripts.msa.textures import load_textures_json

        textures = load_textures_json(textures_json)

    path_json = plan_presentation_path(bo, Path(scene_dir), target, platform, min_route_length_m=min_route_length_m,
                                       require_corridor=require_corridor)
    camera = presentation_camera(bo, path_json, image_size=image_size, wall_height_m=wall_height_m, max_eye_height_m=max_eye_height_m)
    # T15i: the camera's verdict rides in path.json too (the culled edges, the
    # per-point visibility, the frame coverage) - the occluder boxes are in the USD.
    camera_husky = camera.get("camera_husky")
    path_json["presentation_camera"] = {k: v for k, v in camera.items() if k not in ("occluders", "camera_husky")}
    path_json["presentation_camera_husky"] = {k: v for k, v in camera_husky.items() if k != "occluders"} if camera_husky else None
    (out_dir / "path.json").write_text(json.dumps(path_json, indent=2))

    usd_summary = export_presentation_usd(
        bo.scene, bo.floor_y, bo.ceiling_y, out_dir / "scene_presentation.usd",
        floor_polygon=bo.room_polygon,
        path_points_world=[tuple(p) for p in path_json["path_points_world"]],
        target_point=tuple(path_json["target_point"]),
        target_object_point=tuple(path_json["target_object_point"]),
        target_object_id=path_json["target_id"],
        camera=camera, wall_visual_height_m=wall_height_m,
        room_centroid_xz=tuple(camera["room_centroid_xz"]),
        textures=textures,
        wall_thickness_m=wall_thickness_of(bo),
        culled_stub_edge_ids=camera["culled_stub_edges"],
        camera_husky=camera_husky,
        asset_placements=asset_placements,
    )
    # M7 assertion: the presentation USD holds the same objects (count + sorted
    # class list) as the GLB it was built from, and every object that got an
    # asset has a real referenced mesh, not a placeholder box.
    from scripts.msa.usd_assets import compare_glb_usd_objects, placements_by_id

    assets_check = compare_glb_usd_objects(
        assets_glb or (bootstrap_out / "scene.glb"), out_dir / "scene_presentation.usd", list(placements_by_id(asset_placements)),
    )
    (out_dir / "presentation_assets_check.json").write_text(json.dumps(assets_check, indent=2))
    print(f"[export_presentation] GLB/USD identity: ok={assets_check['ok']} objects {assets_check['glb_object_count']}/{assets_check['usd_object_count']}, "
          f"classes identical={assets_check['classes_identical']}, assets checked={assets_check['n_placed_checked']}, "
          f"placed without real mesh={len(assets_check['placed_without_real_mesh'])}")
    if not assets_check["ok"]:
        raise SystemExit(f"scene_presentation.usd disagrees with {assets_glb or 'scene.glb'} - see {out_dir / 'presentation_assets_check.json'}")
    stills = render_presentation_stills(bo, path_json, camera, out_dir, wall_height_m=wall_height_m, image_size=image_size)
    # Morning-4: the same ribbon in a GLB - the bootstrap scene + `Path` (ribbon) /
    # `Path_curve` / `Target`, for the web viewer.
    glb_scene = bo.scene.copy()
    ribbon = path_ribbon_mesh(path_json["path_points_world"], bo.floor_y)
    if ribbon is not None:
        glb_scene.add_geometry(ribbon, node_name="Path")
    glb_scene.add_geometry(trimesh.load_path(np.asarray(path_json["path_points_world"], dtype=float)), node_name="Path_curve")
    target_sphere = trimesh.creation.icosphere(radius=0.1)
    target_sphere.apply_translation(path_json["target_point"])
    glb_scene.add_geometry(target_sphere, node_name="Target")
    glb_path = out_dir / "scene_presentation.glb"
    export_glb(glb_scene, glb_path)
    camera_summary = {k: v for k, v in camera.items() if k not in ("occluders", "camera_husky")}
    camera_summary["camera_husky"] = {k: v for k, v in camera_husky.items() if k != "occluders"} if camera_husky else None
    return {
        "bootstrap_out": str(bootstrap_out), "scene_dir": str(scene_dir), "out_dir": str(out_dir),
        "path_json": str(out_dir / "path.json"),
        "target_id": path_json["target_id"], "n_waypoints": path_json["n_waypoints"],
        "path_length_m": path_json["path_length_m"], "start_xy": path_json["start_xy"],
        "target_point": path_json["target_point"], "target_object_point": path_json["target_object_point"],
        "goal_to_target_object_centroid_m": path_json["goal_to_target_object_centroid_m"],
        "route_quality": path_json["route_quality"],
        "camera": camera_summary, "usd": usd_summary, "stills": stills, "glb": str(glb_path),
        "assets_glb": str(assets_glb) if assets_glb else None,
        "n_asset_placements": len(placements_by_id(asset_placements)),
        "assets_check": {k: v for k, v in assets_check.items() if k != "usd_objects"},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap-out", type=Path, required=True, help="bootstrap output dir (scene.glb + scene_meta.json)")
    ap.add_argument("--scene-dir", type=Path, required=True, help="RAW scene dir (occupancy.npy + occupancy_meta.json)")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="target object class or exact id (default: desk)")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM, help="app/robots.py platform id (default: burger)")
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write path.json / scene_presentation.usd / stills (default: --bootstrap-out)")
    ap.add_argument("--wall-height", type=float, default=PRESENTATION_WALL_HEIGHT_M, help="visible wall stub height above the floor, m")
    ap.add_argument("--textures", type=Path, default=None, help="textures.json from `python -m scripts.msa.textures` - bind its baked PNGs in the USD (T15c-prep)")
    ap.add_argument("--min-route-length", type=float, default=MIN_ROUTE_LENGTH_M,
                    help=f"morning-3: the demo route must be at least this long (default {MIN_ROUTE_LENGTH_M} m) - the planner fails loudly otherwise")
    ap.add_argument("--no-corridor-check", dest="require_corridor", action="store_false", default=True,
                    help="morning-3: do NOT require the route to pass through the corridor between the target and the room boundary")
    ap.add_argument("--max-eye-height", type=float, default=None,
                    help=f"wide-shot eye height cap above the floor (default {WIDE_MAX_EYE_HEIGHT_M} m; the hero's Husky pose needs ~7.4 m for the 95 %% ribbon bar)")
    ap.add_argument("--assets-glb", type=Path, default=None,
                    help=f"M7: assembled GLB whose visuals (real assets) the presentation shows (default: <bootstrap-out>/{ASSETS_GLB_NAME} when present)")
    ap.add_argument("--assets-placed", type=Path, default=None, help=f"M7: {ASSETS_PLACED_NAME} for the USD asset references (default: next to the assets GLB)")
    ap.add_argument("--no-assets", action="store_true", help="M7: ignore scene_assets.glb and keep placeholder boxes")
    args = ap.parse_args(argv)
    assets = None if args.no_assets else (args.assets_glb or "auto")
    summary = run(args.bootstrap_out, args.scene_dir, target=args.target, platform_id=args.platform,
                  out_dir=args.out_dir, wall_height_m=args.wall_height, textures_json=args.textures,
                  min_route_length_m=args.min_route_length, require_corridor=args.require_corridor,
                  max_eye_height_m=args.max_eye_height, assets=assets, assets_placed=args.assets_placed)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
