#!/usr/bin/env python3
"""Stage A - deterministic bootstrap (SPEC.md §4). Reads one scene's occupancy
grid + scene_objects.json + floor_y/ceiling_y and writes s0: GLB, USD, DXF, SVG,
gaps.json and a small report, all under `<out_dir>/`.

CLI: `uv run python -m scripts.msa.bootstrap --scene-dir <path> --out-dir <path>`
`--scene-dir` must contain occupancy.npy, occupancy_meta.json and
scene_objects/scene_objects.json in the same layout `gpu/job_io.py` writes
(see app/services/scene_service.scene_dir for the production location).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scripts.msa.assets import mean_color_uint8
from scripts.msa.gaps import Gap, ObstacleSource, compute_gaps, gaps_to_json
from scripts.msa.geometry import (
    FREE,
    OBSTACLE,
    ROOM_NOTCH_M,
    ROOM_POLYGON_METHOD,
    UNKNOWN,
    WALL_THICKNESS_M,
    compute_dominant_wall_yaw,
    compute_room_polygon_yaw,
    compute_wall_hull_yaw,
    convex_hull_2d,
    extract_room_polygon,
    extract_wall_polygons,
    oriented_min_area_rect,
    regularize_room_polygon,
    room_wall_band,
    rotate_point_xz,
    rotate_wall_polygon,
    simplify_polygon_preserving_passages,
)

OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M = 2.0  # mirrors app/services/usd_export.py's rule (SPEC A3)
# T15a room polygon: an object whose bbox bottom sits within this height above the
# floor is "floor-standing" - its footprint is part of the room floor and is unioned
# into the room polygon (see geometry.extract_room_polygon). Anything higher (a
# wall-mounted TV, a lamp on a shelf) says nothing about where the floor is.
FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M = 0.15


def _default_platforms() -> dict[str, float]:
    """Platform diameters (m), from app/robots.py where importable; a small
    built-in fallback otherwise so this module works standalone in tests without
    the full FastAPI app config (DB settings) needing to load."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from app.robots import ROBOTS

        return {p.id: (p.radius_m or 0.0) * 2 for p in ROBOTS if p.radius_m}
    except Exception:
        return {"burger": 0.20, "go2": 0.40}


class ObjectFootprintInput:
    """Decoupled from PLY I/O so tests can supply pre-extracted hull points
    directly (see tests/fixtures/msa/02_modular_home/README.md)."""

    def __init__(self, id: str, label: str, hull_xz: np.ndarray, bbox_min: list[float], bbox_max: list[float], color_rgb: tuple[int, int, int],
                 point_count: int | None = None):
        self.id = id
        self.label = label
        self.hull_xz = hull_xz
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.color_rgb = color_rgb
        # M7: the detection's real point count (scene_objects.json `point_count`) -
        # the support the duplicate merge and the class caps rank by. None when the
        # input has no count (the hull-only test fixture); those stages then fall
        # back to hull area x height.
        self.point_count = point_count


def load_object_inputs_from_scene_dir(scene_dir: Path, *, hull_sample_max: int = 20000) -> list[ObjectFootprintInput]:
    """Production loader: reads scene_objects.json + each object's own PLY point
    cloud (SPEC A3 wants hull points, not just the bbox)."""
    from scripts.msa.ply_io import read_ply_xyz_rgb

    objects_json = json.loads((scene_dir / "scene_objects" / "scene_objects.json").read_text())
    inputs = []
    rng = np.random.default_rng(0)
    for obj in objects_json:
        ply_path = scene_dir / obj["point_cloud_path"]
        xyz, rgb = read_ply_xyz_rgb(ply_path)
        if len(xyz) > hull_sample_max:
            idx = rng.choice(len(xyz), size=hull_sample_max, replace=False)
            xyz_sample, rgb_sample = xyz[idx], rgb[idx]
        else:
            xyz_sample, rgb_sample = xyz, rgb
        hull_xz = xyz_sample[:, [0, 2]]
        inputs.append(
            ObjectFootprintInput(
                id=obj["id"],
                label=obj["label"],
                hull_xz=hull_xz,
                bbox_min=obj["bbox_min"],
                bbox_max=obj["bbox_max"],
                color_rgb=mean_color_uint8(rgb_sample),
                point_count=int(obj.get("point_count") or len(xyz)),
            )
        )
    return inputs


def load_object_inputs_from_hulls_json(path: Path) -> list[ObjectFootprintInput]:
    """Test-fixture loader: reads a compact {id,label,hull_xz,bbox_min,bbox_max,
    color_rgb} JSON (precomputed convex-hull vertices, not full point clouds -
    see tests/fixtures/msa/02_modular_home/README.md for why)."""
    data = json.loads(path.read_text())
    return [
        ObjectFootprintInput(
            id=o["id"],
            label=o["label"],
            hull_xz=np.array(o["hull_xz"]),
            bbox_min=o["bbox_min"],
            bbox_max=o["bbox_max"],
            color_rgb=tuple(o["color_rgb"]),
            point_count=o.get("point_count"),
        )
        for o in data
    ]


WALL_BACKFILL_THRESHOLD_M = 0.10  # Tier 1: max gap to a wall we'll silently close


def _walls_union(walls):
    """Merge wall polygons into one shapely geometry for distance queries, or
    None if there are no walls (empty scene / all components dropped)."""
    if not walls:
        return None
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = [Polygon(w.vertices) for w in walls if len(w.vertices) >= 3]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [p for p in polys if not p.is_empty]
    if not polys:
        return None
    return unary_union(polys)


def _backfill_footprint_to_walls(
    center_xy: tuple[float, float],
    size_uv: tuple[float, float],
    angle_rad: float,
    walls_union,
    threshold_m: float = WALL_BACKFILL_THRESHOLD_M,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Wall-backfill Tier 1: an object visually flush against a wall (wardrobe,
    headboard, ...) often gets a measured footprint that falls slightly short of
    the actual wall, because the camera never captured the thin sliver of
    floor/object right against it. For each of the oriented rectangle's 4 edges,
    if that edge already comes within `threshold_m` of the wall polygon, extend
    *that edge only* outward (from wherever the visible surface currently ends)
    until it touches the wall - the opposite edge is left untouched. Returns the
    possibly-adjusted (center_xy, size_uv); never shrinks, never touches
    hull_xz (the measured collision hull, which stays exactly as scanned)."""
    if walls_union is None or getattr(walls_union, "is_empty", True):
        return center_xy, size_uv

    from shapely.geometry import Point

    cx, cz = center_xy
    length_u, width_v = size_uv
    u_hat = (np.cos(angle_rad), np.sin(angle_rad))
    v_hat = (-np.sin(angle_rad), np.cos(angle_rad))

    def edge_gap(axis_hat, half_axis, sign):
        # Gap from this edge's *midpoint* (not the whole segment) to the wall.
        # Using the midpoint - rather than the segment's min distance - matters:
        # a perpendicular side edge always has one corner sitting right next to
        # the near edge it shares with the wall-flush edge, which would falsely
        # read as "this side edge is also against the wall" under a whole-segment
        # distance. A wall-flush edge runs parallel to the wall, so its midpoint
        # distance is ~= its true gap; a perpendicular edge's midpoint stays far
        # from the wall even though its corner doesn't.
        mid = (cx + sign * half_axis * axis_hat[0], cz + sign * half_axis * axis_hat[1])
        dist = Point(mid).distance(walls_union)
        return dist if dist <= threshold_m else 0.0

    ext_u_pos = edge_gap(u_hat, length_u / 2, +1)
    ext_u_neg = edge_gap(u_hat, length_u / 2, -1)
    ext_v_pos = edge_gap(v_hat, width_v / 2, +1)
    ext_v_neg = edge_gap(v_hat, width_v / 2, -1)

    if not any((ext_u_pos, ext_u_neg, ext_v_pos, ext_v_neg)):
        return center_xy, size_uv

    new_length_u = length_u + ext_u_pos + ext_u_neg
    new_width_v = width_v + ext_v_pos + ext_v_neg
    shift_u = (ext_u_pos - ext_u_neg) / 2.0
    shift_v = (ext_v_pos - ext_v_neg) / 2.0
    new_center = (
        cx + shift_u * u_hat[0] + shift_v * v_hat[0],
        cz + shift_u * u_hat[1] + shift_v * v_hat[1],
    )
    return new_center, (new_length_u, new_width_v)


def compute_object_footprints(
    inputs: list[ObjectFootprintInput], floor_y: float, walls: list | None = None
) -> tuple[list[dict], list[str]]:
    """Returns (kept object dicts with oriented footprint + height, ids tagged overhead).

    `walls` (the `WallPolygon`s from `extract_wall_polygons`), if given, is used
    for wall-backfill Tier 1 (see `_backfill_footprint_to_walls`): a footprint
    edge already within `WALL_BACKFILL_THRESHOLD_M` of a wall is extended out to
    touch it. Pass `None` (default) to skip backfill entirely (e.g. callers that
    don't have wall polygons handy)."""
    walls_union = _walls_union(walls)
    kept, overhead_ids = [], []
    for obj in inputs:
        if obj.bbox_min[1] >= OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M:
            overhead_ids.append(obj.id)
            continue
        center_xy, size_uv, angle = oriented_min_area_rect(obj.hull_xz)
        center_xy, size_uv = _backfill_footprint_to_walls(center_xy, size_uv, angle, walls_union)
        height = max(obj.bbox_max[1] - obj.bbox_min[1], 0.01)
        point_count = getattr(obj, "point_count", None)
        kept.append(
            {
                "id": obj.id,
                "label": obj.label,
                "center_xy": center_xy,
                "size_uv": size_uv,
                "angle_rad": angle,
                "height": height,
                "bbox_min_y": obj.bbox_min[1],
                "color_rgb": obj.color_rgb,
                # M7: real detection support for dedupe/class_caps (`n_points` is the
                # field name both modules look for first); absent when unknown.
                **({"n_points": int(point_count)} if point_count is not None else {}),
                # Stage E0 collision prim: the *actual* measured convex hull, not
                # the oriented-rectangle footprint used for the visual placeholder
                # (SPEC §5: "Collision geometry ... remains the measured convex
                # hull, never the generated mesh"). Not wall-backfilled: it must
                # stay exactly what was scanned.
                "hull_xz": convex_hull_2d(obj.hull_xz).tolist(),
            }
        )
    return kept, overhead_ids


SMALL_OBJECT_AREA_M2 = 0.04  # SPEC: drop footprints smaller than this ...
SMALL_OBJECT_HEIGHT_M = 0.10  # ... or shorter than this ...
SMALL_OBJECT_TOUCH_TOLERANCE_M = 0.02  # ... unless "touching" something (see _filter_small_isolated_objects)


def _footprint_polygon(obj: dict):
    """The object's oriented-rectangle visual footprint (center_xy/size_uv/
    angle_rad), as a shapely polygon in world XZ - same corner construction as
    `export_dxf.build_floor_plan` and `rasterize_object_footprint`."""
    import math

    from shapely.geometry import Polygon

    cx, cz = obj["center_xy"]
    length_u, width_v = obj["size_uv"]
    angle = obj["angle_rad"]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u, v = length_u / 2, width_v / 2
    local_corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
    world_corners = [(cx + lx * cos_a - lz * sin_a, cz + lx * sin_a + lz * cos_a) for lx, lz in local_corners]
    return Polygon(world_corners)


def _filter_small_isolated_objects(objects: list[dict], walls_union) -> tuple[list[dict], list[str]]:
    """SPEC: drop an object whose footprint area is < `SMALL_OBJECT_AREA_M2` OR
    whose height is < `SMALL_OBJECT_HEIGHT_M`, UNLESS it's "touching the floor,
    a wall, or another object's footprint".

    Interpretation of "touching floor" (this is genuinely ambiguous in the
    spec - documenting the call made here): every object reaching this point
    already sits on the floor - overhead objects (ceiling fans, AC units) were
    already excluded above by `OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M`, and nothing
    else in this pipeline produces a floating footprint. So "touching floor" is
    vacuously true for this entire population and can't be the operative
    condition (a filter that never fires isn't a filter). We therefore read the
    clause as "touching floor" naming the *default* state objects are already
    in, with "wall" and "object" as the two ways something small can still be
    considered legitimately anchored to something real: the practical filter
    is "drop small/short objects UNLESS their footprint touches a wall or
    another object's footprint" - protecting a small item that's flush against
    a wall or wedged next to furniture (probably real), while dropping a tiny
    isolated blob floating alone in open floor space (probably detector/hull
    noise). If a distinct floor-height-band signal is ever added upstream
    (independent of the overhead exclusion), this would need revisiting.

    "Touching" uses `SMALL_OBJECT_TOUCH_TOLERANCE_M` (2cm) rather than exact
    contact (distance == 0), since two independently-measured footprints
    (object-to-object) will rarely land byte-exact flush even when physically
    adjacent - unlike wall-backfilled edges, which get numerically snapped to
    the wall by `_backfill_footprint_to_walls` and so are already ~0m away.
    2cm is deliberately tighter than the 10cm wall-backfill threshold: that
    threshold decides whether to *move* a footprint to close a plausible
    scan gap, whereas this only decides whether two already-placed footprints
    count as adjacent, a much less forgiving question.

    "Another object's footprint" is checked against every *other* object in
    the input list, independent of whether that other object itself survives
    this same filter - i.e. two mutually-adjacent small objects both survive
    (each is "touching an object"), rather than needing a fixed-point/graph
    resolution over which small objects survive first. This keeps the rule
    simple and order-independent.

    Returns (kept objects, dropped ids)."""
    polys = {obj["id"]: _footprint_polygon(obj) for obj in objects}
    kept, dropped_ids = [], []
    for obj in objects:
        length_u, width_v = obj["size_uv"]
        area = length_u * width_v
        if area >= SMALL_OBJECT_AREA_M2 and obj["height"] >= SMALL_OBJECT_HEIGHT_M:
            kept.append(obj)
            continue

        poly = polys[obj["id"]]
        touches_wall = walls_union is not None and not getattr(walls_union, "is_empty", True) and poly.distance(walls_union) <= SMALL_OBJECT_TOUCH_TOLERANCE_M
        touches_other = any(
            poly.distance(polys[other["id"]]) <= SMALL_OBJECT_TOUCH_TOLERANCE_M for other in objects if other["id"] != obj["id"]
        )
        if touches_wall or touches_other:
            kept.append(obj)
        else:
            dropped_ids.append(obj["id"])
    return kept, dropped_ids


OUTSIDE_ROOM_DROP_FRACTION = 0.5  # drop any prim whose footprint is more than this fraction outside the room polygon
# T15b: walls sit ON the room boundary by construction (the room polygon is the
# FREE region + floor-standing footprints; the OBSTACLE strip the occupancy band
# sees for a wall lies just outside it), so a wall polygon is judged against the
# room polygon buffered by about one wall thickness. 0.20 m = 4 cells at the
# pipeline's 5 cm resolution: interior partitions and the band-projected strip of
# a real wall are 1-4 cells thick, so a genuine wall lands fully inside the
# buffer, while an obstacle blob reaching more than ~20 cm past the free-space
# boundary is judged by how much of its area lies beyond that.
WALL_CLIP_MARGIN_M = 0.20
# T15b: doorway-spill cut (geometry.free_room_component). A standard interior
# door is 0.70-0.90 m clear, so a disk of diameter 0.95 m (radius 0.45 m = 9
# cells at 5 cm; the structuring element spans 19 cells) cannot pass through a
# doorway - the opening detaches everything scanned beyond a door - while it
# fits in any room (smallest room dimension is metres). See the T15b entry in
# docs/DECISIONS.md.
ROOM_OPENING_RADIUS_M = 0.45


def _largest_polygon_component(geom):
    """Extract the largest actual `Polygon` out of a shapely op result that may
    come back as a bare `Polygon`, a `MultiPolygon` (e.g. `floor_polygon`
    repaired via `buffer(0)`, or an intersection split by a concave floor
    boundary), or - rarely, when two polygons only touch along an edge/point -
    a `GeometryCollection` mixing in degenerate lines/points. Returns None if
    there's no polygonal piece with non-negligible area."""
    if geom is None or geom.is_empty:
        return None

    from shapely.geometry import Polygon

    if isinstance(geom, Polygon):
        return geom if geom.area > 1e-9 else None
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and g.area > 1e-9]
    if not parts:
        return None
    return max(parts, key=lambda p: p.area)


def _room_polygon_shape(floor_polygon: list[tuple[float, float]] | None):
    """The room reference polygon as one repaired shapely Polygon, or None."""
    if not floor_polygon or len(floor_polygon) < 3:
        return None
    from shapely.geometry import Polygon

    floor_poly = Polygon(floor_polygon)
    if not floor_poly.is_valid:
        floor_poly = floor_poly.buffer(0)
    return _largest_polygon_component(floor_poly)


def _drop_or_clip_walls_outside_room(
    walls: list,
    floor_polygon: list[tuple[float, float]] | None,
    *,
    dropped_records: list[dict] | None = None,
    clip_margin_m: float = WALL_CLIP_MARGIN_M,
) -> tuple[list, list[int]]:
    """T15b: the outside-room drop/clip rule (`_drop_or_clip_objects_outside_room`)
    applied to wall polygons. A wall's reference region is the room polygon
    buffered by `clip_margin_m` (`WALL_CLIP_MARGIN_M` - see there: walls sit on,
    i.e. just outside, the free-space boundary). Per wall: area fraction outside
    that region > `OUTSIDE_ROOM_DROP_FRACTION` -> removed; otherwise the polygon
    is intersected with the region (largest piece kept, `area_m2` becomes the
    clipped polygon's area) - an unclipped wall (fraction 0) is returned as-is.
    Returns (kept walls, indices into `walls` of the kept ones - the caller keeps
    the `wall_<i>` obstacle-source ids in step with the exported list). Dropped
    walls are appended to `dropped_records` as
    `{id, kind: "wall", outside_fraction, area_m2}`."""
    from shapely.geometry import Polygon

    room = _room_polygon_shape(floor_polygon)
    if room is None:
        return list(walls), list(range(len(walls)))
    region = room.buffer(clip_margin_m)

    kept, kept_indices = [], []
    for i, wall in enumerate(walls):
        wall_id = f"wall_{i}"
        poly = Polygon(wall.vertices) if len(wall.vertices) >= 3 else None
        if poly is not None and not poly.is_valid:
            poly = poly.buffer(0)
        poly = _largest_polygon_component(poly) if poly is not None else None
        if poly is None:
            kept.append(wall)  # degenerate - nothing to judge (exporters warn about it)
            kept_indices.append(i)
            continue
        outside_fraction = poly.difference(region).area / poly.area
        if outside_fraction > OUTSIDE_ROOM_DROP_FRACTION:
            if dropped_records is not None:
                dropped_records.append({"id": wall_id, "kind": "wall", "outside_fraction": float(outside_fraction), "area_m2": float(wall.area_m2)})
            continue
        if outside_fraction <= 1e-9:
            kept.append(wall)
            kept_indices.append(i)
            continue
        clipped = _largest_polygon_component(poly.intersection(region))
        if clipped is None:
            if dropped_records is not None:
                dropped_records.append({"id": wall_id, "kind": "wall", "outside_fraction": float(outside_fraction), "area_m2": float(wall.area_m2)})
            continue
        vertices = [(float(x), float(z)) for x, z in clipped.exterior.coords]
        kept.append(replace(wall, vertices=vertices, area_m2=float(clipped.area)))
        kept_indices.append(i)
    return kept, kept_indices


def _drop_or_clip_objects_outside_room(
    objects: list[dict],
    floor_polygon: list[tuple[float, float]] | None,
    *,
    dropped_records: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """Reflection/through-window guard: a footprint that mostly lies outside
    `floor_polygon` (the room reference polygon - since T15a
    `geometry.extract_room_polygon`'s output, which already contains every
    floor-standing object's footprint, so only elevated objects can ever be
    judged outside here) is very likely a mirror-reflection or through-a-window
    false detection -
    something the camera only "saw" because of a reflective or transparent
    surface, not because it's physically in the room - rather than a real
    object that merely extends a bit past the measured free-space edge (which
    wall-backfill and ordinary scan noise both do routinely).

    Rule per object, using its oriented-rectangle visual footprint (the same
    `_footprint_polygon` construction used everywhere else in this module):
      - fraction outside floor_polygon > 50%  -> drop the object entirely.
      - fraction outside floor_polygon <= 50% (including the common 0% case,
        fully inside) -> clip the footprint to floor_polygon and refit
        center_xy/size_uv/angle_rad to the clipped shape's oriented min-area
        rectangle (via `oriented_min_area_rect`, the same routine that built
        every footprint originally), so the object dict's schema stays an
        oriented rectangle rather than gaining a new arbitrary-polygon
        footprint type. A 0%-outside object is returned unchanged (no
        re-fit round-trip through shapely, which could otherwise perturb a
        fully-inside rectangle by float noise) rather than recomputed to a
        numerically-near-identical rectangle.

    Deliberately runs on `objects` *after* both `compute_object_footprints`'s
    wall-backfill and `_filter_small_isolated_objects`, i.e. on their final
    surviving/adjusted footprints, not the raw pre-backfill/pre-filter ones:
    wall-backfill explicitly extends footprint edges outward to touch a wall,
    which can push an already-boundary-hugging object further outside
    floor_polygon than the raw scan was, and the small-object filter changes
    which objects even reach this point at all. Judging "outside the room"
    against anything earlier in the pipeline would use footprints that no
    longer match what actually ships downstream.

    `hull_xz` (the measured 3D collision hull) is never touched here - same
    rule as wall-backfill: only the visual/gap-computation oriented-rectangle
    footprint changes, collision geometry stays exactly as scanned.

    Returns (kept objects - some with refit center_xy/size_uv/angle_rad,
    dropped ids). T15b: each dropped object is also appended to
    `dropped_records` (if given) as `{id, kind: "object", outside_fraction,
    area_m2}` for report.json's `dropped_prims`."""
    floor_poly = _room_polygon_shape(floor_polygon)
    if floor_poly is None:
        return objects, []

    def record_drop(obj: dict, outside_fraction: float, area: float) -> None:
        dropped_ids.append(obj["id"])
        if dropped_records is not None:
            dropped_records.append({"id": obj["id"], "kind": "object", "outside_fraction": float(outside_fraction), "area_m2": float(area)})

    kept, dropped_ids = [], []
    for obj in objects:
        footprint = _footprint_polygon(obj)
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        if footprint.is_empty or footprint.area < 1e-9:
            kept.append(obj)  # degenerate footprint - nothing to judge, leave as-is
            continue

        outside_fraction = footprint.difference(floor_poly).area / footprint.area
        if outside_fraction > OUTSIDE_ROOM_DROP_FRACTION:
            record_drop(obj, outside_fraction, footprint.area)
            continue
        if outside_fraction <= 1e-9:
            kept.append(obj)  # fully inside: intersection == original footprint, no refit needed
            continue

        clipped = _largest_polygon_component(footprint.intersection(floor_poly))
        if clipped is None:
            # Numerically clipped away to nothing despite the <=50% check above
            # (can happen right at the 50% boundary with sliver geometry) - no
            # floor-supported footprint is left, so drop it.
            record_drop(obj, outside_fraction, footprint.area)
            continue

        coords = np.array(clipped.exterior.coords[:-1])
        center_xy, size_uv, angle = oriented_min_area_rect(coords)
        new_obj = dict(obj)
        new_obj["center_xy"] = center_xy
        new_obj["size_uv"] = size_uv
        new_obj["angle_rad"] = angle
        kept.append(new_obj)

    return kept, dropped_ids


YAW_BIN_WIDTH_DEG = 2.0  # compute_dominant_wall_yaw's histogram bin width - see its docstring
# Secondary/diagnostic only (T3' rework): compute_dominant_wall_yaw's own confidence
# gate, still used to decide whether *its* result is trustworthy enough to log as a
# cross-check - no longer gates whether normalization is applied at all (that's now
# compute_room_polygon_yaw, which has no equivalent gate - see its docstring).
YAW_MIN_PEAK_FRACTION = 0.6
YAW_SQUARE_TOLERANCE = 0.02  # compute_room_polygon_yaw's degenerate-square threshold
# T3'' rework (see docs/DECISIONS.md): floor_polygon (FREE+UNKNOWN) turned out to be
# an unreliable yaw source - a doorway into the UNKNOWN halo that always touches the
# occupancy grid's own edges (gpu/stage_occupancy.py builds a percentile-trimmed
# bounding box) drags floor_polygon's bounding rect out to the grid array's own
# axis-aligned extent. The primary source is now extract_free_only_room_polygon
# (FREE cells only), cross-checked against compute_wall_hull_yaw (an independent
# estimate from the raw wall-vertex convex hull that never touches the occupancy
# grid at all). T15a (orchestrator decision, docs/DECISIONS.md): the cross-check no
# longer overrules the primary - the room polygon now contains the floor-standing
# furniture footprints (the strongest structural cue), while the wall hull can be a
# handful of tiny fragments. The threshold is kept only to flag `yaw_disagreement_deg`
# > threshold as `yaw_cross_check_disagrees` in the outputs (diagnostic).
YAW_DISAGREEMENT_THRESHOLD_DEG = 10.0


def _yaw_angle_disagreement_deg(a_deg: float, b_deg: float) -> float:
    """Smallest angular difference between two *undirected axis* angles, in
    [0, 90]. `oriented_min_area_rect`'s angle (which both `compute_room_polygon_yaw`
    and `compute_wall_hull_yaw` report as `long_axis_angle_deg`) describes a line,
    not a direction - it already lives in [-90, 90), so e.g. -89 and 89 describe
    axes only ~2 degrees apart (89 and -89+180=91 wrap to the same line), not the
    ~178 degrees a naive `abs(a - b)` would report. Folding the difference into
    (-90, 90] via mod 180 first, then taking abs, gives the true angular gap
    between the two lines."""
    return abs(((a_deg - b_deg + 90.0) % 180.0) - 90.0)


YAW_METHOD_WALL_HULL = "wall_hull_min_area_rect"
YAW_METHOD_ROOM_POLYGON = "room_polygon_min_area_rect"


def _reconcile_yaw_estimates(
    free_only_result: dict | None,
    wall_hull_result: dict | None,
    *,
    disagreement_threshold_deg: float = YAW_DISAGREEMENT_THRESHOLD_DEG,
) -> tuple[dict | None, bool, float | None]:
    """Morning-1 policy (user decision, 2026-09-07, docs/DECISIONS.md "morning-1/3/5" -
    replaces T15a's "polygon always"): the WALL HULL angle (`wall_hull_result`,
    `compute_wall_hull_yaw` over the pooled wall-vertex convex hull) is the APPLIED
    yaw for the whole scene - walls, room polygon, objects, gaps, cameras, the cloud
    transform `textures.py` reads from scene_meta.json. The room polygon's own
    min-area-rect angle (`free_only_result` - T3'' parameter name kept for the
    callers/meta keys) is a DIAGNOSTIC: on the hero the polygon (-56.76 deg) is
    dragged by the doorway spur while the walls (-50.53) sit on the furniture's
    axis, and applying the polygon left the bed ~5 deg and the nightstands 7-10 deg
    off-axis; the wall angle leaves the bed at ~1.4 deg.

    - `wall_hull_result` present and not degenerate-square -> apply it.
    - No walls (`wall_hull_result` None) or a degenerate (near-square) hull ->
      FALL BACK to `free_only_result` (logged by the caller as
      `yaw_fallback_to_polygon`); a degenerate polygon then applies its own 0
      correction; both missing -> nothing.

    Returns `(result_to_apply, fallback_to_polygon, disagreement_deg)`.
    `fallback_to_polygon` is True whenever the applied result is `free_only_result`
    (or nothing was applicable at all is False). `disagreement_deg`
    (`_yaw_angle_disagreement_deg`) is reported whenever both results exist and
    neither is degenerate - provenance only. `disagreement_threshold_deg` is kept
    for signature compatibility; the caller flags `yaw_cross_check_disagrees`."""
    del disagreement_threshold_deg  # diagnostic only - see docstring
    hull_usable = wall_hull_result is not None and not wall_hull_result["is_degenerate_square"]
    disagreement_deg = None
    if free_only_result is not None and wall_hull_result is not None and not free_only_result["is_degenerate_square"] and hull_usable:
        disagreement_deg = _yaw_angle_disagreement_deg(free_only_result["long_axis_angle_deg"], wall_hull_result["long_axis_angle_deg"])
    if hull_usable:
        return wall_hull_result, False, disagreement_deg
    if free_only_result is not None:
        return free_only_result, True, disagreement_deg
    # No polygon at all: a degenerate hull still carries its 0 correction (nothing to
    # rotate by, but the provenance fields get filled); no hull -> None.
    return wall_hull_result, False, None


def _yaw_rotation_center(floor_polygon: list[tuple[float, float]] | None, walls: list) -> tuple[float, float]:
    """Pivot for yaw normalization. Prefers `floor_polygon`'s centroid (the
    room's actual free-space shape, so the pivot sits inside the room even
    for an irregular floor plan) - falls back to the unweighted mean of every
    wall vertex when there's no usable floor polygon (e.g. a wall-only scene,
    or a floor polygon that collapses to nothing after `buffer(0)` repair).
    Either choice is equally valid geometrically (a rigid rotation about any
    point produces the same axis-aligned result, just translated); this one
    is picked for making the post-rotation coordinates land roughly where the
    pre-rotation ones did, rather than translating the whole scene far from
    its original position."""
    if floor_polygon and len(floor_polygon) >= 3:
        from shapely.geometry import Polygon

        poly = Polygon(floor_polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = _largest_polygon_component(poly)
        if poly is not None:
            c = poly.centroid
            return float(c.x), float(c.y)
    all_verts = [v for w in walls for v in w.vertices]
    if not all_verts:
        return (0.0, 0.0)
    xs = [v[0] for v in all_verts]
    zs = [v[1] for v in all_verts]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


def _rotate_floor_polygon(
    floor_polygon: list[tuple[float, float]] | None, angle_rad: float, center_xz: tuple[float, float]
) -> list[tuple[float, float]] | None:
    if not floor_polygon:
        return floor_polygon
    return [rotate_point_xz(x, z, angle_rad, center_xz) for x, z in floor_polygon]


def _rotate_objects(objects: list[dict], angle_rad: float, center_xz: tuple[float, float]) -> list[dict]:
    """Rotates each object's `center_xy`, `angle_rad`, and `hull_xz` (the
    measured collision hull - also world XZ points, so it must rotate along
    with everything else to stay consistent with the rotated footprint).
    `size_uv` is a rectangle's own (length, width) in its own local frame and
    is rotation-invariant - untouched, per the task's explicit note."""
    rotated = []
    for obj in objects:
        new_obj = dict(obj)
        cx, cz = obj["center_xy"]
        new_obj["center_xy"] = rotate_point_xz(cx, cz, angle_rad, center_xz)
        new_angle = obj["angle_rad"] + angle_rad
        # Same [-pi/2, pi/2) normalization oriented_min_area_rect itself uses.
        new_obj["angle_rad"] = ((new_angle + math.pi / 2) % math.pi) - math.pi / 2
        hull_xz = obj.get("hull_xz")
        if hull_xz:
            new_obj["hull_xz"] = [list(rotate_point_xz(p[0], p[1], angle_rad, center_xz)) for p in hull_xz]
        rotated.append(new_obj)
    return rotated


def _rotate_gaps(gaps: list[Gap], angle_rad: float, center_xz: tuple[float, float]) -> list[Gap]:
    """Rotates each gap's `measurement_point_xy`. `width_m` (and every other
    field) is unaffected - a rigid rotation preserves distances, so a
    passage's measured width doesn't change, only where in space it sits."""
    rotated = []
    for g in gaps:
        new_point = rotate_point_xz(g.measurement_point_xy[0], g.measurement_point_xy[1], angle_rad, center_xz)
        rotated.append(replace(g, measurement_point_xy=new_point))
    return rotated


def _rotate_camera_poses(cameras_json: dict, angle_rad: float, center_xz: tuple[float, float]) -> dict:
    """Rotates every camera pose in a `cameras_aligned.json`-shaped dict (see
    schema_version 2: `cameras` - a flat [x, y, z] world position per camera;
    `extrinsics` - a 4x4 camera-to-world matrix per camera; `intrinsics` - a
    3x3 K matrix per camera, untouched since intrinsics don't depend on world
    orientation) by the same rigid XZ rotation applied to the rest of the
    scene.

    Because `extrinsics` are camera-to-world (not world-to-camera), rotating
    the *world* by `angle_rad` about `center_xz` is a single left-multiply of
    every extrinsic by one fixed homogeneous transform - no per-camera
    quaternion/rotation-matrix re-derivation needed: for a world point
    p' = R @ (p - c) + c = R @ p + (c - R @ c), so the new camera-to-world
    matrix is M @ E where M = [[R, c - R@c], [0, 1]] and E is the original
    camera-to-world matrix."""
    cx, cz = center_xz
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array(
        [
            [cos_a, 0.0, -sin_a],
            [0.0, 1.0, 0.0],
            [sin_a, 0.0, cos_a],
        ]
    )
    center_vec = np.array([cx, 0.0, cz])
    t = center_vec - R @ center_vec
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t

    def rotate_camera_position(pos: list[float]) -> list[float]:
        x, y, z = pos
        nx, nz = rotate_point_xz(x, z, angle_rad, center_xz)
        return [nx, y, nz]

    out = dict(cameras_json)
    out["cameras"] = [rotate_camera_position(pos) for pos in cameras_json.get("cameras", [])]
    out["extrinsics"] = [(M @ np.array(ext)).tolist() for ext in cameras_json.get("extrinsics", [])]
    return out


def rasterize_object_footprint(obj: dict, grid_shape: tuple[int, int], resolution: float, origin_x: float, origin_z: float) -> np.ndarray:
    """Boolean mask of cells inside an object's oriented footprint rectangle, for
    the gaps.json obstacle-source grid (SPEC A5). `grid_shape` is `(nx, nz)` -
    the mask is indexed `[ix, iz]` like `occupancy.npy` (see
    `geometry._cell_to_world`)."""
    import math

    nx, nz = grid_shape
    xs = origin_x + (np.arange(nx) + 0.5) * resolution
    zs = origin_z + (np.arange(nz) + 0.5) * resolution
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    cx, cz = obj["center_xy"]
    angle = obj["angle_rad"]
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    dx, dz = grid_x - cx, grid_z - cz
    local_u = dx * cos_a - dz * sin_a
    local_v = dx * sin_a + dz * cos_a
    length_u, width_v = obj["size_uv"]
    return (np.abs(local_u) <= length_u / 2) & (np.abs(local_v) <= width_v / 2)


def _floor_standing_footprint_masks(
    objects: list[dict],
    floor_y: float,
    grid_shape: tuple[int, int],
    resolution: float,
    origin_x: float,
    origin_z: float,
    *,
    max_bbox_min_above_floor_m: float = FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M,
) -> tuple[list[np.ndarray], list[str]]:
    """Rasterized footprints (`rasterize_object_footprint`, the same grid
    rasterization gaps.json uses) of the *floor-standing* objects - those whose
    `bbox_min_y - floor_y` (Y is up in this frame; XZ is the floor plane) is
    below `max_bbox_min_above_floor_m` - for `geometry.extract_room_polygon`.
    Returns (masks, ids of the objects that qualified) in input order."""
    masks, ids = [], []
    for obj in objects:
        if obj["bbox_min_y"] - floor_y < max_bbox_min_above_floor_m:
            masks.append(rasterize_object_footprint(obj, grid_shape, resolution, origin_x, origin_z))
            ids.append(obj["id"])
    return masks, ids


ROOM_GRID_PADDING_MAX_CELLS = 400  # 20 m at 5 cm - a footprint further out than this is not a room-shape signal


def _pad_grid_to_cover_footprints(
    occupancy: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_z: float,
    objects: list[dict],
    *,
    margin_cells: int = 4,
    max_pad_cells: int = ROOM_GRID_PADDING_MAX_CELLS,
) -> tuple[np.ndarray, float, float, dict]:
    """T15g root-cause fix for "the bed/curtain is outside the room polygon":
    object footprints come from world-frame PLY hulls, but `occupancy.npy` is a
    percentile-trimmed bounding box of the point cloud (`gpu/stage_occupancy.py`),
    so a floor-standing footprint can reach past the grid's edge (hero:
    `curtain_0` to x=4.30 / z=-6.33 against a grid ending at x=4.10 / z=-6.07,
    `bed_0` to z=-6.22). `build_room_mask` can only union cells that exist, so
    the traced polygon stopped at the grid edge and 18% of the curtain (1% of
    the bed) lay outside it. Returns `occupancy` padded with UNKNOWN (never
    FREE - the free component is unchanged) so every footprint corner of
    `objects` is at least `margin_cells` inside, with the shifted origin and a
    record of the padding per side (`[x_lo, x_hi, z_lo, z_hi]`, cells). The
    padding is capped at `max_pad_cells` per side (logged) so a spurious
    far-away detection cannot blow the grid up."""
    nx, nz = occupancy.shape
    if not objects:
        return occupancy, origin_x, origin_z, {"cells": [0, 0, 0, 0], "capped": False}
    xs, zs = [], []
    for obj in objects:
        coords = list(_footprint_polygon(obj).exterior.coords)
        xs += [c[0] for c in coords]
        zs += [c[1] for c in coords]
    min_x, max_x, min_z, max_z = min(xs), max(xs), min(zs), max(zs)
    need = [
        math.ceil((origin_x - min_x) / resolution) + margin_cells,
        math.ceil((max_x - (origin_x + nx * resolution)) / resolution) + margin_cells,
        math.ceil((origin_z - min_z) / resolution) + margin_cells,
        math.ceil((max_z - (origin_z + nz * resolution)) / resolution) + margin_cells,
    ]
    pad = [int(min(max(n, 0), max_pad_cells)) for n in need]
    capped = any(n > max_pad_cells for n in need)
    if capped:
        logging.getLogger(__name__).warning("room grid padding capped at %d cells (requested %s)", max_pad_cells, need)
    if not any(pad):
        return occupancy, origin_x, origin_z, {"cells": pad, "capped": capped}
    padded = np.pad(occupancy, ((pad[0], pad[1]), (pad[2], pad[3])), mode="constant", constant_values=UNKNOWN)
    return padded, origin_x - pad[0] * resolution, origin_z - pad[2] * resolution, {"cells": pad, "capped": capped}


def _floor_standing_inside_fractions(objects: list[dict], floor_standing_ids: list[str], room_polygon) -> dict[str, float]:
    """Area fraction of each floor-standing object's final footprint that lies
    inside `room_polygon` (T15g assertion: >= 0.95 for every one)."""
    room = _room_polygon_shape(room_polygon)
    out: dict[str, float] = {}
    if room is None:
        return out
    ids = set(floor_standing_ids)
    for obj in objects:
        if obj["id"] not in ids:
            continue
        fp = _footprint_polygon(obj)
        if not fp.is_valid:
            fp = fp.buffer(0)
        out[obj["id"]] = float(fp.intersection(room).area / fp.area) if fp.area > 1e-12 else 1.0
    return out


def extract_floor_polygon(occupancy: np.ndarray, resolution: float, origin_x: float, origin_z: float) -> list[tuple[float, float]] | None:
    """Largest connected FREE+UNKNOWN component's boundary. NOT the room
    reference any more (T15a): UNKNOWN cells always touch the grid's own edges
    (`gpu/stage_occupancy.py` builds a bounding box), so through any doorway
    this polygon leaks into the UNKNOWN halo and measures the grid's extent.
    Kept only for external callers/tests; `run_bootstrap` uses
    `geometry.extract_room_polygon`."""
    from scipy import ndimage

    from scripts.msa.geometry import _rasterize_polygon_boundary

    free_mask = occupancy != OBSTACLE  # free-region floor spans FREE + UNKNOWN, per SPEC A2 "floor from the free region"
    labeled, n = ndimage.label(free_mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(free_mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    ring = _rasterize_polygon_boundary(labeled == largest, resolution, origin_x, origin_z)
    if len(ring) < 4:
        return None
    return simplify_polygon_preserving_passages(ring, np.zeros((0, 2)), tolerance_m=0.05, passage_margin_m=0.025)


def _check_grid_frame(occupancy: np.ndarray, meta: dict) -> None:
    """Grid-frame guard (T15b): `occupancy.npy` is `[ix, iz]`, so when
    `occupancy_meta.json` carries `width`/`height` (gpu/stage_occupancy.py's
    nx/nz) the array must be exactly `(width, height)`. A transposed array here
    would silently reflect every wall/room polygon across the x = z diagonal
    relative to the object footprints - the bug T15b fixed. Synthetic test
    scenes that omit width/height are not checked."""
    if "width" not in meta or "height" not in meta:
        return
    expected = (int(meta["width"]), int(meta["height"]))
    if tuple(occupancy.shape) != expected:
        raise ValueError(
            f"occupancy.npy shape {tuple(occupancy.shape)} != (width, height) = {expected} from occupancy_meta.json; "
            "the grid must be indexed [ix, iz] (axis 0 = X) - see geometry._cell_to_world"
        )


def run_bootstrap(scene_dir: Path, out_dir: Path, object_inputs: list[ObjectFootprintInput] | None = None, *,
                  assets_root: Path | None = None) -> dict:
    """M7 stage order: footprints (+ wall backfill) -> duplicate MERGE -> small-object
    filter -> outside-room drop/clip -> SUPPORT RULE -> CLASS CAPS -> gaps -> yaw ->
    exports. `report["stage_counts"]` records `n_objects_kept` after every stage.

    `assets_root` (M7 item 4, opt-in): an asset root with `INVENTORY.json`
    (`asset_index.build_index`) - the final objects get the T15f asset placement
    (`asset_fallback.place_assets`) and `scene.usd` references those assets as
    each object's `visual` (`usd_assets`), so the bootstrap USD shows the same
    furniture as the assembled web GLB. None keeps placeholder boxes."""
    occupancy = np.load(scene_dir / "occupancy.npy")
    meta = json.loads((scene_dir / "occupancy_meta.json").read_text())
    resolution = meta["resolution"]
    origin_x = meta["origin_x"]
    origin_z = meta["origin_z"]
    _check_grid_frame(occupancy, meta)

    # floor_y/ceiling_y: prefer scene_meta.json (floor_y=0.0-normalized, matching
    # the coordinate frame scene_objects.json's bbox_min/max and occupancy.npy are
    # already in) over raw alignment.json - alignment.json's own `floor_y` is in a
    # PRE-normalization frame (verified against the real e1e21716 scene: DB
    # scene.floor_y=0.0 but alignment.json floor_y=-1.279 while object bboxes sit
    # at y~0-2.7, i.e. already floor-normalized) and would silently misplace every
    # wall/floor extrusion if used directly. Production callers (an API endpoint
    # with the DB Scene row) should write scene_meta.json; this is not a
    # hypothetical fallback path exercised only in tests.
    scene_meta_path = scene_dir / "scene_meta.json"
    if scene_meta_path.exists():
        scene_meta = json.loads(scene_meta_path.read_text())
        floor_y = scene_meta["floor_y"]
        ceiling_y = scene_meta["ceiling_y"]
    else:
        alignment = json.loads((scene_dir / "alignment.json").read_text()) if (scene_dir / "alignment.json").exists() else {}
        floor_y = alignment.get("floor_y", 0.0)
        ceiling_y = alignment.get("ceiling_y", 2.5)

    walls, dropped = extract_wall_polygons(occupancy, resolution, origin_x, origin_z)

    if object_inputs is None:
        object_inputs = load_object_inputs_from_scene_dir(scene_dir)
    objects, overhead_ids = compute_object_footprints(object_inputs, floor_y, walls=walls)
    stage_counts: dict[str, int] = {"footprints": len(objects)}
    # M7 item 1: duplicate merge (dedupe.merge_duplicates) right after the
    # footprints + wall backfill and BEFORE the small-object filter and the
    # outside-room check, so a merged hull is judged once by everything below.
    from scripts.msa.dedupe import merge_duplicates

    objects, merges = merge_duplicates(objects)
    stage_counts["after_merge"] = len(objects)
    for m in merges:
        print(f"merged {','.join(m['merged_ids'])} into {m['kept_id']} ({m['reason']}, iou={m['iou']:.3f}, centroid_dist={m['centroid_dist_m']:.3f} m)")
    # Small/short-footprint filter (see _filter_small_isolated_objects docstring
    # for the "touching floor/wall/object" interpretation): runs right after
    # compute_object_footprints so dropped objects never feed the obstacle
    # sources below (gaps.json shouldn't treat detector noise as a real
    # obstacle) or any downstream export.
    walls_union = _walls_union(walls)
    objects, small_object_ids_dropped = _filter_small_isolated_objects(objects, walls_union)
    stage_counts["after_small_filter"] = len(objects)

    # Room polygon (T15a, see geometry.extract_room_polygon and docs/DECISIONS.md):
    # union(largest FREE component, floor-standing object footprints), closed,
    # outer contour only. This one polygon is the room reference for everything
    # below - yaw estimation, the outside-room drop/clip, the floor plate, the
    # USD/GLB floor mesh and the exported `floor_polygon`. It replaces both the
    # FREE+UNKNOWN extract_floor_polygon (leaked into the UNKNOWN halo) and the
    # yaw-only FREE-cell polygon (didn't follow the furniture: a bed/desk whose
    # top sits in the occupancy band is OBSTACLE/UNKNOWN there, so the free
    # region alone is an irregular blob and the hero still rendered ~20 degrees
    # off after "correction"). Built from the post-backfill, post-small-filter
    # footprints so it reflects what actually ships downstream.
    # T15g order of operations: union -> close -> (opening) -> yaw ANGLE -> snap ->
    # drop/clip -> gaps -> rotate everything last. The room mask is built on a
    # grid padded to cover every floor-standing footprint (see
    # _pad_grid_to_cover_footprints - the hero's curtain/bed reach past the
    # occupancy grid, which is why they stuck out of the T15b polygon).
    floor_standing_candidates = [o for o in objects if o["bbox_min_y"] - floor_y < FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M]
    room_occ, room_origin_x, room_origin_z, room_grid_padding = _pad_grid_to_cover_footprints(
        occupancy, resolution, origin_x, origin_z, floor_standing_candidates
    )
    floor_standing_masks, floor_standing_ids = _floor_standing_footprint_masks(
        objects, floor_y, room_occ.shape, resolution, room_origin_x, room_origin_z
    )
    # T15b: doorway spill is cut by a morphological opening (ROOM_OPENING_RADIUS_M,
    # see geometry.detach_corridor_spill) so the largest FREE component stops at
    # the door instead of trailing into whatever was scanned beyond it.
    opening_radius_cells = int(round(ROOM_OPENING_RADIUS_M / resolution))
    room_polygon_raw = extract_room_polygon(
        room_occ, resolution, room_origin_x, room_origin_z, floor_standing_masks, opening_radius_cells=opening_radius_cells
    )

    # Yaw ANGLE from the closed (unsnapped) polygon's min-area rectangle - see the
    # yaw-normalization comment block further down for the method and the
    # diagnostics. Estimated here so the Manhattan snap can happen in that frame;
    # the rotation itself is applied to everything as the LAST step.
    walls_as_scanned = walls
    histogram_result = compute_dominant_wall_yaw(walls_as_scanned, bin_width_deg=YAW_BIN_WIDTH_DEG, min_peak_fraction=YAW_MIN_PEAK_FRACTION)
    # Variable/meta-key names still say "free_only" (T3'' wording, kept so
    # downstream readers of scene_meta.json/report.json keep working) - the
    # polygon behind them is now `room_polygon`.
    free_only_result = compute_room_polygon_yaw(room_polygon_raw, square_tolerance=YAW_SQUARE_TOLERANCE)
    wall_hull_result = compute_wall_hull_yaw(walls_as_scanned, square_tolerance=YAW_SQUARE_TOLERANCE)
    # Morning-1 policy: the wall-hull angle is APPLIED to everything; the polygon
    # angle is the fallback (no walls / degenerate hull) and otherwise a diagnostic.
    applied_result, yaw_fallback_to_polygon, yaw_disagreement_deg = _reconcile_yaw_estimates(
        free_only_result, wall_hull_result, disagreement_threshold_deg=YAW_DISAGREEMENT_THRESHOLD_DEG
    )
    if yaw_fallback_to_polygon:
        why = "no wall polygons" if wall_hull_result is None else "wall hull is degenerate-square (no long axis)"
        print(f"yaw: {why} - falling back to the room polygon's min-area-rect angle ({applied_result['correction_deg']:.2f} deg)")
    elif applied_result is not None and free_only_result is not None and yaw_disagreement_deg is not None:
        print(
            f"yaw: applying the wall-hull angle {applied_result['correction_deg']:.2f} deg "
            f"(room polygon says {free_only_result['correction_deg']:.2f}, {yaw_disagreement_deg:.2f} deg apart)"
        )

    # T15g: Manhattan regularization (geometry.regularize_room_polygon) in the
    # APPLIED yaw frame - 0/90 degree edges only, collinear runs merged,
    # steps < ROOM_NOTCH_M collapsed, genuine appendices (doorway corridor)
    # kept. Returned in the raw frame; rotated below with everything else. The
    # snap frame IS the applied frame, so the outline is axis-aligned after the
    # rotation whichever estimate was applied.
    snap_angle = applied_result["correction_rad"] if applied_result is not None else 0.0
    room_polygon = regularize_room_polygon(room_polygon_raw, snap_angle, notch_m=ROOM_NOTCH_M) if room_polygon_raw else room_polygon_raw
    room_polygon_regularized = room_polygon_raw is not None and room_polygon != room_polygon_raw
    # Reflection/through-window guard (see _drop_or_clip_objects_outside_room
    # docstring): runs *after* both wall-backfill and the small-object filter
    # above, on their final output - both can shift which objects sit near the
    # room boundary (backfill moves edges outward to touch a wall; the
    # small-object filter changes which objects survive at all), so this must
    # judge "outside the room" against what actually ships downstream, not the
    # pre-backfill/pre-filter footprints. Floor-standing objects are part of
    # room_polygon by construction, so only elevated objects can be dropped/clipped.
    dropped_prims: list[dict] = []
    objects, outside_room_ids_dropped = _drop_or_clip_objects_outside_room(objects, room_polygon, dropped_records=dropped_prims)
    # T15b: the same rule for every other footprint-carrying prim - the wall
    # polygons (judged against the room polygon buffered by WALL_CLIP_MARGIN_M,
    # see _drop_or_clip_walls_outside_room). An obstacle fragment beyond a
    # doorway (the hero's "bat") is removed here; `kept_wall_indices` keeps the
    # wall_<i> obstacle-source ids below in step with the exported wall list.
    n_walls_before_room_check = len(walls)
    # The yaw estimates above (wall hull, polygon, edge histogram) read the walls
    # as scanned - before any outside-room drop/clip (which needs the snapped
    # polygon, which needs the angle).
    walls, kept_wall_indices = _drop_or_clip_walls_outside_room(walls, room_polygon, dropped_records=dropped_prims)
    wall_ids_dropped = [d["id"] for d in dropped_prims if d["kind"] == "wall"]
    for d in dropped_prims:
        print(f"dropped {d['kind']} {d['id']}: {d['outside_fraction'] * 100:.1f}% of its {d['area_m2']:.2f} m^2 lies outside the room polygon")

    stage_counts["after_outside_room"] = len(objects)
    # M7 item 2: the support rule (support_rule.apply_support_rule) - floor,
    # wall (scanned fragments + the regularized outline, i.e. the wall band) or
    # another kept object's top face; everything else is floating and dropped.
    # After the merge / small filter / room check, before the class caps.
    from scripts.msa.class_caps import apply_class_caps
    from scripts.msa.support_rule import apply_support_rule

    objects, support_drops, support_keeps = apply_support_rule(objects, floor_y=floor_y, walls=walls, room_polygon=room_polygon)
    stage_counts["after_support_rule"] = len(objects)
    for d in support_drops:
        ns = d["nearest_support"]
        print(f"dropped {d['id']} ({d['label']}): floating at z_min={d['z_min']:.3f} m, nearest support {ns['kind']} {ns.get('id', '')} d={ns.get('distance_m')}")
    # M7 item 3: per-scene class caps (class_caps.apply_class_caps), LAST - ranks
    # by `n_points` (real detection support), hull area x height as the fallback.
    objects, class_cap_drops = apply_class_caps(objects)
    stage_counts["after_class_caps"] = len(objects)
    for d in class_cap_drops:
        print(f"dropped {d['id']} ({d['label']}): class cap {d['cap']} - rank {d['rank']} by support {d['support']:.1f}")

    passable = occupancy != OBSTACLE
    # Wall obstacle sources = original occupancy OBSTACLE connected components
    # (exact cells, not the simplified polygon) that survived the >=0.3m^2 filter
    # in extract_wall_polygons - re-labeled here since that function doesn't
    # return per-component masks, only the final polygons - and (T15b) the
    # outside-room drop above: a dropped wall is not an obstacle source either,
    # and the surviving ones are renumbered exactly like the exported walls.
    from scipy import ndimage as _ndimage

    labeled, n_components = _ndimage.label(occupancy == OBSTACLE, structure=np.ones((3, 3)))
    sources = []
    comp_idx = 0
    kept_wall_index_set = set(kept_wall_indices)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        area = comp_mask.sum() * resolution * resolution
        if area < 0.3:
            continue
        original_idx = comp_idx
        comp_idx += 1
        if original_idx not in kept_wall_index_set:
            continue
        sources.append(ObstacleSource(id=f"wall_{len(sources)}", kind="wall", mask=comp_mask))
    for obj in objects:
        mask = rasterize_object_footprint(obj, occupancy.shape, resolution, origin_x, origin_z)
        sources.append(ObstacleSource(id=obj["id"], kind="object", mask=mask))

    platforms = _default_platforms()
    gaps = compute_gaps(passable, sources, resolution, origin_x, origin_z, platforms=platforms)

    # Yaw normalization post-step (T3''/T15a/morning-1 - see compute_wall_hull_yaw's,
    # compute_room_polygon_yaw's and compute_dominant_wall_yaw's docstrings and
    # docs/DECISIONS.md): the GPU pipeline never corrects the room's yaw, so
    # `walls`/`room_polygon` above may be at an arbitrary rotation.
    #
    # APPLIED (morning-1): compute_wall_hull_yaw's minimum-area bounding rectangle
    # of the pooled wall-vertex convex hull - an estimate that never touches the
    # occupancy grid and, on the hero, sits on the furniture's axis where the
    # polygon is dragged by the doorway spur.
    #
    # Fallback + diagnostic: compute_room_polygon_yaw's minimum-area bounding
    # rectangle of `room_polygon` (T15a: largest FREE component union floor-standing
    # object footprints) - applied only when there are no walls or the hull is
    # degenerate-square (`yaw_fallback_to_polygon`), otherwise logged
    # (`yaw_free_only_*`, `yaw_disagreement_deg`, `yaw_cross_check_disagrees` when
    # > YAW_DISAGREEMENT_THRESHOLD_DEG).
    #
    # Secondary/diagnostic only: compute_dominant_wall_yaw's wall-segment-angle
    # histogram is still computed and logged (scene_meta.json/report.json) as a
    # third cross-check for provenance, but never decides which result is applied.
    #
    # All three are measured against the *pre-rotation* walls/room_polygon, then
    # the applied result's correction is used to rotate every exported geometry -
    # walls, room_polygon, objects, gaps, and (best-effort) camera poses - about
    # the room's own centroid so downstream consumers get an axis-aligned scene.
    # This runs *after* gaps are computed against the (unrotated) occupancy grid:
    # a rigid rotation preserves all distances/areas, so gap widths and object
    # sizes are unaffected - only positions/orientations change, uniformly, so
    # everything stays mutually consistent. (T15g: the estimates themselves were
    # computed above, before the Manhattan snap and the drop/clip; only the
    # rotation happens here, last.)
    yaw_meta = {
        "yaw_method": None,
        "yaw_correction_rad": 0.0,
        "yaw_correction_deg": 0.0,
        "yaw_applied": False,
        "yaw_is_degenerate_square": None,
        "yaw_long_axis_angle_deg": None,
        "yaw_rect_size_m": None,
        "yaw_square_tolerance": YAW_SQUARE_TOLERANCE,
        "yaw_rotation_center_xy": None,
        "yaw_cameras_rotated": False,
        # Full provenance of every signal considered (T3'' rework) - see the
        # yaw-normalization block's comment above.
        "yaw_free_only_polygon_present": room_polygon is not None,
        "yaw_free_only_long_axis_angle_deg": None,
        "yaw_free_only_correction_deg": None,
        "yaw_free_only_is_degenerate_square": None,
        "yaw_free_only_rect_size_m": None,
        "yaw_wall_hull_long_axis_angle_deg": None,
        "yaw_wall_hull_correction_deg": None,
        "yaw_wall_hull_is_degenerate_square": None,
        "yaw_wall_hull_rect_size_m": None,
        "yaw_disagreement_deg": None,
        "yaw_disagreement_threshold_deg": YAW_DISAGREEMENT_THRESHOLD_DEG,
        # Morning-1: which estimate was applied ("wall_hull" | "room_polygon" | None)
        # and whether the polygon was a fallback (no walls / degenerate hull).
        "yaw_applied_source": None,
        "yaw_fallback_to_polygon": False,
        # Secondary/diagnostic cross-check - see docstring above. None when
        # compute_dominant_wall_yaw's own confidence gate wasn't met (that
        # gate still means "no clear histogram peak", it just no longer
        # blocks normalization itself).
        "yaw_histogram_dominant_angle_deg": None,
        "yaw_histogram_correction_deg": None,
        "yaw_histogram_peak_share": None,
        "yaw_histogram_bin_width_deg": YAW_BIN_WIDTH_DEG,
        "yaw_histogram_min_peak_fraction": YAW_MIN_PEAK_FRACTION,
        "yaw_histogram_gate_passed": False,
    }
    if histogram_result is not None:
        yaw_meta.update(
            {
                "yaw_histogram_dominant_angle_deg": histogram_result["dominant_angle_deg"],
                "yaw_histogram_correction_deg": histogram_result["correction_deg"],
                "yaw_histogram_peak_share": histogram_result["peak_share"],
                "yaw_histogram_gate_passed": True,
            }
        )
    if free_only_result is not None:
        yaw_meta.update(
            {
                "yaw_free_only_long_axis_angle_deg": free_only_result["long_axis_angle_deg"],
                "yaw_free_only_correction_deg": free_only_result["correction_deg"],
                "yaw_free_only_is_degenerate_square": free_only_result["is_degenerate_square"],
                "yaw_free_only_rect_size_m": list(free_only_result["rect_size_m"]),
            }
        )
    if wall_hull_result is not None:
        yaw_meta.update(
            {
                "yaw_wall_hull_long_axis_angle_deg": wall_hull_result["long_axis_angle_deg"],
                "yaw_wall_hull_correction_deg": wall_hull_result["correction_deg"],
                "yaw_wall_hull_is_degenerate_square": wall_hull_result["is_degenerate_square"],
                "yaw_wall_hull_rect_size_m": list(wall_hull_result["rect_size_m"]),
            }
        )
    yaw_meta["yaw_disagreement_deg"] = yaw_disagreement_deg
    yaw_meta["yaw_fallback_to_polygon"] = yaw_fallback_to_polygon
    # Diagnostic flag only - the polygon never overrides an applicable wall hull.
    yaw_meta["yaw_cross_check_disagrees"] = yaw_disagreement_deg is not None and yaw_disagreement_deg > YAW_DISAGREEMENT_THRESHOLD_DEG

    if applied_result is not None:
        yaw_meta["yaw_method"] = YAW_METHOD_ROOM_POLYGON if yaw_fallback_to_polygon else YAW_METHOD_WALL_HULL
        yaw_meta["yaw_applied_source"] = "room_polygon" if yaw_fallback_to_polygon else "wall_hull"
        center_xz = _yaw_rotation_center(room_polygon, walls_as_scanned)
        angle_rad = applied_result["correction_rad"]
        walls = [rotate_wall_polygon(w, angle_rad, center_xz) for w in walls]
        room_polygon = _rotate_floor_polygon(room_polygon, angle_rad, center_xz)
        room_polygon_raw = _rotate_floor_polygon(room_polygon_raw, angle_rad, center_xz)
        objects = _rotate_objects(objects, angle_rad, center_xz)
        gaps = _rotate_gaps(gaps, angle_rad, center_xz)
        yaw_meta.update(
            {
                "yaw_correction_rad": angle_rad,
                "yaw_correction_deg": applied_result["correction_deg"],
                "yaw_applied": True,
                "yaw_is_degenerate_square": applied_result["is_degenerate_square"],
                "yaw_long_axis_angle_deg": applied_result["long_axis_angle_deg"],
                "yaw_rect_size_m": list(applied_result["rect_size_m"]),
                "yaw_rotation_center_xy": list(center_xz),
            }
        )

        # Camera poses: best-effort and read-only from scene_dir (never
        # mutates the input scene) - a rotated copy is written to out_dir
        # alongside everything else. Skipped silently (yaw_cameras_rotated
        # stays False) when the scene has no cameras_aligned.json, which is
        # normal for older scenes/synthetic fixtures with no camera data.
        cameras_path = scene_dir / "cameras_aligned.json"
        if cameras_path.exists():
            try:
                cameras_json = json.loads(cameras_path.read_text())
                rotated_cameras = _rotate_camera_poses(cameras_json, angle_rad, center_xz)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "cameras_aligned.json").write_text(json.dumps(rotated_cameras, indent=2))
                yaw_meta["yaw_cameras_rotated"] = True
            except Exception as e:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "cameras_aligned.json.error.txt").write_text(str(e))

    out_dir.mkdir(parents=True, exist_ok=True)
    gaps_path = out_dir / "gaps.json"
    gaps_path.write_text(json.dumps(gaps_to_json(gaps), indent=2))

    # T15g assertion input: every floor-standing footprint, as exported, must
    # be >= 95% inside the (regularized, rotated) room polygon.
    floor_standing_inside = _floor_standing_inside_fractions(objects, floor_standing_ids, room_polygon)
    floor_standing_inside_min = min(floor_standing_inside.values()) if floor_standing_inside else None

    # T15g: ONE wall band for the whole room - the regularized outline buffered
    # outward by WALL_THICKNESS_M, extruded floor_y -> ceiling_y everywhere. The
    # raw wall fragments stay in the data (gaps.json obstacle sources,
    # report.json, /World/Plan/wall_i curves) but are no longer extruded.
    wall_band = room_wall_band(room_polygon, WALL_THICKNESS_M) if room_polygon else None

    scene_meta_out = {
        "floor_y": floor_y,
        "ceiling_y": ceiling_y,
        **yaw_meta,
        # T15a: the (already yaw-rotated) room reference polygon, [x, z] pairs
        # in the same frame as every other exported geometry, plus how it was built.
        "room_polygon": [[float(x), float(z)] for x, z in room_polygon] if room_polygon else None,
        "room_polygon_method": ROOM_POLYGON_METHOD,
        "room_polygon_floor_standing_object_ids": floor_standing_ids,
        "room_polygon_floor_standing_max_bbox_min_above_floor_m": FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M,
        # T15g provenance: the unsnapped outline (rotated), the snap parameters,
        # the grid padding that made the footprints fit, the wall band.
        "room_polygon_raw": [[float(x), float(z)] for x, z in room_polygon_raw] if room_polygon_raw else None,
        "room_polygon_regularized": bool(room_polygon_regularized),
        "room_polygon_notch_m": ROOM_NOTCH_M,
        "room_polygon_n_vertices": len(room_polygon) if room_polygon else 0,
        "room_grid_padding_cells": room_grid_padding["cells"],
        "room_grid_padding_capped": room_grid_padding["capped"],
        "floor_standing_inside_fractions": floor_standing_inside,
        "floor_standing_inside_fraction_min": floor_standing_inside_min,
        "wall_band_thickness_m": WALL_THICKNESS_M,
        "wall_height_m": float(ceiling_y - floor_y),
        "wall_band_present": wall_band is not None,
    }
    (out_dir / "scene_meta.json").write_text(json.dumps(scene_meta_out, indent=2))

    from scripts.msa.export_glb import _extrude_shapely_polygon_xz, build_scene, export_glb

    # `floor_polygon` parameter name kept on every exporter's signature; the
    # value is room_polygon (T15a). `wall_band` (T15g) replaces the per-fragment
    # wall meshes with the single `wall_outline` node.
    scene = build_scene(walls, room_polygon, floor_y, ceiling_y, objects, wall_band=wall_band)
    export_glb(scene, out_dir / "scene.glb")
    wall_band_mesh = _extrude_shapely_polygon_xz(wall_band, floor_y, ceiling_y - floor_y) if wall_band is not None else None

    from scripts.msa.export_dxf import build_floor_plan, export_dxf, export_svg

    doc = build_floor_plan(walls, objects, gaps)
    export_dxf(doc, out_dir / "floor_plan.dxf")
    try:
        export_svg(doc, out_dir / "floor_plan.svg")
    except Exception as e:  # SVG backend is best-effort; DXF is the source of truth
        (out_dir / "floor_plan.svg.error.txt").write_text(str(e))

    try:
        from scripts.msa.export_glb import _floor_plate_mesh, _floor_plate_polygon
        from scripts.msa.export_usd import export_usd

        # Same floor plate as the GLB export (union of room_polygon + every
        # kept object's footprint, see export_glb._floor_plate_polygon) - keeps
        # the USD and GLB floor meshes consistent instead of USD extruding
        # from the bare room_polygon while GLB covers more.
        floor_plate = _floor_plate_polygon(room_polygon, objects)
        floor_mesh = _floor_plate_mesh(floor_plate, floor_y) if floor_plate is not None else None
        # M7 item 4: with an asset root, scene.usd references the same CC0 assets
        # the assembled web GLB uses (placeholders stay for classes without one).
        asset_placements = None
        scene_usd_assets: dict | None = None
        if assets_root is not None:
            try:
                from scripts.msa.asset_fallback import place_assets
                from scripts.msa.asset_index import build_index

                _overrides, asset_placements = place_assets(objects, build_index(assets_root))
                scene_usd_assets = {
                    "assets_root": str(assets_root),
                    "n_placed": sum(1 for p in asset_placements if p["status"] == "placed"),
                    "n_placeholder": sum(1 for p in asset_placements if p["status"] != "placed"),
                }
            except FileNotFoundError as e:
                (out_dir / "scene.usd.assets.error.txt").write_text(str(e))
                asset_placements = None
        export_usd(walls, floor_mesh, floor_y, ceiling_y, objects, out_dir / "scene.usd", floor_polygon=room_polygon, wall_band_mesh=wall_band_mesh,
                   asset_placements=asset_placements)
        usd_ok = True
    except Exception as e:
        (out_dir / "scene.usd.error.txt").write_text(str(e))
        usd_ok = False
        scene_usd_assets = None

    render_ok = True
    try:
        from scripts.msa.render_top_down import render_gaps_top_down

        render_gaps_top_down(walls, objects, gaps, out_dir / "06_gaps_top.png", room_polygon=room_polygon, wall_band=wall_band)
    except Exception as e:
        (out_dir / "06_gaps_top.png.error.txt").write_text(str(e))
        render_ok = False

    report = {
        "n_walls": len(walls),
        "n_dropped_components": len(dropped),
        "dropped_components": [asdict(d) for d in dropped],
        "n_objects_kept": len(objects),
        # M7: object count after every stage, in pipeline order, plus the logs of
        # the three de-clutter stages (merge -> support rule -> class caps).
        "stage_counts": stage_counts,
        "n_merges": len(merges),
        "merges": merges,
        "n_support_dropped": len(support_drops),
        "support_drops": support_drops,
        "support_keeps": support_keeps,
        "n_class_cap_dropped": len(class_cap_drops),
        "class_cap_drops": class_cap_drops,
        "scene_usd_assets": scene_usd_assets,
        "n_objects_overhead_excluded": len(overhead_ids),
        "overhead_object_ids": overhead_ids,
        "n_small_objects_dropped": len(small_object_ids_dropped),
        "small_object_ids_dropped": small_object_ids_dropped,
        "n_objects_outside_room_dropped": len(outside_room_ids_dropped),
        "outside_room_ids_dropped": outside_room_ids_dropped,
        # T15b: the outside-room rule now covers walls too; every dropped prim
        # (object or wall) with its outside fraction and area.
        "n_walls_before_room_check": n_walls_before_room_check,
        "n_walls_outside_room_dropped": len(wall_ids_dropped),
        "wall_ids_dropped": wall_ids_dropped,
        "dropped_prims": dropped_prims,
        "wall_clip_margin_m": WALL_CLIP_MARGIN_M,
        "room_opening_radius_m": ROOM_OPENING_RADIUS_M,
        "room_opening_radius_cells": opening_radius_cells,
        "n_gaps": len(gaps),
        "usd_exported": usd_ok,
        "floor_polygon_present": room_polygon is not None,  # key name kept; the polygon is room_polygon (T15a)
        "room_polygon_method": ROOM_POLYGON_METHOD,
        "n_room_polygon_floor_standing_objects": len(floor_standing_ids),
        # T15g
        "room_polygon_regularized": bool(room_polygon_regularized),
        "room_polygon_n_vertices": len(room_polygon) if room_polygon else 0,
        "room_polygon_n_vertices_raw": len(room_polygon_raw) if room_polygon_raw else 0,
        "room_grid_padding_cells": room_grid_padding["cells"],
        "floor_standing_inside_fractions": floor_standing_inside,
        "floor_standing_inside_fraction_min": floor_standing_inside_min,
        "wall_band_thickness_m": WALL_THICKNESS_M,
        "wall_height_m": float(ceiling_y - floor_y),
        "wall_band_present": wall_band is not None,
        "top_down_render_exported": render_ok,
        "yaw_method": yaw_meta["yaw_method"],
        "yaw_applied": yaw_meta["yaw_applied"],
        "yaw_correction_rad": yaw_meta["yaw_correction_rad"],
        "yaw_correction_deg": yaw_meta["yaw_correction_deg"],
        "yaw_is_degenerate_square": yaw_meta["yaw_is_degenerate_square"],
        "yaw_long_axis_angle_deg": yaw_meta["yaw_long_axis_angle_deg"],
        "yaw_rect_size_m": yaw_meta["yaw_rect_size_m"],
        # Full provenance (T3'' rework): every signal considered and which one
        # was actually applied - see the yaw normalization block's comment.
        "yaw_free_only_polygon_present": yaw_meta["yaw_free_only_polygon_present"],
        "yaw_free_only_long_axis_angle_deg": yaw_meta["yaw_free_only_long_axis_angle_deg"],
        "yaw_free_only_is_degenerate_square": yaw_meta["yaw_free_only_is_degenerate_square"],
        "yaw_wall_hull_long_axis_angle_deg": yaw_meta["yaw_wall_hull_long_axis_angle_deg"],
        "yaw_wall_hull_is_degenerate_square": yaw_meta["yaw_wall_hull_is_degenerate_square"],
        "yaw_disagreement_deg": yaw_meta["yaw_disagreement_deg"],
        "yaw_disagreement_threshold_deg": yaw_meta["yaw_disagreement_threshold_deg"],
        "yaw_applied_source": yaw_meta["yaw_applied_source"],
        "yaw_fallback_to_polygon": yaw_meta["yaw_fallback_to_polygon"],
        "yaw_cross_check_disagrees": yaw_meta["yaw_cross_check_disagrees"],
        # Secondary/diagnostic cross-check against the polygon-derived
        # correction above - see the yaw normalization block's comment.
        "yaw_histogram_correction_deg": yaw_meta["yaw_histogram_correction_deg"],
        "yaw_histogram_peak_share": yaw_meta["yaw_histogram_peak_share"],
        "yaw_histogram_gate_passed": yaw_meta["yaw_histogram_gate_passed"],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # T13: persist the FINAL kept objects (+ walls / room polygon) that scene.glb
    # was built from, so assemble_with_generated.py reads them instead of
    # recomputing footprints (see scripts/msa/objects_io.py).
    from scripts.msa.objects_io import write_objects_json

    write_objects_json(out_dir / "objects.json", objects, walls, room_polygon, floor_y, ceiling_y, yaw_meta=yaw_meta)
    return report


def main() -> None:
    from scripts.msa.asset_index import DEFAULT_ASSETS_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=None,
                        help=f"M7: asset root (INVENTORY.json) whose CC0 assets scene.usd references as object visuals "
                             f"(default: {DEFAULT_ASSETS_ROOT} when it exists)")
    parser.add_argument("--no-assets", action="store_true", help="keep placeholder boxes in scene.usd")
    args = parser.parse_args()
    assets_root = None
    if not args.no_assets:
        assets_root = args.assets_root or (DEFAULT_ASSETS_ROOT if (DEFAULT_ASSETS_ROOT / "INVENTORY.json").exists() else None)
    report = run_bootstrap(args.scene_dir, args.out_dir, assets_root=assets_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
