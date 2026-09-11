"""Scene critic, part 1: deterministic numeric checks on an already-exported MSA scene
(var/scratch/QUEUE.md M14 "Scene critic"). Report-only - every function in this
module opens its inputs read-only (`Path.read_text`/`read_bytes`/`trimesh.load`) and
returns plain dicts; the only write this module performs is `write_critic_rules_json`,
which creates a *new* file (never edits objects.json/assets_placed.json/report.json/the
GLB or anything else already in the scene's out dir). See
tests/test_audit_critic_rules.py::test_never_touches_input_files, which hashes every
input file before and after a full run.

Inputs (all optional except `out_dir` itself - see `load_scene()`):
  - `objects.json`      - per-object hull_xz/center_xy/angle_rad/size_uv/height/
                           room_polygon, written by scripts/msa/bootstrap.py.
  - `assets_placed.json`- per-object placed-asset metadata (yaw_deg, center_xy),
                           written by scripts/msa/assemble_with_generated.py.
  - `report.json`       - support_keeps/support_drops log, written by bootstrap.py.
  - the exported GLB    - scene_assets_textured.glb / scene_textured.glb /
                           scene_assets.glb / scene.glb (first that exists), used ONLY
                           by check_yaw_vs_wall - see that function's docstring for why
                           the metadata fields above are not sufficient on their own.
A scene missing some of these (e.g. a Stage-A0-only bootstrap dir with no
assets_placed.json/objects.json at all) still runs: each check degrades gracefully and
records why it was skipped rather than raising or fabricating a finding.

Each finding is `{check, source, unverified, object_id, severity, measured,
threshold, detail}`. `severity` is one of "high", "medium", "info" (info = not a
violation, e.g. a `split_from_blob` exemption reported for visibility per the task
spec: "an object carrying split_from_blob is exempt - say so in the finding").
`source`/`unverified` (PM decision, 2026-09-07): every finding this module produces
is `"source": "rule"`, `"unverified": False` - rule findings are deterministic,
threshold-checked measurements against the scene's own exported geometry, so they
are the only thing this critic's CI gate (and, per that same decision, the viewer
badge's count/colour) can key off. `scripts/audit/critic_vlm.py`'s findings are the
mirror image - always `"source": "vlm"`, `"unverified": True` - since a VLM's
free-text read of a rendered image is corroborating/advisory only, never gating.
See `scripts/audit/critic_merge.py`'s module docstring for how the two get
combined and reported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Resolution order for the exported GLB, mirroring
# app.services.scene_service.MSA_GLB_CANDIDATES (kept independent - this module must
# not import app.* - but intentionally the same preference order: prefer the most
# fully-baked export, fall back to the plain bootstrap one).
GLB_CANDIDATES = (
    "scene_assets_textured.glb",
    "scene_textured.glb",
    "scene_assets.glb",
    "scene.glb",
)

CLASS_BBOX_LIMITS_M = {
    # label -> (dimension, limit_m). "horizontal" = longest of size_uv's two entries
    # (or, when only hull_xz is available, the longest edge of hull_xz's min-area
    # rectangle). "height" = the object's own `height` field.
    "bed": ("horizontal", 2.3),
    "tv": ("horizontal", 1.6),
    "television": ("horizontal", 1.6),
    "lamp": ("height", 1.9),
}

YAW_WALL_THRESHOLD_DEG = 15.0
ASSET_HULL_CENTROID_THRESHOLD_M = 0.2
# Below this footprint aspect ratio (long/short edge of the min-area rectangle), a
# rectangle fit is unstable (near-square/round objects like most lamps and small
# stools) - min-area-rectangle fitting picks an essentially arbitrary axis on a
# near-square point set, so a "yaw" measured off it is noise, not a bug. Confirmed
# empirically against geo5_out (2026-09-07): lamp_1/lamp_5/nightstand_1/chair_5 all
# have aspect < 1.3 and show 60-120 deg mesh-vs-hull "mismatches" that are pure
# rectangle-fit noise (their point clouds are near-square), while every object with
# aspect >= 1.3 (bed_0, desk_1, television_0, curtain_0, chair_0, chair_4, lamp_0,
# pillow_0, pillow_2, nightstand_0) shows a stable, physically meaningful angle.
YAW_ASPECT_CONFIDENCE_GATE = 1.3


@dataclass
class SceneData:
    out_dir: Path
    room_polygon: list[tuple[float, float]] | None
    objects: list[dict[str, Any]]
    objects_source: str  # "objects.json" | "glb" | "none"
    assets_placed_by_id: dict[str, dict[str, Any]] | None
    support_reason_by_id: dict[str, str] | None  # None = no support log available at all
    glb_path: Path | None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- loading


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def resolve_glb(out_dir: Path) -> Path | None:
    for name in GLB_CANDIDATES:
        p = out_dir / name
        if p.is_file():
            return p
    return None


def _footprint_aspect(hull_xz: list[list[float]] | None, size_uv: list[float] | None) -> float | None:
    """long/short edge ratio of the object's footprint - prefers the cheap `size_uv`
    field (already the (u, v) footprint size objects.json carries) over re-fitting a
    rectangle from hull_xz."""
    if size_uv and len(size_uv) == 2 and min(size_uv) > 1e-9:
        return max(size_uv) / min(size_uv)
    if hull_xz and len(hull_xz) >= 3:
        from shapely.geometry import MultiPoint

        rect = MultiPoint(hull_xz).convex_hull.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)[:4]
        if len(coords) < 4:
            return None
        lens = []
        for i in range(4):
            x1, z1 = coords[i]
            x2, z2 = coords[(i + 1) % 4]
            lens.append(math.hypot(x2 - x1, z2 - z1))
        adjacent = sorted(lens[:2])
        if adjacent[0] <= 1e-9:
            return None
        return adjacent[1] / adjacent[0]
    return None


def _objects_from_glb(glb_path: Path) -> list[dict[str, Any]]:
    """Fallback object list for a bootstrap-only out dir with no objects.json (e.g.
    the rejected own-scenes bootstrap output, which only ever produces `scene.glb` +
    `report.json`) - reconstructs one record per `<id>/collision/hull` node using the
    same `<label>_<index>` naming scripts/msa/export_glb.py already uses, so the rest
    of this module (which only cares about the fields below) doesn't need to know the
    difference."""
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    graph = scene.graph
    ids = sorted({n.rsplit("/", 2)[0] for n in graph.nodes_geometry if n.endswith("/collision/hull")})
    objects = []
    for obj_id in ids:
        hull_node = f"{obj_id}/collision/hull"
        try:
            transform, geom_name = graph.get(hull_node)
        except Exception:
            continue
        mesh = scene.geometry[geom_name]
        verts_world = trimesh.transformations.transform_points(mesh.vertices, transform)
        xz = verts_world[:, [0, 2]]
        hull_xz = [[float(x), float(z)] for x, z in xz]
        cx, cz = float(xz[:, 0].mean()), float(xz[:, 1].mean())
        y_vals = verts_world[:, 1]
        label = obj_id.rsplit("_", 1)[0] if "_" in obj_id else obj_id
        objects.append(
            {
                "id": obj_id,
                "label": label,
                "center_xy": [cx, cz],
                "hull_xz": hull_xz,
                "size_uv": None,
                "angle_rad": None,
                "height": float(y_vals.max() - y_vals.min()),
                "bbox_min_y": float(y_vals.min()),
                "split_from_blob": False,
            }
        )
    return objects


def _wall_directions_from_glb(glb_path: Path) -> list[dict[str, Any]] | None:
    """Fallback "room polygon edges" for a scene with no objects.json room_polygon -
    one pseudo-edge per top-level `wall_N` mesh node, direction = that mesh's own
    footprint long axis (min-area rectangle). Not a closed polygon (the rejected
    scene here only has 2 of an unknown total number of walls - itself very possibly
    part of why it failed validation), so `check_centroid_outside_room` is skipped
    when this fallback is used (see its docstring) but the yaw check can still use
    these as "nearest wall" candidates."""
    import trimesh
    from shapely.geometry import MultiPoint

    scene = trimesh.load(str(glb_path), process=False)
    graph = scene.graph
    wall_nodes = sorted(n for n in graph.nodes_geometry if n.startswith("wall_") and "/" not in n)
    if not wall_nodes:
        return None
    walls = []
    for node in wall_nodes:
        transform, geom_name = graph.get(node)
        mesh = scene.geometry[geom_name]
        verts_world = trimesh.transformations.transform_points(mesh.vertices, transform)
        xz = verts_world[:, [0, 2]]
        rect = MultiPoint(xz).convex_hull.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)[:4]
        if len(coords) < 4:
            continue
        best_len, best_seg = -1.0, None
        for i in range(4):
            x1, z1 = coords[i]
            x2, z2 = coords[(i + 1) % 4]
            length = math.hypot(x2 - x1, z2 - z1)
            if length > best_len:
                best_len, best_seg = length, (x1, z1, x2, z2)
        walls.append({"id": node, "segment": best_seg})
    return walls or None


def load_scene(out_dir: Path) -> SceneData:
    out_dir = Path(out_dir)
    notes: list[str] = []

    objects_json = _read_json(out_dir / "objects.json")
    glb_path = resolve_glb(out_dir)

    room_polygon = None
    objects: list[dict[str, Any]] = []
    objects_source = "none"
    wall_segments_fallback = None

    if objects_json is not None:
        room_polygon = [tuple(p) for p in objects_json.get("room_polygon", [])] or None
        objects = objects_json.get("objects", [])
        objects_source = "objects.json"
    elif glb_path is not None:
        objects = _objects_from_glb(glb_path)
        objects_source = "glb"
        notes.append(
            f"no objects.json in {out_dir} - reconstructed {len(objects)} object record(s) "
            f"from {glb_path.name}'s collision-hull nodes (bootstrap-only out dir)."
        )
        wall_segments_fallback = _wall_directions_from_glb(glb_path)
        if wall_segments_fallback is not None:
            notes.append(
                f"no room_polygon available - using {len(wall_segments_fallback)} wall mesh node(s) "
                "as nearest-wall candidates for the yaw check; check_centroid_outside_room is skipped."
            )
        else:
            notes.append("no room_polygon and no wall_N mesh nodes found - yaw and centroid-in-room checks skipped.")
    else:
        notes.append(f"no objects.json and no exported GLB found in {out_dir} - nothing to check.")

    assets_placed_json = _read_json(out_dir / "assets_placed.json")
    assets_placed_by_id = None
    if isinstance(assets_placed_json, list):
        assets_placed_by_id = {a["id"]: a for a in assets_placed_json if "id" in a}
    elif assets_placed_json is None:
        notes.append("no assets_placed.json - check_asset_centroid_vs_hull skipped.")

    report_json = _read_json(out_dir / "report.json")
    support_reason_by_id = None
    if report_json is not None and ("support_keeps" in report_json or "support_drops" in report_json):
        support_reason_by_id = {}
        for entry in report_json.get("support_keeps", []) or []:
            support_reason_by_id[entry["id"]] = entry.get("reason", "")
        for entry in report_json.get("support_drops", []) or []:
            support_reason_by_id[entry["id"]] = entry.get("reason", "")
    else:
        notes.append("report.json has no support_keeps/support_drops log - check_missing_support_reason skipped.")

    data = SceneData(
        out_dir=out_dir,
        room_polygon=room_polygon,
        objects=objects,
        objects_source=objects_source,
        assets_placed_by_id=assets_placed_by_id,
        support_reason_by_id=support_reason_by_id,
        glb_path=glb_path,
        notes=notes,
    )
    # Stash the wall-segment fallback for check_yaw_vs_wall without growing the
    # dataclass's public surface for what is very much a secondary path.
    data.__dict__["_wall_segments_fallback"] = wall_segments_fallback
    return data


# --------------------------------------------------------------------------- geometry helpers


def _polygon_edges(room_polygon: list[tuple[float, float]]) -> list[tuple[float, float, float, float]]:
    n = len(room_polygon)
    return [(*room_polygon[i], *room_polygon[(i + 1) % n]) for i in range(n)]


def _point_segment_distance(px: float, pz: float, x1: float, z1: float, x2: float, z2: float) -> float:
    dx, dz = x2 - x1, z2 - z1
    length_sq = dx * dx + dz * dz
    if length_sq < 1e-12:
        return math.hypot(px - x1, pz - z1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (pz - z1) * dz) / length_sq))
    proj_x, proj_z = x1 + t * dx, z1 + t * dz
    return math.hypot(px - proj_x, pz - proj_z)


def nearest_wall(
    edges: list[tuple[float, float, float, float]], point_xy: tuple[float, float]
) -> tuple[float, int, float] | None:
    """Returns (direction_deg mod 180, edge_index, distance_m) for the polygon edge
    nearest `point_xy`, or None if there are no edges."""
    if not edges:
        return None
    px, pz = point_xy
    best = None
    for i, (x1, z1, x2, z2) in enumerate(edges):
        dist = _point_segment_distance(px, pz, x1, z1, x2, z2)
        if best is None or dist < best[0]:
            direction = math.degrees(math.atan2(z2 - z1, x2 - x1)) % 180.0
            best = (dist, direction, i)
    dist, direction, idx = best
    return direction, idx, dist


def fold_delta_mod90(a_deg: float, b_deg: float) -> float:
    """Angular distance between two directions, treating both as symmetric mod 90 -
    i.e. "aligned with the wall" and "aligned perpendicular to the wall" both count as
    0 delta (matches this repo's own established convention for this exact
    measurement - see progress/13d_top_v2.png's "worst hull-vs-visual grain delta
    (mod 90 deg)" title, geo5_out/render_15_polygon.py's sibling script). Range
    [0, 45]."""
    da, db = a_deg % 90.0, b_deg % 90.0
    diff = abs(da - db)
    return min(diff, 90.0 - diff)


def _mesh_yaw_from_glb(glb_scene, object_id: str) -> float | None:
    """The ACTUAL orientation of the placed visual mesh for `object_id`, measured
    directly off its baked-in vertex positions in the exported GLB - mod 180, degrees.

    Why not just read `assets_placed.json`'s `yaw_deg` (or objects.json's
    `angle_rad`)? Because of a real, previously-shipped bug (QUEUE.md M12, "bed
    orientation bug"): the object's *hull* yaw was computed in the yaw-ALIGNED frame
    while the visual asset mesh was placed directly into the yaw-EXPORTED frame
    without correcting for that difference - so the recorded yaw_deg/angle_rad fields
    stayed numerically "correct" (matching the hull, which uses the same frame as
    room_polygon) even while the rendered mesh sat roughly `yaw_correction_deg`
    degrees off from where its own metadata claimed. Verified against
    geo5_out/scene_assets_textured.glb (2026-09-07): bed_0's collision hull measures
    1.09 deg (matches objects.json's angle_rad exactly) while its two visual mesh
    parts measure 154.97 deg - a 25 deg delta after the mod-90 wall-alignment fold,
    comfortably past this check's 15 deg threshold; nothing in objects.json or
    assets_placed.json shows this on its own. Reading the GLB is the only way this
    check can catch that class of bug, which is why the GLB is listed as an input at
    the top of this module."""
    import trimesh
    from shapely.geometry import MultiPoint

    graph = glb_scene.graph
    part_nodes = [n for n in graph.nodes_geometry if n.startswith(f"{object_id}/visual/")]
    if not part_nodes:
        return None

    def _rect_angle_deg(xz) -> tuple[float, float] | None:
        """(angle_deg mod 180, footprint area) of one part's own min-area rectangle."""
        if len(xz) < 3:
            return None
        poly = MultiPoint(xz).convex_hull
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)[:4]
        if len(coords) < 4:
            return None
        best_len, best_ang = -1.0, 0.0
        for i in range(4):
            x1, z1 = coords[i]
            x2, z2 = coords[(i + 1) % 4]
            length = math.hypot(x2 - x1, z2 - z1)
            if length > best_len:
                best_len, best_ang = length, math.degrees(math.atan2(z2 - z1, x2 - x1))
        area = max(poly.area, 1e-6)
        return best_ang % 180.0, area

    # IMPORTANT: measure each visual part's own rectangle SEPARATELY rather than
    # pooling every part's vertices into one point set first. For a multi-part
    # object (e.g. a `split_from_blob` bed, k>=2 pieces placed side by side), pooling
    # would measure the direction of the OFFSET BETWEEN pieces, not each piece's own
    # rotation - and for exactly the bed-orientation bug this check exists to catch,
    # those two directions differ: the split placement offsets happen to sit on the
    # correct axis even when each piece's own mesh is rotated ~150 deg off (confirmed
    # against geo5_out's bed_0: pooled vertices measure ~1 deg / correct, while each
    # of the two parts measures 154.97 deg individually - pooling would silently hide
    # the exact bug this check is for). Combine per-part angles with a circular mean
    # (doubled-angle trick, since a rectangle's orientation is 180-periodic),
    # weighted by each part's own footprint area.
    sin_sum = cos_sum = 0.0
    for node in part_nodes:
        try:
            transform, geom_name = graph.get(node)
        except Exception:
            continue
        mesh = glb_scene.geometry[geom_name]
        verts_world = trimesh.transformations.transform_points(mesh.vertices, transform)
        result = _rect_angle_deg(verts_world[:, [0, 2]])
        if result is None:
            continue
        angle_deg, area = result
        theta2 = math.radians(2.0 * angle_deg)
        sin_sum += area * math.sin(theta2)
        cos_sum += area * math.cos(theta2)
    if sin_sum == 0.0 and cos_sum == 0.0:
        return None
    mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0
    return mean_angle % 180.0


# --------------------------------------------------------------------------- checks


def check_yaw_vs_wall(scene: SceneData) -> list[dict[str, Any]]:
    edges = _polygon_edges(scene.room_polygon) if scene.room_polygon else []
    fallback_walls = scene.__dict__.get("_wall_segments_fallback")
    if not edges and fallback_walls:
        edges = [w["segment"] for w in fallback_walls]
    if not edges:
        return []

    glb_scene = None
    if scene.glb_path is not None:
        import trimesh

        glb_scene = trimesh.load(str(scene.glb_path), process=False)

    findings = []
    for obj in scene.objects:
        obj_id = obj.get("id")
        center = obj.get("center_xy")
        if obj_id is None or center is None:
            continue
        aspect = _footprint_aspect(obj.get("hull_xz"), obj.get("size_uv"))
        if aspect is None or aspect < YAW_ASPECT_CONFIDENCE_GATE:
            continue  # footprint too near-square/round for a reliable orientation fit

        measured_yaw = None
        source = None
        if glb_scene is not None:
            measured_yaw = _mesh_yaw_from_glb(glb_scene, obj_id)
            source = "glb_visual_mesh"
        if measured_yaw is None:
            placed = (scene.assets_placed_by_id or {}).get(obj_id)
            if placed and "yaw_deg" in placed:
                measured_yaw = placed["yaw_deg"] % 180.0
                source = "assets_placed.yaw_deg"
            elif obj.get("angle_rad") is not None:
                measured_yaw = math.degrees(obj["angle_rad"]) % 180.0
                source = "objects.angle_rad"
        if measured_yaw is None:
            continue

        wall = nearest_wall(edges, tuple(center))
        if wall is None:
            continue
        wall_dir_deg, wall_idx, wall_dist = wall
        delta = fold_delta_mod90(measured_yaw, wall_dir_deg)
        if delta > YAW_WALL_THRESHOLD_DEG:
            findings.append(
                {
                    "check": "yaw_vs_wall",
                    "source": "rule",
                    "unverified": False,
                    "object_id": obj_id,
                    "severity": "high" if delta > 30 else "medium",
                    "measured": round(delta, 2),
                    "threshold": YAW_WALL_THRESHOLD_DEG,
                    "detail": (
                        f"{obj.get('label', obj_id)} yaw {measured_yaw:.2f} deg (source: {source}) vs nearest wall "
                        f"(edge {wall_idx}, {wall_dist:.2f} m away) direction {wall_dir_deg:.2f} deg - "
                        f"{delta:.2f} deg off after mod-90 wall-alignment fold."
                    ),
                }
            )
    return findings


def check_bbox_vs_class_limit(scene: SceneData) -> list[dict[str, Any]]:
    findings = []
    for obj in scene.objects:
        label = (obj.get("label") or "").lower()
        if label not in CLASS_BBOX_LIMITS_M:
            continue
        dimension, limit = CLASS_BBOX_LIMITS_M[label]
        if dimension == "horizontal":
            size_uv = obj.get("size_uv")
            if size_uv:
                measured = max(size_uv)
            else:
                aspect_hull = obj.get("hull_xz")
                if not aspect_hull:
                    continue
                xs = [p[0] for p in aspect_hull]
                zs = [p[1] for p in aspect_hull]
                measured = max(max(xs) - min(xs), max(zs) - min(zs))
        else:
            measured = obj.get("height")
        if measured is None or measured <= limit:
            continue
        exempt = bool(obj.get("split_from_blob"))
        findings.append(
            {
                "check": "bbox_vs_class_limit",
                "source": "rule",
                "unverified": False,
                "object_id": obj.get("id"),
                "severity": "info" if exempt else ("high" if measured > 1.3 * limit else "medium"),
                "measured": round(measured, 3),
                "threshold": limit,
                "detail": (
                    f"{label} {dimension} {measured:.3f} m > {limit} m limit"
                    + (
                        f" - exempt: split_from_blob=True (split_k={obj.get('split_k')})"
                        if exempt
                        else " - not exempt (no split_from_blob)"
                    )
                ),
            }
        )
    return findings


def check_centroid_outside_room(scene: SceneData) -> list[dict[str, Any]]:
    if not scene.room_polygon or len(scene.room_polygon) < 3:
        return []
    from shapely.geometry import Point, Polygon

    poly = Polygon(scene.room_polygon)
    findings = []
    for obj in scene.objects:
        center = obj.get("center_xy")
        obj_id = obj.get("id")
        if center is None or obj_id is None:
            continue
        point = Point(center)
        if poly.contains(point) or poly.touches(point):
            continue
        distance = point.distance(poly)
        findings.append(
            {
                "check": "centroid_outside_room",
                "source": "rule",
                "unverified": False,
                "object_id": obj_id,
                "severity": "high",
                "measured": round(distance, 3),
                "threshold": 0.0,
                "detail": f"{obj.get('label', obj_id)} centroid {tuple(center)} is {distance:.3f} m outside room_polygon.",
            }
        )
    return findings


def check_asset_centroid_vs_hull(scene: SceneData) -> list[dict[str, Any]]:
    if not scene.assets_placed_by_id:
        return []
    findings = []
    for obj in scene.objects:
        obj_id = obj.get("id")
        hull_xz = obj.get("hull_xz")
        if obj_id is None or not hull_xz:
            continue
        placed = scene.assets_placed_by_id.get(obj_id)
        if not placed or "center_xy" not in placed:
            continue
        hull_cx = sum(p[0] for p in hull_xz) / len(hull_xz)
        hull_cz = sum(p[1] for p in hull_xz) / len(hull_xz)
        asset_cx, asset_cz = placed["center_xy"]
        distance = math.hypot(asset_cx - hull_cx, asset_cz - hull_cz)
        if distance <= ASSET_HULL_CENTROID_THRESHOLD_M:
            continue
        findings.append(
            {
                "check": "asset_centroid_vs_hull",
                "source": "rule",
                "unverified": False,
                "object_id": obj_id,
                "severity": "high" if distance > 2 * ASSET_HULL_CENTROID_THRESHOLD_M else "medium",
                "measured": round(distance, 3),
                "threshold": ASSET_HULL_CENTROID_THRESHOLD_M,
                "detail": (
                    f"placed asset centroid ({asset_cx:.3f}, {asset_cz:.3f}) is {distance:.3f} m from the "
                    f"hull_xz centroid ({hull_cx:.3f}, {hull_cz:.3f})."
                ),
            }
        )
    return findings


def check_missing_support_reason(scene: SceneData) -> list[dict[str, Any]]:
    if scene.support_reason_by_id is None:
        return []
    findings = []
    for obj in scene.objects:
        obj_id = obj.get("id")
        if obj_id is None:
            continue
        if obj_id in scene.support_reason_by_id:
            continue
        findings.append(
            {
                "check": "missing_support_reason",
                "source": "rule",
                "unverified": False,
                "object_id": obj_id,
                "severity": "medium",
                "measured": None,
                "threshold": None,
                "detail": f"{obj.get('label', obj_id)} has no entry in report.json's support_keeps/support_drops log.",
            }
        )
    return findings


ALL_CHECKS = (
    check_yaw_vs_wall,
    check_bbox_vs_class_limit,
    check_centroid_outside_room,
    check_asset_centroid_vs_hull,
    check_missing_support_reason,
)


def run_rules(out_dir: Path) -> dict[str, Any]:
    """Loads `out_dir` and runs every check. Pure function of what's on disk - does
    not write anything; see write_critic_rules_json / the CLI below for that."""
    scene = load_scene(Path(out_dir))
    findings: list[dict[str, Any]] = []
    for check in ALL_CHECKS:
        findings.extend(check(scene))
    return {
        "out_dir": str(scene.out_dir),
        "objects_source": scene.objects_source,
        "n_objects": len(scene.objects),
        "glb_used": str(scene.glb_path) if scene.glb_path else None,
        "notes": scene.notes,
        "findings": findings,
        "n_findings": len(findings),
    }


def write_critic_rules_json(out_dir: Path, dest: Path | None = None) -> Path:
    result = run_rules(out_dir)
    dest = dest or Path(out_dir) / "critic_rules.json"
    dest.write_text(json.dumps(result, indent=2))
    return dest


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="bootstrap/export out dir to audit")
    parser.add_argument("--out", type=Path, default=None, help="where to write critic_rules.json (default: <out_dir>/critic_rules.json)")
    args = parser.parse_args()
    dest = write_critic_rules_json(args.out_dir, args.out)
    result = json.loads(dest.read_text())
    print(f"wrote {dest}: {result['n_findings']} finding(s) over {result['n_objects']} object(s) ({result['objects_source']})")


if __name__ == "__main__":
    _main()
