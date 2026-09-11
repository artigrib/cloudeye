"""3/4-view (angled perspective) PNG renderer for a bootstrapped room.

Until now there was no proper "3/4 view" module in scripts/msa/ - only an
ad-hoc inline trimesh script living outside this package. This module promotes
that idea into a real, tested module alongside `render_top_down.py`.

Rendering backend: like `render_top_down.py` (see its docstring), this VPS has
no Blender/bpy and, checked while building this module, no pyrender/PyOpenGL
and no EGL either - so there is no GPU/headless-OpenGL renderer available here
at all, not just no Blender. Reaching for pyrender would add a new heavy
dependency that can't even be verified to work on this box. So this module
does the same thing `render_top_down.py` does: a pure-Python software
rasterizer on top of PIL. It differs from `render_top_down.py` in three ways:
it projects through a real pinhole camera (not an orthographic top-down
projection), it depth-sorts faces with a painter's algorithm (`_render_scene`)
so nearer geometry draws over farther geometry, and it composites walls at
partial opacity (PIL's `ImageDraw.polygon` does NOT alpha-blend when drawing
directly onto an RGBA image - it overwrites pixels - so translucent faces are
drawn onto a small transparent overlay and `Image.alpha_composite`'d in).

Geometry source: reuses `export_glb.build_scene` (SPEC E0's wall/floor/object
mesh construction - full-height wall extrusion, floor slab extrusion, object
placeholder placement) rather than re-deriving that logic here. That is also
why the floor "plate" is built the same way `bootstrap.py` builds its
USD-export floor mesh (Polygon -> `trimesh.creation.extrude_polygon`): both
paths go through `export_glb._extrude_polygon_xz` / `_largest_polygon`, so
there is exactly one place that logic lives (`export_glb.py`), not two.

`scripts.msa.bootstrap` is intentionally NOT imported anywhere in this module -
it's owned by other in-flight tasks and off-limits to import from or modify
here. That has one consequence worth flagging: the CLI entry point below reads
an already-exported `scene.glb` rather than a raw occupancy scene dir, because
deriving walls/floor_polygon/floor_y/ceiling_y/objects from occupancy.npy the
way `bootstrap.run_bootstrap` does would mean duplicating a large fraction of
Stage A's logic (floor-polygon extraction via connected components, object
footprint/overhead-exclusion rules, etc.) - substantially more than "the small
amount of logic needed" a simple reuse would be. Reading the GLB instead is the
genuinely simpler option the task allows for, and it's also exactly what the
test fixture set up for this task (`a0_out/scene.glb` next to the raw
`scene/`) supports directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from scripts.msa.export_glb import _largest_polygon, build_scene, mesh_base_color_rgb
from scripts.msa.geometry import WallPolygon, convex_hull_2d
from scripts.msa.visibility import TRANSLUCENT_OBJECT_CLASSES as _TRANSLUCENT_OBJECT_CLASSES
from scripts.msa.visibility import TRANSLUCENT_OPACITY as _TRANSLUCENT_OPACITY

# --- camera / render defaults, per task spec --------------------------------
# T3 replaced the old fixed "4m back along the room's oriented-bbox long axis,
# -30deg pitch" camera (compute_camera below) with one derived entirely from
# the room's own geometry (polygon vertices, bed footprint if present) every
# time - see compute_camera's docstring for the full rule. BACK_OFFSET_M/
# PITCH_DEG are gone as tunable knobs; CAMERA_DISTANCE_DIAGONAL_FRAC/
# MIN_CAMERA_DISTANCE_M/PITCH_SEARCH_RANGE_DEG etc. below are the formula's
# own parameters, not per-scene overrides.
CAMERA_HEIGHT_M = 2.6  # camera height ABOVE floor_y (not world Y=0) - was 2.5
FOV_DEG = 55.0  # vertical field of view
CAMERA_DISTANCE_DIAGONAL_FRAC = 0.7  # camera distance from room centroid = max(this * room_diagonal, MIN_CAMERA_DISTANCE_M)
MIN_CAMERA_DISTANCE_M = 3.0
TARGET_FLOOR_SCREEN_FRAC = 0.85  # solved pitch puts the farthest floor point at this fraction of image height
PITCH_SEARCH_RANGE_DEG = (-75.0, -1.0)  # (steepest, shallowest) bisection bracket for the pitch solve
PITCH_SEARCH_TOLERANCE_DEG = 0.05  # bisection convergence tolerance - a twentieth of a degree
FALLBACK_PITCH_DEG = -30.0  # used only when there's no floor polygon to target at all (degenerate/empty scene)
WALL_OPACITY = 0.30
# T15i: the "soft furnishing" classes and their 30% opacity are owned by
# scripts/msa/visibility.py (stdlib-only, so export_usd.py and demo/record_isaac.py
# share them); re-exported here under their historical names.
TRANSLUCENT_OPACITY = _TRANSLUCENT_OPACITY
TRANSLUCENT_OBJECT_CLASSES = _TRANSLUCENT_OBJECT_CLASSES
BACKGROUND_RGB = (8, 8, 12)  # near-black
FLOOR_RGB = (68, 70, 78)
WALL_RGB = (190, 195, 205)
DEFAULT_OBJECT_RGB = (150, 150, 150)
NEAR_M = 0.3  # near-plane clip distance (camera-space z), keeps perspective divide sane
DEFAULT_IMAGE_SIZE = (1280, 960)

_TRAILING_INDEX_RE = re.compile(r"_\d+$")


# --- camera framing ----------------------------------------------------------


def _room_footprint_xz(floor_polygon, walls: list[WallPolygon]) -> np.ndarray:
    """Points (Nx2, world XZ) used to frame the camera. Prefers the floor
    polygon (the actual room outline); falls back to all wall vertices when
    there is no floor polygon, mirroring `build_scene`'s own handling of an
    optional `floor_polygon`."""
    if floor_polygon and len(floor_polygon) >= 3:
        return np.asarray(floor_polygon, dtype=float)
    pts = [v for wall in walls for v in wall.vertices]
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def _room_centroid(points_xz: np.ndarray) -> tuple[float, float]:
    if len(points_xz) >= 3:
        from shapely.geometry import Polygon

        poly = Polygon(points_xz)
        if not poly.is_valid or poly.area < 1e-9:
            poly = poly.buffer(0)
        poly = _largest_polygon(poly)
        if poly is not None:
            c = poly.centroid
            return float(c.x), float(c.y)
    if len(points_xz):
        return float(points_xz[:, 0].mean()), float(points_xz[:, 1].mean())
    return 0.0, 0.0


def _room_diagonal(points_xz: np.ndarray) -> float:
    """Room "diagonal" (task spec step 3): the distance between the two most
    mutually-distant vertices of the room polygon - i.e. the point set's true
    diameter. Chosen over the simpler bounding-box-diagonal alternative the
    task spec explicitly allows because it's no harder to get right for the
    vertex counts a simplified room outline actually has (a `floor_polygon`
    after Douglas-Peucker simplification, or a wall vertex list - typically a
    few dozen points, not per-triangle mesh density), and it's the more
    literal reading of "room diagonal" for an L-shaped or otherwise
    non-rectangular room where a bbox diagonal would overstate the room's
    real extent. O(n^2) over that small n is cheap in absolute terms."""
    n = len(points_xz)
    if n < 2:
        return 0.0
    best = 0.0
    for i in range(n - 1):
        dx = points_xz[i + 1 :, 0] - points_xz[i, 0]
        dz = points_xz[i + 1 :, 1] - points_xz[i, 1]
        d = float(np.max(np.hypot(dx, dz)))
        if d > best:
            best = d
    return best


def _object_class_from_id(obj_id: str) -> str:
    """Derive an object's class from its `id` field (e.g. "curtain_0" ->
    "curtain"), matching `export_glb.build_scene`'s node-naming convention
    (`f"{obj_id}/visual/part_{j}"`, `obj_id = obj["id"]`) and
    `bootstrap.py`'s `id = "<label>_<index>"` convention (see
    `load_object_inputs_from_hulls_json` / `ObjectFootprintInput`, and the
    `02_modular_home` fixture's `scene_objects_hulls.json`, e.g.
    `{"id": "bed_0", "label": "bed"}`). Stripping the trailing `_<index>`
    suffix recovers the label without needing the original `label` field,
    which isn't available once geometry has round-tripped through a GLB (the
    render-from-glb entry point only has the scene graph's node names)."""
    return _TRAILING_INDEX_RE.sub("", obj_id)


def _footprint_centroid_xz(points_xz: np.ndarray) -> tuple[float, float]:
    """Centroid of a point set's convex-hull footprint - order-independent
    (unlike `_room_centroid`, which assumes its input is already an ordered
    ring) and robust to uneven vertex density across an object's mesh parts
    (e.g. a lamp's cylindrical shade contributing far more vertices than its
    thin frame). Reuses `geometry.convex_hull_2d` for the ordered hull ring
    and `_room_centroid` for the actual (shapely, area-weighted) centroid -
    the same centroid routine used for the room polygon itself."""
    hull = convex_hull_2d(np.asarray(points_xz, dtype=float))
    return _room_centroid(hull) if len(hull) >= 3 else _room_centroid(np.asarray(points_xz, dtype=float))


def _bed_centroid_xz(scene: trimesh.Scene) -> tuple[float, float] | None:
    """Footprint centroid of "the bed" (task spec step 1), or None if there is
    no bed object in this scene - a real, expected scenario (not every scene
    has a bed), handled by `compute_camera` falling back to the room
    polygon's own centroid as the anchor reference in that case. Uses the
    object's VISUAL geometry (`<id>/visual/part_*` nodes; always present,
    unlike the optional collision hull) - `_object_class_from_id` picks out
    nodes whose id-derived class is "bed". If more than one object in the
    scene is classed "bed" (seen in the real hero scene: bed_0 and bed_1),
    all of their footprints are pooled into one combined footprint before
    taking the centroid, rather than picking one arbitrarily - a
    deterministic, order-independent way to define "the bed" when a scene
    happens to contain more than one bed-labeled object."""

    def is_bed_visual_node(node_name: str) -> bool:
        if "/visual/part_" not in node_name:
            return False
        obj_id = node_name.split("/", 1)[0]
        return _object_class_from_id(obj_id).lower() == "bed"

    pts = _points_xz_from_nodes(scene, is_bed_visual_node)
    return _footprint_centroid_xz(pts) if len(pts) else None


def _solve_pitch_for_target_screen_y(
    target_point_world: np.ndarray,
    camera_pos: np.ndarray,
    forward_xz: np.ndarray,
    fov_deg: float,
    image_height_px: int,
    target_screen_y: float,
    *,
    pitch_range_deg: tuple[float, float] = PITCH_SEARCH_RANGE_DEG,
    tol_deg: float = PITCH_SEARCH_TOLERANCE_DEG,
    max_iter: int = 60,
) -> float:
    """Task spec step 6: bisection search for the pitch that puts
    `target_point_world` (the farthest floor point along the view direction)
    at `target_screen_y` on screen. Pitch has a monotonic relationship with a
    given world point's screen-Y here (verified numerically, not just
    assumed): a steeper downward pitch (more negative) rotates the view
    direction further down, moving a floor point further from the horizon and
    thus HIGHER up the frame (smaller screen-Y); a shallower pitch (closer to
    0) moves it lower/further down the frame. `pitch_range_deg` defaults to
    (-75, -1) - steep enough to establish a small room, shallow enough to
    still read as "looking into the room" rather than straight down - and the
    search converges to within `tol_deg` (a twentieth of a degree - well
    under a pixel of resulting screen-Y error at this module's focal
    lengths/image sizes).

    Geometric caveat (not a bug): a given floor point's screen-Y has a finite
    supremum as pitch approaches the shallow end of any downward-looking
    range (it does NOT grow without bound as pitch flattens towards
    horizontal) - roughly `target_screen_y_cap ~= image_height/2 +
    (camera_height_m / point_depth) * focal_px`. For a point that's very deep
    relative to `camera_height_m` (a large or elongated room, where task 1's
    distance formula also pushes the camera further back), that cap can sit
    below `target_screen_y`, and the bisection then correctly saturates at
    `pitch_range_deg`'s shallow bound - the best achievable framing - rather
    than converging past it. This is expected geometry, not a solver defect.

    Reuses `_to_camera_space`/`_to_screen` (the module's one projection path)
    rather than re-deriving projection math - only the right/up/forward basis
    is recomputed per candidate pitch, matching what `compute_camera` itself
    does once the search has converged.
    """
    focal_px = (image_height_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cy_px = image_height_px / 2.0
    limit = 10.0 * image_height_px

    def screen_y_for_pitch(pitch_deg: float) -> float:
        pitch = math.radians(pitch_deg)
        forward = np.array([forward_xz[0] * math.cos(pitch), math.sin(pitch), forward_xz[1] * math.cos(pitch)])
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        cam_pt = _to_camera_space(target_point_world[None, :], camera_pos, right, up, forward)[0]
        if cam_pt[2] <= NEAR_M:
            # Degenerate: the point is behind/at the near plane for this
            # pitch. Treat it as far off the bottom of the frame so the
            # bisection steers away from this end of the range.
            return float(image_height_px * 10)
        return _to_screen(cam_pt[None, :], focal_px, 0.0, cy_px, limit)[0][1]

    lo, hi = pitch_range_deg  # lo: steepest (smallest screen_y); hi: shallowest (largest screen_y)
    for _ in range(max_iter):
        if hi - lo < tol_deg:
            break
        mid = (lo + hi) / 2.0
        if screen_y_for_pitch(mid) < target_screen_y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_camera(
    room_points_xz: np.ndarray,
    floor_y: float,
    *,
    bed_centroid_xz: tuple[float, float] | None = None,
    camera_height_m: float = CAMERA_HEIGHT_M,
    fov_deg: float = FOV_DEG,
    image_height_px: int = DEFAULT_IMAGE_SIZE[1],
    target_floor_frac: float = TARGET_FLOOR_SCREEN_FRAC,
    min_distance_m: float = MIN_CAMERA_DISTANCE_M,
    distance_diagonal_frac: float = CAMERA_DISTANCE_DIAGONAL_FRAC,
    pitch_search_range_deg: tuple[float, float] = PITCH_SEARCH_RANGE_DEG,
    pitch_tolerance_deg: float = PITCH_SEARCH_TOLERANCE_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (camera_pos, forward, right, up), all world-space unit vectors
    except camera_pos.

    T3 camera rule (replaces T2's fixed "4m back along the room's oriented-bbox
    long axis, -30deg pitch" - user feedback: that still produced a bad shot,
    an extreme close-up blocked by a curtain, because it ignored room size and
    furniture layout). Every number below comes from THIS scene's own geometry
    - no hardcoded per-scene tuning:

    1. Anchor point: `bed_centroid_xz` if given (the bed's footprint centroid -
       see `_bed_centroid_xz`), else `room_points_xz`'s own centroid as a
       documented fallback for the real, non-hypothetical case of a scene with
       no bed at all.
    2. Direction: the ray from the anchor, through the room polygon's own
       vertex farthest from the anchor ("the farthest corner"), continued
       outward beyond that corner.
    3. Distance: `max(distance_diagonal_frac * room_diagonal, min_distance_m)`
       from the ROOM CENTROID (the look-at point, not the anchor or the
       corner) - `room_diagonal` per `_room_diagonal`.
    4. Look-at: the room polygon's centroid (XZ), at `floor_y` (unchanged from
       T2's behavior - the room "front"/look-at height was always the floor).
    5. Height: `camera_height_m` above `floor_y` (default 2.6m, was 2.5m).
    6. Pitch: solved, not fixed - `_solve_pitch_for_target_screen_y` finds the
       pitch that puts the farthest floor-polygon point along the view
       direction at `target_floor_frac` (0.85) of the frame height, i.e. near
       the bottom of the image - an establishing shot showing most of the
       floor.
    7. The camera may end up outside the room polygon (backing off past the
       farthest corner routinely does this) - expected; T2b's near-wall
       culling (`_wall_occludes_view`) already handles walls between the
       camera and the interior.

    `room_points_xz` is used as-is, including any closing duplicate vertex a
    closed ring may carry (`bootstrap.extract_floor_polygon`'s output repeats
    its first vertex as its last) - a duplicate can only tie, never win, the
    farthest-point/diagonal searches below, so it doesn't need stripping.
    """
    pts = np.asarray(room_points_xz, dtype=float).reshape(-1, 2) if len(room_points_xz) else np.zeros((0, 2))
    cx, cz = _room_centroid(pts)
    room_centroid_xz = np.array([cx, cz])

    anchor_ref = np.array(bed_centroid_xz, dtype=float) if bed_centroid_xz is not None else room_centroid_xz

    if len(pts):
        dists_from_anchor = np.hypot(pts[:, 0] - anchor_ref[0], pts[:, 1] - anchor_ref[1])
        farthest_corner = pts[int(np.argmax(dists_from_anchor))]
    else:
        farthest_corner = anchor_ref

    dir_vec = farthest_corner - anchor_ref
    dir_len = float(np.hypot(*dir_vec))
    # Degenerate fallback (anchor coincides with the farthest corner, e.g. a
    # single-point/degenerate room): an arbitrary but deterministic direction,
    # same spirit as T2's "which end is arbitrary but fixed per room" note.
    direction = dir_vec / dir_len if dir_len > 1e-9 else np.array([1.0, 0.0])

    room_diagonal = _room_diagonal(pts)
    distance = max(distance_diagonal_frac * room_diagonal, min_distance_m)

    camera_xz = room_centroid_xz + direction * distance
    camera_pos = np.array([camera_xz[0], floor_y + camera_height_m, camera_xz[1]])

    forward_xz_vec = room_centroid_xz - camera_xz
    forward_xz_norm = float(np.hypot(*forward_xz_vec))
    forward_xz = forward_xz_vec / forward_xz_norm if forward_xz_norm > 1e-9 else -direction

    if len(pts):
        rel = pts - camera_xz
        depths = rel @ forward_xz
        farthest_floor_xz = pts[int(np.argmax(depths))]
        target_point_world = np.array([farthest_floor_xz[0], floor_y, farthest_floor_xz[1]])
        target_screen_y = target_floor_frac * image_height_px
        pitch_deg = _solve_pitch_for_target_screen_y(
            target_point_world,
            camera_pos,
            forward_xz,
            fov_deg,
            image_height_px,
            target_screen_y,
            pitch_range_deg=pitch_search_range_deg,
            tol_deg=pitch_tolerance_deg,
        )
    else:
        # No floor polygon at all (e.g. a totally empty scene) - nothing to
        # target the pitch solve against; fall back to a fixed reasonable
        # pitch rather than crashing. Not a "per-scene tuning" case - it's a
        # degenerate-input guard, exercised by the empty-scene smoke test.
        pitch_deg = FALLBACK_PITCH_DEG

    pitch = math.radians(pitch_deg)
    forward = np.array([forward_xz[0] * math.cos(pitch), math.sin(pitch), forward_xz[1] * math.cos(pitch)])
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return camera_pos, forward, right, up


# --- projection / clipping ---------------------------------------------------


def _to_camera_space(points_world: np.ndarray, camera_pos: np.ndarray, right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    rel = points_world - camera_pos
    return np.stack([rel @ right, rel @ up, rel @ forward], axis=-1)


def _clip_triangle_near(tri_cam: np.ndarray, near: float) -> list[np.ndarray]:
    """Sutherland-Hodgman clip of one triangle (3x3 camera-space verts)
    against cam_z > near. Returns 0-2 output triangles (each 3x3)."""
    inside = tri_cam[:, 2] > near
    n_in = int(inside.sum())
    if n_in == 0:
        return []
    if n_in == 3:
        return [tri_cam]
    poly: list[np.ndarray] = []
    for i in range(3):
        a, b = tri_cam[i], tri_cam[(i + 1) % 3]
        a_in, b_in = inside[i], inside[(i + 1) % 3]
        if a_in:
            poly.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            poly.append(a + t * (b - a))
    if len(poly) < 3:
        return []
    verts = np.array(poly)
    return [np.array([verts[0], verts[i], verts[i + 1]]) for i in range(1, len(verts) - 1)]


def _to_screen(tri_cam: np.ndarray, focal_px: float, cx_px: float, cy_px: float, limit: float) -> list[tuple[float, float]]:
    z = tri_cam[:, 2]
    sx = cx_px + (tri_cam[:, 0] / z) * focal_px
    sy = cy_px - (tri_cam[:, 1] / z) * focal_px
    sx = np.clip(sx, -limit, limit)
    sy = np.clip(sy, -limit, limit)
    return list(zip(sx.tolist(), sy.tolist()))


# --- scene -> faces -----------------------------------------------------------


def _wall_occludes_view(
    wall_mid_xz: tuple[float, float], camera_pos_xz: tuple[float, float], room_centroid_xz: tuple[float, float]
) -> bool:
    """Near-wall culling test (exact task spec): let `A = room_centroid -
    camera_pos` and `B = wall_midpoint - camera_pos` (both world XZ). The wall
    sits on the near side of the room - between the camera and the centroid,
    and would occlude the view into the room from this camera - iff `A` and
    `B` point the same general direction (`dot(A, B) > 0`) AND `B`'s
    projection length onto `A` is less than `|A|` itself
    (`dot(A, B) / |A| < |A|`), i.e. the wall's midpoint projects onto the
    camera->centroid ray at a point strictly before the centroid."""
    ax, az = room_centroid_xz[0] - camera_pos_xz[0], room_centroid_xz[1] - camera_pos_xz[1]
    bx, bz = wall_mid_xz[0] - camera_pos_xz[0], wall_mid_xz[1] - camera_pos_xz[1]
    a_len = math.hypot(ax, az)
    if a_len < 1e-9:
        return False
    dot = ax * bx + az * bz
    if dot <= 0.0:
        return False
    return (dot / a_len) < a_len


def _dollhouse_faces(
    tris_world: np.ndarray, room_centroid_xz: tuple[float, float], camera_pos_xyz: np.ndarray | None = None
) -> np.ndarray:
    """Faces of a closed wall band to keep for a dollhouse view: those whose
    (winding-derived) normal points toward `room_centroid_xz` in XZ - the
    room-facing side - or up (n_y > 0.5, the top cap), and - when the camera
    position is known - that face the camera (the viewer renders FrontSide, so
    the near wall's room-facing side, seen from behind, is a back-face and
    vanishes; the far wall's room-facing side looks at the camera and stays).
    Mirrors `frontend/src/lib/msaDollhouse.ts::dollhouseWallGeometry` + FrontSide."""
    if len(tris_world) == 0:
        return tris_world
    a, b, c = tris_world[:, 0, :], tris_world[:, 1, :], tris_world[:, 2, :]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-12
    n[ok] /= norm[ok][:, None]
    centers = tris_world.mean(axis=1)
    to_room_x = room_centroid_xz[0] - centers[:, 0]
    to_room_z = room_centroid_xz[1] - centers[:, 2]
    faces_room = to_room_x * n[:, 0] + to_room_z * n[:, 2] > 0
    keep = ok & ((n[:, 1] > 0.5) | ((n[:, 1] > -0.5) & faces_room))
    if camera_pos_xyz is not None:
        to_cam = np.asarray(camera_pos_xyz, dtype=float)[None, :] - centers
        keep &= (to_cam * n).sum(axis=1) > 0  # front-facing only
    return tris_world[keep]


def _iter_scene_node_faces(
    scene: trimesh.Scene,
    camera_pos_xz: tuple[float, float] | None = None,
    room_centroid_xz: tuple[float, float] | None = None,
    *,
    wall_opacity: float = WALL_OPACITY,
    camera_pos_xyz=None,
):
    """Yields (node_name, rgba, triangles_world [Nx3x3]) - one entry per
    renderable NODE (not per triangle) of a `build_scene()`-shaped Scene:
    wall_i -> translucent, floor -> opaque grey, `<id>/visual/part_*` ->
    opaque with the object's own color UNLESS the object's id-derived class
    (`_object_class_from_id`, case-insensitive) is one of
    `TRANSLUCENT_OBJECT_CLASSES` ("curtain", "drape", "blind", "mirror" - T3
    task 2: these soft furnishings/reflective surfaces get the same
    `TRANSLUCENT_OPACITY` (== `WALL_OPACITY`) as walls, since they can just as
    easily block a hero shot's view into the room as a near wall can - this is
    a render-only, cosmetic change; the object's physics/collision
    representation, a separate `<id>/collision/*` node skipped below, is
    untouched). Skips `<id>/collision/*` hulls (those exist for physics, not a
    hero-shot render) and the `Plan/*` outline curves / optional `Path`/
    `Target` markers (not Trimesh geometry, filtered out by the isinstance
    check below).

    Grouped per-node (rather than yielding bare triangles) so a translucent
    wall's own triangles - which self-overlap heavily in screen space at a
    grazing viewing angle, since a wall is extruded with real thickness and a
    3/4 view can end up looking almost along a wall's face - get composited as
    ONE 30%-opacity layer instead of each triangle separately compositing 30%
    on top of the last, which would converge towards a fully opaque wall for
    any camera angle grazing enough to self-overlap many triangles. See
    `_composite_translucent_node`. A translucent OBJECT node gets exactly the
    same treatment "for free": each `<id>/visual/part_*` node is already
    yielded (and later composited in `_render_scene`) as its own entry, so a
    translucent curtain's self-overlapping triangles within one part don't
    compound-darken either - the same bug T2 fixed for walls, not
    reintroduced here for objects.

    Near-wall culling: when `camera_pos_xz`/`room_centroid_xz` are supplied, a
    `wall_i` node whose midpoint - the centroid of its own (already
    GLB-derived) world-space vertices projected to XZ, the simplest thing that
    works here without going back to the original `WallPolygon` - sits between
    the camera and the room centroid (per `_wall_occludes_view`) is dropped
    entirely instead of being yielded at reduced opacity. This closes a gap
    T2 itself flagged: with the camera only a few meters outside the room, the
    nearest wall is viewed close-up and near edge-on, so even at 30% opacity -
    compounded across that wall's own thickness and/or a second wall layer
    along the same ray - it reads as an opaque wall blocking the room's
    interior (reported as a real visual defect). Floor/object nodes and
    far-side walls are unaffected; omitting both args (the default) disables
    culling entirely, preserving old behavior for any other caller.
    """
    can_cull = camera_pos_xz is not None and room_centroid_xz is not None
    for node_name in scene.graph.nodes_geometry:
        if "/collision/" in node_name or node_name.startswith("Plan"):
            continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if geom is None or not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        verts_world = trimesh.transform_points(geom.vertices, transform)
        tris_world = verts_world[geom.faces]

        if node_name.startswith("wall_"):
            if can_cull:
                if node_name == "wall_outline":
                    # T15g: one closed band around the whole room - its vertex
                    # mean IS the room centroid, so the midpoint rule cannot
                    # decide. Cull per face instead, the dollhouse way
                    # (frontend/src/lib/msaDollhouse.ts): keep faces whose normal
                    # points into the room (toward the centroid) or up (top cap),
                    # drop the outward/bottom faces - the near side of the band
                    # opens up, the far side stays.
                    tris_world = _dollhouse_faces(tris_world, room_centroid_xz, camera_pos_xyz)
                    if len(tris_world) == 0:
                        continue
                else:
                    wall_mid_xz = (float(verts_world[:, 0].mean()), float(verts_world[:, 2].mean()))
                    if _wall_occludes_view(wall_mid_xz, camera_pos_xz, room_centroid_xz):
                        continue
            rgba = (*WALL_RGB, int(round(wall_opacity * 255)))
        elif node_name == "floor":
            rgba = (*FLOOR_RGB, 255)
        else:
            # Vertex colours (pre-T15c exports) or the PBR baseColorFactor (T15c
            # `object_material="pbr"` default) - see export_glb.mesh_base_color_rgb.
            base = mesh_base_color_rgb(geom)
            if base is not None:
                r, g, b = base
            else:
                r, g, b = DEFAULT_OBJECT_RGB
            obj_id = node_name.split("/", 1)[0]
            obj_class = _object_class_from_id(obj_id).lower()
            alpha = int(round(TRANSLUCENT_OPACITY * 255)) if obj_class in TRANSLUCENT_OBJECT_CLASSES else 255
            rgba = (int(r), int(g), int(b), alpha)
        yield node_name, rgba, tris_world


def _points_xz_from_nodes(scene: trimesh.Scene, predicate) -> np.ndarray:
    pts = []
    for node_name in scene.graph.nodes_geometry:
        if not predicate(node_name):
            continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
            continue
        verts_world = trimesh.transform_points(geom.vertices, transform)
        pts.append(verts_world[:, [0, 2]])
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


def _estimate_floor_y(scene: trimesh.Scene) -> float:
    """A GLB doesn't carry `floor_y` explicitly. Approximate it as the top
    surface of the `floor` node (`export_glb._floor_mesh` builds the slab from
    floor_y-thickness up to floor_y, i.e. floor_y is its max Y), falling back
    to the lowest wall vertex (walls start exactly at floor_y) when there is no
    floor node, and 0.0 if there's neither."""
    for node_name in scene.graph.nodes_geometry:
        if node_name == "floor":
            transform, geom_name = scene.graph[node_name]
            geom = scene.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices):
                verts_world = trimesh.transform_points(geom.vertices, transform)
                return float(verts_world[:, 1].max())
    ys = []
    for node_name in scene.graph.nodes_geometry:
        if node_name.startswith("wall_"):
            transform, geom_name = scene.graph[node_name]
            geom = scene.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices):
                verts_world = trimesh.transform_points(geom.vertices, transform)
                ys.append(float(verts_world[:, 1].min()))
    return min(ys) if ys else 0.0


# --- painter's-algorithm render ------------------------------------------------


def _composite_translucent_node(
    canvas: Image.Image, screen_tris: list[list[tuple[float, float]]], rgba: tuple[int, int, int, int], width: int, height: int
) -> None:
    """Composite one node's (e.g. one wall's) triangles as a SINGLE translucent
    layer: every triangle is first drawn fully opaque onto a private overlay
    (so triangles belonging to the same surface just overwrite each other,
    never stack), then the whole overlay's alpha is scaled down to the target
    opacity in one shot before compositing onto the canvas. This is what keeps
    a wall at its intended opacity even where its own triangles self-overlap
    heavily in screen space (see `_iter_scene_node_faces`)."""
    xs = [p[0] for tri in screen_tris for p in tri]
    ys = [p[1] for tri in screen_tris for p in tri]
    if not xs:
        return
    x0, x1 = max(0, int(math.floor(min(xs)))), min(width, int(math.ceil(max(xs))) + 1)
    y0, y1 = max(0, int(math.floor(min(ys)))), min(height, int(math.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    r, g, b, a = rgba
    overlay = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for tri in screen_tris:
        local_pts = [(x - x0, y - y0) for x, y in tri]
        draw.polygon(local_pts, fill=(r, g, b, 255))
    arr = np.asarray(overlay).astype(np.float32)
    arr[..., 3] *= a / 255.0
    overlay = Image.fromarray(arr.astype(np.uint8), "RGBA")
    region = canvas.crop((x0, y0, x1, y1))
    region.alpha_composite(overlay)
    canvas.paste(region, (x0, y0))


def _render_scene(
    scene: trimesh.Scene,
    camera_points_xz: np.ndarray,
    floor_y: float,
    out_path: Path,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    camera_height_m: float = CAMERA_HEIGHT_M,
    fov_deg: float = FOV_DEG,
    background_rgb: tuple[int, int, int] = BACKGROUND_RGB,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    wall_opacity: float = WALL_OPACITY,
    cull_near_walls: bool = True,
    opaque_walls: bool = False,
) -> None:
    """`camera` (T15d, optional): an explicit `(camera_pos, forward, right, up)` -
    e.g. `compute_camera`'s own output after `scripts.msa.record_isaac.camera_backoff_for_framing`
    backed it off, or a top-down pose - used verbatim instead of calling
    `compute_camera` here. `wall_opacity`/`cull_near_walls` let the presentation
    still draw its 1.2 m wall stubs near-opaque and keep the near stubs (you look
    over a stub, so culling it would hide the dollhouse edge).

    `opaque_walls=True` (T15i, the "faithful preview" mode export_presentation's
    still uses): walls are drawn fully opaque through the per-triangle painter's
    path and NO near-wall culling happens here - exactly what Isaac does with the
    presentation USD's stubs (displayColor, no opacity), so whatever the exporter
    already culled (invisible stub edges are simply not in the scene) is all that
    opens the room up. T15e run2's preview hid the occlusion because this renderer
    drew the stubs at 0.92 opacity."""
    if opaque_walls:
        wall_opacity, cull_near_walls = 1.0, False
    width, height = image_size
    if camera is None:
        # Bed footprint centroid (task 1's anchor point), or None - `compute_camera`
        # documents the room-centroid fallback for that case. Derived from `scene`
        # itself (not `camera_points_xz`, which is just the room outline) since it
        # needs the object nodes' own geometry.
        bed_centroid_xz = _bed_centroid_xz(scene)
        camera_pos, forward, right, up = compute_camera(
            camera_points_xz,
            floor_y,
            bed_centroid_xz=bed_centroid_xz,
            camera_height_m=camera_height_m,
            fov_deg=fov_deg,
            image_height_px=height,
        )
    else:
        camera_pos, forward, right, up = (np.asarray(v, dtype=float) for v in camera)
    # Reuse `_room_centroid` (the same routine `compute_camera` itself calls)
    # rather than re-deriving the room centroid, so near-wall culling always
    # agrees with wherever `compute_camera` decided to point the camera.
    room_centroid_xz = _room_centroid(camera_points_xz) if cull_near_walls else None
    camera_pos_xz = (float(camera_pos[0]), float(camera_pos[2])) if cull_near_walls else None
    focal_px = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx_px, cy_px = width / 2.0, height / 2.0
    limit = 10.0 * max(width, height)

    canvas = Image.new("RGBA", (width, height), (*background_rgb, 255))
    draw = ImageDraw.Draw(canvas)

    # Opaque geometry (floor, objects) is depth-sorted per TRIANGLE, same as a
    # normal painter's algorithm. Translucent geometry (walls) is instead kept
    # grouped per NODE - see `_iter_scene_node_faces` / `_composite_translucent_node`
    # for why (self-overlapping triangles of one wall must not compound opacity).
    entries: list[tuple[float, str, object, tuple[int, int, int, int]]] = []
    for node_name, rgba, tris_world in _iter_scene_node_faces(
        scene, camera_pos_xz, room_centroid_xz, wall_opacity=wall_opacity, camera_pos_xyz=camera_pos if cull_near_walls else None
    ):
        if rgba[3] >= 255:
            for tri_world in tris_world:
                tri_cam = _to_camera_space(tri_world, camera_pos, right, up, forward)
                for clipped in _clip_triangle_near(tri_cam, NEAR_M):
                    avg_z = float(clipped[:, 2].mean())
                    pts = _to_screen(clipped, focal_px, cx_px, cy_px, limit)
                    entries.append((avg_z, "opaque", pts, rgba))
        else:
            node_tris_screen: list[list[tuple[float, float]]] = []
            zs: list[float] = []
            for tri_world in tris_world:
                tri_cam = _to_camera_space(tri_world, camera_pos, right, up, forward)
                for clipped in _clip_triangle_near(tri_cam, NEAR_M):
                    zs.append(float(clipped[:, 2].mean()))
                    node_tris_screen.append(_to_screen(clipped, focal_px, cx_px, cy_px, limit))
            if node_tris_screen:
                entries.append((sum(zs) / len(zs), "translucent", node_tris_screen, rgba))

    entries.sort(key=lambda e: e[0], reverse=True)  # farthest first (painter's algorithm)

    for _avg_z, kind, payload, rgba in entries:
        if kind == "opaque":
            draw.polygon(payload, fill=rgba)
        else:
            _composite_translucent_node(canvas, payload, rgba, width, height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)


# --- public API ---------------------------------------------------------------


def render_hero_perspective(
    walls: list[WallPolygon],
    floor_polygon: list[tuple[float, float]] | None,
    floor_y: float,
    ceiling_y: float,
    objects: list[dict],
    out_path: Path,
    **camera_kwargs,
) -> None:
    """3/4-view hero render. Same inputs as `export_glb.build_scene` (see its
    docstring for the `objects` dict shape) - this function builds the render
    geometry with `build_scene` itself rather than re-deriving wall/floor
    extrusion, so the perspective render and the GLB export can never disagree
    about wall/floor shape. `camera_kwargs` forwards to `_render_scene`
    (`camera_height_m`, `fov_deg`, `image_size`, `background_rgb`) for callers
    that want to deviate from the module's defaults - camera POSITION/pitch
    are no longer tunable here, since `compute_camera` now derives them
    entirely from the scene's own geometry (see its docstring).
    """
    scene = build_scene(walls, floor_polygon, floor_y, ceiling_y, objects)
    camera_points_xz = _room_footprint_xz(floor_polygon, walls)
    _render_scene(scene, camera_points_xz, floor_y, Path(out_path), **camera_kwargs)


def render_hero_perspective_from_glb(scene_glb_path: Path, out_path: Path, **camera_kwargs) -> None:
    """Alternate entry point for callers that already have a bootstrap-exported
    `scene.glb` (this is what the CLI below uses - see the module docstring for
    why reading the GLB, rather than re-deriving geometry from a raw occupancy
    scene dir, is the intentional choice here)."""
    loaded = trimesh.load(str(scene_glb_path))
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    camera_points_xz = _points_xz_from_nodes(scene, lambda n: n == "floor")
    if len(camera_points_xz) < 3:
        camera_points_xz = _points_xz_from_nodes(scene, lambda n: n.startswith("wall_"))
    floor_y = _estimate_floor_y(scene)
    _render_scene(scene, camera_points_xz, floor_y, Path(out_path), **camera_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 3/4-view hero PNG of a bootstrapped room from its scene.glb.")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Directory containing an already-exported scene.glb (e.g. a prior `bootstrap.py --out-dir`). "
        "NOTE: unlike bootstrap.py's own --scene-dir (a raw occupancy scene dir), this is a bootstrap OUTPUT "
        "dir - see the module docstring for why.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-glb", type=Path, default=None, help="Explicit scene.glb path, overriding <scene-dir>/scene.glb")
    parser.add_argument("--out-name", default="07_hero_perspective.png")
    args = parser.parse_args()

    scene_glb = args.scene_glb or (args.scene_dir / "scene.glb")
    out_path = args.out_dir / args.out_name
    render_hero_perspective_from_glb(scene_glb, out_path)
    print(json.dumps({"scene_glb": str(scene_glb), "out_path": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
