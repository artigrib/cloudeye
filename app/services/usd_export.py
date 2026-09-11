"""Export a scene to OpenUSD for NVIDIA Isaac Sim.

Deliberately does NOT export the point cloud as simulated geometry: `UsdGeomPoints`
renders but has no collision in Isaac, so a robot would drive straight through it. The
simulable scene is instead assembled from the occupancy grid (walls/obstacles) and,
per object, the convex hull of that object's own captured point cloud
(`scene_objects.mesh_path`) - data that's already validated elsewhere in this pipeline
(see scene_validator.py), not re-derived here. A convex hull is a conservative bound
(never smaller than the true geometry, only larger), which is the safe direction of
error for navigation - unlike the axis-aligned bbox it replaces, which could swallow
the empty space under a chair or table. When a point cloud isn't usable (missing file,
too few points, degenerate/coplanar), the object falls back to its axis-aligned bbox,
same as before - see `build_convex_hull` and the `cloudeye:collisionSource` attribute
written on each object prim.

Floor/Structure/Objects are all STATIC collision only (`UsdPhysics.CollisionAPI`, no
`RigidBodyAPI`) - matches what the frontend's own export panel already promises
(`IsaacSimExport.tsx`: "the file has static collision only, no articulation,
materials, or visual mesh beyond the colliders themselves"). Objects briefly carried
`RigidBodyAPI`+`MassAPI` (dynamic furniture) during the initial feat/isaac-demo-look
pass; a real GPU run found this scene's convex-hull colliders dynamically unstable
against each other/Structure (unbounded PhysX divergence, not normal settling - see
docs/DECISIONS.md), and removing dynamics from Objects entirely both matches the
frontend's contract and sidesteps that instability outright, rather than working
around it downstream (as `demo/record_isaac.py` previously had to).

Coordinate frame: this project's data is Y-up, floor at Y=0, metric (see models.py's
Scene.floor_y / Scene.ceiling_y). Isaac Sim expects Z-up. Every point is remapped
`(x, y, z) -> (x, -z, y)` on export (see `to_isaac`) - a +90deg rotation about X, not
an axis swap. This has determinant +1 (a proper rotation): winding order and normals
are preserved automatically, so `_add_hull_mesh` does NOT need to (and must not) flip
triangle winding - see its own comment. (An earlier version of this module used the
naive swap `(x, z, y)`, determinant -1, which mirrored the whole exported scene
left-right; fixed 2026-09-03, see docs/DECISIONS.md.) `Scene.is_reflected` is a
separate, upstream concern - it reflects whether the GPU pipeline's own floor/ceiling
alignment transform (gpu/stage_align.py) was itself a reflection, not anything about
this export step - and is stamped into the stage's custom layer metadata below purely
for observability; this module does not compensate for it.

Pure / DB-free by design (same rationale as scene_ingest.py and pathfinding.py's
`ObjectFootprint`): every input is a plain dataclass or numpy array, not an ORM row, so
this is unit-testable against tests/fixtures/ without a database.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from scipy.spatial import ConvexHull, QhullError

from app.services.ply_reader import PlyParseError, read_ply
from app.services.scene_ingest import GridMeta

logger = logging.getLogger(__name__)

OBSTACLE = 1  # occupancy_path cell value, see Scene.occupancy_path's docstring/schema

# UsdGeomPoints visual layer point cap. The pipeline's own aligned-room .ply files run
# into the millions of vertices (a real capture: 3.17M) - embedding all of them would
# make the "optional visual layer" dominate the file's size for no simulation benefit
# (this layer has no CollisionAPI - see module docstring). Deterministic subsample.
# purpose=guide (not the primary visible layer any more - see POINTCLOUD_INSTANCER_*
# below): kept for consumers other than Isaac's RTX Replicator pipeline, which was
# confirmed (feat/isaac-demo-look, docs/DECISIONS.md) to not render UsdGeomPoints at
# all regardless of purpose/extent/width.
POINTCLOUD_MAX_POINTS = 500_000
_POINTCLOUD_SUBSAMPLE_SEED = 0

# The actual visible-in-Isaac point cloud layer: a UsdGeomPointInstancer of small
# cubes (RTX renders PointInstancer instances fine - unlike raw UsdGeomPoints, see
# above). Voxel-downsampled (not randomly subsampled like POINTCLOUD_MAX_POINTS above)
# to a much lower cap - one cube per occupied voxel is a real geometry instance, not a
# point sprite, so the practical count ceiling for "loads fast, still reads as a
# cloud" is far lower than the 500k Points figure.
POINTCLOUD_INSTANCER_MAX_POINTS = 200_000
POINTCLOUD_INSTANCER_CUBE_SIZE_M = 0.012
POINTCLOUD_INSTANCER_INITIAL_VOXEL_M = 0.02

FLOOR_THICKNESS_M = 0.02

# isaac_validate.py's sphere_drop check drops a 0.1m-radius test sphere from 2m
# (default physics_dt=1/240, ~6 m/s impact speed) and it tunnels straight through
# FLOOR_THICKNESS_M's 2cm slab at that speed/step size. Rather than tune physics config
# (CCD, smaller dt), add a second, thicker static collider immediately beneath the
# floor - purely a collision safety margin, not part of the visible floor surface.
FLOOR_SLAB_THICKNESS_M = 0.1

# Exported scenes had zero light prims until this was added, which made offscreen
# RTX renders (e.g. Isaac Sim demos/screenshots) pitch black regardless of camera
# framing. Intensity matches what an ad hoc demo script confirmed gives a correctly
# lit, shadowed render against these scenes' scale.
DOME_LIGHT_INTENSITY = 1200.0

# A dome alone is flat - no directional shadow to read volume/depth by. Both numbers
# below are GPU-verified, not guessed: the 2026-09-04 Isaac Sim 6.0.1 run (vast.ai
# instance 49860715, see docs/DECISIONS.md's "OPEN - exported USD scenes carry zero
# light prims" entry) confirmed DomeLight(1200) + this exact DistantLight raised a
# pitch-black RTX render's frame mean from 63.73 to ~226-228. That fix was only ever
# applied in a throwaway demo script, not here - reusing its numbers verbatim rather
# than re-deriving new ones that have never actually been rendered.
KEY_LIGHT_INTENSITY = 3000.0
KEY_LIGHT_ROTATE_XYZ_DEG = (-60.0, 20.0, 0.0)

# --- Isaac demo-look colors -----------------------------------------------------------
# Floor/Structure colors are pinned to the 2D occupancy-map legend's free/obstacle
# colors (frontend/src/lib/tokens.ts's OCC_FREE/OCC_OBSTACLE - tokens.test.ts asserts
# those triples match index.css) so the Isaac export reads as a continuation of the
# app's own 2D view rather than an unrelated 3D palette. No shared color-token pipeline
# exists between the TS frontend and this Python module, so `_oklch_to_srgb` below is a
# hand-ported copy of tokens.ts's `oklchToSrgb` - keep them in sync by hand.
FLOOR_OPACITY = 0.35
STRUCTURE_OPACITY = 0.35
OBJECT_OPACITY = 0.5

# Structure boxes are exported at this fixed low height rather than the room's true
# ceiling_y - low, translucent colliders so the point cloud above them stays visible
# through the walls, per the Isaac demo-look spec. `ceiling_y` remains a required
# UsdExportInput field regardless (the /usd endpoint 409s a scene without one) - that
# check guards "scene has valid vertical extents" generally, independent of this
# export's specific choice of visual/collider wall height.
STRUCTURE_HEIGHT_M = 0.6

# Rule: an object whose ENTIRE bbox sits at or above this height (floor is Y=0) is
# excluded from the export outright - no prim, no collider, at all. Found via a real
# GPU render (feat/isaac-demo-look v3, 02_modular_home): two ceiling-mounted "fan"
# objects (bbox y in [2.36, 2.74] and [2.62, 2.74], real ceiling_y ~2.71) rendered as
# two disconnected hulls floating conspicuously above the room outline - correct
# geometry at its true captured height, but visually broken against
# STRUCTURE_HEIGHT_M's low 0.6m walls (the whole point of which is a room silhouette
# a viewer reads at a glance, not literal wall height). An object entirely above 2m is
# also not a real navigation concern for a ground robot regardless of how it's
# rendered - excluding it is a export-content decision, not just a visual patch.
# Deliberately keyed on `bbox_min` (the object's LOWEST point), not its centroid or
# max: an object with a low base that merely EXTENDS above 2m (a tall shelf, a
# floor-to-ceiling curtain) is still a real obstacle at floor level and must stay.
OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M = 2.0

# One consistent color per object "class". SceneObject has no separate category column
# (see models.py) - `name` (a short noun like "chair"/"table" from the ingest
# vocabulary LLM call) already IS the class. Hue is a deterministic hash of the
# normalized name, NOT Python's built-in `hash()` (randomized per-process via
# PYTHONHASHSEED - would make an object's color flicker between runs of the same
# scene); `zlib.crc32` has no such randomization.
_OBJECT_COLOR_L = 0.8  # tokens.ts DATA_OBJECT.l
_OBJECT_COLOR_C = 0.16  # tokens.ts DATA_OBJECT.c


def _srgb_transfer(linear: float) -> float:
    clamped = min(1.0, max(0.0, linear))
    if clamped <= 0.0031308:
        return clamped * 12.92
    return 1.055 * (clamped ** (1 / 2.4)) - 0.055


def _oklch_to_srgb(l: float, c: float, h_deg: float) -> tuple[float, float, float]:
    """Port of frontend/src/lib/tokens.ts's `oklchToSrgb` (OKLCH -> OKLab -> linear
    sRGB -> gamma-encoded sRGB; see that file for the reference/derivation)."""
    h = math.radians(h_deg)
    a = c * math.cos(h)
    b = c * math.sin(h)

    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.291485548 * b

    l3, m3, s3 = l_**3, m_**3, s_**3

    r_lin = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3

    return _srgb_transfer(r_lin), _srgb_transfer(g_lin), _srgb_transfer(b_lin)


FLOOR_COLOR = _oklch_to_srgb(0.24, 0.0, 0.0)  # tokens.ts OCC_FREE
STRUCTURE_COLOR = _oklch_to_srgb(0.62, 0.0, 0.0)  # tokens.ts OCC_OBSTACLE
PATH_COLOR = _oklch_to_srgb(0.7, 0.16, 162.0)  # tokens.ts ACCENT


def _object_class_color(name: str) -> tuple[float, float, float]:
    hue = zlib.crc32(name.strip().lower().encode("utf-8")) % 360
    return _oklch_to_srgb(_OBJECT_COLOR_L, _OBJECT_COLOR_C, float(hue))


def _object_class_material_path(name: str) -> str:
    return f"/World/Materials/object_{_sanitize_prim_name(name.strip().lower())}"


# PhysX's own convex-collider vertex limit is around 256; capping well below that keeps
# the exported mesh small and leaves headroom regardless of PhysX cooking version. An
# object whose full hull exceeds this gets rebuilt from a uniformly subsampled cloud
# instead (see `build_convex_hull`) - a hull built from <= MAX_HULL_VERTICES input
# points can never have more than that many vertices, so the cap is always met.
MAX_HULL_VERTICES = 64
_HULL_MIN_POINTS = 4  # scipy needs >= d+1 = 4 non-coplanar points for a 3D hull
_HULL_DECIMATE_SAMPLE_SIZES = (1024, 256, MAX_HULL_VERTICES)
_HULL_DECIMATE_SEED = 0


@dataclass(frozen=True)
class UsdExportObject:
    """The minimal shape-independent description of a scene object this module needs -
    same decoupling rationale as pathfinding.py's ObjectFootprint."""

    name: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    num_views: int
    num_points: int
    is_fragment: bool
    # Path to the object's own captured point cloud (scene_objects.mesh_path) - the
    # source for its convex-hull collider. None (or an unreadable/too-sparse/degenerate
    # file) falls back to the axis-aligned bbox above.
    mesh_path: str | None = None


@dataclass(frozen=True)
class HullBuildResult:
    """A triangulated convex hull, still in the source Y-up frame (not yet rotated into
    Isaac's Z-up)."""

    vertices: np.ndarray  # (M, 3) float64
    faces: np.ndarray  # (K, 3) int64, indices into `vertices`, outward-oriented
    volume_m3: float
    decimated: bool  # True if built from a subsample rather than the full cloud


@dataclass(frozen=True)
class UsdExportInput:
    grid_cells: np.ndarray  # uint8 (width, height), [ix, iz] - see pathfinding.py
    grid_meta: GridMeta
    ceiling_y: float
    robot_start: tuple[float, float] | None  # (x, z) world meters, or None
    is_reflected: bool
    robot_radius_m: float
    objects: list[UsdExportObject]
    pointcloud_path: str | Path | None = None
    include_fragments: bool = False
    # /World/Path: a visible polyline through these Y-up world (x, z) waypoints (e.g.
    # from pathfinding.plan_to_object/plan_path's PathResult.points). None/empty omits
    # the prim entirely.
    path_points: list[tuple[float, float]] | None = None
    # /World/Target: an Xform at this Y-up world (x, z) goal point. None omits the prim.
    target_point: tuple[float, float] | None = None
    # Index into `objects` this target corresponds to, if any - used to point
    # /World/Target's cloudeye:targetObject relationship at the actual resolved prim.
    # None (or an index that ends up excluded, e.g. a fragment with
    # include_fragments=False) still gets the /World/Target Xform, just without the
    # relationship.
    target_object_index: int | None = None


@dataclass(frozen=True)
class ExportStats:
    obstacle_cell_count: int
    structure_box_count: int
    object_count: int
    fragment_count: int
    pointcloud_point_count: int
    pointcloud_instancer_count: int
    hull_object_count: int
    hull_decimated_object_count: int
    bbox_fallback_object_count: int
    elevated_excluded_object_count: int


# Bumped whenever a change to this module changes the exported geometry/metadata for
# an already-cached scene, so a stale file on disk (written by an older version of this
# module) is never served after the fix that produced it ships - the cache key alone
# (include_pointcloud/include_fragments/robot_radius_m) has no way to express "the code
# changed," only "the query params changed." Bumped 2026-09-03 for the to_isaac
# reflection->rotation fix (see module docstring, docs/DECISIONS.md) - files cached
# before that fix were silently mirrored and must not keep being served.
# Bumped again 2026-09-05 for the Isaac demo-look pass (feat/isaac-demo-look):
# Floor/Structure/Objects now carry displayColor/displayOpacity, Structure boxes are a
# fixed STRUCTURE_HEIGHT_M instead of the room's ceiling_y, the point cloud's purpose
# changed from guide to render (guide geometry is excluded from Hydra's default
# render-purpose set, which was silently hiding it from every RTX render even though
# the data was already in the file), and a DistantLight key light joins the DomeLight.
# Bumped again 2026-09-05 (same day, second pass) for: the point cloud's primary
# visible-in-Isaac layer is now a PointInstancer (`_add_pointcloud_instancer`), not
# UsdGeomPoints - Isaac's RTX Replicator pipeline was confirmed to not render Points at
# all regardless of purpose, so the `guide`->`render` purpose change above didn't
# actually fix visibility; UsdGeomPoints itself reverts to purpose=guide. Objects lost
# RigidBodyAPI/MassAPI (static collision only now, matching the frontend's own
# contract - a real GPU run found dynamic Object colliders unstable). Floor/Structure/
# Object materials gained an actual bound UsdPreviewSurface (displayOpacity alone
# wasn't enough for Isaac's RTX renderer to show them as translucent). New optional
# /World/Path and /World/Target prims.
# Bumped again 2026-09-05 (third pass, pre-merge review): objects entirely above
# OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M (2.0m) are no longer exported at all - see that
# constant's comment (found via real GPU render: two ceiling-mounted "fan" objects
# rendered as disconnected hulls floating above the room's now-low Structure walls).
EXPORT_SCHEMA_VERSION = 5


def cache_filename(*, include_pointcloud: bool, include_fragments: bool, robot_radius_m: float) -> str:
    """Deterministic filename per (EXPORT_SCHEMA_VERSION, include_pointcloud,
    include_fragments, robot_radius_m) combination, so different query params never
    collide on the same cached file (per the /{id}/usd endpoint's "regenerate, don't
    serve another variant's cache" spec) AND a stale file from a previous version of
    this module's export logic is never mistaken for a cache hit against the current one.

    `robot_radius_m` only affects the stamped `cloudeye:robotRadiusM` custom attribute
    today, not any collision geometry - but it still has to be part of the cache key:
    without it, requesting the USD for platform A then platform B would silently keep
    serving platform A's cached file (and its radius) for every platform after the
    first one to hit this endpoint for a given scene - the same class of bug the
    reachability cache key guards against in pathfinding.compute_reachability_cached."""
    return (
        f"scene_v{EXPORT_SCHEMA_VERSION}_pc{int(include_pointcloud)}"
        f"_frag{int(include_fragments)}_r{robot_radius_m:.4f}.usd"
    )


def to_isaac(x: float, y: float, z: float) -> Gf.Vec3d:
    """Y-up (x, y, z) -> Z-up (x, -z, y): a +90deg rotation about X, determinant +1.
    See module docstring for why this is a proper rotation, not a reflection."""
    return Gf.Vec3d(x, -z, y)


# --- Occupancy grid -> merged structure boxes ----------------------------------------


def merge_obstacle_boxes(cells: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy union of OBSTACLE grid cells into axis-aligned rectangles.

    For each row (fixed iz), merges consecutive OBSTACLE cells along ix into a run,
    then extends that run downward (increasing iz) as long as every cell in the next
    row's same ix-range is OBSTACLE and not yet claimed by an earlier box. This is not
    optimal rectangle packing, but it keeps the structure prim count in the hundreds
    for a typical room instead of one cube per cell (thousands) - see the /usd
    endpoint's docstring for why that matters to Isaac's scene load time.

    Returns (ix0, iz0, ix1, iz1) box bounds in grid-cell space, ix1/iz1 EXCLUSIVE.
    """
    width, height = cells.shape
    obstacle = cells == OBSTACLE
    visited = np.zeros_like(obstacle, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for iz in range(height):
        ix = 0
        while ix < width:
            if obstacle[ix, iz] and not visited[ix, iz]:
                ix0 = ix
                while ix < width and obstacle[ix, iz] and not visited[ix, iz]:
                    ix += 1
                ix1 = ix
                iz1 = iz + 1
                while (
                    iz1 < height
                    and obstacle[ix0:ix1, iz1].all()
                    and not visited[ix0:ix1, iz1].any()
                ):
                    iz1 += 1
                visited[ix0:ix1, iz:iz1] = True
                boxes.append((ix0, iz, ix1, iz1))
            else:
                ix += 1
    return boxes


def _cell_box_world_bounds(
    box: tuple[int, int, int, int], meta: GridMeta
) -> tuple[float, float, float, float]:
    ix0, iz0, ix1, iz1 = box
    x0 = meta.origin_x + ix0 * meta.resolution
    x1 = meta.origin_x + ix1 * meta.resolution
    z0 = meta.origin_z + iz0 * meta.resolution
    z1 = meta.origin_z + iz1 * meta.resolution
    return x0, x1, z0, z1


# --- Object point cloud -> convex hull ------------------------------------------------


def _hull_from_points(points: np.ndarray, *, decimated: bool) -> HullBuildResult | None:
    """Run scipy's ConvexHull on `points` and compact the result down to just the hull's
    own vertices (scipy's `.vertices`/`.simplices` still index into the full input
    array). Returns None if the points are degenerate (all coplanar, etc - scipy raises
    QhullError for that, not a return value)."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None

    index_map = {int(orig): i for i, orig in enumerate(hull.vertices)}
    vertices = points[hull.vertices]

    # scipy's simplices aren't guaranteed consistently wound - reorient each triangle
    # so its vertex order is outward-facing (matches `hull.equations`' outward normal).
    # `to_isaac`'s rotation (det=+1) preserves this outward orientation as-is - see
    # `_add_hull_mesh`, which no longer needs to flip winding for the Isaac remap.
    faces = np.empty((hull.simplices.shape[0], 3), dtype=np.int64)
    for i, (tri, eq) in enumerate(zip(hull.simplices, hull.equations)):
        a, b, c = (int(v) for v in tri)
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        if np.dot(normal, eq[:3]) < 0:
            b, c = c, b
        faces[i] = (index_map[a], index_map[b], index_map[c])

    return HullBuildResult(
        vertices=vertices, faces=faces, volume_m3=float(hull.volume), decimated=decimated
    )


def build_convex_hull(points: np.ndarray) -> HullBuildResult | None:
    """Build an outward-oriented triangulated convex hull from an object's point cloud,
    capped at MAX_HULL_VERTICES vertices.

    Returns None if no hull can be built at all - fewer than 4 points, or all points
    coplanar/degenerate - so the caller falls back to the object's bbox.
    """
    if points.shape[0] < _HULL_MIN_POINTS:
        return None

    hull = _hull_from_points(points, decimated=False)
    if hull is None or hull.vertices.shape[0] <= MAX_HULL_VERTICES:
        return hull

    # Too many vertices for a collision hull - rebuild from a uniformly subsampled
    # cloud rather than post-processing the full hull (simpler, and a hull built from
    # <= MAX_HULL_VERTICES points is guaranteed to satisfy the cap, so the last sample
    # size in the sequence always succeeds unless the subsample itself is degenerate).
    rng = np.random.default_rng(_HULL_DECIMATE_SEED)
    for sample_size in _HULL_DECIMATE_SAMPLE_SIZES:
        n = min(sample_size, points.shape[0])
        sample = points[rng.choice(points.shape[0], size=n, replace=False)]
        candidate = _hull_from_points(sample, decimated=True)
        if candidate is not None and candidate.vertices.shape[0] <= MAX_HULL_VERTICES:
            return candidate
    return None


# --- Prim naming -----------------------------------------------------------------------


def _sanitize_prim_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not s or s[0].isdigit():
        s = f"_{s}"
    return s


# --- Stage assembly --------------------------------------------------------------------


def _apply_translucent_material(
    stage: Usd.Stage, prim: Usd.Prim, material_path: str, color: tuple[float, float, float], opacity: float
) -> None:
    """Binds a shared UsdPreviewSurface material for (color, opacity), deduplicated by
    `material_path` (one for Floor, one for Structure, one per distinct object class -
    not one per prim, which would multiply hundreds of Structure boxes into hundreds
    of Material+Shader pairs for no visual difference).

    Kept ALONGSIDE the plain displayColor/displayOpacity primvars `_add_cube`/
    `_add_hull_mesh` already set (harmless, and correct for consumers like Storm that
    honor them directly) because Isaac Sim's RTX renderer needs an actual bound
    material for opacity to render as translucent at all - verified empirically
    (feat/isaac-demo-look, docs/DECISIONS.md): Structure boxes rendered fully opaque
    white despite displayOpacity=0.35 until a material was bound."""
    existing = stage.GetPrimAtPath(material_path)
    if existing.IsValid():
        material = UsdShade.Material(existing)
    else:
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _add_cube(
    stage: Usd.Stage,
    path: str,
    *,
    center_isaac: Gf.Vec3d,
    size_isaac: Gf.Vec3d,
    display_color: tuple[float, float, float] | None = None,
    display_opacity: float | None = None,
    material_path: str | None = None,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)  # unit cube [-0.5, 0.5]^3, so scale == full extents
    # Double precision, not the AddScaleOp()/AddTranslateOp() default of float32 - the
    # numeric-agreement test (test_object_transform_matches_bbox_exactly) requires
    # exact conversion, and float32 rounding alone was enough to break that at scale
    # values around 0.8m (~1e-7 relative error - immaterial for simulation, but not
    # "zero discrepancy" per spec).
    #
    # Translate op MUST be added before the scale op. xformOpOrder lists ops in the
    # order applied to a point (Xformable.GetLocalTransformation()'s row-vector
    # Transform() composes them that way), so [translate, scale] gives
    # world = scale * (local + translate)... no: it gives the intended
    # world = (local * scale) + center. Adding scale first ([scale, translate])
    # instead applies translate to the point BEFORE scale, so `center_isaac` itself
    # gets multiplied by `size_isaac` - every cube's world position silently scales
    # with its own size. That was the root cause of the ~9m Floor/Structure-vs-Objects
    # offset (Objects use raw-coordinate meshes with no scale op, so only cubes drifted).
    cube.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(center_isaac)
    cube.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(size_isaac)
    if display_color is not None:
        cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*display_color)]))
    if display_opacity is not None:
        cube.CreateDisplayOpacityAttr(Vt.FloatArray([display_opacity]))
    prim = cube.GetPrim()
    if material_path is not None and display_color is not None and display_opacity is not None:
        _apply_translucent_material(stage, prim, material_path, display_color, display_opacity)
    return prim


def _add_floor(stage: Usd.Stage, meta: GridMeta) -> None:
    width_m = meta.width * meta.resolution
    depth_m = meta.height * meta.resolution
    center_x = meta.origin_x + width_m / 2
    center_z = meta.origin_z + depth_m / 2

    prim = _add_cube(
        stage,
        "/World/Floor",
        center_isaac=to_isaac(center_x, -FLOOR_THICKNESS_M / 2, center_z),
        size_isaac=Gf.Vec3d(width_m, depth_m, FLOOR_THICKNESS_M),
        display_color=FLOOR_COLOR,
        display_opacity=FLOOR_OPACITY,
        material_path="/World/Materials/Floor",
    )
    UsdPhysics.CollisionAPI.Apply(prim)


def _add_floor_slab(stage: Usd.Stage, meta: GridMeta) -> None:
    """A thicker static collider immediately below `/World/Floor`, same XY footprint -
    see FLOOR_SLAB_THICKNESS_M. Not part of the visible floor surface - purpose=guide
    keeps it a real collider (purpose doesn't affect PhysX) while excluding it from
    Hydra's default render-purpose set, so it doesn't show through the now-translucent
    Floor above it as a mismatched opaque patch."""
    width_m = meta.width * meta.resolution
    depth_m = meta.height * meta.resolution
    center_x = meta.origin_x + width_m / 2
    center_z = meta.origin_z + depth_m / 2
    center_y = -(FLOOR_THICKNESS_M + FLOOR_SLAB_THICKNESS_M / 2)

    prim = _add_cube(
        stage,
        "/World/FloorSlab",
        center_isaac=to_isaac(center_x, center_y, center_z),
        size_isaac=Gf.Vec3d(width_m, depth_m, FLOOR_SLAB_THICKNESS_M),
    )
    UsdGeom.Cube(prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(prim)


def _add_dome_light(stage: Usd.Stage) -> None:
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(DOME_LIGHT_INTENSITY)


def _add_key_light(stage: Usd.Stage) -> None:
    """A single angled-overhead directional light so the low translucent
    Floor/Structure geometry reads as volume, not a flat dome-lit wash. See
    KEY_LIGHT_INTENSITY's comment for why these particular numbers."""
    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(KEY_LIGHT_INTENSITY)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3d(*KEY_LIGHT_ROTATE_XYZ_DEG))


def _add_structure(stage: Usd.Stage, cells: np.ndarray, meta: GridMeta) -> int:
    boxes = merge_obstacle_boxes(cells)
    UsdGeom.Xform.Define(stage, "/World/Structure")
    for i, box in enumerate(boxes):
        x0, x1, z0, z1 = _cell_box_world_bounds(box, meta)
        center = to_isaac((x0 + x1) / 2, STRUCTURE_HEIGHT_M / 2, (z0 + z1) / 2)
        size = Gf.Vec3d(x1 - x0, z1 - z0, STRUCTURE_HEIGHT_M)
        prim = _add_cube(
            stage,
            f"/World/Structure/box_{i}",
            center_isaac=center,
            size_isaac=size,
            display_color=STRUCTURE_COLOR,
            display_opacity=STRUCTURE_OPACITY,
            material_path="/World/Materials/Structure",
        )
        UsdPhysics.CollisionAPI.Apply(prim)
    return len(boxes)


def _add_hull_mesh(
    stage: Usd.Stage,
    path: str,
    hull: HullBuildResult,
    *,
    display_color: tuple[float, float, float],
    display_opacity: float,
    material_path: str,
) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    # Y-up -> Z-up is `to_isaac`'s +90deg rotation about X (det=+1, see module
    # docstring), vectorized: reorder columns to (x, z, y) then negate the new Z
    # column, i.e. (x, y, z) -> (x, -z, y). A proper rotation preserves winding order,
    # so - unlike the reflection this replaced - face indices are NOT reordered here;
    # the triangles scipy already oriented outward in `_hull_from_points` stay outward.
    isaac_points = hull.vertices[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    isaac_faces = hull.faces
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(isaac_points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(isaac_faces.shape[0], 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(isaac_faces.reshape(-1).astype(np.int32))
    )
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*display_color)]))
    mesh.CreateDisplayOpacityAttr(Vt.FloatArray([display_opacity]))
    prim = mesh.GetPrim()
    _apply_translucent_material(stage, prim, material_path, display_color, display_opacity)
    # A convex-mesh collider must be cooked as convex in PhysX, not a raw triangle
    # mesh - the mesh already IS a convex hull, so this just tells PhysX not to run
    # its own (lossier) convex-decomposition on it.
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    return prim


def _object_point_cloud(obj: UsdExportObject) -> np.ndarray | None:
    if not obj.mesh_path:
        return None
    try:
        pc = read_ply(obj.mesh_path)
    except (OSError, PlyParseError) as exc:
        logger.info(
            "usd_export: %s falling back to bbox collider (could not read mesh_path %r: %s)",
            obj.name, obj.mesh_path, exc,
        )
        return None
    return pc.positions.astype(np.float64)


def _add_object_geometry(stage: Usd.Stage, path: str, obj: UsdExportObject) -> tuple[Usd.Prim, str, bool]:
    """Returns (prim, collision_source, decimated)."""
    color = _object_class_color(obj.name)
    material_path = _object_class_material_path(obj.name)
    points = _object_point_cloud(obj)
    hull = build_convex_hull(points) if points is not None else None
    if hull is not None:
        prim = _add_hull_mesh(
            stage, path, hull,
            display_color=color, display_opacity=OBJECT_OPACITY, material_path=material_path,
        )
        return prim, "convex_hull", hull.decimated

    if points is not None:
        logger.info(
            "usd_export: %s falling back to bbox collider (convex hull could not be "
            "built from %d points - too few, or degenerate/coplanar)",
            obj.name, points.shape[0],
        )

    bmin, bmax = obj.bbox_min, obj.bbox_max
    center = to_isaac(
        (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2
    )
    size = Gf.Vec3d(bmax[0] - bmin[0], bmax[2] - bmin[2], bmax[1] - bmin[1])
    prim = _add_cube(
        stage, path, center_isaac=center, size_isaac=size,
        display_color=color, display_opacity=OBJECT_OPACITY, material_path=material_path,
    )
    return prim, "bbox", False


def _add_object(stage: Usd.Stage, path: str, obj: UsdExportObject) -> tuple[str, bool]:
    """Returns (collision_source, decimated) for the caller's stats tally.

    Static collision only (CollisionAPI, no RigidBodyAPI/MassAPI) - see the module
    docstring for why: matches the frontend export panel's own "static collision
    only" promise, and sidesteps a real dynamics-instability finding in this scene's
    convex-hull colliders (docs/DECISIONS.md) rather than working around it downstream."""
    prim, source, decimated = _add_object_geometry(stage, path, obj)

    UsdPhysics.CollisionAPI.Apply(prim)

    prim.CreateAttribute("cloudeye:name", Sdf.ValueTypeNames.String, custom=True).Set(obj.name)
    prim.CreateAttribute("cloudeye:numViews", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(obj.num_views)
    )
    prim.CreateAttribute("cloudeye:numPoints", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(obj.num_points)
    )
    prim.CreateAttribute("cloudeye:isFragment", Sdf.ValueTypeNames.Bool, custom=True).Set(
        bool(obj.is_fragment)
    )
    # "convex_hull" or "bbox" - which one produced this object's collider, since a
    # convex hull can silently fall back per-object (see _add_object_geometry).
    prim.CreateAttribute(
        "cloudeye:collisionSource", Sdf.ValueTypeNames.String, custom=True
    ).Set(source)
    return source, decimated


def _add_objects(
    stage: Usd.Stage,
    objects: list[UsdExportObject],
    include_fragments: bool,
    target_object_index: int | None = None,
) -> tuple[int, int, int, int, int, int, str | None]:
    """Returns (object_count, fragment_count, hull_count, hull_decimated_count,
    bbox_fallback_count, elevated_excluded_count, target_prim_path).
    `target_prim_path` is the resolved prim path for `objects[target_object_index]`
    (for /World/Target's relationship), or None if no index was given or that object
    ended up excluded (a fragment with include_fragments=False, or entirely above
    OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M)."""
    UsdGeom.Xform.Define(stage, "/World/Objects")
    if include_fragments and any(o.is_fragment for o in objects):
        UsdGeom.Xform.Define(stage, "/World/Objects/_fragments")

    counters: dict[str, int] = defaultdict(int)
    object_count = 0
    fragment_count = 0
    hull_count = 0
    hull_decimated_count = 0
    bbox_fallback_count = 0
    elevated_excluded_count = 0
    target_prim_path = None
    for i, obj in enumerate(objects):
        if obj.is_fragment:
            fragment_count += 1
            if not include_fragments:
                continue
        # See OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M's comment - keyed on bbox_min (the
        # object's lowest point), not centroid/max.
        if obj.bbox_min[1] >= OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M:
            elevated_excluded_count += 1
            logger.info(
                "usd_export: excluding %r - entirely above %.1fm (bbox_min_y=%.2f)",
                obj.name, OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M, obj.bbox_min[1],
            )
            continue
        base = _sanitize_prim_name(obj.name)
        idx = counters[base]
        counters[base] += 1
        safe_name = f"{base}_{idx}"

        parent = "/World/Objects/_fragments" if obj.is_fragment else "/World/Objects"
        prim_path = f"{parent}/{safe_name}"
        source, decimated = _add_object(stage, prim_path, obj)
        if i == target_object_index:
            target_prim_path = prim_path
        object_count += 1
        if source == "convex_hull":
            hull_count += 1
            if decimated:
                hull_decimated_count += 1
        else:
            bbox_fallback_count += 1

    logger.info(
        "usd_export objects: %d convex hull (%d decimated), %d bbox fallback, "
        "%d excluded (entirely above %.1fm)",
        hull_count, hull_decimated_count, bbox_fallback_count,
        elevated_excluded_count, OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M,
    )
    return (
        object_count, fragment_count, hull_count, hull_decimated_count,
        bbox_fallback_count, elevated_excluded_count, target_prim_path,
    )


def _load_pointcloud(ply_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    pc = read_ply(ply_path)
    return pc.positions, pc.colors


def _to_isaac_points(positions: np.ndarray) -> np.ndarray:
    """Vectorized `to_isaac` for a whole (N, 3) array - a per-point Python loop over
    hundreds of thousands of Gf.Vec3f objects is the dominant cost otherwise (measured:
    ~500x slower than this). Reorder to (x, z, y) then negate the new Z column:
    (x, y, z) -> (x, -z, y)."""
    return (positions[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])).astype(np.float32)


def _add_pointcloud(stage: Usd.Stage, prim_path: str, positions: np.ndarray, colors: np.ndarray | None) -> int:
    """The full-resolution (randomly subsampled to POINTCLOUD_MAX_POINTS) point cloud
    as UsdGeomPoints, purpose=guide. NOT the primary visible-in-Isaac layer any more -
    see POINTCLOUD_MAX_POINTS's comment and `_add_pointcloud_instancer` below. Kept for
    other consumers that handle Points correctly. No CollisionAPI - visual only."""
    n = positions.shape[0]
    if n > POINTCLOUD_MAX_POINTS:
        rng = np.random.default_rng(_POINTCLOUD_SUBSAMPLE_SEED)
        keep = rng.choice(n, size=POINTCLOUD_MAX_POINTS, replace=False)
        keep.sort()
        positions = positions[keep]
        colors = colors[keep] if colors is not None else None
        logger.info(
            "Subsampled point cloud for USD export: %d -> %d points", n, POINTCLOUD_MAX_POINTS
        )
        n = POINTCLOUD_MAX_POINTS

    isaac_positions = _to_isaac_points(positions)

    points = UsdGeom.Points.Define(stage, prim_path)
    points.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(isaac_positions))
    points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(n, 0.01, dtype=np.float32)))
    # Authored explicitly - UsdGeom.Points doesn't get a correct bound from generic
    # bbox computation without it (confirmed: BBoxCache returned an empty/sentinel
    # range for this prim before this was added, even with real points present).
    extent_min = isaac_positions.min(axis=0).tolist()
    extent_max = isaac_positions.max(axis=0).tolist()
    points.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*extent_min), Gf.Vec3f(*extent_max)]))
    if colors is not None:
        rgb = (colors.astype(np.float32) / 255.0)
        points.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(rgb))
    return n


def _voxel_downsample(
    positions: np.ndarray, colors: np.ndarray | None, *, max_points: int, initial_voxel_m: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Bin points into voxel_m cubes and replace each occupied voxel with the centroid
    of the points inside it - spatially uniform (unlike POINTCLOUD_MAX_POINTS's random
    subsample above), which matters more here since each surviving point becomes a
    real geometry instance a viewer's eye can individually resolve, not a sub-pixel
    sprite. Voxel size grows (not shrinks - starting small and growing converges in
    fewer steps for a cloud already denser than `initial_voxel_m`, the common case)
    until the result fits `max_points`. Deterministic - no RNG, unlike the subsample
    above."""
    voxel_m = initial_voxel_m
    for _ in range(30):
        voxel_idx = np.floor(positions / voxel_m).astype(np.int64)
        uniq, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
        if uniq.shape[0] <= max_points:
            sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
            np.add.at(sums, inverse, positions)
            centroids = (sums / counts[:, None]).astype(np.float32)
            out_colors = None
            if colors is not None:
                csum = np.zeros((uniq.shape[0], colors.shape[1]), dtype=np.float64)
                np.add.at(csum, inverse, colors.astype(np.float64))
                out_colors = csum / counts[:, None]
            return centroids, out_colors
        voxel_m *= 1.4
    raise RuntimeError(
        f"_voxel_downsample: could not reach max_points={max_points} for "
        f"{positions.shape[0]} points within 30 voxel-size growth steps"
    )


def _add_pointcloud_instancer(
    stage: Usd.Stage, prim_path: str, positions: np.ndarray, colors: np.ndarray | None
) -> int:
    """The visible-in-Isaac point cloud layer: a PointInstancer of small cubes,
    voxel-downsampled to POINTCLOUD_INSTANCER_MAX_POINTS. Isaac Sim 6.0.1's RTX
    (RayTracedLighting) Replicator capture pipeline was confirmed (feat/isaac-demo-look,
    see docs/DECISIONS.md) to not render native UsdGeomPoints at all regardless of
    purpose/extent/width - this replaces it as the primary visible representation.
    Per-instance color via `primvars:displayColor` (UsdGeom.PointInstancer isn't a
    Gprim, so it has no CreateDisplayColorAttr convenience method - PrimvarsAPI
    authors the same underlying primvar directly). No CollisionAPI - visual only,
    same as `_add_pointcloud`."""
    downsampled_positions, downsampled_colors = _voxel_downsample(
        positions.astype(np.float64), colors,
        max_points=POINTCLOUD_INSTANCER_MAX_POINTS,
        initial_voxel_m=POINTCLOUD_INSTANCER_INITIAL_VOXEL_M,
    )
    n = downsampled_positions.shape[0]
    isaac_positions = _to_isaac_points(downsampled_positions)

    instancer = UsdGeom.PointInstancer.Define(stage, prim_path)
    proto_path = f"{prim_path}/Prototypes/Cube"
    proto_cube = UsdGeom.Cube.Define(stage, proto_path)
    proto_cube.CreateSizeAttr(POINTCLOUD_INSTANCER_CUBE_SIZE_M)
    # The prototype prim would otherwise also render once on its own, in addition to
    # every instance the PointInstancer places - it's a template, not scene content.
    UsdGeom.Imageable(proto_cube.GetPrim()).MakeInvisible()

    instancer.CreatePrototypesRel().SetTargets([Sdf.Path(proto_path)])
    instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(isaac_positions))
    instancer.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.zeros(n, dtype=np.int32)))

    if downsampled_colors is not None:
        rgb = (downsampled_colors.astype(np.float32) / 255.0)
        color_primvar = UsdGeom.PrimvarsAPI(instancer.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        color_primvar.Set(Vt.Vec3fArray.FromNumpy(rgb))
    return n


def _add_robot_start(stage: Usd.Stage, robot_start: tuple[float, float] | None) -> None:
    xform = UsdGeom.Xform.Define(stage, "/World/RobotStart")
    if robot_start is not None:
        x, z = robot_start
        xform.AddTranslateOp().Set(to_isaac(x, 0.0, z))


# Height above the floor the /World/Path polyline and its waypoint markers sit at -
# just enough to avoid z-fighting with the (already slightly-above-zero) Floor top
# surface, not a real navigation height.
PATH_HEIGHT_ABOVE_FLOOR_M = 0.05
PATH_WIDTH_M = 0.04


def _add_path(stage: Usd.Stage, path_points: list[tuple[float, float]]) -> None:
    """A visible polyline through `path_points` (Y-up world (x, z) waypoints, e.g. from
    app.services.pathfinding.plan_to_object/plan_path's PathResult.points) - purely a
    visual aid for a demo recording to follow, no collision."""
    if len(path_points) < 2:
        return
    isaac_points = [
        to_isaac(x, PATH_HEIGHT_ABOVE_FLOOR_M, z) for x, z in path_points
    ]
    curves = UsdGeom.BasisCurves.Define(stage, "/World/Path")
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(p) for p in isaac_points]))
    curves.CreateCurveVertexCountsAttr(Vt.IntArray([len(isaac_points)]))
    curves.CreateWidthsAttr(Vt.FloatArray([PATH_WIDTH_M] * len(isaac_points)))
    curves.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*PATH_COLOR)]))


def _add_target(stage: Usd.Stage, target_point: tuple[float, float], target_prim_path: str | None) -> None:
    """`/World/Target`: an Xform at the goal point (Y-up world (x, z)), with a
    `cloudeye:targetObject` relationship to the actual target's prim under
    /World/Objects when one is known (e.g. the object a "goto <object>" command
    resolved) - visual/reference only, no collision of its own (the target IS the
    referenced object's own collider, when there is one)."""
    xform = UsdGeom.Xform.Define(stage, "/World/Target")
    x, z = target_point
    xform.AddTranslateOp().Set(to_isaac(x, 0.0, z))
    if target_prim_path is not None:
        xform.GetPrim().CreateRelationship("cloudeye:targetObject", custom=True).SetTargets(
            [Sdf.Path(target_prim_path)]
        )


# --- Top-level entry points --------------------------------------------------------


def build_stage(export_input: UsdExportInput) -> tuple[Usd.Stage, ExportStats]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    stage.GetRootLayer().customLayerData = {
        "cloudeye:robotRadiusM": float(export_input.robot_radius_m),
        "cloudeye:isReflected": bool(export_input.is_reflected),
        "cloudeye:axisConvention": (
            "source data is Y-up; exported (x, y, z) -> (x, -z, y), a +90deg rotation "
            "about X (det=+1)"
        ),
    }

    _add_dome_light(stage)
    _add_key_light(stage)
    _add_floor(stage, export_input.grid_meta)
    _add_floor_slab(stage, export_input.grid_meta)
    box_count = _add_structure(stage, export_input.grid_cells, export_input.grid_meta)
    (
        object_count, fragment_count, hull_count, hull_decimated_count,
        bbox_fallback_count, elevated_excluded_count, target_prim_path,
    ) = _add_objects(
        stage, export_input.objects, export_input.include_fragments, export_input.target_object_index
    )
    _add_robot_start(stage, export_input.robot_start)

    if export_input.path_points:
        _add_path(stage, export_input.path_points)
    if export_input.target_point is not None:
        _add_target(stage, export_input.target_point, target_prim_path)

    point_count = 0
    instancer_point_count = 0
    if export_input.pointcloud_path is not None:
        positions, colors = _load_pointcloud(export_input.pointcloud_path)
        point_count = _add_pointcloud(stage, "/World/PointCloud", positions, colors)
        instancer_point_count = _add_pointcloud_instancer(
            stage, "/World/PointCloudInstancer", positions, colors
        )

    obstacle_cell_count = int((export_input.grid_cells == OBSTACLE).sum())
    stats = ExportStats(
        obstacle_cell_count=obstacle_cell_count,
        structure_box_count=box_count,
        object_count=object_count,
        fragment_count=fragment_count,
        pointcloud_point_count=point_count,
        pointcloud_instancer_count=instancer_point_count,
        hull_object_count=hull_count,
        hull_decimated_object_count=hull_decimated_count,
        bbox_fallback_object_count=bbox_fallback_count,
        elevated_excluded_object_count=elevated_excluded_count,
    )
    return stage, stats


def export_scene_usd(export_input: UsdExportInput, out_path: str | Path) -> ExportStats:
    """Build the stage and write it to `out_path` (creating parent dirs as needed).
    Caller decides whether to call this at all (i.e. does its own cache-hit check
    against `out_path` first, via `cache_filename`)."""
    out_path = Path(out_path)
    stage, stats = build_stage(export_input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_path))
    return stats
