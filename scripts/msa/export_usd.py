"""Stage E0 (USD, SPEC §8): schema-v5 USD export - `/World/Structure` (walls),
`/World/Floor`, `/World/Objects/<class>_<n>` with separate `visual` (placeholder
mesh, or a generated mesh once Stage B lands) and `collision` (the *measured*
convex hull - SPEC §5: this must never be the generated mesh) child prims,
`/World/Plan` (2D wall/floor outlines as curves), and optional `/World/Path` /
`/World/Target` when a planned path is available.

Supersedes Stage A0's minimal single-mesh-per-object export (no visual/collision
split, no Plan/Path/Target) - the object-count/prim-naming logic is unchanged so
existing s0 callers keep working, just with a richer prim tree per object.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from shapely.geometry import MultiPolygon, Polygon

from app.services.usd_export import to_isaac
from scripts.msa.assets import build_placeholder
from scripts.msa.geometry import WALL_THICKNESS_M, WallPolygon
from scripts.msa.usd_assets import DEFAULT_USD_CACHE_ROOT, add_asset_visual, placements_by_id
from scripts.msa.visibility import TRANSLUCENT_OPACITY, is_translucent_class, make_box, transform_box

EXPORT_SCHEMA_VERSION = 5
# T15i: `curtain`/`drape`/`blind`/`mirror` placeholder parts get a UsdPreviewSurface
# with this opacity (visibility.TRANSLUCENT_OPACITY = the render_perspective / web
# viewer value) - Isaac needs a BOUND material for opacity (2026-09-05 v2 entry);
# displayColor alone renders opaque, which is how the 2.7 m near-white curtain boxes
# hid the bed in the T15e run2 still.
TRANSLUCENT_MATERIAL_ROOT = "/World/Looks"
TRANSLUCENT_DEFAULT_RGB = (0.85, 0.85, 0.88)

# T15h: a collider mesh may only be exported with `physics:approximation = convexHull`
# when its convex hull footprint is within this factor of its true footprint - PhysX
# replaces the mesh by its convex hull, so a concave collider (a wall fragment, the
# ring-shaped wall band) would phantom-fill everything inside the hull. Measured on the
# T15e GPU run: hero `wall_1` 5.18 m^2 -> 12.32 m^2 hull covering the desk; the T15g
# band under convexHull would fill the whole room (docs/DECISIONS.md 2026-09-07 T15e).
CONVEX_HULL_MAX_AREA_RATIO = 1.25
# T15h: `/World/TargetObject` marker radius (visual only) - smaller than `/World/Target`'s
# 0.1 so the two read differently in Isaac/usdview.
TARGET_OBJECT_MARKER_RADIUS_M = 0.05


def _sanitize_prim_name(name: str) -> str:
    """USD prim names must be valid identifiers - object labels/ids can contain
    spaces ("coffee maker_0") which SdfPath rejects outright."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"p_{safe}"
    return safe


def _largest_polygon(poly):
    if isinstance(poly, MultiPolygon):
        parts = [p for p in poly.geoms if p.area > 1e-6]
        return max(parts, key=lambda p: p.area) if parts else None
    return poly if not poly.is_empty and poly.area > 1e-6 else None


def _add_trimesh(stage: Usd.Stage, prim_path: str, mesh, world_transform=None) -> Usd.Prim:
    """`mesh.vertices` are in this module's internal Y-up world frame; remapped
    to Isaac's Z-up convention (`to_isaac`) right here at the USD-writing
    boundary, same as `app/services/usd_export.py`'s main-pipeline exporter -
    everything upstream (bootstrap.py, geometry.py, gaps.py) stays Y-up.

    Returns the created prim (visual-only meshes just discard it; collision
    meshes need it to apply UsdPhysics.CollisionAPI - see _add_collision_mesh)."""
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = [to_isaac(*p) for p in mesh.vertices]
    usd_mesh.CreatePointsAttr(points)
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    if world_transform is not None:
        xform = UsdGeom.Xformable(usd_mesh)
        xform.AddTransformOp().Set(Gf.Matrix4d(*world_transform.T.flatten().tolist()))
    return usd_mesh.GetPrim()


def collider_footprint_stats(mesh) -> dict:
    """T15h: how much a `convexHull` approximation would over-fill this (Y-up) mesh.
    `footprint_area_m2` = area of the bottom cap (every face whose vertices all sit
    on the mesh's lowest Y - exact for the extrusions this exporter emits; falls back
    to the XZ projection outline for anything else), `hull_area_m2` = area of the 2D
    convex hull of every vertex's XZ, `ratio` = hull / footprint (1.0 for a convex
    prism)."""
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if len(verts) < 3 or len(faces) == 0:
        return {"footprint_area_m2": 0.0, "hull_area_m2": 0.0, "ratio": 1.0}
    y_min = verts[:, 1].min()
    bottom = np.all(np.isclose(verts[faces][:, :, 1], y_min, atol=1e-6), axis=1)
    footprint = 0.0
    if bottom.any():
        tri = verts[faces[bottom]][:, :, [0, 2]]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        footprint = float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() / 2.0)
    if footprint <= 1e-9:
        from trimesh.path.polygons import projected

        outline = projected(mesh, normal=[0.0, 1.0, 0.0])
        footprint = float(outline.area) if outline is not None else 0.0
    from shapely.geometry import MultiPoint

    hull_area = float(MultiPoint(verts[:, [0, 2]]).convex_hull.area)
    ratio = hull_area / footprint if footprint > 1e-9 else float("inf")
    return {"footprint_area_m2": footprint, "hull_area_m2": hull_area, "ratio": ratio}


def _add_collision_mesh(stage: Usd.Stage, prim_path: str, mesh, world_transform=None, *,
                        approximation: str = "convexHull", strict: bool = False) -> Usd.Prim:
    """Same as `_add_trimesh`, but also marks the prim as a static PhysX
    collider (SPEC §7's collider-coverage check requires `UsdPhysics.CollisionAPI`
    on every structural/object collider - this exporter never applied it before
    Stage E2's GPU validation caught `collider_coverage`/`sphere_drop` failing on
    every MSA scene, see docs/DECISIONS.md). `MeshCollisionAPI` + `convexHull`
    matches `app/services/usd_export.py::_add_hull_mesh` for the object hulls,
    which are convex by construction (extruded `convex_hull_2d` rings).

    T15h guard: `convexHull` is refused for any mesh whose convex-hull footprint
    exceeds its true footprint by more than `CONVEX_HULL_MAX_AREA_RATIO`
    (`collider_footprint_stats`) - PhysX would phantom-fill the concavity (the T15e
    finding: a concave wall fragment became a solid block over the desk; a
    ring-shaped wall band would fill the whole room). Such a mesh is written with
    `approximation = none` (an exact static triangle mesh, valid for static
    colliders) and a warning, or raises when `strict=True`. `approximation` may be
    `"convexHull"` or `"none"`."""
    if approximation not in ("convexHull", "none"):
        raise ValueError(f"{prim_path}: approximation must be 'convexHull' or 'none', got {approximation!r}")
    if approximation == "convexHull":
        stats = collider_footprint_stats(mesh)
        if stats["ratio"] > CONVEX_HULL_MAX_AREA_RATIO:
            msg = (
                f"{prim_path}: convex hull footprint {stats['hull_area_m2']:.3f} m^2 is {stats['ratio']:.2f}x the mesh's "
                f"true footprint {stats['footprint_area_m2']:.3f} m^2 (> {CONVEX_HULL_MAX_AREA_RATIO}x) - a convexHull "
                f"collider would phantom-fill the concavity; use per-edge boxes or approximation='none'"
            )
            if strict:
                raise ValueError(msg)
            logging.getLogger(__name__).warning("%s - exporting as approximation='none' (static triangle mesh)", msg)
            approximation = "none"
    prim = _add_trimesh(stage, prim_path, mesh, world_transform)
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull if approximation == "convexHull" else UsdPhysics.Tokens.none
    )
    return prim


def polygon_edges_ccw(points_xz) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The edges of a closed ring as `[(p0, p1), ...]` with the ring oriented CCW in
    the (x, z) plane (shapely's `orient`), so the outward side of every edge
    `(dx, dz)` is `(dz, -dx)`. A ring with a duplicated closing vertex is fine."""
    from shapely.geometry.polygon import orient

    poly = Polygon([(float(x), float(z)) for x, z in points_xz])
    if not poly.is_valid:
        poly = _largest_polygon(poly.buffer(0))
        if poly is None:
            return []
    ring = list(orient(poly, sign=1.0).exterior.coords)[:-1]
    edges = []
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        if np.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6:
            edges.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return edges


def add_wall_edge_box_colliders(
    stage: Usd.Stage, root_path: str, room_polygon_xz, floor_y: float, ceiling_y: float, *,
    thickness_m: float = WALL_THICKNESS_M, visible: bool = False,
) -> list[dict]:
    """T15h: the wall band's collider as one convex box per edge of the room
    outline - `<root_path>/edge_i` `UsdGeom.Cube` prims with `CollisionAPI` only (a
    Cube is a native PhysX box shape; no mesh approximation involved). Each box
    hugs its edge from the OUTSIDE (the band is `room.buffer(thickness) - room`,
    `geometry.room_wall_band`): centre = edge midpoint + outward normal *
    thickness/2, extents = (edge length + one thickness past every CONVEX end,
    thickness, ceiling - floor). The extension covers the band's mitred corner
    square at a convex corner (adjacent boxes overlap there); at a reflex corner the
    two strips already meet and an extension would poke into the room, so there is
    none. Boxes are invisible by default (the visible stub is a separate prim),
    never deactivated (v2 entry). Returns one record per box (Y-up numbers) for the
    summary."""
    UsdGeom.Xform.Define(stage, root_path)
    height = max(float(ceiling_y - floor_y), 0.01)
    z_mid = float(floor_y) + height / 2.0
    edges = polygon_edges_ccw(room_polygon_xz)
    n = len(edges)
    dirs = []
    for (x0, z0), (x1, z1) in edges:
        length = float(np.hypot(x1 - x0, z1 - z0))
        dirs.append(((x1 - x0) / length, (z1 - z0) / length))
    # A vertex is convex (interior angle < 180 deg) when the CCW turn from the incoming
    # to the outgoing edge is a left turn: cross(d_in, d_out) > 0. Only there does the
    # band's mitred corner square lie beyond the edge ends; at a reflex corner the two
    # strips already overlap, and an extension would poke INTO the room.
    convex_vertex = [
        (dirs[i - 1][0] * dirs[i][1] - dirs[i - 1][1] * dirs[i][0]) > 1e-9 for i in range(n)
    ]  # convex_vertex[i] is about the START vertex of edge i
    records = []
    for i, ((x0, z0), (x1, z1)) in enumerate(edges):
        ux, uz = dirs[i]
        length = float(np.hypot(x1 - x0, z1 - z0))
        # A full thickness past each convex vertex: the mitred corner square of the band
        # (thickness x thickness beyond both edge ends) is then covered by either box.
        ext_start = thickness_m if convex_vertex[i] else 0.0
        ext_end = thickness_m if convex_vertex[(i + 1) % n] else 0.0
        box_len = length + ext_start + ext_end
        nx, nz = uz, -ux  # outward for a CCW ring in the (x, z) plane
        along = (ext_end - ext_start) / 2.0  # shift of the box centre along the edge
        cx = (x0 + x1) / 2.0 + ux * along + nx * thickness_m / 2.0
        cz = (z0 + z1) / 2.0 + uz * along + nz * thickness_m / 2.0
        cube = UsdGeom.Cube.Define(stage, f"{root_path}/edge_{i}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        # Isaac Z-up: (x, y_up, z) -> (x, -z, y_up); the edge direction (ux, uz) -> (ux, -uz).
        xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(-cz), z_mid))
        xf.AddRotateZOp().Set(float(np.degrees(np.arctan2(-uz, ux))))
        xf.AddScaleOp().Set(Gf.Vec3f(float(box_len), float(thickness_m), float(height)))
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()
        records.append({
            "prim": f"{root_path}/edge_{i}", "edge_xz": [[x0, z0], [x1, z1]], "center_xz": [float(cx), float(cz)],
            "length_m": float(box_len), "thickness_m": float(thickness_m), "height_m": height,
        })
    return records


def _edge_convexity(edges) -> tuple[list, list]:
    """Unit directions per edge and, per edge, whether its START vertex is convex
    (CCW left turn from the incoming edge) - shared by the colliders and the stubs."""
    dirs = []
    for (x0, z0), (x1, z1) in edges:
        length = float(np.hypot(x1 - x0, z1 - z0))
        dirs.append(((x1 - x0) / length, (z1 - z0) / length))
    n = len(edges)
    convex_vertex = [(dirs[i - 1][0] * dirs[i][1] - dirs[i - 1][1] * dirs[i][0]) > 1e-9 for i in range(n)]
    return dirs, convex_vertex


def stub_edge_boxes(room_polygon_xz, floor_y: float, height_m: float, thickness_m: float = WALL_THICKNESS_M) -> list[dict]:
    """T15i: the visible wall stub as one box per edge of the regularized outline
    (Y-up records) - the same strips the T15h colliders use, `height_m` tall. Each box
    hugs its edge from the OUTSIDE (centre = midpoint + outward normal * t/2) and
    OWNS the band's corner square at its START vertex: it extends one thickness past
    a convex start vertex (the mitred square lies beyond both edge ends there) and
    retracts one thickness at a reflex start vertex (the two plain strips already
    overlap there). The boxes therefore tile the band exactly - no overlap, nothing
    missing - unlike the T15h colliders, which extend at both convex ends and overlap
    (fine for PhysX, but coplanar overlapping top faces are not what you want to look
    at). Record: `edge` (index), `edge_xz`, `center_xz`, `dir_xz`, `normal_xz`
    (outward), `length_m`, `thickness_m`, `y0`, `y1`."""
    edges = polygon_edges_ccw(room_polygon_xz)
    if not edges:
        return []
    dirs, convex_vertex = _edge_convexity(edges)
    n = len(edges)
    reflex_vertex = [(dirs[i - 1][0] * dirs[i][1] - dirs[i - 1][1] * dirs[i][0]) < -1e-9 for i in range(n)]
    records = []
    for i, ((x0, z0), (x1, z1)) in enumerate(edges):
        ux, uz = dirs[i]
        length = float(np.hypot(x1 - x0, z1 - z0))
        ext_start = thickness_m if convex_vertex[i] else (-thickness_m if reflex_vertex[i] else 0.0)
        nx, nz = uz, -ux
        along = -ext_start / 2.0
        cx = (x0 + x1) / 2.0 + ux * along + nx * thickness_m / 2.0
        cz = (z0 + z1) / 2.0 + uz * along + nz * thickness_m / 2.0
        records.append({
            "edge": i, "edge_xz": [[float(x0), float(z0)], [float(x1), float(z1)]],
            "center_xz": [float(cx), float(cz)], "dir_xz": [float(ux), float(uz)], "normal_xz": [float(nx), float(nz)],
            "length_m": float(length + ext_start), "thickness_m": float(thickness_m),
            "y0": float(floor_y), "y1": float(floor_y + height_m),
        })
    return records


def room_centroid_xz(room_polygon_xz) -> tuple[float, float]:
    poly = Polygon([(float(x), float(z)) for x, z in room_polygon_xz])
    if not poly.is_valid:
        poly = _largest_polygon(poly.buffer(0)) or poly
    c = poly.centroid
    return float(c.x), float(c.y)


def culled_stub_edge_ids(records: list[dict], camera_xz, room_centroid) -> list[int]:
    """T15i dollhouse cull: an edge is made invisible when the camera and the room
    centroid lie on OPPOSITE sides of the edge's line - the wall stands between the
    eye and the room. This is exactly the web viewer's rule (frontend/src/lib/
    msaDollhouse.ts keeps only the face whose normal points toward the room centroid
    and renders it FrontSide, so that face is visible iff the camera is on the
    centroid's side of the wall) and render_perspective._dollhouse_faces. For a
    convex room it reduces to "the camera is on the outer side of the edge" (the
    near walls); for a non-convex outline it also culls the walls of a notch/doorway
    spur that the camera looks across - on the hero the notch's west wall (edge 7,
    outward normal pointing INTO the notch, camera on its inner side) hid the route
    goal at every candidate height, while the viewer shows nothing of that wall from
    that side."""
    cx, cz = float(camera_xz[0]), float(camera_xz[1])
    rx, rz = float(room_centroid[0]), float(room_centroid[1])
    culled = []
    for rec in records:
        (x0, z0), _ = rec["edge_xz"]
        nx, nz = rec["normal_xz"]
        side_cam = (cx - x0) * nx + (cz - z0) * nz
        side_room = (rx - x0) * nx + (rz - z0) * nz
        if side_cam * side_room < -1e-12:
            culled.append(int(rec["edge"]))
    return culled


def stub_edge_trimesh(rec: dict):
    """The Y-up box mesh of one `stub_edge_boxes` record (for the USD and the still)."""
    ux, uz = rec["dir_xz"]
    height = rec["y1"] - rec["y0"]
    box = trimesh.creation.box(extents=(rec["length_m"], height, rec["thickness_m"]))
    box.apply_transform(trimesh.transformations.rotation_matrix(float(np.arctan2(-uz, ux)), [0, 1, 0]))
    box.apply_translation((rec["center_xz"][0], rec["y0"] + height / 2.0, rec["center_xz"][1]))
    return box


def stub_edge_occluder_box(rec: dict, prim: str | None = None) -> dict:
    """The record as a `scripts.msa.visibility` oriented box (Y-up)."""
    ux, uz = rec["dir_xz"]
    nx, nz = rec["normal_xz"]
    height = rec["y1"] - rec["y0"]
    return make_box(
        prim or f"stub_edge_{rec['edge']}",
        (rec["center_xz"][0], rec["y0"] + height / 2.0, rec["center_xz"][1]),
        ((ux, 0.0, uz), (0.0, 1.0, 0.0), (nx, 0.0, nz)),
        (rec["length_m"] / 2.0, height / 2.0, rec["thickness_m"] / 2.0),
    )


def add_wall_stub_edges(stage: Usd.Stage, visual_root: str, records: list[dict], culled_ids, display_color=None) -> list[dict]:
    """`<visual_root>/stub_edge_i` Mesh per record (no collision API - the colliders
    are the T15h edge boxes), `UsdGeom.Imageable.MakeInvisible()` on every culled
    edge. Custom attrs `cloudeye:stubEdgeIndex` / `cloudeye:stubEdgeCulled` ride
    along. Returns `[{prim, edge, culled}]`."""
    from pxr import Sdf

    UsdGeom.Xform.Define(stage, visual_root)
    culled = {int(i) for i in culled_ids}
    out = []
    for rec in records:
        path = f"{visual_root}/stub_edge_{rec['edge']}"
        prim = _add_trimesh(stage, path, stub_edge_trimesh(rec))
        if display_color is not None:
            _set_display_color(prim, display_color)
        is_culled = int(rec["edge"]) in culled
        if is_culled:
            UsdGeom.Imageable(prim).MakeInvisible()
        prim.CreateAttribute("cloudeye:stubEdgeIndex", Sdf.ValueTypeNames.Int, custom=True).Set(int(rec["edge"]))
        prim.CreateAttribute("cloudeye:stubEdgeCulled", Sdf.ValueTypeNames.Bool, custom=True).Set(bool(is_culled))
        out.append({"prim": path, "edge": int(rec["edge"]), "culled": is_culled})
    return out


def author_translucent_material(stage: Usd.Stage, material_path: str, rgb=TRANSLUCENT_DEFAULT_RGB, *,
                                opacity: float = TRANSLUCENT_OPACITY):
    """`UsdShade.Material` + `UsdPreviewSurface` with `opacity` (and the object's own
    colour as diffuseColor, since a bound material overrides displayColor). Same
    authoring pattern as textures.author_texture_material, minus the texture."""
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, material_path)
    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))
    pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def bind_translucent_object_material(stage: Usd.Stage, prim: Usd.Prim, obj_id: str, rgb=None, *,
                                     materials: dict | None = None) -> str | None:
    """Bind the object's translucent material to `prim` when `obj_id`'s class is in
    `visibility.TRANSLUCENT_OBJECT_CLASSES` (one material per object under
    TRANSLUCENT_MATERIAL_ROOT, cached in `materials`). Returns the material path or
    None when the class is opaque."""
    from pxr import UsdShade

    if not is_translucent_class(obj_id):
        return None
    mat_path = f"{TRANSLUCENT_MATERIAL_ROOT}/translucent_{_sanitize_prim_name(obj_id)}"
    cache = materials if materials is not None else {}
    mat = cache.get(mat_path)
    if mat is None:
        mat = author_translucent_material(stage, mat_path, rgb if rgb is not None else TRANSLUCENT_DEFAULT_RGB)
        cache[mat_path] = mat
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    return mat_path


def _extrude_world_polygon_xz(points_xz: np.ndarray, y0: float, height: float):
    """Extrude a polygon given as world-space (x, z) points into a Y-up trimesh
    spanning [y0, y0 + height]. Shared by wall/floor (Stage A0) and, here, the
    per-object collision hull."""
    poly = Polygon(points_xz)
    if not poly.is_valid or poly.area < 1e-6:
        poly = poly.buffer(0)
    poly = _largest_polygon(poly)
    if poly is None:
        return None
    mesh = trimesh.creation.extrude_polygon(poly, height=max(height, 0.01))
    remap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    mesh.apply_transform(remap)
    mesh.apply_translation((0, y0, 0))
    return mesh


def _add_basis_curve_loop(stage: Usd.Stage, prim_path: str, points_xz: np.ndarray, y: float) -> None:
    """A closed 2D outline (wall ring, floor boundary) as a linear BasisCurves at
    a fixed height - SPEC §8's `/World/Plan` ('2D plan as curves')."""
    pts = np.asarray(points_xz, dtype=float)
    if len(pts) < 2:
        return
    closed = np.vstack([pts, pts[:1]])  # close the ring
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(closed)])
    curve.CreatePointsAttr([to_isaac(float(x), y, float(z)) for x, z in closed])
    curve.CreateWidthsAttr([0.02] * len(closed))


def export_usd(
    walls: list[WallPolygon],
    floor_mesh,
    floor_y: float,
    ceiling_y: float,
    objects: list[dict],
    out_path: Path,
    *,
    floor_polygon: list[tuple[float, float]] | None = None,
    path_points_world: list[tuple[float, float, float]] | None = None,
    target_point: tuple[float, float, float] | None = None,
    wall_band_mesh=None,
    textures: dict | None = None,
    asset_placements: list[dict] | None = None,
    asset_cache_root: Path = DEFAULT_USD_CACHE_ROOT,
) -> None:
    """`floor_polygon`/`path_points_world`/`target_point` are optional - Stage
    E0's `/World/Plan`/`/World/Path`/`/World/Target` prims are only emitted when
    the caller has that data (a bootstrap run with no resolved robot start/goal
    for the scene has no path to show, and that is not an error - see
    docs/DECISIONS.md's 2026-09-06 Stage E0 entry).

    `wall_band_mesh` (T15g): the extruded single wall band (Y-up trimesh, see
    `export_glb.build_scene(wall_band=)`). When given, no per-fragment
    `/World/Structure/wall_i` meshes are emitted; `/World/Plan/wall_i` curves
    for the raw fragments are still written for the validator.

    `textures` (T15c-prep,
    opt-in): the dict `scripts.msa.textures.bake_room_textures` returns - baked
    floor/wall PNGs are written next to `out_path` and bound as
    UsdPreviewSurface materials (`textures.apply_usd_textures`).

    T15h: the band is
    ring-shaped, so it is NOT a convexHull collider any more - it is written as
    the visible `/World/Structure/wall_outline/visual/band` (no collision API)
    plus `/World/Structure/wall_outline_collider/edge_i` boxes, one per edge of
    `floor_polygon` (`add_wall_edge_box_colliders`); without a `floor_polygon`
    the band itself is the collider with `approximation = none` (exact static
    triangle mesh).

    M7 (item 4): `asset_placements` - `assets_placed.json` records
    (`asset_fallback.place_assets` / `assemble_with_generated.fit_generated`).
    An object with a `placed`/`generated` record gets its `visual` authored by
    `usd_assets.add_asset_visual`: an Xform carrying the exact GLB placement
    transform and an instanceable reference to the asset converted to USD
    (cached under `asset_cache_root`, synced into `<out dir>/usd_assets/`).
    Objects without a record (and translucent classes) keep the placeholder
    parts. Object prims are named by the sanitized object id (`television_4`
    stays `television_4`), the same key the GLB nodes use, so the two files
    can be compared object for object."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(out_path.as_posix())
    # Isaac Sim expects Z-up (tools/isaac_validate.py's stage_conventions check
    # hard-requires it) - same convention as app/services/usd_export.py, see
    # to_isaac()'s docstring. Internal MSA geometry stays Y-up; only points
    # written into the stage go through to_isaac().
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "cloudeye:msaSchemaVersion": EXPORT_SCHEMA_VERSION,
        "cloudeye:axisConvention": "source data is Y-up; exported (x, y, z) -> (x, -z, y), a +90deg rotation about X (det=+1)",
    }

    UsdGeom.Xform.Define(stage, "/World/Structure")
    for i, wall in enumerate(walls):
        if wall_band_mesh is not None:
            break  # T15g: the single band below is the only structural mesh
        mesh = _extrude_world_polygon_xz(wall.vertices, floor_y, ceiling_y - floor_y)
        if mesh is None:
            # T15b: never skip a wall silently (hero wall_2 had no mesh/collider
            # for a day - only its /World/Plan curve - and nothing said so).
            logging.getLogger(__name__).warning(
                "wall_%d: polygon (%d vertices, area_m2=%.3f) is degenerate - no USD mesh/collider emitted", i, len(wall.vertices), wall.area_m2
            )
            continue
        # T15h: per-fragment walls are concave blobs - the guard downgrades them to `none`.
        _add_collision_mesh(stage, f"/World/Structure/wall_{i}", mesh)
    if wall_band_mesh is not None:
        if floor_polygon and len(floor_polygon) >= 3:
            UsdGeom.Xform.Define(stage, "/World/Structure/wall_outline")
            UsdGeom.Xform.Define(stage, "/World/Structure/wall_outline/visual")
            _add_trimesh(stage, "/World/Structure/wall_outline/visual/band", wall_band_mesh)
            add_wall_edge_box_colliders(stage, "/World/Structure/wall_outline_collider", floor_polygon, floor_y, ceiling_y)
        else:
            _add_collision_mesh(stage, "/World/Structure/wall_outline", wall_band_mesh, approximation="none")

    if floor_mesh is not None:
        # The floor plate may be concave (room polygon + footprints) - the guard picks `none` then.
        _add_collision_mesh(stage, "/World/Floor", floor_mesh)

    UsdGeom.Xform.Define(stage, "/World/Objects")
    translucent_materials: dict = {}
    placed = placements_by_id(asset_placements)
    seen_names: dict[str, int] = {}
    for obj in objects:
        label = obj["label"]
        # M7: prim = the sanitized object id (GLB node key), de-duplicated defensively.
        prim_name = _sanitize_prim_name(str(obj.get("id") or label))
        if prim_name in seen_names:
            seen_names[prim_name] += 1
            prim_name = f"{prim_name}_dup{seen_names[prim_name]}"
        else:
            seen_names[prim_name] = 0
        obj_path = f"/World/Objects/{prim_name}"
        UsdGeom.Xform.Define(stage, obj_path)

        length_u, width_v = obj["size_uv"]
        parts = build_placeholder(label, length_u, width_v, obj["height"])
        angle = obj["angle_rad"]
        cx, cz = obj["center_xy"]
        y0 = obj.get("bbox_min_y", floor_y)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        world_transform = np.array(
            [[cos_a, 0, -sin_a, cx], [0, 1, 0, y0], [sin_a, 0, cos_a, cz], [0, 0, 0, 1]]
        )

        placement = placed.get(obj["id"]) if not is_translucent_class(label) else None
        if placement is not None:
            # M7: the same asset as the web GLB, referenced (see usd_assets).
            add_asset_visual(stage, obj_path, placement, out_dir=out_path.parent, cache_root=asset_cache_root)
            parts = []
        UsdGeom.Xform.Define(stage, f"{obj_path}/visual")
        for j, part in enumerate(parts):
            verts = np.column_stack([part.vertices, np.ones(len(part.vertices))]) @ world_transform.T
            part = part.copy()
            part.vertices = verts[:, :3]
            part_prim = _add_trimesh(stage, f"{obj_path}/visual/part_{j}", part)
            # T15i: curtains/drapes/blinds/mirrors render at 30 % opacity (bound material).
            bind_translucent_object_material(stage, part_prim, prim_name, materials=translucent_materials)

        UsdGeom.Xform.Define(stage, f"{obj_path}/collision")
        hull_xz = obj.get("hull_xz")
        if hull_xz and len(hull_xz) >= 3:
            hull_mesh = _extrude_world_polygon_xz(np.asarray(hull_xz), y0, obj["height"])
            if hull_mesh is not None:
                _add_collision_mesh(stage, f"{obj_path}/collision/hull", hull_mesh)
        # No hull (degenerate/absent) is not an error worth crashing the export
        # over - an object with no collision prim simply isn't a physics
        # obstacle in Isaac, which Stage D's validator is responsible for
        # catching (collider-coverage check, SPEC §7), not this exporter.

    plan = UsdGeom.Xform.Define(stage, "/World/Plan")
    for i, wall in enumerate(walls):
        _add_basis_curve_loop(stage, f"/World/Plan/wall_{i}", wall.vertices, floor_y)
    if floor_polygon and len(floor_polygon) >= 3:
        _add_basis_curve_loop(stage, "/World/Plan/floor", np.asarray(floor_polygon), floor_y)
    if not walls and not floor_polygon:
        stage.RemovePrim(plan.GetPath())

    if path_points_world:
        curve = UsdGeom.BasisCurves.Define(stage, "/World/Path")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(path_points_world)])
        curve.CreatePointsAttr([to_isaac(*p) for p in path_points_world])
        curve.CreateWidthsAttr([0.03] * len(path_points_world))
        add_path_ribbon(stage, path_points_world)  # morning-4: the same 3 cm ribbon as the presentation USD

    if target_point is not None:
        target = UsdGeom.Sphere.Define(stage, "/World/Target")
        target.CreateRadiusAttr(0.1)
        UsdGeom.Xformable(target).AddTranslateOp().Set(to_isaac(*target_point))

    if textures:
        from scripts.msa.textures import apply_usd_textures

        apply_usd_textures(stage, textures, out_path.parent)

    stage.GetRootLayer().Save()


# --- T15d: presentation variant (scene_presentation.usd) -----------------------------
# Same physics content as `export_usd`'s scene.usd, but authored for LOOKING at:
# walls cut to a 1.2 m "dollhouse" stub for the eye while an invisible full-height
# collider keeps the physics identical (PhysX ignores visibility - and NOT
# `SetActive(False)`, see the 2026-09-05 v2 DECISIONS entry), a DomeLight (x3 the
# 1200 usd_export.py/record_isaac.py used, recolored to the dark backdrop rather
# than backed by a backdrop sphere) plus a warm DistantLight key light, and the web
# 3/4 rule's camera baked in so any viewer - and demo/record_isaac.py's wide shot -
# opens on the same framing. Geometry comes from a `build_scene`-shaped trimesh
# Scene (a bootstrap `scene.glb` re-loaded, or a fresh `build_scene` result) so this
# works on any existing bootstrap output without re-running the bootstrap.

PRESENTATION_WALL_HEIGHT_M = 1.2
# = usd_export.DOME_LIGHT_INTENSITY / record_isaac's fallback. T15d shipped 3600 (x3);
# the T15e GPU still (2026-09-06) showed that washes the room out flat-white and turns
# the dark-recolored dome's backdrop light blue-gray (the v2 "boost the dark dome"
# finding) - 1200 is the GPU-verified value next to the 3000 key light.
PRESENTATION_DOME_INTENSITY = 1200.0
# Same magnitude as app/services/usd_export.py's KEY_LIGHT_INTENSITY (3000) - the one
# DistantLight intensity that has been confirmed on a real Isaac GPU render
# (2026-09-04/05 runs) next to a 1200 dome; with the dome now dark-recolored (its
# effective contribution ~5% of nominal) the key light is what actually models the
# stubs and placeholders, so a proven magnitude beats inventing a new one offline.
PRESENTATION_KEY_LIGHT_INTENSITY = 3000.0
PRESENTATION_KEY_LIGHT_COLOR = (1.0, 0.93, 0.82)  # warm (~4000K-ish tint)
PRESENTATION_KEY_LIGHT_ELEVATION_DEG = 45.0
# Swung this far toward the camera's right so the stubs' room-facing faces AND their
# end faces get different shading (a light exactly on the camera axis flattens both).
PRESENTATION_KEY_LIGHT_AZIMUTH_OFFSET_DEG = 30.0
PRESENTATION_BACKDROP_COLOR = (0x0B / 255, 0x0D / 255, 0x12 / 255)  # demo/record_isaac.py BACKDROP_COLOR
PRESENTATION_CAMERA_PATH = "/World/PresentationCamera"
PRESENTATION_CAMERA_HUSKY_PATH = "/World/PresentationCameraHusky"  # two-pose mode: the steeper pose for Husky's pass
PRESENTATION_CAMERA_FOCAL_MM = 24.0
PRESENTATION_WALL_DISPLAY_COLOR = (0.78, 0.80, 0.84)
PRESENTATION_FLOOR_DISPLAY_COLOR = (0.27, 0.28, 0.31)
PRESENTATION_SCHEMA_VERSION = 1


def wall_stub_mesh(mesh, floor_y: float, height_m: float = PRESENTATION_WALL_HEIGHT_M):
    """The visible 1.2 m stub of a full-height wall mesh. Bootstrap walls are straight
    extrusions (every vertex at exactly floor_y or ceiling_y - `_extrude_world_polygon_xz`
    / `export_glb._extrude_wall`), so clamping the top ring down to `floor_y + height_m`
    yields the identical closed topology at the new height - no re-triangulation, no
    polygon-order recovery from the mesh, no extra dependency. A non-extruded mesh is
    clamped the same way (any vertex above the cut lands on the cut plane)."""
    stub = mesh.copy()
    verts = np.array(stub.vertices, dtype=float)
    cut = floor_y + height_m
    verts[:, 1] = np.minimum(verts[:, 1], cut)
    stub.vertices = verts
    return stub


def _set_display_color(prim: Usd.Prim, rgb) -> None:
    from pxr import Vt

    UsdGeom.Gprim(prim).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))]))


def iter_scene_world_meshes(scene):
    """Yields `(node_name, Trimesh)` with the node's world transform already applied,
    for every Trimesh node of a `export_glb.build_scene`-shaped scene (a bootstrap
    scene.glb round-trips its node names: `wall_i`, `floor`, `<id>/visual/part_j`,
    `<id>/collision/hull`; `Plan/*`, `Path`, `Target` are Path3D/not meshes)."""
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        mesh = geom.copy()
        if transform is not None and not np.allclose(transform, np.eye(4)):
            mesh.apply_transform(transform)
        yield node_name, mesh


def add_presentation_lights(stage: Usd.Stage, camera_dir_xz, *, dome_intensity: float = PRESENTATION_DOME_INTENSITY,
                            key_intensity: float = PRESENTATION_KEY_LIGHT_INTENSITY) -> dict:
    """`/World/DomeLight` (dark backdrop color, x3 intensity) + `/World/KeyLight`
    (warm DistantLight at PRESENTATION_KEY_LIGHT_ELEVATION_DEG elevation, from the
    camera's side of the room: `camera_dir_xz` = unit Y-up XZ direction from the room
    centroid toward the camera, swung PRESENTATION_KEY_LIGHT_AZIMUTH_OFFSET_DEG toward
    the camera's right). A DistantLight emits along its local -Z, so the transform's
    local +Z axis is pointed AT the light source."""
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(float(dome_intensity))
    dome.CreateColorAttr(Gf.Vec3f(*PRESENTATION_BACKDROP_COLOR))

    dx, dz = float(camera_dir_xz[0]), float(camera_dir_xz[1])
    n = float(np.hypot(dx, dz))
    dx, dz = (dx / n, dz / n) if n > 1e-9 else (0.0, -1.0)
    # Y-up XZ -> Isaac XY is (x, -z); camera-right in Y-up is cross(forward, +Y) which,
    # for a horizontal forward (-dx, -dz), is (dz, -dx) in XZ - rotate the source
    # direction that way by the azimuth offset.
    az = np.radians(PRESENTATION_KEY_LIGHT_AZIMUTH_OFFSET_DEG)
    sx = dx * np.cos(az) + dz * np.sin(az)
    sz = dz * np.cos(az) - dx * np.sin(az)
    el = np.radians(PRESENTATION_KEY_LIGHT_ELEVATION_DEG)
    source = np.array([sx * np.cos(el), -sz * np.cos(el), np.sin(el)])  # Isaac Z-up, toward the light
    source /= np.linalg.norm(source)
    x_axis = np.cross([0.0, 0.0, 1.0], source)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(source, x_axis)
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(float(key_intensity))
    key.CreateColorAttr(Gf.Vec3f(*PRESENTATION_KEY_LIGHT_COLOR))
    key.CreateAngleAttr(1.0)
    m = Gf.Matrix4d(
        float(x_axis[0]), float(x_axis[1]), float(x_axis[2]), 0.0,
        float(y_axis[0]), float(y_axis[1]), float(y_axis[2]), 0.0,
        float(source[0]), float(source[1]), float(source[2]), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    UsdGeom.Xformable(key.GetPrim()).AddTransformOp().Set(m)
    return {
        "dome": {"path": "/World/DomeLight", "intensity": float(dome_intensity), "color": list(PRESENTATION_BACKDROP_COLOR)},
        "key": {
            "path": "/World/KeyLight", "intensity": float(key_intensity), "color": list(PRESENTATION_KEY_LIGHT_COLOR),
            "elevation_deg": PRESENTATION_KEY_LIGHT_ELEVATION_DEG, "azimuth_offset_deg": PRESENTATION_KEY_LIGHT_AZIMUTH_OFFSET_DEG,
            "emit_direction_isaac": [float(-v) for v in source],
        },
    }


def add_presentation_camera(stage: Usd.Stage, camera: dict, *, aspect: float = 16.0 / 9.0,
                            focal_mm: float = PRESENTATION_CAMERA_FOCAL_MM, path: str = PRESENTATION_CAMERA_PATH) -> Usd.Prim:
    """Bake a Y-up camera (`pos`, `forward`, `right`, `up` unit vectors, `fov_v_deg` -
    scripts.msa.render_perspective.compute_camera's output, optionally already backed
    off per scripts.msa.record_isaac.camera_backoff_for_framing) as a UsdGeom.Camera in the
    stage's Isaac Z-up frame. Apertures are authored so that Isaac's forced
    `verticalAperture = horizontalAperture * h / w` (2026-09-05 v3 entry) reproduces
    exactly `fov_v_deg` at `aspect`. The rule's own numbers ride along as custom
    `cloudeye:camera*` attributes so record_isaac's Phase 1 can report them."""
    from pxr import Sdf

    pos = to_isaac(*[float(v) for v in camera["pos"]])
    fwd = to_isaac(*[float(v) for v in camera["forward"]])
    right = to_isaac(*[float(v) for v in camera["right"]])
    up = to_isaac(*[float(v) for v in camera["up"]])
    cam = UsdGeom.Camera.Define(stage, path)
    prim = cam.GetPrim()
    fov_v_deg = float(camera.get("fov_v_deg", 55.0))
    v_ap = 2.0 * focal_mm * np.tan(np.radians(fov_v_deg) / 2.0)
    cam.CreateFocalLengthAttr(float(focal_mm))
    cam.CreateHorizontalApertureAttr(float(v_ap * aspect))
    cam.CreateVerticalApertureAttr(float(v_ap))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    m = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        -fwd[0], -fwd[1], -fwd[2], 0.0,
        pos[0], pos[1], pos[2], 1.0,
    )
    UsdGeom.Xformable(prim).AddTransformOp().Set(m)
    prim.CreateAttribute("cloudeye:cameraRule", Sdf.ValueTypeNames.String, custom=True).Set(
        str(camera.get("rule", "scripts.msa.render_perspective.compute_camera"))
    )
    prim.CreateAttribute("cloudeye:cameraFovVDeg", Sdf.ValueTypeNames.Double, custom=True).Set(fov_v_deg)
    for key, attr in (("pitch_deg", "cloudeye:cameraPitchDeg"), ("distance_m", "cloudeye:cameraDistanceM"),
                      ("backoff_m", "cloudeye:cameraBackoffM"), ("height_above_floor_m", "cloudeye:cameraHeightAboveFloorM"),
                      ("height_raise_m", "cloudeye:cameraHeightRaiseM"), ("room_frame_coverage", "cloudeye:roomFrameCoverage")):
        if camera.get(key) is not None:
            prim.CreateAttribute(attr, Sdf.ValueTypeNames.Double, custom=True).Set(float(camera[key]))
    # T15i: what record_isaac's worker needs to re-run the SAME visibility rule with the
    # larger platform's ring - the floor aim point (frame centre), the opaque occluder
    # boxes (Isaac frame, 15 doubles each: centre, 3 axes, half extents) and the culled
    # stub edges. See scripts/msa/visibility.py.
    if camera.get("aim") is not None:
        prim.CreateAttribute("cloudeye:cameraAimPoint", Sdf.ValueTypeNames.Double3, custom=True).Set(
            Gf.Vec3d(*to_isaac(*[float(v) for v in camera["aim"]]))
        )
    if camera.get("occluders"):
        from pxr import Vt

        from scripts.msa.visibility import box_to_list

        boxes_isaac = [transform_box(b, lambda p: to_isaac(*p)) for b in camera["occluders"]]
        flat = [v for b in boxes_isaac for v in box_to_list(b)]
        prim.CreateAttribute("cloudeye:occluderBoxes", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(Vt.DoubleArray(flat))
        prim.CreateAttribute("cloudeye:occluderIds", Sdf.ValueTypeNames.StringArray, custom=True).Set(
            Vt.StringArray([str(b["id"]) for b in boxes_isaac])
        )
    if camera.get("culled_stub_edges") is not None:
        from pxr import Vt

        prim.CreateAttribute("cloudeye:culledStubEdges", Sdf.ValueTypeNames.IntArray, custom=True).Set(
            Vt.IntArray([int(i) for i in camera["culled_stub_edges"]])
        )
    if camera.get("visibility_ok") is not None:
        prim.CreateAttribute("cloudeye:cameraVisibilityOk", Sdf.ValueTypeNames.Bool, custom=True).Set(bool(camera["visibility_ok"]))
    # Morning-4 / addendum: the path-ribbon worst-case visibility and the eye clearance
    # the camera was solved with - the worker re-checks both with the same numbers.
    pv = camera.get("path_visibility_worst_case")
    if isinstance(pv, dict) and pv.get("fraction") is not None:
        prim.CreateAttribute("cloudeye:pathVisibilityWorstCase", Sdf.ValueTypeNames.Double, custom=True).Set(float(pv["fraction"]))
        if pv.get("min_fraction") is not None:
            prim.CreateAttribute("cloudeye:pathVisibilityMinFraction", Sdf.ValueTypeNames.Double, custom=True).Set(float(pv["min_fraction"]))
    if camera.get("min_eye_clearance_m") is not None:
        prim.CreateAttribute("cloudeye:cameraMinClearanceM", Sdf.ValueTypeNames.Double, custom=True).Set(float(camera["min_eye_clearance_m"]))
    return prim


def culled_stub_edge_ids_for_camera(records: list[dict], camera: dict | None, room_polygon_xz) -> list[int]:
    """The cull list for a baked camera dict (`pos` Y-up), or none without a camera."""
    if not camera or camera.get("pos") is None:
        return []
    centroid = camera.get("room_centroid_xz") or room_centroid_xz(room_polygon_xz)
    return culled_stub_edge_ids(records, (float(camera["pos"][0]), float(camera["pos"][2])), centroid)


PATH_RIBBON_PRIM = "/World/PathRibbon"
PATH_RIBBON_MATERIAL = "/World/Looks/path_ribbon"
PATH_RIBBON_RGB = (0.25, 0.85, 1.0)


def add_path_ribbon(stage: Usd.Stage, path_points_world, *, floor_y: float | None = None) -> Usd.Prim | None:
    """Morning-4: the route as a 3 cm ribbon Mesh at floor + 0.03 m
    (`export_glb.path_ribbon_mesh`; the path points already sit at that height, so
    `floor_y=None` keeps their Y) - `PATH_RIBBON_PRIM`, a sibling of the
    `/World/Path` BasisCurves (record_isaac reads the curve's points; Hydra draws
    the ribbon reliably where a 3 cm-wide curve may thin out). Bound to a cyan
    `UsdPreviewSurface` that reads unlit: emissiveColor = the cyan, roughness 1,
    so it keeps its colour under any lighting. No collision API - the robots drive
    over it."""
    from pxr import Sdf, UsdShade

    from scripts.msa.export_glb import path_ribbon_mesh

    mesh = path_ribbon_mesh(path_points_world, floor_y)
    if mesh is None:
        return None
    prim = _add_trimesh(stage, PATH_RIBBON_PRIM, mesh)
    UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
    _set_display_color(prim, PATH_RIBBON_RGB)
    mat = UsdShade.Material.Define(stage, PATH_RIBBON_MATERIAL)
    pbr = UsdShade.Shader.Define(stage, f"{PATH_RIBBON_MATERIAL}/PreviewSurface")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*PATH_RIBBON_RGB))
    pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*PATH_RIBBON_RGB))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    prim.CreateAttribute("cloudeye:pathRibbonWidthM", Sdf.ValueTypeNames.Double, custom=True).Set(float(PATH_RIBBON_WIDTH_M_USD))
    return prim


PATH_RIBBON_WIDTH_M_USD = 0.03


def add_path_and_target(stage: Usd.Stage, path_points_world, target_point, target_object_point=None,
                        target_object_id: str | None = None) -> None:
    """`/World/Path` (linear BasisCurves) + `/World/Target` (sphere) - identical
    authoring to `export_usd`'s optional prims (demo/record_isaac.py reads exactly
    these two: the curve's points and the sphere's translate op). T15h: `/World/Target`
    is the route GOAL (where the robot is meant to stop - the acceptance "within
    0.3 m of the target" is measured against it); the target OBJECT's footprint
    centroid, when given, is the separate visual-only marker `/World/TargetObject`
    (custom attr `cloudeye:targetObjectId` = `target_object_id`), so the desk stays
    labelled without the goal sitting inside a hull (T15e finding)."""
    if path_points_world:
        curve = UsdGeom.BasisCurves.Define(stage, "/World/Path")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(path_points_world)])
        curve.CreatePointsAttr([to_isaac(*p) for p in path_points_world])
        curve.CreateWidthsAttr([0.03] * len(path_points_world))
        _set_display_color(curve.GetPrim(), (0.25, 0.85, 1.0))
        add_path_ribbon(stage, path_points_world)
    if target_point is not None:
        target = UsdGeom.Sphere.Define(stage, "/World/Target")
        target.CreateRadiusAttr(0.1)
        UsdGeom.Xformable(target).AddTranslateOp().Set(to_isaac(*target_point))
        _set_display_color(target.GetPrim(), (1.0, 0.3, 0.2))
    if target_object_point is not None:
        marker = UsdGeom.Sphere.Define(stage, "/World/TargetObject")
        marker.CreateRadiusAttr(TARGET_OBJECT_MARKER_RADIUS_M)
        UsdGeom.Xformable(marker).AddTranslateOp().Set(to_isaac(*target_object_point))
        _set_display_color(marker.GetPrim(), (1.0, 0.75, 0.2))
        if target_object_id:
            from pxr import Sdf

            marker.GetPrim().CreateAttribute("cloudeye:targetObjectId", Sdf.ValueTypeNames.String, custom=True).Set(str(target_object_id))


def export_presentation_usd(
    scene,
    floor_y: float,
    ceiling_y: float,
    out_path: Path,
    *,
    floor_polygon: list[tuple[float, float]] | None = None,
    path_points_world: list[tuple[float, float, float]] | None = None,
    target_point: tuple[float, float, float] | None = None,
    camera: dict | None = None,
    wall_visual_height_m: float = PRESENTATION_WALL_HEIGHT_M,
    aspect: float = 16.0 / 9.0,
    room_centroid_xz: tuple[float, float] | None = None,
    textures: dict | None = None,
    target_object_point: tuple[float, float, float] | None = None,
    target_object_id: str | None = None,
    wall_thickness_m: float = WALL_THICKNESS_M,
    culled_stub_edge_ids=None,
    camera_husky: dict | None = None,
    asset_placements: list[dict] | None = None,
    asset_cache_root: Path = DEFAULT_USD_CACHE_ROOT,
) -> dict:
    """Write `scene_presentation.usd` from a `build_scene`-shaped trimesh Scene.

    M7 (item 4): `asset_placements` (`assets_placed.json` records) - every object
    with a `placed`/`generated` record gets `/World/Objects/<id>/visual` from
    `usd_assets.add_asset_visual` (an instanceable reference to the converted
    asset with the GLB's exact placement transform) INSTEAD of the scene's
    `visual/part_j` meshes; other objects (curtains) keep their parts. The
    summary lists them under `asset_visuals`.

    `camera_husky` (orchestrator 2026-09-07, two-pose mode): a second camera dict
    baked as `PRESENTATION_CAMERA_HUSKY_PATH` for Husky's pass.

    T15i: the T15g `wall_outline` stub is authored as one box Mesh per outline edge
    (`/World/Structure/wall_outline/visual/stub_edge_i`, `stub_edge_boxes`) and the
    edges listed in `culled_stub_edge_ids` (default: `culled_stub_edge_ids(records,
    camera XZ)` - the edges whose outward normal faces the camera) are invisible: the
    dollhouse cull the web viewer applies, so the wide shot looks INTO the room over
    the far stubs instead of at the near stub's outer face (T15e run2). Translucent
    object classes get a bound UsdPreviewSurface (`bind_translucent_object_material`).
    `textures` (T15c-prep, opt-in) as in `export_usd`: bound to `/World/Floor` and
    the visible wall stubs (`textures.apply_usd_textures`, geometry-matched).

    Prim layout: `/World/Structure/wall_i/visual/stub` (visible, floor -> floor +
    `wall_visual_height_m`, NO collision API) + `/World/Structure/wall_i_collider`
    (full height, `UsdGeom.Imageable.MakeInvisible()`, `CollisionAPI` +
    `MeshCollisionAPI` via `_add_collision_mesh` - T15h's guard makes a concave
    fragment `approximation = none`). The T15g `wall_outline` band: its stub as
    above, but the collider is `/World/Structure/wall_outline_collider/edge_i` -
    one invisible convex box per edge of `floor_polygon` (`add_wall_edge_box_colliders`,
    thickness `wall_thickness_m`, height ceiling - floor); a ring-shaped band under
    convexHull would have filled the whole room (T15e). `/World/Floor` and
    `/World/Objects/<id>/{visual/part_j,collision/hull}` unchanged from scene.usd;
    `/World/Plan/floor`, `/World/Path`, `/World/Target` (= route goal) and
    `/World/TargetObject` (footprint centroid marker, visual only) as in
    `add_path_and_target`; `/World/DomeLight` + `/World/KeyLight`
    (`add_presentation_lights`) and the camera (`add_presentation_camera`) when
    `camera` is given. The visual stub lives one level under `wall_i` (not AS
    `wall_i`) so tools/isaac_validate.py's collider_coverage - which audits every
    Mesh/Cube under /World/Structure except `/visual/` paths - keeps passing on the
    presentation file too.

    Returns a summary dict (prim counts, collider layout, light/camera numbers)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(out_path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "cloudeye:msaSchemaVersion": EXPORT_SCHEMA_VERSION,
        "cloudeye:msaPresentationVersion": PRESENTATION_SCHEMA_VERSION,
        "cloudeye:presentationWallHeightM": float(wall_visual_height_m),
        "cloudeye:axisConvention": "source data is Y-up; exported (x, y, z) -> (x, -z, y), a +90deg rotation about X (det=+1)",
    }

    UsdGeom.Xform.Define(stage, "/World/Structure")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    counts = {"walls": 0, "objects": set(), "visual_parts": 0, "hulls": 0, "floor": False}
    wall_bottom_xz: list[np.ndarray] = []
    wall_colliders: list[dict] = []
    stub_edges_summary: list[dict] = []
    translucent_materials: dict = {}
    translucent_bound: list[str] = []
    placed = placements_by_id(asset_placements)
    asset_visuals: dict[str, dict] = {}
    for node_name, mesh in iter_scene_world_meshes(scene):
        if node_name.startswith("wall_"):
            wall_id = _sanitize_prim_name(node_name)
            UsdGeom.Xform.Define(stage, f"/World/Structure/{wall_id}")
            UsdGeom.Xform.Define(stage, f"/World/Structure/{wall_id}/visual")
            is_outline = node_name == "wall_outline" and floor_polygon and len(floor_polygon) >= 3
            if is_outline:
                # T15i: per-edge stubs with the dollhouse cull (no single `visual/stub`).
                records = stub_edge_boxes(floor_polygon, floor_y, wall_visual_height_m, wall_thickness_m)
                if culled_stub_edge_ids is None:
                    culled = culled_stub_edge_ids_for_camera(records, camera, floor_polygon)
                else:
                    culled = [int(i) for i in culled_stub_edge_ids]
                stub_edges_summary = add_wall_stub_edges(
                    stage, f"/World/Structure/{wall_id}/visual", records, culled, PRESENTATION_WALL_DISPLAY_COLOR,
                )
            else:
                stub_prim = _add_trimesh(stage, f"/World/Structure/{wall_id}/visual/stub", wall_stub_mesh(mesh, floor_y, wall_visual_height_m))
                _set_display_color(stub_prim, PRESENTATION_WALL_DISPLAY_COLOR)
            if is_outline:
                edges = add_wall_edge_box_colliders(
                    stage, f"/World/Structure/{wall_id}_collider", floor_polygon, floor_y, ceiling_y, thickness_m=wall_thickness_m,
                )
                wall_colliders.append({
                    "wall": wall_id, "kind": "edge_boxes", "prim": f"/World/Structure/{wall_id}_collider",
                    "n_edges": len(edges), "thickness_m": float(wall_thickness_m), "height_m": float(ceiling_y - floor_y),
                    "band_footprint_m2": collider_footprint_stats(mesh)["footprint_area_m2"], "edges": edges,
                })
            else:
                stats = collider_footprint_stats(mesh)
                approximation = "convexHull" if stats["ratio"] <= CONVEX_HULL_MAX_AREA_RATIO else "none"
                collider = _add_collision_mesh(stage, f"/World/Structure/{wall_id}_collider", mesh, approximation=approximation)
                UsdGeom.Imageable(collider).MakeInvisible()
                wall_colliders.append({
                    "wall": wall_id, "kind": "mesh", "prim": f"/World/Structure/{wall_id}_collider", "approximation": approximation,
                    "footprint_m2": stats["footprint_area_m2"], "hull_m2": stats["hull_area_m2"],
                })
            counts["walls"] += 1
            verts = np.asarray(mesh.vertices)
            wall_bottom_xz.append(verts[np.isclose(verts[:, 1], verts[:, 1].min(), atol=1e-6)][:, [0, 2]])
        elif node_name == "floor":
            floor_prim = _add_collision_mesh(stage, "/World/Floor", mesh)
            _set_display_color(floor_prim, PRESENTATION_FLOOR_DISPLAY_COLOR)
            counts["floor"] = True
        elif "/visual/part_" in node_name or node_name.endswith("/collision/hull"):
            obj_id, kind, leaf = node_name.split("/", 2)
            obj_path = f"/World/Objects/{_sanitize_prim_name(obj_id)}"
            if obj_id not in counts["objects"]:
                UsdGeom.Xform.Define(stage, obj_path)
                UsdGeom.Xform.Define(stage, f"{obj_path}/visual")
                UsdGeom.Xform.Define(stage, f"{obj_path}/collision")
                counts["objects"].add(obj_id)
            if kind == "visual" and obj_id in placed and not is_translucent_class(obj_id):
                # M7: the referenced asset replaces every `visual/part_j` of this object.
                if obj_id not in asset_visuals:
                    asset_visuals[obj_id] = add_asset_visual(stage, obj_path, placed[obj_id], out_dir=out_path.parent, cache_root=asset_cache_root)
                continue
            if kind == "visual":
                prim = _add_trimesh(stage, f"{obj_path}/visual/{_sanitize_prim_name(leaf)}", mesh)
                from scripts.msa.export_glb import mesh_base_color_rgb

                rgb = mesh_base_color_rgb(scene.geometry[scene.graph[node_name][1]])
                if rgb is not None:
                    _set_display_color(prim, tuple(c / 255.0 for c in rgb))
                mat_path = bind_translucent_object_material(
                    stage, prim, obj_id, tuple(c / 255.0 for c in rgb) if rgb is not None else None, materials=translucent_materials,
                )
                if mat_path is not None:
                    translucent_bound.append(str(prim.GetPath()))
                counts["visual_parts"] += 1
            else:
                _add_collision_mesh(stage, f"{obj_path}/collision/hull", mesh)
                counts["hulls"] += 1

    plan = UsdGeom.Xform.Define(stage, "/World/Plan")
    if floor_polygon and len(floor_polygon) >= 3:
        _add_basis_curve_loop(stage, "/World/Plan/floor", np.asarray(floor_polygon), floor_y)
    else:
        stage.RemovePrim(plan.GetPath())

    add_path_and_target(stage, path_points_world, target_point, target_object_point, target_object_id)

    summary: dict = {
        "out_path": str(out_path), "n_walls": counts["walls"], "n_objects": len(counts["objects"]),
        "n_visual_parts": counts["visual_parts"], "n_hulls": counts["hulls"], "floor": counts["floor"],
        "wall_visual_height_m": float(wall_visual_height_m), "ceiling_y": float(ceiling_y),
        "path_points": len(path_points_world or []), "target": target_point is not None,
        "target_object": target_object_point is not None,
        "wall_colliders": [{k: v for k, v in wc.items() if k != "edges"} for wc in wall_colliders],
        "wall_collider_edges": [e for wc in wall_colliders for e in wc.get("edges", [])],
        "stub_edges": stub_edges_summary,
        "culled_stub_edges": sorted(e["edge"] for e in stub_edges_summary if e["culled"]),
        "translucent_parts": translucent_bound,
        "translucent_opacity": TRANSLUCENT_OPACITY if translucent_bound else None,
        # M7: objects whose visual is a referenced asset (same as the web GLB).
        "n_asset_visuals": len(asset_visuals),
        "asset_visuals": list(asset_visuals.values()),
    }

    if camera is not None:
        if room_centroid_xz is None:
            pts = np.asarray(floor_polygon, dtype=float) if floor_polygon else (np.vstack(wall_bottom_xz) if wall_bottom_xz else np.zeros((1, 2)))
            room_centroid_xz = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        cam_dir_xz = (float(camera["pos"][0]) - room_centroid_xz[0], float(camera["pos"][2]) - room_centroid_xz[1])
        summary["lights"] = add_presentation_lights(stage, cam_dir_xz)
        if camera.get("culled_stub_edges") is None and stub_edges_summary:
            camera = {**camera, "culled_stub_edges": summary["culled_stub_edges"]}
        add_presentation_camera(stage, camera, aspect=aspect)
        summary["camera"] = {k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in camera.items() if k != "camera_husky"}
        summary["camera"]["prim"] = PRESENTATION_CAMERA_PATH
        if camera_husky is not None:
            # two-pose mode: the steeper Husky-pass pose as a second camera prim
            add_presentation_camera(stage, camera_husky, aspect=aspect, path=PRESENTATION_CAMERA_HUSKY_PATH)
            summary["camera_husky"] = {k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in camera_husky.items()}
            summary["camera_husky"]["prim"] = PRESENTATION_CAMERA_HUSKY_PATH
    else:
        summary["lights"] = add_presentation_lights(stage, (0.0, -1.0))

    if textures:
        from scripts.msa.textures import apply_usd_textures

        summary["textures"] = apply_usd_textures(stage, textures, out_path.parent)

    stage.GetRootLayer().Save()
    return summary
