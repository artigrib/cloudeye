"""Stage E0 (GLB, SPEC §8): mirrors `export_usd.py`'s prim hierarchy as a glTF
node tree for the browser (Three.js `GLTFLoader`) - `wall_i` / `floor` at scene
root, `<obj_id>/visual/part_i` + `<obj_id>/collision/hull` per object, `Plan/*`
outline curves, and optional `Path` / `Target` when a planned path is available.

Supersedes Stage A0's flat single-mesh-per-object scene (`SPEC A6`) - node names
for walls/floor are unchanged so existing s0 callers keep working.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from scripts.msa.assets import build_placeholder
from scripts.msa.geometry import WallPolygon


def _largest_polygon(poly) -> Polygon | None:
    """A rasterized-then-DP-simplified ring can self-intersect and buffer(0) into
    a MultiPolygon; take the largest piece rather than crashing the exporter."""
    if isinstance(poly, MultiPolygon):
        parts = [p for p in poly.geoms if p.area > 1e-6]
        if not parts:
            return None
        return max(parts, key=lambda p: p.area)
    if poly.is_empty or poly.area < 1e-6:
        return None
    return poly


def _extrude_shapely_polygon_xz(poly, y0: float, height: float) -> trimesh.Trimesh | None:
    """Shared tail for extruding an already-built shapely Polygon/MultiPolygon
    into the scene's XZ-ground / Y-up convention - validate, collapse to the
    largest piece if it's a MultiPolygon, extrude, then remap. `_extrude_polygon_xz`
    (build a Polygon from raw XZ points first) and `_floor_plate_mesh` (a
    polygon that may already be a union of several pieces) both funnel through
    this."""
    if not poly.is_valid or poly.area < 1e-6:
        poly = poly.buffer(0)
    poly = _largest_polygon(poly)
    if poly is None:
        return None
    mesh = trimesh.creation.extrude_polygon(poly, height=max(height, 0.01))
    # extrude_polygon builds in XY with the polygon's own axes and extrudes along
    # +Z; remap to the scene's XZ-ground / Y-up convention.
    remap = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    mesh.apply_transform(remap)
    mesh.apply_translation((0, y0, 0))
    return mesh


def _extrude_polygon_xz(points_xz, y0: float, height: float) -> trimesh.Trimesh | None:
    return _extrude_shapely_polygon_xz(Polygon(points_xz), y0, height)


def _extrude_wall(wall: WallPolygon, floor_y: float, ceiling_y: float) -> trimesh.Trimesh | None:
    return _extrude_polygon_xz(wall.vertices, floor_y, ceiling_y - floor_y)


def _floor_mesh(free_polygon_vertices: list[tuple[float, float]], floor_y: float, slab_thickness: float = 0.10) -> trimesh.Trimesh | None:
    """Extrudes the raw free-space `floor_polygon` only, with no knowledge of
    object footprints. Kept as-is (still used by
    `assemble_with_generated.build_scene_with_generated`, out of this change's
    scope) - `build_scene` below no longer calls this; it uses
    `_floor_plate_mesh`/`_floor_plate_polygon` instead so the floor mesh also
    covers boundary-hugging object footprints (see `_floor_plate_polygon`)."""
    return _extrude_polygon_xz(free_polygon_vertices, floor_y - slab_thickness, slab_thickness)


def _object_footprint_polygon(obj: dict) -> Polygon:
    """An object's oriented-rectangle visual footprint as a shapely polygon in
    world XZ - same corner construction as `export_dxf.build_floor_plan` and
    `bootstrap.rasterize_object_footprint`."""
    cx, cz = obj["center_xy"]
    length_u, width_v = obj["size_uv"]
    angle = obj["angle_rad"]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u, v = length_u / 2, width_v / 2
    local_corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
    world_corners = [(cx + lx * cos_a - lz * sin_a, cz + lx * sin_a + lz * cos_a) for lx, lz in local_corners]
    return Polygon(world_corners)


def _floor_plate_polygon(floor_polygon: list[tuple[float, float]] | None, objects: list[dict]):
    """The solid floor plate to extrude a floor mesh/slab from =
    union(free-space `floor_polygon`, every kept object's oriented-rectangle
    footprint). Extruding a floor mesh from `floor_polygon` alone can leave a
    sliver of missing floor under an object whose footprint extends slightly
    outside the originally-computed free-space region - e.g. an object
    wall-backfilled by `bootstrap._backfill_footprint_to_walls`, or generally
    anything close to the free-space boundary - which would make the object
    look like it's floating over a gap in the visual export.

    This only feeds *solid* floor meshes/slabs (this function, `build_scene`'s
    "floor" node, and `run_bootstrap`'s USD floor mesh). It deliberately does
    NOT feed the `Plan/floor` line-outline overlay in `build_scene`/
    `export_usd` (a plan-view boundary curve, not a filled surface - it has no
    missing-fill problem to solve) or `export_dxf.build_floor_plan`'s floor
    plan (which never draws a filled floor area at all, only wall/object
    outlines and gap markers - a different kind of representation this change
    doesn't apply to). Returns a shapely Polygon/MultiPolygon, or None if
    there's no floor geometry at all (no free-space polygon and no objects)."""
    polys = []
    if floor_polygon and len(floor_polygon) >= 3:
        base = Polygon(floor_polygon)
        if not base.is_valid:
            base = base.buffer(0)
        base = _largest_polygon(base)
        if base is not None:
            polys.append(base)
    for obj in objects:
        poly = _object_footprint_polygon(obj)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 1e-9:
            polys.append(poly)
    if not polys:
        return None
    return unary_union(polys)


def _floor_plate_mesh(floor_plate, floor_y: float, slab_thickness: float = 0.10) -> trimesh.Trimesh | None:
    """Extrude a floor slab from an already-computed `_floor_plate_polygon`
    result (shapely Polygon/MultiPolygon), or None if there's nothing to
    extrude."""
    if floor_plate is None:
        return None
    return _extrude_shapely_polygon_xz(floor_plate, floor_y - slab_thickness, slab_thickness)


def _closed_loop_path(points_xz, y: float) -> "trimesh.path.path.Path3D | None":
    pts = np.asarray(points_xz, dtype=float)
    if len(pts) < 2:
        return None
    pts3 = np.column_stack([pts[:, 0], np.full(len(pts), y), pts[:, 1]])
    closed = np.vstack([pts3, pts3[:1]])
    return trimesh.load_path(closed)


PATH_RIBBON_WIDTH_M = 0.03  # morning-4: the route ribbon (visibility.PATH_RIBBON_WIDTH_M)
PATH_RIBBON_HEIGHT_ABOVE_FLOOR_M = 0.03
PATH_RIBBON_RGB = (64, 217, 255)  # export_presentation.PATH_RGB (cyan)


def path_ribbon_mesh(path_points_world, floor_y: float | None = None, *, width_m: float = PATH_RIBBON_WIDTH_M,
                     height_above_floor_m: float = PATH_RIBBON_HEIGHT_ABOVE_FLOOR_M, rgb=PATH_RIBBON_RGB) -> trimesh.Trimesh | None:
    """Morning-4: the route as a flat quad strip (Y-up), `width_m` wide, lying at
    `floor_y + height_above_floor_m` (or at each point's own Y when `floor_y` is
    None), one quad per segment with mitred joints (the offset at a vertex is the
    normal of the two adjoining tangents' mean, scaled to keep the strip width).
    Double-sided by construction (both windings emitted) so it reads from any
    camera in renderers that cull back faces. Cyan vertex colours; None for < 2
    distinct points."""
    pts = np.asarray(path_points_world, dtype=float).reshape(-1, 3)
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i, [0, 2]] - pts[i - 1, [0, 2]]) > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        return None
    y = np.full(len(pts), float(floor_y) + height_above_floor_m) if floor_y is not None else pts[:, 1]
    xz = pts[:, [0, 2]]
    tangents = np.diff(xz, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])  # left of the tangent in XZ
    half = width_m / 2.0
    left, right = [], []
    for i in range(len(pts)):
        if i == 0:
            n, scale = normals[0], 1.0
        elif i == len(pts) - 1:
            n, scale = normals[-1], 1.0
        else:
            m = normals[i - 1] + normals[i]
            if np.linalg.norm(m) < 1e-6:  # a 180 deg turn: fall back to the incoming normal
                n, scale = normals[i - 1], 1.0
            else:
                n = m / np.linalg.norm(m)
                scale = min(1.0 / max(float(n @ normals[i - 1]), 0.25), 4.0)  # mitre length, capped
        left.append((xz[i, 0] + n[0] * half * scale, y[i], xz[i, 1] + n[1] * half * scale))
        right.append((xz[i, 0] - n[0] * half * scale, y[i], xz[i, 1] - n[1] * half * scale))
    verts = np.asarray(left + right, dtype=float)
    n = len(pts)
    faces = []
    for i in range(n - 1):
        l0, l1, r0, r1 = i, i + 1, n + i, n + i + 1
        faces += [(l0, r0, l1), (l1, r0, r1), (l0, l1, r0), (l1, r1, r0)]  # both windings
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile([*rgb, 255], (len(verts), 1)))
    return mesh


def build_scene(
    walls: list[WallPolygon],
    floor_polygon: list[tuple[float, float]] | None,
    floor_y: float,
    ceiling_y: float,
    objects: list[dict],
    *,
    path_points_world: list[tuple[float, float, float]] | None = None,
    target_point: tuple[float, float, float] | None = None,
    textures: dict | None = None,
    object_material: str = "pbr",
    visual_overrides: dict[str, list[trimesh.Trimesh]] | None = None,
    wall_band=None,
) -> trimesh.Scene:
    """`objects` items: {id, label, center_xy, size_uv, angle_rad, height,
    bbox_min_y, color_rgb, hull_xz (optional, world XZ convex hull for the
    collision node)}. `path_points_world`/`target_point` are optional - see
    `export_usd.export_usd`'s docstring for why a bootstrap run may not have
    them.

    `textures` (T15c, opt-in): the dict `scripts.msa.textures.bake_room_textures`
    returns - baked floor/wall base-colour textures get attached to the `floor`
    / `wall_i` geometries as glTF PBR materials with UVs. `object_material`:
    `"pbr"` (default) gives each placeholder part an untextured PBR material
    (baseColorFactor = `color_rgb`, roughness 0.8, metallic 0) so it lights
    properly in the web viewer; `"vertex"` keeps the pre-T15c bare vertex
    colours (COLOR_0 + default material).

    `visual_overrides` (T15f, opt-in): `{obj_id: [meshes]}` already in the
    placeholder LOCAL frame (`scripts.msa.assets.build_placeholder`'s
    convention: XZ-centred at the origin, Y up from 0, X <- size_uv[0],
    Z <- size_uv[1]) - a generated mesh or a real asset. They replace that
    object's placeholder parts under `<id>/visual/` (node `part_<j>`, or
    `metadata["msa_part_name"]` if set), get the very same world transform,
    and keep their own visuals/materials untouched. Everything else (walls,
    floor plate, collision hulls, Plan curves, textures) is unchanged, so an
    assembled scene matches `scene.glb` node for node.

    `wall_band` (T15g): a shapely Polygon/MultiPolygon - `geometry.room_wall_band`
    of the regularized room outline. When given, ONE `wall_outline` node
    (extruded floor_y -> ceiling_y) replaces the per-fragment `wall_i` meshes;
    the raw fragments still get their `Plan/wall_i` curves. `None` (default)
    keeps the pre-T15g per-wall extrusions."""
    scene = trimesh.Scene()
    if wall_band is not None:
        band_mesh = _extrude_shapely_polygon_xz(wall_band, floor_y, ceiling_y - floor_y)
        if band_mesh is None:
            logging.getLogger(__name__).warning("wall_outline: wall band polygon is degenerate - no GLB mesh emitted")
        else:
            scene.add_geometry(band_mesh, node_name="wall_outline")
    else:
        for i, wall in enumerate(walls):
            mesh = _extrude_wall(wall, floor_y, ceiling_y)
            if mesh is None:
                # T15b: never skip a wall silently - the hero's wall_2 (5.19 m^2)
                # exported with no mesh for a whole day this way.
                logging.getLogger(__name__).warning(
                    "wall_%d: polygon (%d vertices, area_m2=%.3f) is degenerate - no GLB mesh emitted", i, len(wall.vertices), wall.area_m2
                )
                continue
            scene.add_geometry(mesh, node_name=f"wall_{i}")

    # Floor plate = union(free-space floor_polygon, every object's footprint) -
    # see _floor_plate_polygon - not floor_polygon alone, so an object whose
    # footprint edges right up against (or slightly past) the free-space
    # boundary still has solid floor mesh underneath it.
    floor_plate = _floor_plate_polygon(floor_polygon, objects)
    if floor_plate is not None:
        floor = _floor_plate_mesh(floor_plate, floor_y)
        if floor is not None:
            scene.add_geometry(floor, node_name="floor")

    for obj in objects:
        obj_id = obj["id"]
        length_u, width_v = obj["size_uv"]
        parts = build_placeholder(obj["label"], length_u, width_v, obj["height"])
        r, g, b = obj.get("color_rgb", (150, 150, 150))
        angle = obj["angle_rad"]
        cx, cz = obj["center_xy"]
        y0 = obj.get("bbox_min_y", floor_y)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        world_transform = np.array(
            [
                [cos_a, 0, -sin_a, cx],
                [0, 1, 0, y0],
                [sin_a, 0, cos_a, cz],
                [0, 0, 0, 1],
            ]
        )

        scene.graph.update(frame_to=obj_id, frame_from=scene.graph.base_frame)
        scene.graph.update(frame_to=f"{obj_id}/visual", frame_from=obj_id)
        override = (visual_overrides or {}).get(obj_id)
        if override is not None:
            for j, part in enumerate(override):
                part = part.copy()
                part.apply_transform(world_transform)
                node_name = f"{obj_id}/visual/{part.metadata.get('msa_part_name', f'part_{j}')}"
                scene.add_geometry(part, node_name=node_name, geom_name=node_name, parent_node_name=f"{obj_id}/visual")
            parts = []
        for j, part in enumerate(parts):
            part = part.copy()
            if object_material == "pbr":
                part.visual = _pbr_color_visual((r, g, b))
            else:
                part.visual.vertex_colors = np.tile([r, g, b, 255], (len(part.vertices), 1))
            part.apply_transform(world_transform)
            scene.add_geometry(part, node_name=f"{obj_id}/visual/part_{j}", parent_node_name=f"{obj_id}/visual")

        hull_xz = obj.get("hull_xz")
        if hull_xz and len(hull_xz) >= 3:
            scene.graph.update(frame_to=f"{obj_id}/collision", frame_from=obj_id)
            hull_mesh = _extrude_polygon_xz(np.asarray(hull_xz), y0, obj["height"])
            if hull_mesh is not None:
                scene.add_geometry(hull_mesh, node_name=f"{obj_id}/collision/hull", parent_node_name=f"{obj_id}/collision")

    if walls or floor_polygon:
        scene.graph.update(frame_to="Plan", frame_from=scene.graph.base_frame)
        for i, wall in enumerate(walls):
            loop = _closed_loop_path(wall.vertices, floor_y)
            if loop is not None:
                scene.add_geometry(loop, node_name=f"Plan/wall_{i}", parent_node_name="Plan")
        if floor_polygon and len(floor_polygon) >= 3:
            loop = _closed_loop_path(floor_polygon, floor_y)
            if loop is not None:
                scene.add_geometry(loop, node_name="Plan/floor", parent_node_name="Plan")

    if path_points_world:
        # Morning-4: `Path` is the 3 cm ribbon mesh at floor + 0.03 m (what must stay
        # visible in the recording); the bare polyline rides along as `Path_curve`.
        ribbon = path_ribbon_mesh(path_points_world, floor_y)
        if ribbon is not None:
            scene.add_geometry(ribbon, node_name="Path")
        path = trimesh.load_path(np.asarray(path_points_world, dtype=float))
        scene.add_geometry(path, node_name="Path_curve")

    if target_point is not None:
        target = trimesh.creation.icosphere(radius=0.1)
        target.apply_translation(target_point)
        scene.add_geometry(target, node_name="Target")

    if textures:
        from scripts.msa.textures import apply_room_textures

        apply_room_textures(scene, textures)

    return scene


def _pbr_color_visual(rgb):
    """Untextured PBR colour material visual for a placeholder part (T15c) - see
    `scripts.msa.textures.pbr_color_material`."""
    from scripts.msa.textures import pbr_color_material

    return trimesh.visual.TextureVisuals(material=pbr_color_material(rgb))


def mesh_base_color_rgb(geom: trimesh.Trimesh) -> tuple[int, int, int] | None:
    """Flat RGB of a GLB mesh however its colour was stored - mean vertex colour
    (pre-T15c `object_material="vertex"` exports) or the PBR `baseColorFactor`
    (T15c default). None if neither is present (e.g. a textured wall)."""
    try:
        kind = geom.visual.kind
    except Exception:
        return None
    if kind == "vertex":
        vc = np.asarray(geom.visual.vertex_colors)
        if len(vc) >= len(geom.vertices):
            r, g, b = vc[:, :3].mean(axis=0).astype(int).tolist()
            return int(r), int(g), int(b)
        return None
    if kind == "texture":
        mat = geom.visual.material
        factor = getattr(mat, "baseColorFactor", None)
        f = None
        if factor is not None:
            f = np.asarray(factor, dtype=float)[:3]
            if f.max() <= 1.0:
                f = f * 255.0
        # T15f: a textured asset (Poly Haven GLB) renders as its mean texture
        # colour in the software renderers (render_perspective / render_top_ortho
        # cannot sample textures) - modulated by baseColorFactor if both exist.
        texture = getattr(mat, "baseColorTexture", None)
        if texture is not None:
            try:
                thumb = texture.convert("RGB")
                thumb.thumbnail((64, 64))
                mean = np.asarray(thumb, dtype=float).reshape(-1, 3).mean(axis=0)
                if f is not None:
                    mean = mean * (f / 255.0)
                return tuple(int(round(min(255.0, max(0.0, c)))) for c in mean)
            except Exception:
                pass
        if f is not None:
            return tuple(int(round(c)) for c in f)
    return None


def export_glb(scene: trimesh.Scene, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path.as_posix(), file_type="glb")
