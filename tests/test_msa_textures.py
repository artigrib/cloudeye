"""T15c: baked floor/wall textures from the RGB cloud (scripts/msa/textures.py)
and the export_glb hook that embeds them as glTF PBR materials."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import trimesh

from scripts.msa.export_glb import build_scene, export_glb, mesh_base_color_rgb
from scripts.msa.geometry import WallPolygon
from scripts.msa.textures import (
    CAP_SWATCH_PX,
    WallAtlas,
    apply_room_textures,
    apply_transform_xyz,
    apply_usd_textures,
    bake_floor_texture,
    bake_room_textures,
    bake_wall_band_texture,
    bake_wall_outline_textures,
    bake_wall_texture,
    detect_outline_wall,
    floor_polygon_from_scene,
    floor_uv,
    load_textures_json,
    outline_bands,
    resolve_cloud_frame,
    rotate_points_xz,
    swap_xz_transform,
    textures_summary,
    wall_frame,
    wall_uv,
    walls_from_scene,
    yaw_from_meta,
)

FLOOR_Y, CEIL_Y = 0.0, 2.5
# 4 m (x) by 3 m (z) room with a thin wall along z = 3 (x in [0, 4]).
FLOOR_POLY = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
WALL = WallPolygon(vertices=[(0.0, 3.0), (4.0, 3.0), (4.0, 3.1), (0.0, 3.1)], area_m2=0.4)


def synthetic_cloud(rng: np.random.Generator, *, n_floor: int = 60000, n_wall: int = 40000):
    """Floor: green everywhere except a red 1 m square at x in [3,4], z in [0,1].
    Wall (z ~ 3.05): blue on the lower half, white on the upper half."""
    fx = rng.uniform(0, 4, n_floor)
    fz = rng.uniform(0, 3, n_floor)
    fy = rng.normal(FLOOR_Y, 0.01, n_floor)
    f_rgb = np.tile([40, 200, 40], (n_floor, 1))
    red = (fx >= 3.0) & (fz <= 1.0)
    f_rgb[red] = [220, 30, 30]
    wx = rng.uniform(0, 4, n_wall)
    wy = rng.uniform(FLOOR_Y, CEIL_Y, n_wall)
    wz = rng.normal(3.05, 0.01, n_wall)
    w_rgb = np.where((wy < (FLOOR_Y + CEIL_Y) / 2)[:, None], [30, 30, 220], [245, 245, 245])
    xyz = np.vstack([np.column_stack([fx, fy, fz]), np.column_stack([wx, wy, wz])])
    rgb = np.vstack([f_rgb, w_rgb]).astype(np.uint8)
    return xyz, rgb


@pytest.fixture(scope="module")
def cloud():
    return synthetic_cloud(np.random.default_rng(0))


class TestFrame:
    def test_rotate_points_xz_matches_geometry_convention(self):
        # CCW by 90 deg about (1, 1): (2, 1) -> (1, 2); Y untouched.
        out = rotate_points_xz(np.array([[2.0, 0.7, 1.0]]), math.pi / 2, (1.0, 1.0))
        assert np.allclose(out, [[1.0, 0.7, 2.0]])

    def test_rotate_identity_is_copy(self):
        src = np.array([[1.0, 2.0, 3.0]])
        out = rotate_points_xz(src, 0.0)
        assert np.array_equal(out, src) and out is not src

    def test_yaw_from_meta(self):
        assert yaw_from_meta({}) == (0.0, (0.0, 0.0))
        assert yaw_from_meta({"yaw_applied": False, "yaw_correction_rad": 1.0}) == (0.0, (0.0, 0.0))
        a, c = yaw_from_meta({"yaw_applied": True, "yaw_correction_rad": 0.5, "yaw_rotation_center_xy": [1.0, -2.0]})
        assert a == 0.5 and c == (1.0, -2.0)

    def test_swap_xz_transform(self):
        t = swap_xz_transform(origin_x=-1.0, origin_z=-6.0)
        out = apply_transform_xyz(np.array([[-1.0, 0.3, -6.0], [0.0, 0.3, -6.0]]), t)
        # Origin corner maps to itself; +1 m in x becomes +1 m in z.
        assert np.allclose(out[0], [-1.0, 0.3, -6.0])
        assert np.allclose(out[1], [-1.0, 0.3, -5.0])

    def test_wall_frame_is_order_independent_and_along_wall(self):
        f1 = wall_frame(WALL.vertices)
        f2 = wall_frame(list(reversed(WALL.vertices)))
        assert f1 == f2
        assert abs(f1.dir_xz[0]) == pytest.approx(1.0, abs=1e-6)
        assert f1.length_m == pytest.approx(4.0, abs=1e-6)
        assert f1.half_thickness_m == pytest.approx(0.05, abs=1e-6)
        assert f1.origin_xz[0] == pytest.approx(0.0, abs=1e-6)


class TestBake:
    def test_floor_texture_has_colour_where_the_cloud_does(self, cloud):
        xyz, rgb = cloud
        ft = bake_floor_texture(xyz, rgb, FLOOR_POLY, FLOOR_Y, px_per_m=50, pad_m=0.0)
        assert ft.image.shape == (150, 200, 3)
        # 60k points over 30k pixels -> Poisson hit rate 1 - e^-2 ~ 0.86.
        assert ft.n_points > 50000 and ft.coverage > 0.8
        # Red square at x in [3,4], z in [0,1] -> columns 150..200, rows 0..50.
        red = ft.image[5:45, 155:195]
        green = ft.image[60:140, 10:140]
        assert red[..., 0].mean() > 180 and red[..., 1].mean() < 80
        assert green[..., 1].mean() > 150 and green[..., 0].mean() < 100

    def test_wall_texture_upright_rows(self, cloud):
        xyz, rgb = cloud
        wt = bake_wall_texture(xyz, rgb, WALL.vertices, FLOOR_Y, CEIL_Y, px_per_m=40)
        h, w = wt.image.shape[:2]
        assert (w, h) == (160, 100)
        assert wt.n_points > 30000
        top = wt.image[: h // 2 - 5]
        bottom = wt.image[h // 2 + 5 :]
        assert top.mean() > 200  # white upper half sits in the top rows
        assert bottom[..., 2].mean() > 180 and bottom[..., 0].mean() < 80  # blue lower half

    def test_holes_filled_and_not_empty(self):
        # A sparse cloud: one red point and one blue point, everything else a hole.
        xyz = np.array([[0.5, 0.0, 0.5], [3.5, 0.0, 2.5]])
        rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        ft = bake_floor_texture(xyz, rgb, FLOOR_POLY, FLOOR_Y, px_per_m=10, pad_m=0.0)
        assert ft.image.min() >= 0 and ft.image.max() > 0
        assert ft.image.std() > 0  # not a flat gray fill
        assert ft.coverage < 0.01

    def test_bake_room_textures_applies_yaw(self, cloud, tmp_path):
        # Rotate the *scene* 90 deg CCW about (2, 1.5); the cloud stays in the
        # original frame and bake_room_textures must apply the same rotation.
        xyz, rgb = cloud
        angle, center = math.pi / 2, (2.0, 1.5)
        rot_floor = [tuple(rotate_points_xz(np.array([[x, 0.0, z]]), angle, center)[0, [0, 2]]) for x, z in FLOOR_POLY]
        rot_wall = WallPolygon(vertices=[tuple(rotate_points_xz(np.array([[x, 0.0, z]]), angle, center)[0, [0, 2]]) for x, z in WALL.vertices], area_m2=0.4)
        tex = bake_room_textures(None, [rot_wall], rot_floor, FLOOR_Y, CEIL_Y, tmp_path, yaw_correction_rad=angle, yaw_rotation_center_xy=center, cloud=(xyz, rgb), px_per_m=50)
        assert tex["floor"] is not None and (tmp_path / "floor_texture.png").exists()
        assert 0 in tex["walls"] and (tmp_path / "wall_0_texture.png").exists()
        assert tex["walls"][0]["n_points"] > 30000, "wall points only land on the wall if the cloud was rotated with the scene"
        # Sample the floor texture at the rotated red square's centre: (3.5, 0.5) -> rotated.
        rx, _, rz = rotate_points_xz(np.array([[3.5, 0.0, 0.5]]), angle, center)[0]
        xmin, zmin, xmax, zmax = tex["floor"]["extent_xz"]
        img = tex["floor"]["image"]
        col = int((rx - xmin) / (xmax - xmin) * img.shape[1])
        row = int((rz - zmin) / (zmax - zmin) * img.shape[0])
        px = img[row, col]
        assert px[0] > 180 and px[1] < 80, f"expected red at rotated square centre, got {px}"

        # Without the yaw the rotated wall's band only catches stray floor points
        # (the real wall points are 90 deg away).
        tex_norot = bake_room_textures(None, [rot_wall], rot_floor, FLOOR_Y, CEIL_Y, tmp_path / "norot", cloud=(xyz, rgb), px_per_m=50)
        assert tex_norot["walls"][0]["n_points"] < tex["walls"][0]["n_points"] * 0.5


class TestUv:
    def test_floor_uv_range_and_orientation(self):
        extent = (0.0, 0.0, 4.0, 3.0)
        uv = floor_uv(np.array([[0.0, 0, 0.0], [4.0, 0, 3.0], [2.0, 0, 1.5]]), extent)
        assert np.allclose(uv[0], [0.0, 1.0])  # z = zmin is image row 0 = top -> v = 1 (trimesh/OpenGL)
        assert np.allclose(uv[1], [1.0, 0.0])
        assert np.allclose(uv[2], [0.5, 0.5])
        assert uv.min() >= 0.0 and uv.max() <= 1.0

    def test_wall_uv_range(self):
        frame = wall_frame(WALL.vertices)
        verts = np.array([[0.0, FLOOR_Y, 3.0], [4.0, CEIL_Y, 3.1], [2.0, 1.25, 3.05]])
        uv = wall_uv(verts, frame, FLOOR_Y, CEIL_Y)
        assert np.allclose(uv[0], [0.0, 0.0]) and np.allclose(uv[1], [1.0, 1.0]) and np.allclose(uv[2], [0.5, 0.5])


class TestExportHook:
    @pytest.fixture(scope="class")
    def textured_glb(self, cloud, tmp_path_factory):
        xyz, rgb = cloud
        out = tmp_path_factory.mktemp("t15c_glb")
        tex = bake_room_textures(None, [WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, out, cloud=(xyz, rgb), px_per_m=50)
        objects = [
            {
                "id": "bed_0",
                "label": "bed",
                "center_xy": (2.0, 1.5),
                "size_uv": (2.0, 1.5),
                "angle_rad": 0.0,
                "height": 0.6,
                "bbox_min_y": 0.0,
                "color_rgb": (200, 100, 50),
                "hull_xz": [(1.0, 0.75), (3.0, 0.75), (3.0, 2.25), (1.0, 2.25)],
            }
        ]
        scene = build_scene([WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, objects, textures=tex)
        glb = out / "scene_textured.glb"
        export_glb(scene, glb)
        return glb

    def test_floor_and_wall_carry_textures_with_uvs(self, textured_glb):
        scene = trimesh.load(str(textured_glb), force="scene")
        for node in ("floor", "wall_0"):
            geom = scene.geometry[scene.graph[node][1]]
            assert geom.visual.kind == "texture", node
            assert geom.visual.material.baseColorTexture is not None, node
            uv = np.asarray(geom.visual.uv)
            assert uv.shape == (len(geom.vertices), 2)
            assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6, node
        floor = scene.geometry[scene.graph["floor"][1]]
        img = np.asarray(floor.visual.material.baseColorTexture.convert("RGB"))
        assert img.shape[0] > 8 and img.std() > 10  # a real texture, not a flat fill

    def test_objects_get_pbr_colour_materials(self, textured_glb):
        scene = trimesh.load(str(textured_glb), force="scene")
        parts = [n for n in scene.graph.nodes_geometry if "visual" in n and "collision" not in n]
        assert parts
        geom = scene.geometry[scene.graph[parts[0]][1]]
        assert geom.visual.kind == "texture"
        mat = geom.visual.material
        assert mat.baseColorTexture is None
        assert mesh_base_color_rgb(geom) == (200, 100, 50)
        assert mat.roughnessFactor == pytest.approx(0.8)
        assert mat.metallicFactor == pytest.approx(0.0)

    def test_glb_json_has_pbr_materials(self, textured_glb):
        import struct

        b = textured_glb.read_bytes()
        length = struct.unpack("<I", b[12:16])[0]
        gltf = json.loads(b[20 : 20 + length])
        textured = [m for m in gltf["materials"] if "baseColorTexture" in m.get("pbrMetallicRoughness", {})]
        assert len(textured) >= 2  # floor + wall
        assert len(gltf.get("images", [])) >= 2
        assert all(m["pbrMetallicRoughness"].get("metallicFactor", 1.0) == 0.0 for m in gltf["materials"] if "pbrMetallicRoughness" in m)

    def test_vertex_material_mode_still_available(self):
        objects = [{"id": "box_0", "label": "box", "center_xy": (1.0, 1.0), "size_uv": (1.0, 1.0), "angle_rad": 0.0, "height": 1.0, "bbox_min_y": 0.0, "color_rgb": (10, 20, 30)}]
        scene = build_scene([], FLOOR_POLY, FLOOR_Y, CEIL_Y, objects, object_material="vertex")
        geom = scene.geometry[scene.graph["box_0/visual/part_0"][1]]
        assert geom.visual.kind == "vertex"
        assert mesh_base_color_rgb(geom) == (10, 20, 30)


class TestRetextureExisting:
    def test_walls_and_floor_recovered_from_glb_and_frame_autodetect(self, cloud, tmp_path):
        xyz, rgb = cloud
        scene = build_scene([WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, [], object_material="vertex")
        glb = tmp_path / "scene.glb"
        export_glb(scene, glb)
        loaded = trimesh.load(str(glb), force="scene")
        walls = walls_from_scene(loaded, FLOOR_Y)
        assert set(walls) == {0}
        assert wall_frame(walls[0].vertices).length_m == pytest.approx(4.0, abs=1e-4)
        poly = floor_polygon_from_scene(loaded)
        assert poly is not None and len(poly) >= 4
        name, t, diag = resolve_cloud_frame("auto", xyz, poly, FLOOR_Y, 0.0, (0.0, 0.0), {"origin_x": 0.0, "origin_z": 0.0})
        assert name == "identity" and t is None
        assert diag["identity"] > 0.95 > diag["swap_xz"]
        tex = bake_room_textures(None, walls, poly, FLOOR_Y, CEIL_Y, tmp_path, cloud=(xyz, rgb), px_per_m=40)
        assert apply_room_textures(loaded, tex) == 2


# --------------------------------------------------------------------------- T15c-prep


def _extrude_ring_band(outline_xz, y0: float, height: float, thickness: float = 0.1) -> trimesh.Trimesh:
    """T15g-style single wall: the closed outline buffered into a thin band and
    extruded (outer ring minus inner ring)."""
    from shapely.geometry import Polygon

    from scripts.msa.export_glb import _extrude_shapely_polygon_xz

    poly = Polygon(outline_xz)
    band = poly.buffer(thickness / 2).difference(poly.buffer(-thickness / 2))
    return _extrude_shapely_polygon_xz(band, y0, height)


def outline_cloud(rng: np.random.Generator, outline_xz, colours, y0: float, height: float, n_per_edge: int = 15000):
    """A coloured vertical plane of points on every edge of `outline_xz`."""
    pts, cols = [], []
    n = len(outline_xz)
    for i in range(n):
        (x0, z0), (x1, z1) = outline_xz[i], outline_xz[(i + 1) % n]
        t = rng.uniform(0, 1, n_per_edge)
        x = x0 + (x1 - x0) * t + rng.normal(0, 0.01, n_per_edge)
        z = z0 + (z1 - z0) * t + rng.normal(0, 0.01, n_per_edge)
        y = rng.uniform(y0, y0 + height, n_per_edge)
        pts.append(np.column_stack([x, y, z]))
        cols.append(np.tile(colours[i], (n_per_edge, 1)))
    return np.vstack(pts), np.vstack(cols).astype(np.uint8)


SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
EDGE_RGB = [(220, 30, 30), (30, 200, 30), (30, 30, 220), (230, 220, 30)]  # z=0 red, x=4 green, z=3 blue, x=0 yellow


class TestFloorPolygon:
    L_POLY = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 3.0), (0.0, 3.0)]

    def test_hole_is_filled_and_outside_polygon_is_neutral(self):
        from shapely import contains_xy
        from shapely.geometry import Polygon

        rng = np.random.default_rng(1)
        x = rng.uniform(0, 4, 120000)
        z = rng.uniform(0, 3, 120000)
        inside = contains_xy(Polygon(self.L_POLY), x, z)
        hole = np.hypot(x - 1.0, z - 1.0) < 0.4  # unobserved disc in the floor
        keep = inside & ~hole
        xyz = np.column_stack([x[keep], rng.normal(0, 0.01, keep.sum()), z[keep]])
        # Speckled green: observed texels keep their per-point noise, the far-field tone is smooth.
        rgb = np.clip(np.array([40, 200, 40]) + rng.normal(0, 40, (keep.sum(), 3)), 0, 255).astype(np.uint8)
        ft = bake_floor_texture(xyz, rgb, self.L_POLY, 0.0, px_per_m=50, pad_m=0.5)
        xmin, zmin, xmax, zmax = ft.extent_xz
        assert (xmin, zmin, xmax, zmax) == pytest.approx((-0.5, -0.5, 4.5, 3.5))

        def px(px_x, px_z):
            return int((px_z - zmin) / (zmax - zmin) * ft.image.shape[0]), int((px_x - xmin) / (xmax - xmin) * ft.image.shape[1])

        r, c = px(1.0, 1.0)  # hole centre: filled with the surrounding green
        hole_block = ft.image[r - 3 : r + 4, c - 3 : c + 4].reshape(-1, 3).astype(float)
        assert ft.mask[r, c] and hole_block[:, 1].mean() > 150 and hole_block[:, 0].mean() < 100
        r, c = px(3.0, 2.6)  # inside the AABB but outside the L: masked, smooth neutral tone
        assert not ft.mask[r, c]
        outside_block = ft.image[r - 5 : r + 6, c - 5 : c + 6].reshape(-1, 3).astype(float)
        r2, c2 = px(3.0, 1.0)  # observed floor: speckled
        inside_block = ft.image[r2 - 5 : r2 + 6, c2 - 5 : c2 + 6].reshape(-1, 3).astype(float)
        assert outside_block.std(axis=0).max() < inside_block.std(axis=0).max() / 4
        assert abs(outside_block[:, 1].mean() - 200) < 40 and outside_block[:, 0].mean() < 100  # still the floor's tone
        r, c = px(-0.3, -0.3)  # padding: outside
        assert not ft.mask[r, c]
        assert 0.5 < ft.coverage <= 1.0 and ft.fill_fraction == pytest.approx(1.0 - ft.coverage)
        uv = ft.uv_for_xz(self.L_POLY)
        assert uv.min() >= 0.0 and uv.max() <= 1.0
        assert np.allclose(uv[0], [0.5 / 5.0, 1.0 - 0.5 / 4.0])  # (0,0) -> u = pad/W, v = 1 - pad/D

    def test_shapely_polygon_with_hole_is_masked(self):
        from shapely.geometry import Polygon

        poly = Polygon(SQUARE, holes=[[(1.5, 1.0), (2.5, 1.0), (2.5, 2.0), (1.5, 2.0)]])
        rng = np.random.default_rng(2)
        xyz = np.column_stack([rng.uniform(0, 4, 20000), np.zeros(20000), rng.uniform(0, 3, 20000)])
        rgb = np.tile([200, 200, 200], (20000, 1)).astype(np.uint8)
        ft = bake_floor_texture(xyz, rgb, poly, 0.0, px_per_m=20, pad_m=0.0)
        assert ft.image.shape == (60, 80, 3)
        assert not ft.mask[30, 40] and ft.mask[5, 5]  # hole centre masked, corner inside


class TestWallBand:
    def test_band_texture_from_a_diagonal_coloured_plane(self):
        rng = np.random.default_rng(3)
        p0, p1 = (0.0, 0.0), (3.0, 4.0)  # length 5
        n = 60000
        s = rng.uniform(0, 5, n)
        y = rng.uniform(0, 2.5, n)
        t = rng.normal(0, 0.01, n)
        d = np.array([0.6, 0.8])
        nrm = np.array([-0.8, 0.6])
        x = p0[0] + s * d[0] + t * nrm[0]
        z = p0[1] + s * d[1] + t * nrm[1]
        rgb = np.where((y >= 1.25)[:, None], [245, 245, 245], np.where((s < 2.5)[:, None], [220, 30, 30], [30, 30, 220])).astype(np.uint8)
        # A green decoy plane 1 m behind the band: must be excluded by band_m.
        decoy = np.column_stack([x + 1.0 * nrm[0], y, z + 1.0 * nrm[1]])
        xyz = np.vstack([np.column_stack([x, y, z]), decoy])
        rgb = np.vstack([rgb, np.tile([30, 220, 30], (n, 1)).astype(np.uint8)])
        wt = bake_wall_band_texture(xyz, rgb, (p0, p1), 0.0, 2.5, px_per_m=40)
        h, w = wt.image.shape[:2]
        assert (w, h) == (200, 100)
        assert 55000 < wt.n_points <= n
        assert wt.image[:45].mean() > 200  # upper half white (row 0 = top)
        lower_left = wt.image[55:, 5:95]
        lower_right = wt.image[55:, 105:195]
        assert lower_left[..., 0].mean() > 180 and lower_left[..., 2].mean() < 80  # red at s < 2.5 = p0 side
        assert lower_right[..., 2].mean() > 180 and lower_right[..., 0].mean() < 80
        assert wt.image[..., 1].mean() < 160  # no green decoy
        assert wt.coverage > 0.8 and wt.fill_fraction == pytest.approx(1.0 - wt.coverage)
        uv = wt.band.uv_for_xyz(np.array([[0.0, 0.0, 0.0], [3.0, 2.5, 4.0], [1.5, 1.25, 2.0]]))
        assert np.allclose(uv, [[0, 0], [1, 1], [0.5, 0.5]])

    def test_outline_bands_skip_degenerate_edges_and_closing_vertex(self):
        bands = outline_bands(SQUARE + [SQUARE[0]], 0.0, 2.5)
        assert len(bands) == 4
        assert bands[1].p0_xz == (4.0, 0.0) and bands[1].p1_xz == (4.0, 3.0)
        assert len(outline_bands([(0, 0), (0, 0), (1, 0)], 0.0, 1.0)) == 2  # (0,0)->(0,0) dropped, closing edge kept

    def test_outline_atlas_layout_and_face_uvs(self):
        rng = np.random.default_rng(4)
        xyz, rgb = outline_cloud(rng, SQUARE, EDGE_RGB, 0.0, 2.5)
        atlas = bake_wall_outline_textures(xyz, rgb, SQUARE, 0.0, 2.5, px_per_m=20, band_m=0.1)
        assert len(atlas.bands) == 4
        h, w = atlas.image.shape[:2]
        assert (w, h) == (80 + 60 + 80 + 60 + CAP_SWATCH_PX, 50)
        # Bands tile [0, cap_u0] left to right without gaps; the cap swatch ends at 1.
        assert atlas.u_ranges[0][0] == 0.0
        for a, b in zip(atlas.u_ranges, atlas.u_ranges[1:]):
            assert a[1] == pytest.approx(b[0])
        assert atlas.u_ranges[-1][1] == pytest.approx(atlas.cap_u[0]) and atlas.cap_u[1] == 1.0
        for j, expect in enumerate(EDGE_RGB):
            u0, u1 = atlas.u_ranges[j]
            cols = atlas.image[:, int(u0 * w) + 2 : int(u1 * w) - 2].reshape(-1, 3).mean(axis=0)
            if j == 3:
                assert cols[0] > 180 and cols[1] > 170 and cols[2] < 100, (j, cols)  # yellow
            else:
                assert np.argmax(cols) == np.argmax(expect), (j, cols)
        assert atlas.n_points > 50000 and atlas.coverage > 0.7
        assert len(atlas.per_band) == 4 and all("fill_fraction" in p for p in atlas.per_band)

        mesh = _extrude_ring_band(SQUARE, 0.0, 2.5)
        idx, cost = atlas.face_band_index(mesh.vertices, mesh.faces)
        assert (idx < 0).sum() > 0  # top + bottom caps
        side = idx >= 0
        assert side.sum() > 0 and cost[side].max() < 0.2  # every side face lies on one of the 4 planes
        centroids = mesh.vertices[mesh.faces].mean(axis=1)
        # Faces in the x = 4 plane, away from the corners (the corner faces at (4 +/- 0.05, 0)
        # lie in the z = 0 / z = 3 planes and rightly belong to bands 0 / 2).
        east = side & (np.abs(centroids[:, 0] - 4.0) < 0.1) & (centroids[:, 2] > 0.2) & (centroids[:, 2] < 2.8)
        assert east.any() and (idx[east] == 1).all()
        # (shapely's buffer rounds the ring's corners into tiny facets whose nearest plane is
        # legitimately either neighbour, so no assertion on those.)
        uv = atlas.uv_for_faces(mesh.vertices, mesh.faces)
        assert uv.shape == (len(mesh.faces) * 3, 2) and uv.min() >= 0.0 and uv.max() <= 1.0
        uv_f = uv.reshape(-1, 3, 2)
        u_east = uv_f[east][..., 0]
        assert u_east.min() >= atlas.u_ranges[1][0] - 1e-9 and u_east.max() <= atlas.u_ranges[1][1] + 1e-9
        caps = idx < 0
        assert np.allclose(uv_f[caps][..., 0], 0.5 * (atlas.cap_u[0] + atlas.cap_u[1]))
        # JSON round trip keeps the layout.
        again = WallAtlas.from_json(atlas.to_json(), atlas.image)
        assert again.u_ranges == atlas.u_ranges and again.cap_u == atlas.cap_u and len(again.bands) == 4

    def test_fragment_wall_goes_through_the_same_atlas_path(self, cloud):
        from shapely.geometry import Polygon

        xyz, rgb = cloud
        wt = bake_wall_texture(xyz, rgb, WALL.vertices, FLOOR_Y, CEIL_Y, px_per_m=40)
        assert wt.atlas is not None and len(wt.atlas.bands) == 1
        assert wt.atlas.image.shape[1] == wt.image.shape[1] + CAP_SWATCH_PX
        mesh = trimesh.creation.extrude_polygon(Polygon(WALL.vertices), height=CEIL_Y - FLOOR_Y)
        mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))
        uv = wt.atlas.uv_for_faces(mesh.vertices, mesh.faces)
        assert uv.min() >= 0.0 and uv.max() <= 1.0


class TestOutlineMesh:
    """T15g readiness: a single `wall_outline` ring-band mesh textured from the
    outline atlas, keyed on geometry in glTF and in USD."""

    @pytest.fixture(scope="class")
    def outline_tex(self, tmp_path_factory):
        rng = np.random.default_rng(5)
        wall_xyz, wall_rgb = outline_cloud(rng, SQUARE, EDGE_RGB, FLOOR_Y, CEIL_Y - FLOOR_Y)
        fx, fz = rng.uniform(0, 4, 40000), rng.uniform(0, 3, 40000)
        floor_xyz = np.column_stack([fx, rng.normal(0, 0.01, 40000), fz])
        floor_rgb = np.tile([120, 90, 60], (40000, 1)).astype(np.uint8)
        xyz, rgb = np.vstack([wall_xyz, floor_xyz]), np.vstack([wall_rgb, floor_rgb])
        out = tmp_path_factory.mktemp("t15c_prep_outline")
        tex = bake_room_textures(None, [], SQUARE, FLOOR_Y, CEIL_Y, out, cloud=(xyz, rgb), px_per_m=20, wall_outline=SQUARE, wall_thickness_m=0.1)
        return tex, out

    def test_bake_room_textures_outline_entry(self, outline_tex):
        tex, out = outline_tex
        assert set(tex["walls"]) == {"outline"}
        e = tex["walls"]["outline"]
        assert e["n_edges"] == 4 and e["frame"] is None and (out / "wall_outline_texture.png").exists()
        assert 0.0 <= e["fill_fraction"] <= 1.0 and e["atlas"]["u_ranges"][0][0] == 0.0
        assert (out / "floor_mask.png").exists() and tex["floor"]["fill_fraction"] == pytest.approx(1.0 - tex["floor"]["coverage"])

    def test_gltf_wall_outline_node_textured_by_geometry(self, outline_tex):
        tex, _ = outline_tex
        scene = build_scene([], SQUARE, FLOOR_Y, CEIL_Y, [])
        scene.add_geometry(_extrude_ring_band(SQUARE, FLOOR_Y, CEIL_Y - FLOOR_Y), node_name="wall_outline")
        assert detect_outline_wall(scene, SQUARE, FLOOR_Y)
        assert apply_room_textures(scene, tex) == 2  # floor + wall_outline
        geom = scene.geometry[scene.graph["wall_outline"][1]]
        assert geom.visual.kind == "texture" and geom.visual.material.baseColorTexture is not None
        uv = np.asarray(geom.visual.uv)
        assert uv.shape == (len(geom.vertices), 2) and uv.min() >= 0 and uv.max() <= 1
        assert len(geom.vertices) == len(geom.faces) * 3  # unwelded for the per-band seams
        # Vertical faces on x = 4 sample the green (edge 1) columns of the atlas.
        atlas = tex["walls"]["outline"]["_atlas"]
        V, F = geom.vertices, geom.faces
        centroids = V[F].mean(axis=1)
        normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        vertical = np.abs(normals[:, 1]) < 1e-6 * np.linalg.norm(normals, axis=1) + 1e-9
        east = vertical & (np.abs(centroids[:, 0] - 4.0) < 0.1) & (centroids[:, 2] > 0.2) & (centroids[:, 2] < 2.8)
        u = uv[F[east]][..., 0]
        assert east.any() and u.min() >= atlas.u_ranges[1][0] - 1e-9 and u.max() <= atlas.u_ranges[1][1] + 1e-9

    def test_named_fragment_wall_node_hits_the_outline_atlas_by_geometry(self, outline_tex):
        # A single `wall_0` node that IS the closed band (T15g may keep the WallPolygon
        # list of one): not in the entries by name -> matched by geometry.
        tex, _ = outline_tex
        scene = trimesh.Scene()
        scene.add_geometry(_extrude_ring_band(SQUARE, FLOOR_Y, CEIL_Y - FLOOR_Y), node_name="wall_0")
        assert apply_room_textures(scene, tex) == 1

    def test_usd_outline_mesh_bound_by_geometry(self, outline_tex, tmp_path):
        from pxr import Usd, UsdGeom, UsdShade

        from scripts.msa.export_usd import export_presentation_usd

        tex, _ = outline_tex
        scene = build_scene([], SQUARE, FLOOR_Y, CEIL_Y, [])
        scene.add_geometry(_extrude_ring_band(SQUARE, FLOOR_Y, CEIL_Y - FLOOR_Y), node_name="wall_outline")
        out = tmp_path / "scene_presentation.usd"
        summary = export_presentation_usd(scene, FLOOR_Y, CEIL_Y, out, floor_polygon=SQUARE, textures=tex)
        # T15i: the stub is one box Mesh per outline edge (`visual/stub_edge_i`); the
        # texture pass matches every visible Structure mesh by geometry, so each edge
        # binds the outline atlas.
        stub_paths = [f"/World/Structure/wall_outline/visual/stub_edge_{i}" for i in range(len(SQUARE))]
        assert all(summary["textures"]["bound"][p] == "outline" for p in stub_paths)
        # T15h: the band collider is a group of invisible `edge_i` Cube boxes (not
        # a Mesh), so the texture pass never binds anything under it - it may be
        # skipped explicitly (pre-T15h Mesh collider) or simply never visited.
        assert not any(p.startswith("/World/Structure/wall_outline_collider") for p in summary["textures"]["bound"])
        stage = Usd.Stage.Open(out.as_posix())
        collider_root = stage.GetPrimAtPath("/World/Structure/wall_outline_collider")
        assert collider_root and all(
            not UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0] for p in Usd.PrimRange(collider_root)
        )
        for stub_path in stub_paths:
            stub = stage.GetPrimAtPath(stub_path)
            mat, _ = UsdShade.MaterialBindingAPI(stub).ComputeBoundMaterial()
            assert mat and mat.GetPath().pathString == "/World/Looks/wall_outline_material"
            st = UsdGeom.PrimvarsAPI(stub).GetPrimvar("st")
            vals = np.asarray(st.Get())
            assert st.GetInterpolation() == UsdGeom.Tokens.faceVarying
            assert len(vals) == len(UsdGeom.Mesh(stub).GetFaceVertexIndicesAttr().Get())
            assert vals.min() >= 0.0 and vals.max() <= 1.0
        assert (tmp_path / "wall_outline_texture.png").exists() and (tmp_path / "floor_texture.png").exists()


class TestUsdBinding:
    @pytest.fixture(scope="class")
    def tex(self, cloud, tmp_path_factory):
        xyz, rgb = cloud
        return bake_room_textures(None, [WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, tmp_path_factory.mktemp("t15c_prep_tex"), cloud=(xyz, rgb), px_per_m=40)

    def test_export_usd_binds_floor_and_walls(self, tex, tmp_path):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        from scripts.msa.export_usd import _extrude_world_polygon_xz, export_usd

        floor_mesh = _extrude_world_polygon_xz(np.asarray(FLOOR_POLY), FLOOR_Y - 0.1, 0.1)
        out = tmp_path / "scene.usd"
        export_usd([WALL], floor_mesh, FLOOR_Y, CEIL_Y, [], out, floor_polygon=FLOOR_POLY, textures=tex)
        stage = Usd.Stage.Open(out.as_posix())
        for path, mat_name in (("/World/Floor", "floor_material"), ("/World/Structure/wall_0", "wall_0_material")):
            prim = stage.GetPrimAtPath(path)
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            assert mat and mat.GetPath().pathString == f"/World/Looks/{mat_name}", path
            st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            vals = np.asarray(st.Get())
            assert st.GetInterpolation() == UsdGeom.Tokens.faceVarying
            assert len(vals) == len(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get())
            assert vals.min() >= 0.0 and vals.max() <= 1.0
            # Shader network: PreviewSurface.diffuseColor <- UsdUVTexture(file=./png).rgb, st <- PrimvarReader(st)
            shaders = {UsdShade.Shader(p).GetIdAttr().Get(): UsdShade.Shader(p) for p in Usd.PrimRange(mat.GetPrim()) if p.IsA(UsdShade.Shader)}
            assert set(shaders) == {"UsdPreviewSurface", "UsdUVTexture", "UsdPrimvarReader_float2"}
            png = shaders["UsdUVTexture"].GetInput("file").Get().path
            assert png.startswith("./") and (tmp_path / png[2:]).exists()
            assert shaders["UsdPrimvarReader_float2"].GetInput("varname").Get() == "st"
            assert shaders["UsdPreviewSurface"].GetInput("diffuseColor").HasConnectedSource()
            assert shaders["UsdUVTexture"].GetInput("st").HasConnectedSource()
        # Physics untouched: the wall keeps its collider API.
        assert stage.GetPrimAtPath("/World/Structure/wall_0").HasAPI(UsdPhysics.CollisionAPI)

    def test_presentation_binds_visible_stubs_not_colliders(self, tex, tmp_path):
        from pxr import Usd, UsdShade

        from scripts.msa.export_usd import export_presentation_usd

        scene = build_scene([WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, [])
        out = tmp_path / "scene_presentation.usd"
        summary = export_presentation_usd(scene, FLOOR_Y, CEIL_Y, out, floor_polygon=FLOOR_POLY, textures=tex)
        assert set(summary["textures"]["bound"]) == {"/World/Floor", "/World/Structure/wall_0/visual/stub"}
        assert summary["textures"]["skipped"] == ["/World/Structure/wall_0_collider"]
        stage = Usd.Stage.Open(out.as_posix())
        collider = stage.GetPrimAtPath("/World/Structure/wall_0_collider")
        mat, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial()
        assert not mat
        assert stage.GetRootLayer().customLayerData["cloudeye:msaPresentationVersion"] == 1

    def test_apply_usd_textures_is_idempotent_on_reopen(self, tex, tmp_path):
        from pxr import Usd, UsdShade

        from scripts.msa.export_usd import _extrude_world_polygon_xz, export_usd

        out = tmp_path / "scene.usd"
        export_usd([WALL], _extrude_world_polygon_xz(np.asarray(FLOOR_POLY), FLOOR_Y - 0.1, 0.1), FLOOR_Y, CEIL_Y, [], out, floor_polygon=FLOOR_POLY)
        stage = Usd.Stage.Open(out.as_posix())
        s1 = apply_usd_textures(stage, tex, tmp_path)
        s2 = apply_usd_textures(stage, tex, tmp_path)
        assert s1["bound"] == s2["bound"]
        stage.GetRootLayer().Save()
        again = Usd.Stage.Open(out.as_posix())
        mat, _ = UsdShade.MaterialBindingAPI(again.GetPrimAtPath("/World/Floor")).ComputeBoundMaterial()
        assert mat


class TestIo:
    def test_textures_json_roundtrip_applies_without_images(self, cloud, tmp_path):
        xyz, rgb = cloud
        tex = bake_room_textures(None, [WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, tmp_path, cloud=(xyz, rgb), px_per_m=40)
        summary = textures_summary(tex)
        assert "image" not in summary["floor"] and "_atlas" not in summary["walls"]["0"] and "atlas" in summary["walls"]["0"]
        (tmp_path / "textures.json").write_text(json.dumps(summary))
        loaded = load_textures_json(tmp_path / "textures.json")
        assert set(loaded["walls"]) == {0} and loaded["floor"]["png"].endswith("floor_texture.png")
        scene = build_scene([WALL], FLOOR_POLY, FLOOR_Y, CEIL_Y, [])
        assert apply_room_textures(scene, loaded) == 2

    def test_ply_reader_subsamples_once(self, tmp_path):
        from scripts.msa.ply_io import read_ply_xyz_rgb

        n = 1000
        xyz = np.arange(n * 3, dtype="<f8").reshape(n, 3)
        rgb = (np.arange(n * 3) % 256).astype("u1").reshape(n, 3)
        dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.empty(n, dtype=dtype)
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["r"], rec["g"], rec["b"] = rgb.T
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        p = tmp_path / "c.ply"
        p.write_bytes(header.encode("ascii") + rec.tobytes())
        full_xyz, full_rgb = read_ply_xyz_rgb(p)
        assert full_xyz.shape == (n, 3) and np.array_equal(full_xyz, xyz) and np.array_equal(full_rgb, rgb)
        sub_xyz, sub_rgb = read_ply_xyz_rgb(p, max_points=100, seed=0)
        assert sub_xyz.shape == (100, 3) and sub_rgb.shape == (100, 3)
        rows = (sub_xyz[:, 0] / 3).astype(int)
        assert len(set(rows.tolist())) == 100 and np.all(np.diff(rows) > 0)  # distinct, file order
        assert np.array_equal(sub_rgb, rgb[rows])
        again, _ = read_ply_xyz_rgb(p, max_points=100, seed=0)
        assert np.array_equal(again, sub_xyz)  # deterministic
        assert read_ply_xyz_rgb(p, max_points=5000)[0].shape == (n, 3)  # cap above N = everything
