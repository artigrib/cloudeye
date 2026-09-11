"""T15b grid-frame regression tests (docs/DECISIONS.md, "T15b" entry).

`occupancy.npy` is indexed `[ix, iz]` - axis 0 = X, axis 1 = Z
(`gpu/stage_occupancy.py` allocates `np.full((nx, nz))`, `occupancy_meta.json`
`width` = nx = shape[0], `height` = nz = shape[1]; `app/services/pathfinding.py`
and `demo/record_isaac.py` read `grid[ix, iz]`). Until T15b `scripts/msa/`
read the grid transposed (row -> z, col -> x), so every wall/room polygon was
reflected across the x = z diagonal relative to the object footprints (built
from world-frame `scene_objects` point clouds) and the aligned point cloud.

  - Synthetic: a single OBSTACLE column at ix=5 must become a wall at world
    x = origin_x + 5*res (not z); footprint rasterization and gap measurement
    points must agree with the same convention.
  - Hero (skipped when the read-only scratch fixture is absent): floor-band
    cloud points must land inside the un-rotated room polygon (>= 0.75; was
    0.375 transposed) and every non-degenerate wall polygon must have > 1000
    cloud points within 0.15 m (wall_1 had 0 transposed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import Polygon

from scripts.msa import bootstrap as bs
from scripts.msa.gaps import ObstacleSource, compute_gaps
from scripts.msa.geometry import FREE, OBSTACLE, UNKNOWN, _cell_to_world, extract_room_polygon, extract_wall_polygons

HERO_SCENE_DIR = Path("var/scratch/run-20260906/hero_scene_input_t3")
HERO_CLOUD = Path("var/scratch/hero-frozen/own_0901_173903/scene/aligned_room.ply")
RES = 0.05
ORIGIN_X, ORIGIN_Z = 10.0, -20.0  # deliberately different so an x/z swap is unmistakable


class TestSyntheticGridFrame:
    def test_cell_to_world_axis0_is_x(self):
        assert _cell_to_world(5, 0, RES, ORIGIN_X, ORIGIN_Z) == (ORIGIN_X + 5 * RES, ORIGIN_Z)
        assert _cell_to_world(0, 7, RES, ORIGIN_X, ORIGIN_Z) == (ORIGIN_X, ORIGIN_Z + 7 * RES)

    def test_obstacle_column_at_ix5_becomes_wall_at_world_x(self):
        # nx=20 (X), nz=40 (Z): non-square so shape alone can't hide a transpose.
        occ = np.full((20, 40), FREE, dtype=np.uint8)
        occ[5, :] = OBSTACLE  # one full column at ix=5 -> a wall along Z at x = ox + 5*res .. ox + 6*res
        walls, dropped = extract_wall_polygons(occ, RES, ORIGIN_X, ORIGIN_Z, min_component_area_m2=0.01)
        assert len(walls) == 1 and dropped == []
        xs = [x for x, _ in walls[0].vertices]
        zs = [z for _, z in walls[0].vertices]
        assert min(xs) == pytest.approx(ORIGIN_X + 5 * RES) and max(xs) == pytest.approx(ORIGIN_X + 6 * RES)
        assert min(zs) == pytest.approx(ORIGIN_Z) and max(zs) == pytest.approx(ORIGIN_Z + 40 * RES)

    def test_room_polygon_spans_x_by_shape0(self):
        occ = np.full((20, 40), UNKNOWN, dtype=np.uint8)
        occ[2:12, 5:35] = FREE  # 10 cells in X, 30 cells in Z
        poly = extract_room_polygon(occ, RES, ORIGIN_X, ORIGIN_Z, [])
        xs = [x for x, _ in poly]
        zs = [z for _, z in poly]
        assert max(xs) - min(xs) == pytest.approx(10 * RES)
        assert max(zs) - min(zs) == pytest.approx(30 * RES)
        assert min(xs) == pytest.approx(ORIGIN_X + 2 * RES) and min(zs) == pytest.approx(ORIGIN_Z + 5 * RES)

    def test_footprint_rasterization_matches_grid_frame(self):
        # A 0.2 m (X) by 0.6 m (Z) box centred on cell (ix=8, iz=20): 4 x 12 cells.
        obj = {
            "center_xy": (ORIGIN_X + 8 * RES, ORIGIN_Z + 20 * RES),
            "size_uv": (0.2, 0.6),
            "angle_rad": 0.0,
        }
        mask = bs.rasterize_object_footprint(obj, (20, 40), RES, ORIGIN_X, ORIGIN_Z)
        ixs, izs = np.where(mask)
        assert ixs.min() == 6 and ixs.max() == 9
        assert izs.min() == 14 and izs.max() == 25

    def test_gap_measurement_point_uses_grid_frame(self):
        # Two walls at ix 0..1 and ix 8..9 -> the gap runs along Z between them; its
        # measurement point must sit at x ~ ORIGIN_X + 5*RES, inside z's range.
        passable = np.full((10, 30), True)
        a = np.zeros((10, 30), dtype=bool)
        b = np.zeros((10, 30), dtype=bool)
        a[:2, :] = True
        b[8:, :] = True
        gaps = compute_gaps(passable, [ObstacleSource("a", "wall", a), ObstacleSource("b", "wall", b)], RES, ORIGIN_X, ORIGIN_Z)
        assert len(gaps) == 1
        px, pz = gaps[0].measurement_point_xy
        assert ORIGIN_X + 2 * RES <= px <= ORIGIN_X + 8 * RES
        assert ORIGIN_Z <= pz <= ORIGIN_Z + 30 * RES

    def test_run_bootstrap_rejects_transposed_grid(self, tmp_path):
        occ = np.full((20, 40), FREE, dtype=np.uint8)
        np.save(tmp_path / "occupancy.npy", occ.T)  # (40, 20) but meta says width=20, height=40
        (tmp_path / "occupancy_meta.json").write_text(json.dumps({"resolution": RES, "origin_x": 0.0, "origin_z": 0.0, "width": 20, "height": 40}))
        (tmp_path / "scene_meta.json").write_text(json.dumps({"floor_y": 0.0, "ceiling_y": 2.5}))
        with pytest.raises(ValueError, match=r"\[ix, iz\]"):
            bs.run_bootstrap(tmp_path, tmp_path / "out", object_inputs=[])


@pytest.mark.skipif(not (HERO_SCENE_DIR.exists() and HERO_CLOUD.exists()), reason="hero scratch fixture not on this machine")
class TestHeroCloudAgreesWithGridFrame:
    @pytest.fixture(scope="class")
    def hero(self):
        from scripts.msa.ply_io import read_ply_xyz_rgb

        occ = np.load(HERO_SCENE_DIR / "occupancy.npy")
        meta = json.loads((HERO_SCENE_DIR / "occupancy_meta.json").read_text())
        assert occ.shape == (meta["width"], meta["height"]), "hero grid is [ix, iz] by construction"
        res, ox, oz = meta["resolution"], meta["origin_x"], meta["origin_z"]
        floor_y = json.loads((HERO_SCENE_DIR / "scene_meta.json").read_text())["floor_y"]

        xyz, _ = read_ply_xyz_rgb(HERO_CLOUD)  # loaded once for the class
        band = xyz[np.abs(xyz[:, 1] - floor_y) <= 0.15]
        rng = np.random.default_rng(0)
        if len(band) > 200_000:
            band = band[rng.choice(len(band), 200_000, replace=False)]
        if len(xyz) > 2_000_000:
            xyz = xyz[rng.choice(len(xyz), 2_000_000, replace=False)]

        walls, _ = extract_wall_polygons(occ, res, ox, oz)
        inputs = bs.load_object_inputs_from_scene_dir(HERO_SCENE_DIR)
        objects, _ = bs.compute_object_footprints(inputs, floor_y, walls=walls)
        objects, _ = bs._filter_small_isolated_objects(objects, bs._walls_union(walls))
        masks, _ = bs._floor_standing_footprint_masks(objects, floor_y, occ.shape, res, ox, oz)
        room = extract_room_polygon(occ, res, ox, oz, masks)  # un-rotated: same frame as the cloud
        return walls, room, band, xyz

    def test_floor_band_points_fall_inside_unrotated_room_polygon(self, hero):
        _, room, band, _ = hero
        poly = Polygon(room)
        if not poly.is_valid:
            poly = poly.buffer(0)
        frac = float(contains_xy(poly, band[:, 0], band[:, 2]).mean())
        assert frac >= 0.75, f"only {frac:.3f} of floor-band cloud points inside the room polygon (transposed read gave 0.375)"

    def test_every_wall_polygon_is_supported_by_cloud_points(self, hero):
        walls, _, _, xyz = hero
        assert walls, "hero has walls"
        for i, w in enumerate(walls):
            poly = Polygon(w.vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)
            assert not poly.is_empty and poly.area > 0, f"wall_{i} ({w.area_m2:.2f} m^2 of cells) exported as a degenerate polygon"
            n = int(contains_xy(poly.buffer(0.15), xyz[:, 0], xyz[:, 2]).sum())
            assert n > 1000, f"wall_{i} has only {n} cloud points within 0.15 m (transposed read gave 0 for wall_1)"
