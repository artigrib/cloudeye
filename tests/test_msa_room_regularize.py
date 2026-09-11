"""T15g: Manhattan regularization of the room outline
(`geometry.regularize_room_polygon`), the union -> close -> snap -> yaw order in
`bootstrap.run_bootstrap` (grid padded so footprints beyond the occupancy grid
still join the room; every floor-standing footprint >= 95% inside the exported
polygon), and the single wall band (`geometry.room_wall_band`, GLB node
`wall_outline`, USD `/World/Structure/wall_outline`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from scripts.msa.bootstrap import (
    FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M,
    ObjectFootprintInput,
    _pad_grid_to_cover_footprints,
    run_bootstrap,
)
from scripts.msa.geometry import (
    FREE,
    OBSTACLE,
    UNKNOWN,
    WALL_THICKNESS_M,
    compute_room_polygon_yaw,
    regularize_room_polygon,
    room_wall_band,
)
from tests.test_msa_room_polygon import _axis_residual_deg, _build_rotated_room_grid, _rect_hull, _rot, _write_scene

HERO_SCENE_DIR = Path("var/scratch/run-20260906/hero_scene_input_t3")
RESOLUTION = 0.05


def _interior_angles_deg(ring: list[tuple[float, float]]) -> list[float]:
    pts = list(ring)
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    out = []
    for i in range(n):
        (x0, z0), (x1, z1), (x2, z2) = pts[i - 1], pts[i], pts[(i + 1) % n]
        ax, az, bx, bz = x0 - x1, z0 - z1, x2 - x1, z2 - z1
        out.append(math.degrees(math.atan2(ax * bz - az * bx, ax * bx + az * bz)) % 360.0)
    return out


def _assert_manhattan(ring, tol_deg: float = 0.5) -> None:
    for a in _interior_angles_deg(ring):
        assert min(abs(a - 90.0), abs(a - 270.0)) <= tol_deg, f"interior angle {a:.3f} is not 90/270"


def _rotated(ring, deg):
    return [_rot(x, z, deg) for x, z in ring]


class TestRegularizeRoomPolygon:
    # 5 m x 3 m room; a 0.2 m deep x 0.4 m wide notch in the bottom edge; a 1.0 m
    # wide x 0.9 m deep doorway appendix on the right edge. CCW in (x, z).
    ROOM = [
        (0.0, 0.0), (2.0, 0.0), (2.0, 0.2), (2.4, 0.2), (2.4, 0.0), (5.0, 0.0),  # bottom edge with notch
        (5.0, 1.0), (5.9, 1.0), (5.9, 2.0), (5.0, 2.0),  # doorway appendix
        (5.0, 3.0), (0.0, 3.0),
    ]

    def test_notch_removed_appendix_kept_on_rotated_room(self):
        deg = 25.0
        rotated = _rotated(self.ROOM, deg)
        # the yaw correction that re-aligns it (compute_room_polygon_yaw's sign convention)
        yaw = compute_room_polygon_yaw(rotated)
        assert _axis_residual_deg(yaw["long_axis_angle_deg"] - deg) < 1e-6  # long axis found at +25 deg
        out = regularize_room_polygon(rotated, yaw["correction_rad"], notch_m=0.30)
        assert len(out) == 8, out
        # rectilinear in the aligned frame ...
        aligned = [_rot(x, z, -deg) for x, z in out]
        _assert_manhattan(aligned)
        # ... and still rectilinear (rotated by exactly -deg) in the input frame
        _assert_manhattan(out)
        poly = Polygon(aligned)
        assert poly.is_valid
        # notch gone (one straight bottom edge), appendix kept (x reaches 5.9).
        # Collapsing the notch merges the two bottom runs at their length-weighted
        # coordinate, so the healed edge sits at most depth*width/length =
        # 0.2*0.4/5.0 = 1.6 cm inside the original line - never further.
        xs = [p[0] for p in aligned]
        zs = [p[1] for p in aligned]
        assert max(xs) == pytest.approx(5.9, abs=1e-6)
        assert 0.0 - 1e-9 <= min(zs) <= 0.2 * 0.4 / 5.0 + 1e-9
        assert poly.area == pytest.approx(15.0 + 0.9, abs=0.1)

    def test_already_rectilinear_axis_aligned_is_unchanged_up_to_order(self):
        rect = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        out = regularize_room_polygon(rect, 0.0)
        assert len(out) == 4
        assert Polygon(out).symmetric_difference(Polygon(rect)).area == pytest.approx(0.0, abs=1e-9)

    def test_staircase_edge_collapses_to_one_edge_within_the_step_band(self):
        # a rasterized 1-cell staircase along the bottom: every 5 cm riser is a
        # sub-notch run and collapses; the one remaining bottom edge lies inside
        # the staircase's own band (collapse merges neighbours at their
        # length-weighted coordinate - a spike cannot drag a 5 m wall with it).
        stairs = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.05), (2.0, 0.05), (2.0, 0.0), (3.0, 0.0), (3.0, 0.05), (4.0, 0.05), (4.0, 0.0), (5.0, 0.0)]
        ring = stairs + [(5.0, 3.0), (0.0, 3.0)]
        out = regularize_room_polygon(ring, 0.0, notch_m=0.30)
        assert len(out) == 4
        assert 0.0 - 1e-9 <= min(p[1] for p in out) <= 0.05 + 1e-9

    def test_near_straight_run_snaps_to_its_outer_line(self):
        # DP-simplified rotated edge: a few vertices wobbling within 3 cm of a
        # line, all with the same dominant axis -> ONE run at its outward extreme
        # (so the room stays a superset of the raw outline along it).
        bottom = [(0.0, 0.0), (1.2, -0.03), (2.6, 0.01), (3.9, -0.02), (5.0, 0.0)]
        ring = bottom + [(5.0, 3.0), (0.0, 3.0)]
        out = regularize_room_polygon(ring, 0.0, notch_m=0.30)
        assert len(out) == 4
        assert min(p[1] for p in out) == pytest.approx(-0.03, abs=1e-9)

    def test_degenerate_input_returned_unchanged(self):
        tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        assert regularize_room_polygon(tri, 0.3) == tri

    def test_deterministic(self):
        rotated = _rotated(self.ROOM, 17.0)
        yaw = compute_room_polygon_yaw(rotated)["correction_rad"]
        assert regularize_room_polygon(rotated, yaw) == regularize_room_polygon(rotated, yaw)


class TestRoomWallBand:
    def test_band_is_a_ring_of_the_given_thickness_outside_the_room(self):
        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        band = room_wall_band(room, 0.1)
        assert band.geom_type == "Polygon" and len(band.interiors) == 1
        assert band.area == pytest.approx((4.2 * 3.2) - 12.0, abs=1e-9)
        assert band.intersection(Polygon(room)).area == pytest.approx(0.0, abs=1e-9)
        # mitre joins keep it Manhattan (a rounded corner would add many vertices)
        assert len(band.exterior.coords) - 1 == 4

    def test_none_for_unusable_room(self):
        assert room_wall_band(None) is None
        assert room_wall_band([(0.0, 0.0), (1.0, 0.0)]) is None


class TestGridPadding:
    def test_footprint_beyond_the_grid_gets_unknown_padding(self):
        occ = np.full((40, 40), UNKNOWN, dtype=np.uint8)
        occ[10:30, 10:30] = FREE
        obj = {"id": "bed", "label": "bed", "center_xy": (2.3, 1.0), "size_uv": (1.0, 0.6), "angle_rad": 0.0, "bbox_min_y": 0.0}
        padded, ox, oz, info = _pad_grid_to_cover_footprints(occ, RESOLUTION, 0.0, 0.0, [obj], margin_cells=2)
        # grid spans x in [0, 2.0]; the bed reaches x = 2.8 -> 16 cells + 2 margin on the +x side only
        assert info["cells"] == [0, 18, 0, 0] and info["capped"] is False
        assert padded.shape == (58, 40) and ox == 0.0 and oz == 0.0
        assert (padded[40:] == UNKNOWN).all() and (padded[:40] == occ).all()

    def test_no_objects_no_padding(self):
        occ = np.full((10, 10), FREE, dtype=np.uint8)
        padded, ox, oz, info = _pad_grid_to_cover_footprints(occ, RESOLUTION, 1.0, 2.0, [])
        assert padded is occ and (ox, oz) == (1.0, 2.0) and info["cells"] == [0, 0, 0, 0]


class TestBootstrapOrderOfOperations:
    """Synthetic 25 deg room whose floor-standing bed pokes past BOTH the free
    region and the occupancy grid's own edge: it must end up inside the
    exported (regularized, rotated) room polygon."""

    ANGLE_DEG = 25.0

    def _run(self, tmp_path):
        occ, (cx, cz) = _build_rotated_room_grid(self.ANGLE_DEG, rect_size=(3.0, 1.6), grid_size=110)
        # shrink the grid on the +x side so the bed below hangs off it
        occ = occ[:88, :]
        scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
        _write_scene(scene_dir, occ)
        bed_center = (cx + _rot(1.3, 0.0, self.ANGLE_DEG)[0], cz + _rot(1.3, 0.0, self.ANGLE_DEG)[1])
        bed = ObjectFootprintInput(
            id="bed_0", label="bed", hull_xz=_rect_hull(bed_center, (1.4, 0.8), self.ANGLE_DEG),
            bbox_min=[0.0, 0.03, 0.0], bbox_max=[0.0, 0.6, 0.0], color_rgb=(100, 100, 100),
        )
        assert max(bed.hull_xz[:, 0]) > 88 * RESOLUTION, "test setup: the bed must reach past the grid's +x edge"
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[bed])
        meta = json.loads((out_dir / "scene_meta.json").read_text())
        return report, meta, out_dir

    def test_floor_standing_footprint_inside_and_polygon_manhattan(self, tmp_path):
        report, meta, _ = self._run(tmp_path)
        assert report["room_grid_padding_cells"][1] > 0  # padded on +x
        assert meta["room_polygon_regularized"] is True
        assert report["floor_standing_inside_fraction_min"] >= 0.95
        assert meta["floor_standing_inside_fractions"]["bed_0"] >= 0.95
        room = [tuple(p) for p in meta["room_polygon"]]
        _assert_manhattan(room)
        # T15a's acceptance test still holds on the snapped polygon
        assert _axis_residual_deg(compute_room_polygon_yaw(room)["long_axis_angle_deg"]) < 2.0
        assert report["yaw_correction_deg"] == pytest.approx(-self.ANGLE_DEG, abs=2.0)

    def test_glb_and_usd_carry_one_wall_band(self, tmp_path):
        from pxr import Usd, UsdPhysics

        report, meta, out_dir = self._run(tmp_path)
        assert meta["wall_band_present"] is True and meta["wall_band_thickness_m"] == WALL_THICKNESS_M
        assert meta["wall_height_m"] == pytest.approx(2.5)
        scene = trimesh.load(str(out_dir / "scene.glb"))
        nodes = set(scene.graph.nodes)
        assert "wall_outline" in nodes and not any(n.startswith("wall_") and n != "wall_outline" for n in nodes)
        # the band hugs the room from the outside: its bottom ring's bbox is the
        # room bbox grown by the thickness
        _, geom_name = scene.graph["wall_outline"]
        verts = np.asarray(scene.geometry[geom_name].vertices)
        room = np.asarray(meta["room_polygon"])
        assert verts[:, 0].min() == pytest.approx(room[:, 0].min() - WALL_THICKNESS_M, abs=1e-6)
        assert verts[:, 0].max() == pytest.approx(room[:, 0].max() + WALL_THICKNESS_M, abs=1e-6)
        assert verts[:, 1].max() == pytest.approx(2.5, abs=1e-6)
        stage = Usd.Stage.Open(str(out_dir / "scene.usd"))
        # T15h: the band is the visual `wall_outline/visual/band`; its collider is one
        # box per outline edge under `wall_outline_collider` (a convexHull of the ring
        # would fill the room - T15e). No per-fragment wall_i anywhere.
        structure = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith("/World/Structure/")]
        assert "/World/Structure/wall_outline/visual/band" in structure
        edge_boxes = [p for p in structure if p.startswith("/World/Structure/wall_outline_collider/edge_")]
        assert len(edge_boxes) == len(meta["room_polygon"])
        assert not any(p.startswith("/World/Structure/wall_") and not p.startswith("/World/Structure/wall_outline") for p in structure)
        assert not stage.GetPrimAtPath("/World/Structure/wall_outline/visual/band").HasAPI(UsdPhysics.CollisionAPI)
        for p in edge_boxes:
            assert stage.GetPrimAtPath(p).HasAPI(UsdPhysics.CollisionAPI)
        # raw fragments kept as data
        assert report["n_walls"] >= 1
        assert any(str(p.GetPath()).startswith("/World/Plan/wall_") for p in stage.Traverse())


@pytest.mark.skipif(not HERO_SCENE_DIR.exists(), reason=f"hero fixture {HERO_SCENE_DIR} not on this machine")
class TestHeroRegularized:
    @pytest.fixture(scope="class")
    def hero(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("hero_g")
        report = run_bootstrap(HERO_SCENE_DIR, out_dir, object_inputs=None)
        return report, json.loads((out_dir / "scene_meta.json").read_text())

    def test_polygon_has_at_most_16_manhattan_vertices(self, hero):
        _, meta = hero
        room = [tuple(p) for p in meta["room_polygon"]]
        assert 4 <= len(room) <= 16, len(room)
        _assert_manhattan(room)
        assert _axis_residual_deg(compute_room_polygon_yaw(room)["long_axis_angle_deg"]) < 2.0

    def test_every_floor_standing_footprint_is_inside(self, hero):
        report, meta = hero
        assert report["floor_standing_inside_fraction_min"] >= 0.95
        fractions = meta["floor_standing_inside_fractions"]
        assert set(fractions) == set(meta["room_polygon_floor_standing_object_ids"])
        assert all(f >= 0.95 for f in fractions.values()), fractions
        # the two footprints that used to hang off the occupancy grid
        assert fractions["curtain_0"] >= 0.95 and fractions["bed_0"] >= 0.95
        assert report["room_grid_padding_cells"] != [0, 0, 0, 0]
