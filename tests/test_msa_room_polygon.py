"""T15a room polygon (see docs/DECISIONS.md "T15a"): `geometry.extract_room_polygon`
= outer contour of union(largest FREE component, floor-standing object
footprints), morphologically closed - the single room reference `bootstrap.
run_bootstrap` feeds to yaw estimation, the outside-room drop/clip, the floor
plate/meshes and the exported `floor_polygon`/`room_polygon`.

Three groups:
  - `TestBuildRoomMask` / `TestFloorStandingSelection`: the construction itself
    (footprint poking outside FREE is unioned in, a 1-2 cell notch is closed,
    an elevated object is NOT unioned in, holes are filled, UNKNOWN never counts).
  - `TestRotatedRoomIsAxisAlignedAfterBootstrap`: an *unpatched* `run_bootstrap`
    over a synthetic rotated-room occupancy grid - the rotated `room_polygon`
    written to scene_meta.json must have a min-area-rect long axis within 2
    degrees of an axis (undirected: distance to the nearest of 0/90).
  - `TestHeroScene`: the same on the real hero inputs, skipped when the
    read-only scratch fixture isn't on this machine.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.msa.bootstrap import (
    FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M,
    ObjectFootprintInput,
    _floor_standing_footprint_masks,
    rasterize_object_footprint,
    run_bootstrap,
)
from scripts.msa.geometry import (
    FREE,
    OBSTACLE,
    ROOM_POLYGON_METHOD,
    UNKNOWN,
    build_room_mask,
    compute_room_polygon_yaw,
    extract_room_polygon,
    rotate_point_xz,
)

HERO_SCENE_DIR = Path("var/scratch/run-20260906/hero_scene_input_t3")
RESOLUTION = 0.05


def _axis_residual_deg(angle_deg: float) -> float:
    """Undirected distance of a line angle to the nearest of 0/90 degrees."""
    a = abs(angle_deg) % 90.0
    return min(a, 90.0 - a)


def _obj(obj_id: str, center_xy, size_uv, *, angle_rad: float = 0.0, bbox_min_y: float = 0.0) -> dict:
    """Minimal object dict in the shape `compute_object_footprints` emits."""
    return {
        "id": obj_id,
        "label": obj_id,
        "center_xy": tuple(center_xy),
        "size_uv": tuple(size_uv),
        "angle_rad": angle_rad,
        "height": 0.8,
        "bbox_min_y": bbox_min_y,
        "color_rgb": (120, 120, 120),
        "hull_xz": [],
    }


def _free_block_grid(size: int = 40, lo: int = 10, hi: int = 30) -> np.ndarray:
    """UNKNOWN grid (touching every edge, like gpu/stage_occupancy.py's
    bounding-box grids always do) with one FREE block occ[lo:hi, lo:hi]."""
    occ = np.full((size, size), UNKNOWN, dtype=np.uint8)
    occ[lo:hi, lo:hi] = FREE
    return occ


class TestBuildRoomMask:
    def test_plain_free_block_is_returned_exactly_and_unknown_halo_is_excluded(self):
        occ = _free_block_grid()
        room = build_room_mask(occ, [])
        assert room is not None
        assert room.shape == occ.shape
        # Closing/filling a solid rectangle is the identity; nothing from the
        # UNKNOWN halo (which touches the grid's edges) is pulled in.
        np.testing.assert_array_equal(room, occ == FREE)

    def test_floor_standing_footprint_poking_outside_free_is_unioned_in(self):
        occ = _free_block_grid()
        # FREE block spans x in [0.5, 1.5]; this 0.4m x 0.4m footprint is
        # centered on the x=1.5 edge, so half of it (4 cells) lies outside FREE.
        # The grid is [ix, iz] (axis 0 = X), so those 4 cells are ix 30..33.
        obj = _obj("wardrobe", (1.5, 1.0), (0.4, 0.4), bbox_min_y=0.02)
        mask = rasterize_object_footprint(obj, occ.shape, RESOLUTION, 0.0, 0.0)
        assert mask.sum() == 64 and mask[30:34, :].sum() == 32, "test setup: 4 x-columns of the footprint lie outside FREE"

        room = build_room_mask(occ, [mask])
        assert room[mask].all(), "every footprint cell must be room"
        assert room[32, 20], "a cell outside the FREE component but inside the footprint must be room"
        # Only the footprint was added - nothing else from the halo.
        assert room.sum() == (occ == FREE).sum() + 32

        polygon = extract_room_polygon(occ, RESOLUTION, 0.0, 0.0, [mask])
        assert polygon is not None
        assert max(x for x, _ in polygon) == pytest.approx(1.7, abs=1e-9), "polygon must reach the footprint's outer edge"

    def test_elevated_object_is_not_unioned_in(self):
        occ = _free_block_grid()
        elevated = _obj("wall_tv", (1.5, 1.0), (0.4, 0.4), bbox_min_y=0.5)  # bbox bottom 0.5m above floor
        masks, ids = _floor_standing_footprint_masks([elevated], 0.0, occ.shape, RESOLUTION, 0.0, 0.0)
        assert masks == [] and ids == []

        polygon = extract_room_polygon(occ, RESOLUTION, 0.0, 0.0, masks)
        assert polygon is not None
        assert max(x for x, _ in polygon) == pytest.approx(1.5, abs=1e-9), "room must stop at the FREE edge, not the elevated footprint"
        room = build_room_mask(occ, masks)
        assert not room[32, 20]

    def test_one_and_two_cell_notches_are_closed(self):
        occ = _free_block_grid()
        occ[15:17, 28:30] = OBSTACLE  # 2x2 notch bitten out of the right edge
        occ[24, 20:30] = OBSTACLE  # 1-cell-wide slit running in from the right edge
        occ[10, 12] = UNKNOWN  # single unscanned cell on the top edge

        room = build_room_mask(occ, [])
        assert room[15:17, 28:30].all(), "2-cell notch must be closed"
        assert room[24, 20:30].all(), "1-cell slit must be closed"
        assert room[10, 12], "single-cell notch must be closed"
        np.testing.assert_array_equal(room, _free_block_grid() == FREE)

    def test_interior_hole_larger_than_closing_reach_is_filled(self):
        occ = _free_block_grid()
        occ[16:24, 16:24] = OBSTACLE  # 8x8 island: closing (reach 2) leaves its 4x4 core; fill_holes must take it
        room = build_room_mask(occ, [])
        assert room[16:24, 16:24].all()
        np.testing.assert_array_equal(room, _free_block_grid() == FREE)

    def test_largest_free_component_wins_and_footprint_can_only_join_it(self):
        occ = _free_block_grid()
        occ[2:6, 2:6] = FREE  # separate 4x4 free islet far away in the halo
        room = build_room_mask(occ, [])
        assert not room[2:6, 2:6].any(), "a disconnected smaller FREE component is not the room"
        assert room[10:30, 10:30].all()

    def test_no_free_cells_returns_none(self):
        occ = np.full((10, 10), UNKNOWN, dtype=np.uint8)
        assert build_room_mask(occ, []) is None
        assert extract_room_polygon(occ, RESOLUTION, 0.0, 0.0, []) is None

    def test_mask_shape_mismatch_raises(self):
        occ = _free_block_grid()
        with pytest.raises(ValueError):
            build_room_mask(occ, [np.zeros((5, 5), dtype=bool)])

    def test_deterministic(self):
        occ = _free_block_grid()
        obj = _obj("bed", (1.4, 0.9), (0.6, 0.4), angle_rad=0.3, bbox_min_y=0.03)
        mask = rasterize_object_footprint(obj, occ.shape, RESOLUTION, 0.0, 0.0)
        assert extract_room_polygon(occ, RESOLUTION, 0.0, 0.0, [mask]) == extract_room_polygon(occ, RESOLUTION, 0.0, 0.0, [mask])


class TestFloorStandingSelection:
    def test_threshold_is_relative_to_floor_y(self):
        objs = [
            _obj("on_floor", (1.0, 1.0), (0.2, 0.2), bbox_min_y=0.0),
            _obj("just_under", (1.0, 1.0), (0.2, 0.2), bbox_min_y=FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M - 0.01),
            _obj("at_threshold", (1.0, 1.0), (0.2, 0.2), bbox_min_y=FLOOR_STANDING_MAX_BBOX_MIN_ABOVE_FLOOR_M),
            _obj("elevated", (1.0, 1.0), (0.2, 0.2), bbox_min_y=0.5),
        ]
        masks, ids = _floor_standing_footprint_masks(objs, 0.0, (40, 40), RESOLUTION, 0.0, 0.0)
        assert ids == ["on_floor", "just_under"]
        assert len(masks) == 2 and all(m.shape == (40, 40) for m in masks)

        # Same objects in a scene whose floor sits at y=1.0: bbox_min_y is
        # measured against floor_y, so 1.1 is floor-standing and 0.0 is not
        # (it would be a metre *below* the floor - still "within 0.15 above").
        objs_high = [_obj("a", (1.0, 1.0), (0.2, 0.2), bbox_min_y=1.1), _obj("b", (1.0, 1.0), (0.2, 0.2), bbox_min_y=1.3)]
        _, ids_high = _floor_standing_footprint_masks(objs_high, 1.0, (40, 40), RESOLUTION, 0.0, 0.0)
        assert ids_high == ["a"]


def _rot(x: float, z: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)


def _build_rotated_room_grid(angle_deg: float, rect_size=(3.0, 1.6), grid_size: int = 110, resolution: float = RESOLUTION):
    """UNKNOWN grid ([ix, iz], axis 0 = X) with one rotated rectangular room:
    2-cell OBSTACLE walls around a FREE interior (no doorway). Returns (occ, center_xz)."""
    occ = np.full((grid_size, grid_size), UNKNOWN, dtype=np.uint8)
    cx = cz = grid_size * resolution / 2.0
    xs = (np.arange(grid_size) + 0.5) * resolution
    grid_x, grid_z = np.meshgrid(xs, xs, indexing="ij")
    dx, dz = grid_x - cx, grid_z - cz
    a = math.radians(angle_deg)
    u = dx * math.cos(a) + dz * math.sin(a)
    v = -dx * math.sin(a) + dz * math.cos(a)
    hw, hh = rect_size[0] / 2, rect_size[1] / 2
    t = 2 * resolution
    interior = (np.abs(u) <= hw) & (np.abs(v) <= hh)
    with_wall = (np.abs(u) <= hw + t) & (np.abs(v) <= hh + t)
    occ[with_wall & ~interior] = OBSTACLE
    occ[interior] = FREE
    return occ, (cx, cz)


def _write_scene(scene_dir: Path, occ: np.ndarray) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "occupancy.npy", occ)
    (scene_dir / "occupancy_meta.json").write_text(json.dumps({"resolution": RESOLUTION, "origin_x": 0.0, "origin_z": 0.0}))
    (scene_dir / "scene_meta.json").write_text(json.dumps({"floor_y": 0.0, "ceiling_y": 2.5}))


def _rect_hull(center, size, angle_deg):
    hw, hh = size[0] / 2, size[1] / 2
    pts = [_rot(x, z, angle_deg) for x, z in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]]
    return np.array([(center[0] + x, center[1] + z) for x, z in pts])


class TestRotatedRoomIsAxisAlignedAfterBootstrap:
    ANGLE_DEG = 25.0

    def _run(self, tmp_path):
        occ, (cx, cz) = _build_rotated_room_grid(self.ANGLE_DEG)
        scene_dir, out_dir = tmp_path / "scene", tmp_path / "out"
        _write_scene(scene_dir, occ)
        # A floor-standing bed against the +U wall (its footprint overlaps the
        # OBSTACLE wall cells, i.e. pokes outside FREE) and an elevated shelf
        # far outside the room that must not become part of it.
        bed_center = (cx + _rot(1.1, 0.0, self.ANGLE_DEG)[0], cz + _rot(1.1, 0.0, self.ANGLE_DEG)[1])
        bed = ObjectFootprintInput(
            id="bed_0", label="bed", hull_xz=_rect_hull(bed_center, (1.0, 0.8), self.ANGLE_DEG),
            bbox_min=[0.0, 0.03, 0.0], bbox_max=[0.0, 0.6, 0.0], color_rgb=(100, 100, 100),
        )
        shelf = ObjectFootprintInput(
            id="shelf_0", label="shelf", hull_xz=_rect_hull((0.4, 0.4), (0.5, 0.3), 0.0),
            bbox_min=[0.0, 1.2, 0.0], bbox_max=[0.0, 1.5, 0.0], color_rgb=(100, 100, 100),
        )
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[bed, shelf])
        meta = json.loads((out_dir / "scene_meta.json").read_text())
        return report, meta

    def test_rotated_room_polygon_long_axis_is_within_2deg_of_an_axis(self, tmp_path):
        report, meta = self._run(tmp_path)
        assert report["yaw_applied"] is True
        assert meta["room_polygon_method"] == ROOM_POLYGON_METHOD
        assert meta["room_polygon_floor_standing_object_ids"] == ["bed_0"]
        assert report["n_room_polygon_floor_standing_objects"] == 1
        # The correction found the synthetic rotation (staircase noise at 5cm
        # keeps it from being exact), from the wall hull (morning-1 policy) - the
        # polygon estimate agrees and is logged.
        assert report["yaw_method"] == "wall_hull_min_area_rect"
        assert report["yaw_fallback_to_polygon"] is False
        assert report["yaw_correction_deg"] == pytest.approx(-self.ANGLE_DEG, abs=2.0)
        assert report["yaw_free_only_long_axis_angle_deg"] == pytest.approx(self.ANGLE_DEG, abs=2.0)

        room_polygon = meta["room_polygon"]
        assert room_polygon and len(room_polygon) >= 4
        fit = compute_room_polygon_yaw([tuple(p) for p in room_polygon])
        assert fit is not None and fit["is_degenerate_square"] is False
        assert _axis_residual_deg(fit["long_axis_angle_deg"]) < 2.0
        # ... and it really was rotated along with everything else: un-rotating
        # it by the applied correction gives back the original ~25deg shape.
        center = tuple(meta["yaw_rotation_center_xy"])
        unrotated = [rotate_point_xz(x, z, -meta["yaw_correction_rad"], center) for x, z in room_polygon]
        assert compute_room_polygon_yaw(unrotated)["long_axis_angle_deg"] == pytest.approx(self.ANGLE_DEG, abs=2.0)

    def test_elevated_object_outside_room_is_dropped_floor_standing_one_kept(self, tmp_path):
        report, _ = self._run(tmp_path)
        assert report["outside_room_ids_dropped"] == ["shelf_0"]
        assert report["n_objects_kept"] == 1


@pytest.mark.skipif(not HERO_SCENE_DIR.exists(), reason=f"hero fixture {HERO_SCENE_DIR} not on this machine")
class TestHeroScene:
    """Real hero inputs (read-only symlinks into var/scratch/hero-frozen/...)."""

    @pytest.fixture(scope="class")
    def hero_out_dir(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("hero_out")
        run_bootstrap(HERO_SCENE_DIR, out_dir, object_inputs=None)
        return out_dir

    @pytest.fixture(scope="class")
    def hero(self, hero_out_dir):
        report = json.loads((hero_out_dir / "report.json").read_text())
        meta = json.loads((hero_out_dir / "scene_meta.json").read_text())
        return report, meta

    def test_room_polygon_is_built_from_free_space_plus_floor_standing_furniture(self, hero):
        report, meta = hero
        assert meta["room_polygon_method"] == ROOM_POLYGON_METHOD
        assert "bed_0" in meta["room_polygon_floor_standing_object_ids"]
        assert "lamp_0" not in meta["room_polygon_floor_standing_object_ids"]  # bbox_min 1.86m up
        assert meta["room_polygon"] and len(meta["room_polygon"]) >= 4
        # Floor-standing objects are inside the room polygon by construction,
        # so they can never be dropped as "outside the room".
        assert not set(report["outside_room_ids_dropped"]) & set(meta["room_polygon_floor_standing_object_ids"])

    def test_room_polygon_method_aligns_its_own_polygon(self, hero):
        """The polygon method's own invariant: rotating the UNSNAPPED room polygon
        (`room_polygon_raw` - the exported `room_polygon` is Manhattan-snapped in
        the applied wall-hull frame, so it no longer carries the polygon's own
        angle) by *its* correction (yaw_free_only_correction_deg - key name kept
        from T3'') leaves its long axis within 2deg of an axis."""
        _, meta = hero
        center = tuple(meta["yaw_rotation_center_xy"])
        # scene_meta's polygon carries the *applied* rotation; undo it, then
        # apply the room-polygon method's own correction.
        own = math.radians(meta["yaw_free_only_correction_deg"]) - meta["yaw_correction_rad"]
        rotated = [rotate_point_xz(x, z, own, center) for x, z in meta["room_polygon_raw"]]
        assert _axis_residual_deg(compute_room_polygon_yaw(rotated)["long_axis_angle_deg"]) < 2.0

    def test_applied_wall_hull_rotation_leaves_the_bed_within_2deg_of_an_axis(self, hero, hero_out_dir):
        """Morning-1 acceptance (replaces T15a's "rotated room polygon < 2 deg"):
        the WALL-HULL angle is applied (-50.53 deg on the hero; the polygon's
        -56.76 left the bed ~5 deg and the nightstands 7-10 deg off-axis), the
        bed's long axis ends up within 2 deg of an axis (measured 1.36), and the
        room polygon is still Manhattan because the T15g snap happens in the
        applied frame. The two estimates still agree within the 10 deg threshold
        (6.23 apart) - the polygon is logged, not applied."""
        report, meta = hero
        assert report["yaw_method"] == "wall_hull_min_area_rect"
        assert report["yaw_applied_source"] == "wall_hull"
        assert report["yaw_fallback_to_polygon"] is False
        assert report["yaw_correction_deg"] == pytest.approx(meta["yaw_wall_hull_correction_deg"])
        assert report["yaw_long_axis_angle_deg"] == pytest.approx(report["yaw_wall_hull_long_axis_angle_deg"])
        assert report["yaw_disagreement_deg"] is not None
        assert report["yaw_disagreement_deg"] < report["yaw_disagreement_threshold_deg"]
        assert report["yaw_cross_check_disagrees"] is False
        # bed long axis within 2 deg of an axis, from the exported objects.json
        objects = json.loads((hero_out_dir / "objects.json").read_text())["objects"]
        bed = next(o for o in objects if o["id"] == "bed_0")
        assert _axis_residual_deg(math.degrees(bed["angle_rad"])) < 2.0
        # room polygon still Manhattan: every interior angle 90 or 270
        room = [tuple(p) for p in meta["room_polygon"]]
        assert meta["room_polygon_regularized"] is True
        for (x0, z0), (x1, z1) in zip(room, room[1:] + room[:1]):
            assert min(abs(x1 - x0), abs(z1 - z0)) < 1e-6, "room polygon edge is not axis-aligned"

    def test_bat_wall_fragment_beyond_the_doorway_is_dropped(self, hero):
        """T15b acceptance: the obstacle fragment in the area beyond the door
        (the "bat", wall_1 as scanned, 0.30 m^2) is not exported."""
        report, _ = hero
        wall_drops = [d for d in report["dropped_prims"] if d["kind"] == "wall"]
        assert wall_drops, "expected the doorway-spill fragment to be dropped"
        assert all(d["outside_fraction"] > 0.5 for d in wall_drops)
        assert report["n_walls"] == report["n_walls_before_room_check"] - len(wall_drops)


class TestDoorwaySpillDetached:
    """T15b: `detach_corridor_spill` - the room polygon stops at a doorway
    instead of trailing through it (radius 9 cells = 0.45 m at 5 cm, a 0.95 m
    disk: wider than any interior door, narrower than any room)."""

    R = 9

    @staticmethod
    def _grid(corridor_half_width_cells: int) -> np.ndarray:
        occ = np.full((130, 120), UNKNOWN, dtype=np.uint8)
        occ[10:70, 10:70] = FREE  # 3 m x 3 m room
        occ[70:100, 40 - corridor_half_width_cells : 40 + corridor_half_width_cells] = FREE  # 1.5 m long passage along +X
        occ[100:125, 10:70] = FREE  # 1.25 m x 3 m area beyond it
        return occ

    def test_door_width_corridor_and_area_beyond_are_cut(self):
        occ = self._grid(corridor_half_width_cells=5)  # 0.5 m wide passage
        room = build_room_mask(occ, [], opening_radius_cells=self.R)
        assert room[10:70, 10:70].all(), "the room itself is intact (opening must not erase its corners)"
        assert not room[100:125, :].any(), "the area beyond the doorway is not room"
        # The kept nub is the reconstruction reach (R) plus however far the disk's
        # own curvature lets the core protrude into the corridor mouth (~2 cells
        # for a 10-cell corridor) - well under 2R and nowhere near the far end.
        assert room[70:75, 35:45].all(), "the doorway itself stays room"
        assert not room[70 + 2 * self.R :, :].any(), "at most a ~radius-deep nub of the passage is kept"
        poly = extract_room_polygon(occ, RESOLUTION, 0.0, 0.0, [], opening_radius_cells=self.R)
        assert max(x for x, _ in poly) <= (70 + 2 * self.R) * RESOLUTION + 1e-9

    def test_wide_opening_keeps_the_area_beyond_it(self):
        occ = self._grid(corridor_half_width_cells=15)  # 1.5 m wide: a real opening, not a door
        room = build_room_mask(occ, [], opening_radius_cells=self.R)
        assert room[100:125, 10:70].all()

    def test_room_narrower_than_the_disk_is_left_unchanged(self):
        occ = np.full((40, 40), UNKNOWN, dtype=np.uint8)
        occ[10:24, 10:24] = FREE  # 0.7 m square: no 0.95 m disk fits -> opening is skipped
        room = build_room_mask(occ, [], opening_radius_cells=self.R)
        np.testing.assert_array_equal(room, occ == FREE)

    def test_zero_radius_disables_the_step(self):
        occ = self._grid(corridor_half_width_cells=5)
        room = build_room_mask(occ, [], opening_radius_cells=0)
        assert room[100:125, 10:70].all()
