"""Integration test for scripts/msa/bootstrap.py on the 02_modular_home fixture
(SPEC.md §11 "Integration ... bootstrap on 02_modular_home fixture -> GLB/DXF/SVG
exist, gaps.json matches golden within 1cm"). No Blender/GPU - CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from scripts.msa.bootstrap import (
    ObjectFootprintInput,
    _drop_or_clip_objects_outside_room,
    _filter_small_isolated_objects,
    _footprint_polygon,
    _walls_union,
    compute_object_footprints,
    load_object_inputs_from_hulls_json,
    run_bootstrap,
)
from scripts.msa.export_glb import _floor_plate_mesh, _floor_plate_polygon
from scripts.msa.geometry import WallPolygon

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "msa" / "02_modular_home"


@pytest.fixture(scope="module")
def bootstrap_report(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("msa_a0_integration")
    inputs = load_object_inputs_from_hulls_json(FIXTURE_DIR / "scene_objects_hulls.json")
    report = run_bootstrap(FIXTURE_DIR, out_dir, object_inputs=inputs)
    return out_dir, report


class TestBootstrapExports:
    def test_all_export_files_created(self, bootstrap_report):
        out_dir, _ = bootstrap_report
        for name in ("scene.glb", "scene.usd", "floor_plan.dxf", "floor_plan.svg", "gaps.json", "report.json"):
            path = out_dir / name
            assert path.exists(), f"missing export: {name}"
            assert path.stat().st_size > 0, f"empty export: {name}"

    def test_report_reflects_known_scene_facts(self, bootstrap_report):
        _, report = bootstrap_report
        # The two ceiling fans and two air conditioners in this scene sit
        # entirely above 2.0m (see docs/DECISIONS.md's isaac-demo-look v3 entry on
        # the same fan-exclusion rule in usd_export.py) - Stage A3 must exclude
        # them too, the same way.
        assert set(report["overhead_object_ids"]) == {"air conditioner_0", "air conditioner_1", "fan_0", "fan_1"}
        assert report["n_walls"] >= 1
        assert report["n_gaps"] > 0
        assert report["usd_exported"] is True

    def test_walls_at_least_one_percent_of_dropped_components_are_tiny(self, bootstrap_report):
        _, report = bootstrap_report
        # Sanity check on the >=0.3m^2 filter itself: every dropped component
        # really is below that threshold (not a filter-direction bug).
        for comp in report["dropped_components"]:
            assert comp["area_m2"] < 0.3


class TestWallBackfill:
    """Wall-backfill Tier 1 (see `compute_object_footprints`/
    `_backfill_footprint_to_walls` in scripts/msa/bootstrap.py): a footprint edge
    within 0.10m of a wall polygon gets extended out to touch it, closing the
    thin sliver of un-scanned floor/object the camera never captured right
    against the wall (e.g. a wardrobe or headboard)."""

    # A solid wall slab occupying x in [-0.2, 0.0], z in [-2, 2] (its
    # room-facing surface is the x=0.0 plane).
    WALL = WallPolygon(vertices=[(-0.2, -2.0), (-0.2, 2.0), (0.0, 2.0), (0.0, -2.0)], area_m2=0.8)

    def test_edge_within_threshold_extends_to_touch_wall(self):
        # True footprint: x in [0.05, 1.0], z in [-0.2, 0.2] - its near edge
        # (x=0.05) sits 0.05m from the wall's x=0.0 face, inside the 0.10m
        # backfill threshold.
        hull = np.array([[0.05, -0.2], [0.05, 0.2], [1.0, 0.2], [1.0, -0.2]])
        obj = ObjectFootprintInput(
            id="wardrobe_0", label="wardrobe", hull_xz=hull, bbox_min=[0.0, 0.0, -0.2], bbox_max=[1.0, 2.0, 0.2], color_rgb=(10, 20, 30)
        )

        kept, overhead = compute_object_footprints([obj], floor_y=0.0, walls=[self.WALL])

        assert overhead == []
        (result,) = kept
        assert result["angle_rad"] == pytest.approx(0.0, abs=1e-9)
        cx, cz = result["center_xy"]
        length_u, width_v = result["size_uv"]
        # Near edge now sits exactly on the wall face (x=0.0); far edge
        # (x=1.0, where the object was actually scanned) is untouched.
        near_edge_x = cx - length_u / 2
        far_edge_x = cx + length_u / 2
        assert near_edge_x == pytest.approx(0.0, abs=1e-9)
        assert far_edge_x == pytest.approx(1.0, abs=1e-9)
        assert cz == pytest.approx(0.0, abs=1e-9)
        assert width_v == pytest.approx(0.4, abs=1e-9)
        # hull_xz (the collision geometry) must stay exactly as measured -
        # backfill only ever touches the visual oriented-rectangle footprint,
        # never the actual scanned hull. Compare as an unordered point set
        # since convex_hull_2d's vertex order/start point is an implementation
        # detail we don't want this test to pin down.
        assert sorted(tuple(p) for p in result["hull_xz"]) == pytest.approx(sorted(tuple(p) for p in hull.tolist()))

    def test_object_far_from_any_wall_is_unchanged(self):
        # Regression guard: an object nowhere near a wall must come out of
        # compute_object_footprints byte-for-byte identical to the
        # no-backfill (walls=None) result - the same rectangle
        # oriented_min_area_rect would produce on its own.
        hull = np.array([[3.0, -0.2], [3.0, 0.2], [4.0, 0.2], [4.0, -0.2]])
        obj = ObjectFootprintInput(
            id="table_0", label="table", hull_xz=hull, bbox_min=[3.0, 0.0, -0.2], bbox_max=[4.0, 1.0, 0.2], color_rgb=(40, 50, 60)
        )

        with_walls, _ = compute_object_footprints([obj], floor_y=0.0, walls=[self.WALL])
        without_walls, _ = compute_object_footprints([obj], floor_y=0.0, walls=None)

        assert with_walls[0]["center_xy"] == pytest.approx(without_walls[0]["center_xy"])
        assert with_walls[0]["size_uv"] == pytest.approx(without_walls[0]["size_uv"])
        # And sanity: it really is the untouched footprint (3.5, 0.0) / (1.0, 0.4).
        assert with_walls[0]["center_xy"] == pytest.approx((3.5, 0.0))
        assert with_walls[0]["size_uv"] == pytest.approx((1.0, 0.4))


def _footprint_obj(id: str, center_xy: tuple[float, float], size_uv: tuple[float, float], *, angle_rad: float = 0.0, height: float = 0.5) -> dict:
    """A minimal `compute_object_footprints`-shaped dict, for exercising
    downstream footprint logic (small-object filter, floor-plate union)
    without going through the full hull -> oriented_min_area_rect pipeline."""
    return {
        "id": id,
        "label": id,
        "center_xy": center_xy,
        "size_uv": size_uv,
        "angle_rad": angle_rad,
        "height": height,
        "bbox_min_y": 0.0,
        "color_rgb": (100, 100, 100),
        "hull_xz": [],
    }


class TestSmallObjectFilter:
    """Small/short-footprint filter (see `_filter_small_isolated_objects` in
    scripts/msa/bootstrap.py): drops a footprint with area < 0.04m^2 OR height
    < 0.10m, UNLESS it touches a wall or another object's footprint (see that
    function's docstring for why "touching floor" isn't a separate check)."""

    WALL = WallPolygon(vertices=[(-0.2, -2.0), (-0.2, 2.0), (0.0, 2.0), (0.0, -2.0)], area_m2=0.8)

    def test_small_isolated_object_is_dropped(self):
        # 0.1m x 0.1m = 0.01m^2 (< 0.04m^2), sitting alone in open space, far
        # from any wall or other object.
        noise = _footprint_obj("noise_0", center_xy=(5.0, 5.0), size_uv=(0.1, 0.1))
        kept, dropped_ids = _filter_small_isolated_objects([noise], walls_union=None)
        assert kept == []
        assert dropped_ids == ["noise_0"]

    def test_equally_small_object_touching_wall_survives(self):
        walls_union = _walls_union([self.WALL])
        # Same tiny footprint (0.01m^2), but flush against the wall's x=0.0
        # face (footprint spans x in [0.0, 0.1]) - e.g. a remote control on a
        # nightstand against the wall.
        remote = _footprint_obj("remote_0", center_xy=(0.05, 0.0), size_uv=(0.1, 0.1))
        kept, dropped_ids = _filter_small_isolated_objects([remote], walls_union=walls_union)
        assert dropped_ids == []
        assert [o["id"] for o in kept] == ["remote_0"]

    def test_equally_small_object_touching_another_object_survives(self):
        # Sofa spans x in [-0.5, 0.5]; a tiny (0.01m^2) cushion sits flush
        # against its right edge (x in [0.5, 0.6]) - wedged next to furniture,
        # not floating alone.
        sofa = _footprint_obj("sofa_0", center_xy=(0.0, 0.0), size_uv=(1.0, 1.0), height=0.8)
        cushion = _footprint_obj("cushion_0", center_xy=(0.55, 0.0), size_uv=(0.1, 0.1), height=0.05)
        kept, dropped_ids = _filter_small_isolated_objects([sofa, cushion], walls_union=None)
        assert dropped_ids == []
        assert {o["id"] for o in kept} == {"sofa_0", "cushion_0"}

    def test_small_object_far_from_wall_is_still_dropped(self):
        # Same tiny footprint as the wall-touching case, but 0.5m from the
        # wall (well past both the 0.02m touch tolerance and the 0.10m
        # wall-backfill threshold) - still isolated, still dropped.
        walls_union = _walls_union([self.WALL])
        far = _footprint_obj("noise_1", center_xy=(0.55, 0.0), size_uv=(0.1, 0.1))
        kept, dropped_ids = _filter_small_isolated_objects([far], walls_union=walls_union)
        assert kept == []
        assert dropped_ids == ["noise_1"]

    def test_normal_sized_object_is_never_dropped(self):
        # Neither area nor height is below threshold - the isolation check
        # shouldn't even matter.
        table = _footprint_obj("table_0", center_xy=(5.0, 5.0), size_uv=(1.0, 0.6), height=0.7)
        kept, dropped_ids = _filter_small_isolated_objects([table], walls_union=None)
        assert dropped_ids == []
        assert [o["id"] for o in kept] == ["table_0"]


class TestFloorPlateUnion:
    """Floor plate for visual export (GLB "floor" node / USD floor mesh) =
    union(free-space `floor_polygon`, every kept object's footprint) - see
    `export_glb._floor_plate_polygon` - so an object footprint that extends
    slightly past the raw free-space polygon (e.g. after wall-backfill, or
    just close to the boundary) still has solid floor underneath it."""

    def test_floor_plate_covers_object_footprint_extending_past_floor_polygon(self):
        floor_polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        # Footprint centered at x=2.05 (half-width 0.15) spans x in
        # [1.9, 2.2] - 0.2m of it pokes past the floor_polygon's x=2.0 edge.
        wardrobe = _footprint_obj("wardrobe_0", center_xy=(2.05, 1.0), size_uv=(0.3, 0.4), height=1.8)
        outside_point = Point(2.1, 1.0)

        assert not Polygon(floor_polygon).contains(outside_point), "test setup: point must be outside the raw floor_polygon"

        floor_plate = _floor_plate_polygon(floor_polygon, [wardrobe])
        assert floor_plate is not None
        assert floor_plate.contains(outside_point), "floor plate union must cover the part of the footprint outside floor_polygon"

        mesh = _floor_plate_mesh(floor_plate, floor_y=0.0)
        assert mesh is not None
        # World X extent (mesh.vertices[:, 0], see _extrude_shapely_polygon_xz's
        # remap) must reach past the original floor_polygon's x=2.0 edge to
        # actually cover the object.
        assert mesh.vertices[:, 0].max() > 2.0

    def test_floor_plate_is_none_without_floor_polygon_or_objects(self):
        assert _floor_plate_polygon(None, []) is None


class TestOutsideRoomFilter:
    """Reflection/through-window guard (see `_drop_or_clip_objects_outside_room`
    in scripts/msa/bootstrap.py): an object footprint mostly outside
    `floor_polygon` is likely a mirror-reflection or through-a-window false
    detection and gets dropped; one only partly outside gets clipped to
    floor_polygon and refit to a new oriented rectangle."""

    FLOOR_POLYGON = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    def test_object_mostly_outside_floor_polygon_is_dropped(self):
        # Footprint x in [2.75, 3.25] - entirely past the floor_polygon's
        # x=2.0 edge, i.e. 100% outside (well past the 50% drop threshold).
        mirror_ghost = _footprint_obj("mirror_ghost_0", center_xy=(3.0, 1.0), size_uv=(0.5, 0.5))
        kept, dropped_ids = _drop_or_clip_objects_outside_room([mirror_ghost], self.FLOOR_POLYGON)
        assert kept == []
        assert dropped_ids == ["mirror_ghost_0"]

    def test_object_partially_outside_is_clipped_not_dropped(self):
        # Footprint x in [1.68, 2.08], z in [0.8, 1.2]: 0.08m of its 0.4m
        # x-extent (20%) pokes past the floor_polygon's x=2.0 edge.
        wardrobe = _footprint_obj("wardrobe_0", center_xy=(1.88, 1.0), size_uv=(0.4, 0.4))
        original_area = _footprint_polygon(wardrobe).area

        kept, dropped_ids = _drop_or_clip_objects_outside_room([wardrobe], self.FLOOR_POLYGON)

        assert dropped_ids == []
        assert [o["id"] for o in kept] == ["wardrobe_0"]
        clipped_footprint = _footprint_polygon(kept[0])
        # Refit rectangle must actually fit inside floor_polygon now (no more
        # than a negligible sliver outside, allowing for oriented-rect refit
        # not being pixel-exact against the clipped polygon it was fit to).
        outside_area = clipped_footprint.difference(Polygon(self.FLOOR_POLYGON)).area
        assert outside_area == pytest.approx(0.0, abs=1e-9)
        # And it really did shrink by about the 20% that was outside.
        assert clipped_footprint.area == pytest.approx(original_area * 0.8, rel=0.05)

    def test_object_fully_inside_is_completely_unchanged(self):
        # Regression guard: fully inside floor_polygon -> intersection ==
        # original footprint, so the object dict must come back untouched.
        table = _footprint_obj("table_0", center_xy=(1.0, 1.0), size_uv=(0.4, 0.4))
        kept, dropped_ids = _drop_or_clip_objects_outside_room([table], self.FLOOR_POLYGON)
        assert dropped_ids == []
        assert kept == [table]

    def test_no_floor_polygon_leaves_objects_untouched(self):
        far_outside = _footprint_obj("mirror_ghost_1", center_xy=(30.0, 30.0), size_uv=(0.5, 0.5))
        kept, dropped_ids = _drop_or_clip_objects_outside_room([far_outside], None)
        assert dropped_ids == []
        assert kept == [far_outside]


class TestGapsMatchGolden:
    def test_gaps_json_matches_golden_within_1cm(self, bootstrap_report):
        out_dir, _ = bootstrap_report
        golden = json.loads((FIXTURE_DIR / "golden_gaps.json").read_text())
        current = json.loads((out_dir / "gaps.json").read_text())

        golden_by_pair = {frozenset((g["a"], g["b"])): g for g in golden}
        current_by_pair = {frozenset((g["a"], g["b"])): g for g in current}

        assert set(golden_by_pair) == set(current_by_pair), "gap pair set changed vs. golden - bootstrap logic drifted"
        for pair, golden_gap in golden_by_pair.items():
            current_gap = current_by_pair[pair]
            assert current_gap["width_m"] == pytest.approx(golden_gap["width_m"], abs=0.01), (
                f"gap {pair} width drifted more than 1cm from golden: "
                f"{current_gap['width_m']} vs {golden_gap['width_m']}"
            )


class TestDropOrClipWallsOutsideRoom:
    """T15b: the outside-room drop/clip rule applied to wall polygons, judged
    against the room polygon buffered by WALL_CLIP_MARGIN_M (0.20 m) because a
    wall sits just outside the free-space boundary by construction."""

    ROOM = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]

    @staticmethod
    def _wall(x0, x1, z0, z1):
        return WallPolygon(vertices=[(x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0)], area_m2=(x1 - x0) * (z1 - z0))

    def test_rule_per_wall(self):
        from scripts.msa.bootstrap import WALL_CLIP_MARGIN_M, _drop_or_clip_walls_outside_room

        assert WALL_CLIP_MARGIN_M == pytest.approx(0.20)
        flush = self._wall(4.0, 4.1, 0.0, 4.0)  # a real wall: on the boundary, within the margin -> untouched
        far = self._wall(6.0, 7.0, 0.0, 1.0)  # fully outside -> dropped
        mostly_out = self._wall(3.9, 4.6, 0.0, 1.0)  # 0.4 of 0.7 beyond x=4.2 (57%) -> dropped
        partly_out = self._wall(3.9, 4.4, 0.0, 1.0)  # 0.2 of 0.5 beyond x=4.2 (40%) -> clipped
        records = []
        kept, kept_idx = _drop_or_clip_walls_outside_room([flush, far, mostly_out, partly_out], self.ROOM, dropped_records=records)
        assert kept_idx == [0, 3]
        assert kept[0] is flush
        assert max(x for x, _ in kept[1].vertices) == pytest.approx(4.2)
        assert kept[1].area_m2 == pytest.approx(0.3)
        assert [(r["id"], r["kind"]) for r in records] == [("wall_1", "wall"), ("wall_2", "wall")]
        assert records[0]["outside_fraction"] == pytest.approx(1.0)
        assert records[1]["outside_fraction"] == pytest.approx(0.4 / 0.7)
        assert records[1]["area_m2"] == pytest.approx(0.7)

    def test_no_room_polygon_keeps_everything(self):
        from scripts.msa.bootstrap import _drop_or_clip_walls_outside_room

        walls = [self._wall(6.0, 7.0, 0.0, 1.0)]
        kept, idx = _drop_or_clip_walls_outside_room(walls, None)
        assert kept == walls and idx == [0]

    def test_objects_record_dropped_prims_too(self):
        obj = _footprint_obj("ghost", (6.0, 0.5), (0.5, 0.5))
        records = []
        kept, dropped = _drop_or_clip_objects_outside_room([obj], self.ROOM, dropped_records=records)
        assert kept == [] and dropped == ["ghost"]
        assert records == [{"id": "ghost", "kind": "object", "outside_fraction": pytest.approx(1.0), "area_m2": pytest.approx(0.25)}]

    def test_report_lists_dropped_prims_and_keeps_wall_ids_in_step(self, bootstrap_report):
        _, report = bootstrap_report
        assert report["n_walls"] == report["n_walls_before_room_check"] - report["n_walls_outside_room_dropped"]
        assert {d["kind"] for d in report["dropped_prims"]} <= {"wall", "object"}
        assert all(d["outside_fraction"] > 0.5 for d in report["dropped_prims"])
        assert [d["id"] for d in report["dropped_prims"] if d["kind"] == "object"] == report["outside_room_ids_dropped"]
        assert [d["id"] for d in report["dropped_prims"] if d["kind"] == "wall"] == report["wall_ids_dropped"]
