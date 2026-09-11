"""T15d: scripts/msa/export_presentation.py - grid-level route planning in the raw
frame, yaw round trip into the exported frame, path.json, and the full CLI on the
02_modular_home fixture (scene_presentation.usd + two stills, no Isaac)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom
from shapely.geometry import Point, Polygon

from app.services.pathfinding import FREE, OBSTACLE, UNKNOWN
from app.services.scene_ingest import GridMeta
from scripts.msa import export_presentation as ep
from scripts.msa.bootstrap import load_object_inputs_from_hulls_json, run_bootstrap

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "msa" / "02_modular_home"


class TestPlanRoute:
    @staticmethod
    def _grid(width=60, height=40, res=0.05):
        cells = np.full((width, height), FREE, dtype=np.uint8)
        cells[:, :2] = OBSTACLE
        cells[:, -2:] = OBSTACLE
        cells[:2, :] = OBSTACLE
        cells[-2:, :] = OBSTACLE
        meta = GridMeta(resolution=res, origin_x=0.0, origin_z=0.0, width=width, height=height)
        return cells, meta

    def test_rasterize_hulls_marks_cells_inside_the_polygon(self):
        cells, meta = self._grid()
        hull = np.array([(1.0, 0.5), (1.5, 0.5), (1.5, 1.0), (1.0, 1.0)])
        n = ep.rasterize_hulls(cells, meta, {"desk_0": hull})
        assert n == 10 * 10
        assert cells[25, 15] == OBSTACLE  # (1.275, 0.775) inside
        assert cells[18, 15] == FREE  # (0.925, 0.775) outside

    def test_route_starts_on_longest_free_run_and_ends_adjacent_to_target(self):
        cells, meta = self._grid()
        hull = np.array([(2.2, 0.8), (2.7, 0.8), (2.7, 1.4), (2.2, 1.4)])
        route = ep.plan_route(cells, meta, {"desk_0": hull}, "desk_0", robot_radius_m=0.1)
        assert len(route["path_cells"]) >= 2
        goal = Point(*route["goal_raw_xz"])
        d = Polygon(hull).distance(goal)
        # T15h: nearest traversable cell that keeps radius + GOAL_CLEARANCE_MARGIN_M to the hull, within a cell of it
        required = 0.1 + ep.GOAL_CLEARANCE_MARGIN_M
        assert required - 1e-9 <= d <= required + 2 * meta.resolution
        assert route["goal_clearance_m"] == pytest.approx(d) and route["goal_required_clearance_m"] == pytest.approx(required)
        assert route["goal_candidates_tried"] == 1
        sx, sz = route["start_raw_xz"]
        assert not Polygon(hull).buffer(0.1).contains(Point(sx, sz))
        assert route["points_raw_xz"][-1] == pytest.approx(route["goal_raw_xz"])
        assert route["path_min_clearance_m"] >= 0.1 - 1e-9  # never grazes the hull
        assert route["room_clip_applied"] is False and route["path_points_inside_room_min_m"] is None

    def test_grid_is_clipped_to_the_room_polygon_eroded_by_the_robot_radius(self):
        """T15e: the grid's FREE region ran past the authored floor (path point 2 was
        0.28 m outside it). With a room polygon the traversable grid is clipped to
        room.buffer(-radius): every path point is inside by >= the radius, the goal
        keeps radius + margin to the boundary, and cells outside were blocked."""
        cells, meta = self._grid()  # FREE everywhere but a 2-cell border: 0.1..2.9 x 0.1..1.9
        room = [(0.6, 0.3), (2.4, 0.3), (2.4, 1.7), (0.6, 1.7)]  # the "floor" is much smaller than the FREE grid
        hull = np.array([(2.0, 0.8), (2.3, 0.8), (2.3, 1.2), (2.0, 1.2)])  # desk near the room's right wall
        r = 0.1
        route = ep.plan_route(cells, meta, {"desk_0": hull}, "desk_0", robot_radius_m=r, room_polygon_raw=room)
        room_poly = Polygon(room)
        eroded = room_poly.buffer(-r)
        assert route["room_clip_applied"] is True and route["n_cells_clipped_outside_room"] > 0
        for x, z in route["points_raw_xz"]:
            assert eroded.covers(Point(x, z)), (x, z)
            assert room_poly.boundary.distance(Point(x, z)) >= r - 1e-6
        assert route["path_points_inside_room_min_m"] >= r - 1e-6
        assert route["goal_distance_to_room_boundary_m"] >= r + ep.GOAL_CLEARANCE_MARGIN_M - 1e-9
        assert route["goal_clearance_m"] >= r + ep.GOAL_CLEARANCE_MARGIN_M - 1e-9
        assert route["path_min_clearance_m"] >= r - 1e-6
        sx, sz = route["start_raw_xz"]
        assert eroded.covers(Point(sx, sz))  # the start moved inside the room too
        without = ep.plan_route(cells, meta, {"desk_0": hull}, "desk_0", robot_radius_m=r)
        assert without["room_clip_applied"] is False
        assert not eroded.covers(Point(*without["start_raw_xz"]))  # ...whereas the unclipped start is outside the room

    def test_goal_needs_clearance_to_every_hull_not_just_the_target(self):
        cells, meta = self._grid()
        target = np.array([(1.5, 0.8), (1.8, 0.8), (1.8, 1.2), (1.5, 1.2)])
        # a second object right next to the target's nearest side: the naive nearest cell sits between them
        neighbour = np.array([(1.5, 1.5), (1.8, 1.5), (1.8, 1.8), (1.5, 1.8)])
        route = ep.plan_route(cells, meta, {"desk_0": target, "chair_0": neighbour}, "desk_0", robot_radius_m=0.1)
        goal = Point(*route["goal_raw_xz"])
        assert Polygon(neighbour).distance(goal) >= 0.2 - 1e-9 and Polygon(target).distance(goal) >= 0.2 - 1e-9
        assert route["goal_clearance_m"] >= 0.2 - 1e-9
        assert route["goal_candidates_with_clearance"] < route["goal_candidates_near_target"]

    def test_clearance_and_sampling_helpers(self):
        room = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
        hull = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        c = ep.clearance_m([(0.5, 1.5), (3.0, 0.5), (1.5, 1.5), (5.0, 1.0)], [hull], room)
        assert c.tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])  # hull side; wall side; inside hull; outside room
        pts = ep.sample_polyline([(0.0, 0.0), (0.1, 0.0)], 0.02)
        assert len(pts) == 6 and pts[-1].tolist() == pytest.approx([0.1, 0.0])

    def test_target_selection_by_class_prefers_largest_footprint_and_exact_id(self):
        hulls = {
            "desk_0": np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float),
            "desk_1": np.array([(0, 0), (2, 0), (2, 1), (0, 1)], dtype=float),
            "bed_0": np.array([(0, 0), (3, 0), (3, 3), (0, 3)], dtype=float),
        }
        assert ep.select_target(hulls, "desk") == "desk_1"
        assert ep.select_target(hulls, "desk_0") == "desk_0"
        with pytest.raises(SystemExit):
            ep.select_target(hulls, "sofa")

    def test_unknown_cells_are_goal_candidates_only_after_free_ones(self):
        cells, meta = self._grid()
        cells[30:40, 2:38] = UNKNOWN  # unknown band right where the target sits
        hull = np.array([(1.6, 0.8), (1.9, 0.8), (1.9, 1.2), (1.6, 1.2)])
        route = ep.plan_route(cells, meta, {"desk_0": hull}, "desk_0", robot_radius_m=0.05)
        gx, gz = route["goal_cell"]
        assert cells[gx, gz] == FREE


class TestMorning3Route:
    """Morning-3: start = traversable cell inside the doorway spur farthest from the
    target; the spur = the deepest piece of the Manhattan outline outside its largest
    inscribed axis-aligned rectangle; corridor report + loud assertions."""

    # an L-ish room: 4 x 3 core with a 1.0 wide, 1.2 deep doorway appendix on the west
    # side and a shallow 0.2 m wall strip on the north side (not a doorway)
    ROOM = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 3.0), (2.0, 3.2), (1.0, 3.2), (1.0, 3.0),
            (0.0, 3.0), (0.0, 2.0), (-1.2, 2.0), (-1.2, 1.0), (0.0, 1.0)]

    def test_core_rectangle_is_the_largest_inscribed_axis_aligned_rectangle(self):
        core = ep.room_core_rectangle(self.ROOM)
        assert core.bounds == pytest.approx((0.0, 0.0, 4.0, 3.0))
        # a plain rectangle is its own core; too few vertices -> None
        assert ep.room_core_rectangle([(0, 0), (2, 0), (2, 1), (0, 1)]).bounds == pytest.approx((0, 0, 2, 1))
        assert ep.room_core_rectangle([(0, 0), (1, 1)]) is None

    def test_doorway_spur_is_the_deepest_piece_outside_the_core(self):
        spur, info = ep.doorway_spur_polygon(self.ROOM)
        assert spur is not None and spur.bounds == pytest.approx((-1.2, 1.0, 0.0, 2.0), abs=1e-6)
        assert info["spur_depth_m"] == pytest.approx(1.2, abs=1e-6)
        # the 0.2 m strip is < SPUR_MIN_AREA_M2 (0.2 m^2) and never a candidate
        assert len(info["pieces"]) == 1
        # no appendix at all -> no spur
        assert ep.doorway_spur_polygon([(0, 0), (4, 0), (4, 3), (0, 3)])[0] is None

    def test_start_is_the_farthest_traversable_cell_in_the_spur_and_the_route_crosses_the_corridor(self):
        res = 0.05
        width, height = 120, 80  # x -1.5..4.5, z -0.5..3.5
        cells = np.full((width, height), FREE, dtype=np.uint8)
        meta = GridMeta(resolution=res, origin_x=-1.5, origin_z=-0.5, width=width, height=height)
        # target desk along the east wall with a 0.45 m corridor between it and the wall
        desk = np.array([(3.0, 0.6), (3.55, 0.6), (3.55, 2.4), (3.0, 2.4)])
        spur, _ = ep.doorway_spur_polygon(self.ROOM)
        route = ep.plan_route(cells, meta, {"desk_0": desk}, "desk_0", 0.1, self.ROOM,
                              start_region_raw=list(spur.exterior.coords)[:-1], start_rule="farthest_from_target")
        sx, sz = route["start_raw_xz"]
        assert spur.covers(Point(sx, sz)) and Polygon(self.ROOM).boundary.distance(Point(sx, sz)) >= 0.1 - 1e-6
        # the farthest cell in the spur from the desk: its west end
        assert sx == pytest.approx(-1.2 + 0.1 + res / 2, abs=res + 1e-6) or sx < -1.0
        assert route["start_distance_to_target_m"] == pytest.approx(route["start_info"]["farthest_distance_to_target_m"])
        assert route["start_rule"] == "farthest_from_target" and route["start_candidates_tried"] == 1
        assert route["path_length_m"] >= 3.5
        # goal = T15h nearest clear cell (radius + 0.10 to the desk AND the wall): the route ends
        # between the desk and a wall - the corridor report finds that passage
        c = route["corridor"]
        assert c["crossed"] is True and 0.4 - 1e-9 <= c["width_m"] <= 2 * ep.CORRIDOR_SEARCH_M
        assert c["d_target_m"] >= 0.2 - 1e-9 and c["d_boundary_m"] >= 0.2 - 1e-9
        assert c["width_m"] == pytest.approx(c["d_target_m"] + c["d_boundary_m"])
        # without a region: the farthest traversable cell anywhere (the room's far west spur end again here)
        anywhere = ep.plan_route(cells, meta, {"desk_0": desk}, "desk_0", 0.1, self.ROOM, start_rule="farthest_from_target")
        assert anywhere["start_info"]["region_applied"] is False
        assert anywhere["start_distance_to_target_m"] >= route["start_distance_to_target_m"] - 1e-9

    def test_corridor_report_reads_the_narrowest_target_to_boundary_passage(self):
        room = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
        target = Polygon([(3.5, 1.0), (3.9, 1.0), (3.9, 2.0), (3.5, 2.0)])  # 0.1 m from the east wall
        chair = Polygon([(1.0, 1.0), (1.4, 1.0), (1.4, 1.4), (1.0, 1.4)])
        # a path that ends just north-east of the target: 0.25 m from it, 0.30 m from the east wall
        samples = ep.sample_polyline([(0.5, 0.5), (3.0, 2.5), (3.7, 2.25)], 0.02)
        rep = ep.corridor_report(samples, target, [chair], room)
        assert rep["crossed"] is True
        assert rep["width_m"] == pytest.approx(0.55, abs=0.03)  # d(target) 0.25 + d(east wall) 0.30 at the end point
        assert rep["d_target_m"] == pytest.approx(0.25, abs=0.03) and rep["d_boundary_m"] == pytest.approx(0.30, abs=0.03)
        # a path that never comes near the target does not cross anything
        rep2 = ep.corridor_report(ep.sample_polyline([(0.5, 0.5), (0.5, 2.5)], 0.02), target, [chair], room)
        assert rep2["crossed"] is False and rep2["closest_approach_target_m"] > 2.0

    def test_fixture_demo_assertions_fail_loudly(self, fixture_presentation):
        """The 02_modular_home table stands mid-room: the corridor requirement cannot be
        met there and the planner must say so (SystemExit with the numbers), never
        silently pick another goal; and a route shorter than the minimum is refused."""
        from scripts.msa.record_isaac import resolve_platform

        out_dir, _ = fixture_presentation
        bo = ep.load_bootstrap_output(out_dir)
        with pytest.raises(SystemExit, match="does not pass through the corridor"):
            ep.plan_presentation_path(bo, FIXTURE_DIR, "table", resolve_platform("burger"), require_corridor=True, min_route_length_m=0.0)
        with pytest.raises(SystemExit, match="demo minimum"):
            ep.plan_presentation_path(bo, FIXTURE_DIR, "table", resolve_platform("burger"), require_corridor=False, min_route_length_m=50.0)
        pj = ep.plan_presentation_path(bo, FIXTURE_DIR, "table", resolve_platform("burger"), require_corridor=False, min_route_length_m=0.0)
        assert pj["start_source"].startswith("doorway_spur_farthest_from_target") or pj["start_source"].startswith("farthest_from_target_no_spur")
        assert pj["route_quality"]["corridor"]["required"] is False
        assert pj["route_quality"]["start_rule"] == "farthest_from_target"


class TestFrameRoundTrip:
    def test_exported_and_raw_frames_invert(self, tmp_path):
        import trimesh

        bo = ep.BootstrapOutput(
            out_dir=tmp_path, scene=trimesh.Scene(), floor_y=0.0, ceiling_y=2.5, room_polygon=[],
            yaw_rad=0.7, yaw_center_xz=(1.5, -2.0), meta={},
        )
        pts = np.array([(0.0, 0.0), (3.0, -1.0), (1.5, -2.0)])
        back = ep.to_raw_frame(ep.to_exported_frame(pts, bo), bo)
        assert back == pytest.approx(pts)
        assert ep.to_exported_frame([(1.5, -2.0)], bo)[0] == pytest.approx((1.5, -2.0))  # center is fixed


@pytest.fixture(scope="module")
def fixture_presentation(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("msa_presentation_cli")
    inputs = load_object_inputs_from_hulls_json(FIXTURE_DIR / "scene_objects_hulls.json")
    run_bootstrap(FIXTURE_DIR, out_dir, object_inputs=inputs)
    # morning-3: the demo assertions (route >= 2.5 m through the target/boundary corridor)
    # are for the hero desk; the fixture's table stands mid-room - see TestMorning3Route
    summary = ep.run(out_dir, FIXTURE_DIR, target="table", platform_id="burger", out_dir=out_dir, min_route_length_m=0.0, require_corridor=False)
    return out_dir, summary


class TestCliOnFixture:
    def test_path_json_written_in_exported_frame_adjacent_to_target(self, fixture_presentation):
        out_dir, summary = fixture_presentation
        path_json = json.loads((out_dir / "path.json").read_text())
        assert path_json["schema"] == ep.PATH_SCHEMA
        assert path_json["target_class"] == "table"
        assert path_json["platform"]["id"] == "burger" and path_json["platform"]["radius_m"] == pytest.approx(0.1)
        assert path_json["n_waypoints"] >= 2 and path_json["path_length_m"] > 0
        assert summary["n_waypoints"] == path_json["n_waypoints"]
        pts = path_json["path_points_world"]
        assert all(len(p) == 3 for p in pts)
        assert pts[0][0] == pytest.approx(path_json["start_xy"][0], abs=0.05)
        assert pts[0][2] == pytest.approx(path_json["start_xy"][1], abs=0.05)
        # the yaw-rotated route ends next to the yaw-rotated target hull (exported frame),
        # T15h: at radius + GOAL_CLEARANCE_MARGIN_M (+ up to two cells), never closer
        hull = Polygon(path_json["target_hull_xz"])
        r = path_json["platform"]["radius_m"]
        d_goal = hull.distance(Point(pts[-1][0], pts[-1][2]))
        assert r + ep.GOAL_CLEARANCE_MARGIN_M - 1e-3 <= d_goal <= r + ep.GOAL_CLEARANCE_MARGIN_M + 2 * path_json["grid"]["resolution"] + 1e-6
        assert path_json["yaw_correction_rad"] != 0.0  # the fixture IS rotated (84.49 deg) - the round trip is exercised
        # T15h target semantics: target_point = the goal (last path point) at floor + 0.05; the centroid is target_object_point
        meta = json.loads((out_dir / "scene_meta.json").read_text())
        assert path_json["target_point"][0] == pytest.approx(pts[-1][0]) and path_json["target_point"][2] == pytest.approx(pts[-1][2])
        assert path_json["target_point"][1] == pytest.approx(meta["floor_y"] + ep.TARGET_HEIGHT_ABOVE_FLOOR_M)
        centroid = hull.centroid
        assert path_json["target_object_point"][0] == pytest.approx(centroid.x) and path_json["target_object_point"][2] == pytest.approx(centroid.y)
        assert path_json["goal_to_target_object_centroid_m"] > 0.2
        # T15h route quality: clipped to the room, every path point inside the room polygon by >= r
        q = path_json["route_quality"]
        assert q["room_clip_applied"] is True and q["path_points_inside_room_min_m"] >= r - 1e-6
        assert q["goal_clearance_m"] >= q["goal_required_clearance_m"] - 1e-9
        room = Polygon(meta["room_polygon"])
        for x, _y, z in pts:
            assert room.boundary.distance(Point(x, z)) >= r - 1e-6 and room.covers(Point(x, z)), (x, z)

    def test_presentation_usd_has_path_target_camera_lights(self, fixture_presentation):
        out_dir, summary = fixture_presentation
        stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
        assert stage.GetPrimAtPath("/World/Path").IsValid()
        assert stage.GetPrimAtPath("/World/Target").IsValid()
        assert stage.GetPrimAtPath("/World/PresentationCamera").IsA(UsdGeom.Camera)
        assert stage.GetPrimAtPath("/World/DomeLight").IsValid() and stage.GetPrimAtPath("/World/KeyLight").IsValid()
        n_path = len(UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Path")).GetPointsAttr().Get())
        assert n_path == summary["n_waypoints"]
        assert summary["usd"]["n_walls"] >= 1 and summary["usd"]["n_hulls"] >= 1
        assert summary["camera"]["framing_ok"] is True
        assert summary["camera"]["camera_outside_room_polygon"] is True
        # T15h: /World/Target = the last path point; /World/TargetObject = the target's centroid, labelled
        path_pts = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Path")).GetPointsAttr().Get()
        target = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Target")).GetOrderedXformOps()[0].Get()
        assert (target[0], target[1]) == pytest.approx((path_pts[-1][0], path_pts[-1][1]), abs=1e-6)
        marker = stage.GetPrimAtPath("/World/TargetObject")
        assert marker.IsValid() and marker.GetAttribute("cloudeye:targetObjectId").Get() == summary["target_id"]
        tox, toy, toz = summary["target_object_point"]
        assert tuple(UsdGeom.Xformable(marker).GetOrderedXformOps()[0].Get()) == pytest.approx((tox, -toz, toy), abs=1e-6)

    def test_presentation_usd_wall_collider_is_per_edge_boxes_not_a_hull(self, fixture_presentation):
        """T15h: the T15g band goes out as `wall_outline_collider/edge_i` boxes (one per
        vertex of the regularized room polygon); nothing under /World/Structure is a
        convexHull, and no collider's footprint exceeds 1.5x the band's own area."""
        from pxr import UsdPhysics

        out_dir, summary = fixture_presentation
        meta = json.loads((out_dir / "scene_meta.json").read_text())
        stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
        colliders = [
            p for p in stage.Traverse()
            if str(p.GetPath()).startswith("/World/Structure/") and p.GetTypeName() in ("Mesh", "Cube") and "/visual/" not in str(p.GetPath())
        ]
        assert colliders and all(p.GetTypeName() == "Cube" and p.HasAPI(UsdPhysics.CollisionAPI) for p in colliders)
        assert len(colliders) == len(meta["room_polygon"]) == summary["usd"]["wall_colliders"][0]["n_edges"]
        assert summary["usd"]["wall_colliders"][0] == pytest.approx({
            **summary["usd"]["wall_colliders"][0], "kind": "edge_boxes", "thickness_m": meta["wall_band_thickness_m"],
            "height_m": meta["ceiling_y"] - meta["floor_y"],
        })
        for p in stage.Traverse():
            if str(p.GetPath()).startswith("/World/Structure/") and p.HasAPI(UsdPhysics.MeshCollisionAPI):
                assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != UsdPhysics.Tokens.convexHull
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        band_area = summary["usd"]["wall_colliders"][0]["band_footprint_m2"]
        for p in colliders:
            size = cache.ComputeWorldBound(p).ComputeAlignedRange().GetSize()
            assert size[0] * size[1] <= 1.5 * band_area
            assert size[2] == pytest.approx(meta["ceiling_y"] - meta["floor_y"], abs=1e-6)
        # floor + object hulls unchanged: still CollisionAPI + MeshCollisionAPI colliders
        assert stage.GetPrimAtPath("/World/Floor").HasAPI(UsdPhysics.MeshCollisionAPI)
        hulls = [p for p in stage.Traverse() if str(p.GetPath()).endswith("/collision/hull")]
        assert hulls and all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull for p in hulls)

    def test_record_isaac_hull_ranges_and_drop_point_from_the_presentation_usd(self, fixture_presentation):
        """T15h Phase-2 helpers, run here with local pxr: every `/collision/hull` yields
        one extent, no `/visual/` part does, and the chosen drop column is free."""
        from scripts.msa.record_isaac import SPHERE_DROP_RADIUS_M, _collect_object_hull_ranges, choose_sphere_drop_point, hulls_blocking_column

        out_dir, summary = fixture_presentation
        stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
        bbcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"])
        ranges = _collect_object_hull_ranges(stage, bbcache)
        prims = {r["prim"] for r in ranges}
        assert prims == {str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith("/collision/hull")}
        assert len(prims) == summary["usd"]["n_hulls"] and not any("/visual/" in p for p in prims)
        path_xy = [(p[0], p[1]) for p in UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Path")).GetPointsAttr().Get()]
        drop = choose_sphere_drop_point(path_xy, ranges, SPHERE_DROP_RADIUS_M)
        assert not hulls_blocking_column(drop["xy"], ranges, SPHERE_DROP_RADIUS_M)
        assert drop["source"] in ("route_start", "shifted_along_path")
        assert tuple(drop["xy"]) != tuple(path_xy[-1])

    def test_stills_rendered_at_the_isaac_frame_with_route_robot_and_ring_visible(self, fixture_presentation):
        """T15i: the 3/4 still is the faithful preview (16:9 Isaac frame, opaque kept
        stubs) and it must SHOW the route: cyan path, orange robot and yellow ring
        pixels are present (T15e run2's Isaac still had 0 cyan / 0 orange)."""
        out_dir, summary = fixture_presentation
        for name in (ep.STILL_34_NAME, ep.STILL_TOP_NAME):
            png = out_dir / name
            assert png.exists() and png.stat().st_size > 0
            with Image.open(png) as img:
                assert img.size == tuple(ep.STILL_IMAGE_SIZE) == (1920, 1080)
        counts = _colour_counts(out_dir / ep.STILL_34_NAME)
        assert counts["cyan_path"] > 0 and counts["orange_robot"] > 0 and counts["yellow_ring"] > 0 and counts["red_goal"] > 0, counts
        assert summary["stills"]["image_size"] == [1920, 1080]

    def test_camera_is_visibility_verified_and_the_room_fills_the_frame(self, fixture_presentation):
        out_dir, summary = fixture_presentation
        cam = summary["camera"]
        assert cam["visibility_ok"] is True and cam["framing_ok"] is True
        assert cam["visibility"]["n_hard_occluded"] == 0 and cam["visibility"]["inside"] == []
        for group in cam["visibility"]["rings"].values():
            assert group["ok"] and group["visible"] >= 4
        assert cam["room_frame_coverage"] >= ep.ROOM_FRAME_COVERAGE_MIN - 1e-9 and cam["room_frame_coverage_ok"] is True
        assert cam["aspect"] == pytest.approx(16 / 9)
        assert 2.6 - 1e-9 <= cam["height_above_floor_m"] <= ep.WIDE_MAX_EYE_HEIGHT_M + 1e-9 and cam["height_raise_m"] >= 0.0
        assert cam["pitch_deg"] < -1.0  # no longer the saturated -1 deg far-floor rule
        assert isinstance(cam["camera_outside_room_polygon"], bool)  # a steep (moved-in) pose may sit over the room
        # orchestrator 2026-09-07: the ribbon criterion (>= 0.95 with a robot anywhere on the
        # route, covered points excluded) is solved single- or two-pose; whether the bar is
        # MET is scene geometry (a 0.4 m Husky next to a short route needs ~7.4 m of eye
        # height on the hero), so the verdict must be consistent, the pose(s) well-formed,
        # and the second pose (when used) checked with Husky's box and a clear transition.
        pv = cam["path_visibility_worst_case"]
        assert cam["path_visibility_ok"] == (pv["fraction"] >= 0.95 - 1e-9)
        assert cam["camera_ok"] == (cam["framing_ok"] and cam["visibility_ok"] and cam["path_visibility_ok"] and cam["eye_clearance"]["ok"])
        assert cam["pose_role"] in ("single_wide", "establish_and_burger")
        assert cam["eye_clearance"]["ok"] is True and cam["search_limits"]["min_width_frac"] in (0.6, 0.5)
        if cam["two_pose"]:
            assert cam["route_robot_platforms"] == ["burger"] and cam["single_pose_attempt"]["path_visibility"] < 0.95
            husky = cam["camera_husky"]
            assert husky["route_robot_platforms"] == ["husky"] and husky["pose_role"] == "husky_pass"
            assert husky["transition_clearance_ok"] is True and husky["eye_clearance"]["ok"] is True
            assert husky["height_above_floor_m"] <= ep.WIDE_MAX_EYE_HEIGHT_M + 1e-9
            assert husky["path_visibility_worst_case"]["fraction"] >= cam["single_pose_attempt"]["path_visibility"] - 1e-9
            assert husky["culled_stub_edges"] == cam["culled_stub_edges"]  # the USD renders ONE cull
            stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
            assert stage.GetPrimAtPath("/World/PresentationCameraHusky").IsA(UsdGeom.Camera)
            assert (out_dir / ep.STILL_HUSKY_NAME).exists()
        else:
            assert cam["route_robot_platforms"] == ["burger", "husky"] and cam["camera_husky"] is None
        # the aim point projects to the frame centre for the solved basis
        rel = np.asarray(cam["aim"]) - np.asarray(cam["pos"])
        assert abs(rel @ np.asarray(cam["right"])) < 1e-6 and abs(rel @ np.asarray(cam["up"])) < 1e-6
        assert cam["n_occluders"] == cam["n_stub_edges"] - len(cam["culled_stub_edges"]) + _n_opaque_parts(out_dir)
        path_json = json.loads((out_dir / "path.json").read_text())
        assert path_json["schema"] == "cloudeye.msa.presentation_path/4"
        assert path_json["presentation_camera"]["culled_stub_edges"] == cam["culled_stub_edges"]
        assert "occluders" not in path_json["presentation_camera"]  # the boxes live in the USD

    def test_presentation_usd_stub_edges_are_culled_per_the_camera(self, fixture_presentation):
        out_dir, summary = fixture_presentation
        meta = json.loads((out_dir / "scene_meta.json").read_text())
        stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
        assert not stage.GetPrimAtPath("/World/Structure/wall_outline/visual/stub").IsValid()
        edges = stage.GetPrimAtPath("/World/Structure/wall_outline/visual").GetChildren()
        assert len(edges) == len(meta["room_polygon"]) == summary["camera"]["n_stub_edges"]
        culled = set(summary["camera"]["culled_stub_edges"])
        assert 0 < len(culled) < len(edges)
        for prim in edges:
            idx = prim.GetAttribute("cloudeye:stubEdgeIndex").Get()
            vis = UsdGeom.Imageable(prim).ComputeVisibility()
            assert (vis == UsdGeom.Tokens.invisible) == (idx in culled)
            assert prim.IsActive() and prim.GetTypeName() == "Mesh"
        assert summary["usd"]["culled_stub_edges"] == sorted(culled)
        cam = stage.GetPrimAtPath("/World/PresentationCamera")
        assert sorted(cam.GetAttribute("cloudeye:culledStubEdges").Get()) == sorted(culled)
        assert cam.GetAttribute("cloudeye:cameraVisibilityOk").Get() is True
        n_boxes = len(cam.GetAttribute("cloudeye:occluderBoxes").Get()) // 15
        assert n_boxes == summary["camera"]["n_occluders"] == len(cam.GetAttribute("cloudeye:occluderIds").Get())
        # kept stub edges are occluders, culled ones are not
        ids = set(cam.GetAttribute("cloudeye:occluderIds").Get())
        assert all(f"stub_edge_{i}" not in ids for i in culled)
        assert all(f"stub_edge_{i}" in ids for i in range(len(edges)) if i not in culled)

    def test_record_isaac_phase1_writes_the_spec_for_the_goal_target(self, fixture_presentation, tmp_path):
        from scripts.msa.record_isaac import build_spec, parse_args

        out_dir, summary = fixture_presentation
        args = parse_args(["fixture", str(tmp_path / "demo.mp4"), "--target", "table",
                           "--usd-path-override", str(out_dir / "scene_presentation.usd")])
        spec = build_spec(args)
        assert spec["usd_source"] == "override" and spec["target_name"] == "table"
        assert [p["id"] for p in spec["platforms"]] == ["burger", "husky"]

    def test_record_isaac_phase1_reads_the_baked_camera(self, fixture_presentation, tmp_path):
        from scripts.msa.record_isaac import PRESENTATION_CAMERA_PATH, build_spec, parse_args

        out_dir, summary = fixture_presentation
        args = parse_args([
            "fixture", str(tmp_path / "demo.mp4"), "--target", "table",
            "--usd-path-override", str(out_dir / "scene_presentation.usd"),
        ])
        spec = build_spec(args)
        wide = spec["wide_camera"]
        assert wide["source"] == f"usd:{PRESENTATION_CAMERA_PATH}"
        px, py, pz = summary["camera"]["pos"]
        assert wide["eye"] == pytest.approx([px, -pz, py], abs=1e-6)
        assert wide["fov_v_deg"] == pytest.approx(55.0)
        assert wide["baked_backoff_m"] == pytest.approx(summary["camera"]["backoff_m"])
        # T15i: the worker's inputs ride along - Isaac frame
        ax, ay, az = summary["camera"]["aim"]
        assert wide["aim"] == pytest.approx([ax, -az, ay], abs=1e-6)
        assert wide["occluder_source"] == "usd:cloudeye:occluderBoxes" and wide["n_occluders"] == summary["camera"]["n_occluders"]
        assert wide["culled_stub_edges"] == summary["camera"]["culled_stub_edges"]
        meta = json.loads((out_dir / "scene_meta.json").read_text())
        assert len(wide["room_outline_isaac"]) == len(meta["room_polygon"])
        assert wide["baked_visibility_ok"] is True
        assert spec["schema"] == "cloudeye.record_isaac/5"
        # morning-4: the ribbon worst case + the 0.3 m eye clearance ride along for the worker
        assert wide["baked_path_visibility_worst_case"] == pytest.approx(summary["camera"]["path_visibility_worst_case"]["fraction"])
        assert wide["path_visibility_min_fraction"] == pytest.approx(0.95)
        assert wide["baked_min_eye_clearance_m"] == pytest.approx(0.3)
        assert summary["camera"]["eye_clearance"]["ok"] is True
        assert summary["camera"]["azimuth_offset_deg"] in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0)
        # the ribbon is in the USD and the GLB
        stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
        ribbon = stage.GetPrimAtPath("/World/PathRibbon")
        assert ribbon.IsA(UsdGeom.Mesh) and ribbon.GetAttribute("cloudeye:pathRibbonWidthM").Get() == pytest.approx(0.03)
        import trimesh

        glb = trimesh.load(str(out_dir / "scene_presentation.glb"), force="scene")
        assert "Path" in glb.graph.nodes_geometry and "Target" in glb.graph.nodes_geometry

    def test_worker_camera_solve_on_the_spec_matches_the_baked_camera(self, fixture_presentation, tmp_path):
        """The Phase-2 camera step, run here with the spec's numbers (no Isaac): the
        same visibility rule with the LARGER platform's ring (Husky) and height must
        accept the baked pose or move it by the documented steps only."""
        from scripts.msa.record_isaac import build_spec, framing_check_points, parse_args
        from scripts.msa import visibility as vis

        out_dir, summary = fixture_presentation
        args = parse_args(["fixture", str(tmp_path / "demo.mp4"), "--target", "table",
                           "--usd-path-override", str(out_dir / "scene_presentation.usd")])
        spec = build_spec(args)
        wide = spec["wide_camera"]
        stage = Usd.Stage.Open(str(out_dir / "scene_presentation.usd"))
        path_pts = [tuple(float(v) for v in p) for p in UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Path")).GetPointsAttr().Get()]
        floor_z = float(wide["floor_z"])
        ring_r = max(np.hypot(p["length_m"], p["width_m"]) / 2.0 for p in spec["platforms"])
        robot_h = max(float(p["height_m"]) for p in spec["platforms"])
        start = (path_pts[0][0], path_pts[0][1], floor_z)
        frame_points = framing_check_points(start, path_pts, ring_r, robot_h, ground_axes=(0, 1), up_axis=2)
        outline = [(x, y, floor_z) for x, y in wide["room_outline_isaac"]] + [(x, y, floor_z + 1.2) for x, y in wide["room_outline_isaac"]]
        vis_points = vis.visibility_check_points((start[0], start[1], floor_z + 0.03), path_pts, path_pts[-1], ring_r, robot_h,
                                                 ground_axes=(0, 1), up_axis=2)
        solved = vis.solve_visible_camera(tuple(wide["eye"]), tuple(wide["aim"]), wide["fov_v_deg"], 16 / 9, frame_points, vis_points,
                                          wide["occluders"], up_axis=2, floor_level=floor_z, outline_points=outline)
        assert solved["ok"] is True
        assert solved["backoff_m"] <= 1.0 and solved["height_raise_m"] <= 1.0  # Husky's bigger ring costs at most a small move
        assert solved["visibility"]["n_hard_occluded"] == 0


def _n_opaque_parts(out_dir: Path) -> int:
    """Non-translucent, non-floor-covering `<id>/visual/part_j` meshes of a bootstrap scene.glb."""
    from scripts.msa.visibility import LOW_OCCLUDER_MAX_TOP_M, is_translucent_class

    bo = ep.load_bootstrap_output(out_dir)
    return sum(
        1 for name, mesh in ep.iter_scene_world_meshes(bo.scene)
        if "/visual/part_" in name and not is_translucent_class(name.split("/", 1)[0]) and mesh.bounds[1][1] - bo.floor_y > LOW_OCCLUDER_MAX_TOP_M
    )


def _colour_counts(png: Path, tol: int = 40) -> dict:
    """Pixels within `tol` (L1 over RGB) of the still's marker colours."""
    a = np.asarray(Image.open(png).convert("RGB")).astype(int)

    def near(rgb):
        return int((np.abs(a - np.array(rgb)).sum(axis=2) <= tol).sum())

    return {"cyan_path": near(ep.PATH_RGB), "orange_robot": near(ep.ROBOT_RGB), "yellow_ring": near(ep.RING_RGB), "red_goal": near(ep.TARGET_RGB)}


class TestOccluderAndCheckPointHelpers:
    def test_object_part_boxes_are_oriented_not_axis_aligned(self):
        """A 1.3 x 0.6 m placeholder at 38 deg: the OBB's half extents are the part's own,
        while its AABB would be ~1.4 x 1.3 m (the fixture's stool_1 swallowed the route
        goal that way)."""
        import trimesh

        scene = trimesh.Scene()
        box = trimesh.creation.box(extents=(1.3, 0.8, 0.6))
        box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(38.0), [0, 1, 0]))
        box.apply_translation((2.0, 0.4, -1.0))
        scene.add_geometry(box, node_name="stool_1/visual/part_0")
        curtain = trimesh.creation.box(extents=(2.0, 2.7, 0.1))
        curtain.apply_translation((0.0, 1.35, 0.0))
        scene.add_geometry(curtain, node_name="curtain_0/visual/part_0")
        scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)), node_name="stool_1/collision/hull")
        rug = trimesh.creation.box(extents=(3.0, 0.04, 2.0))
        rug.apply_translation((2.0, 0.02, -1.0))
        scene.add_geometry(rug, node_name="rug_0/visual/part_0")
        boxes = ep.object_part_occluder_boxes(scene, 0.0)
        assert [b["id"] for b in boxes] == ["object:stool_1/visual/part_0"]  # translucent curtain, hull and the 4 cm rug skipped
        assert len(ep.object_part_occluder_boxes(scene)) == 2  # without a floor level the rug counts
        b = boxes[0]
        assert sorted(round(v, 3) for v in (b["half"][0], b["half"][2])) == [0.3, 0.65]
        assert b["half"][1] == pytest.approx(0.4) and b["center"] == pytest.approx((2.0, 0.4, -1.0), abs=1e-6)
        assert b["axes"][1] == (0.0, 1.0, 0.0)
        for ax in (b["axes"][0], b["axes"][2]):
            assert abs(ax[1]) < 1e-9 and np.hypot(ax[0], ax[2]) == pytest.approx(1.0)
        # the AABB of the same part is much bigger than the OBB's footprint
        lo, hi = box.bounds
        assert (hi[0] - lo[0]) * (hi[2] - lo[2]) > 1.6 * (1.3 * 0.6)

    def test_subdivided_keeps_volume_and_shrinks_triangles(self):
        import trimesh

        wall = trimesh.creation.box(extents=(4.0, 1.2, 0.1))
        out = ep._subdivided(wall, 0.5)
        assert out.volume == pytest.approx(wall.volume)
        edges = out.vertices[out.edges_unique]
        assert np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max() <= 0.5 + 1e-6
        assert len(out.faces) > len(wall.faces)


# --- T15i: the canonical hero output (skips without the scratch run) -----------------

HERO_OUT = Path("var/scratch/run-20260906/t15i_out")


@pytest.mark.skipif(not (HERO_OUT / "path.json").exists() or not (HERO_OUT / "scene_presentation.usd").exists(),
                    reason="hero T15i output not present on this machine")
class TestHeroT15iOutput:
    """The numbers T15e measured on the real GPU, as assertions on the regenerated
    hero: no phantom-filling wall collider, goal off the desk hull and off the floor
    edge with margin, every path point inside the floor by >= the robot radius, the
    sphere-drop column free of every hull, /World/Target = the route goal - and (T15i)
    a camera that actually sees the route, with the still to prove it."""

    def test_camera_sees_the_route_and_the_still_shows_it(self):
        path_json = json.loads((HERO_OUT / "path.json").read_text())
        cam = path_json["presentation_camera"]
        assert cam["visibility_ok"] is True and cam["framing_ok"] is True and cam["visibility"]["n_hard_occluded"] == 0
        assert all(g["visible"] >= 4 for g in cam["visibility"]["rings"].values())
        assert cam["room_frame_coverage"] >= ep.ROOM_FRAME_COVERAGE_MIN - 1e-9
        assert cam["height_above_floor_m"] <= 4.5 and cam["pitch_deg"] < -10.0
        assert len(cam["culled_stub_edges"]) >= 2 and cam["n_stub_edges"] == 16
        counts = _colour_counts(HERO_OUT / ep.STILL_34_NAME)
        assert counts["cyan_path"] > 100 and counts["orange_robot"] > 500 and counts["yellow_ring"] > 20 and counts["red_goal"] > 100, counts
        with Image.open(HERO_OUT / ep.STILL_34_NAME) as img:
            assert img.size == (1920, 1080)
        stage = Usd.Stage.Open(str(HERO_OUT / "scene_presentation.usd"))
        for prim in stage.GetPrimAtPath("/World/Structure/wall_outline/visual").GetChildren():
            culled = prim.GetAttribute("cloudeye:stubEdgeIndex").Get() in cam["culled_stub_edges"]
            assert (UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible) == culled
        # curtains are translucent in the USD (bound material), the bed is not
        from pxr import UsdShade

        curtain = stage.GetPrimAtPath("/World/Objects/curtain_0/visual/part_0")
        assert curtain.IsValid() and UsdShade.MaterialBindingAPI(curtain).GetDirectBindingRel().GetTargets()
        bed = stage.GetPrimAtPath("/World/Objects/bed_0/visual/part_0")
        assert bed.IsValid() and not UsdShade.MaterialBindingAPI(bed).GetDirectBindingRel().GetTargets()

    def test_route_is_inside_the_room_with_clearance(self):
        path_json = json.loads((HERO_OUT / "path.json").read_text())
        meta = json.loads((HERO_OUT / "scene_meta.json").read_text())
        r = path_json["platform"]["radius_m"]
        room = Polygon(meta["room_polygon"])
        pts = path_json["path_points_world"]
        for x, _y, z in pts:
            assert room.covers(Point(x, z)) and room.boundary.distance(Point(x, z)) >= r - 1e-6, (x, z)
        q = path_json["route_quality"]
        assert q["room_clip_applied"] and q["path_points_inside_room_min_m"] >= r - 1e-6
        assert q["goal_clearance_m"] >= r + ep.GOAL_CLEARANCE_MARGIN_M - 1e-9
        assert q["goal_distance_to_footprint_m"] >= r + ep.GOAL_CLEARANCE_MARGIN_M - 1e-9
        assert q["goal_distance_to_room_boundary_m"] >= r + ep.GOAL_CLEARANCE_MARGIN_M - 1e-9
        assert q["path_min_clearance_m"] >= r - 1e-6
        hull = Polygon(path_json["target_hull_xz"])
        assert not hull.covers(Point(path_json["target_point"][0], path_json["target_point"][2]))  # the goal is NOT inside the desk
        assert path_json["target_point"][0] == pytest.approx(pts[-1][0]) and path_json["target_point"][2] == pytest.approx(pts[-1][2])

    def test_presentation_usd_colliders_and_drop_point(self):
        from pxr import UsdPhysics

        from scripts.msa.record_isaac import SPHERE_DROP_RADIUS_M, _collect_object_hull_ranges, choose_sphere_drop_point, hulls_blocking_column

        stage = Usd.Stage.Open(str(HERO_OUT / "scene_presentation.usd"))
        meta = json.loads((HERO_OUT / "scene_meta.json").read_text())
        colliders = [
            p for p in stage.Traverse()
            if str(p.GetPath()).startswith("/World/Structure/") and p.GetTypeName() in ("Mesh", "Cube") and "/visual/" not in str(p.GetPath())
        ]
        assert len(colliders) == len(meta["room_polygon"]) and all(p.GetTypeName() == "Cube" for p in colliders)
        for p in stage.Traverse():
            if str(p.GetPath()).startswith("/World/Structure/") and p.HasAPI(UsdPhysics.MeshCollisionAPI):
                assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != UsdPhysics.Tokens.convexHull
        from scripts.msa.geometry import room_wall_band

        band_area = room_wall_band(meta["room_polygon"], meta["wall_band_thickness_m"]).area
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for p in colliders:
            size = cache.ComputeWorldBound(p).ComputeAlignedRange().GetSize()
            assert size[0] * size[1] <= 1.5 * band_area
        bbcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"])
        ranges = _collect_object_hull_ranges(stage, bbcache)
        path_xy = [(p[0], p[1]) for p in UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Path")).GetPointsAttr().Get()]
        drop = choose_sphere_drop_point(path_xy, ranges, SPHERE_DROP_RADIUS_M)
        assert not hulls_blocking_column(drop["xy"], ranges, SPHERE_DROP_RADIUS_M)
        target = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Target")).GetOrderedXformOps()[0].Get()
        assert (target[0], target[1]) == pytest.approx((path_xy[-1][0], path_xy[-1][1]), abs=1e-6)
        assert stage.GetPrimAtPath("/World/TargetObject").GetAttribute("cloudeye:targetObjectId").Get() == "desk_1"
