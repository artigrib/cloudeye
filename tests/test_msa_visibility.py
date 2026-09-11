"""T15i: scripts/msa/visibility.py - the stdlib-only camera visibility rule shared by
export_presentation (Y-up) and demo/record_isaac.py's Isaac worker (Z-up)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.usd_export import to_isaac
from scripts.msa import visibility as vis
from scripts.msa.render_perspective import compute_camera


def _wall_box_yup(x0, x1, z0, z1, h=1.2, box_id="wall"):
    return vis.aabb_box(box_id, (x0, 0.0, z0), (x1, h, z1))


class TestBoxes:
    def test_segment_hits_and_misses_an_axis_aligned_box(self):
        box = _wall_box_yup(0.0, 0.1, -2.0, 2.0)
        eye = (-3.0, 2.6, 0.0)
        assert vis.segment_hits_box(eye, (2.0, 0.03, 0.0), box)  # ray at ~1.0 m over x=0 -> through the 1.2 m wall
        assert not vis.segment_hits_box((-3.0, 4.5, 0.0), (2.0, 0.03, 0.0), box)  # higher eye clears it (1.8 m at the wall)
        assert not vis.segment_hits_box(eye, (-1.0, 0.03, 0.0), box)  # point before the wall
        assert not vis.segment_hits_box(eye, (2.0, 0.03, 8.0), box)  # passes beside its end (z 4.8 at the wall)
        # a point sitting on the box's face is not "occluded by" that box
        assert not vis.segment_hits_box(eye, (0.0, 0.03, 0.0), box)

    def test_oriented_box_matches_a_rotated_wall(self):
        # a 4 m x 0.1 m x 1.2 m wall rotated 45 deg in XZ, centred at the origin
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        box = vis.make_box("diag", (0.0, 0.6, 0.0), ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), (2.0, 0.6, 0.05))
        eye = (-2.0, 1.0, 2.0)  # on the -u? side, looking across the wall
        assert vis.segment_hits_box(eye, (2.0, 0.03, -2.0), box)
        # the same segment against the wall's AABB would also hit; a segment along the wall's own
        # direction but 0.5 m beside it must miss the OBB (and would hit the 4 x 4 AABB)
        assert not vis.segment_hits_box((-2.0 * c + 0.5 * s, 0.5, -2.0 * s - 0.5 * c), (2.0 * c + 0.5 * s, 0.5, 2.0 * s - 0.5 * c), box)

    def test_flat_round_trip_and_isaac_transform(self):
        box = vis.make_box("b", (1.0, 0.6, -2.0), ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)), (0.4, 0.6, 0.05))
        flat = vis.box_to_list(box)
        assert len(flat) == 15
        back = vis.boxes_from_flat(flat, ["b"])[0]
        assert back == box
        isaac = vis.transform_box(box, lambda p: to_isaac(*p))
        assert isaac["center"] == pytest.approx((1.0, 2.0, 0.6))
        assert isaac["axes"][1] == pytest.approx((0.0, 0.0, 1.0))  # Y-up -> Isaac Z
        assert isaac["half"] == box["half"]
        # the hit test is frame-agnostic: same verdict in both frames
        eye, p = (-3.0, 0.3, -2.0), (3.0, 0.03, -2.0)
        assert vis.segment_hits_box(eye, p, box) == vis.segment_hits_box(to_isaac(*eye), to_isaac(*p), isaac) is True
        with pytest.raises(ValueError):
            vis.boxes_from_flat([1.0] * 14)

    def test_points_inside_boxes_are_reported_not_searched(self):
        box = _wall_box_yup(0.0, 1.0, 0.0, 1.0, h=1.0, box_id="desk")
        pts = [("goal", (0.5, 0.03, 0.5)), ("start", (3.0, 0.03, 0.5))]
        inside = vis.points_inside_boxes(pts, [box])
        assert [row["label"] for row in inside] == ["goal"] and inside[0]["inside"] == "desk"
        solved = vis.solve_visible_camera((-3.0, 2.6, 0.5), (0.5, 0.0, 0.5), 55.0, 16 / 9, [], pts, [box], up_axis=1,
                                          max_backoff_m=0.5)
        assert solved["ok"] is False and solved["visibility"]["inside"] == inside
        assert solved["visibility"]["n_occluded"] == 0  # the start is visible; only the inside point fails
        assert solved["backoff_m"] == 0.0  # no pointless scan for an unsolvable point


class TestReport:
    def test_ring_points_are_soft_hard_points_are_hard(self):
        wall = _wall_box_yup(0.0, 0.1, -0.05, 0.05, h=1.2, box_id="post")  # a thin post
        eye = (-3.0, 0.5, 0.0)
        ring = [(f"ring_goal_{k}", p) for k, p in enumerate(vis.ring_points((1.0, 0.03, 0.0), 0.1, ground_axes=(0, 2)))]
        rep = vis.visibility_report(eye, ring, [wall])
        # the post hides the ring points in its shadow (the ones near z=0), not all 8
        assert 0 < rep["n_occluded"] < 8 and rep["n_hard_occluded"] == 0
        assert rep["rings"]["ring_goal"]["visible"] == 8 - rep["n_occluded"]
        assert rep["ok"] is (rep["rings"]["ring_goal"]["visible"] >= vis.RING_MIN_VISIBLE_POINTS)
        hard = vis.visibility_report(eye, [("goal", (1.0, 0.03, 0.0))], [wall])
        assert hard["ok"] is False and hard["n_hard_occluded"] == 1 and hard["occluded"][0]["by"] == "post"

    def test_visibility_check_points_layout(self):
        pts = vis.visibility_check_points((0.0, 0.03, 0.0), [(0.0, 0.03, 0.0), (1.0, 0.03, 0.0)], (1.0, 0.03, 0.0), 0.1, 0.2,
                                          ground_axes=(0, 2), up_axis=1)
        labels = [label for label, _ in pts]
        assert labels[:5] == ["route_start", "path_0", "path_1", "goal", "robot_top_start"]
        assert labels.count("robot_top_goal") == 1
        assert sum(label.startswith("ring_start_") for label in labels) == vis.RING_VISIBILITY_SEGMENTS
        assert sum(label.startswith("ring_goal_") for label in labels) == vis.RING_VISIBILITY_SEGMENTS
        assert dict(pts)["robot_top_goal"] == pytest.approx((1.0, 0.23, 0.0))
        assert vis.ring_group("ring_goal_7") == "ring_goal" and vis.ring_group("goal") is None


class TestCameraMath:
    def test_basis_matches_render_perspective_convention(self):
        room = np.array([(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)])
        pos, fwd, right, up = compute_camera(room, 0.0)
        f, r, u = vis.camera_basis(tuple(fwd), up_axis=1)
        assert f == pytest.approx(tuple(fwd)) and r == pytest.approx(tuple(right)) and u == pytest.approx(tuple(up))
        # Isaac frame: the same basis mapped through to_isaac
        fi, ri, ui = vis.camera_basis(to_isaac(*fwd), up_axis=2)
        assert ri == pytest.approx(to_isaac(*right)) and ui == pytest.approx(to_isaac(*up))

    def test_aim_point_on_floor_and_pitch(self):
        eye = (0.0, 2.0, 0.0)
        aim = vis.aim_point_on_floor(eye, (0.0, -1.0, 1.0), 0.0, up_axis=1)
        assert aim == pytest.approx((0.0, 0.0, 2.0))
        assert vis.pitch_deg((0.0, -1.0, 1.0), up_axis=1) == pytest.approx(-45.0)
        # looking up: the fallback point ahead
        assert vis.aim_point_on_floor(eye, (0.0, 0.1, 1.0), 0.0, up_axis=1, fallback_distance_m=2.0)[2] == pytest.approx(2.0 / math.hypot(0.1, 1.0))

    def test_centre_points_in_frame_centres_the_bbox(self):
        room = [(x, 0.0, z) for x in (0.0, 6.0) for z in (0.0, 4.0)] + [(x, 1.2, z) for x in (0.0, 6.0) for z in (0.0, 4.0)]
        eye = (-3.0, 2.6, -2.0)
        fwd, bbox = vis.centre_points_in_frame(eye, room, 55.0, 16 / 9, up_axis=1)
        assert bbox is not None
        assert (bbox[0] + bbox[2]) / 2 == pytest.approx(0.0, abs=1e-3) and (bbox[1] + bbox[3]) / 2 == pytest.approx(0.0, abs=1e-3)
        f, r, u = vis.camera_basis(fwd, up_axis=1)
        assert vis.projected_bbox(room, eye, f, r, u, 55.0, 16 / 9) == pytest.approx(bbox)

    def test_first_point_out_of_frame_and_outline_margin(self):
        eye, f, r, u = (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        edge = 10.0 * math.tan(math.radians(30.0))
        p = (edge * 0.95, 10.0, 0.0)  # inside a 2 % margin (usable 0.96), outside a 5 % one (0.90)
        assert vis.first_point_out_of_frame([p], eye, f, r, u, 60.0, 1.0, margin_frac=0.05) == p
        assert vis.first_point_out_of_frame([p], eye, f, r, u, 60.0, 1.0, margin_frac=0.02) is None


class TestSolver:
    """A 6 x 4 m room seen from a corner; a kept 1.2 m stub between the eye and a route
    point on the floor. From 2.6 m the ray crosses the stub below its top; raising the
    eye is what the rule tries first, and it is enough here."""

    ROOM = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]

    def _setup(self):
        eye0 = (-2.0, 2.6, 2.0)
        aim = (3.0, 0.0, 2.0)
        stub = _wall_box_yup(2.0, 2.1, 0.5, 3.5, h=1.2, box_id="stub_edge_x")  # a notch wall inside the room
        start, goal = (0.5, 0.03, 2.0), (4.0, 0.03, 2.0)  # the goal is behind the stub (ray at 0.87 m there from 2.6 m)
        vis_pts = vis.visibility_check_points(start, [start, goal], goal, 0.1, 0.2, ground_axes=(0, 2), up_axis=1)
        frame_pts = [start, goal]
        outline = [(x, 0.0, z) for x, z in self.ROOM] + [(x, 1.2, z) for x, z in self.ROOM]
        return eye0, aim, stub, vis_pts, frame_pts, outline

    def test_raises_height_before_backing_off(self):
        eye0, aim, stub, vis_pts, frame_pts, outline = self._setup()
        base = vis.visibility_report(eye0, vis_pts, [stub])
        assert base["n_hard_occluded"] > 0  # the goal is hidden from the rule's default pose
        solved = vis.solve_visible_camera(eye0, aim, 55.0, 16 / 9, frame_pts, vis_pts, [stub], up_axis=1, floor_level=0.0,
                                          outline_points=outline)
        assert solved["ok"] is True and solved["visibility"]["n_hard_occluded"] == 0
        assert solved["backoff_m"] == 0.0 and solved["height_raise_m"] > 0
        assert solved["height_raise_m"] / vis.HEIGHT_STEP_M == pytest.approx(round(solved["height_raise_m"] / vis.HEIGHT_STEP_M))
        assert solved["eye"][1] == pytest.approx(2.6 + solved["height_raise_m"])
        assert solved["eye"][0] == pytest.approx(eye0[0]) and solved["eye"][2] == pytest.approx(eye0[2])
        # re-aimed at the same floor point: the aim projects to the frame centre
        f, r, u = solved["forward"], solved["right"], solved["up"]
        rel = vis.v_sub(aim, solved["eye"])
        assert vis.v_dot(rel, r) == pytest.approx(0.0, abs=1e-9) and vis.v_dot(rel, u) == pytest.approx(0.0, abs=1e-9)
        # one height step less does not do it (the scan found the minimum)
        lower = (eye0[0], solved["eye"][1] - vis.HEIGHT_STEP_M, eye0[2])
        assert vis.visibility_report(lower, vis_pts, [stub])["ok"] is False
        # deterministic
        again = vis.solve_visible_camera(eye0, aim, 55.0, 16 / 9, frame_pts, vis_pts, [stub], up_axis=1, floor_level=0.0,
                                         outline_points=outline)
        assert again["eye"] == solved["eye"] and again["n_candidates"] == solved["n_candidates"]

    def test_backs_off_horizontally_when_no_height_works(self):
        eye0 = (-1.0, 2.6, 2.0)
        aim = (3.0, 0.0, 2.0)
        # a wide route far too close to fit: framing needs distance, nothing occludes
        pts = [(x, 0.03, z) for x in (0.0, 6.0) for z in (-1.0, 5.0)]
        vis_pts = [("goal", (3.0, 0.03, 2.0))]
        solved = vis.solve_visible_camera(eye0, aim, 55.0, 16 / 9, pts, vis_pts, [], up_axis=1, floor_level=0.0)
        assert solved["ok"] is True and solved["backoff_m"] > 0
        assert solved["eye"][2] == pytest.approx(2.0)  # backed off along the horizontal aim->eye direction (-x)
        assert solved["eye"][0] == pytest.approx(-1.0 - solved["backoff_m"])
        assert solved["backoff_m"] / vis.BACKOFF_STEP_M == pytest.approx(round(solved["backoff_m"] / vis.BACKOFF_STEP_M))

    def test_reports_best_effort_when_nothing_within_limits_works(self):
        eye0, aim = (-2.0, 2.6, 2.0), (3.0, 0.0, 2.0)
        tall = vis.aabb_box("wall_full", (1.0, 0.0, -5.0), (1.1, 10.0, 9.0))  # a full-height wall: nothing gets past
        vis_pts = [("goal", (4.0, 0.03, 2.0))]
        solved = vis.solve_visible_camera(eye0, aim, 55.0, 16 / 9, [], vis_pts, [tall], up_axis=1, floor_level=0.0, max_backoff_m=1.0)
        assert solved["ok"] is False and solved["visibility"]["n_hard_occluded"] == 1
        assert solved["n_candidates"] == len(solved["tried"]) > 1
        assert solved["backoff_m"] == 0.0  # ties broken toward the least-moved pose


class TestRouteRobotOcclusion:
    """Morning-4: robot boxes along the route, ribbon points, worst-case visibility."""

    PATH = [(0.0, 0.0, 0.03), (2.0, 0.0, 0.03)]
    PLATFORMS = [{"id": "husky", "length_m": 1.0, "width_m": 0.6, "height_m": 0.4}, {"id": "burger", "length_m": 0.14, "width_m": 0.18, "height_m": 0.19}]

    def test_samples_boxes_and_ribbon_points(self):
        sets = vis.route_robot_box_sets(self.PATH, self.PLATFORMS, floor_level=0.0, up_axis=2, step_m=0.5)
        assert len(sets) == 5 * 2  # 5 samples (0, 0.5, ..., 2.0) x 2 platforms
        husky0 = sets[0]["box"]
        assert husky0["axes"][0] == pytest.approx((1.0, 0.0, 0.0)) and husky0["half"] == pytest.approx((0.5, 0.3, 0.2))
        assert husky0["center"] == pytest.approx((0.0, 0.0, 0.2))
        pts = vis.ribbon_sample_points(self.PATH, floor_level=0.0, up_axis=2, step_m=0.1)
        assert len(pts) == 21 and all(p[2] == pytest.approx(0.03) for p in pts)
        assert vis.point_under_box_footprint((0.4, 0.1, 0.03), husky0) and not vis.point_under_box_footprint((0.6, 0.0, 0.03), husky0)

    def test_camera_along_the_route_hides_the_ribbon_behind_the_robot_a_high_side_camera_does_not(self):
        pts = vis.ribbon_sample_points(self.PATH, floor_level=0.0, up_axis=2, step_m=0.1)
        sets = vis.route_robot_box_sets(self.PATH, self.PLATFORMS[:1], floor_level=0.0, up_axis=2, step_m=0.5)
        # a low camera looking along the route from beyond its start
        along = vis.path_visibility_worst_case((-3.0, 0.0, 0.8), pts, [], sets)
        assert along["fraction"] < 0.7 and along["worst"]["platform"] == "husky" and along["worst"]["covered"] >= 6
        # a high camera off to the side: only the covered points are hidden
        side = vis.path_visibility_worst_case((1.0, -4.0, 4.0), pts, [], sets)
        assert side["fraction"] >= 0.95
        # the static occluders still count (a wall between the eye and the ribbon)
        wall = vis.aabb_box("stub", (-0.5, -1.0, 0.0), (2.5, -0.9, 1.2))
        blocked = vis.path_visibility_worst_case((1.0, -4.0, 0.5), pts, [wall], sets)
        assert blocked["static_fraction"] == 0.0 and blocked["fraction"] == 0.0

    def test_solver_uses_the_path_criterion_and_the_eye_clearance(self):
        pts = vis.ribbon_sample_points(self.PATH, floor_level=0.0, up_axis=2, step_m=0.1)
        sets = vis.route_robot_box_sets(self.PATH, self.PLATFORMS[:1], floor_level=0.0, up_axis=2, step_m=0.5)
        frame = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        labelled = [("route_start", (0.0, 0.0, 0.03)), ("goal", (2.0, 0.0, 0.03))]
        # from beyond the start, low: the robot hides the ribbon behind it -> the solver raises the eye
        solved = vis.solve_visible_camera((-3.0, 0.0, 1.0), (1.0, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [], up_axis=2,
                                          ribbon_points=pts, robot_box_sets=sets, max_backoff_m=1.0)
        assert solved["path_visibility"]["fraction"] >= 0.95 or solved["ok"] is False
        assert solved["height_above_floor_m"] > 1.0 or solved["ok"] is False
        # eye clearance: a box around the starting eye forces the solver away from it
        blob = [{"prim": "blob", "min": (-3.2, -0.2, 0.8), "max": (-2.8, 0.2, 1.2)}]
        solved2 = vis.solve_visible_camera((-3.0, 0.0, 1.0), (1.0, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [], up_axis=2,
                                           clearance_ranges=blob, min_eye_clearance_m=0.3)
        assert solved2["eye_clearance"]["ok"] is True and vis.aabb_distance(solved2["eye"], blob[0]["min"], blob[0]["max"]) >= 0.3 - 1e-9
        assert vis.aabb_distance((-3.0, 0.0, 1.0), blob[0]["min"], blob[0]["max"]) == 0.0


class TestSteepSearch:
    """Orchestrator 2026-09-07: the solver may move the eye IN toward the aim (steep
    pitch), require a room-width fraction, and reject candidates whose straight
    transition from another eye breaks the clearance."""

    def test_move_in_reaches_a_steep_pitch_and_respects_min_aim_distance(self):
        frame = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        labelled = [("goal", (2.0, 0.0, 0.03))]
        # a tall thin wall between a far low eye and the goal: only a steep look over it works
        wall = vis.aabb_box("stub", (0.9, -3.0, 0.0), (1.1, 3.0, 2.0))
        solved = vis.solve_visible_camera((-3.0, 0.0, 1.0), (1.5, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [wall], up_axis=2,
                                          max_height_above_floor_m=6.0, max_backoff_m=0.0, move_in_max_m=8.0, min_aim_distance_m=0.5)
        assert solved["ok"] is True and solved["backoff_m"] < 0.0  # it moved in
        assert solved["pitch_deg"] < -45.0
        assert math.hypot(solved["eye"][0] - 1.5, solved["eye"][1]) >= 0.5 - 1e-9

    def test_min_outline_width_and_transition_clearance(self):
        frame = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        labelled = [("goal", (2.0, 0.0, 0.03))]
        outline = [(0.0, -1.0, 0.0), (2.0, -1.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        far = vis.solve_visible_camera((-8.0, 0.0, 2.0), (1.0, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [], up_axis=2,
                                       outline_points=outline, min_outline_width_frac=0.5, max_backoff_m=0.0, move_in_max_m=8.0)
        assert far["ok"] is True and far["outline_width_frac"] >= 0.5 and far["backoff_m"] <= -7.0 + 1e-9  # had to move in 7 m for width
        # 0.6 is not reachable here: at that distance the 55 deg frame no longer holds the far frame point
        assert vis.solve_visible_camera((-8.0, 0.0, 2.0), (1.0, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [], up_axis=2,
                                        outline_points=outline, min_outline_width_frac=0.6, max_backoff_m=0.0, move_in_max_m=8.0)["ok"] is False
        # a prim right where a moved-in eye would land (width 0.2 needs the eye ~4 m closer):
        # the eye must keep 0.3 m from it AND the straight move from the old eye must stay
        # clear -> the solver picks a higher / further-in eye
        near = vis.solve_visible_camera((-8.0, 0.0, 2.0), (1.0, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [], up_axis=2,
                                        outline_points=outline, min_outline_width_frac=0.2, max_backoff_m=0.0, move_in_max_m=8.0)
        assert near["ok"] is True and near["backoff_m"] < 0.0
        ex, ey, ez = near["eye"]
        blob = [{"prim": "blob", "min": (ex - 0.2, ey - 0.3, ez - 0.5), "max": (ex + 0.2, ey + 0.3, ez + 0.2)}]
        moved = vis.solve_visible_camera((-8.0, 0.0, 2.0), (1.0, 0.0, 0.0), 55.0, 16 / 9, frame, labelled, [], up_axis=2,
                                         outline_points=outline, min_outline_width_frac=0.2, max_backoff_m=0.0, move_in_max_m=8.0,
                                         clearance_ranges=blob, transition_from=(-8.0, 0.0, 2.0))
        assert moved["ok"] is True and moved["eye_clearance"]["ok"] is True
        assert vis.camera_clearance_violation(moved["eye"], blob) is None
        assert vis.segment_clearance_violation((-8.0, 0.0, 2.0), moved["eye"], blob) is None
        assert moved["eye"] != near["eye"]


class TestTranslucentClasses:
    def test_class_from_id_and_membership(self):
        assert vis.object_class_from_id("curtain_0") == "curtain"
        assert vis.object_class_from_id("coffee maker_12/visual/part_0") == "coffee maker"
        assert vis.object_class_from_id("floor") == "floor"
        assert vis.is_translucent_class("Curtain_3") and vis.is_translucent_class("mirror") and not vis.is_translucent_class("bed_0")
        assert vis.TRANSLUCENT_OBJECT_CLASSES == frozenset({"curtain", "drape", "blind", "mirror"})
        assert vis.TRANSLUCENT_OPACITY == 0.3
