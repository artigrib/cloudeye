"""demo/record_isaac.py's Phase-1 (driver) logic only: argument parsing and platform
resolution. No Isaac Sim/pxr import happens anywhere in this file or in the code paths
it exercises - see record_isaac.py's module docstring for why that's a hard
requirement (the file must import cleanly under both this repo's venv and, later,
Isaac Sim's own bundled interpreter, which has neither pytest nor pxr in common)."""

import numpy as np
import pytest

from app.services.scene_ingest import GridMeta
from scripts.msa.record_isaac import (
    DEFAULT_TARGET_NAME,
    DEMO_PLATFORM_IDS,
    _FALLBACK_REGISTRY,
    SPHERE_DROP_RADIUS_M,
    SPHERE_DROP_SHIFT_STEP_M,
    STATIC_CAMERA_HEIGHT_ABOVE_FLOOR_M,
    TARGET_REACH_TOLERANCE_M,
    _build_legend_filters,
    _drive_toward_waypoints,
    _longest_free_run_start,
    _polyline_samples,
    _rasterize_object_obstacles,
    _slugify,
    _smootherstep,
    choose_sphere_drop_point,
    hulls_blocking_column,
    parse_args,
    resolve_platform,
    robot_footprint_ranges,
    visibility,
)

FREE, OBSTACLE, UNKNOWN = 0, 1, 2


def test_parse_args_requires_scene_id_and_output_without_worker_spec():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults():
    args = parse_args(["2e45d2b6-4208-4a04-ba2d-ecb5a2f227b2", "out.mp4"])
    assert args.target == DEFAULT_TARGET_NAME
    assert args.fps == 60
    assert args.width == 1920
    assert args.height == 1080
    assert args.establish_s == 3.0
    assert args.pass_s == 4.0
    assert args.drop_s == 1.0
    assert args.static_s == 2.0
    assert args.isaac_worker_spec is None


def test_parse_args_worker_spec_mode_does_not_require_scene_id():
    args = parse_args(["--_isaac-worker-spec", "spec.json"])
    assert args.scene_id is None
    assert str(args.isaac_worker_spec) == "spec.json"


@pytest.mark.parametrize("flag", ["--establish-s", "--pass-s", "--drop-s", "--static-s"])
def test_parse_args_rejects_non_positive_phase_durations(flag):
    with pytest.raises(SystemExit):
        parse_args(["scene-id", "out.mp4", flag, "0"])
    with pytest.raises(SystemExit):
        parse_args(["scene-id", "out.mp4", flag, "-1"])


def test_demo_platform_ids_are_burger_then_husky():
    """The whole point of the demo: the SAME planned route, driven first by the small
    reference platform, then by a much wider one - order matters (Burger must clear
    the route before Husky is expected to get blocked on it)."""
    assert DEMO_PLATFORM_IDS == ("burger", "husky")


def test_slugify():
    assert _slugify("Coffee Table") == "coffee_table"
    assert _slugify("  weird!! Name--123  ") == "weird_name_123"
    assert _slugify("") == "target"


def test_resolve_platform_default_burger_matches_real_registry():
    # app.robots IS importable in this test env (full backend venv), so this exercises
    # the real registry, not the fallback snapshot.
    platform = resolve_platform("burger")
    assert platform.display_name == "TurtleBot3 Burger"
    assert platform.radius_m == pytest.approx(0.10)
    assert platform.height_m == pytest.approx(0.192)
    assert platform.height_source == "measured"


def test_resolve_platform_unknown_id_raises():
    with pytest.raises(SystemExit):
        resolve_platform("not-a-real-platform")


def test_resolve_platform_every_registry_id_resolves():
    for platform_id in _FALLBACK_REGISTRY:
        platform = resolve_platform(platform_id)
        assert platform.radius_m > 0
        assert platform.height_m > 0
        assert platform.length_m > 0
        assert platform.width_m > 0


def test_fallback_registry_matches_real_registry_snapshot():
    # Guards the two tables from silently drifting apart - if app/robots.py changes a
    # measured value, this fails until _FALLBACK_REGISTRY is updated to match.
    for platform_id, (display_name, radius_m, height_m, length_m, width_m) in _FALLBACK_REGISTRY.items():
        real = resolve_platform(platform_id)
        assert real.display_name == display_name
        assert real.radius_m == pytest.approx(radius_m)
        assert real.height_m == pytest.approx(height_m)
        assert real.length_m == pytest.approx(length_m)
        assert real.width_m == pytest.approx(width_m)


# --- _longest_free_run_start (grid-based robot start, feat/isaac-demo-look) ----------


def test_longest_free_run_picks_the_longer_of_two_runs():
    # ix=2 has a 7-cell free run (iz 1..7 inclusive), ix=1 has a shorter one.
    cells = np.full((5, 10), OBSTACLE, dtype=np.uint8)
    cells[2, 1:8] = FREE
    cells[1, 2:5] = FREE
    meta = GridMeta(resolution=0.1, origin_x=0.0, origin_z=0.0, width=5, height=10)

    result = _longest_free_run_start(cells, meta, robot_radius_m=0.1)
    assert result is not None
    assert result["run_len_m"] == pytest.approx(0.7)
    assert result["wall_confirmed"] is True
    assert result["x"] == pytest.approx(0.25)  # (ix=2 + 0.5) * 0.1
    # wall boundary at iz=8 -> z=0.8; inset = min(0.1+0.1, 0.7*0.5) = 0.2
    assert result["z"] == pytest.approx(0.6)


def test_longest_free_run_treats_unknown_as_not_free():
    """A run must be actually mapped-and-clear (FREE), not merely unexplored
    (UNKNOWN) - an UNKNOWN cell must not extend or count toward a run."""
    cells = np.full((3, 6), UNKNOWN, dtype=np.uint8)
    cells[1, 1:5] = FREE  # the only real free run
    meta = GridMeta(resolution=0.2, origin_x=0.0, origin_z=0.0, width=3, height=6)

    result = _longest_free_run_start(cells, meta, robot_radius_m=0.1)
    assert result is not None
    assert result["run_len_m"] == pytest.approx(0.8)
    # bounded by UNKNOWN, not OBSTACLE - no confirmed wall at either end
    assert result["wall_confirmed"] is False


def test_longest_free_run_ties_break_toward_center_column():
    # Two equal-length runs (ix=0 and ix=4), grid width=5 (columns 0-4, center=2).
    cells = np.full((5, 4), OBSTACLE, dtype=np.uint8)
    cells[0, 0:3] = FREE
    cells[4, 0:3] = FREE
    cells[1, 0:3] = FREE  # closer to center (ix=1, dist=1) than either candidate above
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_z=0.0, width=5, height=4)

    result = _longest_free_run_start(cells, meta, robot_radius_m=0.1)
    assert result is not None
    assert result["x"] == pytest.approx(1.5)  # (ix=1 + 0.5) * 1.0


def test_longest_free_run_returns_none_when_grid_has_no_free_cells():
    cells = np.full((4, 4), OBSTACLE, dtype=np.uint8)
    meta = GridMeta(resolution=0.1, origin_x=0.0, origin_z=0.0, width=4, height=4)
    assert _longest_free_run_start(cells, meta, robot_radius_m=0.1) is None


# --- static-hold camera height / flythrough easing ------------------------------------


def test_static_camera_height_matches_spec():
    assert STATIC_CAMERA_HEIGHT_ABOVE_FLOOR_M == pytest.approx(1.2)


def test_smootherstep_endpoints_and_monotonic():
    assert _smootherstep(0.0) == pytest.approx(0.0)
    assert _smootherstep(1.0) == pytest.approx(1.0)
    assert _smootherstep(-1.0) == pytest.approx(0.0)  # clamped
    assert _smootherstep(2.0) == pytest.approx(1.0)  # clamped
    xs = [i / 10 for i in range(11)]
    ys = [_smootherstep(x) for x in xs]
    assert ys == sorted(ys)  # monotonically non-decreasing


def test_smootherstep_has_zero_derivative_at_endpoints_unlike_smoothstep():
    """The whole point of the change (spec: 'пролёт к роботу мягче') - smootherstep's
    slope near t=0/1 is essentially flat, cubic smoothstep's isn't."""
    eps = 1e-4
    smootherstep_slope_at_0 = _smootherstep(eps) / eps
    smoothstep_slope_at_0 = (eps * eps * (3.0 - 2.0 * eps)) / eps
    assert smootherstep_slope_at_0 < smoothstep_slope_at_0 * 0.05


# --- _rasterize_object_obstacles (grid-based start must see Object colliders too) ----


class _FakeObject:
    def __init__(self, bbox_min_x, bbox_max_x, bbox_min_z, bbox_max_z, is_fragment=False):
        self.bbox_min_x, self.bbox_max_x = bbox_min_x, bbox_max_x
        self.bbox_min_z, self.bbox_max_z = bbox_min_z, bbox_max_z
        self.is_fragment = is_fragment


def test_rasterize_object_obstacles_marks_bbox_footprint():
    """A real regression: occupancy.npy is a height-band point-density grid (see
    gpu/stage_occupancy.py), not a per-object rasterization - a chair whose seat sits
    above that band showed as clear in the grid while its actual exported collider (the
    convex hull of its own full point cloud) blocked the robot a few tenths of a meter
    into what the grid alone called a 3m+ clear run (feat/isaac-demo-look,
    02_modular_home)."""
    cells = np.full((10, 10), FREE, dtype=np.uint8)
    meta = GridMeta(resolution=0.5, origin_x=0.0, origin_z=0.0, width=10, height=10)
    obj = _FakeObject(bbox_min_x=1.0, bbox_max_x=2.0, bbox_min_z=1.0, bbox_max_z=1.6)

    _rasterize_object_obstacles(cells, meta, [obj])

    assert (cells[2:5, 2:4] == OBSTACLE).all()  # ix 1.0-2.0 -> [2,4), iz 1.0-1.6 -> [2,4)
    assert (cells[0:2, :] == FREE).all()
    assert (cells[:, 0:2] == FREE).all()


def test_rasterize_object_obstacles_skips_fragments():
    """usd_export's Isaac path always exports with include_fragments=False (see
    _build_demo_export_async) - a fragment has no collider in the actual scene, so it
    must not block a candidate run either."""
    cells = np.full((10, 10), FREE, dtype=np.uint8)
    meta = GridMeta(resolution=0.5, origin_x=0.0, origin_z=0.0, width=10, height=10)
    obj = _FakeObject(bbox_min_x=1.0, bbox_max_x=2.0, bbox_min_z=1.0, bbox_max_z=1.6, is_fragment=True)

    _rasterize_object_obstacles(cells, meta, [obj])

    assert (cells == FREE).all()


# --- _drive_toward_waypoints (path-following, feat/isaac-demo-look v2) ---------------


class _FakeRigid:
    """Minimal stand-in for isaacsim.core.prims.RigidPrim - just enough surface for
    _drive_toward_waypoints, which only ever calls get_world_poses/
    set_linear_velocities on it (duck-typed, no pxr/Isaac import needed to test)."""

    def __init__(self, xy):
        self.xy = list(xy)
        self.last_velocity = None

    def get_world_poses(self):
        return [[self.xy[0], self.xy[1], 0.0]], [[0, 0, 0, 1]]

    def set_linear_velocities(self, velocities):
        self.last_velocity = list(velocities[0])

    def step(self, dt):
        if self.last_velocity is not None:
            self.xy[0] += self.last_velocity[0] * dt
            self.xy[1] += self.last_velocity[1] * dt


def _new_state(stall_threshold_frames=6):
    return {
        "wp_idx": 0, "stall_frames": 0, "prev_xy": None,
        "blocked": False, "finished": False,
        "stall_threshold_frames": stall_threshold_frames,
    }


def test_drive_toward_waypoints_moves_toward_current_target():
    rigid = _FakeRigid((0.0, 0.0))
    state = _new_state()
    _drive_toward_waypoints(rigid, [(1.0, 0.0)], state, speed_mps=1.0)
    assert rigid.last_velocity == pytest.approx([1.0, 0.0, 0.0])
    assert not state["finished"]


def test_drive_toward_waypoints_advances_to_next_waypoint_on_arrival():
    rigid = _FakeRigid((0.99, 0.0))  # within WAYPOINT_ARRIVE_EPS_M of the first waypoint
    state = _new_state()
    _drive_toward_waypoints(rigid, [(1.0, 0.0), (1.0, 5.0)], state, speed_mps=1.0)
    assert state["wp_idx"] == 1
    assert not state["finished"]
    # velocity now aims at the SECOND waypoint
    assert rigid.last_velocity[1] > 0


def test_drive_toward_waypoints_finishes_after_last_waypoint():
    rigid = _FakeRigid((0.99, 0.0))
    state = _new_state()
    _drive_toward_waypoints(rigid, [(1.0, 0.0)], state, speed_mps=1.0)
    assert state["finished"]
    assert rigid.last_velocity == pytest.approx([0.0, 0.0, 0.0])


def test_drive_toward_waypoints_marks_blocked_when_stalled():
    rigid = _FakeRigid((0.0, 0.0))
    state = _new_state(stall_threshold_frames=3)
    for _ in range(3):
        _drive_toward_waypoints(rigid, [(10.0, 0.0)], state, speed_mps=1.0)
        # deliberately never call rigid.step() - the robot never actually moves,
        # simulating a real physics block against real geometry
    assert not state["blocked"]  # stall_frames counts from the SECOND call onward
    _drive_toward_waypoints(rigid, [(10.0, 0.0)], state, speed_mps=1.0)
    assert state["blocked"]
    assert rigid.last_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert not state["finished"]


def test_drive_toward_waypoints_no_op_once_finished_or_blocked():
    rigid = _FakeRigid((0.0, 0.0))
    state = _new_state()
    state["finished"] = True
    _drive_toward_waypoints(rigid, [(10.0, 0.0)], state, speed_mps=1.0)
    assert rigid.last_velocity is None  # never touched


# --- _build_legend_filters -------------------------------------------------------------


def test_build_legend_filters_one_row_per_class_sorted():
    class_colors = {"chair": [255, 0, 0], "bed": [0, 255, 0]}
    filters = _build_legend_filters(class_colors, base_y=100, font="/fake/font.ttf")
    assert len(filters) == 2
    assert "text='bed'" in filters[0]  # alphabetical
    assert "fontcolor=0x00ff00" in filters[0]
    assert "text='chair'" in filters[1]
    assert "fontcolor=0xff0000" in filters[1]
    assert "y=100" in filters[0]
    assert "y=122" in filters[1]


def test_build_legend_filters_caps_row_count():
    class_colors = {f"class_{i}": [i, i, i] for i in range(20)}
    filters = _build_legend_filters(class_colors, base_y=0, font="/fake/font.ttf")
    from scripts.msa.record_isaac import MAX_LEGEND_ROWS
    assert len(filters) == MAX_LEGEND_ROWS


def test_build_legend_filters_escapes_quotes_in_class_names():
    filters = _build_legend_filters({"o'brien's desk": [1, 2, 3]}, base_y=0, font="/fake/font.ttf")
    assert "'" not in filters[0].split("text=")[1].split(":fontsize")[0].strip("'’").replace("’", "")


# --- T15d: framing check / camera back-off (pure stdlib math, shared with the local still)


import math

from scripts.msa.record_isaac import (
    CAMERA_BACKOFF_STEP_M,
    camera_backoff_for_framing,
    clearance_ring_points,
    framing_check_points,
    points_in_frame,
)

_EYE = (0.0, 0.0, 0.0)
_FWD = (0.0, 1.0, 0.0)
_RIGHT = (1.0, 0.0, 0.0)
_UP = (0.0, 0.0, 1.0)


def test_points_in_frame_respects_margin_and_near_plane():
    limit = 10.0 * math.tan(math.radians(30.0)) * 0.9  # fov 60, aspect 1, 5% margin per side
    assert points_in_frame([(0.0, 10.0, 0.0)], _EYE, _FWD, _RIGHT, _UP, 60.0, 1.0)
    assert points_in_frame([(limit - 0.01, 10.0, 0.0)], _EYE, _FWD, _RIGHT, _UP, 60.0, 1.0)
    assert not points_in_frame([(limit + 0.01, 10.0, 0.0)], _EYE, _FWD, _RIGHT, _UP, 60.0, 1.0)
    assert not points_in_frame([(0.0, 10.0, limit + 0.01)], _EYE, _FWD, _RIGHT, _UP, 60.0, 1.0)
    assert not points_in_frame([(0.0, -1.0, 0.0)], _EYE, _FWD, _RIGHT, _UP, 60.0, 1.0)  # behind
    # a 2:1 aspect doubles the horizontal budget only
    assert points_in_frame([(2 * limit - 0.01, 10.0, 0.0)], _EYE, _FWD, _RIGHT, _UP, 60.0, 2.0)
    assert not points_in_frame([(0.0, 10.0, limit + 0.01)], _EYE, _FWD, _RIGHT, _UP, 60.0, 2.0)


def test_camera_backoff_is_zero_when_already_framed():
    backoff, eye = camera_backoff_for_framing(_EYE, _FWD, _RIGHT, _UP, 60.0, 1.0, [(0.0, 5.0, 0.0), (1.0, 5.0, 0.5)])
    assert backoff == 0.0
    assert eye == _EYE


def test_camera_backoff_moves_along_view_axis_until_synthetic_path_fits():
    """A 3/4-style camera 2.6 m up looking down at a route that's too wide for the
    frame: the eye backs off along -forward (look-at/pitch unchanged) by a multiple
    of the step, after which every path point, the robot start and the ring fit."""
    eye = (0.0, -3.0, 2.6)
    fwd_raw = (0.0, 3.0, -2.6)
    n = math.sqrt(sum(c * c for c in fwd_raw))
    fwd = tuple(c / n for c in fwd_raw)
    right = (1.0, 0.0, 0.0)
    up = (0.0, fwd[2] * -1.0, fwd[1])  # cross(right, forward) for right=(1,0,0)
    path = [(-6.0, 0.0, 0.0), (-3.0, 2.0, 0.0), (0.0, 4.0, 0.0), (6.0, 5.0, 0.0)]
    points = framing_check_points(path[0], path, 0.1, 0.2, ground_axes=(0, 1), up_axis=2)
    assert not points_in_frame(points, eye, fwd, right, up, 55.0, 16 / 9)

    backoff, new_eye = camera_backoff_for_framing(eye, fwd, right, up, 55.0, 16 / 9, points)
    assert backoff > 0
    assert backoff / CAMERA_BACKOFF_STEP_M == pytest.approx(round(backoff / CAMERA_BACKOFF_STEP_M))
    assert new_eye == pytest.approx(tuple(e - f * backoff for e, f in zip(eye, fwd)))
    assert points_in_frame(points, new_eye, fwd, right, up, 55.0, 16 / 9)
    # one step less does NOT fit - the scan found the minimum
    one_less = tuple(e - f * (backoff - CAMERA_BACKOFF_STEP_M) for e, f in zip(eye, fwd))
    assert not points_in_frame(points, one_less, fwd, right, up, 55.0, 16 / 9)
    # deterministic
    assert camera_backoff_for_framing(eye, fwd, right, up, 55.0, 16 / 9, points) == (backoff, new_eye)


# --- T15h: sphere-drop point = route start, verified free of object hulls -----------


def _hull(prim, x0, y0, x1, y1, z0=0.0, z1=0.8):
    return {"prim": prim, "min": (x0, y0, z0), "max": (x1, y1, z1)}


def test_target_reach_tolerance_and_drop_constants_are_the_documented_numbers():
    assert TARGET_REACH_TOLERANCE_M == 0.3
    assert SPHERE_DROP_RADIUS_M == 0.1  # same as tools/isaac_validate.py's default sphere
    assert SPHERE_DROP_SHIFT_STEP_M == 0.05


def test_hulls_blocking_column_grows_extents_by_the_sphere_radius():
    hulls = [_hull("/World/Objects/desk_0/collision/hull", 1.0, 1.0, 2.0, 1.5)]
    assert hulls_blocking_column((1.5, 1.2), hulls, 0.1) == ["/World/Objects/desk_0/collision/hull"]
    assert hulls_blocking_column((0.95, 1.2), hulls, 0.1) == ["/World/Objects/desk_0/collision/hull"]  # within r of the edge
    assert hulls_blocking_column((0.85, 1.2), hulls, 0.1) == []
    assert hulls_blocking_column((1.5, 1.65), hulls, 0.1) == []


def test_choose_sphere_drop_point_uses_the_route_start_when_its_column_is_free():
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    hulls = [_hull("/World/Objects/lamp_0/collision/hull", 0.8, 0.8, 1.3, 1.3, 1.8, 2.5)]  # over the GOAL, not the start
    drop = choose_sphere_drop_point(path, hulls, 0.1)
    assert drop["source"] == "route_start" and drop["xy"] == [0.0, 0.0]
    assert drop["shift_along_path_m"] == 0.0 and drop["blocking_hulls_at_start"] == []
    # the goal itself would have been blocked - the very T15e case
    assert hulls_blocking_column(path[-1], hulls, 0.1) == ["/World/Objects/lamp_0/collision/hull"]


def test_choose_sphere_drop_point_shifts_along_the_path_until_the_column_is_free():
    path = [(0.0, 0.0), (2.0, 0.0)]
    hulls = [_hull("/World/Objects/chair_0/collision/hull", -0.3, -0.3, 0.4, 0.3)]
    drop = choose_sphere_drop_point(path, hulls, 0.1, step_m=0.05)
    assert drop["source"] == "shifted_along_path"
    assert drop["blocking_hulls_at_start"] == ["/World/Objects/chair_0/collision/hull"]
    # first sample past x = 0.4 + 0.1 (extent + radius), on the 5 cm lattice
    assert drop["xy"] == pytest.approx([0.55, 0.0]) and drop["shift_along_path_m"] == pytest.approx(0.55)
    assert not hulls_blocking_column(drop["xy"], hulls, 0.1)
    assert drop["xy"] != list(path[-1])  # never the goal


def test_choose_sphere_drop_point_falls_back_to_start_when_nothing_on_the_path_is_free():
    path = [(0.0, 0.0), (1.0, 0.0)]
    hulls = [_hull("/World/Objects/bed_0/collision/hull", -1.0, -1.0, 2.0, 1.0)]
    drop = choose_sphere_drop_point(path, hulls, 0.1)
    assert drop["source"].startswith("route_start_unverified") and drop["xy"] == [0.0, 0.0]
    assert drop["blocking_hulls_at_start"] == ["/World/Objects/bed_0/collision/hull"]


# --- T15i: drop point chosen at drop time against both robots' current footprints ----


def test_robots_blocking_point_uses_half_diagonal_plus_margin():
    from scripts.msa.record_isaac import SPHERE_DROP_ROBOT_MARGIN_M, robots_blocking_point

    husky = {"name": "husky", "xy": [0.0, 0.0], "half_diagonal_m": 0.595}
    assert SPHERE_DROP_ROBOT_MARGIN_M == 0.15
    assert robots_blocking_point((0.7, 0.0), [husky]) == ["husky"]  # 0.7 < 0.595 + 0.15
    assert robots_blocking_point((0.75, 0.0), [husky]) == []


def test_choose_sphere_drop_point_at_drop_time_avoids_the_blocked_robot():
    """T15e run2: the start lay 0.045 m behind blocked Husky's rear face. With the
    robots' current footprints given, the first route point clear of every hull AND
    every robot wins, logged as `route_point_clear_of_robots`."""
    path = [(0.0, 0.0), (3.0, 0.0)]
    husky = {"name": "husky", "xy": [0.5, 0.0], "half_diagonal_m": 0.595}  # parked over the start
    burger = {"name": "burger", "xy": [20.0, 20.0], "half_diagonal_m": 0.113}  # parked far away
    drop = choose_sphere_drop_point(path, [], 0.1, step_m=0.05, robot_footprints=[burger, husky])
    assert drop["source"] == "route_point_clear_of_robots"
    assert drop["blocking_robots_at_start"] == ["husky"] and drop["blocking_hulls_at_start"] == []
    # first 5 cm sample past 0.5 + 0.595 + 0.15 = 1.245
    assert drop["xy"] == pytest.approx([1.25, 0.0]) and drop["shift_along_path_m"] == pytest.approx(1.25)
    assert drop["robots_checked"] == ["burger", "husky"]
    # hulls still count: a hull over that stretch pushes it further
    hull = _hull("/World/Objects/chair_0/collision/hull", 1.0, -0.3, 1.6, 0.3)
    drop2 = choose_sphere_drop_point(path, [hull], 0.1, step_m=0.05, robot_footprints=[burger, husky])
    assert drop2["xy"][0] == pytest.approx(1.75) and drop2["source"] == "route_point_clear_of_robots"


def test_choose_sphere_drop_point_at_drop_time_keeps_the_start_when_it_is_clear():
    path = [(0.0, 0.0), (1.0, 0.0)]
    husky = {"name": "husky", "xy": [3.0, 0.0], "half_diagonal_m": 0.595}
    drop = choose_sphere_drop_point(path, [], 0.1, robot_footprints=[husky])
    assert drop["source"] == "route_point_clear_of_robots" and drop["xy"] == [0.0, 0.0] and drop["shift_along_path_m"] == 0.0
    assert drop["blocking_robots_at_start"] == []


def test_choose_sphere_drop_point_at_drop_time_reports_when_nothing_is_clear():
    path = [(0.0, 0.0), (0.5, 0.0)]
    husky = {"name": "husky", "xy": [0.25, 0.0], "half_diagonal_m": 0.595}
    drop = choose_sphere_drop_point(path, [], 0.1, robot_footprints=[husky])
    assert drop["source"].startswith("route_start_unverified") and drop["xy"] == [0.0, 0.0]
    assert drop["blocking_robots_at_start"] == ["husky"]


def test_choose_sphere_drop_point_without_robots_is_the_t15h_behaviour():
    path = [(0.0, 0.0), (1.0, 0.0)]
    assert choose_sphere_drop_point(path, [], 0.1)["source"] == "route_start"
    assert choose_sphere_drop_point(path, [], 0.1, robot_footprints=None)["source"] == "route_start"


# --- morning-5: off-route drop point, close-shot eye, camera clearance ---------------


_ROOM = [(-1.0, -1.0), (4.0, -1.0), (4.0, 3.0), (-1.0, 3.0)]


def test_drop_point_goes_off_route_when_a_robot_parks_on_the_whole_route():
    """T15e run3: Husky's 0.745 m exclusion covered the whole 0.89 m route and the
    chooser fell back to the unverified start. Now: the nearest free floor cell to
    the goal (outline eroded 0.2 m, clear of hulls and both robots)."""
    path = [(0.0, 0.0), (0.8, 0.0)]
    husky = {"name": "husky", "xy": [0.4, 0.0], "half_diagonal_m": 0.595}
    burger = {"name": "burger", "xy": [20.0, 20.0], "half_diagonal_m": 0.113}
    drop = choose_sphere_drop_point(path, [], 0.1, robot_footprints=[burger, husky], room_outline_xy=_ROOM)
    assert drop["source"] == "off_route_free_cell_nearest_goal"
    x, y = drop["xy"]
    assert math.hypot(x - 0.4, y - 0.0) >= 0.745 - 1e-9  # clear of Husky
    assert -0.8 <= x <= 3.8 and -0.8 <= y <= 2.8  # inside the eroded room
    assert drop["off_route"]["n_free_cells"] > 0 and drop["blocking_robots_at_start"] == ["husky"]
    # nearest such cell to the goal (0.8, 0): just past the exclusion circle
    assert math.hypot(x - 0.8, y) <= 0.5
    # a hull over the free area pushes it elsewhere, never unverified
    hull = _hull("/World/Objects/bed_0/collision/hull", 0.8, -0.6, 1.6, 0.6)
    drop2 = choose_sphere_drop_point(path, [hull], 0.1, robot_footprints=[burger, husky], room_outline_xy=_ROOM)
    assert drop2["source"] == "off_route_free_cell_nearest_goal" and not hulls_blocking_column(drop2["xy"], [hull], 0.1)
    # without an outline the T15i fallback still names itself unverified
    assert choose_sphere_drop_point(path, [], 0.1, robot_footprints=[burger, husky])["source"].startswith("route_start_unverified")


def test_point_in_polygon_and_boundary_distance():
    from scripts.msa.record_isaac import distance_to_polygon_boundary_xy, point_in_polygon_xy

    assert point_in_polygon_xy((0.0, 0.0), _ROOM) and not point_in_polygon_xy((5.0, 0.0), _ROOM)
    assert distance_to_polygon_boundary_xy((0.0, 0.0), _ROOM) == pytest.approx(1.0)
    assert distance_to_polygon_boundary_xy((3.9, 2.9), _ROOM) == pytest.approx(0.1)


def test_close_shot_eye_avoids_the_robot_and_sees_the_drop_point():
    """A robot parked right where the old fixed back-off put the eye: the chooser
    must pick an eye outside its footprint (+0.3 m clearance) that sees the drop
    point and the rest area, and the eye must move away from the robot."""
    from scripts.msa.record_isaac import choose_close_shot_eye

    drop = (0.0, 0.0)
    husky = {"name": "husky", "xy": [0.0, -1.2], "half_diagonal_m": 0.595, "height_m": 0.4}  # south of the drop point
    res = choose_close_shot_eye(drop, 0.0, occluders=[], hull_ranges=[], robot_footprints=[husky], room_outline_xy=_ROOM)
    assert res["ok"] is True and res["n_valid"] > 0
    ex, ey, ez = res["eye"]
    assert math.hypot(ex - 0.0, ey + 1.2) >= 0.595 + 0.15 - 1e-9  # not inside Husky's footprint + margin
    assert res["clearance_violation"] is None and res["eye_blocked_by"] == []
    assert res["radius_m"] == pytest.approx(1.5) and 1.2 <= res["height_m"] <= 2.6  # the closest valid ring wins
    assert res["look_at"] == pytest.approx([0.0, 0.0, 0.3])
    # T15e run3's eye sat over Husky: an eye 0.1 m above the box, or inside it, is a clearance violation
    ranges = robot_footprint_ranges([husky], 0.0)
    assert visibility.camera_clearance_violation((0.0, -1.2, 0.5), ranges)["distance_m"] == pytest.approx(0.1)
    assert visibility.camera_clearance_violation((0.0, -1.2, 0.2), ranges)["inside"] is True
    assert res["rest_points_inside_boxes"] == ["rest_6"]  # the rest point 0.5 m south lies inside Husky's margin box


def test_camera_clearance_check_fails_at_0_2_m_and_passes_when_clear():
    """Addendum: an eye 0.2 m from a robot box fails the 0.3 m clearance check; a
    pose 0.5 m away passes."""
    husky = {"name": "husky", "xy": [1.0, 1.0], "half_diagonal_m": 0.5, "height_m": 0.4}
    ranges = robot_footprint_ranges([husky], 0.0)
    assert ranges[0]["min"] == (0.5, 0.5, 0.0) and ranges[0]["max"] == (1.5, 1.5, 0.4)
    v = visibility.camera_clearance_violation((1.7, 1.0, 0.2), ranges)  # 0.2 m east of the box
    assert v is not None and v["prim"] == "robot:husky" and v["distance_m"] == pytest.approx(0.2) and v["inside"] is False
    assert visibility.camera_clearance_violation((1.0, 1.0, 0.2), ranges)["inside"] is True
    assert visibility.camera_clearance_violation((2.0, 1.0, 0.2), ranges) is None  # 0.5 m away
    assert visibility.camera_clearance_violation((1.0, 1.0, 0.75), ranges) is None  # 0.35 m above
    # the transition segment is checked along its length
    assert visibility.segment_clearance_violation((3.0, 1.0, 0.5), (-1.0, 1.0, 0.5), ranges) is not None  # passes 0.1 m over the box
    assert visibility.segment_clearance_violation((3.0, 1.0, 1.0), (-1.0, 1.0, 1.0), ranges) is None  # 0.6 m over it
    assert visibility.segment_clearance_violation((3.0, 1.0, 0.5), (3.0, -2.0, 0.5), ranges) is None
    # the close-shot chooser refuses a candidate whose eye or transition is within 0.3 m of a prim
    from scripts.msa.record_isaac import choose_close_shot_eye

    wall = {"prim": "/World/Structure/wall_outline_collider/edge_0", "min": (-2.0, 2.0, 0.0), "max": (2.0, 2.1, 3.0)}
    res = choose_close_shot_eye((0.0, 0.0), 0.0, occluders=[], hull_ranges=[], robot_footprints=[],
                                clearance_ranges=[wall], transition_from=(0.0, -3.0, 2.0))
    assert res["ok"] is True and visibility.camera_clearance_violation(tuple(res["eye"]), [wall]) is None
    assert visibility.segment_clearance_violation((0.0, -3.0, 2.0), tuple(res["eye"]), [wall]) is None


def test_camera_clearance_violation_exception_carries_the_frame_record():
    from scripts.msa.record_isaac import CameraClearanceViolation

    exc = CameraClearanceViolation({"frame": 700, "camera": [0, 0, 1], "prim": "robot:husky", "distance_m": 0.2, "min_clearance_m": 0.3})
    assert exc.violation["frame"] == 700 and "robot:husky" in str(exc)


# --- T15i: ring / body visibility schedule -------------------------------------------


def test_proxy_visibility_schedule_shows_exactly_one_robot_and_its_ring():
    from scripts.msa.record_isaac import proxy_visibility_schedule

    est, pas = 180, 240
    b0, h0, d0 = est, est + pas, est + 2 * pas
    for frame in (0, b0 - 1, b0, h0 - 1):
        s = proxy_visibility_schedule(frame, b0, h0)
        assert s == {"burger": {"body": True, "ring": True}, "husky": {"body": False, "ring": False}}, frame
    for frame in (h0, h0 + 17, d0 - 1, d0, d0 + 100):
        s = proxy_visibility_schedule(frame, b0, h0)
        assert s == {"burger": {"body": False, "ring": False}, "husky": {"body": True, "ring": True}}, frame
    # never both rings, never neither (a ring is visible in every frame)
    for frame in range(0, d0 + 200, 7):
        s = proxy_visibility_schedule(frame, b0, h0)
        assert s["burger"]["ring"] != s["husky"]["ring"]
        assert s["burger"]["ring"] == s["burger"]["body"] and s["husky"]["ring"] == s["husky"]["body"]


def test_visibility_module_is_the_shared_one_and_points_in_frame_delegates():
    from scripts.msa import record_isaac
    from scripts.msa import visibility

    assert record_isaac.visibility is visibility
    assert record_isaac.points_in_frame([(0.0, 5.0, 0.0)], _EYE, _FWD, _RIGHT, _UP, 60.0, 1.0) is True
    assert record_isaac.SCHEMA == "cloudeye.record_isaac/5"


def test_polyline_samples_include_vertices_and_cumulative_distance():
    samples = _polyline_samples([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)], 0.05)
    xy = [(round(x, 6), round(y, 6)) for x, y, _ in samples]
    assert xy == [(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.1, 0.05), (0.1, 0.1)]
    assert [round(d, 6) for _, _, d in samples] == [0.0, 0.05, 0.1, 0.15, 0.2]


def test_framing_check_points_cover_start_ring_lifted_ring_and_path():
    path = [(1.0, 2.0, 0.0), (2.0, 3.0, 0.0)]
    pts = framing_check_points((1.0, 2.0, 0.0), path, 0.25, 0.4, ground_axes=(0, 1), up_axis=2)
    ring = clearance_ring_points((1.0, 2.0, 0.0), 0.25, ground_axes=(0, 1))
    assert len(pts) == len(path) + 1 + 2 * len(ring)
    for p in ring:
        assert math.hypot(p[0] - 1.0, p[1] - 2.0) == pytest.approx(0.25)
        assert p[2] == 0.0
    assert any(p[2] == pytest.approx(0.4) for p in pts)
    # Y-up variant: ring in XZ, lift along Y
    ring_yup = clearance_ring_points((1.0, 0.0, 2.0), 0.25, ground_axes=(0, 2))
    assert all(p[1] == 0.0 for p in ring_yup)
    assert math.hypot(ring_yup[3][0] - 1.0, ring_yup[3][2] - 2.0) == pytest.approx(0.25)
