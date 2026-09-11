"""Tests for app/services/footprint.py: the rectangle-footprint SE(2) planner.

Covers, on synthetic occupancy grids: the rotated-rectangle mask itself, the 8-
connected/no-corner-cutting SE(2) A*, and the three-level reachability classification
(any_orientation / requires_alignment / not_reachable) - including the concrete case
that motivates having three levels at all: a gap too narrow for the robot at every
heading except a couple of near-perpendicular ones. Plus one check against the real
02_modular_home fixture (see scene_modular_home_dir), mirroring how
test_pathfinding_reachability_cache.py checks the circle planner against
scene_horizontal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services import footprint as fp
from app.services import scene_ingest
from app.services.pathfinding import (
    FREE,
    OBSTACLE,
    NoPathError,
    ObjectFootprint,
    OccupancyGrid,
    build_cost_grid,
)
from app.services.scene_ingest import GridMeta


def make_grid(rows: list[str], *, resolution: float = 0.1) -> tuple[np.ndarray, GridMeta]:
    """Same convention as test_pathfinding_connectivity.py's helper: rows are Z
    (top=iz=0), columns are X, one character per cell ('.' free, '#' obstacle)."""
    height = len(rows)
    width = len(rows[0])
    char_to_val = {".": FREE, "#": OBSTACLE}
    cells = np.zeros((width, height), dtype=np.uint8)
    for iz, row in enumerate(rows):
        for ix, ch in enumerate(row):
            cells[ix, iz] = char_to_val[ch]
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)
    return cells, meta


def object_at(meta: GridMeta, x: float, z: float, obj_id_suffix: str = "") -> tuple[str, ObjectFootprint]:
    return f"obj{obj_id_suffix}", ObjectFootprint(x=x, z=z, bbox_min_x=x, bbox_min_z=z, bbox_max_x=x, bbox_max_z=z)


# --- Rectangle mask ------------------------------------------------------------------


def test_footprint_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        fp.RobotFootprint(length_m=0.0, width_m=0.2)
    with pytest.raises(ValueError):
        fp.RobotFootprint(length_m=0.2, width_m=-0.1)


def test_mask_offsets_include_the_center_cell():
    offsets = fp.footprint_cell_offsets(0.3, 0.2, 0.1, theta_rad=0.0)
    assert (0, 0) in offsets


def test_mask_extent_grows_with_footprint_size():
    small = set(fp.footprint_cell_offsets(0.1, 0.1, 0.1, theta_rad=0.0))
    large = set(fp.footprint_cell_offsets(0.6, 0.6, 0.1, theta_rad=0.0))
    assert len(large) > len(small)


def test_mask_rotated_90_degrees_swaps_length_and_width_footprint():
    # A long-and-narrow rectangle (length along x) at theta=0 extends further in x
    # than z; rotated a quarter turn (index 4 of 16 == 90 degrees) it should extend
    # further in z than x instead.
    mask_cache = fp.FootprintMaskCache(fp.RobotFootprint(length_m=0.5, width_m=0.1), resolution_m=0.05)
    theta0 = mask_cache.masks[0]
    theta90 = mask_cache.masks[4]
    max_dx_0 = max(abs(dx) for dx, _ in theta0)
    max_dz_0 = max(abs(dz) for _, dz in theta0)
    max_dx_90 = max(abs(dx) for dx, _ in theta90)
    max_dz_90 = max(abs(dz) for _, dz in theta90)
    assert max_dx_0 > max_dz_0
    assert max_dz_90 > max_dx_90


def test_is_free_false_when_any_covered_cell_is_blocked():
    cells, meta = make_grid(["....", ".#..", "....", "...."])
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.3)  # big enough to cover (1,1)
    mask_cache = fp.FootprintMaskCache(footprint, meta.resolution)
    checker = fp.PoseChecker(mask_cache, fp.blocked_cells(cost), meta.width, meta.height)
    assert checker.is_free(0, 0, 0) is False  # centered near the obstacle - mask covers it


def test_is_free_out_of_bounds_counts_as_blocked():
    cells, meta = make_grid(["...", "...", "..."])
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.5, width_m=0.5)
    mask_cache = fp.FootprintMaskCache(footprint, meta.resolution)
    checker = fp.PoseChecker(mask_cache, fp.blocked_cells(cost), meta.width, meta.height)
    assert checker.is_free(0, 0, 0) is False  # footprint would hang off the grid edge


# --- Three-level reachability classification (per cell) ------------------------------


def _wall_with_gap(gap_cols: int, *, width: int = 20, height: int = 14, wall_iz: int = 7, resolution: float = 0.1):
    cells = np.zeros((width, height), dtype=np.uint8)
    cells[:, wall_iz] = OBSTACLE
    start = (width // 2) - (gap_cols // 2)
    for i in range(gap_cols):
        cells[start + i, wall_iz] = FREE
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)
    return cells, meta, wall_iz


def test_classify_cell_any_orientation_in_open_room():
    cells, meta = make_grid(["......", "......", "......", "......", "......"])
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.2)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    assert checker.classify_cell(3, 2) == "any_orientation"


def test_classify_cell_requires_alignment_in_a_gap_that_only_fits_lengthwise():
    # 0.3m gap: exactly the footprint's length_m, so only headings close to a quarter
    # turn (length axis perpendicular to the wall) let the narrower width_m dimension
    # present to the gap instead.
    cells, meta, wall_iz = _wall_with_gap(gap_cols=3)
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    ix, iz = meta.width // 2, wall_iz
    count = checker.free_theta_count(ix, iz)
    assert checker.classify_cell(ix, iz) == "requires_alignment"
    assert 0 < count < fp.THETA_STEPS
    # The orientations that DO work must be near a quarter turn (index 4 or 12 of 16),
    # not near theta=0/8 (length axis parallel to the gap opening).
    free_thetas = {k for k in range(fp.THETA_STEPS) if checker.is_free(ix, iz, k)}
    assert 4 in free_thetas
    assert 12 in free_thetas
    assert 0 not in free_thetas
    assert 8 not in free_thetas


def test_classify_cell_not_reachable_when_gap_too_narrow_for_any_heading():
    cells, meta, wall_iz = _wall_with_gap(gap_cols=1)  # 0.1m gap
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    ix, iz = meta.width // 2, wall_iz
    assert checker.classify_cell(ix, iz) == "not_reachable"
    assert checker.free_theta_count(ix, iz) == 0


def test_classify_cell_any_orientation_once_gap_is_wide_enough():
    cells, meta, wall_iz = _wall_with_gap(gap_cols=5)  # 0.5m gap
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    ix, iz = meta.width // 2, wall_iz
    assert checker.classify_cell(ix, iz) == "any_orientation"


# --- SE(2) A* --------------------------------------------------------------------------


def test_astar_se2_finds_path_in_open_room():
    cells, meta = make_grid(["." * 14] * 10)
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.2, width_m=0.2)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    start = (3, 3, 0)
    goal = (10, 6)
    path = fp.astar_se2(cost, checker, start, goal)
    assert path[0] == start
    assert path[-1][:2] == goal


def test_astar_se2_raises_when_start_pose_collides():
    cells, meta = make_grid([".#", "..", ".."])
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.3)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    with pytest.raises(NoPathError):
        fp.astar_se2(cost, checker, (0, 0, 0), (0, 2))


def test_astar_se2_uses_rotation_to_pass_a_heading_dependent_gap():
    """The concrete "requires alignment" scenario end-to-end: a straight-line path at
    theta=0 is blocked by the gap, but astar_se2 finds a path by rotating to one of
    the free headings before crossing, then rotating back if needed."""
    cells, meta, wall_iz = _wall_with_gap(gap_cols=3, width=14, height=14, wall_iz=7)
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    start = (meta.width // 2, 2, 0)  # theta=0, well before the wall
    goal = (meta.width // 2, 11)  # past the wall
    path = fp.astar_se2(cost, checker, start, goal)
    assert path[-1][:2] == goal
    # somewhere along the path the robot must be at one of the gap-passable headings
    gap_states = [p for p in path if p[1] == wall_iz]
    assert gap_states, "path must actually cross the wall row"
    assert all(checker.is_free(*state) for state in gap_states)


def test_astar_se2_no_path_when_gap_impassable_at_every_heading():
    cells, meta, wall_iz = _wall_with_gap(gap_cols=1, width=14, height=14, wall_iz=7)
    cost = build_cost_grid(cells)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    checker = fp.PoseChecker(fp.FootprintMaskCache(footprint, meta.resolution), fp.blocked_cells(cost), meta.width, meta.height)
    start = (meta.width // 2, 2, 0)
    goal = (meta.width // 2, 11)
    with pytest.raises(NoPathError):
        fp.astar_se2(cost, checker, start, goal)


# --- compute_footprint_reachability (per object, 3-level) -----------------------------


def test_compute_footprint_reachability_any_orientation_object_in_open_room():
    cells, meta = make_grid(["." * 14] * 14)
    grid = OccupancyGrid(cells=cells, meta=meta)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.2)
    start_world = (0.45, 0.45)
    objects = [object_at(meta, 0.95, 0.95)]
    result = fp.compute_footprint_reachability(grid, start_world, 0.0, objects, footprint)
    assert result.any_orientation_ids == {"obj"}
    assert result.requires_alignment_ids == frozenset()
    assert result.unreachable_reasons == {}
    assert result.reachable_theta_counts["obj"] == fp.THETA_STEPS


def test_compute_footprint_reachability_requires_alignment_object_beyond_a_tight_gap():
    cells, meta, wall_iz = _wall_with_gap(gap_cols=3, width=14, height=14, wall_iz=7)
    grid = OccupancyGrid(cells=cells, meta=meta)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    start_world = (0.7, 0.2)
    goal_x, goal_z = 0.7, 1.15
    objects = [object_at(meta, goal_x, goal_z)]
    result = fp.compute_footprint_reachability(grid, start_world, 0.0, objects, footprint)
    assert result.requires_alignment_ids == {"obj"}
    assert result.any_orientation_ids == frozenset()
    assert result.unreachable_reasons == {}
    assert 0 < result.reachable_theta_counts["obj"] < fp.THETA_STEPS


def test_compute_footprint_reachability_not_reachable_when_gap_seals_everything():
    cells, meta, wall_iz = _wall_with_gap(gap_cols=1, width=14, height=14, wall_iz=7)
    grid = OccupancyGrid(cells=cells, meta=meta)
    footprint = fp.RobotFootprint(length_m=0.3, width_m=0.12)
    start_world = (0.7, 0.2)
    objects = [object_at(meta, 0.7, 1.15)]
    result = fp.compute_footprint_reachability(grid, start_world, 0.0, objects, footprint)
    assert result.reachable_ids == frozenset()
    assert result.unreachable_reasons["obj"] in ("disconnected", "robot_does_not_fit")


def test_compute_footprint_reachability_outside_grid_object_does_not_abort_others():
    cells, meta = make_grid(["." * 10] * 10)
    grid = OccupancyGrid(cells=cells, meta=meta)
    footprint = fp.RobotFootprint(length_m=0.2, width_m=0.2)
    objects = [
        object_at(meta, 0.55, 0.55, "_in"),
        ("obj_out", ObjectFootprint(x=999.0, z=999.0, bbox_min_x=999.0, bbox_min_z=999.0, bbox_max_x=999.0, bbox_max_z=999.0)),
    ]
    result = fp.compute_footprint_reachability(grid, (0.35, 0.35), 0.0, objects, footprint)
    assert "obj_in" in result.reachable_ids
    assert result.unreachable_reasons["obj_out"] == "outside_grid"


def test_three_levels_are_monotonic_as_footprint_grows_on_the_same_gap():
    """A sanity property tying all three levels together on one fixture: as the
    footprint's width grows past the gap's tolerance, the SAME goal cell's
    reachability can only get worse (any_orientation -> requires_alignment ->
    not-reachable), never better - a wider robot never gains an orientation a
    narrower one lacked, for the same gap."""
    cells, meta, wall_iz = _wall_with_gap(gap_cols=5, width=14, height=14, wall_iz=7)  # 0.5m gap
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = (0.7, 0.2)
    goal = object_at(meta, 0.7, 1.15)

    widths = [0.1, 0.3, 0.45, 0.6]  # last one should no longer fit at gap_cols=5+margin
    levels = []
    for width_m in widths:
        footprint = fp.RobotFootprint(length_m=0.3, width_m=width_m)
        result = fp.compute_footprint_reachability(grid, start_world, 0.0, [goal], footprint)
        if "obj" in result.any_orientation_ids:
            levels.append(2)
        elif "obj" in result.requires_alignment_ids:
            levels.append(1)
        else:
            levels.append(0)
    assert levels == sorted(levels, reverse=True), f"levels must be non-increasing as width grows: {list(zip(widths, levels))}"
    assert levels[0] == 2
    assert levels[-1] == 0


# --- Real fixture: 02_modular_home ----------------------------------------------------


def test_footprint_reachability_on_real_modular_home_fixture(scene_modular_home_dir):
    """Rectangle-footprint reachability against the real 02_modular_home occupancy
    grid (151x92 @ 0.05m) and its real object list - the one required real-fixture
    check for this planner (see this PR's report for the full before/after table
    across all 8 platforms). Only asserts structural properties (no crash, every
    object classified, counts add up) rather than exact object ids, since those would
    be brittle against any future re-measurement of the fixture scene."""
    grid_meta = scene_ingest.read_grid_metadata(scene_modular_home_dir)
    cameras = scene_ingest.read_camera_track(scene_modular_home_dir)
    start_world = scene_ingest.choose_robot_start(cameras, grid_meta)
    cells = np.load(scene_modular_home_dir / "occupancy.npy")
    grid = OccupancyGrid(cells=cells, meta=grid_meta)

    parsed_objects = scene_ingest.parse_scene_objects(
        scene_modular_home_dir / "scene_objects" / "scene_objects.json", scene_modular_home_dir
    )
    objects = [
        (
            f"o{i}",
            ObjectFootprint(
                x=o.pos[0],
                z=o.pos[2],
                bbox_min_x=o.bbox_min[0],
                bbox_min_z=o.bbox_min[2],
                bbox_max_x=o.bbox_max[0],
                bbox_max_z=o.bbox_max[2],
            ),
        )
        for i, o in enumerate(parsed_objects)
        if not o.is_fragment
    ]
    assert objects, "fixture must have at least one non-fragment object to make this test meaningful"

    # TurtleBot3 Burger's real dimensions_m (see app/robots.py) - small footprint,
    # expect most objects reachable.
    burger = fp.RobotFootprint(length_m=0.138, width_m=0.178)
    burger_result = fp.compute_footprint_reachability(grid, start_world, 0.0, objects, burger)
    classified = len(burger_result.reachable_ids) + len(burger_result.unreachable_reasons)
    assert classified == len(objects)
    assert burger_result.reachable_ids, "burger should reach at least some real objects in this fixture"

    # Husky A200's real dimensions_m - much bigger footprint, must never reach MORE
    # objects than the small Burger does on the identical grid/start.
    husky = fp.RobotFootprint(length_m=0.985, width_m=0.6693)
    husky_result = fp.compute_footprint_reachability(grid, start_world, 0.0, objects, husky)
    assert len(husky_result.reachable_ids) <= len(burger_result.reachable_ids)
