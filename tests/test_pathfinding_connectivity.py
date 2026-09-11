"""Connectivity-aware start resolution - the fix for the "robot locked at its start
cell" bug: a cell can be individually traversable yet still be sealed off from most of
the room once obstacles are inflated by the robot's radius (a doorway narrower than the
robot's diameter, closed shut by inflation). `nearest_free_cell`/`snap_if_blocked` only
ask "is this cell blocked" and can't catch that - these tests are for the functions that
can: `connected_component`, `largest_connected_component`, and `resolve_robot_start`.
"""

import numpy as np
import pytest

from app.services import pathfinding
from app.services.pathfinding import FREE, NoPathError, OBSTACLE, ObjectFootprint, OccupancyGrid
from app.services.scene_ingest import GridMeta


def make_grid(rows: list[str], *, resolution=0.1) -> tuple[np.ndarray, GridMeta]:
    """Same convention as test_pathfinding_astar.py's helper: rows are Z (top=iz=0),
    columns are X, one character per cell ('.' free, '#' obstacle)."""
    height = len(rows)
    width = len(rows[0])
    char_to_val = {".": FREE, "#": OBSTACLE}
    cells = np.zeros((width, height), dtype=np.uint8)
    for iz, row in enumerate(rows):
        for ix, ch in enumerate(row):
            cells[ix, iz] = char_to_val[ch]
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)
    return cells, meta


def footprint_at(meta: GridMeta, ix: int, iz: int) -> ObjectFootprint:
    x, z = pathfinding.cell_to_world(ix, iz, meta)
    return ObjectFootprint(x=x, z=z, bbox_min_x=x, bbox_min_z=z, bbox_max_x=x, bbox_max_z=z)


def test_connected_component_stops_at_obstacles():
    cells, meta = make_grid(
        [
            "...#...",
            "...#...",
            "...#...",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    left = pathfinding.connected_component(cost, (0, 0))
    assert (2, 0) in left
    assert all(ix < 3 for ix, _ in left), "the obstacle column must not be crossed"


def test_connected_component_of_blocked_cell_is_empty():
    cells, meta = make_grid(["#.."])
    cost = pathfinding.build_cost_grid(cells)
    assert pathfinding.connected_component(cost, (0, 0)) == set()


def test_connected_component_respects_no_corner_cutting():
    # (0,0) and (1,1) are diagonal neighbors but both orthogonal cells between them are
    # obstacles - same rule astar's neighbors8 enforces, so component membership must
    # agree with what astar could actually reach.
    cells, meta = make_grid(
        [
            ".#",
            "#.",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    component = pathfinding.connected_component(cost, (0, 0))
    assert component == {(0, 0)}


def test_largest_connected_component_picks_the_biggest_island():
    cells, meta = make_grid(
        [
            "..#....",
            "..#....",
            "..#....",
            "..#....",
            "#######",  # ensure the two islands don't wrap-connect around the grid edge
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    largest = pathfinding.largest_connected_component(cost)
    # left island is 2x4=8 cells, right island is 4x4=16 cells
    assert len(largest) == 16
    assert all(ix >= 3 for ix, _ in largest)


def _isolated_pocket_grid():
    """(1,1) is a single free cell boxed in on all 8 sides by obstacles - individually
    traversable, but with zero traversable neighbors, same shape as a wide robot's
    radius sealing it in on every side (the reported "Husky spawns touching the point
    cloud and can't move at all, even though there's real space nearby" bug). A
    separate 5-cell free strip at ix=3..7 is the "real space nearby" it should be
    relocated into - not connected to (1,1) at all, by construction."""
    cells, meta = make_grid(
        [
            "########",
            "#.#.....",
            "########",
        ]
    )
    return cells, meta


def test_relocate_if_isolated_leaves_a_well_connected_cell_alone():
    cells, meta = make_grid(["....."])
    cost = pathfinding.build_cost_grid(cells)
    cell, relocated = pathfinding.relocate_if_isolated(cost, (2, 0))
    assert (cell, relocated) == ((2, 0), False)


def test_relocate_if_isolated_escapes_a_boxed_in_single_cell():
    cells, meta = _isolated_pocket_grid()
    cost = pathfinding.build_cost_grid(cells)
    cell, relocated = pathfinding.relocate_if_isolated(cost, (1, 1))
    assert relocated is True
    assert cell[0] >= 3  # landed in the 5-cell strip, not still at the boxed-in cell


def test_resolve_robot_start_relocates_out_of_an_isolated_pocket_with_no_objects():
    # The exact blind spot the object-fraction logic alone has: total_objects == 0 used
    # to make resolve_robot_start accept ANY individually-free start outright (fraction
    # defaults to 1.0 when there's nothing to check against), even a start with zero
    # traversable neighbors at all. Regression test for that - would fail before the
    # relocate_if_isolated check was added ahead of the total==0 early return.
    cells, meta = _isolated_pocket_grid()
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(1, 1, meta)

    result = pathfinding.resolve_robot_start(grid, start_world, [], robot_radius_m=0.0)

    assert result.relocated is True
    assert result.snapped is True
    assert result.cell[0] >= 3
    assert result.total_objects == 0


def test_resolve_robot_start_relocates_out_of_an_isolated_pocket_even_with_a_tie():
    # Same pocket, but now with objects split so relocating doesn't *strictly* improve
    # the object-reachable fraction (both candidates see 0 of the 1 object) - the other
    # blind spot: `alt_fraction > fraction` being false must not by itself keep a
    # start that has zero traversable neighbors.
    cells, meta = _isolated_pocket_grid()
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(1, 1, meta)
    # An object far outside the grid - "outside_grid" for both candidates, so neither
    # improves on the other by the object-fraction proxy alone.
    objects = [_far_outside_footprint(meta)]

    result = pathfinding.resolve_robot_start(grid, start_world, objects, robot_radius_m=0.0)

    assert result.relocated is True
    assert result.cell[0] >= 3


def test_plan_to_object_succeeds_from_an_isolated_start_pocket():
    # Before relocate_if_isolated was wired into plan_to_object, this raised NoPathError
    # ("goal is unreachable from start") every time - the exact "can't move at all"
    # symptom, even though the object sits in real, reachable space just past the seal.
    cells, meta = _isolated_pocket_grid()
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(1, 1, meta)
    target = footprint_at(meta, 7, 1)

    result = pathfinding.plan_to_object(grid, start_world, target, robot_radius_m=0.0, speed_mps=1.0)

    assert result.start_snapped is True
    assert result.points[0] != start_world  # actually moved off the sealed cell
    assert result.cells[-1][0] >= 3


def test_compute_reachability_relocates_out_of_an_isolated_start_pocket():
    # Keeps the reachability report honest about what a command would actually achieve:
    # without relocating here too, this would report 0/1 reachable ("disconnected") even
    # though the very next goto command succeeds (see the plan_to_object test above).
    cells, meta = _isolated_pocket_grid()
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(1, 1, meta)
    objects = [("chair", footprint_at(meta, 7, 1))]

    result = pathfinding.compute_reachability(grid, start_world, objects, robot_radius_m=0.0)

    assert "chair" in result.reachable_ids
    assert result.start_component_size > 1


def test_resolve_robot_start_keeps_a_well_connected_start():
    cells, meta = make_grid(["." * 10 for _ in range(10)])
    grid = OccupancyGrid(cells=cells, meta=meta)
    objects = [footprint_at(meta, 9, 9), footprint_at(meta, 0, 9), footprint_at(meta, 9, 0)]
    start_world = pathfinding.cell_to_world(0, 0, meta)

    result = pathfinding.resolve_robot_start(grid, start_world, objects, robot_radius_m=0.0)

    assert result.relocated is False
    assert result.reachable_fraction == 1.0
    assert result.point == start_world


def test_resolve_robot_start_relocates_out_of_a_sealed_pocket():
    # A small pocket separated from a big room by a wall with a single-cell gap. The
    # gap is only reachable raw, not after a 1.5-cell inflation radius seals it (and
    # the columns either side of the wall) shut - this is the real ikport bug's
    # shape: the start cell itself stays individually free, but it's cut off from
    # almost every object.
    #
    # Physically bigger than the original (tiny, 1.6m x 0.8m) version of this fixture
    # on purpose: resolve_robot_start now rasterizes every object's own bbox as an
    # obstacle too (app.services.pathfinding.rasterize_footprints_as_obstacles - see
    # the task that added this), and app.services.goal_resolution's SEARCH_RADIUS_M
    # (2.0m) would otherwise dwarf a grid that small, letting a goal "resolve" to a
    # nearby-in-Euclidean-distance-only cell across the sealed wall (same class of
    # issue documented on test_reachable_count_is_non_increasing_as_radius_grows's
    # own fixture). The pocket is also kept notably SMALLER than the big room here:
    # the big room loses some open area to its own now-rasterized objects, so it must
    # start out with enough margin to still end up the larger (and therefore correct
    # relocation target) connected component.
    POCKET_W = 40  # cells (2.0m at resolution=0.1)
    BIG_ROOM_W = 140  # cells (7.0m) - see the margin note above
    width = POCKET_W + 1 + BIG_ROOM_W
    height = 80
    wall_col, gap_row = POCKET_W, 40
    rows = [
        "".join("#" if ix == wall_col and iz != gap_row else "." for ix in range(width))
        for iz in range(height)
    ]
    cells, meta = make_grid(rows, resolution=0.1)
    grid = OccupancyGrid(cells=cells, meta=meta)

    start_world = pathfinding.cell_to_world(20, 40, meta)  # inside the small pocket
    objects = [
        footprint_at(meta, POCKET_W + 1 + ix, iz) for ix in range(10, 130, 10) for iz in (20, 60)
    ]  # all in the big room

    result = pathfinding.resolve_robot_start(grid, start_world, objects, robot_radius_m=0.15)

    assert result.relocated is True
    assert result.reachable_fraction > 0.9
    # relocated cell must land on the big-room side, not still in the sealed pocket
    assert result.cell[0] >= POCKET_W + 1


def test_resolve_robot_start_keeps_start_when_relocating_would_not_help():
    # Two disconnected rooms of equal size, objects split evenly between them - no
    # relocation does strictly better than the other, so the original start (already
    # snapped/valid) must be kept rather than churned for no gain. Mirrors the real
    # "ikport" scene: the grid itself is fragmented, not the start point.
    cells, meta = make_grid(
        [
            "...#...",
            "...#...",
            "...#...",
        ]
    )
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(0, 0, meta)
    objects = [footprint_at(meta, 1, 0), footprint_at(meta, 5, 0)]  # one per side

    result = pathfinding.resolve_robot_start(grid, start_world, objects, robot_radius_m=0.0)

    assert result.relocated is False
    assert result.reachable_fraction == pytest.approx(0.5)
    assert result.point == start_world


def test_resolve_robot_start_snaps_a_literally_blocked_start():
    cells, meta = make_grid(["#...."])
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(0, 0, meta)  # sits on an obstacle cell
    objects = [footprint_at(meta, 4, 0)]

    result = pathfinding.resolve_robot_start(grid, start_world, objects, robot_radius_m=0.0)

    assert result.snapped is True
    assert result.cell != (0, 0)
    assert result.reachable_fraction == 1.0


def test_snap_if_blocked_leaves_a_free_cell_untouched():
    cells, meta = make_grid(["....."])
    cost = pathfinding.build_cost_grid(cells)
    cell, snapped = pathfinding.snap_if_blocked(cost, (2, 0), max_radius_cells=3)
    assert (cell, snapped) == ((2, 0), False)


def test_resolve_goal_cell_snaps_off_an_objects_own_footprint():
    cells, meta = make_grid(
        [
            ".....",
            ".###.",
            ".###.",
            ".###.",
            ".....",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    footprint = footprint_at(meta, 2, 2)  # dead center of the obstacle block
    cell, snapped = pathfinding.resolve_goal_cell(cells, meta, footprint, 0.0)
    assert snapped is True
    assert np.isfinite(cost[cell])
    assert cell != (2, 2)  # not inside the obstacle itself


# --- Out-of-grid objects: one bad object degrades gracefully, robot start does not ------


def _far_outside_footprint(meta: GridMeta) -> ObjectFootprint:
    """An object whose centroid is nowhere near the grid - simulates the coordinate-frame
    mismatch bug (bad data for that one object), not constructed via footprint_at since
    that always produces an in-bounds point."""
    x = meta.origin_x - 100.0
    z = meta.origin_z - 100.0
    return ObjectFootprint(x=x, z=z, bbox_min_x=x - 0.1, bbox_min_z=z - 0.1, bbox_max_x=x + 0.1, bbox_max_z=z + 0.1)


def test_try_resolve_goal_cell_returns_none_for_out_of_grid_footprint():
    cells, meta = make_grid(["....."])
    assert pathfinding.try_resolve_goal_cell(cells, meta, _far_outside_footprint(meta), 0.0) is None
    # in-bounds footprint still resolves normally
    assert pathfinding.try_resolve_goal_cell(cells, meta, footprint_at(meta, 2, 0), 0.0) is not None


def test_compute_reachability_tolerates_an_out_of_grid_object():
    cells, meta = make_grid(
        [
            ".....",
            ".....",
            ".....",
        ]
    )
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(0, 0, meta)
    objects = [
        ("in_grid", footprint_at(meta, 4, 2)),
        ("out_of_grid", _far_outside_footprint(meta)),
    ]

    result = pathfinding.compute_reachability(grid, start_world, objects, robot_radius_m=0.0)

    assert result.unreachable_reasons["out_of_grid"] == "outside_grid"
    assert "out_of_grid" not in result.reachable_ids
    assert "in_grid" in result.reachable_ids
    assert "in_grid" not in result.unreachable_reasons


def test_compute_reachability_marks_a_walled_off_object_disconnected():
    cells, meta = make_grid(
        [
            "...#...",
            "...#...",
            "...#...",
        ]
    )
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(0, 0, meta)
    objects = [("far_side", footprint_at(meta, 6, 0))]  # other side of the unbroken wall

    result = pathfinding.compute_reachability(grid, start_world, objects, robot_radius_m=0.0)

    assert result.unreachable_reasons["far_side"] == "disconnected"
    assert "far_side" not in result.reachable_ids


def test_resolve_robot_start_tolerates_an_out_of_grid_object():
    cells, meta = make_grid(["." * 10 for _ in range(10)])
    grid = OccupancyGrid(cells=cells, meta=meta)
    objects = [footprint_at(meta, 9, 9), footprint_at(meta, 0, 9), _far_outside_footprint(meta)]
    start_world = pathfinding.cell_to_world(0, 0, meta)

    result = pathfinding.resolve_robot_start(grid, start_world, objects, robot_radius_m=0.0)

    assert result.total_objects == 3
    assert result.reachable_count == 2  # the two in-grid objects only
    assert result.reachable_fraction == pytest.approx(2 / 3)


def test_plan_to_object_raises_no_path_error_for_out_of_grid_object():
    cells, meta = make_grid(["....."])
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(0, 0, meta)

    with pytest.raises(NoPathError):
        pathfinding.plan_to_object(
            grid, start_world, _far_outside_footprint(meta), robot_radius_m=0.0, speed_mps=1.0
        )


def test_robot_start_position_itself_still_raises_when_far_outside_grid():
    # The robot's own start point being off the map must stay a loud failure - unlike a
    # bad object, silently treating this as "nothing reachable" would look like a normal
    # (if empty) scene rather than the coordinate-frame bug it actually is.
    cells, meta = make_grid(["....."])
    grid = OccupancyGrid(cells=cells, meta=meta)
    bad_start = (meta.origin_x - 100.0, meta.origin_z - 100.0)
    objects = [footprint_at(meta, 2, 0)]

    with pytest.raises(ValueError):
        pathfinding.resolve_robot_start(grid, bad_start, objects, robot_radius_m=0.0)

    with pytest.raises(ValueError):
        pathfinding.compute_reachability(
            grid, bad_start, [("a", footprint_at(meta, 2, 0))], robot_radius_m=0.0
        )


# --- resolve_start_for_radius --------------------------------------------------------
#
# The per-request, per-radius start (see its docstring). Distinct from
# resolve_robot_start above, which runs once at scene creation and is persisted.


def test_start_for_radius_keeps_an_anchor_that_already_fits():
    cells, meta = make_grid(
        [
            ".....",
            ".....",
            ".....",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    anchor = pathfinding.cell_to_world(2, 1, meta)
    start = pathfinding.resolve_start_for_radius(cost, meta, anchor)
    assert start.status == "original"
    assert start.cell == (2, 1)
    assert start.moved_m == 0.0
    # The caller's exact world point is preserved, not snapped to the cell centre.
    assert start.point == anchor


def test_start_for_radius_prefers_the_largest_component_over_the_nearest_cell():
    """The (e2) rule, and the reason it replaced plain "nearest".

    The anchor sits in a one-cell pocket, walled off from a large open area. Nearest-
    overall picks the pocket - it is 0 cells away - and the robot can then reach nothing.
    This is the shape of what own_0902_140657 does at Husky A200's 0.5528m: the nearest
    fitting cell was in a 423-cell pocket reaching 1 of 24 objects while the largest
    component held 2,687 cells. Picking the largest component moved it to 17 of 24.
    """
    cells, meta = make_grid(
        [
            ".#....",
            "##....",
            "......",
            "......",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    anchor = pathfinding.cell_to_world(0, 0, meta)  # the sealed pocket

    pocket = pathfinding.connected_component(cost, (0, 0))
    assert pocket == {(0, 0)}, "the anchor must really be sealed off for this test to mean anything"

    start = pathfinding.resolve_start_for_radius(cost, meta, anchor)
    assert start.status == "moved"
    assert start.cell != (0, 0)
    # It lands in the big area, and at the closest point of it to the anchor - the anchor
    # still has pull, it just cannot drag the start into a dead end.
    assert start.cell in pathfinding.largest_connected_component(cost)
    assert start.moved_m > 0
    assert start.cell == (0, 2)


def test_start_for_radius_reports_none_when_nothing_fits():
    """Not an error: own_0901_173903_15fps at Husky A200's 0.5528m has zero cells with
    that clearance, and "No start position for this platform" is the honest answer where
    0/N would read as a broken scene."""
    cells, meta = make_grid(
        [
            "###",
            "###",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    start = pathfinding.resolve_start_for_radius(cost, meta, (0.05, 0.05))
    assert start.status == "none"
    assert start.cell is None
    assert start.point is None
