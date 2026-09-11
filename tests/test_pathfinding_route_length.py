"""`ReachabilityResult.path_length_m` - the route length the object list prints per row.

The object list says how far away each reachable object is. That number has to be the
route the robot would actually drive, not an estimate of it: the command panel prints
the planner's own `total_length_m` for the same object the moment someone double-clicks
the row, and two different numbers for one route is worse than no number at all.

So these tests pin the contract rather than the arithmetic: the keys are exactly the
reachable ids, and each value equals what `plan_to_object` returns from the same start
over the same grid.
"""

import pytest

from app.services import pathfinding
from app.services.pathfinding import OccupancyGrid

from tests.test_pathfinding_connectivity import footprint_at, make_grid


def _reachability(rows, objects, *, robot_radius_m=0.0, start_cell=(0, 0)):
    cells, meta = make_grid(rows)
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(*start_cell, meta)
    footprints = [(name, footprint_at(meta, ix, iz)) for name, (ix, iz) in objects.items()]
    result = pathfinding.compute_reachability(grid, start_world, footprints, robot_radius_m=robot_radius_m)
    return grid, result


def test_every_reachable_object_gets_a_length_and_no_unreachable_one_does():
    grid, result = _reachability(
        [
            "...#...",
            "...#...",
            "...#...",
        ],
        {"near": (2, 0), "far_side": (6, 0)},
    )

    assert "far_side" in result.unreachable_reasons
    # Exactly the reachable ids - an unreachable object has no route, and a 0.0 here
    # would read as "it is already there".
    assert set(result.path_length_m) == set(result.reachable_ids) == {"near"}
    assert result.path_length_m["near"] > 0.0


def test_the_length_is_the_route_the_planner_would_drive():
    grid, result = _reachability(
        [
            ".........",
            "....##...",
            ".........",
        ],
        {"corner": (8, 2)},
    )

    assert "corner" in result.reachable_ids
    planned = pathfinding.plan_to_object(
        grid,
        result.start.point,
        footprint_at(grid.meta, 8, 2),
        robot_radius_m=0.0,
        speed_mps=0.5,
        objects=[footprint_at(grid.meta, 8, 2)],
    )
    assert result.path_length_m["corner"] == pytest.approx(planned.length_m, abs=1e-9)


def test_a_platform_with_no_start_reports_no_lengths_at_all():
    # Radius far larger than the room: no cell anywhere has that much clearance, so the
    # robot never stands anywhere and there is nothing to measure a route from.
    _grid, result = _reachability(["...", "...", "..."], {"thing": (2, 2)}, robot_radius_m=5.0)

    assert result.start.cell is None
    assert result.path_length_m == {}
    assert result.unreachable_reasons["thing"] == "robot_does_not_fit"


def test_an_out_of_grid_object_is_absent_rather_than_zero():
    cells, meta = make_grid(["....."])
    grid = OccupancyGrid(cells=cells, meta=meta)
    far_x = meta.origin_x - 100.0
    outside = pathfinding.ObjectFootprint(
        x=far_x, z=meta.origin_z - 100.0,
        bbox_min_x=far_x - 0.1, bbox_min_z=meta.origin_z - 100.1,
        bbox_max_x=far_x + 0.1, bbox_max_z=meta.origin_z - 99.9,
    )
    result = pathfinding.compute_reachability(
        grid, pathfinding.cell_to_world(0, 0, meta),
        [("in_grid", footprint_at(meta, 4, 0)), ("out_of_grid", outside)],
        robot_radius_m=0.0,
    )

    assert result.unreachable_reasons["out_of_grid"] == "outside_grid"
    assert "out_of_grid" not in result.path_length_m
    assert "in_grid" in result.path_length_m
