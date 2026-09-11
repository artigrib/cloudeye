"""Two things the layer switch needed from pathfinding: rasterising a smoothed route back
into cells, and clamping a start anchor that sits just outside the grid.

Both exist because a route is now judged against ground truth about what was OBSERVED, not
only about what is passable - unknown is traversable at 1.5x (see build_cost_grid for the
0/17 that blocking it measures), so the honest reporting of a reachable count has to say
how much of it crossed ground nobody scanned.
"""

import numpy as np
import pytest

from app.services import pathfinding
from app.services.pathfinding import (
    FREE,
    OBSTACLE,
    UNKNOWN,
    ObjectFootprint,
    OccupancyGrid,
    clamp_anchor_to_grid,
    compute_reachability,
    route_cells,
    segment_cells,
)
from app.services.scene_ingest import GridMeta

META = GridMeta(resolution=0.1, origin_x=0.0, origin_z=0.0, width=10, height=10)


# --- segment_cells / route_cells ------------------------------------------------------


def test_segment_cells_walks_every_cell_between_two_waypoints():
    assert segment_cells((0, 0), (3, 0)) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert segment_cells((0, 0), (0, 0)) == [(0, 0)]


def test_segment_cells_is_the_same_walk_line_of_sight_makes():
    """The two must agree or a route can be reported as crossing cells the smoother
    believed it did not - they are deliberately separate functions (line_of_sight runs
    O(n^2) per route and stays allocation-free) so this pins them together."""
    cost = np.ones((10, 10))
    for a, b in (((0, 0), (9, 4)), ((9, 4), (0, 0)), ((2, 7), (7, 2))):
        walked = segment_cells(a, b)
        assert walked[0] == a and walked[-1] == b
        cost_blocked = cost.copy()
        cost_blocked[walked[len(walked) // 2]] = np.inf
        assert not pathfinding.line_of_sight(cost_blocked, a, b)


def test_route_cells_fills_the_gaps_a_smoothed_route_leaves():
    """`smooth_path` returns waypoints, not a cell walk. Counting the waypoints alone
    inspects a handful of corners and calls that the route."""
    path = [(0, 0), (5, 0), (5, 5)]
    cells = route_cells(path)
    assert len(cells) == 11          # 6 along x + 5 more along z, corner shared
    assert (3, 0) in cells and (5, 3) in cells


def test_route_cells_of_an_empty_route_is_empty():
    assert route_cells([]) == set()


# --- unknown_cells_on_route -----------------------------------------------------------


def _corridor_with_unknown_middle():
    """A 10x1 corridor whose middle three cells were never observed. The only way across
    is through them, which is exactly the situation `unknown_cells_on_route` reports."""
    cells = np.full((10, 10), OBSTACLE, dtype=np.uint8)
    cells[:, 5] = FREE
    cells[4:7, 5] = UNKNOWN
    return OccupancyGrid(cells=cells, meta=META)


def test_a_route_across_unobserved_ground_is_counted():
    grid = _corridor_with_unknown_middle()
    far = ObjectFootprint(x=0.95, z=0.55, bbox_min_x=0.9, bbox_min_z=0.5,
                          bbox_max_x=1.0, bbox_max_z=0.6)
    result = compute_reachability(grid, (0.05, 0.55), [("far", far)], robot_radius_m=0.0)

    assert "far" in result.reachable_ids
    assert result.unknown_cells_on_route == 3


def test_a_route_over_observed_ground_only_counts_zero():
    cells = np.full((10, 10), OBSTACLE, dtype=np.uint8)
    cells[:, 5] = FREE
    grid = OccupancyGrid(cells=cells, meta=META)
    far = ObjectFootprint(x=0.95, z=0.55, bbox_min_x=0.9, bbox_min_z=0.5,
                          bbox_max_x=1.0, bbox_max_z=0.6)
    result = compute_reachability(grid, (0.05, 0.55), [("far", far)], robot_radius_m=0.0)

    assert "far" in result.reachable_ids
    assert result.unknown_cells_on_route == 0


def test_the_count_is_distinct_cells_not_a_sum_over_routes():
    """Two objects down the same corridor cross the SAME three unknown cells. The number
    is 'how much of this room's floor is unverified', not 'how many times we drove over
    it' - summing would triple a single doorway."""
    grid = _corridor_with_unknown_middle()
    objs = [
        ("a", ObjectFootprint(x=0.85, z=0.55, bbox_min_x=0.8, bbox_min_z=0.5,
                              bbox_max_x=0.9, bbox_max_z=0.6)),
        ("b", ObjectFootprint(x=0.95, z=0.55, bbox_min_x=0.9, bbox_min_z=0.5,
                              bbox_max_x=1.0, bbox_max_z=0.6)),
    ]
    result = compute_reachability(grid, (0.05, 0.55), objs, robot_radius_m=0.0)
    assert len(result.reachable_ids) == 2
    assert result.unknown_cells_on_route == 3


# --- clamp_anchor_to_grid -------------------------------------------------------------


def test_an_anchor_just_outside_the_grid_is_clamped_to_the_edge():
    """own_0901_161054's first camera pose sits 0.62 m past the z edge - the operator
    started in the doorway, outside the percentile-trimmed extent - and it used to 500
    every reachability call on that scene."""
    assert clamp_anchor_to_grid((-0.62, 0.05), META) == (0, 0)
    assert clamp_anchor_to_grid((1.6, 0.55), META) == (9, 5)


def test_an_anchor_inside_the_grid_is_unchanged():
    assert clamp_anchor_to_grid((0.55, 0.35), META) == (5, 3)


def test_an_anchor_far_outside_the_grid_still_raises():
    """The loud failure stands past the margin: 100 m out is a coordinate-frame bug, and
    clamping it into a corner cell would answer it as if it were a normal scene."""
    with pytest.raises(ValueError, match="coordinate-frame mismatch"):
        clamp_anchor_to_grid((-100.0, 0.5), META)
    with pytest.raises(ValueError, match="coordinate-frame mismatch"):
        clamp_anchor_to_grid((0.5, 100.0), META)


def test_the_margin_is_the_documented_one_metre():
    inside = pathfinding.ANCHOR_CLAMP_MARGIN_M - 0.01
    outside = pathfinding.ANCHOR_CLAMP_MARGIN_M + 0.01
    assert clamp_anchor_to_grid((-inside, 0.5), META) == (0, 5)
    with pytest.raises(ValueError):
        clamp_anchor_to_grid((-outside, 0.5), META)
