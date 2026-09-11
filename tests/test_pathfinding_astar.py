"""A* correctness on small, hand-built grids: straight paths, obstacle detours,
genuinely unreachable goals, inflation, smoothing, and no diagonal corner-cutting.
"""

import numpy as np
import pytest

from app.services import pathfinding
from app.services.pathfinding import FREE, OBSTACLE, UNKNOWN
from app.services.scene_ingest import GridMeta


def make_grid(rows: list[str], *, resolution=0.1) -> tuple[np.ndarray, GridMeta]:
    """Build a (width, height) uint8 grid from a list of row-strings, each character
    one cell: '.' free, '#' obstacle, '?' unknown. Rows are Z (top row = iz=0), columns
    are X - so we transpose from the natural [row][col] reading order into [ix, iz]."""
    height = len(rows)
    width = len(rows[0])
    char_to_val = {".": FREE, "#": OBSTACLE, "?": UNKNOWN}
    cells = np.zeros((width, height), dtype=np.uint8)
    for iz, row in enumerate(rows):
        for ix, ch in enumerate(row):
            cells[ix, iz] = char_to_val[ch]
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)
    return cells, meta


def test_straight_path_no_obstacles():
    cells, meta = make_grid(["....."])
    cost = pathfinding.build_cost_grid(cells)
    path = pathfinding.astar(cost, (0, 0), (4, 0))
    assert path[0] == (0, 0)
    assert path[-1] == (4, 0)


def test_detour_around_obstacle():
    cells, meta = make_grid(
        [
            ".....",
            ".###.",
            ".....",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    path = pathfinding.astar(cost, (0, 1), (4, 1))
    # must go around, not through, the obstacle row
    assert all(cells[ix, iz] != OBSTACLE for ix, iz in path)


def test_unreachable_goal_raises_no_path_error():
    cells, meta = make_grid(
        [
            ".#.",
            "###",
            ".#.",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    with pytest.raises(pathfinding.NoPathError):
        pathfinding.astar(cost, (0, 0), (2, 2))


def test_start_or_goal_inside_obstacle_raises():
    cells, meta = make_grid(["..#.."])
    cost = pathfinding.build_cost_grid(cells)
    with pytest.raises(pathfinding.NoPathError):
        pathfinding.astar(cost, (2, 0), (4, 0))


def test_inflation_closes_a_narrow_corridor():
    # A 1-cell-wide gap between two obstacle blocks - should be passable unflated,
    # blocked once inflated by a radius bigger than the gap.
    cells, meta = make_grid(
        [
            "##.##",
            "##.##",
            "##.##",
        ]
    )
    cost_raw = pathfinding.build_cost_grid(cells)
    assert np.isfinite(cost_raw[2, 1])  # the gap itself is free

    inflated = pathfinding.inflate(cells, meta, radius_m=0.15)  # 1.5 cells at 0.1m resolution
    cost_inflated = pathfinding.build_cost_grid(inflated)
    assert not np.isfinite(cost_inflated[2, 1]), "a wide-enough inflation should close a 1-cell gap"


def test_no_diagonal_corner_cutting():
    cells, meta = make_grid(
        [
            ".#",
            "#.",
        ]
    )
    cost = pathfinding.build_cost_grid(cells)
    # (0,0) and (1,1) are diagonal neighbors, but both orthogonal cells between them
    # ((1,0) and (0,1)) are obstacles - a real robot can't cut that corner.
    neighbors = list(pathfinding.neighbors8(0, 0, cost))
    assert (1, 1, pytest.approx(pathfinding.SQRT2)) not in [
        (nx, nz, pytest.approx(c)) for nx, nz, c in neighbors
    ]
    assert all((nx, nz) != (1, 1) for nx, nz, _ in neighbors)


def test_unknown_cells_traversable_but_cost_more():
    cells, meta = make_grid(["....."])
    cells[2, 0] = UNKNOWN
    cost = pathfinding.build_cost_grid(cells)
    assert cost[2, 0] == pathfinding.UNKNOWN_COST_MULTIPLIER
    path = pathfinding.astar(cost, (0, 0), (4, 0))
    assert (2, 0) in path  # still traversable, just costs more - not blocked


def test_smoothing_shortens_a_staircase_path():
    cells, meta = make_grid(["....." for _ in range(5)])
    cost = pathfinding.build_cost_grid(cells)
    raw_path = pathfinding.astar(cost, (0, 0), (4, 4))
    smoothed = pathfinding.smooth_path(raw_path, cost)
    assert len(smoothed) <= len(raw_path)
    assert smoothed[0] == raw_path[0]
    assert smoothed[-1] == raw_path[-1]
    # every hop in the smoothed path must still be a real, obstacle-free line
    for a, b in zip(smoothed, smoothed[1:]):
        assert pathfinding.line_of_sight(cost, a, b)


def test_nearest_free_cell_finds_something_outside_a_blocked_target():
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
    ix, iz = pathfinding.nearest_free_cell(cost, (2, 2))  # dead center of the block
    assert np.isfinite(cost[ix, iz])


def test_path_length_m_is_euclidean_sum():
    points = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    assert pathfinding.path_length_m(points) == pytest.approx(7.0)
