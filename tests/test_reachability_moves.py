"""The "what if that were somewhere else" branch of GET /reachability.

The `move` parameter answers a question the screen can otherwise only gesture at - "can
this robot get through if I shift the nightstand?" - by cutting each named object's bbox
out of the occupancy grid and pasting it back dx/dz metres away, then computing
reachability over the result exactly as usual.

Two properties matter more than any particular number:

  1. With no `move`, nothing about the computation changes. The room's real numbers are
     the product's claim; a what-if must not be able to move them.
  2. A move that seals a doorway makes what is behind it unreachable, and moving it back
     restores the original answer - i.e. the cut and the paste are inverses.
"""

import numpy as np
import pytest

from app.services import pathfinding
from app.services.pathfinding import FREE, OBSTACLE, ObjectFootprint, OccupancyGrid

from tests.test_pathfinding_connectivity import footprint_at, make_grid


def _box(meta, ix0, iz0, ix1, iz1) -> ObjectFootprint:
    x0, z0 = pathfinding.cell_to_world(ix0, iz0, meta)
    x1, z1 = pathfinding.cell_to_world(ix1, iz1, meta)
    return ObjectFootprint(
        x=(x0 + x1) / 2, z=(z0 + z1) / 2,
        bbox_min_x=min(x0, x1), bbox_min_z=min(z0, z1),
        bbox_max_x=max(x0, x1), bbox_max_z=max(z0, z1),
    )


def test_cut_clears_exactly_the_bbox_and_leaves_the_rest_alone():
    cells, meta = make_grid(["#####", "#...#", "#####"])
    out = pathfinding.cut_footprints_from_grid(cells, meta, [_box(meta, 0, 0, 0, 0)])

    assert out[0, 0] == FREE                     # the cut cell
    assert out[1, 0] == OBSTACLE                 # its neighbour, untouched
    assert cells[0, 0] == OBSTACLE               # the input is not mutated
    assert np.array_equal(out[1:, :], cells[1:, :])


def test_cut_with_no_footprints_is_a_plain_copy():
    cells, meta = make_grid(["#.#", ".#."])
    out = pathfinding.cut_footprints_from_grid(cells, meta, [])
    assert np.array_equal(out, cells)
    assert out is not cells


def test_translate_footprint_slides_the_box_without_resizing_it():
    _cells, meta = make_grid(["....."])
    fp = _box(meta, 1, 0, 3, 0)
    moved = pathfinding.translate_footprint(fp, 0.5, -0.25)

    assert moved.bbox_min_x == pytest.approx(fp.bbox_min_x + 0.5)
    assert moved.bbox_max_x == pytest.approx(fp.bbox_max_x + 0.5)
    assert moved.bbox_min_z == pytest.approx(fp.bbox_min_z - 0.25)
    assert moved.x == pytest.approx(fp.x + 0.5)
    # Same size, so a moved object never becomes a bigger or smaller obstacle.
    assert moved.bbox_max_x - moved.bbox_min_x == pytest.approx(fp.bbox_max_x - fp.bbox_min_x)
    assert moved.bbox_max_z - moved.bbox_min_z == pytest.approx(fp.bbox_max_z - fp.bbox_min_z)


def _doorway():
    """A room split by a wall with one gap, at cell (3, 1). An object sits well clear of
    the gap; the target is on the far side."""
    cells, meta = make_grid(
        [
            "...#...",
            ".......",
            "...#...",
        ]
    )
    grid = OccupancyGrid(cells=cells, meta=meta)
    start = pathfinding.cell_to_world(0, 1, meta)
    objects = [("blocker", _box(meta, 0, 0, 0, 0)), ("far", footprint_at(meta, 6, 1))]
    return grid, meta, start, objects


def test_a_move_that_seals_the_only_gap_makes_the_far_side_unreachable():
    grid, meta, start, objects = _doorway()

    before = pathfinding.compute_reachability(grid, start, objects, robot_radius_m=0.0)
    assert "far" in before.reachable_ids

    # Slide the blocker into the gap: three cells right, one cell down.
    dx = 3 * meta.resolution
    dz = 1 * meta.resolution
    cut = pathfinding.cut_footprints_from_grid(grid.cells, meta, [objects[0][1]])
    moved_objects = [("blocker", pathfinding.translate_footprint(objects[0][1], dx, dz)), objects[1]]
    after = pathfinding.compute_reachability(
        OccupancyGrid(cells=cut, meta=meta), start, moved_objects, robot_radius_m=0.0
    )

    assert "far" not in after.reachable_ids
    assert "far" in after.unreachable_reasons
    # And no route length is reported for something there is no route to.
    assert "far" not in after.path_length_m


def test_moving_it_back_restores_the_original_answer():
    grid, meta, start, objects = _doorway()
    before = pathfinding.compute_reachability(grid, start, objects, robot_radius_m=0.0)

    # Out and back: the cut is applied on the original grid both times, so this is the
    # identity move, and it must not drift.
    cut = pathfinding.cut_footprints_from_grid(grid.cells, meta, [objects[0][1]])
    same = [("blocker", pathfinding.translate_footprint(objects[0][1], 0.0, 0.0)), objects[1]]
    after = pathfinding.compute_reachability(
        OccupancyGrid(cells=cut, meta=meta), start, same, robot_radius_m=0.0
    )

    assert after.reachable_ids == before.reachable_ids
    assert after.path_length_m.keys() == before.path_length_m.keys()
    for obj_id, length in before.path_length_m.items():
        assert after.path_length_m[obj_id] == pytest.approx(length)
