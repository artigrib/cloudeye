"""world_to_cell / cell_to_world are the two functions most likely to have a silent
off-by-origin bug - this project hit exactly this class of coordinate-frame error
repeatedly on the GPU side, so these are tested against the REAL recorded grid edges
from an actual pipeline run, not synthetic round numbers.
"""

import json

import numpy as np
import pytest

from app.services import pathfinding, scene_ingest


@pytest.fixture
def grid_meta(scene_horizontal_dir):
    return scene_ingest.read_grid_metadata(scene_horizontal_dir)


def test_grid_meta_has_negative_origin(grid_meta):
    # The real fixture's origin is negative on both axes - if a future refactor
    # accidentally assumes origin >= 0 (e.g. clamping instead of offsetting), this
    # fixture will catch it immediately.
    assert grid_meta.origin_x < 0
    assert grid_meta.origin_z < 0


def test_round_trip_cell_center(grid_meta):
    for ix in range(0, grid_meta.width, 5):
        for iz in range(0, grid_meta.height, 7):
            x, z = pathfinding.cell_to_world(ix, iz, grid_meta)
            ix2, iz2 = pathfinding.world_to_cell(x, z, grid_meta)
            assert (ix2, iz2) == (ix, iz), f"round trip failed for cell ({ix},{iz})"


def test_axis_order_matches_occupancy_grid_convention(scene_horizontal_dir, grid_meta):
    """occupancy.npy is indexed [ix, iz] - axis 0 is X, axis 1 is Z. Assert this
    directly against the real array shape rather than trusting a comment."""
    cells = np.load(scene_horizontal_dir / "occupancy.npy")
    assert cells.shape == (grid_meta.width, grid_meta.height)
    assert cells.shape[0] != cells.shape[1]  # this fixture's grid isn't square, so a
    # transpose bug would be caught by the shape assertion above, not silently pass


def test_cell_to_world_returns_cell_center(grid_meta):
    x0, z0 = pathfinding.cell_to_world(0, 0, grid_meta)
    assert x0 == pytest.approx(grid_meta.origin_x + grid_meta.resolution / 2)
    assert z0 == pytest.approx(grid_meta.origin_z + grid_meta.resolution / 2)


def test_world_to_cell_clamps_small_out_of_bounds(grid_meta):
    # A hair outside the grid (within the 2-cell margin) should clamp, not raise -
    # this is expected floating-point slop right at an edge.
    x = grid_meta.origin_x - grid_meta.resolution * 0.5
    z = grid_meta.origin_z - grid_meta.resolution * 0.5
    ix, iz = pathfinding.world_to_cell(x, z, grid_meta)
    assert ix == 0
    assert iz == 0


def test_world_to_cell_raises_far_out_of_bounds(grid_meta):
    with pytest.raises(ValueError):
        pathfinding.world_to_cell(grid_meta.origin_x - 100, grid_meta.origin_z, grid_meta)


def test_grid_meta_matches_raw_npz_edges(scene_horizontal_dir, grid_meta):
    """Cross-check the JSON-derived GridMeta against the underlying occupancy_meta.json
    directly (belt and suspenders vs a future change to how it's written)."""
    raw = json.loads((scene_horizontal_dir / "occupancy_meta.json").read_text())
    assert grid_meta.resolution == raw["resolution"]
    assert grid_meta.origin_x == raw["origin_x"]
    assert grid_meta.origin_z == raw["origin_z"]
    assert grid_meta.width == raw["width"]
    assert grid_meta.height == raw["height"]
