"""The one navigation layer: resolving it, deriving a robot's grid from it, refusing.

The point of `app/services/nav_layer.py` is that a scene has exactly ONE answer to "where
can this robot stand", and a scene that has none says so instead of quietly producing a
second one. These tests hold both halves - the derivation, and the refusal.
"""

import json

import numpy as np
import pytest

from app.services import nav_layer
from app.services.nav_layer import FREE, OBSTACLE, UNKNOWN, NavLayerUnavailableError

META = {
    "resolution": 0.05,
    "origin_x": -1.0,
    "origin_z": -2.0,
    "width": 4,
    "height": 3,
    "band_min": 0.1,
    "band_max": 1.5,
    "height_margin_m": 0.05,
    "source": "test",
}


def _write_layer(scene_dir, name="backfill", *, heights=None, unobserved=None, cells=None):
    d = scene_dir / "layers" / name
    d.mkdir(parents=True)
    (d / "occupancy_meta.json").write_text(json.dumps(META))
    if cells is None:
        cells = np.zeros((4, 3), dtype=np.uint8)
    np.save(d / "occupancy.npy", cells)
    if heights is not None:
        np.save(d / "obstacle_min_height.npy", heights.astype(np.float32))
    if unobserved is not None:
        np.save(d / "unobserved_mask.npy", unobserved)
    return d


def test_resolve_reads_the_five_geometry_keys(tmp_path):
    _write_layer(tmp_path)
    layer = nav_layer.resolve(tmp_path)
    assert (layer.meta.resolution, layer.meta.width, layer.meta.height) == (0.05, 4, 3)
    assert (layer.meta.origin_x, layer.meta.origin_z) == (-1.0, -2.0)
    assert layer.band_m == (0.1, 1.5)


def test_no_layer_refuses_and_names_the_fix(tmp_path):
    """The refusal is the feature. A fallback to the old occupancy.npy is how a scene
    silently gets a second answer, which is the defect this module closed."""
    with pytest.raises(NavLayerUnavailableError) as exc:
        nav_layer.resolve(tmp_path)
    assert "backfill_occupancy.py" in str(exc.value)


def test_a_layers_dir_without_the_two_required_files_is_not_a_layer(tmp_path):
    (tmp_path / "layers" / "backfill").mkdir(parents=True)
    with pytest.raises(NavLayerUnavailableError):
        nav_layer.resolve(tmp_path)


def test_a_live_layer_wins_over_the_backfill(tmp_path):
    """A scene reconstructed after the band shipped has its own layer, from its own mesh.
    The backfill was derived from a point cloud with no observed/unobserved mask of its
    own, so it loses whenever the real thing is present."""
    _write_layer(tmp_path, "backfill")
    _write_layer(tmp_path, "own_0901_161054")
    assert nav_layer.resolve(tmp_path).grid_path.parent.name == "own_0901_161054"


def test_cells_for_height_lets_a_short_robot_under_a_tall_surface(tmp_path):
    """The whole reason the layer ships a height map instead of a boolean mask.

    One cell holds a surface at 0.69 m - a bed top. It is a wall to nothing that drives
    under it, and a wall to everything that does not."""
    heights = np.full((4, 3), np.nan, dtype=np.float32)
    heights[1, 1] = 0.69   # bed top
    heights[2, 1] = 0.15   # a chair leg: low enough to block both
    _write_layer(tmp_path, heights=heights)
    layer = nav_layer.resolve(tmp_path)

    short = nav_layer.cells_for_height(layer, 0.192)   # TurtleBot3 Burger
    tall = nav_layer.cells_for_height(layer, 0.75)

    assert short[1, 1] == FREE and tall[1, 1] == OBSTACLE
    assert short[2, 1] == OBSTACLE and tall[2, 1] == OBSTACLE


def test_unobserved_is_never_free_at_any_height(tmp_path):
    """Carried over from the shipped grid rather than recomputed: the height map says
    nothing about what was observed, so deriving at a robot's height must not turn a cell
    nobody ever saw into clear floor."""
    cells = np.zeros((4, 3), dtype=np.uint8)
    cells[0, 0] = UNKNOWN
    heights = np.full((4, 3), np.nan, dtype=np.float32)
    _write_layer(tmp_path, heights=heights, cells=cells)
    layer = nav_layer.resolve(tmp_path)

    for h in (0.192, 0.40, 1.6):
        assert nav_layer.cells_for_height(layer, h)[0, 0] == UNKNOWN


def test_no_height_map_serves_the_shipped_default_height_grid(tmp_path):
    cells = np.array([[FREE, OBSTACLE, UNKNOWN]] * 4, dtype=np.uint8)
    _write_layer(tmp_path, cells=cells)
    layer = nav_layer.resolve(tmp_path)
    assert layer.height_path is None
    np.testing.assert_array_equal(nav_layer.cells_for_height(layer, 0.192), cells)


def test_unobserved_mask_prefers_the_file_and_falls_back_to_the_grid(tmp_path):
    """The fallback is about a FILE inside the one layer, never about another layer, and
    it is exact rather than approximate: the producer applies unobserved last, over
    everything else, so `grid == UNKNOWN` is the same set."""
    cells = np.zeros((4, 3), dtype=np.uint8)
    cells[3, 2] = UNKNOWN
    _write_layer(tmp_path, cells=cells)
    assert nav_layer.unobserved_mask(nav_layer.resolve(tmp_path))[3, 2]

    mask = np.zeros((4, 3), dtype=bool)
    mask[0, 1] = True
    d = tmp_path / "layers" / "backfill"
    np.save(d / "unobserved_mask.npy", mask)
    got = nav_layer.unobserved_mask(nav_layer.resolve(tmp_path))
    assert got[0, 1] and not got[3, 2]


def test_derived_cells_are_cached_per_height(tmp_path):
    heights = np.full((4, 3), np.nan, dtype=np.float32)
    heights[1, 1] = 0.69
    _write_layer(tmp_path, heights=heights)
    layer = nav_layer.resolve(tmp_path)
    a = nav_layer.cells_for_height(layer, 0.192)
    b = nav_layer.cells_for_height(layer, 0.192)
    assert a is b
    assert nav_layer.cells_for_height(layer, 0.75) is not a
