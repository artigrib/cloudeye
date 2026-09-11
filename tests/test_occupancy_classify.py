"""Task 4 (phantom-obstacle fix): rule (a) connected-component cluster-size filtering,
rule (b) multi-view support filtering, and the fallback logic that picks between them -
see gpu/occupancy_classify.py's module docstring for the full rationale. Pure numpy,
synthetic fixtures only (no GPU/open3d needed - this module is deliberately importable
from the plain test venv, same pattern as test_confidence_filter.py/test_camera_convention.py).
"""

import json

import numpy as np
import pytest

from gpu.occupancy_classify import (
    FREE,
    MIN_CLUSTER_CELLS,
    MIN_VIEWS_FOR_OBSTACLE,
    OBSTACLE,
    UNKNOWN,
    classify_phantom_obstacles,
    connected_components,
    filter_by_view_support,
    filter_small_obstacle_clusters,
    object_footprint_mask,
)


def make_edges(n_cells: int, cell_m: float = 0.1, origin: float = 0.0) -> np.ndarray:
    return origin + np.arange(n_cells + 1) * cell_m


def make_grid(rows: list[str], cell_m: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same row/column convention as test_pathfinding_astar.py's helper: rows are Z
    (top=iz=0), columns are X, one character per cell ('.' free, '#' obstacle, '?' unknown)."""
    height = len(rows)
    width = len(rows[0])
    char_to_val = {".": FREE, "#": OBSTACLE, "?": UNKNOWN}
    grid = np.zeros((width, height), dtype=np.uint8)
    for iz, row in enumerate(rows):
        for ix, ch in enumerate(row):
            grid[ix, iz] = char_to_val[ch]
    x_edges = make_edges(width, cell_m)
    z_edges = make_edges(height, cell_m)
    return grid, x_edges, z_edges


# --- connected_components -----------------------------------------------------------


def test_connected_components_8_connected():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True  # diagonal neighbor - 8-connected, same component
    mask[4, 4] = True  # far away - separate component
    components = connected_components(mask)
    sizes = sorted(len(c) for c in components)
    assert sizes == [1, 2]


def test_connected_components_empty_mask():
    assert connected_components(np.zeros((3, 3), dtype=bool)) == []


# --- rule (a): small-cluster filter --------------------------------------------------


def test_small_unprotected_cluster_is_reclassified_unknown():
    """A 2-cell OBSTACLE blob (< MIN_CLUSTER_CELLS) with no object nearby is phantom
    noise - demoted to UNKNOWN."""
    grid, x_edges, z_edges = make_grid(
        [
            "........",
            "..##....",
            "........",
        ]
    )
    out, n_reclassified = filter_small_obstacle_clusters(grid, x_edges, z_edges, object_bboxes_xz=[])
    assert n_reclassified == 2
    assert out[2, 1] == UNKNOWN
    assert out[3, 1] == UNKNOWN
    assert (out == FREE).sum() == (grid == FREE).sum()  # FREE cells untouched


def test_large_cluster_survives_even_without_object():
    """A cluster at/above MIN_CLUSTER_CELLS is kept as OBSTACLE regardless of object
    overlap - only SMALL clusters are treated as noise."""
    rows = ["." * 8 for _ in range(6)]
    grid, x_edges, z_edges = make_grid(rows)
    # a 2x2 block = 4 cells = MIN_CLUSTER_CELLS exactly - must survive.
    assert MIN_CLUSTER_CELLS == 4
    grid[2:4, 2:4] = OBSTACLE
    out, n_reclassified = filter_small_obstacle_clusters(grid, x_edges, z_edges, object_bboxes_xz=[])
    assert n_reclassified == 0
    assert (out[2:4, 2:4] == OBSTACLE).all()


def test_small_cluster_protected_by_real_object_survives():
    """The same small blob as above, but now a scene_objects.json bbox overlaps it -
    this is a real small object (e.g. a lamp), not noise, and must be kept OBSTACLE."""
    grid, x_edges, z_edges = make_grid(
        [
            "........",
            "..##....",
            "........",
        ]
    )
    # object bbox in world meters covering roughly the same cells as the blob
    # (cells (2,1) and (3,1), cell_m=0.1 => x in [0.2,0.4], z in [0.1,0.2])
    object_bboxes_xz = [(0.2, 0.1, 0.4, 0.2)]
    out, n_reclassified = filter_small_obstacle_clusters(
        grid, x_edges, z_edges, object_bboxes_xz=object_bboxes_xz
    )
    assert n_reclassified == 0
    assert out[2, 1] == OBSTACLE
    assert out[3, 1] == OBSTACLE


def test_object_footprint_mask_marks_overlapping_cells_only():
    x_edges = make_edges(5, 0.1)
    z_edges = make_edges(5, 0.1)
    mask = object_footprint_mask([(0.15, 0.15, 0.25, 0.25)], x_edges, z_edges)
    assert mask[1, 1] or mask[2, 2]  # overlaps somewhere in the 0.1-0.3 region
    assert not mask[4, 4]  # far corner, untouched


# --- rule (b): multi-view support filter ---------------------------------------------


def test_obstacle_cell_with_one_view_is_demoted():
    grid = np.array([[OBSTACLE]], dtype=np.uint8)
    view_counts = np.array([[1]], dtype=np.int32)
    out, n = filter_by_view_support(grid, view_counts, min_views=MIN_VIEWS_FOR_OBSTACLE)
    assert n == 1
    assert out[0, 0] == UNKNOWN


def test_obstacle_cell_with_two_views_survives():
    grid = np.array([[OBSTACLE]], dtype=np.uint8)
    view_counts = np.array([[2]], dtype=np.int32)
    out, n = filter_by_view_support(grid, view_counts, min_views=MIN_VIEWS_FOR_OBSTACLE)
    assert n == 0
    assert out[0, 0] == OBSTACLE


def test_view_support_filter_never_touches_non_obstacle_cells():
    grid = np.array([[FREE, UNKNOWN]], dtype=np.uint8)
    view_counts = np.array([[0, 0]], dtype=np.int32)
    out, n = filter_by_view_support(grid, view_counts, min_views=2)
    assert n == 0
    assert out[0, 0] == FREE
    assert out[0, 1] == UNKNOWN


def test_view_support_filter_shape_mismatch_raises():
    grid = np.zeros((2, 2), dtype=np.uint8)
    bad_counts = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        filter_by_view_support(grid, bad_counts)


# --- classify_phantom_obstacles: fallback wiring -------------------------------------


def test_classify_without_per_view_data_falls_back_to_cluster_rule_only():
    """No view_count_grid given (the common case: retain_per_view_artifacts is off by
    default) -> rule (b) must not run at all, only rule (a)."""
    grid, x_edges, z_edges = make_grid(
        [
            "........",
            "..##....",
            "........",
        ]
    )
    result = classify_phantom_obstacles(grid, x_edges, z_edges, object_bboxes_xz=[], view_count_grid=None)
    assert result["used_view_filter"] is False
    assert result["n_cells_demoted_by_view_support"] == 0
    assert result["n_cells_demoted_by_cluster_size"] == 2
    assert (result["grid"] == OBSTACLE).sum() == 0


def test_classify_with_per_view_data_runs_both_rules_in_order():
    """A single large OBSTACLE blob (>= MIN_CLUSTER_CELLS, so rule (a) alone would keep
    it) where every cell is backed by only 1 view: rule (b) demotes it all to UNKNOWN
    first, and rule (a) then sees no OBSTACLE cells left to touch."""
    rows = ["." * 8 for _ in range(6)]
    grid, x_edges, z_edges = make_grid(rows)
    grid[2:5, 2:5] = OBSTACLE  # 3x3 = 9 cells, comfortably >= MIN_CLUSTER_CELLS
    view_counts = np.ones(grid.shape, dtype=np.int32)  # every cell: exactly 1 view

    result = classify_phantom_obstacles(
        grid, x_edges, z_edges, object_bboxes_xz=[], view_count_grid=view_counts
    )
    assert result["used_view_filter"] is True
    assert result["n_cells_demoted_by_view_support"] == 9
    assert (result["grid"] == OBSTACLE).sum() == 0


def test_classify_with_per_view_data_keeps_well_supported_real_obstacle():
    """The same 3x3 blob, but every cell has 2+ views - a real, well-observed obstacle
    must survive both rules untouched."""
    rows = ["." * 8 for _ in range(6)]
    grid, x_edges, z_edges = make_grid(rows)
    grid[2:5, 2:5] = OBSTACLE
    view_counts = np.full(grid.shape, 3, dtype=np.int32)

    result = classify_phantom_obstacles(
        grid, x_edges, z_edges, object_bboxes_xz=[], view_count_grid=view_counts
    )
    assert result["n_cells_demoted_by_view_support"] == 0
    assert result["n_cells_demoted_by_cluster_size"] == 0
    assert (result["grid"][2:5, 2:5] == OBSTACLE).all()


def test_classify_thinned_cluster_can_still_lose_remaining_fragment_to_cluster_rule():
    """Rule (b) demotes most of a blob's cells (1 view each) but leaves a small
    well-supported (2-view) fragment behind that's now, post-thinning, itself under
    MIN_CLUSTER_CELLS and unprotected by any object - rule (a) should still clean it up,
    demonstrating the two rules compose (b) then (a), not just (b) alone."""
    rows = ["." * 8 for _ in range(6)]
    grid, x_edges, z_edges = make_grid(rows)
    grid[2:5, 2:5] = OBSTACLE  # 3x3 block
    view_counts = np.ones(grid.shape, dtype=np.int32)
    # Leave a 2-cell fragment (< MIN_CLUSTER_CELLS) well-supported so rule (b) keeps it,
    # but it's still noise-shaped once isolated from the rest of the (now-gone) blob.
    view_counts[2, 2] = 3
    view_counts[2, 3] = 3

    result = classify_phantom_obstacles(
        grid, x_edges, z_edges, object_bboxes_xz=[], view_count_grid=view_counts
    )
    # rule (b): 7 of the 9 cells (all but (2,2) and (2,3)) drop below min_views
    assert result["n_cells_demoted_by_view_support"] == 7
    # rule (a) then finds the surviving 2-cell fragment too small and unprotected
    assert result["n_cells_demoted_by_cluster_size"] == 2
    assert (result["grid"] == OBSTACLE).sum() == 0


# --- real-scene fixture: no per_view data -> fallback path ---------------------------


def test_real_scene_fixture_falls_back_to_rule_a_and_never_increases_obstacles(scene_horizontal_dir):
    """scene_horizontal is a real captured scene (see conftest.py) with no per_view/
    data (retain_per_view_artifacts was off when it was captured, same as essentially
    every scene on this deployment today) - classify_phantom_obstacles must run rule
    (a) alone without crashing on real geometry/object data, and must never turn an
    OBSTACLE cell into anything other than UNKNOWN (no promotions, no new obstacles)."""
    meta = json.loads((scene_horizontal_dir / "occupancy_meta.json").read_text())
    grid = np.load(scene_horizontal_dir / "occupancy.npy")
    assert grid.shape == (meta["width"], meta["height"])

    x_edges = meta["origin_x"] + np.arange(meta["width"] + 1) * meta["resolution"]
    z_edges = meta["origin_z"] + np.arange(meta["height"] + 1) * meta["resolution"]

    objects = json.loads((scene_horizontal_dir / "scene_objects" / "scene_objects.json").read_text())
    object_bboxes_xz = [
        (o["bbox_min"][0], o["bbox_min"][2], o["bbox_max"][0], o["bbox_max"][2]) for o in objects
    ]

    n_obstacle_before = int((grid == OBSTACLE).sum())
    result = classify_phantom_obstacles(
        grid, x_edges, z_edges, object_bboxes_xz=object_bboxes_xz, view_count_grid=None
    )

    assert result["used_view_filter"] is False  # no per_view data for this fixture
    out = result["grid"]
    n_obstacle_after = int((out == OBSTACLE).sum())
    assert n_obstacle_after <= n_obstacle_before
    assert n_obstacle_before - n_obstacle_after == result["n_cells_demoted_by_cluster_size"]
    # FREE cells are never touched by either rule.
    assert (out == FREE).sum() == (grid == FREE).sum()
    # Every OBSTACLE->UNKNOWN change lands exactly on cells that WERE obstacle before.
    changed = (grid == OBSTACLE) & (out != OBSTACLE)
    assert changed.sum() == result["n_cells_demoted_by_cluster_size"]
    assert (out[changed] == UNKNOWN).all()
