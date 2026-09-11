"""Unit tests for scripts/msa/geometry.py - no Blender/GPU (SPEC.md A1/A3, section
11 "Unit (no Blender/GPU)")."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.msa.geometry import (
    FREE,
    OBSTACLE,
    extract_wall_polygons,
    oriented_min_area_rect,
    simplify_polygon_preserving_passages,
)


def _make_corridor_grid() -> np.ndarray:
    """20x20 grid (indexed [ix, iz]): two obstacle blocks with a 3-cell-wide
    (0.15 m) corridor between them (iz 8..10, i.e. z in [0.4, 0.55]) along a
    jagged (non-straight) obstacle edge, so a naive Douglas-Peucker
    simplification (ignoring the passage invariant) would cut the corner and
    narrow the corridor."""
    grid = np.full((20, 20), FREE, dtype=np.uint8)
    grid[0:20, 0:8] = OBSTACLE
    # Jagged notch cut into the first block's edge, right at the corridor mouth,
    # ix 8-11 - a straight-line DP simplification of the block's boundary would
    # cut across this notch and eat into the corridor.
    grid[8:12, 8:9] = FREE
    grid[0:20, 11:20] = OBSTACLE
    return grid


class TestPassagePreservingSimplify:
    def test_never_narrows_corridor_below_original_minus_margin(self):
        grid = _make_corridor_grid()
        resolution = 0.05
        walls, dropped = extract_wall_polygons(
            grid, resolution, origin_x=0.0, origin_z=0.0, min_component_area_m2=0.01
        )
        assert len(walls) == 2, "expected two separate obstacle components either side of the corridor"

        free_ixs, free_izs = np.where(grid == FREE)
        free_points = np.array(
            [(ix * resolution + resolution / 2, iz * resolution + resolution / 2) for ix, iz in zip(free_ixs, free_izs)]
        )

        def min_dist_to_polygon(p, polygon):
            from scripts.msa.geometry import _point_to_segment_distance

            return min(_point_to_segment_distance(p, polygon[i], polygon[i + 1]) for i in range(len(polygon) - 1))

        # For every free cell, distance to each simplified wall must not have
        # shrunk more than the 2.5cm margin relative to what the *unsimplified*
        # (exact grid) obstacle boundary would give - i.e. the invariant SPEC A1
        # requires holds for the corridor region specifically.
        corridor_free_points = free_points[(free_points[:, 1] > 0.35) & (free_points[:, 1] < 0.6)]
        assert len(corridor_free_points) > 0
        for wall in walls:
            for p in corridor_free_points:
                simplified_dist = min_dist_to_polygon((float(p[0]), float(p[1])), wall.vertices)
                # The true minimum corridor half-width is 1 cell = 0.05m from the
                # straight part of each wall; simplification must not claim the
                # free cell is closer than (original - margin).
                assert simplified_dist >= 0.05 - 0.025 - 1e-9

    def test_small_components_dropped_and_logged(self):
        grid = np.full((10, 10), FREE, dtype=np.uint8)
        grid[2:3, 2:3] = OBSTACLE  # 1 cell = 0.0025 m^2 << 0.3 m^2 threshold
        walls, dropped = extract_wall_polygons(grid, 0.05, 0.0, 0.0, min_component_area_m2=0.3)
        assert walls == []
        assert len(dropped) == 1
        assert dropped[0].cell_count == 1

    def test_simplification_reduces_vertex_count_on_smooth_wall(self):
        # A large rectangular block's boundary should simplify to ~4 corners
        # (8 ring points incl. closing vertex) even though the raw rasterized
        # boundary has many collinear points.
        grid = np.full((30, 30), FREE, dtype=np.uint8)
        grid[5:25, 5:25] = OBSTACLE
        walls, dropped = extract_wall_polygons(grid, 0.05, 0.0, 0.0, min_component_area_m2=0.3)
        assert len(walls) == 1
        assert len(walls[0].vertices) <= 6, "expected the four corners plus closing vertex, not every raw grid point"


class TestOrientedMinAreaRect:
    def test_axis_aligned_rectangle_recovered(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        center, size, angle = oriented_min_area_rect(pts)
        assert center == pytest.approx((1.0, 0.5), abs=1e-6)
        assert sorted(size) == pytest.approx(sorted((2.0, 1.0)), abs=1e-6)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_rotated_rectangle_recovers_true_angle_not_aabb(self):
        # A 2x1 rectangle rotated 30 degrees. Its AABB would be larger than 2x1;
        # the oriented min-area rect must recover the true 2x1 size and 30deg tilt.
        theta = np.radians(30.0)
        w, h = 2.0, 1.0
        local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        world = local @ rot.T + np.array([5.0, -3.0])

        center, size, angle = oriented_min_area_rect(world)
        assert center == pytest.approx((5.0, -3.0), abs=1e-6)
        assert sorted(size) == pytest.approx(sorted((w, h)), abs=1e-6)
        assert min(abs(angle - theta), abs(angle - theta + np.pi), abs(angle - theta - np.pi)) < 1e-6

    def test_dense_point_cloud_hull_matches_bounding_rectangle(self):
        rng = np.random.default_rng(0)
        # Points sampled inside a known rotated rectangle (dense fill, not just
        # corners) - the hull-based min-area rect must still recover it, unlike a
        # plain AABB over these points which would be axis-aligned and larger.
        theta = np.radians(-20.0)
        w, h = 3.0, 1.2
        local = rng.uniform([-w / 2, -h / 2], [w / 2, h / 2], size=(500, 2))
        # Force exact corners into the sample so the hull touches the true extents.
        local = np.vstack([local, [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        world = local @ rot.T

        _, size, angle = oriented_min_area_rect(world)
        assert sorted(size) == pytest.approx(sorted((w, h)), abs=1e-3)

        aabb_area = (world[:, 0].max() - world[:, 0].min()) * (world[:, 1].max() - world[:, 1].min())
        assert size[0] * size[1] < aabb_area - 1e-6, "oriented rect must be tighter than the AABB for a rotated shape"


class TestWallPolygonNeverDegenerates:
    """T15b (docs/DECISIONS.md): a wall with real cell area must never export
    as a degenerate polygon. On the hero scene `wall_2` (2077 cells, 5.19 m^2,
    twelve 4-connected pieces touching at corners) traced to a 5-vertex ring of
    0.002 m^2 and simplified to an invalid 3-vertex polygon; the exporters then
    silently emitted no mesh/collider for the largest wall in the scene."""

    def test_thick_l_shaped_wall_keeps_its_area_after_simplification(self):
        from shapely.geometry import Polygon

        grid = np.full((40, 40), FREE, dtype=np.uint8)
        grid[5:35, 5:9] = OBSTACLE  # 4 cells (0.20 m) thick leg along Z
        grid[5:9, 5:35] = OBSTACLE  # 4 cells thick leg along X
        cells = int((grid == OBSTACLE).sum())
        walls, dropped = extract_wall_polygons(grid, 0.05, 0.0, 0.0)
        assert len(walls) == 1 and dropped == []
        poly = Polygon(walls[0].vertices)
        assert poly.is_valid
        assert walls[0].area_m2 == pytest.approx(cells * 0.05 * 0.05)
        assert poly.area == pytest.approx(walls[0].area_m2, rel=0.02)
        # 6 corners + the ring's (mid-edge) start vertex + the closing vertex.
        assert len(walls[0].vertices) <= 8, "an L simplifies to (about) its 6 corners"

    def test_corner_touching_pieces_export_as_one_valid_polygon_with_full_area(self):
        """Three square blocks touching only at corners form ONE 8-connected
        component (the labeling extract_wall_polygons uses) - the exact pinch
        geometry that broke the old Moore tracer."""
        from shapely.geometry import Polygon

        grid = np.full((40, 40), FREE, dtype=np.uint8)
        grid[5:15, 5:15] = OBSTACLE
        grid[15:25, 15:25] = OBSTACLE
        grid[25:35, 5:15] = OBSTACLE
        cells = 300
        walls, dropped = extract_wall_polygons(grid, 0.05, 0.0, 0.0)
        assert len(walls) == 1 and dropped == []
        poly = Polygon(walls[0].vertices)
        assert poly.is_valid, "ring must be a valid polygon, not a self-touching one"
        assert poly.area >= 0.95 * cells * 0.05 * 0.05
        assert poly.area <= 1.05 * cells * 0.05 * 0.05, "the 1 mm pinch bridge must not add visible area"

    def test_degenerate_simplification_falls_back_and_warns(self, monkeypatch, caplog):
        import logging

        from scripts.msa import geometry
        from scripts.msa.geometry import simplify_wall_ring_validated

        ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5), (0.0, 0.0)]
        monkeypatch.setattr(geometry, "simplify_polygon_preserving_passages", lambda *a, **k: [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
        with caplog.at_level(logging.WARNING, logger="scripts.msa.geometry"):
            out = simplify_wall_ring_validated(ring, np.zeros((0, 2)), cell_area_m2=0.5, wall_id="wall_7")
        assert out == ring, "raw outline is the last resort"
        assert any("wall_7" in r.message and "degenerate" in r.message for r in caplog.records)
