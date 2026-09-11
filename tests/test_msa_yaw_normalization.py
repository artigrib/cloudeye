"""Yaw-normalization post-step (T3'/T3'' rework - see scripts/msa/geometry.py's
`compute_room_polygon_yaw`/`extract_free_only_room_polygon`/`compute_wall_hull_yaw`
and scripts/msa/bootstrap.py's `run_bootstrap` integration): the GPU pipeline
never corrects the room's yaw about the vertical axis, so MSA's occupancy
grid/walls come in at whatever arbitrary rotation MapAnything's raw inference
happened to produce that run.

Primary method (T3'' rework): `compute_room_polygon_yaw` fits a minimum-area
bounding rectangle to `extract_free_only_room_polygon`'s output (FREE cells
only) and rotates the whole scene (walls, floor_polygon, objects, gaps, camera
poses) so that rectangle's long axis lands on +X. There is no confidence gate
for this method - a bounding rectangle always has a well-defined long axis
except in the degenerate near-square/near-circular case, which is handled
explicitly (no rotation, since either axis is equally "correct").

T3'' superseded T3's use of `bootstrap.extract_floor_polygon` (FREE+UNKNOWN)
as the yaw source: `gpu/stage_occupancy.py` builds the occupancy grid as a
percentile-trimmed axis-aligned bounding box of the point cloud, so UNKNOWN
cells reliably touch the grid array's own edges. Wherever the room's FREE
interior connects to that UNKNOWN halo through a doorway/gap,
`extract_floor_polygon`'s largest FREE+UNKNOWN component bounds to the *grid
array's own axis-aligned extent*, not the true (possibly rotated) room shape -
`TestExtractFreeOnlyRoomPolygon` below is the regression test for exactly this
bug. `extract_floor_polygon` itself is unchanged and still used for everything
else (object-footprint containment, floor-plate union).

Cross-check (T3''): `compute_wall_hull_yaw` fits the same min-area-rect method
to the convex hull of every wall polygon's pooled vertices - a signal that
never touches the occupancy grid at all. `bootstrap._reconcile_yaw_estimates`
applies the free-only-polygon method's angle unless it disagrees with this
cross-check by more than 10 degrees, in which case the wall-hull angle is
applied instead (`TestReconcileYawEstimates` below).

Tertiary/diagnostic method (T3' rework, still kept): `compute_dominant_wall_yaw`
histograms wall-segment edge directions mod 90 degrees and requires a >=60%
confidence peak. Its result is still computed and logged in
scene_meta.json/report.json for provenance, but never decides which result is
applied.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from scripts.msa.bootstrap import (
    _reconcile_yaw_estimates,
    _rotate_camera_poses,
    _rotate_gaps,
    _rotate_objects,
    _yaw_rotation_center,
    run_bootstrap,
)
from scripts.msa.gaps import Gap
from scripts.msa.geometry import (
    FREE,
    OBSTACLE,
    UNKNOWN,
    WallPolygon,
    compute_dominant_wall_yaw,
    compute_room_polygon_yaw,
    compute_wall_hull_yaw,
    extract_free_only_room_polygon,
    rotate_point_xz,
    rotate_wall_polygon,
)


def _rot_deg(x: float, z: float, deg: float) -> tuple[float, float]:
    """Test-only rotation helper, independent of `rotate_point_xz` (the
    function under test), used to *construct* misaligned synthetic input
    without relying on the code being tested."""
    a = math.radians(deg)
    return x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)


def _rect_wall(width: float, height: float, offset_deg: float) -> WallPolygon:
    """A single closed-ring rectangle (as if it were one traced wall polygon),
    axis-aligned at 0 degrees, then rotated by `offset_deg` about the origin
    using the independent `_rot_deg` helper above."""
    local = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    verts = [_rot_deg(x, z, offset_deg) for x, z in local]
    verts.append(verts[0])
    # Shoelace area (rotation/translation-invariant) for the WallPolygon's area_m2.
    area = 0.0
    for (x0, z0), (x1, z1) in zip(verts[:-1], verts[1:]):
        area += x0 * z1 - x1 * z0
    return WallPolygon(vertices=verts, area_m2=abs(area) / 2.0)


def _rect_polygon(width: float, height: float, offset_deg: float) -> list[tuple[float, float]]:
    """A closed rectangular ring (as `extract_floor_polygon` would return),
    axis-aligned at 0 degrees then rotated by `offset_deg` about the origin."""
    local = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    verts = [_rot_deg(x, z, offset_deg) for x, z in local]
    verts.append(verts[0])
    return verts


class TestComputeDominantWallYaw:
    """Now the secondary/diagnostic method (see module docstring) - unchanged
    behavior, still covered directly since bootstrap still computes and logs
    it as a cross-check."""

    def test_clearly_misaligned_room_is_detected(self):
        # A single rectangular room, all 4 walls consistently rotated 15
        # degrees off-axis - the textbook "staircase wall" case the task
        # describes.
        wall = _rect_wall(4.0, 3.0, offset_deg=15.0)
        result = compute_dominant_wall_yaw([wall])

        assert result is not None
        assert result["peak_share"] == pytest.approx(1.0, abs=1e-9)
        assert result["dominant_angle_deg"] == pytest.approx(15.0, abs=1e-6)
        # Correction should bring 15 degrees back to axis-aligned (0/90).
        assert result["correction_deg"] == pytest.approx(-15.0, abs=1e-6)

        # Applying the correction actually axis-aligns the room: re-running
        # the detector on the rotated result finds a ~0-degree dominant angle.
        rotated = rotate_wall_polygon(wall, result["correction_rad"], center_xz=(0.0, 0.0))
        verified = compute_dominant_wall_yaw([rotated])
        assert verified is not None
        assert verified["dominant_angle_deg"] == pytest.approx(0.0, abs=1e-6)
        assert verified["correction_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_octagonal_room_has_no_dominant_direction(self):
        # A regular octagon's edges collapse, mod 90 degrees, into exactly
        # two equally-weighted (50/50) direction families - neither reaches
        # the 60% confidence gate, so this (secondary/diagnostic) method
        # reports "no clear direction". The primary polygon method (below)
        # has no such gate and still produces a well-defined result for the
        # same shape (a regular octagon's bounding rect is near-square, so it
        # lands in the explicit degenerate-square case instead).
        n, radius = 8, 3.0
        verts = [
            (radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n)) for k in range(n)
        ]
        verts.append(verts[0])
        octagon = WallPolygon(vertices=verts, area_m2=1.0)  # area_m2 unused by the detector

        result = compute_dominant_wall_yaw([octagon])
        assert result is None

    def test_no_walls_returns_none(self):
        assert compute_dominant_wall_yaw([]) is None

    def test_already_axis_aligned_room_needs_no_correction(self):
        wall = _rect_wall(5.0, 2.0, offset_deg=0.0)
        result = compute_dominant_wall_yaw([wall])
        assert result is not None
        assert result["dominant_angle_deg"] == pytest.approx(0.0, abs=1e-9)
        assert result["correction_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_deterministic_across_repeated_calls(self):
        wall = _rect_wall(4.0, 3.0, offset_deg=15.0)
        r1 = compute_dominant_wall_yaw([wall])
        r2 = compute_dominant_wall_yaw([wall])
        assert r1 == r2


class TestComputeRoomPolygonYaw:
    """Primary yaw-detection method (T3' rework): minimum-area bounding
    rectangle of `floor_polygon`, long axis -> +X, no confidence gate."""

    def test_clearly_misaligned_room_is_detected(self):
        floor_polygon = _rect_polygon(4.0, 3.0, offset_deg=15.0)
        result = compute_room_polygon_yaw(floor_polygon)

        assert result is not None
        assert result["is_degenerate_square"] is False
        assert result["long_axis_angle_deg"] == pytest.approx(15.0, abs=1e-6)
        assert result["correction_deg"] == pytest.approx(-15.0, abs=1e-6)
        assert result["rect_size_m"] == pytest.approx((4.0, 3.0), abs=1e-6)

        # Applying the correction lands the long side on +X: re-fitting the
        # rotated polygon reports ~0-degree long-axis angle.
        rotated = [rotate_point_xz(x, z, result["correction_rad"], (0.0, 0.0)) for x, z in floor_polygon]
        verified = compute_room_polygon_yaw(rotated)
        assert verified is not None
        assert verified["long_axis_angle_deg"] == pytest.approx(0.0, abs=1e-6)
        assert verified["correction_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_no_confidence_gate_still_applies_to_low_histogram_confidence_shape(self):
        # A regular octagon: compute_dominant_wall_yaw refuses (see above),
        # but its bounding rectangle is near-square (a regular octagon is
        # close to circular) - which is exactly the explicit degenerate case,
        # not a gated failure. There is no "None, no clear direction" outcome
        # for this method at all.
        n, radius = 8, 3.0
        verts = [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n)) for k in range(n)]
        verts.append(verts[0])
        result = compute_room_polygon_yaw(verts)
        assert result is not None
        assert result["is_degenerate_square"] is True
        assert result["correction_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_square_room_is_degenerate_no_rotation(self):
        floor_polygon = _rect_polygon(3.0, 3.0, offset_deg=27.0)
        result = compute_room_polygon_yaw(floor_polygon)
        assert result is not None
        assert result["is_degenerate_square"] is True
        assert result["correction_rad"] == 0.0
        assert result["correction_deg"] == 0.0

    def test_near_square_within_tolerance_is_degenerate(self):
        # 1% length/width difference, tolerance defaults to 2% - counts as square.
        floor_polygon = _rect_polygon(3.0, 2.97, offset_deg=10.0)
        result = compute_room_polygon_yaw(floor_polygon)
        assert result is not None
        assert result["is_degenerate_square"] is True

    def test_outside_tolerance_is_not_degenerate(self):
        # 10% length/width difference - clearly outside the default 2% tolerance.
        floor_polygon = _rect_polygon(3.0, 2.7, offset_deg=10.0)
        result = compute_room_polygon_yaw(floor_polygon)
        assert result is not None
        assert result["is_degenerate_square"] is False
        assert result["correction_deg"] == pytest.approx(-10.0, abs=1e-6)

    def test_already_axis_aligned_room_needs_no_correction(self):
        floor_polygon = _rect_polygon(5.0, 2.0, offset_deg=0.0)
        result = compute_room_polygon_yaw(floor_polygon)
        assert result is not None
        assert result["long_axis_angle_deg"] == pytest.approx(0.0, abs=1e-9)
        assert result["correction_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_none_floor_polygon_returns_none(self):
        assert compute_room_polygon_yaw(None) is None

    def test_too_few_vertices_returns_none(self):
        assert compute_room_polygon_yaw([(0.0, 0.0), (1.0, 0.0)]) is None

    def test_deterministic_across_repeated_calls(self):
        floor_polygon = _rect_polygon(4.0, 3.0, offset_deg=15.0)
        r1 = compute_room_polygon_yaw(floor_polygon)
        r2 = compute_room_polygon_yaw(floor_polygon)
        assert r1 == r2


class TestExtractFreeOnlyRoomPolygon:
    """Regression coverage for the diagnosed bug (T3'' rework):
    `bootstrap.extract_floor_polygon` builds its polygon from the largest
    connected FREE+UNKNOWN component. `gpu/stage_occupancy.py` builds the
    occupancy grid as a percentile-trimmed axis-aligned bounding box of the
    point cloud, so UNKNOWN cells reliably touch the grid array's own edges.
    Wherever the room's FREE interior connects to that UNKNOWN halo through a
    doorway/gap (no OBSTACLE wall in between), the largest FREE+UNKNOWN
    component's bounding box ends up measuring the *grid array's own
    axis-aligned extent*, not the true (possibly rotated) room shape.
    `extract_free_only_room_polygon` fixes the yaw signal's source by counting
    FREE cells only - UNKNOWN never counts as room, so this doorway-to-halo
    connection can't drag the polygon out to the grid's bounding box."""

    @staticmethod
    def _build_rotated_room_with_doorway_to_unknown_halo(
        resolution: float = 0.05,
        grid_size: int = 80,
        rect_size: tuple[float, float] = (2.0, 1.4),
        angle_deg: float = 25.0,
        doorway_width_cells: int = 3,
    ):
        """A `grid_size` x `grid_size` occupancy grid, entirely UNKNOWN by
        default - touching every array edge, exactly as `gpu/stage_occupancy.py`'s
        percentile-trimmed bounding box always does - containing one rotated
        rectangular room: OBSTACLE walls tracing the rectangle's boundary,
        FREE cells filling its interior, except for a `doorway_width_cells`-
        wide gap in one wall where there's no OBSTACLE at all. The room's FREE
        interior is directly adjacent, through that gap, to the UNKNOWN cells
        immediately outside it - which are themselves connected, all the way
        out, to the grid's UNKNOWN border. This is exactly the shape of the
        diagnosed bug: FREE interior -> (no wall) -> UNKNOWN -> grid edge."""
        origin_x, origin_z = 0.0, 0.0
        occ = np.full((grid_size, grid_size), UNKNOWN, dtype=np.uint8)

        cx, cz = (grid_size * resolution) / 2.0, (grid_size * resolution) / 2.0
        half_w, half_h = rect_size[0] / 2.0, rect_size[1] / 2.0
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        xs = origin_x + (np.arange(grid_size) + 0.5) * resolution
        zs = origin_z + (np.arange(grid_size) + 0.5) * resolution
        grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")  # occupancy is [ix, iz]
        dx, dz = grid_x - cx, grid_z - cz
        # Inverse-rotate world coords into the rectangle's own local (u, v) frame.
        local_u = dx * cos_a + dz * sin_a
        local_v = -dx * sin_a + dz * cos_a

        wall_thickness = 2 * resolution
        interior = (np.abs(local_u) <= half_w) & (np.abs(local_v) <= half_h)
        with_wall = (np.abs(local_u) <= half_w + wall_thickness) & (np.abs(local_v) <= half_h + wall_thickness)
        wall_ring = with_wall & ~interior

        occ[interior] = FREE
        occ[wall_ring] = OBSTACLE

        # Carve a doorway gap in the +U wall, centered on its midpoint:
        # revert those wall cells back to UNKNOWN so the FREE interior
        # touches UNKNOWN directly, with no OBSTACLE in between.
        doorway_half = (doorway_width_cells * resolution) / 2.0
        doorway = with_wall & ~interior & (local_u > half_w - 1e-9) & (np.abs(local_v) <= doorway_half)
        occ[doorway] = UNKNOWN

        return occ, resolution, origin_x, origin_z

    def test_free_only_polygon_ignores_unknown_halo_and_finds_true_angle(self):
        angle_deg = 25.0
        rect_size = (2.0, 1.4)
        occ, resolution, origin_x, origin_z = self._build_rotated_room_with_doorway_to_unknown_halo(
            angle_deg=angle_deg, rect_size=rect_size
        )

        # Sanity check the bug actually reproduces on this synthetic grid:
        # extract_floor_polygon's FREE+UNKNOWN mask should indeed bound to
        # (approximately) the grid's own square axis-aligned extent, not the
        # true 2.0m x 1.4m rotated rectangle - a near-square bounding rect
        # here is itself evidence of the bug, not a real finding.
        from scripts.msa.bootstrap import extract_floor_polygon

        buggy_polygon = extract_floor_polygon(occ, resolution, origin_x, origin_z)
        assert buggy_polygon is not None
        buggy_result = compute_room_polygon_yaw(buggy_polygon)
        assert buggy_result is not None
        assert buggy_result["is_degenerate_square"] is True

        free_only_polygon = extract_free_only_room_polygon(occ, resolution, origin_x, origin_z)
        assert free_only_polygon is not None
        result = compute_room_polygon_yaw(free_only_polygon)
        assert result is not None
        assert result["is_degenerate_square"] is False
        # 5cm-grid rasterization noise means this won't be exact - within a
        # couple of degrees/centimeters of the true synthetic shape confirms
        # this measured the room, not the 4.0m x 4.0m grid extent.
        assert result["long_axis_angle_deg"] == pytest.approx(angle_deg, abs=2.0)
        assert result["rect_size_m"] == pytest.approx(rect_size, abs=0.15)

    def test_deterministic_across_repeated_calls(self):
        occ, resolution, origin_x, origin_z = self._build_rotated_room_with_doorway_to_unknown_halo()
        p1 = extract_free_only_room_polygon(occ, resolution, origin_x, origin_z)
        p2 = extract_free_only_room_polygon(occ, resolution, origin_x, origin_z)
        assert p1 == p2

    def test_closing_bridges_a_narrow_split_in_the_free_region(self):
        """A single thin OBSTACLE sliver splitting an otherwise-rectangular
        FREE room into two nearly-equal-sized halves (e.g. a mis-rasterized
        doorway threshold or an occluded scan cell) would, without closing
        happening *before* the largest-component selection, corrupt the
        measured shape down to just one half. `closing_iterations=2` bridges
        gaps up to 2 cells wide, healing the split so the true full-room
        extent is recovered."""
        resolution = 0.05
        grid_size = 60
        occ = np.full((grid_size, grid_size), UNKNOWN, dtype=np.uint8)
        occ[10:50, 10:50] = FREE
        # 2-cell-wide slit at iz 29..30 (constant z) splitting the room into
        # two halves along Z (the grid is [ix, iz]).
        occ[10:50, 29:31] = OBSTACLE

        polygon = extract_free_only_room_polygon(occ, resolution, 0.0, 0.0)
        assert polygon is not None
        poly_points = np.array(polygon)
        # The closed, healed polygon should span the full 40-cell-deep
        # square in Z, not just one ~19-cell-deep half.
        depth = poly_points[:, 1].max() - poly_points[:, 1].min()
        assert depth == pytest.approx(40 * resolution, abs=1e-9)

    def test_no_free_cells_returns_none(self):
        occ = np.full((10, 10), UNKNOWN, dtype=np.uint8)
        assert extract_free_only_room_polygon(occ, 0.05, 0.0, 0.0) is None


class TestComputeWallHullYaw:
    """Independent cross-check (T3'' rework): fits the same min-area-rect
    method to the convex hull of every wall polygon's pooled vertices,
    reusing `compute_room_polygon_yaw` itself against that hull."""

    def test_matches_room_polygon_yaw_for_a_single_rectangular_wall(self):
        wall = _rect_wall(4.0, 3.0, offset_deg=18.0)
        result = compute_wall_hull_yaw([wall])
        assert result is not None
        assert result["is_degenerate_square"] is False
        assert result["long_axis_angle_deg"] == pytest.approx(18.0, abs=1e-6)
        assert result["rect_size_m"] == pytest.approx((4.0, 3.0), abs=1e-6)

    def test_no_walls_returns_none(self):
        assert compute_wall_hull_yaw([]) is None

    def test_deterministic_across_repeated_calls(self):
        wall = _rect_wall(4.0, 3.0, offset_deg=18.0)
        r1 = compute_wall_hull_yaw([wall])
        r2 = compute_wall_hull_yaw([wall])
        assert r1 == r2


class TestReconcileYawEstimates:
    """`bootstrap._reconcile_yaw_estimates` (morning-1 policy, replaces T15a): the
    WALL-HULL estimate is applied whenever it exists and has a long axis; the
    room-polygon estimate is the fallback (no walls / degenerate hull) and is
    otherwise only compared and logged."""

    def _result(self, angle_deg: float, is_degenerate_square: bool = False):
        return {
            "long_axis_angle_deg": angle_deg,
            "correction_deg": -angle_deg,
            "correction_rad": math.radians(-angle_deg),
            "is_degenerate_square": is_degenerate_square,
            "rect_size_m": (4.0, 3.0),
            "square_tolerance": 0.02,
        }

    def test_agreement_within_threshold_applies_wall_hull(self):
        free_only = self._result(20.0)
        wall_hull = self._result(25.0)  # 5 degrees apart - within the 10-degree threshold
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, wall_hull)
        assert applied is wall_hull
        assert fallback_to_polygon is False
        assert disagreement_deg == pytest.approx(5.0, abs=1e-9)

    def test_disagreement_over_threshold_still_applies_wall_hull_and_reports_it(self):
        # The disagreement is logged (the caller flags it), never acted on - the
        # wall hull is the applied estimate whatever the polygon says.
        free_only = self._result(20.0)
        wall_hull = self._result(50.0)  # 30 degrees apart - over the 10-degree threshold
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, wall_hull)
        assert applied is wall_hull
        assert fallback_to_polygon is False
        assert disagreement_deg == pytest.approx(30.0, abs=1e-9)

    def test_disagreement_wraps_correctly_near_90_degrees(self):
        # -85 and 85 degrees describe axes only 10 degrees apart (an
        # undirected line, not a direction) - not the ~170 a naive abs(a - b)
        # would report.
        free_only = self._result(-85.0)
        wall_hull = self._result(85.0)
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, wall_hull)
        assert disagreement_deg == pytest.approx(10.0, abs=1e-9)
        assert applied is wall_hull
        assert fallback_to_polygon is False

    def test_degenerate_polygon_does_not_matter_when_the_wall_hull_is_usable(self):
        free_only = self._result(0.0, is_degenerate_square=True)
        wall_hull = self._result(40.0)
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, wall_hull)
        assert applied is wall_hull
        assert fallback_to_polygon is False
        assert disagreement_deg is None

    def test_degenerate_polygon_without_wall_hull_applies_nothing(self):
        free_only = self._result(0.0, is_degenerate_square=True)
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, None)
        assert applied is free_only and applied["correction_deg"] == 0.0
        assert fallback_to_polygon is True
        assert disagreement_deg is None

    def test_degenerate_wall_hull_falls_back_to_polygon(self):
        # A near-square hull has no long axis -> the polygon is applied (logged
        # as a fallback), no disagreement is measured.
        free_only = self._result(40.0)
        wall_hull = self._result(0.0, is_degenerate_square=True)
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, wall_hull)
        assert applied is free_only
        assert fallback_to_polygon is True
        assert disagreement_deg is None

    def test_missing_free_only_applies_wall_hull(self):
        wall_hull = self._result(40.0)
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(None, wall_hull)
        assert applied is wall_hull
        assert fallback_to_polygon is False
        assert disagreement_deg is None

    def test_missing_wall_hull_falls_back_to_polygon(self):
        free_only = self._result(40.0)
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(free_only, None)
        assert applied is free_only
        assert fallback_to_polygon is True
        assert disagreement_deg is None

    def test_both_missing_returns_none(self):
        applied, fallback_to_polygon, disagreement_deg = _reconcile_yaw_estimates(None, None)
        assert applied is None
        assert fallback_to_polygon is False
        assert disagreement_deg is None


class TestSceneRotationHelpers:
    """Exercise the pieces `run_bootstrap` composes: rotation center,
    object/gap rotation, and camera pose rotation. Unaffected by the T3'
    method swap - these operate on a caller-supplied angle/center regardless
    of which yaw-detection method produced it."""

    def test_yaw_rotation_center_prefers_floor_polygon_centroid(self):
        floor_polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)]
        center = _yaw_rotation_center(floor_polygon, walls=[])
        assert center == pytest.approx((1.0, 2.0))

    def test_yaw_rotation_center_falls_back_to_wall_vertex_mean(self):
        wall = WallPolygon(vertices=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)], area_m2=4.0)
        center = _yaw_rotation_center(None, walls=[wall])
        # Mean of the 5 listed vertices (closing vertex duplicated, matching
        # the function's straightforward "mean of every wall vertex" contract).
        xs = [0.0, 2.0, 2.0, 0.0, 0.0]
        zs = [0.0, 0.0, 2.0, 2.0, 0.0]
        assert center == pytest.approx((sum(xs) / len(xs), sum(zs) / len(zs)))

    def test_rotate_objects_updates_center_and_angle_not_size(self):
        objects = [
            {
                "id": "table_0",
                "center_xy": (1.0, 0.0),
                "size_uv": (0.5, 0.3),
                "angle_rad": 0.0,
                "hull_xz": [[0.8, -0.1], [1.2, -0.1], [1.2, 0.1], [0.8, 0.1]],
            }
        ]
        angle_rad = math.radians(40.0)
        rotated = _rotate_objects(objects, angle_rad, center_xz=(0.0, 0.0))

        obj = rotated[0]
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        assert obj["center_xy"][0] == pytest.approx(1.0 * cos_a, abs=1e-9)
        assert obj["center_xy"][1] == pytest.approx(1.0 * sin_a, abs=1e-9)
        # size_uv is rotation-invariant - untouched, same tuple contents.
        assert obj["size_uv"] == (0.5, 0.3)
        # angle_rad shifted by 40 degrees (well within [-pi/2, pi/2), no
        # periodic wraparound ambiguity to worry about at this angle).
        assert obj["angle_rad"] == pytest.approx(math.radians(40.0), abs=1e-9)
        # hull_xz vertices rotated the same way as center_xy.
        rotated_hull = np.array(obj["hull_xz"])
        original_hull = np.array(objects[0]["hull_xz"])
        expected = np.column_stack(
            [
                original_hull[:, 0] * cos_a - original_hull[:, 1] * sin_a,
                original_hull[:, 0] * sin_a + original_hull[:, 1] * cos_a,
            ]
        )
        assert rotated_hull == pytest.approx(expected, abs=1e-9)

        # Original input must be untouched (no in-place mutation).
        assert objects[0]["center_xy"] == (1.0, 0.0)

    def test_rotate_gaps_moves_point_preserves_width(self):
        gap = Gap(
            a_id="wall_0",
            b_id="table_0",
            a_kind="wall",
            b_kind="object",
            width_m=0.42,
            measurement_point_xy=(1.0, 0.0),
            platform_verdicts=[],
        )
        rotated = _rotate_gaps([gap], math.radians(90.0), center_xz=(0.0, 0.0))
        assert rotated[0].width_m == gap.width_m
        assert rotated[0].a_id == gap.a_id
        assert rotated[0].measurement_point_xy[0] == pytest.approx(0.0, abs=1e-9)
        assert rotated[0].measurement_point_xy[1] == pytest.approx(1.0, abs=1e-9)

    def test_rotate_camera_poses_matches_manual_position_rotation(self):
        cameras_json = {
            "schema_version": 2,
            "cameras": [[1.0, 1.5, 0.0]],
            "intrinsics": [[[280.0, 0.0, 150.0], [0.0, 280.0, 260.0], [0.0, 0.0, 1.0]]],
            "extrinsics": [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
        }
        angle_rad = math.radians(90.0)
        rotated = _rotate_camera_poses(cameras_json, angle_rad, center_xz=(0.0, 0.0))

        # Position rotates the same way as any other world XZ point.
        expected_x, expected_z = rotate_point_xz(1.0, 0.0, angle_rad, (0.0, 0.0))
        assert rotated["cameras"][0][0] == pytest.approx(expected_x, abs=1e-9)
        assert rotated["cameras"][0][1] == pytest.approx(1.5)  # y (up) untouched
        assert rotated["cameras"][0][2] == pytest.approx(expected_z, abs=1e-9)

        # The extrinsic's translation column must match the rotated position.
        new_extrinsic = np.array(rotated["extrinsics"][0])
        assert new_extrinsic[0, 3] == pytest.approx(expected_x, abs=1e-9)
        assert new_extrinsic[1, 3] == pytest.approx(1.5, abs=1e-9)
        assert new_extrinsic[2, 3] == pytest.approx(expected_z, abs=1e-9)

        # Intrinsics are untouched (world orientation doesn't affect them).
        assert rotated["intrinsics"] == cameras_json["intrinsics"]

    def test_rotate_camera_poses_is_deterministic(self):
        cameras_json = {
            "schema_version": 2,
            "cameras": [[1.0, 1.5, 0.5], [-0.5, 1.2, 2.0]],
            "intrinsics": [
                [[280.0, 0.0, 150.0], [0.0, 280.0, 260.0], [0.0, 0.0, 1.0]],
                [[280.0, 0.0, 150.0], [0.0, 280.0, 260.0], [0.0, 0.0, 1.0]],
            ],
            "extrinsics": [
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]],
                [[0.7, 0.0, 0.7, -0.5], [0.0, 1.0, 0.0, 1.2], [-0.7, 0.0, 0.7, 2.0], [0.0, 0.0, 0.0, 1.0]],
            ],
        }
        angle_rad = math.radians(12.3)
        r1 = _rotate_camera_poses(copy.deepcopy(cameras_json), angle_rad, (0.3, -0.2))
        r2 = _rotate_camera_poses(copy.deepcopy(cameras_json), angle_rad, (0.3, -0.2))
        assert r1 == r2


def _build_simple_room_occupancy(resolution: float = 0.05, width: float = 4.0, height: float = 3.0, thickness: float = 0.2):
    """A plain axis-aligned rectangular room (FREE interior, OBSTACLE border),
    just so `run_bootstrap`'s other stages (floor_polygon extraction, gap
    computation) have something sane to chew on. Deliberately NOT the source
    of the rotated geometry used by `TestRunBootstrapYawIntegration` below -
    see that class's docstring for why `extract_wall_polygons` and
    `extract_floor_polygon` are monkeypatched instead of relying on
    rasterizing truly rotated geometry."""
    from scripts.msa.geometry import FREE, OBSTACLE

    nx = int(round((width + 2 * thickness) / resolution))  # occupancy is [ix, iz]: axis 0 = X
    nz = int(round((height + 2 * thickness) / resolution))
    occ = np.full((nx, nz), OBSTACLE, dtype=np.uint8)
    inner_c = int(round(thickness / resolution))
    occ[inner_c : nx - inner_c, inner_c : nz - inner_c] = FREE
    origin_x, origin_z = 0.0, 0.0
    return occ, resolution, origin_x, origin_z


def _write_simple_room_scene(scene_dir) -> None:
    occ, resolution, origin_x, origin_z = _build_simple_room_occupancy()
    np.save(scene_dir / "occupancy.npy", occ)
    (scene_dir / "occupancy_meta.json").write_text(
        json.dumps({"resolution": resolution, "origin_x": origin_x, "origin_z": origin_z})
    )
    (scene_dir / "scene_meta.json").write_text(json.dumps({"floor_y": 0.0, "ceiling_y": 2.5}))


class TestRunBootstrapYawIntegration:
    """End-to-end: `run_bootstrap` should derive its correction from
    `floor_polygon`'s minimum-area bounding rectangle (primary method), log
    `compute_dominant_wall_yaw`'s histogram result as a secondary/diagnostic
    cross-check, and rotate walls/floor_polygon/objects/gaps/scene_meta.json
    consistently - reproducibly across repeated runs.

    `extract_wall_polygons` and `extract_room_polygon` (T15a: the room
    reference polygon `run_bootstrap` now feeds the yaw step - see
    tests/test_msa_room_polygon.py for its own construction tests and an
    unpatched end-to-end rotated-grid run) (real occupancy-grid
    rasterizers + Douglas-Peucker simplifiers - already covered by their own
    tests, out of scope for this change) are monkeypatched to return clean,
    exactly-known rotated geometry, rather than rasterizing genuinely-rotated
    shapes onto a 5cm grid and hoping DP simplification collapses them
    cleanly enough - rasterizing a small rotated rectangle at 5cm resolution
    produces real "staircase" quantization noise, a real characteristic of
    those functions' output on small synthetic geometry, not a bug in
    either yaw-detection method; this class exercises `run_bootstrap`'s *use*
    of their results, so it controls the inputs directly instead of fighting
    rasterization noise. (Both detection methods, including behavior on
    real-world-shaped noisy input, are covered directly above and against the
    real 02_modular_home fixture in `tests/test_msa_bootstrap.py`.)"""

    OFFSET_DEG = 20.0

    def _rotated_wall(self, offset_deg=OFFSET_DEG):
        return _rect_wall(4.0, 3.0, offset_deg=offset_deg)

    def _rotated_floor_polygon(self, offset_deg=OFFSET_DEG):
        return _rect_polygon(4.0, 3.0, offset_deg=offset_deg)

    def test_yaw_detected_and_applied_from_free_only_polygon(self, tmp_path, monkeypatch):
        # Walls and free_only_room_polygon agree on the same 20-degree offset
        # here - the straightforward case where both the primary method and
        # its wall-hull cross-check concur (well within the 10-degree
        # disagreement threshold - they're exactly equal).
        monkeypatch.setattr("scripts.msa.bootstrap.extract_wall_polygons", lambda *a, **k: ([self._rotated_wall()], []))
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_room_polygon", lambda *a, **k: self._rotated_floor_polygon()
        )

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        _write_simple_room_scene(scene_dir)

        out_dir = tmp_path / "out"
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[])

        assert report["yaw_method"] == "wall_hull_min_area_rect"
        assert report["yaw_applied_source"] == "wall_hull"
        assert report["yaw_applied"] is True
        assert report["yaw_is_degenerate_square"] is False
        assert report["yaw_correction_deg"] == pytest.approx(-self.OFFSET_DEG, abs=1e-6)
        assert report["yaw_fallback_to_polygon"] is False
        assert report["yaw_disagreement_deg"] == pytest.approx(0.0, abs=1e-6)
        assert report["yaw_free_only_long_axis_angle_deg"] == pytest.approx(self.OFFSET_DEG, abs=1e-6)
        assert report["yaw_wall_hull_long_axis_angle_deg"] == pytest.approx(self.OFFSET_DEG, abs=1e-6)
        # Secondary/diagnostic histogram cross-check agrees in this scenario.
        assert report["yaw_histogram_gate_passed"] is True
        assert report["yaw_histogram_peak_share"] == pytest.approx(1.0, abs=1e-9)
        assert report["yaw_histogram_correction_deg"] == pytest.approx(-self.OFFSET_DEG, abs=1e-6)

        scene_meta = json.loads((out_dir / "scene_meta.json").read_text())
        assert scene_meta["yaw_applied"] is True
        assert scene_meta["yaw_correction_rad"] == pytest.approx(report["yaw_correction_rad"])
        assert scene_meta["yaw_long_axis_angle_deg"] == pytest.approx(self.OFFSET_DEG, abs=1e-6)
        assert scene_meta["floor_y"] == 0.0 and scene_meta["ceiling_y"] == 2.5

    def test_yaw_applied_even_when_histogram_confidence_gate_fails(self, tmp_path, monkeypatch):
        # This is the headline behavior change (T3' rework, still true under
        # T3''): walls are an octagon (the secondary/diagnostic histogram
        # method refuses - < 60% peak share, and the wall hull is
        # degenerate-square), but free_only_room_polygon is a clean
        # rotated rectangle - morning-1: a degenerate hull falls back to the
        # polygon, which has no confidence gate, so normalization still applies.
        n, radius = 8, 3.0
        octagon_verts = [
            (radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n)) for k in range(n)
        ]
        octagon_verts.append(octagon_verts[0])
        octagon_wall = WallPolygon(vertices=octagon_verts, area_m2=1.0)

        monkeypatch.setattr("scripts.msa.bootstrap.extract_wall_polygons", lambda *a, **k: ([octagon_wall], []))
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_room_polygon", lambda *a, **k: self._rotated_floor_polygon()
        )

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        _write_simple_room_scene(scene_dir)

        out_dir = tmp_path / "out"
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[])

        assert report["yaw_applied"] is True
        assert report["yaw_correction_deg"] == pytest.approx(-self.OFFSET_DEG, abs=1e-6)
        assert report["yaw_method"] == "room_polygon_min_area_rect"
        assert report["yaw_applied_source"] == "room_polygon"
        assert report["yaw_fallback_to_polygon"] is True
        # The wall hull is degenerate-square (an octagon's convex hull is
        # near-circular) - _reconcile_yaw_estimates falls back to the polygon's
        # (non-degenerate) result; no disagreement is measured against an
        # ambiguous hull.
        assert report["yaw_wall_hull_is_degenerate_square"] is True
        assert report["yaw_disagreement_deg"] is None
        # Diagnostic cross-check correctly records that the histogram method
        # itself found no clear dominant direction.
        assert report["yaw_histogram_gate_passed"] is False
        assert report["yaw_histogram_correction_deg"] is None
        assert report["yaw_histogram_peak_share"] is None

    def test_degenerate_square_room_polygon_is_irrelevant_when_walls_exist(self, tmp_path, monkeypatch):
        # A square room polygon has no well-defined long axis, but the wall-hull
        # estimate (from the real rotated-rectangle walls, 20 degrees) is the
        # applied one anyway - the polygon's degeneracy is only recorded.
        monkeypatch.setattr("scripts.msa.bootstrap.extract_wall_polygons", lambda *a, **k: ([self._rotated_wall()], []))
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_room_polygon",
            lambda *a, **k: _rect_polygon(3.0, 3.0, offset_deg=33.0),
        )

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        _write_simple_room_scene(scene_dir)

        out_dir = tmp_path / "out"
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[])

        assert report["yaw_free_only_is_degenerate_square"] is True
        assert report["yaw_method"] == "wall_hull_min_area_rect"
        assert report["yaw_fallback_to_polygon"] is False
        assert report["yaw_is_degenerate_square"] is False
        assert report["yaw_correction_deg"] == pytest.approx(-self.OFFSET_DEG, abs=1e-6)
        assert report["yaw_disagreement_deg"] is None

    def test_no_walls_falls_back_to_room_polygon(self, tmp_path, monkeypatch):
        # Morning-1's only fallback: no wall polygons at all -> the room polygon's
        # angle is applied and the fallback is recorded.
        monkeypatch.setattr("scripts.msa.bootstrap.extract_wall_polygons", lambda *a, **k: ([], []))
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_room_polygon", lambda *a, **k: self._rotated_floor_polygon()
        )

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        _write_simple_room_scene(scene_dir)

        out_dir = tmp_path / "out"
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[])

        assert report["yaw_method"] == "room_polygon_min_area_rect"
        assert report["yaw_applied_source"] == "room_polygon"
        assert report["yaw_fallback_to_polygon"] is True
        assert report["yaw_wall_hull_long_axis_angle_deg"] is None
        assert report["yaw_correction_deg"] == pytest.approx(-self.OFFSET_DEG, abs=1e-6)
        assert report["yaw_disagreement_deg"] is None

    def test_disagreement_over_threshold_still_applies_wall_hull(self, tmp_path, monkeypatch):
        # The room polygon and the wall hull disagree by 60 degrees (well over
        # the 10-degree threshold). Morning-1: the wall-hull angle is what gets
        # applied; the disagreement is only logged/flagged.
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_wall_polygons", lambda *a, **k: ([self._rotated_wall(offset_deg=20.0)], [])
        )
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_room_polygon",
            lambda *a, **k: self._rotated_floor_polygon(offset_deg=-40.0),
        )

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        _write_simple_room_scene(scene_dir)

        out_dir = tmp_path / "out"
        report = run_bootstrap(scene_dir, out_dir, object_inputs=[])

        assert report["yaw_method"] == "wall_hull_min_area_rect"
        assert report["yaw_fallback_to_polygon"] is False
        assert report["yaw_cross_check_disagrees"] is True
        assert report["yaw_free_only_long_axis_angle_deg"] == pytest.approx(-40.0, abs=1e-6)
        assert report["yaw_wall_hull_long_axis_angle_deg"] == pytest.approx(20.0, abs=1e-6)
        assert report["yaw_disagreement_deg"] == pytest.approx(60.0, abs=1e-6)
        # Applied correction matches the wall hull's angle, not the room polygon's.
        assert report["yaw_correction_deg"] == pytest.approx(-20.0, abs=1e-6)
        assert report["yaw_long_axis_angle_deg"] == pytest.approx(20.0, abs=1e-6)
        # ... and the Manhattan snap happened in the APPLIED frame: the exported
        # (rotated) room polygon is axis-aligned even though its own estimate
        # was 60 degrees off.
        scene_meta = json.loads((out_dir / "scene_meta.json").read_text())
        room = scene_meta["room_polygon"]
        assert scene_meta["room_polygon_regularized"] is True
        for (x0, z0), (x1, z1) in zip(room, room[1:] + room[:1]):
            assert min(abs(x1 - x0), abs(z1 - z0)) < 1e-6, "edge is not axis-aligned in the applied frame"

    def test_yaw_normalization_is_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.msa.bootstrap.extract_wall_polygons", lambda *a, **k: ([self._rotated_wall()], []))
        monkeypatch.setattr(
            "scripts.msa.bootstrap.extract_room_polygon", lambda *a, **k: self._rotated_floor_polygon()
        )

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        _write_simple_room_scene(scene_dir)

        out_dir_1 = tmp_path / "out1"
        out_dir_2 = tmp_path / "out2"
        report_1 = run_bootstrap(scene_dir, out_dir_1, object_inputs=[])
        report_2 = run_bootstrap(scene_dir, out_dir_2, object_inputs=[])

        assert report_1 == report_2
        assert (out_dir_1 / "gaps.json").read_text() == (out_dir_2 / "gaps.json").read_text()
        assert (out_dir_1 / "scene_meta.json").read_text() == (out_dir_2 / "scene_meta.json").read_text()
