"""Pure-unit tests for app.services.goal_resolution - the shared "nearest free cell
to an object's footprint EDGE, with real clearance" rule now used by both
app.services.pathfinding.resolve_goal_cell (live /reachability, POST /command) and
scripts.audit.fit_prob (the Monte-Carlo audit). No DB, no GPU, no network for the
synthetic-grid tests; the hero cross-check below reads real (checked-in-by-path,
skipped if absent) scratch artifacts, same convention as
tests/test_msa_room_polygon.py's `HERO_SCENE_DIR`/`TestHeroScene`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, box

from app.services import goal_resolution
from app.services.pathfinding import cell_to_world
from app.services.scene_ingest import GridMeta

# --- Constants must match the exporter's own (scripts/msa is a geo agent's file -
# never edited here; this asserts the shared module didn't drift from it). ---


def test_constants_match_the_exporters_own():
    from scripts.msa import export_presentation as ep

    assert goal_resolution.CLEARANCE_MARGIN_M == ep.GOAL_CLEARANCE_MARGIN_M
    assert goal_resolution.SEARCH_RADIUS_M == ep.GOAL_SEARCH_RADIUS_M


# --- Synthetic-grid behavioural tests -----------------------------------------------


def _uniform_grid(width: int, height: int, resolution: float = 0.1) -> GridMeta:
    return GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)


def test_resolves_to_the_footprint_edge_not_the_centroid():
    """A large object (a 2m x 2m footprint) in an open room: the resolved goal must
    sit near its EDGE, not collapse toward the centroid the old rule used to anchor
    on - the centroid itself is ~1m from any edge here, well past the required
    clearance bar at this radius. `dist_m` here is a REAL distance transform (the
    target rasterized as the only obstacle in an otherwise open room), not a flat
    array - a flat "clearance is fine everywhere, including right at the target's own
    edge" array would trivially let the resolved cell touch the boundary (edge_dist
    -> 0), which is not realistic: right next to a real object IS right next to an
    obstacle, so real clearance necessarily degrades approaching it."""
    meta = _uniform_grid(60, 60, resolution=0.1)  # 6m x 6m
    target = box(2.0, 2.0, 4.0, 4.0)  # footprint centred at (3, 3), edges at 2 and 4

    xs = meta.origin_x + (np.arange(meta.width) + 0.5) * meta.resolution
    zs = meta.origin_z + (np.arange(meta.height) + 0.5) * meta.resolution
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    inside_target = (gx >= 2.0) & (gx <= 4.0) & (gz >= 2.0) & (gz <= 4.0)
    from scipy import ndimage

    dist_m = ndimage.distance_transform_edt(~inside_target) * meta.resolution

    (ix, iz), _ = goal_resolution.resolve_goal_cell(dist_m, meta, target, robot_radius_m=0.1)
    x, z = cell_to_world(ix, iz, meta)

    centroid_dist = math.hypot(x - 3.0, z - 3.0)
    edge_dist = target.distance(Point(x, z))
    assert edge_dist < centroid_dist
    assert edge_dist == pytest.approx(0.1 + goal_resolution.CLEARANCE_MARGIN_M, abs=0.15)  # near the required bar


def test_raises_goal_unreachable_when_nothing_meets_clearance():
    meta = _uniform_grid(20, 20, resolution=0.1)
    dist_m = np.full((meta.width, meta.height), 0.05)  # nothing anywhere clears even a tiny radius + margin
    target = box(0.9, 0.9, 1.1, 1.1)

    with pytest.raises(goal_resolution.GoalUnreachableError):
        goal_resolution.resolve_goal_cell(dist_m, meta, target, robot_radius_m=0.1)


def test_raises_target_out_of_bounds_when_footprint_is_off_grid():
    meta = _uniform_grid(20, 20, resolution=0.1)  # covers world [0, 2) x [0, 2)
    dist_m = np.full((meta.width, meta.height), 5.0)
    target = box(-100.1, -100.1, -99.9, -99.9)  # nowhere near the grid

    with pytest.raises(goal_resolution.TargetOutOfBoundsError):
        goal_resolution.resolve_goal_cell(dist_m, meta, target, robot_radius_m=0.1)


def test_a_bigger_robot_needs_more_clearance_and_can_end_up_farther_away():
    """A door-sized gap: a small robot's goal can sit right at the target edge, but a
    bigger robot's own clearance+margin bar pushes its resolved cell farther out (or
    fails to resolve at all) - clearance is compared, not just "does A cell exist"."""
    meta = _uniform_grid(60, 20, resolution=0.1)
    dist_m = np.full((meta.width, meta.height), 5.0)
    # A corridor pinch near the target: clearance drops to 0.3m for x in [2.9, 3.1].
    xs = meta.origin_x + (np.arange(meta.width) + 0.5) * meta.resolution
    pinch = (xs >= 2.9) & (xs <= 3.1)
    dist_m[pinch, :] = 0.3
    target = box(3.4, 0.9, 3.6, 1.1)

    (small_ix, _), small_target_dist = goal_resolution.resolve_goal_cell(dist_m, meta, target, robot_radius_m=0.1)
    (big_ix, _), big_target_dist = goal_resolution.resolve_goal_cell(dist_m, meta, target, robot_radius_m=0.3)

    assert small_target_dist[small_ix, 10] <= big_target_dist[big_ix, 10]


def test_target_dist_m_is_reused_when_passed_in():
    """Passing a precomputed target_dist_m (as scripts.audit.fit_prob does once per
    trial, reused across every platform) must give the identical result to letting
    the function compute it fresh - it's a pure function of the target polygon and
    grid, independent of robot_radius_m."""
    meta = _uniform_grid(30, 30, resolution=0.1)
    dist_m = np.full((meta.width, meta.height), 5.0)
    target = box(1.4, 1.4, 1.6, 1.6)

    _cell_a, target_dist_m = goal_resolution.resolve_goal_cell(dist_m, meta, target, robot_radius_m=0.1)
    _cell_b, target_dist_m_2 = goal_resolution.resolve_goal_cell(
        dist_m, meta, target, robot_radius_m=0.2, target_dist_m=target_dist_m
    )
    assert np.array_equal(target_dist_m, target_dist_m_2)


# --- Real hero cross-check: does this shared implementation land near where
# scripts/msa/export_presentation.py's OWN (not-touched-here) plan_route did, given
# the same target object and platform radius? Not an exact match (different
# obstacle rasterization - oriented-rectangle object footprints + wall bands here vs.
# point-cloud hull rasterization there - see the tolerance below), but close enough
# to show this is a faithful port of the SAME rule, evidence a later geo task can use
# to switch the exporter over to this shared implementation. ---

M7_OUT = Path("var/scratch/run-20260906/m7_out")

# Burger's recorded goal on the hero (m7_out/path.json's target_point XZ, radius 0.1m)
# and the platform's own reported goal_clearance_m - see this task's report for how
# these were read off the real file, not guessed.
_HERO_BURGER_RADIUS_M = 0.1
_HERO_BURGER_GOAL_XZ = (0.0627308024359603, -2.1593595892691178)
_HERO_GOAL_TOLERANCE_M = 0.5  # generous: different obstacle rasterization, see module docstring


@pytest.mark.skipif(not M7_OUT.exists(), reason=f"hero bootstrap-out fixture {M7_OUT} not on this machine")
def test_reproduces_the_hero_burger_goal_cell_within_tolerance():
    from scripts.audit.fit_prob import (
        _derive_grid_meta,
        _distance_to_obstacle_m,
        _footprint_polygon_shapely,
        _rasterize_polygon_mask,
        _cell_center_points,
    )
    from app.services.pathfinding import FREE, OBSTACLE
    from scripts.msa.bootstrap import rasterize_object_footprint
    from scripts.msa.objects_io import load_objects_json

    objects_json = M7_OUT / "objects_audited.json"
    if not objects_json.is_file():
        objects_json = M7_OUT / "objects.json"
    bo = load_objects_json(objects_json)

    resolution = 0.05
    meta = _derive_grid_meta(bo.room_polygon, bo.objects, resolution)
    cell_points = _cell_center_points(meta)
    shape = (meta.width, meta.height)
    room_mask = _rasterize_polygon_mask(bo.room_polygon, cell_points, shape)
    cells = np.full(shape, OBSTACLE, dtype=np.uint8)
    cells[room_mask] = FREE
    for obj in bo.objects:
        cells[rasterize_object_footprint(obj, shape, meta.resolution, meta.origin_x, meta.origin_z)] = OBSTACLE
    for wall in bo.walls:
        cells[_rasterize_polygon_mask(wall.vertices, cell_points, shape)] = OBSTACLE

    dist_m = _distance_to_obstacle_m(cells, meta.resolution)
    target = next(o for o in bo.objects if o["id"] == "desk_1")
    target_polygon = _footprint_polygon_shapely(target)

    (ix, iz), _ = goal_resolution.resolve_goal_cell(dist_m, meta, target_polygon, robot_radius_m=_HERO_BURGER_RADIUS_M)
    x, z = cell_to_world(ix, iz, meta)

    d = math.hypot(x - _HERO_BURGER_GOAL_XZ[0], z - _HERO_BURGER_GOAL_XZ[1])
    assert d <= _HERO_GOAL_TOLERANCE_M, (
        f"shared goal_resolution landed {d:.3f}m from the exporter's own recorded hero goal "
        f"{_HERO_BURGER_GOAL_XZ} (got {(x, z)}) - farther than the {_HERO_GOAL_TOLERANCE_M}m tolerance"
    )
