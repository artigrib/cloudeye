"""Pure-unit tests for scripts.audit.fit_prob - no DB, no GPU, no network (matches
this suite's convention, see tests/conftest.py's module docstring). Builds small
synthetic `--bootstrap-out`/`--scene-dir` fixture trees on disk via tmp_path (real
objects.json/scene_meta.json/path.json/occupancy_meta.json, written through the same
scripts.msa.objects_io helpers the real bootstrap pipeline uses) rather than
depending on a real scene - a corridor pinched between two wall blocks, wide or
narrow enough to trivially pass/fail for ANY registered platform's radius, is a much
faster and more targeted probe of this module's own jitter/A*/percentile logic than
a full hero-scene fixture would be.

The fixture's goal object ("desk_test") sits well clear of any wall/other object
(see TARGET_OBJECT below) specifically so goal RESOLUTION itself (see
app.services.goal_resolution, imported/reused by fit_prob.py) always succeeds in
these tests - what's being exercised here is the corridor/jitter/percentile
machinery. The two NEW fit criteria (goal proximity, path-width) get their own
direct, non-Monte-Carlo tests further down (test_fit_requires_*), since fighting
jitter variance to hit those specific edge cases reliably is its own kind of flaky
(see test_narrow_corridor_gives_zero_fit_rate's own comment on that lesson).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from app.services import goal_resolution
from app.services.pathfinding import FREE, OBSTACLE, UNKNOWN
from app.services.scene_ingest import GridMeta
from scripts.audit.fit_prob import (
    GOAL_FALLBACK_PROXIMITY_MARGIN_M,
    GOAL_PRIMARY_PROXIMITY_MARGIN_M,
    TrialWorld,
    _build_planner_base_grid,
    _clip_to_room_polygon,
    _evaluate_trial,
    _generate_trial_worlds,
    _load_resolution,
    run_fit_prob,
)
from scripts.msa.geometry import WallPolygon
from scripts.msa.objects_io import write_objects_json

ROOM_POLYGON = [(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)]
START_XY = (1.0, 2.0)
RESOLUTION = 0.1

TARGET_OBJECT = {
    "id": "desk_test",
    "label": "desk",
    "center_xy": (7.0, 2.0),
    "size_uv": (0.6, 0.6),
    "angle_rad": 0.0,
    "height": 0.7,
    "bbox_min_y": 0.0,
    "color_rgb": (90, 70, 50),
    "n_points": 100,
    "hull_xz": [(6.7, 1.7), (7.3, 1.7), (7.3, 2.3), (6.7, 2.3)],
}

_FILLER_OBJECT = {
    "id": "table_0",
    "label": "table",
    "center_xy": (6.0, 0.6),
    "size_uv": (0.6, 0.6),
    "angle_rad": 0.0,
    "height": 0.7,
    "bbox_min_y": 0.0,
    "color_rgb": (120, 100, 80),
    "n_points": 100,
    "hull_xz": [(5.7, 0.3), (6.3, 0.3), (6.3, 0.9), (5.7, 0.9)],
}


def _corridor_walls(gap_m: float) -> list[WallPolygon]:
    """Two solid wall blocks spanning x in [3.5, 4.5], leaving a `gap_m`-wide gap
    centred on z=2 (the room's midline, and the same z START_XY/TARGET_OBJECT both
    sit on) - the only way through the room from START_XY to the target."""
    z_lo = 2.0 - gap_m / 2.0
    z_hi = 2.0 + gap_m / 2.0

    def _rect(x0: float, x1: float, z0: float, z1: float) -> WallPolygon:
        verts = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        return WallPolygon(vertices=verts, area_m2=(x1 - x0) * (z1 - z0))

    walls = []
    if z_lo > 0.0:
        walls.append(_rect(3.5, 4.5, 0.0, z_lo))
    if z_hi < 4.0:
        walls.append(_rect(3.5, 4.5, z_hi, 4.0))
    return walls


def _write_scene(tmp_path: Path, name: str, gap_m: float) -> tuple[Path, Path]:
    bootstrap_out = tmp_path / name / "bootstrap_out"
    scene_dir = tmp_path / name / "scene_dir"
    bootstrap_out.mkdir(parents=True)
    scene_dir.mkdir(parents=True)

    write_objects_json(
        bootstrap_out / "objects.json",
        objects=[TARGET_OBJECT, _FILLER_OBJECT],
        walls=_corridor_walls(gap_m),
        room_polygon=ROOM_POLYGON,
        floor_y=0.0,
        ceiling_y=2.5,
    )
    (bootstrap_out / "scene_meta.json").write_text(
        json.dumps({"start_xy": list(START_XY), "start_yaw_rad": 0.0})
    )
    (bootstrap_out / "path.json").write_text(json.dumps({"target_id": "desk_test"}))
    (scene_dir / "occupancy_meta.json").write_text(
        json.dumps({"resolution": RESOLUTION, "origin_x": 0.0, "origin_z": 0.0, "width": 1, "height": 1})
    )
    return bootstrap_out, scene_dir


def test_wide_corridor_gives_full_fit_rate(tmp_path):
    bootstrap_out, scene_dir = _write_scene(tmp_path, "wide", gap_m=2.0)

    result = run_fit_prob(bootstrap_out, scene_dir, ["burger"], n_trials=20, seed=0, grid_source="synthetic")

    platform = result["platforms"]["burger"]
    assert platform["trials"] == 20
    assert platform["fits_count"] == 20
    assert platform["first_block_point"] is None
    assert platform["goal_unresolved_trials"] == 0
    assert platform["no_path_trials"] == 0
    assert platform["path_too_narrow_trials"] == 0
    assert platform["fallback_trials"] == 0
    assert platform["corridor_width_p5_m"] is not None


def test_narrow_corridor_gives_zero_fit_rate(tmp_path):
    # Negative "gap" = the two wall blocks OVERLAP by 0.6m at the room's midline, so
    # there is no corridor at all, by a margin (0.6m) far bigger than the per-vertex
    # jitter (sigma 0.08m) could plausibly close - unlike a small POSITIVE gap
    # (e.g. 0.05m), which per-vertex wall jitter routinely blows open into something
    # much wider than nominal (each of the gap's two bounding edges jitters
    # independently by ~0.08m, so a 5cm nominal gap is not a stable "always narrower
    # than the robot" fixture - this asserts genuine, jitter-robust closure instead).
    bootstrap_out, scene_dir = _write_scene(tmp_path, "narrow", gap_m=-0.6)

    result = run_fit_prob(bootstrap_out, scene_dir, ["burger"], n_trials=20, seed=0, grid_source="synthetic")

    platform = result["platforms"]["burger"]
    assert platform["trials"] == 20
    assert platform["fits_count"] == 0
    assert platform["first_block_point"] is not None
    # The block point is at the (fully closed) corridor, far from either object - the
    # goal itself resolves fine and close (the desk sits in open space on the far
    # side), so every failure is REASON_NO_PATH, not a goal-resolution/proximity one.
    assert platform["goal_unresolved_trials"] == 0
    assert platform["no_path_trials"] == 20
    assert platform["corridor_width_p5_m"] is None  # no fits at all - not 0.0 (see the p5 bug fix)
    assert platform["first_block_point"]["label"] in ("room boundary", "table corridor", "desk corridor", "start")


def test_jitter_is_reproducible_under_a_fixed_seed(tmp_path):
    bootstrap_out, scene_dir = _write_scene(tmp_path, "repro", gap_m=0.6)
    meta_resolution = _load_resolution(scene_dir)
    assert meta_resolution == RESOLUTION

    result_a = run_fit_prob(bootstrap_out, scene_dir, ["burger", "husky"], n_trials=15, seed=42, grid_source="synthetic")
    result_b = run_fit_prob(bootstrap_out, scene_dir, ["burger", "husky"], n_trials=15, seed=42, grid_source="synthetic")

    assert result_a["platforms"] == result_b["platforms"]

    result_c = run_fit_prob(bootstrap_out, scene_dir, ["burger", "husky"], n_trials=15, seed=1, grid_source="synthetic")
    assert result_a["platforms"] != result_c["platforms"]


def test_generate_trial_worlds_is_reproducible_under_a_fixed_seed(tmp_path):
    from scripts.msa.objects_io import load_objects_json

    bootstrap_out, scene_dir = _write_scene(tmp_path, "worlds", gap_m=0.6)
    bootstrap_objects = load_objects_json(bootstrap_out / "objects.json")
    resolution = _load_resolution(scene_dir)

    from scripts.audit.fit_prob import _derive_grid_meta

    meta = _derive_grid_meta(bootstrap_objects.room_polygon, bootstrap_objects.objects, resolution)

    worlds_a = _generate_trial_worlds(
        meta, bootstrap_objects.room_polygon, bootstrap_objects.walls, bootstrap_objects.objects, "desk_test", n_trials=5, seed=7
    )
    worlds_b = _generate_trial_worlds(
        meta, bootstrap_objects.room_polygon, bootstrap_objects.walls, bootstrap_objects.objects, "desk_test", n_trials=5, seed=7
    )

    for wa, wb in zip(worlds_a, worlds_b):
        assert np.array_equal(wa.cells, wb.cells)
        assert np.array_equal(wa.dist_m, wb.dist_m)
        assert np.array_equal(wa.target_dist_m, wb.target_dist_m)
        assert wa.target_polygon.equals(wb.target_polygon)


def test_corridor_width_percentiles_are_monotonic(tmp_path):
    # A gap right around burger's fit threshold (0.20m diameter) so jitter pushes
    # some of the 30 trials over the line and some under it - real variance in
    # fits_count to check against, not a degenerate all-pass/all-fail run. (p5/p50/p95
    # themselves can legitimately be IDENTICAL here even with that variance: they are
    # now computed ONLY over trials that fit - see the module's "BUG FIXED" docstring
    # section - and every one of THOSE happens to thread the exact same physical
    # bottleneck at the exact same width in this synthetic fixture. That's the fix
    # working correctly, not a weaker test: before the fix, this same run would have
    # shown a much lower p5 purely from mixing in blocked trials' near-zero pseudo-
    # widths, which is exactly the misleading behaviour that was reported as a bug.)
    bootstrap_out, scene_dir = _write_scene(tmp_path, "mixed", gap_m=0.22)

    result = run_fit_prob(bootstrap_out, scene_dir, ["burger"], n_trials=30, seed=3, grid_source="synthetic")

    platform = result["platforms"]["burger"]
    assert 0 < platform["fits_count"] < 30  # real mixed pass/fail, not a degenerate run
    p5, p50, p95 = (
        platform["corridor_width_p5_m"],
        platform["corridor_width_p50_m"],
        platform["corridor_width_p95_m"],
    )
    assert p5 is not None and p50 is not None and p95 is not None
    assert p5 <= p50 <= p95


def test_run_fit_prob_covers_all_registered_platforms_by_default(tmp_path):
    bootstrap_out, scene_dir = _write_scene(tmp_path, "all_platforms", gap_m=2.0)

    result = run_fit_prob(bootstrap_out, scene_dir, None, n_trials=5, seed=0, grid_source="synthetic")

    from app.robots import list_robots

    assert set(result["platforms"].keys()) == {p.id for p in list_robots()}
    assert result["seed"] == 0
    assert result["trials"] == 5
    assert result["goal_target_id"] == "desk_test"
    assert result["grid"]["grid_source"] == "synthetic_from_footprints"
    assert result["grid"]["grid_extent_covers_room_polygon_pct"] == 100.0
    assert result["goal_rule"]["shared_module"] == "app.services.goal_resolution"


# --- Direct, non-Monte-Carlo tests of the two NEW fit criteria ---------------------
#
# Both hand-build a TrialWorld so the geometry is exact and the specific edge case
# (a goal that resolves but sits too far away; a path A* connects but that's narrower
# than the platform's own diameter) is guaranteed to trigger, rather than relying on
# jitter to occasionally produce it (see test_narrow_corridor_gives_zero_fit_rate's
# own comment on why that's the wrong tool for a guaranteed edge case).


def _open_room_world_with_one_near_target_cell(distance_to_edge_m: float) -> tuple[TrialWorld, GridMeta]:
    """A `size x size` open room (dist_m uniformly 5.0 - clearance is never the
    limiting factor) with exactly ONE cell within goal_resolution.SEARCH_RADIUS_M of
    the target polygon, at `distance_to_edge_m` from its edge - everywhere else is
    10.0m away (out of range). Used by the two fallback-tier tests below to place the
    single resolvable candidate at an exact, controlled distance."""
    size = 40
    meta = GridMeta(resolution=0.1, origin_x=0.0, origin_z=0.0, width=size, height=size)
    dist_m = np.full((size, size), 5.0)
    target_dist_m = np.full((size, size), 10.0)
    target_dist_m[20, 20] = distance_to_edge_m
    target_polygon = Polygon([(1.9, 1.9), (2.1, 1.9), (2.1, 2.1), (1.9, 2.1)])  # placed near (2, 2) world
    world = TrialWorld(cells=np.zeros((size, size), dtype=np.uint8), dist_m=dist_m, target_polygon=target_polygon, target_dist_m=target_dist_m)
    return world, meta


def test_fit_accepts_the_fallback_proximity_margin_and_tags_the_trial():
    """Criterion (a), fallback tier: the only clearance-respecting cell near the
    target is farther than the PRIMARY proximity bar (radius_m + 0.15m) but still
    within the FALLBACK bar (radius_m + 0.30m) - accepted (fits=True, criterion (b)
    is trivially satisfied in this open room), but tagged
    used_fallback_goal_margin=True so the fallback rate is visible."""
    radius_m = 0.1  # primary bar 0.25m, fallback bar 0.40m
    world, meta = _open_room_world_with_one_near_target_cell(distance_to_edge_m=0.35)
    assert radius_m + GOAL_PRIMARY_PROXIMITY_MARGIN_M < 0.35 <= radius_m + GOAL_FALLBACK_PROXIMITY_MARGIN_M

    outcome = _evaluate_trial(world, meta, start_world=(0.5, 0.5), radius_m=radius_m, snap_radius_cells=1)

    assert outcome.fits is True
    assert outcome.used_fallback_goal_margin is True
    assert outcome.reason is None


def test_fit_requires_goal_within_the_fallback_proximity_margin():
    """Criterion (a): a goal that resolves (clearance-wise) but sits farther than
    EVEN the fallback bar (radius_m + GOAL_FALLBACK_PROXIMITY_MARGIN_M) from the
    target's own edge - despite a clearance-respecting cell existing well within
    app.services.goal_resolution.SEARCH_RADIUS_M (2.0m) - does not count as a fit:
    REASON_GOAL_UNRESOLVED (not close enough to count as "reached" at any tier)."""
    radius_m = 0.1  # primary bar 0.25m, fallback bar 0.40m
    world, meta = _open_room_world_with_one_near_target_cell(distance_to_edge_m=1.0)
    assert 1.0 > radius_m + GOAL_FALLBACK_PROXIMITY_MARGIN_M

    outcome = _evaluate_trial(world, meta, start_world=(0.5, 0.5), radius_m=radius_m, snap_radius_cells=1)

    assert outcome.fits is False
    assert outcome.reason == "goal_unresolved"
    assert outcome.used_fallback_goal_margin is False
    # block_cell is the pinch point along a relaxed best-effort route toward the
    # nearest-to-target cell (20, 20) - in this uniformly-open room every cell along
    # that route has the same width, so the specific cell isn't pinned here, just
    # that a real point was produced.
    assert outcome.block_cell is not None
    assert outcome.block_point is not None


def test_fit_requires_full_diameter_along_the_actual_path():
    """Criterion (b): A* connecting start to a valid, close-enough goal is not
    sufficient - the actual path's own minimum corridor width (world.dist_m, this
    module's own continuous distance transform, independent of the discrete cells A*
    ran on) must also be >= the platform's full diameter. Built directly: `cells` has
    no obstacles at all (so A* trivially finds the straight-line start->goal path),
    but `dist_m` is deliberately set low at one cell that path must cross - dist_m is
    an independent field here, not re-derived from cells, so this isolates criterion
    (b) from criterion (a)/A* connectivity entirely."""
    width, height = 20, 5
    resolution = 0.1
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)

    cells = np.zeros((width, height), dtype=np.uint8)  # FREE everywhere - inflate() finds nothing to dilate
    dist_m = np.full((width, height), 5.0)  # generous clearance everywhere...
    pinch_ix = 10
    dist_m[pinch_ix, 2] = 0.05  # ...except this one cell, width 0.1m - under burger's 0.2m diameter

    start_world = (0.15, 0.25)  # cell (1, 2)
    target_polygon = Polygon([(1.75, 0.15), (1.85, 0.15), (1.85, 0.35), (1.75, 0.35)])  # near cell (18, 2)
    target_dist_m = goal_resolution.target_distance_grid(meta, target_polygon)

    world = TrialWorld(cells=cells, dist_m=dist_m, target_polygon=target_polygon, target_dist_m=target_dist_m)

    radius_m = 0.1  # diameter 0.2m > the 0.1m available at the pinch cell
    snap_radius_cells = 2
    outcome = _evaluate_trial(world, meta, start_world, radius_m, snap_radius_cells)

    assert outcome.fits is False
    assert outcome.reason == "path_too_narrow"
    assert outcome.block_cell == (pinch_ix, 2)
    assert outcome.min_corridor_width_m == pytest.approx(0.1, abs=1e-9)


# --- Planner-grid mode (--grid-source planner, the default) ------------------------


def test_clip_to_room_polygon_blocks_outside_leaves_inside_alone():
    """The fix for the "raw-grid-obstacles-only misses real walls" finding: a cell
    outside room_polygon becomes OBSTACLE regardless of what the raw grid said
    there (FREE and UNKNOWN both get overridden) - a cell INSIDE keeps its own
    original classification untouched, UNKNOWN included (still traversable-at-a-
    penalty in app.services.pathfinding.build_cost_grid)."""
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_z=0.0, width=10, height=10)
    cells = np.full((10, 10), FREE, dtype=np.uint8)
    cells[8, 8] = UNKNOWN  # outside the room polygon below - must become OBSTACLE
    cells[2, 2] = UNKNOWN  # inside the room polygon below - must stay UNKNOWN
    room_polygon = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]  # world [0,5) x [0,5)

    clipped = _clip_to_room_polygon(cells, meta, room_polygon)

    assert clipped[2, 2] == UNKNOWN  # inside, unchanged
    assert clipped[3, 3] == FREE  # inside, unchanged
    assert clipped[8, 8] == OBSTACLE  # outside, was UNKNOWN, now forced OBSTACLE
    assert clipped[7, 7] == OBSTACLE  # outside, was FREE, now forced OBSTACLE


def test_build_planner_base_grid_samples_the_correct_rotated_cell():
    """Direct, isolated test of the inverse-warp rotation+resample math: a single
    OBSTACLE cell in the raw grid, rotated 90 degrees CCW about the origin, must
    land at the correspondingly-rotated world point in the output grid - and a
    point that rotates to somewhere the raw grid never covered must come back
    UNKNOWN with covered=False, not OBSTACLE."""
    raw_meta = GridMeta(resolution=1.0, origin_x=0.0, origin_z=0.0, width=10, height=10)
    raw_cells = np.full((10, 10), FREE, dtype=np.uint8)
    raw_cells[7, 2] = OBSTACLE  # world centre (7.5, 2.5)

    angle_rad = math.pi / 2  # CCW 90 degrees about (0, 0): (x, z) -> (-z, x)
    center = (0.0, 0.0)
    out_meta = GridMeta(resolution=1.0, origin_x=-10.0, origin_z=-10.0, width=20, height=20)

    cells, covered = _build_planner_base_grid(raw_cells, raw_meta, out_meta, angle_rad, center)

    # (7.5, 2.5) rotated by +90 degrees about the origin -> (-2.5, 7.5).
    out_ix = int((-2.5 - out_meta.origin_x) / out_meta.resolution)
    out_iz = int((7.5 - out_meta.origin_z) / out_meta.resolution)
    assert cells[out_ix, out_iz] == OBSTACLE
    assert covered[out_ix, out_iz]

    # A point far outside anywhere the raw grid could have rotated into.
    far_ix = int((9.5 - out_meta.origin_x) / out_meta.resolution)
    far_iz = int((-9.5 - out_meta.origin_z) / out_meta.resolution)
    assert cells[far_ix, far_iz] == UNKNOWN
    assert not covered[far_ix, far_iz]


def _write_planner_scene(tmp_path: Path, name: str, gap_m: float) -> tuple[Path, Path]:
    """Same corridor shape as `_write_scene`, but as a REAL raw occupancy.npy (no
    yaw correction - identity transform, angle=0 - so the raw and exported frames
    coincide and the fixture stays simple) instead of a room_polygon/walls vector
    description, so this exercises the actual `--grid-source planner` (default)
    code path end to end."""
    bootstrap_out = tmp_path / name / "bootstrap_out"
    scene_dir = tmp_path / name / "scene_dir"
    bootstrap_out.mkdir(parents=True)
    scene_dir.mkdir(parents=True)

    # The raw grid deliberately covers MORE than ROOM_POLYGON's own 8m x 4m extent -
    # margin_cells of padding on every side, well past _derive_grid_meta's own
    # GRID_MARGIN_M (0.5m = 5 cells at this resolution). Cells outside the raw grid's
    # bounds default to UNKNOWN (traversable) in planner-grid mode - see
    # _build_planner_base_grid - so if the wall pinch below only spanned the bare
    # room height, a trial could route AROUND it through that traversable margin
    # instead of actually threading the gap. A real occupancy scan's own wall
    # detection typically extends across whatever the sensor covered, well past the
    # room polygon's own tight boundary, so this is what a real raw grid looks like,
    # not a fixture-only workaround (same lesson as the earlier synthetic-mode
    # narrow-corridor margin-leak bug this module already fixed once).
    resolution = 0.1
    margin_cells = 15
    room_w_cells, room_h_cells = 80, 40  # 8m x 4m
    width, height = room_w_cells + 2 * margin_cells, room_h_cells + 2 * margin_cells
    origin_x = origin_z = -margin_cells * resolution
    cells = np.full((width, height), FREE, dtype=np.uint8)
    z_lo = 2.0 - gap_m / 2.0
    z_hi = 2.0 + gap_m / 2.0
    x0_ix = margin_cells + int(3.5 / resolution)
    x1_ix = margin_cells + int(4.5 / resolution)
    for iz in range(height):
        z = origin_z + (iz + 0.5) * resolution
        if z_lo > 0.0 and z < z_lo:
            cells[x0_ix:x1_ix, iz] = OBSTACLE
        if z_hi < 4.0 and z >= z_hi:
            cells[x0_ix:x1_ix, iz] = OBSTACLE
    np.save(scene_dir / "occupancy.npy", cells)
    (scene_dir / "occupancy_meta.json").write_text(
        json.dumps({"resolution": resolution, "origin_x": origin_x, "origin_z": origin_z, "width": width, "height": height})
    )

    write_objects_json(
        bootstrap_out / "objects.json",
        objects=[TARGET_OBJECT, _FILLER_OBJECT],
        walls=[],  # walls are now the raw grid's own OBSTACLE cells - see the module docstring
        room_polygon=ROOM_POLYGON,
        floor_y=0.0,
        ceiling_y=2.5,
    )
    (bootstrap_out / "scene_meta.json").write_text(json.dumps({"start_xy": list(START_XY), "start_yaw_rad": 0.0}))
    (bootstrap_out / "path.json").write_text(json.dumps({"target_id": "desk_test"}))
    return bootstrap_out, scene_dir


def test_planner_grid_wide_corridor_gives_full_fit_rate_and_real_coverage(tmp_path):
    bootstrap_out, scene_dir = _write_planner_scene(tmp_path, "planner_wide", gap_m=2.0)

    result = run_fit_prob(bootstrap_out, scene_dir, ["burger"], n_trials=15, seed=0)  # grid_source defaults to "planner"

    assert result["grid"]["grid_source"] == "planner_grid_rotated_occupancy_clipped_to_room_plus_hulls"
    # The raw grid was built to exactly cover ROOM_POLYGON, so coverage should be at
    # (or extremely close to) 100% - a REAL measurement here, not 100% by
    # construction the way synthetic mode's always is.
    assert result["grid"]["grid_extent_covers_room_polygon_pct"] > 99.0
    assert result["grid"]["yaw_correction_rad"] == 0.0

    platform = result["platforms"]["burger"]
    assert platform["trials"] == 15
    assert platform["fits_count"] == 15


def test_planner_grid_narrow_corridor_gives_zero_fit_rate(tmp_path):
    bootstrap_out, scene_dir = _write_planner_scene(tmp_path, "planner_narrow", gap_m=-0.6)

    result = run_fit_prob(bootstrap_out, scene_dir, ["burger"], n_trials=15, seed=0)

    platform = result["platforms"]["burger"]
    assert platform["fits_count"] == 0
    assert platform["no_path_trials"] == 15


def test_planner_grid_coverage_reflects_a_real_gap_in_the_raw_scan(tmp_path):
    """Unlike synthetic mode (always 100% by construction), planner-grid mode's
    coverage figure must drop when the raw occupancy grid genuinely doesn't reach
    part of room_polygon - built here by shrinking the raw grid's own height so it
    only covers half of ROOM_POLYGON's z-extent."""
    bootstrap_out, scene_dir = _write_planner_scene(tmp_path, "planner_gap", gap_m=2.0)
    # Truncate the raw occupancy grid to only the first half (z < 2.0m) after the fact.
    resolution = 0.1
    full = np.load(scene_dir / "occupancy.npy")
    half_height = full.shape[1] // 2
    np.save(scene_dir / "occupancy.npy", full[:, :half_height])
    meta_json = json.loads((scene_dir / "occupancy_meta.json").read_text())
    meta_json["height"] = half_height
    (scene_dir / "occupancy_meta.json").write_text(json.dumps(meta_json))

    result = run_fit_prob(bootstrap_out, scene_dir, ["burger"], n_trials=5, seed=0)

    coverage = result["grid"]["grid_extent_covers_room_polygon_pct"]
    assert 30.0 < coverage < 70.0, f"expected roughly half coverage, got {coverage}"
