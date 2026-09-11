"""`compute_reachability_cached`'s cache key, and reachability's monotonicity in
robot radius - both matter a lot more now that requests can come from any of eight
registered platforms (app/robots.py) instead of one hardcoded radius: a cache key that
doesn't fully key off radius would silently serve one platform's answer for another
(pick Husky, see Burger's reachable-object list) - see pathfinding.py's
compute_reachability_cached docstring and app/robots.py's resolve_radius_m.
"""

import math

import numpy as np
import pytest

from app.services import pathfinding, scene_ingest
from app.services.pathfinding import (
    FREE,
    OBSTACLE,
    NoPathError,
    ObjectFootprint,
    OccupancyGrid,
    build_cost_grid,
    inflate,
    snap_if_blocked,
    world_to_cell,
)
from app.services.scene_ingest import GridMeta


def make_grid(rows: list[str], *, resolution=0.1) -> tuple[np.ndarray, GridMeta]:
    """Same convention as test_pathfinding_astar.py / test_pathfinding_connectivity.py's
    helper: rows are Z (top=iz=0), columns are X, one character per cell."""
    height = len(rows)
    width = len(rows[0])
    char_to_val = {".": FREE, "#": OBSTACLE}
    cells = np.zeros((width, height), dtype=np.uint8)
    for iz, row in enumerate(rows):
        for ix, ch in enumerate(row):
            cells[ix, iz] = char_to_val[ch]
    meta = GridMeta(resolution=resolution, origin_x=0.0, origin_z=0.0, width=width, height=height)
    return cells, meta


def footprint_at(meta: GridMeta, ix: int, iz: int) -> ObjectFootprint:
    x, z = pathfinding.cell_to_world(ix, iz, meta)
    return ObjectFootprint(x=x, z=z, bbox_min_x=x, bbox_min_z=z, bbox_max_x=x, bbox_max_z=z)


# A small pocket (cols 0-3) separated from a big room (cols 5-15) by a wall at col=4
# with a single one-cell gap (row=4) - same shape as
# test_pathfinding_connectivity.test_resolve_robot_start_relocates_out_of_a_sealed_pocket,
# reused here because it's already verified to behave well (room on both sides stays
# individually non-empty) at both radius=0.0 (gap open) and radius=0.15 (gap sealed by
# a 1.5-cell inflation) - unlike a bare 1-cell corridor, which collapses the entire grid
# to nothing once inflated and makes start-cell snapping itself fail.
def _pocket_and_room_grid() -> tuple[np.ndarray, GridMeta]:
    width, height, wall_col, gap_row = 16, 8, 4, 4
    rows = [
        "".join("#" if ix == wall_col and iz != gap_row else "." for ix in range(width))
        for iz in range(height)
    ]
    return make_grid(rows, resolution=0.1)


def _pocket_grid_and_far_object(tmp_path, scene_id):
    cells, meta = _pocket_and_room_grid()
    npy_path = tmp_path / f"{scene_id}.npy"
    np.save(npy_path, cells)
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(1, 1, meta)  # inside the small pocket
    far_object = [("far_side", footprint_at(meta, 10, 1))]  # deep in the big room
    return npy_path, grid, start_world, far_object


def test_cache_key_already_includes_radius(tmp_path):
    """Regression test for the exact bug the task called out: two different radii
    against the identical (scene_id, grid, mtime) must NOT collide in the cache. At
    radius=0.0 the gap is open (object reachable); at radius=0.15 (1.5 cells)
    inflation seals it (object disconnected) - if the cache key omitted radius, the
    second call would wrongly return the first radius's cached (reachable) result."""
    scene_id = "gap-scene"
    npy_path, grid, start_world, objects = _pocket_grid_and_far_object(tmp_path, scene_id)

    low = pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects, robot_radius_m=0.0
    )
    high = pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects, robot_radius_m=0.15
    )

    assert "far_side" in low.reachable_ids
    assert "far_side" not in high.reachable_ids
    assert high.unreachable_reasons["far_side"] == "disconnected"

    # And re-visiting the low radius afterwards must still return the low-radius
    # answer, not whatever was computed last (proves the key round-trips both ways,
    # not just "first write wins").
    low_again = pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects, robot_radius_m=0.0
    )
    assert "far_side" in low_again.reachable_ids


def test_cache_key_literally_contains_the_rounded_radius_and_height(tmp_path):
    """Direct check on the key shape itself (not just the observable behavior above) -
    documents exactly why the behavioral test passes: `round(robot_radius_m, 4)` is
    part of the tuple key, not just the scene id and grid mtime.

    HEIGHT joined it when the grid stopped being one array per scene: `grid` is now
    derived from the layer's obstacle-height map at the requesting robot's own roof
    (`nav_layer.cells_for_height`), so two platforms that share a radius but not a height
    are asking about two different grids. `None` is the "no height given" slot, which is
    what every pre-layer caller lands on."""
    scene_id = "gap-scene-key-shape"
    npy_path, grid, start_world, objects = _pocket_grid_and_far_object(tmp_path, scene_id)
    mtime_ns = npy_path.stat().st_mtime_ns

    pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects, robot_radius_m=0.123456
    )
    assert (scene_id, mtime_ns, 0.1235, None) in pathfinding._REACHABILITY_CACHE

    pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects,
        robot_radius_m=0.123456, robot_height_m=0.404321,
    )
    assert (scene_id, mtime_ns, 0.1235, 0.4043) in pathfinding._REACHABILITY_CACHE


def test_two_platforms_of_equal_radius_and_different_height_do_not_share_an_answer(tmp_path):
    """The collision the height slot exists to stop.

    A tall robot and a short one of the SAME radius see different grids - the short one
    drives under a surface that walls the tall one off. Before height entered the key the
    second caller was handed the first one's ReachabilityResult, and the object list
    silently showed one platform's answer under the other's name."""
    scene_id = "same-radius-different-height"
    npy_path, grid, start_world, objects = _pocket_grid_and_far_object(tmp_path, scene_id)

    short = pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects,
        robot_radius_m=0.1, robot_height_m=0.192,
    )
    # Same radius, same file, taller robot - and a grid where the far object is walled off.
    walled = pathfinding.OccupancyGrid(
        cells=grid.cells.copy(), meta=grid.meta
    )
    walled.cells[:, :] = pathfinding.OBSTACLE
    tall = pathfinding.compute_reachability_cached(
        scene_id, npy_path, walled, start_world, objects,
        robot_radius_m=0.1, robot_height_m=0.400,
    )

    assert "far_side" in short.reachable_ids
    assert tall.reachable_ids != short.reachable_ids


def test_revisiting_a_radius_is_served_from_cache_not_recomputed(tmp_path, monkeypatch):
    scene_id = "gap-scene-cache-hit"
    npy_path, grid, start_world, objects = _pocket_grid_and_far_object(tmp_path, scene_id)

    calls = []
    real_compute = pathfinding.compute_reachability

    def counting_compute(*args, **kwargs):
        calls.append(1)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(pathfinding, "compute_reachability", counting_compute)

    pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects, robot_radius_m=0.05
    )
    pathfinding.compute_reachability_cached(
        scene_id, npy_path, grid, start_world, objects, robot_radius_m=0.05
    )
    assert len(calls) == 1  # second call was a cache hit, not a recompute


# --- Monotonicity: reachable-object count must never increase with radius -----------
#
# Two doors of different widths off a shared start room, with both destination objects
# placed well inside their own room (not near any wall) so their resolved goal cell
# stays the same physical cell across every radius tested - isolating the property
# under test (does more inflation only ever remove paths) from a separate, real
# phenomenon: resolve_goal_cell/nearest_free_cell can snap an object's goal to a
# *different* nearby cell as radius grows and its usual cell's neighborhood changes,
# which can occasionally make one specific object's reachability non-monotonic on
# messy real-world data even though the underlying free-space area only ever shrinks
# (see test_start_component_size_is_non_increasing_on_real_fixture below, and this
# task's report for a worked real-fixture example).


ROOM_W = 90  # cells (4.5m at resolution=0.05) - see test_reachable_count_is_non_increasing_as_radius_grows


def _two_doors_grid() -> tuple[np.ndarray, GridMeta]:
    """Three rooms in a row, each `ROOM_W` cells (4.5m) wide/deep - NOT the tiny
    (17x11 cell, 0.85m x 0.55m) grid this used before goal resolution moved to the
    edge+clearance rule (`app.services.goal_resolution`, `SEARCH_RADIUS_M=2.0m`):
    on a grid smaller than that search radius, "nearest cell meeting clearance
    within 2.0m of the target" can end up nowhere near the target at all - the ONLY
    surviving high-clearance cell in the whole tiny grid, wherever that happens to
    be (observed: it could land right on the START's own cell, making an object look
    "reachable" again at a LARGER radius than one where it wasn't - the exact
    monotonicity violation this test now guards against). Each room here is deep
    enough (4.5m > 2.0m search radius) that an object placed at its centre resolves
    its goal within its OWN room at every radius tested, so the only thing that
    changes with radius is genuine door-inflation connectivity - the property this
    test actually means to check."""
    resolution = 0.05
    door1_col = ROOM_W
    door2_col = ROOM_W + 1 + ROOM_W
    width = ROOM_W * 3 + 2
    height = 80
    door1_gap = set(range(37, 43))  # 6 rows tall (0.3m) - the wider door
    door2_gap = set(range(39, 41))  # 2 rows tall (0.1m) - the narrower door

    def row(iz: int) -> str:
        chars = []
        for ix in range(width):
            if ix == door1_col:
                chars.append("." if iz in door1_gap else "#")
            elif ix == door2_col:
                chars.append("." if iz in door2_gap else "#")
            else:
                chars.append(".")
        return "".join(chars)

    return make_grid([row(iz) for iz in range(height)], resolution=resolution)


def test_reachable_count_is_non_increasing_as_radius_grows():
    """Room A (start) connects to room B through a wide door and to room C through a
    narrower one. As radius grows: both reachable -> only room B (wide door survives
    longer) -> neither - a clean, deliberately monotonic staircase."""
    cells, meta = _two_doors_grid()
    grid = OccupancyGrid(cells=cells, meta=meta)
    start_world = pathfinding.cell_to_world(45, 40, meta)  # room A centre, away from either door
    objects = [
        ("room_b", footprint_at(meta, ROOM_W + 1 + 45, 40)),  # room B centre
        ("room_c", footprint_at(meta, ROOM_W + 1 + ROOM_W + 1 + 45, 40)),  # room C centre
    ]

    radii = [0.0, 0.05, 0.10, 0.15, 0.20]
    counts = [
        len(pathfinding.compute_reachability(grid, start_world, objects, robot_radius_m=r).reachable_ids)
        for r in radii
    ]

    assert counts == sorted(counts, reverse=True), (
        f"reachable-object count must be non-increasing in radius, got {list(zip(radii, counts))}"
    )
    assert counts[0] == 2  # both doors open at radius 0 - the property is exercised, not vacuous
    assert counts[-1] == 0  # both doors sealed by the largest radius tested


def test_start_component_size_is_non_increasing_on_real_fixture(scene_horizontal_dir):
    """The aggregate, snap-independent monotonic property, checked against the real
    scene_horizontal fixture (same fixture test_scene_ingest.py / test_usd_export.py
    use): the robot start's connected-component size can only shrink or hold steady as
    obstacles are inflated by a bigger radius - inflation only ever adds obstacle
    cells, never removes any, so a cell (or the start cell itself, snapped to the same
    nearby free cell every time since it's far from any wall in this fixture) that was
    in a component of size N at radius r can only be in a component of size <= N at
    any radius > r.

    Deliberately NOT asserting reachable_ids/reachable count is non-increasing here:
    on this real, cluttered fixture, individual objects' resolved goal cells do
    sometimes hop to a different nearby free cell as radius grows (see this module's
    docstring above and this task's report) - a real, separate effect from the
    property under test."""
    grid_meta = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    cameras = scene_ingest.read_camera_track(scene_horizontal_dir)
    start_world = scene_ingest.choose_robot_start(cameras, grid_meta)
    cells = np.load(scene_horizontal_dir / "occupancy.npy")
    grid = OccupancyGrid(cells=cells, meta=grid_meta)

    radii = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sizes = [
        pathfinding.compute_reachability(grid, start_world, [], robot_radius_m=r).start_component_size
        for r in radii
    ]

    assert sizes == sorted(sizes, reverse=True), (
        f"start_component_size must be non-increasing in radius, got {list(zip(radii, sizes))}"
    )
    assert sizes[0] > sizes[-1]  # the property is exercised, not vacuously true


# --- Start cell doesn't fit anywhere near itself at this radius: reachability must ----
# --- still return a normal answer, not raise (unlike resolve_robot_start) -------------


def test_compute_reachability_reports_nothing_reachable_instead_of_raising(scene_horizontal_dir):
    """Regression test for the /reachability 500: at a large enough robot_radius_m,
    obstacle inflation can eat every free cell within 0.5m of the start point, and
    `snap_if_blocked` (via `nearest_free_cell`) raises `NoPathError` in that case - see
    pathfinding.py's `nearest_free_cell`/`snap_if_blocked`. That's the right behavior for
    `resolve_robot_start` (a one-off computation persisted at scene creation - see its
    docstring), but `compute_reachability` backs a live UI radius slider/platform picker
    (GET /scenes/{id}/reachability), which had no handler for `NoPathError` anywhere on
    that path (app/routers/scenes.py only catches GridUnavailableError) - so any request
    landing in this zone became an unhandled 500. Confirmed live: selecting the Husky
    platform (radius_m=0.5528, app/robots.py) hit exactly this on a real scene.

    radius_m=1.0 is well past the fixture's break point (empirically ~0.90-0.92m on
    this occupancy grid, resolution=0.05m - unrelated to the 0.5528m that triggered it
    live, since fixture grids differ from the scene the frontend agent hit this on) and
    reproduces the crash reliably: before the fix in compute_reachability, this exact
    call raised `NoPathError("no free cell found within 10 cells of ...")` instead of
    returning - verified by temporarily reverting the fix and observing the raise."""
    grid_meta = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    cameras = scene_ingest.read_camera_track(scene_horizontal_dir)
    start_world = scene_ingest.choose_robot_start(cameras, grid_meta)
    cells = np.load(scene_horizontal_dir / "occupancy.npy")
    grid = OccupancyGrid(cells=cells, meta=grid_meta)
    objects = [("some_object", footprint_at(grid_meta, 20, 125))]

    result = pathfinding.compute_reachability(grid, start_world, objects, robot_radius_m=1.0)

    assert result.component_count == 0
    assert result.start_component_size == 0
    assert result.reachable_ids == frozenset()
    assert result.unreachable_reasons == {"some_object": "robot_does_not_fit"}
    assert not result.reachable_cells.any()

    # Sanity check that this radius really is past the break point on the raw
    # snap_if_blocked call the fix wraps, not just "no objects happened to resolve" -
    # i.e. this test is exercising the actual bug, not vacuously passing.
    cost_grid = build_cost_grid(inflate(grid.cells, grid.meta, 1.0))
    snap_radius_cells = max(1, int(math.ceil(0.5 / grid.meta.resolution)))
    with pytest.raises(NoPathError):
        snap_if_blocked(cost_grid, world_to_cell(*start_world, grid.meta), snap_radius_cells)


# --- An individual object's goal cell doesn't fit near itself at this radius: that ----
# --- one object must be excluded, not raise for the whole request ---------------------


def test_compute_reachability_excludes_object_with_no_free_cell_near_goal_instead_of_raising(
    scene_horizontal_dir,
):
    """Second, distinct regression test for the /reachability 500 - same bug class as
    `test_compute_reachability_reports_nothing_reachable_instead_of_raising` above, but
    a different code path. That test covers the robot's own START cell having no free
    cell nearby (handled directly in `compute_reachability`, via the try/except around
    the initial `snap_if_blocked` call). This one covers a single OBJECT's goal cell
    having no free cell nearby - resolved via `try_resolve_goal_cell` deeper in the same
    function's per-object loop.

    `try_resolve_goal_cell` only caught `ValueError` (centroid off the grid entirely)
    before this fix - it did NOT catch `NoPathError`, which `resolve_goal_cell` can also
    raise when no cell meets the goal's requirements. That `NoPathError` propagated
    unhandled out of `compute_reachability` - a real 500, live-reproduced on a real
    scene at `robot_radius_m=0.5528` (Husky, app/robots.py) hitting exactly this for
    one cluttered object, with six other, smaller-radius platforms on the very same
    scene returning 200 for every object.

    cell (32, 0) on this fixture and robot_radius_m=0.8 reproduce it reliably under
    the current goal rule (`app.services.goal_resolution`: nearest cell to the
    footprint's own edge with clearance >= radius + 0.10m, searched within 2.0m) -
    confirmed empirically (see this task's report) that no cell within 2.0m of this
    footprint meets a 0.90m clearance bar at this radius, while the robot's start
    point (a much more open area of this scene) is still perfectly fine (see
    test_start_component_size_is_non_increasing_on_real_fixture), so this exercises
    the object-goal-cell path specifically, not the start-cell path above. (0.3, the
    radius this test used under the OLD centroid+1.0m-ring goal rule, no longer
    reproduces the failure: the new rule's wider 2.0m search radius and edge anchor
    find a valid, if disconnected, cell at that radius instead - see
    test_compute_reachability_marks_a_walled_off_object_disconnected's sibling
    coverage of THAT outcome instead.)"""
    grid_meta = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    cameras = scene_ingest.read_camera_track(scene_horizontal_dir)
    start_world = scene_ingest.choose_robot_start(cameras, grid_meta)
    cells = np.load(scene_horizontal_dir / "occupancy.npy")
    grid = OccupancyGrid(cells=cells, meta=grid_meta)
    objects = [("far_object", footprint_at(grid_meta, 32, 0))]

    result = pathfinding.compute_reachability(grid, start_world, objects, robot_radius_m=0.8)

    assert result.reachable_ids == frozenset()
    # Label corrected 2026-09-09: this case is "the object is ON the grid but no cell
    # within 2m of its edge is wide enough", which is what this test's own name says -
    # it was reported as "outside_grid", the same label a genuinely off-grid footprint
    # gets. On the hero scene that mislabelled four objects that all sat inside the grid
    # with a free cell 0.01-0.08m away. Behaviour is unchanged; only the reason is now
    # distinguishable.
    assert result.unreachable_reasons == {"far_object": "no_free_cell_within_2m"}
    # The start itself is fine at this radius - unlike the sibling test above, this is
    # NOT the "robot doesn't fit anywhere" case; only the one object's goal resolution
    # fails, and the grid's overall connectivity is otherwise reported normally.
    assert result.start_component_size > 0

    # Sanity check that this radius/cell really is past the goal-cell break point on
    # the raw snap_if_blocked call the fix wraps, not just "the object happened not to
    # resolve" - i.e. this test is exercising the actual bug, not vacuously passing.
    cost_grid = build_cost_grid(inflate(grid.cells, grid.meta, 0.3))
    goal_snap_radius_cells = max(1, int(math.ceil(1.0 / grid.meta.resolution)))
    with pytest.raises(NoPathError):
        snap_if_blocked(cost_grid, (32, 0), goal_snap_radius_cells)
