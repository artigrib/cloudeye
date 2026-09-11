"""The shared "nearest free cell to an object's footprint EDGE, with real clearance"
goal-resolution rule.

This is the SAME rule `scripts/msa/export_presentation.py`'s `plan_route` defined and
uses for the presentation/demo route (`GOAL_CLEARANCE_MARGIN_M`/`GOAL_SEARCH_RADIUS_M`
values reused verbatim here - see `CLEARANCE_MARGIN_M`/`SEARCH_RADIUS_M` below), now
extracted into one shared implementation used by BOTH:

  - `app.services.pathfinding.resolve_goal_cell` - the live `/reachability` and
    `POST /command` planner (previously: snap from the target's CENTROID, 1.0m ring
    search, no clearance margin beyond bare radius-inflation - see that function's
    own docstring for the exact diff this replaces);
  - `scripts.audit.fit_prob` - the Monte-Carlo fit-probability audit.

`scripts/msa/export_presentation.py` itself is NOT changed here (a geo agent owns
that file) - `tests/test_goal_resolution.py` instead asserts this module's constants
match its `GOAL_CLEARANCE_MARGIN_M`/`GOAL_SEARCH_RADIUS_M` and that this module
reproduces (within tolerance - the two implementations rasterize obstacles
differently, see that test) its recorded hero goal, so a later task can switch the
exporter over to this shared implementation with evidence it behaves the same.

Deliberately grid-convention-agnostic: takes a precomputed `dist_to_obstacle_m` array
(metres, "distance to the nearest obstacle cell" - the same distance-transform
quantity both callers already compute for their own purposes) rather than a raw
tri-state occupancy grid, so this module has no dependency on either caller's own
FREE/OBSTACLE cell-value convention and - critically - creates no import cycle with
`app.services.pathfinding` (which imports THIS module, not the other way around).
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Polygon, box

from app.services.scene_ingest import GridMeta

# Copied verbatim from scripts.msa.export_presentation.py's GOAL_CLEARANCE_MARGIN_M /
# GOAL_SEARCH_RADIUS_M - see that module's own T15h comment for the original
# rationale (a validator's 0.2m sphere wedged between the desk and the floor and fell
# through with a bare radius clearance, hence the +0.10m margin).
CLEARANCE_MARGIN_M = 0.10
SEARCH_RADIUS_M = 2.0

Cell = tuple[int, int]


class TargetOutOfBoundsError(Exception):
    """The target polygon's bounding box does not overlap the grid at all - bad data
    for this one object (e.g. a coordinate-frame mismatch), not a "nothing nearby is
    wide enough" result. Callers map this to whatever their own "outside_grid"-style
    classification is (see `app.services.pathfinding.try_resolve_goal_cell`)."""


class StartDisconnectedError(Exception):
    """The START cell itself is not traversable at this radius, so it belongs to no
    component at all - nothing can be reached from it. Distinct from
    `no_connected_cell_within_2m` (raised as `GoalUnreachableError`), which means the start
    is fine but this particular target sits in another component.

    This case used to be invisible: the old rule picked the cell nearest the target edge
    without regard to connectivity, so a SMALL robot - whose lower clearance bar admits more
    cells - could be handed a nearer goal sitting in an isolated pocket and be reported
    unreachable, while a LARGER robot, for which those pocket cells fail the bar, got a
    farther but connected goal and succeeded. That is how TurtleBot (0.10 m) scored 11/17
    against Go2's (0.2496 m) 13/17 on the hero scene."""


class GoalUnreachableError(Exception):
    """The target IS on the grid, but no cell within `search_radius_m` of its
    footprint meets `robot_radius_m + clearance_margin_m` clearance - the target is
    real, just not approachable by a robot this wide (at least not within the search
    radius). Callers map this to whatever their own "goal doesn't fit"-style
    classification is."""


def _cell_center_points(meta: GridMeta) -> np.ndarray:
    """(width*height, 2) array of every cell's world-space centre, row-major in the
    same `[ix, iz]` order `dist_to_obstacle_m.reshape((meta.width, meta.height))`
    expects - identical convention to `app.services.pathfinding.cell_to_world`."""
    xs = meta.origin_x + (np.arange(meta.width) + 0.5) * meta.resolution
    zs = meta.origin_z + (np.arange(meta.height) + 0.5) * meta.resolution
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_z.ravel()])


def target_distance_grid(meta: GridMeta, target_polygon: Polygon) -> np.ndarray:
    """Distance (metres) from every cell centre to `target_polygon`'s own edge -
    shape `(meta.width, meta.height)`, `[ix, iz]`. Exposed separately from
    `resolve_goal_cell` so a caller resolving the SAME target across many radii in one
    run (`scripts.audit.fit_prob`: one call per platform per trial) can compute this
    once and pass it back in via `resolve_goal_cell`'s `target_dist_m` - it does not
    depend on `robot_radius_m`."""
    points = _cell_center_points(meta)
    dist = shapely.distance(target_polygon, shapely.points(points[:, 0], points[:, 1]))
    return dist.reshape((meta.width, meta.height))


def resolve_goal_cell(
    dist_to_obstacle_m: np.ndarray,
    meta: GridMeta,
    target_polygon: Polygon,
    robot_radius_m: float,
    *,
    clearance_margin_m: float = CLEARANCE_MARGIN_M,
    search_radius_m: float = SEARCH_RADIUS_M,
    target_dist_m: np.ndarray | None = None,
    start_cell: Cell | None = None,
) -> tuple[Cell, np.ndarray]:
    """The nearest traversable cell to `target_polygon`'s own EDGE (not its centroid)
    whose clearance - `dist_to_obstacle_m`, i.e. distance to the nearest obstacle
    cell, covering every other object, the target object itself, walls, and the room
    boundary alike (whatever the caller's own distance-transform was built from) - is
    at least `robot_radius_m + clearance_margin_m`, among cells within
    `search_radius_m` of the target polygon. Ties (equal distance to the target)
    broken by cell scan order (`np.argmin`'s own tie-break) - rare enough in practice
    (continuous float distances) not to warrant a more elaborate rule.

    Returns `(cell, target_dist_m)` - the distance grid is returned (whether it was
    passed in or computed here) so a caller that also wants to know the RESOLVED
    cell's own distance to the target edge (e.g. `scripts.audit.fit_prob`'s stricter
    fit criterion: is the resolved goal actually close enough to count as "reaching"
    the object, separately from whether it merely clears every obstacle) can read
    `target_dist_m[cell]` without a second query.

    Raises `TargetOutOfBoundsError` if `target_polygon`'s bounding box doesn't
    overlap the grid at all, `GoalUnreachableError` if it does but nothing within
    `search_radius_m` meets the clearance bar.
    """
    if target_polygon.is_empty:
        # A zero-area (degenerate point/line) footprint is NOT rejected here - shapely
        # reports one as `is_valid=False` even though `.distance()`/`.intersects()`
        # both still work correctly against it (verified), and a real caller can
        # legitimately have a footprint this small (or a test fixture standing in for
        # an object as a bare point - see tests/test_goal_resolution.py). Only a
        # genuinely EMPTY geometry (e.g. NaN/malformed coordinates) is bad data.
        raise TargetOutOfBoundsError("target polygon is empty")

    grid_box = box(
        meta.origin_x, meta.origin_z, meta.origin_x + meta.width * meta.resolution, meta.origin_z + meta.height * meta.resolution
    )
    if not grid_box.intersects(target_polygon):
        raise TargetOutOfBoundsError(
            f"target polygon bounds {target_polygon.bounds} do not overlap the grid bounds {grid_box.bounds}"
        )

    if target_dist_m is None:
        target_dist_m = target_distance_grid(meta, target_polygon)

    required = robot_radius_m + clearance_margin_m
    near = target_dist_m <= search_radius_m
    clearance_ok = dist_to_obstacle_m >= required - 1e-9
    if not (near & clearance_ok).any():
        raise GoalUnreachableError(
            f"no_free_cell_within_2m: no cell within {search_radius_m}m of the target meets "
            f"clearance >= {required:.3f}m (robot_radius_m={robot_radius_m}, "
            f"clearance_margin_m={clearance_margin_m})"
        )

    # A cell only counts as a goal if it can SEE the object: the straight segment from
    # the cell to the nearest point on the target's edge must not cross an occupied cell.
    # Without this, "within 2.0m of the edge and in the start's component" was satisfiable
    # from the WRONG SIDE of a sealed wall - on a 0.1m grid a pocket 0.9m away from the
    # object passes the distance test, so a robot walled into that pocket was reported as
    # reaching the object (tests/test_pathfinding_reachability_cache.py's `far_side`).
    # Occupied cells are read off the distance transform itself (distance 0 == obstacle);
    # the target's OWN cells are excluded, or nothing could ever see it.
    # Connectivity is decided BEFORE the goal is picked, not checked afterwards. The
    # component is labelled over the clearance mask at THIS robot's radius (8-connected,
    # matching the planner's own 8-neighbour A*), and the goal is chosen only from the
    # start's own component - so a nearer cell in an isolated pocket can never win.
    if start_cell is not None:
        # Connectivity is labelled over TRAVERSABILITY (clearance >= robot_radius_m), not
        # over the stricter goal bar (radius + margin). The margin exists so the robot can
        # STOP next to the target without touching it; it must not decide which corridors
        # the robot may DRIVE through. Labelling on the strict mask instead made every
        # object unreachable for Go2 (0/17) because the start itself failed the 0.35 m bar
        # while the planner, which inflates by 0.2496 m, considers it perfectly traversable.
        traversable = dist_to_obstacle_m >= robot_radius_m - 1e-9
        labels, _n = ndimage.label(traversable, structure=np.ones((3, 3), dtype=int))
        start_label = int(labels[start_cell[0], start_cell[1]])
        if start_label == 0:
            raise StartDisconnectedError(
                f"start_disconnected: the start cell {start_cell} does not itself meet "
                f"clearance >= {robot_radius_m:.3f}m, so it belongs to no traversable component"
            )
        component = labels == start_label
    else:
        component = np.ones_like(clearance_ok, dtype=bool)

    # Criteria in order: clearance -> start's component -> nearest edge. A visibility
    # criterion was tried and REVERTED on 2026-09-09 - see the known-defect note in
    # var/scratch/STATE.md for the numbers.
    ok = near & clearance_ok & component
    if not ok.any():
        raise GoalUnreachableError(
            f"no_connected_cell_within_2m: cells within {search_radius_m}m of the target meet "
            f"clearance >= {required:.3f}m, but none is in the start's connected component - "
            f"the goal is fine, the route to it does not exist"
        )
    masked = np.where(ok, target_dist_m, np.inf)
    ix, iz = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return (int(ix), int(iz)), target_dist_m
