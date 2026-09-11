"""Rectangle-footprint SE(2) planning: A* over (x, y, theta) with 16 discrete
orientations, and three-level reachability (any orientation / requires alignment /
not reachable) - the footprint-aware sibling of `app/services/pathfinding.py`'s
circle-radius-based `inflate`/`astar`/`compute_reachability`.

Why this exists as a separate module, not a rewrite of pathfinding.py: the circle
planner's `inflate()` dilates OBSTACLE cells by a scalar radius once, then runs a
single 2D A* - correct for a robot whose collision shape is orientation-independent.
A rectangle isn't: whether a given (x, y) pose collides depends on theta, so
"inflation" isn't a single grid anymore, it's a per-orientation mask evaluated at
query time. Every registered platform's `dimensions_m` (length_m x width_m) already
exists in `app/robots.py`, sourced from the same vendor-Nav2/mesh-measurement pass as
`radius_m` (see that module's docstring and each platform's `notes` field) - this
module is the first consumer of `dimensions_m` for anything other than display.

Grid convention, coordinate frame, and cost values (`FREE`/`OBSTACLE`/`UNKNOWN`,
`build_cost_grid`'s unknown-penalty multiplier) are all reused unchanged from
`pathfinding.py` - a footprint-aware plan should agree with the circle planner about
what a cell's cost even means, differing only in HOW MUCH of the grid a given pose
occupies.

Orientation convention: theta=0 means the robot's `length_m` axis (local +x, "front-
back") is aligned with the grid's world +x axis, and `width_m` (local z, "left-right")
with world +z; theta increases counter-clockwise in the grid's (x, z) plane, same
right-handed sense `world_to_cell`/`cell_to_world` use. 16 steps -> 22.5 degrees apart.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.services.pathfinding import (
    FREE,
    SQRT2,
    Cell,
    NoPathError,
    ObjectFootprint,
    OccupancyGrid,
    WorldPoint,
    build_cost_grid,
    cell_to_world,
    world_to_cell,
)
from app.services.scene_ingest import GridMeta

THETA_STEPS = 16
THETA_STEP_RAD = 2 * math.pi / THETA_STEPS

# Rotating the footprint in place costs less than a full cell translation (1.0 at the
# cheapest/FREE cost) so the planner prefers "turn then go straight" over a long
# detour around a gap only passable at one heading, but a full 180-degree turn (8
# steps) still costs more than a couple of cell-moves, so it won't spin in place
# gratuitously either. Tunable; no vendor/measured source, this is a planning-cost
# choice not a physical quantity.
ROTATE_STEP_COST = 0.3

ReachabilityLevel = Literal["any_orientation", "requires_alignment", "not_reachable"]

Pose = tuple[int, int, int]  # (ix, iz, itheta)


def _theta_for_index(itheta: int) -> float:
    return itheta * THETA_STEP_RAD


def nearest_theta_index(theta_rad: float, theta_steps: int = THETA_STEPS) -> int:
    """The discrete orientation index closest to a continuous heading (radians)."""
    step = 2 * math.pi / theta_steps
    return int(round(theta_rad / step)) % theta_steps


@dataclass(frozen=True)
class RobotFootprint:
    """A rectangle centered on the robot's origin. `length_m` is the local-x
    (front/back) extent, `width_m` the local-z (left/right) extent - matches
    `app.robots.RobotDimensions`' field names so a platform's `dimensions_m` can be
    passed straight through (`RobotFootprint(p.dimensions_m.length_m,
    p.dimensions_m.width_m)`)."""

    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError(f"footprint dimensions must be positive, got {self.length_m}x{self.width_m}")


def footprint_cell_offsets(
    length_m: float, width_m: float, resolution_m: float, theta_rad: float
) -> list[Cell]:
    """Integer (dx, dz) cell offsets (relative to a center cell) covered by a
    `length_m` x `width_m` rectangle rotated by `theta_rad`, rasterized by testing
    each candidate cell's CENTER against the rotated rectangle with a half-cell
    margin. The margin means a cell whose center sits just outside the mathematical
    rectangle, but within half a cell of its boundary, is still counted as occupied -
    deliberately conservative (never under-covers the true footprint at this
    resolution), same "err toward marking too much OBSTACLE, not too little" spirit
    as pathfinding.inflate's disk kernel."""
    half_l = length_m / 2.0
    half_w = width_m / 2.0
    margin = resolution_m / 2.0
    max_extent = math.hypot(half_l, half_w)
    r_cells = int(math.ceil(max_extent / resolution_m)) + 1

    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    offsets: list[Cell] = []
    for dx in range(-r_cells, r_cells + 1):
        for dz in range(-r_cells, r_cells + 1):
            wx = dx * resolution_m
            wz = dz * resolution_m
            # Rotate the world-space offset into the robot's local frame (inverse
            # rotation of theta) so it can be compared against the axis-aligned
            # half-extents.
            lx = wx * cos_t + wz * sin_t
            lz = -wx * sin_t + wz * cos_t
            if abs(lx) <= half_l + margin and abs(lz) <= half_w + margin:
                offsets.append((dx, dz))
    return offsets


@dataclass(frozen=True)
class FootprintMaskCache:
    """Precomputed rectangle masks for all `theta_steps` discrete orientations of one
    footprint at one grid resolution - built once per (footprint, resolution) pair and
    reused for every pose query, since `footprint_cell_offsets` itself is not cheap."""

    footprint: RobotFootprint
    resolution_m: float
    theta_steps: int = THETA_STEPS
    masks: list[list[Cell]] = field(init=False)

    def __post_init__(self) -> None:
        masks = [
            footprint_cell_offsets(self.footprint.length_m, self.footprint.width_m, self.resolution_m, _theta_for_index(k))
            for k in range(self.theta_steps)
        ]
        object.__setattr__(self, "masks", masks)


@dataclass
class PoseChecker:
    """Binds a `FootprintMaskCache` to one occupancy grid's blocked-cell set, so
    `is_free`/`any_theta_free`/`free_theta_count` don't need the grid passed at every
    call site. `blocked` is the set of cells the footprint may never cover (OBSTACLE
    after `build_cost_grid`, i.e. infinite cost) - UNKNOWN cells are allowed (same
    traversable-at-a-penalty treatment `pathfinding.build_cost_grid` uses), matching
    the circle planner's semantics exactly."""

    mask_cache: FootprintMaskCache
    blocked: frozenset[Cell]
    width: int
    height: int

    def is_free(self, ix: int, iz: int, itheta: int) -> bool:
        for dx, dz in self.mask_cache.masks[itheta % self.mask_cache.theta_steps]:
            nx, nz = ix + dx, iz + dz
            if nx < 0 or nx >= self.width or nz < 0 or nz >= self.height or (nx, nz) in self.blocked:
                return False
        return True

    def any_theta_free(self, ix: int, iz: int) -> bool:
        return any(self.is_free(ix, iz, k) for k in range(self.mask_cache.theta_steps))

    def free_theta_count(self, ix: int, iz: int) -> int:
        return sum(1 for k in range(self.mask_cache.theta_steps) if self.is_free(ix, iz, k))

    def classify_cell(self, ix: int, iz: int) -> ReachabilityLevel:
        """Three-level reachability of ONE cell, orientation-only (no path
        connectivity involved - see module docstring / compute_footprint_reachability
        for the connectivity-aware version used per object):

        - "any_orientation": every one of the 16 discrete headings is collision-free
          here - the robot can stand at this cell facing any way.
        - "requires_alignment": at least one heading is free, but not all - the robot
          fits, but only at specific headings (e.g. squeezing through a doorway
          straight-on but not diagonally).
        - "not_reachable": no heading at all is collision-free here - the footprint
          simply does not fit at this cell, regardless of orientation.
        """
        count = self.free_theta_count(ix, iz)
        if count == 0:
            return "not_reachable"
        if count == self.mask_cache.theta_steps:
            return "any_orientation"
        return "requires_alignment"


def blocked_cells(cost_grid: np.ndarray) -> frozenset[Cell]:
    """The set of (ix, iz) cells a footprint may never cover - infinite-cost
    (OBSTACLE-after-inflation-equivalent) cells of `cost_grid`. Building this once as a
    Python set (rather than repeated `np.isfinite` array indexing) is what makes
    `PoseChecker.is_free`'s plain-Python offset loop fast enough for A*/reachability
    over the (x, y, theta) lattice - see this module's docstring."""
    ix, iz = np.nonzero(~np.isfinite(cost_grid))
    return frozenset(zip(ix.tolist(), iz.tolist()))


def classify_grid(checker: PoseChecker) -> np.ndarray:
    """Per-cell three-level classification for the WHOLE grid, as a `uint8` array
    (0=not_reachable, 1=requires_alignment, 2=any_orientation), shape (width, height) -
    same axis convention as `pathfinding.OccupancyGrid.cells`. Orientation-only, like
    `PoseChecker.classify_cell` - no start-connectivity. Meant for small/medium grids
    or offline reporting (e.g. this PR's before/after report); `compute_footprint_reachability`
    below evaluates specific goal cells only, which is cheap enough for interactive use
    on real scene-sized grids without paying for the whole grid."""
    levels = np.zeros((checker.width, checker.height), dtype=np.uint8)
    for ix in range(checker.width):
        for iz in range(checker.height):
            count = checker.free_theta_count(ix, iz)
            if count == checker.mask_cache.theta_steps:
                levels[ix, iz] = 2
            elif count > 0:
                levels[ix, iz] = 1
    return levels


# --- SE(2) A* -------------------------------------------------------------------------


def _se2_neighbors(cost_grid: np.ndarray, checker: PoseChecker, ix: int, iz: int, itheta: int):
    """Edges out of one (ix, iz, itheta) node: two in-place rotations (+-1 orientation
    step) plus up to 8 translations at the SAME orientation. A translation is only
    offered if the rectangle mask is collision-free at the destination cell/orientation
    AND (for a diagonal move) neither orthogonal cell it would cut past is itself
    blocked - the same no-corner-cutting rule `pathfinding.neighbors8` enforces,
    approximated here at cell-center granularity rather than re-checking the whole
    swept rectangle (a known simplification - see this module's docstring)."""
    theta_steps = checker.mask_cache.theta_steps
    for d in (-1, 1):
        nt = (itheta + d) % theta_steps
        if checker.is_free(ix, iz, nt):
            yield (ix, iz, nt), ROTATE_STEP_COST

    for dx, dz in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        nx, nz = ix + dx, iz + dz
        if not (0 <= nx < checker.width and 0 <= nz < checker.height):
            continue
        if dx != 0 and dz != 0:
            if (ix + dx, iz) in checker.blocked or (ix, iz + dz) in checker.blocked:
                continue
        if not checker.is_free(nx, nz, itheta):
            continue
        base_cost = cost_grid[nx, nz]
        if not np.isfinite(base_cost):
            continue
        step_cost = base_cost * (SQRT2 if dx != 0 and dz != 0 else 1.0)
        yield (nx, nz, itheta), step_cost


def _octile(a: Cell, b: Cell) -> float:
    dx, dz = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dz) + (SQRT2 - 2) * min(dx, dz)


def astar_se2(cost_grid: np.ndarray, checker: PoseChecker, start: Pose, goal_xy: Cell) -> list[Pose]:
    """A* over the (ix, iz, itheta) lattice. `start` is a specific pose (position AND
    orientation); the goal is a CELL, reachable at any orientation - the search ends as
    soon as any theta arrives at `goal_xy`, since an object/command goal has no
    required final heading. Heuristic (translation-only octile distance to `goal_xy`,
    ignoring rotation) is admissible: every edge cost is >= 0 and the true remaining
    cost can only be >= the translation-only lower bound."""
    if not checker.is_free(*start):
        raise NoPathError(f"start pose {start} collides with the robot footprint")
    if not checker.any_theta_free(*goal_xy):
        raise NoPathError(f"goal cell {goal_xy} has no collision-free orientation for this footprint")

    open_heap: list[tuple[float, float, Pose]] = [(_octile(start[:2], goal_xy), 0.0, start)]
    came_from: dict[Pose, Pose] = {}
    g_score: dict[Pose, float] = {start: 0.0}
    visited: set[Pose] = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current[0] == goal_xy[0] and current[1] == goal_xy[1]:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for neighbor, cost in _se2_neighbors(cost_grid, checker, *current):
            if neighbor in visited:
                continue
            tentative = g + cost
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + _octile(neighbor[:2], goal_xy), tentative, neighbor))

    raise NoPathError(f"no SE(2) path found from {start} to cell {goal_xy} - goal is unreachable from start")


def se2_path_to_world(path: list[Pose], meta: GridMeta) -> list[tuple[float, float, float]]:
    """(ix, iz, itheta) poses -> (x, z, theta_rad) world poses, cell centers."""
    return [(*cell_to_world(ix, iz, meta), _theta_for_index(it)) for ix, iz, it in path]


# --- Start/goal pose resolution --------------------------------------------------------


def _nearest_free_pose_near_cell(
    checker: PoseChecker, start_cell: Cell, preferred_theta: int, max_radius_cells: int = 12
) -> Pose | None:
    """Nearest (by cell ring, then angular distance from `preferred_theta`)
    collision-free pose to `start_cell`. Used to resolve the robot's own start when
    the raw start cell/heading collides with the footprint - mirrors
    `pathfinding.nearest_free_cell`'s ring search, extended to also search
    orientation."""
    tx, tz = start_cell
    theta_steps = checker.mask_cache.theta_steps

    def best_theta_at(ix: int, iz: int) -> int | None:
        best: tuple[int, int] | None = None
        for k in range(theta_steps):
            if not checker.is_free(ix, iz, k):
                continue
            ang_dist = min((k - preferred_theta) % theta_steps, (preferred_theta - k) % theta_steps)
            if best is None or ang_dist < best[0]:
                best = (ang_dist, k)
        return best[1] if best else None

    theta = best_theta_at(tx, tz)
    if theta is not None:
        return tx, tz, theta

    for r in range(1, max_radius_cells + 1):
        candidates: list[tuple[int, int, int, int]] = []  # (dist2, ix, iz, theta)
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                nx, nz = tx + dx, tz + dz
                if not (0 <= nx < checker.width and 0 <= nz < checker.height):
                    continue
                theta = best_theta_at(nx, nz)
                if theta is not None:
                    candidates.append((dx * dx + dz * dz, nx, nz, theta))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            _, nx, nz, theta = candidates[0]
            return nx, nz, theta
    return None


def _nearest_cell_with_free_orientation(checker: PoseChecker, target: Cell, max_radius_cells: int = 12) -> Cell | None:
    """Like `_nearest_free_pose_near_cell` but only cares about the CELL, not which
    orientation - used to snap an object's goal cell (which has no required heading)
    the way `pathfinding.resolve_goal_cell` snaps a blocked centroid cell."""
    tx, tz = target
    if 0 <= tx < checker.width and 0 <= tz < checker.height and checker.any_theta_free(tx, tz):
        return target
    for r in range(1, max_radius_cells + 1):
        best: tuple[int, int, int] | None = None
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                nx, nz = tx + dx, tz + dz
                if not (0 <= nx < checker.width and 0 <= nz < checker.height):
                    continue
                if checker.any_theta_free(nx, nz):
                    dist2 = dx * dx + dz * dz
                    if best is None or dist2 < best[0]:
                        best = (dist2, nx, nz)
        if best is not None:
            return best[1], best[2]
    return None


# --- Reachability (three-level, per object) --------------------------------------------


@dataclass(frozen=True)
class FootprintReachabilityResult:
    footprint: RobotFootprint
    theta_steps: int
    reachable_ids: frozenset[str]  # any_orientation_ids | requires_alignment_ids
    any_orientation_ids: frozenset[str]
    requires_alignment_ids: frozenset[str]
    # Same closed-set convention as pathfinding.ReachabilityResult.unreachable_reasons:
    # "outside_grid", "robot_does_not_fit" (no free orientation anywhere near the goal
    # cell, or the start itself doesn't fit at any orientation), or "disconnected" (a
    # free orientation exists at the goal cell, but it's in none of the fixed-
    # orientation connected components the start belongs to).
    unreachable_reasons: dict[str, str]
    # obj_id -> how many of the 16 discrete headings have a FIXED-orientation path
    # (see compute_footprint_reachability's docstring) from start to that object's
    # (possibly snapped) goal cell. 0 for every id in unreachable_reasons.
    reachable_theta_counts: dict[str, int]


def _fixed_orientation_cost_grid(passable_theta: np.ndarray) -> np.ndarray:
    """`passable_theta` (bool, shape (width, height)) -> a cost grid `pathfinding`'s
    `connected_component` can flood-fill: 1.0 where the footprint fits at this fixed
    orientation, inf where it doesn't. No FREE/UNKNOWN cost distinction here (unlike
    `pathfinding.build_cost_grid`) - this function answers pure connectivity, not path
    cost; `astar_se2` (which DOES use the real cost grid) is what a caller wanting an
    actual weighted path/duration should use instead."""
    return np.where(passable_theta, 1.0, np.inf)


def compute_footprint_reachability(
    grid: OccupancyGrid,
    start_world: WorldPoint,
    start_theta_rad: float,
    objects: list[tuple[str, ObjectFootprint]],
    footprint: RobotFootprint,
    *,
    theta_steps: int = THETA_STEPS,
    goal_snap_radius_m: float = 1.0,
) -> FootprintReachabilityResult:
    """The rectangle-footprint analogue of `pathfinding.compute_reachability`, with
    THREE outcomes per object instead of two:

    For each of the `theta_steps` discrete headings k, build the set of cells the
    footprint fits at k (`PoseChecker.is_free(..., k)` for every cell) and flood-fill
    (`pathfinding.connected_component`, reused as-is) from the start cell within that
    set - exactly what the circle planner's `compute_reachability` does with
    `inflate()`, just with a per-orientation rectangle mask standing in for the
    scalar-radius disk. An object is:

    - "any_orientation" if its goal cell is in the start's connected component at
      EVERY one of the 16 headings - the robot can travel there facing any way the
      whole trip, no realignment ever forced.
    - "requires_alignment" if it's in the component for SOME but not all headings -
      reachable, but only by moving the whole way at one of those specific headings
      (see `reachable_theta_counts` for exactly how many/which - callers wanting the
      indices can recompute via `PoseChecker.is_free` for the object's goal cell).
    - not reachable (folded into `unreachable_reasons`, same closed-set convention as
      `pathfinding.ReachabilityResult`) if it's in the component for NONE of them.

    KNOWN SIMPLIFICATION, deliberate: this fixed-orientation-per-pass connectivity
    model is more conservative than the full (x, y, theta) lattice `astar_se2`
    searches - a route that needs to rotate MID-WAY (e.g. an L-shaped corridor too
    tight to turn a corner in without a shuffle) can be false-negative here even
    though `astar_se2` would actually find it, since this check never blends headings
    within one flood-fill pass. Chosen anyway over running `astar_se2` per object (an
    earlier version of this function did): `theta_steps` flood-fills, computed ONCE
    for the whole grid and reused for every object, is far cheaper on real scenes'
    object counts (tens to low hundreds) than `theta_steps`x the (x, y, theta) state
    space explored per-object - see this PR's report for real-scene timings. A caller
    that actually needs to execute a route (not just classify reachability) should
    plan it with `astar_se2` directly, which does not share this limitation.

    Deliberately does NOT call `pathfinding.inflate` first - the rectangle mask IS the
    footprint-aware equivalent of inflation, evaluated per-orientation; inflating by a
    scalar radius on top would double-count.
    """
    from app.services.pathfinding import connected_component  # local import: avoid a cycle risk if pathfinding ever imports this module

    cost_grid = build_cost_grid(grid.cells)
    mask_cache = FootprintMaskCache(footprint, grid.meta.resolution, theta_steps)
    checker = PoseChecker(mask_cache, blocked_cells(cost_grid), grid.meta.width, grid.meta.height)
    width, height = grid.meta.width, grid.meta.height

    snap_radius_cells = max(1, int(math.ceil(goal_snap_radius_m / grid.meta.resolution)))

    def _empty(reason_for_all: str) -> FootprintReachabilityResult:
        return FootprintReachabilityResult(
            footprint=footprint,
            theta_steps=theta_steps,
            reachable_ids=frozenset(),
            any_orientation_ids=frozenset(),
            requires_alignment_ids=frozenset(),
            unreachable_reasons={obj_id: reason_for_all for obj_id, _ in objects},
            reachable_theta_counts={obj_id: 0 for obj_id, _ in objects},
        )

    try:
        start_cell = world_to_cell(*start_world, grid.meta)
    except ValueError:
        # The robot's own start being off the grid is bad data for every object, not a
        # per-object one - mirrors resolve_robot_start's "keep raising loudly for the
        # start" stance, but this function reports rather than raises
        # (compute_reachability's own contract - see its docstring) since it's called
        # on every radius/footprint UI tick, not just once at scene creation.
        return _empty("outside_grid")

    if not checker.any_theta_free(*start_cell):
        # Snap the start to the nearest pose with ANY free heading (same spirit as
        # pathfinding.snap_if_blocked), so a start cell that's merely inconvenient
        # (not a genuine "footprint doesn't fit anywhere near here") doesn't report
        # every object unreachable.
        preferred_theta = nearest_theta_index(start_theta_rad, theta_steps)
        snapped_start = _nearest_free_pose_near_cell(checker, start_cell, preferred_theta)
        if snapped_start is None:
            return _empty("robot_does_not_fit")
        start_cell = snapped_start[0], snapped_start[1]

    # One flood-fill per heading, computed once for the whole grid/start and reused
    # for every object below - see docstring for why this beats per-object astar_se2.
    components: list[set[Cell]] = []
    for k in range(theta_steps):
        passable_k = np.zeros((width, height), dtype=bool)
        for ix in range(width):
            for iz in range(height):
                passable_k[ix, iz] = checker.is_free(ix, iz, k)
        if not passable_k[start_cell]:
            components.append(set())
            continue
        components.append(connected_component(_fixed_orientation_cost_grid(passable_k), start_cell))

    any_o: set[str] = set()
    align: set[str] = set()
    reasons: dict[str, str] = {}
    theta_counts: dict[str, int] = {}

    for obj_id, obj_footprint in objects:
        try:
            goal_cell = world_to_cell(obj_footprint.x, obj_footprint.z, grid.meta)
        except ValueError:
            reasons[obj_id] = "outside_grid"
            theta_counts[obj_id] = 0
            continue

        if not checker.any_theta_free(*goal_cell):
            snapped = _nearest_cell_with_free_orientation(checker, goal_cell, snap_radius_cells)
            if snapped is None:
                reasons[obj_id] = "robot_does_not_fit"
                theta_counts[obj_id] = 0
                continue
            goal_cell = snapped

        reachable_count = sum(1 for component in components if goal_cell in component)
        theta_counts[obj_id] = reachable_count

        if reachable_count == 0:
            reasons[obj_id] = "disconnected"
        elif reachable_count == theta_steps:
            any_o.add(obj_id)
        else:
            align.add(obj_id)

    return FootprintReachabilityResult(
        footprint=footprint,
        theta_steps=theta_steps,
        reachable_ids=frozenset(any_o | align),
        any_orientation_ids=frozenset(any_o),
        requires_alignment_ids=frozenset(align),
        unreachable_reasons=reasons,
        reachable_theta_counts=theta_counts,
    )
