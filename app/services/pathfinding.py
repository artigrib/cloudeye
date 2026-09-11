"""World-meters <-> occupancy-grid pathfinding: A* over the tri-state occupancy grid,
obstacle inflation by robot radius, and line-of-sight path smoothing.

Own `heapq`-based 8-connected A*, not `pyastar2d` - avoids a Cython/C-extension build
dependency for grids this small (~80x80 typically), and the inflation/unknown-cost/
no-corner-cutting semantics need custom code regardless of which core A* is used.

Grid convention (verified against real pipeline output, not assumed): `cells` is
`uint8 (width, height)`, indexed `[ix, iz]` - axis 0 is X, axis 1 is Z, NOT transposed.
`world_to_cell`/`cell_to_world` are the two functions most likely to have a silent
off-by-origin bug (this project has hit exactly this class of coordinate-frame error
repeatedly), which is why they're covered by dedicated tests against the real recorded
grid edges, not just hand-picked numbers.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import box as shapely_box

from app.services import goal_resolution
from app.services.scene_ingest import GridMeta

logger = logging.getLogger(__name__)

FREE, OBSTACLE, UNKNOWN = 0, 1, 2
UNKNOWN_COST_MULTIPLIER = 1.5
SQRT2 = math.sqrt(2)
# A start/goal cell whose own connected component is this small or smaller is treated
# as sealed off, not just "a small room" - see relocate_if_isolated. 1 means "the cell
# itself is fine but has zero traversable neighbors", the exact "can't move at all"
# failure mode a wide robot's inflation radius can produce even where real, walkable
# space exists nearby (the inflated obstacle regions on either side of a gap just
# happen to touch/overlap at that one point). Deliberately not larger: this should only
# ever override a genuinely-stuck start, never second-guess a real, if tight, room.
ISOLATED_COMPONENT_MAX_CELLS = 1

#: How far outside the grid a START ANCHOR may sit and still be clamped in rather than
#: raising - see clamp_anchor_to_grid for the measured reason (own_0901_161054's first
#: camera pose, 0.62 m past the z edge, 500'd every reachability call on that scene).
ANCHOR_CLAMP_MARGIN_M = 1.0

_GRID_CACHE_MAXSIZE = 8
_GRID_CACHE: "OrderedDict[tuple[str, int], OccupancyGrid]" = OrderedDict()

_REACHABILITY_CACHE_MAXSIZE = 32
# Keyed by (scene_id, grid mtime, radius, height - the last two rounded to avoid
# float-noise cache misses) - see compute_reachability_cached.
_REACHABILITY_CACHE: "OrderedDict[tuple[str, int, float, float | None], ReachabilityResult]" = OrderedDict()

Cell = tuple[int, int]
WorldPoint = tuple[float, float]


class NoPathError(Exception):
    """No path exists between start and goal - carries a specific reason (start
    blocked, goal unreachable, etc), never raised for a silently-empty path."""


class GridUnavailableError(Exception):
    """The occupancy grid file is missing or unreadable."""


@dataclass
class OccupancyGrid:
    cells: np.ndarray  # uint8 (width, height), [ix, iz]
    meta: GridMeta


@dataclass(frozen=True)
class ObjectFootprint:
    """The minimal shape-independent description of an object `plan_to_object` needs -
    callers build this from whatever object representation they have (an ORM row, a
    `scene_ingest.ParsedObject`, ...), decoupling pathfinding from any one of them."""

    x: float
    z: float
    bbox_min_x: float
    bbox_min_z: float
    bbox_max_x: float
    bbox_max_z: float


@dataclass(frozen=True)
class PathResult:
    cells: list[Cell]
    points: list[WorldPoint]
    length_m: float
    duration_sec: float
    start_snapped: bool = False
    goal_snapped: bool = False


# --- Coordinate conversion ---------------------------------------------------------


def world_to_cell(x: float, z: float, meta: GridMeta) -> Cell:
    """World meters -> grid cell indices. Small out-of-bounds (within 2 cells) is
    clamped as floating-point slop right at an edge; anything further out raises -
    that's a real bug (coordinate-frame mismatch), not something to silently clamp."""
    ix = int(math.floor((x - meta.origin_x) / meta.resolution))
    iz = int(math.floor((z - meta.origin_z) / meta.resolution))
    if 0 <= ix < meta.width and 0 <= iz < meta.height:
        return ix, iz

    margin_m = 2 * meta.resolution
    x_max = meta.origin_x + meta.width * meta.resolution
    z_max = meta.origin_z + meta.height * meta.resolution
    if (meta.origin_x - margin_m <= x <= x_max + margin_m) and (
        meta.origin_z - margin_m <= z <= z_max + margin_m
    ):
        return max(0, min(ix, meta.width - 1)), max(0, min(iz, meta.height - 1))

    raise ValueError(
        f"world point ({x:.3f}, {z:.3f}) is far outside the grid bounds "
        f"(origin=({meta.origin_x:.3f},{meta.origin_z:.3f}), size={meta.width}x{meta.height} "
        f"@ {meta.resolution}m) - likely a coordinate-frame mismatch, not clamped"
    )


def clamp_anchor_to_grid(anchor_world: WorldPoint, meta: GridMeta) -> Cell:
    """The cell a START SEARCH should be anchored on, clamped into the grid.

    `world_to_cell` raises past a 2-cell margin, and for an object centroid or a goal that
    is correct: out there it means a coordinate-frame mismatch, and silence would be worse
    than an error. An anchor is a weaker thing - "roughly where the operator was standing
    when the scan started" - and the search it seeds does the real work, so a little slop
    should move the search, not fail the request.

    Measured, own_0901_161054: the first camera pose sits **0.62 m** past the grid's z
    edge, because the operator started in the doorway and the reconstruction's extent is a
    percentile-trimmed box that does not reach it. 40 of that scene's 47 camera poses are
    inside the grid, so the track is plainly in the right frame - yet every
    `/reachability` call on the scene answered **500** on this one point.

    ANCHOR_CLAMP_MARGIN_M is the line between the two readings. A metre is room-scale
    slop: a pose just outside a trimmed extent. Beyond it the loud failure stands, which
    is what `test_robot_start_position_itself_still_raises_when_far_outside_grid` exists
    to hold - a start 100 m out is a frame bug and must not be quietly clamped into a
    corner cell and answered as if it were a normal scene.
    """
    x, z = anchor_world
    x_max = meta.origin_x + meta.width * meta.resolution
    z_max = meta.origin_z + meta.height * meta.resolution
    m = ANCHOR_CLAMP_MARGIN_M
    if not (meta.origin_x - m <= x <= x_max + m and meta.origin_z - m <= z <= z_max + m):
        raise ValueError(
            f"start anchor ({x:.3f}, {z:.3f}) is more than {m:.2f}m outside the grid "
            f"(origin=({meta.origin_x:.3f},{meta.origin_z:.3f}), "
            f"size={meta.width}x{meta.height} @ {meta.resolution}m) - "
            f"likely a coordinate-frame mismatch, not clamped"
        )
    return (
        max(0, min(int(math.floor((x - meta.origin_x) / meta.resolution)), meta.width - 1)),
        max(0, min(int(math.floor((z - meta.origin_z) / meta.resolution)), meta.height - 1)),
    )


def cell_to_world(ix: int, iz: int, meta: GridMeta) -> WorldPoint:
    """Grid cell indices -> world meters, at the CELL CENTER."""
    return (
        meta.origin_x + (ix + 0.5) * meta.resolution,
        meta.origin_z + (iz + 0.5) * meta.resolution,
    )


# --- Grid loading / caching ---------------------------------------------------------


def load_grid(scene_id: object, npy_path: str | Path, meta: GridMeta) -> OccupancyGrid:
    """Load (and cache) the occupancy grid array. Cache key includes the file's mtime,
    so a re-processed scene (which never happens today per scene idempotency, but keeps
    this correct if that ever changes) never serves stale data."""
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise GridUnavailableError(f"occupancy grid file missing: {npy_path}")

    mtime_ns = npy_path.stat().st_mtime_ns
    key = (str(scene_id), mtime_ns)
    if key in _GRID_CACHE:
        _GRID_CACHE.move_to_end(key)
        return _GRID_CACHE[key]

    cells = np.load(npy_path)
    grid = OccupancyGrid(cells=cells, meta=meta)
    _GRID_CACHE[key] = grid
    if len(_GRID_CACHE) > _GRID_CACHE_MAXSIZE:
        _GRID_CACHE.popitem(last=False)
    return grid


def invalidate_grid(scene_id: object) -> None:
    """Drop any cached grid(s) (and reachability results, which key off the same grid)
    for this scene - called by scene deletion."""
    for key in [k for k in _GRID_CACHE if k[0] == str(scene_id)]:
        del _GRID_CACHE[key]
    for key in [k for k in _REACHABILITY_CACHE if k[0] == str(scene_id)]:
        del _REACHABILITY_CACHE[key]


# --- Inflation + cost grid -----------------------------------------------------------


def rasterize_footprints_as_obstacles(cells: np.ndarray, meta: GridMeta, footprints: list[ObjectFootprint]) -> np.ndarray:
    """A COPY of `cells` with every footprint's own axis-aligned bbox
    (`bbox_min_x/z`..`bbox_max_x/z`) marked OBSTACLE - mirrors
    `scripts/msa/export_presentation.py`'s `rasterize_hulls` (not imported: that
    module is a geo agent's, and this project's convention is read-only reuse of the
    pattern, not a cross-module import), which layers every object's own point-cloud
    convex hull on top of the raw occupancy grid as an extra obstacle source, because
    `occupancy.npy` is a height-band density scan, not object-aware - a desk whose top
    sits above the scanned band, or a shelf whose legs sit below it, can read as clear
    floor space even though a real object is there.

    Uses each object's BBOX, not a true hull: unlike the offline MSA/bootstrap
    pipeline (`scripts/msa/bootstrap.py`), which computes a real 2D convex hull from
    each object's own point cloud and persists it (`objects.json`'s `hull_xz`), the
    LIVE ingestion pipeline's `SceneObject` rows carry no hull at all - only
    `bbox_min_x/z`/`bbox_max_x/z` (see `app/services/scene_ingest.parse_scene_objects`/
    `ParsedObject`, and `scene_objects.json`'s own schema, checked directly: no hull
    field exists there). Computing a real hull live would mean reading each object's
    `.ply` point cloud on every `/reachability`/`POST /command` call - a new,
    per-request I/O cost this function deliberately avoids. A bbox is also, by
    construction, never SMALLER than the true hull it stands in for (same
    conservative-by-construction reasoning as `app.robots.resolve_footprint_m`'s own
    square-from-radius fallback), so this never under-counts collision risk versus a
    tighter hull - it can only be more cautious.

    `footprints` should normally include EVERY object in the scene, the one currently
    being planned toward included (matching `rasterize_hulls`'s own "every hull, no
    exceptions" behaviour) - `resolve_goal_cell`'s own clearance search already finds
    a cell OUTSIDE any obstacle (including the target's own bbox), so rasterizing it
    too does not make it unreachable, just correctly excludes its own interior."""
    out = cells.copy()
    if not footprints:
        return out
    ix, iz = np.meshgrid(np.arange(meta.width), np.arange(meta.height), indexing="ij")
    xs = meta.origin_x + (ix + 0.5) * meta.resolution
    zs = meta.origin_z + (iz + 0.5) * meta.resolution
    for fp in footprints:
        inside = (xs >= fp.bbox_min_x) & (xs <= fp.bbox_max_x) & (zs >= fp.bbox_min_z) & (zs <= fp.bbox_max_z)
        out[inside] = OBSTACLE
    return out


def cut_footprints_from_grid(cells: np.ndarray, meta: GridMeta, footprints: list[ObjectFootprint]) -> np.ndarray:
    """A COPY of `cells` with every footprint's own axis-aligned bbox set back to FREE -
    the "cut" half of the bbox cut/paste that answers "what if this were somewhere else".

    The inverse of `rasterize_footprints_as_obstacles`, and deliberately just as blunt.
    `occupancy.npy` is a height-band density scan, not object-aware: the cells an object
    occupies are not labelled as that object's, so the only way to take it out of the
    grid is to clear the rectangle it stands in. That rectangle also contains whatever
    else happened to be inside it - a bit of wall the bbox overlaps, floor clutter below
    the scanned band - and clearing it says those are free floor too.

    That is why the UI calls this operation APPROXIMATE and says so next to its answer.
    It is a defensible approximation in one direction: it can only make the room look
    MORE open than it is, so a "still blocked" answer after moving something is
    trustworthy, and a "now reachable" answer is a hypothesis worth re-scanning to
    confirm. Nothing in the normal reachability path calls this - no move, no cut.
    """
    out = cells.copy()
    if not footprints:
        return out
    ix, iz = np.meshgrid(np.arange(meta.width), np.arange(meta.height), indexing="ij")
    xs = meta.origin_x + (ix + 0.5) * meta.resolution
    zs = meta.origin_z + (iz + 0.5) * meta.resolution
    for fp in footprints:
        inside = (xs >= fp.bbox_min_x) & (xs <= fp.bbox_max_x) & (zs >= fp.bbox_min_z) & (zs <= fp.bbox_max_z)
        out[inside] = FREE
    return out


def translate_footprint(fp: ObjectFootprint, dx: float, dz: float) -> ObjectFootprint:
    """The same footprint, dx/dz metres away. Pure translation - the bbox keeps its size
    and its axis alignment, which is the whole approximation: an object is a rectangle
    that slides, never one that turns."""
    return ObjectFootprint(
        x=fp.x + dx,
        z=fp.z + dz,
        bbox_min_x=fp.bbox_min_x + dx,
        bbox_min_z=fp.bbox_min_z + dz,
        bbox_max_x=fp.bbox_max_x + dx,
        bbox_max_z=fp.bbox_max_z + dz,
    )


def inflate(cells: np.ndarray, meta: GridMeta, radius_m: float) -> np.ndarray:
    """Dilate OBSTACLE cells by `radius_m` (a disk kernel, pure numpy - no scipy
    dependency for this). Cells within the radius of an obstacle become OBSTACLE too,
    including previously-UNKNOWN ones: the robot shouldn't graze close to a wall just
    because that particular cell was never directly observed."""
    radius_cells = int(math.ceil(radius_m / meta.resolution))
    if radius_cells <= 0:
        return cells.copy()

    width, height = cells.shape
    obstacle_ix, obstacle_iz = np.nonzero(cells == OBSTACLE)
    if len(obstacle_ix) == 0:
        return cells.copy()

    offsets = np.array(
        [
            (dx, dz)
            for dx in range(-radius_cells, radius_cells + 1)
            for dz in range(-radius_cells, radius_cells + 1)
            if dx * dx + dz * dz <= radius_cells * radius_cells
        ]
    )

    all_ix = (obstacle_ix[:, None] + offsets[None, :, 0]).ravel()
    all_iz = (obstacle_iz[:, None] + offsets[None, :, 1]).ravel()
    valid = (all_ix >= 0) & (all_ix < width) & (all_iz >= 0) & (all_iz < height)

    inflated = cells.copy()
    inflated[all_ix[valid], all_iz[valid]] = OBSTACLE
    return inflated


def build_cost_grid(inflated_cells: np.ndarray) -> np.ndarray:
    """OBSTACLE -> inf (impassable), UNKNOWN -> 1.5x, FREE -> 1x.

    Blocking unknown outright was tried, on 2026-09-09, at the owner's instruction, and
    MEASURED on hero-74 before shipping. It gives ZERO reachable objects for every robot:

        unknown rule   min_points   unknown cells   burger   go2     husky
        blocked            1        6686 (42%)      0/17     0/17    0/17
        blocked            3        6770 (43%)      0/17     0/17    0/17
        1.5x               1        6686 (42%)      13/17    10/17   10/17
        1.5x               3        6770 (43%)      13/17    10/17   10/17

    43% of that grid was never directly observed, so a robot that may not cross unknown
    cannot leave the pocket it starts in - it reaches nothing at all, which is a worse
    answer than any of the ones being argued about. The original docstring made this
    argument from 56% observed coverage; the number above is the same argument, measured
    on the layer that exists now rather than the one that existed then.

    So unknown stays traversable-at-a-penalty. It is not free: the multiplier is a real
    signal that the robot is heading into unverified space, and, unlike OBSTACLE, unknown
    is NOT dilated by the robot's radius in `inflate`, so a drift cell blocks itself
    without sealing the corridor around it.

    Reverting this is one line if the owner reaffirms it - set the UNKNOWN row to np.inf -
    but it should not be done without the 0/17 above in view.
    """
    cost = np.ones(inflated_cells.shape, dtype=np.float64)
    cost[inflated_cells == UNKNOWN] = UNKNOWN_COST_MULTIPLIER
    cost[inflated_cells == OBSTACLE] = np.inf
    return cost


# --- A* -------------------------------------------------------------------------------


def _octile_heuristic(a: Cell, b: Cell) -> float:
    dx, dz = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dz) + (SQRT2 - 2) * min(dx, dz)


def neighbors8(ix: int, iz: int, cost_grid: np.ndarray):
    """8-connected neighbors with their move cost, skipping impassable cells and
    forbidding diagonal corner-cutting (a diagonal move is only allowed if BOTH
    orthogonal neighbors it would cut past are themselves traversable)."""
    width, height = cost_grid.shape
    for dx, dz in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        nx, nz = ix + dx, iz + dz
        if not (0 <= nx < width and 0 <= nz < height):
            continue
        step_cost = cost_grid[nx, nz]
        if not np.isfinite(step_cost):
            continue
        if dx != 0 and dz != 0:
            if not np.isfinite(cost_grid[ix + dx, iz]) or not np.isfinite(cost_grid[ix, iz + dz]):
                continue
            yield nx, nz, step_cost * SQRT2
        else:
            yield nx, nz, step_cost


def astar(cost_grid: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    if not np.isfinite(cost_grid[start]):
        raise NoPathError(f"start cell {start} is blocked (obstacle after inflation)")
    if not np.isfinite(cost_grid[goal]):
        raise NoPathError(f"goal cell {goal} is blocked (obstacle after inflation)")
    if start == goal:
        return [start]

    open_heap: list[tuple[float, float, Cell]] = [(_octile_heuristic(start, goal), 0.0, start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    visited: set[Cell] = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for nx, nz, move_cost in neighbors8(current[0], current[1], cost_grid):
            neighbor = (nx, nz)
            if neighbor in visited:
                continue
            tentative_g = g + move_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + _octile_heuristic(neighbor, goal), tentative_g, neighbor))

    raise NoPathError(f"no path found from {start} to {goal} - goal is unreachable from start")


def line_of_sight(cost_grid: np.ndarray, a: Cell, b: Cell) -> bool:
    """Bresenham line between two cells; True iff every cell along it is traversable."""
    x0, z0 = a
    x1, z1 = b
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz
    x, z = x0, z0
    while True:
        if not np.isfinite(cost_grid[x, z]):
            return False
        if x == x1 and z == z1:
            return True
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x += sx
        if e2 < dx:
            err += dx
            z += sz


def segment_cells(a: Cell, b: Cell) -> list[Cell]:
    """Every cell the straight line a->b passes through, endpoints included.

    The same Bresenham walk `line_of_sight` does, but collecting cells instead of testing
    them. Deliberately NOT shared with it: `smooth_path` calls `line_of_sight` O(n^2) times
    per route and it must stay allocation-free, while this runs once per accepted segment.
    """
    x0, z0 = a
    x1, z1 = b
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz
    x, z = x0, z0
    out: list[Cell] = []
    while True:
        out.append((x, z))
        if x == x1 and z == z1:
            return out
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x += sx
        if e2 < dx:
            err += dx
            z += sz


def route_cells(path: list[Cell]) -> set[Cell]:
    """The cells a SMOOTHED route actually crosses.

    `smooth_path` returns waypoints, not a cell walk: two consecutive waypoints can be
    twenty cells apart. Anything that asks "what does this route drive over" - how much of
    it crosses ground nobody observed, say - has to rasterise the segments between them,
    or it inspects a handful of corners and calls that the route.
    """
    if not path:
        return set()
    cells: set[Cell] = {path[0]}
    for a, b in zip(path, path[1:]):
        cells.update(segment_cells(a, b))
    return cells


def smooth_path(path_cells: list[Cell], cost_grid: np.ndarray) -> list[Cell]:
    """Greedy line-of-sight shortcutting: from each kept point, jump as far ahead as
    possible while a straight line to that point stays fully traversable."""
    if len(path_cells) <= 2:
        return path_cells
    smoothed = [path_cells[0]]
    i = 0
    while i < len(path_cells) - 1:
        j = len(path_cells) - 1
        while j > i + 1 and not line_of_sight(cost_grid, path_cells[i], path_cells[j]):
            j -= 1
        smoothed.append(path_cells[j])
        i = j
    return smoothed


def nearest_free_cell(cost_grid: np.ndarray, target: Cell, max_radius_cells: int = 40) -> Cell:
    """Ring-search (Chebyshev rings, nearest-by-Euclidean-distance within each ring)
    outward from `target` for the closest traversable cell. Used both to snap a
    blocked start/goal, and (via plan_to_object) to find a reachable spot near an
    object whose own centroid cell is typically inside its obstacle footprint."""
    width, height = cost_grid.shape
    tx, tz = target
    if 0 <= tx < width and 0 <= tz < height and np.isfinite(cost_grid[tx, tz]):
        return target

    for r in range(1, max_radius_cells + 1):
        best: tuple[int, int, int] | None = None
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                nx, nz = tx + dx, tz + dz
                if 0 <= nx < width and 0 <= nz < height and np.isfinite(cost_grid[nx, nz]):
                    dist2 = dx * dx + dz * dz
                    if best is None or dist2 < best[0]:
                        best = (dist2, nx, nz)
        if best is not None:
            return best[1], best[2]

    raise NoPathError(f"no free cell found within {max_radius_cells} cells of {target}")


def snap_if_blocked(cost_grid: np.ndarray, cell: Cell, max_radius_cells: int) -> tuple[Cell, bool]:
    """`cell` unchanged if already traversable, else the nearest traversable cell within
    `max_radius_cells` (via `nearest_free_cell`) - the one snapping rule every start/goal
    resolution in this module goes through, so there's exactly one place that decides
    what "blocked" means."""
    if np.isfinite(cost_grid[cell]):
        return cell, False
    return nearest_free_cell(cost_grid, cell, max_radius_cells), True


def _footprint_distance_to_obstacle_m(cells: np.ndarray, meta: GridMeta) -> np.ndarray:
    """`app.services.goal_resolution.resolve_goal_cell`'s required clearance
    reference: distance (metres) from every cell centre to the nearest OBSTACLE cell
    in the RAW (not radius-inflated) grid - see that module's docstring for why the
    clearance metric must be computed against the raw grid, not an already-inflated
    `cost_grid` (inflating first would double-count the radius)."""
    return ndimage.distance_transform_edt(cells != OBSTACLE) * meta.resolution


def resolve_goal_cell(cells: np.ndarray, meta: GridMeta, footprint: ObjectFootprint, robot_radius_m: float, start_cell: Cell | None = None) -> tuple[Cell, bool]:
    """The cell to path toward for a given object - the nearest traversable cell to
    the object's own footprint EDGE (its `bbox_min_x/z`..`bbox_max_x/z` rectangle, the
    only footprint shape `ObjectFootprint` carries) whose clearance to every obstacle
    is at least `robot_radius_m + app.services.goal_resolution.CLEARANCE_MARGIN_M`,
    via the shared `app.services.goal_resolution.resolve_goal_cell` rule (the same
    rule `scripts/msa/export_presentation.py`'s presentation-route planner and
    `scripts.audit.fit_prob`'s Monte-Carlo audit use - see that module's own
    docstring).

    Previously (before this shared rule): snapped from the object's CENTROID
    (`world_to_cell(footprint.x, footprint.z, meta)`) to the nearest traversable cell
    within a 1.0m ring search, with no clearance margin beyond bare radius-inflation
    traversability. That could plan a robot to stop right at an obstacle's inflated
    boundary (zero safety margin) and anchored on a point that, for an elongated
    object, needn't be anywhere near where the robot could usefully interact with it.

    `cells` must be the RAW (not radius-inflated) tri-state grid - this function does
    its own inflation-equivalent clearance check internally via the shared rule.

    Raises `ValueError` if the footprint's bbox doesn't overlap the grid at all
    (mirrors the old centroid-out-of-bounds case `world_to_cell` used to raise for),
    `NoPathError` if it does but nothing within
    `goal_resolution.SEARCH_RADIUS_M` meets the clearance bar (mirrors the old
    snap-radius-exceeded case) - both exception TYPES kept identical to before this
    rule changed, so every existing caller's `except ValueError`/`except NoPathError`
    (`plan_to_object`, `try_resolve_goal_cell`) keeps working unchanged."""
    target_polygon = shapely_box(footprint.bbox_min_x, footprint.bbox_min_z, footprint.bbox_max_x, footprint.bbox_max_z)
    dist_to_obstacle_m = _footprint_distance_to_obstacle_m(cells, meta)
    try:
        # Passing start_cell turns on the connectivity constraint: the goal is chosen from
        # the START's own traversable component, decided BEFORE the nearest cell is picked.
        # Without it a smaller robot could be handed a nearer goal in an isolated pocket and
        # be reported unreachable while a larger one succeeded (TurtleBot 11/17 vs Go2 13/17
        # on the hero scene; with it, 13/17 and 13/17 and the subset invariant holds).
        cell, _target_dist_m = goal_resolution.resolve_goal_cell(
            dist_to_obstacle_m, meta, target_polygon, robot_radius_m, start_cell=start_cell
        )
    except goal_resolution.TargetOutOfBoundsError as exc:
        raise ValueError(str(exc)) from exc
    except goal_resolution.GoalUnreachableError as exc:
        raise NoPathError(str(exc)) from exc
    # Every resolution via this rule involved a search away from the object's own
    # interior (never the literal centroid) - "snapped" is always True here, unlike
    # the old centroid-based rule where it distinguished "the centroid itself was
    # already clear" from "had to search". See PathResult.goal_snapped's docstring.
    return cell, True


def try_resolve_goal_cell(
    cells: np.ndarray,
    meta: GridMeta,
    footprint: ObjectFootprint,
    robot_radius_m: float,
    *,
    start_cell: Cell | None = None,
    obj_id: str | None = None,
    scene_id: object = None,
) -> tuple[Cell, bool] | None:
    """Like `resolve_goal_cell`, but returns None instead of raising for either of the
    two ways a single object's goal cell can fail to resolve:

    - `ValueError`: the object's footprint is outside the grid bounds entirely.
    - `NoPathError`: the footprint IS on the grid, but no cell within
      `app.services.goal_resolution.SEARCH_RADIUS_M` (2.0m) of its edge meets the
      required clearance (`robot_radius_m + goal_resolution.CLEARANCE_MARGIN_M`) - a
      small object sitting in a tight spot that a big robot's own clearance
      requirement seals off entirely, even though the robot's own start point still
      has room to stand (see this project's Husky/0.5528m reachability-500 report).

    Both are bad data/geometry for that one object, not a failed request for everyone
    else - callers that resolve goal cells for many objects at once
    (`compute_reachability`, `resolve_robot_start`) should use this instead of
    `resolve_goal_cell` so one bad object degrades gracefully rather than aborting the
    whole computation. Deliberately not distinguished in the return value (still just
    `tuple | None`) - callers that need to report why already have exactly one bucket
    for "goal cell didn't resolve" (`ReachabilityResult.unreachable_reasons`'s
    "outside_grid", documented as one of a closed set of exact strings alongside
    "disconnected"/"robot_does_not_fit" - see app/robots.py and
    ReachabilityResponse.unreachable_reasons in schemas.py); the ERROR log below is
    where the two cases are told apart when that matters.

    Deliberately NOT used for the robot's own start position anywhere - that one should
    keep raising loudly (see resolve_robot_start's docstring / call site).

    Logs at ERROR (with scene/object context via `extra`) so the failure is visible in
    `journalctl -p err` rather than silently disappearing."""
    try:
        return resolve_goal_cell(cells, meta, footprint, robot_radius_m, start_cell=start_cell)
    except ValueError as exc:
        logger.error(
            "object %s footprint (bbox centred near %.3f, %.3f) is outside the grid bounds - excluding it "
            "from reachability/goal resolution instead of failing the whole request: %s",
            obj_id if obj_id is not None else "<unknown>",
            footprint.x,
            footprint.z,
            exc,
            extra={"scene_id": str(scene_id) if scene_id is not None else "-"},
        )
        return None
    except NoPathError as exc:
        logger.error(
            "object %s footprint (bbox centred near %.3f, %.3f) is on the grid but no cell "
            "meeting the required clearance was found near its edge - excluding it from "
            "reachability/goal resolution instead of failing the whole request: %s",
            obj_id if obj_id is not None else "<unknown>",
            footprint.x,
            footprint.z,
            exc,
            extra={"scene_id": str(scene_id) if scene_id is not None else "-"},
        )
        return None


def path_length_m(points: list[WorldPoint]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])
    )


#: Same function under a name that is not shadowed by ReachabilityResult's
#: `path_length_m` field (and by the local dict of the same name that builds it) inside
#: compute_reachability. Kept as an alias rather than a rename so no existing caller,
#: test or import has to change.
path_length_m_of = path_length_m


# --- High-level planning ---------------------------------------------------------


def plan_path(
    grid: OccupancyGrid,
    start_world: WorldPoint,
    goal_world: WorldPoint,
    *,
    robot_radius_m: float,
    speed_mps: float,
) -> PathResult:
    """Plan a path between two world-space points. Snaps a blocked start/goal cell to
    the nearest free one within 0.5m (a common case: `ROBOT_RADIUS_M`-inflation makes a
    79x79 grid's edge cells frequently blocked)."""
    inflated = inflate(grid.cells, grid.meta, robot_radius_m)
    cost_grid = build_cost_grid(inflated)
    snap_radius_cells = max(1, int(math.ceil(0.5 / grid.meta.resolution)))

    start_cell, start_snapped = snap_if_blocked(
        cost_grid, world_to_cell(*start_world, grid.meta), snap_radius_cells
    )
    if start_snapped:
        logger.info("start cell blocked, snapped to %s", start_cell)
    start_cell, start_relocated = relocate_if_isolated(cost_grid, start_cell)
    if start_relocated:
        logger.warning(
            "start cell was individually free but had no traversable neighbors at "
            "radius=%.3fm (sealed off by obstacle inflation) - relocated to %s",
            robot_radius_m, start_cell,
        )
    start_snapped = start_snapped or start_relocated

    goal_cell, goal_snapped = snap_if_blocked(
        cost_grid, world_to_cell(*goal_world, grid.meta), snap_radius_cells
    )
    if goal_snapped:
        logger.info("goal cell blocked, snapped to %s", goal_cell)

    path_cells = smooth_path(astar(cost_grid, start_cell, goal_cell), cost_grid)
    points = [cell_to_world(ix, iz, grid.meta) for ix, iz in path_cells]
    length_m = path_length_m(points)
    return PathResult(
        cells=path_cells,
        points=points,
        length_m=length_m,
        duration_sec=length_m / speed_mps if speed_mps > 0 else 0.0,
        start_snapped=start_snapped,
        goal_snapped=goal_snapped,
    )


def plan_to_object(
    grid: OccupancyGrid,
    start_world: WorldPoint,
    footprint: ObjectFootprint,
    *,
    robot_radius_m: float,
    speed_mps: float,
    objects: list[ObjectFootprint] = (),
) -> PathResult:
    """Plan a path to the nearest reachable cell near an object, not its centroid - the
    centroid is usually inside the object's own obstacle-flagged footprint.

    `objects` (default empty - every existing test/call site that doesn't pass it
    keeps its old, occupancy-grid-only behaviour) is every object footprint in the
    scene, rasterized as an extra OBSTACLE source on top of the raw occupancy grid
    before inflation - see `rasterize_footprints_as_obstacles`'s own docstring for
    why (the occupancy grid alone is a height-band scan, not object-aware).
    `command_service.build_steps` always passes the scene's full object list; this
    default only matters for direct/test callers that don't care about that."""
    obstacle_cells = rasterize_footprints_as_obstacles(grid.cells, grid.meta, list(objects))
    inflated = inflate(obstacle_cells, grid.meta, robot_radius_m)
    cost_grid = build_cost_grid(inflated)
    snap_radius_cells = max(1, int(math.ceil(1.0 / grid.meta.resolution)))

    # The start is resolved FIRST now: the goal rule needs it to pick a cell from the
    # start's own connected component (see goal_resolution.resolve_goal_cell's start_cell
    # parameter). Before that constraint existed the order did not matter.
    start_cell, start_snapped = snap_if_blocked(
        cost_grid, world_to_cell(*start_world, grid.meta), snap_radius_cells
    )
    start_cell, start_relocated = relocate_if_isolated(cost_grid, start_cell)
    try:
        goal_cell, goal_snapped = resolve_goal_cell(obstacle_cells, grid.meta, footprint, robot_radius_m, start_cell=start_cell)
    except ValueError as exc:
        raise NoPathError(
            f"cannot plan to the target object: its footprint (bbox centred near {footprint.x:.3f}, "
            f"{footprint.z:.3f}) lies outside the occupancy grid bounds - {exc}"
        ) from exc
    if start_relocated:
        logger.warning(
            "start cell was individually free but had no traversable neighbors at "
            "radius=%.3fm (sealed off by obstacle inflation) - relocated to %s",
            robot_radius_m, start_cell,
        )
    start_snapped = start_snapped or start_relocated

    path_cells = smooth_path(astar(cost_grid, start_cell, goal_cell), cost_grid)
    points = [cell_to_world(ix, iz, grid.meta) for ix, iz in path_cells]
    length_m = path_length_m(points)
    return PathResult(
        cells=path_cells,
        points=points,
        length_m=length_m,
        duration_sec=length_m / speed_mps if speed_mps > 0 else 0.0,
        start_snapped=start_snapped,
        goal_snapped=goal_snapped,
    )


# --- Connectivity / robot_start resolution -----------------------------------------
#
# A cell can be individually traversable (finite cost) yet still be unreachable from
# most of the room, if it sits in a pocket that inflation has sealed off from the rest
# of the free space - `snap_if_blocked` alone can't catch that, since it only asks "is
# THIS cell blocked", not "can THIS cell get anywhere". These functions answer the
# second question.


def connected_component(cost_grid: np.ndarray, start: Cell) -> set[Cell]:
    """Every cell reachable from `start` by the same 8-connected, no-corner-cutting
    adjacency A* uses (`neighbors8`). Empty if `start` itself is blocked."""
    if not np.isfinite(cost_grid[start]):
        return set()
    seen: set[Cell] = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        cur = queue.popleft()
        for nx, nz, _cost in neighbors8(cur[0], cur[1], cost_grid):
            neighbor = (nx, nz)
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def all_connected_components(cost_grid: np.ndarray) -> list[set[Cell]]:
    """Every connected traversable area in the grid, in no particular order - the one
    full-grid scan `largest_connected_component` and `compute_reachability` both build on
    top of, so the room's connectivity is only ever computed one way."""
    width, height = cost_grid.shape
    seen_global: set[Cell] = set()
    components: list[set[Cell]] = []
    for ix in range(width):
        for iz in range(height):
            cell = (ix, iz)
            if cell in seen_global or not np.isfinite(cost_grid[cell]):
                continue
            component = connected_component(cost_grid, cell)
            seen_global |= component
            components.append(component)
    return components


def largest_connected_component(cost_grid: np.ndarray) -> set[Cell]:
    """The single largest connected traversable area anywhere in the grid - the fallback
    spot for `resolve_robot_start` when the camera-track start can't reach most objects."""
    return max(all_connected_components(cost_grid), key=len, default=set())


def _nearest_member(component: set[Cell], target: Cell) -> Cell:
    """The cell in `component` closest to `target` - used to place the relocated start
    near the room's overall centroid rather than at an arbitrary corner of it."""
    return min(component, key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2)


def relocate_if_isolated(cost_grid: np.ndarray, cell: Cell) -> tuple[Cell, bool]:
    """`cell` unchanged if its own connected component has more than
    `ISOLATED_COMPONENT_MAX_CELLS` cells, else relocated to the nearest member of the
    grid's largest connected component.

    Catches what `snap_if_blocked` can't: a cell that is individually traversable (not
    itself an obstacle) but has been sealed off from the rest of the room by obstacle
    inflation - e.g. a robot wide enough that its radius closes a real, narrow gap on
    both sides of a start point, leaving the robot able to stand there but not move at
    all, even though a genuinely passable room exists just past the gap. Distinct from
    `resolve_robot_start`'s own reachable-*fraction* relocation (object-count based,
    decided once at scene creation, and known to accept an isolated cell outright when
    there are zero objects to prove it bad - see that function's own use of this
    helper): this is a pure connectivity check with no object-count blind spot, cheap
    enough to run on every request at whatever radius is currently selected, not just
    once at ingest time with whatever radius happened to be the default then."""
    component = connected_component(cost_grid, cell)
    if len(component) > ISOLATED_COMPONENT_MAX_CELLS:
        return cell, False
    largest = largest_connected_component(cost_grid)
    if not largest:
        return cell, False
    return _nearest_member(largest, cell), True


@dataclass(frozen=True)
class StartForRadius:
    """Where the robot actually stands at ONE radius, re-derived per request.

    Distinct from `RobotStartResolution`, which is decided once at scene creation with
    whatever radius was default then and persisted to `Scene.robot_start_x/z`. A wider
    platform can make that persisted point stop fitting, and the answer for a 0.55 m
    Husky is not the answer for a 0.10 m Burger.
    """

    cell: Cell | None
    point: WorldPoint | None
    #: Metres from `anchor_world` to `point`. 0.0 when the anchor itself fits.
    moved_m: float
    #: "original" - the anchor fits as-is. "moved" - relocated, see `moved_m`.
    #: "none" - no cell in the whole grid has clearance >= r; this platform has nowhere
    #: to stand in this scene at all, which is a real answer, not an error.
    status: str


def resolve_start_for_radius(
    cost_grid: np.ndarray,
    meta: GridMeta,
    anchor_world: WorldPoint,
) -> StartForRadius:
    """The cell nearest `anchor_world` that the robot actually fits in at this radius.

    `cost_grid` is already inflated by the radius, so "fits" is exactly "finite cost".
    The search is over the WHOLE grid, not a fixed window - the window is what this
    replaces. `compute_reachability` used `snap_if_blocked` with a 0.5 m radius and
    reported every object `robot_does_not_fit` when nothing was found inside it, which
    conflates two very different situations:

      own_0902_140657 at Husky's 0.5528 m has 3,419 cells with clearance >= r (largest
      component 2,687), and the nearest one to the first camera pose is 0.83 m away.
      The room is perfectly traversable for a Husky; the start just sat outside a 0.5 m
      window, and the panel said 0/24.

      own_0901_173903_15fps at the same radius has ZERO such cells - its layer pack is a
      single 0.3 m ESDF slice, so clearance is capped near the slice height (EDT max
      0.600 m) and a 0.5528 m platform fits nowhere. 0 reachable is the truth there, and
      it deserves to be said as "no start position for this platform", not as 0/N.

    Candidates are restricted to the LARGEST connected component at this radius, and the
    nearest cell of that component to the anchor is chosen. Nearest-overall was tried
    first and is worse: on own_0902_140657 at Husky's 0.5528 m the nearest fitting cell
    of any component sits in a 423-cell pocket, and standing there reaches 1 of 24
    objects, while the grid's largest Husky component holds 2,687 cells. "Nearest" on its
    own optimises the distance the robot is teleported, which is not what anyone wants -
    what they want is the start it can actually work from. This keeps the anchor's pull
    (of all the cells in that component, the closest one) without letting a sealed-off
    pocket win just for being near.

    Ties in component size are broken by whichever `all_connected_components` returns
    first, which is deterministic (it scans ix then iz), so this is stable across runs.
    """
    finite = np.isfinite(cost_grid)
    if not finite.any():
        return StartForRadius(cell=None, point=None, moved_m=0.0, status="none")

    components = all_connected_components(cost_grid)
    largest = max(components, key=len)

    anchor_cell = clamp_anchor_to_grid(anchor_world, meta)
    members = np.array(sorted(largest), dtype=np.int64)
    d2 = (members[:, 0] - anchor_cell[0]) ** 2 + (members[:, 1] - anchor_cell[1]) ** 2
    best = int(np.argmin(d2))
    cell = (int(members[best, 0]), int(members[best, 1]))

    if cell == anchor_cell:
        # Keep the caller's exact world point rather than snapping it to the cell centre -
        # the robot is already standing somewhere legal.
        return StartForRadius(cell=cell, point=anchor_world, moved_m=0.0, status="original")

    point = cell_to_world(*cell, meta)
    moved = math.hypot(point[0] - anchor_world[0], point[1] - anchor_world[1])
    return StartForRadius(cell=cell, point=point, moved_m=moved, status="moved")


@dataclass(frozen=True)
class RobotStartResolution:
    point: WorldPoint
    cell: Cell
    snapped: bool  # start cell was individually blocked and got snapped to the nearest free one
    relocated: bool  # start was reassigned to the largest connected area (reachability was too low)
    reachable_count: int
    total_objects: int
    reachable_fraction: float


def resolve_robot_start(
    grid: OccupancyGrid,
    start_world: WorldPoint,
    objects: list[ObjectFootprint],
    *,
    robot_radius_m: float,
    min_reachable_fraction: float = 0.5,
) -> RobotStartResolution:
    """Pick the robot's actual usable starting point, run once at scene creation and
    persisted (see pipeline_orchestrator._run) so every command plans from a start that's
    known to reach the room, not from the raw camera-track position:

    1. Snap the start cell if it's individually blocked (obstacle after inflation) - the
       same `snap_if_blocked` every other start/goal resolution uses.
    2. Check what fraction of this scene's (non-fragment) objects are reachable from
       there. If that's at least `min_reachable_fraction`, keep it.
    3. Otherwise, re-home to the connected area nearest the original start's own
       component, and use that instead if it does strictly better. If it doesn't (the
       whole grid is this fragmented), keep the original - see the log message below.

    Both candidates and the reasoning are logged with numbers either way, per the spec's
    "и оба варианта, и причину выбора".
    """
    obstacle_cells = rasterize_footprints_as_obstacles(grid.cells, grid.meta, objects)
    inflated = inflate(obstacle_cells, grid.meta, robot_radius_m)
    cost_grid = build_cost_grid(inflated)
    snap_radius_cells = max(1, int(math.ceil(0.5 / grid.meta.resolution)))

    # The robot's own start position must keep raising loudly on failure (see this
    # function's docstring / the module docstring): a bad start point silently reported
    # as "nothing reachable" would look like a normal empty scene rather than the
    # coordinate-frame bug it actually is - worse than an obvious crash. Only the
    # per-object goal cells (below) get the tolerant treatment.
    start_cell, snapped = snap_if_blocked(
        cost_grid, world_to_cell(*start_world, grid.meta), snap_radius_cells
    )

    # An isolated start (individually free, but sealed off from the rest of the room by
    # obstacle inflation - see relocate_if_isolated) is unconditionally bad, regardless
    # of how many objects there are to prove it: the object-fraction logic below has two
    # blind spots that would otherwise accept it outright - `total == 0` (no objects to
    # check against) and a tie in `alt_fraction > fraction` (relocating wouldn't improve
    # the object count, even though the current start can't reach ANY of them, including
    # zero). Checked and, if needed, relocated before any of that logic runs.
    start_cell, isolation_relocated = relocate_if_isolated(cost_grid, start_cell)
    if isolation_relocated:
        snapped = True
        logger.warning(
            "robot_start relocated %s -> %s: original start was individually free but "
            "had no traversable neighbors (sealed off by obstacle inflation at "
            "radius=%.3fm)",
            world_to_cell(*start_world, grid.meta), start_cell, robot_radius_m,
        )

    # A bad object here is just that one object being unreachable, not a failed
    # scene-creation run for every other object - see try_resolve_goal_cell.
    resolved_goals = [try_resolve_goal_cell(obstacle_cells, grid.meta, o, robot_radius_m) for o in objects]  # noqa: E501 - start not known here
    total = len(resolved_goals)

    component = connected_component(cost_grid, start_cell)
    reachable = sum(1 for g in resolved_goals if g is not None and g[0] in component)
    fraction = reachable / total if total else 1.0
    logger.info(
        "robot_start candidate (camera track): cell=%s snapped=%s reachable=%d/%d objects (%.0f%%)",
        start_cell, snapped, reachable, total, fraction * 100,
    )

    if total == 0 or fraction >= min_reachable_fraction:
        point = cell_to_world(*start_cell, grid.meta) if snapped else start_world
        return RobotStartResolution(point, start_cell, snapped, isolation_relocated, reachable, total, fraction)

    largest = largest_connected_component(cost_grid)
    alt_cell = _nearest_member(largest, start_cell) if largest else start_cell
    alt_reachable = sum(1 for g in resolved_goals if g is not None and g[0] in largest)
    alt_fraction = alt_reachable / total
    logger.info(
        "robot_start candidate (largest connected area, size=%d): cell=%s reachable=%d/%d objects (%.0f%%)",
        len(largest), alt_cell, alt_reachable, total, alt_fraction * 100,
    )

    if alt_fraction > fraction:
        logger.warning(
            "robot_start relocated %s -> %s: camera-track start only reached %.0f%% of "
            "objects, the largest connected area reaches %.0f%%",
            start_cell, alt_cell, fraction * 100, alt_fraction * 100,
        )
        return RobotStartResolution(
            cell_to_world(*alt_cell, grid.meta), alt_cell, snapped, True, alt_reachable, total, alt_fraction
        )

    logger.warning(
        "robot_start kept at %s despite low reachability (%.0f%%) - the largest connected "
        "area does no better (%.0f%%); this looks like real gaps in the occupancy grid, "
        "not a bad start point",
        start_cell, fraction * 100, alt_fraction * 100,
    )
    point = cell_to_world(*start_cell, grid.meta) if snapped else start_world
    return RobotStartResolution(point, start_cell, snapped, isolation_relocated, reachable, total, fraction)


# --- Reachability (radius exploration) ----------------------------------------------
#
# Answers "at this robot radius, what can I actually reach" - for a UI slider letting
# someone explore the ROBOT_RADIUS_M/clutter tradeoff live, and for pre-flight checks
# (dim an object in the list before the user tries a command that can only fail).


@dataclass(frozen=True)
class ReachabilityResult:
    radius: float
    component_count: int
    largest_component_size: int
    start_component_size: int
    reachable_ids: frozenset[str]
    # object id -> why it's NOT in reachable_ids: "outside_grid" (centroid off the grid
    # entirely - bad data for that one object), "disconnected" (goal cell resolved
    # fine, just not in the start's connected component), or "robot_does_not_fit"
    # (every object, when the start itself has no free cell within 0.5m at this
    # radius - see compute_reachability's docstring).
    unreachable_reasons: dict[str, str]
    reachable_cells: np.ndarray  # bool, shape (width, height), [ix, iz] - True = in start's component
    #: Where the robot actually stands at THIS radius - see resolve_start_for_radius.
    #: Defaulted so older constructions (and tests) stay valid; `cell is None` means the
    #: platform fits nowhere in this scene.
    start: StartForRadius = StartForRadius(cell=None, point=None, moved_m=0.0, status="none")
    #: object id -> the length in metres of the route from `start` to that object, for
    #: every id in `reachable_ids` and for no other. Keys are exactly `reachable_ids`:
    #: an unreachable object has no route, and a zero would read as "it is right here".
    #:
    #: Planned with the same astar + smooth_path over the same cost grid `plan_to_object`
    #: uses, so this is the route itself measured, not an estimate of it - a straight-line
    #: or grid-distance figure would disagree with the length the command panel prints for
    #: the same object, and two different numbers for one route is worse than none.
    #: It answers from THIS PLATFORM'S START, which is what the object list is asking;
    #: a command issued later plans from wherever the robot currently stands.
    path_length_m: dict[str, float] = field(default_factory=dict)
    #: How many DISTINCT cells, across every route counted above, the robot would cross
    #: that nobody ever observed - UNKNOWN in the layer, traversable only because
    #: `build_cost_grid` prices unknown at 1.5x rather than blocking it (see that
    #: docstring for the 0/17 that blocking it measures). It is the honest footnote on the
    #: reachable count: on a scene where 43% of the grid was never seen, "13 of 17" leans
    #: on ground the scan never covered, and the verdict bar says so.
    unknown_cells_on_route: int = 0
    #: How many cells those same routes cross that the layer calls OBSTACLE. **This is
    #: always 0**, and it is here so that "always" is a number something can assert on
    #: rather than a claim in a docstring.
    #:
    #: It is not tautological. The route is planned on the INFLATED cost grid; this counts
    #: against the layer's own tri-state grid at this robot's height - the same array
    #: `GET /map?robot_id=...` serves to draw the picture. So it is a cross-check between
    #: the grid a route was planned on and the grid a user is looking at, which is exactly
    #: what was broken before the band layer: the old 0.1-0.5 m histogram could not see a
    #: table top, and routes were drawn straight through furniture (hero-74: 54 / 64 / 31
    #: such cells for burger / go2 / husky). If those two ever come apart again, this
    #: stops being 0 and demo/probe_route_obstacles.py fails.
    obstacle_cells_on_route: int = 0


def compute_reachability(
    grid: OccupancyGrid,
    start_world: WorldPoint,
    objects: list[tuple[str, ObjectFootprint]],
    *,
    robot_radius_m: float,
    scene_id: object = None,
) -> ReachabilityResult:
    """Which of `objects` (id, footprint pairs) the robot can path to at `robot_radius_m`,
    plus the grid's overall connectivity at that radius. `objects` decides reachability by
    the same goal-cell resolution `plan_to_object` uses (`resolve_goal_cell`), so this
    agrees with what an actual command would do - it never reports an object reachable
    that a real `goto` would then fail to reach, or vice versa.

    An object whose centroid is outside the grid bounds (bad data for that one object,
    e.g. a coordinate-frame mismatch) does NOT abort reachability for every other object -
    it's classified into `unreachable_reasons` as "outside_grid" instead. `scene_id` is
    optional and only used to stamp log context on that failure (via try_resolve_goal_cell).

    Unlike `resolve_robot_start` (see its docstring), a start that doesn't fit at this
    radius is NOT an error here: this runs on every radius-slider tick/platform switch a
    live user makes, not once at scene creation.

    The start is re-derived per radius by `resolve_start_for_radius`, which searches the
    WHOLE grid for the nearest cell the robot fits in rather than a fixed 0.5 m window.
    Only when the grid holds no such cell at all is every object marked
    "robot_does_not_fit" - and then that is the literal truth, reported in `start` as
    status "none". See `resolve_start_for_radius` for the two measured scenes that
    separate those cases."""
    obstacle_cells = rasterize_footprints_as_obstacles(grid.cells, grid.meta, [fp for _oid, fp in objects])
    inflated = inflate(obstacle_cells, grid.meta, robot_radius_m)
    cost_grid = build_cost_grid(inflated)

    start = resolve_start_for_radius(cost_grid, grid.meta, start_world)
    if start.cell is None:
        logger.info(
            "reachability: no cell anywhere in the grid has clearance >= %.3fm "
            "(scene_id=%s) - this platform has no start position in this scene",
            robot_radius_m, scene_id,
        )
        return ReachabilityResult(
            radius=robot_radius_m,
            component_count=0,
            largest_component_size=0,
            start_component_size=0,
            reachable_ids=frozenset(),
            unreachable_reasons={obj_id: "robot_does_not_fit" for obj_id, _ in objects},
            reachable_cells=np.zeros(cost_grid.shape, dtype=bool),
            start=start,
        )
    if start.status == "moved":
        logger.info(
            "reachability start relocated %.2fm to %s at radius=%.3fm (scene_id=%s) - the "
            "persisted start does not fit this platform",
            start.moved_m, start.cell, robot_radius_m, scene_id,
        )
    start_cell = start.cell

    components = all_connected_components(cost_grid)
    start_component = next((c for c in components if start_cell in c), set())
    largest_size = max((len(c) for c in components), default=0)

    reachable_ids: set[str] = set()
    unreachable_reasons: dict[str, str] = {}
    path_length_m: dict[str, float] = {}
    unobserved_on_routes: set[Cell] = set()
    obstacles_on_routes: set[Cell] = set()
    for obj_id, footprint in objects:
        # Every failure used to collapse into "outside_grid", which was actively
        # misleading: on the hero scene four objects reported that way were all INSIDE
        # the grid with a free cell 0.01-0.08m away, they simply had no cell wide enough
        # for the robot within 2m. goal_resolution now prefixes its message with a
        # machine-readable label, so the reason survives the trip through NoPathError.
        # "disconnected" is kept as the label for the no-route case, unchanged contract.
        try:
            resolved = resolve_goal_cell(obstacle_cells, grid.meta, footprint,
                                         robot_radius_m, start_cell=start_cell)
        except ValueError:
            resolved = None
            # Genuinely off the grid keeps its original label - that one was never
            # wrong; only the "on the grid but nothing fits nearby" case was.
            unreachable_reasons[obj_id] = "outside_grid"
        except NoPathError as exc:
            resolved = None
            msg = str(exc)
            for label in ("no_connected_cell_within_2m", "no_visible_cell_within_2m",
                          "no_free_cell_within_2m"):
                if label in msg:
                    unreachable_reasons[obj_id] = (
                        "disconnected" if label == "no_connected_cell_within_2m" else label
                    )
                    break
            else:
                unreachable_reasons[obj_id] = "outside_grid"
        if resolved is None:
            pass
        elif resolved[0] in start_component:
            reachable_ids.add(obj_id)
            # The goal cell is in the start's component, so astar always finds a route;
            # smoothing it is what makes this the same number plan_to_object reports.
            route = smooth_path(astar(cost_grid, start_cell, resolved[0]), cost_grid)
            path_length_m[obj_id] = path_length_m_of(
                [cell_to_world(ix, iz, grid.meta) for ix, iz in route]
            )
            # Counted against the layer's OWN tri-state grid, not the inflated one: the
            # inflation turns unknown cells inside an obstacle's dilation disc into
            # OBSTACLE, which would undercount exactly the cells worth reporting.
            crossed = route_cells(route)
            unobserved_on_routes.update(c for c in crossed if grid.cells[c] == UNKNOWN)
            obstacles_on_routes.update(c for c in crossed if grid.cells[c] == OBSTACLE)
        else:
            unreachable_reasons[obj_id] = "disconnected"

    if obstacles_on_routes:
        # Never seen, and if it is ever seen it is the layer and the planner disagreeing -
        # the defect the band layer closed, come back. Loud here as well as in the gate,
        # because a probe only runs when somebody runs it.
        logger.error(
            "reachability: %d route cell(s) cross OBSTACLE in the layer's own grid "
            "(scene_id=%s, radius=%.4f) - the planned grid and the served grid disagree",
            len(obstacles_on_routes), scene_id, robot_radius_m,
        )

    reachable_cells = np.zeros(cost_grid.shape, dtype=bool)
    for cell in start_component:
        reachable_cells[cell] = True

    return ReachabilityResult(
        radius=robot_radius_m,
        component_count=len(components),
        largest_component_size=largest_size,
        start_component_size=len(start_component),
        reachable_ids=frozenset(reachable_ids),
        unreachable_reasons=unreachable_reasons,
        reachable_cells=reachable_cells,
        start=start,
        path_length_m=path_length_m,
        unknown_cells_on_route=len(unobserved_on_routes),
        obstacle_cells_on_route=len(obstacles_on_routes),
    )


def compute_reachability_cached(
    scene_id: object,
    npy_path: str | Path,
    grid: OccupancyGrid,
    start_world: WorldPoint,
    objects: list[tuple[str, ObjectFootprint]],
    *,
    robot_radius_m: float,
    robot_height_m: float | None = None,
) -> ReachabilityResult:
    """`compute_reachability`, cached by (scene_id, grid mtime, radius, height) - a radius
    slider firing on every debounced tick shouldn't re-inflate and re-flood-fill the grid
    every time, and re-visiting a radius (slider dragged back) should be free.

    HEIGHT is in the key because `grid` is no longer one array per scene. It is derived
    from the layer's obstacle-height map at the requesting robot's own roof
    (`nav_layer.cells_for_height`), so two platforms that happen to share a radius but not
    a height are asking about two different grids, and a three-part key would hand the
    second one the first one's answer. Same failure this file's `test_cache_key_*` pair
    documents for radius, one axis over.
    """
    mtime_ns = Path(npy_path).stat().st_mtime_ns
    key = (str(scene_id), mtime_ns, round(robot_radius_m, 4),
           None if robot_height_m is None else round(robot_height_m, 4))
    if key in _REACHABILITY_CACHE:
        _REACHABILITY_CACHE.move_to_end(key)
        return _REACHABILITY_CACHE[key]

    result = compute_reachability(grid, start_world, objects, robot_radius_m=robot_radius_m, scene_id=scene_id)
    _REACHABILITY_CACHE[key] = result
    if len(_REACHABILITY_CACHE) > _REACHABILITY_CACHE_MAXSIZE:
        _REACHABILITY_CACHE.popitem(last=False)
    return result
