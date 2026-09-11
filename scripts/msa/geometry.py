"""Stage A1/A3 geometry: wall polygons from an occupancy grid (with a
passage-preserving Douglas-Peucker simplification), and oriented minimum-area
footprints for measured objects. See var/scratch/msa/SPEC.md sections A1/A3.

All coordinates in this module are in the scene's metric XZ plane (Y is up and
handled separately by the caller) unless noted otherwise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FREE, OBSTACLE, UNKNOWN = 0, 1, 2


@dataclass(frozen=True)
class WallPolygon:
    """One simplified wall footprint polygon (closed ring, CCW, metres, XZ)."""

    vertices: list[tuple[float, float]]
    area_m2: float


@dataclass(frozen=True)
class DroppedComponent:
    """A connected obstacle component too small to be a real wall (SPEC A1:
    components < 0.3 m^2 are dropped and logged as phantom candidates)."""

    area_m2: float
    centroid_xy: tuple[float, float]
    cell_count: int


def _cell_to_world(ix: int, iz: int, resolution: float, origin_x: float, origin_z: float) -> tuple[float, float]:
    """Grid index `[ix, iz]` -> world (x, z) at the cell's lower corner.

    Grid-frame convention (T15b, docs/DECISIONS.md): `occupancy.npy` is indexed
    `[ix, iz]` - **axis 0 is X, axis 1 is Z** - exactly as `gpu/stage_occupancy.py`
    allocates it (`np.full((nx, nz))`, `occupancy_meta.json`'s `width` = nx =
    `shape[0]`, `height` = nz = `shape[1]`) and as `app/services/pathfinding.py`
    and `demo/record_isaac.py` read it (`grid[ix, iz]`). Every occupancy <-> world
    conversion in `scripts/msa/` goes through this function or mirrors it
    (`bootstrap.rasterize_object_footprint`, `gaps.compute_gaps`). Until T15b this
    module read the grid transposed (row -> z, col -> x), which reflected every
    wall/room polygon across the x = z diagonal relative to the object footprints
    and the point cloud.
    """
    return origin_x + ix * resolution, origin_z + iz * resolution


# T15b: an 8-connected obstacle component can consist of several 4-connected
# pieces that only touch at cell corners ("pinch" vertices). As shapely
# polygons, corner-touching squares are *separate* polygons, so the union of
# such a component is a MultiPolygon; buffering it by this much (1 mm - two
# orders of magnitude below the 5 cm cell, invisible in any export) fuses the
# pieces through a 2 mm neck into ONE valid polygon that keeps every cell's
# area, so the wall exports as a single mesh/collider under its single id.
PINCH_BRIDGE_M = 0.001


def _rasterize_polygon_boundary(mask: np.ndarray, resolution: float, origin_x: float, origin_z: float) -> list[tuple[float, float]]:
    """Outer boundary polygon of one connected cell mask (indexed `[ix, iz]`,
    see `_cell_to_world`) as a closed ring (first vertex repeated last, CCW) in
    world metres. Holes are dropped - outer contour only, as every caller wants.

    T15b: built with shapely (`unary_union` of the cells' squares) instead of
    the previous hand-rolled Moore corner tracer. That tracer stopped as soon
    as it returned to its start vertex, and at a "pinch" corner (two cells
    touching diagonally - legal under the 8-connected labeling
    `extract_wall_polygons` uses) it did so after a single cell: on the hero
    scene `wall_2` (2077 cells, 5.19 m^2, 12 four-connected pieces) traced as a
    5-vertex ring of 0.002 m^2 that simplified to an invalid 3-vertex polygon,
    and `wall_0` (162 cells) kept 0.058 of its 0.41 m^2 - both then silently
    exported with no mesh or collider. Pinch-connected pieces are fused with a
    `PINCH_BRIDGE_M` buffer (see there); a component with no pinch is returned
    with exactly its cell-corner outline (no buffer is applied then).
    """
    from shapely import box
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    ixs, izs = np.where(mask)
    if len(ixs) == 0:
        return []
    cells = [
        box(origin_x + ix * resolution, origin_z + iz * resolution, origin_x + (ix + 1) * resolution, origin_z + (iz + 1) * resolution)
        for ix, iz in zip(ixs.tolist(), izs.tolist())
    ]
    union = unary_union(cells)
    if isinstance(union, MultiPolygon) or not isinstance(union, Polygon):
        union = unary_union(union.buffer(PINCH_BRIDGE_M, join_style="mitre"))
        if isinstance(union, MultiPolygon):
            # Genuinely disconnected input (caller passed more than one
            # component): keep the largest piece rather than crash.
            union = max(union.geoms, key=lambda p: p.area)
    if union.is_empty:
        return []
    union = orient(union, sign=1.0)  # CCW exterior
    return [(float(x), float(z)) for x, z in union.exterior.coords]


def _point_to_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


def simplify_polygon_preserving_passages(
    polygon: list[tuple[float, float]],
    free_points: np.ndarray,
    *,
    tolerance_m: float = 0.05,
    passage_margin_m: float = 0.025,
    original_distance_fn=None,
) -> list[tuple[float, float]]:
    """Douglas-Peucker simplification of a closed polygon ring, with the SPEC A1
    passage-preserving invariant: a simplified segment is only accepted if, for
    every nearby free cell, distance-to-simplified-wall >= distance-to-original-wall
    - passage_margin_m. Segments that would violate this are recursively split and
    kept closer to the original vertices, down to consecutive original vertices in
    the worst case (i.e. simplification never expands a passage's wall boundary
    inward by more than the margin).

    `free_points` is an (N, 2) array of free-cell world XZ centers used to check the
    invariant against. `original_distance_fn(point) -> float` gives the distance
    from a world point to the *original* (unsimplified) polygon boundary; if not
    given, it's computed from `polygon` itself (fine when called with the true
    original ring).
    """
    if len(polygon) < 4:
        return list(polygon)

    ring = polygon if polygon[0] == polygon[-1] else [*polygon, polygon[0]]

    if original_distance_fn is None:
        original_ring = ring

        def original_distance_fn(p: tuple[float, float]) -> float:
            return min(_point_to_segment_distance(p, original_ring[i], original_ring[i + 1]) for i in range(len(original_ring) - 1))

    def nearby_free_points(a: tuple[float, float], b: tuple[float, float], radius: float) -> np.ndarray:
        if free_points.size == 0:
            return free_points
        min_x, max_x = min(a[0], b[0]) - radius, max(a[0], b[0]) + radius
        min_y, max_y = min(a[1], b[1]) - radius, max(a[1], b[1]) + radius
        mask = (
            (free_points[:, 0] >= min_x)
            & (free_points[:, 0] <= max_x)
            & (free_points[:, 1] >= min_y)
            & (free_points[:, 1] <= max_y)
        )
        return free_points[mask]

    def segment_is_safe(a: tuple[float, float], b: tuple[float, float], search_radius: float) -> bool:
        pts = nearby_free_points(a, b, search_radius)
        for p in pts:
            p_t = (float(p[0]), float(p[1]))
            simplified_dist = _point_to_segment_distance(p_t, a, b)
            original_dist = original_distance_fn(p_t)
            if simplified_dist < original_dist - passage_margin_m:
                return False
        return True

    def dp_recurse(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points
        a, b = points[0], points[-1]
        max_dist, max_idx = -1.0, -1
        for i in range(1, len(points) - 1):
            d = _point_to_segment_distance(points[i], a, b)
            if d > max_dist:
                max_dist, max_idx = d, i
        if max_dist <= tolerance_m and segment_is_safe(a, b, search_radius=max(tolerance_m, passage_margin_m) + 0.1):
            return [a, b]
        left = dp_recurse(points[: max_idx + 1])
        right = dp_recurse(points[max_idx:])
        return left[:-1] + right

    simplified = dp_recurse(ring)
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def extract_wall_polygons(
    occupancy: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_z: float,
    *,
    min_component_area_m2: float = 0.3,
    dp_tolerance_m: float = 0.05,
    passage_margin_m: float = 0.025,
) -> tuple[list[WallPolygon], list[DroppedComponent]]:
    """SPEC A1. `occupancy` is an `[ix, iz]` grid (axis 0 = X, axis 1 = Z - see
    `_cell_to_world`) of FREE/OBSTACLE/UNKNOWN.
    Returns (kept wall polygons, dropped small components)."""
    obstacle_mask = occupancy == OBSTACLE
    free_mask = occupancy == FREE
    free_ixs, free_izs = np.where(free_mask)
    free_points = np.array(
        [_cell_to_world(ix, iz, resolution, origin_x, origin_z) for ix, iz in zip(free_ixs, free_izs)]
        + [_cell_to_world(ix + 1, iz + 1, resolution, origin_x, origin_z) for ix, iz in zip(free_ixs, free_izs)]
    ) if len(free_ixs) else np.zeros((0, 2))

    labeled, n_components = ndimage.label(obstacle_mask, structure=np.ones((3, 3)))
    cell_area = resolution * resolution

    walls: list[WallPolygon] = []
    dropped: list[DroppedComponent] = []

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        cell_count = int(comp_mask.sum())
        area = cell_count * cell_area
        if area < min_component_area_m2:
            ixs, izs = np.where(comp_mask)
            centroid = (
                origin_x + (ixs.mean() + 0.5) * resolution,
                origin_z + (izs.mean() + 0.5) * resolution,
            )
            dropped.append(DroppedComponent(area_m2=area, centroid_xy=centroid, cell_count=cell_count))
            continue

        original_ring = _rasterize_polygon_boundary(comp_mask, resolution, origin_x, origin_z)
        if len(original_ring) < 4:
            continue
        simplified = simplify_wall_ring_validated(
            original_ring,
            free_points,
            cell_area_m2=area,
            tolerance_m=dp_tolerance_m,
            passage_margin_m=passage_margin_m,
            wall_id=f"wall_{len(walls)}",
        )
        walls.append(WallPolygon(vertices=simplified, area_m2=area))

    return walls, dropped


# T15b: a simplified wall ring must stay a valid polygon with at least this
# fraction of the component's cell-count area, else the simplification is
# retried with a tighter tolerance and finally the raw cell outline is used.
SIMPLIFIED_WALL_MIN_AREA_FRACTION = 0.5


def polygon_area_if_valid(ring: list[tuple[float, float]]) -> float:
    """Area of a closed ring as a polygon, or 0.0 if it has < 3 distinct
    vertices or is not a valid simple polygon (self-touching/crossing rings
    included - `buffer(0)` would "repair" those by splitting them into pieces,
    which is exactly the silent material loss T15b's wall_2 fix guards against)."""
    from shapely.geometry import Polygon

    if len(ring) < 4:
        return 0.0
    poly = Polygon(ring)
    if not poly.is_valid:
        return 0.0
    return float(poly.area)


def simplify_wall_ring_validated(
    original_ring: list[tuple[float, float]],
    free_points: np.ndarray,
    *,
    cell_area_m2: float,
    tolerance_m: float = 0.05,
    passage_margin_m: float = 0.025,
    wall_id: str = "wall",
    min_area_fraction: float = SIMPLIFIED_WALL_MIN_AREA_FRACTION,
) -> list[tuple[float, float]]:
    """`simplify_polygon_preserving_passages` with a validity ladder (T15b,
    docs/DECISIONS.md): the result must be a valid polygon whose area is at
    least `min_area_fraction` of `cell_area_m2` (the component's cell count x
    cell area - the ground truth the wall's `area_m2` reports). If the
    requested tolerance fails that, halve the tolerance and retry (twice),
    then fall back to the raw cell outline itself, which is valid by
    construction (`_rasterize_polygon_boundary`). Every fallback is logged
    with the wall id so a degenerate wall is never silent again."""
    import logging

    log = logging.getLogger(__name__)
    tolerances = [tolerance_m, tolerance_m / 2.0, tolerance_m / 4.0]
    for i, tol in enumerate(tolerances):
        simplified = simplify_polygon_preserving_passages(original_ring, free_points, tolerance_m=tol, passage_margin_m=passage_margin_m)
        simplified_area = polygon_area_if_valid(simplified)
        if simplified_area >= min_area_fraction * cell_area_m2:
            if i > 0:
                log.warning("%s: DP simplification at %.3f m tolerance was degenerate; kept %.3f m tolerance (area %.3f of %.3f m^2)", wall_id, tolerance_m, tol, simplified_area, cell_area_m2)
            return simplified
    raw_area = polygon_area_if_valid(original_ring)
    log.warning("%s: every DP tolerance produced a degenerate polygon; exporting the raw cell outline (%d vertices, area %.3f of %.3f m^2)", wall_id, len(original_ring), raw_area, cell_area_m2)
    return list(original_ring)


def extract_free_only_room_polygon(
    occupancy: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_z: float,
    *,
    closing_iterations: int = 2,
) -> list[tuple[float, float]] | None:
    """Room-shape polygon for **yaw detection only** (see `compute_room_polygon_yaw`
    in this module and the yaw-normalization step in `bootstrap.run_bootstrap`).

    Why this exists, alongside `bootstrap.extract_floor_polygon`: that function
    builds its polygon from the largest connected FREE+UNKNOWN component of the
    occupancy grid. `gpu/stage_occupancy.py` builds the grid as a
    percentile-trimmed axis-aligned bounding box of the point cloud, so UNKNOWN
    cells reliably touch the grid array's own edges. Wherever the room's FREE
    interior connects to that UNKNOWN halo through a doorway/gap, the largest
    FREE+UNKNOWN component's bounding box ends up measuring the *grid array's
    own axis-aligned extent* - not the true (possibly rotated) room shape. This
    function fixes the yaw signal's SOURCE by counting **FREE cells only** -
    UNKNOWN never counts as room, so a doorway into the UNKNOWN halo can't drag
    the polygon out to the grid's bounding box. `extract_floor_polygon` itself
    is untouched: other callers (object-footprint containment, floor-plate
    union) may depend on its existing FREE+UNKNOWN behavior.

    Method (mirrors `extract_wall_polygons`/`bootstrap.extract_floor_polygon`'s
    own pipeline so the three extractors stay consistent):
      1. `occupancy == FREE` mask (never UNKNOWN).
      2. Morphological closing (`scipy.ndimage.binary_closing`) with a full
         3x3 (8-connected) structuring element, `closing_iterations=2` by
         default, applied to that mask *before* picking a component: closing
         bridges a gap of up to `closing_iterations` cells between two parts
         of the free region that would otherwise fragment it into separate
         components or bite a large notch out of it (a doorway threshold
         rasterized as a thin non-FREE sliver, a single occluded/misclassified
         cell, ...) - closing *after* selecting a single component could never
         recover material a real fragmentation had already discarded into a
         different, smaller component. 2 iterations bridges gaps up to 2 cells
         wide - 10cm at the pipeline's standard 5cm grid resolution - the same
         order of magnitude as `bootstrap.WALL_BACKFILL_THRESHOLD_M` (0.10m),
         which this mirrors as "a gap this small is scan noise, not real room
         structure" rather than picking an unrelated constant. A real doorway
         into a genuinely separate room is far wider than a couple of cells at
         typical door widths, so it stays a separate component after closing
         and is correctly excluded by the next step.
      3. Largest 8-connected component of the *closed* mask
         (`scipy.ndimage.label`), same connectivity as `extract_wall_polygons`
         uses for obstacle components.
      4. Trace that component's outer boundary (`_rasterize_polygon_boundary`,
         the same corner-tracing helper `extract_wall_polygons` uses) and
         Douglas-Peucker simplify it (`simplify_polygon_preserving_passages`,
         same as `extract_floor_polygon`).

    A real doorway/passage can still leave a visible notch in the returned
    polygon where the closing step doesn't fully smooth it over (e.g. a
    passage wider than `closing_iterations` cells) - that's expected: this
    polygon feeds a *yaw* estimate (`oriented_min_area_rect`'s bounding
    rectangle only cares about the convex hull), not a rendered room outline,
    so a concave notch that doesn't reach the hull has no effect on the
    result.

    Returns None if there's no FREE cell in `occupancy`, or if the traced
    boundary has fewer than 4 vertices (degenerate/too-small component) -
    same failure contract as `extract_floor_polygon`.
    """
    free_mask = occupancy == FREE
    # Close *before* labeling, not after: closing is what keeps a room whose
    # free space is fragmented by a small gap (a doorway threshold rasterized
    # as a thin non-FREE sliver, a single occluded/misclassified cell, ...)
    # from being cut down to just its largest surviving fragment - closing an
    # already-selected single fragment could never recover material that
    # labeling discarded into a different, smaller component. A real doorway
    # into a genuinely separate room is (at typical door widths) far wider
    # than `closing_iterations` cells, so it stays a separate component after
    # closing and is correctly left out by the largest-component selection
    # below - only sub-`closing_iterations`-cell gaps get bridged.
    closed_free_mask = ndimage.binary_closing(free_mask, structure=np.ones((3, 3), dtype=bool), iterations=closing_iterations)

    labeled, n = ndimage.label(closed_free_mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(closed_free_mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    closed = labeled == largest

    ring = _rasterize_polygon_boundary(closed, resolution, origin_x, origin_z)
    if len(ring) < 4:
        return None
    return simplify_polygon_preserving_passages(ring, np.zeros((0, 2)), tolerance_m=0.05, passage_margin_m=0.025)


ROOM_POLYGON_METHOD = "largest_free_component_union_floor_standing_footprints_closed_outer_contour"


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Largest 8-connected True component of a boolean mask, or None if empty."""
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def _disk_structure(radius_cells: int) -> np.ndarray:
    r = int(radius_cells)
    span = np.arange(-r, r + 1)
    dx, dz = np.meshgrid(span, span, indexing="ij")
    return (dx * dx + dz * dz) <= r * r


def detach_corridor_spill(solid: np.ndarray, opening_radius_cells: int) -> np.ndarray:
    """Cut doorway spill off a *solid* (closed, hole-free) room mask (T15b,
    docs/DECISIONS.md).

    The largest FREE component leaks through any doorway into whatever was
    scanned beyond it - on the hero scene a corridor a few cells wide runs from
    the room to a second area whose obstacle fragment (the "bat") then counted
    as inside the room. A binary opening with a disk of radius
    `opening_radius_cells` erases every part of the mask that cannot contain a
    disk of that diameter (a door-width corridor and everything only reachable
    through it, once the largest opened component is kept), while a room, whose
    smallest dimension is metres, survives. The opening is used for
    *connectivity only*: its largest component is the room core, and the result
    is the original `solid` cells reachable from that core by at most
    `opening_radius_cells` steps of 8-connected geodesic dilation *inside
    `solid`*, so the boundary detail an opening would erase (a rasterized rotated
    corner, a strip between a bed and the wall) comes back, only the far side of
    a doorway (further than the radius from the core) is cut, and nothing outside
    `solid` is ever pulled in.

    It must run on the solid mask, not the raw FREE mask: the hero's FREE cells
    are riddled with UNKNOWN holes under and around furniture, and no 0.95 m
    disk fits in them at all - measured: an opening of the raw FREE mask at
    radius 9, 15 or 20 cells has zero components. If the opening empties the
    mask (a synthetic room narrower than the disk) `solid` is returned unchanged.
    `opening_radius_cells <= 0` disables the step.
    """
    if opening_radius_cells <= 0:
        return solid
    opened = ndimage.binary_opening(solid, structure=_disk_structure(opening_radius_cells))
    core = _largest_component(opened)
    if core is None:
        return solid
    grown = core & solid
    step = np.ones((3, 3), dtype=bool)
    for _ in range(int(opening_radius_cells)):
        grown = ndimage.binary_dilation(grown, structure=step) & solid
    kept = _largest_component(grown)
    return solid if kept is None else kept


def build_room_mask(
    occupancy: np.ndarray,
    footprint_masks: list[np.ndarray] | tuple[np.ndarray, ...] = (),
    *,
    closing_iterations: int = 2,
    opening_radius_cells: int = 0,
) -> np.ndarray | None:
    """Cell mask of the room (T15a - see `extract_room_polygon` for why this
    replaced both `bootstrap.extract_floor_polygon` and
    `extract_free_only_room_polygon` as the room reference):

      1. Largest 8-connected component of `occupancy == FREE` (UNKNOWN never
         counts - see `extract_free_only_room_polygon` for the halo bug).
      2. Union with every mask in `footprint_masks` - the caller passes the
         rasterized footprints of *floor-standing* objects only (see
         `bootstrap._floor_standing_footprint_masks`): a bed/desk/wardrobe is
         part of the room floor even where the occupancy band saw it as
         OBSTACLE or UNKNOWN, and its edges are what the room's orientation
         should follow. Elevated objects (a wall lamp, a TV on a bracket) say
         nothing about where the floor is and are deliberately excluded by the
         caller.
      3. Morphological closing, full 3x3 structuring element,
         `closing_iterations` (default 2 = 2 cells = 10cm at 5cm resolution -
         same reasoning as `extract_free_only_room_polygon`): bridges 1-2 cell
         notches/slits between the free region and the footprints (a doorway
         threshold sliver, an occluded cell alongside a footprint). Done on a
         padded copy so cells on the grid's own edge are not eroded away by
         the closing's erosion pass treating outside-the-grid as empty.
      4. Largest component of the closed union, holes filled (`binary_fill_holes`)
         - the OUTER contour only. Interior islands (an object seen as OBSTACLE
         in the middle of the free floor, an unscanned patch under a table)
         are room, not holes in it; this also guarantees the mask is
         simply-connected, which `_rasterize_polygon_boundary` requires.
      5. (T15b) `detach_corridor_spill` with `opening_radius_cells` (bootstrap
         passes `ROOM_OPENING_RADIUS_M / resolution`; 0 disables): a doorway-
         width corridor and everything beyond it are cut off the solid mask,
         then holes are filled once more. Runs on the solid mask, not the raw
         FREE cells - see `detach_corridor_spill` for why. Consequence: a
         floor-standing object beyond the doorway is no longer part of the room
         by construction and can be dropped by the outside-room rule.

    Returns the boolean mask (same shape as `occupancy`), or None if there is
    no FREE cell at all.
    """
    free_component = _largest_component(occupancy == FREE)
    if free_component is None:
        return None

    union = free_component.copy()
    for mask in footprint_masks:
        if mask is None:
            continue
        if mask.shape != occupancy.shape:
            raise ValueError(f"footprint mask shape {mask.shape} != occupancy shape {occupancy.shape}")
        union |= mask.astype(bool)

    pad = closing_iterations + 1
    padded = np.pad(union, pad, mode="constant", constant_values=False)
    closed = ndimage.binary_closing(padded, structure=np.ones((3, 3), dtype=bool), iterations=closing_iterations)
    closed = closed[pad:-pad, pad:-pad]

    # Closing is extensive (never removes a set cell), so the original largest
    # free component is still inside exactly one closed component - the one to
    # keep, even if a footprint elsewhere happened to grow its own separate blob.
    labeled, n = ndimage.label(closed, structure=np.ones((3, 3)))
    if n == 0:
        return None
    ids_touching_free = np.unique(labeled[free_component & (labeled > 0)])
    if len(ids_touching_free) == 0:
        room = _largest_component(closed)
    else:
        sizes = ndimage.sum(closed, labeled, ids_touching_free)
        room = labeled == int(ids_touching_free[int(np.argmax(sizes))])
    if room is None:
        return None
    room = ndimage.binary_fill_holes(room)
    if opening_radius_cells > 0:
        room = ndimage.binary_fill_holes(detach_corridor_spill(room, opening_radius_cells))
    return room


def extract_room_polygon(
    occupancy: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_z: float,
    footprint_masks: list[np.ndarray] | tuple[np.ndarray, ...] = (),
    *,
    closing_iterations: int = 2,
    opening_radius_cells: int = 0,
) -> list[tuple[float, float]] | None:
    """The room reference polygon (T15a, see docs/DECISIONS.md): outer contour
    of `build_room_mask` (largest FREE component union floor-standing object
    footprints, closed, holes filled), traced with `_rasterize_polygon_boundary`
    and Douglas-Peucker simplified with the same helper/tolerances every other
    room/wall polygon in this module uses.

    This single polygon is what `bootstrap.run_bootstrap` now uses for
    everything that used to read `floor_polygon` (FREE+UNKNOWN - leaked into
    the UNKNOWN halo that always touches the grid's edges) or the yaw-only
    `extract_free_only_room_polygon` (FREE only - but a bed/desk whose top is
    in the occupancy band shows up as OBSTACLE/UNKNOWN, so the free region
    alone is an irregular blob whose min-area rectangle does not follow the
    furniture, and the hero scene still rendered ~20 degrees off after a
    "correction"): yaw estimation, the outside-room drop/clip check, the
    floor plate, the USD/GLB floor mesh and the exported `floor_polygon`.

    Returns None if there is no FREE cell, or the traced boundary is degenerate
    (< 4 vertices) - same contract as the two functions it replaces.
    """
    room = build_room_mask(occupancy, footprint_masks, closing_iterations=closing_iterations, opening_radius_cells=opening_radius_cells)
    if room is None:
        return None
    ring = _rasterize_polygon_boundary(room, resolution, origin_x, origin_z)
    if len(ring) < 4:
        return None
    return simplify_polygon_preserving_passages(ring, np.zeros((0, 2)), tolerance_m=0.05, passage_margin_m=0.025)


# --- T15g: Manhattan regularization of the room outline + one wall band -----------

ROOM_NOTCH_M = 0.30  # steps shorter than this in the snapped outline are scan artefacts, not room shape
WALL_THICKNESS_M = 0.10  # the single wall band's thickness, outward from the room outline


def _rectilinear_runs(ring_xz: list[tuple[float, float]]) -> list[list[float]] | None:
    """Group a closed ring's edges (already in the axis-aligned frame, oriented
    CCW - interior to the left of every edge) into runs of consecutive edges
    sharing a dominant axis. Returns a cyclic, strictly alternating list of
    `[axis, coord, weight]` where axis 0 = edge runs along X (its constant
    coordinate is z), axis 1 = along Z (constant x); `weight` is the run's total
    projected length.

    `coord` is the run's OUTWARD extreme (the vertex coordinate farthest from
    the interior side), not its mean: the room polygon is the union of the free
    space and the floor-standing footprints, and a snapped edge through the mean
    of a rasterized/DP-simplified edge cuts a few cm into whatever sits against
    it (hero `nightstand_1` fell to 94.6% inside with the mean). Taking the outer
    line keeps the snapped room a superset of the raw outline along each run
    (bounded by the DP tolerance, ~5 cm); the walls sit outside it anyway."""
    pts = [tuple(map(float, p)) for p in ring_xz]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    runs: list[list[float]] = []
    n = len(pts)

    def merge_into(r: list[float], coord: float, weight: float, outward_sign: float) -> None:
        # outward_sign: +1 -> outward is the larger coordinate, -1 -> the smaller
        r[1] = max(r[1], coord) if outward_sign > 0 else min(r[1], coord)
        r[2] += weight

    for i in range(n):
        (x0, z0), (x1, z1) = pts[i], pts[(i + 1) % n]
        dx, dz = x1 - x0, z1 - z0
        if abs(dx) < 1e-12 and abs(dz) < 1e-12:
            continue
        axis = 0 if abs(dx) >= abs(dz) else 1
        # CCW: interior is to the left. Moving +x, left is +z -> outward is -z
        # (smaller z); moving +z, left is -x -> outward is +x (larger x).
        if axis == 0:
            outward = -1.0 if dx > 0 else 1.0
            coord = min(z0, z1) if outward < 0 else max(z0, z1)
            weight = abs(dx)
        else:
            outward = 1.0 if dz > 0 else -1.0
            coord = max(x0, x1) if outward > 0 else min(x0, x1)
            weight = abs(dz)
        if runs and runs[-1][0] == axis:
            merge_into(runs[-1], coord, weight, runs[-1][3])
        else:
            runs.append([axis, coord, weight, outward])
    # cyclic wrap: first and last run may share an axis
    if len(runs) >= 2 and runs[0][0] == runs[-1][0]:
        last = runs.pop()
        merge_into(runs[0], last[1], last[2], runs[0][3])
    runs = [r[:3] for r in runs]
    if len(runs) < 4 or len(runs) % 2:
        return None if len(runs) < 4 else runs[:-1] if runs[0][0] == runs[-1][0] else runs
    return runs


def _runs_to_vertices(runs: list[list[float]]) -> list[tuple[float, float]]:
    """Corner between run i and run i+1: x from whichever run is along Z, z from
    the run along X. Emits the vertex at the START of run i+1, i.e. one vertex per run."""
    verts = []
    n = len(runs)
    for i in range(n):
        a, b = runs[i], runs[(i + 1) % n]
        if a[0] == b[0]:  # should not happen for alternating runs
            continue
        x = a[1] if a[0] == 1 else b[1]
        z = a[1] if a[0] == 0 else b[1]
        verts.append((float(x), float(z)))
    return verts


def _run_lengths(runs: list[list[float]]) -> list[float]:
    """Current length of each run = distance between the coordinates of its two
    (parallel) neighbours."""
    n = len(runs)
    return [abs(runs[(i + 1) % n][1] - runs[(i - 1) % n][1]) for i in range(n)]


def _collapse_short_runs(runs: list[list[float]], notch_m: float) -> list[list[float]]:
    """Repeatedly remove the shortest run below `notch_m` by merging its two
    (parallel) neighbours into one run at their length-weighted coordinate -
    a notch's two short risers vanish and the wall line it interrupted heals;
    an appendix whose edges are all >= notch_m survives. Stops at 4 runs."""
    runs = [list(r) for r in runs]
    while len(runs) > 4:
        lengths = _run_lengths(runs)
        i = min(range(len(runs)), key=lambda k: lengths[k])
        if lengths[i] >= notch_m:
            break
        n = len(runs)
        a, b = runs[(i - 1) % n], runs[(i + 1) % n]
        la, lb = max(lengths[(i - 1) % n], 1e-9), max(lengths[(i + 1) % n], 1e-9)
        merged = [a[0], (a[1] * la + b[1] * lb) / (la + lb), a[2] + b[2]]
        # rebuild cyclically: ... , merged, ... (drop i-1, i, i+1; insert merged at i-1)
        order = [(i - 1 + k) % n for k in range(n)]  # starts at a
        rest = [runs[j] for j in order[3:]]  # everything after b
        runs = [merged, *rest]
        # the merge may leave the wrap-around neighbours with the same axis
        if len(runs) >= 2 and runs[0][0] == runs[-1][0]:
            last = runs.pop()
            total = runs[0][2] + last[2]
            runs[0][1] = (runs[0][1] * runs[0][2] + last[1] * last[2]) / total if total > 0 else runs[0][1]
            runs[0][2] = total
    return runs


def regularize_room_polygon(
    polygon_xz: list[tuple[float, float]],
    angle_rad: float,
    notch_m: float = ROOM_NOTCH_M,
) -> list[tuple[float, float]]:
    """T15g: snap a room outline to Manhattan geometry in the yaw-normalized frame.

    `angle_rad` is the yaw correction (`compute_room_polygon_yaw`'s
    `correction_rad`) that makes the room axis-aligned. The ring is rotated by it
    about its own centroid, every edge is classified by dominant axis
    (|dx| >= |dz| -> along X), consecutive same-axis edges are merged into one run
    at their length-weighted coordinate, runs shorter than `notch_m` are collapsed
    into their neighbours (`_collapse_short_runs` - a 1-2 cell raster notch goes,
    a doorway appendix 1 m x 0.9 m stays), the alternating runs are turned back
    into corners (every interior angle exactly 90 or 270 degrees), and the ring
    is rotated back by `-angle_rad` about the same pivot - so the result lives in
    the INPUT frame, and the caller applies the yaw to it together with walls,
    objects, gaps and cameras as the last step (a rotation by -a then +a about
    different pivots is a translation, so the exported ring stays rectilinear).

    The result is validated with shapely; a self-touching ring is repaired with
    `buffer(0)` (largest piece, still rectilinear). If nothing valid comes out
    (< 4 runs, or a degenerate repair) the input polygon is returned unchanged
    with a warning - never a crash in the bootstrap.
    """
    pts = [tuple(map(float, p)) for p in polygon_xz]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 4:
        return list(polygon_xz)
    from shapely.geometry import Polygon

    src = Polygon(pts)
    pivot = (float(src.centroid.x), float(src.centroid.y)) if src.is_valid and src.area > 0 else (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
    aligned = [rotate_point_xz(x, z, angle_rad, pivot) for x, z in pts]
    from shapely.geometry.polygon import orient

    aligned_poly = Polygon(aligned)
    if aligned_poly.is_valid and aligned_poly.area > 0:
        aligned = list(orient(aligned_poly, sign=1.0).exterior.coords)[:-1]  # CCW: interior on the left

    runs = _rectilinear_runs(aligned)
    if runs is None:
        logging.getLogger(__name__).warning("regularize_room_polygon: fewer than 4 axis runs - outline left unsnapped")
        return list(polygon_xz)
    runs = _collapse_short_runs(runs, notch_m)
    verts = _runs_to_vertices(runs)
    if len(verts) < 4:
        logging.getLogger(__name__).warning("regularize_room_polygon: collapsed to < 4 corners - outline left unsnapped")
        return list(polygon_xz)

    snapped = Polygon(verts)
    if not snapped.is_valid or snapped.area <= 0:
        repaired = snapped.buffer(0)
        parts = [g for g in getattr(repaired, "geoms", [repaired]) if isinstance(g, Polygon) and g.area > 1e-9]
        if not parts:
            logging.getLogger(__name__).warning("regularize_room_polygon: snapped ring invalid and not repairable - outline left unsnapped")
            return list(polygon_xz)
        snapped = max(parts, key=lambda g: g.area)
    from shapely.geometry.polygon import orient

    snapped = orient(snapped, sign=1.0)  # CCW like every other ring in this module
    out = [rotate_point_xz(float(x), float(z), -angle_rad, pivot) for x, z in list(snapped.exterior.coords)[:-1]]
    return out


def room_wall_band(room_polygon_xz: list[tuple[float, float]], thickness_m: float = WALL_THICKNESS_M):
    """T15g: the single wall footprint = `room.buffer(thickness, mitre) - room`, a
    ring-shaped shapely Polygon (one hole) hugging the regularized outline from
    the outside. Mitre joins keep a Manhattan outline Manhattan (round joins would
    add arcs at every corner). None if the room polygon is unusable."""
    from shapely.geometry import JOIN_STYLE, Polygon

    if not room_polygon_xz or len(room_polygon_xz) < 3:
        return None
    room = Polygon(room_polygon_xz)
    if not room.is_valid:
        room = room.buffer(0)
    if room.is_empty or room.area <= 0:
        return None
    outer = room.buffer(thickness_m, join_style=JOIN_STYLE.mitre, mitre_limit=5.0)
    band = outer.difference(room)
    return None if band.is_empty else band


def oriented_min_area_rect(points_xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    """SPEC A3: minimum-area oriented rectangle of a 2D point set (rotating
    calipers over the convex hull). Returns (center_xy, (length_u, width_v),
    angle_rad) where angle is the rotation of the rectangle's first axis (length)
    from the world +X axis, in [-pi/2, pi/2].
    """
    points_xy = np.asarray(points_xy, dtype=float)
    unique = np.unique(points_xy, axis=0)
    if len(unique) == 1:
        x, y = unique[0]
        return (float(x), float(y)), (0.0, 0.0), 0.0
    if len(unique) == 2:
        (x0, y0), (x1, y1) = unique
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        length = float(np.hypot(x1 - x0, y1 - y0))
        angle = float(np.arctan2(y1 - y0, x1 - x0))
        return center, (length, 0.0), angle

    from scipy.spatial import ConvexHull

    hull = ConvexHull(unique)
    hull_points = unique[hull.vertices]

    best_area = float("inf")
    best = None
    n = len(hull_points)
    for i in range(n):
        a = hull_points[i]
        b = hull_points[(i + 1) % n]
        edge = b - a
        edge_len = np.hypot(*edge)
        if edge_len < 1e-12:
            continue
        u = edge / edge_len
        v = np.array([-u[1], u[0]])
        proj_u = hull_points @ u
        proj_v = hull_points @ v
        min_u, max_u = proj_u.min(), proj_u.max()
        min_v, max_v = proj_v.min(), proj_v.max()
        area = (max_u - min_u) * (max_v - min_v)
        if area < best_area:
            best_area = area
            center_local = ((min_u + max_u) / 2.0, (min_v + max_v) / 2.0)
            center_world = u * center_local[0] + v * center_local[1]
            best = (
                (float(center_world[0]), float(center_world[1])),
                (float(max_u - min_u), float(max_v - min_v)),
                float(np.arctan2(u[1], u[0])),
            )
    assert best is not None
    center, size, angle = best
    # Normalize: length is the longer side, angle in [-pi/2, pi/2).
    length_u, width_v = size
    if length_u < width_v:
        length_u, width_v = width_v, length_u
        angle += np.pi / 2
    angle = ((angle + np.pi / 2) % np.pi) - np.pi / 2
    return center, (length_u, width_v), angle


def compute_room_polygon_yaw(
    floor_polygon: list[tuple[float, float]] | np.ndarray | None,
    *,
    square_tolerance: float = 0.02,
) -> dict | None:
    """Primary yaw-detection method (T3' rework - see docs/DECISIONS.md):
    derives the room's yaw from the room/floor free-space polygon
    (`extract_floor_polygon`'s output in `bootstrap.py`, the same polygon
    `_drop_or_clip_objects_outside_room` clips object footprints against),
    rather than from the wall-segment angle histogram. `compute_dominant_wall_yaw`
    (this module, above) is kept and still computed by the caller, but only as
    a secondary/diagnostic cross-check logged alongside this method's result -
    this function's output is what actually gets applied.

    Method: `oriented_min_area_rect` (the same rotating-calipers routine this
    module already uses for every measured-object footprint) over the floor
    polygon's own boundary vertices. A polygon's minimum-area bounding
    rectangle is exactly that of its convex hull, and a convex hull is fully
    determined by boundary vertices alone (no filled interior point cloud
    needed) - so calling `oriented_min_area_rect` directly on `floor_polygon`
    is exact, not an approximation, and reuses the existing rotating-calipers
    implementation rather than a second one.

    No confidence gate (unlike `compute_dominant_wall_yaw`): a minimum-area
    bounding rectangle always has a well-defined long axis, *except* the
    degenerate case where the room's bounding rectangle is (near-)square -
    "long" and "short" side are then interchangeable, so there's no
    principled choice of which one to rotate onto +X. A near-circular room
    also bounds to a near-square rectangle, so this one check covers both
    degenerate shapes the task calls out. `square_tolerance` (default 2%,
    relative to the longer side) sets how close counts as "square"; within
    that tolerance this returns `is_degenerate_square=True` and forces
    `correction_deg`/`correction_rad` to 0.0 (no rotation) instead of picking
    one of two equally-arbitrary axes.

    Returns None only if `floor_polygon` is missing or has fewer than 3
    vertices (no polygon to measure - mirrors the existing
    `not floor_polygon or len(floor_polygon) < 3` guard used elsewhere in
    this module/`bootstrap.py` for the same input). Otherwise always returns
    a dict - there is no "no clear dominant direction" failure mode here the
    way there was for the histogram method:
      - `long_axis_angle_deg`: `oriented_min_area_rect`'s angle (the long
        side's angle from world +X), in degrees, in [-90, 90).
      - `correction_deg` / `correction_rad`: the rotation to apply (about
        whatever center the caller chooses - this function is agnostic to
        translation, it only measures direction) to bring the long axis onto
        +X, i.e. exactly `-long_axis_angle_deg` - forced to 0.0 in the
        degenerate-square case.
      - `is_degenerate_square`: True if the bounding rectangle is
        (near-)square per `square_tolerance`.
      - `rect_size_m`: `(length, width)` of the bounding rectangle
        (length >= width), for provenance/logging.
      - `square_tolerance`: echoed back for provenance/logging.

    Deterministic by construction: `oriented_min_area_rect` is a closed-form
    rotating-calipers computation over the (already-deterministic) floor
    polygon vertices - no randomness, no confidence threshold to tip either
    way on borderline input.
    """
    if not floor_polygon or len(floor_polygon) < 3:
        return None

    points = np.asarray(floor_polygon, dtype=float)
    _center, (length, width), angle_rad = oriented_min_area_rect(points)

    is_degenerate_square = length <= 1e-12 or ((length - width) / length) < square_tolerance
    correction_rad = 0.0 if is_degenerate_square else -angle_rad

    return {
        "long_axis_angle_deg": math.degrees(angle_rad),
        "correction_deg": math.degrees(correction_rad),
        "correction_rad": correction_rad,
        "is_degenerate_square": is_degenerate_square,
        "rect_size_m": (length, width),
        "square_tolerance": square_tolerance,
    }


def compute_wall_hull_yaw(
    walls: list[WallPolygon],
    *,
    square_tolerance: float = 0.02,
) -> dict | None:
    """Independent cross-check for `compute_room_polygon_yaw`/
    `extract_free_only_room_polygon` (see `bootstrap.run_bootstrap`'s yaw
    step): pools every wall polygon's own vertices (not the traced
    free-space boundary), takes their 2D convex hull, and fits the same
    min-area bounding rectangle to *that*. Reuses `compute_room_polygon_yaw`
    itself rather than re-implementing the degenerate-square handling - a
    convex hull is just another polygon as far as that function's contract is
    concerned (`floor_polygon` was never required to be `extract_floor_polygon`'s
    exact output, only >=3 vertices).

    This is deliberately a *different* signal from the free-only room
    polygon: it never touches the occupancy grid at all, so it can't be
    dragged out to the grid array's own extent the way the original
    FREE+UNKNOWN bug was - but it has its own failure mode (a wall
    component's rasterization noise, or a spurious small dropped-then-kept
    component, can skew the hull). Agreement between the two independent
    methods is the actual confidence signal; see `bootstrap.run_bootstrap`
    for the >10 degree disagreement fallback that uses this method's angle
    instead of the free-only-polygon method's when they diverge.

    Returns None if there are no walls or fewer than 3 total vertices pooled
    across them (mirrors `compute_room_polygon_yaw`'s own guard)."""
    all_verts = [v for w in walls for v in w.vertices]
    if len(all_verts) < 3:
        return None
    hull_points = convex_hull_2d(np.asarray(all_verts, dtype=float))
    if len(hull_points) < 3:
        return None
    return compute_room_polygon_yaw(hull_points.tolist(), square_tolerance=square_tolerance)


def convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """Ordered 2D convex-hull boundary of a point set, in world coordinates.

    Stage E0 (SPEC §5, §8) needs this to build the *collision* prim from the
    object's actual measured hull - distinct from the oriented-rectangle
    footprint used for the *visual* placeholder mesh. Degenerate inputs (<3
    unique points, or all-collinear) fall back to the point set itself so
    callers can still build a degenerate prism rather than crashing.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    unique = np.unique(points_xy, axis=0)
    if len(unique) < 3:
        return unique
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(unique)
    except QhullError:
        return unique
    return unique[hull.vertices]


def rotate_point_xz(x: float, z: float, angle_rad: float, center_xz: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Rotate a world XZ point by `angle_rad` (standard CCW convention - same
    sign as every other angle in this module, e.g. `oriented_min_area_rect`'s
    `angle_rad` and the corner construction in `bootstrap._footprint_polygon`)
    about `center_xz`."""
    cx, cz = center_xz
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx, dz = x - cx, z - cz
    return cx + dx * cos_a - dz * sin_a, cz + dx * sin_a + dz * cos_a


def rotate_wall_polygon(wall: WallPolygon, angle_rad: float, center_xz: tuple[float, float]) -> WallPolygon:
    """Rigid-rotate a wall polygon's vertices about `center_xz`. `area_m2` is
    rotation-invariant, so it's carried over unchanged rather than
    recomputed."""
    rotated = [rotate_point_xz(x, z, angle_rad, center_xz) for x, z in wall.vertices]
    return WallPolygon(vertices=rotated, area_m2=wall.area_m2)


def compute_dominant_wall_yaw(
    walls: list[WallPolygon],
    *,
    bin_width_deg: float = 2.0,
    min_peak_fraction: float = 0.6,
) -> dict | None:
    """Yaw-normalization post-step (see docs/DECISIONS.md and
    var/scratch/STATE.md "Open questions"): the GPU pipeline never estimates
    the room's yaw about the vertical axis, so MSA's occupancy grid/walls come
    in at whatever arbitrary rotation MapAnything's raw inference happened to
    produce that run. This finds the room's true dominant wall direction from
    `extract_wall_polygons`'s output, so the caller can rotate the whole scene
    to axis-align it as a deterministic post-step - no GPU rerun needed.

    Method: histogram every wall-polygon edge's direction, mod 90 degrees.
    Walls are (almost always) rectilinear, so a wall running at angle theta is
    indistinguishable in "squareness" from one at theta+90/180/270 - folding
    the full 360 degrees down to [0, 90) merges a room's two perpendicular
    wall families into one signal instead of splitting it into two
    competing peaks. Each edge is weighted by its segment length (in metres),
    so a handful of long structural walls dominate a much larger number of
    short jagged stair-step segments the 5cm-grid rasterization can produce
    along a wall that's already close to axis-aligned.

    Bin width: `bin_width_deg` (default 2 degrees) - fine enough to resolve a
    real few-degree yaw error, coarse enough that the stair-step segments
    along one real wall (which cluster tightly around that wall's true angle
    after `simplify_polygon_preserving_passages`'s Douglas-Peucker pass, not
    spread randomly) still land in a single bin instead of being split by
    quantization noise. Bins are centered on multiples of `bin_width_deg`
    (bin 0 spans [-bin_width_deg/2, +bin_width_deg/2) mod 90) rather than
    starting at 0, so a scene that's already axis-aligned isn't artificially
    split across the 0-degree edge between two adjacent bins.

    Confidence gate: if the peak bin's weighted length share of the total is
    below `min_peak_fraction` (default 60%), there is no single dominant wall
    direction - a curved or highly irregular room (SPEC example: a regular
    octagon's 8 edges collapse, mod 90, into exactly two equally-weighted 50%
    families - neither ever reaches 60%). Returns None in that case, meaning
    "do not normalize this scene" - an expected, legitimate outcome, not an
    error.

    Returns None, or a dict:
      - `dominant_angle_deg`: the peak bin's length-weighted mean edge angle,
        mod 90, in [0, 90).
      - `correction_deg` / `correction_rad`: the rotation to apply (about
        whatever center the caller chooses - this function is agnostic to
        translation, it only measures direction) to bring
        `dominant_angle_deg` to the nearest axis - wraps `dominant_angle_deg`
        into (-45, 45] and negates it. A rectilinear room's two axes (0 and
        90 degrees) are the same "axis-aligned" outcome, so there's no
        separate 90-degree case.
      - `peak_share`: the winning bin's weighted fraction of total edge
        length, in [`min_peak_fraction`, 1.0].
      - `bin_width_deg`: echoed back for provenance/logging.

    Deterministic by construction: pure closed-form arithmetic over the
    (already-deterministic) wall polygons - no randomness, no dependence on
    input ordering (each edge's bin index is a closed-form function of its
    own angle, and ties in `weights` are broken by the first-seen index via
    `max`, which is itself a fixed function of the deterministic input
    order - not that ties are expected in practice with real metric wall
    lengths)."""
    segments: list[tuple[float, float]] = []  # (angle_deg mod 90, length_m)
    for wall in walls:
        verts = wall.vertices
        if len(verts) < 2:
            continue
        ring = verts if verts[0] == verts[-1] else [*verts, verts[0]]
        for (x0, z0), (x1, z1) in zip(ring[:-1], ring[1:]):
            dx, dz = x1 - x0, z1 - z0
            length = math.hypot(dx, dz)
            if length < 1e-9:
                continue
            # Rounded to 6 decimal places: two edges that are mathematically
            # identical directions (e.g. a rectangle's two long sides) can
            # differ by ~1e-14 degrees of float noise depending on which
            # vertex order/atan2 path computed them, which - unrounded - can
            # push them across a bin boundary from each other purely by
            # floating-point luck. Real-world angle differences this fix
            # could ever matter for are many orders of magnitude larger than
            # the rounding step, so this doesn't change real behavior.
            angle_deg = round(math.degrees(math.atan2(dz, dx)) % 90.0, 6)
            segments.append((angle_deg, length))

    if not segments:
        return None

    n_bins = max(1, round(90.0 / bin_width_deg))
    weights = [0.0] * n_bins
    members: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for angle_deg, length in segments:
        idx = int(math.floor((angle_deg + bin_width_deg / 2.0) / bin_width_deg)) % n_bins
        weights[idx] += length
        members[idx].append((angle_deg, length))

    total = sum(weights)
    if total <= 0:
        return None
    peak_idx = max(range(n_bins), key=lambda i: weights[i])
    peak_share = weights[peak_idx] / total
    if peak_share < min_peak_fraction:
        return None

    bin_center = (peak_idx * bin_width_deg) % 90.0
    weighted_sum = 0.0
    weight_total = 0.0
    for angle_deg, length in members[peak_idx]:
        offset = ((angle_deg - bin_center + 45.0) % 90.0) - 45.0  # unwrap near bin_center, mod-90 safe
        weighted_sum += (bin_center + offset) * length
        weight_total += length
    dominant_angle_deg = (weighted_sum / weight_total) % 90.0

    wrapped = ((dominant_angle_deg + 45.0) % 90.0) - 45.0
    correction_deg = -wrapped
    return {
        "dominant_angle_deg": dominant_angle_deg,
        "correction_deg": correction_deg,
        "correction_rad": math.radians(correction_deg),
        "peak_share": peak_share,
        "bin_width_deg": bin_width_deg,
    }
