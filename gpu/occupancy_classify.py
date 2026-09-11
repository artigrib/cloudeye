#!/usr/bin/env python3
"""Phantom-obstacle post-filter for stage_occupancy.py's tri-state grid.

Pure numpy, no open3d/matplotlib/torch - deliberately importable from both the
mapanything venv (stage_occupancy.py, which builds the raw grid from
`aligned_room.ply`) and the plain `uv run pytest` venv (this repo's test suite has no
open3d, see job_io.py's module docstring for why shared cross-venv code stays
stdlib/numpy-only).

Problem this fixes: `stage_occupancy.py` classifies a cell OBSTACLE purely from raw
point density in a fixed [0.1, 0.5]m height band (`occ_grid >= min_points_per_cell`,
default 3). Two failure modes produce OBSTACLE cells that are not real obstacles
("phantom obstacles"):

(a) Sensor/reconstruction noise (MapAnything fly-away points that survive stage_align's
    SOR filter, floor-band reflections past the below-floor clip's margin, a handful of
    points landing in the band from an object's edge) forms a small, spatially-isolated
    obstacle cluster with no matching entry in `scene_objects.json` - nothing a real
    object could be. Fixed by `filter_small_obstacle_clusters`: an 8-connected OBSTACLE
    component smaller than `MIN_CLUSTER_CELLS` cells is reclassified to UNKNOWN, unless
    it spatially overlaps a real, scene-exported object's own XZ footprint (a small
    genuine object - a lamp, a vase - must not be swept away just for being small).

(b) A single view's own depth noise (one frame's misestimated depth at an edge, a
    reflection only that one camera angle catches) can, on its own, clear
    `min_points_per_cell` in one cell despite being backed by no other viewpoint at all.
    Fixed by `filter_by_view_support`: when `per_view/*.npz` data exists for this job
    (see stage_infer.py/stage_align.py), an OBSTACLE cell is kept only if points from at
    least `MIN_VIEWS_FOR_OBSTACLE` distinct views land in it (within the same height
    band the base grid used) - a real obstacle should be visible from more than one
    walkthrough frame; single-view support is exactly the phantom-obstacle shape. When
    no per_view data survived the job (the common case today: `retain_per_view_artifacts`
    defaults to False - see app/config.py - so most scenes have none), this rule cannot
    run at all and `classify_phantom_obstacles` falls back to (a) alone.

Both rules only ever demote OBSTACLE -> UNKNOWN, never touch FREE cells, and never
promote anything - a cell already reported as ambiguous (never directly observed) isn't
this module's problem to solve, and a cell the base grid found genuinely empty
(FREE, ever-observed with zero points in the band) has no obstacle claim to weaken in
the first place. Downstream (`app/services/pathfinding.py`), UNKNOWN is traversable at a
penalty rather than blocked outright - see build_cost_grid's docstring - so a
reclassified phantom stops sealing off real, walkable space instead of becoming an
equally-wrong "definitely free" claim.
"""

from __future__ import annotations

from collections import deque

import numpy as np

FREE, OBSTACLE, UNKNOWN = 0, 1, 2

# An OBSTACLE connected component this size or smaller, with no real object's footprint
# overlapping any of its cells, is treated as sensor noise rather than a genuine small
# obstacle - see module docstring's (a). Chosen as "a handful of cells", not "a single
# cell": a lone stray point can clear min_points_per_cell in one cell by chance, but a
# multi-cell cluster is a stronger noise signal worth still catching (a real small
# object is instead protected by the object-footprint check, not by raising this past 1).
MIN_CLUSTER_CELLS = 4

# An OBSTACLE cell must be backed by points from at least this many distinct
# keyframes/views to survive rule (b) - see module docstring. 2, not 1: the whole point
# is "more than a single view's own noise attests to this".
MIN_VIEWS_FOR_OBSTACLE = 2

_NEIGHBORS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Every 8-connected component of True cells in a 2D boolean mask, as lists of
    (ix, iz) cell coordinates. Same 8-connectivity `app/services/pathfinding.py` uses
    for its own connectivity checks, kept as an independent implementation here since
    gpu/ stage scripts must not import the `app` package (they run on a separate GPU
    host with no backend installed - see job_io.py's module docstring)."""
    width, height = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for ix in range(width):
        for iz in range(height):
            if not mask[ix, iz] or seen[ix, iz]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(ix, iz)])
            seen[ix, iz] = True
            while queue:
                cx, cz = queue.popleft()
                component.append((cx, cz))
                for dx, dz in _NEIGHBORS8:
                    nx, nz = cx + dx, cz + dz
                    if 0 <= nx < width and 0 <= nz < height and mask[nx, nz] and not seen[nx, nz]:
                        seen[nx, nz] = True
                        queue.append((nx, nz))
            components.append(component)
    return components


def object_footprint_mask(
    object_bboxes_xz: list[tuple[float, float, float, float]],
    x_edges: np.ndarray,
    z_edges: np.ndarray,
) -> np.ndarray:
    """Boolean grid (width, height) marking every cell whose world-space extent
    overlaps at least one real, scene-exported object's XZ bounding box
    (`(bbox_min_x, bbox_min_z, bbox_max_x, bbox_max_z)`, aligned-frame meters - the same
    fields `scene_objects.json` carries per object, at indices 0/2 of its 3D bbox_min/
    bbox_max). Used to protect a genuinely small real object from rule (a)'s cluster-
    size filter - a cell overlapping a real object's footprint is never reclassified,
    regardless of how small its own connected component is."""
    nx, nz = len(x_edges) - 1, len(z_edges) - 1
    mask = np.zeros((nx, nz), dtype=bool)
    if not object_bboxes_xz:
        return mask
    for bbox_min_x, bbox_min_z, bbox_max_x, bbox_max_z in object_bboxes_xz:
        ix_lo = np.searchsorted(x_edges, bbox_min_x, side="right") - 1
        ix_hi = np.searchsorted(x_edges, bbox_max_x, side="left")
        iz_lo = np.searchsorted(z_edges, bbox_min_z, side="right") - 1
        iz_hi = np.searchsorted(z_edges, bbox_max_z, side="left")
        ix_lo = max(0, min(ix_lo, nx - 1))
        iz_lo = max(0, min(iz_lo, nz - 1))
        ix_hi = max(0, min(ix_hi, nx - 1))
        iz_hi = max(0, min(iz_hi, nz - 1))
        if ix_hi < ix_lo or iz_hi < iz_lo:
            continue
        mask[ix_lo : ix_hi + 1, iz_lo : iz_hi + 1] = True
    return mask


def filter_small_obstacle_clusters(
    grid: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    object_bboxes_xz: list[tuple[float, float, float, float]] | None = None,
    *,
    min_cluster_cells: int = MIN_CLUSTER_CELLS,
) -> tuple[np.ndarray, int]:
    """Rule (a): reclassify to UNKNOWN every OBSTACLE connected component smaller than
    `min_cluster_cells` cells that does not overlap any real object's footprint (see
    `object_footprint_mask`). Returns (new_grid, n_cells_reclassified) - never mutates
    `grid` in place."""
    protected = object_footprint_mask(object_bboxes_xz or [], x_edges, z_edges)
    out = grid.copy()
    obstacle_mask = grid == OBSTACLE
    n_reclassified = 0
    for component in connected_components(obstacle_mask):
        if len(component) >= min_cluster_cells:
            continue
        if any(protected[cell] for cell in component):
            continue
        for ix, iz in component:
            out[ix, iz] = UNKNOWN
        n_reclassified += len(component)
    return out, n_reclassified


def filter_by_view_support(
    grid: np.ndarray,
    view_count_grid: np.ndarray,
    *,
    min_views: int = MIN_VIEWS_FOR_OBSTACLE,
) -> tuple[np.ndarray, int]:
    """Rule (b): reclassify to UNKNOWN every OBSTACLE cell backed by fewer than
    `min_views` distinct keyframes/views. `view_count_grid` is an int array, same shape
    as `grid`, giving the number of distinct source views with at least one point
    landing in that cell within the occupancy band (see stage_occupancy.py's
    `_per_view_count_grid`). Cells that are not OBSTACLE are left untouched regardless
    of their view count. Returns (new_grid, n_cells_reclassified)."""
    if view_count_grid.shape != grid.shape:
        raise ValueError(
            f"view_count_grid shape {view_count_grid.shape} != grid shape {grid.shape}"
        )
    out = grid.copy()
    demote_mask = (grid == OBSTACLE) & (view_count_grid < min_views)
    out[demote_mask] = UNKNOWN
    return out, int(demote_mask.sum())


def classify_phantom_obstacles(
    grid: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    object_bboxes_xz: list[tuple[float, float, float, float]] | None = None,
    view_count_grid: np.ndarray | None = None,
    min_cluster_cells: int = MIN_CLUSTER_CELLS,
    min_views: int = MIN_VIEWS_FOR_OBSTACLE,
) -> dict:
    """Top-level entry point stage_occupancy.py calls after building the raw
    FREE/OBSTACLE/UNKNOWN grid.

    - `view_count_grid` given (per_view data existed for this job): rule (b) runs
      first - single-view-backed OBSTACLE cells are demoted to UNKNOWN - then rule (a)'s
      cluster-size filter runs on the RESULT, so a real obstacle that rule (b) partially
      thins out (e.g. a real object seen from 3 views but with one edge cell only ever
      caught by 1) doesn't leave a now-tiny, disconnected orphan fragment behind either.
    - `view_count_grid` is None (no per_view data - the common case today, see module
      docstring's (b)): rule (b) is skipped entirely and rule (a) is the fallback,
      exactly as specified.

    Returns a dict with the final `grid`, which rule(s) actually ran, and how many
    cells each one reclassified - written into occupancy_meta.json's phantom_obstacle_*
    fields so a scene's classification provenance is inspectable after the fact."""
    working = grid
    n_view_filtered = 0
    used_view_filter = view_count_grid is not None
    if used_view_filter:
        working, n_view_filtered = filter_by_view_support(working, view_count_grid, min_views=min_views)

    working, n_cluster_filtered = filter_small_obstacle_clusters(
        working, x_edges, z_edges, object_bboxes_xz, min_cluster_cells=min_cluster_cells
    )

    return {
        "grid": working,
        "used_view_filter": used_view_filter,
        "n_cells_demoted_by_view_support": n_view_filtered,
        "n_cells_demoted_by_cluster_size": n_cluster_filtered,
        "min_cluster_cells": min_cluster_cells,
        "min_views_for_obstacle": min_views if used_view_filter else None,
    }
