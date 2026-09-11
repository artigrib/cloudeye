"""Stage A5 (gaps.json): minimum passage widths between obstacle pairs (object-
object and object-wall), via a generalized-Voronoi ridge analysis over the free
region's distance transform. See var/scratch/msa/SPEC.md section A5.

Approach: label every obstacle source (walls + each object footprint) on a combined
grid, then for every free cell find its nearest obstacle label via
`scipy.ndimage.distance_transform_edt(..., return_indices=True)`. Where two
neighbouring free cells have different nearest labels, that boundary is a ridge of
the generalized Voronoi diagram between those two obstacles - the standard
approach for corridor/gap width estimation. The minimum, over all such ridge
cells for a given obstacle pair, of (distance to A + distance to B) is that pair's
passage width estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FREE, OBSTACLE, UNKNOWN = 0, 1, 2


@dataclass(frozen=True)
class ObstacleSource:
    """One obstacle contributing to the gap analysis: a wall component or an
    object footprint, rasterized onto the grid as a boolean mask."""

    id: str
    kind: str  # "wall" | "object"
    mask: np.ndarray  # same shape as the occupancy grid, True where this obstacle occupies a cell


@dataclass(frozen=True)
class PlatformVerdict:
    platform_id: str
    fits: bool
    required_clearance_m: float


@dataclass(frozen=True)
class Gap:
    a_id: str
    b_id: str
    a_kind: str
    b_kind: str
    width_m: float
    measurement_point_xy: tuple[float, float]
    platform_verdicts: list[PlatformVerdict]


def compute_gaps(
    passable_mask: np.ndarray,
    sources: list[ObstacleSource],
    resolution: float,
    origin_x: float,
    origin_z: float,
    *,
    platforms: dict[str, float] | None = None,
) -> list[Gap]:
    """`passable_mask` is True for cells a robot could ever occupy (FREE, and by
    convention also UNKNOWN unless the caller excludes it - SPEC treats gaps as
    computed "over the free region"). `sources` gives every obstacle whose pairwise
    clearance should be measured. `platforms` maps platform_id -> required
    diameter_m (circle clearance, per SPEC A5 "circle diameter now").
    """
    if not sources:
        return []

    label_grid = np.zeros(passable_mask.shape, dtype=np.int32)
    id_by_label: dict[int, tuple[str, str]] = {}
    for i, src in enumerate(sources, start=1):
        label_grid[src.mask] = i
        id_by_label[i] = (src.id, src.kind)

    obstacle_mask = label_grid > 0
    # EDT distance (in cells) from every passable cell to the nearest obstacle
    # cell, plus the index of that nearest obstacle cell.
    free_like = passable_mask & ~obstacle_mask
    dist_cells, nearest_idx = ndimage.distance_transform_edt(~obstacle_mask, return_indices=True)
    nearest_label = label_grid[nearest_idx[0], nearest_idx[1]]

    best: dict[tuple[int, int], tuple[float, tuple[int, int]]] = {}
    rows, cols = np.where(free_like)
    for r, c in zip(rows, cols):
        my_label = nearest_label[r, c]
        my_dist = dist_cells[r, c]
        if my_label == 0:
            continue
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if nr >= free_like.shape[0] or nc >= free_like.shape[1] or not free_like[nr, nc]:
                continue
            other_label = nearest_label[nr, nc]
            if other_label == 0 or other_label == my_label:
                continue
            other_dist = dist_cells[nr, nc]
            width_cells = my_dist + other_dist
            key = (min(my_label, other_label), max(my_label, other_label))
            candidate = (width_cells, (r, c))
            if key not in best or candidate[0] < best[key][0]:
                best[key] = candidate

    platforms = platforms or {}
    gaps: list[Gap] = []
    for (label_a, label_b), (width_cells, (r, c)) in best.items():
        width_m = float(width_cells) * resolution
        a_id, a_kind = id_by_label[label_a]
        b_id, b_kind = id_by_label[label_b]
        # Grid is [ix, iz] (axis 0 = X) - see geometry._cell_to_world.
        point_xy = (origin_x + (r + 0.5) * resolution, origin_z + (c + 0.5) * resolution)
        verdicts = [
            PlatformVerdict(platform_id=pid, fits=width_m >= diameter, required_clearance_m=diameter)
            for pid, diameter in platforms.items()
        ]
        gaps.append(
            Gap(
                a_id=a_id,
                b_id=b_id,
                a_kind=a_kind,
                b_kind=b_kind,
                width_m=width_m,
                measurement_point_xy=point_xy,
                platform_verdicts=verdicts,
            )
        )
    gaps.sort(key=lambda g: g.width_m)
    return gaps


def gaps_to_json(gaps: list[Gap]) -> list[dict]:
    return [
        {
            "a": g.a_id,
            "a_kind": g.a_kind,
            "b": g.b_id,
            "b_kind": g.b_kind,
            "width_m": round(g.width_m, 4),
            "measurement_point_xy": [round(g.measurement_point_xy[0], 4), round(g.measurement_point_xy[1], 4)],
            "platforms": [
                {"platform_id": v.platform_id, "fits": v.fits, "required_clearance_m": v.required_clearance_m}
                for v in g.platform_verdicts
            ],
        }
        for g in gaps
    ]
