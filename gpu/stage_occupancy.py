#!/usr/bin/env python3
"""Stage 6: 2D top-down occupancy grid from the aligned point cloud.

Runs under the mapanything venv (has open3d + matplotlib + numpy together). No script
producing this ever existed before this project's work: the original
occupancy_2d_grid.npz/.png were made by a one-off inline heredoc that was never saved.
The exact logic below was reverse-engineered and bit-verified against that saved
output (histogram2d over X/Z, 5cm cells, [0.1, 0.5]m inclusive height band for the
"occupied" grid) - reproduced faithfully here, with the one real change: the original's
hardcoded room-crop box (`X,Z in [-3.2, 0.7]`, specific to one demo room) is replaced
with a data-derived percentile trim on this run's own point cloud.

Only ever run on `aligned_room.ply` (stage 4's output) - the [0.1, 0.5]m height band is
meaningless on anything not already floor-aligned to Y=0.

Important limitation, not swept under the rug: this is a point-density histogram, NOT a
free/occupied/unknown grid with proper ray-based free-space carving. A cell with zero
points could be genuinely free space or simply never observed - occupancy.npy encodes
that ambiguity explicitly as a third "unknown" state rather than silently guessing.
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import OCCUPANCY_GRID_RESOLUTION_M, job_paths, load_params  # noqa: E402
from occupancy_classify import classify_phantom_obstacles  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

FREE, OBSTACLE, UNKNOWN = 0, 1, 2

DEFAULT_CELL = OCCUPANCY_GRID_RESOLUTION_M
DEFAULT_BAND_MIN = 0.1
DEFAULT_BAND_MAX = 0.5
DEFAULT_EXTENT_PERCENTILE = 0.5  # trims [p, 100-p] on X and Z; 0 disables
DEFAULT_MIN_POINTS_PER_CELL = 3


def _load_object_bboxes_xz(paths: dict) -> list[tuple[float, float, float, float]]:
    """(bbox_min_x, bbox_min_z, bbox_max_x, bbox_max_z) for every object in
    scene_objects.json (stage_objects.py, which runs before this stage - see
    docs/DECISIONS.md's stage-order entry), used to protect a real small object from
    occupancy_classify.filter_small_obstacle_clusters. All objects count as "real,
    scene-exported" here regardless of `likely_fragment` - that flag is informational
    (see job_io.finalize_result's solid_object_count), not an exclusion from the export
    itself. Returns [] (no protection, rule (a) alone still runs) if the file is
    missing - stage_objects.py can legitimately find zero objects."""
    path = paths["scene_objects_json"]
    if not path.exists():
        return []
    objects = json.loads(path.read_text())
    bboxes = []
    for o in objects:
        bbox_min, bbox_max = o["bbox_min"], o["bbox_max"]
        bboxes.append((float(bbox_min[0]), float(bbox_min[2]), float(bbox_max[0]), float(bbox_max[2])))
    return bboxes


def _load_view_count_grid(
    paths: dict,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    band_min: float,
    band_max: float,
) -> np.ndarray | None:
    """Per-cell count of distinct per_view/*.npz views with >=1 point landing in that
    cell within the occupancy height band, for occupancy_classify's rule (b). None if
    no per_view data survived this job (see stage_infer.py's per_view write and
    app/config.py's retain_per_view_artifacts - off by default, so most scenes today
    have none) - the caller then falls back to rule (a) alone, per this task's spec.

    Each view's raw pts3d/mask is transformed into the SAME aligned frame
    aligned_room.ply is in (p_aligned = R @ p_raw - [0, floor_y, 0], read from
    alignment_transform.npz - see stage_align.py's own comment deriving this from its
    R/floor_y). Deliberately does NOT re-apply stage_align's room-bbox/confidence/SOR
    filters per view (those need the full pooled cloud to compute percentile cutoffs
    against) - an approximation, fine for this purpose since multi-view support only
    needs "did >=2 views see roughly this spot", not point-perfect parity with the final
    filtered cloud. Points outside [x_edges[0], x_edges[-1]] x [z_edges[0], z_edges[-1]]
    are silently excluded by histogram2d, same as the main grid computation below."""
    view_files = sorted(glob.glob(str(paths["per_view_dir"] / "view_*.npz")))
    if not view_files or not paths["transform_npz"].exists():
        return None

    xform = np.load(paths["transform_npz"])
    R, floor_y = xform["R"], float(xform["floor_y"])

    nx, nz = len(x_edges) - 1, len(z_edges) - 1
    view_count_grid = np.zeros((nx, nz), dtype=np.int32)
    for f in view_files:
        d = np.load(f)
        pts3d, mask = d["pts3d"], d["mask"]
        pts = pts3d[mask]
        if len(pts) == 0:
            continue
        pts_aligned = pts @ R.T
        pts_aligned[:, 1] -= floor_y
        band_mask = (pts_aligned[:, 1] >= band_min) & (pts_aligned[:, 1] <= band_max)
        if not band_mask.any():
            continue
        view_hist, _, _ = np.histogram2d(
            pts_aligned[band_mask, 0], pts_aligned[band_mask, 2], bins=[x_edges, z_edges]
        )
        view_count_grid += view_hist > 0

    return view_count_grid


def main() -> None:
    job_dir = sys.argv[1]
    paths = job_paths(job_dir)
    params = load_params(job_dir)

    cell = float(params.get("occupancy_cell_m", DEFAULT_CELL))
    band_min = float(params.get("occupancy_band_min_m", DEFAULT_BAND_MIN))
    band_max = float(params.get("occupancy_band_max_m", DEFAULT_BAND_MAX))
    extent_pct = float(params.get("occupancy_extent_percentile", DEFAULT_EXTENT_PERCENTILE))
    min_pts_per_cell = int(params.get("occupancy_min_points_per_cell", DEFAULT_MIN_POINTS_PER_CELL))

    pcd = o3d.io.read_point_cloud(str(paths["aligned_ply"]))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError(f"{paths['aligned_ply']} has zero points - nothing to grid")
    print(f"loaded {len(pts)} points from {paths['aligned_ply'].name}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    if extent_pct > 0:
        # Data-derived replacement for the original's hardcoded room-crop box. Without
        # SOME trim, a sightline through an open doorway stretches the grid over empty
        # hallway/outdoor space (as seen in this project's own reference room); with a
        # light trim, a genuinely long room only loses its extreme 0.5% either way.
        x_lo, x_hi = np.percentile(x, [extent_pct, 100 - extent_pct])
        z_lo, z_hi = np.percentile(z, [extent_pct, 100 - extent_pct])
        keep = (x >= x_lo) & (x <= x_hi) & (z >= z_lo) & (z <= z_hi)
        n_before = len(x)
        x, y, z = x[keep], y[keep], z[keep]
        print(
            f"extent trim (percentile={extent_pct}): kept {len(x)}/{n_before} pts, "
            f"X[{x_lo:.2f},{x_hi:.2f}] Z[{z_lo:.2f},{z_hi:.2f}]"
        )

    x_edges = np.arange(x.min() - cell, x.max() + cell, cell)
    z_edges = np.arange(z.min() - cell, z.max() + cell, cell)

    band_mask = (y >= band_min) & (y <= band_max)
    occ_grid, _, _ = np.histogram2d(x[band_mask], z[band_mask], bins=[x_edges, z_edges])
    full_grid, _, _ = np.histogram2d(x, z, bins=[x_edges, z_edges])

    nx, nz = occ_grid.shape
    print(
        f"grid: {nx}x{nz} cells @ {cell}m, band [{band_min},{band_max}]m -> "
        f"{int((occ_grid > 0).sum())}/{occ_grid.size} band-occupied cells, "
        f"{int((full_grid > 0).sum())}/{full_grid.size} ever-observed cells"
    )

    # occupancy.npy: [ix, iz] axis order (axis 0 = X, axis 1 = Z) - verified against the
    # reference artifact via exact recomputation, NOT transposed. Getting this backwards
    # is exactly the kind of coordinate-frame bug this project has hit repeatedly.
    grid = np.full((nx, nz), UNKNOWN, dtype=np.uint8)
    grid[full_grid > 0] = FREE
    grid[occ_grid >= min_pts_per_cell] = OBSTACLE
    n_obstacle_raw = int((grid == OBSTACLE).sum())

    # Phantom-obstacle post-filter (see occupancy_classify.py's module docstring): a
    # small OBSTACLE cluster with no matching real object is reconstruction noise, not
    # geometry - and where per_view/*.npz survived this job, a cell backed by only one
    # view is the same shape of noise even if it clears min_points_per_cell. Both rules
    # only ever demote OBSTACLE -> UNKNOWN.
    object_bboxes_xz = _load_object_bboxes_xz(paths)
    view_count_grid = _load_view_count_grid(paths, x_edges, z_edges, band_min=band_min, band_max=band_max)
    phantom_filter_enabled = params.get("phantom_obstacle_filter_enabled", True)
    if phantom_filter_enabled:
        classification = classify_phantom_obstacles(
            grid, x_edges, z_edges, object_bboxes_xz=object_bboxes_xz, view_count_grid=view_count_grid
        )
        grid = classification["grid"]
        print(
            f"phantom-obstacle filter: view_support={'used' if classification['used_view_filter'] else 'no per_view data - rule (a) fallback'}, "
            f"demoted by view support: {classification['n_cells_demoted_by_view_support']}, "
            f"demoted by cluster size: {classification['n_cells_demoted_by_cluster_size']}, "
            f"{n_obstacle_raw} -> {int((grid == OBSTACLE).sum())} obstacle cells"
        )
    else:
        classification = {
            "used_view_filter": False,
            "n_cells_demoted_by_view_support": 0,
            "n_cells_demoted_by_cluster_size": 0,
            "min_cluster_cells": None,
            "min_views_for_obstacle": None,
        }
        print("phantom-obstacle filter: disabled via params.json (phantom_obstacle_filter_enabled=false)")

    np.save(paths["occupancy_npy"], grid)

    np.savez(
        paths["occupancy_npz"], occ_grid=occ_grid, full_grid=full_grid, x_edges=x_edges, z_edges=z_edges
    )

    meta = {
        "resolution": cell,
        "origin_x": float(x_edges[0]),
        "origin_z": float(z_edges[0]),
        "width": int(nx),
        "height": int(nz),
        "band_min": band_min,
        "band_max": band_max,
        "min_points_per_cell": min_pts_per_cell,
        "extent_percentile": extent_pct,
        "n_free": int((grid == FREE).sum()),
        "n_obstacle": int((grid == OBSTACLE).sum()),
        "n_unknown": int((grid == UNKNOWN).sum()),
        "n_obstacle_before_phantom_filter": n_obstacle_raw,
        "phantom_obstacle_filter_enabled": phantom_filter_enabled,
        "phantom_obstacle_used_view_filter": classification["used_view_filter"],
        "phantom_obstacle_cells_demoted_by_view_support": classification["n_cells_demoted_by_view_support"],
        "phantom_obstacle_cells_demoted_by_cluster_size": classification["n_cells_demoted_by_cluster_size"],
    }
    paths["occupancy_meta"].write_text(json.dumps(meta, indent=2))

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(
        (grid.T == OBSTACLE),
        origin="lower",
        cmap="gray_r",
        extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
    )
    ax1.set_title(
        f"Top-down occupancy (point density, {band_min}-{band_max}m band)\n"
        f"{cell * 100:.0f}cm cells, {meta['n_obstacle']}/{nx * nz} occupied "
        f"(raw density: {n_obstacle_raw}, phantom-filtered)"
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(
        np.log1p(full_grid.T),
        origin="lower",
        cmap="viridis",
        extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
    )
    ax2.set_title("Top-down point density (all heights, log scale)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")

    plt.tight_layout()
    plt.savefig(paths["map_preview"], dpi=110)
    plt.close(fig)

    print(f"DONE: {paths['occupancy_npy'].name}, {paths['occupancy_npz'].name}, {paths['map_preview'].name}")


if __name__ == "__main__":
    main()
