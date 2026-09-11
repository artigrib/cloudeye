#!/usr/bin/env python3
"""Give the scenes that predate the band layer a band layer, from their own point cloud.

Removing the old nav2d source from the router orphans every scene processed before it: 40
`done` rows, each with a stage_occupancy `occupancy.npy` (a 0.1-0.5 m point-density
histogram) and no nvblox mesh, so `band_mask.py` cannot be pointed at them. Rather than
leave the router reading two layers - the thing §9.5 exists to stop - each old scene gets
the SAME 0.1-1.5 m band rule applied to the artifact it does have: `aligned_room.ply`,
stage 4's floor-aligned point cloud.

Two differences from band_mask.py, both forced by the input and both recorded in the
metadata rather than glossed:

  * points, not mesh vertices. There is no mesh for these scenes. A point cloud rasterises
    the same way - a cell is an obstacle if any point stands in the band - but it has no
    connected components, so `min_component_tri` has no analogue. `--min-points-per-cell`
    takes its place: a cell backed by fewer than that many points in the band is noise, the
    same rule stage_occupancy already used (its default was 3).
  * the frame is the pipeline's aligned frame (Y up, floor at y = 0, axes world X and Z),
    not a fitted floor plane's (u, v). So the grid keeps the scene's EXISTING GridMeta -
    same resolution, origin and shape - and every world coordinate the app has already
    persisted stays valid. Nothing is re-registered.

UNOBSERVED IS STILL NEVER FREE, and here it has to be derived rather than read: a point
cloud carries no observed/unobserved mask. A cell with no points at all is UNKNOWN, exactly
as stage_occupancy treated it - that ambiguity is why occupancy.npy has always been
tri-state - so the rule is unchanged and only the band moved.

`--dry-run` is the DEFAULT and there is no flag to remember: with no arguments this prints
every scene, the cell counts before and after, and writes nothing.

    python3 pipeline/steps/layers/backfill_occupancy.py            # list, write nothing
    python3 pipeline/steps/layers/backfill_occupancy.py --apply    # write the layers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FREE, OBSTACLE, UNKNOWN = 0, 1, 2
DEFAULT_BAND = (0.1, 1.5)
DEFAULT_MIN_POINTS = 3
#: Same two constants band_mask.py uses, and for the same reason - see its docstring.
DEFAULT_MARGIN = 0.05
DEFAULT_ROBOT_HEIGHT_M = 0.192

#: Written into occupancy_meta.json so a reader can tell a backfilled layer from one
#: band_mask.py produced, and never has to guess which rule made a grid.
SOURCE = "pipeline/steps/layers/backfill_occupancy.py"


def band_from_cloud(ply: Path, meta: dict, band=DEFAULT_BAND,
                    min_points: int = DEFAULT_MIN_POINTS):
    """Tri-state grid on the scene's existing GridMeta, from its aligned point cloud.

    Returns (cells, obstacle_mask, unknown_mask, counts).
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply))
    p = np.asarray(pcd.points)
    if not len(p):
        raise SystemExit(f"backfill: {ply} has no points (Open3D returns an EMPTY cloud "
                         f"for a file it cannot open - check the path)")
    res = float(meta["resolution"])
    ox, oz = float(meta["origin_x"]), float(meta["origin_z"])
    w, h = int(meta["width"]), int(meta["height"])

    ix = np.floor((p[:, 0] - ox) / res).astype(int)
    iz = np.floor((p[:, 2] - oz) / res).astype(int)
    inside = (ix >= 0) & (ix < w) & (iz >= 0) & (iz < h)
    y = p[:, 1]

    # Any point at all -> the cell was observed. Points in the band -> obstacle.
    seen = np.zeros((w, h), dtype=np.int32)
    np.add.at(seen, (ix[inside], iz[inside]), 1)
    in_band = inside & (y >= band[0]) & (y <= band[1])
    hits = np.zeros((w, h), dtype=np.int32)
    np.add.at(hits, (ix[in_band], iz[in_band]), 1)

    obstacle = hits >= min_points
    # 1..min_points-1 band points is drift, not furniture. "Any point at all = obstacle"
    # promotes reconstruction noise to a wall, and a wall is INFLATED by the robot radius:
    # measured on hero-74, 534 such cells dilated at Go2's 0.2496 m sealed a corridor a
    # 0.1 m robot walked straight down. UNKNOWN is the state that exists for this - we
    # cannot claim a wall from two points, and we cannot claim clear floor either.
    thin = (hits > 0) & ~obstacle
    unknown = (seen == 0) | thin

    # The LOWEST band surface per cell - the layer's source of truth, thresholded per robot
    # rather than fixed at 1.5 m. Only cells that clear min_points get a height: a height
    # read off one or two stray points is noise, and it would block a robot.
    lowest = np.full((w, h), np.inf)
    np.minimum.at(lowest, (ix[in_band], iz[in_band]), y[in_band])
    # A thin cell keeps no height: thresholding drift against a robot's roof would decide
    # whether to trust two points by how tall the robot is.
    lowest[~obstacle] = np.nan
    lowest[~np.isfinite(lowest)] = np.nan

    cells = np.full((w, h), FREE, dtype=np.uint8)
    cells[obstacle & (lowest <= DEFAULT_ROBOT_HEIGHT_M + DEFAULT_MARGIN)] = OBSTACLE
    # Applied LAST, same order and same reason as band_mask.tri_state: a cell nobody
    # looked at is never free, and never claimed to be an obstacle either.
    cells[unknown] = UNKNOWN
    return cells, obstacle, unknown, lowest.astype(np.float32), {
        "points_total": int(len(p)), "points_in_grid": int(inside.sum()),
        "points_in_band": int(in_band.sum()),
        "thin_cells_demoted_to_unknown": int(thin.sum())}


def counts(cells: np.ndarray) -> dict:
    return {"free": int((cells == FREE).sum()),
            "obstacle": int((cells == OBSTACLE).sum()),
            "unknown": int((cells == UNKNOWN).sum())}


def backfill_one(scene_dir: Path, *, band=DEFAULT_BAND, min_points=DEFAULT_MIN_POINTS,
                 apply: bool = False) -> dict:
    meta_path = scene_dir / "occupancy_meta.json"
    ply = scene_dir / "aligned_room.ply"
    row = {"scene_dir": str(scene_dir), "written": False}
    if not meta_path.exists() or not ply.exists():
        row["skipped"] = ("no occupancy_meta.json" if not meta_path.exists()
                          else "no aligned_room.ply")
        return row
    meta = json.loads(meta_path.read_text())
    old = np.load(scene_dir / "occupancy.npy") if (scene_dir / "occupancy.npy").exists() \
        else None
    cells, obstacle, unknown, lowest, pc = band_from_cloud(ply, meta, band, min_points)

    row["grid"] = [int(meta["width"]), int(meta["height"])]
    row["before"] = counts(old) if old is not None else None
    row["before_band_m"] = [meta.get("band_min"), meta.get("band_max")]
    row["after"] = counts(cells)
    row["after_band_m"] = list(band)
    row.update(pc)
    row["min_points_per_cell"] = int(min_points)

    if not apply:
        return row

    out = scene_dir / "layers" / "backfill"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "occupancy.npy", cells)
    np.save(out / "obstacle_min_height.npy", lowest)
    np.save(out / "obstacle_mask.npy", obstacle)
    np.save(out / "unobserved_mask.npy", unknown)
    new_meta = dict(meta)
    new_meta.update({
        "resolution": float(meta["resolution"]),
        "origin_x": float(meta["origin_x"]), "origin_z": float(meta["origin_z"]),
        "width": int(meta["width"]), "height": int(meta["height"]),
        "source": SOURCE, "band_min": float(band[0]), "band_max": float(band[1]),
        "min_points_per_cell": int(min_points),
        "thin_cells_demoted_to_unknown": pc["thin_cells_demoted_to_unknown"],
        "layer_kind": "obstacle_min_height",
        "height_margin_m": DEFAULT_MARGIN,
        "default_robot_height_m": DEFAULT_ROBOT_HEIGHT_M,
        "frame": "pipeline aligned frame (Y up, floor y = 0); axes are world X and Z",
        "backfilled_from": str(ply),
        "n_free": row["after"]["free"], "n_obstacle": row["after"]["obstacle"],
        "n_unknown": row["after"]["unknown"],
    })
    (out / "occupancy_meta.json").write_text(json.dumps(new_meta, indent=2))
    row["written"] = True
    row["out"] = str(out)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes-root", default=None,
                    help="directory of <scene_id>/ dirs; default is the app's upload_dir")
    ap.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND))
    ap.add_argument("--min-points-per-cell", type=int, default=DEFAULT_MIN_POINTS)
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without it this is a dry run - the default, "
                         "because rewriting the layer 40 live scenes are served from is "
                         "the owner's call, not this script's.")
    a = ap.parse_args()

    root = Path(a.scenes_root) if a.scenes_root else None
    if root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from app.config import settings
        root = Path(settings.upload_dir) / "scenes"

    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    print(f"scenes root: {root}  ({len(dirs)} scene dirs)")
    done = skipped = 0
    for d in dirs:
        row = backfill_one(d, band=tuple(a.band), min_points=a.min_points_per_cell,
                           apply=a.apply)
        if "skipped" in row:
            skipped += 1
            continue
        done += 1
        b, af = row["before"], row["after"]
        print(f"  {d.name}  grid {row['grid'][0]}x{row['grid'][1]}  "
              f"band {row['before_band_m']} -> {row['after_band_m']}")
        print(f"      before free/obst/unknown "
              f"{b['free']}/{b['obstacle']}/{b['unknown']}" if b else "      before: -")
        print(f"      after  free/obst/unknown "
              f"{af['free']}/{af['obstacle']}/{af['unknown']}   "
              f"points in band {row['points_in_band']}/{row['points_in_grid']}"
              + ("   WRITTEN" if row["written"] else ""))
    print(f"\n{done} scene(s) with both artifacts, {skipped} skipped.")
    if not a.apply:
        print("DRY RUN (default): nothing was written. Re-run with --apply, "
              "on the owner's word.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
