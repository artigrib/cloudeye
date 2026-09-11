#!/usr/bin/env python3
"""The obstacle layer a ground robot actually needs: a height BAND above the fitted floor.

Replaces `esdf_slice_0.3m.npy` as the layer LAYERS ships and REACH reads.

Why the ESDF slice had to go, with the number that settles it: it is the 3D ESDF sampled on
a plane 0.3 m above the floor, so it is capped at 0.30 m BY CONSTRUCTION - the floor itself
is 0.30 m below every query point, and no cell can report more clearance than that. Measured
on both hero (`own_0901_173903__step2`) and `own_0901_161054__optimal_step2`: max 0.3421 m.
A Husky (r = 0.5528 m) is therefore "blocked" in every cell of every scene, forever, for a
reason that is about the sampling height and not about the room. Free cells at that radius:
0. That is not a hard room; it is an unusable layer.

The band rule is copied from `nvblox_v2/reachability.py::obstacle_grid` (read-only there,
this is a copy and not an import - same convention as `steps/nvblox_scenes.py`): a cell is
an obstacle when any reconstructed surface stands between `band` metres above the floor
plane. Floor and ceiling are excluded by the band itself; `--min-component-tri` drops
isolated blobs first, because a 10-triangle speck of reconstruction noise floating in a room
is treated as solid and silently destroys the clearance field around it.

Unobserved is NOT free. It keeps nvblox's own +100 sentinel
(`constants.esdf_unknown_distance()`, see nvblox_scenes.py's header) in the float layer and
becomes UNKNOWN in the tri-state grid. A cell nobody looked at is a cell nobody may claim is
clear - the whole reason `occupancy.npy` has ever been tri-state.

Two arrays, one grid:

  obstacle_mask.npy  bool  (nu, nv)   True = something stands in the band. Floor-plane
                                      (u, v) frame, the same one floor_plane.json defines.
  occupancy.npy      uint8 (nu, nv)   0 FREE / 1 OBSTACLE / 2 UNKNOWN, indexed [ix, iz]
                                      exactly as app/services/pathfinding.py expects, with
                                      ix the u index and iz the v index.

They are the same grid, so REACH and the scene page read one layer rather than two that
have to be kept in step. `occupancy_meta.json` carries the GridMeta fields the app already
reads (resolution / origin_x / origin_z / width / height) - unchanged in shape - plus the
band bounds, so a reader can tell which rule produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Same names and values as app/services/pathfinding.py. Written out rather than imported:
# this runs under the CPU venv, which does not have the app on its path.
FREE, OBSTACLE, UNKNOWN = 0, 1, 2

#: nvblox's own "never observed" distance. Kept in the float layer so the sentinel a reader
#: meets here is the sentinel nvblox wrote, not a second convention invented downstream.
UNKNOWN_SENTINEL = 100.0

DEFAULT_BAND = (0.1, 1.5)
DEFAULT_MIN_COMPONENT_TRI = 100

#: A band cell backed by fewer than this many surface samples is drift, not furniture.
#: 3 is stage_occupancy.py's own phantom-obstacle threshold (DEFAULT_MIN_POINTS_PER_CELL),
#: kept identical so the two producers do not disagree about what counts as a thing.
DEFAULT_MIN_POINTS_PER_CELL = 3

#: Headroom over a robot's own height before a surface counts as blocking it. One cell of
#: vertical slack: reconstruction puts a surface within a few centimetres, and a robot that
#: fits under a shelf by 1 cm on paper does not fit under it in a room.
DEFAULT_MARGIN = 0.05

#: The height used when nobody says which robot - the registry's default platform. Recorded
#: in the metadata so a grid never silently means "some robot".
DEFAULT_ROBOT_HEIGHT_M = 0.192   # TurtleBot3 burger, app/robots.py dimensions_m.height_m


def obstacle_grid(mesh_path: Path, fp: dict, shape: tuple[int, int],
                  band: tuple[float, float], min_tri: int = 0):
    """Rasterise mesh vertices standing between band[0] and band[1] above the floor.

    Copied from nvblox_v2/reachability.py:132. `min_tri` drops connected components smaller
    than that many triangles first - see the module docstring.

    Returns (mask, n_in_band, n_total, lowest) where `lowest` is the LOWEST height above the
    floor of any surface in each cell, NaN where the cell has none. That map is the layer's
    source of truth: a boolean mask answers "is something there" for one fixed band, and the
    band that matters is the robot's own height - see the module docstring.
    """
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(m.vertices):
        raise SystemExit(f"band_mask: {mesh_path} has no vertices (Open3D returns an EMPTY "
                         f"mesh for a file it cannot open, so check the path exists)")
    if min_tri > 0 and len(m.triangles):
        labels, counts, _ = m.cluster_connected_triangles()
        counts = np.asarray(counts)
        keep = np.asarray(labels) >= 0
        keep &= counts[np.asarray(labels)] >= min_tri
        m.remove_triangles_by_mask(~keep)
        m.remove_unreferenced_vertices()
    v = np.asarray(m.vertices)
    up = np.asarray(fp["normal_up"])
    c = np.asarray(fp["point_on_plane"])
    au, av = np.asarray(fp["axis_u"]), np.asarray(fp["axis_v"])
    org = np.asarray(fp["slice_origin_world"])
    hgt = (v - c) @ up
    sel = (hgt >= band[0]) & (hgt <= band[1])
    rel = v[sel] - org
    res = fp["slice_resolution_m"]
    u0, v0 = fp["u_range"][0], fp["v_range"][0]
    iu = np.round((rel @ au - u0) / res).astype(int)
    iv = np.round((rel @ av - v0) / res).astype(int)
    ok = (iu >= 0) & (iu < shape[0]) & (iv >= 0) & (iv < shape[1])
    grid = np.zeros(shape, bool)
    grid[iu[ok], iv[ok]] = True
    hits = np.zeros(shape, dtype=np.int32)
    np.add.at(hits, (iu[ok], iv[ok]), 1)
    lowest = np.full(shape, np.inf, dtype=np.float64)
    np.minimum.at(lowest, (iu[ok], iv[ok]), hgt[sel][ok])
    lowest[~np.isfinite(lowest)] = np.nan
    return grid, int(sel.sum()), int(len(v)), lowest.astype(np.float32), hits


def obstacles_for_height(lowest: np.ndarray, robot_height_m: float,
                         margin_m: float = DEFAULT_MARGIN) -> np.ndarray:
    """Which cells block a robot this tall: something whose LOWEST surface is at or below
    the robot's own height plus a margin.

    This is the whole per-robot rule, in one line, and it is derived from the shipped map
    rather than from a second file - so REACH and the scene page cannot disagree.

    A fixed 1.5 m upper bound made a bed top at 0.694 m and a hanging curtain at 0.876 m
    into walls for a 0.192 m TurtleBot. Measured on hero-74: the three objects TurtleBot
    lost were disconnected by cells whose lowest surface sits at 0.684-0.876 m, and 3107 of
    the 3879 newly-blocked cells had nothing below 0.4 m at all.
    """
    with np.errstate(invalid="ignore"):
        return np.isfinite(lowest) & (lowest <= robot_height_m + margin_m)


def tri_state(obstacle: np.ndarray, unobserved: np.ndarray,
              thin: np.ndarray | None = None) -> np.ndarray:
    """FREE / OBSTACLE / UNKNOWN, with unobserved applied LAST so it can never read free.

    Order matters and is the point: a cell that is both unobserved and in the band is
    UNKNOWN, not OBSTACLE - we did not see it, so we do not get to say what is there. A cell
    that is unobserved and not in the band is UNKNOWN, not FREE, for the same reason.

    `thin` is a cell with SOME band surface but fewer than min_points of it: one or two
    stray points. "Any point at all = obstacle" promotes reconstruction drift to a wall,
    and a wall is inflated by the robot's radius - measured on hero-74, 534 such cells
    dilated at Go2's 0.2496 m sealed a corridor that a 0.1 m robot walked straight down.
    Neither OBSTACLE (we cannot claim a wall from two points) nor FREE (we cannot claim
    clear floor either): UNKNOWN, the state that exists for exactly this.
    """
    out = np.full(obstacle.shape, FREE, dtype=np.uint8)
    out[obstacle] = OBSTACLE
    if thin is not None:
        out[thin] = UNKNOWN
    out[unobserved] = UNKNOWN
    return out


def build(scene_dir: Path, out_dir: Path, *, band=DEFAULT_BAND,
          min_component_tri=DEFAULT_MIN_COMPONENT_TRI, mesh_name="mesh.ply",
          min_points_per_cell=DEFAULT_MIN_POINTS_PER_CELL) -> dict:
    fp = json.loads((scene_dir / "floor_plane.json").read_text())
    unobs = np.load(scene_dir / "unobserved_slice.npy").astype(bool)
    nu, nv = unobs.shape
    res = fp["slice_resolution_m"]

    obst_any, n_band, n_vert, lowest, hits = obstacle_grid(
        scene_dir / mesh_name, fp, (nu, nv), band, min_component_tri)
    obst_raw, _, _, _, _ = obstacle_grid(scene_dir / mesh_name, fp, (nu, nv), band, 0)
    solid = hits >= min_points_per_cell
    thin = obst_any & ~solid
    obst = solid
    # A thin cell keeps no height: thresholding drift against a robot's roof would decide
    # whether to trust two points by how tall the robot is.
    lowest = np.where(solid, lowest, np.nan).astype(np.float32)
    # The default view, for a reader that names no robot. The SOURCE is `lowest`; this is
    # derived from it by the same function REACH and the router use, so it can never mean
    # something the per-robot answer does not.
    cells = tri_state(obstacles_for_height(lowest, DEFAULT_ROBOT_HEIGHT_M), unobs, thin)

    # The float companion, in nvblox's own units and with nvblox's own sentinel: 0.0 where
    # something stands in the band, +100.0 where nothing was ever observed.
    band_f = np.where(obst, 0.0, np.nan).astype(np.float32)
    band_f[unobs] = UNKNOWN_SENTINEL

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "obstacle_min_height.npy", lowest)
    np.save(out_dir / "obstacle_mask.npy", obst)
    np.save(out_dir / "thin_mask.npy", thin)
    np.save(out_dir / "occupancy.npy", cells)
    np.save(out_dir / "band_distance.npy", band_f)
    np.save(out_dir / "unobserved_mask.npy", unobs)

    meta = {
        # Exactly the five fields app/services/scene_ingest.read_grid_metadata reads. ix is
        # the u index and iz the v index, so the app's [ix, iz] IS this grid's (u, v).
        "resolution": res,
        "origin_x": fp["u_range"][0],
        "origin_z": fp["v_range"][0],
        "width": int(nu),
        "height": int(nv),
        # ...and what produced it, so a reader can tell this from a stage_occupancy grid.
        "source": "pipeline/steps/layers/band_mask.py",
        "band_min": float(band[0]),
        "band_max": float(band[1]),
        "min_component_tri": int(min_component_tri),
        "min_points_per_cell": int(min_points_per_cell),
        "thin_cells_demoted_to_unknown": int(thin.sum()),
        "layer_kind": "obstacle_min_height",
        "height_margin_m": DEFAULT_MARGIN,
        "default_robot_height_m": DEFAULT_ROBOT_HEIGHT_M,
        "occupancy_npy_note": ("derived from obstacle_min_height.npy at "
                               f"default_robot_height_m; the MAP is the source of truth, "
                               f"threshold it per robot with obstacles_for_height()"),
        "unknown_sentinel": UNKNOWN_SENTINEL,
        "n_free": int((cells == FREE).sum()),
        "n_obstacle": int((cells == OBSTACLE).sum()),
        "n_unknown": int((cells == UNKNOWN).sum()),
        "obstacle_cells": int(obst.sum()),
        "obstacle_cells_before_component_filter": int(obst_raw.sum()),
        "mesh_vertices_in_band": n_band,
        "mesh_vertices_total": n_vert,
        "unobserved_frac": round(float(unobs.mean()), 4),
        # The (u, v) frame is not world XZ. Anything drawing this in world space needs
        # these, and floor_plane.json is copied next to it for exactly that reason.
        "frame": "scene's own fitted floor plane (u, v); NOT world X/Z",
        "origin_world": fp["slice_origin_world"],
        "axis_u": fp["axis_u"], "axis_v": fp["axis_v"],
        "up_axis_world": fp["normal_up"],
    }
    (out_dir / "occupancy_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene-dir", required=True, help="an nvblox output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND),
                    help="height band above the floor treated as obstacles")
    ap.add_argument("--min-component-tri", type=int, default=DEFAULT_MIN_COMPONENT_TRI)
    ap.add_argument("--min-points-per-cell", type=int, default=DEFAULT_MIN_POINTS_PER_CELL,
                    help="band cells with fewer surface samples than this become UNKNOWN "
                         "rather than OBSTACLE - see tri_state")
    ap.add_argument("--mesh-name", default="mesh.ply")
    a = ap.parse_args()
    m = build(Path(a.scene_dir), Path(a.out), band=tuple(a.band),
              min_component_tri=a.min_component_tri, mesh_name=a.mesh_name,
              min_points_per_cell=a.min_points_per_cell)
    print(f"band {a.band} -> obstacle {m['obstacle_cells']} cells "
          f"(raw {m['obstacle_cells_before_component_filter']}), "
          f"free {m['n_free']} / obstacle {m['n_obstacle']} / unknown {m['n_unknown']}, "
          f"grid {m['width']}x{m['height']} @ {m['resolution']} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
