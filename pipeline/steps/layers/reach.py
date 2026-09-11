#!/usr/bin/env python3
"""Reachability on the layer LAYERS shipped - and on nothing else.

A copy of nvblox_v2/reachability.py (read-only there; a copy, not an import, same convention
as steps/nvblox_scenes.py) with ONE change of substance: it no longer re-derives the
obstacle band from mesh.ply. It loads `obstacle_mask.npy` out of the layer directory.

Why that matters more than it sounds. The band rule used to live here, in a script the
front end never runs, while LAYERS shipped an ESDF slice to everyone else. Two different
answers to "where can the robot stand" existed for the same scene, and only one of them was
ever looked at in the UI. Now the band is computed once, in band_mask.py, and this reads it -
so a number reported here is the number the scene page shows.

The clearance field is the 2D Euclidean distance transform of that mask, and free space for
a disc of radius r is (clearance >= r) AND observed. Unobserved is not free, here as
everywhere: `unobserved_mask.npy` comes from the same layer and gates every answer.

The esdf3d comparison the original carried is gone with the slice it compared against. It
was reported for completeness and explicitly not used for planning - "capped at 0.30 m BY
CONSTRUCTION", in its own words - and keeping a field nothing may act on invites acting on
it.

Then, unchanged from the original: scipy.ndimage.label (8-connectivity) -> are start and
goal in the same component? -> A* (8-neighbour, Euclidean cost) -> path, length, minimum
clearance along it and where that minimum sits.
"""

from __future__ import annotations
import argparse, heapq, json
from pathlib import Path

import numpy as np
from scipy import ndimage

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- radius convention
def astar(free: np.ndarray, start, goal, res: float):
    nu, nv = free.shape
    g = {start: 0.0}
    def hcost(c):
        return float(np.hypot(c[0] - goal[0], c[1] - goal[1]) * res)
    openq = [(hcost(start), start)]
    came, seen = {}, set()
    while openq:
        _, cur = heapq.heappop(openq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        for du, dv in NEIGH:
            n = (cur[0] + du, cur[1] + dv)
            if not (0 <= n[0] < nu and 0 <= n[1] < nv) or not free[n]:
                continue
            ng = g[cur] + float(np.hypot(du, dv)) * res
            if ng < g.get(n, np.inf):
                g[n] = ng; came[n] = cur
                heapq.heappush(openq, (ng + hcost(n), n))
    return None


def nearest(mask: np.ndarray, cell):
    if mask[cell]:
        return cell, 0.0
    d, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
    return (int(idx[0][cell]), int(idx[1][cell])), float(d[cell])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-dir", required=True,
                    help="a LAYERS output directory - the one band_mask.py wrote")
    ap.add_argument("--heights", type=float, nargs="+", default=None,
                    help="robot height per --radii entry, metres. A surface higher than "
                         "this (plus --height-margin) is driven UNDER, not around. "
                         "Omit to use the layer's fixed mask.")
    ap.add_argument("--height-margin", type=float, default=0.05,
                    help="headroom over the robot's own height before a surface blocks it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radii", type=float, nargs="+", default=[0.3, 0.6])
    a = ap.parse_args()

    sd = Path(a.layers_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # The whole point of this file: the obstacle band is READ, never re-derived. If the
    # layer is not there, say so by name rather than silently falling back to a mesh.
    need = ["obstacle_mask.npy", "unobserved_mask.npy", "occupancy.npy", "floor_plane.json",
            "occupancy_meta.json"]
    missing = [f for f in need if not (sd / f).exists()]
    if missing:
        raise SystemExit(f"reach: {sd} is not a layer directory - missing {missing}. "
                         f"Run LAYERS (pipeline/steps/layers/band_mask.py) first.")
    # The SOURCE: lowest surface height per cell, NaN where the cell has none. The boolean
    # mask next to it is the default-height view; thresholding this per robot is the point.
    lowest = (np.load(sd / "obstacle_min_height.npy")
              if (sd / "obstacle_min_height.npy").exists() else None)
    obst = np.load(sd / "obstacle_mask.npy").astype(bool)
    unobs = np.load(sd / "unobserved_mask.npy").astype(bool)
    cells = np.load(sd / "occupancy.npy")
    om = json.loads((sd / "occupancy_meta.json").read_text())
    fp = json.loads((sd / "floor_plane.json").read_text())
    res = fp["slice_resolution_m"]
    org = np.asarray(fp["slice_origin_world"])
    au, av = np.asarray(fp["axis_u"]), np.asarray(fp["axis_v"])
    u0, v0 = fp["u_range"][0], fp["v_range"][0]
    nu, nv = obst.shape
    # A "thin" cell - some band surface but fewer than min_points of it - is UNKNOWN in the
    # shipped tri-state grid, and it has to be UNKNOWN here too. Without this, REACH's
    # `observed` came only from unobserved_mask.npy and a path could cross a cell the layer
    # itself calls UNKNOWN: measured on own_0901_161054, exactly one TurtleBot path cell.
    thin = (np.load(sd / "thin_mask.npy").astype(bool)
            if (sd / "thin_mask.npy").exists() else np.zeros_like(unobs))
    observed = ~unobs & ~thin

    conv = {"convention": "not_applicable",
            "clearance_rule_used": "clearance(r) = EDT(band obstacle mask) >= r, AND observed"}

    # distance from every cell to the nearest unobserved cell: how much observed margin a
    # path keeps. inf-safe: if a slice were fully observed there would be no boundary.
    dist_to_unobs = (ndimage.distance_transform_edt(~unobs) * res
                     if unobs.any() else np.full((nu, nv), np.inf))
    def band_for(height_m):
        """Cells this robot cannot drive under. Falls back to the fixed mask when the
        layer predates the height map."""
        if lowest is None or height_m is None:
            return obst
        with np.errstate(invalid="ignore"):
            return np.isfinite(lowest) & (lowest <= height_m + a.height_margin)

    nav2d = ndimage.distance_transform_edt(~obst) * res
    np.save(out / "obstacle_grid.npy", obst)
    np.save(out / "clearance_nav2d.npy", nav2d.astype(np.float32))

    def cell_to_uv(c):
        return (u0 + c[0] * res, v0 + c[1] * res)

    def cell_to_world(c):
        u, v = cell_to_uv(c)
        return (org + u * au + v * av).tolist()

    # Anchors: the most open observed cell in each QUADRANT of the slice. Taking the
    # nearest non-obstacle cell to a bbox corner puts every anchor against a wall; taking
    # a global "most open cell near the corner" collapses two corners onto one cell.
    mu, mv = nu // 2, nv // 2
    quads = {"SW": (slice(0, mu), slice(0, mv)), "SE": (slice(mu, nu), slice(0, mv)),
             "NW": (slice(0, mu), slice(mv, nv)), "NE": (slice(mu, nu), slice(mv, nv))}
    anchors = {}
    for k, (su, sv) in quads.items():
        score = np.where(observed[su, sv], nav2d[su, sv], -1.0)
        iu, iv = np.unravel_index(int(np.argmax(score)), score.shape)
        anchors[k] = (int(iu + su.start), int(iv + sv.start))

    # all six unordered corner pairs rather than an arbitrary five - the five obvious
    # ones happened to omit SE->NE, the only pair that runs straight down the corridor
    pairs = [("SW", "NE"), ("SE", "NW"), ("SW", "SE"), ("NW", "NE"), ("SW", "NW"), ("SE", "NE")]

    results = {"scene_dir": str(sd), "slice_shape": [nu, nv], "resolution_m": res,
               "obstacle_band_m": [om.get("band_min"), om.get("band_max")],
               "layer_dir": str(sd), "layer_source": om.get("source"),
               "radius_convention": conv,
               "obstacle_cells": int(obst.sum()),
               "mesh_vertices_in_band": om.get("mesh_vertices_in_band"),
               "mesh_vertices_total": om.get("mesh_vertices_total"),
               "tri_state": {"free": om.get("n_free"), "obstacle": om.get("n_obstacle"),
                             "unknown": om.get("n_unknown")},
               "observed_frac": float(observed.mean()),
               "unobserved_frac": float(unobs.mean()),
               "min_component_tri_dropped": om.get("min_component_tri"),
               "obstacle_cells_before_component_filter":
                   om.get("obstacle_cells_before_component_filter"),
               "nav2d_clearance_max_m": float(nav2d[observed].max()),
               # The layer ships one already-filtered mask, so there is no unfiltered
               # variant to compare against here; band_mask.py records the raw count.
               "nav2d_clearance_max_m_before_component_filter": None,
               "anchors": {k: {"cell": list(v), "uv": [round(x, 3) for x in cell_to_uv(v)],
                               "world": [round(x, 3) for x in cell_to_world(v)]}
                           for k, v in anchors.items()},
               "runs": []}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = dict(zip(a.radii, a.heights)) if a.heights else {}
    for r in a.radii:
        h = heights.get(r)
        obst_r = band_for(h)
        nav2d = ndimage.distance_transform_edt(~obst_r) * res
        free_nav = observed & (nav2d >= r)
        lab, nlab = ndimage.label(free_nav, structure=np.ones((3, 3), int))
        # optimistic variant: unobserved cells treated as traversable. 27% of this slice
        # is unobserved, and treating it as blocked is what splits the room up, so both
        # readings are reported rather than one being presented as the answer.
        free_opt = nav2d >= r
        lab_o, nlab_o = ndimage.label(free_opt, structure=np.ones((3, 3), int))
        for k0, k1 in pairs:
            s_, g_ = anchors[k0], anchors[k1]
            row = {"radius_m": r, "from": k0, "to": k1,
                   "start_cell": list(s_), "goal_cell": list(g_),
                   "free_frac_nav2d": round(float(free_nav.sum() / observed.sum()), 4),
                   # Two fields the original carried are gone rather than kept as copies
                   # of free_frac_nav2d under their old names: free_frac_esdf3d described a
                   # slice this no longer reads, and the "no component filter" variant
                   # needs an unfiltered mask the layer does not ship. A field that silently
                   # became a duplicate is worse than an absent one.
                   "free_cells_nav2d": int(free_nav.sum()),
                   "observed_cells": int(observed.sum()),
                   "n_free_components": int(nlab),
                   "start_free": bool(free_nav[s_]), "goal_free": bool(free_nav[g_]),
                   "free_frac_if_unknown_traversable":
                       round(float(free_opt.sum() / free_opt.size), 4),
                   "reachable_if_unknown_traversable":
                       bool(free_opt[s_] and free_opt[g_] and lab_o[s_] == lab_o[g_])}
            if not (free_nav[s_] and free_nav[g_]):
                row["reachable"] = False
                row["reason"] = ("start corner has less than %.2f m clearance" % r
                                 if not free_nav[s_] else
                                 "goal corner has less than %.2f m clearance" % r)
            elif lab[s_] != lab[g_]:
                row["reachable"] = False
                row["reason"] = "start and goal are in different free components"
            else:
                path = astar(free_nav, s_, g_, res)
                if path is None:
                    row["reachable"] = False; row["reason"] = "same component but A* found no path"
                else:
                    p = np.asarray(path)
                    seg = np.hypot(*(np.diff(p, axis=0).T)) * res
                    cl = nav2d[p[:, 0], p[:, 1]]
                    i = int(np.argmin(cl))
                    puno = unobs[p[:, 0], p[:, 1]]
                    row.update({"reachable": True, "path_cells": int(len(path)),
                                "path_cells_in_unobserved": int(puno.sum()),
                                "path_frac_in_unobserved": round(float(puno.mean()), 4),
                                "path_min_dist_to_unobserved_m":
                                    round(float(dist_to_unobs[p[:, 0], p[:, 1]].min()), 3),
                                "path_median_dist_to_unobserved_m":
                                    round(float(np.median(dist_to_unobs[p[:, 0], p[:, 1]])), 3),
                                "path_length_m": round(float(seg.sum()), 3),
                                "straight_line_m": round(float(np.hypot(*(np.asarray(s_) - np.asarray(g_))) * res), 3),
                                "min_clearance_m": round(float(cl[i]), 3),
                                "bottleneck_cell": [int(p[i, 0]), int(p[i, 1])],
                                "bottleneck_uv": [round(x, 3) for x in cell_to_uv(tuple(p[i]))],
                                "bottleneck_world": [round(x, 3) for x in cell_to_world(tuple(p[i]))]})
                    row["_path"] = p.tolist()
            results["runs"].append(row)

        fig, ax = plt.subplots(figsize=(6.5, min(6.5 * nv / max(nu, 1) + 1.2, 14)))
        ext = [u0, u0 + nu * res, v0, v0 + nv * res]
        disp = np.where(observed, nav2d, np.nan)
        im = ax.imshow(disp.T, origin="lower", cmap="viridis", extent=ext)
        ax.imshow(np.where(unobs, 1.0, np.nan).T, origin="lower", cmap="Greys",
                  vmin=0, vmax=1.6, extent=ext)                      # unobserved = grey
        ax.imshow(np.where(obst, 1.0, np.nan).T, origin="lower", cmap="autumn",
                  extent=ext, alpha=0.9)                             # obstacles = red
        fig.colorbar(im, ax=ax, shrink=0.8, label="2D clearance [m]")
        for row in results["runs"]:
            if row["radius_m"] != r or "_path" not in row:
                continue
            p = np.asarray(row["_path"])
            ax.plot(u0 + p[:, 0] * res, v0 + p[:, 1] * res, lw=2.2,
                    label=f'{row["from"]}->{row["to"]}  {row["path_length_m"]} m, '
                          f'min {row["min_clearance_m"]} m')
            b = row["bottleneck_cell"]
            ax.plot(u0 + b[0] * res, v0 + b[1] * res, "x", ms=9, mew=2, c="k")
        for k, c in anchors.items():
            u, v = cell_to_uv(c)
            ax.plot(u, v, "o", ms=8, mfc="none", mec="w", mew=2)
            ax.annotate(k, (u, v), color="w", fontsize=10, fontweight="bold")
        n_ok = sum(1 for x in results["runs"] if x["radius_m"] == r and x.get("reachable"))
        short = sd.name.replace("__optimal_step2", "").replace("__step2", "")
        ax.set_title(f'{short}   r = {r} m   {n_ok}/{len(pairs)} pairs reachable\n'
                     f'grey = unobserved, red = obstacle, x = bottleneck', fontsize=10)
        ax.set_xlabel("u [m]"); ax.set_ylabel("v [m]")
        if n_ok:
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout(); fig.savefig(out / f"reachability_r{r}.png", dpi=130)
        plt.close(fig)

    for row in results["runs"]:
        if "_path" in row:
            row["path"] = row.pop("_path")
    (out / "reachability.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
