#!/usr/bin/env python3
"""Room measurements over a shipped band layer: free space per robot, the widest passage,
and a scale sanity check against furniture heights.

Three numbers the batch table needs, none of which exists in any commit. `38b7498`'s message
quotes doorway widths (0.450 m on hero-15fps at cell (37,112), 0.050 m on filtered hero-74)
but the code that produced them was an ephemeral heredoc that was never saved, so this is a
rewrite from the description rather than a recovered script.

**The one free parameter, stated rather than buried.** "Widest passage between open areas"
needs a definition of *open area*, and the original threshold was not recorded anywhere. Here
an open area is a connected component of cells whose clearance is at least `--open-clearance`
metres (default 0.50). Change it and the answer changes: a lower threshold merges rooms
through the very gap you are trying to measure, a higher one can leave fewer than two areas
and no passage to report at all. The default is calibrated in the repo's own terms - it
reproduces hero-15fps's published 0.450 m - and every result records the threshold it used.

Widths follow the handoff's convention `width = (2d - 1) * res`, where `d` is the bottleneck
clearance in cells along the max-min path. A one-cell-wide gap (d=1) reads as 1 cell, not 2.

    room_measure.py --layer-dir <dir> [--radii 0.10 0.2496 0.5528] [--json out.json]
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

FREE, OBSTACLE, UNKNOWN = 0, 1, 2

#: Clearance a cell needs before it counts as standing in "open" floor rather than in a gap.
#: See the module docstring: this is the measurement's one judgement call.
DEFAULT_OPEN_CLEARANCE_M = 0.50

#: What the scale check expects to find, in metres above the floor. From the owner's brief.
SCALE_PRIORS = {"bed": (0.60, 0.70), "desk": (0.74, 0.80)}
SCALE_TOLERANCE_M = 0.05


def load_layer(d: Path) -> tuple[np.ndarray, dict, np.ndarray | None]:
    meta = json.loads((d / "occupancy_meta.json").read_text())
    cells = np.load(d / "occupancy.npy")
    hp = d / "obstacle_min_height.npy"
    lowest = np.load(hp) if hp.exists() else None
    return cells, meta, lowest


def free_after_inflation(cells: np.ndarray, res: float, radius_m: float) -> np.ndarray:
    """Cells a robot of this radius can stand in.

    UNKNOWN is traversable - `build_cost_grid` prices it at 1.5x rather than blocking it,
    and blocking it measures 0/17 reachable for every robot on hero-74, where 43% of the grid
    was never directly observed. So only OBSTACLE inflates. That choice is the app's, and
    this mirrors it rather than inventing a stricter one for the report.
    """
    obstacle = cells == OBSTACLE
    if radius_m <= 0:
        return ~obstacle
    dist = ndimage.distance_transform_edt(~obstacle) * res
    return dist > radius_m


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    return lab, int(n)


def widest_passage(cells: np.ndarray, res: float, open_clearance_m: float) -> dict:
    """Max-min (bottleneck) search on the clearance field, between distinct open areas.

    Dijkstra with `max` in place of `sum`: the value of a path is its TIGHTEST cell, and we
    want the path whose tightest cell is as wide as possible. A max-heap keyed on that value
    settles each cell with the best bottleneck any path can deliver to it, so the first time
    a second open area is settled, its key is the widest passage into it from the first.
    """
    obstacle = cells == OBSTACLE
    clear = ndimage.distance_transform_edt(~obstacle)          # in CELLS
    open_cells = (clear * res) >= open_clearance_m
    lab, n = components(open_cells)
    if n < 2:
        return {"widest_passage_m": None, "n_open_areas": int(n),
                "reason": f"fewer than 2 open areas at >= {open_clearance_m} m clearance",
                "open_clearance_m": open_clearance_m}

    sizes = np.bincount(lab.ravel())[1:]
    home = int(np.argmax(sizes)) + 1                            # start from the biggest area

    best = np.full(cells.shape, -np.inf)
    seen = np.zeros(cells.shape, bool)
    heap: list[tuple[float, int, int]] = []
    for iy, ix in zip(*np.nonzero(lab == home)):
        best[iy, ix] = clear[iy, ix]
        heapq.heappush(heap, (-clear[iy, ix], int(iy), int(ix)))

    H, W = cells.shape
    while heap:
        negd, y, x = heapq.heappop(heap)
        d = -negd
        if seen[y, x]:
            continue
        seen[y, x] = True
        if lab[y, x] not in (0, home):
            # First other open area settled: d is the bottleneck of the widest path to it.
            return {"widest_passage_m": float((2 * d - 1) * res),
                    "bottleneck_cells": float(d), "at_cell": [int(y), int(x)],
                    "n_open_areas": int(n), "open_clearance_m": open_clearance_m,
                    "between_areas": [home, int(lab[y, x])]}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W) or seen[ny, nx] or obstacle[ny, nx]:
                    continue
                cand = min(d, clear[ny, nx])
                if cand > best[ny, nx]:
                    best[ny, nx] = cand
                    heapq.heappush(heap, (-cand, ny, nx))
    return {"widest_passage_m": None, "n_open_areas": int(n),
            "reason": "open areas are not connected through free space at all",
            "open_clearance_m": open_clearance_m}


def flat_surface_heights(lowest: np.ndarray, res: float, min_area_m2: float = 0.30) -> list:
    """Heights of the largest flat surfaces above the floor, for the scale check.

    `obstacle_min_height.npy` is the LOWEST surface per cell, so a desk top reads as the desk
    top only where nothing lower is under it. Surfaces are found by binning heights at 1 cm
    and keeping bins whose cells form a component of at least `min_area_m2` - a table top is
    a plateau, and a plateau is what a histogram bin plus a connectivity test finds.
    """
    if lowest is None:
        return []
    finite = np.isfinite(lowest)
    if not finite.any():
        return []
    out = []
    hs = lowest[finite]
    for lo in np.arange(0.20, min(2.0, float(hs.max())) + 0.01, 0.01):
        band = finite & (lowest >= lo) & (lowest < lo + 0.03)
        if not band.any():
            continue
        lab, n = components(band)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        big = sizes.max() * res * res
        if big >= min_area_m2:
            out.append({"height_m": round(float(lo + 0.015), 3),
                        "area_m2": round(float(big), 3)})
    out.sort(key=lambda r: -r["area_m2"])
    # Collapse near-duplicate bins - a 0.75 m desk lights up 0.74, 0.75 and 0.76.
    kept: list[dict] = []
    for r in out:
        if all(abs(r["height_m"] - k["height_m"]) > 0.04 for k in kept):
            kept.append(r)
    return kept[:8]


def scale_verdict(surfaces: list) -> dict:
    """Does any measured surface sit where a bed or a desk should? Reported per prior, and
    a miss on BOTH is what 'scale suspect' means - one missing item is furniture, two is a
    scale error."""
    out = {}
    for name, (lo, hi) in SCALE_PRIORS.items():
        hit = next((s for s in surfaces
                    if lo - SCALE_TOLERANCE_M <= s["height_m"] <= hi + SCALE_TOLERANCE_M), None)
        out[name] = {"expected_m": [lo, hi], "found_m": hit["height_m"] if hit else None,
                     "area_m2": hit["area_m2"] if hit else None, "ok": hit is not None}
    out["scale_suspect"] = not any(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def measure(layer_dir: Path, radii: list[float], open_clearance_m: float) -> dict:
    cells, meta, lowest = load_layer(layer_dir)
    res = float(meta["resolution"])
    per_robot = {}
    for r in radii:
        free = free_after_inflation(cells, res, r)
        lab, n = components(free)
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([], int)
        per_robot[f"r{r}"] = {
            "radius_m": r, "free_cells": int(free.sum()), "components": int(n),
            "largest_component_cells": int(sizes.max()) if sizes.size else 0,
            "largest_component_m2": round(float(sizes.max() * res * res), 3) if sizes.size else 0.0,
        }
    surfaces = flat_surface_heights(lowest, res)
    return {
        "layer_dir": str(layer_dir),
        "grid": {"width": meta["width"], "height": meta["height"], "resolution_m": res,
                 "extent_m": [round(meta["width"] * res, 2), round(meta["height"] * res, 2)]},
        "cells": {"free": int((cells == FREE).sum()), "obstacle": int((cells == OBSTACLE).sum()),
                  "unknown": int((cells == UNKNOWN).sum())},
        "obstacle_cells_meta": meta.get("obstacle_cells"),
        "band_free_per_robot": per_robot,
        "passage": widest_passage(cells, res, open_clearance_m),
        "flat_surfaces": surfaces,
        "scale_check": scale_verdict(surfaces),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--radii", type=float, nargs="+", default=[0.10, 0.2496, 0.5528])
    ap.add_argument("--open-clearance", type=float, default=DEFAULT_OPEN_CLEARANCE_M)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    doc = measure(Path(a.layer_dir), a.radii, a.open_clearance)
    text = json.dumps(doc, indent=2)
    if a.json:
        Path(a.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
