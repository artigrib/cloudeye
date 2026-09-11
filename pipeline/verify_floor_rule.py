#!/usr/bin/env python3
"""Exercise the floor-selection rule on real scenes, on CPU, without a GPU box.

`fit_floor`'s input is the nvblox mesh's vertices (every scene's mesh is 121k-834k
vertices, well over the 20 000 threshold that would fall back to the depth cloud), and both
the mesh and the camera poses are already on this VPS. So the rule can be swept over seeds
and scenes for free, which is the only honest way to claim it fixed anything: the failure it
fixes is a 4-in-9 coin flip, and a single run proves nothing either way.

`pipeline/steps/nvblox_scenes.py` imports torch at module scope and cannot be imported here,
so the functions under test are extracted from its own source with `ast` - nothing is
retyped, and a drift between this harness and the shipped script is impossible.

    $PIPELINE_CPU_PYTHON pipeline/verify_floor_rule.py --scenes <name> [--seeds 0 1 ...]
    (default ~/venvs/o3d-cpu/bin/python - see pipeline/cpu_env.py)
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # the pin the real script applies

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

STEPS = Path(__file__).resolve().parent / "steps" / "nvblox_scenes.py"
NVBLOX = Path("var/nvblox_v2")
WANT = ("fit_floor", "median_camera_up", "angle_to_deg")


def load_from_source():
    src = STEPS.read_text()
    tree = ast.parse(src)
    ns: dict = {"np": np, "DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M": (0.8, 2.0)}
    for name in ("FloorFitError", "FloorImplausible"):
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(STEPS), "exec"), ns)
    for name in WANT:
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(STEPS), "exec"), ns)
    return ns


def run_scene(ns, scene: str, seeds: list[int], support_frac: float, gate_deg: float,
              iters: int = 30000, union_deg: float = 0.0):
    import open3d as o3d
    mesh_p = NVBLOX / "results" / "scenes_out" / scene / "mesh.ply"
    meta_p = NVBLOX / "packed" / scene / "meta.json"
    if not (mesh_p.exists() and meta_p.exists()):
        return [{"scene": scene, "error": "missing mesh or meta"}]
    mv = np.asarray(o3d.io.read_triangle_mesh(str(mesh_p)).vertices)
    views = json.loads(meta_p.read_text())["views"]
    cam_up = ns["median_camera_up"](views)
    cams = np.asarray([v["camera_pose"] for v in views], float)[:, :3, 3]
    out = []
    for seed in seeds:
        t0 = time.time()
        row = {"scene": scene, "seed": seed, "n_verts": int(len(mv)),
               "n_views": len(views)}
        try:
            best, cands = ns["fit_floor"](mv, cams.mean(0), seed=seed, camera_up=cam_up,
                                          support_frac=support_frac, gate_deg=gate_deg,
                                          ransac_iterations=iters,
                                          union_deg=union_deg)
            raw_up = best.get("raw_up")
            raw_ang = (ns["angle_to_deg"](np.asarray(raw_up), cam_up)
                       if raw_up is not None else None)
            row.update(verdict="FLOOR", angle_deg=round(best["angle_to_camera_up_deg"], 3),
                       inlier=round(best["inlier_frac"], 4),
                       cam_h=round(best["cam_height_m"], 4),
                       pool=best["support_pool_size"], n_cands=len(cands),
                       raw_angle_deg=round(raw_ang, 3) if raw_ang is not None else None,
                       raw_cam_h=round(best["raw_cam_height_m"], 4)
                       if best.get("raw_cam_height_m") is not None else None,
                       rms_inlier_dist_m=round(best["rms_inlier_dist_m"], 5)
                       if best.get("rms_inlier_dist_m") is not None else None,
                       n_inliers=best.get("n_inliers_refined"),
                       union_n_cands=best.get("union_n_candidates"),
                       union_n_points=best.get("union_n_points"),
                       union_rms_m=(round(best["union_rms_m"], 5)
                                    if best.get("union_rms_m") is not None else None))
            # Would the tighter pool have excluded the plane a 0.2 pool would have found?
            # Reported as a number rather than an opinion about the threshold.
            angles = [c["angle_to_camera_up_deg"] for c in cands]
            best_sup = max(c["inlier_frac"] for c in cands)
            wide = [c for c in cands if c["inlier_frac"] >= 0.2 * best_sup]
            row["best_angle_in_0.2_pool"] = round(
                min(c["angle_to_camera_up_deg"] for c in wide), 2)
            row["min_angle_any_candidate"] = round(min(angles), 2)
        except Exception as e:
            row.update(verdict=type(e).__name__, error=str(e)[:200])
        row["wall_s"] = round(time.time() - t0, 1)
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--support-frac", type=float, default=0.8)
    ap.add_argument("--gate-deg", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--union-deg", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ns = load_from_source()
    rows = []
    for sc in a.scenes:
        rows += run_scene(ns, sc, a.seeds, a.support_frac, a.gate_deg, a.iters,
                          a.union_deg)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print("WROTE", a.out)
    bad = [r for r in rows if r.get("verdict") != "FLOOR"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} runs found a floor")
    for r in bad:
        print("  NOT FLOOR:", r.get("scene"), r.get("seed"), r.get("verdict"),
              r.get("error", "")[:120])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
