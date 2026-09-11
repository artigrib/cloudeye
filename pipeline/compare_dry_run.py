#!/usr/bin/env python3
"""Control checks for the dry run: compare what the pipeline produced against the
already-published reference outputs for the same scene.

Honest scope, stated up front and repeated in the JSON it writes: LAYERS / REACH / EXPORT
read the SAME `results/scenes_out/<scene>/` the references were built from (NVBLOX cannot be
re-run here - this VPS has no GPU), so those three are a determinism / reproducibility check,
not an independent validation of the geometry. PACK is the one link that is genuinely
recomputed from the 1148 per-view npz, so its comparison is a real end-to-end check.

Nothing here rounds, clips or tolerates a difference: every number is reported as measured.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np


def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def nan_aware_max_abs_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """max|a-b| over cells finite in both, plus the count of cells where NaN-ness differs."""
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    nan_pattern_mismatch = int((fa != fb).sum())
    diff = np.abs(a[both] - b[both]) if both.any() else np.array([0.0])
    return {"shape": list(a.shape),
            "n_cells": int(a.size),
            "n_finite_both": int(both.sum()),
            "nan_pattern_mismatch_cells": nan_pattern_mismatch,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-name", default="own_0901_173903__step2")
    ap.add_argument("--nvblox-root", default="var/nvblox_v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sd = Path(a.scene_dir)
    nv = Path(a.nvblox_root)
    short = a.scene_name.replace("__optimal_step2", "").replace("__step2", "")
    R = {"scene": a.scene_name,
         "caveat": ("LAYERS/REACH/EXPORT read the same scenes_out the references were built "
                    "from; those three are a determinism check, not independent validation. "
                    "PACK is recomputed from the per-view npz and is a real check."),
         "checks": {}}

    # 1. PACK - genuinely recomputed
    ours = sd / "packed" / a.scene_name / "depth_u16.npy"
    ref = nv / "packed" / a.scene_name / "depth_u16.npy"
    c = {"ours": str(ours), "reference": str(ref)}
    if ours.exists() and ref.exists():
        c["ours_bytes"], c["reference_bytes"] = ours.stat().st_size, ref.stat().st_size
        c["ours_md5"], c["reference_md5"] = md5(ours), md5(ref)
        c["md5_equal"] = c["ours_md5"] == c["reference_md5"]
        A = np.load(ours, mmap_mode="r"); B = np.load(ref, mmap_mode="r")
        c["shape_ours"], c["shape_ref"] = list(A.shape), list(B.shape)
        if A.shape == B.shape:
            m = 0
            for i in range(0, A.shape[0], 128):     # chunked: 350 MB uint16
                m = max(m, int(np.abs(A[i:i+128].astype(np.int32)
                                      - B[i:i+128].astype(np.int32)).max()))
            c["max_abs_diff_mm"] = m
            c["n_mismatched_voxels"] = None if m == 0 else int(
                sum(int((A[i:i+128] != B[i:i+128]).sum()) for i in range(0, A.shape[0], 128)))
        mo = json.loads((ours.parent / "meta.json").read_text())
        mr = json.loads((ref.parent / "meta.json").read_text())
        c["n_views_ours"], c["n_views_ref"] = mo["n_views"], mr["n_views"]
    else:
        c["error"] = "missing input"
    R["checks"]["PACK_depth_u16"] = c

    # 2. LAYERS - esdf slice + unobserved mask
    lo = sd / "layers" / short
    lr = nv / "results" / "frontend_layers" / short
    c = {"ours": str(lo), "reference": str(lr)}
    if (lo / "esdf_slice_0.3m.npy").exists() and (lr / "esdf_slice_0.3m.npy").exists():
        c["esdf_slice"] = nan_aware_max_abs_diff(
            np.load(lo / "esdf_slice_0.3m.npy").astype(np.float64),
            np.load(lr / "esdf_slice_0.3m.npy").astype(np.float64))
        uo = np.load(lo / "unobserved_mask.npy")
        ur = np.load(lr / "unobserved_mask.npy")
        if uo.shape == ur.shape:
            mism = int((uo != ur).sum())
            c["unobserved_mask"] = {"shape": list(uo.shape), "n_cells": int(uo.size),
                                    "mismatched_cells": mism,
                                    "mismatched_frac": mism / uo.size,
                                    "unobserved_frac_ours": float(uo.mean()),
                                    "unobserved_frac_ref": float(ur.mean())}
        else:
            c["unobserved_mask"] = {"shape_mismatch": [list(uo.shape), list(ur.shape)]}
    else:
        c["error"] = "missing input"
    R["checks"]["LAYERS"] = c

    # 3. REACH
    ro = sd / "reach" / "reachability.json"
    rr = nv / "results" / "reach" / short / "reachability.json"
    c = {"ours": str(ro), "reference": str(rr)}
    if ro.exists() and rr.exists():
        do, dr = json.loads(ro.read_text()), json.loads(rr.read_text())
        for k in ["slice_shape", "obstacle_cells", "nav2d_clearance_max_m", "observed_frac",
                  "mesh_vertices_in_band", "obstacle_cells_before_component_filter"]:
            c[k] = {"ours": do.get(k), "reference": dr.get(k), "equal": do.get(k) == dr.get(k)}
    else:
        c["error"] = "missing input"
    R["checks"]["REACH"] = c

    # 4. EXPORT - USD triangles + size
    eo = sd / "export" / "export_summary.json"
    er = nv / "isaac_export" / "export_summary.json"
    c = {"ours": str(eo), "reference": str(er)}
    if eo.exists() and er.exists():
        row_o = next((r for r in json.loads(eo.read_text())
                      if r.get("scene") == a.scene_name), None)
        row_r = next((r for r in json.loads(er.read_text())
                      if r.get("scene") == a.scene_name), None)
        for k in ["source_mesh", "triangles_in", "triangles_after_decimation", "triangles_out",
                  "vertices_out", "usd_bytes", "glb_bytes", "wall_s"]:
            c[k] = {"ours": (row_o or {}).get(k), "reference": (row_r or {}).get(k),
                    "equal": (row_o or {}).get(k) == (row_r or {}).get(k)}
        usd_o = sd / "export" / short / "scene.usd"
        usd_r = nv / "isaac_export" / short / "scene.usd"
        if usd_o.exists() and usd_r.exists():
            c["usd_file_bytes"] = {"ours": usd_o.stat().st_size,
                                   "reference": usd_r.stat().st_size,
                                   "equal": usd_o.stat().st_size == usd_r.stat().st_size}
            c["usd_md5"] = {"ours": md5(usd_o), "reference": md5(usd_r)}
            c["usd_md5"]["equal"] = c["usd_md5"]["ours"] == c["usd_md5"]["reference"]
        c["usd_validation_errors_ours"] = (row_o or {}).get("usd_validation", {}).get("n_errors")
    else:
        c["error"] = "missing input"
    R["checks"]["EXPORT"] = c

    txt = json.dumps(R, indent=2, ensure_ascii=False)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
