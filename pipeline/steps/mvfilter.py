#!/usr/bin/env python3
"""Multi-view consistency filter over a PACKED depth stack, applied BEFORE nvblox fuses it.

    mvfilter.py --packed <scene_dir>/packed/<scene> --m 2 --tau 0.10 \
                --neighbours nearest --k 8

A point that only one view ever saw cannot be distinguished from reconstruction drift, and
nvblox fuses it into surface either way - which is how a band cell becomes a wall and a wall
becomes a robot's dead end. This step asks each valid pixel whether the cameras that should
also see it agree, and zeroes the ones that fail. Rejected pixels become 0, which
`pack_scenes.py` already defines as "invalid" and `nvblox_scenes.py:459-460` already skips,
so nothing downstream needs a new convention.

The rule is ported unchanged from the CPU experiment in var/scratch/mvfilter/mvcore.py
(`neighbours()` and `vote_view()`), whose results are written up in
var/scratch/mvfilter/REPORT.md. One-sided, exactly as specified there:

    delta = z_proj - z_own    |delta| <= tau -> AGREE (a vote)
                               delta  >  tau -> disagree (behind the neighbour's surface)
                               delta  < -tau -> no vote (in front of it, i.e. occluding)

A pixel is kept iff it collects at least `m` votes among its `k` neighbour views. The
occlusion asymmetry is the point: a point in FRONT of a neighbour's surface is not evidence
against itself, it is a thing the neighbour has not seen through yet.

`--neighbours` selects which views are asked, and it matters more than tau does:

    nearest      the k nearest camera centres, no exclusion
    baseline15   the k nearest at least 0.15 m away - a real baseline, so agreement means
                 triangulated agreement rather than two frames of the same viewpoint
    framegap8    the k nearest at least 8 raw video frames apart

At 15 fps `nearest` picks a view's own temporal neighbours at near-zero baseline, which is
the WEAKEST form of this test: REPORT.md section 1 measures residual mean 0.060 m for
`nearest` against 0.098 m for `baseline15` on hero-15fps. All three are ported so the
neighbour rule is a parameter of the experiment and not a thing rewritten to change it.

Reads `depth_u16.npy` (uint16 millimetres, 0 = invalid) + `meta.json` from a PACK output and
writes `depth_u16_mvfilter.npy` beside it, same shape and dtype, plus `mvfilter.json`
recording the parameters and what they cost. The input is never modified: the unfiltered
stack stays on disk so the same run can fuse both arms and compare.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

#: Written next to the input. Named for the step, not for its parameters: the parameters are
#: recorded inside mvfilter.json, and a filename carrying them invites two stacks that differ
#: in a way no reader can see (packed_conf/ already learned this - see nvblox_v2/pack_conf.sh).
OUT_DEPTH = "depth_u16_mvfilter.npy"
OUT_META = "mvfilter.json"

#: mvcore.py's own baseline and frame-gap thresholds, kept as constants so the two
#: implementations can be diffed rather than compared by eye.
BASELINE_MIN_M = 0.15
FRAME_GAP_MIN = 8

NEIGHBOUR_KINDS = ("nearest", "baseline15", "framegap8")


def neighbour_table(centres: np.ndarray, raw_frame_index: np.ndarray, kind: str,
                    k: int) -> np.ndarray:
    """(n, k) int32 neighbour view indices, -1 padded. Ported from mvcore.py:46 neighbours().

    -1 means "this view has fewer than k eligible neighbours", which only `baseline15` and
    `framegap8` can produce. It is padding, not a view: a row of -1 is a view nobody can
    corroborate, and its pixels collect no votes and are dropped. That is the honest answer
    for a view with no usable partner, and `k_eff_min` in the metadata says how often it
    happened rather than leaving it to be inferred from the kept fraction.
    """
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if kind == "baseline15":
        d[d < BASELINE_MIN_M] = np.inf
    elif kind == "framegap8":
        gap = np.abs(raw_frame_index[:, None] - raw_frame_index[None, :])
        d[gap < FRAME_GAP_MIN] = np.inf
    elif kind != "nearest":
        raise ValueError(f"unknown --neighbours {kind!r}; expected one of {NEIGHBOUR_KINDS}")
    order = np.argsort(d, axis=1)[:, :k]
    out = order.astype(np.int32)
    out[np.take_along_axis(d, order, axis=1) == np.inf] = -1
    return out


def keep_mask_for_view(i, depth, poses, intrinsics, nbr_row, m, tau):
    """(keep 2-D bool, n_valid, votes-per-pixel histogram, k_eff) for one view.

    Ported from mvcore.py:65 vote_view(), reduced to the single (m, tau) this step applies:
    the experiment accumulated three taus in one pass to build a grid, and a run that filters
    needs one threshold, not a grid.
    """
    h, w = depth.shape[1:]
    z = np.asarray(depth[i], np.float32) / 1000.0
    valid = z > 0
    yy, xx = np.nonzero(valid)
    n_valid = yy.size
    keep2d = np.zeros((h, w), bool)
    if n_valid == 0:
        return keep2d, 0, np.zeros(1, np.int64), 0

    zz = z[yy, xx].astype(np.float64)
    ki = intrinsics[i]
    cam = np.stack([(xx - ki[0, 2]) / ki[0, 0] * zz,
                    (yy - ki[1, 2]) / ki[1, 1] * zz, zz], 1)
    world = cam @ poses[i, :3, :3].T + poses[i, :3, 3]

    votes = np.zeros(n_valid, np.uint8)
    k_eff = 0
    for j in nbr_row:
        if j < 0:
            continue
        k_eff += 1
        rj, tj = poses[j, :3, :3], poses[j, :3, 3]
        cj = (world - tj) @ rj
        zc = cj[:, 2]
        # Behind the neighbour's optical centre: it cannot have an opinion about this point.
        front = zc > 1e-6
        oi = np.nonzero(front)[0]
        if oi.size == 0:
            continue
        kj = intrinsics[j]
        u = np.round(cj[oi, 0] / zc[oi] * kj[0, 0] + kj[0, 2]).astype(np.int32)
        v = np.round(cj[oi, 1] / zc[oi] * kj[1, 1] + kj[1, 2]).astype(np.int32)
        ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        oi = oi[ins]
        if oi.size == 0:
            continue
        z_own = np.asarray(depth[j][v[ins], u[ins]], np.float64) / 1000.0
        has_depth = z_own > 0
        oi = oi[has_depth]
        if oi.size == 0:
            continue
        # One-sided: |delta| <= tau votes, delta < -tau (in front) abstains rather than
        # counting against the point. See the module docstring.
        votes[oi] += np.abs(zc[oi] - z_own[has_depth]) <= tau

    kept = votes >= m
    keep2d[yy[kept], xx[kept]] = True
    return keep2d, n_valid, np.bincount(votes, minlength=1), k_eff


def run(packed: Path, m: int, tau: float, kind: str, k: int,
        depth_file: str = "depth_u16.npy", out_depth: str = OUT_DEPTH,
        out_meta: str = OUT_META, progress_every: int = 100) -> dict:
    meta = json.loads((packed / "meta.json").read_text())
    views = meta["views"]
    n = len(views)
    h, w = meta["shape_hw"]
    depth = np.load(packed / depth_file, mmap_mode="r")
    if depth.shape != (n, h, w):
        raise SystemExit(f"mvfilter: {depth_file} is {depth.shape}, meta.json says "
                         f"{(n, h, w)} - refusing to filter a stack that does not match "
                         f"its own metadata")
    if depth.dtype != np.uint16:
        raise SystemExit(f"mvfilter: {depth_file} is {depth.dtype}, expected uint16 "
                         f"millimetres (pack_scenes.py:73)")
    if k >= n:
        raise SystemExit(f"mvfilter: --k {k} needs at least {k + 1} views, this scene has {n}")

    poses = np.array([v["camera_pose"] for v in views], np.float64)
    intrinsics = np.array([v["intrinsics"] for v in views], np.float64)
    raw_frame_index = np.array([v["raw_frame_index"] for v in views], np.int64)
    nbr = neighbour_table(poses[:, :3, 3], raw_frame_index, kind, k)
    k_eff_row = (nbr >= 0).sum(1)

    # Written straight to disk rather than assembled in RAM: hero-15fps's stack is 350 MB and
    # this step runs on the same VPS as the API.
    out = np.lib.format.open_memmap(packed / out_depth, mode="w+", dtype=np.uint16,
                                    shape=(n, h, w))
    t0 = time.time()
    valid_before = valid_after = 0
    per_view_kept = np.zeros(n, np.float64)
    vote_hist = np.zeros(k + 1, np.int64)
    for i in range(n):
        keep2d, n_valid, hist, _ = keep_mask_for_view(i, depth, poses, intrinsics,
                                                      nbr[i], m, tau)
        out[i] = np.where(keep2d, depth[i], np.uint16(0))
        n_kept = int(keep2d.sum())
        valid_before += n_valid
        valid_after += n_kept
        per_view_kept[i] = (n_kept / n_valid) if n_valid else float("nan")
        vote_hist[:hist.size] += hist[:k + 1]
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n} views, kept {100 * valid_after / max(valid_before, 1):.2f}% "
                  f"so far [{time.time() - t0:.0f}s]", flush=True)
    out.flush()
    wall_s = time.time() - t0

    finite = per_view_kept[np.isfinite(per_view_kept)]
    doc = {
        "source": str(packed / depth_file),
        "output": str(packed / out_depth),
        "rule": "one-sided multi-view agreement vote, ported from "
                "var/scratch/mvfilter/mvcore.py (neighbours, vote_view); measured in "
                "var/scratch/mvfilter/REPORT.md",
        "m": m, "tau": tau, "neighbours": kind, "k": k,
        "n_views": n, "shape_hw": [h, w],
        "valid_px_before": int(valid_before),
        "valid_px_after": int(valid_after),
        "kept_frac": round(valid_after / valid_before, 6) if valid_before else None,
        "kept_pct": round(100 * valid_after / valid_before, 3) if valid_before else None,
        "per_view_kept_frac": {
            "min": round(float(finite.min()), 6) if finite.size else None,
            "median": round(float(np.median(finite)), 6) if finite.size else None,
            "max": round(float(finite.max()), 6) if finite.size else None,
            "views_fully_dropped": int((finite == 0).sum()),
        },
        "k_eff": {"min": int(k_eff_row.min()), "mean": round(float(k_eff_row.mean()), 3),
                  "rows_below_k": int((k_eff_row < k).sum())},
        # How many neighbours agreed, over every valid pixel. The m-th bucket onwards is what
        # survived, so a run can be re-thresholded on paper before it is re-run on a box.
        "vote_histogram": [int(x) for x in vote_hist],
        "wall_s": round(wall_s, 2),
    }
    (packed / out_meta).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packed", required=True,
                    help="a PACK output directory: <scene_dir>/packed/<scene>, holding "
                         "depth_u16.npy and meta.json")
    ap.add_argument("--m", type=int, default=2,
                    help="agreeing neighbour views a pixel needs to survive. 2 is the only "
                         "setting in REPORT.md's 9-config grid under which no control object "
                         "loses all of its points")
    ap.add_argument("--tau", type=float, default=0.10,
                    help="agreement tolerance in metres on the reprojected depth difference")
    ap.add_argument("--neighbours", default="nearest", choices=NEIGHBOUR_KINDS,
                    help="which views are asked; see the module docstring - `nearest` is the "
                         "weakest of the three at high frame rates")
    ap.add_argument("--k", type=int, default=8, help="neighbour views per view")
    ap.add_argument("--depth-file", default="depth_u16.npy",
                    help="the stack to filter, relative to --packed")
    ap.add_argument("--out-depth", default=OUT_DEPTH)
    ap.add_argument("--out-meta", default=OUT_META)
    ap.add_argument("--progress-every", type=int, default=100,
                    help="views between progress lines; 0 silences them")
    a = ap.parse_args()

    packed = Path(a.packed)
    for f in ("meta.json", a.depth_file):
        if not (packed / f).exists():
            print(f"mvfilter: {packed / f} does not exist - --packed must be a PACK output "
                  f"directory", file=sys.stderr)
            return 2
    if a.m < 1:
        print(f"mvfilter: --m {a.m} keeps everything; omit the filter instead",
              file=sys.stderr)
        return 2
    if a.tau <= 0:
        print(f"mvfilter: --tau {a.tau} must be positive", file=sys.stderr)
        return 2

    print(f"mvfilter: {packed}/{a.depth_file}  m={a.m} tau={a.tau} "
          f"neighbours={a.neighbours} k={a.k}", flush=True)
    doc = run(packed, a.m, a.tau, a.neighbours, a.k, a.depth_file, a.out_depth, a.out_meta,
              a.progress_every)
    print(f"mvfilter: kept {doc['valid_px_after']:,}/{doc['valid_px_before']:,} valid px "
          f"= {doc['kept_pct']}%  (per-view min {doc['per_view_kept_frac']['min']}, "
          f"median {doc['per_view_kept_frac']['median']}, "
          f"{doc['per_view_kept_frac']['views_fully_dropped']} views emptied)  "
          f"k_eff min {doc['k_eff']['min']} mean {doc['k_eff']['mean']}  "
          f"[{doc['wall_s']}s]")
    print(f"mvfilter: wrote {doc['output']} and {packed / a.out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
