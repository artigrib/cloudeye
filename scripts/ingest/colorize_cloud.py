#!/usr/bin/env python3
"""Colour the 15 fps hero run's per-view points from the source video and reduce them to a
voxel-downsampled cloud in the nvblox slice frame (floor = 0, Z-up).

Colour source: the frames of the original video put through MapAnything's OWN preprocessing
("scale short side to 294, centre-crop the long side to 518" ->
`scale=294:523:flags=bicubic,crop=294:518:0:2`). That recipe is not a guess: it was
calibrated in the parallel nvblox session against the real `img_no_norm` images of the 74-view
pipeline run (NCC 0.9637, vs 0.9579 for a plain resize and 0.9389 for pad-to-width), this
session reproduced it byte-for-byte (md5 match on 5 frames) and re-verified the NCC itself -
see scripts/ingest/ncc_gate.py. Because the frame comes out at exactly the (518, 294) grid of
`pts3d`/`mask`, the sampling is a plain `frame[mask]`, with no interpolation to get wrong.

Frame: points land in the frame `results/frontend_layers/<scene>/grid_meta.json` describes -
u along `axis_u`, v along `axis_v`, height above the FITTED floor plane along `up_axis_world`
(the raw MapAnything world frame is not floor-aligned; the floor must be fitted, never
assumed - HANDOFF_nvblox.md). That is the same "floor = 0, Z-up" convention as
`transform_world_to_zup_floor0` in nvblox_v2/isaac_export/*/transform.json.

Downsampling: ~160 M input points do not fit an Open3D voxel hash on this box (23 GB RAM), so
the 0.02 m reduction is done as a two-level numpy reduction - each batch of views reduces to
its own occupied voxels, then the partials are reduced again. Sums are accumulated in float64
and divided by the count, so the result is the voxel MEAN, same as voxel_down_sample.
"""
from __future__ import annotations

import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

VOXEL_M = 0.02


def load_frame_transform(layers_dir: Path):
    """Read the slice frame out of grid_meta.json + floor_plane.json.

    Height is measured from the FITTED FLOOR (floor_plane.json's `point_on_plane`), not from
    grid_meta's `origin_world` - the latter is the slice origin, which sits
    `slice_height_above_floor_m` above the floor.
    """
    gm = json.loads((layers_dir / "grid_meta.json").read_text())
    fp = json.loads((layers_dir / "floor_plane.json").read_text())
    ours = gm["our_grid"] if "our_grid" in gm else gm
    return {
        "origin": np.asarray(ours["origin_world"], dtype=np.float64),
        "axis_u": np.asarray(ours["axis_u"], dtype=np.float64),
        "axis_v": np.asarray(ours["axis_v"], dtype=np.float64),
        "up": np.asarray(fp["normal_up"], dtype=np.float64),
        "floor_point": np.asarray(fp["point_on_plane"], dtype=np.float64),
        "cell": float(ours["cell_m"]),
        "u_range": [float(x) for x in ours["u_range"]],
        "v_range": [float(x) for x in ours["v_range"]],
        "grid_meta": gm,
    }


def to_slice_frame(pts: np.ndarray, T: dict) -> np.ndarray:
    """World (raw MapAnything) -> (u, v, height-above-floor). float32 out, float64 inside."""
    d = pts.astype(np.float64) - T["origin"]
    u = d @ T["axis_u"]
    v = d @ T["axis_v"]
    h = (pts.astype(np.float64) - T["floor_point"]) @ T["up"]
    return np.stack([u, v, h], axis=1).astype(np.float32)


def _reduce(keys: np.ndarray, xyz: np.ndarray, rgb_sum: np.ndarray, cnt: np.ndarray):
    """Sum xyz/rgb per unique voxel key. Inputs already carry per-key sums and counts."""
    uk, inv = np.unique(keys, return_inverse=True)
    n = uk.size
    out_xyz = np.empty((n, 3), dtype=np.float64)
    out_rgb = np.empty((n, 3), dtype=np.float64)
    for c in range(3):
        out_xyz[:, c] = np.bincount(inv, weights=xyz[:, c], minlength=n)
        out_rgb[:, c] = np.bincount(inv, weights=rgb_sum[:, c], minlength=n)
    out_cnt = np.bincount(inv, weights=cnt, minlength=n)
    return uk, out_xyz, out_rgb, out_cnt


def _voxel_keys(p: np.ndarray) -> np.ndarray:
    """Integer voxel index packed into one int64. The +2**20 bias keeps every axis
    non-negative for a scene up to +/- 20971 m, far past any room."""
    idx = np.floor(p / VOXEL_M).astype(np.int64) + (1 << 20)
    return (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]


def process_batch(args):
    npz_paths, frames_dir, T = args
    keys_l, xyz_l, rgb_l, cnt_l = [], [], [], []
    n_valid = n_coloured = 0
    for np_path in npz_paths:
        d = np.load(np_path)
        mask = d["mask"]
        raw = int(d["raw_frame_index"])
        img_path = Path(frames_dir) / f"frame_{raw:06d}.jpg"
        frame = np.asarray(Image.open(img_path).convert("RGB"))  # (518, 294, 3) uint8
        if frame.shape[:2] != mask.shape:
            raise RuntimeError(f"{img_path.name} is {frame.shape[:2]}, mask is {mask.shape}")
        pts = d["pts3d"][mask]
        col = frame[mask].astype(np.float64)
        n_valid += int(mask.sum())
        n_coloured += col.shape[0]
        p = to_slice_frame(pts, T)
        finite = np.isfinite(p).all(axis=1)
        p, col = p[finite], col[finite]
        k, x, r, c = _reduce(_voxel_keys(p), p.astype(np.float64), col,
                             np.ones(p.shape[0], dtype=np.float64))
        keys_l.append(k); xyz_l.append(x); rgb_l.append(r); cnt_l.append(c)
    return (np.concatenate(keys_l), np.concatenate(xyz_l), np.concatenate(rgb_l),
            np.concatenate(cnt_l), n_valid, n_coloured)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-view", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--out", required=True, help="npz with xyz float32 (N,3) + rgb uint8 (N,3)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    T = load_frame_transform(Path(a.layers_dir))
    files = sorted(Path(a.per_view).glob("view_*.npz"))
    print(f"views: {len(files)}   voxel {VOXEL_M} m   workers {a.workers}")
    batches = [(files[i:i + a.batch], a.frames_dir, T) for i in range(0, len(files), a.batch)]

    t0 = time.time()
    parts, n_valid, n_coloured = [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (k, x, r, c, nv, nc) in enumerate(ex.map(process_batch, batches), 1):
            parts.append((k, x, r, c)); n_valid += nv; n_coloured += nc
            print(f"  batch {i:3d}/{len(batches)}  voxels {k.size:>9,}  "
                  f"{time.time()-t0:6.1f}s", flush=True)

    keys = np.concatenate([p[0] for p in parts])
    xyz = np.concatenate([p[1] for p in parts])
    rgb = np.concatenate([p[2] for p in parts])
    cnt = np.concatenate([p[3] for p in parts])
    del parts
    print(f"merging {keys.size:,} partial voxels ...", flush=True)
    _, xyz, rgb, cnt = _reduce(keys, xyz, rgb, cnt)

    xyz = (xyz / cnt[:, None]).astype(np.float32)
    rgb = np.clip(np.round(rgb / cnt[:, None]), 0, 255).astype(np.uint8)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))   # deterministic output order
    xyz, rgb = xyz[order], rgb[order]

    np.savez(a.out, xyz=xyz, rgb=rgb)
    stats = {
        "n_views": len(files),
        "n_points_in": int(n_valid),
        "n_points_out": int(xyz.shape[0]),
        "voxel_m": VOXEL_M,
        "coloured_frac_of_valid": n_coloured / n_valid if n_valid else 0.0,
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "grid_u_range": T["u_range"],
        "grid_v_range": T["v_range"],
        "wall_sec": round(time.time() - t0, 1),
    }
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
