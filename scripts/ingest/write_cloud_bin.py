#!/usr/bin/env python3
"""Pack a coloured point cloud into the binary the viewer's worker parses.

Not JSON and not ascii PLY on purpose: at a couple of million points JSON costs roughly 6x
the bytes and has to be parsed a character at a time, and ascii PLY has the same problem plus
a header the browser would have to tokenise. This layout is two contiguous typed-array slices
the worker can hand to the main thread as transferables with no per-point work at all.

Layout, little-endian (CEPC v1):
    0   char[4]   magic "CEPC"
    4   uint32    version = 1
    8   uint32    count
   12   uint32    flags        bit0 = rgb present
   16   float32[6] bbox  xmin, ymin, zmin, xmax, ymax, zmax
   40   float32[3*count]  xyz    (u, v, height) - floor = 0, Z-up
        uint8[3*count]    rgb    starts at 40 + 12*count, no padding needed:
                                 12*count is a multiple of 4, so the Float32Array view is
                                 always aligned; the Uint8Array after it needs no alignment.
Total = 40 + 15*count.
"""
from __future__ import annotations

import argparse, json, struct
from pathlib import Path

import numpy as np

MAGIC = b"CEPC"
VERSION = 1
HEADER_BYTES = 40


def write_cloud_bin(xyz: np.ndarray, rgb: np.ndarray | None, out: Path) -> dict:
    xyz = np.ascontiguousarray(xyz, dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    count = xyz.shape[0]
    if rgb is not None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.shape != (count, 3):
            raise ValueError(f"rgb must be ({count},3), got {rgb.shape}")
    bbox = np.concatenate([xyz.min(axis=0), xyz.max(axis=0)]).astype("<f4")

    with open(out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<III", VERSION, count, 1 if rgb is not None else 0))
        fh.write(bbox.tobytes())
        fh.write(xyz.tobytes())
        if rgb is not None:
            fh.write(rgb.tobytes())

    size = out.stat().st_size
    expect = HEADER_BYTES + (15 if rgb is not None else 12) * count
    if size != expect:
        raise RuntimeError(f"wrote {size} bytes, layout says {expect}")
    return {"path": str(out), "count": int(count), "bytes": size,
            "bytes_per_point": size / count if count else 0,
            "bbox_min": bbox[:3].tolist(), "bbox_max": bbox[3:].tolist(),
            "has_rgb": rgb is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="npz with xyz (N,3) float32 and rgb (N,3) uint8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.npz)
    stats = write_cloud_bin(d["xyz"], d["rgb"] if "rgb" in d.files else None, Path(a.out))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
