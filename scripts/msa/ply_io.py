"""Minimal reader for the binary_little_endian PLY files this project's `gpu/`
stage writes per detected object (Open3D's writer: double x/y/z + uchar r/g/b, no
other properties). Not a general PLY parser - deliberately narrow, matching the one
format this repo actually produces (see gpu/stage_objects.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_VERTEX_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])


def _read_header(f) -> tuple[int, int]:
    """Returns (vertex_count, header_byte_length)."""
    header_lines = []
    while True:
        line = f.readline().decode("ascii").strip()
        header_lines.append(line)
        if line == "end_header":
            break
    header = "\n".join(header_lines)
    if "format binary_little_endian" not in header:
        raise ValueError(f"only binary_little_endian PLY is supported, got header: {header[:80]!r}")
    vertex_count = 0
    for line in header_lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
    return vertex_count, f.tell()


def read_ply_xyz_rgb(path: Path, *, max_points: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz: (N,3) float64, rgb: (N,3) uint8).

    `max_points` (T15c-prep): a uniform random subsample of at most that many
    points, drawn through a memory map so a 271 MB hero cloud is touched once
    and never fully materialised (the texture bake needs ~2 M points, not 10 M).
    The subsample is deterministic for a given `seed` and keeps file order."""
    with open(path, "rb") as f:
        vertex_count, offset = _read_header(f)
        if max_points is None or vertex_count <= max_points:
            raw = np.frombuffer(f.read(vertex_count * _VERTEX_DTYPE.itemsize), dtype=_VERTEX_DTYPE, count=vertex_count)
        else:
            mm = np.memmap(path, dtype=_VERTEX_DTYPE, mode="r", offset=offset, shape=(vertex_count,))
            idx = np.sort(np.random.default_rng(seed).choice(vertex_count, int(max_points), replace=False))
            raw = np.array(mm[idx])
            del mm
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1)
    rgb = np.stack([raw["r"], raw["g"], raw["b"]], axis=1)
    return xyz, rgb
