"""Minimal PLY point-cloud reader.

Covers exactly what Open3D's `write_point_cloud` emits for this project's pipeline
(`gpu/stage_align.py`, `gpu/stage_objects.py`): a `binary_little_endian` vertex element
with `double x/y/z` plus `uchar red/green/blue` - verified against a real
`aligned_room_run2.ply` (3.17M vertices). ASCII and big-endian are also handled since
the PLY spec allows them and they cost little extra code, but this project's own
writers never produce them.

No third-party dependency (open3d, plyfile, trimesh) is worth adding for read-only
access to a format this narrow - none of those are already installed in this backend's
venv (open3d is GPU-side only, see gpu/job_io.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PlyParseError(Exception):
    """Raised when a .ply file doesn't match the subset of the format this reader
    supports (see module docstring)."""


@dataclass(frozen=True)
class PointCloud:
    positions: np.ndarray  # (N, 3) float32, columns are (x, y, z)
    colors: np.ndarray | None  # (N, 3) uint8 RGB, or None if the file had no color


# PLY property type name -> numpy dtype code. Includes both the short PLY-spec names
# (float, uchar, ...) and the long ones some writers use (float32, uint8, ...).
_NP_TYPE = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def read_ply(path: str | Path) -> PointCloud:
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise PlyParseError(f"{path}: not a PLY file (missing 'ply' magic line)")

        fmt: str | None = None
        properties: list[tuple[str, str]] = []
        vertex_count = 0
        in_vertex_element = False

        while True:
            line = f.readline()
            if not line:
                raise PlyParseError(f"{path}: truncated header (no end_header)")
            line = line.strip()
            if line.startswith(b"format"):
                fmt = line.split()[1].decode("ascii")
            elif line.startswith(b"element"):
                parts = line.split()
                in_vertex_element = parts[1] == b"vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif line.startswith(b"property") and in_vertex_element:
                parts = line.split()
                if parts[1] == b"list":
                    raise PlyParseError(f"{path}: list properties on vertex element unsupported")
                properties.append((parts[2].decode("ascii"), parts[1].decode("ascii")))
            elif line == b"end_header":
                break

        names = [name for name, _ in properties]
        for required in ("x", "y", "z"):
            if required not in names:
                raise PlyParseError(f"{path}: vertex element missing required property '{required}'")
        has_color = all(c in names for c in ("red", "green", "blue"))

        if fmt == "ascii":
            positions = np.empty((vertex_count, 3), dtype=np.float32)
            colors = np.empty((vertex_count, 3), dtype=np.uint8) if has_color else None
            idx = {n: i for i, n in enumerate(names)}
            for row in range(vertex_count):
                fields = f.readline().split()
                positions[row] = (
                    float(fields[idx["x"]]),
                    float(fields[idx["y"]]),
                    float(fields[idx["z"]]),
                )
                if has_color and colors is not None:
                    colors[row] = (
                        int(fields[idx["red"]]),
                        int(fields[idx["green"]]),
                        int(fields[idx["blue"]]),
                    )
            return PointCloud(positions=positions, colors=colors)

        if fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt == "binary_little_endian" else ">"
            for _, ty in properties:
                if ty not in _NP_TYPE:
                    raise PlyParseError(f"{path}: unsupported property type '{ty}'")
            dtype = np.dtype([(name, endian + _NP_TYPE[ty]) for name, ty in properties])

            raw = f.read(dtype.itemsize * vertex_count)
            if len(raw) < dtype.itemsize * vertex_count:
                raise PlyParseError(f"{path}: truncated vertex data")
            records = np.frombuffer(raw, dtype=dtype, count=vertex_count)

            positions = np.stack(
                [records["x"], records["y"], records["z"]], axis=1
            ).astype(np.float32)
            colors = None
            if has_color:
                colors = np.stack(
                    [records["red"], records["green"], records["blue"]], axis=1
                ).astype(np.uint8)
            return PointCloud(positions=positions, colors=colors)

        raise PlyParseError(f"{path}: unsupported format '{fmt}'")
