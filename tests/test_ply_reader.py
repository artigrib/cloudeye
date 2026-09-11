"""ply_reader parses the narrow PLY subset Open3D's write_point_cloud emits (see its
module docstring) - tested against synthetic files built here (both binary and ascii)
rather than checked-in binaries, since the point is exercising the parser's header/
dtype logic, not storing real point-cloud data in the repo."""

import struct

import numpy as np
import pytest

from app.services.ply_reader import PlyParseError, read_ply


def _write_binary_ply(path, positions: np.ndarray, colors: np.ndarray | None) -> None:
    n = len(positions)
    lines = [b"ply\n", b"format binary_little_endian 1.0\n", f"element vertex {n}\n".encode()]
    lines += [b"property double x\n", b"property double y\n", b"property double z\n"]
    if colors is not None:
        lines += [b"property uchar red\n", b"property uchar green\n", b"property uchar blue\n"]
    lines.append(b"end_header\n")
    with open(path, "wb") as f:
        f.writelines(lines)
        for i in range(n):
            f.write(struct.pack("<ddd", *positions[i]))
            if colors is not None:
                f.write(struct.pack("<BBB", *colors[i]))


def _write_ascii_ply(path, positions: np.ndarray, colors: np.ndarray | None) -> None:
    n = len(positions)
    lines = ["ply", "format ascii 1.0", f"element vertex {n}"]
    lines += ["property float x", "property float y", "property float z"]
    if colors is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    for i in range(n):
        row = list(positions[i])
        if colors is not None:
            row += list(colors[i])
        lines.append(" ".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


POSITIONS = np.array([[0.1, 0.2, 0.3], [-1.5, 2.5, -3.5], [10.0, 0.0, -10.0]])
COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)


def test_binary_with_color(tmp_path):
    p = tmp_path / "cloud.ply"
    _write_binary_ply(p, POSITIONS, COLORS)
    pc = read_ply(p)
    assert pc.positions.shape == (3, 3)
    np.testing.assert_allclose(pc.positions, POSITIONS, atol=1e-6)
    assert pc.colors is not None
    np.testing.assert_array_equal(pc.colors, COLORS)


def test_binary_without_color(tmp_path):
    p = tmp_path / "cloud.ply"
    _write_binary_ply(p, POSITIONS, None)
    pc = read_ply(p)
    np.testing.assert_allclose(pc.positions, POSITIONS, atol=1e-6)
    assert pc.colors is None


def test_ascii_with_color(tmp_path):
    p = tmp_path / "cloud.ply"
    _write_ascii_ply(p, POSITIONS, COLORS)
    pc = read_ply(p)
    np.testing.assert_allclose(pc.positions, POSITIONS, atol=1e-4)
    np.testing.assert_array_equal(pc.colors, COLORS)


def test_missing_xyz_property_raises(tmp_path):
    p = tmp_path / "bad.ply"
    p.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        b"end_header\n0.0 0.0\n"
    )
    with pytest.raises(PlyParseError, match="missing required"):
        read_ply(p)


def test_not_a_ply_file_raises(tmp_path):
    p = tmp_path / "notply.txt"
    p.write_text("hello\n")
    with pytest.raises(PlyParseError, match="not a PLY file"):
        read_ply(p)


def test_real_pipeline_fixture_if_available(tmp_path):
    """Cross-check against a real Open3D-written .ply from the GPU backup, if this
    checkout has one - skipped otherwise (that backup directory isn't checked into
    the repo, see ~/gpu-backup/ in the project README)."""
    from pathlib import Path

    real = Path.home() / "gpu-backup/data/output_horizontal/run2/scene_objects/sink_1.ply"
    if not real.is_file():
        pytest.skip("no local gpu-backup fixture available")
    pc = read_ply(real)
    assert pc.positions.shape[0] == 73
    assert pc.colors is not None
    assert pc.colors.shape == (73, 3)
