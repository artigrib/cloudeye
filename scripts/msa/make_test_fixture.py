#!/usr/bin/env python3
"""One-off generator for tests/fixtures/msa/02_modular_home/ - run manually, not
part of the test suite itself. Pulls occupancy.npy/occupancy_meta.json unmodified
from the real `02_modular_home` scene (e1e21716-841e-449e-b973-07c9020094e2, see
docs/DECISIONS.md) and writes a compact `scene_objects_hulls.json` (convex-hull
XZ vertices per object, NOT the full multi-hundred-MB per-object point clouds -
those can't live in git; oriented-rect-from-hull logic itself is already unit-
tested against synthetic point sets in tests/test_msa_geometry.py, so a
precomputed hull is a faithful, honest stand-in here). See fixtures README.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from scripts.msa.assets import mean_color_uint8
from scripts.msa.ply_io import read_ply_xyz_rgb

SRC = Path("var/uploads/scenes/e1e21716-841e-449e-b973-07c9020094e2")
DST = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "msa" / "02_modular_home"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "occupancy.npy", DST / "occupancy.npy")
    shutil.copy(SRC / "occupancy_meta.json", DST / "occupancy_meta.json")

    (DST / "scene_meta.json").write_text(
        json.dumps({"floor_y": 0.0, "ceiling_y": 2.7076803001776657, "source": "DB scene e1e21716, 2026-09-05"}, indent=2)
    )

    objects = json.loads((SRC / "scene_objects" / "scene_objects.json").read_text())
    compact = []
    for obj in objects:
        xyz, rgb = read_ply_xyz_rgb(SRC / obj["point_cloud_path"])
        xz = xyz[:, [0, 2]]
        if len(np.unique(xz, axis=0)) >= 3:
            hull = ConvexHull(xz)
            hull_xz = xz[hull.vertices].tolist()
        else:
            hull_xz = xz.tolist()
        compact.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "hull_xz": hull_xz,
                "bbox_min": obj["bbox_min"],
                "bbox_max": obj["bbox_max"],
                "color_rgb": list(mean_color_uint8(rgb)),
            }
        )
    (DST / "scene_objects_hulls.json").write_text(json.dumps(compact, indent=2))
    print(f"wrote {len(compact)} object hulls, occupancy {DST / 'occupancy.npy'}")


if __name__ == "__main__":
    main()
