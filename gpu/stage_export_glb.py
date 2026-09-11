#!/usr/bin/env python3
"""Stage 7: export the aligned point cloud as a points-primitive GLB.

Runs under the mapanything venv. This is a colored POINT CLOUD in glTF form, not a
triangulated mesh - there is no meshing step in this pipeline yet (per-object meshing
was flagged as real follow-up work, not silently claimed as done). It exists so
`scenes.mesh_path` / `GET /api/scenes/{id}/mesh` have something genuinely useful to
serve that's in the SAME coordinate frame as the objects and occupancy grid.

Deliberately does NOT reuse the legacy `demo_images_only_inference.py --save_glb` path
from earlier in this project - that runs a second, independent `.infer()` call, which
lands in a different, non-co-registered world frame. Exactly the bug this whole
pipeline redesign exists to avoid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import job_paths, load_params  # noqa: E402

import numpy as np
import open3d as o3d
import trimesh

# Fallback only for a manual/direct stage invocation with no params.json override -
# app/services/pipeline_orchestrator.py always sets glb_max_points explicitly (to
# settings.glb_max_points, default 400_000, matching the frontend's render budget).
DEFAULT_MAX_POINTS = 1_500_000


def main() -> None:
    job_dir = sys.argv[1]
    paths = job_paths(job_dir)
    params = load_params(job_dir)
    max_points = int(params.get("glb_max_points", DEFAULT_MAX_POINTS))

    pcd = o3d.io.read_point_cloud(str(paths["aligned_ply"]))
    n_in = len(pcd.points)
    print(f"loaded {n_in} points from {paths['aligned_ply'].name}")

    if n_in > max_points:
        # Binary-search the voxel size that gets us under max_points, rather than
        # guessing one - point density varies a lot between rooms/captures.
        lo, hi = 0.001, 0.5
        for _ in range(20):
            mid = (lo + hi) / 2
            n = len(pcd.voxel_down_sample(mid).points)
            if n > max_points:
                lo = mid
            else:
                hi = mid
        pcd = pcd.voxel_down_sample(hi)
        print(f"voxel-downsampled to {len(pcd.points)} points (voxel_size={hi:.4f})")

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.shape[0] != points.shape[0]:
        colors = np.ones_like(points) * 0.5
    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)

    cloud = trimesh.PointCloud(vertices=points, colors=colors_u8)
    cloud.export(str(paths["glb"]))

    print(f"DONE: {paths['glb'].name} ({len(points)} points, {paths['glb'].stat().st_size} bytes)")


if __name__ == "__main__":
    main()
