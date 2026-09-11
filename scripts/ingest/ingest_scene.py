#!/usr/bin/env python3
"""Run the application's OWN ingest over a prepared scene directory.

This is a driver, not a second ingest: every step below is the same call, in the same order,
that `app/services/pipeline_orchestrator.py::_run` makes after `gpu_client.fetch_results`
returns - `parse_scene_objects`, `read_alignment`, `read_grid_metadata`, `read_camera_track`,
`choose_robot_start`, `read_vocab_*`, `validate_scene`, `find_oversized_objects`,
`pathfinding.resolve_robot_start`, `apply_scene_artifacts`, `replace_scene_objects`,
`mark_done`. Nothing here parses an artifact itself.

What it replaces is only the part that cannot apply: the GPU round-trip. The 15 fps run was
computed on a rented B200 that no longer exists, so the artifacts are staged on disk by
scripts/ingest/build_scene_dir.py instead of rsynced off a live box.
"""
from __future__ import annotations

import argparse, asyncio, json, uuid
from pathlib import Path

import numpy as np

from app.config import settings
from app.database import async_session_maker
from app.robots import resolve_radius_m
from app.services import pathfinding, project_service, scene_ingest, scene_service, video_service
from app.services.footprint import ObjectFootprint
from app.services.pathfinding import OccupancyGrid
from app.services.scene_validator import find_oversized_objects, validate_scene


def preflight(staged: Path) -> None:
    """Run the app's own readers + validator against the staged dir BEFORE any DB row is
    created, so a rejected scene leaves nothing behind to roll back. Same calls, same order
    as the real sequence below - just done twice, cheaply."""
    objects = scene_ingest.parse_scene_objects(
        staged / "scene_objects" / "scene_objects.json", staged)
    report = validate_scene(scene_ingest.read_alignment(staged),
                            objects,
                            scene_ingest.read_grid_metadata(staged),
                            scene_ingest.read_camera_track(staged))
    print("preflight validation:", report.summary())
    if not report.passed:
        raise SystemExit(f"staged dir fails validation, nothing written: {report.summary()}")


async def run(name: str, video_path: Path, staged: Path, label: str | None) -> dict:
    preflight(staged)
    async with async_session_maker() as session:
        project = await project_service.create_project(
            session, name=name, description="hero 15 fps (step 2, 1148 views) ingested from "
            "the 2026-09-08 B200 run; colour from the source video, layers from nvblox_v2")
        video = await video_service.create_video(
            session, project_id=project.id, filename=video_path.name, filepath=video_path,
            size_bytes=video_path.stat().st_size, fmt="mp4",
            duration_sec=76.528133, resolution="1080x1920")
        scene, created = await scene_service.get_or_create_scene(
            session, video_id=video.id, project_id=project.id, label=label)
        assert created, "scene already existed for a freshly created video"

        local_dir = scene_service.scene_dir(scene.id)
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(local_dir)
        print(f"scene {scene.id}  dir {local_dir}")

        # --- from here down: pipeline_orchestrator._run's post-fetch sequence, verbatim ---
        objects = scene_ingest.parse_scene_objects(
            local_dir / "scene_objects" / "scene_objects.json", local_dir)
        alignment = scene_ingest.read_alignment(local_dir)
        grid = scene_ingest.read_grid_metadata(local_dir)
        cameras = scene_ingest.read_camera_track(local_dir)
        robot_start = scene_ingest.choose_robot_start(cameras, grid)
        vocab_source = scene_ingest.read_vocab_source(local_dir)
        vocab_model = scene_ingest.read_vocab_model(local_dir)
        vocab_entries = scene_ingest.read_vocab_entries(local_dir)

        report = validate_scene(alignment, objects, grid, cameras)
        print("validation:", report.summary())
        if not report.passed:
            raise SystemExit(f"scene failed validation: {report.summary()}")

        oversized = find_oversized_objects(objects)
        oversized_object_warning = "; ".join(oversized) if oversized else None

        occupancy_cells = np.load(local_dir / "occupancy.npy")
        footprints = [
            ObjectFootprint(x=o.pos[0], z=o.pos[2],
                            bbox_min_x=o.bbox_min[0], bbox_min_z=o.bbox_min[2],
                            bbox_max_x=o.bbox_max[0], bbox_max_z=o.bbox_max[2])
            for o in objects if not o.is_fragment
        ]
        start_resolution = pathfinding.resolve_robot_start(
            OccupancyGrid(cells=occupancy_cells, meta=grid), robot_start, footprints,
            robot_radius_m=resolve_radius_m(None, None))
        print(f"robot_start {robot_start} -> {start_resolution.point} "
              f"(relocated={start_resolution.relocated}, snapped={start_resolution.snapped})")
        robot_start = start_resolution.point

        await scene_service.apply_scene_artifacts(
            session, scene,
            mesh_path=str(local_dir / "scene_points.glb"),
            pointcloud_path=str(local_dir / "aligned_room.ply"),
            occupancy_path=str(local_dir / "occupancy.npy"),
            map_preview_path=str(local_dir / "map_preview.png"),
            transform_path=str(local_dir / "alignment_transform.npz"),
            grid_resolution=grid.resolution, grid_origin_x=grid.origin_x,
            grid_origin_z=grid.origin_z, grid_width=grid.width, grid_height=grid.height,
            floor_y=0.0, ceiling_y=alignment.ceiling_y,
            is_reflected=alignment.is_reflection,
            points_below_floor_frac=alignment.points_below_floor_frac,
            reflective_floor_suspected=alignment.reflective_floor_suspected,
            oversized_object_warning=oversized_object_warning,
            vocab_source=vocab_source, vocab_model=vocab_model, vocab_entries=vocab_entries,
            robot_start_x=robot_start[0], robot_start_z=robot_start[1])
        await scene_service.replace_scene_objects(session, scene, [])
        await scene_service.mark_done(session, scene)
        return {"project_id": str(project.id), "video_id": str(video.id),
                "scene_id": str(scene.id), "scene_dir": str(local_dir),
                "n_objects": len(objects), "n_cameras": len(cameras),
                "ceiling_y": alignment.ceiling_y, "grid": [grid.width, grid.height],
                "robot_start": list(robot_start), "validation": report.summary()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--staged-dir", required=True, help="dir built by build_scene_dir.py")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(
        run(a.name, Path(a.video), Path(a.staged_dir), a.label)), indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
