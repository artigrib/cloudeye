#!/usr/bin/env python3
"""Stdlib-only job-directory helpers shared across every pipeline stage.

Deliberately has zero third-party imports: the mapanything venv runs numpy 2.5.2, the
sam3 venv runs numpy 1.26.4, and vidmap has its own opencv/numpy pin - any module
imported by more than one stage's venv must not depend on a specific numpy/torch/etc
version, or it becomes a landmine the moment two stages' pinned versions diverge further.

Usable both as an importable module (`from job_io import job_paths, write_status`) and as
a CLI (`python3 job_io.py status <job_dir> <state> --stage ... --message ...`) - the CLI
form is what run_pipeline.sh calls between stages, since bash has no direct access to the
Python helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Single source of truth for the pipeline's spatial resolution, shared across stages
# that otherwise run in different venvs (mapanything/vidmap/sam3) and can't import each
# other's modules directly - see this module's own docstring for why. Used by:
# stage_occupancy.py (grid cell size), stage_vocab.py (per-size-class DBSCAN eps,
# expressed as a multiple of this instead of an independent flat constant), and
# stage_objects.py (voxel-downsample size before clustering). Changing this one value
# changes all three consistently instead of three independent constants drifting apart.
OCCUPANCY_GRID_RESOLUTION_M = 0.05


def job_paths(job_dir: str | Path) -> dict[str, Path]:
    """The full, fixed set of paths inside one job directory. Every stage script should
    build its input/output paths from this, never hand-roll a path string, so the layout
    only needs to change in one place."""
    job_dir = Path(job_dir)
    return {
        "job_dir": job_dir,
        "input": job_dir / "input.mp4",
        "params": job_dir / "params.json",
        "status": job_dir / "status.json",
        "result": job_dir / "result.json",
        "keyframes_dir": job_dir / "keyframes",
        "keyframes_json": job_dir / "keyframes.json",
        "vocab_json": job_dir / "vocab.json",
        "per_view_dir": job_dir / "per_view",
        "per_view_png_dir": job_dir / "per_view_png",
        "infer_meta": job_dir / "infer_meta.json",
        "camera_convention_report": job_dir / "camera_convention_report.json",
        "aligned_ply": job_dir / "aligned_room.ply",
        "transform_npz": job_dir / "alignment_transform.npz",
        "alignment_json": job_dir / "alignment.json",
        "cameras_json": job_dir / "cameras_aligned.json",
        "scene_objects_dir": job_dir / "scene_objects",
        "scene_objects_json": job_dir / "scene_objects" / "scene_objects.json",
        "occupancy_npz": job_dir / "occupancy_grid.npz",
        "occupancy_npy": job_dir / "occupancy.npy",
        "occupancy_meta": job_dir / "occupancy_meta.json",
        "map_preview": job_dir / "map_preview.png",
        "glb": job_dir / "scene_points.glb",
        "logs_dir": job_dir / "logs",
        "pid_file": job_dir / "pipeline.pid",
        "job_env": job_dir / ".env",
    }


def load_params(job_dir: str | Path) -> dict:
    """Read params.json (the per-job overrides the backend writes alongside input.mp4),
    or {} if absent - every stage script should apply its own defaults on top of this."""
    p = job_paths(job_dir)["params"]
    if p.exists():
        return json.loads(p.read_text())
    return {}


def write_status(
    job_dir: str | Path,
    *,
    state: str,
    stage: str | None = None,
    stage_index: int | None = None,
    n_stages: int | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    """Atomically overwrite status.json (write to a .tmp file, then os.replace) so a
    concurrent poller (services/gpu_client.py, polling every GPU_POLL_INTERVAL_SEC) can
    never observe a half-written file."""
    status_path = job_paths(job_dir)["status"]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    started_at = now
    if status_path.exists():
        try:
            started_at = json.loads(status_path.read_text()).get("started_at", now)
        except (json.JSONDecodeError, OSError):
            pass

    payload = {
        "state": state,
        "stage": stage,
        "stage_index": stage_index,
        "n_stages": n_stages,
        "message": message,
        "error": error,
        "started_at": started_at,
        "updated_at": now,
    }
    tmp_path = status_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, status_path)


def write_result(job_dir: str | Path, **fields) -> None:
    """Write result.json - the final artifact-summary the backend reads after fetching
    the job directory. Not atomic (written once, at the very end, after `state=done`)."""
    job_paths(job_dir)["result"].write_text(json.dumps(fields, indent=2))


def finalize_result(job_dir: str | Path) -> dict:
    """Assemble result.json from whatever artifacts stages 1-7 actually produced. Safe to
    call even if some are missing (a stage failed) - reports what exists."""
    paths = job_paths(job_dir)
    result: dict = {}

    if paths["scene_objects_json"].exists():
        objects = json.loads(paths["scene_objects_json"].read_text())
        result["object_count"] = len(objects)
        result["solid_object_count"] = sum(1 for o in objects if not o.get("likely_fragment"))

    if paths["keyframes_json"].exists():
        kf = json.loads(paths["keyframes_json"].read_text())
        result["keyframe_count"] = kf.get("kept_count")

    if paths["alignment_json"].exists():
        align = json.loads(paths["alignment_json"].read_text())
        result["floor_y"] = align.get("floor_y")
        result["ceiling_y"] = align.get("ceiling_y")
        result["is_reflected"] = align.get("is_reflection")
        result["residual_tilt_deg"] = align.get("residual_tilt_deg")

    if paths["occupancy_meta"].exists():
        result["occupancy_meta"] = json.loads(paths["occupancy_meta"].read_text())

    for key, path_key in (
        ("pointcloud_path", "aligned_ply"),
        ("transform_path", "transform_npz"),
        ("occupancy_path", "occupancy_npy"),
        ("map_preview_path", "map_preview"),
        ("mesh_path", "glb"),
    ):
        if paths[path_key].exists():
            result[key] = str(paths[path_key].relative_to(paths["job_dir"]))

    write_result(job_dir, **result)
    return result


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="atomically write status.json")
    st.add_argument("job_dir")
    st.add_argument("state", choices=["running", "done", "failed"])
    st.add_argument("--stage")
    st.add_argument("--stage-index", type=int)
    st.add_argument("--n-stages", type=int)
    st.add_argument("--message")
    st.add_argument("--error")

    fin = sub.add_parser("finalize", help="assemble result.json from whatever exists")
    fin.add_argument("job_dir")

    args = ap.parse_args()
    if args.cmd == "status":
        write_status(
            args.job_dir,
            state=args.state,
            stage=args.stage,
            stage_index=args.stage_index,
            n_stages=args.n_stages,
            message=args.message,
            error=args.error,
        )
        print(f"status written: state={args.state} stage={args.stage}")
    elif args.cmd == "finalize":
        result = finalize_result(args.job_dir)
        print(f"result.json written: {json.dumps(result)}")


if __name__ == "__main__":
    _main()
