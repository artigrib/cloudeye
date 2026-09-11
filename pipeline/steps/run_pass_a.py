#!/usr/bin/env python3
"""Pass A (coarse): run MapAnything over the ~80 motion-sampled frames spread across
the whole video to get one set of GLOBAL, metric-scale camera poses.

Runs on the GPU box under the `mapanything` venv (same one gpu/stage_infer.py uses -
this script mirrors that stage's model-call convention exactly, see its own docstring
for the camera_pose/intrinsics/pixel-grid convention this trusts without re-checking).

Not a pipeline change: this is new code under scripts/exp/dense_recon/, invoked
manually for this spike, not wired into gpu/run_pipeline.sh.

Metric scale note: MapAnything's own `.infer()` output is trusted as metric as-is -
the production pipeline's own stage_align.py never rescales points/poses, it only
rotates+translates to align the floor plane (see gpu/stage_align.py's `run_align`,
around the `A[:3,:3]=R; A[1,3]=-floor_y` construction) - so Pass A here does the same:
no extra scale calibration step.

Output format (poses.json), one dict, documented here since nothing upstream defines
this schema for a standalone experiment:
{
  "model": "facebook/map-anything-apache",
  "frame": "passA_global",                 # coordinate frame name, referenced by Pass B
  "n_views": int,
  "wall_clock_sec": float,
  "peak_vram_bytes": int,
  "views": [
    {
      "raw_frame_index": int,               # ties back to motion_sampled_frames.json
      "t_sec": float,
      "image_file": "frame_000000.jpg",
      "camera_pose": [[4x4]],                # camera-to-world, OpenCV convention (+X right, +Y down, +Z forward)
      "intrinsics": [[3x3]],
      "shape_hw": [H, W]
    }, ...
  ]
}
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

if torch.cuda.is_available():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.geometry import depthmap_to_world_frame  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402

MODEL_NAME = "facebook/map-anything-apache"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-json", required=True, help="passA_frames.json (has raw_frame_index/t_sec per view)")
    ap.add_argument("--frames-dir", required=True, help="dir containing frame_{raw_frame_index:06d}.jpg")
    ap.add_argument("--out", required=True, help="poses.json output path")
    ap.add_argument(
        "--save-per-view-npz-dir",
        default=None,
        help="if given, also save view_{raw_frame_index:06d}.npz per view "
        "(pts3d/mask/depth_z/conf/raw_frame_index; pts3d is in this run's OWN "
        "arbitrary world frame, not the hero's) for later coverage analysis via "
        "align_and_coverage.py",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.frames_json).read_text())
    frame_entries = data["frames"]
    frames_dir = Path(args.frames_dir)
    image_paths = [str(frames_dir / f"frame_{f['raw_frame_index']:06d}.jpg") for f in frame_entries]
    for p in image_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"loading {MODEL_NAME} on {device}")
    model = MapAnything.from_pretrained(MODEL_NAME).to(device)

    views = load_images(image_paths)
    print(f"loaded {len(views)} views from {args.frames_dir}")

    t0 = time.time()
    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    wall_clock = time.time() - t0
    print(f"inference complete in {wall_clock:.1f}s")

    # Confidence is REQUIRED by this run (2026-09-08 task change): it is saved raw and
    # filtered downstream, never inside the model (apply_confidence_mask stays False).
    # facebook/map-anything-apache's config.json has
    #   pred_head_config.adaptor_type = "raydirs+depth+pose+confidence+mask"
    # and model.py sets self.scene_rep_type from adaptor_type, then emits res[i]["conf"]
    # iff "confidence" in self.scene_rep_type -- so conf must be present. Do NOT trust
    # pred_head_config.adaptor_config.scene_rep_type ("raydirs+depth+pose", no
    # confidence): that nested field is not what the emission guard reads.
    # Fail loudly rather than silently writing npz without conf.
    probe = outputs[0]
    if "conf" not in probe:
        raise RuntimeError(
            "pred['conf'] missing from model output. Present keys: "
            f"{sorted(probe.keys())}. Refusing to run: this job requires confidence."
        )
    _c = probe["conf"][0]
    _m = probe["mask"][0].squeeze(-1)
    if tuple(_c.shape) != tuple(_m.shape):
        raise RuntimeError(
            f"pred['conf'] shape {tuple(_c.shape)} != mask shape {tuple(_m.shape)}"
        )
    print(f"conf OK: shape={tuple(_c.shape)} dtype={_c.dtype} "
          f"min={float(_c.min()):.4f} max={float(_c.max()):.4f}")

    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    npz_dir = None
    if args.save_per_view_npz_dir:
        npz_dir = Path(args.save_per_view_npz_dir)
        npz_dir.mkdir(parents=True, exist_ok=True)

    out_views = []
    for i, (pred, entry) in enumerate(zip(outputs, frame_entries)):
        camera_pose = pred["camera_poses"][0].cpu().numpy().astype(np.float64)
        intrinsics = pred["intrinsics"][0].cpu().numpy().astype(np.float64)
        depthmap = pred["depth_z"][0].squeeze(-1)
        shape_hw = list(depthmap.shape[-2:])

        if npz_dir is not None:
            pts3d, valid_mask = depthmap_to_world_frame(depthmap, pred["intrinsics"][0], pred["camera_poses"][0])
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & valid_mask.cpu().numpy()
            conf = pred["conf"][0].squeeze(-1) if pred["conf"][0].ndim == 3 else pred["conf"][0]
            np.savez_compressed(
                npz_dir / f"view_{entry['raw_frame_index']:06d}.npz",
                pts3d=pts3d.cpu().numpy().astype(np.float32),
                mask=mask,
                # conf: RAW learned confidence, unmasked and unthresholded, float16
                # (user decision 2026-09-08 -- filtering is percentile-based, so the
                # mantissa loss is immaterial). Range is [1, inf) (config:
                # confidence_type "exp", vmin 1), NOT a 0..1 probability.
                # depth_z deliberately NOT stored: it is recoverable exactly from
                # pts3d + camera_pose (median reprojection error 0.011 px, measured).
                conf=conf.cpu().numpy().astype(np.float16),
                # source frame index in the original video, so a npz is self-describing
                # and does not depend on its filename to be traced back.
                raw_frame_index=np.int32(entry["raw_frame_index"]),
            )

        out_views.append(
            {
                "raw_frame_index": entry["raw_frame_index"],
                "t_sec": entry["t_sec"],
                "image_file": Path(image_paths[i]).name,
                "camera_pose": camera_pose.tolist(),
                "intrinsics": intrinsics.tolist(),
                "shape_hw": shape_hw,
            }
        )

    result = {
        "model": MODEL_NAME,
        "frame": "passA_global",
        "n_views": len(out_views),
        "wall_clock_sec": round(wall_clock, 2),
        "peak_vram_bytes": int(peak_vram),
        "peak_vram_mib": round(peak_vram / (1024 * 1024), 1),
        "views": out_views,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        f"DONE: {len(out_views)} views, {wall_clock:.1f}s wall clock, "
        f"peak VRAM {peak_vram / (1024 * 1024):.1f} MiB -> {args.out}"
    )


if __name__ == "__main__":
    main()
