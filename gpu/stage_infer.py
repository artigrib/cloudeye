#!/usr/bin/env python3
"""Stage 3: the single, canonical MapAnything inference call for this job.

Runs under the mapanything venv. Everything downstream (alignment, objects, occupancy,
GLB export) must be derived from THIS ONE inference call's output and nothing else -
MapAnything does not share a world coordinate frame (origin, rotation, or even axis
direction) across separate `.infer()` invocations, confirmed empirically earlier in this
project. Mixing outputs across two calls silently produces wrong 3D coordinates.

Generalized from the working `extract_per_view_run2.py` reference: argparse instead of
hardcoded `output_horizontal/run2` paths, and never assumes a fixed (294, 518) shape -
portrait sources transpose to (518, 294).

Camera convention (verified against MapAnything's own `mapanything.utils.geometry`
source, and empirically against real capture data - see
tests/test_camera_convention.py): `camera_pose` is camera-to-world, 4x4; the camera
frame is OpenCV (+X right, +Y down, +Z forward - points in front of the camera have
positive camera-frame Z); `intrinsics` is a standard 3x3 pinhole K; the pixel grid is
INTEGER coordinates, `pts3d`/`mask` indexed `[row, col]` (shape `(H, W, ...)`), no
`+0.5` pixel-center offset. Every stage downstream of this one trusts this convention
without re-checking it, so it is verified once, right here, per view, via
`gpu/camera_convention.py`.

A view whose reprojection error is merely elevated (not the ~140-160px signature of a
real convention bug - see that module's docstring) is DROPPED from the pooled output,
not treated as a whole-job failure: gpu/floor_ceiling.py's RANSAC-based fitting over
the pooled cloud degrades gracefully with a few percent fewer points. The job as a
whole still fails if too large a fraction of views were dropped, too few views survive
for a reliable fit regardless of fraction, or any single view hits the unambiguous
convention-mismatch signature - see camera_convention.summarize_job_convention. Every
view's classification (clean/warn/dropped/signature, median/p95 px error) is written
to `camera_convention_report.json` in the job root regardless of outcome - unlike
`per_view/*.npz`, this is not cleaned up after a normal run, so it survives to
diagnose a job whether it passed or failed (see docs/DECISIONS.md's 2026-09-04 entry
for why this was missing before and what that cost).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_convention import (  # noqa: E402
    DEFAULT_DROP_MEDIAN_PX,
    DEFAULT_DROP_P95_PX,
    DEFAULT_MAX_DROPPED_FRACTION,
    DEFAULT_MIN_SURVIVING_VIEWS,
    DEFAULT_NEAR_SIGNATURE_PX,
    DEFAULT_WARN_MEDIAN_PX,
    CameraConventionError,
    build_convention_report,
    check_camera_convention,
    summarize_job_convention,
)
from job_io import job_paths, load_params  # noqa: E402

import numpy as np
import torch
from PIL import Image

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.geometry import depthmap_to_world_frame  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402

DEFAULT_MODEL_NAME = "facebook/map-anything-apache"

# Only Apache-2.0-licensed weights belong here. In particular, "facebook/map-anything"
# (no "-apache" suffix) is the CC-BY-NC-4.0 variant and must never be reachable through
# a job's params.json - see docs/MODELS.md.
ALLOWED_MODEL_NAMES = {DEFAULT_MODEL_NAME}

# MapAnything's per-pixel confidence output - the actual key name in `pred` has not
# been independently confirmed against a real installed copy of the model in this
# environment (no GPU/mapanything venv reachable here, same limitation noted throughout
# docs/DECISIONS.md's 2026-09-03 entries). Feature-detected across the plausible names
# used by DUSt3R-family models (map-anything is one) rather than guessed: whichever of
# these is actually present in `pred[0]` wins, and if none are, `conf` is simply omitted
# from the written npz - every downstream stage already treats missing confidence as
# "fall back to binary mask only" (see floor_ceiling.load_run, gpu/stage_align.py,
# gpu/stage_objects.py), so a wrong/renamed key degrades gracefully instead of silently
# reading and persisting the wrong tensor.
CONF_KEY_CANDIDATES = ("conf", "confidence", "pts3d_conf", "depth_conf")


def main() -> None:
    job_dir = sys.argv[1]
    paths = job_paths(job_dir)
    params = load_params(job_dir)
    model_name = params.get("mapanything_model", DEFAULT_MODEL_NAME)
    if model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(
            f"mapanything_model {model_name!r} is not in ALLOWED_MODEL_NAMES "
            f"{sorted(ALLOWED_MODEL_NAMES)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {model_name} on {device}")
    model = MapAnything.from_pretrained(model_name).to(device)

    views = load_images(str(paths["keyframes_dir"]))
    n_views = len(views)
    print(f"loaded {n_views} views from {paths['keyframes_dir']}")
    if n_views < 3:
        raise RuntimeError(f"only {n_views} views loaded - too few for reconstruction")

    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("inference complete")

    paths["per_view_dir"].mkdir(parents=True, exist_ok=True)
    paths["per_view_png_dir"].mkdir(parents=True, exist_ok=True)

    # Camera-convention thresholds - see gpu/camera_convention.py's module docstring
    # for how each is derived (from the ~140-160px swap signature, not the small
    # clean-data sample the old single 0.5px threshold came from) and
    # docs/DECISIONS.md's 2026-09-04 entry for the full account. Overridable via
    # params.json, same pattern as every other per-job threshold in this pipeline
    # (see gpu/stage_align.py's floor_*/conf_threshold_percentile params).
    warn_median_px = params.get("camera_convention_warn_median_px", DEFAULT_WARN_MEDIAN_PX)
    drop_median_px = params.get("camera_convention_drop_median_px", DEFAULT_DROP_MEDIAN_PX)
    drop_p95_px = params.get("camera_convention_drop_p95_px", DEFAULT_DROP_P95_PX)
    near_signature_px = params.get(
        "camera_convention_near_signature_px", DEFAULT_NEAR_SIGNATURE_PX
    )
    max_dropped_fraction = params.get(
        "camera_convention_max_dropped_fraction", DEFAULT_MAX_DROPPED_FRACTION
    )
    min_surviving_views = params.get(
        "camera_convention_min_surviving_views", DEFAULT_MIN_SURVIVING_VIEWS
    )

    # Pass 1: classify EVERY view's camera-convention fit before writing anything or
    # deciding pass/fail - the model inference call above already ran for all views at
    # once (memory_efficient_inference batches internally), so there is no compute
    # cost to fully classifying every view even if an early one turns out to fail the
    # whole job; doing so is what makes the report below complete regardless of the
    # outcome, instead of stopping at the first bad view the way the old fail-fast
    # code did (see module docstring).
    per_view_data = []
    conf_key_used = None
    for i, pred in enumerate(outputs):
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()
        camera_pose_np = camera_pose_torch.cpu().numpy().astype(np.float32)
        intrinsics_np = intrinsics_torch.cpu().numpy().astype(np.float32)

        result = check_camera_convention(
            pts3d_np,
            mask,
            intrinsics_np,
            camera_pose_np,
            warn_median_px=warn_median_px,
            drop_median_px=drop_median_px,
            drop_p95_px=drop_p95_px,
            near_signature_px=near_signature_px,
        )

        conf_np = None
        for key in CONF_KEY_CANDIDATES:
            if key in pred:
                conf_t = pred[key][0]
                if conf_t.ndim == 3 and conf_t.shape[-1] == 1:
                    conf_t = conf_t.squeeze(-1)
                candidate = conf_t.cpu().numpy().astype(np.float32)
                if candidate.shape != mask.shape:
                    print(
                        f"  view {i}: found conf candidate key {key!r} but shape "
                        f"{candidate.shape} != mask shape {mask.shape} - skipping it"
                    )
                    continue
                conf_np, conf_key_used = candidate, key
                break
        if i == 0:
            print(
                f"confidence key detected: {conf_key_used!r}"
                if conf_key_used
                else f"confidence key NOT found (tried {CONF_KEY_CANDIDATES}) - "
                "conf will be omitted; downstream stages fall back to binary mask only"
            )

        print(
            f"view {i}: convention={result.status} median={result.median_px_error:.2f}px "
            f"p95={result.p95_px_error:.2f}px"
            + (f" - {result.detail}" if result.status != "clean" else "")
        )

        per_view_data.append(
            {
                "i": i,
                "result": result,
                "pts3d_np": pts3d_np,
                "mask": mask,
                "image_np": image_np,
                "camera_pose_np": camera_pose_np,
                "intrinsics_np": intrinsics_np,
                "conf_np": conf_np,
            }
        )

    results = [d["result"] for d in per_view_data]
    summary = summarize_job_convention(
        results,
        max_dropped_fraction=max_dropped_fraction,
        min_surviving_views=min_surviving_views,
    )

    thresholds = {
        "warn_median_px": warn_median_px,
        "drop_median_px": drop_median_px,
        "drop_p95_px": drop_p95_px,
        "near_signature_px": near_signature_px,
        "max_dropped_fraction": max_dropped_fraction,
        "min_surviving_views": min_surviving_views,
    }
    report = build_convention_report(
        [f"view {d['i']}" for d in per_view_data], results, summary, thresholds=thresholds
    )
    # Written to the job root, NOT per_view/ - survives the worker's normal
    # fetch-then-cleanup of per_view/*.npz regardless of pass/fail (see module
    # docstring and docs/DECISIONS.md's 2026-09-04 entry).
    paths["camera_convention_report"].write_text(json.dumps(report, indent=2))
    print(
        f"camera convention: {summary.n_clean} clean, {summary.n_warned} warned, "
        f"{summary.n_dropped} dropped, {summary.kept_views}/{summary.n_views} views kept "
        f"-> {paths['camera_convention_report']}"
    )

    if summary.should_fail:
        raise CameraConventionError(
            f"stage_infer: {summary.fail_reason} - see "
            f"{paths['camera_convention_report']} for full per-view detail"
        )

    # Pass 2: write per_view/*.npz + PNG only for views that survived (clean or warn -
    # dropped views are excluded from the pooled output entirely, per
    # summarize_job_convention's per-view/job-level split).
    view_meta = []
    for d in per_view_data:
        i, result = d["i"], d["result"]
        if result.should_drop:
            print(f"view {i}: DROPPED from output ({result.detail})")
            continue

        pts3d_np, mask, image_np = d["pts3d_np"], d["mask"], d["image_np"]
        camera_pose_np, intrinsics_np, conf_np = (
            d["camera_pose_np"],
            d["intrinsics_np"],
            d["conf_np"],
        )

        save_kwargs = dict(
            pts3d=pts3d_np.astype(np.float32),
            mask=mask,
            img_no_norm=image_np.astype(np.float32),
            camera_pose=camera_pose_np,
            intrinsics=intrinsics_np,
        )
        if conf_np is not None:
            save_kwargs["conf"] = conf_np
        np.savez_compressed(paths["per_view_dir"] / f"view_{i:03d}.npz", **save_kwargs)

        img_u8 = (
            np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            if image_np.max() <= 1.5
            else np.clip(image_np, 0, 255).astype(np.uint8)
        )
        Image.fromarray(img_u8).save(paths["per_view_png_dir"] / f"view_{i:03d}.png")

        valid_frac = float(mask.sum()) / mask.size
        view_meta.append(
            {
                "view": i,
                "shape": list(image_np.shape[:2]),
                "valid_fraction": round(valid_frac, 4),
                "valid_pixels": int(mask.sum()),
                "conf_available": conf_np is not None,
                "convention_status": result.status,
            }
        )

    n_conf_views = sum(1 for v in view_meta if v["conf_available"])
    paths["infer_meta"].write_text(
        json.dumps(
            {
                "model": model_name,
                "n_views": n_views,
                "kept_views": len(view_meta),
                "conf_available_views": n_conf_views,
                "views": view_meta,
            },
            indent=2,
        )
    )
    print(f"DONE: {len(view_meta)}/{n_views} views written to {paths['per_view_dir']}")


if __name__ == "__main__":
    main()
