"""Stage B1 (SPEC.md §5): batch real-mesh generation for scene objects via
TRELLIS.2 (`scripts/msa/trellis_client.py`), replacing Stage A0's placeholder
primitives where a generated mesh passes the containment test.

Three stages, each running where its dependencies actually live:

- **SAM 3 scan** (`scripts/msa/sam3_scan_cli.py`, pipeline-box GPU, `sam3` venv):
  runs real SAM 3 open-vocabulary segmentation on real keyframe PNGs with each
  object class's own text prompt, across every retained view. Produces real
  per-pixel 2D masks - the actual thing SPEC §5 asks for.
- **Crop building** (this module's `match_best_detection`/`crop_from_mask` +
  `scripts/msa/build_real_crops.py`, CPU, VPS): matches each real SAM 3
  detection back to a specific object id (there can be several objects of the
  same class in one scene) and crops the real photo to a real RGBA mask.
- **Generation** (`scripts/msa/generate_from_manifest_cli.py`, GPU instance,
  needs `Trellis2Client`): runs TRELLIS.2 on each crop, applies the SPEC §5
  containment test.

**A real, previously-shipped bug this module now fixes**: `scene_objects/*.ply`
and `scene_objects.json`'s `centroid_xyz`/`bbox_min`/`bbox_max` are in the
*aligned* room frame (`gpu/stage_objects.py`: `aligned = raw_pts @ R.T;
aligned[:, 1] -= floor_y`, using `alignment_transform.npz`'s `R`/`floor_y`) -
but `per_view/*.npz`'s `camera_pose`/`pts3d` are in the pipeline's *raw*,
pre-alignment frame. Projecting an aligned-frame point with a raw-frame camera
pose (what an earlier version of this module did) silently produces a
scrambled, non-metric 2D position - it "worked" often enough by accident to
look plausible (a real photo, a real-ish blob) while actually being wrong,
which is far more dangerous than an obvious crash. Confirmed and fixed by
comparing reprojected-and-transformed centroids against independently-obtained
real SAM 3 detection centroids: distances dropped from 140-1000+ px (scrambled)
to 1-52px (correct) once `to_raw_frame` was applied first. `load_alignment_transform`/
`to_raw_frame` below are the fix - every caller projecting `scene_objects` data
into a `per_view` camera MUST apply `to_raw_frame` first.

**Superseded, not just historical**: an earlier approach reconstructed a mask
by reprojecting an object's own 3D point cloud into 2D and filling its convex
hull, instead of using a real SAM 3 mask (no real 2D SAM 3 masks are persisted
to disk - `gpu/stage_objects.py` computes them on the fly and immediately
backprojects into 3D). That approach failed on two different real scenes for
two different reasons (imprecise/overbroad 3D clustering in one scene;
low-resolution grazing-angle crops in another) *before* the coordinate-frame
bug above was even found - it is not used by this module's current functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def project_points_to_view(
    xyz: np.ndarray, camera_pose: np.ndarray, intrinsics: np.ndarray, *, min_z: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space (raw pipeline frame - see module docstring) points
    into one view's pixel coordinates.

    `camera_pose` is camera-to-world (4x4, OpenCV convention - positive
    camera-frame Z is forward - see `gpu/camera_convention.py`). Returns
    `(uv, in_front)`: `uv` is (N,2) pixel coordinates for ALL input points
    (meaningless for `~in_front` rows), `in_front` is a bool mask for points
    with positive camera-frame Z beyond `min_z`.
    """
    world_to_cam = np.linalg.inv(np.asarray(camera_pose, dtype=np.float64))
    pts_h = np.hstack([xyz, np.ones((len(xyz), 1))])
    pts_cam = (world_to_cam @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > min_z
    z_safe = np.where(in_front, pts_cam[:, 2], 1.0)
    px = (np.asarray(intrinsics, dtype=np.float64) @ pts_cam.T).T
    uv = px[:, :2] / z_safe[:, None]
    return uv, in_front


def load_alignment_transform(scene_dir: Path) -> tuple[np.ndarray, float]:
    """Reads `alignment_transform.npz`'s `R` (3,3 rotation) and `floor_y`
    (scalar) - the same transform `gpu/stage_objects.py` applies when it
    writes `scene_objects/*.ply` and `scene_objects.json`'s
    `centroid_xyz`/`bbox_min`/`bbox_max` (aligned frame)."""
    d = np.load(scene_dir / "alignment_transform.npz")
    return np.asarray(d["R"], dtype=np.float64), float(d["floor_y"])


def to_raw_frame(aligned_xyz: np.ndarray, R: np.ndarray, floor_y: float) -> np.ndarray:
    """Inverse of `gpu/stage_objects.py`'s `aligned = raw @ R.T; aligned[:,1] -=
    floor_y`: `raw = aligned_with_floor_restored @ R` (R is a rotation matrix,
    so `R.T` is its own inverse: `raw = aligned @ (R.T)^-1 = aligned @ R`).
    Points from `scene_objects/*.ply`/`scene_objects.json` MUST go through this
    before `project_points_to_view` - see module docstring."""
    a = np.asarray(aligned_xyz, dtype=np.float64).copy()
    a[:, 1] += floor_y
    return a @ R


def match_best_detection(
    object_centroid_raw: np.ndarray,
    detections: list[dict],
    view_data: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    *,
    min_npix: int = 200,
    max_dist_px: float = 100.0,
) -> dict | None:
    """Disambiguates which real SAM 3 detection (across possibly many views
    and many same-class objects) belongs to THIS object: projects the
    object's own measured centroid (already in raw frame - see
    `to_raw_frame`) into each detection's view and picks the detection whose
    own 2D centroid is closest. Filters out detections smaller than
    `min_npix` (near-noise) and matches farther than `max_dist_px` (not
    actually this object). `detections` is a list of dicts with at least
    `view_id`, `centroid_uv` (2-list), `npix`. Returns the winning detection
    dict (with an added `match_dist_px` key) or None if nothing qualifies."""
    centroid = np.asarray(object_centroid_raw, dtype=np.float64).reshape(1, 3)
    best: tuple[float, dict] | None = None
    for det in detections:
        if det["npix"] < min_npix:
            continue
        view_id = det["view_id"]
        if view_id not in view_data:
            continue
        camera_pose, intrinsics, _hw = view_data[view_id]
        uv, in_front = project_points_to_view(centroid, camera_pose, intrinsics)
        if not in_front[0]:
            continue
        dist = float(np.linalg.norm(uv[0] - np.asarray(det["centroid_uv"], dtype=np.float64)))
        if dist > max_dist_px:
            continue
        if best is None or dist < best[0]:
            best = (dist, det)
    if best is None:
        return None
    dist, det = best
    return {**det, "match_dist_px": dist}


def crop_from_mask(image: np.ndarray, mask: np.ndarray, *, pad_frac: float = 0.15) -> np.ndarray | None:
    """Crops `image` (H,W,3 uint8) to `mask`'s (H,W bool or 0/255) bounding
    box + padding, returns an (h,w,4) uint8 RGBA array with `mask` as alpha.
    Returns None if `mask` is empty."""
    mask_bool = np.asarray(mask) > 0
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    H, W = mask_bool.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pad = int(pad_frac * max(x1 - x0, y1 - y0, 1))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop_img = image[y0:y1, x0:x1]
    crop_mask = (mask_bool[y0:y1, x0:x1] * 255).astype(np.uint8)
    return np.dstack([crop_img, crop_mask])


def bbox_volume(bbox_min: list[float] | np.ndarray, bbox_max: list[float] | np.ndarray) -> float:
    size = np.asarray(bbox_max, dtype=np.float64) - np.asarray(bbox_min, dtype=np.float64)
    return float(max(size[0], 0.0) * max(size[1], 0.0) * max(size[2], 0.0))


def _sorted_extents(bbox_min, bbox_max) -> np.ndarray:
    """Bbox extents (raw, same units as input) sorted descending -
    orientation-invariant, since a single-photo generated mesh has no
    guaranteed axis correspondence to the measured object's world-frame axes
    (TRELLIS.2 outputs in its own canonical frame, not the scene's)."""
    return np.sort(np.maximum(np.asarray(bbox_max, dtype=np.float64) - np.asarray(bbox_min, dtype=np.float64), 0.0))[::-1]


def containment_ratio(generated_bbox_min, generated_bbox_max, measured_bbox_min, measured_bbox_max) -> float:
    """SPEC §5: reject (fall back to primitive) if the generated mesh's shape
    is <20% or >150% of the measured hull bbox's shape.

    Two real, confirmed-on-GPU findings shaped this (2026-09-06, real Stage
    B1 output on 7 objects - see docs/DECISIONS.md):

    1. A raw bbox-volume comparison is meaningless: TRELLIS.2 outputs a mesh
       in its own arbitrary local frame (uniform-scaled to fit its longest
       dimension near 1.0, NOT real-world meters - only fit to the real
       footprint later, by `assemble_with_generated.py`). Comparing that
       directly against a meters-scale measured bbox rejects almost
       everything by construction (raw-volume ratios of 0.06-5.9 for objects
       whose actual proportions were fine). Fix: scale the generated mesh's
       sorted extents so its LARGEST dimension matches the measured object's
       largest dimension, before comparing anything.
    2. Even after that fix, comparing the product of all 3 (sorted, scaled)
       dimensions still over-weights the SMALLEST one - which is the least
       reliable measurement on both sides (photogrammetry depth noise on the
       measured side, single-image depth ambiguity on the generated side).
       On real objects this alone flipped otherwise-clearly-correct
       generations (a real desk, a real chair) to "reject". Fix: the primary
       accept/reject signal is the ratio of the top-2 (largest) sorted
       dimensions' product - the footprint-relevant area, which is what
       SPEC's placement/collision logic actually cares about. The smallest
       dimension only vetoes as a *degenerate-geometry* guard (a genuinely
       flat sliver where a solid object was measured, or vice versa) via a
       much wider band - it does not enter the returned ratio.

    Returns the top-2-dimension area ratio (what `passes_containment`'s
    [0.2, 1.5] band applies to); `float('inf')` if the smallest-dimension
    guard fails, either measured extent is degenerate, or the two shapes are
    otherwise incomparable.
    """
    gen_sorted = _sorted_extents(generated_bbox_min, generated_bbox_max)
    measured_sorted = _sorted_extents(measured_bbox_min, measured_bbox_max)
    if measured_sorted[0] <= 1e-9 or gen_sorted[0] <= 1e-9:
        return float("inf")
    scale = measured_sorted[0] / gen_sorted[0]
    scaled_gen = gen_sorted * scale

    if measured_sorted[1] <= 1e-9:
        return float("inf")
    top2_ratio = float((scaled_gen[0] * scaled_gen[1]) / (measured_sorted[0] * measured_sorted[1]))

    if measured_sorted[2] > 1e-9:
        thin_ratio = scaled_gen[2] / measured_sorted[2]
        if thin_ratio < 0.05 or thin_ratio > 20.0:
            return float("inf")
    return top2_ratio


def passes_containment(ratio: float, *, low: float = 0.2, high: float = 1.5) -> bool:
    return low <= ratio <= high


def write_manifest(entries: list[dict], out_path: Path) -> None:
    out_path.write_text(json.dumps(entries, indent=2))
