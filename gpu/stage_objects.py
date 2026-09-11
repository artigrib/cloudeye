#!/usr/bin/env python3
"""Stage 5: SAM3 open-vocabulary object extraction + 3D localization.

Runs under the sam3 venv. Generalized from the working `build_scene_objects_run2.py`
reference: the object vocabulary comes from vocab.json (stage 2's OpenRouter call, or
its default-vocabulary fallback) instead of a hardcoded CONCEPTS table, and N_VIEWS is
read from however many per_view/*.npz files actually exist instead of a hardcoded 26.

Coordinates: pts3d from stage 3 (stage_infer.py) are in that run's own raw frame -
every point gets `@ R.T` then `-= floor_y` applied using stage 4's alignment transform
before pooling, so object coordinates land in the same aligned frame as
aligned_room.ply and the occupancy grid. Never skip this step or use a different run's
transform - see floor_ceiling.py's module docstring for why that's unsafe.

Resource handling (2026-09-01, hotel-own-batch diagnosis): per-concept voxel
downsampling (voxel_size_for_eps) + a hard post-downsample point cap
(MAX_CLUSTER_POINTS) + density-aware min_samples/min_cluster rescaling
(rescale_cluster_params). Deliberately does NOT add spatial chunking (splitting a
concept's points into overlapping regions, clustering each separately, merging at
boundaries) even though that was also on the table: the two real failures this was
built to fix (room 1's "curtain" at 1,336,153 raw points hitting the 900s
STAGE_TIMEOUT; room 3's "bed" at 1,610,369 raw points getting SIGKILLed, OOM-consistent)
are both bounded by downsampling + the point cap alone - 300k points is a size sklearn's
DBSCAN clusters in low tens of seconds. Chunking's own failure mode (a single real
object straddling a chunk boundary needs correct merge logic, which is genuinely easy
to get subtly wrong) isn't worth taking on for a problem the simpler fix already covers.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import OCCUPANCY_GRID_RESOLUTION_M, job_paths, load_params  # noqa: E402

import numpy as np

# Hard ceiling on points fed into DBSCAN per concept, applied AFTER voxel downsampling
# (below) - defense in depth, not the primary fix. Confirmed real failures this guards
# against (hotel-own-batch, 2026-09-01, own recordings - much longer/higher-bitrate than
# prior YouTube-shorts material): room 1's "curtain" pooled 1,336,153 raw points and hit
# the 900s STAGE_TIMEOUT mid-DBSCAN; room 3's "bed" pooled 1,610,369 raw points and was
# SIGKILLed (exit 137, OOM-consistent - host memory traced to ~248GB right before the
# kill). 300k points is comfortably within what sklearn's DBSCAN clusters in low tens of
# seconds even at these eps values, on this project's own GPU-host CPU.
MAX_CLUSTER_POINTS = 300_000

# Floors for the density-aware rescaling in rescale_cluster_params() below - never
# shrink a threshold to the point that pure sensor noise could pass it.
MIN_SAMPLES_FLOOR = 3
MIN_CLUSTER_FLOOR = 5

# Extra slack applied on top of room_bbox_min/room_bbox_max (which already carries its
# own room_bbox_margin, default 0.3m - see gpu/stage_align.py) when rejecting individual
# points, not just cluster centroids. Same order of magnitude as
# gpu/stage_align.py's DEFAULT_FLOOR_CLIP_MARGIN_M (0.05m) and app/config.py's
# ROBOT_RADIUS_M (0.10m) - this is about tolerating per-point reconstruction noise on an
# object flush against a wall (e.g. a door frame), not about compensating for a tight
# room boundary, which room_bbox_margin already handles.
POINT_ROOM_MARGIN_M = 0.10


def room_bbox_point_mask(
    pts: np.ndarray, room_bbox_min: np.ndarray, room_bbox_max: np.ndarray, margin: float = POINT_ROOM_MARGIN_M
) -> np.ndarray:
    """Boolean mask, one per row of `pts`, True where the point falls within
    room_bbox_min/room_bbox_max padded by `margin` on every axis.

    Kept free of GPU/venv-specific imports (torch/open3d/sam3/sklearn) so it can be
    unit-tested from the main app venv, unlike the rest of this stage script.
    """
    lo = np.asarray(room_bbox_min, dtype=float) - margin
    hi = np.asarray(room_bbox_max, dtype=float) + margin
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def voxel_size_for_eps(eps: float, *, grid_resolution: float = OCCUPANCY_GRID_RESOLUTION_M) -> float:
    """Per-concept downsample voxel size: small enough (eps/3) that voxel downsampling
    can never itself break DBSCAN's ability to link points within a real cluster at this
    concept's eps, but never coarser than the occupancy grid's own resolution - tied to
    that resolution, not an independently-chosen constant. See SIZE_CLASS_EPS_VOXEL_MULTIPLE
    in stage_vocab.py for the matching eps side of this."""
    return min(grid_resolution, eps / 3)


def voxel_downsample_indices(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Indices into `pts` to keep after grid voxel downsampling: one representative
    point per occupied voxel of side `voxel_size` (the first point encountered in each
    voxel, in input order) - deterministic, no synthetic averaged points, so the kept
    points/colors/view-ids stay real and mutually consistent by simple fancy-indexing.
    Collapses near-duplicate points contributed by many overlapping camera views of the
    same physical surface, which is the actual source of the multi-million-point pools
    that caused real timeout/OOM failures (see MAX_CLUSTER_POINTS's comment) - this is
    the primary fix, MAX_CLUSTER_POINTS is the backstop for whatever it doesn't catch."""
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    _, first_seen = np.unique(voxel_idx, axis=0, return_index=True)
    return np.sort(first_seen)


def cap_points_uniform(n: int, max_points: int) -> np.ndarray | None:
    """If n > max_points, indices for a uniform subsample down to max_points (same
    "step and round" approach as stage_keyframes.py's max_keyframes truncation, for
    consistency) - otherwise None, meaning "no truncation needed, use everything"."""
    if n <= max_points:
        return None
    step = n / max_points
    return np.array(sorted({int(i * step) for i in range(max_points)}))


def rescale_cluster_params(
    min_samples: int,
    min_cluster: int,
    compression_ratio: float,
    *,
    min_samples_floor: int = MIN_SAMPLES_FLOOR,
    min_cluster_floor: int = MIN_CLUSTER_FLOOR,
) -> tuple[int, int]:
    """`min_samples`/`min_cluster` (from vocab.json, via SIZE_CLASS_PARAMS) were
    calibrated against RAW, multi-view-redundant point density. After voxel
    downsampling to `compression_ratio` (= downsampled_count / raw_count for THIS
    concept in THIS room - measured, not assumed, since it varies with how much view
    overlap the footage actually has), both thresholds are point-count thresholds
    against a now-sparser cloud, so they're rescaled by the same measured ratio rather
    than reused unchanged or re-guessed as new constants. Floor-clamped so downsampling
    can never make either threshold low enough that noise passes it."""
    new_samples = max(min_samples_floor, round(min_samples * compression_ratio))
    new_cluster = max(min_cluster_floor, round(min_cluster * compression_ratio))
    return new_samples, new_cluster


def main() -> None:
    # Deferred: these only exist in the sam3 venv (gpu/run_pipeline.sh's SAM3_PY), not
    # the main app venv - importing them at module level would break unit-testing
    # room_bbox_point_mask() above from the main test suite.
    import open3d as o3d
    import torch
    from PIL import Image
    from sklearn.cluster import DBSCAN

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    job_dir = sys.argv[1]
    paths = job_paths(job_dir)
    params = load_params(job_dir)
    min_det_pixels = params.get("min_detection_pixels", 5)

    paths["scene_objects_dir"].mkdir(parents=True, exist_ok=True)

    vocab = json.loads(paths["vocab_json"].read_text())
    concepts = {
        e["name"]: (e["name"], e["score_threshold"], e["dbscan_eps"], e["dbscan_min_samples"], e["min_cluster"])
        for e in vocab["objects"]
    }
    print(f"vocabulary ({vocab['source']}): {list(concepts.keys())}")

    xform = np.load(paths["transform_npz"])
    R, floor_y = xform["R"], float(xform["floor_y"])
    room_bbox_min, room_bbox_max = xform["bbox_min"], xform["bbox_max"]
    # conf_available/conf_cutoff come from stage_align.py's own percentile filter (schema
    # 5+, absent entirely on an older alignment_transform.npz) - reusing THAT run's
    # cutoff here keeps object extraction and room alignment agreeing on what "low
    # confidence" means for this scene, rather than each stage computing its own
    # independent percentile over a different (per-detection vs whole-room) point pool.
    conf_available = bool(xform["conf_available"]) if "conf_available" in xform.files else False
    conf_cutoff = float(xform["conf_cutoff"]) if conf_available else None
    print(
        f"loaded alignment transform: floor_y={floor_y:.4f}, "
        + (f"conf_cutoff={conf_cutoff:.4f}" if conf_available else "no confidence data (binary mask only)")
    )

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    view_ids = sorted(p.stem.split("_")[1] for p in paths["per_view_dir"].glob("view_*.npz"))
    print(f"{len(view_ids)} views to process")

    pooled = {k: {"pts": [], "cols": [], "view_ids": []} for k in concepts}

    for vid in view_ids:
        png_path = paths["per_view_png_dir"] / f"view_{vid}.png"
        npz_path = paths["per_view_dir"] / f"view_{vid}.npz"
        img = Image.open(png_path).convert("RGB")
        d = np.load(npz_path)
        pts3d, valid_mask, img_no_norm = d["pts3d"], d["mask"], d["img_no_norm"]
        conf = d["conf"] if conf_available and "conf" in d.files else None

        state = processor.set_image(img)
        for concept, (prompt, thr, *_rest) in concepts.items():
            out = processor.set_text_prompt(state=state, prompt=prompt)
            masks, scores = out["masks"], out["scores"]
            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            for j, sc in enumerate(scores_list):
                if sc < thr:
                    continue
                m = masks[j]
                m_np = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(bool)
                if m_np.ndim == 3:
                    m_np = m_np[0]
                if m_np.shape != valid_mask.shape:
                    print(f"  SKIP {concept} view {vid} det{j}: mask shape {m_np.shape} != {valid_mask.shape}")
                    continue
                combined = m_np & valid_mask
                if conf is not None:
                    combined = combined & (conf >= conf_cutoff)
                n = combined.sum()
                if n < min_det_pixels:
                    continue
                raw_pts = pts3d[combined]  # this run's own raw frame - R/floor_y from THIS run
                aligned = raw_pts @ R.T
                aligned[:, 1] -= floor_y
                cols = img_no_norm[combined]
                pooled[concept]["pts"].append(aligned)
                pooled[concept]["cols"].append(cols)
                pooled[concept]["view_ids"].append(np.full(n, int(vid), dtype=np.int32))
        print(f"view {vid} done")

    print("=== pooling complete, clustering ===")

    objects = []
    for concept, (prompt, thr, eps, min_samples, min_cluster) in concepts.items():
        p = pooled[concept]
        if not p["pts"]:
            print(f"CONCEPT DROPPED: {concept!r} (prompt {prompt!r}) - zero detections above threshold {thr}")
            continue
        pts = np.concatenate(p["pts"], axis=0)
        cols = np.concatenate(p["cols"], axis=0)
        vids = np.concatenate(p["view_ids"], axis=0)

        # SAM3's mask for an open doorway/mirror-type prompt can include the opening
        # itself; MapAnything's monocular depth has no anchor past it and produces
        # runaway depth for those pixels. The centroid check below (kept as a second,
        # cheap layer) doesn't catch this: a cluster whose mass is real and in-room but
        # has a leaked tail still averages to an in-bounds centroid, while its
        # bbox/point_count/.ply all carry the leaked points untouched. Confirmed on real
        # production data (see docs/GPU_SETUP_LOG.md) - drop the leaked points
        # themselves, before they can affect centroid/bbox/count/DBSCAN at all.
        in_room = room_bbox_point_mask(pts, room_bbox_min, room_bbox_max)
        n_leaked = len(pts) - int(in_room.sum())
        if n_leaked:
            print(
                f"  {concept}: dropped {n_leaked}/{len(pts)} pooled points outside room "
                f"bbox (+{POINT_ROOM_MARGIN_M}m margin) - likely depth leak through an "
                f"opening/mirror"
            )
        pts, cols, vids = pts[in_room], cols[in_room], vids[in_room]
        if len(pts) == 0:
            print(f"CONCEPT DROPPED: {concept!r} (prompt {prompt!r}) - all pooled points were outside room bbox")
            continue
        n_raw = len(pts)
        print(f"{concept}: {n_raw} pooled pts from prompt {prompt!r}")

        voxel = voxel_size_for_eps(eps)
        keep = voxel_downsample_indices(pts, voxel)
        pts, cols, vids = pts[keep], cols[keep], vids[keep]
        compression_ratio = len(pts) / n_raw
        print(
            f"  voxel downsample (size={voxel:.4f}m, tied to eps={eps}): "
            f"{n_raw} -> {len(pts)} pts ({compression_ratio:.1%} kept)"
        )

        cap_idx = cap_points_uniform(len(pts), MAX_CLUSTER_POINTS)
        if cap_idx is not None:
            print(
                f"  exceeded MAX_CLUSTER_POINTS={MAX_CLUSTER_POINTS} after downsampling: "
                f"uniformly subsampled {len(pts)} -> {len(cap_idx)}"
            )
            pts, cols, vids = pts[cap_idx], cols[cap_idx], vids[cap_idx]

        eff_min_samples, eff_min_cluster = rescale_cluster_params(min_samples, min_cluster, compression_ratio)
        if (eff_min_samples, eff_min_cluster) != (min_samples, min_cluster):
            print(
                f"  density-rescaled cluster params ({compression_ratio:.1%} of raw density): "
                f"min_samples {min_samples}->{eff_min_samples}, min_cluster {min_cluster}->{eff_min_cluster}"
            )

        db = DBSCAN(eps=eps, min_samples=eff_min_samples).fit(pts)
        labels = db.labels_
        unique = sorted(set(labels) - {-1})
        print(f"  DBSCAN(eps={eps}, min_samples={eff_min_samples}): {len(unique)} raw clusters")

        kept = 0
        for lbl in unique:
            cmask = labels == lbl
            if cmask.sum() < eff_min_cluster:
                continue
            cpts, ccols, cvids = pts[cmask], cols[cmask], vids[cmask]
            centroid = cpts.mean(axis=0)
            bbox_min = cpts.min(axis=0)
            bbox_max = cpts.max(axis=0)

            if np.any(centroid < room_bbox_min) or np.any(centroid > room_bbox_max):
                print(
                    f"  DROPPED {concept}_{kept} (would-be): centroid {centroid.round(2).tolist()} "
                    f"outside room bbox - likely false-positive on scenery outside the room, "
                    f"{cmask.sum()} pts, {len(set(cvids.tolist()))} views"
                )
                continue

            n_views = len(set(cvids.tolist()))
            fragment = cmask.sum() < 300 or n_views <= 1
            obj_id = f"{concept}_{kept}"
            ply_path = paths["scene_objects_dir"] / f"{obj_id}.ply"

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cpts)
            pcd.colors = o3d.utility.Vector3dVector(np.clip(ccols, 0, 1))
            o3d.io.write_point_cloud(str(ply_path), pcd)

            objects.append(
                {
                    "id": obj_id,
                    "label": concept,
                    "prompt_used": prompt,
                    "centroid_xyz": centroid.tolist(),
                    "bbox_min": bbox_min.tolist(),
                    "bbox_max": bbox_max.tolist(),
                    "point_count": int(cmask.sum()),
                    "source_view_count": n_views,
                    "likely_fragment": bool(fragment),
                    # Relative to the job dir, not an absolute GPU path - the backend
                    # rebases this onto its own scene directory after fetching results.
                    "point_cloud_path": str(ply_path.relative_to(paths["job_dir"])),
                }
            )
            kept += 1
            flag = " [LIKELY FRAGMENT]" if fragment else ""
            print(
                f"  {obj_id}: {cmask.sum()} pts, centroid={centroid.round(3).tolist()}, "
                f"views={n_views}{flag}"
            )
        if kept == 0:
            print(f"CONCEPT DROPPED after clustering: {concept!r} - no cluster reached min_cluster={eff_min_cluster}")

    paths["scene_objects_json"].write_text(json.dumps(objects, indent=2))

    print(f"\n=== DONE: {len(objects)} objects written to {paths['scene_objects_json']} ===")
    for o in objects:
        flag = " [FRAGMENT]" if o["likely_fragment"] else ""
        print(f"  {o['id']}: centroid_y={o['centroid_xyz'][1]:.3f} pts={o['point_count']} views={o['source_view_count']}{flag}")


if __name__ == "__main__":
    main()
