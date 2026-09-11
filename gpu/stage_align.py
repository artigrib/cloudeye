#!/usr/bin/env python3
"""Stage 4: floor/ceiling disambiguation + alignment transform WRITER for this job's
single MapAnything inference run.

Runs under the mapanything venv. The highest-risk stage in the pipeline. Uses
floor_ceiling.py's camera-support-based disambiguation (see that module's docstring for
why a naive low-Y-band RANSAC isn't safe) and writes a single, complete, consistent
alignment_transform.npz + alignment.json - something that never existed as a real,
reusable artifact before this: the original floor_ceiling_disambiguate.py only
diagnosed floor_y/ceiling_y/tilt, it never computed a rotation or wrote anything.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import job_paths, load_params  # noqa: E402
from floor_ceiling import (  # noqa: E402
    DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M,
    DEFAULT_FLOOR_EXTENT_FRAC,
    DEFAULT_FLOOR_MIN_SUPPORT_FRAC,
    DEFAULT_RANDOM_SEED,
    confidence_percentile_cutoff,
    find_floor_and_ceiling,
    load_run,
    rotation_to_up,
    seed_ransac,
)

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# Below-floor phantom points: on a glossy/reflective floor, the depth model can read a
# floor-level object's mirror reflection as real geometry past the reflective plane -
# see setup-log.md's "reflective-floor phantom points" entry for the corridor case this
# was found on. The floor plane itself is RANSAC-fit with real majority support (not a
# guess), so once floor_y is established, anything below it minus a small RANSAC/plane-
# thickness margin cannot be real room geometry.
DEFAULT_FLOOR_CLIP_MARGIN_M = 0.05
# Fraction of room points clipped below floor above which the floor is flagged as likely
# reflective (a matte floor should clip a small fraction of a percent; a glossy one
# noticeably more - see setup-log.md for the corridor/bathroom numbers this was set from).
DEFAULT_REFLECTIVE_FLOOR_FRAC_THRESHOLD = 0.03
# Percentile of THIS RUN'S OWN pooled confidence distribution below which a point is
# dropped - relative, not a fixed absolute conf value, since MapAnything's confidence
# scale is not calibrated to be comparable across separate .infer() runs any more than
# its coordinate frame is (see floor_ceiling.py's module docstring). No-ops entirely
# when a run has no confidence data at all - see floor_ceiling.load_run's docstring.
DEFAULT_CONF_THRESHOLD_PERCENTILE = 10.0


def run_align(job_dir: str | Path, *, out_dir: str | Path | None = None) -> dict:
    """The actual alignment computation, factored out of `main()` so it can be re-run
    against an existing job's already-present `per_view/*.npz` without going through
    the full CLI/`run_pipeline.sh` flow - see `rerun_align.py`, which is the only other
    caller today. `out_dir` defaults to `job_dir` itself (this stage's normal, in-place
    behavior); passing a different directory (e.g. a `aligned_v2/` sibling) writes the
    same four output files there instead, leaving the job's existing artifacts alone -
    the point being an apples-to-apples before/after on real data without destroying
    the "before".

    Returns a plain-dict summary (floor_y/ceiling_y/floor_likely_wrong/clip fraction/
    the full per-candidate diagnostics list) so callers get these values directly
    instead of re-parsing the JSON this also writes to disk.
    """
    paths = job_paths(job_dir)
    params = load_params(job_dir)
    out_dir = Path(out_dir) if out_dir is not None else paths["job_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_ply_path = out_dir / "aligned_room.ply"
    transform_npz_path = out_dir / "alignment_transform.npz"
    alignment_json_path = out_dir / "alignment.json"
    cameras_json_path = out_dir / "cameras_aligned.json"

    min_view_support = params.get("min_view_support", 3)
    bbox_margin = params.get("room_bbox_margin", 0.3)
    floor_clip_margin = params.get("floor_clip_margin", DEFAULT_FLOOR_CLIP_MARGIN_M)
    reflective_floor_frac_threshold = params.get(
        "reflective_floor_frac_threshold", DEFAULT_REFLECTIVE_FLOOR_FRAC_THRESHOLD
    )
    floor_min_support_frac = params.get(
        "floor_min_support_frac", DEFAULT_FLOOR_MIN_SUPPORT_FRAC
    )
    floor_camera_height_range = (
        params.get("floor_camera_height_min_m", DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M[0]),
        params.get("floor_camera_height_max_m", DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M[1]),
    )
    floor_extent_frac = params.get("floor_extent_frac", DEFAULT_FLOOR_EXTENT_FRAC)
    conf_threshold_percentile = params.get(
        "conf_threshold_percentile", DEFAULT_CONF_THRESHOLD_PERCENTILE
    )

    # Determinism. Both RNG sources in this stage's floor-selection path are global and
    # were unseeded: numpy's (the room-bbox DBSCAN subsample below) and Open3D's
    # (segment_plane's RANSAC, reached via floor_ceiling.fit_plane from
    # extract_horizontal_plane_candidates - see seed_ransac's docstring). Repeat runs of
    # identical input were observed to report a real spread in below-floor clip fraction
    # purely from RANSAC sampling - the pass/fail verdict was stable, but "which run's
    # number is real" was not answerable, which is untenable for the pipeline's highest-
    # risk stage. Always seeded (there is no "off"); `random_seed` only chooses WHICH
    # seed, so a deliberate sensitivity study is a params.json edit rather than a code edit.
    random_seed = params.get("random_seed", DEFAULT_RANDOM_SEED)
    rng = np.random.default_rng(random_seed)
    seeded_o3d = seed_ransac(random_seed)
    print(
        f"random seed {random_seed} (numpy: yes, open3d RANSAC: "
        f"{'yes' if seeded_o3d else 'NO - this open3d build has no o3d.utility.random.seed'})"
    )

    all_pts, all_cols, view_idx, cam_positions, all_conf, cam_poses, cam_intrinsics = load_run(
        paths["per_view_dir"], include_poses=True
    )
    print(f"pooled {len(all_pts)} valid points across {cam_positions.shape[0]} views")

    # Step A: DBSCAN on a subsample just to find the room cluster's spatial bounding
    # box - well-conditioned on a subsample, unlike the floor-plane fit itself.
    if len(all_pts) > 300_000:
        sub_idx = rng.choice(len(all_pts), 300_000, replace=False)
    else:
        sub_idx = np.arange(len(all_pts))
    sub_pts = all_pts[sub_idx]

    db_room = DBSCAN(eps=0.08, min_samples=15).fit(sub_pts)
    labels_room = db_room.labels_
    uniq_room, counts_room = np.unique(labels_room[labels_room >= 0], return_counts=True)
    if len(uniq_room) == 0:
        raise RuntimeError(
            "DBSCAN found no cluster at all in the pooled point cloud - reconstruction "
            "likely failed upstream (stage_infer), not something to work around here"
        )
    main_room_label = uniq_room[np.argmax(counts_room)]
    room_core_sub = sub_pts[labels_room == main_room_label]
    print(
        f"main room cluster (subsample): {len(room_core_sub)}/{len(sub_pts)} pts "
        f"({100 * len(room_core_sub) / len(sub_pts):.1f}%)"
    )

    # Step B: apply that cluster's bounding box (small margin) to the FULL point
    # population, not the subsample - random subsampling was found to unreliably wash
    # out the sparse floor-band signal (two runs of a subsample-only approach found
    # wildly different floor tilts, 1.89deg vs 79deg, purely from which random points
    # happened to get picked).
    margin = 0.05
    bb_min = room_core_sub.min(axis=0) - margin
    bb_max = room_core_sub.max(axis=0) + margin
    room_box_mask = np.all((all_pts >= bb_min) & (all_pts <= bb_max), axis=1)
    room_pts = all_pts[room_box_mask]
    room_cols = all_cols[room_box_mask]
    room_view_idx = view_idx[room_box_mask]
    room_conf = all_conf[room_box_mask] if all_conf is not None else None
    print(f"room bounding-box filter on full cloud: {len(room_pts)}/{len(all_pts)} pts")

    room_bbox_xy_area = float(bb_max[0] - bb_min[0]) * float(bb_max[2] - bb_min[2])
    fc = find_floor_and_ceiling(
        room_pts,
        room_view_idx,
        cam_positions,
        min_view_support=min_view_support,
        room_bbox_xy_area=room_bbox_xy_area,
        floor_min_support_frac=floor_min_support_frac,
        floor_camera_height_range=floor_camera_height_range,
        floor_extent_frac=floor_extent_frac,
    )
    for w in fc.warnings:
        print(f"WARNING: {w}")
    print(
        f"chosen floor: {fc.chosen} band, tilt={fc.floor.tilt_deg:.2f}deg, "
        f"support={fc.floor.support} pts / {fc.floor.n_views_supporting} views"
    )
    print("diagnostics:", json.dumps(fc.diagnostics, indent=2))

    R = rotation_to_up(fc.up_normal)
    det_r = float(np.linalg.det(R))
    is_reflection = det_r < 0
    print(f"det(R)={det_r:.6f} ({'REFLECTION' if is_reflection else 'proper rotation'})")

    room_pts_rot = room_pts @ R.T
    floor_inliers_rot = fc.floor.inlier_points @ R.T
    ceiling_inliers_rot = fc.ceiling.inlier_points @ R.T

    # floor_y/ceiling_y derived from each SELECTED plane's own validated inlier points,
    # never a fresh density-percentile re-threshold of the whole cloud - that was the
    # exact bug that produced a bogus ~86deg "residual" earlier in this project (a
    # percentile threshold silently re-selects wall points, since floor points are
    # sparse relative to walls).
    floor_y = float(np.median(floor_inliers_rot[:, 1]))
    ceiling_y_raw = float(np.median(ceiling_inliers_rot[:, 1]))

    room_pts_aligned = room_pts_rot.copy()
    room_pts_aligned[:, 1] -= floor_y
    ceiling_y = ceiling_y_raw - floor_y  # ceiling height in the now-floor-at-0 frame

    print(f"floor_y={floor_y:.4f} ceiling_y={ceiling_y:.4f} (aligned frame, floor=0)")
    print(
        f"post-align Y range: {room_pts_aligned[:, 1].min():.3f} to "
        f"{room_pts_aligned[:, 1].max():.3f}"
    )

    # Below-floor phantom clip. The floor plane was RANSAC-fit with real majority
    # support (not assumed), so anything below it - past a small margin for RANSAC
    # noise and the plane's own thickness - cannot be real geometry. On a reflective
    # floor this removes the mirror-artifact points; on a normal floor it removes
    # essentially nothing. Never clips above the ceiling - see setup-log.md for why
    # that side has no equally strong physical argument.
    n_before_clip = len(room_pts_aligned)
    floor_keep_mask = room_pts_aligned[:, 1] >= -floor_clip_margin
    n_points_below_floor = n_before_clip - int(floor_keep_mask.sum())
    points_below_floor_frac = n_points_below_floor / n_before_clip if n_before_clip else 0.0
    reflective_floor_suspected = points_below_floor_frac > reflective_floor_frac_threshold

    print(
        f"floor clip (margin={floor_clip_margin:.3f}m): removed "
        f"{n_points_below_floor}/{n_before_clip} points below floor "
        f"({points_below_floor_frac:.2%})"
    )
    # A reflective floor clips a modestly elevated fraction (mirror-artifact points are
    # a thin band right below the true floor). A WILDLY elevated fraction instead means
    # the clip is removing real geometry that sits below a floor_y that is itself wrong
    # - e.g. the furniture-plane-as-floor failure this margin was never meant to catch
    # (see floor_ceiling.py's select_floor_candidate). Distinguish the two rather than
    # silently applying the same "probably reflective" story to both.
    floor_likely_wrong = points_below_floor_frac > reflective_floor_frac_threshold * 3
    if floor_likely_wrong:
        print(
            f"WARNING: points_below_floor_frac={points_below_floor_frac:.2%} exceeds "
            f"3x the reflective-floor threshold ({3 * reflective_floor_frac_threshold:.0%}) "
            f"- floor may be wrong (selected plane is likely not the true floor), not "
            f"just reflective. Check the per-candidate diagnostics printed above."
        )
    elif reflective_floor_suspected:
        print(
            f"WARNING: points_below_floor_frac={points_below_floor_frac:.2%} exceeds "
            f"threshold {reflective_floor_frac_threshold:.0%} - floor is likely "
            f"reflective/glossy; reconstruction quality near the floor plane may be "
            f"degraded (see setup-log.md - this is a known limitation, not something "
            f"this clip fully corrects)"
        )

    room_pts_aligned = room_pts_aligned[floor_keep_mask]
    room_cols = room_cols[floor_keep_mask]
    room_conf = room_conf[floor_keep_mask] if room_conf is not None else None

    # Percentile-based confidence filter, applied before SOR (SOR is a spatial-outlier
    # check; this is upstream data-quality, and low-confidence points can BE the "fly-
    # away" points SOR later removes - filtering by confidence first when it's available
    # gives SOR a cleaner set to reason about). See DEFAULT_CONF_THRESHOLD_PERCENTILE's
    # comment for why this is percentile-based, not a fixed absolute conf value.
    conf_available = room_conf is not None and not np.all(np.isnan(room_conf))
    conf_cutoff = float("nan")
    if conf_available:
        conf_cutoff, conf_keep_mask = confidence_percentile_cutoff(
            room_conf, conf_threshold_percentile
        )
        n_before_conf = len(room_pts_aligned)
        room_pts_aligned = room_pts_aligned[conf_keep_mask]
        room_cols = room_cols[conf_keep_mask]
        room_conf = room_conf[conf_keep_mask]
        print(
            f"conf filter (percentile={conf_threshold_percentile:.1f}, "
            f"cutoff={conf_cutoff:.4f}): removed "
            f"{n_before_conf - len(room_pts_aligned)}/{n_before_conf} low-confidence points"
        )
    else:
        print("conf filter: skipped - no per-view confidence data for this run (binary-mask fallback)")

    # Statistical outlier removal: MapAnything produces sparse fly-away points (mostly
    # near depth discontinuities/object edges) that survive the below-floor clip because
    # they're not below the floor, just isolated. These are the single biggest
    # contributor to the point cloud's "snowy" look. nb_neighbors=20/std_ratio=2.0 are
    # Open3D's own documented defaults - not tuned per-scene, since point density varies
    # a lot between captures and a fixed distance threshold wouldn't generalise.
    # Runs on the SAME floor_keep_mask-filtered set the residual-tilt check and bbox
    # below use, so occupancy (stage 6) and the GLB (stage 7) both see the cleaned cloud.
    if params.get("sor_enabled", True):
        n_before_sor = len(room_pts_aligned)
        pcd_sor = o3d.geometry.PointCloud()
        pcd_sor.points = o3d.utility.Vector3dVector(room_pts_aligned)
        _, sor_keep_idx = pcd_sor.remove_statistical_outlier(
            nb_neighbors=params.get("sor_nb_neighbors", 20),
            std_ratio=params.get("sor_std_ratio", 2.0),
        )
        room_pts_aligned = room_pts_aligned[sor_keep_idx]
        room_cols = room_cols[sor_keep_idx]
        print(
            f"SOR: removed {n_before_sor - len(room_pts_aligned)}/{n_before_sor} outlier "
            f"points ({(n_before_sor - len(room_pts_aligned)) / n_before_sor:.2%})"
        )

    # Residual-tilt verification: re-fit on the SAME validated floor inlier set
    # (rotated + shifted), never a fresh re-threshold of the aligned cloud - see the
    # comment above for why a fresh threshold is unsafe here.
    floor_check = floor_inliers_rot.copy()
    floor_check[:, 1] -= floor_y
    pcd_check = o3d.geometry.PointCloud()
    pcd_check.points = o3d.utility.Vector3dVector(floor_check)
    plane_model, _inliers = pcd_check.segment_plane(
        distance_threshold=0.025, ransac_n=3, num_iterations=2000
    )
    n2 = np.array(plane_model[:3])
    n2 /= np.linalg.norm(n2)
    residual_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(n2[1]), -1, 1))))
    print(f"RESIDUAL tilt after alignment: {residual_tilt_deg:.3f} deg (should be small)")

    bbox_min = (room_pts_aligned.min(axis=0) - bbox_margin).tolist()
    bbox_max = (room_pts_aligned.max(axis=0) + bbox_margin).tolist()

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(room_pts_aligned)
    pcd_out.colors = o3d.utility.Vector3dVector(np.clip(room_cols, 0, 1))
    o3d.io.write_point_cloud(str(aligned_ply_path), pcd_out)

    np.savez(
        transform_npz_path,
        R=R,
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        bbox_min=np.array(bbox_min),
        bbox_max=np.array(bbox_max),
        residual_tilt_deg=residual_tilt_deg,
        det_r=det_r,
        is_reflection=is_reflection,
        n_views=cam_positions.shape[0],
        frac_cameras_between=fc.frac_cameras_between,
        floor_support=fc.floor.support,
        ceiling_support=fc.ceiling.support,
        points_below_floor=n_points_below_floor,
        points_below_floor_frac=points_below_floor_frac,
        reflective_floor_suspected=reflective_floor_suspected,
        floor_likely_wrong=floor_likely_wrong,
        conf_available=conf_available,
        conf_cutoff=conf_cutoff,
        conf_threshold_percentile=conf_threshold_percentile,
        schema_version=5,
    )

    # Plain-JSON mirror of the same content, so scene_ingest.py on the backend never
    # needs to know npz key names/dtypes for the common case - it's a direct read.
    alignment_json_path.write_text(
        json.dumps(
            {
                "floor_y": floor_y,
                "ceiling_y": ceiling_y,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "residual_tilt_deg": residual_tilt_deg,
                "det_r": det_r,
                "is_reflection": is_reflection,
                "n_views": int(cam_positions.shape[0]),
                "frac_cameras_between": fc.frac_cameras_between,
                "floor_support": fc.floor.support,
                "ceiling_support": fc.ceiling.support,
                "chosen_band": fc.chosen,
                "warnings": fc.warnings,
                "points_below_floor": n_points_below_floor,
                "points_below_floor_frac": points_below_floor_frac,
                "reflective_floor_suspected": reflective_floor_suspected,
                "floor_likely_wrong": floor_likely_wrong,
                "floor_hull_area": fc.floor.hull_area,
                "room_bbox_xy_area": room_bbox_xy_area,
                "conf_available": conf_available,
                "conf_cutoff": conf_cutoff if conf_available else None,
                "conf_threshold_percentile": conf_threshold_percentile,
                "schema_version": 5,
            },
            indent=2,
        )
    )

    # Cameras in the aligned frame - the backend uses these to choose robot_start and
    # to independently re-verify "floor below cameras, ceiling above" in the validator.
    cams_aligned = cam_positions @ R.T
    cams_aligned[:, 1] -= floor_y

    # Full 4x4 extrinsics (camera-to-world) + 3x3 intrinsics (K), per frame, in the SAME
    # aligned room frame as `cameras`/`aligned_room.ply` - pose/intrinsics retention,
    # originally added for the TSDF fusion spike (docs/DECISIONS.md's 2026-09-03
    # Workstream C/D entries; TSDF itself was spiked and rejected, but this retention is
    # independent, validated infrastructure kept for any future per-view-pose consumer).
    # This is additive: the pre-existing "cameras" key (position-only) is unchanged, so
    # `scene_ingest.read_camera_track` and every other existing consumer keeps working
    # with zero changes.
    #
    # Math: stage_align's alignment is p_aligned = R @ p_world - [0, floor_y, 0], applied
    # here identically to cam_positions above. As a single rigid transform on homogeneous
    # world points, that's A = [[R, t], [0, 0, 0, 1]] with t = [0, -floor_y, 0].
    # `camera_pose` from stage_infer.py's per-view .npz is camera-to-world
    # (p_world = camera_pose @ p_cam), so the same extrinsic matrix re-expressed in the
    # aligned frame is A @ camera_pose (world points transform via A, and camera_pose
    # already maps camera-local points to world points, so composing on the left carries
    # that through unchanged). Intrinsics (K) are untouched by any rigid transform of the
    # world frame - each camera's own image-formation model doesn't change, only where
    # it's *placed*.
    A = np.eye(4)
    A[:3, :3] = R
    A[1, 3] = -floor_y
    # numpy's `@` broadcasts a single (4,4) matrix against a (V,4,4) stack correctly
    # (verified: equivalent to np.einsum("ij,vjk->vik", A, cam_poses)).
    cam_extrinsics_aligned = A @ cam_poses

    cameras_json_path.write_text(
        json.dumps(
            {
                "cameras": cams_aligned.tolist(),
                "extrinsics": cam_extrinsics_aligned.tolist(),
                "intrinsics": cam_intrinsics.tolist(),
                "schema_version": 2,
            },
            indent=2,
        )
    )

    print(
        f"DONE: saved {aligned_ply_path.name} ({len(room_pts_aligned)} pts) to {out_dir}, "
        f"{transform_npz_path.name}, {alignment_json_path.name}"
    )

    return {
        "out_dir": str(out_dir),
        "floor_y": floor_y,
        "ceiling_y": ceiling_y,
        "chosen_band": fc.chosen,
        "floor_support": fc.floor.support,
        "ceiling_support": fc.ceiling.support,
        "points_below_floor": n_points_below_floor,
        "points_below_floor_frac": points_below_floor_frac,
        "reflective_floor_suspected": reflective_floor_suspected,
        "floor_likely_wrong": floor_likely_wrong,
        "residual_tilt_deg": residual_tilt_deg,
        "conf_available": conf_available,
        "conf_cutoff": conf_cutoff,
        "conf_threshold_percentile": conf_threshold_percentile,
        "warnings": fc.warnings,
        "diagnostics": fc.diagnostics,
    }


def main() -> None:
    run_align(sys.argv[1])


if __name__ == "__main__":
    main()
