#!/usr/bin/env python3
"""Steps 5-6: nvblox TSDF+ESDF over a set of packed 15 fps scenes.

Per scene it writes, into <out-root>/<scene>/:
  mesh.ply                nvblox marching-cubes mesh
  esdf_slice_0.3m.npy     float32, NaN where the ESDF is unobserved
  unobserved_slice.npy    bool, True where unobserved (same grid)
  esdf_3d.npy             float16 dense ESDF over the map AABB, NaN = unobserved
  esdf_3d_meta.json       origin/shape/resolution for the above
  map.nvblx               nvblox's own serialized map (all layers), if save_map works
  floor_plane.json        RANSAC floor: normal, point, camera height, all candidates
  stats.json              timings, VRAM, mesh counts, surface probe, slice occupancy

Conventions verified in step 2 and reused here unchanged:
  add_depth_frame wants t_w_c = world <- camera (cam->world), float32, pose on CPU;
  query_layer(QueryType.ESDF, q) needs q as Nx4 (x,y,z,radius) despite the docstring;
  the unknown sentinel is constants.esdf_unknown_distance() = +100.0.

The reference "fused" surface points are re-derived here by back-projecting the same
depth maps that are integrated - the same points an Open3D TSDF would be fed - so the
probe does not depend on an external fused cloud that only the hero scene has.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------------
# COPY of nvblox_v2/nvblox_scenes.py with ONE change: the floor RANSAC is made reproducible.
# nvblox_v2 itself is read-only from this repo, hence the copy rather than an edit there.
#
# Why: the 2026-09-09 live run reproduced yesterday's mesh EXACTLY (358 958 verts / 545 622
# tris / 8767 components, byte-identical mesh.ply) from byte-identical input on the same box,
# but produced a DIFFERENT floor plane - normal [0.018956, -0.981482, -0.190615] against
# [0.016326, -0.981632, -0.190083], inlier_frac 0.05464 vs 0.05605. The slice frame that
# plane defines then shifted by ~3 cm, which moved 38 of 10 500 slice cells between observed
# and unobserved and put max|diff| 0.21 m into esdf_slice_0.3m.npy. Integration and meshing
# are deterministic; the floor fit was not.
#
# What fixes it, measured rather than assumed (4 fresh processes each):
#   * `o3d.utility.random.seed(n)` ALONE is NOT enough - results still varied run to run.
#   * seed + a single OpenMP thread IS reproducible - 3/3 identical planes, both the floor
#     and the second candidate.
# `segment_plane` in open3d 0.19 takes no `seed` argument (checked its signature), so the
# global RNG is the only handle, and its multi-threaded reduction order is the other half of
# the problem. OMP_NUM_THREADS must be set before open3d is imported, so it is set here at
# module scope; only this script's own Open3D work is affected (floor fit + component
# counting). nvblox's own integrate/ESDF/mesh run on CUDA and are untouched by it.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from nvblox_torch.constants import constants
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.mapper_params import (EsdfIntegratorParams, MapperParams,
                                        ProjectiveIntegratorParams)
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

UNKNOWN = constants.esdf_unknown_distance()
DEV = "cuda"


def build_mapper(voxel, trunc_vox, max_int_dist, esdf_max):
    pi = ProjectiveIntegratorParams()
    pi.projective_integrator_max_integration_distance_m = max_int_dist
    pi.projective_integrator_truncation_distance_vox = trunc_vox
    ei = EsdfIntegratorParams()
    ei.esdf_integrator_max_distance_m = esdf_max
    p = MapperParams()
    p.set_projective_integrator_params(pi)
    p.set_esdf_integrator_params(ei)
    m = Mapper(voxel_sizes_m=voxel, integrator_types=ProjectiveIntegratorType.TSDF,
               mapper_parameters=p)
    e = m.params().get_projective_integrator_params()
    return m, {"voxel_size_m": voxel,
               "max_integration_distance_m": e.projective_integrator_max_integration_distance_m,
               "truncation_distance_vox": e.projective_integrator_truncation_distance_vox,
               "weighting_mode": str(e.projective_integrator_weighting_mode),
               "esdf_max_distance_m": m.params().get_esdf_integrator_params()
                                       .esdf_integrator_max_distance_m}


def query_esdf(mapper, pts, radius=0.0, chunk=500000) -> np.ndarray:
    out = []
    for i in range(0, len(pts), chunk):
        p = np.asarray(pts[i:i + chunk], np.float32)
        q = np.concatenate([p, np.full((len(p), 1), radius, np.float32)], axis=1)
        t = torch.as_tensor(np.ascontiguousarray(q), device=DEV)
        out.append(mapper.query_layer(QueryType.ESDF, t, mapper_id=0)
                   .detach().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def fused_points_from_depth(stack, views, n_points, max_int_dist, seed=42):
    """Back-project the integrated depth maps - the same surface an Open3D TSDF sees."""
    rng = np.random.default_rng(seed)
    vi = rng.choice(len(views), min(300, len(views)), replace=False)
    per = max(1, n_points // len(vi) * 4)
    pts = []
    for i in vi:
        d = stack[i].astype(np.float32) / 1000.0
        d[d > max_int_dist] = 0.0
        ys, xs = np.nonzero(d)
        if len(ys) == 0:
            continue
        sel = rng.choice(len(ys), min(per, len(ys)), replace=False)
        ys, xs = ys[sel], xs[sel]
        z = d[ys, xs]
        K = np.asarray(views[i]["intrinsics"], np.float64)
        x = (xs - K[0, 2]) / K[0, 0] * z
        y = (ys - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z, np.ones_like(z)], 1)
        P = np.asarray(views[i]["camera_pose"], np.float64)
        pts.append((P @ cam.T).T[:, :3])
    all_pts = np.concatenate(pts)
    idx = rng.choice(len(all_pts), min(n_points, len(all_pts)), replace=False)
    return all_pts[idx].astype(np.float32), all_pts.astype(np.float32)


# The camera-height band gpu/floor_ceiling.py validated for this exact purpose
# (its DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M). Used here as a plausibility check on the
# chosen plane, never as the thing that chooses it.
DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M = (0.8, 2.0)


class FloorFitError(RuntimeError):
    """No plane could be fitted at all. Caught in run_scene, which writes a stats.json
    carrying `floor_fit_failed` and returns - preserved from the original."""


def _rotation_hint(by_rot: dict, gate_deg: float) -> str:
    """Append the four-rotation angles to a failure message, and say plainly when one of the
    other three would have passed. States the diagnosis; applies nothing."""
    if not by_rot:
        return ""
    txt = "; angle under each frame rotation: " + ", ".join(
        f"{d} deg -> {a:.1f}" for d, a in sorted(by_rot.items()))
    better = sorted(d for d, a in by_rot.items() if d != 0 and a <= gate_deg)
    quarter = [d for d in better if d in (90, 270)]
    if quarter:
        txt += (f". NOTE: rotation(s) {quarter} would pass the {gate_deg:.0f} deg gate - "
                "that is the signature of video rotation metadata dropped at the FRAMES "
                "stage, not a bad floor. Nothing was applied; fix it at the source")
    elif better == [180]:
        # A 180 deg frame rotation flips the camera's own "up" to "down", which is exactly
        # what picking the CEILING instead of the floor also looks like from here. The two
        # are degenerate in this test and must not be reported as one.
        txt += (". NOTE: 180 deg would pass the gate, but that is ambiguous - an upside-down "
                "video and a ceiling picked instead of the floor are indistinguishable by "
                "this test. Check the frame orientation recorded at FRAMES before assuming "
                "either. Nothing was applied")
    return txt


class FloorImplausible(RuntimeError):
    """A plane WAS fitted but does not look like a floor. Deliberately NOT caught: the
    step must fail loudly rather than hand downstream stages a wall to build a slice on.
    """


def median_camera_up(views) -> np.ndarray:
    """World-space "up", as the cameras themselves define it.

    OpenCV camera convention (x right, y DOWN, z forward), so "up" in the camera frame is
    -y; `camera_pose` is cam->world, hence -R[:, 1] per view. Median over all views, then
    normalised - a median rather than a mean so a handful of views pointing at the ceiling
    or the floor cannot drag it.

    This is the same principle `gpu/floor_ceiling.py` uses ("orient 'up' using the actual
    direction from the chosen floor toward the cameras - never an assumption about raw-Y
    sign"); that module cannot be reused here because its API needs the full per-view run
    tree, and the box only receives depth_u16.npy + meta.json.
    """
    ups = np.asarray([-np.asarray(v["camera_pose"], float)[:3, 1] for v in views], float)
    med = np.median(ups, axis=0)
    n = float(np.linalg.norm(med))
    if n < 1e-9:
        raise FloorFitError("median camera up is degenerate - cameras have no common up")
    return med / n


def camera_up_variants(views) -> dict:
    """The world "up" the cameras would define under each of the four frame rotations.

    A video whose rotation metadata was ignored yields frames turned 90, 180 or 270 degrees
    about the optical axis, which in the OpenCV frame is z. Rotating the image by k*90 turns
    the camera's own "up" from -y to +x, +y, -x in turn, so the four world candidates are
    -R[:,1], R[:,0], R[:,1], -R[:,0]. Diagnosis only: nothing here applies a rotation or
    corrects anything - a wrong rotation is a FRAMES-stage bug and belongs fixed at the
    source, not compensated three stages downstream.
    """
    R = np.asarray([np.asarray(v["camera_pose"], float)[:3, :3] for v in views])
    out = {}
    for deg, vec in ((0, -R[:, :, 1]), (90, R[:, :, 0]),
                     (180, R[:, :, 1]), (270, -R[:, :, 0])):
        med = np.median(vec, axis=0)
        n = float(np.linalg.norm(med))
        out[deg] = (med / n) if n > 1e-9 else None
    return out


def angle_to_deg(a: np.ndarray, b: np.ndarray) -> float:
    """SIGNED angle between two oriented unit vectors, in degrees - never abs().

    Using the absolute angle would treat a ceiling as a floor: on the hero scene the
    ceiling candidates sit at 174.14 deg and 174.62 deg from the median camera up, which
    abs() folds to 5.86 and 5.38 - indistinguishable from a real floor at 0.00 deg. That
    floor/ceiling confusion is the exact failure `gpu/floor_ceiling.py` was written for
    (its Motivation #1). Candidate normals are already oriented toward the cameras, so a
    real floor lands near 0 deg, a ceiling near 180, a wall near 90.
    """
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def fit_floor(points: np.ndarray, cam_centroid: np.ndarray, seed: int = 42,
              camera_up: np.ndarray | None = None, support_frac: float = 0.2,
              gate_deg: float = 30.0,
              cam_height_range: tuple = DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M,
              ransac_iterations: int = 9000, up_variants: dict | None = None,
              union_deg: float = 0.0):
    """Floor = lowest substantial near-horizontal plane below the camera path.

    `seed` pins Open3D's global RNG, which is what `segment_plane` draws its RANSAC samples
    from (it exposes no seed parameter of its own in 0.19). This is only half of what makes
    the fit reproducible - see the module header: OMP_NUM_THREADS=1 is the other half.

    **How the reference "up" is chosen, and why it changed.** The original rule took the
    single best-supported candidate as the reference. On the hero scene the real floor
    (inlier 0.0560) and a wall (0.0502) are within 4% of each other, so which one wins is
    decided by the random draw: measured over 9 seeds, **4 of 9 picked the wall**. Seeding
    alone does not fix that - it only freezes the coin, and seed 42 froze it on the wall.

    The rule now is: among candidates supported at least `support_frac` of the best, take
    the one whose (camera-oriented) normal is closest to `camera_up`. This is the same
    shape of fix `gpu/floor_ceiling.py:select_floor_candidate` already applies for the same
    reason - "not the single most-supported one - that was the bug".

    `support_frac` is 0.2, the same value `gpu/floor_ceiling.py` settled on. It was tried
    at 0.8 first and measured: on `own_0829_000840` the real floor IS among the candidates
    at 5.9 deg, but its support falls below 0.8 x the best, so it never enters the pool,
    the reference becomes a wall at 85.3 deg and the gate fails a scene that used to work.
    At 0.2 that scene returns a floor at 5.89 deg with camera height 1.271 m against the
    published 1.293 m. Widening changed nothing on the scenes that already passed: hero and
    own_0828_152850 give bit-identical angles at 0.2 and 0.8. Wider is also structurally
    safer here, because the pick is by minimum angle - adding candidates can only offer a
    better-aligned one, never a worse one.

    Then a plausibility gate: if the CHOSEN plane sits more than `gate_deg` from the camera
    up, raise `FloorImplausible` rather than return a wall. Measured separation on real
    data is not marginal - floors 0.0-10.1 deg, ceilings ~174 deg, walls ~90 deg.
    """
    import open3d as o3d
    o3d.utility.random.seed(int(seed))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pc = pc.voxel_down_sample(0.03)
    cands = []
    for _ in range(8):
        if len(pc.points) < 5000:
            break
        pl, inl = pc.segment_plane(0.03, 3, ransac_iterations)
        n = np.asarray(pl[:3], float); nn = n / np.linalg.norm(n)
        pts = np.asarray(pc.points)[inl]
        c = pts.mean(0)
        up = nn if np.dot(cam_centroid - c, nn) > 0 else -nn
        cands.append({"up": up, "centroid": c, "inlier_frac": len(inl) / len(pc.points),
                      "cam_height_m": float(np.dot(cam_centroid - c, up)),
                      "_inliers": pts})
        pc = pc.select_by_index(inl, invert=True)
    if not cands:
        raise FloorFitError("no plane could be fitted - too few points survive")
    if camera_up is None:
        # Only reachable from a caller that has no poses; keep the old behaviour rather
        # than silently inventing an up direction.
        ref = max(cands, key=lambda x: x["inlier_frac"])["up"]
        pool_n = len(cands)
    else:
        best_support = max(c["inlier_frac"] for c in cands)
        pool = [c for c in cands if c["inlier_frac"] >= support_frac * best_support]
        for c in cands:
            c["angle_to_camera_up_deg"] = angle_to_deg(np.asarray(c["up"]), camera_up)
        ref = min(pool, key=lambda c: c["angle_to_camera_up_deg"])["up"]
        pool_n = len(pool)

    # --- pick among the floor-oriented candidates by SUPPORT, not by height -------------
    # The old rule took the greatest camera height among near-horizontal candidates, i.e.
    # the lowest plane. Measured consequence: different RANSAC draws find different portions
    # of the same floor (hero inlier counts range 5149-10809 across 9 seeds) and the height
    # tiebreak then settles on a different patch, which is where the whole residual spread
    # came from - the sparsest draws were exactly the ones furthest off camera up.
    #
    # Support is the stabler key: the largest patch of floor the draw found is the one most
    # likely to be the floor proper rather than a fragment of it. Height stops being the
    # selector and becomes a plausibility CHECK, over the band gpu/floor_ceiling.py already
    # validated for this purpose (DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M = (0.8, 2.0)).
    if camera_up is not None:
        oriented = [c for c in cands
                    if c["angle_to_camera_up_deg"] < gate_deg
                    and c["inlier_frac"] >= 0.015]
        best = max(oriented, key=lambda x: x["inlier_frac"]) if oriented else \
            max(cands, key=lambda x: x["inlier_frac"])
    else:
        horiz = [c for c in cands if np.dot(c["up"], ref) > 0.9
                 and c["inlier_frac"] >= 0.015 and 0.5 <= c["cam_height_m"] <= 2.5]
        best = max(horiz, key=lambda x: x["cam_height_m"]) if horiz else \
            max(cands, key=lambda x: x["inlier_frac"])

    # --- least-squares refit of the chosen plane, and what it measured -------------------
    # Refit the chosen candidate's plane to ALL of its inliers by PCA: the
    # smallest-singular-vector direction of the centred inlier matrix is the least-squares
    # normal, and the inlier mean is a point on it. This never re-decides WHICH plane was
    # chosen, only re-fits the one already picked.
    #
    # MEASURED RESULT: it changes nothing. Over 9 seeds x 2 scenes the refit moved the normal
    # by at most 0.001 deg and the camera height by under 0.1 mm, and the seed-to-seed spread
    # was unchanged - hero 4.338 -> 4.339 deg and 4.57 -> 4.57 cm, own_0828_152850 1.333 ->
    # 1.333 deg and 12.09 -> 12.09 cm. Open3D 0.19's `segment_plane` already returns the
    # least-squares fit over its final inlier set, so this reproduces its own answer; that is
    # a direct measurement (same inliers in, same plane out to 4 decimals), not a claim about
    # its source.
    #
    # The residual spread is therefore plane SELECTION, not plane FITTING: on hero the inlier
    # count ranges 5149-10809 across seeds, and the sparsest draws (seed 0 at 6089, seed 99
    # at 5149) are exactly the ones furthest off camera-up (4.49 and 4.67 deg). Different
    # draws find different portions of the same floor, and the `max(cam_height_m)` tiebreak
    # then settles on a different patch. Reducing that would mean changing the tiebreak, which
    # is a separate decision and is NOT done here.
    #
    # What the block is kept for: `rms_inlier_dist_m` (0.013-0.017 m here) and
    # `n_inliers_refined` are real diagnostics - they say how tight the floor is and how much
    # of it this draw actually found, which is what explains the spread above.
    pts = best.get("_inliers")
    if pts is not None and len(pts) >= 3:
        centroid = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        n_ref = vt[-1] / np.linalg.norm(vt[-1])
        if np.dot(n_ref, best["up"]) < 0:      # keep the camera-facing orientation
            n_ref = -n_ref
        d = (pts - centroid) @ n_ref
        best["raw_up"] = np.asarray(best["up"]).tolist()
        best["raw_centroid"] = np.asarray(best["centroid"]).tolist()
        best["raw_cam_height_m"] = best["cam_height_m"]
        best["rms_inlier_dist_m"] = float(np.sqrt(np.mean(d ** 2)))
        best["n_inliers_refined"] = int(len(pts))
        best["up"] = n_ref
        best["centroid"] = centroid
        best["cam_height_m"] = float(np.dot(cam_centroid - centroid, n_ref))
        best["refined_normal"] = n_ref.tolist()
        best["refined_height_m"] = best["cam_height_m"]

    # --- OPTIONAL: refit over the union of near-parallel candidates ----------------------
    # NOT the default. The measured cause of the remaining spread is a fragmented floor -
    # different draws find different patches of the same surface - so the idea is to stop
    # choosing between patches and fit one plane to all of them at once: take every candidate
    # whose normal is within `union_deg` of the winner's, pool their inliers, and fit that.
    # Enabled with --floor-union-deg; 0 (default) leaves the chosen candidate alone.
    if union_deg and union_deg > 0 and best.get("_inliers") is not None:
        bn = np.asarray(best["up"])
        pool_pts, pool_n = [], 0
        for c in cands:
            if c.get("_inliers") is None:
                continue
            if angle_to_deg(np.asarray(c["up"]), bn) <= union_deg:
                pool_pts.append(c["_inliers"]); pool_n += 1
        if pool_n > 1:
            allp = np.concatenate(pool_pts, axis=0)
            centroid = allp.mean(0)
            _, _, vt = np.linalg.svd(allp - centroid, full_matrices=False)
            n_u = vt[-1] / np.linalg.norm(vt[-1])
            if np.dot(n_u, bn) < 0:
                n_u = -n_u
            d = (allp - centroid) @ n_u
            best["union_normal"] = n_u.tolist()
            best["union_n_candidates"] = pool_n
            best["union_n_points"] = int(len(allp))
            best["union_rms_m"] = float(np.sqrt(np.mean(d ** 2)))
            best["pre_union_up"] = bn.tolist()
            best["pre_union_cam_height_m"] = best["cam_height_m"]
            best["up"] = n_u
            best["centroid"] = centroid
            best["cam_height_m"] = float(np.dot(cam_centroid - centroid, n_u))
        else:
            best["union_n_candidates"] = pool_n

    for c in cands:
        c.pop("_inliers", None)

    if camera_up is not None:
        ang = angle_to_deg(np.asarray(best["up"]), camera_up)
        best["angle_to_camera_up_deg"] = ang
        best["support_pool_size"] = pool_n
        # Angle to the "up" each of the four frame rotations would imply. Written to
        # floor_plane.json on every run and quoted whenever the gate trips: a plane 87 deg
        # off under the recorded rotation but 3 deg off under 90 is not a bad floor fit, it
        # is a video whose rotation metadata was dropped at FRAMES.
        by_rot = {}
        if up_variants:
            for deg, v in up_variants.items():
                if v is not None:
                    by_rot[int(deg)] = round(angle_to_deg(np.asarray(best["up"]), v), 3)
        best["angle_by_frame_rotation_deg"] = by_rot

        h = best["cam_height_m"]
        lo, hi = cam_height_range
        if not (lo <= h <= hi):
            raise FloorImplausible(
                f"floor not found: chosen plane puts the cameras {h:.3f} m above it, "
                f"outside the plausible band {lo}-{hi} m; angle to camera up {ang:.1f} deg, "
                f"inlier_frac {best['inlier_frac']:.4f}, "
                f"{len(cands)} candidates with heights "
                + ", ".join(f"{c['cam_height_m']:.2f}" for c in cands) + " m"
                + _rotation_hint(by_rot, gate_deg))
        if ang > gate_deg:
            raise FloorImplausible(
                f"floor not found: best candidate angle {ang:.1f} deg to median camera up "
                f"(gate {gate_deg:.0f} deg); {len(cands)} candidates, {pool_n} in the "
                f">= {support_frac:g}x-support pool, angles "
                + ", ".join(f"{c['angle_to_camera_up_deg']:.1f}" for c in cands) + " deg"
                + _rotation_hint(by_rot, gate_deg))
    return best, cands


def small_components(mesh_path: Path, max_tri: int = 100):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(m.triangles) == 0:
        return {"components": 0, "small_components": 0, "small_component_triangles": 0}
    labels, counts, _ = m.cluster_connected_triangles()
    counts = np.asarray(counts)
    small = counts < max_tri
    return {"components": int(len(counts)), "small_components": int(small.sum()),
            "small_component_triangles": int(counts[small].sum()),
            "largest_component_triangles": int(counts.max())}


def run_scene(name, packed: Path, out: Path, args, depth_file: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((packed / "meta.json").read_text())
    views = meta["views"]
    stack = np.load(packed / depth_file, mmap_mode="r")
    h, w = meta["shape_hw"]
    keep = None
    if args.mask_file:
        mp = Path(args.mask_root or packed) / name / args.mask_file
        mmeta = json.loads((mp.parent / "mask_meta.json").read_text())
        keep = np.unpackbits(np.load(mp))[: len(views) * h * w].reshape(len(views), h, w).astype(bool)
        meta["conf_pct_dropped"] = mmeta["pct"]
        meta["conf_threshold"] = args.mask_file
        meta["conf_dropped_frac_of_valid"] = (mmeta["dropped_frac_rank10"]
                                              if "rank10" in args.mask_file
                                              else mmeta["dropped_frac_gt1"])
    torch.cuda.reset_peak_memory_stats()

    # --- colour ---------------------------------------------------------------------
    # Depth-only runs still call update_color_mesh()/get_color_mesh(), so mesh.ply has always
    # HAD an RGB property - every vertex just carries nvblox's unset colour voxel, 127/127/127.
    # That is why `source_has_vertex_colors: True` in the export was believed and shipped: it
    # tests for the presence of the property, not for variance. Measured: 1 unique colour on
    # every live run, against 98,225 in nvblox_v2/results/hero_color/mesh_color.ply.
    rgb_stack = None
    if args.rgb_file:
        rgb_path = packed / args.rgb_file
        if not rgb_path.exists():
            raise SystemExit(f"--rgb-file {args.rgb_file} not found at {rgb_path}; PACK_RGB "
                             f"must run before NVBLOX, or drop the flag for a grey mesh")
        rgb_stack = np.load(rgb_path, mmap_mode="r")
        if rgb_stack.shape != (len(views), h, w, 3):
            raise SystemExit(f"{rgb_path}: shape {rgb_stack.shape} does not match "
                             f"{(len(views), h, w, 3)} from meta.json - the RGB stack and the "
                             f"depth stack must be the same views in the same order")
        if rgb_stack.dtype != np.uint8:
            raise SystemExit(f"{rgb_path}: dtype {rgb_stack.dtype}, expected uint8")

    # nvblox's sphere tracer asserts image_width % raycast_subsampling_factor == 0
    # (sphere_tracer.cu:398). It is a glog CHECK - a process ABORT, not an exception - the
    # factor defaults to 4, and MapAnything's images are 294 wide, so 294 % 4 == 2 and the
    # COLOUR path kills the process outright. Depth-only runs never reach that code, which is
    # why this has never been hit before. Lowering the factor through ViewCalculatorParams
    # reads back correctly in Python but does not reach the C++ tracer. So every image is
    # padded by one pixel on each side to 296x520 (both divisible by 4) and the principal
    # point is shifted to match. Geometry is untouched: the border is invalid depth. This
    # costs 337 of 545,622 source triangles (-0.06%), which survives decimation intact.
    # Verbatim from nvblox_v2/nvblox_hero_color.py, which measured all of the above.
    PAD = 1 if rgb_stack is not None else 0
    hp, wp = h + 2 * PAD, w + 2 * PAD

    mapper, settings = build_mapper(args.voxel_size, args.truncation_vox,
                                    args.max_integration_distance, args.esdf_max_distance)
    t0 = time.time()
    n_color = 0
    color_dtype = None
    for i, v in enumerate(views):
        d0 = np.ascontiguousarray(stack[i].astype(np.float32) / 1000.0)
        d0[d0 > args.max_integration_distance] = 0.0
        if keep is not None:
            d0[~keep[i]] = 0.0
        if PAD:
            d = np.zeros((hp, wp), np.float32); d[PAD:-PAD, PAD:-PAD] = d0
        else:
            d = d0
        Kp = np.asarray(v["intrinsics"], np.float32).copy()
        Kp[0, 2] += PAD; Kp[1, 2] += PAD
        K = torch.as_tensor(np.ascontiguousarray(Kp))
        sensor = Sensor.from_camera_matrix(K, int(wp), int(hp))
        P = torch.as_tensor(np.ascontiguousarray(np.asarray(v["camera_pose"], np.float32)))
        mapper.add_depth_frame(torch.as_tensor(d, device=DEV), P, sensor)

        if rgb_stack is None:
            continue
        rgb = np.zeros((hp, wp, 3), np.uint8)
        rgb[PAD:-PAD, PAD:-PAD] = rgb_stack[i]
        if color_dtype is None:
            # Settle the dtype the wheel accepts ONCE, on the first frame, and refuse the
            # whole run if neither works. The alternative - catching per frame and carrying
            # on - is how you get a grey mesh that reports success.
            last = None
            for cand in (np.uint8, np.float32):
                try:
                    mapper.add_color_frame(
                        torch.as_tensor(np.ascontiguousarray(rgb.astype(cand)), device=DEV),
                        P, sensor)
                    color_dtype = cand
                    break
                except Exception as e:                                       # noqa: BLE001
                    last = e
            if color_dtype is None:
                raise SystemExit(f"add_color_frame rejected both uint8 and float32: {last}")
        else:
            mapper.add_color_frame(
                torch.as_tensor(np.ascontiguousarray(rgb.astype(color_dtype)), device=DEV),
                P, sensor)
        n_color += 1
    integrate_s = time.time() - t0
    if rgb_stack is not None and n_color != len(views):
        raise SystemExit(f"only {n_color} of {len(views)} colour frames integrated; a "
                         f"partially coloured fusion is not a result worth keeping")
    t1 = time.time(); mapper.update_esdf(); esdf_s = time.time() - t1
    t2 = time.time(); mapper.update_color_mesh(); mesh = mapper.get_color_mesh()
    mesh_path = out / "mesh.ply"; mesh.save(str(mesh_path)); mesh_s = time.time() - t2
    mv = mesh.vertices().detach().cpu().numpy()

    # --- colour, as a NUMBER rather than a boolean --------------------------------------
    # `unique` is the whole point. A flat-grey mesh has vertex_colors() of the right length,
    # all-nonzero and with a perfectly plausible mean - every presence-shaped check passes on
    # it, which is exactly how the grey export shipped. Only the unique count separates
    # "integrated" from "never touched", so that is what goes in stats.json for the
    # acceptance gate to read. Computed here, on the GPU's own mesh, costing nothing.
    try:
        vc = mesh.vertex_colors().detach().cpu().numpy()
        vc2 = vc.reshape(len(vc), -1)
        colour = {"unique_vertex_colors": int(np.unique(vc2, axis=0).shape[0]),
                  "n_vertices_with_color": int(vc.shape[0]),
                  "mean_rgb": [round(float(x), 1) for x in vc2.mean(0)],
                  "frac_nonzero": round(float((vc2.sum(1) > 0).mean()), 4)}
    except Exception as e:                                                   # noqa: BLE001
        colour = {"vertex_colors_failed": str(e)}
    colour |= {"rgb_file": args.rgb_file,
               "n_color_frames_integrated": n_color,
               "color_dtype": None if color_dtype is None else np.dtype(color_dtype).name,
               "image_padded_to": [hp, wp] if PAD else None}
    if args.rgb_file and colour.get("unique_vertex_colors", 0) <= 1:
        raise SystemExit(
            f"colour was requested and {n_color} frames were integrated, but the mesh has "
            f"{colour.get('unique_vertex_colors')} unique vertex colour(s) - that is the "
            f"grey-mesh failure this flag exists to prevent, so this run fails instead of "
            f"shipping a mesh that merely LOOKS coloured to a presence check")

    # --- surface probe on back-projected depth points ---
    probe_pts, cloud = fused_points_from_depth(stack, views, args.probe_points,
                                               args.max_integration_distance)
    sdf = query_esdf(mapper, probe_pts)
    unk = sdf >= UNKNOWN - 1e-3
    ok = sdf[~unk]
    probe = {"n": int(sdf.size), "unobserved_frac": float(unk.mean()),
             "median_abs_sdf_m": float(np.median(np.abs(ok))) if ok.size else None,
             "p95_abs_sdf_m": float(np.percentile(np.abs(ok), 95)) if ok.size else None}

    # --- floor + 2D slice ---
    cams = np.asarray([v["camera_pose"] for v in views], float)[:, :3, 3]
    cam_up = median_camera_up(views)
    up_variants = camera_up_variants(views)
    try:
        floor, cands = fit_floor(mv if len(mv) > 20000 else cloud, cams.mean(0),
                                 seed=args.floor_seed, camera_up=cam_up,
                                 support_frac=args.floor_support_frac,
                                 gate_deg=args.floor_gate_deg,
                                 ransac_iterations=args.ransac_iterations,
                                 up_variants=up_variants,
                                 union_deg=args.floor_union_deg)
    except FloorFitError as e:
        stats = {"scene": name, "depth_file": depth_file, "n_views": len(views),
                 "settings": settings, "integrate_wall_s": round(integrate_s, 2),
                 "esdf_wall_s": round(esdf_s, 2), "mesh_wall_s": round(mesh_s, 2),
                 "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                 "tsdf_blocks": int(mapper.tsdf_layer_view().num_allocated_blocks()),
                 "mesh": {"vertices": int(mv.shape[0]),
                          "triangles": int(mesh.triangles().shape[0])},
                 "colour": colour,
                 "mesh_components": small_components(mesh_path, args.small_component_tri),
                 "surface_probe": probe, "slice": None, "esdf_3d": {},
                 "floor_fit_failed": str(e),
                 "conf_filter": {"pct": meta.get("conf_pct_dropped"),
                                 "threshold": meta.get("conf_threshold"),
                                 "dropped_frac_of_valid": meta.get("conf_dropped_frac_of_valid")}}
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        del mapper; torch.cuda.empty_cache()
        return stats
    up = floor["up"]
    a0 = np.array([1.0, 0, 0]) - up * up[0]
    if np.linalg.norm(a0) < 1e-6:
        a0 = np.array([0, 1.0, 0]) - up * up[1]
    a0 /= np.linalg.norm(a0); a1 = np.cross(up, a0)
    org = floor["centroid"] + args.slice_height * up
    rel = cloud - org
    u, vv = rel @ a0, rel @ a1
    lo = np.array([np.percentile(u, 0.5), np.percentile(vv, 0.5)])
    hi = np.array([np.percentile(u, 99.5), np.percentile(vv, 99.5)])
    nu = int(np.ceil((hi[0] - lo[0]) / args.slice_res)) + 1
    nv = int(np.ceil((hi[1] - lo[1]) / args.slice_res)) + 1
    gu, gv = np.meshgrid(lo[0] + np.arange(nu) * args.slice_res,
                         lo[1] + np.arange(nv) * args.slice_res, indexing="ij")
    pts = org + gu.ravel()[:, None] * a0 + gv.ravel()[:, None] * a1
    s = query_esdf(mapper, pts)
    unobs = (s >= UNKNOWN - 1e-3).reshape(nu, nv)
    sl = np.where(unobs, np.nan, s.reshape(nu, nv)).astype(np.float32)
    np.save(out / "esdf_slice_0.3m.npy", sl)
    np.save(out / "unobserved_slice.npy", unobs)
    (out / "floor_plane.json").write_text(json.dumps({
        "normal_up": up.tolist(), "point_on_plane": floor["centroid"].tolist(),
        "camera_height_above_plane_m": floor["cam_height_m"],
        "inlier_frac": floor["inlier_frac"],
        # Provenance for reproducibility: BOTH of these are needed to reproduce this plane.
        "floor_seed": args.floor_seed,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "angle_to_camera_up_deg": floor.get("angle_to_camera_up_deg"),
        "angle_by_frame_rotation_deg": floor.get("angle_by_frame_rotation_deg"),
        "union_normal": floor.get("union_normal"),
        "union_n_candidates": floor.get("union_n_candidates"),
        "union_n_points": floor.get("union_n_points"),
        "union_rms_m": floor.get("union_rms_m"),
        "refined_normal": floor.get("refined_normal"),
        "refined_height_m": floor.get("refined_height_m"),
        "rms_inlier_dist_m": floor.get("rms_inlier_dist_m"),
        "n_inliers_refined": floor.get("n_inliers_refined"),
        "raw_normal_up": floor.get("raw_up"),
        "raw_camera_height_m": floor.get("raw_cam_height_m"),
        "median_camera_up": cam_up.tolist(),
        "n_candidates": len(cands),
        "support_pool_size": floor.get("support_pool_size"),
        "floor_support_frac": args.floor_support_frac,
        "floor_gate_deg": args.floor_gate_deg,
        "slice_height_above_floor_m": args.slice_height,
        "slice_origin_world": org.tolist(), "axis_u": a0.tolist(), "axis_v": a1.tolist(),
        "slice_resolution_m": args.slice_res, "slice_shape": [nu, nv],
        "u_range": [float(lo[0]), float(hi[0])], "v_range": [float(lo[1]), float(hi[1])],
        "candidates": [{"up": [round(float(x), 3) for x in c["up"]],
                        "cam_height_m": round(c["cam_height_m"], 3),
                        "inlier_frac": round(c["inlier_frac"], 4)} for c in cands],
    }, indent=2))

    # --- dense 3D ESDF over the map AABB ---
    esdf3d = {}
    if args.esdf_3d:
        layer = mapper.tsdf_layer_view()
        bmin, bmax = layer.get_block_limits()
        vs = layer.voxel_size()
        amin = (bmin.cpu().numpy() * layer.block_dim_in_voxels + 0.5) * vs
        amax = ((bmax.cpu().numpy() + 1) * layer.block_dim_in_voxels + 0.5) * vs
        res = args.esdf_3d_res
        dims = np.maximum(((amax - amin) / res).astype(int) + 1, 1)
        if int(np.prod(dims)) <= args.esdf_3d_max_voxels:
            gx, gy, gz = [amin[i] + np.arange(dims[i]) * res for i in range(3)]
            G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
            v3 = query_esdf(mapper, G)
            v3 = np.where(v3 >= UNKNOWN - 1e-3, np.nan, v3).astype(np.float16)
            np.save(out / "esdf_3d.npy", v3.reshape(dims))
            esdf3d = {"path": str(out / "esdf_3d.npy"), "origin_m": amin.tolist(),
                      "resolution_m": res, "shape": dims.tolist(),
                      "unobserved_frac": float(np.isnan(v3).mean())}
        else:
            esdf3d = {"skipped": f"{int(np.prod(dims))} voxels > {args.esdf_3d_max_voxels}"}
    try:
        mapper.save_map(str(out / "map.nvblx"), 0)
        esdf3d["map_nvblx_bytes"] = (out / "map.nvblx").stat().st_size
    except Exception as e:
        esdf3d["map_nvblx"] = f"failed: {type(e).__name__}: {e}"

    stats = {"scene": name, "depth_file": depth_file, "n_views": len(views),
             "settings": settings,
             "integrate_wall_s": round(integrate_s, 2), "esdf_wall_s": round(esdf_s, 2),
             "mesh_wall_s": round(mesh_s, 2),
             "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
             "tsdf_blocks": int(mapper.tsdf_layer_view().num_allocated_blocks()),
             "mesh": {"vertices": int(mv.shape[0]),
                      "triangles": int(mesh.triangles().shape[0]),
                      "bbox_min": [round(float(x), 3) for x in mv.min(0)],
                      "bbox_max": [round(float(x), 3) for x in mv.max(0)]},
             "colour": colour,
             "mesh_components": small_components(mesh_path, args.small_component_tri),
             "surface_probe": probe,
             "slice": {"shape": [nu, nv], "resolution_m": args.slice_res,
                       "unobserved_frac": float(unobs.mean()),
                       "observed_sdf_median_m": float(np.nanmedian(sl)),
                       "observed_sdf_min_m": float(np.nanmin(sl)),
                       "observed_sdf_max_m": float(np.nanmax(sl))},
             "esdf_3d": esdf3d,
             "conf_filter": {"pct": meta.get("conf_pct_dropped"),
                             "threshold": meta.get("conf_threshold"),
                             "dropped_frac_of_valid": meta.get("conf_dropped_frac_of_valid")}}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    del mapper
    torch.cuda.empty_cache()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--depth-file", default="depth_u16.npy")
    ap.add_argument("--rgb-file", default=None,
                    help="a uint8 (N, h, w, 3) stack beside the depth stack, in meta.json's "
                         "view order, produced by pipeline/steps/pack_rgb.py. Given it, "
                         "add_color_frame runs and mesh.ply carries real vertex colours; "
                         "omitted, the mesh is nvblox's unset 127/127/127 for every vertex "
                         "- which still LOOKS coloured to any presence check, so this run "
                         "fails loudly rather than shipping grey when the flag is set.")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--voxel-size", type=float, default=0.03)
    ap.add_argument("--truncation-vox", type=float, default=4.0)
    ap.add_argument("--max-integration-distance", type=float, default=5.0)
    ap.add_argument("--esdf-max-distance", type=float, default=2.0)
    ap.add_argument("--slice-height", type=float, default=0.3)
    ap.add_argument("--slice-res", type=float, default=0.05)
    ap.add_argument("--probe-points", type=int, default=1000)
    ap.add_argument("--small-component-tri", type=int, default=100)
    ap.add_argument("--mask-file", default=None,
                    help="packed-bit keep-mask from pack_conf_masks.py, e.g. mask_rank10.npy")
    ap.add_argument("--mask-root", default=None, help="root holding <scene>/<mask-file>")
    ap.add_argument("--esdf-3d", action="store_true")
    ap.add_argument("--esdf-3d-res", type=float, default=0.05)
    ap.add_argument("--esdf-3d-max-voxels", type=int, default=40_000_000)
    ap.add_argument("--floor-support-frac", type=float, default=0.2,
                    help="candidates supported at least this fraction of the best-supported "
                         "one form the pool the reference plane is picked from. 0.2 is the "
                         "value gpu/floor_ceiling.py settled on; 0.8 was measured and is "
                         "too tight (see fit_floor's docstring).")
    ap.add_argument("--floor-gate-deg", type=float, default=30.0,
                    help="max angle between the chosen floor normal and the median camera "
                         "up before the run fails outright")
    ap.add_argument("--floor-union-deg", type=float, default=0.0,
                    help="if > 0, refit the floor over the pooled inliers of every candidate "
                         "within this angle of the winner. NOT the default - see the sweep "
                         "in PIPELINE.md before enabling it.")
    ap.add_argument("--ransac-iterations", type=int, default=9000,
                    help="segment_plane iterations. Was 3000; 3x after measuring that the "
                         "plane a draw FINDS, not the fit of it, drives the spread. x3 "
                         "halved hero's seed-to-seed angle spread (6.290 -> 2.674 deg) for "
                         "~90 s of extra CPU per scene; x10 was measured as ~9 min/seed and "
                         "extrapolates to ~1.5-2 deg, so it does not reach the 1 deg bar "
                         "either and was not taken.")
    ap.add_argument("--floor-seed", type=int, default=42,
                    help="seed for Open3D's global RNG, used by the floor RANSAC. Needs "
                         "OMP_NUM_THREADS=1 (set at the top of this file) to actually "
                         "reproduce - the seed alone does not.")
    args = ap.parse_args()

    root, outroot = Path(args.packed_root), Path(args.out_root)
    all_stats = []
    for s in args.scenes:
        print(f"=== {s} ===", flush=True)
        st = run_scene(s, root / s, outroot / (s + args.suffix), args, args.depth_file)
        all_stats.append(st)
        print(json.dumps({k: st[k] for k in ["n_views", "integrate_wall_s", "esdf_wall_s",
                                             "peak_vram_mib", "mesh", "surface_probe",
                                             "slice"]}, indent=1), flush=True)
    (outroot / f"all_stats{args.suffix or ''}.json").write_text(json.dumps(all_stats, indent=2))
    print("WROTE", outroot / f"all_stats{args.suffix or ''}.json")


if __name__ == "__main__":
    main()
