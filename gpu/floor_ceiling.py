"""Direction-agnostic floor/ceiling disambiguation + robust plane fitting, for one
MapAnything inference run.

Extends the original `floor_ceiling_disambiguate.py` (written and validated earlier in
this project against two real failure cases) from a diagnostic-only tool - it returned
`(floor_y, ceiling_y, floor_tilt, diagnostics)` and wrote nothing - into the actual
method `stage_align.py` uses to compute and persist a real alignment transform.

Motivation #1 (real incident this project hit twice): a plane found via RANSAC on a
"low-Y" band can be a perfectly clean, low-tilt fit that is nonetheless the CEILING,
not the floor - MapAnything does not guarantee any fixed axis convention between
separate `.infer()` calls, not even which raw-Y direction is "up". This failure is
worse than an outright axis inversion because every downstream number still looks
internally consistent (geometry, scale, tilt) - only the absolute reference is wrong.
Fixed by disambiguating floor-vs-ceiling by aggregate support on each side, then
orienting "up" using the actual direction from the chosen floor toward the cameras -
never an assumption about raw-Y sign.

Motivation #2 (hotel-room scene, 2026-09): a SECOND, narrower failure survives fix #1.
A single band-wide RANSAC fit picks whichever plane has the most inliers in the band -
on a room where the real floor is heavily occluded (furniture covers most of it) but a
bed or table top sits within the same low-Y band, that furniture plane can out-support
the sparse, real floor and win outright, even though it is well above true floor
height. The exported scene then starts at bed/table height with everything below it
gone. Fixed by extracting ALL near-horizontal plane candidates on the floor side (not
just the single best-supported one) and choosing among them by height first, subject
to a minimum-support floor and two physical sanity checks - see `select_floor_candidate`.

Method: find the two dominant large horizontal planes (floor/ceiling candidates),
verify camera positions are physically between them (you cannot be your own camera
outside your own room), disambiguate which SIDE is the floor by aggregate support
(floor gets 600-900x more support in a walkthrough where the operator tilts down for
floor coverage - validated on two real cases), pick the actual floor PLANE on that
side by height + support-floor + camera-height-prior + extent (see
`select_floor_candidate`), then orient "up" using the actual direction from the floor
toward the cameras - never an assumption about raw-Y sign.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Selection defaults for `select_floor_candidate` - overridable via job params in
# stage_align.py, same pattern as DEFAULT_FLOOR_CLIP_MARGIN_M there.
DEFAULT_FLOOR_MIN_SUPPORT_FRAC = 0.2
DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M = (0.8, 2.0)
# Originally spec'd at 0.4 (40%); recalibrated down after measuring real hotel-batch
# scenes (job 4ae57385 and 3 others, recovered via inverting their saved transforms -
# see docs/DECISIONS.md's 2026-09-03 real-scene-verification entry): a Y-band-
# restricted candidate's own hull area, even for a floor later confirmed correct
# (matched the old run's independently-plausible room height), covered only 7.2%-14.2%
# of the FULL room bbox area on real data - nowhere near 40%. 0.4 rejected every real
# floor candidate tested and forced a LOW CONFIDENCE fallback on every single one of
# them, including scenes with no evidence of a wrong floor. 0.05 sits comfortably below
# that observed 7.2% floor while still rejecting a pathologically small footprint (a
# synthetic nightstand-sized surface at ~1.3% of room area - see
# tests/test_floor_ceiling.py::test_extent_check_rejects_small_footprint_candidate).
DEFAULT_FLOOR_EXTENT_FRAC = 0.05

# Reproducibility: below-floor clip-fraction numbers reported for a real scene were
# found to vary 1.8%-2.7% across repeated runs of otherwise-identical code, on the same
# fixture - the pass/fail verdict against the thresholds above was stable across every
# run, but "which run's number is the real one" was not answerable. Root cause:
# fit_plane's RANSAC (segment_plane, below) draws from Open3D's own global C++ RNG,
# entirely separate from numpy's, and has no per-call seed parameter - seeding numpy
# alone does nothing for it.
DEFAULT_RANDOM_SEED = 0


def seed_ransac(seed: int = DEFAULT_RANDOM_SEED) -> bool:
    """Seed Open3D's global RANSAC RNG, if this build exposes the hook (added in some
    0.19.x builds, not guaranteed present on every host - see fit_plane's own deferred-
    import comment for why open3d isn't imported at module scope). Callers should call
    this ONCE, early, before any fit_plane call - never from inside fit_plane itself,
    which would make a library function silently mutate global process state on every
    invocation (and, worse, reseeding mid-run would restart the RNG stream a prior fit
    already consumed, making THAT fit's result depend on how many fits ran before it).

    Returns whether the seed actually took, so a caller can log the difference between
    "seeded" and "this Open3D build can't be seeded" rather than silently assuming success.

    Known limitation, not fixed here: Open3D's RANSAC is OpenMP-parallel, so a fixed seed
    gives run-to-run reproducibility on one host/thread-count, not guaranteed bit-exact
    reproducibility across machines with different core counts. Pinning OMP_NUM_THREADS
    would close that gap at a real throughput cost, for a guarantee this project doesn't
    need (single-host reproducibility, not cross-machine bit-exactness) - deliberately not
    done."""
    import open3d as o3d  # deferred - see module docstring's note near the imports above

    if hasattr(o3d.utility, "random") and hasattr(o3d.utility.random, "seed"):
        o3d.utility.random.seed(seed)
        return True
    return False


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    inlier_idx: np.ndarray
    inlier_points: np.ndarray
    tilt_deg: float
    support: int
    n_views_supporting: int
    plane_y: float
    hull_area: float


@dataclass
class FloorCeilingResult:
    floor: PlaneFit
    ceiling: PlaneFit
    up_normal: np.ndarray
    frac_cameras_between: float
    chosen: str
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def load_run(per_view_dir: str | Path, *, include_poses: bool = False):
    """Pool valid points+colors+source-view-index+confidence, plus full 3D camera
    positions, from one MapAnything inference run's `per_view/*.npz` files.

    Returns (points (N,3), colors (N,3), view_idx (N,) int32, camera_positions (V,3),
    conf (N,) float32 or None) by default - a 5-tuple.

    `conf` is MapAnything's per-pixel confidence (see stage_infer.py's feature-detection
    of the model's actual confidence key - not guaranteed present in every run). A view
    whose npz lacks "conf" contributes NaN for its points rather than being dropped, so
    a run with partial confidence coverage can still filter the points that DO have it.
    `conf` is None outright only when NO view in this run has confidence data at all -
    callers must treat that as "fall back to binary-mask-only" (see stage_align.py's
    conf-percentile filter and stage_objects.py's conf-cutoff pass-through).

    With `include_poses=True` (used by stage_align.py to persist full pose/intrinsics
    into cameras_aligned.json - see docs/SPRINT-poisson-meshing.md's Workstream C),
    appends camera_poses (V,4,4) float64 - each view's full camera-to-world extrinsic
    matrix, of which `camera_positions` is just column 3's top 3 rows - and
    camera_intrinsics (V,3,3) float64, the per-view K matrix, making it a 7-tuple. Both
    are in the SAME raw per-run frame as `camera_positions`/`points` (i.e. pre-
    alignment); stage_align.py applies its own rotation+floor-shift to these afterward,
    same as it already does for `camera_positions`.
    """
    files = sorted(glob.glob(str(Path(per_view_dir) / "view_*.npz")))
    if not files:
        raise FileNotFoundError(f"no view_*.npz files found in {per_view_dir}")

    all_pts, all_cols, all_view_idx, all_conf, cam_positions = [], [], [], [], []
    cam_poses, cam_intrinsics = [], []
    any_conf = False
    for i, f in enumerate(files):
        d = np.load(f)
        pts3d, mask, img_no_norm = d["pts3d"], d["mask"], d["img_no_norm"]
        pts = pts3d[mask]
        all_pts.append(pts)
        all_cols.append(img_no_norm[mask])
        all_view_idx.append(np.full(len(pts), i, dtype=np.int32))
        if "conf" in d.files:
            all_conf.append(d["conf"][mask].astype(np.float32))
            any_conf = True
        else:
            all_conf.append(np.full(len(pts), np.nan, dtype=np.float32))
        cam_positions.append(d["camera_pose"][:3, 3])
        if include_poses:
            cam_poses.append(d["camera_pose"])
            cam_intrinsics.append(d["intrinsics"])

    result = (
        np.concatenate(all_pts, axis=0),
        np.concatenate(all_cols, axis=0),
        np.concatenate(all_view_idx, axis=0),
        np.asarray(cam_positions, dtype=np.float64),
        np.concatenate(all_conf, axis=0) if any_conf else None,
    )
    if include_poses:
        result = result + (
            np.asarray(cam_poses, dtype=np.float64),
            np.asarray(cam_intrinsics, dtype=np.float64),
        )
    return result


def _convex_hull_xy_area(points_2d: np.ndarray) -> float:
    """Convex hull area of a 2D point set. 0.0 for degenerate (collinear/too-few-point)
    input rather than raising - a real but tiny/sliver plane candidate should just lose
    the extent check, not crash the whole alignment stage."""
    if len(points_2d) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError  # deferred: keep this module

    try:
        return float(ConvexHull(points_2d).volume)  # 2D ConvexHull.volume IS the area
    except QhullError:
        return 0.0


def confidence_percentile_cutoff(conf: np.ndarray, percentile: float) -> tuple[float, np.ndarray]:
    """Given a pooled per-point confidence array (may contain NaN for points whose
    source view had no confidence data at all - see `load_run`), compute the
    `percentile`-th percentile cutoff over only the non-NaN values, and a boolean
    keep-mask the same length as `conf`. A point with NO confidence value passes
    through (True) rather than being dropped - "unknown" is not the same claim as "low
    confidence", and shouldn't be penalized as one just because a view lacked the data.

    Percentile-based (relative to THIS run's own distribution), not a fixed absolute
    conf value - MapAnything's confidence scale is not calibrated to be comparable
    across separate `.infer()` runs any more than its coordinate frame is (see this
    module's own docstring).

    Raises ValueError if every value is NaN - callers should already have checked
    `conf is not None and not np.all(np.isnan(conf))` before calling this (both current
    callers, stage_align.py and stage_objects.py's cutoff-reuse, do).
    """
    valid = ~np.isnan(conf)
    if not valid.any():
        raise ValueError("confidence_percentile_cutoff: conf is entirely NaN")
    cutoff = float(np.percentile(conf[valid], percentile))
    keep_mask = np.where(valid, conf >= cutoff, True)
    return cutoff, keep_mask


def fit_plane(points: np.ndarray, distance_threshold=0.025, ransac_n=3, num_iterations=2000):
    """RANSAC-fit a single dominant plane. Imports open3d locally, not at module level,
    so the rest of this module (in particular `select_floor_candidate`, the pure
    height/support/camera-height/extent selection logic) stays importable and unit-
    testable from a venv that never has open3d installed - see gpu/stage_objects.py's
    build_sam3_image_model for the same pattern and why."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    a, b, c, dd = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(normal[1]), -1, 1))))
    return normal, dd, np.asarray(inliers), tilt_deg


def extract_horizontal_plane_candidates(
    points: np.ndarray,
    view_idx: np.ndarray,
    *,
    tilt_max_deg=45.0,
    min_support=150,
    max_planes=8,
    distance_threshold=0.025,
) -> list[PlaneFit]:
    """Iteratively RANSAC-fit dominant planes out of `points` (removing each fit's
    inliers before the next iteration, same idea as multi-plane extraction anywhere
    else), keeping every near-horizontal (tilt <= tilt_max_deg) one with enough
    support as a genuine candidate.

    This is the fix for the hotel-room failure: the old code fit exactly ONE plane per
    Y-band and trusted it as *the* floor. A band can contain more than one real
    horizontal surface (true floor + a bed/table top), and RANSAC on the whole band
    just returns whichever has more inliers - which is not necessarily the floor. By
    extracting every candidate instead of only the winner, `select_floor_candidate` gets
    a real choice to make instead of inheriting whatever RANSAC happened to prefer.

    `tilt_max_deg` defaults to a generous 45deg, not a tight one: measured against a
    real hotel-batch capture (job 4ae57385, recovered via inverting its saved
    transform), that scene's genuine floor/ceiling planes sit at 20-30deg tilt in
    MapAnything's raw frame, while its actual walls sit at 70-83deg - MapAnything's raw
    Y axis is evidently only LOOSELY correlated with true vertical, not tightly, for at
    least some real captures. A tight cutoff (originally 15deg) filtered out every real
    candidate for that scene and made this function return empty - worse than the
    original band-based code, which never applied an absolute tilt cutoff at all (it
    only ever compared candidates against each other, "most horizontal wins"). 45deg
    still leaves a wide margin below real wall tilts measured on that same scene.

    Non-horizontal dominant planes (walls) are still removed from the working set so
    the sweep doesn't get stuck re-fitting the same wall - they're just not returned.

    Callers should restrict `points`/`view_idx` to a Y-band near the room's floor or
    ceiling extreme before calling this (see `find_floor_and_ceiling`'s `_y_band`) -
    RANSAC over an UNRESTRICTED whole-room cloud was measured to take minutes on a
    real multi-million-point capture; a band cuts that to a small fraction of the room.
    """
    remaining_mask = np.ones(len(points), dtype=bool)
    candidates: list[PlaneFit] = []
    for _ in range(max_planes):
        remaining_idx = np.nonzero(remaining_mask)[0]
        if len(remaining_idx) < min_support:
            break
        try:
            normal, d, inliers_local, tilt = fit_plane(
                points[remaining_idx], distance_threshold=distance_threshold
            )
        except RuntimeError:
            break
        if len(inliers_local) < min_support:
            break
        global_inliers = remaining_idx[inliers_local]
        if tilt <= tilt_max_deg:
            inlier_pts = points[global_inliers]
            inlier_views = view_idx[global_inliers]
            candidates.append(
                PlaneFit(
                    normal=normal,
                    d=d,
                    inlier_idx=global_inliers,
                    inlier_points=inlier_pts,
                    tilt_deg=tilt,
                    support=len(global_inliers),
                    n_views_supporting=len(set(inlier_views.tolist())),
                    plane_y=float(np.median(inlier_pts[:, 1])),
                    hull_area=_convex_hull_xy_area(inlier_pts[:, [0, 2]]),
                )
            )
        remaining_mask[global_inliers] = False
    return candidates


def _y_band(
    points: np.ndarray,
    view_idx: np.ndarray,
    *,
    low: bool,
    fracs=(0.10, 0.15, 0.22, 0.30, 0.40),
    min_band_pts=2000,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Restrict to a Y-band near y_min (if `low`) or y_max (otherwise), sweeping
    increasingly generous fractions of the room's own Y extent until at least
    `min_band_pts` points are found - same band-sweep idea the original band-based
    code used, kept here so `extract_horizontal_plane_candidates` runs RANSAC over a
    bounded subset of the room (hundreds of thousands of points, not millions) instead
    of the whole pooled cloud. Falls back to the most generous frac's band even if it's
    still under `min_band_pts` - a small band is still better than none for a sparse
    reconstruction; `extract_horizontal_plane_candidates`'s own min_support guards
    against fitting garbage out of too few points.
    """
    y = points[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    extent = y_max - y_min
    if extent <= 1e-6:
        return None, None
    band_mask = None
    for frac in fracs:
        band_mask = (y <= y_min + frac * extent) if low else (y >= y_max - frac * extent)
        if int(band_mask.sum()) >= min_band_pts:
            break
    return points[band_mask], view_idx[band_mask]


def select_floor_candidate(
    candidates: list[PlaneFit],
    *,
    extreme: str,
    cam_positions: np.ndarray,
    room_bbox_xy_area: float | None,
    min_support_frac: float = DEFAULT_FLOOR_MIN_SUPPORT_FRAC,
    camera_height_range: tuple[float, float] = DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M,
    extent_frac_threshold: float = DEFAULT_FLOOR_EXTENT_FRAC,
) -> tuple[PlaneFit, list[str]]:
    """Pick the real floor plane out of same-side candidates. Pure numpy/logic, no
    open3d dependency - this is the part of the hotel-room fix that is actually worth
    unit-testing in isolation from RANSAC.

    `extreme` is "min" if this side's floor candidate should be the LOWEST-y one
    (this side sits below the other, in raw Y), "max" if it should be the HIGHEST-y one
    (this side sits above the other). The caller (`find_floor_and_ceiling`) already
    knows which, from splitting all candidates by the room's own Y midpoint.

    Rule (the actual fix): among candidates with support >= min_support_frac * the
    best-supported candidate's support (not the single most-supported one - that was
    the bug, since a bed/table top can out-support a heavily-occluded real floor), pick
    the one closest to the room's true floor extreme, i.e. the most extreme in `extreme`
    direction - but only if it also passes two physical sanity checks: cameras must sit
    a plausible walking height above it, and its footprint must cover a real fraction
    of the room (a shelf or nightstand top can be flat, well-supported, and still not be
    a floor). A candidate failing either check is skipped in favor of the next-most-
    extreme eligible one; if none pass, this falls back to the old max-support behavior
    with an explicit low-confidence warning, rather than raising - a scene this
    ambiguous should still get *a* floor and clear diagnostics, not a hard failure.
    """
    if not candidates:
        raise ValueError("select_floor_candidate: no candidates to choose from")
    if extreme not in ("min", "max"):
        raise ValueError(f"extreme must be 'min' or 'max', got {extreme!r}")

    best_support = max(c.support for c in candidates)
    eligible = [c for c in candidates if c.support >= min_support_frac * best_support]
    eligible.sort(key=lambda c: c.plane_y, reverse=(extreme == "max"))

    cam_centroid = cam_positions.mean(axis=0)
    warnings: list[str] = []

    for c in eligible:
        plane_point = c.inlier_points.mean(axis=0)
        # Provisional "up" for THIS candidate only: which side the cameras are on
        # relative to it. Independent of the final up_normal computed later in
        # find_floor_and_ceiling (that one uses the WINNING floor) - this is just the
        # same physical test applied per-candidate, to sanity-check candidates before
        # one is chosen.
        up_guess = c.normal.copy()
        if np.dot(up_guess, cam_centroid - plane_point) < 0:
            up_guess = -up_guess
        median_cam_height = float(np.median((cam_positions - plane_point) @ up_guess))
        camera_ok = camera_height_range[0] <= median_cam_height <= camera_height_range[1]

        extent_ok = (
            room_bbox_xy_area is None
            or c.hull_area >= extent_frac_threshold * room_bbox_xy_area
        )

        if camera_ok and extent_ok:
            return c, warnings

        reasons = []
        if not camera_ok:
            reasons.append(
                f"median camera height {median_cam_height:.2f}m outside "
                f"[{camera_height_range[0]:.1f}, {camera_height_range[1]:.1f}]m"
            )
        if not extent_ok:
            reasons.append(
                f"hull_area {c.hull_area:.2f}m^2 < {extent_frac_threshold:.0%} of "
                f"room bbox area {room_bbox_xy_area:.2f}m^2"
            )
        warnings.append(
            f"floor candidate at y={c.plane_y:.3f} (support={c.support}) rejected: "
            + "; ".join(reasons)
        )

    fallback = max(eligible, key=lambda c: c.support)
    warnings.append(
        "no candidate passed camera-height/extent checks - falling back to max-support "
        f"candidate at y={fallback.plane_y:.3f} (support={fallback.support}); floor "
        "selection is LOW CONFIDENCE for this scene"
    )
    return fallback, warnings


def find_floor_and_ceiling(
    pts: np.ndarray,
    view_idx: np.ndarray,
    cam_positions: np.ndarray,
    *,
    min_view_support=3,
    room_bbox_xy_area: float | None = None,
    floor_min_support_frac: float = DEFAULT_FLOOR_MIN_SUPPORT_FRAC,
    floor_camera_height_range: tuple[float, float] = DEFAULT_FLOOR_CAMERA_HEIGHT_RANGE_M,
    floor_extent_frac: float = DEFAULT_FLOOR_EXTENT_FRAC,
) -> FloorCeilingResult:
    """Direction-agnostic floor/ceiling identification for one inference run's own
    point cloud + camera positions."""
    warnings: list[str] = []

    low_pts, low_view_idx = _y_band(pts, view_idx, low=True)
    high_pts, high_view_idx = _y_band(pts, view_idx, low=False)
    if low_pts is None or high_pts is None:
        raise RuntimeError(
            "room point cloud has ~zero Y extent - genuine data limitation, not "
            "something to silently work around"
        )

    group_lo = extract_horizontal_plane_candidates(low_pts, low_view_idx)
    group_hi = extract_horizontal_plane_candidates(high_pts, high_view_idx)
    if not group_lo or not group_hi:
        missing = "low side" if not group_lo else "high side"
        raise RuntimeError(
            f"could not fit a horizontal-plane candidate on the {missing} of the room - "
            "genuine data limitation (floor/ceiling too sparse or absent in this "
            "reconstruction), not something to silently work around"
        )
    candidates = group_lo + group_hi

    # Disambiguate which SIDE is the floor by aggregate (best) support, not by raw Y
    # value: the floor side gets far more inlier points in a walkthrough where the
    # operator deliberately tilts the camera down for floor coverage - validated at a
    # 600-900x support ratio on two real failure cases in this project. This is
    # unaffected by the hotel-room fix below: it only decides which SIDE is the floor,
    # not which specific plane on that side is the real floor surface.
    lo_best_support = max(c.support for c in group_lo)
    hi_best_support = max(c.support for c in group_hi)
    floor_is_lo = lo_best_support >= hi_best_support
    floor_group, ceiling_group = (group_lo, group_hi) if floor_is_lo else (group_hi, group_lo)
    chosen = "low" if floor_is_lo else "high"

    floor, floor_warnings = select_floor_candidate(
        floor_group,
        extreme=("min" if floor_is_lo else "max"),
        cam_positions=cam_positions,
        room_bbox_xy_area=room_bbox_xy_area,
        min_support_frac=floor_min_support_frac,
        camera_height_range=floor_camera_height_range,
        extent_frac_threshold=floor_extent_frac,
    )
    warnings.extend(floor_warnings)
    ceiling = max(ceiling_group, key=lambda c: c.support)

    lo_y, hi_y = sorted([floor.plane_y, ceiling.plane_y])
    frac_between = float(np.mean((cam_positions[:, 1] >= lo_y) & (cam_positions[:, 1] <= hi_y)))
    if frac_between < 0.9:
        warnings.append(
            f"only {frac_between:.0%} of camera positions lie between the chosen floor "
            "and ceiling planes - one or both may not actually be floor/ceiling"
        )

    if floor.n_views_supporting < min_view_support:
        warnings.append(
            f"chosen floor candidate is only supported by {floor.n_views_supporting} "
            f"view(s) (min_view_support={min_view_support}) - low confidence"
        )

    # THE original fix: orient "up" using where the cameras actually are relative to
    # the floor plane, never an assumption about which raw-Y direction is up. A plane's
    # own normal only tells you it's horizontal, not which side people were standing on.
    up = floor.normal.copy()
    cam_centroid = cam_positions.mean(axis=0)
    floor_point = floor.inlier_points.mean(axis=0)
    if np.dot(up, cam_centroid - floor_point) < 0:
        up = -up

    lo_ids = {id(c) for c in group_lo}
    diagnostics = {
        "n_horizontal_candidates": len(candidates),
        "candidates": [
            {
                "side": "low" if id(c) in lo_ids else "high",
                "plane_y": c.plane_y,
                "tilt_deg": c.tilt_deg,
                "support": c.support,
                "n_views": c.n_views_supporting,
                "hull_area": c.hull_area,
                "is_chosen_floor": c is floor,
                "is_chosen_ceiling": c is ceiling,
            }
            for c in candidates
        ],
        "floor_y": floor.plane_y,
        "floor_tilt_deg": floor.tilt_deg,
        "floor_support": floor.support,
        "floor_n_views": floor.n_views_supporting,
        "floor_hull_area": floor.hull_area,
        "ceiling_y": ceiling.plane_y,
        "ceiling_tilt_deg": ceiling.tilt_deg,
        "ceiling_support": ceiling.support,
        "ceiling_n_views": ceiling.n_views_supporting,
        "room_bbox_xy_area": room_bbox_xy_area,
        "frac_cameras_between": frac_between,
        "chosen": chosen,
        "cam_y_mean": float(cam_positions[:, 1].mean()),
        "cam_y_range": [float(cam_positions[:, 1].min()), float(cam_positions[:, 1].max())],
    }

    return FloorCeilingResult(
        floor=floor,
        ceiling=ceiling,
        up_normal=up,
        frac_cameras_between=frac_between,
        chosen=chosen,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def rotation_to_up(up_normal: np.ndarray) -> np.ndarray:
    """Rodrigues rotation mapping `up_normal` -> +Y.

    Handles the antiparallel case explicitly (up_normal ~= -Y): this is a REAL
    possibility here, not a formality - `up_normal` is expressed in MapAnything's
    arbitrary raw frame, which has no reason to already be close to +Y, and the whole
    point of this function is correcting that. Rodrigues' formula is singular exactly
    there (cross product is zero), so it needs its own branch.
    """
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(up_normal, target)
    s = np.linalg.norm(v)
    cos_a = float(np.dot(up_normal, target))
    if s < 1e-8:
        if cos_a > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees around any axis perpendicular to Y (world X
        # works). det(diag(1,-1,-1)) = +1, a genuine proper rotation, not a reflection.
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - cos_a) / (s**2))


if __name__ == "__main__":
    # Retained CLI entry point for standalone diagnostics (matches the original
    # floor_ceiling_disambiguate.py's usage) - NOT used by the pipeline itself, which
    # goes through stage_align.py for the full transform-writing behavior.
    import sys

    per_view_dir = sys.argv[1]
    pts, _cols, view_idx, cam_positions, _conf = load_run(per_view_dir)
    result = find_floor_and_ceiling(pts, view_idx, cam_positions)
    print(f"chosen={result.chosen} floor_y={result.floor.plane_y:.4f} "
          f"ceiling_y={result.ceiling.plane_y:.4f} floor_tilt={result.floor.tilt_deg:.2f}deg")
    for w in result.warnings:
        print(f"WARNING: {w}")
    for k, v in result.diagnostics.items():
        print(f"  {k}: {v}")
