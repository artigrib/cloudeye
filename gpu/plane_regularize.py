"""Bounded RANSAC plane-position regularization for the aligned room point cloud.

Ports the `mode=project` (position-projection) result of the Workstream A spike
(`gpu/spikes/plane_regularize_spike.py`, see docs/SPRINT-poisson-meshing.md's
"Workstream A results" section) into production. That spike proved that snapping
each plane-inlier point onto its own fitted plane - by moving it only along the
plane's normal, only by its own signed distance to the plane, never sideways, never
past the inlier set's own footprint - visibly fixes "puffy"/rippled noise on flat
surfaces (walls/floor/ceiling) and on a genuinely non-flat curtain, without ever
extrapolating past observed points. `mode=normal` (overriding a point's *normal*
instead of moving it) is NOT ported here: production `aligned_room.ply` has no
normals at all (the frontend renders raw colored points, not a mesh), so that mode
would be a no-op in this pipeline.

Guardrail (unchanged from the spike): this module never creates a new point and
never extends a plane past the points RANSAC actually assigned to it as inliers. No
point ever moves more than `dist_thresh` from where it was observed, and every
displacement is purely along that plane's own fitted normal.

Numpy-only at module level (mirrors gpu/stage_objects.py's `room_bbox_point_mask`
pattern) so `regularize_planes` can be unit-tested from the main app `uv` venv, which
does not have open3d/sklearn installed. The only open3d import in this file lives
inside `_segment_plane_o3d`, used solely as `regularize_planes`'s default
`segment_plane` implementation - never imported at module scope.
"""

import numpy as np

DEFAULT_DIST_THRESH_M = 0.04
DEFAULT_MAX_PLANES = 8
DEFAULT_MIN_INLIER_FRAC = 0.01
# Absolute floor of 30 (not 500 - that was tuned for the room-scale ~600k-point cloud
# and silently made min_inlier_frac inert for anything below ~50k points, e.g. a
# 538-point per-object cloud needs min_inliers=500 just to try, meaning ~93% of the
# object would already have to be one plane before RANSAC even counts a candidate -
# found via a 0-planes-detected false negative on curtain_1.ply in the spike). 30 is
# comfortably above segment_plane's own ransac_n=3 minimum while staying well below
# any object/room's real plane sizes.
MIN_INLIERS_ABSOLUTE = 30
RANSAC_N = 3
RANSAC_NUM_ITERATIONS = 1000


def _segment_plane_o3d(points: np.ndarray, dist_thresh: float) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Default `segment_plane` implementation for `regularize_planes` - imports
    open3d INSIDE this function only (never at module scope, so this module stays
    importable from a venv without open3d installed).

    Takes and returns plain numpy: `points` is a plain (k,3) array (the current
    "remaining" subset, not the full cloud), and the returned `local_inlier_indices`
    are indices into THAT array (0..k-1), not global indices into the full cloud -
    `regularize_planes` does all global/local index bookkeeping itself.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, local_inlier_indices = pcd.segment_plane(
        distance_threshold=dist_thresh, ransac_n=RANSAC_N, num_iterations=RANSAC_NUM_ITERATIONS
    )
    return tuple(model), np.asarray(local_inlier_indices, dtype=np.int64)


def _plane_report(plane_id, points_xyz, n, d_hat, n_inliers_returned):
    """Diagnostic only (never feeds back into geometry, mirrors the spike's
    `_plane_report`): reports how tightly this plane's PROJECTED inliers cluster, in
    the plane's own local 2D frame, plus the actual residual (pre-projection signed
    distance to the fitted plane, from `points_xyz @ n + d_hat` - not the spike's
    centroid-relative version) so callers can confirm/audit that no point moved more
    than the configured distance threshold."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    centroid = points_xyz.mean(axis=0)
    local = points_xyz - centroid
    u_coord = local @ u
    v_coord = local @ v
    resid = points_xyz @ n + d_hat  # actual signed distance to the fitted plane
    return {
        "plane_id": int(plane_id),
        "model_abcd": [float(n[0]), float(n[1]), float(n[2]), float(d_hat)],
        "n_inliers_returned": int(n_inliers_returned),
        "n_inliers_projected": int(len(points_xyz)),
        "extent_u_m": float(u_coord.max() - u_coord.min()) if len(points_xyz) else 0.0,
        "extent_v_m": float(v_coord.max() - v_coord.min()) if len(points_xyz) else 0.0,
        "residual_rms_m": float(np.sqrt(np.mean(resid ** 2))) if len(points_xyz) else 0.0,
        "residual_max_m": float(np.max(np.abs(resid))) if len(points_xyz) else 0.0,
        "centroid": centroid.tolist(),
    }


def regularize_planes(
    points: np.ndarray,
    *,
    dist_thresh: float = DEFAULT_DIST_THRESH_M,
    max_planes: int = DEFAULT_MAX_PLANES,
    min_inlier_frac: float = DEFAULT_MIN_INLIER_FRAC,
    segment_plane=_segment_plane_o3d,
) -> tuple[np.ndarray, list[dict]]:
    """Iterative RANSAC plane extraction (detect -> project -> remove -> repeat), the
    `mode=project` half of the Workstream A spike's `detect_and_regularize_planes`
    (see module docstring). For each detected plane, its inlier points are snapped
    onto the plane by removing only their own out-of-plane component (moved along the
    plane's normal by exactly their own signed distance to it) - never moved
    sideways, never extended past the inlier set's own footprint, never more than
    `dist_thresh`. Points never assigned to any plane are returned unchanged.

    Returns a NEW (N,3) float array (same N, same row order as `points` - rows are
    never reordered or dropped, and `points` itself is never mutated) plus a
    `list[dict]` of per-plane diagnostic reports (see `_plane_report`).

    `segment_plane` is a test seam (this repo has no unittest.mock/monkeypatch usage
    anywhere in tests/, so a default-arg callable is used instead of mocking): a
    callable `(points_subset, dist_thresh) -> (model_abcd, local_inlier_indices)`,
    called with a plain (k,3) numpy array of the CURRENT remaining points (never an
    o3d.geometry.PointCloud - that type never crosses this boundary) and returning
    indices local to that subset (0..k-1). Defaults to `_segment_plane_o3d`.
    """
    out = points.astype(np.float64, copy=True)
    n_total = len(out)
    min_inliers = max(MIN_INLIERS_ABSOLUTE, int(min_inlier_frac * n_total))

    remaining_idx = np.arange(n_total)
    reports: list[dict] = []

    for plane_id in range(max_planes):
        if len(remaining_idx) < min_inliers:
            break

        remaining_pts = out[remaining_idx]
        model_abcd, local_inlier_indices = segment_plane(remaining_pts, dist_thresh)
        n_inliers_returned = len(local_inlier_indices)
        if n_inliers_returned < min_inliers:
            break

        a, b, c, d = model_abcd
        abc = np.array([a, b, c], dtype=np.float64)
        k = np.linalg.norm(abc)
        if k == 0:
            # Degenerate model (segment_plane should never actually return this, but
            # guard rather than divide by zero / silently corrupt data).
            break
        n = abc / k
        d_hat = d / k

        global_inliers = remaining_idx[local_inlier_indices]

        # Local guardrail: don't blindly trust that every index segment_plane handed
        # back actually satisfies OUR dist_thresh - recompute the signed distance
        # ourselves and only keep what passes. This makes "no point ever moves more
        # than dist_thresh" a property of this function's own code, not an inherited
        # assumption about segment_plane's behavior.
        signed = out[global_inliers] @ n + d_hat
        within = np.abs(signed) <= dist_thresh
        global_inliers = global_inliers[within]
        signed = signed[within]

        # Report is built from the PRE-projection positions/residuals (mirrors the
        # spike's ordering: it generates its report from the points as observed,
        # before moving them) - residual_rms_m/residual_max_m measure exactly how far
        # the actual projection step below will move each point, proving the edit
        # stayed bounded within dist_thresh, not the trivial ~0 a post-projection
        # residual would show.
        report = _plane_report(plane_id, out[global_inliers], n, d_hat, n_inliers_returned)
        reports.append(report)

        # Project: remove only the out-of-plane component along the plane's own
        # normal, by exactly each point's own signed distance - never sideways,
        # never past this inlier set's own footprint, never more than dist_thresh
        # (enforced by the guardrail above).
        out[global_inliers] = out[global_inliers] - signed[:, None] * n[None, :]

        # Remove ALL of this plane's originally-returned inliers from `remaining_idx`
        # (not just the ones that passed the guardrail) - a point segment_plane
        # claimed but that failed our own distance check is still a point RANSAC
        # associated with this plane's model, not a candidate for a later plane's fit.
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[local_inlier_indices] = False
        remaining_idx = remaining_idx[mask]

    return out, reports
