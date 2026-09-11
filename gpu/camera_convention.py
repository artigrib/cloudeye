#!/usr/bin/env python3
"""Verifies that a MapAnything-shaped per-view artifact (pts3d, mask, intrinsics,
camera_pose) actually matches the camera convention every downstream pipeline stage
assumes: OpenCV (+X right, +Y down, +Z forward - points in front of the camera have
positive camera-frame Z), `camera_pose` is camera-to-world, `intrinsics` is a standard
3x3 pinhole K, and the pixel grid is INTEGER coordinates (pts3d[row, col], no +0.5
pixel-center offset).

Why this exists: gpu/stage_infer.py's own docstring already flags that "MapAnything
does not share a world coordinate frame ... across separate .infer() calls" - but until
now nothing checked the convention WITHIN one call either. Every stage downstream of
stage_infer.py (gpu/floor_ceiling.py's floor/ceiling fit, gpu/stage_objects.py's 3D
object localization, gpu/stage_occupancy.py's grid) blindly reuses pts3d/camera_pose/
intrinsics with no re-check. A wrong convention - a transposed pixel grid, world-to-
camera instead of camera-to-world, or an OpenGL (-Z-forward) camera - would silently
corrupt every one of them with no error: the same failure shape this project has
already been bitten by twice for the floor/ceiling axis (see floor_ceiling.py's module
docstring, "a plane found via RANSAC ... can be a perfectly clean, low-tilt fit that is
nonetheless the CEILING"). This module exists so a future model swap fails loudly at
ingestion instead of producing plausible-looking, silently wrong geometry.

This validates the ARTIFACT CONTRACT (what pts3d/mask/intrinsics/camera_pose are
documented to mean), not MapAnything's internals specifically - it makes no assumption
about which model produced the arrays, only about what they claim to represent.

Numpy-only (no torch/open3d) by design, the same rationale as gpu/stage_objects.py:
stage_infer.py and stage_align.py both already run under the mapanything venv (see
gpu/run_pipeline.sh), so a numpy dependency here creates no cross-venv conflict -
unlike gpu/job_io.py, which is imported by three venvs on different numpy pins and
stays stdlib-only for exactly that reason. Being numpy-only also means this module is
unit-testable from the backend venv with no GPU pipeline dependencies at all.

Verified empirically against real production per-view data (2026-09-03, see
tests/test_camera_convention.py): median reprojection error 0.05-0.15px under this
module's assumed convention across three independent real capture sets, versus
~140-160px under a row/col-swapped hypothesis on the same data.

Four-tier severity (reworked 2026-09-04, see docs/DECISIONS.md's "camera-convention
threshold derivation" entry for the full account of why): a single-threshold pass/fail
either failed the whole job on one noisy-but-harmless view (room.mp4: 0.50px, exactly
at the old 0.5px boundary) or silently passed a borderline one with zero record
(street1.mp4: 1.55px, 3x the old threshold, never logged anywhere since a job that
passes leaves no per-view trace). CLEAN/WARN/DROP view-level tiers, plus a job-level
aggregate decision in `summarize_job_convention`, replace that - see the STATUS_*
constants and threshold derivation comments below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Per-view severity tiers -------------------------------------------------------
STATUS_CLEAN = "clean"
STATUS_WARN = "warn"
STATUS_DROP = "drop"
STATUS_SIGNATURE = "signature"

# The actual convention-mismatch signature, measured empirically (see module
# docstring): a synthetic row/col transposition produces ~140-160px median
# reprojection error on real data. Every threshold below is derived from THIS number
# with an explicit, stated safety margin - not from the 3 clean captures available
# when this module was first written (that gave a real but thin margin over clean
# data: 0.5px was only ~3.3x the worst observed clean median). See docs/DECISIONS.md's
# 2026-09-04 entry for the room.mp4/street1.mp4 numbers that prompted this rework.
SIGNATURE_PX = 140.0  # the lower end of the observed 140-160px swap signature

# Below this: real per-pixel noise on real data (motion blur, a rolling-shutter frame,
# a depth discontinuity) - measured 0.05-0.15px median / up to 0.45px p95 across 3 real
# capture sets (~7x and ~2x those worst-observed numbers respectively - a real margin,
# not a formula, but the two numbers behind it are on record here, unlike the old
# single hardcoded 0.5px). Views at or below this are CLEAN; nothing is logged.
DEFAULT_WARN_MEDIAN_PX = 1.0

# Above this: DROPPED from pooling, not trusted, but does not fail the job by itself -
# derived from SIGNATURE_PX with a 10x safety margin (140 / 10 = 14, rounded up to 15
# for a round number), i.e. still ~100x the worst observed clean median. A view this
# bad could plausibly still be ordinary per-frame noise (a badly motion-blurred
# keyframe) rather than a convention bug, so it is excluded from the pooled point
# cloud rather than aborting the whole run - gpu/floor_ceiling.py's RANSAC-based
# fitting over the pooled cloud degrades gracefully with a few percent fewer points
# (see summarize_job_convention for the job-level fraction/count checks that DO fail
# the job if too many views land here).
DEFAULT_DROP_MEDIAN_PX = 15.0
# p95 companion: 3x the drop-median threshold (same ratio precedent as the old
# 0.5/5.0 median/p95 pair), which still leaves a >3x margin below SIGNATURE_PX even
# at the noisier p95 percentile.
DEFAULT_DROP_P95_PX = 45.0

# At or above this, a SINGLE view is unambiguous evidence of a real convention bug -
# roughly 1/3 of SIGNATURE_PX, but still ~330x the worst clean median observed, and
# well above anything a single bad frame's per-pixel noise plausibly produces. Fails
# the whole job immediately regardless of how many other views are clean: a genuine
# convention mismatch (wrong axis, transposed grid, world-to-camera instead of
# camera-to-world) is structural - it comes from how the whole run's data was
# produced, not from one frame - so it is expected to show up this badly on
# effectively every view, not just one. Distinguishing "this one view is corrupt" from
# "the whole run's convention is corrupt" is exactly the point of this tier.
DEFAULT_NEAR_SIGNATURE_PX = 50.0

# --- Job-level aggregate defaults (see summarize_job_convention) -------------------
# A real convention bug fails ~100% of views (see DEFAULT_NEAR_SIGNATURE_PX above), so
# any reasonable fraction bound cleanly separates "isolated per-frame noise" (a few
# dropped views out of 100+) from "systemic corruption" - not tuned finer than a round
# 10%.
DEFAULT_MAX_DROPPED_FRACTION = 0.10
# Independent of fraction: gpu/floor_ceiling.py's RANSAC-based plane fitting needs a
# minimum absolute number of views to be reliable regardless of the job's total size -
# a 5-view job with 1 dropped view is only a 20% fraction (would fail the fraction
# check above anyway) but a 30-view job with 3 dropped (10%, right at the fraction
# limit) still leaves 27 views, plenty; the fraction check alone under-protects short
# clips. 20 is a round floor comfortably above the handful of views RANSAC's own
# min_support/min_view_support floors already require per plane.
DEFAULT_MIN_SURVIVING_VIEWS = 20


class CameraConventionError(Exception):
    """Raised when a job-level camera-convention decision fails the whole job - either
    a single STATUS_SIGNATURE view (see ConventionCheckResult.should_fail_job) or the
    aggregate check in summarize_job_convention. The message names exactly what looked
    wrong - never a bare "check failed", matching this project's "log every check's
    actual numbers" standard (see app/services/scene_validator.py's module
    docstring)."""


@dataclass(frozen=True)
class ConventionCheckResult:
    status: str  # one of STATUS_CLEAN / STATUS_WARN / STATUS_DROP / STATUS_SIGNATURE
    median_px_error: float
    p95_px_error: float
    positive_z_fraction: float  # fraction of sampled points with camera-frame Z > 0
    n_sampled: int
    swapped_median_px_error: float | None = None  # median error if (row, col) were swapped
    detail: str = ""

    @property
    def should_drop(self) -> bool:
        """True if this view should be excluded from pooling but the JOB can still
        continue - STATUS_DROP only. STATUS_SIGNATURE is deliberately NOT included
        here; that is a hard job failure (see should_fail_job), not a per-view drop."""
        return self.status == STATUS_DROP

    @property
    def should_fail_job(self) -> bool:
        """True if this view ALONE is sufficient grounds to fail the whole job,
        regardless of how many other views are clean - STATUS_SIGNATURE, or one of the
        structural artifact failures (_fail() below: bad shapes, bad intrinsics,
        camera_pose that isn't camera-to-world) which always map to STATUS_SIGNATURE."""
        return self.status == STATUS_SIGNATURE


@dataclass(frozen=True)
class JobConventionSummary:
    """Aggregate, job-level decision over every view's ConventionCheckResult - see
    summarize_job_convention."""

    n_views: int
    n_clean: int
    n_warned: int
    n_dropped: int
    kept_views: int
    dropped_fraction: float
    should_fail: bool
    fail_reason: str = ""  # empty iff should_fail is False


def _validate_intrinsics(K: np.ndarray) -> str | None:
    """Returns an error string if K doesn't look like a standard 3x3 pinhole matrix,
    else None."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        return f"intrinsics has shape {K.shape}, expected (3, 3)"
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    off_diag = (K[0, 1], K[1, 0], K[2, 0], K[2, 1])
    if fx <= 0 or fy <= 0:
        return f"intrinsics fx={fx:.3f}, fy={fy:.3f} - expected both positive"
    if any(abs(v) > 1e-3 for v in off_diag):
        return f"intrinsics has non-zero off-diagonal entries {off_diag} - not a standard pinhole K"
    if abs(K[2, 2] - 1.0) > 1e-3:
        return f"intrinsics[2,2]={K[2, 2]:.4f}, expected 1.0"
    if cx <= 0 or cy <= 0:
        return f"intrinsics principal point ({cx:.2f}, {cy:.2f}) is non-positive"
    return None


def _fail(detail: str, *, positive_z_fraction: float = float("nan"), n_sampled: int = 0) -> ConventionCheckResult:
    """A structural artifact defect (bad shapes, bad intrinsics, camera_pose that
    isn't camera-to-world) - always STATUS_SIGNATURE. These aren't "a bit noisy", they
    mean the artifact doesn't even have the shape/meaning downstream code assumes, so
    there's nothing to drop-and-continue with; the job fails, same as the old
    unconditional behavior for these specific checks."""
    return ConventionCheckResult(
        status=STATUS_SIGNATURE,
        median_px_error=float("nan"),
        p95_px_error=float("nan"),
        positive_z_fraction=positive_z_fraction,
        n_sampled=n_sampled,
        detail=detail,
    )


def check_camera_convention(
    pts3d: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    *,
    sample_size: int = 500,
    warn_median_px: float = DEFAULT_WARN_MEDIAN_PX,
    drop_median_px: float = DEFAULT_DROP_MEDIAN_PX,
    drop_p95_px: float = DEFAULT_DROP_P95_PX,
    near_signature_px: float = DEFAULT_NEAR_SIGNATURE_PX,
    min_positive_z_fraction: float = 0.95,
    rng_seed: int = 0,
) -> ConventionCheckResult:
    """Back-projects a deterministic random sample of valid pixels through `pts3d` and
    `camera_pose` into the camera frame, reprojects through `intrinsics`, and compares
    against each sample's own (row, col) pixel index. See module docstring for what
    this validates, why, and the four-tier severity (CLEAN/WARN/DROP/SIGNATURE) this
    classifies into.

    `drop_p95_px` is looser than `drop_median_px` deliberately: a handful of sampled
    pixels can legitimately land near a depth discontinuity or object edge (see
    gpu/stage_align.py's statistical-outlier-removal comment on "sparse fly-away
    points ... near depth discontinuities/object edges") without the convention itself
    being wrong - the median is the number that actually discriminates a real
    convention mismatch (measured at ~140-160px under a swap) from real per-pixel
    noise (measured at well under 1px on real data)."""
    pts3d = np.asarray(pts3d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    camera_pose = np.asarray(camera_pose, dtype=np.float64)

    if pts3d.ndim != 3 or pts3d.shape[2] != 3:
        return _fail(f"pts3d has shape {pts3d.shape}, expected (H, W, 3)")
    if mask.shape != pts3d.shape[:2]:
        return _fail(f"mask has shape {mask.shape}, pts3d is {pts3d.shape[:2]} - shapes must match")
    if camera_pose.shape != (4, 4):
        return _fail(f"camera_pose has shape {camera_pose.shape}, expected (4, 4)")
    intrinsics_err = _validate_intrinsics(intrinsics)
    if intrinsics_err:
        return _fail(intrinsics_err)
    K = np.asarray(intrinsics, dtype=np.float64)

    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return _fail("mask has no valid pixels - nothing to verify")

    rng = np.random.default_rng(rng_seed)
    n = min(sample_size, len(rows))
    idx = rng.choice(len(rows), size=n, replace=False)
    sample_rows, sample_cols = rows[idx], cols[idx]
    p_world = pts3d[sample_rows, sample_cols]  # (n, 3)

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    # cam2world assumption: p_world = R @ p_cam + t  =>  p_cam = R.T @ (p_world - t).
    # In row-vector form, (R.T @ v) == (v @ R), which is what lets this stay a single
    # vectorized matmul over all n samples rather than a per-point loop.
    p_cam = (p_world - t) @ R

    z = p_cam[:, 2]
    positive_z_fraction = float(np.mean(z > 0))
    if positive_z_fraction < min_positive_z_fraction:
        return _fail(
            f"camera_pose does not look like camera-to-world under an OpenCV "
            f"(+Z-forward) convention - only {positive_z_fraction:.1%} of sampled "
            f"points have positive camera-frame Z (expected "
            f">={min_positive_z_fraction:.0%}). This looks like world-to-camera, or "
            f"an OpenGL (-Z-forward) camera convention.",
            positive_z_fraction=positive_z_fraction,
            n_sampled=n,
        )

    # Guard against near-zero Z for a degenerate sample that slipped past the majority
    # check above - report those points' error as non-finite (dropped below) rather
    # than crashing on a division by ~0.
    safe_z = np.where(np.abs(z) < 1e-9, np.nan, z)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u_pred = fx * p_cam[:, 0] / safe_z + cx  # predicted column
    v_pred = fy * p_cam[:, 1] / safe_z + cy  # predicted row

    err = np.sqrt((u_pred - sample_cols) ** 2 + (v_pred - sample_rows) ** 2)
    finite = np.isfinite(err)
    err = err[finite]
    if len(err) == 0:
        return _fail(
            "every sampled point had near-zero camera-frame Z - cannot reproject",
            positive_z_fraction=positive_z_fraction,
            n_sampled=n,
        )
    median_err = float(np.median(err))
    p95_err = float(np.percentile(err, 95))

    # Transposition diagnostic: reuse the SAME u_pred/v_pred (they don't change - only
    # which ground-truth index they're compared against does) and check whether
    # comparing against the swapped (row, col) pair fits dramatically better. This is
    # exactly the real failure mode a transposed pixel grid produces.
    swapped_err = None
    if median_err > warn_median_px:
        swap = np.sqrt((u_pred[finite] - sample_rows[finite]) ** 2 + (v_pred[finite] - sample_cols[finite]) ** 2)
        if len(swap):
            swapped_err = float(np.median(swap))

    if median_err >= near_signature_px:
        status = STATUS_SIGNATURE
    elif median_err > drop_median_px or p95_err > drop_p95_px:
        status = STATUS_DROP
    elif median_err > warn_median_px:
        status = STATUS_WARN
    else:
        status = STATUS_CLEAN

    detail = ""
    if status != STATUS_CLEAN:
        detail = (
            f"reprojection error under the assumed convention (OpenCV, camera-to-"
            f"world, integer pixel grid) is {status.upper()}: median={median_err:.2f}px, "
            f"p95={p95_err:.2f}px (warn>{warn_median_px}px, drop>{drop_median_px}px "
            f"median or >{drop_p95_px}px p95, signature>={near_signature_px}px)."
        )
        if swapped_err is not None and swapped_err < median_err / 5:
            detail += (
                f" Swapping (row, col) drops the median to {swapped_err:.2f}px - the "
                f"pixel grid looks TRANSPOSED (row/col swapped)."
            )
    return ConventionCheckResult(
        status=status,
        median_px_error=median_err,
        p95_px_error=p95_err,
        positive_z_fraction=positive_z_fraction,
        n_sampled=len(err),
        swapped_median_px_error=swapped_err,
        detail=detail,
    )


def summarize_job_convention(
    results: list[ConventionCheckResult],
    *,
    max_dropped_fraction: float = DEFAULT_MAX_DROPPED_FRACTION,
    min_surviving_views: int = DEFAULT_MIN_SURVIVING_VIEWS,
) -> JobConventionSummary:
    """Aggregates one job's per-view check_camera_convention results into a job-level
    pass/fail decision - the (b) half of the 2026-09-04 rework (see module docstring):
    one bad view drops and the job continues, but too MANY bad views (or too few
    survivors regardless of fraction) still fails the whole job.

    Pure function, numpy-only inputs, no I/O - takes already-computed
    ConventionCheckResult objects rather than raw per-view arrays, so it is testable
    with synthetic fixtures and no GPU/MapAnything/real per-view data at all (the
    2026-09-04 DIAG.md finding that prompted this rework was specifically that NEITHER
    a passing NOR a failing real job left any reconstructible per-view trace - this
    function's aggregate reasoning shouldn't depend on that same missing data either).

    Order of checks matters: a single STATUS_SIGNATURE view fails the job immediately,
    before the fraction/count checks even run - unambiguous evidence of real
    corruption (see DEFAULT_NEAR_SIGNATURE_PX) shouldn't be diluted by how many other
    views happen to be clean.
    """
    n_views = len(results)
    for i, r in enumerate(results):
        if r.should_fail_job:
            return JobConventionSummary(
                n_views=n_views,
                n_clean=sum(1 for x in results if x.status == STATUS_CLEAN),
                n_warned=sum(1 for x in results if x.status == STATUS_WARN),
                n_dropped=sum(1 for x in results if x.should_drop),
                kept_views=0,
                dropped_fraction=1.0,
                should_fail=True,
                fail_reason=(
                    f"view {i}: {r.detail or 'unambiguous camera-convention signature'} "
                    f"(median={r.median_px_error:.2f}px) - this alone fails the job "
                    f"regardless of other views"
                ),
            )

    n_dropped = sum(1 for r in results if r.should_drop)
    n_warned = sum(1 for r in results if r.status == STATUS_WARN)
    n_clean = sum(1 for r in results if r.status == STATUS_CLEAN)
    kept_views = n_views - n_dropped
    dropped_fraction = (n_dropped / n_views) if n_views else 0.0

    if dropped_fraction > max_dropped_fraction:
        return JobConventionSummary(
            n_views=n_views,
            n_clean=n_clean,
            n_warned=n_warned,
            n_dropped=n_dropped,
            kept_views=kept_views,
            dropped_fraction=dropped_fraction,
            should_fail=True,
            fail_reason=(
                f"{n_dropped}/{n_views} views dropped ({dropped_fraction:.1%}) exceeds "
                f"max_dropped_fraction={max_dropped_fraction:.0%} - looks systemic, not "
                f"isolated per-frame noise"
            ),
        )

    if kept_views < min_surviving_views:
        return JobConventionSummary(
            n_views=n_views,
            n_clean=n_clean,
            n_warned=n_warned,
            n_dropped=n_dropped,
            kept_views=kept_views,
            dropped_fraction=dropped_fraction,
            should_fail=True,
            fail_reason=(
                f"only {kept_views} views survived (need >= {min_surviving_views}) for "
                f"reliable RANSAC-based plane fitting downstream, regardless of dropped "
                f"fraction"
            ),
        )

    return JobConventionSummary(
        n_views=n_views,
        n_clean=n_clean,
        n_warned=n_warned,
        n_dropped=n_dropped,
        kept_views=kept_views,
        dropped_fraction=dropped_fraction,
        should_fail=False,
    )


def build_convention_report(
    view_labels: list[str],
    results: list[ConventionCheckResult],
    summary: JobConventionSummary,
    *,
    thresholds: dict | None = None,
) -> dict:
    """Plain-dict, JSON-serializable per-job camera-convention report - the artifact
    docs/DECISIONS.md's 2026-09-04 entry says didn't exist (a job that passed left no
    numeric trace at all; a job that failed left only the one view that tripped it,
    and only in a log line, not a file - both per_view/*.npz, which this data would
    otherwise have had to be reconstructed from, get cleaned up by the worker's normal
    fetch-then-cleanup). Written to the job root (job_paths()['camera_convention_report'],
    NOT under per_view/) so it survives that cleanup regardless of pass/fail."""
    return {
        "n_views": summary.n_views,
        "n_clean": summary.n_clean,
        "n_warned": summary.n_warned,
        "n_dropped": summary.n_dropped,
        "kept_views": summary.kept_views,
        "dropped_fraction": round(summary.dropped_fraction, 4),
        "should_fail": summary.should_fail,
        "fail_reason": summary.fail_reason,
        "thresholds": thresholds or {},
        "views": [
            {
                "view": label,
                "status": r.status,
                "median_px_error": r.median_px_error,
                "p95_px_error": r.p95_px_error,
                "positive_z_fraction": r.positive_z_fraction,
                "dropped": r.should_drop,
                "detail": r.detail,
            }
            for label, r in zip(view_labels, results)
        ],
    }


def assert_camera_convention(
    pts3d: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    *,
    view_label: str = "",
    **kwargs,
) -> ConventionCheckResult:
    """Same check as check_camera_convention, but raises CameraConventionError instead
    of returning a result when this ONE view alone is grounds to fail the job
    (`result.should_fail_job` - STATUS_SIGNATURE only). A STATUS_DROP or STATUS_WARN
    result is returned normally, not raised - callers that need per-view drop/warn
    handling (gpu/stage_infer.py) should call check_camera_convention directly instead
    of this wrapper; this form is for a caller that only cares about the immediate,
    unambiguous "the whole job is dead" case."""
    result = check_camera_convention(pts3d, mask, intrinsics, camera_pose, **kwargs)
    if result.should_fail_job:
        label = f" ({view_label})" if view_label else ""
        raise CameraConventionError(f"camera convention check failed{label}: {result.detail}")
    return result
