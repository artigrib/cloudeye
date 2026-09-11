"""camera_convention verifies a per-view artifact (pts3d, mask, intrinsics,
camera_pose) matches the OpenCV / camera-to-world / integer-pixel-grid convention every
downstream GPU stage assumes. Synthetic fixtures cover the three failure shapes this
was built to catch (a self-consistent baseline, a transposed pixel grid, and an
OpenGL-style -Z-forward camera); a real-data cross-check runs opportunistically against
per-view data already present on this machine, skipped (not failed) when absent - same
pattern as test_ply_reader.py::test_real_pipeline_fixture_if_available, and no binary
data is checked into the repo for it.

2026-09-04 rework (see docs/DECISIONS.md's "camera-convention threshold derivation"
entry): a single pass/fail threshold either failed the whole job on one harmless noisy
view or silently passed a borderline one with zero record. This file now also covers
the four-tier per-view severity (CLEAN/WARN/DROP/SIGNATURE) and the job-level
aggregate decision in summarize_job_convention - entirely with synthetic
ConventionCheckResult fixtures, no GPU/MapAnything/real per-view data needed for any
of it (see test_one_noisy_view_is_dropped_but_job_survives and friends below)."""

from pathlib import Path

import numpy as np
import pytest

from gpu.camera_convention import (
    STATUS_CLEAN,
    STATUS_DROP,
    STATUS_SIGNATURE,
    STATUS_WARN,
    ConventionCheckResult,
    check_camera_convention,
    summarize_job_convention,
)

# A fixed, non-trivial cam2world pose (rotation about Y + translation) shared by the
# synthetic fixtures below, so each test only has to vary the one thing it's checking.
_THETA = 0.3
_ROTATION = np.array(
    [
        [np.cos(_THETA), 0, np.sin(_THETA)],
        [0, 1, 0],
        [-np.sin(_THETA), 0, np.cos(_THETA)],
    ]
)
_TRANSLATION = np.array([1.0, 2.0, 3.0])
_CAMERA_POSE = np.eye(4)
_CAMERA_POSE[:3, :3] = _ROTATION
_CAMERA_POSE[:3, 3] = _TRANSLATION

_FX, _FY, _CX, _CY = 80.0, 82.0, 48.0, 32.0
_K = np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]])
_H, _W = 64, 96


def _synthetic_view(*, opengl_forward: bool = False, pixel_noise_std: float = 0.0, rng_seed: int = 0):
    """A self-consistent per-view artifact built by definition: pick a depth per pixel,
    back-project through K under the OpenCV convention (optionally negated, to simulate
    an OpenGL -Z-forward camera), then forward-project through the SAME cam2world pose
    the check will be given - so a correct implementation must find ~zero error.
    `pixel_noise_std` > 0 perturbs the pixel indices used for back-projection (not the
    ground-truth ones the check compares against), simulating a real noisy/degraded
    view without touching the convention itself - the WARN/DROP tier fixtures below."""
    rng = np.random.default_rng(rng_seed)
    rows, cols = np.meshgrid(np.arange(_H), np.arange(_W), indexing="ij")
    depth = rng.uniform(1.0, 5.0, size=(_H, _W))
    noisy_rows = rows + rng.normal(0, pixel_noise_std, size=rows.shape) if pixel_noise_std else rows
    noisy_cols = cols + rng.normal(0, pixel_noise_std, size=cols.shape) if pixel_noise_std else cols
    x_cam = (noisy_cols - _CX) * depth / _FX
    y_cam = (noisy_rows - _CY) * depth / _FY
    z_cam = -depth if opengl_forward else depth
    p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    p_world = p_cam @ _ROTATION.T + _TRANSLATION
    mask = np.ones((_H, _W), dtype=bool)
    return p_world, mask


def test_self_consistent_view_passes_with_near_zero_error():
    p_world, mask = _synthetic_view()
    result = check_camera_convention(p_world, mask, _K, _CAMERA_POSE)
    assert result.status == STATUS_CLEAN
    assert result.median_px_error < 1e-6
    assert result.positive_z_fraction == 1.0


def test_transposed_pixel_grid_is_detected_and_named():
    """Simulate a row/col transposition bug: same square-shaped data, but the world
    points at [row, col] are actually what belongs at [col, row]. This is the real
    ~140-160px signature - must land in STATUS_SIGNATURE, the tier that fails the job
    regardless of any other view. A production-scale image (300x300, not the other
    tests' small 64x96 fixture) is used deliberately: a transpose's error scales with
    image size (how far a swapped (row, col) can land from the true one), and this
    project's real 140-160px signature was measured on ~300-500px-scale real frames -
    a small toy grid understates it (the 80x80 grid this test used before 2026-09-04
    only reached ~32px, well under a real signature but still clearly not clean)."""
    n, fx, fy, cx, cy = 300, 300.0, 305.0, 150.0, 150.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rng = np.random.default_rng(0)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    depth = rng.uniform(1.0, 5.0, size=(n, n))
    x_cam = (cols - cx) * depth / fx
    y_cam = (rows - cy) * depth / fy
    p_cam = np.stack([x_cam, y_cam, depth], axis=-1)
    p_world = p_cam @ _ROTATION.T + _TRANSLATION
    p_world_transposed = np.transpose(p_world, (1, 0, 2))
    mask = np.ones((n, n), dtype=bool)

    result = check_camera_convention(p_world_transposed, mask, K, _CAMERA_POSE)
    assert result.status == STATUS_SIGNATURE
    assert result.should_fail_job
    assert result.median_px_error > 100  # in the real ~140-160px signature's range
    assert result.swapped_median_px_error is not None
    assert result.swapped_median_px_error < 1.0  # swapping (row, col) fits almost exactly
    assert "TRANSPOSED" in result.detail


def test_opengl_style_negative_z_forward_is_detected_and_named():
    """A model whose camera convention is OpenGL (-Z forward) instead of OpenCV
    (+Z forward), with camera_pose still a genuine cam2world transform - the majority-
    negative-camera-frame-Z signal, not the reprojection-error signal, is what must
    catch this (a uniform sign flip on Z alone wouldn't reliably blow up pixel error at
    the image center). A structural artifact defect, always STATUS_SIGNATURE."""
    p_world, mask = _synthetic_view(opengl_forward=True)
    result = check_camera_convention(p_world, mask, _K, _CAMERA_POSE)
    assert result.status == STATUS_SIGNATURE
    assert result.should_fail_job
    assert result.positive_z_fraction < 0.05
    assert "OpenGL" in result.detail or "world-to-camera" in result.detail


def test_bad_intrinsics_shape_fails_without_a_crash():
    p_world, mask = _synthetic_view()
    result = check_camera_convention(p_world, mask, np.eye(4), _CAMERA_POSE)
    assert result.status == STATUS_SIGNATURE
    assert "3, 3" in result.detail


def test_bad_camera_pose_shape_fails_without_a_crash():
    p_world, mask = _synthetic_view()
    result = check_camera_convention(p_world, mask, _K, np.eye(3))
    assert result.status == STATUS_SIGNATURE
    assert "4, 4" in result.detail


def test_empty_mask_fails_without_a_crash():
    p_world, mask = _synthetic_view()
    result = check_camera_convention(p_world, np.zeros_like(mask), _K, _CAMERA_POSE)
    assert result.status == STATUS_SIGNATURE
    assert "no valid pixels" in result.detail


def test_moderately_noisy_view_is_dropped_not_signature():
    """A view degraded enough to fail the DROP threshold (default 15px median) but
    nowhere near the real ~140px swap signature - room.mp4/street1.mp4-shaped: real,
    but not evidence of a convention bug. Must land in STATUS_DROP, not
    STATUS_SIGNATURE, and must NOT set should_fail_job."""
    p_world, mask = _synthetic_view(pixel_noise_std=20.0)
    result = check_camera_convention(p_world, mask, _K, _CAMERA_POSE)
    assert result.status == STATUS_DROP
    assert result.should_drop
    assert not result.should_fail_job
    assert result.median_px_error > 15.0
    assert result.median_px_error < 50.0


def test_mildly_noisy_view_is_warned_not_dropped():
    """Elevated but small - the street1.mp4 shape (1.55px, 3x the old 0.5px threshold
    but far under any real-corruption signature). Must be STATUS_WARN: logged, but
    kept in the pool (should_drop is False)."""
    p_world, mask = _synthetic_view(pixel_noise_std=1.5)
    result = check_camera_convention(p_world, mask, _K, _CAMERA_POSE)
    assert result.status == STATUS_WARN
    assert not result.should_drop
    assert not result.should_fail_job
    assert result.median_px_error > 1.0


# --- Job-level aggregate (summarize_job_convention) ---------------------------------


def _clean_result(median=0.1) -> ConventionCheckResult:
    return ConventionCheckResult(
        status=STATUS_CLEAN, median_px_error=median, p95_px_error=median * 2,
        positive_z_fraction=1.0, n_sampled=500,
    )


def _drop_result(median=20.0) -> ConventionCheckResult:
    return ConventionCheckResult(
        status=STATUS_DROP, median_px_error=median, p95_px_error=median * 2,
        positive_z_fraction=1.0, n_sampled=500, detail="dropped",
    )


def _signature_result(median=145.0) -> ConventionCheckResult:
    return ConventionCheckResult(
        status=STATUS_SIGNATURE, median_px_error=median, p95_px_error=median * 1.2,
        positive_z_fraction=1.0, n_sampled=500, detail="TRANSPOSED",
    )


def test_synthetic_swap_signature_fails_the_job_even_alone():
    """The (a)/(c) case from the threshold-derivation discussion: a single view at the
    real ~140px signature fails the whole job outright, regardless of how many other
    views are clean."""
    results = [_clean_result()] * 99 + [_signature_result()]
    summary = summarize_job_convention(results)
    assert summary.should_fail
    assert "regardless of other views" in summary.fail_reason


def test_one_noisy_view_is_dropped_but_job_survives():
    """The (a) case: one bad view out of 100+ is excluded, the scene/job continues -
    matches room.mp4/street1.mp4's shape (a single elevated view, well under 10% of a
    100+-view job)."""
    results = [_clean_result()] * 99 + [_drop_result()]
    summary = summarize_job_convention(results)
    assert not summary.should_fail
    assert summary.n_dropped == 1
    assert summary.kept_views == 99


def test_ten_percent_dropped_fails_the_job():
    """The (b) case: dropped fraction over the 10% default max_dropped_fraction fails
    the whole job even though no single view hit the signature tier - looks systemic,
    not isolated noise."""
    results = [_clean_result()] * 88 + [_drop_result()] * 12  # 12/100 = 12% > 10%
    summary = summarize_job_convention(results)
    assert summary.should_fail
    assert "exceeds max_dropped_fraction" in summary.fail_reason
    assert summary.n_dropped == 12


def test_below_ten_percent_dropped_survives():
    results = [_clean_result()] * 91 + [_drop_result()] * 9  # 9% < 10%
    summary = summarize_job_convention(results)
    assert not summary.should_fail
    assert summary.n_dropped == 9
    assert summary.kept_views == 91


def test_short_clip_fails_on_absolute_survivor_floor_even_at_zero_percent_dropped():
    """min_surviving_views protects short clips independently of the fraction check:
    19 views, ZERO dropped (0% - nowhere near the 10% fraction bound), but 19 is still
    below the default floor of 20 - RANSAC-based plane fitting downstream needs a
    minimum absolute number of views regardless of how proportionally clean the run
    was."""
    results = [_clean_result()] * 19  # 19 kept, 0 dropped - fraction check can't fire
    summary = summarize_job_convention(results)
    assert summary.should_fail
    assert "reliable RANSAC-based plane fitting" in summary.fail_reason


# --- Real-data cross-check --------------------------------------------------------------

_REAL_PER_VIEW_DIRS = [
    Path("/data/spike_fixtures/pose_validation_hotel_room_20260902/per_view"),
    Path("<gpu-backup>/data/output_horizontal/per_view"),  # the one landscape (H<W)
    # set on this machine - the shape most likely to expose a row/col transposition.
    Path("<gpu-backup>/data/output_vertical/run1/per_view"),
]


def test_real_pipeline_fixtures_if_available():
    """Cross-check against real MapAnything per-view output, if this checkout has any -
    skipped (not failed) otherwise, since these directories aren't checked into the
    repo (see test_ply_reader.py::test_real_pipeline_fixture_if_available for the same
    pattern). Confirms every view classifies as CLEAN with real margin under the new
    default thresholds: measured on this machine at worst-case median 0.15px / p95
    0.45px across three independent real capture sets, both comfortably under
    DEFAULT_WARN_MEDIAN_PX=1.0."""
    any_found = False
    for per_view_dir in _REAL_PER_VIEW_DIRS:
        files = sorted(per_view_dir.glob("view_*.npz"))
        if not files:
            continue
        any_found = True
        for f in files:
            d = np.load(f)
            result = check_camera_convention(d["pts3d"], d["mask"], d["intrinsics"], d["camera_pose"])
            assert result.status == STATUS_CLEAN, f"{f}: {result.detail}"
            assert result.median_px_error < 0.5, f"{f}: median={result.median_px_error}"
    if not any_found:
        pytest.skip("no local per-view fixture data available (none of _REAL_PER_VIEW_DIRS exist)")
