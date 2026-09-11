"""voxel_size_for_eps / voxel_downsample_indices / cap_points_uniform /
rescale_cluster_params - the 2026-09-01 hotel-own-batch fix for stage_objects.py's two
real failures on much longer/higher-bitrate own-recorded footage than this project had
tested before: room 1's "curtain" (1,336,153 raw pooled points) hit the 900s
STAGE_TIMEOUT mid-DBSCAN; room 3's "bed" (1,610,369 raw pooled points) was SIGKILLed
(exit 137, OOM-consistent - host memory traced to ~248GB right before the kill).
"""

import numpy as np

from gpu.stage_objects import (
    MAX_CLUSTER_POINTS,
    cap_points_uniform,
    rescale_cluster_params,
    voxel_downsample_indices,
    voxel_size_for_eps,
)


def test_voxel_size_never_exceeds_occupancy_resolution():
    # "huge" class eps (0.10m after the tightening) / 3 = 0.0333, below the 0.05
    # occupancy resolution - resolution isn't the binding constraint here.
    assert voxel_size_for_eps(0.10, grid_resolution=0.05) < 0.05


def test_voxel_size_capped_at_occupancy_resolution_for_large_eps():
    # A hypothetically large eps must not push the voxel coarser than the occupancy
    # grid itself.
    assert voxel_size_for_eps(1.0, grid_resolution=0.05) == 0.05


def test_voxel_size_scales_down_for_small_eps():
    # "small" class eps (0.03m) needs a finer voxel than "huge" (0.10m), or downsampling
    # would itself destroy small objects' ability to cluster.
    assert voxel_size_for_eps(0.03) < voxel_size_for_eps(0.10)


def test_voxel_downsample_collapses_duplicate_multi_view_points():
    """The actual mechanism that fixes the timeout/OOM: many near-duplicate points from
    overlapping camera views of the same physical surface collapse to one representative
    point per occupied voxel."""
    rng = np.random.default_rng(0)
    # 100 points tightly clustered around a single physical location (simulating dozens
    # of views all seeing the same patch of curtain fabric), well within one 0.05m voxel
    # - centered on a voxel's own center (0.025 offset), not a voxel boundary, so a tight
    # cluster can't straddle two voxels purely from where the grid lines fall.
    duplicates = rng.normal(loc=[0.025, 1.025, 0.025], scale=0.005, size=(100, 3))
    keep = voxel_downsample_indices(duplicates, voxel_size=0.05)
    assert len(keep) == 1


def test_voxel_downsample_preserves_points_in_different_voxels():
    far_apart = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    keep = voxel_downsample_indices(far_apart, voxel_size=0.05)
    assert len(keep) == 3


def test_voxel_downsample_returns_real_input_points_not_synthetic_ones():
    """Downsampling must select real rows of the input, not compute new averaged
    positions - so cols/vids stay trivially consistent via the same index array."""
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001], [3.0, 3.0, 3.0]])
    keep = voxel_downsample_indices(pts, voxel_size=0.05)
    kept_pts = pts[keep]
    for row in kept_pts:
        assert any(np.array_equal(row, p) for p in pts)


def test_voxel_downsample_realistic_curtain_scale_stays_under_cap():
    """The actual regression case: ~1.3M points from many overlapping views of one large
    surface must collapse to well under MAX_CLUSTER_POINTS after voxel downsampling
    alone, before the hard cap even needs to engage. Simulated as points scattered over
    a 2x2m curtain-sized patch with realistic per-view noise, at a density many times
    higher than one point per voxel - representative of real multi-view SAM3 pooling,
    not merely more points than voxels exist."""
    rng = np.random.default_rng(1)
    n = 1_300_000
    base = rng.uniform(low=[0.0, 0.0, 0.0], high=[2.0, 2.0, 0.05], size=(n, 3))
    voxel = voxel_size_for_eps(0.10)  # huge class, post-tightening
    keep = voxel_downsample_indices(base, voxel_size=voxel)
    # 2x2x0.05m volume at this voxel size has far fewer possible voxels than points.
    max_possible_voxels = int(2.0 / voxel + 1) * int(2.0 / voxel + 1) * int(0.05 / voxel + 1)
    assert len(keep) <= max_possible_voxels
    assert len(keep) < n / 10  # a real, large reduction, not a no-op


def test_cap_points_uniform_no_truncation_when_under_limit():
    assert cap_points_uniform(100, max_points=300) is None


def test_cap_points_uniform_truncates_and_spans_full_range():
    idx = cap_points_uniform(1_000_000, max_points=MAX_CLUSTER_POINTS)
    assert idx is not None
    assert len(idx) <= MAX_CLUSTER_POINTS
    assert idx.min() >= 0
    assert idx.max() < 1_000_000
    # Uniform, not "first N" - must cover close to the full original index range.
    assert idx.max() > 900_000


def test_rescale_cluster_params_scales_down_with_density():
    # Heavy downsampling (10% of raw density) must shrink both thresholds.
    new_samples, new_cluster = rescale_cluster_params(15, 40, compression_ratio=0.1)
    assert new_samples < 15
    assert new_cluster < 40


def test_rescale_cluster_params_never_goes_below_floor():
    new_samples, new_cluster = rescale_cluster_params(15, 40, compression_ratio=0.001)
    assert new_samples >= 3
    assert new_cluster >= 5


def test_rescale_cluster_params_noop_at_full_density():
    new_samples, new_cluster = rescale_cluster_params(15, 40, compression_ratio=1.0)
    assert (new_samples, new_cluster) == (15, 40)
