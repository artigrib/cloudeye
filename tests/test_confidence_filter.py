"""Task 4 (confidence-aware filtering): MapAnything's per-pixel confidence, when
available, is pooled per-point by floor_ceiling.load_run and turned into a keep-mask by
confidence_percentile_cutoff (used by stage_align.py before SOR, and whose cutoff value
stage_objects.py reuses for its own per-detection masks). Both are pure numpy - no
open3d/torch/sklearn - so they're unit-testable directly, unlike the rest of the
alignment stage.
"""

import numpy as np
import pytest

from gpu.floor_ceiling import confidence_percentile_cutoff, load_run


def test_percentile_cutoff_drops_bottom_fraction():
    conf = np.concatenate([np.full(90, 0.9), np.full(10, 0.1)])
    cutoff, keep_mask = confidence_percentile_cutoff(conf, percentile=10.0)

    # cutoff sits at the 10th percentile of a 90x0.9 / 10x0.1 distribution - the low
    # cluster should be right at/below it and get dropped, the high cluster kept.
    assert keep_mask[:90].all()
    assert not keep_mask[90:].any()
    assert 0.1 <= cutoff <= 0.9


def test_points_without_confidence_pass_through_unfiltered():
    """A point whose source view had no confidence data (NaN) is not the same claim as
    a point that WAS scored and scored low - it must survive the filter regardless of
    the cutoff."""
    conf = np.array([0.9, 0.9, 0.9, 0.1, np.nan, np.nan])
    _cutoff, keep_mask = confidence_percentile_cutoff(conf, percentile=50.0)

    assert keep_mask[4] and keep_mask[5]  # the two NaN entries always pass through
    assert keep_mask[0] and keep_mask[1] and keep_mask[2]  # well above median, kept
    assert not keep_mask[3]  # well below median, dropped


def test_all_nan_raises():
    with pytest.raises(ValueError):
        confidence_percentile_cutoff(np.full(5, np.nan), percentile=10.0)


def _write_view_npz(path, *, n_pts, valid_frac, conf=None):
    h, w = 4, 4
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[: int(h * w * valid_frac)] = True
    pts3d = np.random.default_rng(0).normal(size=(h, w, 3)).astype(np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    camera_pose = np.eye(4, dtype=np.float32)
    kwargs = dict(pts3d=pts3d, mask=mask, img_no_norm=img, camera_pose=camera_pose)
    if conf is not None:
        kwargs["conf"] = conf
    np.savez(path, **kwargs)
    return int(mask.sum())


def test_load_run_pools_confidence_when_all_views_have_it(tmp_path):
    n0 = _write_view_npz(tmp_path / "view_000.npz", n_pts=16, valid_frac=1.0, conf=np.full((4, 4), 0.7, dtype=np.float32))
    n1 = _write_view_npz(tmp_path / "view_001.npz", n_pts=16, valid_frac=0.5, conf=np.full((4, 4), 0.3, dtype=np.float32))

    _pts, _cols, _view_idx, _cams, conf = load_run(tmp_path)

    assert conf is not None
    assert len(conf) == n0 + n1
    assert not np.isnan(conf).any()


def test_load_run_returns_none_when_no_view_has_confidence(tmp_path):
    _write_view_npz(tmp_path / "view_000.npz", n_pts=16, valid_frac=1.0)
    _write_view_npz(tmp_path / "view_001.npz", n_pts=16, valid_frac=0.5)

    _pts, _cols, _view_idx, _cams, conf = load_run(tmp_path)

    assert conf is None


def test_load_run_fills_nan_for_views_missing_confidence(tmp_path):
    """Partial coverage (some views ran conf, some didn't - e.g. a mixed job or a model
    upgrade mid-project) must not silently drop the views lacking it; their points get
    NaN, and confidence_percentile_cutoff already knows to pass those through."""
    n0 = _write_view_npz(tmp_path / "view_000.npz", n_pts=16, valid_frac=1.0, conf=np.full((4, 4), 0.9, dtype=np.float32))
    n1 = _write_view_npz(tmp_path / "view_001.npz", n_pts=16, valid_frac=0.5)  # no conf

    _pts, _cols, view_idx, _cams, conf = load_run(tmp_path)

    assert conf is not None
    assert len(conf) == n0 + n1
    assert not np.isnan(conf[view_idx == 0]).any()
    assert np.isnan(conf[view_idx == 1]).all()
