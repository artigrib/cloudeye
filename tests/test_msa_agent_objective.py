"""Stage C objective tests on synthetic data (SPEC.md §11: "objective
computation on synthetic renders/masks")."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from scripts.msa.agent_loop.objective import (
    InvariantViolation,
    ObjectiveBreakdown,
    invariant_penalty,
    mask_iou_loss,
    photo_loss,
    point_fit_cm,
)


def test_mask_iou_loss_perfect_overlap_is_zero():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    assert mask_iou_loss(mask, mask) == pytest.approx(0.0)


def test_mask_iou_loss_no_overlap_is_one():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert mask_iou_loss(a, b) == pytest.approx(1.0)


def test_mask_iou_loss_known_partial_overlap():
    # a: 4x4=16 cells, b: 4x4=16 cells, overlap 2x4=8 cells -> union=16+16-8=24, iou=8/24=1/3
    a = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:4] = True
    b = np.zeros((10, 10), dtype=bool)
    b[0:4, 2:6] = True
    loss = mask_iou_loss(a, b)
    assert loss == pytest.approx(1.0 - 1.0 / 3.0, abs=1e-9)


def test_mask_iou_loss_both_empty_is_zero():
    empty = np.zeros((5, 5), dtype=bool)
    assert mask_iou_loss(empty, empty) == pytest.approx(0.0)


def test_mask_iou_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mask_iou_loss(np.zeros((5, 5), dtype=bool), np.zeros((4, 4), dtype=bool))


def test_point_fit_cm_points_on_surface_is_zero():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # a point exactly on a face center is 0 distance from the surface
    points = np.array([[0.5, 0.0, 0.0]])
    assert point_fit_cm(points, box) == pytest.approx(0.0, abs=1e-6)


def test_point_fit_cm_known_offset_distance():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))  # centered at origin, faces at +-0.5
    # 3cm outside the +x face
    points = np.array([[0.53, 0.0, 0.0]])
    assert point_fit_cm(points, box) == pytest.approx(3.0, abs=1e-3)


def test_point_fit_cm_empty_points_is_zero():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    assert point_fit_cm(np.zeros((0, 3)), box) == pytest.approx(0.0)


def test_photo_loss_identical_images_is_zero():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    assert photo_loss(img, img, mask) == pytest.approx(0.0)


def test_photo_loss_known_constant_offset():
    kf = np.zeros((4, 4, 3), dtype=np.uint8)
    render = np.zeros((4, 4, 3), dtype=np.uint8)
    render[:] = 255  # max possible per-channel diff
    mask = np.ones((4, 4), dtype=bool)
    # normalized diff is 1.0 everywhere -> sqrt(mean(1.0^2)) == 1.0
    assert photo_loss(kf, render, mask) == pytest.approx(1.0, abs=1e-9)


def test_photo_loss_unmasked_region_ignored():
    kf = np.zeros((4, 4, 3), dtype=np.uint8)
    render = np.zeros((4, 4, 3), dtype=np.uint8)
    render[0, 0] = 255  # only in a masked-out cell
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    assert photo_loss(kf, render, mask) == pytest.approx(0.0)


def test_photo_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        photo_loss(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((5, 5, 3), dtype=np.uint8), np.ones((4, 4), dtype=bool))


def test_invariant_penalty_no_violations_is_zero():
    assert invariant_penalty([]) == 0.0


def test_invariant_penalty_any_violation_is_inf():
    assert invariant_penalty([InvariantViolation("walls changed")]) == float("inf")


def test_objective_breakdown_total_sums_terms():
    b = ObjectiveBreakdown(mask_iou_loss=0.2, point_fit_cm=10.0, photo_loss=0.1, invariant_penalty=0.0)
    assert b.total == pytest.approx(0.2 + 0.1 + 0.1)


def test_objective_breakdown_total_is_inf_when_invariant_violated():
    b = ObjectiveBreakdown(mask_iou_loss=0.0, point_fit_cm=0.0, photo_loss=0.0, invariant_penalty=float("inf"))
    assert b.total == float("inf")
