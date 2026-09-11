"""Stage C Verifier objective (SPEC.md §6, "Objective (lower is better)").
Computed by tool code, never by an LLM - each term below is a pure function
over arrays/meshes the caller supplies; none of them call out to a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INVARIANT_PENALTY = float("inf")


def mask_iou_loss(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """1 - IoU between a rendered object silhouette (pred_mask) and its SAM3
    ground-truth mask (gt_mask), both boolean (or 0/1) arrays of the same shape.
    Both-empty is defined as 0.0 loss (no disagreement), not NaN/1.0."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"mask shape mismatch: {pred_mask.shape} vs {gt_mask.shape}")
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(pred, gt).sum())
    return 1.0 - intersection / union


def point_fit_cm(points_xyz: np.ndarray, mesh) -> float:
    """Mean distance (cm) from measured points to the nearest surface of a mesh.
    `mesh` is duck-typed (needs `.nearest.on_surface(points) -> (closest, dist,
    tri_id)`, trimesh.Trimesh's own interface) so tests can pass a lightweight
    stand-in instead of a real Stage B mesh."""
    points_xyz = np.asarray(points_xyz, dtype=float)
    if len(points_xyz) == 0:
        return 0.0
    _, distances, _ = mesh.nearest.on_surface(points_xyz)
    return float(np.mean(distances)) * 100.0  # metres -> cm


def photo_loss(keyframe_rgb: np.ndarray, render_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Masked photometric discrepancy between a real keyframe crop and a render
    from the same camera pose (SPEC §6: "masked LPIPS/SSIM between keyframe and
    render").

    LIGHTWEIGHT STAND-IN, not real LPIPS/SSIM: this VPS has no torch/lpips/
    scikit-image install (CPU-only, no GPU budget for Stage C0 - the same
    constraint already documented for Stage A0's renderer in
    docs/DECISIONS.md). Real LPIPS needs a torch forward pass (GPU or slow CPU);
    scikit-image's SSIM needs an install this environment doesn't have. This
    implements a masked normalized-RMSE proxy instead: correlates with
    photometric discrepancy for coarse material/lighting comparisons, but is
    NOT numerically interchangeable with LPIPS - Stage C1/C2 should swap this
    for the real thing on the GPU instance (which has more headroom for a torch
    install) if the bake-off (C1) finds this proxy too coarse.
    """
    keyframe_rgb = np.asarray(keyframe_rgb)
    render_rgb = np.asarray(render_rgb)
    mask = np.asarray(mask)
    if keyframe_rgb.shape != render_rgb.shape:
        raise ValueError(f"image shape mismatch: {keyframe_rgb.shape} vs {render_rgb.shape}")
    if mask.sum() == 0:
        return 0.0
    kf = keyframe_rgb.astype(np.float64) / 255.0
    rn = render_rgb.astype(np.float64) / 255.0
    diff_sq = (kf - rn) ** 2
    if mask.ndim == 2 and diff_sq.ndim == 3:
        mask_bool = np.repeat(mask.astype(bool)[:, :, None], diff_sq.shape[2], axis=2)
    else:
        mask_bool = mask.astype(bool)
    return float(np.sqrt(np.mean(diff_sq[mask_bool])))


@dataclass(frozen=True)
class InvariantViolation:
    reason: str


def invariant_penalty(violations: list[InvariantViolation]) -> float:
    """SPEC §6: invariant_penalty = inf if ANY hard constraint is violated, else
    0. `violations` is produced by the caller (tools.SceneState, which knows
    the s0 baseline to diff against) - kept as a pure function so the
    "any violation -> inf" rule is trivially unit-testable on its own."""
    return INVARIANT_PENALTY if violations else 0.0


@dataclass(frozen=True)
class ObjectiveBreakdown:
    mask_iou_loss: float
    point_fit_cm: float
    photo_loss: float
    invariant_penalty: float

    @property
    def total(self) -> float:
        """Naive unweighted sum (point_fit_cm normalized to metres to keep unit
        scale roughly comparable to the other [0,1]-ish terms). SPEC §6 doesn't
        specify term weights - real weighting is an empirical call for the C1
        Generator/Verifier bake-off, not this module. `invariant_penalty`
        dominates (inf) whenever any hard constraint is violated, by design."""
        return self.mask_iou_loss + self.point_fit_cm / 100.0 + self.photo_loss + self.invariant_penalty
