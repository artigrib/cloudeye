"""Stage D0 - MSA validation gate (SPEC.md §7). Decides whether an agent-loop
refined scene `sn` (Stage C) may replace the deterministic bootstrap `s0`
(Stage A0). Reuses `app.services.scene_validator`'s `ValidationCheck`/
`ValidationReport` pattern rather than inventing a parallel one.

Five checks, in SPEC §7 order:
1. Passage widths (gaps.json) unchanged within +-2.5cm, recomputed from the
   final collision geometry (not just diffing two static JSON files).
2. Every object's final footprint inside its measured (s0) footprint dilated
   by 5cm; height within +-5cm.
3. Walls identical to s0 (hash) - reuses `agent_loop.tools._walls_hash`.
4. USD/Isaac physical checks (collider coverage, sphere-drop settle, robot
   proxy stop point) - STUBBED here: these need a live Isaac Sim run on the
   GPU instance, which this check does not have access to. Always non-fatal,
   always reports `requires_isaac=True` rather than inventing numbers.
5. Reprojection: mean mask IoU over the scene must not regress (sn >= s0).

If sn fails any FATAL check, the caller should keep s0 (SPEC §7: "accept sn or
fall back to s0" - this module only judges, the fallback decision itself
belongs to the Stage C/D orchestration, not implemented yet since Stage C2
(the full loop on a real scene) is deferred - see
var/scratch/run-20260906/QUESTIONS.md #4).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from app.services.scene_validator import ValidationCheck, ValidationReport
from scripts.msa.agent_loop.objective import mask_iou_loss
from scripts.msa.agent_loop.tools import MutableObjectState, _walls_hash
from scripts.msa.bootstrap import rasterize_object_footprint
from scripts.msa.gaps import Gap, ObstacleSource, compute_gaps, gaps_to_json
from scripts.msa.geometry import OBSTACLE

PASSAGE_TOLERANCE_M = 0.025
FOOTPRINT_DILATION_M = 0.05
HEIGHT_TOLERANCE_M = 0.05
MIN_WALL_COMPONENT_AREA_M2 = 0.3  # mirrors bootstrap.py's SPEC A1 threshold

# SPEC §5 Stage B fallback rule (not yet exercised by any Stage B run in this
# sprint - B0 is blocked on a gated HF dependency, see docs/MODELS.md - but the
# rule itself has no other home in the codebase yet, so it lives here pending
# Stage B1 actually generating meshes to apply it to).
MESH_VOLUME_MIN_FRACTION = 0.20
MESH_VOLUME_MAX_FRACTION = 1.50


# --------------------------------------------------------------------------
# 3. Walls identical to s0
# --------------------------------------------------------------------------


def check_walls_identical(s0_walls, sn_walls) -> ValidationCheck:
    s0_hash = _walls_hash(s0_walls)
    sn_hash = _walls_hash(sn_walls)
    passed = s0_hash == sn_hash
    return ValidationCheck(
        "msa_walls_identical",
        passed=passed,
        fatal=True,
        detail="walls hash unchanged from s0" if passed else f"walls hash changed: s0={s0_hash[:12]}.. sn={sn_hash[:12]}..",
        values={},
    )


# --------------------------------------------------------------------------
# 2. Footprint containment + height
# --------------------------------------------------------------------------


def apply_object_state_to_footprint(obj: dict, obj_state: MutableObjectState) -> dict:
    """Returns a new s0-shaped object-footprint dict (same keys `bootstrap.
    compute_object_footprints` produces) with `obj_state`'s cumulative
    yaw/scale applied on top of s0's baseline. Only `size_uv` and `angle_rad`
    move - `center_xy` is untouched because SPEC §6's whitelist has no
    translate action at all (Stage C cannot move an object's center)."""
    length_u, width_v = obj["size_uv"]
    factor = obj_state.cumulative_scale_factor
    new = dict(obj)
    new["size_uv"] = (length_u * factor, width_v * factor)
    new["angle_rad"] = obj["angle_rad"] + math.radians(obj_state.cumulative_yaw_deg)
    return new


def _footprint_polygon(obj: dict):
    from shapely.affinity import rotate as shapely_rotate
    from shapely.affinity import translate
    from shapely.geometry import Polygon

    length_u, width_v = obj["size_uv"]
    cx, cz = obj["center_xy"]
    rect = Polygon(
        [
            (-length_u / 2, -width_v / 2),
            (length_u / 2, -width_v / 2),
            (length_u / 2, width_v / 2),
            (-length_u / 2, width_v / 2),
        ]
    )
    rect = shapely_rotate(rect, math.degrees(obj["angle_rad"]), origin=(0, 0), use_radians=False)
    return translate(rect, cx, cz)


def check_object_footprint_containment(s0_objects: list[dict], sn_objects: list[dict]) -> list[ValidationCheck]:
    """One check per s0 object: is sn's footprint (accounting for whatever
    rotate/scale Stage C applied) fully inside s0's measured footprint dilated
    by 5cm, and is the height delta within 5cm. A rotation alone can swing a
    rectangle's corners outside an axis-unaware dilation even when each
    individual action stayed within `tools.py`'s own per-step/cumulative
    bounds - this is deliberate defense-in-depth, not a duplicate check."""
    checks = []
    sn_by_id = {o["id"]: o for o in sn_objects}
    for s0_obj in s0_objects:
        oid = s0_obj["id"]
        sn_obj = sn_by_id.get(oid)
        if sn_obj is None:
            checks.append(
                ValidationCheck(
                    f"msa_footprint_containment:{oid}",
                    passed=False,
                    fatal=True,
                    detail=f"object {oid} missing from sn (add/delete is forbidden by SPEC §6)",
                    values={},
                )
            )
            continue
        s0_dilated = _footprint_polygon(s0_obj).buffer(FOOTPRINT_DILATION_M)
        sn_poly = _footprint_polygon(sn_obj)
        contained = s0_dilated.contains(sn_poly) or s0_dilated.buffer(1e-9).contains(sn_poly)
        height_delta_m = abs(sn_obj.get("height", s0_obj["height"]) - s0_obj["height"])
        height_ok = height_delta_m <= HEIGHT_TOLERANCE_M
        passed = contained and height_ok
        checks.append(
            ValidationCheck(
                f"msa_footprint_containment:{oid}",
                passed=passed,
                fatal=True,
                detail=(
                    f"footprint {'inside' if contained else 'OUTSIDE'} s0+{FOOTPRINT_DILATION_M * 100:.0f}cm; "
                    f"height delta {height_delta_m * 100:.1f}cm ({'ok' if height_ok else 'EXCEEDS'} "
                    f"{HEIGHT_TOLERANCE_M * 100:.0f}cm)"
                ),
                values={"height_delta_m": height_delta_m, "contained": float(contained)},
            )
        )
    return checks


# --------------------------------------------------------------------------
# 1. Passage widths - recomputed from the final collision geometry
# --------------------------------------------------------------------------


def _wall_obstacle_sources(occupancy: np.ndarray, resolution: float) -> list[ObstacleSource]:
    """Re-derives wall obstacle masks from the raw occupancy grid, the same
    way `bootstrap.run_bootstrap` does (it doesn't persist these masks in its
    output, only the simplified polygons + gaps.json) - safe to redo here
    because Stage C's whitelist forbids editing walls at all, so s0's and
    sn's wall obstacle masks are always identical by construction; this
    function only needs to run once per validation, not once per candidate."""
    labeled, n_components = ndimage.label(occupancy == OBSTACLE, structure=np.ones((3, 3)))
    sources = []
    comp_idx = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        area = comp_mask.sum() * resolution * resolution
        if area < MIN_WALL_COMPONENT_AREA_M2:
            continue
        sources.append(ObstacleSource(id=f"wall_{comp_idx}", kind="wall", mask=comp_mask))
        comp_idx += 1
    return sources


def recompute_gaps_for_objects(
    occupancy: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_z: float,
    objects: list[dict],
    platforms: dict[str, float],
) -> list[Gap]:
    """SPEC §7: "recomputed from the final collision geometry" - reruns the
    same Stage A5 gap analysis (`gaps.compute_gaps`) against `objects`'
    footprints (s0's own, or s0's with Stage C's deltas applied via
    `apply_object_state_to_footprint`), not against a stale gaps.json."""
    passable = occupancy != OBSTACLE
    sources = list(_wall_obstacle_sources(occupancy, resolution))
    for obj in objects:
        mask = rasterize_object_footprint(obj, occupancy.shape, resolution, origin_x, origin_z)
        sources.append(ObstacleSource(id=obj["id"], kind="object", mask=mask))
    return compute_gaps(passable, sources, resolution, origin_x, origin_z, platforms=platforms)


def check_passage_widths(s0_gaps: list[Gap], sn_gaps: list[Gap]) -> ValidationCheck:
    """Matches gaps by their unordered (a_id, b_id) pair (object ids are
    invariant under Stage C - add/delete is forbidden - and wall ids are
    invariant too since walls can't be edited, so pairing by id is exact, not
    a nearest-match heuristic)."""

    def _key(g: Gap) -> frozenset[str]:
        return frozenset((g.a_id, g.b_id))

    s0_by_key = {_key(g): g for g in s0_gaps}
    sn_by_key = {_key(g): g for g in sn_gaps}

    missing = sorted(str(k) for k in s0_by_key if k not in sn_by_key)
    extra = sorted(str(k) for k in sn_by_key if k not in s0_by_key)
    deltas = {k: abs(sn_by_key[k].width_m - g.width_m) for k, g in s0_by_key.items() if k in sn_by_key}
    worst_key = max(deltas, key=deltas.get) if deltas else None
    worst_delta = deltas[worst_key] if worst_key is not None else 0.0

    passed = not missing and not extra and worst_delta <= PASSAGE_TOLERANCE_M
    if missing or extra:
        detail = f"gap set changed: {len(missing)} missing, {len(extra)} extra (obstacle topology changed)"
    else:
        worst_pair = ", ".join(sorted(worst_key)) if worst_key else "n/a"
        detail = (
            f"max passage-width delta {worst_delta * 100:.2f}cm ({worst_pair}) "
            f"vs {PASSAGE_TOLERANCE_M * 100:.1f}cm tolerance"
        )
    return ValidationCheck(
        "msa_passage_widths",
        passed=passed,
        fatal=True,
        detail=detail,
        values={"max_delta_m": worst_delta, "n_missing": len(missing), "n_extra": len(extra)},
    )


# --------------------------------------------------------------------------
# 4. USD/Isaac physical checks - stubbed, GPU-only
# --------------------------------------------------------------------------


def check_usd_isaac_physical() -> ValidationCheck:
    """SPEC §7: collider coverage 100%, floor/object footprint overlap,
    sphere-drop settle, robot-proxy stop point within 10cm of s0. All four
    need a live Isaac Sim run (`tools/isaac_validate.py` on the GPU instance)
    that this CPU-only validator does not have access to. Deliberately
    non-fatal with an explicit `requires_isaac=True` flag rather than
    inventing pass/fail numbers - a caller with real Isaac output should
    replace this check's result, not trust this stub as a real pass."""
    return ValidationCheck(
        "msa_usd_isaac_physical",
        passed=True,
        fatal=False,
        detail="not evaluated - requires a live Isaac Sim run on the GPU instance (see tools/isaac_validate.py)",
        values={"requires_isaac": 1.0},
    )


# --------------------------------------------------------------------------
# 5. Reprojection - mean mask IoU must not regress
# --------------------------------------------------------------------------


def check_reprojection_iou(s0_masks: dict[str, tuple[np.ndarray, np.ndarray]], sn_masks: dict[str, tuple[np.ndarray, np.ndarray]]) -> ValidationCheck:
    """`s0_masks`/`sn_masks`: object_id -> (rendered_silhouette, sam3_gt_mask).
    Mean IoU (1 - mask_iou_loss) across objects present in both must not
    regress. Objects missing a render in either (Stage C's render_from_camera
    only covers view_idx=0 - SPEC A7/§6's known limitation) are skipped, not
    counted as a failure - this check judges silhouette quality where it CAN
    be measured, it doesn't invent a verdict for what it can't render."""
    common = set(s0_masks) & set(sn_masks)
    if not common:
        return ValidationCheck(
            "msa_reprojection_iou",
            passed=True,
            fatal=False,
            detail="no objects with renders in both s0 and sn, skipped",
            values={},
        )
    s0_iou = [1.0 - mask_iou_loss(*s0_masks[oid]) for oid in common]
    sn_iou = [1.0 - mask_iou_loss(*sn_masks[oid]) for oid in common]
    s0_mean, sn_mean = float(np.mean(s0_iou)), float(np.mean(sn_iou))
    passed = sn_mean >= s0_mean - 1e-9
    return ValidationCheck(
        "msa_reprojection_iou",
        passed=passed,
        fatal=True,
        detail=f"mean mask IoU s0={s0_mean:.4f} -> sn={sn_mean:.4f} ({'ok' if passed else 'REGRESSED'})",
        values={"s0_mean_iou": s0_mean, "sn_mean_iou": sn_mean},
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def validate_msa_transition(
    *,
    s0_walls,
    sn_walls,
    s0_objects: list[dict],
    sn_objects: list[dict],
    s0_gaps: list[Gap],
    sn_gaps: list[Gap],
    s0_masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    sn_masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> ValidationReport:
    """Runs all five SPEC §7 checks and returns one report. `sn == s0` (no
    Stage C edits applied) trivially passes every check - this is
    deliberate, so the gate is safe to run even when Stage C2 hasn't
    (yet) produced a real sn (see docs/DECISIONS.md 2026-09-06 Stage C1
    entry - Stage C2 is currently deferred)."""
    checks: list[ValidationCheck] = [
        check_passage_widths(s0_gaps, sn_gaps),
        *check_object_footprint_containment(s0_objects, sn_objects),
        check_walls_identical(s0_walls, sn_walls),
        check_usd_isaac_physical(),
        check_reprojection_iou(s0_masks or {}, sn_masks or {}),
    ]
    return ValidationReport(checks=checks)


# --------------------------------------------------------------------------
# SPEC §5 Stage B fallback rule (adversarial test target - see module docstring)
# --------------------------------------------------------------------------


def mesh_passes_containment_test(mesh_volume_m3: float, hull_bbox_volume_m3: float) -> bool:
    """SPEC §5: "Reject and fall back to primitive if: mesh volume < 20% or >
    150% of hull bbox volume". Pure function so it's unit-testable without a
    real Stage B mesh; `hull_bbox_volume_m3` must be > 0 (a zero-volume hull
    bbox means the caller has a bad measurement, not a bad mesh - that's a
    different failure this function doesn't try to classify)."""
    if hull_bbox_volume_m3 <= 0:
        raise ValueError("hull_bbox_volume_m3 must be > 0")
    ratio = mesh_volume_m3 / hull_bbox_volume_m3
    return MESH_VOLUME_MIN_FRACTION <= ratio <= MESH_VOLUME_MAX_FRACTION


def is_mesh_non_manifold_beyond_repair(mesh) -> bool:
    """Duck-typed on trimesh.Trimesh's own `.is_watertight`/`.fill_holes()`
    interface, mirroring SPEC §5's "non-manifold beyond repair" fallback
    trigger: attempt trimesh's own hole-filling repair, then check whether
    the result is watertight. Returns True (fallback to primitive) if repair
    didn't fix it."""
    if mesh.is_watertight:
        return False
    repaired = mesh.copy()
    try:
        repaired.fill_holes()
    except Exception:
        return True
    return not repaired.is_watertight
