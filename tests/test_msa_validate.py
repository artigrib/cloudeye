"""Stage D0 validator tests (SPEC.md §7, §11). Unit tests on synthetic data for
each of the 5 checks, one integration test cross-checking `recompute_gaps_for_objects`
against the 02_modular_home golden fixture, and adversarial tests for the SPEC §5
mesh-fallback rule this module also hosts (see validate_msa.py's module docstring)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.msa.agent_loop.tools import MutableObjectState
from scripts.msa.bootstrap import compute_object_footprints, load_object_inputs_from_hulls_json
from scripts.msa.gaps import Gap, PlatformVerdict
from scripts.msa.geometry import extract_wall_polygons
from scripts.msa.validate_msa import (
    apply_object_state_to_footprint,
    check_object_footprint_containment,
    check_passage_widths,
    check_reprojection_iou,
    check_usd_isaac_physical,
    check_walls_identical,
    is_mesh_non_manifold_beyond_repair,
    mesh_passes_containment_test,
    recompute_gaps_for_objects,
    validate_msa_transition,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "msa" / "02_modular_home"


def _make_gap(a, b, width_m, a_kind="object", b_kind="object") -> Gap:
    return Gap(a_id=a, b_id=b, a_kind=a_kind, b_kind=b_kind, width_m=width_m, measurement_point_xy=(0.0, 0.0), platform_verdicts=[PlatformVerdict("burger", width_m > 0.2, 0.2)])


def _simple_object(oid="obj_0", cx=0.0, cz=0.0, length=1.0, width=0.5, height=0.5, angle=0.0) -> dict:
    return {
        "id": oid,
        "label": "table",
        "center_xy": (cx, cz),
        "size_uv": (length, width),
        "angle_rad": angle,
        "height": height,
        "bbox_min_y": 0.0,
        "color_rgb": (128, 128, 128),
    }


class FakeWall:
    """Minimal stand-in matching `_walls_hash`'s only requirement: a
    `.vertices` attribute of (x, z) tuples."""

    def __init__(self, vertices):
        self.vertices = vertices


# --------------------------------------------------------------------------
# 3. Walls identical
# --------------------------------------------------------------------------


class TestWallsIdentical:
    def test_identical_walls_pass(self):
        walls = [FakeWall([(0, 0), (1, 0), (1, 1), (0, 1)])]
        check = check_walls_identical(walls, walls)
        assert check.passed
        assert check.fatal

    def test_moved_wall_fails(self):
        s0 = [FakeWall([(0, 0), (1, 0), (1, 1), (0, 1)])]
        sn = [FakeWall([(0, 0), (1.1, 0), (1.1, 1), (0, 1)])]
        check = check_walls_identical(s0, sn)
        assert not check.passed


# --------------------------------------------------------------------------
# 2. Footprint containment + height
# --------------------------------------------------------------------------


class TestFootprintContainment:
    def test_unchanged_object_passes(self):
        s0 = [_simple_object()]
        sn = [_simple_object()]
        checks = check_object_footprint_containment(s0, sn)
        assert len(checks) == 1
        assert checks[0].passed

    def test_small_scale_within_5cm_passes(self):
        obj_state = MutableObjectState(id="obj_0", label="table", size_uv=(1.0, 0.5), cumulative_scale_factor=1.02)
        s0 = [_simple_object(length=1.0, width=0.5)]
        sn_obj = apply_object_state_to_footprint(s0[0], obj_state)
        checks = check_object_footprint_containment(s0, [sn_obj])
        assert checks[0].passed, checks[0].detail

    def test_large_rotation_pushes_corners_outside_dilation(self):
        # A long thin object (2m x 0.2m) rotated 45deg swings its far corners
        # well outside a 5cm dilation of the original axis-aligned rect -
        # this is exactly the "each per-step bound was satisfied but the
        # actual footprint moved too far" case the module docstring calls out.
        obj_state = MutableObjectState(id="obj_0", label="shelf", size_uv=(2.0, 0.2), cumulative_yaw_deg=45.0)
        s0 = [_simple_object(length=2.0, width=0.2)]
        sn_obj = apply_object_state_to_footprint(s0[0], obj_state)
        checks = check_object_footprint_containment(s0, [sn_obj])
        assert not checks[0].passed
        assert "OUTSIDE" in checks[0].detail

    def test_height_delta_beyond_tolerance_fails(self):
        s0 = [_simple_object(height=0.5)]
        sn = [_simple_object(height=0.58)]  # 8cm > 5cm tolerance
        checks = check_object_footprint_containment(s0, sn)
        assert not checks[0].passed
        assert "height" in checks[0].detail

    def test_missing_object_is_fatal(self):
        s0 = [_simple_object(oid="obj_0")]
        checks = check_object_footprint_containment(s0, [])
        assert not checks[0].passed
        assert checks[0].fatal


# --------------------------------------------------------------------------
# 1. Passage widths
# --------------------------------------------------------------------------


class TestPassageWidths:
    def test_unchanged_gaps_pass(self):
        gaps = [_make_gap("a", "b", 0.40), _make_gap("a", "wall_0", 0.60, b_kind="wall")]
        check = check_passage_widths(gaps, gaps)
        assert check.passed

    def test_small_delta_within_tolerance_passes(self):
        s0 = [_make_gap("a", "b", 0.400)]
        sn = [_make_gap("a", "b", 0.415)]  # 1.5cm delta
        check = check_passage_widths(s0, sn)
        assert check.passed

    def test_delta_over_2_5cm_fails(self):
        s0 = [_make_gap("a", "b", 0.400)]
        sn = [_make_gap("a", "b", 0.440)]  # 4cm delta
        check = check_passage_widths(s0, sn)
        assert not check.passed
        assert check.fatal

    def test_missing_gap_fails(self):
        s0 = [_make_gap("a", "b", 0.400), _make_gap("a", "c", 0.500)]
        sn = [_make_gap("a", "b", 0.400)]
        check = check_passage_widths(s0, sn)
        assert not check.passed
        assert "missing" in check.detail


# --------------------------------------------------------------------------
# 4. USD/Isaac stub
# --------------------------------------------------------------------------


def test_usd_isaac_physical_is_nonfatal_stub():
    check = check_usd_isaac_physical()
    assert check.passed
    assert not check.fatal
    assert check.values["requires_isaac"] == 1.0


# --------------------------------------------------------------------------
# 5. Reprojection IoU
# --------------------------------------------------------------------------


class TestReprojectionIou:
    def test_identical_masks_pass(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        s0 = {"obj_0": (mask, mask)}
        sn = {"obj_0": (mask, mask)}
        check = check_reprojection_iou(s0, sn)
        assert check.passed

    def test_regressed_iou_fails(self):
        gt = np.zeros((10, 10), dtype=bool)
        gt[2:8, 2:8] = True
        good_pred = gt.copy()
        bad_pred = np.zeros((10, 10), dtype=bool)
        bad_pred[2:4, 2:4] = True  # much smaller overlap
        s0 = {"obj_0": (good_pred, gt)}
        sn = {"obj_0": (bad_pred, gt)}
        check = check_reprojection_iou(s0, sn)
        assert not check.passed
        assert "REGRESSED" in check.detail

    def test_no_common_objects_skips_non_fatally(self):
        mask = np.ones((4, 4), dtype=bool)
        check = check_reprojection_iou({"a": (mask, mask)}, {"b": (mask, mask)})
        assert check.passed
        assert not check.fatal


# --------------------------------------------------------------------------
# Orchestration: sn == s0 trivially passes everything
# --------------------------------------------------------------------------


def test_identity_transition_passes_all_fatal_checks():
    walls = [FakeWall([(0, 0), (1, 0), (1, 1), (0, 1)])]
    objects = [_simple_object()]
    gaps = [_make_gap("obj_0", "wall_0", 0.5, b_kind="wall")]
    report = validate_msa_transition(
        s0_walls=walls, sn_walls=walls,
        s0_objects=objects, sn_objects=objects,
        s0_gaps=gaps, sn_gaps=gaps,
    )
    assert report.passed, report.summary()


# --------------------------------------------------------------------------
# Integration: recompute_gaps_for_objects vs golden fixture (identity case)
# --------------------------------------------------------------------------


def test_recompute_gaps_matches_golden_for_unchanged_objects():
    """Sanity check that this module's own gap-recompute path (used for the
    "recomputed from the final collision geometry" requirement) agrees with
    bootstrap.py's own gap computation when nothing changed - if these drift
    apart, Stage D0 would reject every honestly-unchanged scene."""
    occupancy = np.load(FIXTURE_DIR / "occupancy.npy")
    meta = json.loads((FIXTURE_DIR / "occupancy_meta.json").read_text())
    scene_meta = json.loads((FIXTURE_DIR / "scene_meta.json").read_text())
    inputs = load_object_inputs_from_hulls_json(FIXTURE_DIR / "scene_objects_hulls.json")
    objects, _overhead = compute_object_footprints(inputs, scene_meta["floor_y"])

    platforms = {"burger": 0.20, "go2": 0.40}
    gaps = recompute_gaps_for_objects(
        occupancy, meta["resolution"], meta["origin_x"], meta["origin_z"], objects, platforms
    )
    current = {frozenset((g.a_id, g.b_id)): g for g in gaps}

    golden = json.loads((FIXTURE_DIR / "golden_gaps.json").read_text())
    golden_by_pair = {frozenset((g["a"], g["b"])): g for g in golden}

    assert set(current) == set(golden_by_pair)
    for pair, golden_gap in golden_by_pair.items():
        assert current[pair].width_m == pytest.approx(golden_gap["width_m"], abs=0.01)


# --------------------------------------------------------------------------
# Adversarial: SPEC §5 mesh fallback rule
# --------------------------------------------------------------------------


class TestMeshContainmentFallback:
    def test_mesh_within_bounds_passes(self):
        assert mesh_passes_containment_test(mesh_volume_m3=0.5, hull_bbox_volume_m3=1.0)

    def test_mesh_too_small_falls_back(self):
        assert not mesh_passes_containment_test(mesh_volume_m3=0.1, hull_bbox_volume_m3=1.0)

    def test_mesh_too_large_falls_back(self):
        assert not mesh_passes_containment_test(mesh_volume_m3=2.0, hull_bbox_volume_m3=1.0)

    def test_boundary_values_are_inclusive(self):
        assert mesh_passes_containment_test(mesh_volume_m3=0.20, hull_bbox_volume_m3=1.0)
        assert mesh_passes_containment_test(mesh_volume_m3=1.50, hull_bbox_volume_m3=1.0)

    def test_zero_hull_volume_raises(self):
        with pytest.raises(ValueError):
            mesh_passes_containment_test(mesh_volume_m3=0.5, hull_bbox_volume_m3=0.0)

    def test_watertight_box_is_not_non_manifold(self):
        box = trimesh.creation.box(extents=(1, 1, 1))
        assert not is_mesh_non_manifold_beyond_repair(box)

    def test_single_triangle_is_non_manifold_beyond_repair(self):
        broken = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        assert is_mesh_non_manifold_beyond_repair(broken)
