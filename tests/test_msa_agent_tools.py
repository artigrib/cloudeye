"""Stage C tool-interface tests (SceneState.apply cumulative bounds,
get_scene_state/score/render_from_camera - SPEC.md §6)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.msa.agent_loop.actions import ActionRejected, Flag, Rotate, Scale, SetExposure, SetLight, SetMaterial, SwapAsset
from scripts.msa.agent_loop.objective import InvariantViolation
from scripts.msa.agent_loop.tools import SceneState
from scripts.msa.geometry import WallPolygon


def _small_object(obj_id="bed_1", label="bed", size_uv=(1.0, 1.0)):
    return {"id": obj_id, "label": label, "size_uv": size_uv}


def _walls():
    return [WallPolygon(vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], area_m2=1.2)]


def test_from_bootstrap_builds_baseline():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    assert state.baseline_object_ids == frozenset({"bed_1"})
    assert "bed_1" in state.objects
    assert state.objects["bed_1"].cumulative_yaw_deg == 0.0


def test_apply_rotate_accumulates():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    state.apply(Rotate(object_id="bed_1", yaw_delta_deg=10.0))
    state.apply(Rotate(object_id="bed_1", yaw_delta_deg=10.0))
    assert state.objects["bed_1"].cumulative_yaw_deg == pytest.approx(20.0)


def test_apply_rotate_cumulative_over_45_rejected():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    state.apply(Rotate(object_id="bed_1", yaw_delta_deg=15.0))
    state.apply(Rotate(object_id="bed_1", yaw_delta_deg=15.0))
    state.apply(Rotate(object_id="bed_1", yaw_delta_deg=15.0))  # cumulative 45, exactly at bound - OK
    assert state.objects["bed_1"].cumulative_yaw_deg == pytest.approx(45.0)
    with pytest.raises(ActionRejected):
        state.apply(Rotate(object_id="bed_1", yaw_delta_deg=1.0))  # would push to 46 > 45
    # rejection must not mutate state
    assert state.objects["bed_1"].cumulative_yaw_deg == pytest.approx(45.0)


def test_apply_scale_within_footprint_delta_accepted():
    # 1m object: cumulative factor 1.05 -> edge moves by 1.0*0.05/2 = 2.5cm <= 5cm
    state = SceneState.from_bootstrap(_walls(), [_small_object(size_uv=(1.0, 1.0))])
    state.apply(Scale(object_id="bed_1", factor=1.05))
    assert state.objects["bed_1"].cumulative_scale_factor == pytest.approx(1.05)


def test_apply_scale_exceeding_footprint_delta_rejected():
    # 3m object: cumulative factor 1.05 -> edge moves by 3.0*0.05/2 = 7.5cm > 5cm
    state = SceneState.from_bootstrap(_walls(), [_small_object(size_uv=(3.0, 3.0))])
    with pytest.raises(ActionRejected):
        state.apply(Scale(object_id="bed_1", factor=1.05))
    assert state.objects["bed_1"].cumulative_scale_factor == pytest.approx(1.0)  # untouched


def test_apply_on_unknown_object_rejected():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    with pytest.raises(ActionRejected):
        state.apply(Rotate(object_id="does_not_exist", yaw_delta_deg=5.0))


def test_apply_swap_asset_set_material_flag():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    state.apply(SwapAsset(object_id="bed_1", candidate_id="trellis_v2"))
    state.apply(SetMaterial(object_id="bed_1", albedo=(0.4, 0.3, 0.2), roughness=0.7))
    state.apply(Flag(object_id="bed_1", reason="check headboard orientation"))
    obj = state.objects["bed_1"]
    assert obj.asset_candidate_id == "trellis_v2"
    assert obj.albedo == (0.4, 0.3, 0.2)
    assert obj.roughness == pytest.approx(0.7)
    assert obj.flags == ["check headboard orientation"]


def test_apply_set_light_and_exposure():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    state.apply(SetLight(light_id="key", intensity=750.0))
    state.apply(SetExposure(value=1.3))
    assert state.lights["key"]["intensity"] == pytest.approx(750.0)
    assert state.exposure == pytest.approx(1.3)


def test_get_scene_state_is_json_serializable():
    import json

    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    state.apply(Rotate(object_id="bed_1", yaw_delta_deg=5.0))
    snapshot = state.get_scene_state()
    json.dumps(snapshot)  # must not raise
    assert snapshot["objects"]["bed_1"]["cumulative_yaw_deg"] == pytest.approx(5.0)


def test_check_invariants_no_change_is_clean():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    violations = state.check_invariants(_walls(), frozenset({"bed_1"}))
    assert violations == []


def test_check_invariants_walls_changed_flagged():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    mutated_walls = [WallPolygon(vertices=[(0.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)], area_m2=1.5)]
    violations = state.check_invariants(mutated_walls, frozenset({"bed_1"}))
    assert any("walls" in v.reason for v in violations)


def test_check_invariants_object_set_changed_flagged():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    violations = state.check_invariants(_walls(), frozenset({"bed_1", "extra_obj"}))
    assert any("object set" in v.reason for v in violations)


def test_score_assembles_breakdown():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    breakdown = state.score(mask_iou=0.1, point_fit_cm=2.0, photo=0.05)
    assert breakdown.mask_iou_loss == pytest.approx(0.1)
    assert breakdown.invariant_penalty == 0.0
    assert breakdown.total < float("inf")


def test_score_with_violations_is_inf():
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    breakdown = state.score(mask_iou=0.0, point_fit_cm=0.0, photo=0.0, extra_violations=[InvariantViolation("walls changed")])
    assert breakdown.total == float("inf")


def test_render_from_camera_top_down_writes_png(tmp_path):
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    objects = [{"id": "bed_1", "label": "bed", "center_xy": (2.0, 1.5), "size_uv": (1.0, 1.0), "angle_rad": 0.0}]
    out = tmp_path / "top_down.png"
    state.render_from_camera(0, out, walls=_walls(), objects=objects, gaps=[])
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_from_camera_non_top_down_view_not_implemented(tmp_path):
    state = SceneState.from_bootstrap(_walls(), [_small_object()])
    with pytest.raises(NotImplementedError):
        state.render_from_camera(1, tmp_path / "eye_level.png", walls=_walls(), objects=[], gaps=[])
