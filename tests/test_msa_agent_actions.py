"""Stage C whitelist/static-checker tests (SPEC.md §11: "action whitelist
static checker", adversarial: "запрещённое действие агента отклоняется")."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.msa.agent_loop.actions import (
    ActionRejected,
    Flag,
    Rotate,
    Scale,
    SetExposure,
    SetLight,
    SetMaterial,
    SwapAsset,
    check_action_static,
)


def test_rotate_within_per_step_bound_accepted():
    check_action_static(Rotate(object_id="bed_1", yaw_delta_deg=15.0))
    check_action_static(Rotate(object_id="bed_1", yaw_delta_deg=-15.0))
    check_action_static(Rotate(object_id="bed_1", yaw_delta_deg=0.0))


def test_rotate_exceeding_per_step_bound_rejected():
    with pytest.raises(ActionRejected):
        check_action_static(Rotate(object_id="bed_1", yaw_delta_deg=15.01))
    with pytest.raises(ActionRejected):
        check_action_static(Rotate(object_id="bed_1", yaw_delta_deg=-20.0))


def test_scale_within_per_step_bound_accepted():
    check_action_static(Scale(object_id="bed_1", factor=0.95))
    check_action_static(Scale(object_id="bed_1", factor=1.05))
    check_action_static(Scale(object_id="bed_1", factor=1.0))


def test_scale_exceeding_per_step_bound_rejected():
    with pytest.raises(ActionRejected):
        check_action_static(Scale(object_id="bed_1", factor=0.94))
    with pytest.raises(ActionRejected):
        check_action_static(Scale(object_id="bed_1", factor=1.5))


def test_swap_asset_empty_candidate_rejected():
    with pytest.raises(ActionRejected):
        check_action_static(SwapAsset(object_id="bed_1", candidate_id=""))


def test_swap_asset_valid_accepted():
    check_action_static(SwapAsset(object_id="bed_1", candidate_id="trellis_bed_v2"))


def test_set_material_out_of_range_rejected():
    with pytest.raises(ActionRejected):
        check_action_static(SetMaterial(object_id="bed_1", albedo=(0.5, 0.5, 0.5), roughness=1.5))
    with pytest.raises(ActionRejected):
        check_action_static(SetMaterial(object_id="bed_1", albedo=(1.2, 0.5, 0.5), roughness=0.5))


def test_set_material_valid_accepted():
    check_action_static(SetMaterial(object_id="bed_1", albedo=(0.5, 0.4, 0.3), roughness=0.6))


def test_set_light_negative_intensity_rejected():
    with pytest.raises(ActionRejected):
        check_action_static(SetLight(light_id="key_light", intensity=-1.0))


def test_set_light_valid_accepted():
    check_action_static(SetLight(light_id="key_light", intensity=800.0, colour=(1.0, 0.95, 0.9)))


def test_set_exposure_accepted():
    check_action_static(SetExposure(value=1.2))


def test_flag_empty_reason_rejected():
    with pytest.raises(ActionRejected):
        check_action_static(Flag(object_id="bed_1", reason=""))


def test_flag_valid_accepted():
    check_action_static(Flag(object_id="bed_1", reason="mesh looks upside down"))


def test_unauthorized_action_type_rejected():
    """Adversarial (SPEC §11): translate() is NOT in the SPEC §6 whitelist at
    all - simulate a rogue action type (as if a bug let something non-whitelisted
    through) and confirm it's rejected rather than silently handled."""

    @dataclass(frozen=True)
    class Translate:  # not part of the Action union on purpose
        object_id: str
        dx_m: float
        dz_m: float

    with pytest.raises(ActionRejected):
        check_action_static(Translate(object_id="bed_1", dx_m=0.10, dz_m=0.0))


def test_unauthorized_plain_object_rejected():
    with pytest.raises(ActionRejected):
        check_action_static("delete bed_1")  # type: ignore[arg-type]
