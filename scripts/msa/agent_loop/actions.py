"""Stage C (SPEC.md §6): the Generator's whitelisted action set + a static
checker that rejects anything else BEFORE it ever touches scene state.

Each allowed action is its own frozen dataclass. Anything not one of these
types is rejected by `check_action_static` on sight - there is no generic
"action" base class instances of a rogue type could subclass their way into;
the check is a plain isinstance() ladder over the exact SPEC §6 whitelist.

Per-step numeric bounds (rotate |delta|<=15deg, scale factor in [0.95, 1.05])
are checked here. Cumulative bounds (rotate <=45deg total, scale within
footprint +-5cm) need running per-object state and are checked by
`SceneState.apply` in tools.py, which calls this module's
`check_action_static` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


class ActionRejected(Exception):
    """Raised for any action outside the SPEC §6 whitelist, or an allowed
    action with parameters outside its per-step bound."""


@dataclass(frozen=True)
class Rotate:
    object_id: str
    yaw_delta_deg: float


@dataclass(frozen=True)
class Scale:
    object_id: str
    factor: float


@dataclass(frozen=True)
class SwapAsset:
    object_id: str
    candidate_id: str


@dataclass(frozen=True)
class SetMaterial:
    object_id: str
    albedo: tuple[float, float, float]
    roughness: float


@dataclass(frozen=True)
class SetLight:
    light_id: str
    intensity: float | None = None
    colour: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SetExposure:
    value: float


@dataclass(frozen=True)
class Flag:
    object_id: str
    reason: str


Action = Union[Rotate, Scale, SwapAsset, SetMaterial, SetLight, SetExposure, Flag]

MAX_ROTATE_STEP_DEG = 15.0
MAX_ROTATE_CUMULATIVE_DEG = 45.0
SCALE_STEP_MIN = 0.95
SCALE_STEP_MAX = 1.05
MAX_FOOTPRINT_DELTA_M = 0.05  # cumulative scale bound, checked in tools.py (needs the object's own size)


def check_action_static(action: Action) -> None:
    """Whitelist membership + per-step bound check. Raises ActionRejected;
    returns None if the action is allowed at the per-step level (cumulative
    bounds are a separate, stateful check - see tools.SceneState.apply)."""
    if isinstance(action, Rotate):
        if not (-MAX_ROTATE_STEP_DEG <= action.yaw_delta_deg <= MAX_ROTATE_STEP_DEG):
            raise ActionRejected(f"rotate: |yaw_delta|={abs(action.yaw_delta_deg)}deg exceeds per-step limit {MAX_ROTATE_STEP_DEG}deg")
    elif isinstance(action, Scale):
        if not (SCALE_STEP_MIN <= action.factor <= SCALE_STEP_MAX):
            raise ActionRejected(f"scale: factor={action.factor} outside per-step [{SCALE_STEP_MIN}, {SCALE_STEP_MAX}]")
    elif isinstance(action, SwapAsset):
        if not action.candidate_id:
            raise ActionRejected("swap_asset: empty candidate_id")
    elif isinstance(action, SetMaterial):
        if not (0.0 <= action.roughness <= 1.0):
            raise ActionRejected(f"set_material: roughness={action.roughness} outside [0,1]")
        if not all(0.0 <= c <= 1.0 for c in action.albedo):
            raise ActionRejected(f"set_material: albedo={action.albedo} components must each be in [0,1]")
    elif isinstance(action, SetLight):
        if action.intensity is not None and action.intensity < 0:
            raise ActionRejected(f"set_light: negative intensity {action.intensity}")
    elif isinstance(action, SetExposure):
        pass  # SPEC §6 doesn't bound exposure explicitly; any finite value is accepted here
    elif isinstance(action, Flag):
        if not action.reason:
            raise ActionRejected("flag: empty reason")
    else:
        raise ActionRejected(f"action type {type(action).__name__!r} is not in the SPEC §6 whitelist")
