"""Stage C tool interfaces for the Generator (SPEC.md §6): `render_from_camera`,
`get_scene_state`, `apply`, `score`. Wraps the mutable per-object deltas
(cumulative yaw/scale, material, asset-candidate, flags) that accumulate across
an agent loop's steps on top of Stage A0's frozen s0 (`scripts/msa/bootstrap.py`
output), applies SPEC §6's cumulative bounds (45deg yaw, +-5cm footprint from
scale) on top of `actions.check_action_static`'s per-step bounds, and exposes
the "Forbidden (hard)" invariant check `objective.invariant_penalty` needs.

Does NOT do the OpenRouter Generator/Verifier integration - that's Stage C1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from scripts.msa.agent_loop.actions import (
    Action,
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
from scripts.msa.agent_loop.objective import InvariantViolation, ObjectiveBreakdown, invariant_penalty

MAX_ROTATE_CUMULATIVE_DEG = 45.0
MAX_FOOTPRINT_DELTA_M = 0.05


def _walls_hash(walls) -> str:
    walls_repr = repr([tuple(w.vertices) for w in walls]).encode()
    return hashlib.sha256(walls_repr).hexdigest()


@dataclass
class MutableObjectState:
    id: str
    label: str
    size_uv: tuple[float, float]  # s0's baseline footprint (length_u, width_v), immutable reference for cumulative-scale bound
    cumulative_yaw_deg: float = 0.0
    cumulative_scale_factor: float = 1.0
    asset_candidate_id: str | None = None
    albedo: tuple[float, float, float] | None = None
    roughness: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class SceneState:
    """s0's frozen baseline (wall hash, object id set) plus the mutable
    per-object deltas an agent loop accumulates. Construct via
    `SceneState.from_bootstrap` with Stage A0's `walls`/`objects` (the same
    values `bootstrap.run_bootstrap` computes internally)."""

    baseline_object_ids: frozenset[str]
    baseline_walls_hash: str
    objects: dict[str, MutableObjectState]
    lights: dict[str, dict] = field(default_factory=dict)
    exposure: float = 1.0

    @classmethod
    def from_bootstrap(cls, walls, objects: list[dict]) -> SceneState:
        obj_states = {o["id"]: MutableObjectState(id=o["id"], label=o["label"], size_uv=tuple(o["size_uv"])) for o in objects}
        return cls(baseline_object_ids=frozenset(obj_states), baseline_walls_hash=_walls_hash(walls), objects=obj_states)

    def get_scene_state(self) -> dict:
        """SPEC §6 tool `get_scene_state()` -> JSON of objects/transforms/materials/lights."""
        return {
            "objects": {
                oid: {
                    "label": s.label,
                    "cumulative_yaw_deg": s.cumulative_yaw_deg,
                    "cumulative_scale_factor": s.cumulative_scale_factor,
                    "asset_candidate_id": s.asset_candidate_id,
                    "albedo": s.albedo,
                    "roughness": s.roughness,
                    "flags": list(s.flags),
                }
                for oid, s in self.objects.items()
            },
            "lights": dict(self.lights),
            "exposure": self.exposure,
        }

    def _require_object(self, object_id: str) -> MutableObjectState:
        if object_id not in self.objects:
            raise ActionRejected(f"unknown object_id {object_id!r} (not in s0's object set - SPEC §6 forbids add/delete)")
        return self.objects[object_id]

    def apply(self, action: Action) -> None:
        """SPEC §6 tool `apply(action)`: executes only whitelisted actions.
        Validated BEFORE any mutation - a rejected action leaves state
        untouched. Raises ActionRejected for anything check_action_static or
        the cumulative-bound checks below reject."""
        check_action_static(action)  # per-step bound + whitelist membership

        if isinstance(action, Rotate):
            obj = self._require_object(action.object_id)
            new_cumulative = obj.cumulative_yaw_deg + action.yaw_delta_deg
            if abs(new_cumulative) > MAX_ROTATE_CUMULATIVE_DEG:
                raise ActionRejected(f"rotate: cumulative yaw {new_cumulative}deg would exceed {MAX_ROTATE_CUMULATIVE_DEG}deg for {action.object_id}")
            obj.cumulative_yaw_deg = new_cumulative

        elif isinstance(action, Scale):
            obj = self._require_object(action.object_id)
            new_cumulative = obj.cumulative_scale_factor * action.factor
            length_u, width_v = obj.size_uv
            delta_u = length_u * abs(new_cumulative - 1.0) / 2
            delta_v = width_v * abs(new_cumulative - 1.0) / 2
            if max(delta_u, delta_v) > MAX_FOOTPRINT_DELTA_M:
                raise ActionRejected(
                    f"scale: cumulative factor {new_cumulative:.4f} would move {action.object_id}'s footprint edge "
                    f"by {max(delta_u, delta_v) * 100:.1f}cm > {MAX_FOOTPRINT_DELTA_M * 100:.0f}cm"
                )
            obj.cumulative_scale_factor = new_cumulative

        elif isinstance(action, SwapAsset):
            self._require_object(action.object_id).asset_candidate_id = action.candidate_id

        elif isinstance(action, SetMaterial):
            obj = self._require_object(action.object_id)
            obj.albedo = action.albedo
            obj.roughness = action.roughness

        elif isinstance(action, SetLight):
            light = self.lights.setdefault(action.light_id, {})
            if action.intensity is not None:
                light["intensity"] = action.intensity
            if action.colour is not None:
                light["colour"] = action.colour

        elif isinstance(action, SetExposure):
            self.exposure = action.value

        elif isinstance(action, Flag):
            self._require_object(action.object_id).flags.append(action.reason)

        else:  # unreachable: check_action_static already rejected any other type
            raise ActionRejected(f"unhandled action type {type(action).__name__}")

    def check_invariants(self, current_walls, current_object_ids: frozenset[str]) -> list[InvariantViolation]:
        """SPEC §6 'Forbidden (hard)': wall/floor edits, object add/delete.
        Passage-width invariant (SPEC §7, Stage D0's job) is intentionally NOT
        checked here - that needs a live gaps.json recompute against Stage A0's
        occupancy grid. This is defense-in-depth for what Stage C's OWN action
        whitelist could violate if a bug let something through (neither a wall
        action nor an add/delete action exists in the whitelist, so this should
        never fire via `apply` alone - tests exercise it by diffing against a
        deliberately mutated walls/object-id set to simulate that bug)."""
        violations = []
        if _walls_hash(current_walls) != self.baseline_walls_hash:
            violations.append(InvariantViolation("walls changed since s0"))
        if current_object_ids != self.baseline_object_ids:
            violations.append(InvariantViolation("object set changed since s0 (add/delete)"))
        return violations

    def score(
        self,
        *,
        mask_iou: float,
        point_fit_cm: float,
        photo: float,
        extra_violations: list[InvariantViolation] | None = None,
    ) -> ObjectiveBreakdown:
        """SPEC §6 tool `score()`: current objective breakdown, computed by
        this tool code (not the LLM). Caller supplies the three
        render/measurement-dependent terms (mask_iou_loss, point_fit,
        photo_loss come from `objective.py`'s functions applied to actual
        render output - out of this method's scope, since it needs a render);
        this assembles them with the invariant check."""
        return ObjectiveBreakdown(
            mask_iou_loss=mask_iou,
            point_fit_cm=point_fit_cm,
            photo_loss=photo,
            invariant_penalty=invariant_penalty(list(extra_violations or [])),
        )

    def render_from_camera(self, view_idx: int, out_path: Path, *, walls, objects: list[dict], gaps) -> None:
        """SPEC §6 tool `render_from_camera(view_idx)` -> PNG.

        DEVIATION (same root cause as Stage A0's `render_top_down.py` - no
        Blender/bpy on this VPS, see docs/DECISIONS.md '2026-09-06: MSA Stage
        A0 bootstrap'): view_idx=0 (top-down) reuses Stage A0's PIL renderer as
        a real, working stand-in. Any other view_idx (SPEC A7's eye-level/
        close-up angles) needs actual 3D perspective rendering, which PIL
        cannot produce - this raises NotImplementedError rather than silently
        returning a wrong top-down image for a requested eye-level shot.
        Documented as a hard blocker for Stage C2 (the full agent loop on a
        real scene, whose `photo_loss` term needs a matching-pose render)
        until Blender headless or an equivalent 3D renderer is available on
        this VPS or the Isaac GPU instance."""
        if view_idx != 0:
            raise NotImplementedError(
                f"render_from_camera(view_idx={view_idx}): only view_idx=0 (top-down) is implemented - "
                "no Blender/bpy on this VPS for eye-level/close-up perspective renders (see docstring)"
            )
        from scripts.msa.render_top_down import render_gaps_top_down

        render_gaps_top_down(walls, objects, gaps, out_path)
