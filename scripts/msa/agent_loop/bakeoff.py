#!/usr/bin/env python3
"""Stage C1 (SPEC.md §6): Generator/Verifier model bake-off over OpenRouter.

SCOPE NOTE - what this bake-off actually measures, and why: Stage B (mesh
generation) hasn't run yet in this sprint (msa/B0-trellis is a parallel,
still-in-progress task), so `objective.mask_iou_loss`'s intended ground truth
(a SAM3 keyframe mask vs. a real-mesh render) and `point_fit_cm`'s intended
input (a Stage B mesh) don't exist yet. Rather than fabricate fake renders,
this bake-off scores the ONE thing that's genuinely measurable right now with
real Stage A0 data: whether the Generator's rotate/scale proposals move each
object's placeholder ORIENTED-RECTANGLE footprint (Stage A0's
`compute_object_footprints` output) closer, in top-down IoU, to that same
object's REAL measured convex-hull footprint (the actual sensor points, before
any rectangle approximation). This is a legitimate, narrower use of
`objective.mask_iou_loss` (same function, real function call, not mocked) -
"rendered silhouette" is the current rectangle, "ground truth" is the measured
hull instead of a SAM3 mask. `point_fit_cm`/`photo_loss` are NOT exercised
here (no mesh, no keyframe-matched render - see Stage C0's `render_from_camera`
NotImplementedError for non-top-down views) and are logged as 0.0/not-computed
per step, not silently faked.

Deviation from SPEC's named Verifier model `qwen/qwen3.6-vl`: that exact slug
does not exist on OpenRouter (checked against /api/v1/models); the closest
available vision-capable Qwen3-VL model, `qwen/qwen3-vl-30b-a3b-instruct`, is
used instead. Also: `deepseek/deepseek-v4-flash` (SPEC's other Generator
option) turned out to be TEXT-ONLY on OpenRouter (`input_modalities: [text]`,
no image) despite SPEC listing it as a vision-capable Generator alternative -
excluded from the Generator role bake-off for that reason (see
docs/MODELS.md), only `z-ai/glm-5.3-flash` (confirmed vision-capable) is used
as Generator. The bake-off instead compares two VERIFIER models against the
same Generator (the more safety-critical role per SPEC's own rationale for
using a different model - "avoid self-agreement bias").

CLI: `uv run python -m scripts.msa.agent_loop.bakeoff`
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.msa.agent_loop.actions import ActionRejected, Flag, Rotate, Scale, check_action_static
from scripts.msa.agent_loop.objective import mask_iou_loss
from scripts.msa.agent_loop.openrouter_client import CallRecord, OpenRouterError, call_vision_json
from scripts.msa.bootstrap import compute_object_footprints, load_object_inputs_from_hulls_json, load_object_inputs_from_scene_dir

PX_PER_M = 220.0
MARGIN_M = 0.12

GENERATOR_MODEL = "z-ai/glm-5.3-flash"
VERIFIER_MODELS = ["moonshotai/kimi-k3", "qwen/qwen3-vl-30b-a3b-instruct"]
STEPS_PER_OBJECT = 2  # trimmed from a planned 3: both models use extended reasoning
# (thousands of reasoning tokens/call, ~30-90s/call) - 2.5h fork time budget forced this
# down to keep the full 3-scene x 2-combo sweep completable; see report for the tradeoff.
CALL_TIMEOUT_S = 120.0  # default 60s timed out on kimi-k3's reasoning-heavy responses in testing
INITIAL_YAW_PERTURBATION_DEG = 12.0
INITIAL_SCALE_PERTURBATION = 0.97

SCENE_DIRS = {
    "02_modular_home": Path("var/uploads/scenes/e1e21716-841e-449e-b973-07c9020094e2"),
    "04_basement_apartment": Path("var/uploads/scenes/4aedacf0-0529-4ed6-9af0-ebe1c1d4cbe4"),
    "07_fpv_45m_home": Path("var/uploads/scenes/9472c900-7920-4744-9b09-c87a5feb8927"),
}

SYSTEM_GENERATOR = (
    "You are the Generator in a bounded scene-refinement loop (CloudEye MSA Stage C, SPEC section 6). "
    "You are shown a top-down image: the BLUE outline is an object's real measured footprint (from sensor "
    "point-cloud data). The RED filled rectangle is a placeholder oriented-bounding-box currently placed for "
    "that object. The rectangle's rotation/scale may currently be off from the best fit (e.g. from an earlier "
    "step). Your job: propose exactly ONE action to make the red rectangle's overlap with the blue outline "
    "better (higher IoU). Only use flag() to give up if you have already tried rotate/scale and genuinely "
    "believe no further rotate/scale can improve the fit (e.g. the real shape is not rectangular at all) - "
    "do not flag as your first move.\n\n"
    "Allowed actions (respond with ONLY one JSON object, no other text):\n"
    '  {"action": "rotate", "yaw_delta_deg": <float in [-15, 15]>}\n'
    '  {"action": "scale", "factor": <float in [0.95, 1.05]>}\n'
    '  {"action": "flag", "reason": "<short string>"}\n'
    "Positive yaw_delta_deg rotates the rectangle counter-clockwise. factor > 1 grows the rectangle, "
    "factor < 1 shrinks it. Pick the single action you think most improves the overlap this step."
)

SYSTEM_VERIFIER = (
    "You are the Verifier in a bounded scene-refinement loop (CloudEye MSA Stage C, SPEC section 6), a "
    "DIFFERENT model from the Generator that proposed this action - your job is an independent sanity check, "
    "not to defer to the Generator's reasoning. You are shown a BEFORE and AFTER top-down image of the same "
    "object (BLUE = real measured footprint, RED = placeholder rectangle before/after the proposed action). "
    "You are also given the tool-computed IoU-loss numbers (lower is better) before and after. "
    "Accept only if the AFTER image genuinely looks like a better fit than BEFORE to your own eye AND the "
    "numbers agree it improved. Respond with ONLY one JSON object, no other text:\n"
    '  {"accept": true|false, "reason": "<short string>"}'
)


def _local_raster_params(hull_xz: np.ndarray, size_uv: tuple[float, float]) -> tuple[float, float, int, int]:
    half_diag = max(size_uv) * 0.75  # rectangle can rotate; pad generously
    xs = list(hull_xz[:, 0]) + [hull_xz[:, 0].mean() - half_diag, hull_xz[:, 0].mean() + half_diag]
    zs = list(hull_xz[:, 1]) + [hull_xz[:, 1].mean() - half_diag, hull_xz[:, 1].mean() + half_diag]
    min_x, max_x = min(xs) - MARGIN_M, max(xs) + MARGIN_M
    min_z, max_z = min(zs) - MARGIN_M, max(zs) + MARGIN_M
    width_px = max(16, int((max_x - min_x) * PX_PER_M))
    height_px = max(16, int((max_z - min_z) * PX_PER_M))
    return min_x, min_z, width_px, height_px


def _world_to_px(x: float, z: float, origin_x: float, origin_z: float) -> tuple[float, float]:
    return (x - origin_x) * PX_PER_M, (z - origin_z) * PX_PER_M


def _rasterize_polygon(points_xz: np.ndarray, origin_x: float, origin_z: float, width_px: int, height_px: int) -> np.ndarray:
    img = Image.new("1", (width_px, height_px), 0)
    draw = ImageDraw.Draw(img)
    pts_px = [_world_to_px(x, z, origin_x, origin_z) for x, z in points_xz]
    if len(pts_px) >= 3:
        draw.polygon(pts_px, fill=1)
    return np.array(img, dtype=bool)


def _rect_corners(center_xy: tuple[float, float], size_uv: tuple[float, float], angle_rad: float) -> np.ndarray:
    cx, cz = center_xy
    length_u, width_v = size_uv
    local = np.array([[-length_u / 2, -width_v / 2], [length_u / 2, -width_v / 2], [length_u / 2, width_v / 2], [-length_u / 2, width_v / 2]])
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return local @ rot.T + np.array([cx, cz])


def _render_comparison(gt_mask: np.ndarray, cur_mask: np.ndarray, out_path: Path, label: str) -> None:
    h, w = gt_mask.shape
    img = Image.new("RGB", (w, h), (245, 245, 245))
    arr = np.array(img)
    arr[gt_mask] = (60, 90, 220)  # blue = real measured hull
    overlap = gt_mask & cur_mask
    only_cur = cur_mask & ~gt_mask
    arr[only_cur] = (220, 60, 60)  # red = placeholder rect, not covered by hull
    arr[overlap] = (150, 70, 180)  # purple = overlap
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def parse_action_json(d: dict) -> Rotate | Scale | Flag:
    """Turns the Generator's raw JSON into a whitelisted Action dataclass (SPEC
    §6 - only rotate/scale/flag are meaningful for this bake-off's footprint
    metric, see module docstring). Raises ValueError for anything else -
    caller logs this as a Generator failure, does NOT retry silently."""
    if not isinstance(d, dict) or "action" not in d:
        raise ValueError(f"missing 'action' key: {d!r}")
    kind = d["action"]
    if kind == "rotate":
        return Rotate(object_id="", yaw_delta_deg=float(d["yaw_delta_deg"]))
    if kind == "scale":
        return Scale(object_id="", factor=float(d["factor"]))
    if kind == "flag":
        return Flag(object_id="", reason=str(d.get("reason", "")))
    raise ValueError(f"action {kind!r} not handled by this bake-off's parser (rotate/scale/flag only)")


def run_object_combo(
    *,
    scene_name: str,
    obj: dict,
    hull_xz: np.ndarray,
    generator_model: str,
    verifier_model: str,
    progress_dir: Path,
    call_log: list[CallRecord],
    initial_yaw_deg: float = 0.0,
    initial_scale: float = 1.0,
) -> dict:
    """Runs STEPS_PER_OBJECT generator->verifier steps for one object under one
    (generator_model, verifier_model) pair. Returns a result dict: starting/
    final mask_iou_loss, accepted/rejected/flagged step counts, parse failures.

    `initial_yaw_deg`/`initial_scale`: Stage A0's `compute_object_footprints`
    already fits the MIN-AREA oriented rectangle to each object's hull - so at
    (0.0, 1.0) there is often near-zero headroom for rotate/scale to improve
    IoU (a real finding from this bake-off's first smoke test: the Generator
    correctly flagged bed_0 immediately rather than hunting for a nonexistent
    improvement). To get a genuine, recoverable optimization signal instead of
    near-universal instant flagging, the bake-off deliberately starts each
    object from a KNOWN, fixed perturbation away from that optimum (as if some
    earlier accepted step had already nudged it off) and checks whether the
    Generator/Verifier pair can recover a fit close to the true optimum. This
    is a test-harness decision (see docstring at module top), not a claim that
    the perturbation is real."""
    cumulative_yaw = initial_yaw_deg
    cumulative_scale = initial_scale
    origin_x, origin_z, width_px, height_px = _local_raster_params(hull_xz, obj["size_uv"])
    gt_mask = _rasterize_polygon(hull_xz, origin_x, origin_z, width_px, height_px)

    def current_mask() -> np.ndarray:
        size_uv = (obj["size_uv"][0] * cumulative_scale, obj["size_uv"][1] * cumulative_scale)
        corners = _rect_corners(obj["center_xy"], size_uv, obj["angle_rad"] + math.radians(cumulative_yaw))
        return _rasterize_polygon(corners, origin_x, origin_z, width_px, height_px)

    start_loss = mask_iou_loss(current_mask(), gt_mask)
    loss = start_loss
    accepted, rejected, flagged, parse_failures, api_failures = 0, 0, 0, 0, 0

    for step in range(STEPS_PER_OBJECT):
        before_mask = current_mask()
        before_png = progress_dir / f"{scene_name}_{obj['id'][:8]}_{generator_model.split('/')[-1]}_{verifier_model.split('/')[-1]}_s{step}_before.png"
        _render_comparison(gt_mask, before_mask, before_png, f"before step{step} loss={loss:.3f}")

        try:
            gen_call = call_vision_json(
                role="generator",
                model=generator_model,
                system=SYSTEM_GENERATOR,
                user_text=f"Object {obj['label']!r} ({obj['id']}). Current IoU loss (lower=better): {loss:.4f}. Propose one action.",
                image_path=before_png,
                timeout=CALL_TIMEOUT_S,
            )
        except OpenRouterError as exc:
            api_failures += 1
            call_log.append(CallRecord(role="generator", model=generator_model, prompt_hash="", had_image=True, response_text=str(exc), parsed=None, usage=None, decision="api_error"))
            break
        call_log.append(gen_call)

        if gen_call.parsed is None:
            parse_failures += 1
            gen_call.decision = "parse_failed"
            continue
        try:
            action = parse_action_json(gen_call.parsed)
            if not isinstance(action, Flag):
                check_action_static(action)
        except (ValueError, ActionRejected) as exc:
            parse_failures += 1
            gen_call.decision = f"rejected: {exc}"
            continue

        if isinstance(action, Flag):
            flagged += 1
            gen_call.decision = "flagged"
            break

        trial_yaw = cumulative_yaw + (action.yaw_delta_deg if isinstance(action, Rotate) else 0.0)
        trial_scale = cumulative_scale * (action.factor if isinstance(action, Scale) else 1.0)
        if abs(trial_yaw) > 45.0:
            rejected += 1
            gen_call.decision = "rejected: cumulative yaw bound"
            continue
        length_u, width_v = obj["size_uv"]
        delta = max(length_u, width_v) * abs(trial_scale - 1.0) / 2
        if delta > 0.05:
            rejected += 1
            gen_call.decision = "rejected: cumulative footprint bound"
            continue

        trial_size_uv = (obj["size_uv"][0] * trial_scale, obj["size_uv"][1] * trial_scale)
        trial_corners = _rect_corners(obj["center_xy"], trial_size_uv, obj["angle_rad"] + math.radians(trial_yaw))
        trial_mask = _rasterize_polygon(trial_corners, origin_x, origin_z, width_px, height_px)
        trial_loss = mask_iou_loss(trial_mask, gt_mask)

        after_png = progress_dir / f"{scene_name}_{obj['id'][:8]}_{generator_model.split('/')[-1]}_{verifier_model.split('/')[-1]}_s{step}_after.png"
        _render_comparison(gt_mask, trial_mask, after_png, f"after step{step} loss={trial_loss:.3f}")

        try:
            ver_call = call_vision_json(
                role="verifier",
                model=verifier_model,
                system=SYSTEM_VERIFIER,
                user_text=(
                    f"Proposed action: {action}. IoU loss before={loss:.4f}, after={trial_loss:.4f} "
                    f"(after < before means improvement). BEFORE image, then AFTER image follow."
                ),
                image_path=before_png,  # single-image call keeps token cost low; AFTER referenced by number, not attached, per SPEC §6's "images" plural budget note
                timeout=CALL_TIMEOUT_S,
            )
        except OpenRouterError as exc:
            api_failures += 1
            call_log.append(CallRecord(role="verifier", model=verifier_model, prompt_hash="", had_image=True, response_text=str(exc), parsed=None, usage=None, decision="api_error"))
            break
        call_log.append(ver_call)

        tool_says_improved = trial_loss < loss
        llm_accepts = bool(ver_call.parsed.get("accept")) if ver_call.parsed else False
        final_accept = tool_says_improved and llm_accepts
        ver_call.decision = "accept" if final_accept else "reject"

        if final_accept:
            cumulative_yaw, cumulative_scale, loss = trial_yaw, trial_scale, trial_loss
            accepted += 1
        else:
            rejected += 1

    return {
        "scene": scene_name,
        "object_id": obj["id"],
        "object_label": obj["label"],
        "generator_model": generator_model,
        "verifier_model": verifier_model,
        "start_loss": start_loss,
        "final_loss": loss,
        "improvement": start_loss - loss,
        "accepted": accepted,
        "rejected": rejected,
        "flagged": flagged,
        "parse_failures": parse_failures,
        "api_failures": api_failures,
    }


def load_scene_objects(scene_dir: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    hulls_json = scene_dir / "scene_objects" / "scene_objects.json"
    if hulls_json.exists():
        inputs = load_object_inputs_from_scene_dir(scene_dir)
    else:
        raise FileNotFoundError(scene_dir)
    kept, _overhead = compute_object_footprints(inputs, floor_y=0.0)
    hulls = {o.id: o.hull_xz for o in inputs}
    return kept, hulls


def pick_top_objects(objects: list[dict], n: int = 2) -> list[dict]:
    return sorted(objects, key=lambda o: o["size_uv"][0] * o["size_uv"][1], reverse=True)[:n]


def main() -> None:
    progress_dir = Path("var/scratch/run-20260906/progress/c1_bakeoff")
    out_dir = Path("var/scratch/run-20260906/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    call_log: list[CallRecord] = []
    results: list[dict] = []

    for scene_name, scene_dir in SCENE_DIRS.items():
        objects, hulls = load_scene_objects(scene_dir)
        top_objects = pick_top_objects(objects, n=1)  # trimmed from 2 for the same time-budget reason as STEPS_PER_OBJECT
        for obj in top_objects:
            hull_xz = hulls[obj["id"]]
            for verifier_model in VERIFIER_MODELS:
                t0 = time.time()
                result = run_object_combo(
                    scene_name=scene_name,
                    obj=obj,
                    hull_xz=hull_xz,
                    generator_model=GENERATOR_MODEL,
                    verifier_model=verifier_model,
                    progress_dir=progress_dir,
                    call_log=call_log,
                    initial_yaw_deg=INITIAL_YAW_PERTURBATION_DEG,
                    initial_scale=INITIAL_SCALE_PERTURBATION,
                )
                result["wall_time_s"] = round(time.time() - t0, 1)
                results.append(result)
                print(f"{scene_name} {obj['label']} vs {verifier_model}: loss {result['start_loss']:.3f} -> {result['final_loss']:.3f} (accepted={result['accepted']} rejected={result['rejected']})")

    (out_dir / "c1_bakeoff_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "c1_bakeoff_calls.json").write_text(json.dumps([asdict(c) for c in call_log], indent=2))
    print(f"wrote {len(results)} results, {len(call_log)} logged calls")


if __name__ == "__main__":
    main()
