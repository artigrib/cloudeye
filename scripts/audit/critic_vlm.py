"""Scene critic, part 2: paired real-frame-vs-rendered-GLB VLM discrepancy check
(var/scratch/QUEUE.md M14 "Scene critic"). Report-only, like critic_rules.py -
every function here opens its inputs read-only and the only file this module writes
is its own JSON output (never any scene artifact).

Pipeline, per scene:
  1. Pick 6 camera poses from `cameras_aligned.json`, evenly spread across the whole
     video's view index range - `sample_camera_indices()`.
  2. For each, render the exported GLB from that EXACT pose with the same intrinsics
     as the real frame - `render_view_pair()`. Rendering reuses
     `scripts.msa.render_perspective`'s existing pure-Python/PIL software rasterizer
     (read-only import; chosen over headless three.js/pyrender because this box has
     no GPU/EGL/OSMesa and render_perspective.py's own docstring already documents
     that dead end - see its module docstring - while the rasterizer is proven,
     already tested (tests/test_msa_render_perspective.py), and needs nothing extra
     installed). Camera pose comes from `cameras_aligned.json`'s per-view extrinsics,
     converted from MapAnything's OpenCV-style convention (camera looks down +Z, X
     right, Y down) into render_perspective's world-space (camera_pos, forward,
     right, up) convention, then rotated into the scene's yaw-CORRECTED/exported
     frame with the exact same `rotate_point_xz` transform bootstrap.py applied to
     every other point in the scene - see `camera_pose_from_extrinsics()`.
  3. One extra pair: the aligned point cloud's top-down view vs. the exported hull
     outlines, in the style of geo5_out/render_15_polygon.py - `render_cloud_vs_hull_pair()`.
  4. Send all 7 pairs to OpenRouter `google/gemma-4-31b-it` (temperature 0, a fixed
     seed recorded on every call), asking only for discrepancies as JSON.

Cost: 7 calls/scene x 2 scenes = 14 calls total, per the task's budget.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw

from scripts.msa import render_perspective as rp
from scripts.msa.agent_loop.openrouter_client import OPENROUTER_BASE_URL, OpenRouterError, extract_json_object
from scripts.msa.geometry import rotate_point_xz
from scripts.msa.ply_io import read_ply_xyz_rgb

MODEL = "google/gemma-4-31b-it"
TEMPERATURE = 0.0
SEED = 0
N_POSES = 6
MAX_CLOUD_POINTS = 400_000

PROMPT_PATH = Path(__file__).parent / "prompts" / "critic_vlm.md"


# --------------------------------------------------------------------------- camera sampling


def sample_camera_indices(n_cameras: int, k: int = N_POSES) -> list[int]:
    """Evenly spread across the video's whole view-index range: `np.linspace(0,
    n_cameras - 1, k)`, rounded to the nearest integer, de-duplicated keeping order.
    That's the entire rule - the most literal reading of the task spec's "sample
    evenly across the view index"."""
    if n_cameras <= 0:
        return []
    raw = np.linspace(0, n_cameras - 1, min(k, n_cameras))
    seen: list[int] = []
    for v in raw:
        idx = int(round(float(v)))
        if idx not in seen:
            seen.append(idx)
    return seen


# --------------------------------------------------------------------------- camera pose conversion


def camera_pose_from_extrinsics(
    extrinsics_4x4: list[list[float]] | np.ndarray,
    yaw_correction_rad: float,
    yaw_rotation_center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converts one MapAnything camera-to-world extrinsics matrix (OpenCV convention:
    local +Z = forward, +X = right, +Y = down - confirmed against this project's own
    `camera_convention_report.json`, `positive_z_fraction: 1.0` for every clean view)
    into `render_perspective._render_scene`'s `(camera_pos, forward, right, up)`
    convention (world-space vectors, camera space Z = depth-along-forward, Y = up),
    then rotates all four into the scene's yaw-corrected EXPORTED frame with the same
    `rotate_point_xz` transform bootstrap.py used on every other point/vector in the
    scene (points translate about `yaw_rotation_center_xy`; direction vectors only
    rotate, about the origin).

    Verified empirically (2026-09-07) by rendering geo5_out's hero from view 30's
    real pose this way and comparing to `per_view_png/view_030.png`: matching
    composition (bed + pillow + wall + lamp pole in the same relative positions and
    scale) - see this task's write-up for the side-by-side."""
    m = np.asarray(extrinsics_4x4, dtype=float)
    r = m[:3, :3]
    t = m[:3, 3]
    forward = r[:, 2]
    right = r[:, 0]
    up = -r[:, 1]

    def _rotate_vector(v: np.ndarray) -> np.ndarray:
        x, z = rotate_point_xz(float(v[0]), float(v[2]), yaw_correction_rad, (0.0, 0.0))
        return np.array([x, v[1], z])

    forward_r = _rotate_vector(forward)
    right_r = _rotate_vector(right)
    up_r = _rotate_vector(up)
    pos_x, pos_z = rotate_point_xz(float(t[0]), float(t[2]), yaw_correction_rad, tuple(yaw_rotation_center_xy))
    pos_r = np.array([pos_x, t[1], pos_z])
    return pos_r, forward_r, right_r, up_r


def _lookat_camera_pose(camera_pos_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback pose for a scene whose `cameras_aligned.json` has no rotation/
    intrinsics at all (only camera positions - see `render_pairs_for_scene`'s
    `has_full_pose=False` branch): look from `camera_pos_xyz` toward `target_xyz`,
    with world-up as the up reference. Strictly less faithful than
    `camera_pose_from_extrinsics` (no real orientation data exists to be faithful
    to) - callers using this must say so in their output, which `render_view_pair`
    does via `PosedRender.pose_source`."""
    forward = target_xyz - camera_pos_xyz
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return camera_pos_xyz, forward, right, up


# --------------------------------------------------------------------------- rendering


@dataclass
class VlmPair:
    view: str
    left_image: Path  # "real"
    right_image: Path  # "rendered"
    detail: str = ""


def render_view_pair(
    glb_path: Path,
    real_image_path: Path,
    camera_pos: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    image_size: tuple[int, int],
    out_path: Path,
    view_label: str,
    pose_source: str,
) -> VlmPair:
    rp.render_hero_perspective_from_glb(
        Path(glb_path),
        Path(out_path),
        camera=(camera_pos, forward, right, up),
        fov_deg=fov_deg,
        image_size=image_size,
    )
    return VlmPair(
        view=view_label,
        left_image=Path(real_image_path),
        right_image=Path(out_path),
        detail=f"pose_source={pose_source}",
    )


def render_pairs_for_scene(
    *,
    glb_path: Path,
    cameras_aligned: dict[str, Any],
    per_view_dir: Path,
    yaw_correction_rad: float,
    yaw_rotation_center_xy: tuple[float, float],
    room_centroid_xy: tuple[float, float],
    out_dir: Path,
    k: int = N_POSES,
) -> list[VlmPair]:
    """Builds up to `k` (real frame, rendered GLB) pairs. Degrades gracefully when
    `cameras_aligned.json` is missing rotation/intrinsics (only a `cameras` list of
    positions - the rejected own-scenes bootstrap output's actual shape, schema-less
    and far sparser than the hero's `schema_version: 2` file): falls back to a
    look-at-room-centroid pose and a fixed FOV, and records that in the pair's
    `detail` so the merged report can say plainly that pose/intrinsics weren't
    matched exactly for that scene."""
    cameras = cameras_aligned.get("cameras", [])
    extrinsics = cameras_aligned.get("extrinsics")
    intrinsics = cameras_aligned.get("intrinsics")
    has_full_pose = bool(extrinsics) and bool(intrinsics) and len(extrinsics) == len(cameras)

    indices = sample_camera_indices(len(cameras), k)
    pairs: list[VlmPair] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for view_idx in indices:
        real_path = per_view_dir / f"view_{view_idx:03d}.png"
        if not real_path.is_file():
            continue
        with Image.open(real_path) as im:
            width, height = im.size

        if has_full_pose:
            intr = intrinsics[view_idx]
            fy = float(np.asarray(intr)[1][1])
            fov_deg = 2.0 * math.degrees(math.atan((height / 2.0) / fy))
            camera_pos, forward, right, up = camera_pose_from_extrinsics(
                extrinsics[view_idx], yaw_correction_rad, yaw_rotation_center_xy
            )
            pose_source = "cameras_aligned.extrinsics+intrinsics (matched pose+FOV)"
        else:
            raw_pos = np.asarray(cameras[view_idx], dtype=float)
            pos_x, pos_z = rotate_point_xz(float(raw_pos[0]), float(raw_pos[2]), yaw_correction_rad, tuple(yaw_rotation_center_xy))
            camera_pos_r = np.array([pos_x, raw_pos[1], pos_z])
            target = np.array([room_centroid_xy[0], raw_pos[1] - 0.3, room_centroid_xy[1]])
            camera_pos, forward, right, up = _lookat_camera_pose(camera_pos_r, target)
            fov_deg = rp.FOV_DEG
            pose_source = "camera position only, no rotation/intrinsics in cameras_aligned.json - look-at-room-centroid fallback, FOV approximate"

        out_path = out_dir / f"render_view_{view_idx:03d}.png"
        pairs.append(
            render_view_pair(
                glb_path,
                real_path,
                camera_pos,
                forward,
                right,
                up,
                fov_deg,
                (width, height),
                out_path,
                view_label=f"view_{view_idx:03d}",
                pose_source=pose_source,
            )
        )
    return pairs


# --------------------------------------------------------------------------- cloud-vs-hull top view


def render_cloud_vs_hull_pair(
    *,
    ply_path: Path,
    room_polygon: list[list[float]] | None,
    objects: list[dict[str, Any]],
    yaw_correction_rad: float,
    yaw_rotation_center_xy: tuple[float, float],
    floor_y: float,
    out_dir: Path,
    image_size: tuple[int, int] = (900, 900),
) -> VlmPair:
    """One extra pair (task spec item 2's "plus one extra pair"): LEFT = the aligned
    point cloud's own top-down view (obstacle band 0.1-1.4 m above floor_y, rotated
    into the exported frame), RIGHT = the same crop with room_polygon + every
    object's hull_xz drawn as outlines - same style as
    geo5_out/render_15_polygon.py (PIL top-down, not a 3D renderer, since this is a
    2D top-view comparison by design, not a perspective shot)."""
    xyz, _rgb = read_ply_xyz_rgb(ply_path, max_points=MAX_CLOUD_POINTS, seed=0)
    x_al, y_al, z_al = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    band = (y_al - floor_y >= 0.1) & (y_al - floor_y <= 1.4)
    x_al, z_al = x_al[band], z_al[band]

    cx = np.empty_like(x_al)
    cz = np.empty_like(z_al)
    for i in range(len(x_al)):
        cx[i], cz[i] = rotate_point_xz(float(x_al[i]), float(z_al[i]), yaw_correction_rad, tuple(yaw_rotation_center_xy))

    all_x = list(cx) + ([p[0] for p in room_polygon] if room_polygon else [])
    all_z = list(cz) + ([p[1] for p in room_polygon] if room_polygon else [])
    x0, x1 = min(all_x) - 0.3, max(all_x) + 0.3
    z0, z1 = min(all_z) - 0.3, max(all_z) + 0.3
    width, height = image_size
    scale = min((width - 40) / max(x1 - x0, 1e-6), (height - 40) / max(z1 - z0, 1e-6))

    def px(x: float, z: float) -> tuple[float, float]:
        return (20 + (x - x0) * scale, height - 20 - (z - z0) * scale)

    left = Image.new("RGB", image_size, (18, 18, 22))
    draw_left = ImageDraw.Draw(left)
    step = max(1, len(cx) // 200_000)
    for x, z in zip(cx[::step], cz[::step]):
        draw_left.point(px(x, z), fill=(150, 155, 165))

    right = left.copy()
    draw_right = ImageDraw.Draw(right)
    if room_polygon:
        pts = [px(*p) for p in room_polygon] + [px(*room_polygon[0])]
        draw_right.line(pts, fill=(235, 235, 235), width=2)
    for obj in objects:
        hull = obj.get("hull_xz")
        if not hull:
            continue
        pts = [px(*p) for p in hull] + [px(*hull[0])]
        draw_right.line(pts, fill=(90, 220, 120), width=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    left_path = out_dir / "cloud_top_view.png"
    right_path = out_dir / "hull_overlay_top_view.png"
    left.save(left_path)
    right.save(right_path)
    return VlmPair(
        view="cloud_top_view_vs_hull_overlay",
        left_image=left_path,
        right_image=right_path,
        detail=f"{len(cx)} cloud points (band {floor_y + 0.1:.2f}-{floor_y + 1.4:.2f} m), {len(objects)} object hull outline(s)",
    )


# --------------------------------------------------------------------------- prompt loading


def load_prompt() -> tuple[str, str]:
    """Returns (system, user_template) parsed out of prompts/critic_vlm.md's
    "## System" / "## User" sections."""
    text = PROMPT_PATH.read_text()
    system_marker, user_marker = "## System", "## User"
    system_start = text.index(system_marker) + len(system_marker)
    user_start = text.index(user_marker)
    system = text[system_start:user_start].strip()
    user_template = text[user_start + len(user_marker) :].strip()
    # Drop the "(template - ...)" parenthetical line if present, keep the rest.
    lines = user_template.splitlines()
    if lines and lines[0].startswith("(template"):
        lines = lines[1:]
    return system, "\n".join(lines).strip()


# --------------------------------------------------------------------------- OpenRouter call (multi-image)


@dataclass
class VlmCallRecord:
    view: str
    model: str
    prompt_hash: str
    seed: int
    temperature: float
    findings: list[dict[str, Any]]
    usage: dict[str, Any] | None
    raw_response_text: str
    error: str | None = None


def _image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def call_vlm_pair(
    pair: VlmPair,
    *,
    system: str,
    user_template: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
    timeout: float = 90.0,
) -> VlmCallRecord:
    """One OpenRouter chat-completion call with TWO images (left=real, right=
    rendered) - `scripts.msa.agent_loop.openrouter_client.call_vision_json` only
    supports a single image, so this is a small local variant of the same
    request/response handling (reuses its `extract_json_object`/`OpenRouterError`
    rather than duplicating them) that also sets `seed` for reproducibility. Never
    prints/logs the API key - reads it from `OPENROUTER_API_KEY` exactly like
    `openrouter_client.call_vision_json` does."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY not set in environment")

    user_text = user_template.format(view=pair.view)
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": _image_data_url(pair.left_image)}},
        {"type": "image_url", "image_url": {"url": _image_data_url(pair.right_image)}},
    ]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    prompt_hash = hashlib.sha256(json.dumps({"system": system, "user_text": user_text}, sort_keys=True).encode()).hexdigest()[:16]

    body = {"model": model, "messages": messages, "temperature": temperature, "seed": seed}
    try:
        resp = httpx.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return VlmCallRecord(
            view=pair.view, model=model, prompt_hash=prompt_hash, seed=seed, temperature=temperature,
            findings=[], usage=None, raw_response_text="", error=f"OpenRouter request failed: {exc}",
        )

    result = resp.json()
    if "error" in result:
        return VlmCallRecord(
            view=pair.view, model=model, prompt_hash=prompt_hash, seed=seed, temperature=temperature,
            findings=[], usage=None, raw_response_text="", error=f"OpenRouter API error: {result['error']}",
        )
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return VlmCallRecord(
            view=pair.view, model=model, prompt_hash=prompt_hash, seed=seed, temperature=temperature,
            findings=[], usage=result.get("usage"), raw_response_text=json.dumps(result)[:2000],
            error="unexpected OpenRouter response shape",
        )

    parsed = extract_json_object(text)
    findings = []
    error = None
    if parsed is None:
        error = "could not parse a JSON object out of the model response"
    else:
        raw_findings = parsed.get("findings", [])
        if isinstance(raw_findings, list):
            for f in raw_findings:
                if isinstance(f, dict):
                    f.setdefault("view", pair.view)
                    # PM decision (2026-09-07): every VLM finding is tagged
                    # source="vlm" + unverified=True (forced, not setdefault - the
                    # model's own JSON must never be able to claim otherwise) so
                    # nothing downstream (the CI gate, the viewer badge's
                    # count/colour) can mistake a VLM's free-text read of a render
                    # for a verified, gating finding. Mirrors critic_rules.py's
                    # findings, which are always source="rule"/unverified=False.
                    f["source"] = "vlm"
                    f["unverified"] = True
                    findings.append(f)
        else:
            error = "response JSON had no 'findings' list"

    return VlmCallRecord(
        view=pair.view, model=model, prompt_hash=prompt_hash, seed=seed, temperature=temperature,
        findings=findings, usage=result.get("usage"), raw_response_text=text, error=error,
    )


# --------------------------------------------------------------------------- top-level orchestration


def run_vlm(
    *,
    out_dir: Path,
    source_scene_dir: Path,
    render_out_dir: Path,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
) -> dict[str, Any]:
    """Full pipeline for one scene: build the 7 pairs, call the VLM on each, return
    the merged result dict (also written by the CLI below). `out_dir` = the scene's
    bootstrap/export out dir (objects.json, GLB, ...); `source_scene_dir` = the raw
    MapAnything scene dir (cameras_aligned.json, per_view_png/, aligned_room.ply)."""
    from scripts.audit.critic_rules import resolve_glb

    glb_path = resolve_glb(out_dir)
    if glb_path is None:
        raise FileNotFoundError(f"no exported GLB found in {out_dir}")

    objects_json = json.loads((out_dir / "objects.json").read_text()) if (out_dir / "objects.json").is_file() else {}
    room_polygon = objects_json.get("room_polygon")
    objects = objects_json.get("objects", [])

    scene_meta_path = out_dir / "scene_meta.json"
    scene_meta = json.loads(scene_meta_path.read_text()) if scene_meta_path.is_file() else {}
    yaw_correction_rad = float(scene_meta.get("yaw_correction_rad", 0.0) or 0.0)
    yaw_rotation_center_xy = tuple(scene_meta.get("yaw_rotation_center_xy") or (0.0, 0.0))
    floor_y = float(scene_meta.get("floor_y", 0.0) or 0.0)

    if room_polygon:
        room_centroid_xy = (
            sum(p[0] for p in room_polygon) / len(room_polygon),
            sum(p[1] for p in room_polygon) / len(room_polygon),
        )
    else:
        # Bootstrap-only out dir (no objects.json/room_polygon - see
        # critic_rules.py's GLB fallback for the same situation): fall back to the
        # exported GLB's own overall XZ bounding-box centre as a stand-in room
        # centroid, purely so the look-at fallback camera in render_pairs_for_scene
        # points somewhere inside the scene instead of the origin.
        import trimesh as _trimesh

        glb_scene = _trimesh.load(str(glb_path), process=False)
        bounds = glb_scene.bounds  # (2, 3): [min, max] over x, y, z
        room_centroid_xy = (float((bounds[0][0] + bounds[1][0]) / 2.0), float((bounds[0][2] + bounds[1][2]) / 2.0))

    cameras_aligned_path = source_scene_dir / "cameras_aligned.json"
    cameras_aligned = json.loads(cameras_aligned_path.read_text()) if cameras_aligned_path.is_file() else {"cameras": []}
    per_view_dir = source_scene_dir / "per_view_png"

    pairs = render_pairs_for_scene(
        glb_path=glb_path,
        cameras_aligned=cameras_aligned,
        per_view_dir=per_view_dir,
        yaw_correction_rad=yaw_correction_rad,
        yaw_rotation_center_xy=yaw_rotation_center_xy,
        room_centroid_xy=room_centroid_xy,
        out_dir=render_out_dir,
    )

    ply_path = source_scene_dir / "aligned_room.ply"
    if ply_path.is_file():
        pairs.append(
            render_cloud_vs_hull_pair(
                ply_path=ply_path,
                room_polygon=room_polygon,
                objects=objects,
                yaw_correction_rad=yaw_correction_rad,
                yaw_rotation_center_xy=yaw_rotation_center_xy,
                floor_y=floor_y,
                out_dir=render_out_dir,
            )
        )

    system, user_template = load_prompt()
    calls: list[VlmCallRecord] = []
    for pair in pairs:
        calls.append(call_vlm_pair(pair, system=system, user_template=user_template, model=model, temperature=temperature, seed=seed))

    all_findings = []
    for call in calls:
        all_findings.extend(call.findings)

    return {
        "out_dir": str(out_dir),
        "source_scene_dir": str(source_scene_dir),
        "glb_used": str(glb_path),
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "n_pairs": len(pairs),
        "pairs": [
            {"view": p.view, "left_image": str(p.left_image), "right_image": str(p.right_image), "detail": p.detail}
            for p in pairs
        ],
        "calls": [
            {
                "view": c.view,
                "prompt_hash": c.prompt_hash,
                "usage": c.usage,
                "error": c.error,
                "n_findings": len(c.findings),
                "raw_response_text": c.raw_response_text,
            }
            for c in calls
        ],
        "findings": all_findings,
        "n_findings": len(all_findings),
    }


def write_critic_vlm_json(*, out_dir: Path, source_scene_dir: Path, render_out_dir: Path, dest: Path | None = None, **kwargs) -> Path:
    result = run_vlm(out_dir=out_dir, source_scene_dir=source_scene_dir, render_out_dir=render_out_dir, **kwargs)
    dest = dest or Path(out_dir) / "critic_vlm.json"
    dest.write_text(json.dumps(result, indent=2))
    return dest


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="bootstrap/export out dir (objects.json, GLB)")
    parser.add_argument("source_scene_dir", type=Path, help="raw MapAnything scene dir (cameras_aligned.json, per_view_png/, aligned_room.ply)")
    parser.add_argument("--render-out", type=Path, default=None, help="where to write rendered PNGs (default: <out_dir>/critic_vlm_renders)")
    parser.add_argument("--out", type=Path, default=None, help="where to write critic_vlm.json")
    args = parser.parse_args()
    render_out = args.render_out or (args.out_dir / "critic_vlm_renders")
    dest = write_critic_vlm_json(out_dir=args.out_dir, source_scene_dir=args.source_scene_dir, render_out_dir=render_out, dest=args.out)
    result = json.loads(dest.read_text())
    print(f"wrote {dest}: {result['n_findings']} finding(s) over {result['n_pairs']} pair(s), model={result['model']}")


if __name__ == "__main__":
    _main()
