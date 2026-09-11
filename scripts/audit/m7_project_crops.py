"""M7 audit helper (item 1/3 of the 13c audit): projects each of the 18 final
`m7_out/objects.json` hulls into the hero's 74 raw camera views
(`cameras_aligned.json` + `per_view_png/`) and crops+annotates the best view.

Read-only against the hero scan; writes only under `progress/` in the run dir
passed on the command line (default: var/scratch/run-20260906/progress).

M7's `bootstrap.py` applies one rigid XZ rotation (`_rotate_objects` /
`_rotate_camera_poses`, see scripts/msa/bootstrap.py) to take the aligned
scan frame (the frame `cameras_aligned.json` and `scene_objects.json` are
both in) to the final de-cluttered frame objects.json ships in. This script
inverts that same rotation (angle -> -angle, same center) to bring each
object's extruded hull back into the aligned frame before projecting with
`scripts/msa/object_generation.py::project_points_to_view` against
`cameras_aligned.json`'s per-camera-to-world extrinsics + intrinsics (NOT
`load_alignment_transform`/`to_raw_frame` - those convert aligned points into
the *pipeline's raw pre-alignment* frame that `per_view/*.npz` cameras use;
`cameras_aligned.json` is already in the aligned frame, which is what M7's
yaw rotation was applied on top of).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from msa.object_generation import project_points_to_view  # noqa: E402

M7_OUT = Path("var/scratch/run-20260906/m7_out")
HERO_SCENE = Path("var/scratch/hero-frozen/own_0901_173903/scene")
B1_MANIFEST = Path("var/scratch/run-20260906/hero_b1/crops/manifest.json")


def rotate_point_xz(x: float, z: float, angle_rad: float, center: tuple[float, float]) -> tuple[float, float]:
    cx, cz = center
    dx, dz = x - cx, z - cz
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    nx = cos_a * dx - sin_a * dz
    nz = sin_a * dx + cos_a * dz
    return nx + cx, nz + cz


def load_cameras() -> list[dict]:
    d = json.loads((HERO_SCENE / "cameras_aligned.json").read_text())
    n = len(d["cameras"])
    cams = []
    for i in range(n):
        cams.append(
            {
                "view_id": f"view_{i:03d}",
                "extrinsics": np.array(d["extrinsics"][i], dtype=np.float64),
                "intrinsics": np.array(d["intrinsics"][i], dtype=np.float64),
            }
        )
    return cams


def project_object(obj: dict, scene_meta: dict, cameras: list[dict]) -> dict:
    """Returns dict(view_id, bbox_px=(x0,y0,x1,y1), fully_in_frame, area_px, image_wh)
    for the best camera, or dict(failed=True, reason=...) if nothing usable."""
    hull_xz = obj["hull_xz"]
    y0 = obj["bbox_min_y"]
    y1 = obj["bbox_min_y"] + obj["height"]

    angle = scene_meta.get("yaw_correction_rad") or 0.0
    center = tuple(scene_meta.get("yaw_rotation_center_xy") or (0.0, 0.0))
    applied = scene_meta.get("yaw_applied", False)
    inv_angle = -angle if applied else 0.0

    pts_aligned = []
    for x, z in hull_xz:
        ax, az = rotate_point_xz(x, z, inv_angle, center) if applied else (x, z)
        pts_aligned.append((ax, y0, az))
        pts_aligned.append((ax, y1, az))
    pts = np.array(pts_aligned, dtype=np.float64)

    candidates = []
    for cam in cameras:
        uv, in_front = project_points_to_view(pts, cam["extrinsics"], cam["intrinsics"])
        if not np.all(in_front):
            continue
        x0, y0px = uv[:, 0].min(), uv[:, 1].min()
        x1, y1px = uv[:, 0].max(), uv[:, 1].max()
        candidates.append(
            {
                "view_id": cam["view_id"],
                "bbox_px": (float(x0), float(y0px), float(x1), float(y1px)),
            }
        )

    if not candidates:
        return {"failed": True, "reason": "no camera has the whole hull in front of it (in_front test failed for every view)"}

    img_probe = Image.open(HERO_SCENE / "per_view_png" / f"{candidates[0]['view_id']}.png")
    W, H = img_probe.size

    best = None
    best_score = -1.0
    for c in candidates:
        x0, y0px, x1, y1px = c["bbox_px"]
        fully_in = (x0 >= 0 and y0px >= 0 and x1 <= W and y1px <= H)
        area = max(0.0, (min(x1, W) - max(x0, 0))) * max(0.0, (min(y1px, H) - max(y0px, 0)))
        score = area if fully_in else area * 0.3  # prefer fully-in-frame, but keep partials as fallback
        c["fully_in_frame"] = fully_in
        c["area_px"] = area
        c["image_wh"] = (W, H)
        if score > best_score:
            best_score = score
            best = c
    return best


def draw_crop(view_id: str, bbox_px: tuple[float, float, float, float], out_path: Path, pad_frac: float = 0.15) -> None:
    img = Image.open(HERO_SCENE / "per_view_png" / f"{view_id}.png").convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = bbox_px
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    pad = pad_frac * max(x1c - x0c, y1c - y0c, 1)
    cx0, cy0 = max(0, x0c - pad), max(0, y0c - pad)
    cx1, cy1 = min(W, x1c + pad), min(H, y1c + pad)
    crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    draw = ImageDraw.Draw(crop)
    rx0, ry0 = x0 - cx0, y0 - cy0
    rx1, ry1 = x1 - cx0, y1 - cy0
    draw.rectangle([rx0, ry0, rx1, ry1], outline=(255, 32, 32), width=3)
    crop.save(out_path)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("var/scratch/run-20260906/progress")
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = json.loads((M7_OUT / "objects.json").read_text())["objects"]
    scene_meta = json.loads((M7_OUT / "scene_meta.json").read_text())
    cameras = load_cameras()

    order = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [o["id"] for o in objects]
    by_id = {o["id"]: o for o in objects}

    results = {}
    for i, oid in enumerate(order, start=1):
        obj = by_id[oid]
        res = project_object(obj, scene_meta, cameras)
        out_png = out_dir / f"13c_obj_{i:02d}.png"
        if res.get("failed"):
            results[oid] = {"ok": False, "index": i, "reason": res["reason"]}
            print(f"{i:02d} {oid:14s} PROJECTION FAILED: {res['reason']}")
            continue
        draw_crop(res["view_id"], res["bbox_px"], out_png)
        results[oid] = {
            "ok": True,
            "index": i,
            "view_id": res["view_id"],
            "bbox_px": res["bbox_px"],
            "fully_in_frame": res["fully_in_frame"],
            "area_px": res["area_px"],
            "image_wh": res["image_wh"],
            "png": str(out_png),
        }
        print(f"{i:02d} {oid:14s} view={res['view_id']} fully_in_frame={res['fully_in_frame']} "
              f"area_px={res['area_px']:.0f} -> {out_png.name}")

    (out_dir / "13c_projection_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
