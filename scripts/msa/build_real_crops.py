"""CLI (VPS, CPU-only): consume a `scripts/msa/sam3_scan_cli.py` scan result
(real SAM 3 masks) and build one real RGBA crop + `manifest.json` per scene
object, matching each object to its own detection via
`object_generation.match_best_detection` (coordinate-frame-corrected - see
that module's docstring). The GPU half (`generate_from_manifest_cli.py`)
consumes this manifest unchanged.

    uv run python -m scripts.msa.build_real_crops \
        --scene-dir var/uploads/scenes/<id> \
        --scan-dir <sam3_scan_cli.py's --out-dir, copied back from the pipeline-box> \
        --object-ids bed_0 chair_0 ... --prompts bed chair ... \
        --out-dir <dir>

`--object-ids`/`--prompts` are paired positionally (same length, same order) -
this is what lets several objects share one prompt (e.g. two nightstands).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.msa.object_generation import (
    bbox_volume,
    crop_from_mask,
    load_alignment_transform,
    match_best_detection,
    to_raw_frame,
)


def load_view_meta(scene_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    """view_id -> (camera_pose, intrinsics, (H, W)) for every retained per_view/*.npz."""
    view_data = {}
    for npz_path in sorted((scene_dir / "per_view").glob("view_*.npz")):
        d = np.load(npz_path)
        H, W = d["mask"].shape
        view_data[npz_path.stem] = (d["camera_pose"], d["intrinsics"], (H, W))
    return view_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--object-ids", nargs="+", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.object_ids) != len(args.prompts):
        raise SystemExit("--object-ids and --prompts must be the same length (paired positionally)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    view_data = load_view_meta(args.scene_dir)
    scan = json.loads((args.scan_dir / "scan_results.json").read_text())
    objects_json = json.loads((args.scene_dir / "scene_objects" / "scene_objects.json").read_text())
    by_id = {o["id"]: o for o in objects_json}
    R, floor_y = load_alignment_transform(args.scene_dir)

    manifest = []
    for obj_id, prompt in zip(args.object_ids, args.prompts):
        obj = by_id.get(obj_id)
        if obj is None:
            print(f"SKIP {obj_id}: not in scene_objects.json")
            continue
        detections = scan.get(prompt, [])
        centroid_raw = to_raw_frame(np.array([obj["centroid_xyz"]]), R, floor_y)[0]
        match = match_best_detection(centroid_raw, detections, view_data)
        if match is None:
            print(f"SKIP {obj_id}: no matching SAM3 detection for prompt {prompt!r}")
            continue

        view_id = match["view_id"]
        mask = np.load(args.scan_dir / match["mask_path"])
        image = np.array(Image.open(args.scene_dir / "per_view_png" / f"{view_id}.png").convert("RGB"))
        rgba = crop_from_mask(image, mask)
        if rgba is None:
            print(f"SKIP {obj_id}: empty mask")
            continue

        crop_path = args.out_dir / f"{obj_id}.png"
        Image.fromarray(rgba, "RGBA").save(crop_path)
        manifest.append(
            {
                "id": obj_id,
                "label": obj["label"],
                "crop_path": crop_path.name,
                "view_id": view_id,
                "match_dist_px": match["match_dist_px"],
                "sam3_score": match["score"],
                "bbox_min": obj["bbox_min"],
                "bbox_max": obj["bbox_max"],
                "measured_volume_m3": bbox_volume(obj["bbox_min"], obj["bbox_max"]),
            }
        )
        print(
            f"OK {obj_id}: view={view_id} score={match['score']:.3f} "
            f"dist={match['match_dist_px']:.1f}px crop={rgba.shape[:2]}"
        )

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest with {len(manifest)}/{len(args.object_ids)} objects")


if __name__ == "__main__":
    main()
