"""CLI (pipeline-box GPU only, `sam3` venv): run real SAM 3 open-vocabulary
segmentation for a set of text prompts across every retained keyframe of a
scene, saving every above-threshold detection's mask + score + centroid.

This is Stage B1's real-mask half (see `scripts/msa/object_generation.py`'s
module docstring for why a reprojected-3D-point-cloud mask was abandoned).
Uses the exact same model-call pattern as `gpu/stage_objects.py`
(`build_sam3_image_model`/`Sam3Processor`/`set_text_prompt`) - not a
reimplementation, the same real segmentation the production pipeline uses,
just scanning ALL views for a handful of prompts instead of building the
full scene vocabulary.

    /workspace/envs/sam3/bin/python -m scripts.msa.sam3_scan_cli \
        --views-dir <dir of view_NNN.png> --prompts bed chair desk \
        --out-dir <dir> [--score-threshold 0.3]

Writes `<out-dir>/scan_results.json`: `{prompt: [{view_id, score, centroid_uv,
npix, mask_path}, ...]}`, one `.npy` (H,W bool) per detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", type=Path, required=True, help="dir of view_NNN.png")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise RuntimeError(
            "sam3 is not installed - this must run under the pipeline-box's sam3 "
            "venv (/workspace/envs/sam3/bin/python), not a generic Python."
        ) from e

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("model loaded")

    view_paths = sorted(args.views_dir.glob("view_*.png"))
    print(f"{len(view_paths)} views, {len(args.prompts)} prompts")

    results: dict[str, list[dict]] = {p: [] for p in args.prompts}
    for view_path in view_paths:
        view_id = view_path.stem
        img = Image.open(view_path).convert("RGB")
        state = processor.set_image(img)
        for prompt in args.prompts:
            out = processor.set_text_prompt(state=state, prompt=prompt)
            masks, scores = out["masks"], out["scores"]
            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            for j, sc in enumerate(scores_list):
                if sc < args.score_threshold:
                    continue
                m = masks[j]
                m_np = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(bool)
                if m_np.ndim == 3:
                    m_np = m_np[0]
                ys, xs = np.where(m_np)
                if len(xs) == 0:
                    continue
                mask_path = args.out_dir / f"{prompt}_{view_id}_det{j}.npy"
                np.save(mask_path, m_np)
                results[prompt].append(
                    {
                        "view_id": view_id,
                        "score": float(sc),
                        "centroid_uv": [float(xs.mean()), float(ys.mean())],
                        "npix": int(m_np.sum()),
                        "mask_path": mask_path.name,
                    }
                )
        print(f"{view_id} done")

    for prompt, dets in results.items():
        dets.sort(key=lambda d: -d["score"])
        top = dets[0] if dets else None
        print(f"{prompt}: {len(dets)} detections, top={top}")

    (args.out_dir / "scan_results.json").write_text(json.dumps(results, indent=2))
    print("SAM3_SCAN_SUCCESS")


if __name__ == "__main__":
    main()
