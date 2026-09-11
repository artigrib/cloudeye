"""CLI (GPU instance only): run `Trellis2Client.generate` for every crop in a
manifest.json (produced by `build_crops_batch.py`), apply the SPEC §5
containment test, export accepted meshes to GLB, and write a results.json.

    uv run python -m scripts.msa.generate_from_manifest_cli --manifest-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()

    from PIL import Image

    from scripts.msa.object_generation import containment_ratio, passes_containment
    from scripts.msa.trellis_client import Trellis2Client

    manifest = json.loads((args.manifest_dir / "manifest.json").read_text())
    client = Trellis2Client()

    results = []
    for entry in manifest:
        obj_id = entry["id"]
        t0 = time.time()
        try:
            image = Image.open(args.manifest_dir / entry["crop_path"]).convert("RGBA")
            mesh = client.generate(image)
            # mesh.vertices is a torch.Tensor here (TRELLIS.2 keeps it on-device
            # until `export_glb`'s o_voxel call) - torch's .min(axis=...)/.max(axis=...)
            # return a `torch.return_types.min`/`max` named tuple (values, indices),
            # not a plain tensor, so a bare `.tolist()` raises AttributeError. Convert
            # to numpy first.
            verts_np = mesh.vertices.detach().cpu().numpy() if hasattr(mesh.vertices, "detach") else mesh.vertices
            gen_min = verts_np.min(axis=0).tolist()
            gen_max = verts_np.max(axis=0).tolist()
            ratio = containment_ratio(gen_min, gen_max, entry["bbox_min"], entry["bbox_max"])
            accepted = passes_containment(ratio)
            glb_path = None
            if accepted:
                glb_path = args.manifest_dir / f"{obj_id}.glb"
                client.export_glb(mesh, glb_path)
            elapsed = time.time() - t0
            result = {
                "id": obj_id,
                "label": entry["label"],
                "accepted": accepted,
                "containment_ratio": ratio,
                "generated_bbox_min": gen_min,
                "generated_bbox_max": gen_max,
                "glb_path": glb_path.name if glb_path else None,
                "elapsed_s": elapsed,
                "error": None,
            }
            print(f"{'ACCEPT' if accepted else 'REJECT'} {obj_id}: ratio={ratio:.3f} elapsed={elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "id": obj_id,
                "label": entry["label"],
                "accepted": False,
                "containment_ratio": None,
                "generated_bbox_min": None,
                "generated_bbox_max": None,
                "glb_path": None,
                "elapsed_s": elapsed,
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"ERROR {obj_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
        results.append(result)
        (args.manifest_dir / "results.json").write_text(json.dumps(results, indent=2))

    n_accept = sum(1 for r in results if r["accepted"])
    print(f"DONE: {n_accept}/{len(results)} accepted")


if __name__ == "__main__":
    main()
