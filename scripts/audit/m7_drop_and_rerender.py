"""M7 audit item 3/5: drop bogus detections (`desk_0` = a door-frame fragment
mis-tagged as a desk, `television_1` = a wall mirror mis-tagged as a TV - see
progress/13c_m7_audit.md) from the already-assembled, already-textured hero
GLB and re-export a web GLB + top/3-4 screenshots for comparison, without
re-running TRELLIS/asset-fitting for the 16 objects that are untouched.

Also writes `m7_out/objects_audited.json` (objects.json filtered to the 16
kept ids, for provenance / any future full re-assembly) - `m7_out/objects.json`
itself is left untouched per the task's explicit instruction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import trimesh

M7_OUT = Path("var/scratch/run-20260906/m7_out")
DROP_IDS = ["desk_0", "television_1"]


def write_audit_drop_ids() -> None:
    reason = {
        "desk_0": "SAM3/point-cloud fragment (88 points, 0.766x0.100x0.246 m) near a door frame "
                  "(hero camera view_025) - the projected crop shows a door edge, no desk visible "
                  "anywhere in the source frame. Not a real desk.",
        "television_1": "Projected crop (view_018) shows a wall-mounted MIRROR next to a wall lamp "
                         "above a kettle console, not a television screen - the bright rectangle is "
                         "the mirror's reflection, not a display. 2905 points, support reason wall. "
                         "Also the object nearest the doorway alcove among the two TV detections "
                         "(item 5): center_xy (1.568, -1.556) vs television_0's (2.301, -1.807).",
    }
    out = {"drop_ids": DROP_IDS, "reason": reason}
    (M7_OUT / "audit_drop_ids.json").write_text(json.dumps(out, indent=2))
    print("wrote", M7_OUT / "audit_drop_ids.json")


def write_objects_audited() -> None:
    data = json.loads((M7_OUT / "objects.json").read_text())
    kept = [o for o in data["objects"] if o["id"] not in DROP_IDS]
    dropped = [o["id"] for o in data["objects"] if o["id"] in DROP_IDS]
    new_data = dict(data)
    new_data["objects"] = kept
    (M7_OUT / "objects_audited.json").write_text(json.dumps(new_data, indent=1))
    print(f"objects_audited.json: {len(kept)} kept, dropped {dropped}")


def rerender_web_glb(out_dir: Path) -> Path:
    """Removes the dropped objects' node subtrees from the already-textured
    scene_assets_textured.glb (the file the T15c `?glb=` web viewer loads) and
    re-exports it - this keeps the 16 untouched objects byte-identical to the
    accepted M7 render instead of re-running asset fitting/texturing."""
    src = M7_OUT / "scene_assets_textured.glb"
    scene = trimesh.load(src, process=False)

    to_delete = []
    for node_name in scene.graph.nodes:
        if any(node_name == did or node_name.startswith(f"{did}/") for did in DROP_IDS):
            transform, geom_name = scene.graph.get(node_name) if node_name in scene.graph.nodes else (None, None)
            if geom_name:
                to_delete.append(geom_name)
    to_delete = sorted(set(to_delete))
    print("deleting geometries:", to_delete)
    scene.delete_geometry(to_delete)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_assets_textured_audited.glb"
    scene.export(out_path)
    print("wrote", out_path)
    return out_path


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else M7_OUT
    write_audit_drop_ids()
    write_objects_audited()
    rerender_web_glb(out_dir)
