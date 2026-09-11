"""T15f: strict top-down orthographic render of an exported MSA GLB (walls,
floor, object visuals) - pure Python/PIL like `render_perspective`, no GPU.
Painter's algorithm by triangle height (highest last), flat colour per mesh
via `export_glb.mesh_base_color_rgb` (vertex colours / PBR base colour /
mean texture colour); collision hulls and Plan curves are skipped, walls are
drawn translucent so the room interior stays readable.

    uv run python -m scripts.msa.render_top_ortho --scene-glb <scene.glb> --out <png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from scripts.msa.export_glb import mesh_base_color_rgb

DEFAULT_OBJECT_RGB = (150, 150, 150)
FLOOR_RGB = (225, 222, 214)
WALL_RGB = (90, 90, 90)
WALL_ALPHA = 110
BACKGROUND_RGB = (255, 255, 255)


def _node_triangles(scene: trimesh.Scene):
    for node_name in scene.graph.nodes_geometry:
        if "/collision/" in node_name or node_name.startswith("Plan"):
            continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        verts = trimesh.transform_points(geom.vertices, transform)
        yield node_name, geom, verts[geom.faces]


def render_top_ortho(scene_glb: Path, out_path: Path, *, pixels_per_meter: float = 120.0, margin_px: int = 30) -> dict:
    loaded = trimesh.load(Path(scene_glb).as_posix())
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    nodes = list(_node_triangles(scene))
    if not nodes:
        raise ValueError(f"{scene_glb}: nothing to render")
    all_pts = np.concatenate([tris.reshape(-1, 3) for _n, _g, tris in nodes])
    xmin, zmin = all_pts[:, 0].min(), all_pts[:, 2].min()
    xmax, zmax = all_pts[:, 0].max(), all_pts[:, 2].max()
    width = int(np.ceil((xmax - xmin) * pixels_per_meter)) + 2 * margin_px
    height = int(np.ceil((zmax - zmin) * pixels_per_meter)) + 2 * margin_px

    def to_px(x, z):
        return (margin_px + (x - xmin) * pixels_per_meter, margin_px + (z - zmin) * pixels_per_meter)

    opaque: list[tuple[float, list, tuple]] = []
    walls: list[tuple[float, list, tuple]] = []
    for node_name, geom, tris in nodes:
        if node_name.startswith("wall_"):
            rgba = (*WALL_RGB, WALL_ALPHA)
            bucket = walls
        elif node_name == "floor":
            rgba = (*FLOOR_RGB, 255)
            bucket = opaque
        else:
            base = mesh_base_color_rgb(geom) or DEFAULT_OBJECT_RGB
            rgba = (*base, 255)
            bucket = opaque
        for tri in tris:
            bucket.append((float(tri[:, 1].max()), [to_px(p[0], p[2]) for p in tri], rgba))

    canvas = Image.new("RGBA", (width, height), (*BACKGROUND_RGB, 255))
    draw = ImageDraw.Draw(canvas)
    opaque.sort(key=lambda e: e[0])
    for _y, pts, rgba in opaque:
        draw.polygon(pts, fill=rgba)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for _y, pts, rgba in walls:
        odraw.polygon(pts, fill=(rgba[0], rgba[1], rgba[2], 255))
    arr = np.asarray(overlay).astype(np.float32)
    arr[..., 3] *= WALL_ALPHA / 255.0
    canvas.alpha_composite(Image.fromarray(arr.astype(np.uint8), "RGBA"))
    # +Z is down in the image (looking down -Y from above with +X right); flip so
    # the view matches render_gaps_top_down's convention (Z up the page).
    canvas = canvas.transpose(Image.FLIP_TOP_BOTTOM)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    return {"scene_glb": str(scene_glb), "out_path": str(out_path), "size_px": [width, height], "n_nodes": len(nodes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixels-per-meter", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(render_top_ortho(args.scene_glb, args.out, pixels_per_meter=args.pixels_per_meter), indent=2))


if __name__ == "__main__":
    main()
