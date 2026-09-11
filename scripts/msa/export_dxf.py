"""Stage A6 (DXF + SVG floor plan): walls, object footprints with class labels,
and passage dimension lines with per-platform verdict text, 1:1 metres, via
ezdxf (SPEC A6)."""

from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend, layout
from ezdxf.addons.drawing.svg import SVGBackend

from scripts.msa.gaps import Gap
from scripts.msa.geometry import WallPolygon


def build_floor_plan(
    walls: list[WallPolygon],
    objects: list[dict],
    gaps: list[Gap],
) -> "ezdxf.document.Drawing":
    doc = ezdxf.new("R2018", setup=True)
    msp = doc.modelspace()
    doc.layers.add("WALLS", color=7)
    doc.layers.add("OBJECTS", color=3)
    doc.layers.add("GAPS_PASS", color=3)
    doc.layers.add("GAPS_FAIL", color=1)
    doc.layers.add("LABELS", color=2)

    for wall in walls:
        msp.add_lwpolyline(wall.vertices, close=True, dxfattribs={"layer": "WALLS"})

    for obj in objects:
        cx, cz = obj["center_xy"]
        length_u, width_v = obj["size_uv"]
        angle = obj["angle_rad"]
        import math

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        u, v = length_u / 2, width_v / 2
        local_corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
        world_corners = [(cx + lx * cos_a - lz * sin_a, cz + lx * sin_a + lz * cos_a) for lx, lz in local_corners]
        msp.add_lwpolyline(world_corners, close=True, dxfattribs={"layer": "OBJECTS"})
        msp.add_text(
            obj["label"],
            height=0.08,
            dxfattribs={"layer": "LABELS", "insert": (cx, cz)},
        ).set_placement((cx, cz))

    for gap in gaps:
        all_pass = all(v.fits for v in gap.platform_verdicts) if gap.platform_verdicts else None
        layer = "GAPS_PASS" if all_pass else "GAPS_FAIL" if all_pass is not None else "GAPS_PASS"
        px, py = gap.measurement_point_xy
        verdict_text = f"{gap.a_id}-{gap.b_id}: {gap.width_m:.2f}m"
        if gap.platform_verdicts:
            verdict_text += " (" + ", ".join(f"{v.platform_id}:{'OK' if v.fits else 'FAIL'}" for v in gap.platform_verdicts) + ")"
        msp.add_circle((px, py), radius=0.03, dxfattribs={"layer": layer})
        msp.add_text(verdict_text, height=0.06, dxfattribs={"layer": layer, "insert": (px, py + 0.05)}).set_placement((px, py + 0.05))

    return doc


def export_dxf(doc, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path.as_posix())


def export_svg(doc, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msp = doc.modelspace()
    context = RenderContext(doc)
    backend = SVGBackend()
    frontend = Frontend(context, backend)
    frontend.draw_layout(msp, finalize=True)
    page = layout.Page(0, 0, layout.Units.mm, margins=layout.Margins.all(0))
    out_path.write_text(backend.get_string(page))
