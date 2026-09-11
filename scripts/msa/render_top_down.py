"""Stage A7 (renders) - DEVIATION FROM SPEC, documented here and in
docs/DECISIONS.md: SPEC.md A7 asks for Blender `bpy` headless renders (top-down
60deg, eye-level, close-up, plus a dimensioned `06_gaps_top.png`). This VPS has
no Blender/bpy install and getting one working headless was out of budget for
this pass (Stage A0, 2.5h wall-clock cap) - see the budget note in SPEC.md §12
("dominated by weights/GPU/limits rather than coding"), which this VPS doesn't
have GPU access on to install/verify a real Blender headless render anyway.

What this module DOES produce: a pure-Python (PIL, no Blender) strict top-down
2D orthographic render of walls + object footprints + gap measurement points
colored by verdict - functionally equivalent to SPEC A7's `06_gaps_top.png`,
just not rendered through Blender/Cycles. The angled top-down (60deg), eye-level,
and close-up 3D renders from SPEC A7 are NOT produced by this module - those
need an actual 3D renderer with lighting, which PIL cannot do. This is a
one-view stand-in for the four-render set A7 asks for, not full compliance.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from scripts.msa.gaps import Gap
from scripts.msa.geometry import WallPolygon


def render_gaps_top_down(
    walls: list[WallPolygon],
    objects: list[dict],
    gaps: list[Gap],
    out_path: Path,
    *,
    pixels_per_meter: float = 80.0,
    margin_px: int = 40,
    room_polygon: list[tuple[float, float]] | None = None,
    room_outline_rgb: tuple[int, int, int] = (30, 90, 220),
    room_outline_width_px: int = 2,
    wall_band=None,
) -> None:
    """`room_polygon` (T15a: `bootstrap.run_bootstrap`'s already-rotated room
    reference polygon), if given, is drawn as a `room_outline_width_px`-wide
    outline in `room_outline_rgb` on top of everything else and counts toward
    the image extent, so the room's orientation is visible in the render.
    `wall_band` (T15g): the single wall band (shapely Polygon/MultiPolygon from
    `geometry.room_wall_band`); when given it is drawn INSTEAD of the raw wall
    fragments, which stay data-only (gaps.json, Plan curves)."""
    all_x, all_z = [], []
    band_rings: list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]] = []
    if wall_band is not None and not wall_band.is_empty:
        for part in getattr(wall_band, "geoms", [wall_band]):
            if part.geom_type != "Polygon":
                continue
            ext = [(float(x), float(z)) for x, z in part.exterior.coords]
            holes = [[(float(x), float(z)) for x, z in ring.coords] for ring in part.interiors]
            band_rings.append((ext, holes))
            for x, z in ext:
                all_x.append(x)
                all_z.append(z)
    else:
        for wall in walls:
            for x, z in wall.vertices:
                all_x.append(x)
                all_z.append(z)
    for x, z in room_polygon or []:
        all_x.append(x)
        all_z.append(z)
    for obj in objects:
        cx, cz = obj["center_xy"]
        length_u, width_v = obj["size_uv"]
        r = (length_u**2 + width_v**2) ** 0.5 / 2
        all_x += [cx - r, cx + r]
        all_z += [cz - r, cz + r]
    if not all_x:
        return

    min_x, max_x = min(all_x), max(all_x)
    min_z, max_z = min(all_z), max(all_z)
    width_px = int((max_x - min_x) * pixels_per_meter) + 2 * margin_px
    height_px = int((max_z - min_z) * pixels_per_meter) + 2 * margin_px

    def to_px(x: float, z: float) -> tuple[float, float]:
        return (margin_px + (x - min_x) * pixels_per_meter, margin_px + (z - min_z) * pixels_per_meter)

    img = Image.new("RGB", (max(width_px, 1), max(height_px, 1)), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    if band_rings:
        for ext, holes in band_rings:
            draw.polygon([to_px(x, z) for x, z in ext], outline=(30, 30, 30), fill=(90, 90, 90))
            for hole in holes:
                draw.polygon([to_px(x, z) for x, z in hole], outline=(30, 30, 30), fill=(245, 245, 245))
    else:
        for wall in walls:
            pts = [to_px(x, z) for x, z in wall.vertices]
            draw.polygon(pts, outline=(30, 30, 30), fill=(90, 90, 90))

    import math

    for obj in objects:
        cx, cz = obj["center_xy"]
        length_u, width_v = obj["size_uv"]
        angle = obj["angle_rad"]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        u, v = length_u / 2, width_v / 2
        local_corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
        world_corners = [(cx + lx * cos_a - lz * sin_a, cz + lx * sin_a + lz * cos_a) for lx, lz in local_corners]
        pts = [to_px(x, z) for x, z in world_corners]
        r, g, b = obj.get("color_rgb", (150, 150, 150))
        draw.polygon(pts, outline=(0, 0, 0), fill=(r, g, b))

    for gap in gaps:
        px, py = to_px(*gap.measurement_point_xy)
        all_pass = all(v.fits for v in gap.platform_verdicts) if gap.platform_verdicts else None
        color = (0, 160, 0) if all_pass else (200, 0, 0) if all_pass is False else (200, 160, 0)
        radius = 4
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)

    if room_polygon and len(room_polygon) >= 2:
        ring = list(room_polygon)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        draw.line([to_px(x, z) for x, z in ring], fill=room_outline_rgb, width=room_outline_width_px)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
