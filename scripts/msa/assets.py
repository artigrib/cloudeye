"""Stage A4: placeholder parametric primitives by object class. SPEC A4: bed =
mattress + headboard; table = top + 4 legs; chair = seat + back; sofa = seat +
back + arms; rug/pillow/blanket = rounded box; unknown = box. Colour = mean
colour of the object's points (passed in by the caller, not computed here).

Each builder returns a list of (trimesh.Trimesh, local_transform) parts already
scaled/positioned to fill a footprint of size (length_u, width_v, height) centred
at the origin on the local XZ plane with Y up, height starting at y=0.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _box(length: float, width: float, height: float, *, center_xz: tuple[float, float] = (0.0, 0.0), y0: float = 0.0) -> trimesh.Trimesh:
    length = max(length, 0.01)
    width = max(width, 0.01)
    height = max(height, 0.01)
    mesh = trimesh.creation.box(extents=(length, height, width))
    mesh.apply_translation((center_xz[0], y0 + height / 2.0, center_xz[1]))
    return mesh


def build_placeholder(label: str, length_u: float, width_v: float, height: float) -> list[trimesh.Trimesh]:
    """`length_u`/`width_v` are the footprint's oriented rectangle sides (local U/V,
    U mapped to local X and V to local Z before the caller's world transform is
    applied), `height` is the object's measured bbox height."""
    label = label.lower()
    if label == "bed":
        mattress_h = height * 0.55
        headboard_h = height
        headboard_depth = min(0.08, length_u * 0.15)
        return [
            _box(length_u - headboard_depth, width_v, mattress_h, center_xz=(headboard_depth / 2, 0.0)),
            _box(headboard_depth, width_v, headboard_h, center_xz=(-(length_u - headboard_depth) / 2, 0.0)),
        ]
    if label == "table" or label == "desk":
        top_h = max(0.04, height * 0.08)
        leg_h = height - top_h
        leg_w = min(0.05, length_u * 0.08, width_v * 0.08)
        parts = [_box(length_u, width_v, top_h, y0=leg_h)]
        for sx in (-1, 1):
            for sz in (-1, 1):
                cx = sx * (length_u / 2 - leg_w / 2)
                cz = sz * (width_v / 2 - leg_w / 2)
                leg = _box(leg_w, leg_w, leg_h, center_xz=(cx, cz))
                parts.append(leg)
        return parts
    if label == "chair":
        seat_h = height * 0.45
        back_h = height - seat_h
        back_depth = min(0.05, length_u * 0.2)
        return [
            _box(length_u, width_v, seat_h),
            _box(back_depth, width_v, back_h, center_xz=(-(length_u - back_depth) / 2, 0.0), y0=seat_h),
        ]
    if label == "sofa":
        seat_h = height * 0.45
        back_h = height - seat_h
        back_depth = min(0.15, length_u * 0.2)
        arm_w = min(0.12, width_v * 0.15)
        parts = [
            _box(length_u, width_v - 2 * arm_w, seat_h, center_xz=(0.0, 0.0)),
            _box(back_depth, width_v, back_h, center_xz=(-(length_u - back_depth) / 2, 0.0), y0=seat_h),
        ]
        for sz in (-1, 1):
            cz = sz * (width_v / 2 - arm_w / 2)
            parts.append(_box(length_u, arm_w, height, center_xz=(0.0, cz)))
        return parts
    if label in ("rug", "pillow", "blanket"):
        return [_box(length_u, width_v, min(height, 0.08))]
    return [_box(length_u, width_v, height)]


def mean_color_uint8(rgb: np.ndarray) -> tuple[int, int, int]:
    if rgb.size == 0:
        return (128, 128, 128)
    mean = rgb.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))
