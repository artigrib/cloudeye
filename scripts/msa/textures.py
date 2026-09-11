"""T15c / T15c-prep "dollhouse" textures: bake floor/wall base-colour textures
from the aligned RGB point cloud (`aligned_room.ply`) and attach them to the MSA
room meshes - the GLB's `floor` / `wall_*` nodes as glTF PBR materials
(`apply_room_textures`) and the USD's `/World/Floor` + `/World/Structure/**`
meshes as `UsdPreviewSurface` materials (`apply_usd_textures`) - so the web
viewer (`frontend/src/components/MsaSceneViewer.tsx`) and Isaac show a lit,
textured room instead of flat gray boxes.

Coordinate frame: the cloud is in the *pre-yaw* scene frame (Y up, floor at
`floor_y`). Bootstrap (`scripts/msa/bootstrap.py`, yaw-normalization post-step)
rigidly rotates walls/floor/objects about `yaw_rotation_center_xy` by
`yaw_correction_rad` (standard CCW `rotate_point_xz` convention: x' = cx + dx*cos
- dz*sin, z' = cz + dx*sin + dz*cos) before exporting. `bake_room_textures`
applies the same rotation to the cloud so baked pixels land where the exported
geometry is - pass the two `scene_meta.json` keys through (see `yaw_from_meta`).
`--cloud-frame auto|identity|swap_xz` (T15c) additionally guards against a
bootstrap output produced before T15b's occupancy [ix, iz] fix; on a post-T15b
output `auto` resolves to identity.

Texture spaces (T15c-prep generalisation):
- floor (`bake_floor_texture`): orthographic top-down splat of points within
  `band_m` of `floor_y` onto the ARBITRARY floor polygon's XZ bounding box
  (padded by `pad_m`; the AABB is the UV domain). Image row = +Z, column = +X.
  Pixels outside the polygon are masked to a neutral tone (the far-field blur
  of the observed floor - `FloorTexture.mask` says which); the floor plate mesh
  (room polygon + object footprints) can extend a little past the polygon.
  `FloorTexture.uv_for_xz` maps world XZ -> UV so ANY floor mesh can be UV'd.
- wall band (`bake_wall_band_texture`): a band is `(segment_xz, y0, height)` -
  the vertical rectangle above a 2D segment. Points within `band_m` of the
  vertical plane through the segment and within the segment's extent (+/-
  `margin_m`) are splatted in (along-segment s, height y): column = +s from p0
  to p1, row 0 = top (upright image).
- wall atlas (`bake_wall_atlas` / `bake_wall_outline_textures`): several bands
  in ONE image, laid out left to right at a common px/m, each band owning the
  u-range [u0, u1] of its pixel columns, plus a `CAP_SWATCH_PX`-wide flat-colour
  column at the right edge for the extrusion's top/bottom cap faces. A closed
  outline polygon becomes one atlas with one band per edge (edge i = outline[i]
  -> outline[i+1]). ONE atlas rather than one texture per edge because a single
  `wall_outline` mesh is one glTF primitive (one material) and one USD Mesh
  (one material binding) - a per-edge texture would need GeomSubsets in USD and
  a split mesh in glTF. UVs are authored per FACE-VERTEX: every side face is
  assigned to the band whose vertical plane it lies in (`WallAtlas.
  face_band_index` - by geometry, never by prim/node name), so the same code
  textures the current per-fragment `wall_i` boxes (one band each = the PCA
  frame T15c used, see `band_from_wall_frame`) and T15g's single closed band
  extruded from the regularized outline (`/World/Structure/wall_outline`,
  bands = outline edges). In glTF the wall mesh is unwelded first
  (`Trimesh.unmerge_vertices`) so the per-face UV seams at corners survive;
  in USD `primvars:st` is authored `faceVarying`.

Holes (no points) are filled with the nearest baked pixel within
`fill_radius_m`, farther pixels fade to a wide Gaussian average of the nearest
fill; a 3x3 median + light blur then runs on the FILLED pixels only (observed
pixels are left as splatted).

UV convention: trimesh's `TextureVisuals` uses OpenGL (v=0 at the image's
*bottom* row); its glTF exporter flips to glTF's top-left origin on write
(verified: trimesh 5.1 writes `1 - v` into TEXCOORD_0). USD's `UsdUVTexture`
also samples with (0, 0) at the image's bottom-left, so the SAME UVs are
authored as `primvars:st`. `floor_uv` / `WallAtlas.uv_for_faces` produce these.

CLI: `python -m scripts.msa.textures --scene-dir <scene> --bootstrap-out <out>
--out <dir> [--usd scene_presentation.usd]` re-textures an existing bootstrap
output: `<dir>/scene_textured.glb`, `<dir>/scene_textured.usd` (a copy of the
input USD with materials bound), PNGs and `textures.json`. In-pipeline hooks:
`export_glb.build_scene(..., textures=)`, `export_usd.export_usd(...,
textures=)`, `export_usd.export_presentation_usd(..., textures=)`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.msa.geometry import WallPolygon

DEFAULT_BAND_M = 0.15
DEFAULT_PX_PER_M = 64.0  # ~1.6 cm/px
DEFAULT_MAX_PX = 1024
DEFAULT_MAX_ATLAS_PX = 4096
MIN_PX = 8
FLOOR_PAD_M = 0.25
OBJECT_ROUGHNESS = 0.8
ROOM_ROUGHNESS = 0.9
DEFAULT_FILL_RADIUS_M = 0.30  # nearest-neighbour fill reach; farther holes take the far-field tone
DEFAULT_MAX_POINTS = 2_500_000  # hero cloud is 10 M points / 271 MB - a 2.5 M subsample bakes the same textures
DEFAULT_SEGMENT_MARGIN_M = 0.05
AUTO_TARGET_POINTS_PER_PX = 1.5
AUTO_MIN_PX_PER_M = 16.0
CAP_SWATCH_PX = 4
CAP_NORMAL_Y_MIN = 0.5  # |normal.y| above this -> top/bottom cap face
USD_LOOKS_ROOT = "/World/Looks"


# --------------------------------------------------------------------------- frame


def rotate_points_xz(xyz: np.ndarray, angle_rad: float, center_xz: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Vectorized twin of `geometry.rotate_point_xz` (same CCW convention, same
    pivot semantics) for an (N,3) XYZ array - Y untouched. Returns a copy."""
    xyz = np.asarray(xyz, dtype=float)
    out = xyz.copy()
    if abs(angle_rad) < 1e-12:
        return out
    cx, cz = center_xz
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx = xyz[:, 0] - cx
    dz = xyz[:, 2] - cz
    out[:, 0] = cx + dx * cos_a - dz * sin_a
    out[:, 2] = cz + dx * sin_a + dz * cos_a
    return out


def swap_xz_transform(origin_x: float = 0.0, origin_z: float = 0.0) -> np.ndarray:
    """4x4 that maps a cloud point to the frame a bootstrap run lands in when it
    reads `occupancy.npy` as `[row=z, col=x]` while the file is actually laid
    out `[ix, iz]` (`occupancy_meta.json` width x height): X' = z + (ox - oz),
    Z' = x + (oz - ox), Y unchanged. Found on the hero scene 2026-09-06 (walls /
    floor polygon transposed relative to the cloud and to the object hulls,
    which bootstrap computes directly from `scene_objects/*.ply`). A mirror,
    not a rotation - only ever applied to the *cloud* so baked textures land
    on the exported floor/wall meshes; the geometry read itself was fixed by
    T15b, so on current outputs `--cloud-frame auto` picks identity (see
    docs/DECISIONS.md, T15c and T15b entries)."""
    t = np.eye(4)
    t[0, 0], t[0, 2] = 0.0, 1.0
    t[2, 0], t[2, 2] = 1.0, 0.0
    t[0, 3] = origin_x - origin_z
    t[2, 3] = origin_z - origin_x
    return t


def apply_transform_xyz(xyz: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return np.asarray(xyz, dtype=float)
    t = np.asarray(transform, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    return xyz @ t[:3, :3].T + t[:3, 3]


def yaw_from_meta(scene_meta: dict) -> tuple[float, tuple[float, float]]:
    """`(angle_rad, center_xz)` the bootstrap applied, from a bootstrap-output
    `scene_meta.json` dict. Pre-yaw bootstraps (no keys / `yaw_applied` False)
    -> identity."""
    if not scene_meta.get("yaw_applied"):
        return 0.0, (0.0, 0.0)
    angle = float(scene_meta.get("yaw_correction_rad") or 0.0)
    center = scene_meta.get("yaw_rotation_center_xy") or (0.0, 0.0)
    return angle, (float(center[0]), float(center[1]))


@dataclass(frozen=True)
class WallFrame:
    """A wall polygon's along-wall texture frame in world XZ (T15c; a
    per-fragment wall is one `WallBand` along this frame, see
    `band_from_wall_frame`)."""

    origin_xz: tuple[float, float]
    dir_xz: tuple[float, float]  # unit, along the wall
    normal_xz: tuple[float, float]  # unit, across the wall
    length_m: float
    half_thickness_m: float

    def to_json(self) -> dict:
        return {
            "origin_xz": list(self.origin_xz),
            "dir_xz": list(self.dir_xz),
            "normal_xz": list(self.normal_xz),
            "length_m": self.length_m,
            "half_thickness_m": self.half_thickness_m,
        }

    @classmethod
    def from_json(cls, d: dict) -> "WallFrame":
        return cls(
            origin_xz=tuple(d["origin_xz"]),
            dir_xz=tuple(d["dir_xz"]),
            normal_xz=tuple(d["normal_xz"]),
            length_m=float(d["length_m"]),
            half_thickness_m=float(d["half_thickness_m"]),
        )


def wall_frame(vertices_xz) -> WallFrame:
    """Principal-axis frame of a wall footprint (any vertex order - PCA over the
    unique vertices). `origin_xz` is the polygon's extreme point along `dir_xz`
    so along-wall coordinate s runs [0, length_m]."""
    pts = np.unique(np.asarray(vertices_xz, dtype=float).reshape(-1, 2), axis=0)
    center = pts.mean(axis=0)
    d = pts - center
    if len(pts) >= 2:
        cov = d.T @ d
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
    else:
        axis = np.array([1.0, 0.0])
    if abs(axis[0]) < 1e-12 and abs(axis[1]) < 1e-12:
        axis = np.array([1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    # Deterministic sign: +X-ish (then +Z-ish) so re-baking gives the same image.
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    s = d @ axis
    t = d @ normal
    s_min, s_max = float(s.min()), float(s.max())
    origin = center + axis * s_min
    return WallFrame(
        origin_xz=(float(origin[0]), float(origin[1])),
        dir_xz=(float(axis[0]), float(axis[1])),
        normal_xz=(float(normal[0]), float(normal[1])),
        length_m=max(s_max - s_min, 1e-3),
        half_thickness_m=float(np.abs(t).max()) if len(t) else 0.0,
    )


# --------------------------------------------------------------------------- wall bands


@dataclass(frozen=True)
class WallBand:
    """The vertical rectangle above the 2D segment p0 -> p1 spanning
    [y0, y0 + height]. Texture u runs 0 at p0 to 1 at p1, v 0 at y0 to 1 at the top."""

    p0_xz: tuple[float, float]
    p1_xz: tuple[float, float]
    y0: float
    height: float

    @property
    def length_m(self) -> float:
        return max(math.hypot(self.p1_xz[0] - self.p0_xz[0], self.p1_xz[1] - self.p0_xz[1]), 1e-6)

    @property
    def dir_xz(self) -> tuple[float, float]:
        L = self.length_m
        return ((self.p1_xz[0] - self.p0_xz[0]) / L, (self.p1_xz[1] - self.p0_xz[1]) / L)

    @property
    def normal_xz(self) -> tuple[float, float]:
        dx, dz = self.dir_xz
        return (-dz, dx)

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(s along the segment in metres from p0, t signed perpendicular offset)."""
        xyz = np.asarray(xyz, dtype=float)
        dx = xyz[:, 0] - self.p0_xz[0]
        dz = xyz[:, 2] - self.p0_xz[1]
        d, n = self.dir_xz, self.normal_xz
        return dx * d[0] + dz * d[1], dx * n[0] + dz * n[1]

    def uv_for_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Band-local UVs (trimesh/USD convention, clipped to [0, 1])."""
        xyz = np.asarray(xyz, dtype=float)
        s, _ = self.project(xyz)
        u = np.clip(s / self.length_m, 0.0, 1.0)
        v = np.clip((xyz[:, 1] - self.y0) / max(self.height, 1e-9), 0.0, 1.0)
        return np.column_stack([u, v])

    def to_json(self) -> dict:
        return {"p0_xz": list(self.p0_xz), "p1_xz": list(self.p1_xz), "y0": self.y0, "height": self.height}

    @classmethod
    def from_json(cls, d: dict) -> "WallBand":
        return cls(p0_xz=(float(d["p0_xz"][0]), float(d["p0_xz"][1])), p1_xz=(float(d["p1_xz"][0]), float(d["p1_xz"][1])), y0=float(d["y0"]), height=float(d["height"]))


def band_from_wall_frame(frame: WallFrame, y0: float, height: float) -> WallBand:
    """The single band of a per-fragment wall: its PCA axis from origin to
    origin + length (what T15c splatted along)."""
    ox, oz = frame.origin_xz
    dx, dz = frame.dir_xz
    return WallBand(p0_xz=(ox, oz), p1_xz=(ox + dx * frame.length_m, oz + dz * frame.length_m), y0=float(y0), height=float(height))


def outline_bands(outline_xz, y0: float, height: float, *, closed: bool = True, min_length_m: float = 0.01) -> list[WallBand]:
    """One band per edge of a polyline/polygon: edge i = outline[i] -> outline[i+1]
    (and the closing edge when `closed`). Degenerate edges are skipped; a
    repeated closing vertex is tolerated."""
    pts = [(float(x), float(z)) for x, z in np.asarray(outline_xz, dtype=float).reshape(-1, 2)]
    if len(pts) >= 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-9:
        pts = pts[:-1]
    n = len(pts)
    if n < 2:
        return []
    pairs = [(pts[i], pts[(i + 1) % n]) for i in range(n if closed else n - 1)]
    return [WallBand(a, b, float(y0), float(height)) for a, b in pairs if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_length_m]


# --------------------------------------------------------------------------- baking primitives


def _texture_size(extent_m: float, px_per_m: float, max_px: int) -> int:
    return int(min(max(MIN_PX, round(extent_m * px_per_m)), max_px))


def auto_px_per_m(n_points: int, area_m2: float, *, target_pts_per_px: float = AUTO_TARGET_POINTS_PER_PX, lo: float = AUTO_MIN_PX_PER_M, hi: float = DEFAULT_PX_PER_M) -> float:
    """Pixel density such that a surface gets ~`target_pts_per_px` cloud points
    per texel (sparser -> mostly hole fill = blur; denser -> wasted pixels),
    clamped to [lo, hi]."""
    if area_m2 <= 1e-9 or n_points <= 0:
        return float(lo)
    return float(min(hi, max(lo, math.sqrt(n_points / (area_m2 * target_pts_per_px)))))


def _splat(u_px: np.ndarray, v_px: np.ndarray, rgb: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean colour per pixel via bincount. Returns (image HxWx3 float, hit mask)."""
    u = np.clip(u_px.astype(np.int64), 0, width - 1)
    v = np.clip(v_px.astype(np.int64), 0, height - 1)
    idx = v * width + u
    n = width * height
    counts = np.bincount(idx, minlength=n).astype(float)
    img = np.zeros((n, 3), dtype=float)
    rgb = np.asarray(rgb, dtype=float)
    for c in range(3):
        img[:, c] = np.bincount(idx, weights=rgb[:, c], minlength=n)
    hit = counts > 0
    img[hit] /= counts[hit, None]
    return img.reshape(height, width, 3), hit.reshape(height, width)


def _far_field(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy import ndimage

    return np.stack([ndimage.gaussian_filter(img[..., c], sigma=max(sigma, 1.0), mode="nearest") for c in range(3)], axis=-1)


def _fill_holes(img: np.ndarray, hit: np.ndarray, *, fill_radius_px: float = 12.0, far_px: float | None = None, median_size: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Hole fill. Empty pixels within `fill_radius_px` of data take the nearest
    baked pixel (keeps edges crisp); beyond `far_px` (default 4x the radius)
    they fade to a wide Gaussian average of the nearest-filled image (a smooth
    tone instead of long nearest-neighbour streaks across big unobserved areas,
    e.g. floor under the bed or the padded bbox outside the room). A 3x3 median
    + light blur then run on the FILLED pixels only - observed pixels keep their
    splatted colour. Returns (image, filled mask). No data at all -> flat mid
    gray."""
    from scipy import ndimage

    if not hit.any():
        return np.full(img.shape, 128.0), ~hit
    if hit.all():
        return img, ~hit
    far_px = float(far_px if far_px is not None else max(4.0 * fill_radius_px, fill_radius_px + 1.0))
    dist, (iy, ix) = ndimage.distance_transform_edt(~hit, return_indices=True)
    nearest = img[iy, ix]
    smooth = _far_field(nearest, far_px / 2.0)
    w = np.clip((dist - fill_radius_px) / max(far_px - fill_radius_px, 1e-6), 0.0, 1.0)[..., None]
    out = nearest * (1.0 - w) + smooth * w
    filled = ~hit
    if median_size and median_size > 1:
        med = np.stack([ndimage.median_filter(out[..., c], size=median_size) for c in range(3)], axis=-1)
        soft = np.stack([ndimage.gaussian_filter(med[..., c], sigma=1.0, mode="nearest") for c in range(3)], axis=-1)
        out = np.where(filled[..., None], soft, out)
    return out, filled


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _as_shapely_polygon(floor_polygon):
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.geometry.base import BaseGeometry

    if isinstance(floor_polygon, BaseGeometry):
        poly = floor_polygon
    else:
        poly = Polygon(np.asarray(floor_polygon, dtype=float).reshape(-1, 2))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        parts = [p for p in poly.geoms if p.area > 1e-9]
        poly = max(parts, key=lambda p: p.area) if parts else poly
    return poly


def polygon_mask(floor_polygon, extent_xz, width: int, height: int) -> np.ndarray:
    """HxW bool: pixel centre inside the polygon (image row = +Z, col = +X)."""
    from shapely import contains_xy

    poly = _as_shapely_polygon(floor_polygon)
    xmin, zmin, xmax, zmax = extent_xz
    xs = xmin + (np.arange(width) + 0.5) / width * (xmax - xmin)
    zs = zmin + (np.arange(height) + 0.5) / height * (zmax - zmin)
    gx, gz = np.meshgrid(xs, zs)
    return contains_xy(poly, gx.ravel(), gz.ravel()).reshape(height, width)


# --------------------------------------------------------------------------- floor


@dataclass
class FloorTexture:
    image: np.ndarray  # HxWx3 uint8, row = +Z, col = +X; neutral outside the polygon
    extent_xz: tuple[float, float, float, float]  # xmin, zmin, xmax, zmax = the UV domain
    n_points: int
    coverage: float  # fraction of in-polygon pixels hit by at least one point before hole fill
    mask: np.ndarray | None = None  # HxW bool, inside the floor polygon
    px_per_m: float = DEFAULT_PX_PER_M

    @property
    def fill_fraction(self) -> float:
        return 1.0 - self.coverage

    def uv_for_xz(self, xz) -> np.ndarray:
        """World XZ -> UV (trimesh/USD convention: v = 0 at the image's bottom row =
        zmax). Works for any floor mesh - feed it the vertices' (x, z)."""
        xz = np.asarray(xz, dtype=float).reshape(-1, 2)
        xmin, zmin, xmax, zmax = self.extent_xz
        u = (xz[:, 0] - xmin) / max(xmax - xmin, 1e-9)
        v = 1.0 - (xz[:, 1] - zmin) / max(zmax - zmin, 1e-9)
        return np.column_stack([np.clip(u, 0, 1), np.clip(v, 0, 1)])

    def to_json(self) -> dict:
        return {
            "extent_xz": list(self.extent_xz),
            "n_points": self.n_points,
            "coverage": self.coverage,
            "fill_fraction": self.fill_fraction,
            "px_per_m": self.px_per_m,
            "size_px": [self.image.shape[1], self.image.shape[0]],
            "inside_polygon_fraction": float(self.mask.mean()) if self.mask is not None else None,
        }


def floor_extent(floor_polygon, pad_m: float = FLOOR_PAD_M) -> tuple[float, float, float, float]:
    poly = _as_shapely_polygon(floor_polygon)
    xmin, zmin, xmax, zmax = poly.bounds
    return (float(xmin - pad_m), float(zmin - pad_m), float(xmax + pad_m), float(zmax + pad_m))


def bake_floor_texture(
    xyz: np.ndarray,
    rgb: np.ndarray,
    floor_polygon,
    floor_y: float,
    *,
    px_per_m: float | None = DEFAULT_PX_PER_M,
    band_m: float = DEFAULT_BAND_M,
    max_px: int = DEFAULT_MAX_PX,
    pad_m: float = FLOOR_PAD_M,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
    mask_outside: bool = True,
) -> FloorTexture:
    """Top-down bake of the floor band onto the polygon's padded AABB. `xyz`
    must already be in the exported (post-yaw) frame. `floor_polygon` is any
    ring (list of (x, z)) or shapely polygon (holes respected by the mask).
    `px_per_m=None` picks the density from the point count (`auto_px_per_m`).
    Outside-polygon pixels are set to the far-field tone (`mask_outside`)."""
    xyz = np.asarray(xyz, dtype=float)
    xmin, zmin, xmax, zmax = floor_extent(floor_polygon, pad_m)
    w_m, d_m = xmax - xmin, zmax - zmin
    y = xyz[:, 1]
    sel = (y >= floor_y - band_m) & (y <= floor_y + band_m) & (xyz[:, 0] >= xmin) & (xyz[:, 0] < xmax) & (xyz[:, 2] >= zmin) & (xyz[:, 2] < zmax)
    P = xyz[sel]
    if px_per_m is None:
        px_per_m = auto_px_per_m(len(P), _as_shapely_polygon(floor_polygon).area)
    scale = min(float(px_per_m), max_px / max(w_m, d_m, 1e-6))
    width = _texture_size(w_m, scale, max_px)
    height = _texture_size(d_m, scale, max_px)
    u = (P[:, 0] - xmin) / w_m * width
    v = (P[:, 2] - zmin) / d_m * height
    img, hit = _splat(u, v, np.asarray(rgb)[sel], width, height)
    img, _filled = _fill_holes(img, hit, fill_radius_px=fill_radius_m * scale)
    mask = polygon_mask(floor_polygon, (xmin, zmin, xmax, zmax), width, height)
    if mask_outside and hit.any() and not mask.all():
        neutral = _far_field(img, max(width, height) / 8.0)
        img = np.where(mask[..., None], img, neutral)
    inside = mask if mask.any() else np.ones_like(mask)
    coverage = float(hit[inside].mean())
    return FloorTexture(image=_to_uint8(img), extent_xz=(xmin, zmin, xmax, zmax), n_points=int(sel.sum()), coverage=coverage, mask=mask, px_per_m=float(scale))


# --------------------------------------------------------------------------- wall band / atlas


@dataclass
class WallBandTexture:
    image: np.ndarray  # HxWx3 uint8, row 0 = top, col = +s (p0 -> p1)
    band: WallBand
    n_points: int
    coverage: float
    px_per_m: float

    @property
    def fill_fraction(self) -> float:
        return 1.0 - self.coverage

    def to_json(self) -> dict:
        return {"band": self.band.to_json(), "n_points": self.n_points, "coverage": self.coverage, "fill_fraction": self.fill_fraction, "px_per_m": self.px_per_m, "size_px": [self.image.shape[1], self.image.shape[0]]}


def bake_wall_band_texture(
    xyz: np.ndarray,
    rgb: np.ndarray,
    segment_xz,
    y0: float,
    height: float,
    *,
    px_per_m: float | None = DEFAULT_PX_PER_M,
    band_m: float = DEFAULT_BAND_M,
    max_px: int = DEFAULT_MAX_PX,
    margin_m: float = DEFAULT_SEGMENT_MARGIN_M,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
    size_px: tuple[int, int] | None = None,
) -> WallBandTexture:
    """Project cloud points within `band_m` of the vertical plane through
    `segment_xz = ((x0, z0), (x1, z1))`, within the segment's extent +/-
    `margin_m` and within [y0, y0 + height], onto (u along the segment, v
    height) texture space. `size_px=(w, h)` forces the image size (atlas
    packing); otherwise `px_per_m` (None = auto from the point count)."""
    (x0, z0), (x1, z1) = segment_xz
    band = WallBand(p0_xz=(float(x0), float(z0)), p1_xz=(float(x1), float(z1)), y0=float(y0), height=max(float(height), 1e-3))
    xyz = np.asarray(xyz, dtype=float)
    s, t = band.project(xyz)
    y = xyz[:, 1]
    sel = (np.abs(t) <= band_m) & (s >= -margin_m) & (s <= band.length_m + margin_m) & (y >= band.y0) & (y < band.y0 + band.height)
    n_sel = int(sel.sum())
    if size_px is not None:
        width, height_px = int(size_px[0]), int(size_px[1])
        scale = width / band.length_m
    else:
        if px_per_m is None:
            px_per_m = auto_px_per_m(n_sel, band.length_m * band.height)
        scale = min(float(px_per_m), max_px / max(band.length_m, band.height, 1e-6))
        width = _texture_size(band.length_m, scale, max_px)
        height_px = _texture_size(band.height, scale, max_px)
    u = np.clip(s[sel] / band.length_m, 0.0, 1.0 - 1e-9) * width
    v = (band.y0 + band.height - y[sel]) / band.height * height_px  # row 0 = top
    img, hit = _splat(u, v, np.asarray(rgb)[sel], width, height_px)
    img, _filled = _fill_holes(img, hit, fill_radius_px=fill_radius_m * scale)
    return WallBandTexture(image=_to_uint8(img), band=band, n_points=n_sel, coverage=float(hit.mean()), px_per_m=float(scale))


@dataclass
class WallAtlas:
    """Several `WallBand`s in one image + a flat cap swatch (see module doc)."""

    image: np.ndarray  # HxWx3 uint8
    bands: list[WallBand]
    u_ranges: list[tuple[float, float]]  # per band: [u0, u1] of its columns
    y0: float
    height: float
    cap_u: tuple[float, float] | None  # u range of the cap swatch column, or None
    n_points: int = 0
    coverage: float = 0.0
    px_per_m: float = DEFAULT_PX_PER_M
    per_band: list[dict] = field(default_factory=list)  # n_points / coverage per band

    @property
    def fill_fraction(self) -> float:
        return 1.0 - self.coverage

    # -- face assignment (by geometry) -----------------------------------------
    def face_band_index(self, vertices_xyz: np.ndarray, faces: np.ndarray, *, margin_m: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """For each face: the band it belongs to (-1 = horizontal cap face) and
        its matching cost (perpendicular distance to the band's plane plus how
        far outside the segment's extent its centroid lies). Side faces are
        assigned to the closest band; `margin_m` only softens the extent
        penalty so a face straddling a corner still lands on a neighbour."""
        V = np.asarray(vertices_xyz, dtype=float)
        F = np.asarray(faces, dtype=int)
        if len(F) == 0:
            return np.zeros(0, dtype=int), np.zeros(0)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n, axis=1)
        ny = np.abs(n[:, 1]) / np.where(nn > 1e-12, nn, 1.0)
        centroid = V[F].mean(axis=1)
        idx = np.full(len(F), -1, dtype=int)
        cost = np.zeros(len(F))
        side = ny < CAP_NORMAL_Y_MIN
        if side.any() and self.bands:
            costs = np.empty((int(side.sum()), len(self.bands)))
            cen = centroid[side]
            for j, band in enumerate(self.bands):
                s, t = band.project(cen)
                outside = np.maximum(0.0, np.maximum(-s, s - band.length_m) - margin_m)
                costs[:, j] = np.abs(t) + outside
            best = np.argmin(costs, axis=1)
            idx[side] = best
            cost[side] = costs[np.arange(len(best)), best]
        return idx, cost

    def uv_for_faces(self, vertices_xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Per FACE-VERTEX UVs, shape (F*3, 2) in `faces.ravel()` order
        (trimesh/USD convention). Cap faces map into the cap swatch."""
        V = np.asarray(vertices_xyz, dtype=float)
        F = np.asarray(faces, dtype=int)
        idx, _ = self.face_band_index(V, F)
        uv = np.zeros((len(F), 3, 2))
        cap_u = 0.5 * (self.cap_u[0] + self.cap_u[1]) if self.cap_u else 1.0
        for j, band in enumerate(self.bands):
            sel = idx == j
            if not sel.any():
                continue
            u0, u1 = self.u_ranges[j]
            local = band.uv_for_xyz(V[F[sel]].reshape(-1, 3)).reshape(-1, 3, 2)
            local[..., 0] = u0 + local[..., 0] * (u1 - u0)
            uv[sel] = local
        caps = idx < 0
        uv[caps, :, 0] = cap_u
        uv[caps, :, 1] = 0.5
        return uv.reshape(-1, 2)

    def to_json(self) -> dict:
        return {
            "bands": [b.to_json() for b in self.bands],
            "u_ranges": [list(r) for r in self.u_ranges],
            "y0": self.y0,
            "height": self.height,
            "cap_u": list(self.cap_u) if self.cap_u else None,
            "n_points": self.n_points,
            "coverage": self.coverage,
            "fill_fraction": self.fill_fraction,
            "px_per_m": self.px_per_m,
            "size_px": [self.image.shape[1], self.image.shape[0]],
            "per_band": self.per_band,
        }

    @classmethod
    def from_json(cls, d: dict, image: np.ndarray | None = None) -> "WallAtlas":
        if image is None:
            w, h = d["size_px"]
            image = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        return cls(
            image=image,
            bands=[WallBand.from_json(b) for b in d["bands"]],
            u_ranges=[(float(r[0]), float(r[1])) for r in d["u_ranges"]],
            y0=float(d["y0"]),
            height=float(d["height"]),
            cap_u=tuple(d["cap_u"]) if d.get("cap_u") else None,
            n_points=int(d.get("n_points", 0)),
            coverage=float(d.get("coverage", 0.0)),
            px_per_m=float(d.get("px_per_m", DEFAULT_PX_PER_M)),
            per_band=list(d.get("per_band", [])),
        )


def bake_wall_atlas(
    xyz: np.ndarray,
    rgb: np.ndarray,
    bands: list[WallBand],
    *,
    px_per_m: float | None = DEFAULT_PX_PER_M,
    band_m: float = DEFAULT_BAND_M,
    max_px: int = DEFAULT_MAX_PX,
    max_atlas_px: int = DEFAULT_MAX_ATLAS_PX,
    margin_m: float = DEFAULT_SEGMENT_MARGIN_M,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
    cap_rgb: tuple[int, int, int] | None = None,
) -> WallAtlas:
    """Bake every band at ONE common px/m and pack them left to right (+ the cap
    swatch). The common scale is `px_per_m` (None = auto from the total point
    count over all bands) reduced so the atlas fits `max_atlas_px` wide and
    `max_px` tall. Cap colour defaults to the mean baked wall colour."""
    if not bands:
        raise ValueError("bake_wall_atlas needs at least one band")
    y0 = bands[0].y0
    height = max(b.height for b in bands)
    total_len = sum(b.length_m for b in bands)
    if px_per_m is None:
        xyz_a = np.asarray(xyz, dtype=float)
        n_total = 0
        for b in bands:
            s, t = b.project(xyz_a)
            n_total += int(((np.abs(t) <= band_m) & (s >= -margin_m) & (s <= b.length_m + margin_m) & (xyz_a[:, 1] >= b.y0) & (xyz_a[:, 1] < b.y0 + b.height)).sum())
        px_per_m = auto_px_per_m(n_total, total_len * height)
    scale = min(float(px_per_m), (max_atlas_px - CAP_SWATCH_PX) / max(total_len, 1e-6), max_px / max(height, 1e-6))
    height_px = _texture_size(height, scale, max_px)
    widths = [max(2, int(round(b.length_m * scale))) for b in bands]
    tiles: list[WallBandTexture] = []
    for b, w in zip(bands, widths):
        tiles.append(bake_wall_band_texture(xyz, rgb, (b.p0_xz, b.p1_xz), b.y0, b.height, band_m=band_m, margin_m=margin_m, fill_radius_m=fill_radius_m, size_px=(w, height_px)))
    total_w = sum(widths) + CAP_SWATCH_PX
    atlas = np.zeros((height_px, total_w, 3), dtype=np.uint8)
    u_ranges: list[tuple[float, float]] = []
    x = 0
    n_points = 0
    hit_px = 0.0
    per_band = []
    for tile, w in zip(tiles, widths):
        img = tile.image
        if img.shape[0] != height_px:  # band shorter than the tallest: pad at the top (upright images share y0)
            pad = np.zeros((height_px, w, 3), dtype=np.uint8)
            pad[height_px - img.shape[0] :] = img
            img = pad
        atlas[:, x : x + w] = img
        u_ranges.append((x / total_w, (x + w) / total_w))
        x += w
        n_points += tile.n_points
        hit_px += tile.coverage * w * height_px
        per_band.append({"n_points": tile.n_points, "coverage": tile.coverage, "fill_fraction": tile.fill_fraction, "width_px": w})
    if cap_rgb is None:
        cap_rgb = tuple(int(v) for v in atlas[:, : x or 1].reshape(-1, 3).mean(axis=0)) if x else (180, 180, 180)
    atlas[:, x:] = np.asarray(cap_rgb, dtype=np.uint8)
    coverage = hit_px / max(x * height_px, 1)
    return WallAtlas(image=atlas, bands=list(bands), u_ranges=u_ranges, y0=y0, height=height, cap_u=(x / total_w, 1.0), n_points=n_points, coverage=float(coverage), px_per_m=float(scale), per_band=per_band)


def bake_wall_outline_textures(
    xyz: np.ndarray,
    rgb: np.ndarray,
    outline_xz,
    y0: float,
    height: float,
    *,
    closed: bool = True,
    px_per_m: float | None = DEFAULT_PX_PER_M,
    band_m: float = DEFAULT_BAND_M,
    max_px: int = DEFAULT_MAX_PX,
    max_atlas_px: int = DEFAULT_MAX_ATLAS_PX,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
) -> WallAtlas:
    """Every edge of the (closed) outline polygon is one wall band; returns the
    single atlas a `wall_outline` mesh binds (edge i -> `atlas.bands[i]` ->
    `atlas.u_ranges[i]`; the mesh's faces are assigned to edges by geometry in
    `WallAtlas.uv_for_faces`). `band_m` is the half-width of the slab of cloud
    points taken per edge (wall thickness / 2 + tolerance)."""
    bands = outline_bands(outline_xz, y0, height, closed=closed)
    if not bands:
        raise ValueError("bake_wall_outline_textures: outline has no edges")
    return bake_wall_atlas(xyz, rgb, bands, px_per_m=px_per_m, band_m=band_m, max_px=max_px, max_atlas_px=max_atlas_px, fill_radius_m=fill_radius_m)


# --------------------------------------------------------------------------- T15c compat: per-fragment wall


@dataclass
class WallTexture:
    """T15c's per-fragment wall texture: the single band along the PCA frame
    (image WITHOUT the cap swatch) plus the `WallAtlas` that carries it."""

    image: np.ndarray  # HxWx3 uint8, row 0 = ceiling, col = +s
    frame: WallFrame
    y0: float
    y1: float
    n_points: int
    coverage: float
    atlas: WallAtlas | None = None

    @property
    def fill_fraction(self) -> float:
        return 1.0 - self.coverage

    def to_json(self) -> dict:
        return {"frame": self.frame.to_json(), "y0": self.y0, "y1": self.y1, "n_points": self.n_points, "coverage": self.coverage, "fill_fraction": self.fill_fraction, "size_px": [self.image.shape[1], self.image.shape[0]]}


def bake_wall_texture(
    xyz: np.ndarray,
    rgb: np.ndarray,
    wall_vertices_xz,
    floor_y: float,
    ceiling_y: float,
    *,
    band_m: float = DEFAULT_BAND_M,
    px_per_m: float | None = DEFAULT_PX_PER_M,
    max_px: int = DEFAULT_MAX_PX,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
) -> WallTexture:
    """A per-fragment wall polygon = one band along its PCA axis; the slab of
    points is the wall's own half-thickness + `band_m`."""
    frame = wall_frame(wall_vertices_xz)
    band = band_from_wall_frame(frame, floor_y, ceiling_y - floor_y)
    bt = bake_wall_band_texture(xyz, rgb, (band.p0_xz, band.p1_xz), band.y0, band.height, band_m=frame.half_thickness_m + band_m, px_per_m=px_per_m, max_px=max_px, fill_radius_m=fill_radius_m, margin_m=0.0)
    atlas = bake_wall_atlas(xyz, rgb, [band], px_per_m=bt.px_per_m, band_m=frame.half_thickness_m + band_m, max_px=max_px, fill_radius_m=fill_radius_m, margin_m=0.0)
    return WallTexture(image=bt.image, frame=frame, y0=float(floor_y), y1=float(ceiling_y), n_points=bt.n_points, coverage=bt.coverage, atlas=atlas)


# --------------------------------------------------------------------------- room


def _entry_atlas(entry: dict) -> WallAtlas | None:
    """The `WallAtlas` of a wall entry from `bake_room_textures` (in-memory or
    re-hydrated from `textures.json`)."""
    atlas = entry.get("atlas")
    if isinstance(atlas, WallAtlas):
        return atlas
    if isinstance(atlas, dict):
        return WallAtlas.from_json(atlas)
    if entry.get("frame") is not None:  # pre-T15c-prep entry: PCA frame only
        frame = entry["frame"] if isinstance(entry["frame"], WallFrame) else WallFrame.from_json(entry["frame"])
        band = band_from_wall_frame(frame, float(entry["y0"]), float(entry["y1"]) - float(entry["y0"]))
        return WallAtlas(image=np.zeros((1, 1, 3), np.uint8), bands=[band], u_ranges=[(0.0, 1.0)], y0=band.y0, height=band.height, cap_u=None)
    return None


def bake_room_textures(
    ply_path: Path | str | None,
    walls,
    floor_polygon,
    floor_y: float,
    ceiling_y: float,
    out_dir: Path | str,
    *,
    yaw_correction_rad: float = 0.0,
    yaw_rotation_center_xy: tuple[float, float] | None = None,
    band_m: float = DEFAULT_BAND_M,
    px_per_m: float | None = DEFAULT_PX_PER_M,
    max_px: int = DEFAULT_MAX_PX,
    cloud: tuple[np.ndarray, np.ndarray] | None = None,
    cloud_to_scene: np.ndarray | None = None,
    wall_outline=None,
    wall_outline_key: str = "outline",
    wall_thickness_m: float = 0.0,
    max_points: int | None = DEFAULT_MAX_POINTS,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
) -> dict:
    """Bake every texture for one room. `walls` is a list of `WallPolygon` (index
    i <-> GLB node `wall_i` / USD `/World/Structure/wall_i`) or a `{i:
    WallPolygon}` dict for sparse indices - each becomes a one-band atlas along
    its PCA frame. `wall_outline` (T15g): a closed (x, z) ring whose edges are
    baked as ONE atlas under key `wall_outline_key` (-> node `wall_outline`);
    `wall_thickness_m / 2 + band_m` is the slab half-width per edge.
    `floor_polygon` is the world-XZ ring (post-yaw frame) or None to skip the
    floor. The cloud is read ONCE from `ply_path` (subsampled to `max_points`)
    or taken from `cloud=(xyz, rgb)`, optionally pushed through
    `cloud_to_scene` (4x4, see `swap_xz_transform`) and then rotated by the
    bootstrap yaw - the same order bootstrap applies to its own geometry.
    `px_per_m=None` = auto density.

    Returns the "textures" dict `export_glb.build_scene(..., textures=)` /
    `apply_room_textures` / `apply_usd_textures` consume:
      {"floor": {"png", "mask_png", "extent_xz", "coverage", "fill_fraction", ...} | None,
       "walls": {i | "outline": {"png", "atlas", "frame" | None, "y0", "y1", "coverage", ...}},
       "yaw_correction_rad", "yaw_rotation_center_xy", "n_cloud_points", ...}
    plus in-memory `image` arrays (and the floor `mask`) under the same keys."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cloud is None:
        if ply_path is None:
            raise ValueError("bake_room_textures needs ply_path or cloud=(xyz, rgb)")
        from scripts.msa.ply_io import read_ply_xyz_rgb

        xyz, rgb = read_ply_xyz_rgb(Path(ply_path), max_points=max_points)
    else:
        xyz, rgb = cloud
    xyz = apply_transform_xyz(xyz, cloud_to_scene)
    rgb = np.asarray(rgb)
    center = tuple(yaw_rotation_center_xy) if yaw_rotation_center_xy is not None else (0.0, 0.0)
    xyz = rotate_points_xz(xyz, yaw_correction_rad, center)

    from PIL import Image

    result: dict = {
        "floor": None,
        "walls": {},
        "yaw_correction_rad": float(yaw_correction_rad),
        "yaw_rotation_center_xy": list(center),
        "cloud_to_scene": None if cloud_to_scene is None else np.asarray(cloud_to_scene, dtype=float).tolist(),
        "n_cloud_points": int(len(xyz)),
        "band_m": band_m,
        "px_per_m": px_per_m,
        "fill_radius_m": fill_radius_m,
    }
    if floor_polygon is not None and len(floor_polygon) >= 3:
        ft = bake_floor_texture(xyz, rgb, floor_polygon, floor_y, band_m=band_m, px_per_m=px_per_m, max_px=max_px, fill_radius_m=fill_radius_m)
        png = out_dir / "floor_texture.png"
        Image.fromarray(ft.image).save(png)
        mask_png = out_dir / "floor_mask.png"
        Image.fromarray((ft.mask.astype(np.uint8) * 255)).save(mask_png)
        result["floor"] = {**ft.to_json(), "png": str(png), "mask_png": str(mask_png), "image": ft.image, "mask": ft.mask}

    wall_items = walls.items() if isinstance(walls, dict) else enumerate(walls or [])
    for i, wall in wall_items:
        verts = wall.vertices if isinstance(wall, WallPolygon) else wall
        if verts is None or len(verts) < 2:
            continue
        wt = bake_wall_texture(xyz, rgb, verts, floor_y, ceiling_y, band_m=band_m, px_per_m=px_per_m, max_px=max_px, fill_radius_m=fill_radius_m)
        png = out_dir / f"wall_{i}_texture.png"
        Image.fromarray(wt.atlas.image).save(png)
        result["walls"][int(i) if not isinstance(i, str) else i] = {
            **wt.to_json(),
            "atlas": wt.atlas.to_json(),
            "size_px": [wt.atlas.image.shape[1], wt.atlas.image.shape[0]],
            "png": str(png),
            "image": wt.atlas.image,
            "_atlas": wt.atlas,
        }

    if wall_outline is not None and len(wall_outline) >= 2:
        atlas = bake_wall_outline_textures(xyz, rgb, wall_outline, floor_y, ceiling_y - floor_y, px_per_m=px_per_m, band_m=wall_thickness_m / 2.0 + band_m, max_px=max_px, fill_radius_m=fill_radius_m)
        png = out_dir / f"wall_{wall_outline_key}_texture.png"
        Image.fromarray(atlas.image).save(png)
        result["walls"][wall_outline_key] = {
            "frame": None,
            "y0": float(floor_y),
            "y1": float(ceiling_y),
            "n_points": atlas.n_points,
            "coverage": atlas.coverage,
            "fill_fraction": atlas.fill_fraction,
            "n_edges": len(atlas.bands),
            "atlas": atlas.to_json(),
            "size_px": [atlas.image.shape[1], atlas.image.shape[0]],
            "png": str(png),
            "image": atlas.image,
            "_atlas": atlas,
        }
    return result


def textures_summary(textures: dict) -> dict:
    """`textures.json`-able copy (drops the in-memory arrays)."""
    skip = {"image", "mask", "_atlas"}
    out = {k: v for k, v in textures.items() if k not in ("floor", "walls")}
    out["floor"] = {k: v for k, v in (textures.get("floor") or {}).items() if k not in skip} or None
    out["walls"] = {str(i): {k: v for k, v in e.items() if k not in skip} for i, e in (textures.get("walls") or {}).items()}
    return out


def load_textures_json(path: Path | str) -> dict:
    """Re-hydrate a `textures.json` (written by the CLI) into the dict the apply
    functions take - PNG paths are resolved relative to the JSON's directory."""
    path = Path(path)
    data = json.loads(path.read_text())
    base = path.parent

    def _fix(entry):
        if not entry:
            return entry
        for key in ("png", "mask_png"):
            p = entry.get(key)
            if p and not Path(p).is_absolute():
                entry[key] = str(base / p)
            if p and not Path(entry[key]).exists() and (base / Path(p).name).exists():
                entry[key] = str(base / Path(p).name)
        return entry

    data["floor"] = _fix(data.get("floor"))
    walls = {}
    for k, e in (data.get("walls") or {}).items():
        key: int | str = int(k) if str(k).isdigit() else k
        walls[key] = _fix(e)
    data["walls"] = walls
    return data


# --------------------------------------------------------------------------- UVs / glTF materials


def floor_uv(vertices_xyz: np.ndarray, extent_xz) -> np.ndarray:
    """trimesh-convention UVs (v=0 at image bottom) for floor mesh vertices.
    Image row r = (z - zmin)/D * H, so v = 1 - (z - zmin)/D."""
    xmin, zmin, xmax, zmax = extent_xz
    v_ = np.asarray(vertices_xyz, dtype=float)
    u = (v_[:, 0] - xmin) / max(xmax - xmin, 1e-9)
    v = 1.0 - (v_[:, 2] - zmin) / max(zmax - zmin, 1e-9)
    return np.column_stack([u, v])


def wall_uv(vertices_xyz: np.ndarray, frame: WallFrame, y0: float, y1: float) -> np.ndarray:
    """trimesh-convention per-vertex UVs for a one-band wall (T15c): u =
    along-wall s / length, v = height fraction (image row 0 is the ceiling, so
    v = (y - y0)/(y1 - y0))."""
    v_ = np.asarray(vertices_xyz, dtype=float)
    dx = v_[:, 0] - frame.origin_xz[0]
    dz = v_[:, 2] - frame.origin_xz[1]
    s = dx * frame.dir_xz[0] + dz * frame.dir_xz[1]
    u = s / max(frame.length_m, 1e-9)
    v = (v_[:, 1] - y0) / max(y1 - y0, 1e-9)
    return np.column_stack([u, v])


def _image_of(entry: dict):
    from PIL import Image

    img = entry.get("image")
    if img is not None:
        return Image.fromarray(np.asarray(img, dtype=np.uint8))
    return Image.open(entry["png"]).convert("RGB")


def pbr_texture_material(image, roughness: float = ROOM_ROUGHNESS):
    from trimesh.visual.material import PBRMaterial

    return PBRMaterial(baseColorTexture=image, metallicFactor=0.0, roughnessFactor=roughness, doubleSided=False)


def pbr_color_material(rgb, roughness: float = OBJECT_ROUGHNESS):
    """Untextured PBR material (per-class colour from `color_rgb`) - what object
    placeholder meshes get instead of bare vertex colours, so Three.js builds a
    lit MeshStandardMaterial with a real roughness."""
    from trimesh.visual.material import PBRMaterial

    r, g, b = (int(c) for c in rgb[:3])
    return PBRMaterial(baseColorFactor=[r, g, b, 255], metallicFactor=0.0, roughnessFactor=roughness, doubleSided=False)


def textured_visual(mesh, uv: np.ndarray, image, roughness: float = ROOM_ROUGHNESS):
    from trimesh.visual import TextureVisuals

    return TextureVisuals(uv=uv, material=pbr_texture_material(image, roughness))


def _wall_entries(textures: dict) -> dict:
    """`{key: (entry, WallAtlas)}` for every wall entry that has an atlas."""
    out = {}
    for key, entry in (textures.get("walls") or {}).items():
        atlas = _entry_atlas(entry)
        if atlas is not None and atlas.bands:
            out[key] = (entry, atlas)
    return out


def match_wall_entry(vertices_xyz: np.ndarray, faces: np.ndarray, entries: dict, *, max_cost_m: float = 1.0):
    """Pick the wall texture entry whose bands the mesh's side faces lie in
    (mean face cost, `WallAtlas.face_band_index`) - geometry, not names.
    Returns (key, entry, atlas) or None when nothing is within `max_cost_m`."""
    best = None
    for key, (entry, atlas) in entries.items():
        idx, cost = atlas.face_band_index(vertices_xyz, faces)
        side = idx >= 0
        if not side.any():
            continue
        mean_cost = float(cost[side].mean())
        if mean_cost <= max_cost_m and (best is None or mean_cost < best[0]):
            best = (mean_cost, key, entry, atlas)
    return None if best is None else best[1:]


def texture_wall_geometry(geom, transform, entry: dict, atlas: WallAtlas) -> None:
    """Unweld a trimesh wall geometry (so UV seams between bands survive) and
    give it face-vertex UVs from `atlas` + the entry's image. In place."""
    import trimesh

    geom.unmerge_vertices()
    world = trimesh.transform_points(geom.vertices, transform) if transform is not None else np.asarray(geom.vertices)
    faces = np.asarray(geom.faces)
    uv_fv = atlas.uv_for_faces(world, faces)
    uv = np.zeros((len(geom.vertices), 2))
    uv[faces.ravel()] = uv_fv
    geom.visual = textured_visual(geom, uv, _image_of(entry))


def apply_room_textures(scene, textures: dict, *, floor_node: str = "floor", wall_node_fmt: str = "wall_{i}") -> int:
    """Attach baked textures (from `bake_room_textures` / `load_textures_json`)
    to a trimesh Scene's `floor` and wall geometries in place. Wall nodes are
    every `wall_*` Trimesh node (the current `wall_i` fragments or T15g's
    `wall_outline`): a node named `wall_node_fmt.format(i=key)` takes entry
    `key` directly, any other wall node is matched to the entry whose bands
    its side faces lie in (`match_wall_entry`). Geometry vertices are read in
    the node's world frame. Returns the number of geometries textured."""
    import trimesh

    applied = 0
    floor_entry = textures.get("floor")
    if floor_entry and floor_node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[floor_node]
        geom = scene.geometry[geom_name]
        if isinstance(geom, trimesh.Trimesh):
            world = trimesh.transform_points(geom.vertices, transform)
            geom.visual = textured_visual(geom, floor_uv(world, floor_entry["extent_xz"]), _image_of(floor_entry))
            applied += 1
    entries = _wall_entries(textures)
    if not entries:
        return applied
    by_node = {wall_node_fmt.format(i=key): key for key in entries}
    prefix = wall_node_fmt.split("{", 1)[0]
    done = set()
    for node in list(scene.graph.nodes_geometry):
        if not node.startswith(prefix) or "/" in node:
            continue
        transform, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or geom_name in done:
            continue
        if node in by_node:
            key = by_node[node]
            entry, atlas = entries[key]
        else:
            world = trimesh.transform_points(geom.vertices, transform)
            m = match_wall_entry(world, np.asarray(geom.faces), entries)
            if m is None:
                continue
            _key, entry, atlas = m
        texture_wall_geometry(geom, transform, entry, atlas)
        done.add(geom_name)
        applied += 1
    return applied


def apply_object_materials(scene, roughness: float = OBJECT_ROUGHNESS) -> int:
    """Re-materialize vertex-coloured object visual parts (an existing
    bootstrap GLB, pre-T15c) with a PBR colour material (mean vertex colour).
    Skips `floor`/`wall_*`/collision/Plan nodes. Returns count."""
    import trimesh

    n = 0
    for node in list(scene.graph.nodes_geometry):
        if node == "floor" or node.startswith("wall_") or "collision" in node or node.startswith("Plan"):
            continue
        _, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        try:
            kind = geom.visual.kind
        except Exception:
            kind = None
        if kind == "vertex":
            rgb = np.asarray(geom.visual.vertex_colors)[:, :3].mean(axis=0)
            geom.visual = trimesh.visual.TextureVisuals(material=pbr_color_material(rgb, roughness))
            n += 1
        elif kind == "texture" and getattr(geom.visual.material, "baseColorTexture", None) is None:
            mat = geom.visual.material
            if hasattr(mat, "roughnessFactor") and mat.roughnessFactor is None:
                mat.roughnessFactor = roughness
                mat.metallicFactor = 0.0
                n += 1
    return n


# --------------------------------------------------------------------------- USD materials


def _usd_mesh_yup(mesh_prim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points Y-up (V,3), faceVertexIndices flat, faceVertexCounts) of a USD
    Mesh authored by `export_usd` (Isaac Z-up: (x, y, z)_yup -> (x, -z, y), so
    back is (X, Z, -Y))."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
    yup = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]]) if len(pts) else pts
    fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
    fvc = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=int)
    return yup, fvi, fvc


def _triangles_of(fvi: np.ndarray, fvc: np.ndarray) -> np.ndarray:
    """First three indices of every face (all our faces ARE triangles; a polygon
    gets its first triangle for the normal/centroid test)."""
    starts = np.concatenate([[0], np.cumsum(fvc)[:-1]]) if len(fvc) else np.zeros(0, dtype=int)
    tri = np.column_stack([fvi[starts], fvi[starts + 1], fvi[starts + 2]]) if len(fvc) else np.zeros((0, 3), dtype=int)
    return tri


def _ensure_png(entry: dict, out_dir: Path, name: str) -> Path:
    """The PNG lives next to the USD (relative asset path): copy it there if it
    is elsewhere, or write it from the in-memory image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = entry.get("png")
    dst = out_dir / (Path(src).name if src else name)
    if src and Path(src).exists():
        if Path(src).resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    else:
        from PIL import Image

        Image.fromarray(np.asarray(entry["image"], dtype=np.uint8)).save(dst)
    return dst


def author_texture_material(stage, material_path: str, png_relpath: str, *, roughness: float = ROOM_ROUGHNESS):
    """`UsdShade.Material` with a `UsdPreviewSurface` whose diffuseColor is a
    `UsdUVTexture` (clamp, sRGB) fed by a `UsdPrimvarReader_float2` on `st`."""
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, material_path)
    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader_st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png_relpath)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader_out)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def _author_st(mesh_prim, uv_face_varying: np.ndarray) -> None:
    from pxr import Gf, Sdf, UsdGeom, Vt

    pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uv_face_varying]))


def _bind(mesh_prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)


def _is_visible(prim) -> bool:
    from pxr import UsdGeom

    try:
        return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
    except Exception:
        return True


def apply_usd_textures(
    stage,
    textures: dict,
    out_dir: Path | str,
    *,
    structure_root: str = "/World/Structure",
    floor_path: str = "/World/Floor",
    looks_root: str = USD_LOOKS_ROOT,
    skip_invisible: bool = True,
) -> dict:
    """Bind baked textures to an `export_usd`/`export_presentation_usd` stage.
    PNGs are written/copied into `out_dir` (the USD's directory - asset paths
    are relative `./name.png`), one `UsdShade.Material` per texture under
    `looks_root`, `primvars:st` (faceVarying) authored on every bound mesh.
    Floor: the `floor_path` Mesh, UVs from world XZ via the floor extent. Walls:
    EVERY visible Mesh under `structure_root` (scene.usd `wall_i`, the
    presentation's `wall_i/visual/stub`, T15g's `wall_outline` - never by
    name), matched to the wall entry whose bands its side faces lie in; side
    faces get (u along the band, v height), cap faces the atlas' cap swatch.
    Invisible colliders are skipped (`skip_invisible`). Returns a summary."""
    from pxr import UsdGeom

    out_dir = Path(out_dir)
    summary: dict = {"materials": {}, "bound": {}, "skipped": []}
    materials = {}

    floor_entry = textures.get("floor")
    floor_prim = stage.GetPrimAtPath(floor_path) if floor_entry else None
    if floor_prim is not None and floor_prim.IsValid() and floor_prim.IsA(UsdGeom.Mesh):
        png = _ensure_png(floor_entry, out_dir, "floor_texture.png")
        mat_path = f"{looks_root}/floor_material"
        mat = author_texture_material(stage, mat_path, f"./{png.name}")
        materials["floor"] = mat
        summary["materials"]["floor"] = {"path": mat_path, "png": str(png)}
        pts, fvi, _fvc = _usd_mesh_yup(floor_prim)
        uv = floor_uv(pts, floor_entry["extent_xz"])
        _author_st(floor_prim, np.clip(uv[fvi], 0.0, 1.0))
        _bind(floor_prim, mat)
        summary["bound"][floor_path] = "floor"

    entries = _wall_entries(textures)
    root = stage.GetPrimAtPath(structure_root)
    if entries and root.IsValid():
        from pxr import Usd

        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if skip_invisible and not _is_visible(prim):
                summary["skipped"].append(prim.GetPath().pathString)
                continue
            pts, fvi, fvc = _usd_mesh_yup(prim)
            if len(fvc) == 0:
                continue
            tri = _triangles_of(fvi, fvc)
            m = match_wall_entry(pts, tri, entries)
            if m is None:
                summary["skipped"].append(prim.GetPath().pathString)
                continue
            key, entry, atlas = m
            if key not in materials:
                png = _ensure_png(entry, out_dir, f"wall_{key}_texture.png")
                mat_path = f"{looks_root}/wall_{key}_material"
                materials[key] = author_texture_material(stage, mat_path, f"./{png.name}")
                summary["materials"][str(key)] = {"path": mat_path, "png": str(png)}
            if (fvc == 3).all():
                uv_fv = atlas.uv_for_faces(pts, tri)
            else:  # general polygons: per-face band from the first triangle, UVs per face-vertex
                idx, _ = atlas.face_band_index(pts, tri)
                uv_fv = np.zeros((len(fvi), 2))
                pos = 0
                cap_u = 0.5 * (atlas.cap_u[0] + atlas.cap_u[1]) if atlas.cap_u else 1.0
                for f, n in enumerate(fvc):
                    verts = pts[fvi[pos : pos + n]]
                    if idx[f] < 0:
                        uv_fv[pos : pos + n] = (cap_u, 0.5)
                    else:
                        u0, u1 = atlas.u_ranges[idx[f]]
                        local = atlas.bands[idx[f]].uv_for_xyz(verts)
                        local[:, 0] = u0 + local[:, 0] * (u1 - u0)
                        uv_fv[pos : pos + n] = local
                    pos += n
            _author_st(prim, uv_fv)
            _bind(prim, materials[key])
            summary["bound"][prim.GetPath().pathString] = str(key)
    return summary


def retexture_usd(usd_in: Path | str, usd_out: Path | str, textures: dict) -> dict:
    """Copy `usd_in` to `usd_out` and bind the textures there (PNGs land next to
    `usd_out`). Returns `apply_usd_textures`' summary + paths."""
    from pxr import Usd

    usd_in, usd_out = Path(usd_in), Path(usd_out)
    usd_out.parent.mkdir(parents=True, exist_ok=True)
    if usd_in.resolve() != usd_out.resolve():
        shutil.copyfile(usd_in, usd_out)
    stage = Usd.Stage.Open(usd_out.as_posix())
    summary = apply_usd_textures(stage, textures, usd_out.parent)
    layer = stage.GetRootLayer()
    layer.customLayerData = {**dict(layer.customLayerData), "cloudeye:msaTextures": "scripts.msa.textures (T15c-prep)"}
    layer.Save()
    return {"usd_in": str(usd_in), "usd_out": str(usd_out), **summary}


# --------------------------------------------------------------------------- recover geometry from a bootstrap GLB


def walls_from_scene(scene, floor_y: float, tol: float = 1e-3) -> dict[int, WallPolygon]:
    """`{i: WallPolygon}` from a bootstrap `scene.glb`'s `wall_i` nodes - the
    bottom-cap vertices (any order; `wall_frame` is order-independent).
    Non-numeric wall nodes (`wall_outline`) are not fragments and are skipped."""
    import trimesh

    out: dict[int, WallPolygon] = {}
    for node in scene.graph.nodes_geometry:
        if not node.startswith("wall_"):
            continue
        try:
            i = int(node.split("_", 1)[1])
        except ValueError:
            continue
        transform, geom_name = scene.graph[node]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        world = trimesh.transform_points(geom.vertices, transform)
        bottom = world[np.abs(world[:, 1] - floor_y) < tol]
        if len(bottom) < 2:
            bottom = world[world[:, 1] <= world[:, 1].min() + tol]
        verts = [(float(x), float(z)) for x, z in bottom[:, [0, 2]]]
        out[i] = WallPolygon(vertices=verts, area_m2=0.0)
    return out


def floor_polygon_from_scene(scene) -> list[tuple[float, float]] | None:
    """The `Plan/floor` outline's XZ vertices (falls back to the `floor` mesh's
    footprint vertices)."""
    import trimesh

    for node in ("Plan/floor", "Planfloor"):
        if node in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node]
            geom = scene.geometry[geom_name]
            verts = trimesh.transform_points(np.asarray(geom.vertices, dtype=float), transform)
            return [(float(x), float(z)) for x, z in verts[:, [0, 2]]]
    if "floor" in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph["floor"]
        geom = scene.geometry[geom_name]
        verts = trimesh.transform_points(np.asarray(geom.vertices, dtype=float), transform)
        return [(float(x), float(z)) for x, z in verts[:, [0, 2]]]
    return None


def detect_outline_wall(scene, floor_polygon, floor_y: float, *, min_span_frac: float = 0.6) -> bool:
    """True when the scene's walls look like T15g's single closed band around
    the room (a `wall_outline` node, or exactly one `wall_*` mesh whose
    footprint spans >= `min_span_frac` of the room polygon's bbox in BOTH axes)
    rather than per-fragment boxes."""
    import trimesh

    nodes = [n for n in scene.graph.nodes_geometry if n.startswith("wall_") and "/" not in n and isinstance(scene.geometry.get(scene.graph[n][1]), trimesh.Trimesh)]
    if "wall_outline" in nodes:
        return True
    if len(nodes) != 1 or not floor_polygon or len(floor_polygon) < 3:
        return False
    transform, geom_name = scene.graph[nodes[0]]
    world = trimesh.transform_points(scene.geometry[geom_name].vertices, transform)
    poly = np.asarray(floor_polygon, dtype=float)
    span_w = (world[:, 0].max() - world[:, 0].min()) / max(poly[:, 0].max() - poly[:, 0].min(), 1e-6)
    span_d = (world[:, 2].max() - world[:, 2].min()) / max(poly[:, 1].max() - poly[:, 1].min(), 1e-6)
    return bool(span_w >= min_span_frac and span_d >= min_span_frac)


def cloud_alignment_check(xyz_rotated: np.ndarray, floor_polygon, floor_y: float, band_m: float = DEFAULT_BAND_M, sample: int = 50000) -> float:
    """Diagnostic: fraction of floor-band cloud points that fall inside the
    exported floor polygon. A wrong yaw sign/pivot shows up as a drop vs the
    unrotated cloud; printed by the CLI, not a hard gate."""
    from shapely import contains_xy

    poly = _as_shapely_polygon(floor_polygon)
    band = np.flatnonzero((xyz_rotated[:, 1] >= floor_y - band_m) & (xyz_rotated[:, 1] <= floor_y + band_m))
    if len(band) == 0:
        return 0.0
    if len(band) > sample:
        band = np.random.default_rng(0).choice(band, sample, replace=False)
    return float(contains_xy(poly, xyz_rotated[band, 0], xyz_rotated[band, 2]).mean())


CLOUD_FRAMES = ("auto", "identity", "swap_xz")
WALL_MODES = ("auto", "fragments", "outline")


def resolve_cloud_frame(
    cloud_frame: str,
    xyz: np.ndarray,
    floor_polygon,
    floor_y: float,
    angle: float,
    center: tuple[float, float],
    occupancy_meta: dict | None,
    band_m: float = DEFAULT_BAND_M,
) -> tuple[str, np.ndarray | None, dict]:
    """Pick the cloud->scene transform for `retexture_bootstrap_output`.
    `identity` = trust the cloud frame; `swap_xz` = `swap_xz_transform` with the
    occupancy origins; `auto` = whichever puts more floor-band points inside the
    exported floor polygon (identity wins ties). Returns (name, 4x4 | None,
    diagnostics)."""
    ox = float((occupancy_meta or {}).get("origin_x", 0.0))
    oz = float((occupancy_meta or {}).get("origin_z", 0.0))
    candidates: dict[str, np.ndarray | None] = {"identity": None, "swap_xz": swap_xz_transform(ox, oz)}
    if cloud_frame not in CLOUD_FRAMES:
        raise ValueError(f"cloud_frame must be one of {CLOUD_FRAMES}, got {cloud_frame!r}")
    diag: dict = {}
    if floor_polygon is not None and len(floor_polygon) >= 3:
        for name, t in candidates.items():
            rotated = rotate_points_xz(apply_transform_xyz(xyz, t), angle, center)
            diag[name] = cloud_alignment_check(rotated, floor_polygon, floor_y, band_m)
    if cloud_frame == "auto":
        if not diag:
            chosen = "identity"
        else:
            chosen = max(("identity", "swap_xz"), key=lambda n: (diag[n], n == "identity"))
    else:
        chosen = cloud_frame
    return chosen, candidates[chosen], diag


def locate_ply(scene_dir: Path, ply_name: str, ply_path: Path | None = None) -> Path:
    """`ply_path` if given; else `scene_dir/ply_name`; else the same name next to
    the real `occupancy.npy` (the hero raw scene dir is a folder of symlinks
    into the frozen GPU output, which holds the 271 MB cloud)."""
    if ply_path is not None:
        return Path(ply_path)
    direct = Path(scene_dir) / ply_name
    if direct.exists():
        return direct
    occ = Path(scene_dir) / "occupancy.npy"
    if occ.exists():
        sibling = Path(os.path.realpath(occ)).parent / ply_name
        if sibling.exists():
            return sibling
    raise FileNotFoundError(f"{ply_name} not found in {scene_dir} (nor next to the real occupancy.npy); pass --ply")


def retexture_bootstrap_output(
    scene_dir: Path,
    bootstrap_out: Path,
    out_dir: Path,
    *,
    ply_name: str = "aligned_room.ply",
    ply_path: Path | None = None,
    band_m: float = DEFAULT_BAND_M,
    px_per_m: float | None = None,
    max_px: int = DEFAULT_MAX_PX,
    cloud_frame: str = "auto",
    usd: Path | str | None = "auto",
    max_points: int | None = DEFAULT_MAX_POINTS,
    wall_mode: str = "auto",
    wall_thickness_m: float = 0.0,
    fill_radius_m: float = DEFAULT_FILL_RADIUS_M,
) -> dict:
    """CLI body: bake textures for an existing bootstrap output and write
    `out_dir/scene_textured.glb`, `out_dir/scene_textured.usd` (a copy of the
    input USD - `usd="auto"` = `scene_presentation.usd` if present else
    `scene.usd`, a name relative to `bootstrap_out`, an absolute path, or None
    to skip), PNGs and `textures.json`. Walls/room polygon come from
    `objects.json` (T15f) when present, else from the GLB (`walls_from_scene`,
    `Plan/floor`) and `scene_meta.json`'s `room_polygon`. `wall_mode`: `fragments`
    = one PCA band per `wall_i`; `outline` = ONE atlas over the room polygon's
    edges (T15g's single band - the polygon is the regularized outline);
    `auto` = `detect_outline_wall`. Returns the summary written to textures.json."""
    import trimesh

    from scripts.msa.export_glb import export_glb
    from scripts.msa.ply_io import read_ply_xyz_rgb

    t_start = time.perf_counter()
    scene_dir, bootstrap_out, out_dir = Path(scene_dir), Path(bootstrap_out), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((bootstrap_out / "scene_meta.json").read_text())
    floor_y = float(meta["floor_y"])
    ceiling_y = float(meta["ceiling_y"])
    angle, center = yaw_from_meta(meta)
    occ_meta_path = scene_dir / "occupancy_meta.json"
    occupancy_meta = json.loads(occ_meta_path.read_text()) if occ_meta_path.exists() else None
    if wall_mode not in WALL_MODES:
        raise ValueError(f"wall_mode must be one of {WALL_MODES}, got {wall_mode!r}")

    scene = trimesh.load(str(bootstrap_out / "scene.glb"), force="scene")
    objects_json = bootstrap_out / "objects.json"
    geometry_source = "scene.glb"
    walls: dict | list
    if objects_json.exists():
        from scripts.msa.objects_io import load_objects_json

        bo = load_objects_json(objects_json)
        walls = list(bo.walls)
        room_polygon = bo.room_polygon
        geometry_source = "objects.json"
    else:
        walls = walls_from_scene(scene, floor_y)
        room_polygon = [tuple(p) for p in (meta.get("room_polygon") or meta.get("floor_polygon") or [])] or None
    floor_polygon = room_polygon if room_polygon and len(room_polygon) >= 3 else floor_polygon_from_scene(scene)

    ply = locate_ply(scene_dir, ply_name, ply_path)
    t0 = time.perf_counter()
    xyz, rgb = read_ply_xyz_rgb(ply, max_points=max_points)
    t_load = time.perf_counter() - t0
    frame_name, cloud_to_scene, frame_diag = resolve_cloud_frame(cloud_frame, xyz, floor_polygon, floor_y, angle, center, occupancy_meta, band_m)
    if frame_name == "swap_xz":
        print(
            "WARNING: cloud frame 'swap_xz' selected - the bootstrap's floor/walls are transposed "
            f"(x<->z + occupancy origin offset) relative to {ply.name}; floor-band inside fraction "
            f"identity={frame_diag.get('identity', float('nan')):.3f} vs swap_xz={frame_diag.get('swap_xz', float('nan')):.3f}. "
            "Textures are baked to match the exported meshes; this bootstrap output predates T15b's occupancy [ix, iz] fix.",
            file=sys.stderr,
        )

    outline_mode = wall_mode == "outline" or (wall_mode == "auto" and detect_outline_wall(scene, floor_polygon, floor_y))
    wall_outline = floor_polygon if outline_mode and floor_polygon else None
    t0 = time.perf_counter()
    textures = bake_room_textures(
        None,
        {} if outline_mode else walls,
        floor_polygon,
        floor_y,
        ceiling_y,
        out_dir,
        yaw_correction_rad=angle,
        yaw_rotation_center_xy=center,
        band_m=band_m,
        px_per_m=px_per_m,
        max_px=max_px,
        cloud=(xyz, rgb),
        cloud_to_scene=cloud_to_scene,
        wall_outline=wall_outline,
        wall_thickness_m=wall_thickness_m,
        fill_radius_m=fill_radius_m,
    )
    t_bake = time.perf_counter() - t0
    alignment = {
        "cloud_frame": frame_name,
        "inside_floor_polygon_by_frame": frame_diag,
        "inside_floor_polygon_unrotated": cloud_alignment_check(xyz, floor_polygon, floor_y, band_m) if floor_polygon else None,
    }
    del xyz, rgb

    n_tex = apply_room_textures(scene, textures)
    n_obj = apply_object_materials(scene)
    glb_path = out_dir / "scene_textured.glb"
    export_glb(scene, glb_path)

    usd_summary = None
    if usd is not None:
        if usd == "auto":
            cand = [bootstrap_out / "scene_presentation.usd", bootstrap_out / "scene.usd"]
            usd_in = next((p for p in cand if p.exists()), None)
        else:
            usd_in = Path(usd) if Path(usd).is_absolute() or Path(usd).exists() else bootstrap_out / usd
        if usd_in is not None and usd_in.exists():
            usd_summary = retexture_usd(usd_in, out_dir / "scene_textured.usd", textures)
        elif usd != "auto":
            raise FileNotFoundError(f"--usd {usd}: not found (looked at {usd_in})")

    summary = {
        **textures_summary(textures),
        "glb": str(glb_path),
        "usd": usd_summary,
        "n_textured_geometries": n_tex,
        "n_object_materials": n_obj,
        "wall_mode": "outline" if outline_mode else "fragments",
        "geometry_source": geometry_source,
        "ply": str(ply),
        "max_points": max_points,
        "alignment": alignment,
        "timing_s": {"load_cloud": round(t_load, 2), "bake": round(t_bake, 2), "total": round(time.perf_counter() - t_start, 2)},
    }
    (out_dir / "textures.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene-dir", required=True, type=Path, help="raw scene dir (aligned_room.ply or a symlinked occupancy.npy next to it)")
    ap.add_argument("--bootstrap-out", required=True, type=Path, help="bootstrap output dir (scene.glb + scene_meta.json [+ objects.json, scene*.usd])")
    ap.add_argument("--out", required=True, type=Path, help="output dir for scene_textured.glb/.usd + PNGs + textures.json")
    ap.add_argument("--ply-name", default="aligned_room.ply")
    ap.add_argument("--ply", type=Path, default=None, help="explicit cloud path (overrides --scene-dir/--ply-name)")
    ap.add_argument("--usd", default="auto", help="USD to re-texture: 'auto' (scene_presentation.usd else scene.usd in --bootstrap-out), a name relative to --bootstrap-out, a path, or 'none'")
    ap.add_argument("--band-m", type=float, default=DEFAULT_BAND_M)
    ap.add_argument("--px-per-m", default="auto", help="texels per metre, or 'auto' (from the point density, <= %.0f)" % DEFAULT_PX_PER_M)
    ap.add_argument("--max-px", type=int, default=DEFAULT_MAX_PX)
    ap.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="subsample the cloud to this many points (0 = all)")
    ap.add_argument("--wall-mode", choices=WALL_MODES, default="auto", help="fragments = one band per wall_i; outline = one atlas over the room polygon's edges; auto = detect")
    ap.add_argument("--wall-thickness-m", type=float, default=0.0, help="outline mode: wall thickness (slab half-width = thickness/2 + band)")
    ap.add_argument("--fill-radius-m", type=float, default=DEFAULT_FILL_RADIUS_M)
    ap.add_argument(
        "--cloud-frame",
        choices=CLOUD_FRAMES,
        default="auto",
        help="cloud->bootstrap frame: identity, swap_xz (pre-T15b transposed occupancy read, see swap_xz_transform), or auto-pick by floor alignment",
    )
    args = ap.parse_args(argv)
    px = None if str(args.px_per_m).lower() == "auto" else float(args.px_per_m)
    summary = retexture_bootstrap_output(
        args.scene_dir,
        args.bootstrap_out,
        args.out,
        ply_name=args.ply_name,
        ply_path=args.ply,
        band_m=args.band_m,
        px_per_m=px,
        max_px=args.max_px,
        cloud_frame=args.cloud_frame,
        usd=None if str(args.usd).lower() == "none" else args.usd,
        max_points=args.max_points or None,
        wall_mode=args.wall_mode,
        wall_thickness_m=args.wall_thickness_m,
        fill_radius_m=args.fill_radius_m,
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("floor", "walls", "usd")}, indent=2))
    for i, w in summary["walls"].items():
        print(f"wall_{i}: {w['size_px'][0]}x{w['size_px'][1]} px, {w['n_points']} pts, coverage {w['coverage']:.2f}, fill {w['fill_fraction']:.2f}, bands {len(w['atlas']['bands'])}")
    if summary["floor"]:
        f = summary["floor"]
        print(f"floor: {f['size_px'][0]}x{f['size_px'][1]} px, {f['n_points']} pts, coverage {f['coverage']:.2f}, fill {f['fill_fraction']:.2f}, {f['px_per_m']:.1f} px/m")
    if summary["usd"]:
        print(f"usd: {summary['usd']['usd_out']} - bound {list(summary['usd']['bound'])}, skipped {summary['usd']['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
