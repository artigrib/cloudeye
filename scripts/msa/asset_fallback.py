"""T15f Part B: real CC0 furniture models (see `asset_index.py`) as the
`visual` geometry of every object the generator (Stage B1) did not produce an
accepted mesh for.

Placement rule, per object (`bootstrap.compute_object_footprints` dict):

1. **Pick** the candidate of the object's class (label -> class via
   `asset_index.LABEL_SYNONYMS`; Poly Haven first, Kenney only when Poly Haven
   lacks the class) whose *footprint aspect ratio* (long/short of the
   canonical width x depth) is closest (in log space) to the measured
   `size_uv` long/short; ties break on height similarity, then id. No
   candidate -> the placeholder box stays and the object is logged.
2. **Orient**: the asset's long horizontal axis is rotated (a 90 deg
   pre-rotation about Y when needed) onto the footprint's long axis, so after
   the object's own yaw (`angle_rad`, applied by `export_glb.build_scene`'s
   world transform exactly as for placeholders) the asset's long side lies
   along the measured long side.
3. **Scale per axis into the measured box**: X <- `size_uv[0]` (U), Y <-
   `height`, Z <- `size_uv[1]` (V). Round classes (`asset_index.ROUND_CLASSES`,
   lamps) get ONE uniform factor = min of the three ratios so they are never
   squashed; they are XZ-centred in the footprint.
4. **Floor**: the asset's own min-Y goes to local y=0, which the world
   transform lifts to `bbox_min_y` - same convention as `assets.build_placeholder`.

**T16b small tier** (Google Scanned Objects, `tier="small"`): the same rule,
with two extra gates in `select_candidate` - a small-tier candidate is only
accepted when (a) the object's detected label is a small-object label
(`asset_index.is_small_label`; a furniture label can never pick up a mug) and
(b) every measured extent is <= `SMALL_TIER_MAX_EXTENT_M` (SAM3 "glass" also
fires on window panes). Small assets are placed ONLY at detected objects -
`place_assets` iterates the bootstrap's object list and nothing else, so no
asset is ever added where there was no detection. Cups/bottles are round
classes (uniform scale).

The collision node is NOT touched: it stays the measured hull extrusion
(`hull_xz`), identical to placeholder and generated-mesh objects (SPEC §5).
The GLB's own PBR materials/textures are kept (each sub-mesh of the asset
becomes its own `<id>/visual/part_j` node with its material; verified to
survive a GLB export + reload round-trip).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh

from scripts.msa.asset_index import SMALL_TIER_MAX_EXTENT_M, TIER_SMALL, AssetCandidate, AssetIndex, footprint_aspect, is_small_label

# `export_glb.build_scene(visual_overrides=...)` names each override mesh's node
# `<id>/visual/<metadata[PART_NAME_KEY]>` (default `part_<j>`). Keeping the
# `part_` prefix matters: `render_perspective._bed_centroid_xz` and the web
# viewer's layer classifier key on `/visual/part_`.
PART_NAME_KEY = "msa_part_name"
MIN_TARGET_M = 0.01


def select_candidate(
    label: str, size_uv: tuple[float, float], height: float, index: AssetIndex
) -> tuple[str | None, AssetCandidate | None, float]:
    """(index_class, best candidate or None, measured footprint aspect).
    Small-tier candidates (T16b) are dropped unless the label is a small-object
    label AND the measured box is small enough (see module docstring)."""
    cls, cands = index.candidates_for_label(label, height_m=height)
    aspect_measured = footprint_aspect(*size_uv)
    small_ok = is_small_label(label) and max(float(size_uv[0]), float(size_uv[1]), float(height)) <= SMALL_TIER_MAX_EXTENT_M
    cands = [c for c in cands if c.tier != TIER_SMALL or small_ok]
    if not cands:
        return cls, None, aspect_measured

    def key(c: AssetCandidate):
        aspect_term = abs(math.log(c.footprint_aspect) - math.log(aspect_measured))
        height_term = abs(math.log(max(c.height_m, 1e-3) / max(float(height), 1e-3)))
        return (round(aspect_term, 6), round(height_term, 6), c.id)

    return cls, min(cands, key=key), aspect_measured


def load_asset_meshes(path: str | Path) -> list[trimesh.Trimesh]:
    """Every sub-mesh of a GLB with its node transform baked in and its own
    visual (PBR material + texture/UVs) kept - `Scene.dump(concatenate=False)`."""
    scene = trimesh.load(Path(path).as_posix(), force="scene")
    meshes = [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
    if not meshes:
        raise ValueError(f"{path}: no triangle geometry")
    return meshes


def _joint_bounds(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    b = np.array([m.bounds for m in meshes])
    return b[:, 0].min(axis=0), b[:, 1].max(axis=0)


def fit_meshes_to_box(
    meshes: list[trimesh.Trimesh],
    length_u: float,
    height: float,
    width_v: float,
    *,
    uniform: bool = False,
    align_long_axis: bool = True,
) -> tuple[list[trimesh.Trimesh], dict]:
    """Fit a group of meshes (treated as ONE rigid asset, joint bbox) into the
    placeholder local frame: XZ-centred at the origin, Y up from 0, X <- U
    (`length_u`), Z <- V (`width_v`). `align_long_axis` pre-rotates the asset
    90 deg about Y when its long horizontal axis does not already match the
    footprint's long axis. `uniform` uses min(ratios) on all three axes.
    Returns (fitted copies, info dict: scale_xyz, pre_rotation_deg,
    source_extents, fitted_extents)."""
    length_u = max(float(length_u), MIN_TARGET_M)
    width_v = max(float(width_v), MIN_TARGET_M)
    height = max(float(height), MIN_TARGET_M)
    meshes = [m.copy() for m in meshes]
    bmin, bmax = _joint_bounds(meshes)
    extents = np.maximum(bmax - bmin, 1e-6)

    pre_rotation_deg = 0.0
    if align_long_axis and (extents[0] >= extents[2]) != (length_u >= width_v):
        rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0.0, 1.0, 0.0])
        for m in meshes:
            m.apply_transform(rot)
        pre_rotation_deg = 90.0
        bmin, bmax = _joint_bounds(meshes)
        extents = np.maximum(bmax - bmin, 1e-6)

    ratios = np.array([length_u / extents[0], height / extents[1], width_v / extents[2]], dtype=float)
    scale = np.full(3, float(ratios.min())) if uniform else ratios
    center = (bmin + bmax) / 2.0
    to_origin = trimesh.transformations.translation_matrix([-center[0], -bmin[1], -center[2]])
    scaling = np.diag([scale[0], scale[1], scale[2], 1.0])
    transform = scaling @ to_origin
    for m in meshes:
        m.apply_transform(transform)
    fmin, fmax = _joint_bounds(meshes)
    return meshes, {
        "scale_xyz": [float(s) for s in scale],
        "uniform": bool(uniform),
        "pre_rotation_deg": pre_rotation_deg,
        "source_extents": [float(e) for e in extents],
        "fitted_extents": [float(e) for e in (fmax - fmin)],
    }


def place_assets(
    objects: list[dict], index: AssetIndex, *, skip_ids=()
) -> tuple[dict[str, list[trimesh.Trimesh]], list[dict]]:
    """Apply the module's placement rule to every object not in `skip_ids`
    (those with an accepted generated mesh). Returns (visual overrides for
    `export_glb.build_scene`, per-object placement records - one per object
    considered, including the ones that kept their placeholder)."""
    skip = set(skip_ids)
    overrides: dict[str, list[trimesh.Trimesh]] = {}
    placements: list[dict] = []
    mesh_cache: dict[str, list[trimesh.Trimesh]] = {}
    for obj in objects:
        obj_id = obj["id"]
        if obj_id in skip:
            continue
        length_u, width_v = (float(v) for v in obj["size_uv"])
        height = float(obj["height"])
        cls, cand, aspect_measured = select_candidate(obj["label"], (length_u, width_v), height, index)
        record = {
            "id": obj_id,
            "label": obj["label"],
            "index_class": cls,
            "source": None,
            "tier": None,
            "asset_id": None,
            "asset_path": None,
            "scale_xyz": None,
            "uniform": None,
            "yaw_deg": math.degrees(float(obj["angle_rad"])),
            "aspect_measured": aspect_measured,
            "aspect_asset": None,
            "footprint_uv": [length_u, width_v],
            "height": height,
            "bbox_min_y": float(obj.get("bbox_min_y", 0.0)),
            # M7: the placeholder world transform inputs, so `usd_assets` can
            # re-place the asset in the USD from this record alone.
            "center_xy": [float(v) for v in obj["center_xy"]],
            "angle_rad": float(obj["angle_rad"]),
            "status": "no_candidate_placeholder_kept",
        }
        if cand is None:
            if index.candidates_for_label(obj["label"], height_m=height)[1]:
                record["status"] = "small_tier_gated_placeholder_kept"  # candidates existed but the T16b gates refused them
            placements.append(record)
            continue
        if cand.path not in mesh_cache:
            mesh_cache[cand.path] = load_asset_meshes(cand.path)
        fitted, info = fit_meshes_to_box(mesh_cache[cand.path], length_u, height, width_v, uniform=cand.is_round)
        for j, m in enumerate(fitted):
            m.metadata[PART_NAME_KEY] = f"part_{j}"
            m.metadata["msa_asset"] = f"{cand.source}:{cand.id}"
        overrides[obj_id] = fitted
        record.update(
            {
                "source": cand.source,
                "tier": cand.tier,
                "asset_id": cand.id,
                "asset_path": cand.path,
                "scale_xyz": info["scale_xyz"],
                "uniform": info["uniform"],
                "yaw_deg": math.degrees(float(obj["angle_rad"])) + info["pre_rotation_deg"],
                "pre_rotation_deg": info["pre_rotation_deg"],
                "aspect_asset": cand.footprint_aspect,
                "asset_canonical_aabb_m": list(cand.canonical_aabb_m),
                "fitted_extents": info["fitted_extents"],
                "n_parts": len(fitted),
                "status": "placed",
            }
        )
        placements.append(record)
    return overrides, placements


def write_assets_placed(placements: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(placements, indent=2))
    return path


def format_placements_table(placements: list[dict]) -> str:
    header = f"{'id':<16}{'label':<12}{'source':<10}{'asset':<28}{'scale_xyz':<26}{'yaw':>7}  {'aspect meas/asset':<18}status"
    lines = [header, "-" * len(header)]
    for p in placements:
        scale = "-" if p["scale_xyz"] is None else "x".join(f"{s:.2f}" for s in p["scale_xyz"]) + (" (uni)" if p.get("uniform") else "")
        aspect = f"{p['aspect_measured']:.2f}/" + ("-" if p["aspect_asset"] is None else f"{p['aspect_asset']:.2f}")
        lines.append(
            f"{p['id']:<16}{p['label']:<12}{str(p['source'] or '-'):<10}{str(p['asset_id'] or '-'):<28}{scale:<26}{p['yaw_deg']:>7.1f}  {aspect:<18}{p['status']}"
        )
    return "\n".join(lines)
