"""CLI: assemble a final room GLB whose object `visual` geometry is, per object,
(1) Stage B1's generated mesh when `results.json` accepted one, else (2) a real
CC0 furniture asset fitted into the measured footprint (T15f, see
`asset_fallback.py`), else (3) Stage A0's placeholder primitive. `collision`
stays the measured hull always (SPEC §5), exactly as Stage A0/E0 do.

Inputs (T13): the bootstrap's `objects.json` (`--bootstrap-out <dir>`), i.e.
the FINAL kept objects + walls + room polygon `scene.glb` itself was built from
- wall-backfilled, small-object-filtered, outside-room-dropped/clipped and
yaw-rotated. The pre-T13 path (`--scene-dir`, recompute footprints from the
raw occupancy scene) skipped every one of those steps and produced objects
that did not match `scene.glb`; it is kept only as a deprecated fallback that
warns loudly.

Generated-mesh alignment is a deliberate, documented heuristic, not a solved
pose-estimation problem: the mesh (TRELLIS.2's local frame, Y assumed up) is
non-uniformly scaled to exactly fill the measured oriented-footprint box
(length_u x height x width_v), then placed with the SAME world transform (yaw
`angle_rad` about Y, translate to `center_xy`/`bbox_min_y`) Stage A0 uses for
its placeholders. This guarantees the correct measured volume and position but
NOT that its yaw matches the real photographed object's - TRELLIS.2 cannot
infer "real-world forward" from one masked crop. Real assets get the extra
long-axis alignment step (`asset_fallback.fit_meshes_to_box`), which is all a
footprint can tell us.

    uv run python -m scripts.msa.assemble_with_generated \
        --bootstrap-out <dir with objects.json> \
        [--manifest-dir <dir with results.json + *.glb>] \
        [--out <dir>/scene_assets.glb] [--no-assets]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import trimesh

from scripts.msa.asset_fallback import PART_NAME_KEY, fit_meshes_to_box, format_placements_table, place_assets, write_assets_placed
from scripts.msa.asset_index import DEFAULT_ASSETS_ROOT, AssetIndex, build_index, footprint_aspect
from scripts.msa.export_glb import build_scene
from scripts.msa.geometry import WALL_THICKNESS_M, WallPolygon, room_wall_band
from scripts.msa.objects_io import OBJECTS_JSON_NAME, BootstrapObjects, load_objects_json

GENERATED_GLB_SUBDIRS = ("", "crops")


def _fit_generated_mesh_to_footprint(mesh: trimesh.Trimesh, length_u: float, height: float, width_v: float) -> trimesh.Trimesh:
    """Stage B1 fit (kept for callers/tests): per-axis scale of ONE mesh's bbox
    into the local footprint box, no long-axis alignment (the generated mesh
    was produced from a crop of the object in its measured orientation)."""
    fitted, _info = fit_meshes_to_box([mesh], length_u, height, width_v, uniform=False, align_long_axis=False)
    return fitted[0]


def load_generated(manifest_dir: Path) -> dict[str, Path]:
    """`{obj_id: glb_path}` for every accepted entry of `results.json` whose GLB
    exists (looked up in `manifest_dir` then `manifest_dir/crops`, where the
    B1 CLI actually writes them)."""
    manifest_dir = Path(manifest_dir)
    results_path = manifest_dir / "results.json"
    if not results_path.exists():
        return {}
    results = json.loads(results_path.read_text())
    generated: dict[str, Path] = {}
    for r in results:
        if not (r.get("accepted") and r.get("glb_path")):
            continue
        for sub in GENERATED_GLB_SUBDIRS:
            candidate = manifest_dir / sub / r["glb_path"] if sub else manifest_dir / r["glb_path"]
            if candidate.exists():
                generated[r["id"]] = candidate
                break
    return generated


def fit_generated(objects: list[dict], generated: dict[str, Path]) -> tuple[dict[str, list[trimesh.Trimesh]], list[dict]]:
    """Local-frame visual overrides for `export_glb.build_scene` from the
    accepted generated GLBs - every sub-mesh kept with its own material - plus
    (M7) one `assets_placed.json`-shaped record per generated object
    (`status = "generated"`, `source = "generated"`, `scale_xyz`,
    `pre_rotation_deg = 0`, placeholder transform inputs) so `usd_assets` can
    put the same mesh into the USD."""
    overrides: dict[str, list[trimesh.Trimesh]] = {}
    records: list[dict] = []
    for obj in objects:
        glb_path = generated.get(obj["id"])
        if glb_path is None:
            continue
        loaded = trimesh.load(Path(glb_path).as_posix(), force="scene")
        meshes = [m for m in loaded.dump(concatenate=False) if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
        if not meshes:
            continue
        length_u, width_v = (float(v) for v in obj["size_uv"])
        height = float(obj["height"])
        fitted, info = fit_meshes_to_box(meshes, length_u, height, width_v, uniform=False, align_long_axis=False)
        for j, m in enumerate(fitted):
            m.metadata[PART_NAME_KEY] = f"part_{j}"
            m.metadata["msa_asset"] = f"generated:{Path(glb_path).name}"
        overrides[obj["id"]] = fitted
        records.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "index_class": None,
                "source": "generated",
                "tier": None,
                "asset_id": Path(glb_path).stem,
                "asset_path": str(Path(glb_path)),
                "scale_xyz": info["scale_xyz"],
                "uniform": False,
                "yaw_deg": math.degrees(float(obj["angle_rad"])),
                "pre_rotation_deg": 0.0,
                "aspect_measured": footprint_aspect(length_u, width_v),
                "aspect_asset": None,
                "footprint_uv": [length_u, width_v],
                "height": height,
                "bbox_min_y": float(obj.get("bbox_min_y", 0.0)),
                "center_xy": [float(v) for v in obj["center_xy"]],
                "angle_rad": float(obj["angle_rad"]),
                "fitted_extents": info["fitted_extents"],
                "n_parts": len(fitted),
                "status": "generated",
            }
        )
    return overrides, records


def fit_generated_overrides(objects: list[dict], generated: dict[str, Path]) -> dict[str, list[trimesh.Trimesh]]:
    """Pre-M7 entry point, kept: overrides only."""
    return fit_generated(objects, generated)[0]


def assemble(
    bootstrap: BootstrapObjects,
    generated: dict[str, Path] | None = None,
    *,
    assets: bool = True,
    asset_index: AssetIndex | None = None,
    assets_root: Path = DEFAULT_ASSETS_ROOT,
    small_root: Path | None = None,
    textures: dict | None = None,
) -> tuple[trimesh.Scene, list[dict]]:
    """Build the assembled scene from a loaded `objects.json`. Returns (scene,
    placement records: one `status="generated"` record per accepted B1 mesh
    (M7) plus, when `assets=True`, one record per remaining object - `placed` or
    a placeholder status). `small_root` (T16b) adds the `tier="small"` GSO
    candidates when it holds an INVENTORY.json."""
    generated = generated or {}
    overrides, placements = fit_generated(bootstrap.objects, generated)
    if assets:
        index = asset_index or build_index(assets_root, small_root=small_root)
        asset_overrides, asset_placements = place_assets(bootstrap.objects, index, skip_ids=set(overrides))
        overrides.update(asset_overrides)
        placements = placements + asset_placements
    # T15h: the same single `wall_outline` band as the bootstrap's scene.glb (T15g,
    # `geometry.room_wall_band` of the regularized room polygon in objects.json) -
    # this file used to extrude the per-fragment `wall_i` blobs itself, so
    # scene_assets.glb and scene.glb disagreed on the walls. No polygon -> the
    # pre-T15g per-wall extrusions, as before.
    wall_band = room_wall_band(bootstrap.room_polygon, WALL_THICKNESS_M) if bootstrap.room_polygon else None
    scene = build_scene(
        bootstrap.walls,
        bootstrap.room_polygon,
        bootstrap.floor_y,
        bootstrap.ceiling_y,
        bootstrap.objects,
        textures=textures,
        visual_overrides=overrides,
        wall_band=wall_band,
    )
    return scene, placements


def build_scene_with_generated(
    walls: list[WallPolygon],
    floor_polygon,
    floor_y: float,
    ceiling_y: float,
    objects: list[dict],
    generated: dict[str, Path],
    *,
    asset_index: AssetIndex | None = None,
) -> trimesh.Scene:
    """Stage B1 entry point, kept: same as `assemble` but from bare inputs.
    Passing `asset_index` enables the T15f asset fallback."""
    bootstrap = BootstrapObjects(objects=objects, walls=walls, room_polygon=floor_polygon, floor_y=floor_y, ceiling_y=ceiling_y)
    scene, _ = assemble(bootstrap, generated, assets=asset_index is not None, asset_index=asset_index)
    return scene


def export_assets_usd(bootstrap: BootstrapObjects, placements: list[dict], out_usd: Path, *, asset_cache_root: Path | None = None) -> Path:
    """M7 item 4: the schema-v5 USD twin of `scene_assets.glb` - same walls / floor
    plate / wall band / measured hull colliders as `bootstrap.run_bootstrap`'s
    `scene.usd`, and every object's `visual` = the SAME asset (or generated
    mesh) the GLB got, referenced from `<out dir>/usd_assets/` (`usd_assets`)."""
    from scripts.msa.export_glb import _extrude_shapely_polygon_xz, _floor_plate_mesh, _floor_plate_polygon
    from scripts.msa.export_usd import export_usd
    from scripts.msa.usd_assets import DEFAULT_USD_CACHE_ROOT

    floor_plate = _floor_plate_polygon(bootstrap.room_polygon, bootstrap.objects)
    floor_mesh = _floor_plate_mesh(floor_plate, bootstrap.floor_y) if floor_plate is not None else None
    wall_band = room_wall_band(bootstrap.room_polygon, WALL_THICKNESS_M) if bootstrap.room_polygon else None
    wall_band_mesh = _extrude_shapely_polygon_xz(wall_band, bootstrap.floor_y, bootstrap.ceiling_y - bootstrap.floor_y) if wall_band is not None else None
    export_usd(
        bootstrap.walls, floor_mesh, bootstrap.floor_y, bootstrap.ceiling_y, bootstrap.objects, Path(out_usd),
        floor_polygon=bootstrap.room_polygon, wall_band_mesh=wall_band_mesh, asset_placements=placements,
        asset_cache_root=Path(asset_cache_root) if asset_cache_root is not None else DEFAULT_USD_CACHE_ROOT,
    )
    return Path(out_usd)


def _deprecated_recompute_from_scene_dir(scene_dir: Path) -> BootstrapObjects:
    warnings.warn(
        "--scene-dir recomputes footprints with compute_object_footprints only and BYPASSES wall-backfill, "
        "the small-object filter, the outside-room drop/clip and the yaw rotation - its objects will NOT match "
        "scene.glb. Run bootstrap and pass --bootstrap-out <dir> (objects.json) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("WARNING: --scene-dir is deprecated; objects will not match scene.glb (use --bootstrap-out).", file=sys.stderr)
    from scripts.msa.bootstrap import compute_object_footprints, extract_floor_polygon, load_object_inputs_from_scene_dir
    from scripts.msa.geometry import extract_wall_polygons

    occupancy = np.load(scene_dir / "occupancy.npy")
    meta = json.loads((scene_dir / "occupancy_meta.json").read_text())
    resolution, origin_x, origin_z = meta["resolution"], meta["origin_x"], meta["origin_z"]
    scene_meta = json.loads((scene_dir / "scene_meta.json").read_text())
    walls, _dropped = extract_wall_polygons(occupancy, resolution, origin_x, origin_z)
    floor_polygon = extract_floor_polygon(occupancy, resolution, origin_x, origin_z)
    objects, _overhead = compute_object_footprints(load_object_inputs_from_scene_dir(scene_dir), scene_meta["floor_y"])
    return BootstrapObjects(objects=objects, walls=walls, room_polygon=floor_polygon, floor_y=scene_meta["floor_y"], ceiling_y=scene_meta["ceiling_y"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-out", type=Path, default=None, help=f"bootstrap --out-dir containing {OBJECTS_JSON_NAME} (preferred)")
    parser.add_argument("--scene-dir", type=Path, default=None, help="DEPRECATED: raw occupancy scene dir; recomputes footprints (objects will not match scene.glb)")
    parser.add_argument("--manifest-dir", type=Path, default=None, help="Stage B1 dir with results.json + generated GLBs (optional)")
    parser.add_argument("--out", type=Path, default=None, help="output GLB (default: <bootstrap-out>/scene_assets.glb)")
    parser.add_argument("--assets", dest="assets", action="store_true", default=True, help="fit real CC0 assets into every object without an accepted generated mesh (default)")
    parser.add_argument("--no-assets", dest="assets", action="store_false", help="keep placeholder boxes instead of assets")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--small-root", type=Path, default=None, help="T16b: small-object (GSO) asset root with INVENTORY.json; off unless given")
    parser.add_argument("--usd", dest="usd", action="store_true", default=True, help="M7: also write scene_assets.usd referencing the same assets (default)")
    parser.add_argument("--no-usd", dest="usd", action="store_false")
    args = parser.parse_args()

    if args.bootstrap_out is not None:
        objects_json = args.bootstrap_out / OBJECTS_JSON_NAME
        if not objects_json.exists():
            parser.error(f"{objects_json} not found - re-run scripts.msa.bootstrap (T13+) so it writes objects.json")
        bootstrap = load_objects_json(objects_json)
        print(f"objects.json: {len(bootstrap.objects)} objects, {len(bootstrap.walls)} walls, room_polygon={'yes' if bootstrap.room_polygon else 'no'}")
    elif args.scene_dir is not None:
        bootstrap = _deprecated_recompute_from_scene_dir(args.scene_dir)
    else:
        parser.error("pass --bootstrap-out <dir> (or the deprecated --scene-dir)")

    generated = load_generated(args.manifest_dir) if args.manifest_dir else {}
    out = args.out or ((args.bootstrap_out or args.scene_dir) / "scene_assets.glb")
    scene, placements = assemble(bootstrap, generated, assets=args.assets, assets_root=args.assets_root, small_root=args.small_root)

    n_placed = sum(1 for p in placements if p["status"] == "placed")
    n_generated = sum(1 for p in placements if p["status"] == "generated")
    print(f"objects: {len(bootstrap.objects)}, generated (accepted, found): {len(generated)}, assets placed: {n_placed}, "
          f"placeholders kept: {len(placements) - n_placed - n_generated}")
    if placements:
        print(format_placements_table(placements))
        placed_path = write_assets_placed(placements, out.parent / "assets_placed.json")
        print(f"wrote {placed_path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out.as_posix(), file_type="glb")
    print(f"wrote {out}")
    if args.usd:
        from scripts.msa.usd_assets import compare_glb_usd_objects, placements_by_id

        usd_path = export_assets_usd(bootstrap, placements, out.with_suffix(".usd"))
        check = compare_glb_usd_objects(out, usd_path, list(placements_by_id(placements)))
        (out.parent / "assets_usd_check.json").write_text(json.dumps(check, indent=2))
        print(f"wrote {usd_path} - GLB/USD identity: ok={check['ok']} objects {check['glb_object_count']}/{check['usd_object_count']}, "
              f"classes identical={check['classes_identical']}, placed without real mesh={len(check['placed_without_real_mesh'])}")
        if not check["ok"]:
            raise SystemExit(f"scene_assets.glb and {usd_path.name} disagree - see {out.parent / 'assets_usd_check.json'}")


if __name__ == "__main__":
    main()
