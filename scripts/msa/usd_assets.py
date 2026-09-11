"""M7 item 4: the Isaac USD references the SAME furniture assets as the web GLB.

`assemble_with_generated --assets` puts Poly Haven / Kenney / GSO GLBs (and the
Stage B1 generated meshes) into `scene_assets.glb`, fitted into the measured
footprints by `asset_fallback.fit_meshes_to_box`. Until M7 the USD exporters
(`export_usd.export_usd`, `export_usd.export_presentation_usd`) authored
placeholder boxes for every object, so the recording showed white boxes where
the web viewer showed furniture. This module closes that gap:

1. **GLB -> USD once, cached** (`convert_glb_to_usd`): no glTF->USD converter is
   available offline (checked: `usdcat`/`usdzip` ship with usd-core but have no
   glTF plugin; `gltf2usd`/`guc`/Omniverse's asset converter are not installed
   and need the network or a Kit install), so the GLB is read with trimesh and
   every sub-mesh is authored as a `UsdGeom.Mesh` (points, triangles, vertex
   normals, `primvars:st`, extent, `subdivisionScheme = none`, doubleSided)
   under a `/Asset` default prim, with a `UsdPreviewSurface` per material
   (baseColor texture written next to the .usd as PNG and wired through
   `UsdUVTexture` + `UsdPrimvarReader_float2`; untextured materials get the
   constant `baseColorFactor` / vertex-colour mean as `diffuseColor`; roughness
   and metallic factors carried over). The cache lives under
   `var/assets/furniture/usd/<source>/<id>/<id>.usd` (+ `.json` sidecar with
   the asset's joint bounds, needed for the fit transform); Stage B1 generated
   meshes are per scene and are converted straight into the output dir.

2. **Same placement as the GLB** (`asset_stage_transform`): the transform of
   `/World/Objects/<id>/visual` is `YUP_TO_ZUP @ world(yaw, center_xy, bbox_min_y)
   @ fit(pre_rotation, scale_xyz, asset bounds)` - `fit` reproduces
   `asset_fallback.fit_meshes_to_box` exactly from the numbers `assets_placed.json`
   records (`pre_rotation_deg`, `scale_xyz`) plus the cached asset bounds, and
   `world` is the placeholder transform `export_glb.build_scene` applies. Points
   land where `to_isaac` puts every other prim's points.

3. **Reference, don't copy** (`add_asset_visual`): the asset .usd (+ its
   textures) is synced into `<out_dir>/usd_assets/<source>/<id>/` and referenced
   RELATIVELY from `/World/Objects/<id>/visual/asset` (instanceable), so the
   output directory is self-contained - copy the whole directory to the Isaac
   host, not just the .usd. Colliders are untouched (the measured hulls);
   translucent classes (curtains) keep their placeholder parts.

4. **Assertion** (`compare_glb_usd_objects`): the object ids and the sorted class
   list of `scene_assets.glb` and the USD are identical, and every object that
   got an asset has a real mesh (a referenced `visual/asset` whose composed
   subtree holds Mesh prims with more than a box's 12 triangles and a bound
   material) - not a placeholder box.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

DEFAULT_USD_CACHE_ROOT = Path("var/assets/furniture/usd")
USD_ASSETS_SUBDIR = "usd_assets"
ASSET_ROOT_PRIM = "/Asset"
ASSET_LOOKS_PRIM = "/Asset/Looks"
ASSET_PRIM_NAME = "asset"  # /World/Objects/<id>/visual/asset
GENERATED_SOURCE = "generated"
PLACED_STATUSES = ("placed", GENERATED_SOURCE)
BOX_FACE_COUNT = 12  # a placeholder box is 12 triangles; anything richer is "a real mesh"

# (x, y, z)_Y-up -> (x, -z, y)_Z-up, the same map as app.services.usd_export.to_isaac.
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _sanitize_prim_name(name: str) -> str:
    """Same rule as `export_usd._sanitize_prim_name` (kept local: export_usd imports this module)."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"p_{safe}"
    return safe


def object_class_from_id(obj_id: str) -> str:
    """`television_2` -> `television` (the `render_perspective._object_class_from_id` rule)."""
    return re.sub(r"_\d+$", "", obj_id)


# --------------------------------------------------------------------------- GLB -> USD


def _texture_image(material):
    """The base-colour PIL image of a trimesh material, if any."""
    if material is None:
        return None
    img = getattr(material, "baseColorTexture", None)
    if img is None:
        img = getattr(material, "image", None)
    return img


def _material_constants(material, mesh) -> tuple[tuple[float, float, float], float, float]:
    """(diffuse rgb 0..1, roughness, metallic) for an untextured or textured material."""
    rgb = (0.8, 0.8, 0.8)
    roughness, metallic = 0.8, 0.0
    if material is not None:
        factor = getattr(material, "baseColorFactor", None)
        if factor is None:
            factor = getattr(material, "diffuse", None)
        if factor is not None:
            f = np.asarray(factor, dtype=float).reshape(-1)[:3]
            if f.max() > 1.0:
                f = f / 255.0
            rgb = (float(f[0]), float(f[1]), float(f[2]))
        r = getattr(material, "roughnessFactor", None)
        m = getattr(material, "metallicFactor", None)
        roughness = float(r) if r is not None else roughness
        metallic = float(m) if m is not None else metallic
    elif isinstance(mesh.visual, trimesh.visual.ColorVisuals) and mesh.visual.kind is not None:
        mean = np.asarray(mesh.visual.vertex_colors, dtype=float)[:, :3].mean(axis=0) / 255.0
        rgb = (float(mean[0]), float(mean[1]), float(mean[2]))
    return rgb, roughness, metallic


def _author_material(stage: Usd.Stage, path: str, *, diffuse, roughness, metallic, texture_file: str | None) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if texture_file is not None:
        reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/diffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        diffuse_input.ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        diffuse_input.Set(Gf.Vec3f(*[float(c) for c in diffuse]))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def load_glb_parts(glb_path: str | Path) -> list[trimesh.Trimesh]:
    """Every triangle sub-mesh of a GLB with its node transform baked in and its
    own visual kept (`asset_fallback.load_asset_meshes` convention)."""
    scene = trimesh.load(Path(glb_path).as_posix(), force="scene")
    parts = [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
    if not parts:
        raise ValueError(f"{glb_path}: no triangle geometry")
    return parts


def joint_bounds(parts: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    b = np.array([m.bounds for m in parts])
    return b[:, 0].min(axis=0), b[:, 1].max(axis=0)


def convert_glb_to_usd(glb_path: str | Path, out_usd: str | Path, *, force: bool = False) -> dict:
    """Author `out_usd` (+ textures next to it, + `<stem>.json` sidecar) from a GLB;
    returns the sidecar dict. Skipped when the cached .usd is newer than the GLB
    and its sidecar exists (`force=True` re-converts)."""
    glb_path, out_usd = Path(glb_path), Path(out_usd)
    sidecar = out_usd.with_suffix(".json")
    if not force and out_usd.exists() and sidecar.exists() and out_usd.stat().st_mtime >= glb_path.stat().st_mtime:
        try:
            info = json.loads(sidecar.read_text())
            if info.get("glb") == str(glb_path):
                return info
        except (json.JSONDecodeError, KeyError):
            pass

    parts = load_glb_parts(glb_path)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale textures from a previous conversion of the same asset.
    for old in out_usd.parent.glob(f"{out_usd.stem}_tex_*.png"):
        old.unlink()
    stage = Usd.Stage.CreateNew(out_usd.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ASSET_ROOT_PRIM)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, ASSET_LOOKS_PRIM)

    texture_files: dict[int, str] = {}
    materials: dict[str, UsdShade.Material] = {}
    material_summaries: list[dict] = []
    n_faces = 0
    for j, mesh in enumerate(parts):
        prim_path = f"{ASSET_ROOT_PRIM}/part_{j}"
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
        usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if len(normals) == len(verts):
            usd_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd_mesh.CreateDoubleSidedAttr(True)
        lo, hi = mesh.bounds
        usd_mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in lo]), Gf.Vec3f(*[float(v) for v in hi])]))
        n_faces += len(faces)

        visual = mesh.visual
        material = getattr(visual, "material", None)
        uv = getattr(visual, "uv", None)
        has_uv = uv is not None and len(uv) == len(verts)
        if has_uv:
            st = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
        if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
            colors = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            dc = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            dc.Set(Vt.Vec3fArray.FromNumpy(colors))

        image = _texture_image(material) if has_uv else None
        texture_file = None
        if image is not None:
            key = id(image)
            if key not in texture_files:
                fname = f"{out_usd.stem}_tex_{len(texture_files)}.png"
                img = image.convert("RGBA") if image.mode not in ("RGB", "RGBA") else image
                img.save(out_usd.parent / fname)
                texture_files[key] = fname
            texture_file = f"./{texture_files[key]}"
        diffuse, roughness, metallic = _material_constants(material, mesh)
        mat_key = texture_file or f"rgb:{diffuse}:{roughness}:{metallic}"
        if mat_key not in materials:
            mat_path = f"{ASSET_LOOKS_PRIM}/mat_{len(materials)}"
            materials[mat_key] = _author_material(stage, mat_path, diffuse=diffuse, roughness=roughness, metallic=metallic, texture_file=texture_file)
            material_summaries.append({"path": mat_path, "texture": texture_file, "diffuse": [round(c, 4) for c in diffuse], "roughness": roughness, "metallic": metallic})
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(materials[mat_key])

    bmin, bmax = joint_bounds(parts)
    stage.GetRootLayer().customLayerData = {"cloudeye:msaAssetSource": str(glb_path), "cloudeye:msaAssetConverter": "scripts.msa.usd_assets (trimesh -> UsdGeom.Mesh + UsdPreviewSurface)"}
    stage.GetRootLayer().Save()
    info = {
        "glb": str(glb_path),
        "usd": str(out_usd),
        "n_parts": len(parts),
        "n_faces": int(n_faces),
        "n_textures": len(texture_files),
        "textures": sorted(texture_files.values()),
        "materials": material_summaries,
        "bounds_min": [float(v) for v in bmin],
        "bounds_max": [float(v) for v in bmax],
    }
    sidecar.write_text(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------- transforms


def fit_transform(bounds_min, bounds_max, scale_xyz, pre_rotation_deg: float = 0.0) -> np.ndarray:
    """The 4x4 (column-vector convention) `asset_fallback.fit_meshes_to_box`
    applies to the asset's own frame: optional 90 deg pre-rotation about Y,
    then XZ-centre + own min-Y to the origin, then per-axis scale."""
    rot = np.eye(4)
    if pre_rotation_deg:
        rot = trimesh.transformations.rotation_matrix(math.radians(float(pre_rotation_deg)), [0.0, 1.0, 0.0])
    lo, hi = np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float)
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rotated = (rot @ corners.T).T[:, :3]
    bmin, bmax = rotated.min(axis=0), rotated.max(axis=0)
    center = (bmin + bmax) / 2.0
    to_origin = trimesh.transformations.translation_matrix([-center[0], -bmin[1], -center[2]])
    s = np.asarray(scale_xyz, dtype=float)
    scaling = np.diag([s[0], s[1], s[2], 1.0])
    return scaling @ to_origin @ rot


def object_world_transform(center_xy, angle_rad: float, y0: float) -> np.ndarray:
    """`export_glb.build_scene` / `export_usd.export_usd` placeholder world
    transform (Y-up): yaw about Y, translate to the footprint centre and bbox_min_y."""
    cx, cz = float(center_xy[0]), float(center_xy[1])
    cos_a, sin_a = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    return np.array([[cos_a, 0.0, -sin_a, cx], [0.0, 1.0, 0.0, float(y0)], [sin_a, 0.0, cos_a, cz], [0.0, 0.0, 0.0, 1.0]])


def asset_stage_transform(placement: dict, bounds_min, bounds_max) -> np.ndarray:
    """Stage-frame (Z-up) transform of `/World/Objects/<id>/visual` for one
    `assets_placed.json` record: `YUP_TO_ZUP @ world @ fit`."""
    fit = fit_transform(bounds_min, bounds_max, placement["scale_xyz"], float(placement.get("pre_rotation_deg") or 0.0))
    world = object_world_transform(placement["center_xy"], placement["angle_rad"], placement.get("bbox_min_y", 0.0))
    return YUP_TO_ZUP @ world @ fit


def _gf_matrix(m: np.ndarray) -> Gf.Matrix4d:
    """USD stores row-vector matrices: `p_row @ M_usd`; ours are column-vector, so transpose."""
    return Gf.Matrix4d(*np.asarray(m, dtype=float).T.flatten().tolist())


# --------------------------------------------------------------------------- authoring into a scene stage


def placements_by_id(placements) -> dict[str, dict]:
    """`{obj_id: record}` for the records that actually carry an asset
    (`status` in `PLACED_STATUSES`, with an `asset_path`)."""
    out: dict[str, dict] = {}
    for rec in placements or []:
        if rec.get("status") in PLACED_STATUSES and rec.get("asset_path"):
            out[rec["id"]] = rec
    return out


def load_assets_placed(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def _sync_dir(src: Path, dst: Path) -> int:
    """Copy files of `src` into `dst` when missing or older; returns files copied."""
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in src.iterdir():
        if not f.is_file():
            continue
        target = dst / f.name
        if not target.exists() or target.stat().st_mtime < f.stat().st_mtime:
            shutil.copy2(f, target)
            copied += 1
    return copied


def asset_usd_location(placement: dict, *, out_dir: Path, cache_root: Path = DEFAULT_USD_CACHE_ROOT) -> tuple[str, str, Path, Path]:
    """(source, asset_id, cache dir, local dir under out_dir/usd_assets)."""
    src_glb = Path(placement["asset_path"])
    source = str(placement.get("source") or GENERATED_SOURCE)
    asset_id = _sanitize_prim_name(str(placement.get("asset_id") or src_glb.stem))
    local_dir = Path(out_dir) / USD_ASSETS_SUBDIR / source / asset_id
    cache_dir = local_dir if source == GENERATED_SOURCE else Path(cache_root) / source / asset_id
    return source, asset_id, cache_dir, local_dir


def add_asset_visual(stage: Usd.Stage, obj_path: str, placement: dict, *, out_dir: str | Path,
                     cache_root: str | Path = DEFAULT_USD_CACHE_ROOT) -> dict:
    """Author `<obj_path>/visual` (Xform with the placement transform) and
    `<obj_path>/visual/asset` (instanceable reference to the converted asset,
    synced into `<out_dir>/usd_assets/`). Returns a summary record."""
    out_dir = Path(out_dir)
    source, asset_id, cache_dir, local_dir = asset_usd_location(placement, out_dir=out_dir, cache_root=Path(cache_root))
    usd_path = cache_dir / f"{asset_id}.usd"
    info = convert_glb_to_usd(placement["asset_path"], usd_path)
    if cache_dir != local_dir:
        _sync_dir(cache_dir, local_dir)
    relative = f"./{USD_ASSETS_SUBDIR}/{source}/{asset_id}/{asset_id}.usd"

    visual = UsdGeom.Xform.Define(stage, f"{obj_path}/visual")
    xformable = UsdGeom.Xformable(visual)
    xformable.ClearXformOpOrder()
    matrix = asset_stage_transform(placement, info["bounds_min"], info["bounds_max"])
    xformable.AddTransformOp().Set(_gf_matrix(matrix))
    prim = visual.GetPrim()
    prim.CreateAttribute("cloudeye:msaAsset", Sdf.ValueTypeNames.String).Set(f"{source}:{asset_id}")
    prim.CreateAttribute("cloudeye:msaAssetStatus", Sdf.ValueTypeNames.String).Set(str(placement.get("status")))
    prim.CreateAttribute("cloudeye:msaAssetScaleXYZ", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*[float(s) for s in placement["scale_xyz"]]))

    asset_prim = stage.DefinePrim(f"{obj_path}/visual/{ASSET_PRIM_NAME}")
    asset_prim.GetReferences().AddReference(relative)
    asset_prim.SetInstanceable(True)
    return {
        "id": placement["id"], "source": source, "asset_id": asset_id, "reference": relative,
        "n_parts": info["n_parts"], "n_faces": info["n_faces"], "n_textures": info["n_textures"],
        "scale_xyz": [float(s) for s in placement["scale_xyz"]], "pre_rotation_deg": float(placement.get("pre_rotation_deg") or 0.0),
    }


# --------------------------------------------------------------------------- assertion: GLB vs USD


def glb_object_inventory(glb_path: str | Path) -> dict[str, dict]:
    """`{obj_id: {class, n_visual_parts, n_faces}}` from a `build_scene`-shaped GLB."""
    scene = trimesh.load(Path(glb_path).as_posix(), force="scene")
    out: dict[str, dict] = {}
    for node_name in scene.graph.nodes_geometry:
        if "/visual/" not in node_name and "/collision/" not in node_name:
            continue
        obj_id = node_name.split("/", 1)[0]
        rec = out.setdefault(obj_id, {"class": object_class_from_id(obj_id), "n_visual_parts": 0, "n_faces": 0})
        if "/visual/" in node_name:
            geom = scene.geometry.get(scene.graph[node_name][1])
            rec["n_visual_parts"] += 1
            rec["n_faces"] += int(len(geom.faces)) if isinstance(geom, trimesh.Trimesh) else 0
    return out


def usd_object_inventory(usd_path: str | Path) -> dict[str, dict]:
    """`{prim_name: {class, n_meshes, n_faces, has_reference, bound_materials}}` for
    every child of `/World/Objects`, composed (referenced asset prims included)."""
    stage = Usd.Stage.Open(Path(usd_path).as_posix())
    objects_root = stage.GetPrimAtPath("/World/Objects")
    out: dict[str, dict] = {}
    if not objects_root.IsValid():
        return out
    for obj in objects_root.GetChildren():
        rec = {"class": object_class_from_id(obj.GetName()), "n_meshes": 0, "n_faces": 0, "has_reference": False, "bound_materials": 0}
        visual = stage.GetPrimAtPath(f"{obj.GetPath()}/visual")
        if visual.IsValid():
            for prim in Usd.PrimRange(visual, Usd.TraverseInstanceProxies()):
                if prim.HasAuthoredReferences():
                    rec["has_reference"] = True
                if prim.IsA(UsdGeom.Mesh):
                    counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
                    rec["n_meshes"] += 1
                    rec["n_faces"] += len(counts) if counts else 0
                    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    if bound and bound.GetPrim().IsValid():
                        rec["bound_materials"] += 1
        out[obj.GetName()] = rec
    return out


def compare_glb_usd_objects(glb_path: str | Path, usd_path: str | Path, placed_ids=None) -> dict:
    """The M7 identity assertion. `placed_ids` (object ids that got an asset in the
    GLB, from `assets_placed.json`) must each have a referenced, non-box, material-
    bound mesh in the USD. Returns a dict with `ok` and every number."""
    glb = glb_object_inventory(glb_path)
    usd = usd_object_inventory(usd_path)
    # Both sides keyed by the sanitized prim name ("coffee maker_1" -> "coffee_maker_1")
    # and classed from THAT, so a label with a space compares equal.
    glb_names = {_sanitize_prim_name(k): {**v, "class": object_class_from_id(_sanitize_prim_name(k))} for k, v in glb.items()}
    glb_classes = sorted(v["class"] for v in glb_names.values())
    usd_classes = sorted(v["class"] for v in usd.values())
    missing_in_usd = sorted(set(glb_names) - set(usd))
    missing_in_glb = sorted(set(usd) - set(glb_names))
    not_real_mesh = []
    for obj_id in placed_ids or []:
        name = _sanitize_prim_name(obj_id)
        rec = usd.get(name)
        if rec is None or not rec["has_reference"] or rec["n_meshes"] == 0 or rec["n_faces"] <= BOX_FACE_COUNT or rec["bound_materials"] == 0:
            not_real_mesh.append({"id": obj_id, **(rec or {"missing": True})})
    ok = len(glb_names) == len(usd) and glb_classes == usd_classes and not missing_in_usd and not missing_in_glb and not not_real_mesh
    return {
        "ok": bool(ok),
        "glb": str(glb_path), "usd": str(usd_path),
        "glb_object_count": len(glb_names), "usd_object_count": len(usd),
        "glb_classes": glb_classes, "usd_classes": usd_classes, "classes_identical": glb_classes == usd_classes,
        "missing_in_usd": missing_in_usd, "missing_in_glb": missing_in_glb,
        "n_placed_checked": len(list(placed_ids or [])), "placed_without_real_mesh": not_real_mesh,
        "usd_objects": usd,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Convert a GLB to USD (cache) or compare a GLB with a USD (M7 assertion).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    conv = sub.add_parser("convert")
    conv.add_argument("glb", type=Path)
    conv.add_argument("out_usd", type=Path)
    conv.add_argument("--force", action="store_true")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("glb", type=Path)
    cmp_.add_argument("usd", type=Path)
    cmp_.add_argument("--assets-placed", type=Path, default=None)
    cmp_.add_argument("--out", type=Path, default=None, help="write the comparison JSON here")
    args = ap.parse_args(argv)
    if args.cmd == "convert":
        print(json.dumps(convert_glb_to_usd(args.glb, args.out_usd, force=args.force), indent=2))
        return 0
    placed = list(placements_by_id(load_assets_placed(args.assets_placed))) if args.assets_placed else []
    result = compare_glb_usd_objects(args.glb, args.usd, placed)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "usd_objects"}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
