"""M7 item 4: scripts/msa/usd_assets.py - GLB -> USD conversion, the placement
transform, referenced asset visuals in both USD exporters, and the GLB-vs-USD
identity assertion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

from app.services.usd_export import to_isaac
from scripts.msa.asset_fallback import fit_meshes_to_box, place_assets
from scripts.msa.asset_index import AssetCandidate, AssetIndex
from scripts.msa.assemble_with_generated import assemble, export_assets_usd
from scripts.msa.geometry import WallPolygon
from scripts.msa.objects_io import BootstrapObjects
from scripts.msa.usd_assets import (
    BOX_FACE_COUNT,
    asset_stage_transform,
    compare_glb_usd_objects,
    convert_glb_to_usd,
    fit_transform,
    glb_object_inventory,
    load_glb_parts,
    placements_by_id,
    usd_object_inventory,
)


def _textured_sphere_glb(path: Path, *, extents=(0.6, 1.2, 0.4), offset=(1.0, 0.5, -2.0)) -> Path:
    """A real-mesh asset: an icosphere (320 faces) squashed to `extents`, moved off
    the origin, with a 4x4 checker base-colour texture and per-vertex UVs."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    mesh.apply_scale([extents[0], extents[1], extents[2]])
    mesh.apply_translation(offset)
    uv = np.column_stack([(mesh.vertices[:, 0] - mesh.bounds[0][0]) / extents[0], (mesh.vertices[:, 1] - mesh.bounds[0][1]) / extents[1]])
    img = Image.fromarray((np.indices((4, 4)).sum(axis=0) % 2 * 255).astype(np.uint8)).convert("RGB")
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=img, roughnessFactor=0.6, metallicFactor=0.1)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    trimesh.Scene(mesh).export(path.as_posix(), file_type="glb")
    return path


def _plain_box_glb(path: Path, extents) -> Path:
    trimesh.Scene(trimesh.creation.box(extents=extents)).export(path.as_posix(), file_type="glb")
    return path


def _plain_cylinder_glb(path: Path, radius: float, height: float) -> Path:
    """An untextured but non-box asset (16 sections -> 60+ faces), Y up."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0]))
    trimesh.Scene(cyl).export(path.as_posix(), file_type="glb")
    return path


def _compact(check: dict) -> dict:
    return {k: v for k, v in check.items() if k != "usd_objects"}


class TestConvert:
    def test_textured_glb_becomes_mesh_with_normals_st_and_preview_surface(self, tmp_path: Path):
        glb = _textured_sphere_glb(tmp_path / "sphere.glb")
        info = convert_glb_to_usd(glb, tmp_path / "cache" / "sphere.usd")
        assert info["n_parts"] == 1 and info["n_faces"] == 320 and info["n_textures"] == 1
        stage = Usd.Stage.Open(str(tmp_path / "cache" / "sphere.usd"))
        assert stage.GetDefaultPrim().GetPath() == "/Asset"
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/part_0"))
        assert len(mesh.GetPointsAttr().Get()) == len(load_glb_parts(glb)[0].vertices)
        assert len(mesh.GetNormalsAttr().Get()) == len(mesh.GetPointsAttr().Get())
        assert mesh.GetSubdivisionSchemeAttr().Get() == "none"
        st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        assert st and st.GetInterpolation() == "vertex" and len(st.Get()) == len(mesh.GetPointsAttr().Get())
        bound, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
        assert bound.GetPrim().IsValid()
        surface = UsdShade.Shader(stage.GetPrimAtPath(str(bound.GetPath()) + "/PreviewSurface"))
        assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
        assert surface.GetInput("roughness").Get() == pytest.approx(0.6) and surface.GetInput("metallic").Get() == pytest.approx(0.1)
        tex = UsdShade.Shader(stage.GetPrimAtPath(str(bound.GetPath()) + "/diffuseTexture"))
        assert tex.GetIdAttr().Get() == "UsdUVTexture"
        tex_file = tex.GetInput("file").Get().path
        assert tex_file.startswith("./") and (tmp_path / "cache" / tex_file[2:]).exists()
        assert Image.open(tmp_path / "cache" / tex_file[2:]).size == (4, 4)
        # bounds in the sidecar = the GLB's own bounds (off-origin kept)
        parts = load_glb_parts(glb)
        assert np.allclose(info["bounds_min"], parts[0].bounds[0], atol=1e-5) and np.allclose(info["bounds_max"], parts[0].bounds[1], atol=1e-5)
        assert json.loads((tmp_path / "cache" / "sphere.json").read_text())["n_faces"] == 320

    def test_cache_is_reused_unless_forced(self, tmp_path: Path):
        glb = _plain_box_glb(tmp_path / "box.glb", (1.0, 1.0, 1.0))
        out = tmp_path / "box.usd"
        convert_glb_to_usd(glb, out)
        mtime = out.stat().st_mtime
        convert_glb_to_usd(glb, out)
        assert out.stat().st_mtime == mtime
        convert_glb_to_usd(glb, out, force=True)
        assert out.stat().st_mtime >= mtime

    def test_untextured_material_gets_constant_diffuse(self, tmp_path: Path):
        glb = _plain_box_glb(tmp_path / "box.glb", (1.0, 1.0, 1.0))
        info = convert_glb_to_usd(glb, tmp_path / "box.usd")
        assert info["n_textures"] == 0 and len(info["materials"]) == 1 and info["materials"][0]["texture"] is None
        stage = Usd.Stage.Open(str(tmp_path / "box.usd"))
        bound, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Asset/part_0")).ComputeBoundMaterial()
        assert bound.GetPrim().IsValid()


class TestTransform:
    @pytest.mark.parametrize("pre_rotation_deg,uniform", [(0.0, False), (90.0, False), (0.0, True)])
    def test_fit_transform_reproduces_fit_meshes_to_box(self, tmp_path: Path, pre_rotation_deg, uniform):
        parts = load_glb_parts(_textured_sphere_glb(tmp_path / "s.glb", extents=(0.6, 1.2, 0.4)))
        # length_u < width_v with a 0.6 x 0.4 asset forces the 90 deg pre-rotation.
        length_u, height, width_v = (0.5, 0.9, 0.8) if pre_rotation_deg else (0.8, 0.9, 0.5)
        fitted, info = fit_meshes_to_box(parts, length_u, height, width_v, uniform=uniform)
        assert info["pre_rotation_deg"] == pre_rotation_deg
        lo, hi = parts[0].bounds
        M = fit_transform(lo, hi, info["scale_xyz"], info["pre_rotation_deg"])
        ours = parts[0].copy()
        ours.apply_transform(M)
        assert np.allclose(ours.vertices, fitted[0].vertices, atol=1e-6)

    def test_stage_transform_matches_glb_world_placement_in_isaac_frame(self, tmp_path: Path):
        parts = load_glb_parts(_textured_sphere_glb(tmp_path / "s.glb"))
        obj = {"center_xy": (2.0, -1.5), "angle_rad": math.radians(35.0), "bbox_min_y": 0.12}
        fitted, info = fit_meshes_to_box(parts, 0.8, 0.9, 0.5)
        # What export_glb.build_scene does: local fitted mesh -> world (Y-up).
        cos_a, sin_a = math.cos(obj["angle_rad"]), math.sin(obj["angle_rad"])
        world = np.array([[cos_a, 0, -sin_a, 2.0], [0, 1, 0, 0.12], [sin_a, 0, cos_a, -1.5], [0, 0, 0, 1]])
        expected = fitted[0].copy()
        expected.apply_transform(world)
        expected_isaac = np.array([to_isaac(*p) for p in expected.vertices])
        placement = {"scale_xyz": info["scale_xyz"], "pre_rotation_deg": info["pre_rotation_deg"], **obj}
        M = asset_stage_transform(placement, *parts[0].bounds)
        ours = (M @ np.column_stack([parts[0].vertices, np.ones(len(parts[0].vertices))]).T).T[:, :3]
        assert np.allclose(ours, expected_isaac, atol=1e-6)


@pytest.fixture
def room(tmp_path: Path):
    """A bootstrap-like objects.json world with one asset per class in a fake index."""
    (tmp_path / "assets").mkdir()
    sphere = _textured_sphere_glb(tmp_path / "assets" / "chair_asset.glb")
    cylinder = _plain_cylinder_glb(tmp_path / "assets" / "table_asset.glb", 0.4, 0.7)
    index = AssetIndex(assets_root=str(tmp_path / "assets"))
    index.classes["chair"] = [AssetCandidate(source="polyhaven", id="chair_asset", path=str(sphere), cls="chair", canonical_aabb_m=(0.6, 1.2, 0.4), width_m=0.6, depth_m=0.4, height_m=1.2, is_round=False)]
    index.classes["table"] = [AssetCandidate(source="kenney", id="table_asset", path=str(cylinder), cls="table", canonical_aabb_m=(0.8, 0.7, 0.8), width_m=0.8, depth_m=0.8, height_m=0.7, is_round=False)]
    walls = [WallPolygon(vertices=[(0.0, 0.0), (6.0, 0.0), (6.0, 0.2), (0.0, 0.2)], area_m2=1.2)]
    room_polygon = [(0.2, 0.2), (5.8, 0.2), (5.8, 4.8), (0.2, 4.8)]

    def hull(cx, cz, r):
        return [[cx - r, cz - r], [cx + r, cz - r], [cx + r, cz + r], [cx - r, cz + r]]

    objects = [
        {"id": "chair_0", "label": "chair", "center_xy": (1.0, 1.0), "size_uv": (0.5, 0.5), "angle_rad": 0.3, "height": 0.9, "bbox_min_y": 0.0, "color_rgb": (50, 50, 50), "hull_xz": hull(1.0, 1.0, 0.25)},
        {"id": "chair_3", "label": "chair", "center_xy": (4.0, 1.0), "size_uv": (0.5, 0.6), "angle_rad": 0.0, "height": 0.9, "bbox_min_y": 0.0, "color_rgb": (50, 50, 50), "hull_xz": hull(4.0, 1.0, 0.25)},
        {"id": "desk_1", "label": "desk", "center_xy": (3.0, 3.0), "size_uv": (1.4, 0.7), "angle_rad": 0.0, "height": 0.75, "bbox_min_y": 0.0, "color_rgb": (120, 90, 60), "hull_xz": hull(3.0, 3.0, 0.5)},
        {"id": "curtain_0", "label": "curtain", "center_xy": (5.5, 2.5), "size_uv": (2.0, 0.1), "angle_rad": math.radians(90.0), "height": 2.4, "bbox_min_y": 0.05, "color_rgb": (200, 200, 200), "hull_xz": hull(5.5, 2.5, 0.1)},
    ]
    return BootstrapObjects(objects=objects, walls=walls, room_polygon=room_polygon, floor_y=0.0, ceiling_y=2.7), index


class TestExporters:
    def test_scene_assets_usd_references_the_same_assets_as_the_glb(self, room, tmp_path: Path):
        bootstrap, index = room
        scene, placements = assemble(bootstrap, {}, assets=True, asset_index=index)
        out = tmp_path / "out"
        out.mkdir()
        glb = out / "scene_assets.glb"
        scene.export(glb.as_posix(), file_type="glb")
        usd = export_assets_usd(bootstrap, placements, out / "scene_assets.usd", asset_cache_root=tmp_path / "usdcache")
        placed_ids = list(placements_by_id(placements))
        assert sorted(placed_ids) == ["chair_0", "chair_3", "desk_1"]  # desk -> table class via LABEL_SYNONYMS
        check = compare_glb_usd_objects(glb, usd, placed_ids)
        assert check["ok"], _compact(check)
        assert check["glb_object_count"] == check["usd_object_count"] == 4
        assert check["glb_classes"] == check["usd_classes"] == ["chair", "chair", "curtain", "desk"]
        inv = usd_object_inventory(usd)
        assert inv["chair_0"]["has_reference"] and inv["chair_0"]["n_faces"] == 320 and inv["chair_0"]["bound_materials"] == 1
        assert not inv["curtain_0"]["has_reference"] and inv["curtain_0"]["n_faces"] == BOX_FACE_COUNT  # placeholder kept, translucent
        # The referenced files live under out/usd_assets and are referenced relatively.
        stage = Usd.Stage.Open(str(usd))
        asset_prim = stage.GetPrimAtPath("/World/Objects/chair_0/visual/asset")
        assert asset_prim.IsInstanceable() and asset_prim.HasAuthoredReferences()
        assert (out / "usd_assets" / "polyhaven" / "chair_asset" / "chair_asset.usd").exists()
        assert (out / "usd_assets" / "kenney" / "table_asset" / "table_asset.usd").exists()
        # Collider untouched: the measured hull, still a collider.
        hull = stage.GetPrimAtPath("/World/Objects/chair_0/collision/hull")
        assert hull.IsValid() and hull.HasAPI(__import__("pxr").UsdPhysics.CollisionAPI)
        # The referenced mesh lands where the GLB put it (world AABB in the Isaac frame).
        glb_scene = trimesh.load(glb.as_posix(), force="scene")
        node = "chair_0/visual/part_0"
        T, geom_name = glb_scene.graph[node]
        part = glb_scene.geometry[geom_name].copy()
        part.apply_transform(T)
        glb_isaac = np.array([to_isaac(*p) for p in part.vertices])
        # Compare the composed mesh POINTS pushed through the visual's transform (a
        # BBoxCache bound of a yawed mesh is the AABB of its rotated extent box, i.e.
        # deliberately larger than the point AABB - not a placement error).
        visual = stage.GetPrimAtPath("/World/Objects/chair_0/visual")
        M = np.array(UsdGeom.Xformable(visual).GetLocalTransformation())  # row-vector convention
        meshes = [p for p in Usd.PrimRange(visual, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        assert len(meshes) == 1
        pts = np.array(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get())
        usd_world = (np.column_stack([pts, np.ones(len(pts))]) @ M)[:, :3]
        assert np.allclose(usd_world.min(axis=0), glb_isaac.min(axis=0), atol=1e-4)
        assert np.allclose(usd_world.max(axis=0), glb_isaac.max(axis=0), atol=1e-4)
        # ...and the cache bound contains the points (sanity that the reference resolves).
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(visual).ComputeAlignedRange()
        assert np.all(np.array(rng.GetMin()) <= glb_isaac.min(axis=0) + 1e-4) and np.all(np.array(rng.GetMax()) >= glb_isaac.max(axis=0) - 1e-4)

    def test_compare_flags_placeholder_boxes_and_count_mismatch(self, room, tmp_path: Path):
        bootstrap, index = room
        scene, placements = assemble(bootstrap, {}, assets=True, asset_index=index)
        out = tmp_path / "out"
        out.mkdir()
        glb = out / "scene_assets.glb"
        scene.export(glb.as_posix(), file_type="glb")
        # A USD written WITHOUT the placements: same objects, but boxes -> the assertion fails.
        usd_boxes = export_assets_usd(bootstrap, [], out / "boxes.usd", asset_cache_root=tmp_path / "usdcache")
        check = compare_glb_usd_objects(glb, usd_boxes, list(placements_by_id(placements)))
        assert not check["ok"] and {d["id"] for d in check["placed_without_real_mesh"]} == {"chair_0", "chair_3", "desk_1"}, _compact(check)
        assert check["classes_identical"] and check["glb_object_count"] == check["usd_object_count"]
        # Drop an object from the USD side: count / class mismatch.
        fewer = BootstrapObjects(objects=bootstrap.objects[:-1], walls=bootstrap.walls, room_polygon=bootstrap.room_polygon, floor_y=0.0, ceiling_y=2.7)
        usd_fewer = export_assets_usd(fewer, placements, out / "fewer.usd", asset_cache_root=tmp_path / "usdcache")
        check = compare_glb_usd_objects(glb, usd_fewer, list(placements_by_id(placements)))
        assert not check["ok"] and check["missing_in_usd"] == ["curtain_0"] and not check["classes_identical"]

    def test_presentation_usd_uses_the_assets_and_keeps_the_curtain_placeholder(self, room, tmp_path: Path):
        from scripts.msa.export_usd import export_presentation_usd

        bootstrap, index = room
        scene, placements = assemble(bootstrap, {}, assets=True, asset_index=index)
        out = tmp_path / "pres"
        out.mkdir()
        glb = out / "scene_assets.glb"
        scene.export(glb.as_posix(), file_type="glb")
        summary = export_presentation_usd(
            scene, 0.0, 2.7, out / "scene_presentation.usd", floor_polygon=bootstrap.room_polygon,
            path_points_world=[(1.0, 0.03, 2.0), (2.5, 0.03, 2.5)], target_point=(2.5, 0.05, 2.5), asset_placements=placements,
            asset_cache_root=tmp_path / "usdcache",
        )
        assert summary["n_asset_visuals"] == 3 and {a["id"] for a in summary["asset_visuals"]} == {"chair_0", "chair_3", "desk_1"}
        assert summary["n_objects"] == 4 and summary["n_visual_parts"] == 1  # only the curtain's placeholder part is authored as a mesh
        check = compare_glb_usd_objects(glb, out / "scene_presentation.usd", list(placements_by_id(placements)))
        assert check["ok"], _compact(check)
        assert summary["translucent_parts"] == ["/World/Objects/curtain_0/visual/part_0"]
        assert glb_object_inventory(glb)["chair_0"]["n_faces"] == 320
