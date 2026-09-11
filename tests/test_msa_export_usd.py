"""Stage E0 (SPEC §8): USD schema v5 prim hierarchy - visual/collision split per
object, Plan outline curves, optional Path/Target. Reuses the same
02_modular_home fixture as Stage A0's integration test (tests/test_msa_bootstrap.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from scripts.msa.bootstrap import load_object_inputs_from_hulls_json, run_bootstrap
from scripts.msa.export_usd import EXPORT_SCHEMA_VERSION, export_usd
from scripts.msa.geometry import WallPolygon

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "msa" / "02_modular_home"


@pytest.fixture(scope="module")
def bootstrap_stage(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("msa_e0_usd")
    inputs = load_object_inputs_from_hulls_json(FIXTURE_DIR / "scene_objects_hulls.json")
    run_bootstrap(FIXTURE_DIR, out_dir, object_inputs=inputs)
    stage = Usd.Stage.Open(str(out_dir / "scene.usd"))
    return stage


class TestSchemaVersion:
    def test_custom_layer_data_records_schema_v5(self, bootstrap_stage):
        assert bootstrap_stage.GetRootLayer().customLayerData["cloudeye:msaSchemaVersion"] == 5
        assert EXPORT_SCHEMA_VERSION == 5


class TestObjectHierarchy:
    def test_every_object_has_visual_and_collision_children(self, bootstrap_stage):
        object_prims = [p for p in bootstrap_stage.Traverse() if str(p.GetPath()).count("/") == 3 and str(p.GetPath()).startswith("/World/Objects/")]
        assert len(object_prims) > 0
        for prim in object_prims:
            path = str(prim.GetPath())
            assert bootstrap_stage.GetPrimAtPath(f"{path}/visual").IsValid(), f"{path} missing /visual"
            # collision may legitimately be absent-of-geometry (degenerate hull),
            # but the Xform prim itself is always created.
            assert bootstrap_stage.GetPrimAtPath(f"{path}/collision").IsValid(), f"{path} missing /collision"

    def test_collision_hull_differs_from_visual_placeholder_geometry(self, bootstrap_stage):
        """SPEC §5: collision must be the measured hull, never the generated/
        placeholder visual mesh - assert they are not the same points, for at
        least one real object where the hull isn't itself a perfect rectangle."""
        found_a_difference = False
        for prim in bootstrap_stage.Traverse():
            path = str(prim.GetPath())
            if not path.endswith("/collision/hull"):
                continue
            hull_mesh = prim
            visual_path = path.replace("/collision/hull", "/visual/part_0")
            visual_prim = bootstrap_stage.GetPrimAtPath(visual_path)
            if not visual_prim.IsValid():
                continue
            hull_points = np.array(hull_mesh.GetAttribute("points").Get())
            visual_points = np.array(visual_prim.GetAttribute("points").Get())
            if hull_points.shape != visual_points.shape or not np.allclose(hull_points, visual_points):
                found_a_difference = True
                break
        assert found_a_difference, "expected at least one object where the measured hull is not identical to the placeholder mesh"


class TestPhysicsColliders:
    """Stage E2 (GPU validation): tools/isaac_validate.py's collider_coverage and
    sphere_drop both failed on every MSA scene until this - export_usd.py never
    applied UsdPhysics.CollisionAPI to anything. See docs/DECISIONS.md."""

    def test_walls_and_floor_are_colliders(self, bootstrap_stage):
        """Every Mesh/Cube under /World/Structure that is not a `/visual/` part, plus
        the floor - exactly tools/isaac_validate.py's collider_coverage selection."""
        structure_meshes = [
            p for p in bootstrap_stage.Traverse()
            if (str(p.GetPath()).startswith("/World/Structure/") or str(p.GetPath()) == "/World/Floor")
            and p.GetTypeName() in ("Mesh", "Cube") and "/visual/" not in str(p.GetPath())
        ]
        assert structure_meshes, "expected at least one wall collider or the floor"
        for prim in structure_meshes:
            assert prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} missing CollisionAPI"

    def test_collision_hulls_are_colliders(self, bootstrap_stage):
        hull_prims = [p for p in bootstrap_stage.Traverse() if str(p.GetPath()).endswith("/collision/hull")]
        assert hull_prims, "expected at least one object with a collision hull"
        for prim in hull_prims:
            assert prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} missing CollisionAPI"
            assert prim.HasAPI(UsdPhysics.MeshCollisionAPI), f"{prim.GetPath()} missing MeshCollisionAPI"

    def test_visual_meshes_are_not_colliders(self, bootstrap_stage):
        """SPEC §5: the collider must always be the measured hull, never the
        visual/placeholder/generated mesh - visual/* must NOT carry CollisionAPI."""
        visual_prims = [p for p in bootstrap_stage.Traverse() if "/visual/" in str(p.GetPath())]
        assert visual_prims, "expected at least one visual mesh part"
        for prim in visual_prims:
            assert not prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} unexpectedly has CollisionAPI"


def _structure_collider_prims(stage):
    return [
        p for p in stage.Traverse()
        if str(p.GetPath()).startswith("/World/Structure/") and p.GetTypeName() in ("Mesh", "Cube") and "/visual/" not in str(p.GetPath())
    ]


def _footprint_area_xy(stage, prim) -> float:
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    size = rng.GetMax() - rng.GetMin()
    return float(size[0] * size[1])


class TestWallBand:
    """T15g: the bootstrap USD carries ONE structural wall, `wall_outline` (full
    height = ceiling_y - floor_y), instead of per-fragment `wall_i` meshes; the
    fragments stay as `/World/Plan/wall_i` curves. T15h: the band is ring-shaped,
    so its collider is `wall_outline_collider/edge_i` boxes (one per outline edge)
    and the band mesh itself is the visual `wall_outline/visual/band` - never a
    convexHull collider (which would fill the whole room, T15e)."""

    def test_structure_has_the_band_visual_and_per_edge_box_colliders(self, bootstrap_stage):
        import json

        meta = json.loads((Path(bootstrap_stage.GetRootLayer().realPath).parent / "scene_meta.json").read_text())
        band = bootstrap_stage.GetPrimAtPath("/World/Structure/wall_outline/visual/band")
        assert band.IsValid() and band.GetTypeName() == "Mesh" and not band.HasAPI(UsdPhysics.CollisionAPI)
        colliders = _structure_collider_prims(bootstrap_stage)
        assert colliders and all(p.GetTypeName() == "Cube" for p in colliders)
        assert all(str(p.GetPath()).startswith("/World/Structure/wall_outline_collider/edge_") for p in colliders)
        assert len(colliders) == len(meta["room_polygon"])
        for prim in colliders:
            assert prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert not any(
            p.HasAPI(UsdPhysics.MeshCollisionAPI) and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
            for p in bootstrap_stage.Traverse() if str(p.GetPath()).startswith("/World/Structure/")
        )

    def test_wall_outline_spans_floor_to_ceiling(self, bootstrap_stage):
        import json

        meta = json.loads((Path(bootstrap_stage.GetRootLayer().realPath).parent / "scene_meta.json").read_text())
        pts = np.array(bootstrap_stage.GetPrimAtPath("/World/Structure/wall_outline/visual/band").GetAttribute("points").Get())
        assert pts[:, 2].min() == pytest.approx(meta["floor_y"], abs=1e-6)  # Isaac Z-up: z is height
        assert pts[:, 2].max() == pytest.approx(meta["ceiling_y"], abs=1e-6)
        assert meta["wall_height_m"] == pytest.approx(meta["ceiling_y"] - meta["floor_y"])
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for prim in _structure_collider_prims(bootstrap_stage):
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            assert rng.GetMin()[2] == pytest.approx(meta["floor_y"], abs=1e-6)
            assert rng.GetMax()[2] == pytest.approx(meta["ceiling_y"], abs=1e-6)

    def test_no_structure_collider_footprint_exceeds_1_5x_the_band(self, bootstrap_stage):
        """The T15e failure mode, as a number: a convexHull of the ring would be the
        whole room (~10x the band); every T15h box is a sliver of it."""
        from scripts.msa.export_usd import collider_footprint_stats

        band_prim = bootstrap_stage.GetPrimAtPath("/World/Structure/wall_outline/visual/band")
        pts = np.array(band_prim.GetAttribute("points").Get())
        yup = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])  # Isaac -> Y-up
        import trimesh

        faces = np.array(band_prim.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)
        stats = collider_footprint_stats(trimesh.Trimesh(yup, faces, process=False))
        band_area = stats["footprint_area_m2"]
        assert stats["ratio"] > 1.25  # the ring IS the case the guard exists for
        for prim in _structure_collider_prims(bootstrap_stage):
            assert _footprint_area_xy(bootstrap_stage, prim) <= 1.5 * band_area
        assert sum(_footprint_area_xy(bootstrap_stage, p) for p in _structure_collider_prims(bootstrap_stage)) <= 1.5 * band_area


class TestConvexHullGuard:
    """T15h: `_add_collision_mesh` refuses convexHull for a concave mesh."""

    def test_concave_mesh_is_downgraded_to_none_and_convex_kept(self, tmp_path):
        from scripts.msa.export_usd import CONVEX_HULL_MAX_AREA_RATIO, _add_collision_mesh, _extrude_world_polygon_xz, collider_footprint_stats

        l_shape = np.array([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)], dtype=float)  # 5 m^2, hull 7 m^2
        box = np.array([(0, 0), (2, 0), (2, 1), (0, 1)], dtype=float)
        concave = _extrude_world_polygon_xz(l_shape, 0.0, 2.0)
        convex = _extrude_world_polygon_xz(box, 0.0, 0.7)
        assert collider_footprint_stats(concave)["ratio"] == pytest.approx(1.4, abs=1e-6)
        assert collider_footprint_stats(convex)["ratio"] == pytest.approx(1.0, abs=1e-6)
        assert CONVEX_HULL_MAX_AREA_RATIO == 1.25
        stage = Usd.Stage.CreateNew(str(tmp_path / "g.usd"))
        UsdGeom.Xform.Define(stage, "/World")
        a = _add_collision_mesh(stage, "/World/concave", concave)
        b = _add_collision_mesh(stage, "/World/convex", convex)
        assert UsdPhysics.MeshCollisionAPI(a).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
        assert UsdPhysics.MeshCollisionAPI(b).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
        with pytest.raises(ValueError):
            _add_collision_mesh(stage, "/World/strict", concave, strict=True)
        with pytest.raises(ValueError):
            _add_collision_mesh(stage, "/World/bad", convex, approximation="boundingCube")

    def test_edge_boxes_cover_the_band_and_only_the_band(self, tmp_path):
        """Per-edge boxes of an L-shaped Manhattan room: every box lies inside the
        band's outer rectangle and outside the room, adjacent boxes overlap at
        the corners, the union covers the band."""
        from shapely.geometry import Polygon, box as shp_box
        from shapely.ops import unary_union

        from scripts.msa.export_usd import add_wall_edge_box_colliders, polygon_edges_ccw
        from scripts.msa.geometry import room_wall_band

        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 3.0), (0.0, 3.0)]  # L, 6 edges, one reflex corner
        t = 0.1
        assert len(polygon_edges_ccw(list(reversed(room)))) == 6  # orientation-agnostic
        stage = Usd.Stage.CreateNew(str(tmp_path / "e.usd"))
        UsdGeom.Xform.Define(stage, "/World")
        recs = add_wall_edge_box_colliders(stage, "/World/wc", room, 0.0, 2.5, thickness_m=t)
        assert len(recs) == 6 and all(r["height_m"] == pytest.approx(2.5) for r in recs)
        boxes = []
        for r in recs:
            (x0, z0), (x1, z1) = r["edge_xz"]
            cx, cz = r["center_xz"]
            half_len, half_t = r["length_m"] / 2.0, t / 2.0
            if abs(x1 - x0) > abs(z1 - z0):  # along X
                boxes.append(shp_box(cx - half_len, cz - half_t, cx + half_len, cz + half_t))
            else:
                boxes.append(shp_box(cx - half_t, cz - half_len, cx + half_t, cz + half_len))
        room_poly = Polygon(room)
        band = room_wall_band(room, t)
        union = unary_union(boxes)
        assert union.intersection(room_poly).area < 1e-9  # nothing inside the room
        assert band.difference(union).area < 1e-9  # the whole band is covered (corners included)
        assert union.area <= band.area * 1.5
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Cube":
                assert prim.HasAPI(UsdPhysics.CollisionAPI)
                assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
                assert prim.IsActive()


class TestPlanCurves:
    def test_plan_has_one_curve_per_wall_plus_floor(self, bootstrap_stage):
        plan_children = [
            p for p in bootstrap_stage.Traverse() if str(p.GetPath()).startswith("/World/Plan/")
        ]
        names = {str(p.GetPath()).rsplit("/", 1)[-1] for p in plan_children}
        assert "floor" in names
        assert any(n.startswith("wall_") for n in names)


class TestOptionalPathAndTarget:
    def test_path_and_target_absent_when_not_provided(self, bootstrap_stage):
        assert not bootstrap_stage.GetPrimAtPath("/World/Path").IsValid()
        assert not bootstrap_stage.GetPrimAtPath("/World/Target").IsValid()

    def test_path_and_target_present_when_provided(self, tmp_path):
        out_path = tmp_path / "scene_with_path.usd"
        wall = WallPolygon(vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], area_m2=1.0)
        export_usd(
            [wall],
            None,
            0.0,
            2.5,
            [],
            out_path,
            path_points_world=[(0.0, 0.05, 0.0), (0.5, 0.05, 0.5), (1.0, 0.05, 1.0)],
            target_point=(1.0, 0.05, 1.0),
        )
        stage = Usd.Stage.Open(str(out_path))
        assert stage.GetPrimAtPath("/World/Path").IsValid()
        assert stage.GetPrimAtPath("/World/Target").IsValid()
        radius = stage.GetPrimAtPath("/World/Target").GetAttribute("radius").Get()
        assert radius == pytest.approx(0.1)
        # morning-4: the 3 cm ribbon Mesh next to the curve, cyan emissive UsdPreviewSurface, no collision
        from pxr import UsdPhysics, UsdShade

        ribbon = stage.GetPrimAtPath("/World/PathRibbon")
        assert ribbon.IsA(UsdGeom.Mesh) and not ribbon.HasAPI(UsdPhysics.CollisionAPI)
        pts = np.asarray(UsdGeom.Mesh(ribbon).GetPointsAttr().Get())
        assert len(pts) == 6 and np.allclose(pts[:, 2], 0.05)  # 3 path points x 2 sides, at the path's height (Isaac z-up)
        mat_targets = UsdShade.MaterialBindingAPI(ribbon).GetDirectBindingRel().GetTargets()
        assert [str(t) for t in mat_targets] == ["/World/Looks/path_ribbon"]
        shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/path_ribbon/PreviewSurface"))
        assert tuple(shader.GetInput("emissiveColor").Get()) == pytest.approx((0.25, 0.85, 1.0))


# --- T15d: presentation variant --------------------------------------------------------


def _look_at_camera(pos, look_at):
    pos = np.asarray(pos, dtype=float)
    forward = np.asarray(look_at, dtype=float) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return {
        "pos": pos.tolist(), "forward": forward.tolist(), "right": right.tolist(), "up": up.tolist(),
        "fov_v_deg": 55.0, "pitch_deg": float(np.degrees(np.arcsin(forward[1]))), "distance_m": 5.0, "backoff_m": 0.5,
    }


@pytest.fixture(scope="module")
def presentation_case(tmp_path_factory):
    from scripts.msa.export_glb import build_scene
    from scripts.msa.export_usd import export_presentation_usd

    out_path = tmp_path_factory.mktemp("msa_presentation") / "scene_presentation.usd"
    floor_y, ceiling_y = 0.0, 2.7
    walls = [
        WallPolygon(vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 0.1), (0.0, 0.1)], area_m2=0.4),
        WallPolygon(vertices=[(0.0, 0.0), (0.1, 0.0), (0.1, 3.0), (0.0, 3.0)], area_m2=0.3),
    ]
    floor_polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    objects = [{
        "id": "desk_0", "label": "desk", "center_xy": (3.0, 1.5), "size_uv": (1.2, 0.6), "angle_rad": 0.0,
        "height": 0.75, "bbox_min_y": 0.0, "color_rgb": (120, 90, 60),
        "hull_xz": [(2.4, 1.2), (3.6, 1.2), (3.6, 1.8), (2.4, 1.8)],
    }]
    scene = build_scene(walls, floor_polygon, floor_y, ceiling_y, objects)
    camera = _look_at_camera((6.0, 2.6, -2.0), (2.0, 0.0, 1.5))
    summary = export_presentation_usd(
        scene, floor_y, ceiling_y, out_path, floor_polygon=floor_polygon,
        path_points_world=[(0.5, 0.03, 0.5), (2.0, 0.03, 1.0)], target_point=(3.0, 0.05, 1.5), camera=camera,
    )
    return {"stage": Usd.Stage.Open(str(out_path)), "summary": summary, "floor_y": floor_y, "ceiling_y": ceiling_y, "camera": camera}


class TestPresentationVariant:
    """T15d: walls cut to 1.2 m for the eye, full-height invisible colliders for
    physics, lights + camera baked in. Points are in Isaac's Z-up frame (z = Y-up y)."""

    @staticmethod
    def _z_extent(prim):
        pts = np.array(prim.GetAttribute("points").Get())
        return float(pts[:, 2].min()), float(pts[:, 2].max())

    def test_visual_wall_is_a_1_2m_stub_without_collision(self, presentation_case):
        from scripts.msa.export_usd import PRESENTATION_WALL_HEIGHT_M

        stage, floor_y = presentation_case["stage"], presentation_case["floor_y"]
        stub = stage.GetPrimAtPath("/World/Structure/wall_0/visual/stub")
        assert stub.IsValid() and stub.GetTypeName() == "Mesh"
        zmin, zmax = self._z_extent(stub)
        assert zmin == pytest.approx(floor_y, abs=1e-6)
        assert zmax <= floor_y + PRESENTATION_WALL_HEIGHT_M + 1e-6
        assert zmax == pytest.approx(floor_y + PRESENTATION_WALL_HEIGHT_M, abs=1e-6)
        assert not stub.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(stub).ComputeVisibility() == UsdGeom.Tokens.inherited

    def test_collider_wall_is_full_height_invisible_collider(self, presentation_case):
        stage, ceiling_y = presentation_case["stage"], presentation_case["ceiling_y"]
        for i in (0, 1):
            collider = stage.GetPrimAtPath(f"/World/Structure/wall_{i}_collider")
            assert collider.IsValid() and collider.GetTypeName() == "Mesh"
            _, zmax = self._z_extent(collider)
            assert zmax == pytest.approx(ceiling_y, abs=1e-6)
            assert collider.HasAPI(UsdPhysics.CollisionAPI)
            assert collider.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert UsdGeom.Imageable(collider).ComputeVisibility() == UsdGeom.Tokens.invisible
            # visibility is what hides it - never deactivation (PhysX tensor-view crash, v2 entry)
            assert collider.IsActive()

    def test_floor_and_object_hull_unchanged_visual_part_not_a_collider(self, presentation_case):
        stage = presentation_case["stage"]
        floor = stage.GetPrimAtPath("/World/Floor")
        assert floor.IsValid() and floor.HasAPI(UsdPhysics.CollisionAPI)
        hull = stage.GetPrimAtPath("/World/Objects/desk_0/collision/hull")
        assert hull.IsValid() and hull.HasAPI(UsdPhysics.CollisionAPI) and hull.HasAPI(UsdPhysics.MeshCollisionAPI)
        part = stage.GetPrimAtPath("/World/Objects/desk_0/visual/part_0")
        assert part.IsValid() and not part.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Gprim(part).GetDisplayColorAttr().Get() is not None

    def test_both_lights_present_with_presentation_intensities(self, presentation_case):
        from pxr import UsdLux

        from scripts.msa.export_usd import (
            PRESENTATION_BACKDROP_COLOR, PRESENTATION_DOME_INTENSITY, PRESENTATION_KEY_LIGHT_INTENSITY,
        )

        stage = presentation_case["stage"]
        dome = stage.GetPrimAtPath("/World/DomeLight")
        assert dome.IsValid() and dome.IsA(UsdLux.DomeLight)
        assert UsdLux.DomeLight(dome).GetIntensityAttr().Get() == pytest.approx(PRESENTATION_DOME_INTENSITY)
        # 1200 = usd_export.DOME_LIGHT_INTENSITY; T15d's 3 x 1200 washed the room out
        # on the real RTX render (T15e GPU still, docs/DECISIONS.md 2026-09-07).
        assert PRESENTATION_DOME_INTENSITY == pytest.approx(1200.0)
        color = UsdLux.DomeLight(dome).GetColorAttr().Get()
        assert tuple(color) == pytest.approx(PRESENTATION_BACKDROP_COLOR, abs=1e-6)
        key = stage.GetPrimAtPath("/World/KeyLight")
        assert key.IsValid() and key.IsA(UsdLux.DistantLight)
        assert UsdLux.DistantLight(key).GetIntensityAttr().Get() == pytest.approx(PRESENTATION_KEY_LIGHT_INTENSITY)
        # emits downward (Z-up) at 45 deg elevation
        m = UsdGeom.Xformable(key).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        emit = m.TransformDir((0.0, 0.0, -1.0))
        assert emit[2] == pytest.approx(-np.sin(np.radians(45.0)), abs=1e-6)
        assert sum(1 for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)) == 1
        assert sum(1 for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)) == 1

    def test_camera_prim_present_and_points_along_the_rule_forward(self, presentation_case):
        from scripts.msa.export_usd import PRESENTATION_CAMERA_PATH

        stage, camera = presentation_case["stage"], presentation_case["camera"]
        cam = stage.GetPrimAtPath(PRESENTATION_CAMERA_PATH)
        assert cam.IsValid() and cam.IsA(UsdGeom.Camera)
        m = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = m.Transform((0.0, 0.0, 0.0))
        fwd = m.TransformDir((0.0, 0.0, -1.0))
        px, py, pz = camera["pos"]
        fx, fy, fz = camera["forward"]
        assert tuple(eye) == pytest.approx((px, -pz, py), abs=1e-6)
        assert tuple(fwd) == pytest.approx((fx, -fz, fy), abs=1e-6)
        geom = UsdGeom.Camera(cam)
        h_ap, v_ap = geom.GetHorizontalApertureAttr().Get(), geom.GetVerticalApertureAttr().Get()
        assert h_ap / v_ap == pytest.approx(16.0 / 9.0, abs=1e-6)
        fov_v = np.degrees(2 * np.arctan(v_ap / (2 * geom.GetFocalLengthAttr().Get())))
        assert fov_v == pytest.approx(55.0, abs=1e-6)
        assert cam.GetAttribute("cloudeye:cameraBackoffM").Get() == pytest.approx(0.5)

    def test_path_target_and_plan_present(self, presentation_case):
        stage = presentation_case["stage"]
        assert stage.GetPrimAtPath("/World/Path").IsValid()
        assert stage.GetPrimAtPath("/World/Target").IsValid()
        assert stage.GetPrimAtPath("/World/Plan/floor").IsValid()
        assert presentation_case["summary"]["n_walls"] == 2

    def test_wall_band_scene_exports_outline_stub_and_edge_box_colliders(self, tmp_path):
        """T15g/T15h: a `build_scene(wall_band=...)` scene (one `wall_outline` node)
        goes through the presentation exporter as `wall_outline/visual/stub`
        (1.2 m, no collision) + `wall_outline_collider/edge_i` boxes (one per
        outline edge, full height, invisible, CollisionAPI only - never a convexHull
        of the ring). Floor and object hulls keep their convexHull colliders."""
        from scripts.msa.export_glb import build_scene
        from scripts.msa.export_usd import PRESENTATION_WALL_HEIGHT_M, export_presentation_usd
        from scripts.msa.geometry import room_wall_band

        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.5, 2.0), (2.5, 3.0), (0.0, 3.0)]
        objects = [{
            "id": "desk_0", "label": "desk", "center_xy": (3.0, 1.0), "size_uv": (1.2, 0.6), "angle_rad": 0.0,
            "height": 0.75, "bbox_min_y": 0.0, "color_rgb": (120, 90, 60),
            "hull_xz": [(2.4, 0.7), (3.6, 0.7), (3.6, 1.3), (2.4, 1.3)],
        }]
        band = room_wall_band(room, 0.1)
        scene = build_scene([], room, 0.0, 2.7, objects, wall_band=band)
        summary = export_presentation_usd(
            scene, 0.0, 2.7, tmp_path / "p.usd", floor_polygon=room, path_points_world=[(0.5, 0.03, 0.5), (2.0, 0.03, 1.0)],
            target_point=(2.0, 0.05, 1.0), target_object_point=(3.0, 0.05, 1.0), target_object_id="desk_0",
        )
        stage = Usd.Stage.Open(str(tmp_path / "p.usd"))
        # T15i: the stub is one box Mesh per outline edge (no single `visual/stub`)
        assert not stage.GetPrimAtPath("/World/Structure/wall_outline/visual/stub").IsValid()
        stubs = [p for p in stage.GetPrimAtPath("/World/Structure/wall_outline/visual").GetChildren()]
        assert len(stubs) == 6 and all(p.GetTypeName() == "Mesh" and p.GetName().startswith("stub_edge_") for p in stubs)
        for stub in stubs:
            assert not stub.HasAPI(UsdPhysics.CollisionAPI)
            assert self._z_extent(stub)[1] == pytest.approx(PRESENTATION_WALL_HEIGHT_M, abs=1e-6)
            # no camera -> nothing culled, every edge visible
            assert UsdGeom.Imageable(stub).ComputeVisibility() == UsdGeom.Tokens.inherited
        assert summary["culled_stub_edges"] == [] and len(summary["stub_edges"]) == 6
        root = stage.GetPrimAtPath("/World/Structure/wall_outline_collider")
        assert root.IsValid() and root.GetTypeName() == "Xform"
        edges = [p for p in root.GetChildren()]
        assert len(edges) == 6 and all(p.GetTypeName() == "Cube" for p in edges)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for prim in edges:
            assert prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
            assert prim.IsActive()
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            assert rng.GetMin()[2] == pytest.approx(0.0, abs=1e-6) and rng.GetMax()[2] == pytest.approx(2.7, abs=1e-6)
            assert _footprint_area_xy(stage, prim) <= 1.5 * band.area
        assert sum(_footprint_area_xy(stage, p) for p in _structure_collider_prims(stage)) <= 1.5 * band.area
        # no convexHull anywhere under /World/Structure
        for p in stage.Traverse():
            if str(p.GetPath()).startswith("/World/Structure/") and p.HasAPI(UsdPhysics.MeshCollisionAPI):
                assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != UsdPhysics.Tokens.convexHull
        # floor + object hull unchanged (convexHull colliders as before)
        floor = stage.GetPrimAtPath("/World/Floor")
        assert floor.HasAPI(UsdPhysics.CollisionAPI) and floor.HasAPI(UsdPhysics.MeshCollisionAPI)
        hull = stage.GetPrimAtPath("/World/Objects/desk_0/collision/hull")
        assert UsdPhysics.MeshCollisionAPI(hull).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
        assert summary["n_walls"] == 1
        assert summary["wall_colliders"] == [pytest.approx({
            "wall": "wall_outline", "kind": "edge_boxes", "prim": "/World/Structure/wall_outline_collider",
            "n_edges": 6, "thickness_m": 0.1, "height_m": 2.7, "band_footprint_m2": band.area,
        })]
        # T15h target semantics: /World/Target = goal, /World/TargetObject = the desk's centroid
        target = stage.GetPrimAtPath("/World/Target")
        marker = stage.GetPrimAtPath("/World/TargetObject")
        assert tuple(UsdGeom.Xformable(target).GetOrderedXformOps()[0].Get()) == pytest.approx((2.0, -1.0, 0.05))
        assert tuple(UsdGeom.Xformable(marker).GetOrderedXformOps()[0].Get()) == pytest.approx((3.0, -1.0, 0.05))
        assert marker.GetAttribute("cloudeye:targetObjectId").Get() == "desk_0"
        assert not marker.HasAPI(UsdPhysics.CollisionAPI) and not target.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Sphere(marker).GetRadiusAttr().Get() < UsdGeom.Sphere(target).GetRadiusAttr().Get()

    def test_stub_edge_boxes_tile_the_band_once_and_cull_by_camera_side(self):
        """T15i: one box per outline edge, extended one thickness past its convex START
        vertex only, so the union is the band (mitred corners covered) with no
        overlap and nothing inside the room; `culled_stub_edge_ids` drops exactly the
        edges whose line separates the camera from the room centroid."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        from scripts.msa.export_usd import culled_stub_edge_ids, room_centroid_xz, stub_edge_boxes, stub_edge_occluder_box, stub_edge_trimesh
        from scripts.msa.geometry import room_wall_band

        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.5, 2.0), (2.5, 3.0), (0.0, 3.0)]  # L room: 5 convex + 1 reflex vertex
        t = 0.1
        recs = stub_edge_boxes(room, 0.0, 1.2, t)
        assert [r["edge"] for r in recs] == list(range(6))
        footprints = []
        for r in recs:
            ux, uz = r["dir_xz"]
            nx, nz = r["normal_xz"]
            cx, cz = r["center_xz"]
            hl, ht = r["length_m"] / 2.0, r["thickness_m"] / 2.0
            corners = [(cx + ux * a * hl + nx * b * ht, cz + uz * a * hl + nz * b * ht) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            footprints.append(Polygon(corners))
            assert r["y0"] == 0.0 and r["y1"] == pytest.approx(1.2)
            # the trimesh box and the occluder box describe the same solid
            mesh = stub_edge_trimesh(r)
            box = stub_edge_occluder_box(r)
            assert np.asarray(mesh.bounds)[:, 1].tolist() == pytest.approx([0.0, 1.2])
            assert box["center"][1] == pytest.approx(0.6) and box["half"][1] == pytest.approx(0.6)
            assert mesh.volume == pytest.approx(8 * box["half"][0] * box["half"][1] * box["half"][2])
        band = room_wall_band(room, t)
        union = unary_union(footprints)
        assert union.area == pytest.approx(band.area, rel=1e-6)  # covers the band exactly ...
        assert sum(f.area for f in footprints) == pytest.approx(band.area, rel=1e-6)  # ... with no overlap
        assert union.intersection(Polygon(room)).area == pytest.approx(0.0, abs=1e-9)  # ... nothing inside the room
        # convex start vertex -> extended by t; the reflex vertex (2.5, 2.0) starts edge 3 -> retracted by t
        assert recs[3]["edge_xz"][0] == pytest.approx([2.5, 2.0]) and recs[3]["length_m"] == pytest.approx(1.0 - t)
        assert recs[0]["length_m"] == pytest.approx(4.0 + t)
        centroid = room_centroid_xz(room)
        # camera outside the bottom-left corner: the bottom (y=0) and left (x=0) edges face it
        culled = culled_stub_edge_ids(recs, (-2.0, -1.5), centroid)
        culled_edges = {tuple(map(tuple, recs[i]["edge_xz"])) for i in culled}
        assert culled_edges == {((0.0, 0.0), (4.0, 0.0)), ((0.0, 3.0), (0.0, 0.0))}
        # camera far to the right, level with the notch: the right edge AND the notch's inner
        # edge (2.5, 2.0)-(2.5, 3.0), whose outward normal points away from the room centroid,
        # are on the far side of their lines from the centroid -> culled (the web viewer rule)
        culled_right = culled_stub_edge_ids(recs, (7.0, 2.5), centroid)
        assert ((4.0, 0.0), (4.0, 2.0)) in {tuple(map(tuple, recs[i]["edge_xz"])) for i in culled_right}
        assert ((2.5, 2.0), (2.5, 3.0)) in {tuple(map(tuple, recs[i]["edge_xz"])) for i in culled_right}
        # inside the room nothing is culled
        assert culled_stub_edge_ids(recs, (1.0, 1.0), centroid) == []

    def test_presentation_culls_the_edges_facing_the_camera_and_records_them(self, tmp_path):
        from scripts.msa.export_glb import build_scene
        from scripts.msa.export_usd import export_presentation_usd
        from scripts.msa.geometry import room_wall_band

        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        scene = build_scene([], room, 0.0, 2.7, [], wall_band=room_wall_band(room, 0.1))
        camera = _look_at_camera((-3.0, 2.6, -2.0), (2.0, 0.0, 1.5))
        camera["room_centroid_xz"] = [2.0, 1.5]
        summary = export_presentation_usd(scene, 0.0, 2.7, tmp_path / "c.usd", floor_polygon=room, camera=camera)
        stage = Usd.Stage.Open(str(tmp_path / "c.usd"))
        culled = summary["culled_stub_edges"]
        assert len(culled) == 2  # the two near edges of a rectangle
        for entry in summary["stub_edges"]:
            prim = stage.GetPrimAtPath(entry["prim"])
            expect = UsdGeom.Tokens.invisible if entry["edge"] in culled else UsdGeom.Tokens.inherited
            assert UsdGeom.Imageable(prim).ComputeVisibility() == expect
            assert prim.GetAttribute("cloudeye:stubEdgeCulled").Get() is (entry["edge"] in culled)
            assert prim.IsActive()
        # the colliders are untouched by the cull: 4 invisible boxes, all active
        boxes = stage.GetPrimAtPath("/World/Structure/wall_outline_collider").GetChildren()
        assert len(boxes) == 4 and all(p.HasAPI(UsdPhysics.CollisionAPI) and p.IsActive() for p in boxes)
        cam = stage.GetPrimAtPath("/World/PresentationCamera")
        assert list(cam.GetAttribute("cloudeye:culledStubEdges").Get()) == culled

    def test_translucent_classes_get_a_bound_preview_surface_in_both_exporters(self, tmp_path):
        from pxr import UsdShade

        from scripts.msa.export_glb import build_scene
        from scripts.msa.export_usd import TRANSLUCENT_OPACITY, export_presentation_usd

        objects = [
            {"id": "curtain_0", "label": "curtain", "center_xy": (0.3, 1.5), "size_uv": (0.1, 2.0), "angle_rad": 0.0,
             "height": 2.7, "bbox_min_y": 0.0, "color_rgb": (230, 230, 235), "hull_xz": [(0.25, 0.5), (0.35, 0.5), (0.35, 2.5), (0.25, 2.5)]},
            {"id": "bed_0", "label": "bed", "center_xy": (2.0, 1.5), "size_uv": (2.0, 1.5), "angle_rad": 0.0,
             "height": 0.5, "bbox_min_y": 0.0, "color_rgb": (120, 90, 60), "hull_xz": [(1.0, 0.75), (3.0, 0.75), (3.0, 2.25), (1.0, 2.25)]},
        ]
        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

        def check(stage, curtain_path, bed_path):
            curtain = stage.GetPrimAtPath(curtain_path)
            bed = stage.GetPrimAtPath(bed_path)
            assert curtain.IsValid() and bed.IsValid()
            rel = UsdShade.MaterialBindingAPI(curtain).GetDirectBindingRel()
            targets = rel.GetTargets()
            assert len(targets) == 1 and str(targets[0]).startswith("/World/Looks/translucent_curtain_0")
            shader = UsdShade.Shader(stage.GetPrimAtPath(f"{targets[0]}/PreviewSurface"))
            assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
            assert shader.GetInput("opacity").Get() == pytest.approx(TRANSLUCENT_OPACITY)
            assert not UsdShade.MaterialBindingAPI(bed).GetDirectBindingRel().GetTargets()
            assert not curtain.HasAPI(UsdPhysics.CollisionAPI)

        export_usd([], None, 0.0, 2.7, objects, tmp_path / "scene.usd", floor_polygon=room)
        check(Usd.Stage.Open(str(tmp_path / "scene.usd")), "/World/Objects/curtain_0/visual/part_0", "/World/Objects/bed_0/visual/part_0")
        scene = build_scene([], room, 0.0, 2.7, objects)
        summary = export_presentation_usd(scene, 0.0, 2.7, tmp_path / "p.usd", floor_polygon=room)
        stage = Usd.Stage.Open(str(tmp_path / "p.usd"))
        check(stage, "/World/Objects/curtain_0/visual/part_0", "/World/Objects/bed_0/visual/part_0")
        assert summary["translucent_parts"] == ["/World/Objects/curtain_0/visual/part_0"]
        assert summary["translucent_opacity"] == pytest.approx(0.3)
        # the collision hull of the curtain is still a normal collider
        hull = stage.GetPrimAtPath("/World/Objects/curtain_0/collision/hull")
        assert hull.HasAPI(UsdPhysics.CollisionAPI)

    def test_camera_bakes_aim_and_occluder_boxes_in_isaac_frame(self, tmp_path):
        from scripts.msa.export_glb import build_scene
        from scripts.msa.export_usd import export_presentation_usd
        from scripts.msa.visibility import boxes_from_flat, make_box

        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        scene = build_scene([], room, 0.0, 2.7, [])
        camera = _look_at_camera((-3.0, 2.6, -2.0), (2.0, 0.0, 1.5))
        camera["aim"] = [2.0, 0.0, 1.5]
        camera["occluders"] = [make_box("stub_edge_1", (4.05, 0.6, 1.5), ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)), (1.6, 0.6, 0.05))]
        camera["height_raise_m"] = 0.5
        camera["room_frame_coverage"] = 0.66
        camera["visibility_ok"] = True
        export_presentation_usd(scene, 0.0, 2.7, tmp_path / "a.usd", camera=camera)
        stage = Usd.Stage.Open(str(tmp_path / "a.usd"))
        cam = stage.GetPrimAtPath("/World/PresentationCamera")
        assert tuple(cam.GetAttribute("cloudeye:cameraAimPoint").Get()) == pytest.approx((2.0, -1.5, 0.0))  # to_isaac
        boxes = boxes_from_flat(list(cam.GetAttribute("cloudeye:occluderBoxes").Get()), list(cam.GetAttribute("cloudeye:occluderIds").Get()))
        assert len(boxes) == 1 and boxes[0]["id"] == "stub_edge_1"
        assert boxes[0]["center"] == pytest.approx((4.05, -1.5, 0.6))
        assert boxes[0]["axes"][1] == pytest.approx((0.0, 0.0, 1.0))  # Y-up height axis -> Isaac Z
        assert boxes[0]["half"] == pytest.approx((1.6, 0.6, 0.05))
        assert cam.GetAttribute("cloudeye:cameraHeightRaiseM").Get() == pytest.approx(0.5)
        assert cam.GetAttribute("cloudeye:roomFrameCoverage").Get() == pytest.approx(0.66)
        assert cam.GetAttribute("cloudeye:cameraVisibilityOk").Get() is True

    def test_wall_stub_mesh_clamps_only_the_top_ring(self):
        from scripts.msa.export_glb import _extrude_wall
        from scripts.msa.export_usd import wall_stub_mesh

        wall = WallPolygon(vertices=[(0.0, 0.0), (2.0, 0.0), (2.0, 0.2), (0.0, 0.2)], area_m2=0.4)
        mesh = _extrude_wall(wall, 0.5, 3.2)
        stub = wall_stub_mesh(mesh, 0.5, 1.2)
        ys = np.unique(np.round(np.asarray(stub.vertices)[:, 1], 6))
        assert ys.tolist() == pytest.approx([0.5, 1.7])
        assert len(stub.faces) == len(mesh.faces)
        assert np.asarray(mesh.vertices)[:, 1].max() == pytest.approx(3.2)  # input untouched
