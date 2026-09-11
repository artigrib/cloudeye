import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.msa.asset_index import AssetCandidate, AssetIndex
from scripts.msa.assemble_with_generated import _fit_generated_mesh_to_footprint, assemble, load_generated
from scripts.msa.geometry import WallPolygon
from scripts.msa.objects_io import OBJECTS_JSON_SCHEMA_VERSION, load_objects_json, write_objects_json


def test_fit_matches_build_placeholder_local_convention():
    """scripts/msa/assets.py's build_placeholder convention: centered at origin
    on XZ, Y up starting at 0. A fitted generated mesh must land in the same
    local frame so it can share the identical world_transform."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))  # arbitrary source bbox
    fitted = _fit_generated_mesh_to_footprint(box, length_u=1.2, height=0.8, width_v=0.6)
    bmin, bmax = fitted.bounds
    assert np.allclose(bmin, [-0.6, 0.0, -0.3], atol=1e-6)
    assert np.allclose(bmax, [0.6, 0.8, 0.3], atol=1e-6)


def test_fit_handles_off_center_source_bbox():
    # Source mesh not centered at its own origin - must still normalize correctly.
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation((5.0, -3.0, 10.0))
    fitted = _fit_generated_mesh_to_footprint(box, length_u=2.0, height=1.0, width_v=4.0)
    bmin, bmax = fitted.bounds
    assert np.allclose(bmin, [-1.0, 0.0, -2.0], atol=1e-6)
    assert np.allclose(bmax, [1.0, 1.0, 2.0], atol=1e-6)


def test_fit_handles_nonuniform_source_bbox():
    # Source mesh already non-cubic (e.g. a tall thin generated mesh) - each
    # axis scales independently to fill the measured footprint exactly.
    box = trimesh.creation.box(extents=(4.0, 1.0, 2.0))
    fitted = _fit_generated_mesh_to_footprint(box, length_u=1.0, height=1.0, width_v=1.0)
    bmin, bmax = fitted.bounds
    assert np.allclose(bmin, [-0.5, 0.0, -0.5], atol=1e-6)
    assert np.allclose(bmax, [0.5, 1.0, 0.5], atol=1e-6)


# --- T13: objects.json round trip + assembly from it ---------------------------


def _square_hull(cx, cz, r):
    return [[cx - r, cz - r], [cx + r, cz - r], [cx + r, cz + r], [cx - r, cz + r]]


@pytest.fixture
def bootstrap_like(tmp_path: Path):
    walls = [
        WallPolygon(vertices=[(0.0, 0.0), (6.0, 0.0), (6.0, 0.2), (0.0, 0.2)], area_m2=1.2),
        WallPolygon(vertices=[(0.0, 4.8), (6.0, 4.8), (6.0, 5.0), (0.0, 5.0)], area_m2=1.2),
    ]
    room_polygon = [(0.2, 0.2), (5.8, 0.2), (5.8, 4.8), (0.2, 4.8)]
    objects = [
        {
            "id": "bed_0",
            "label": "bed",
            "center_xy": (np.float64(2.0), np.float64(2.5)),
            "size_uv": (np.float64(2.0), np.float64(1.5)),
            "angle_rad": np.float64(math.radians(10.0)),
            "height": np.float64(0.6),
            "bbox_min_y": np.float64(0.02),
            "color_rgb": (120, 80, 60),
            "hull_xz": np.array(_square_hull(2.0, 2.5, 0.7)).tolist(),
        },
        {
            "id": "chair_0",
            "label": "chair",
            "center_xy": (4.5, 1.0),
            "size_uv": (0.5, 0.5),
            "angle_rad": 0.0,
            "height": 0.9,
            "bbox_min_y": 0.0,
            "color_rgb": (50, 50, 50),
            "hull_xz": _square_hull(4.5, 1.0, 0.25),
        },
        {
            "id": "curtain_0",
            "label": "curtain",
            "center_xy": (5.5, 2.5),
            "size_uv": (2.0, 0.1),
            "angle_rad": math.radians(90.0),
            "height": 2.4,
            "bbox_min_y": 0.05,
            "color_rgb": (200, 200, 200),
            "hull_xz": _square_hull(5.5, 2.5, 0.1),
        },
    ]
    path = write_objects_json(tmp_path / "objects.json", objects, walls, room_polygon, 0.0, 2.7, yaw_meta={"yaw_correction_deg": 12.5})
    return path, objects, walls, room_polygon


def test_objects_json_round_trips_all_fields(bootstrap_like):
    path, objects, walls, room_polygon = bootstrap_like
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == OBJECTS_JSON_SCHEMA_VERSION
    assert raw["floor_polygon"] == raw["room_polygon"]
    loaded = load_objects_json(path)
    assert loaded.floor_y == 0.0 and loaded.ceiling_y == 2.7
    assert loaded.room_polygon == [tuple(p) for p in room_polygon] and loaded.floor_polygon == loaded.room_polygon
    assert [w.vertices for w in loaded.walls] == [w.vertices for w in walls]
    assert [w.area_m2 for w in loaded.walls] == [w.area_m2 for w in walls]
    assert loaded.meta["yaw"] == {"yaw_correction_deg": 12.5}
    assert [o["id"] for o in loaded.objects] == [o["id"] for o in objects]
    for got, want in zip(loaded.objects, objects):
        assert set(got) == set(want)
        assert got["center_xy"] == pytest.approx(tuple(float(v) for v in want["center_xy"]))
        assert got["size_uv"] == pytest.approx(tuple(float(v) for v in want["size_uv"]))
        assert got["angle_rad"] == pytest.approx(float(want["angle_rad"]))
        assert got["height"] == pytest.approx(float(want["height"]))
        assert got["bbox_min_y"] == pytest.approx(float(want["bbox_min_y"]))
        assert got["color_rgb"] == tuple(want["color_rgb"])
        assert np.allclose(got["hull_xz"], want["hull_xz"])


def _box_glb(path: Path, extents) -> Path:
    trimesh.Scene(trimesh.creation.box(extents=extents)).export(path.as_posix(), file_type="glb")
    return path


def test_assembly_node_count_walls_floor_visual_collision(bootstrap_like, tmp_path: Path):
    path, objects, walls, _room = bootstrap_like
    bootstrap = load_objects_json(path)
    # Generated mesh for bed_0, an asset for chair_0, nothing for curtain_0.
    manifest = tmp_path / "manifest"
    (manifest / "crops").mkdir(parents=True)
    _box_glb(manifest / "crops" / "bed_0.glb", (1.0, 0.4, 2.0))
    (manifest / "results.json").write_text(json.dumps([{"id": "bed_0", "accepted": True, "glb_path": "bed_0.glb"}, {"id": "chair_0", "accepted": False, "glb_path": None}]))
    generated = load_generated(manifest)
    assert set(generated) == {"bed_0"}

    index = AssetIndex(assets_root=tmp_path.as_posix())
    chair_glb = _box_glb(tmp_path / "chair.glb", (0.45, 0.9, 0.5))
    index.classes["chair"] = [AssetCandidate(source="polyhaven", id="chair_x", path=chair_glb.as_posix(), cls="chair", canonical_aabb_m=(0.45, 0.9, 0.5), width_m=0.45, depth_m=0.5, height_m=0.9, is_round=False)]

    scene, placements = assemble(bootstrap, generated, assets=True, asset_index=index)
    nodes = set(scene.graph.nodes_geometry)
    mesh_nodes = {n for n in nodes if not n.startswith("Plan")}
    n_walls = 1  # T15h: ONE `wall_outline` band (T15g), not the per-fragment wall_i blobs
    n_visual = 1 + 1 + 1  # bed_0: one generated part; chair_0: one asset part; curtain_0: one placeholder box
    n_collision = len(objects)
    assert len(mesh_nodes) == n_walls + 1 + n_visual + n_collision, sorted(mesh_nodes)
    assert {"wall_outline", "floor", "bed_0/visual/part_0", "chair_0/visual/part_0", "curtain_0/visual/part_0"} <= mesh_nodes
    assert not any(re.fullmatch(r"wall_\d+", n) for n in mesh_nodes), sorted(mesh_nodes)
    assert {f"{o['id']}/collision/hull" for o in objects} <= mesh_nodes
    # the band is the same geometry scene.glb carries: room.buffer(0.10) - room, floor -> ceiling
    band_tf, band_geom = scene.graph["wall_outline"]
    band_verts = trimesh.transform_points(scene.geometry[band_geom].vertices, band_tf)
    assert band_verts[:, 1].min() == pytest.approx(0.0, abs=1e-6) and band_verts[:, 1].max() == pytest.approx(2.7, abs=1e-6)
    assert band_verts[:, 0].min() == pytest.approx(0.2 - 0.1, abs=1e-6) and band_verts[:, 0].max() == pytest.approx(5.8 + 0.1, abs=1e-6)
    assert band_verts[:, 2].min() == pytest.approx(0.2 - 0.1, abs=1e-6) and band_verts[:, 2].max() == pytest.approx(4.8 + 0.1, abs=1e-6)
    assert {str(n) for n in nodes if str(n).startswith("Plan/wall_")} == {"Plan/wall_0", "Plan/wall_1"}  # raw fragments keep their Plan curves
    # M7: the generated mesh gets a record too (so the USD can reference it), then the T15f asset placements.
    assert [(p["id"], p["status"]) for p in placements] == [("bed_0", "generated"), ("chair_0", "placed"), ("curtain_0", "no_candidate_placeholder_kept")]
    generated_rec = placements[0]
    assert generated_rec["source"] == "generated" and generated_rec["asset_id"] == "bed_0" and generated_rec["pre_rotation_deg"] == 0.0
    assert generated_rec["center_xy"] == pytest.approx([2.0, 2.5]) and generated_rec["scale_xyz"] == pytest.approx([2.0, 1.5, 0.75])

    # The generated bed fills its measured box in world space, at bbox_min_y.
    transform, geom_name = scene.graph["bed_0/visual/part_0"]
    verts = trimesh.transform_points(scene.geometry[geom_name].vertices, transform)
    assert verts[:, 1].min() == pytest.approx(0.02, abs=1e-6) and verts[:, 1].max() == pytest.approx(0.62, abs=1e-6)
    assert verts[:, [0, 2]].mean(axis=0) == pytest.approx([2.0, 2.5], abs=1e-6)

    # Round trip through GLB keeps every node.
    out = tmp_path / "scene_assets.glb"
    scene.export(out.as_posix(), file_type="glb")
    reloaded = trimesh.load(out.as_posix())
    assert mesh_nodes <= set(reloaded.graph.nodes_geometry)


def test_assembly_without_assets_keeps_placeholders(bootstrap_like):
    path, objects, walls, _room = bootstrap_like
    scene, placements = assemble(load_objects_json(path), {}, assets=False)
    assert placements == []
    nodes = set(scene.graph.nodes_geometry)
    assert {"bed_0/visual/part_0", "bed_0/visual/part_1", "chair_0/visual/part_0", "chair_0/visual/part_1"} <= nodes  # A4 placeholders
