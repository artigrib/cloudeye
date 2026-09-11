"""T15f Part B: the asset placement rule on synthetic box assets."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.msa.asset_fallback import PART_NAME_KEY, fit_meshes_to_box, place_assets, select_candidate
from scripts.msa.asset_index import AssetCandidate, AssetIndex
from scripts.msa.export_glb import build_scene


def _box_glb(path: Path, extents_xyz, *, offset=(0.0, 0.0, 0.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents_xyz)
    mesh.apply_translation(offset)
    trimesh.Scene(mesh).export(path.as_posix(), file_type="glb")
    return path


def _cand(source, cid, path, extents, cls, *, is_round=False) -> AssetCandidate:
    x, y, z = extents
    return AssetCandidate(source=source, id=cid, path=path.as_posix(), cls=cls, canonical_aabb_m=(x, y, z), width_m=x, depth_m=z, height_m=y, is_round=is_round)


@pytest.fixture
def index(tmp_path: Path) -> AssetIndex:
    idx = AssetIndex(assets_root=tmp_path.as_posix())
    idx.classes["bed"] = [
        _cand("polyhaven", "bed_long", _box_glb(tmp_path / "bed_long.glb", (1.0, 0.5, 2.0)), (1.0, 0.5, 2.0), "bed"),  # aspect 2.0
        _cand("polyhaven", "bed_square", _box_glb(tmp_path / "bed_square.glb", (1.8, 0.5, 2.0)), (1.8, 0.5, 2.0), "bed"),  # aspect 1.11
        _cand("kenney", "bed_kenney", _box_glb(tmp_path / "bed_kenney.glb", (1.0, 0.5, 2.0)), (1.0, 0.5, 2.0), "bed"),
    ]
    idx.classes["lamp"] = [
        _cand("polyhaven", "lamp_tall", _box_glb(tmp_path / "lamp.glb", (0.4, 1.0, 0.4), offset=(3.0, 5.0, -2.0)), (0.4, 1.0, 0.4), "lamp", is_round=True),
    ]
    return idx


def _obj(obj_id, label, center, size_uv, angle_rad, height, bbox_min_y=0.0):
    return {
        "id": obj_id,
        "label": label,
        "center_xy": center,
        "size_uv": size_uv,
        "angle_rad": angle_rad,
        "height": height,
        "bbox_min_y": bbox_min_y,
        "color_rgb": (10, 20, 30),
        "hull_xz": [[center[0] - 0.5, center[1] - 0.5], [center[0] + 0.5, center[1] - 0.5], [center[0] + 0.5, center[1] + 0.5], [center[0] - 0.5, center[1] + 0.5]],
    }


def test_select_closest_footprint_aspect_and_polyhaven_first(index: AssetIndex):
    cls, cand, aspect = select_candidate("bed", (2.1, 1.0), 0.6, index)  # measured aspect 2.1
    assert cls == "bed" and cand.id == "bed_long" and aspect == pytest.approx(2.1)
    cls, cand, _ = select_candidate("bed", (1.0, 1.9), 0.6, index)  # sorted ratio ~1.9... wait, 1.9 -> bed_long (2.0) beats square (1.11)
    assert cand.id == "bed_long"
    cls, cand, _ = select_candidate("bed", (1.9, 1.7), 0.6, index)  # aspect 1.12 -> bed_square
    assert cand.id == "bed_square"
    assert all(c.source == "polyhaven" for c in index.candidates("bed"))  # kenney never mixed in


def test_no_candidate_keeps_placeholder(index: AssetIndex):
    cls, cand, _ = select_candidate("curtain", (2.0, 0.1), 2.5, index)
    assert cls is None and cand is None
    overrides, placements = place_assets([_obj("curtain_0", "curtain", (0, 0), (2.0, 0.1), 0.0, 2.5)], index)
    assert overrides == {}
    assert placements[0]["status"] == "no_candidate_placeholder_kept"


def test_per_axis_scale_fills_measured_box():
    src = [trimesh.creation.box(extents=(1.0, 0.5, 2.0))]
    fitted, info = fit_meshes_to_box(src, length_u=3.0, height=0.75, width_v=1.0)
    bmin, bmax = fitted[0].bounds
    assert np.allclose(bmin, [-1.5, 0.0, -0.5], atol=1e-6)
    assert np.allclose(bmax, [1.5, 0.75, 0.5], atol=1e-6)
    # source long axis was Z, footprint long axis is U (X): a 90 deg pre-rotation
    # then X scale 3/2, Y 0.75/0.5, Z 1/1.
    assert info["pre_rotation_deg"] == 90.0
    assert info["scale_xyz"] == pytest.approx([1.5, 1.5, 1.0])


def test_uniform_scale_for_round_class_never_distorts():
    src = [trimesh.creation.box(extents=(0.4, 1.0, 0.4))]
    fitted, info = fit_meshes_to_box(src, length_u=0.3, height=1.5, width_v=0.6, uniform=True)
    assert info["uniform"] is True
    s = min(0.3 / 0.4, 1.5 / 1.0, 0.6 / 0.4)
    assert info["scale_xyz"] == pytest.approx([s, s, s])
    ext = fitted[0].extents
    assert ext == pytest.approx([0.4 * s, 1.0 * s, 0.4 * s])
    bmin, bmax = fitted[0].bounds
    assert bmin[1] == pytest.approx(0.0)  # on the floor
    assert (bmin[0] + bmax[0]) / 2 == pytest.approx(0.0) and (bmin[2] + bmax[2]) / 2 == pytest.approx(0.0)  # XZ-centred


def test_off_origin_multi_mesh_asset_is_treated_as_one_rigid_body():
    a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    a.apply_translation((10.0, 4.0, 10.0))
    b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    b.apply_translation((12.0, 4.0, 10.0))  # joint bbox 3 x 1 x 1, X is long
    fitted, info = fit_meshes_to_box([a, b], length_u=6.0, height=2.0, width_v=1.0)
    lo = np.min([m.bounds[0] for m in fitted], axis=0)
    hi = np.max([m.bounds[1] for m in fitted], axis=0)
    assert np.allclose(lo, [-3.0, 0.0, -0.5], atol=1e-6) and np.allclose(hi, [3.0, 2.0, 0.5], atol=1e-6)
    assert info["pre_rotation_deg"] == 0.0
    # The gap between the two boxes is preserved (rigid, not per-mesh fitted).
    assert fitted[0].bounds[1][0] < fitted[1].bounds[0][0]


def test_long_axis_ends_up_along_angle_rad_in_world(index: AssetIndex, tmp_path: Path):
    """A 2x1 asset placed into a footprint whose long side (size_uv[0]) is
    rotated by angle_rad must have its 2-side along (cos a, sin a) in world XZ."""
    angle = math.radians(30.0)
    obj = _obj("bed_0", "bed", (4.0, -2.0), (2.4, 1.2), angle, 0.6, bbox_min_y=0.05)
    overrides, placements = place_assets([obj], index)
    assert placements[0]["asset_id"] == "bed_long"  # a 1 x 2 (X x Z) box: long axis is Z -> pre-rotated
    assert placements[0]["pre_rotation_deg"] == 90.0
    assert placements[0]["yaw_deg"] == pytest.approx(30.0 + 90.0)

    scene = build_scene([], None, 0.0, 2.5, [obj], visual_overrides=overrides)
    node = "bed_0/visual/part_0"
    assert node in scene.graph.nodes_geometry
    transform, geom_name = scene.graph[node]
    verts = trimesh.transform_points(scene.geometry[geom_name].vertices, transform)
    centered = verts[:, [0, 2]] - verts[:, [0, 2]].mean(axis=0)
    # PCA of the box's corner cloud: principal direction == footprint long axis.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    principal = vt[0] / np.linalg.norm(vt[0])
    long_dir = np.array([math.cos(angle), math.sin(angle)])
    assert abs(float(principal @ long_dir)) == pytest.approx(1.0, abs=1e-6)
    # Extent along the long direction is size_uv[0], across it size_uv[1].
    along = centered @ long_dir
    across = centered @ np.array([-long_dir[1], long_dir[0]])
    assert along.max() - along.min() == pytest.approx(2.4, abs=1e-6)
    assert across.max() - across.min() == pytest.approx(1.2, abs=1e-6)
    # Floor: bottom at bbox_min_y, top at bbox_min_y + height, centred on center_xy.
    assert verts[:, 1].min() == pytest.approx(0.05, abs=1e-6)
    assert verts[:, 1].max() == pytest.approx(0.65, abs=1e-6)
    assert verts[:, [0, 2]].mean(axis=0) == pytest.approx([4.0, -2.0], abs=1e-6)


def test_round_lamp_placed_on_floor_and_centred_even_when_source_is_off_origin(index: AssetIndex):
    obj = _obj("lamp_1", "lamp", (1.0, 1.0), (0.3, 0.6), 0.0, 1.5, bbox_min_y=0.7)
    overrides, placements = place_assets([obj], index)
    p = placements[0]
    assert p["asset_id"] == "lamp_tall" and p["uniform"] is True
    s = min(0.3 / 0.4, 1.5 / 1.0, 0.6 / 0.4)
    assert p["scale_xyz"] == pytest.approx([s, s, s])
    scene = build_scene([], None, 0.0, 2.5, [obj], visual_overrides=overrides)
    transform, geom_name = scene.graph["lamp_1/visual/part_0"]
    verts = trimesh.transform_points(scene.geometry[geom_name].vertices, transform)
    assert verts[:, 1].min() == pytest.approx(0.7, abs=1e-6)
    assert verts[:, [0, 2]].mean(axis=0) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_skip_ids_and_collision_untouched(index: AssetIndex):
    objs = [_obj("bed_0", "bed", (0, 0), (2.0, 1.0), 0.0, 0.5), _obj("bed_1", "bed", (5, 0), (2.0, 1.0), 0.0, 0.5)]
    overrides, placements = place_assets(objs, index, skip_ids={"bed_1"})
    assert set(overrides) == {"bed_0"} and [p["id"] for p in placements] == ["bed_0"]
    assert overrides["bed_0"][0].metadata[PART_NAME_KEY] == "part_0"
    scene = build_scene([], None, 0.0, 2.5, objs, visual_overrides=overrides)
    nodes = set(scene.graph.nodes_geometry)
    assert {"bed_0/visual/part_0", "bed_0/collision/hull", "bed_1/visual/part_0", "bed_1/visual/part_1", "bed_1/collision/hull"} <= nodes
    # bed_1 kept its 2-part placeholder (mattress + headboard); bed_0 is the asset.
    assert "bed_0/visual/part_1" not in nodes
    hull = scene.geometry[scene.graph["bed_0/collision/hull"][1]]
    assert hull.extents[0] == pytest.approx(1.0, abs=1e-6)  # the measured hull_xz square, not the asset box


# --- T16b: small tier gating ----------------------------------------------------


@pytest.fixture
def index_with_small(index: AssetIndex, tmp_path: Path) -> AssetIndex:
    mug = _cand("gso", "Mug_A", _box_glb(tmp_path / "Mug_A.glb", (0.13, 0.10, 0.09)), (0.13, 0.10, 0.09), "cup", is_round=True)
    mug.tier = "small"
    index.classes["cup"] = [mug]
    return index


def test_small_asset_placed_only_for_small_labels_and_small_boxes(index_with_small: AssetIndex):
    objects = [
        _obj("glass_0", "glass", (0.0, 0.0), (0.08, 0.06), 0.0, 0.11, bbox_min_y=0.75),  # a real glass on a desk
        _obj("glass_19", "glass", (3.0, 0.0), (1.31, 1.19), 0.0, 2.4),  # SAM3 "glass" = a window pane
        _obj("chair_0", "chair", (5.0, 0.0), (0.5, 0.5), 0.0, 0.9),  # furniture label: no cup for it, ever
    ]
    overrides, placements = place_assets(objects, index_with_small)
    by_id = {p["id"]: p for p in placements}
    assert set(overrides) == {"glass_0"}
    assert by_id["glass_0"]["status"] == "placed" and by_id["glass_0"]["tier"] == "small" and by_id["glass_0"]["asset_id"] == "Mug_A"
    assert by_id["glass_0"]["uniform"] is True  # round class: never squashed
    fitted = np.array(by_id["glass_0"]["fitted_extents"])
    assert np.all(fitted <= np.array([0.08, 0.11, 0.06]) + 1e-6)  # fits inside the measured box on every axis
    assert by_id["glass_19"]["status"] == "small_tier_gated_placeholder_kept" and by_id["glass_19"]["asset_id"] is None
    assert by_id["chair_0"]["status"] == "no_candidate_placeholder_kept" and by_id["chair_0"]["tier"] is None
    # one record per detected object and nothing else: no asset is ever added where there was no detection
    assert [p["id"] for p in placements] == ["glass_0", "glass_19", "chair_0"]
    scene = build_scene([], None, 0.0, 2.5, objects, visual_overrides=overrides)
    hull = scene.geometry[scene.graph["glass_0/collision/hull"][1]]
    assert hull.extents[0] == pytest.approx(1.0, abs=1e-6)  # collider stays the measured hull_xz, not the mug


def test_small_selection_gates(index_with_small: AssetIndex):
    cls, cand, _ = select_candidate("glass", (0.08, 0.06), 0.11, index_with_small)
    assert cls == "cup" and cand is not None and cand.tier == "small"
    cls, cand, _ = select_candidate("glass", (0.08, 0.06), 0.9, index_with_small)  # too tall for a small object
    assert cls == "cup" and cand is None
    _, cand, _ = select_candidate("table", (0.08, 0.06), 0.05, index_with_small)  # furniture label with a tiny box: still no cup
    assert cand is None
