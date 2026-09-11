"""Pure-unit tests for scene_service.msa_meta_path / read_msa_meta - no DB, no network,
same convention as tests/test_scene_service_msa_glb.py (whose router endpoint this one's
sibling, app.routers.scenes.get_scene_msa_meta, mirrors: a thin 404-or-JSON wrapper
around these two helpers)."""

import json
import uuid
from pathlib import Path

import pytest

from app.services.scene_service import MSA_META_FIELDS, msa_glb_path, msa_meta_path, read_msa_meta


def test_msa_meta_path_sits_next_to_the_served_glb():
    scene_id = uuid.uuid4()
    assert msa_meta_path(scene_id) == msa_glb_path(scene_id).with_name("scene_meta.json")
    assert msa_meta_path(scene_id).parent == msa_glb_path(scene_id).parent


def test_read_msa_meta_returns_none_when_absent(tmp_path: Path):
    assert read_msa_meta(tmp_path / "msa" / "scene_meta.json") is None


def test_read_msa_meta_keeps_only_the_exposed_fields(tmp_path: Path):
    full = {
        "floor_y": 0.0,
        "ceiling_y": 2.95,
        "yaw_method": "room_polygon_min_area_rect",
        "yaw_correction_rad": 0.99,
        "yaw_correction_deg": 56.76,
        "yaw_applied": True,
        "yaw_rotation_center_xy": [1.57, -3.24],
        "room_polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        # Diagnostics the API deliberately does not expose:
        "yaw_wall_hull_correction_deg": 50.5,
        "yaw_histogram_gate_passed": False,
        "objects": [{"id": "bed_0"}],
    }
    path = tmp_path / "scene_meta.json"
    path.write_text(json.dumps(full))
    meta = read_msa_meta(path)
    assert meta == {k: full[k] for k in MSA_META_FIELDS}
    assert "objects" not in meta and "yaw_wall_hull_correction_deg" not in meta


def test_read_msa_meta_omits_missing_keys_for_a_pre_yaw_export(tmp_path: Path):
    # A scene_meta.json from before the yaw work has no yaw_* keys at all - they must
    # be absent (the frontend treats missing yaw_applied as "not rotated"), not nulls.
    path = tmp_path / "scene_meta.json"
    path.write_text(json.dumps({"floor_y": 0.1, "ceiling_y": 2.4}))
    assert read_msa_meta(path) == {"floor_y": 0.1, "ceiling_y": 2.4}


def test_read_msa_meta_rejects_a_non_object_file(tmp_path: Path):
    path = tmp_path / "scene_meta.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        read_msa_meta(path)
