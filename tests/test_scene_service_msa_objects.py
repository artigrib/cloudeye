"""Pure-unit test for scene_service.resolve_msa_objects - no DB, no network, matches
this test suite's convention (see tests/conftest.py's module docstring and
tests/test_scene_service_msa_glb.py, which this file mirrors for the sibling
GET /{scene_id}/msa-objects endpoint - app.routers.scenes.get_scene_msa_objects is a
thin wrapper around resolve_msa_objects() and isn't separately covered here, same
reasoning as that file's docstring)."""

import json
import uuid
from pathlib import Path

import pytest

from app.config import settings
from app.services.scene_service import MSA_OBJECTS_CANDIDATES, resolve_msa_objects, scene_dir


@pytest.fixture
def scene_id(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return uuid.uuid4()


def _write(scene_id: uuid.UUID, filename: str, payload: object) -> Path:
    path = scene_dir(scene_id) / "msa" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _objects(*labels_by_id: tuple[str, str]) -> dict:
    return {"objects": [{"id": oid, "label": label, "n_points": 1} for oid, label in labels_by_id]}


def test_prefers_audited_over_plain_bootstrap(scene_id):
    _write(scene_id, "objects.json", _objects(("bed_0", "bed"), ("desk_0", "desk")))
    _write(scene_id, "objects_audited.json", _objects(("bed_0", "bed")))

    resolved = resolve_msa_objects(scene_id)

    assert resolved == ([{"id": "bed_0", "label": "bed"}], "objects_audited.json")


def test_falls_back_to_plain_bootstrap_when_no_audited_file(scene_id):
    _write(scene_id, "objects.json", _objects(("bed_0", "bed"), ("chair_0", "chair")))

    resolved = resolve_msa_objects(scene_id)

    assert resolved == (
        [{"id": "bed_0", "label": "bed"}, {"id": "chair_0", "label": "chair"}],
        "objects.json",
    )


def test_returns_none_when_no_candidate_exists(scene_id):
    assert resolve_msa_objects(scene_id) is None


def test_returns_none_for_malformed_json_rather_than_raising(scene_id):
    path = scene_dir(scene_id) / "msa" / "objects_audited.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")
    _write(scene_id, "objects.json", _objects(("bed_0", "bed")))

    # Falls through the broken audited file to the good plain one, doesn't raise.
    resolved = resolve_msa_objects(scene_id)

    assert resolved == ([{"id": "bed_0", "label": "bed"}], "objects.json")


def test_returns_none_when_objects_key_missing_or_wrong_shape(scene_id):
    _write(scene_id, "objects_audited.json", {"not_objects": []})
    _write(scene_id, "objects.json", _objects(("bed_0", "bed")))

    resolved = resolve_msa_objects(scene_id)

    assert resolved == ([{"id": "bed_0", "label": "bed"}], "objects.json")


def test_skips_malformed_individual_entries(scene_id):
    payload = {
        "objects": [
            {"id": "bed_0", "label": "bed"},
            {"id": "chair_0"},  # missing label
            {"label": "lamp"},  # missing id
            "not-a-dict",
            {"id": "desk_0", "label": "desk"},
        ]
    }
    _write(scene_id, "objects.json", payload)

    resolved = resolve_msa_objects(scene_id)

    assert resolved == (
        [{"id": "bed_0", "label": "bed"}, {"id": "desk_0", "label": "desk"}],
        "objects.json",
    )


def test_candidate_order_prefers_audited(scene_id):
    assert MSA_OBJECTS_CANDIDATES.index("objects_audited.json") < MSA_OBJECTS_CANDIDATES.index("objects.json")
