"""read_vocab_entries(): the vocab-cache read side. Reduces vocab.json's "objects"
array (which also carries the derived SAM3 threshold fields - score_threshold,
dbscan_eps, dbscan_min_samples, min_cluster, all a pure function of size_class per
gpu/stage_vocab.py's SIZE_CLASS_PARAMS) down to just {"name", "size_class"} per entry -
the shape gpu/stage_vocab.py's own `vocab_override` job param accepts, so this is
exactly what pipeline_orchestrator.resolve_vocab_override needs to feed back in on a
`rerun` to reuse a scene's previously-resolved vocabulary instead of making a fresh
OpenRouter/Vertex call.

Pure-unit, no DB/GPU/network - matches this suite's convention (see conftest.py):
writes a synthetic vocab.json to tmp_path and reads it back, same style as
test_scene_ingest.py's existing read_vocab_source/read_vocab_model coverage.
"""

import json

from app.services.scene_ingest import read_vocab_entries


def _write_vocab_json(tmp_path, payload):
    (tmp_path / "vocab.json").write_text(json.dumps(payload))
    return tmp_path


def test_reduces_objects_to_name_and_size_class(tmp_path):
    scene_dir = _write_vocab_json(
        tmp_path,
        {
            "source": "openrouter",
            "model": "z-ai/glm-5.3-flash",
            "objects": [
                {
                    "name": "chair",
                    "size_class": "medium",
                    "score_threshold": 0.5,
                    "dbscan_eps": 0.1,
                    "dbscan_min_samples": 15,
                    "min_cluster": 30,
                },
                {"name": "door", "size_class": "huge"},
            ],
        },
    )
    assert read_vocab_entries(scene_dir) == [
        {"name": "chair", "size_class": "medium"},
        {"name": "door", "size_class": "huge"},
    ]


def test_missing_file_returns_none(tmp_path):
    assert read_vocab_entries(tmp_path) is None


def test_malformed_json_returns_none(tmp_path):
    (tmp_path / "vocab.json").write_text("{not json")
    assert read_vocab_entries(tmp_path) is None


def test_missing_objects_field_returns_none(tmp_path):
    scene_dir = _write_vocab_json(tmp_path, {"source": "openrouter"})
    assert read_vocab_entries(scene_dir) is None


def test_empty_objects_list_returns_none(tmp_path):
    scene_dir = _write_vocab_json(tmp_path, {"source": "override", "objects": []})
    assert read_vocab_entries(scene_dir) is None


def test_skips_malformed_entries(tmp_path):
    scene_dir = _write_vocab_json(
        tmp_path,
        {
            "source": "override",
            "objects": [
                {"name": "chair", "size_class": "medium"},
                {"size_class": "large"},  # no name - dropped
                "not-a-dict",  # dropped
                {"name": "", "size_class": "small"},  # empty name - dropped
                {"name": "sofa"},  # no size_class - kept, key omitted
            ],
        },
    )
    assert read_vocab_entries(scene_dir) == [
        {"name": "chair", "size_class": "medium"},
        {"name": "sofa"},
    ]


def test_roundtrips_through_vocab_override_shape(tmp_path):
    """The output must be exactly the entry shape gpu/stage_vocab.py's
    vocab_override param accepts (build_vocab_entries() there reads item["name"] /
    item.get("size_class")) - this is the contract a rerun's cached vocabulary relies
    on."""
    from gpu.stage_vocab import build_vocab_entries

    scene_dir = _write_vocab_json(
        tmp_path,
        {"source": "openrouter", "objects": [{"name": "Chair", "size_class": "medium"}]},
    )
    entries = read_vocab_entries(scene_dir)
    rebuilt = build_vocab_entries(entries)
    assert rebuilt == [
        {"name": "chair", "size_class": "medium"}
    ]  # build_vocab_entries lowercases the name
