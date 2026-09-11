"""T15f Part A: asset index on a tiny synthetic inventory (no dependency on
var/assets/furniture)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.msa.asset_index import (
    INDEX_FILENAME,
    INDEX_SCHEMA_VERSION,
    KENNEY_UNIT_TO_M,
    ROUND_CLASSES,
    TIER_SMALL,
    AssetIndex,
    build_index,
    canonical_from_api_dimensions,
    footprint_aspect,
    is_small_label,
    resolve_label,
)


def _write_box_glb(path: Path, extents_xyz) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents_xyz)
    trimesh.Scene(mesh).export(path.as_posix(), file_type="glb")


@pytest.fixture
def synthetic_assets_root(tmp_path: Path) -> Path:
    root = tmp_path / "furniture"
    # Poly Haven: API dims in mm in a DIFFERENT order than the GLB axes
    # (x=1.5 wide, y=0.5 tall, z=2.0 deep bed -> api lists [1500, 2000, 500]).
    _write_box_glb(root / "polyhaven" / "bed_x" / "bed_x.glb", (1.5, 0.5, 2.0))
    _write_box_glb(root / "polyhaven" / "lamp_x" / "lamp_x.glb", (0.2, 0.8, 0.2))
    # Kenney: native units, 1 u = KENNEY_UNIT_TO_M.
    kenney = root / "kenney_furniture_kit" / "Models" / "GLTF format"
    _write_box_glb(kenney / "lampRoundFloor.glb", (0.15, 0.86, 0.15))
    _write_box_glb(kenney / "chair.glb", (0.2, 0.47, 0.2))
    _write_box_glb(kenney / "wall.glb", (1.0, 1.0, 0.1))  # deliberately unmapped
    inventory = {
        "polyhaven": {
            "bed_x": {
                "class_hint": "bed",
                "api_dimensions": [1500.0, 2000.0, 500.0],
                "glb_path": (root / "polyhaven" / "bed_x" / "bed_x.glb").as_posix(),
                "aabb_extents_m": [1.5, 0.5, 2.0],
            },
            "lamp_x": {
                "class_hint": "lamp (floor/table lamp)",
                "api_dimensions": [200.0, 200.0, 800.0],
                "glb_path": (root / "polyhaven" / "lamp_x" / "lamp_x.glb").as_posix(),
                "aabb_extents_m": [0.2, 0.8, 0.2],
            },
        },
        "kenney_furniture_kit": {
            "obtained": True,
            "model_files": [
                {"path": "Models/GLTF format/lampRoundFloor.glb", "format": "GLB"},
                {"path": "Models/GLTF format/chair.glb", "format": "GLB"},
                {"path": "Models/GLTF format/wall.glb", "format": "GLB"},
                {"path": "Models/DAE format/chair.dae", "format": "DAE"},
            ],
        },
        "mastjie_household_goods": {"obtained": False, "count": 0},
    }
    (root / "INVENTORY.json").write_text(json.dumps(inventory))
    return root


def test_canonical_from_api_dimensions_matches_by_sorted_rank():
    canonical, perm, mismatch = canonical_from_api_dimensions([1500.0, 2000.0, 500.0], [1.5, 0.5, 2.0])
    assert canonical == pytest.approx((1.5, 0.5, 2.0))
    assert perm == (0, 2, 1)
    assert mismatch == pytest.approx(0.0)


def test_canonical_flags_mismatch_but_still_maps():
    # desk_lamp_arm_01 case: the api triple and the GLB disagree on one axis.
    canonical, _perm, mismatch = canonical_from_api_dimensions([617.0, 408.0, 879.0], [0.202, 0.893, 0.614])
    assert canonical == pytest.approx((0.408, 0.879, 0.617))
    assert mismatch > 0.5


def test_build_index_classes_sizes_and_sources(synthetic_assets_root: Path):
    index = build_index(synthetic_assets_root)
    assert set(index.classes) == {"bed", "lamp", "floor_lamp", "chair"}
    bed = index.candidates("bed")[0]
    assert bed.source == "polyhaven"
    assert bed.canonical_aabb_m == pytest.approx((1.5, 0.5, 2.0))
    assert (bed.width_m, bed.height_m, bed.depth_m) == pytest.approx((1.5, 0.5, 2.0))
    assert bed.axis_permutation == (0, 2, 1)
    assert bed.tier == "furniture"
    assert not bed.is_round

    lamp = index.candidates("lamp")[0]
    assert lamp.is_round and "lamp" in ROUND_CLASSES

    floor_lamp = index.candidates("floor_lamp")[0]
    assert floor_lamp.source == "kenney"
    assert floor_lamp.unit_factor == KENNEY_UNIT_TO_M
    assert floor_lamp.height_m == pytest.approx(0.86 * KENNEY_UNIT_TO_M, rel=1e-3)
    assert floor_lamp.is_round

    assert index.unmapped["kenney"] == ["wall"]
    assert index.sources["mastjie"]["obtained"] is False
    assert index.stats()["floor_lamp"] == {"kenney": 1}


def test_label_resolution_and_source_preference(synthetic_assets_root: Path):
    index = build_index(synthetic_assets_root)
    assert resolve_label("desk") == ("desk", "table")
    assert resolve_label("television") == ("tv",)
    assert resolve_label("wardrobe") == ("cabinet",)
    assert resolve_label("nightstand_3") == ("nightstand",)
    assert resolve_label("lamp", height_m=1.4) == ("floor_lamp", "lamp")
    assert resolve_label("lamp", height_m=0.4) == ("lamp",)

    cls, cands = index.candidates_for_label("lamp", height_m=0.5)
    assert cls == "lamp" and [c.source for c in cands] == ["polyhaven"]
    cls, cands = index.candidates_for_label("lamp", height_m=1.5)
    assert cls == "floor_lamp" and [c.id for c in cands] == ["lampRoundFloor"]
    cls, cands = index.candidates_for_label("curtain")
    assert cls is None and cands == []
    cls, cands = index.candidates_for_label("Chair_7")
    assert cls == "chair" and cands[0].source == "kenney"


def test_index_cache_round_trip_and_regeneration(synthetic_assets_root: Path):
    index = build_index(synthetic_assets_root)
    cache = synthetic_assets_root / INDEX_FILENAME
    assert cache.exists()
    reloaded = AssetIndex.from_json(json.loads(cache.read_text()))
    assert reloaded.stats() == index.stats()
    assert reloaded.candidates("bed")[0].canonical_aabb_m == pytest.approx(index.candidates("bed")[0].canonical_aabb_m)
    # The cache is used while fresh ...
    cached = build_index(synthetic_assets_root)
    assert cached.stats() == index.stats()
    # ... and regenerated once INVENTORY.json is newer than it.
    inventory_path = synthetic_assets_root / "INVENTORY.json"
    data = json.loads(inventory_path.read_text())
    del data["polyhaven"]["lamp_x"]
    inventory_path.write_text(json.dumps(data))
    future = time.time() + 10
    os.utime(inventory_path, (future, future))
    rebuilt = build_index(synthetic_assets_root)
    assert "lamp" not in rebuilt.classes


def test_footprint_aspect_is_orientation_free():
    assert footprint_aspect(2.0, 1.0) == pytest.approx(footprint_aspect(1.0, 2.0)) == pytest.approx(2.0)
    assert footprint_aspect(0.0, 1.0) > 1e5


@pytest.mark.skipif(not Path("var/assets/furniture/INVENTORY.json").exists(), reason="real asset packs not present")
def test_real_index_covers_hero_classes():
    index = build_index("var/assets/furniture")
    for label in ("bed", "chair", "desk", "nightstand", "lamp", "television"):
        cls, cands = index.candidates_for_label(label)
        assert cands, label
        assert all(c.source == "polyhaven" for c in cands), label
    for c in index.candidates("bed"):
        assert 1.5 < max(c.width_m, c.depth_m) < 2.5  # real beds are ~2 m long
    assert np.all([c.height_m > 0 for cands in index.classes.values() for c in cands])


# --- T16b: small tier (Google Scanned Objects) ---------------------------------


@pytest.fixture
def synthetic_small_root(tmp_path: Path) -> Path:
    root = tmp_path / "small_objects"
    _write_box_glb(root / "gso" / "Mug_A" / "Mug_A.glb", (0.13, 0.10, 0.09))  # handle along X
    _write_box_glb(root / "gso" / "Bottle_B" / "Bottle_B.glb", (0.06, 0.22, 0.06))
    _write_box_glb(root / "gso" / "Shark_C" / "Shark_C.glb", (0.3, 0.1, 0.1))  # class not in SMALL_CLASSES
    inventory = {
        "license": "CC BY 4.0",
        "models": {
            "Mug_A": {"class": "cup", "glb_path": (root / "gso" / "Mug_A" / "Mug_A.glb").as_posix(), "aabb_extents_m": [0.13, 0.10, 0.09]},
            "Bottle_B": {"class": "bottle", "glb_path": (root / "gso" / "Bottle_B" / "Bottle_B.glb").as_posix()},  # no cached AABB -> measured
            "Shark_C": {"class": "animal", "glb_path": (root / "gso" / "Shark_C" / "Shark_C.glb").as_posix()},
            "Missing_D": {"class": "cup", "glb_path": (root / "gso" / "Missing_D" / "Missing_D.glb").as_posix()},
        },
    }
    (root / "INVENTORY.json").write_text(json.dumps(inventory))
    return root


def test_small_tier_indexed_with_glb_aabb_and_unmapped(synthetic_assets_root: Path, synthetic_small_root: Path):
    index = build_index(synthetic_assets_root, small_root=synthetic_small_root, use_cache=False)
    cups = index.candidates("cup")
    assert [c.id for c in cups] == ["Mug_A"]
    assert cups[0].tier == TIER_SMALL and cups[0].source == "gso" and cups[0].is_round
    assert cups[0].canonical_aabb_m == (0.13, 0.10, 0.09) and cups[0].height_m == 0.10
    bottle = index.candidates("bottle")[0]
    assert np.allclose(bottle.canonical_aabb_m, (0.06, 0.22, 0.06), atol=1e-6)  # measured from the GLB
    assert sorted(index.unmapped["gso"]) == ["Missing_D", "Shark_C"]
    assert index.sources["gso"]["n_indexed"] == 2 and index.sources["gso"]["n_models"] == 4
    assert all(c.tier == "furniture" for c in index.candidates("bed") + index.candidates("chair"))  # furniture tier untouched


def test_small_labels_resolve_only_to_small_classes(synthetic_assets_root: Path, synthetic_small_root: Path):
    index = build_index(synthetic_assets_root, small_root=synthetic_small_root, use_cache=False)
    assert resolve_label("glass_12") == ("cup",)
    assert resolve_label("water bottle") == ("bottle",)
    assert resolve_label("laptop")[0] == "laptop"  # furniture (Kenney) class first
    cls, cands = index.candidates_for_label("glass_3")
    assert cls == "cup" and cands[0].id == "Mug_A"
    assert index.candidates_for_label("pillow_0") == (None, [])  # no pillow in GSO, none in this synthetic furniture root
    assert index.candidates_for_label("chair")[1][0].tier == "furniture"
    assert is_small_label("glass_7") and is_small_label("Water Bottle") and is_small_label("cell phone")
    assert not is_small_label("chair") and not is_small_label("pillow") and not is_small_label("curtain_0")


def test_small_root_optional_and_cache_keyed_on_it(synthetic_assets_root: Path, synthetic_small_root: Path, tmp_path: Path):
    no_small = build_index(synthetic_assets_root, use_cache=True)
    assert "cup" not in no_small.classes and no_small.small_root is None
    with_small = build_index(synthetic_assets_root, small_root=synthetic_small_root, use_cache=True)
    assert "cup" in with_small.classes and with_small.small_root == synthetic_small_root.as_posix()
    cached = json.loads((synthetic_assets_root / INDEX_FILENAME).read_text())
    assert cached["small_root"] == synthetic_small_root.as_posix() and cached["schema_version"] == INDEX_SCHEMA_VERSION
    reloaded = build_index(synthetic_assets_root, small_root=synthetic_small_root, use_cache=True)
    assert reloaded.candidates("cup")[0].tier == TIER_SMALL
    # a small root without INVENTORY.json is silently skipped, not an error
    assert "cup" not in build_index(synthetic_assets_root, small_root=tmp_path / "nowhere", use_cache=False).classes
