"""M7 item 3: per-scene class caps (scripts/msa/class_caps.py)."""

from __future__ import annotations

from scripts.msa.class_caps import CLASS_CAPS, apply_class_caps, normalize_label, object_support

SQUARE_1M = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]  # area 1.0
SQUARE_2M = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]  # area 4.0


def _make(id, label, *, n_points=None, hull_xz=SQUARE_1M, height=1.0):
    obj = {
        "id": id,
        "label": label,
        "center_xy": [0.0, 0.0],
        "size_uv": [1.0, 1.0],
        "angle_rad": 0.0,
        "height": height,
        "bbox_min_y": 0.0,
        "color_rgb": [0, 0, 0],
        "hull_xz": hull_xz,
    }
    if n_points is not None:
        obj["n_points"] = n_points
    return obj


def test_five_lamps_capped_to_four_lowest_support_dropped():
    lamps = [
        _make("lamp_0", "lamp", n_points=10),
        _make("lamp_1", "lamp", n_points=20),
        _make("lamp_2", "lamp", n_points=30),
        _make("lamp_3", "lamp", n_points=40),
        _make("lamp_4", "lamp", n_points=5),  # lowest support
    ]
    kept, dropped = apply_class_caps(lamps)

    assert [o["id"] for o in kept] == ["lamp_0", "lamp_1", "lamp_2", "lamp_3"]
    assert len(dropped) == 1
    entry = dropped[0]
    assert entry["id"] == "lamp_4"
    assert entry["label"] == "lamp"
    assert entry["support"] == 5.0
    assert entry["rank"] == 5
    assert entry["cap"] == 4
    assert entry["reason"] == "class_cap"

    # kept objects preserve original order (not sorted by support)
    assert kept[0]["id"] == "lamp_0"


def test_tv_television_synonym_grouped_and_capped_together():
    objs = [
        _make("tv_0", "tv", n_points=100),
        _make("tv_1", "television", n_points=50),
        _make("tv_2", "tv", n_points=10),  # lowest support of the three -> dropped
    ]
    kept, dropped = apply_class_caps(objs)

    assert {o["id"] for o in kept} == {"tv_0", "tv_1"}
    assert len(dropped) == 1
    assert dropped[0]["id"] == "tv_2"
    assert dropped[0]["cap"] == 2
    assert dropped[0]["rank"] == 3


def test_uncapped_class_untouched():
    sofas = [_make(f"sofa_{i}", "sofa", n_points=1) for i in range(10)]
    kept, dropped = apply_class_caps(sofas)

    assert len(kept) == 10
    assert dropped == []


def test_tie_break_by_hull_area_then_id():
    # Equal support (same n_points); larger hull area wins the tie and is kept.
    objs = [
        _make("desk_a", "desk", n_points=100, hull_xz=SQUARE_1M),  # smaller area
        _make("desk_b", "desk", n_points=100, hull_xz=SQUARE_2M),  # larger area
        _make("desk_c", "desk", n_points=100, hull_xz=SQUARE_1M),
    ]
    kept, dropped = apply_class_caps(objs)  # cap for desk is 2

    kept_ids = {o["id"] for o in kept}
    assert "desk_b" in kept_ids  # largest area always survives the tie
    assert len(kept) == 2
    assert len(dropped) == 1
    # among desk_a/desk_c (equal support and area), id-order tie-break drops the higher id
    assert dropped[0]["id"] == "desk_c"


def test_empty_input():
    kept, dropped = apply_class_caps([])
    assert kept == []
    assert dropped == []


def test_normalize_label_synonyms():
    assert normalize_label("television") == "tv"
    assert normalize_label("TV") == "tv"
    assert normalize_label("Bedside") == "nightstand"
    assert normalize_label("closet") == "wardrobe"
    assert normalize_label("chair") == "chair"
    assert normalize_label(None) == ""


def test_object_support_falls_back_to_hull_area_times_height_when_no_point_field():
    obj = _make("lamp_x", "lamp", hull_xz=SQUARE_2M, height=1.5)  # no n_points field
    assert "n_points" not in obj
    assert object_support(obj) == 4.0 * 1.5


def test_object_support_prefers_dedicated_point_field_over_geometry():
    obj = _make("lamp_y", "lamp", n_points=42, hull_xz=SQUARE_2M, height=1.5)
    assert object_support(obj) == 42.0


def test_class_caps_default_dict_shape():
    assert CLASS_CAPS["bed"] == 2
    assert CLASS_CAPS["tv"] == 2
    assert CLASS_CAPS["television"] == 2
    assert CLASS_CAPS["desk"] == 2
    assert CLASS_CAPS["chair"] == 4
    assert CLASS_CAPS["lamp"] == 4
    assert CLASS_CAPS["nightstand"] == 2
    assert CLASS_CAPS["wardrobe"] == 2
    assert "sofa" not in CLASS_CAPS


def test_apply_class_caps_does_not_mutate_input():
    lamps = [_make(f"lamp_{i}", "lamp", n_points=i) for i in range(5)]
    original_ids = [o["id"] for o in lamps]
    apply_class_caps(lamps)
    assert [o["id"] for o in lamps] == original_ids
    assert len(lamps) == 5
