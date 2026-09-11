"""Unit tests for M7-1 (scripts/msa/dedupe.py): duplicate-detection merge of
same-class object footprints (IoU or centroid-proximity rule, union-find
transitive clustering)."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from scripts.msa.dedupe import _footprint_polygon, merge_duplicates


def _rect_object(obj_id, label, cx, cz, length_u, width_v, *, angle_rad=0.0, height=1.0, bbox_min_y=0.0):
    """Build an object dict whose hull_xz is exactly its oriented-rectangle
    footprint corners, so support (hull area x height) is easy to reason
    about and the footprint used for IoU/centroid checks matches the hull
    used for the merge's convex-hull-union check."""
    obj = {
        "id": obj_id,
        "label": label,
        "center_xy": (cx, cz),
        "size_uv": (length_u, width_v),
        "angle_rad": angle_rad,
        "height": height,
        "bbox_min_y": bbox_min_y,
        "color_rgb": (128, 128, 128),
    }
    obj["hull_xz"] = list(_footprint_polygon(obj).exterior.coords)[:-1]
    return obj


def test_overlapping_same_class_objects_merge_by_iou():
    # Two unit squares offset by dx=0.25 along x: intersection=0.75, union=1.25
    # -> IoU = 0.6, comfortably above the 0.3 threshold.
    a = _rect_object("tv_1", "television", 0.0, 0.0, 1.0, 1.0)
    b = _rect_object("tv_2", "television", 0.25, 0.0, 1.0, 1.0)

    merged, log = merge_duplicates([a, b])

    assert len(merged) == 1
    assert len(log) == 1
    assert log[0]["reason"] == "iou"
    assert log[0]["iou"] == pytest.approx(0.6, rel=1e-3)
    assert sorted(log[0]["merged_ids"] + [log[0]["kept_id"]]) == sorted(["tv_1", "tv_2"])
    assert merged[0]["merged_from"] == ["tv_1", "tv_2"]


def test_nearby_same_class_objects_merge_by_centroid_and_sum_support():
    # Oriented with the long side (0.7) perpendicular to the separation axis
    # so IoU stays below threshold (~0.14) while centroid distance (0.3) is
    # still within 0.5 * 0.7 = 0.35 of the larger object's long side.
    low = _rect_object("nightstand_low", "nightstand", 0.0, 0.0, 0.4, 0.7, height=0.5)
    high = _rect_object("nightstand_high", "nightstand", 0.3, 0.0, 0.4, 0.7, height=0.8)

    merged, log = merge_duplicates([low, high])

    assert len(merged) == 1
    assert len(log) == 1
    assert log[0]["reason"] == "centroid"
    assert log[0]["centroid_dist_m"] == pytest.approx(0.3, rel=1e-3)
    # higher support (hull_area * height) wins the id
    support_low = 0.4 * 0.7 * 0.5
    support_high = 0.4 * 0.7 * 0.8
    assert log[0]["kept_id"] == "nightstand_high"
    assert merged[0]["id"] == "nightstand_high"
    assert merged[0]["support"] == pytest.approx(support_low + support_high, rel=1e-3)


def test_different_class_overlap_not_merged():
    chair = _rect_object("chair_1", "chair", 0.0, 0.0, 1.0, 1.0)
    desk = _rect_object("desk_1", "desk", 0.0, 0.0, 1.0, 1.0)  # fully overlapping (IoU=1.0)

    merged, log = merge_duplicates([chair, desk])

    assert len(merged) == 2
    assert log == []
    ids = {o["id"] for o in merged}
    assert ids == {"chair_1", "desk_1"}
    assert "merged_from" not in merged[0]
    assert "merged_from" not in merged[1]


def test_chained_boxes_merge_transitively():
    # A-B and B-C each satisfy IoU > 0.3 (dx=0.35 -> iou ~0.481), but A-C
    # (dx=0.70) does not satisfy IoU (~0.176) or the centroid rule
    # (0.70 > 0.5*1.0) directly - only transitive union-find chaining
    # produces a single cluster.
    a = _rect_object("box_a", "box", 0.0, 0.0, 1.0, 1.0)
    b = _rect_object("box_b", "box", 0.35, 0.0, 1.0, 1.0)
    c = _rect_object("box_c", "box", 0.70, 0.0, 1.0, 1.0)

    merged, log = merge_duplicates([a, b, c])

    assert len(merged) == 1
    assert len(log) == 1
    assert sorted(merged[0]["merged_from"]) == ["box_a", "box_b", "box_c"]


def test_disjoint_same_class_objects_untouched():
    a = _rect_object("chair_1", "chair", 0.0, 0.0, 0.5, 0.5)
    b = _rect_object("chair_2", "chair", 10.0, 10.0, 0.5, 0.5)

    merged, log = merge_duplicates([a, b])

    assert merged == [a, b]
    assert log == []
    # identity preserved, not just equality - untouched objects pass through
    assert merged[0] is a
    assert merged[1] is b


def test_merged_hull_is_convex_and_contains_all_member_hulls():
    a = _rect_object("tv_1", "television", 0.0, 0.0, 1.0, 1.0)
    b = _rect_object("tv_2", "television", 0.25, 0.3, 1.0, 1.0, angle_rad=0.2)

    merged, _ = merge_duplicates([a, b])
    merged_poly = Polygon(merged[0]["hull_xz"])

    assert merged_poly.is_valid
    # a convex polygon equals its own convex hull
    assert merged_poly.convex_hull.equals(merged_poly)

    for member in (a, b):
        member_poly = Polygon(member["hull_xz"])
        # allow tiny floating-point slack at the boundary
        assert merged_poly.buffer(1e-9).contains(member_poly)
