"""M7 item 2: the support rule (scripts/msa/support_rule.py)."""

from __future__ import annotations

from scripts.msa.geometry import WallPolygon
from scripts.msa.support_rule import (
    REASON_FLOATING,
    REASON_FLOOR,
    REASON_STACKED,
    REASON_WALL,
    apply_support_rule,
    format_support_table,
    wall_geometry,
)

FLOOR_Y = 0.0
# A 6 x 5 m room: outline ring + two scanned wall fragments along z=0 and x=6.
ROOM = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]
WALLS = [
    WallPolygon(vertices=[(0.0, -0.1), (6.0, -0.1), (6.0, 0.0), (0.0, 0.0)], area_m2=0.6),
    WallPolygon(vertices=[(6.0, 0.0), (6.1, 0.0), (6.1, 5.0), (6.0, 5.0)], area_m2=0.5),
]


def _obj(obj_id, label, center, size_uv, bbox_min_y, height, angle_rad=0.0):
    return {
        "id": obj_id, "label": label, "center_xy": center, "size_uv": size_uv, "angle_rad": angle_rad,
        "height": height, "bbox_min_y": bbox_min_y, "color_rgb": (100, 100, 100), "hull_xz": [],
    }


def _run(objects):
    return apply_support_rule(objects, floor_y=FLOOR_Y, walls=WALLS, room_polygon=ROOM)


def test_floor_object_kept_with_floor_reason():
    bed = _obj("bed_0", "bed", (3.0, 2.5), (2.0, 1.5), bbox_min_y=0.05, height=0.6)
    kept, drops, keeps = _run([bed])
    assert [o["id"] for o in kept] == ["bed_0"] and drops == []
    assert keeps[0]["reason"] == REASON_FLOOR and keeps[0]["nearest_support"]["kind"] == "floor"
    assert keeps[0]["nearest_support"]["distance_m"] == 0.05


def test_wall_hugging_shelf_at_1_2m_kept():
    # 1.2 m off the floor, footprint 0.04 m from the z=0 wall fragment / outline.
    shelf = _obj("shelf_0", "shelf", (3.0, 0.19), (1.0, 0.3), bbox_min_y=1.2, height=0.3)
    kept, drops, keeps = _run([shelf])
    assert [o["id"] for o in kept] == ["shelf_0"] and drops == []
    assert keeps[0]["reason"] == REASON_WALL
    assert abs(keeps[0]["nearest_support"]["distance_m"] - 0.04) < 1e-3


def test_lamp_on_a_nightstand_kept_as_stacked():
    nightstand = _obj("nightstand_0", "nightstand", (2.0, 2.5), (0.5, 0.5), bbox_min_y=0.0, height=0.6)
    lamp = _obj("lamp_0", "lamp", (2.05, 2.55), (0.2, 0.2), bbox_min_y=0.62, height=0.4)
    kept, drops, keeps = _run([nightstand, lamp])
    assert [o["id"] for o in kept] == ["nightstand_0", "lamp_0"] and drops == []
    lamp_keep = next(k for k in keeps if k["id"] == "lamp_0")
    assert lamp_keep["reason"] == REASON_STACKED
    assert lamp_keep["nearest_support"]["id"] == "nightstand_0"
    assert abs(lamp_keep["nearest_support"]["dz_m"] - 0.02) < 1e-6
    assert lamp_keep["nearest_support"]["overlap_fraction"] > 0.5


def test_lamp_floating_half_a_metre_above_nothing_dropped():
    lamp = _obj("lamp_1", "lamp", (3.0, 2.5), (0.2, 0.2), bbox_min_y=0.5, height=0.4)
    kept, drops, keeps = _run([lamp])
    assert kept == [] and keeps == []
    assert drops[0]["id"] == "lamp_1" and drops[0]["reason"] == REASON_FLOATING
    assert drops[0]["z_min"] == 0.5
    assert drops[0]["nearest_support"]["kind"] == "floor" and drops[0]["nearest_support"]["distance_m"] == 0.5


def test_ceiling_lamp_dropped_even_over_a_bed():
    bed = _obj("bed_0", "bed", (3.0, 2.5), (2.0, 1.5), bbox_min_y=0.0, height=0.6)
    ceiling_lamp = _obj("lamp_2", "lamp", (3.0, 2.5), (0.3, 0.3), bbox_min_y=1.9, height=0.3)
    kept, drops, _ = _run([bed, ceiling_lamp])
    assert [o["id"] for o in kept] == ["bed_0"]
    assert drops[0]["nearest_support"] == {"kind": "object", "distance_m": 1.3, "id": "bed_0", "label": "bed", "other_top_y": 0.6, "overlap_fraction": 1.0}


def test_stacked_support_is_transitive_a_lamp_on_a_floating_shelf_goes_with_it():
    shelf = _obj("shelf_1", "shelf", (3.0, 2.5), (1.0, 0.3), bbox_min_y=1.0, height=0.05)  # mid-room, nothing under it
    lamp = _obj("lamp_3", "lamp", (3.0, 2.5), (0.2, 0.2), bbox_min_y=1.06, height=0.3)
    kept, drops, _ = _run([shelf, lamp])
    assert kept == []
    by_id = {d["id"]: d for d in drops}
    assert by_id["shelf_1"]["nearest_support"]["kind"] == "floor"
    assert by_id["lamp_3"]["nearest_support"] == {"kind": "support_dropped", "id": "shelf_1", "distance_m": None}


def test_every_kept_object_has_a_reason_and_order_is_preserved():
    objs = [
        _obj("chair_0", "chair", (1.0, 1.0), (0.5, 0.5), 0.1, 0.9),
        _obj("tv_0", "television", (5.9, 2.5), (0.1, 1.2), 1.1, 0.7, angle_rad=0.0),  # on the x=6 wall
        _obj("lamp_4", "lamp", (3.0, 2.5), (0.2, 0.2), 0.8, 0.3),  # floating mid-room, nothing under it
        _obj("desk_0", "desk", (1.0, 4.0), (1.2, 0.6), 0.0, 0.75),
        _obj("monitor_0", "monitor", (1.0, 4.0), (0.5, 0.2), 0.78, 0.4),  # on the desk
    ]
    kept, drops, keeps = _run(objs)
    assert [o["id"] for o in kept] == ["chair_0", "tv_0", "desk_0", "monitor_0"]
    assert {k["id"]: k["reason"] for k in keeps} == {"chair_0": REASON_FLOOR, "tv_0": REASON_WALL, "desk_0": REASON_FLOOR, "monitor_0": REASON_STACKED}
    assert [d["id"] for d in drops] == ["lamp_4"]
    assert all(k["reason"] in (REASON_FLOOR, REASON_WALL, REASON_STACKED) for k in keeps)
    table = format_support_table(keeps, drops)
    assert "lamp_4" in table and "DROPPED" in table


def test_wall_geometry_uses_outline_when_there_are_no_wall_polygons():
    geom = wall_geometry([], ROOM)
    assert geom is not None and geom.length > 0
    assert wall_geometry([], None) is None
    tv = _obj("tv_1", "television", (3.0, 4.95), (1.0, 0.1), bbox_min_y=1.2, height=0.6)
    kept, _, keeps = apply_support_rule([tv], floor_y=FLOOR_Y, walls=[], room_polygon=ROOM)
    assert kept and keeps[0]["reason"] == REASON_WALL


def test_inputs_not_mutated():
    lamp = _obj("lamp_5", "lamp", (3.0, 2.5), (0.2, 0.2), 0.5, 0.4)
    before = dict(lamp)
    apply_support_rule([lamp], floor_y=FLOOR_Y, walls=WALLS, room_polygon=ROOM)
    assert lamp == before
