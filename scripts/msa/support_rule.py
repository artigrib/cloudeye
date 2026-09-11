"""M7 item 2: the support rule - an object is kept only if something holds it up.

Three supports are recognised, checked in this order (the first that applies
is the logged `reason`):

- `floor`   - `bbox_min_y - floor_y < FLOOR_TOUCH_M` (0.15 m).
- `wall`    - the object's oriented-rectangle footprint comes within
              `WALL_TOUCH_M` (0.10 m) of the wall geometry: the union of the
              scanned wall polygons and the room outline (the T15g regularized
              room polygon's boundary ring, which is where the wall band sits).
              A wall-hung shelf or TV at any height qualifies.
- `stacked` - it sits on another KEPT object's top face:
              `|bbox_min_y - other_top_y| < STACK_TOUCH_M` (0.10 m) AND the two
              footprints overlap. "Kept" matters: support is re-evaluated to a
              fixed point, so a lamp on a floating shelf goes when the shelf goes
              (logged `nearest_support.kind = "support_dropped"`).

Everything else is `floating` and dropped - the ceiling-lamp detections that
SAM3 labels `lamp` at 1.9 m, reflections in a window, the "television" slab in
mid-air. Every decision (kept or dropped) is logged with the numbers that made
it, so a keep without a reason cannot exist by construction.

Runs on the `bootstrap.compute_object_footprints` dicts (`objects.json`
schema: `id`, `label`, `center_xy`, `size_uv`, `angle_rad`, `height`,
`bbox_min_y`, `hull_xz`, ...). Pure function, no I/O; wired into
`bootstrap.run_bootstrap` AFTER the duplicate merge (`dedupe.py`), the
small-object filter and the outside-room drop/clip, and BEFORE the class caps
(`class_caps.py`), so a merged hull is judged once and the caps rank only
supported objects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from scripts.msa.dedupe import _footprint_polygon

FLOOR_TOUCH_M = 0.15
WALL_TOUCH_M = 0.10
STACK_TOUCH_M = 0.10

REASON_FLOOR = "floor"
REASON_WALL = "wall"
REASON_STACKED = "stacked"
REASON_FLOATING = "floating"


def wall_geometry(walls=None, room_polygon=None):
    """One shapely geometry to measure wall distance against: the union of the
    wall polygons (`WallPolygon`s or `{"vertices": ...}` dicts) and the room
    outline ring. None when there is nothing to measure against."""
    parts = []
    for w in walls or []:
        vertices = w.vertices if hasattr(w, "vertices") else w["vertices"]
        if len(vertices) < 3:
            continue
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            parts.append(poly)
    if room_polygon is not None and len(room_polygon) >= 3:
        ring = [tuple(p) for p in room_polygon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        parts.append(LineString(ring))
    if not parts:
        return None
    return unary_union(parts)


def _valid_footprint(obj: dict) -> Polygon:
    fp = _footprint_polygon(obj)
    return fp if fp.is_valid else fp.buffer(0)


def _top_y(obj: dict) -> float:
    return float(obj["bbox_min_y"]) + float(obj["height"])


def _stack_candidate(obj: dict, fp: Polygon, others: list[tuple[dict, Polygon]], stack_touch_m: float) -> dict | None:
    """The best supporter among `others`: overlapping footprint and the smallest
    |z_min - other_top|. Returns its record (with `supports` = whether it is
    within `stack_touch_m`) or None when no footprint overlaps."""
    z_min = float(obj["bbox_min_y"])
    best = None
    for other, other_fp in others:
        if other is obj or other["id"] == obj["id"]:
            continue
        if fp.area <= 1e-12 or other_fp.area <= 1e-12:
            continue
        inter = fp.intersection(other_fp).area
        if inter <= 1e-9:
            continue
        dz = abs(z_min - _top_y(other))
        rec = {
            "kind": "object",
            "id": other["id"],
            "label": other.get("label"),
            "other_top_y": round(_top_y(other), 4),
            "dz_m": round(dz, 4),
            "overlap_fraction": round(inter / fp.area, 4),
            "supports": dz < stack_touch_m,
        }
        if best is None or dz < best["dz_m"]:
            best = rec
    return best


def apply_support_rule(
    objects: list[dict],
    *,
    floor_y: float,
    walls=None,
    room_polygon=None,
    floor_touch_m: float = FLOOR_TOUCH_M,
    wall_touch_m: float = WALL_TOUCH_M,
    stack_touch_m: float = STACK_TOUCH_M,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns `(kept objects in original order, drop log, keep log)`.

    Drop log entries: `{id, label, reason: "floating", z_min, nearest_support}`
    where `nearest_support` is the closest thing that ALMOST held it up
    (`{kind: "floor"|"wall"|"object"|"support_dropped"|"none", distance_m, ...}`).
    Keep log entries: `{id, label, reason: "floor"|"wall"|"stacked", z_min,
    nearest_support}` with the measured gap. Pure: input dicts are not mutated.
    """
    walls_geom = wall_geometry(walls, room_polygon)
    footprints = [(o, _valid_footprint(o)) for o in objects]

    # Floor and wall support never change; stacked support depends on which
    # objects survive, hence the fixed-point loop below.
    static: dict[str, dict] = {}
    for obj, fp in footprints:
        z_min = float(obj["bbox_min_y"])
        floor_gap = z_min - float(floor_y)
        wall_dist = float(fp.distance(walls_geom)) if walls_geom is not None and fp.area > 1e-12 else math.inf
        static[obj["id"]] = {"floor_gap_m": floor_gap, "wall_distance_m": wall_dist}

    kept_ids = {o["id"] for o in objects}
    reasons: dict[str, dict] = {}
    dropped_by_support: dict[str, str] = {}  # id -> the supporter that later fell away
    while True:
        changed = False
        alive = [(o, fp) for o, fp in footprints if o["id"] in kept_ids]
        for obj, fp in alive:
            s = static[obj["id"]]
            z_min = float(obj["bbox_min_y"])
            if s["floor_gap_m"] < floor_touch_m:
                reasons[obj["id"]] = {
                    "reason": REASON_FLOOR,
                    "nearest_support": {"kind": "floor", "distance_m": round(max(s["floor_gap_m"], 0.0), 4)},
                }
                continue
            if s["wall_distance_m"] < wall_touch_m:
                reasons[obj["id"]] = {
                    "reason": REASON_WALL,
                    "nearest_support": {"kind": "wall", "distance_m": round(s["wall_distance_m"], 4)},
                }
                continue
            cand = _stack_candidate(obj, fp, alive, stack_touch_m)
            if cand is not None and cand["supports"]:
                reasons[obj["id"]] = {
                    "reason": REASON_STACKED,
                    "nearest_support": {**{k: v for k, v in cand.items() if k != "supports"}, "distance_m": cand["dz_m"]},
                }
                continue
            # Floating. Record what came closest so the log explains the drop.
            previous = reasons.pop(obj["id"], None)
            if previous is not None and previous["reason"] == REASON_STACKED:
                dropped_by_support[obj["id"]] = previous["nearest_support"]["id"]
            kept_ids.discard(obj["id"])
            changed = True
            if z_min < 0 and False:  # pragma: no cover - placeholder for future below-floor handling
                pass
        if not changed:
            break

    kept = [o for o in objects if o["id"] in kept_ids]
    keep_log = [
        {"id": o["id"], "label": o.get("label"), "reason": reasons[o["id"]]["reason"], "z_min": round(float(o["bbox_min_y"]), 4),
         "nearest_support": reasons[o["id"]]["nearest_support"]}
        for o in kept
    ]
    drop_log = []
    for obj, fp in footprints:
        if obj["id"] in kept_ids:
            continue
        s = static[obj["id"]]
        z_min = float(obj["bbox_min_y"])
        supporter = dropped_by_support.get(obj["id"])
        if supporter is not None:
            nearest = {"kind": "support_dropped", "id": supporter, "distance_m": None}
        else:
            # The closest near-miss among floor / wall / any overlapping object.
            candidates = [("floor", max(s["floor_gap_m"], 0.0), {})]
            if math.isfinite(s["wall_distance_m"]):
                candidates.append(("wall", s["wall_distance_m"], {}))
            cand = _stack_candidate(obj, fp, [(o, f) for o, f in footprints if o["id"] in kept_ids], stack_touch_m)
            if cand is not None:
                candidates.append(("object", cand["dz_m"], {k: cand[k] for k in ("id", "label", "other_top_y", "overlap_fraction")}))
            kind, dist, extra = min(candidates, key=lambda c: c[1])
            nearest = {"kind": kind, "distance_m": round(float(dist), 4), **extra}
        drop_log.append({"id": obj["id"], "label": obj.get("label"), "reason": REASON_FLOATING, "z_min": round(z_min, 4),
                         "floor_gap_m": round(s["floor_gap_m"], 4),
                         "wall_distance_m": None if not math.isfinite(s["wall_distance_m"]) else round(s["wall_distance_m"], 4),
                         "nearest_support": nearest})
    return kept, drop_log, keep_log


def format_support_table(keep_log: list[dict], drop_log: list[dict]) -> str:
    header = f"{'id':<20}{'label':<12}{'z_min':>7}  {'verdict':<9}{'reason':<9}nearest support"
    lines = [header, "-" * len(header)]
    for e in keep_log:
        ns = e["nearest_support"]
        detail = f"{ns['kind']} {ns.get('id', '')} d={ns.get('distance_m')}".strip()
        lines.append(f"{e['id']:<20}{str(e['label']):<12}{e['z_min']:>7.3f}  {'kept':<9}{e['reason']:<9}{detail}")
    for e in drop_log:
        ns = e["nearest_support"]
        detail = f"{ns['kind']} {ns.get('id', '')} d={ns.get('distance_m')}".strip()
        lines.append(f"{e['id']:<20}{str(e['label']):<12}{e['z_min']:>7.3f}  {'DROPPED':<9}{e['reason']:<9}{detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the M7 support rule to an objects.json (offline).")
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    from scripts.msa.objects_io import load_objects_json, write_objects_json

    data = load_objects_json(args.objects)
    kept, drop_log, keep_log = apply_support_rule(data.objects, floor_y=data.floor_y, walls=data.walls, room_polygon=data.room_polygon)
    write_objects_json(args.out, kept, data.walls, data.room_polygon, data.floor_y, data.ceiling_y, yaw_meta=data.meta.get("yaw"))
    print(format_support_table(keep_log, drop_log))
    print(f"\n{len(data.objects)} objects -> {len(kept)} kept, {len(drop_log)} dropped as floating")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
