"""M7 item 1: duplicate-detection merge - collapse near-duplicate object
detections of the same class into a single object, before the (separate,
not-yet-built) support rule and class-cap stages run. The geo agent wires
these three in order: merge (this module) -> support rule -> class caps
(`scripts/msa/class_caps.py`, written concurrently by another agent). This
module does NOT wire into `bootstrap.py` - it is a standalone, pure function
plus a CLI for offline use on an existing `objects.json`.

Objects here are the dicts `bootstrap.compute_object_footprints` writes (see
`scripts/msa/objects_io.py` for the `objects.json` schema): `id`, `label`,
`center_xy`, `size_uv` (length_u, width_v), `angle_rad`, `height`,
`bbox_min_y`, `hull_xz` (the measured convex hull in world XZ, distinct from
the oriented-rectangle visual footprint), `color_rgb`.

Support field (M7 integration, 2026-09-07): `bootstrap.compute_object_footprints`
now copies the detection's real point count into the object dict as `n_points`
(from `scene_objects.json`'s `point_count`; absent for the hull-only test
fixture). Support is `class_caps.object_support`: `n_points` when present, else
the documented `hull area x height` proxy. A merged object carries the SUM of
its members' `n_points` (when every member has one) so the class caps rank the
merged detection by all the evidence that went into it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from scripts.msa.geometry import convex_hull_2d, oriented_min_area_rect

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_CENTROID_FRAC = 0.5


def _footprint_polygon(obj: dict) -> Polygon:
    """The object's oriented-rectangle visual footprint, as a shapely polygon
    in world XZ. Same corner construction as `bootstrap._footprint_polygon` /
    `export_dxf.build_floor_plan` - kept independent (not imported) so this
    module has no dependency on `bootstrap.py`."""
    cx, cz = obj["center_xy"]
    length_u, width_v = obj["size_uv"]
    angle = obj["angle_rad"]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u, v = length_u / 2, width_v / 2
    local_corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
    world_corners = [(cx + lx * cos_a - lz * sin_a, cz + lx * sin_a + lz * cos_a) for lx, lz in local_corners]
    return Polygon(world_corners)


def _support(obj: dict) -> float:
    """Scanned evidence: the detection's `n_points` when the object dict has it,
    else hull area x height - one definition shared with the class caps
    (`class_caps.object_support`), see module docstring."""
    from scripts.msa.class_caps import object_support

    return float(object_support(obj))


def _iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    inter = poly_a.intersection(poly_b).area
    if inter <= 0:
        return 0.0
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def _centroid_dist(a: dict, b: dict) -> float:
    ax, az = a["center_xy"]
    bx, bz = b["center_xy"]
    return math.hypot(ax - bx, az - bz)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _merge_cluster(members: list[dict]) -> dict:
    """Collapse one cluster (size > 1) into a single merged object dict."""
    supports = [_support(m) for m in members]
    winner_pos = max(range(len(members)), key=lambda k: supports[k])
    winner = members[winner_pos]

    hull_points = np.concatenate([np.asarray(m["hull_xz"], dtype=float) for m in members], axis=0)
    merged_hull = convex_hull_2d(hull_points)
    center_xy, size_uv, angle = oriented_min_area_rect(merged_hull)

    bbox_min_y = min(float(m["bbox_min_y"]) for m in members)
    top_y = max(float(m["bbox_min_y"]) + float(m["height"]) for m in members)
    height = max(top_y - bbox_min_y, 0.01)

    merged = dict(winner)  # keeps winner's id, label, color_rgb (and any other pass-through fields)
    merged["center_xy"] = center_xy
    merged["size_uv"] = size_uv
    merged["angle_rad"] = angle
    merged["height"] = height
    merged["bbox_min_y"] = bbox_min_y
    merged["hull_xz"] = merged_hull.tolist()
    merged["merged_from"] = [m["id"] for m in members]
    merged["support"] = float(sum(supports))  # synthetic field - see module docstring
    if all("n_points" in m for m in members):
        merged["n_points"] = int(sum(int(m["n_points"]) for m in members))
    return merged


def merge_duplicates(
    objects: list[dict],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    centroid_frac: float = DEFAULT_CENTROID_FRAC,
) -> tuple[list[dict], list[dict]]:
    """Cluster same-`label` objects whose oriented-rectangle footprints have
    IoU > `iou_threshold` OR whose centroid distance is < `centroid_frac` x
    max(w, l) of the larger of the pair (by footprint area), using union-find
    so chains of pairwise-qualifying detections merge transitively even when
    the endpoints of the chain don't themselves satisfy either rule.

    Pure function, no I/O. Objects that don't merge with anything are
    returned unchanged (same dict identity) at their original position;
    merged clusters appear once, at the position of their earliest member.

    Returns (merged objects in original relative order, merge log entries:
    `{kept_id, merged_ids, reason: "iou"|"centroid", iou, centroid_dist_m}`).
    """
    n = len(objects)
    if n == 0:
        return [], []

    uf = _UnionFind(n)
    polys = [_footprint_polygon(o) for o in objects]
    # (i, j) -> (reason, iou, centroid_dist_m) for every pair that actually
    # triggered a union - kept so the merge log can report real evidence
    # rather than a value recomputed after the fact.
    pair_evidence: dict[tuple[int, int], tuple[str, float, float]] = {}

    for i in range(n):
        for j in range(i + 1, n):
            if objects[i]["label"] != objects[j]["label"]:
                continue
            iou = _iou(polys[i], polys[j])
            dist = _centroid_dist(objects[i], objects[j])
            area_i = objects[i]["size_uv"][0] * objects[i]["size_uv"][1]
            area_j = objects[j]["size_uv"][0] * objects[j]["size_uv"][1]
            larger = objects[i] if area_i >= area_j else objects[j]
            centroid_limit = centroid_frac * max(larger["size_uv"])

            if iou > iou_threshold:
                uf.union(i, j)
                pair_evidence[(i, j)] = ("iou", iou, dist)
            elif dist < centroid_limit:
                uf.union(i, j)
                pair_evidence[(i, j)] = ("centroid", iou, dist)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    result: list[dict] = []
    log: list[dict] = []
    processed_roots: set[int] = set()
    for i in range(n):
        root = uf.find(i)
        if root in processed_roots:
            continue
        processed_roots.add(root)
        idxs = groups[root]
        if len(idxs) == 1:
            result.append(objects[idxs[0]])
            continue

        members = [objects[k] for k in idxs]
        merged = _merge_cluster(members)
        result.append(merged)

        edges = [pair_evidence[(a, b)] for a in idxs for b in idxs if a < b and (a, b) in pair_evidence]
        reason = "iou" if any(r == "iou" for r, _, _ in edges) else "centroid"
        iou_val = max((v for _, v, _ in edges), default=0.0)
        centroid_val = min((d for _, _, d in edges), default=0.0)
        log.append(
            {
                "kept_id": merged["id"],
                "merged_ids": [m["id"] for m in members if m["id"] != merged["id"]],
                "reason": reason,
                "iou": round(iou_val, 4),
                "centroid_dist_m": round(centroid_val, 4),
            }
        )

    return result, log


def _print_log_table(log: list[dict]) -> None:
    if not log:
        print("No duplicates merged.")
        return
    header = f"{'kept_id':<24} {'merged_ids':<40} {'reason':<10} {'iou':>8} {'centroid_dist_m':>16}"
    print(header)
    print("-" * len(header))
    for entry in log:
        print(
            f"{entry['kept_id']:<24} {','.join(entry['merged_ids']):<40} "
            f"{entry['reason']:<10} {entry['iou']:>8.3f} {entry['centroid_dist_m']:>16.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True, help="input objects.json")
    parser.add_argument("--out", type=Path, required=True, help="output objects.json (merged)")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--centroid-frac", type=float, default=DEFAULT_CENTROID_FRAC)
    args = parser.parse_args()

    from collections import Counter

    from scripts.msa.objects_io import load_objects_json, write_objects_json

    data = load_objects_json(args.objects)
    merged, log = merge_duplicates(data.objects, iou_threshold=args.iou_threshold, centroid_frac=args.centroid_frac)

    write_objects_json(
        args.out,
        merged,
        data.walls,
        data.room_polygon,
        data.floor_y,
        data.ceiling_y,
        yaw_meta=data.meta.get("yaw"),
    )

    _print_log_table(log)
    before = Counter(o["label"] for o in data.objects)
    after = Counter(o["label"] for o in merged)
    print()
    print(f"{len(data.objects)} objects -> {len(merged)} after merge ({len(log)} clusters merged)")
    print(f"{'label':<16} {'before':>8} {'after':>8}")
    for label in sorted(set(before) | set(after)):
        print(f"{label:<16} {before.get(label, 0):>8} {after.get(label, 0):>8}")


if __name__ == "__main__":
    main()
