"""M7 item 3: per-scene class caps.

Some scenes over-detect a class (e.g. 13 "lamp" instances in one bedroom scan)
because the upstream point-cloud segmentation over-splits a single physical
object, or genuinely-implausible duplicates survive earlier stages. This module
applies a small per-class prior cap: for any label with more instances than its
cap, keep the `cap` highest-"support" instances and drop the rest.

Runs on the object dicts `bootstrap.compute_object_footprints` writes to
`objects.json` (see `scripts/msa/objects_io.py` for the schema): `id`, `label`,
`center_xy`, `size_uv`, `angle_rad`, `height`, `bbox_min_y`, `color_rgb`,
`hull_xz`. NOTE: the upstream `scene_objects.json` input carries a
`point_count` per object, but `compute_object_footprints` does not copy it
into the persisted object dict (checked against the real schema in
`scripts/msa/bootstrap.py` and the hero fixture at
`var/scratch/run-20260906/morning_out/objects.json` - neither has any
point-count/support field). So there is no measured point-cloud support left
by the time objects reach this stage. Support here therefore falls back to
`hull_xz` polygon area x `height` (a proxy for "how much of the scan this
object accounts for"). If a future object dict *does* carry a point-count
field (checked via `SUPPORT_FIELD_CANDIDATES` below), that field is preferred
over the geometric proxy.

Pure function; does not mutate its input and is not wired into
`bootstrap.run_bootstrap` - a later integration step composes this with the
duplicate-merge and support-rule stages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# A small prior. Classes not listed here are left uncapped.
CLASS_CAPS: dict[str, int] = {
    "bed": 2,
    "tv": 2,
    "television": 2,
    "desk": 2,
    "chair": 4,
    "lamp": 4,
    "nightstand": 2,
    "wardrobe": 2,
}

# Synonyms collapsed to one canonical class before caps are looked up/applied,
# so e.g. 2 "tv" + 2 "television" instances of the same physical class are
# capped together, not 2-and-2 independently.
_SYNONYMS: dict[str, str] = {
    "television": "tv",
    "bedside": "nightstand",
    "closet": "wardrobe",
}

# Preferred order of dedicated support/point-count fields, if an object dict
# ever carries one (see module docstring - the real schema currently doesn't).
SUPPORT_FIELD_CANDIDATES: tuple[str, ...] = ("n_points", "num_points", "point_count", "n_pts", "npoints")


def normalize_label(label: str | None) -> str:
    """Canonical class name for cap lookup/grouping: lower-cased, synonym-
    collapsed. Unknown labels pass through unchanged (uncapped)."""
    key = (label or "").strip().lower()
    return _SYNONYMS.get(key, key)


def _hull_area(hull_xz) -> float:
    """Shoelace area of the (possibly unclosed) hull polygon. Returns 0.0 for
    degenerate hulls (< 3 points), never raises."""
    pts = list(hull_xz or [])
    n = len(pts)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, z1 = pts[i]
        x2, z2 = pts[(i + 1) % n]
        area2 += x1 * z2 - x2 * z1
    return abs(area2) / 2.0


def _support_field_name(obj: dict) -> str | None:
    for name in SUPPORT_FIELD_CANDIDATES:
        if name in obj:
            return name
    return None


def object_support(obj: dict) -> float:
    """A dedicated point-count field if the object dict has one; otherwise
    hull area x height, per the module docstring."""
    field = _support_field_name(obj)
    if field is not None:
        try:
            return float(obj[field])
        except (TypeError, ValueError):
            pass
    return _hull_area(obj.get("hull_xz")) * float(obj.get("height") or 0.0)


def apply_class_caps(objects: list[dict], caps: dict[str, int] = CLASS_CAPS) -> tuple[list[dict], list[dict]]:
    """Per class (after synonym normalization), keep the `cap` highest-support
    instances and drop the rest. Ties broken by larger hull area, then by id
    (ascending, for determinism). Classes with no entry in `caps` (after
    normalization) are left untouched.

    Returns (kept objects in original order, drop log). Each drop log entry is
    `{id, label, support, rank, cap, reason: "class_cap"}` where `rank` is the
    instance's 1-indexed support rank within its class (so `rank > cap` is why
    it was dropped). Pure: does not mutate `objects` or its dicts.
    """
    normalized_caps: dict[str, int] = {}
    for raw_label, cap in caps.items():
        normalized_caps[normalize_label(raw_label)] = cap

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, obj in enumerate(objects):
        groups[normalize_label(obj.get("label"))].append(idx)

    drop_indices: set[int] = set()
    drop_log: list[dict] = []
    for norm_label, indices in groups.items():
        cap = normalized_caps.get(norm_label)
        if cap is None or len(indices) <= cap:
            continue

        def _sort_key(i: int, _objects=objects):
            obj = _objects[i]
            return (-object_support(obj), -_hull_area(obj.get("hull_xz")), str(obj.get("id", "")))

        ranked = sorted(indices, key=_sort_key)
        for rank, i in enumerate(ranked, start=1):
            if rank <= cap:
                continue
            obj = objects[i]
            drop_indices.add(i)
            drop_log.append(
                {
                    "id": obj.get("id"),
                    "label": obj.get("label"),
                    "support": object_support(obj),
                    "rank": rank,
                    "cap": cap,
                    "reason": "class_cap",
                }
            )

    kept = [obj for i, obj in enumerate(objects) if i not in drop_indices]
    drop_log.sort(key=lambda d: (normalize_label(d["label"]), d["rank"]))
    return kept, drop_log


def _print_drop_table(objects: list[dict], kept: list[dict], drop_log: list[dict], caps: dict[str, int]) -> None:
    normalized_caps: dict[str, int] = {}
    for raw_label, cap in caps.items():
        normalized_caps[normalize_label(raw_label)] = cap

    before = Counter(normalize_label(o.get("label")) for o in objects)
    after = Counter(normalize_label(o.get("label")) for o in kept)

    print("class caps applied:")
    for label in sorted(before):
        cap = normalized_caps.get(label)
        cap_str = str(cap) if cap is not None else "-"
        print(f"  {label:<12s} before={before[label]:>3d} after={after.get(label, 0):>3d} cap={cap_str}")

    if drop_log:
        print("dropped:")
        for d in drop_log:
            print(f"  id={d['id']!s:<24s} label={d['label']!s:<12s} support={d['support']:>10.4f} rank={d['rank']} cap={d['cap']}")
    else:
        print("dropped: none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply per-scene class caps to an objects.json file.")
    parser.add_argument("--objects", required=True, type=Path, help="Path to objects.json (or a bare objects list JSON)")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the capped result")
    parser.add_argument(
        "--cap",
        action="append",
        default=[],
        metavar="LABEL=N",
        help="Override or add a class cap, e.g. --cap tv=1. Repeatable.",
    )
    args = parser.parse_args(argv)

    caps = dict(CLASS_CAPS)
    for item in args.cap:
        label, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--cap expects LABEL=N, got {item!r}")
        caps[normalize_label(label.strip())] = int(value)

    payload = json.loads(args.objects.read_text())
    is_wrapped = isinstance(payload, dict) and "objects" in payload
    objects = payload["objects"] if is_wrapped else payload

    kept, drop_log = apply_class_caps(objects, caps)

    if is_wrapped:
        out_payload = dict(payload)
        out_payload["objects"] = kept
    else:
        out_payload = kept

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2))

    _print_drop_table(objects, kept, drop_log, caps)
    print(f"wrote {len(kept)}/{len(objects)} objects to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
