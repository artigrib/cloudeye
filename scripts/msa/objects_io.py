"""T13: `objects.json` - the bootstrap's FINAL kept object list persisted next
to `report.json`, so downstream assembly (`assemble_with_generated.py`) reads
exactly what `scene.glb` was built from instead of recomputing footprints
itself and silently bypassing wall-backfill (T4), the small-object filter
(T5), the outside-room drop/clip (T5b/T15a) and the yaw rotation (T3''/T15a).

Payload (schema_version 1):
    floor_y, ceiling_y
    room_polygon   [[x, z], ...] | null  - the (yaw-rotated) T15a room polygon
    floor_polygon  same list - alias kept under the pre-T15a name
    walls          [{"vertices": [[x, z], ...], "area_m2": float}, ...]
    objects        [{id, label, center_xy, size_uv, angle_rad, height,
                     bbox_min_y, color_rgb, hull_xz, ...}, ...]  (every field
                    `bootstrap.compute_object_footprints` + later stages set)
    yaw            the bootstrap's yaw_meta (provenance; optional)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.msa.geometry import WallPolygon

OBJECTS_JSON_SCHEMA_VERSION = 1
OBJECTS_JSON_NAME = "objects.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def objects_json_payload(
    objects: list[dict],
    walls: list[WallPolygon],
    room_polygon: list[tuple[float, float]] | None,
    floor_y: float,
    ceiling_y: float,
    *,
    yaw_meta: dict | None = None,
) -> dict:
    polygon = [[float(x), float(z)] for x, z in room_polygon] if room_polygon else None
    return {
        "schema_version": OBJECTS_JSON_SCHEMA_VERSION,
        "floor_y": float(floor_y),
        "ceiling_y": float(ceiling_y),
        "room_polygon": polygon,
        "floor_polygon": polygon,
        "walls": [{"vertices": [[float(x), float(z)] for x, z in w.vertices], "area_m2": float(w.area_m2)} for w in walls],
        "objects": [_jsonable(obj) for obj in objects],
        "yaw": _jsonable(yaw_meta) if yaw_meta else None,
    }


def write_objects_json(
    path: Path,
    objects: list[dict],
    walls: list[WallPolygon],
    room_polygon: list[tuple[float, float]] | None,
    floor_y: float,
    ceiling_y: float,
    *,
    yaw_meta: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(objects_json_payload(objects, walls, room_polygon, floor_y, ceiling_y, yaw_meta=yaw_meta), indent=2))
    return path


@dataclass
class BootstrapObjects:
    objects: list[dict]
    walls: list[WallPolygon]
    room_polygon: list[tuple[float, float]] | None
    floor_y: float
    ceiling_y: float
    meta: dict = field(default_factory=dict)

    @property
    def floor_polygon(self):  # pre-T15a name, same polygon
        return self.room_polygon


def load_objects_json(path: Path) -> BootstrapObjects:
    data = json.loads(Path(path).read_text())
    version = data.get("schema_version")
    if version != OBJECTS_JSON_SCHEMA_VERSION:
        raise ValueError(f"{path}: objects.json schema_version {version!r}, expected {OBJECTS_JSON_SCHEMA_VERSION}")
    polygon = data.get("room_polygon") or data.get("floor_polygon")
    objects = []
    for obj in data["objects"]:
        obj = dict(obj)
        obj["center_xy"] = tuple(float(v) for v in obj["center_xy"])
        obj["size_uv"] = tuple(float(v) for v in obj["size_uv"])
        if obj.get("color_rgb") is not None:
            obj["color_rgb"] = tuple(int(v) for v in obj["color_rgb"])
        objects.append(obj)
    return BootstrapObjects(
        objects=objects,
        walls=[WallPolygon(vertices=[(float(x), float(z)) for x, z in w["vertices"]], area_m2=float(w["area_m2"])) for w in data.get("walls", [])],
        room_polygon=[(float(x), float(z)) for x, z in polygon] if polygon else None,
        floor_y=float(data["floor_y"]),
        ceiling_y=float(data["ceiling_y"]),
        meta={"yaw": data.get("yaw")},
    )
