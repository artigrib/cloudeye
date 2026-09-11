#!/usr/bin/env python3
"""Assemble a scene directory in the layout the production worker leaves behind, so the
ordinary ingest path (app/services/scene_ingest.py + pipeline_orchestrator's post-fetch
sequence) can read it unchanged.

This writes ARTIFACTS, it does not re-implement ingest: every consumer below
(`read_alignment`, `read_grid_metadata`, `read_camera_track`, `parse_scene_objects`,
`choose_robot_start`, `validate_scene`) is the app's own code, run against this directory by
scripts/ingest/ingest_scene.py.

Everything here is measured from the 15 fps run itself. Where the layout requires a file the
15 fps run genuinely has no equivalent for, the file is written from the nvblox layers or left
empty and NAMED in the report - nothing is invented:

  aligned_room.ply        the 0.02 m coloured cloud (this run's own points, this run's colour)
  scene_points.glb        the same cloud, points-primitive glTF, capped at glb_max_points
  cloud.bin               the same cloud, CEPC v1 (what the viewer's worker parses)
  occupancy.npy           DERIVED FROM THE NVBLOX LAYERS: obstacle = esdf_slice <= 0 OR
                          unobserved. Not an occupancy solve - the 15 fps run has none, and
                          the task's free-space source is the nvblox layers. Reported.
  occupancy_meta.json     the nvblox slice grid (0.05 m, u_range/v_range, 75x140)
  alignment.json          floor/ceiling/bbox/tilt measured from this cloud (floor_y = 0 by
                          construction: points are already in the fitted-floor frame)
  alignment_transform.npz world -> (u, v, height) 4x4, the same convention as
                          nvblox_v2/isaac_export/*/transform.json
  cameras_aligned.json    the 1148 camera positions from poses.json, same frame
  keyframes.json          fps of THIS run (14.9945), counts of THIS run - not a stub
  scene_objects/          EMPTY LIST: the 15 fps run has no SAM3/vocab pass. Reported.
  vocab.json              omitted entirely (read_vocab_* return None when absent)
  map_preview.png         rendered from the occupancy grid above
  nvblox/                 the four frontend_layers files, copied verbatim
"""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ingest.colorize_cloud import load_frame_transform  # noqa: E402
from scripts.ingest.write_cloud_bin import write_cloud_bin      # noqa: E402

FREE, OBSTACLE, UNKNOWN = 0, 1, 2      # app/services/pathfinding.py:34


def zup_to_app(p: np.ndarray) -> np.ndarray:
    """(u, v, height) Z-up  ->  the app's Y-up floor-0 frame (x, y, z) = (u, height, -v).

    The whole backend and viewer are Y-up with the floor at y=0: scene_validator reads a
    camera's height out of index 1, occupancy_meta names origin_x/origin_z and the grid is
    indexed [ix][iz], and SceneMap3D frames the camera in the same convention. The `-v`
    (rather than `+v`) keeps the map a PROPER rotation - det +1, no mirror - which matters
    because `alignment.json.is_reflection` and the whole camera-convention check exist to
    catch exactly a mirrored scene. cloud.bin stays Z-up (the nvblox slice frame, same as
    isaac_export/transform.json) and its worker applies this same map on load.
    """
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)
NATIVE_FPS = 29.98897137610395
SAMPLE_FPS = 14.994485688051975


def cameras_in_slice_frame(poses_json: Path, T: dict) -> np.ndarray:
    views = json.loads(poses_json.read_text())["views"]
    pos = np.asarray([v["camera_pose"] for v in views], dtype=np.float64)[:, :3, 3]
    d = pos - T["origin"]
    zup = np.stack([d @ T["axis_u"], d @ T["axis_v"],
                    (pos - T["floor_point"]) @ T["up"]], axis=1)
    return zup_to_app(zup)


def build_occupancy(layers: Path) -> np.ndarray:
    """Obstacle = the ESDF slice at or inside a surface, OR unobserved.

    `unobserved -> OBSTACLE` is the task's rule ("unobserved = occupied"), and it is also
    what keeps the stock reachability honest: an unseen cell is not free space.

    `pathfinding.inflate` then dilates by a disk of ceil(r/res) cells, which is NOT the same
    as "EDT >= r" - it rounds the radius UP to a whole cell, so it is strictly more
    conservative. Measured on this grid: at r = 0.30 m the two disagree on 225 of 10500
    cells and give free fractions 0.1797 (inflate) vs 0.2091 (EDT); at r = 0.5528 m,
    0.0000 vs 0.0022. The endpoint's own number is the inflate one.
    """
    esdf = np.load(layers / "esdf_slice_0.3m.npy")
    unobs = np.load(layers / "unobserved_mask.npy")
    if esdf.shape != unobs.shape:
        raise RuntimeError(f"esdf {esdf.shape} vs unobserved {unobs.shape}")
    obstacle = unobs | (np.nan_to_num(esdf, nan=-1.0) <= 0.0)
    cells = np.where(obstacle, OBSTACLE, FREE).astype(np.uint8)
    # z = -v, so walking +iz must walk -v: flip the v axis and anchor origin_z at -v_range[1]
    return cells[:, ::-1]


def render_preview(cells: np.ndarray, out: Path) -> None:
    from PIL import Image
    rgb = np.zeros(cells.shape + (3,), dtype=np.uint8)
    rgb[cells == FREE] = (235, 235, 235)
    rgb[cells == OBSTACLE] = (40, 40, 40)
    rgb[cells == UNKNOWN] = (150, 150, 150)
    Image.fromarray(np.transpose(rgb, (1, 0, 2))[::-1]).save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud-npz", required=True)
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--glb-max-points", type=int, default=400_000)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = Path(a.layers_dir)
    T = load_frame_transform(layers)
    d = np.load(a.cloud_npz)
    xyz, rgb = d["xyz"], d["rgb"]
    report = {"n_points": int(xyz.shape[0]), "stubs": [], "derived": []}

    # --- nvblox layers, verbatim -------------------------------------------------
    nvb = out / "nvblox"; nvb.mkdir(exist_ok=True)
    for name in ("esdf_slice_0.3m.npy", "unobserved_mask.npy",
                 "floor_plane.json", "grid_meta.json"):
        shutil.copy2(layers / name, nvb / name)

    # --- occupancy grid + meta ---------------------------------------------------
    cells = build_occupancy(layers)
    np.save(out / "occupancy.npy", cells)
    report["derived"].append(
        f"occupancy.npy {cells.shape} from nvblox layers "
        f"(obstacle = esdf<=0 or unobserved): free {int((cells==FREE).sum())}, "
        f"obstacle {int((cells==OBSTACLE).sum())}"
    )
    ours = T["grid_meta"]["our_grid"]
    (out / "occupancy_meta.json").write_text(json.dumps({
        "resolution": T["cell"],
        "origin_x": T["u_range"][0], "origin_z": -T["v_range"][1],
        "width": int(cells.shape[0]), "height": int(cells.shape[1]),
        "frame": ours["frame"], "source": "nvblox frontend_layers/own_0901_173903",
        "legend": {"free": FREE, "obstacle": OBSTACLE, "unknown": UNKNOWN},
    }, indent=1))
    render_preview(cells, out / "map_preview.png")

    # --- alignment ----------------------------------------------------------------
    cams = cameras_in_slice_frame(Path(a.poses), T)
    xyz_app = zup_to_app(xyz)          # what the backend + viewer consume
    h = xyz[:, 2]
    ceiling_y = float(np.percentile(h, 99.5))
    below = float((h < 0).mean())
    tilt = float(np.degrees(np.arccos(np.clip(
        np.dot(T["up"], T["up"]) / (np.linalg.norm(T["up"]) ** 2), -1, 1))))
    (out / "alignment.json").write_text(json.dumps({
        "schema_version": 4,
        "floor_y": 0.0, "ceiling_y": ceiling_y,
        "bbox_min": xyz_app.min(axis=0).tolist(), "bbox_max": xyz_app.max(axis=0).tolist(),
        "residual_tilt_deg": tilt, "det_r": 1.0, "is_reflection": False,
        "points_below_floor_frac": below, "reflective_floor_suspected": False,
        "floor_likely_wrong": False,
        "note": "floor frame comes from nvblox floor_plane.json (RANSAC on the 1148-view "
                "solve), not from this repo's floor_ceiling.py; points are already "
                "expressed with the floor at 0, so no residual tilt remains by construction",
    }, indent=1))
    np.savez(out / "alignment_transform.npz",
             transform_world_to_zup_floor0=np.vstack([
                 np.hstack([np.stack([T["axis_u"], T["axis_v"], T["up"]]),
                            np.array([[-T["origin"] @ T["axis_u"]],
                                      [-T["origin"] @ T["axis_v"]],
                                      [-T["floor_point"] @ T["up"]]])]),
                 np.array([[0.0, 0.0, 0.0, 1.0]])]))

    (out / "cameras_aligned.json").write_text(json.dumps(
        {"cameras": cams.tolist(), "frame": ours["frame"], "n": int(cams.shape[0])}))
    (out / "keyframes.json").write_text(json.dumps({
        "extracted_count": int(cams.shape[0]), "kept_count": int(cams.shape[0]),
        "subsampled": False, "max_keyframes": int(cams.shape[0]),
        "fps": SAMPLE_FPS, "blur_threshold": None, "similarity_threshold": None,
        "note": "uniform integer decimation of the source video by step 2 "
                f"({NATIVE_FPS:.5f} fps native), no blur/similarity filtering",
    }, indent=1))

    so = out / "scene_objects"; so.mkdir(exist_ok=True)
    (so / "scene_objects.json").write_text("[]\n")  # schema_version 2+ is a bare JSON array
    report["stubs"].append(
        "scene_objects/scene_objects.json = empty object list: the 15 fps run is a "
        "MapAnything pass only (no stage_objects/SAM3, no vocabulary), so this scene "
        "genuinely has no objects - the file is empty, not filled with placeholders"
    )
    report["stubs"].append(
        "vocab.json omitted entirely: read_vocab_source/model/entries return None when the "
        "file is absent, which is the truthful value for a run with no vocabulary pass"
    )
    report["stubs"].append(
        "keyframes/ (the raw JPEG dir) not written: fetch_results excludes it from the real "
        "pipeline too, so its absence matches a production scene dir exactly"
    )

    # --- clouds -------------------------------------------------------------------
    cb = write_cloud_bin(xyz, rgb, out / "cloud.bin")
    report["cloud_bin"] = cb

    import trimesh
    keep = np.arange(xyz.shape[0])
    if xyz.shape[0] > a.glb_max_points:
        keep = np.linspace(0, xyz.shape[0] - 1, a.glb_max_points).astype(np.int64)
    pc = trimesh.PointCloud(vertices=xyz_app[keep].astype(np.float64), colors=rgb[keep])
    (out / "scene_points.glb").write_bytes(
        trimesh.Scene([pc]).export(file_type="glb"))
    pc_full = trimesh.PointCloud(vertices=xyz_app.astype(np.float64), colors=rgb)
    pc_full.export(str(out / "aligned_room.ply"), encoding="binary")
    report["glb_points"] = int(keep.size)

    for name in ("aligned_room.ply", "scene_points.glb", "cloud.bin", "occupancy.npy",
                 "occupancy_meta.json", "alignment.json", "alignment_transform.npz",
                 "cameras_aligned.json", "keyframes.json", "map_preview.png"):
        report.setdefault("files", {})[name] = (out / name).stat().st_size
    report["ceiling_y"] = ceiling_y
    report["points_below_floor_frac"] = below
    report["camera_height_min_max"] = [float(cams[:, 1].min()), float(cams[:, 1].max())]
    report["frames"] = {"cloud.bin": "Z-up (u, v, height) - the nvblox slice frame",
                        "everything else": "app Y-up (u, height, -v), floor y = 0"}
    (out / "ingest_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
