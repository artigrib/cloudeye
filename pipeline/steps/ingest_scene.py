#!/usr/bin/env python3
"""INGEST: the artifacts `app/services/pipeline_orchestrator.py` reads, from what the driver made.

The driver produces a complete scene — a fused mesh, a band layer, reachability, a USD — and
the app could not read any of it. `ingest_scene_directory` wants a different set of files at
the scene-dir root, written by the retired `gpu/stage_*.py` pipeline, and died on the first
one missing. This closes that gap for the files that are genuinely read, and ONLY those:
`docs/DECISIONS.md` has the inventory and the grep behind each call.

Nothing here is invented. Every number is measured from this scene's own fused mesh, its own
fitted floor plane, or its own camera poses. Where the old pipeline recorded something this
one cannot know — `floor_support`, `chosen_band`, `frac_cameras_between`, the `conf_*` fields,
all of them products of `gpu/floor_ceiling.py` — the key is OMITTED rather than filled with a
plausible-looking default. `read_alignment` already treats those as optional; a stub with
invented numbers would read as measurement to everyone downstream, forever.

FRAMES, because they are the one thing that can be silently wrong here. The driver works in
the scene's own fitted floor frame `(u, v, height)`: rows `[axis_u, axis_v, normal_up]` from
`floor_plane.json`, origin at `point_on_plane`. That is the frame `layers/<name>/occupancy.npy`
is rasterised in, so `occupancy_meta.json`'s `origin_x`/`origin_z` are `u_range[0]`/`v_range[0]`.
The app is Y-up with the floor at y=0 and indexes the grid by `(x, z)`. So the mapping is
exactly `(x, y, z) = (u, height, v)` and it must be applied to cameras and bboxes alike — get
it wrong and the camera track lands off-grid, `world_to_cell` raises, and `/reachability`
answers 500 for the whole scene.

    ingest_scene.py --scene-dir <dir> --nvblox-dir <dir> --layers-dir <dir>
                    --packed <dir> --export-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def floor_basis(fp: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """(R, t, was_reflection) mapping world -> (u, v, height), the driver's own floor frame.

    Identical construction to `steps/isaac_export.py::floor_transform`, deliberately: the
    cameras this writes and the mesh the exporter writes have to end up in the same frame,
    and two copies of a rotation that must agree is how they stop agreeing.
    """
    au = np.asarray(fp["axis_u"], float)
    av = np.asarray(fp["axis_v"], float)
    up = np.asarray(fp["normal_up"], float)
    R = np.stack([au, av, up])
    reflection = bool(np.linalg.det(R) < 0)
    if reflection:                                   # keep it a right-handed rotation
        R[1] = -R[1]
    t = -R @ np.asarray(fp["point_on_plane"], float)
    return R, t, reflection


def to_app_frame(pts_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """world -> the app's GRID frame, (u, height, v), floor at y=0.

    This is the frame the BACKEND indexes: `occupancy_meta.json`'s `origin_x`/`origin_z` are
    `u_range[0]`/`v_range[0]`, so a camera's (x, z) has to be its (u, v) or `world_to_cell`
    lands in the wrong cell. Used for `cameras_aligned.json` and `alignment.json`.

    It is NOT the frame the viewer renders in - see `to_viewer_frame`. The two genuinely
    differ, and conflating them is what mirrored the first mesh that reached the screen.
    """
    f = (R @ pts_world.T).T + t                      # (u, v, height)
    return np.stack([f[:, 0], f[:, 2], f[:, 1]], axis=1)


#: (u, v, height) -> (u, height, -v). The viewer's own `zUpToYUp` (frontend/src/lib/
#: cloudBin.ts), applied to `cloud.bin` on the worker thread. Determinant +1: a PROPER
#: rotation, which is the whole point - `(u, h, +v)` is the same footprint MIRRORED about z,
#: and a bounding box cannot tell the two apart. Measured on both scenes against the cloud
#: the viewer actually draws: this candidate scores median 0.0000 m / p90 0.0021-0.0076 m
#: nearest-neighbour, the three alternatives 0.166-0.362 m median.
VIEWER_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def to_viewer_frame(pts_uvh: np.ndarray) -> np.ndarray:
    """floor frame (u, v, height) -> the frame the 3D viewer renders, (u, height, -v)."""
    return (VIEWER_R @ np.asarray(pts_uvh).T).T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene-dir", required=True, help="the app scene dir; artifacts land here")
    ap.add_argument("--nvblox-dir", required=True, help="holds mesh.ply and floor_plane.json")
    ap.add_argument("--layers-dir", required=True, help="the shipped band layer")
    ap.add_argument("--packed", required=True, help="holds meta.json with the camera poses")
    ap.add_argument("--export-dir", required=True, help="holds scene.glb")
    ap.add_argument("--mesh-name", default="mesh.ply")
    a = ap.parse_args()

    scene_dir = Path(a.scene_dir)
    layers = Path(a.layers_dir)
    fp_path = layers / "floor_plane.json"
    if not fp_path.is_file():
        fp_path = Path(a.nvblox_dir) / "floor_plane.json"
    fp = json.loads(fp_path.read_text())
    R, t, reflection = floor_basis(fp)

    written: dict[str, str] = {}

    # --- 1. objects: none, and that is a fact about this build, not a gap ----------------
    # SEMANTICS is a declared slot with no implementation and there is no SAM3 stage, so the
    # honest object list is empty. An empty array is legal (schema_version 2+ wants an array)
    # and passes validation: every object check in scene_validator self-skips with no
    # eligible objects, leaving `room_dimensions` as the only fatal one.
    obj_dir = scene_dir / "scene_objects"
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "scene_objects.json").write_text("[]\n", encoding="utf-8")
    written["scene_objects/scene_objects.json"] = "0 objects (no SAM3 stage in this build)"

    # --- 1b. the grid geometry, at the root, because a SECOND app reads it there ---------
    # `GET /api/scenes/<id>/layers` (scene-screen-take3, routers/scenes.py:815) calls
    # read_grid_metadata on the SCENE ROOT, and 500s on a missing file. That endpoint does
    # not exist on pipeline-v1, which is why grepping pipeline-v1's app/ said the root copy
    # had no reader outside ingest - a true answer about the wrong branch. :8010 serves
    # take3, and :5173 proxies to :8010, so take3's app/ is the contract the scene page
    # actually has.
    #
    # These are the layer's own five measured numbers, not a synthetic grid, and `source`
    # names the directory they came from so the two can be compared rather than trusted.
    # Only the METADATA goes here: /layers reads nothing else from the root, so root
    # occupancy.npy stays absent and layers/<name>/ remains the single planner truth.
    lmeta = json.loads((layers / "occupancy_meta.json").read_text())
    (scene_dir / "occupancy_meta.json").write_text(json.dumps({
        "resolution": lmeta["resolution"], "origin_x": lmeta["origin_x"],
        "origin_z": lmeta["origin_z"], "width": lmeta["width"], "height": lmeta["height"],
        "source": f"copied from {layers}/occupancy_meta.json by ingest_scene.py",
        "note": "geometry only. The planner reads layers/<name>/, never this file; this "
                "exists because GET /scenes/<id>/layers reads grid metadata at the root.",
    }, indent=2), encoding="utf-8")
    written["occupancy_meta.json"] = (f"{lmeta['width']}x{lmeta['height']} @ "
                                      f"{lmeta['resolution']} m, from the layer")

    # --- 2. the camera track ------------------------------------------------------------
    # Required, and more load-bearing than it looks: GET /camera-track calls read_camera_track
    # UNGUARDED (a missing file is a 500, not a 404), and /reachability anchors each robot's
    # per-radius start on cameras[0] rather than the persisted robot_start, because the
    # persisted one already has one radius's snapping baked into it.
    meta = json.loads((Path(a.packed) / "meta.json").read_text())
    C = np.array([np.asarray(v["camera_pose"], float)[:3, 3] for v in meta["views"]])
    Capp = to_app_frame(C, R, t)
    (scene_dir / "cameras_aligned.json").write_text(
        json.dumps({"cameras": [[float(x) for x in row] for row in Capp]}, indent=1),
        encoding="utf-8")
    written["cameras_aligned.json"] = f"{len(Capp)} real camera poses"

    # --- 3. alignment: measured fields only ---------------------------------------------
    import open3d as o3d                                     # only this section needs it
    mesh = o3d.io.read_triangle_mesh(str(Path(a.nvblox_dir) / a.mesh_name))
    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        raise SystemExit(f"{Path(a.nvblox_dir) / a.mesh_name}: 0 vertices; nothing to measure")
    Vapp = to_app_frame(V, R, t)
    lo, hi = Vapp.min(0), Vapp.max(0)

    # The floor is at y=0 BY CONSTRUCTION - the frame is built from the plane itself - and
    # ingest hardcodes floor_y=0.0 anyway. `residual_tilt_deg` is then the numerical residual
    # of that construction, computed rather than assumed to be zero, so a basis that ever
    # stops being orthonormal shows up as a number instead of silence.
    up_after = R @ np.asarray(fp["normal_up"], float)
    tilt = float(np.degrees(np.arccos(np.clip(
        up_after @ np.array([0.0, 0.0, 1.0]) / np.linalg.norm(up_after), -1.0, 1.0))))

    alignment = {
        "floor_y": 0.0,
        "ceiling_y": float(hi[1]),
        "bbox_min": [float(x) for x in lo],
        "bbox_max": [float(x) for x in hi],
        "residual_tilt_deg": tilt,
        "det_r": float(np.linalg.det(R)),
        "is_reflection": reflection,
        "n_views": int(meta["n_views"]),
        "source": "pipeline/steps/ingest_scene.py from this scene's fused mesh + fitted floor",
        "omitted_fields_note": (
            "floor_support, ceiling_support, chosen_band, frac_cameras_between, "
            "points_below_floor*, conf_* are products of the retired gpu/floor_ceiling.py "
            "stage. This pipeline cannot measure them, so they are omitted rather than "
            "stubbed - read_alignment treats every one of them as optional."),
    }
    (scene_dir / "alignment.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    written["alignment.json"] = (f"ceiling_y {alignment['ceiling_y']:.3f} m, "
                                f"tilt {tilt:.2e} deg, {len(V)} mesh vertices")

    # --- 4. vocabulary: the SEMANTICS stub, said in the file the app reads ---------------
    # `source` is deliberately not one of the real provider ids. A reader that switches on
    # "openrouter"/"vertex"/"override" must not silently match this, and Scene.vocab_source
    # ends up echoing the truth to the API instead of a provider that never ran.
    (scene_dir / "vocab.json").write_text(json.dumps({
        "source": "not_implemented",
        "objects": [],
        "note": "SEMANTICS is a declared slot with no implementation in this build; no "
                "vocabulary provider was called, so there is no provider_used to report",
    }, indent=2), encoding="utf-8")
    written["vocab.json"] = "SEMANTICS stub (no provider called)"

    # --- 5. the 3D view, which wants POINTS and will not say so -------------------------
    # The viewer asks for `cloud.bin` FIRST and falls back to `/mesh` on a 404
    # (`useSceneGlb(sceneId, meshUrl, enabled, cloudUrl)`). Both paths end in
    # `scene3dPoints.buildDecimatedPointCloud`, which calls `findPointsObject` and returns
    # null when the GLB has no `.isPoints` node - and null renders NOTHING, with no error,
    # because the file loaded perfectly well. Copying EXPORT's triangle-mesh scene.glb to
    # `scene_points.glb` therefore produced a 200 that drew nothing at all: 2D correct, 3D
    # showing only the clearance plane. The comment above findPointsObject anticipates
    # exactly this - "if a triangulated mesh ever gets exported instead".
    #
    # So: write the binary cloud the viewer actually prefers, and make the GLB fallback a
    # real points primitive rather than a mesh wearing a points filename.
    colours = np.asarray(mesh.vertex_colors)
    has_rgb = len(colours) == len(V) and len(V) > 0
    rgb = (np.clip(colours, 0, 1) * 255).astype(np.uint8) if has_rgb else None

    # CEPC v1 (frontend/src/lib/cloudBin.ts), little-endian:
    #   "CEPC" | uint32 version=1 | uint32 count | uint32 flags | float32[6] bbox | xyz | rgb
    # Coordinates are the nvblox SLICE frame - (u, v, height), Z-up, floor at 0 - NOT the
    # app's Y-up frame used for cameras above. The worker applies (x, y, z) = (u, h, -v)
    # itself; handing it pre-rotated data would mirror the scene, which is the very thing
    # `alignment.json.is_reflection` exists to catch.
    Vz = (R @ V.T).T + t                                   # (u, v, height)
    bbox = np.concatenate([Vz.min(0), Vz.max(0)]).astype(np.float32)
    blob = bytearray()
    blob += b"CEPC"
    blob += struct.pack("<III", 1, len(Vz), 1 if has_rgb else 0)
    blob += bbox.tobytes()
    blob += np.ascontiguousarray(Vz, dtype=np.float32).tobytes()
    if has_rgb:
        blob += np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    (scene_dir / "cloud.bin").write_bytes(bytes(blob))
    written["cloud.bin"] = (f"CEPC v1, {len(Vz)} points, rgb={has_rgb}, "
                            f"{len(blob)} B, frame (u,v,height) Z-up")

    # The GLB fallback, as POINTS. `mesh_path` is written blind at this exact name, so the
    # file has to be here; it is now what the name has always claimed it is.
    import trimesh
    # THE CONTRACT (owner, 2026-09-10): /mesh serves the nvblox TRIANGLE mesh in the app
    # frame; /cloud serves points and only points. `mesh_path` is written blind at
    # `scene_points.glb`, so that filename is what /mesh serves and the name is now a
    # historical artefact of the old points-GLB pipeline rather than a description.
    #
    # Pre-rotated into the render frame, because the GLB path is NOT rotated by the viewer -
    # only cloud.bin is. It was briefly written in to_app_frame's (u, h, +v), which is the
    # same footprint MIRRORED about z; measured against the cloud the viewer draws, that
    # scored 0.166-0.321 m nearest-neighbour where (u, h, -v) scores 0.0000 m.
    import trimesh
    sc_src = trimesh.load(str(Path(a.export_dir) / "scene.glb"), force="scene")
    tri_parts = []
    for geom in sc_src.geometry.values():
        g = geom.copy()
        g.vertices = to_viewer_frame(np.asarray(g.vertices))     # det +1: winding preserved
        tri_parts.append(g)
    tri = trimesh.util.concatenate(tri_parts) if len(tri_parts) > 1 else tri_parts[0]
    (scene_dir / "scene_points.glb").write_bytes(
        trimesh.Scene({"nvblox_visual": tri}).export(file_type="glb"))
    written["scene_points.glb"] = (
        f"nvblox TRIANGLE mesh in app frame, {len(tri.faces)} faces, node 'nvblox_visual', "
        f"{(scene_dir / 'scene_points.glb').stat().st_size} B - this is what /mesh serves")

    # --- 6. msa/: NOT written -----------------------------------------------------------
    # `/msa-glb` is the MSA (Measured Scene Assembly) layout - furniture and wall stubs from
    # scripts/msa/export_*.py, meant to be overlaid ON TOP of this scene's point cloud. A
    # driver scene has no MSA layout, so the honest answer is the 404 the route already gives
    # ("most scenes won't", per its own docstring).
    #
    # This briefly wrote the nvblox mesh there so the viewer's mesh checkbox had something to
    # show. That was wrong twice over: it put reconstruction geometry behind a control that
    # means "show the ASSEMBLY", and `X-Msa-Glb-Variant` then reported `scene.glb` for a file
    # no MSA script produced. The nvblox mesh belongs at /mesh, which is where it now is.
    written["msa/scene.glb"] = "not written - a driver scene has no MSA layout; /msa-glb 404s"

    doc = {"scene_dir": str(scene_dir), "written": written,
           "layer_dir": str(layers),
           "frame": "app (x, y, z) = floor-frame (u, height, v); floor at y=0"}
    (scene_dir / "ingest.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for k, v in written.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
