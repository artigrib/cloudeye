#!/usr/bin/env python3
"""Step 10: nvblox meshes -> Isaac-Sim-ready GLB + USD, one directory per scene.

Pipeline per scene:
  1. rotate into a Z-up frame whose floor is z=0, using that scene's own fitted floor
     (`floor_plane.json`): rows [axis_u, axis_v, normal_up] map world -> (u, v, height),
     so the transform is exactly the frame the ESDF slice already lives in;
  2. quadric-decimate to at most --max-tri triangles (skipped if already under);
  3. order: components below --min-component-tri triangles are dropped FIRST, then the
     mesh is decimated. The reverse (decimate, then filter) was measured and rejected:
     quadric decimation with a global budget shreds large surfaces into sub-100-triangle
     pieces, so the filter afterwards deletes most of the budget it just allocated -
     own_0828_221137 came out at 35 915 triangles instead of 199 990, a 0.6 MB "room";
  4. write `scene.glb` (trimesh) and `scene.usd` (pxr), the latter with
     metersPerUnit = 1, Z up-axis, and UsdPhysics.CollisionAPI +
     MeshCollisionAPI(approximation = "meshSimplification") on the mesh prim;
  5. write `transform.json` with the 4x4 actually applied and the before/after counts;
  6. validate the stage.

Note on validation: the `usd-core` wheel does not ship the `usdchecker` executable. The
checks are run through the same `pxr.UsdValidation` framework that modern usdchecker
drives, over all registered validators, and the per-file result is recorded.
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdValidation


def floor_transform(fp: dict) -> np.ndarray:
    """4x4 mapping world -> (u, v, height above floor), i.e. Z-up with the floor at z=0."""
    au = np.asarray(fp["axis_u"], float); av = np.asarray(fp["axis_v"], float)
    up = np.asarray(fp["normal_up"], float)
    R = np.stack([au, av, up])                      # rows
    if np.linalg.det(R) < 0:                        # keep it a right-handed rotation
        R[1] = -R[1]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.asarray(fp["point_on_plane"], float)
    return T


def drop_small_components(m: o3d.geometry.TriangleMesh, min_tri: int):
    if min_tri <= 0 or len(m.triangles) == 0:
        return m, 0
    labels, counts, _ = m.cluster_connected_triangles()
    counts = np.asarray(counts); lab = np.asarray(labels)
    keep = (lab >= 0) & (counts[lab] >= min_tri)
    removed = int((~keep).sum())
    m.remove_triangles_by_mask(~keep)
    m.remove_unreferenced_vertices()
    return m, removed


def usd_prim_name(scene_name: str) -> str:
    """A scene name turned into a legal USD prim name, losslessly.

    USD identifiers are `[A-Za-z_][A-Za-z0-9_]*`. Every scene this script had ever been run
    on was named `own_0901_161054__optimal_step2`, which already satisfies that, so the raw
    name went straight into the path. The worker names a scene by its SCENE UUID -
    `1be86e2f-9247-4c74-bc5b-28e00e41b75c` - which starts with a digit and is full of
    hyphens, and `UsdGeom.Mesh.Define` threw `Ill-formed SdfPath ... at character 8 ('1'):
    expected prim name`.

    `Tf.MakeValidIdentifier` is the obvious tool and is wrong here: it REPLACES the leading
    digit rather than prefixing, so `1be86e2f...` becomes `_be86e2f...` and a hypothetical
    `2be86e2f...` collapses onto the same name. A scene identifier that can collide with
    another scene is worse than one that is ugly. Prefixing keeps every character.

    Names that were already valid are returned untouched, so the published `own_*` exports
    keep the prim names they have and nothing downstream has to be re-pointed.
    """
    ident = re.sub(r"[^A-Za-z0-9_]", "_", scene_name)
    if not ident or not (ident[0].isalpha() or ident[0] == "_"):
        ident = "scene_" + ident
    return ident


def write_usd(path: Path, verts: np.ndarray, tris: np.ndarray, colors: np.ndarray | None,
              scene_name: str) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"/World/{usd_prim_name(scene_name)}")
    # Gf.Vec3f rejects numpy scalars, so hand it plain Python floats
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in verts])
    mesh.CreateFaceVertexIndicesAttr(tris.reshape(-1).astype(np.int32).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = verts.min(0), verts.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if colors is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in colors])
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mc.CreateApproximationAttr(UsdPhysics.Tokens.meshSimplification)
    stage.GetRootLayer().Save()


def validate(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path))
    reg = UsdValidation.ValidationRegistry()
    validators = reg.GetOrLoadAllValidators()
    ctx = UsdValidation.ValidationContext(validators)
    errors = ctx.Validate(stage)
    msgs = [e.GetMessage() for e in errors]
    return {"n_validators": len(validators), "n_errors": len(msgs), "errors": msgs[:20]}


def export_scene(scene: str, src: Path, out: Path, args) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    fp = json.loads((src / "floor_plane.json").read_text())
    mesh_file = src / args.mesh_name
    m = o3d.io.read_triangle_mesh(str(mesh_file))
    n_tri_in, n_vert_in = len(m.triangles), len(m.vertices)
    # `has_color` used to mean "the mesh has an RGB property", which is true of every mesh
    # nvblox writes - update_color_mesh()/get_color_mesh() run whether or not a single
    # colour frame was integrated, and an untouched colour layer reads back as 127/127/127
    # on every vertex. So the flag was True on a flat-grey mesh, commit b2457e0 read it as
    # "the live mesh is already coloured", and seven of the eight published USDs shipped
    # grey. Presence is not the question; VARIANCE is, and it is one np.unique away.
    n_unique_colors = (int(np.unique(np.asarray(m.vertex_colors), axis=0).shape[0])
                       if len(m.vertex_colors) == n_vert_in and n_vert_in > 0 else 0)
    has_color = n_unique_colors > 1

    T = floor_transform(fp)
    m.transform(T)

    n_tri_rot = len(m.triangles)
    if args.order == "decimate_then_filter":
        if n_tri_rot > args.max_tri:
            m = m.simplify_quadric_decimation(target_number_of_triangles=args.max_tri)
        n_tri_dec = len(m.triangles)
        m, removed = drop_small_components(m, args.min_component_tri)
    else:                                    # filter_then_decimate
        m, removed = drop_small_components(m, args.min_component_tri)
        if len(m.triangles) > args.max_tri:
            m = m.simplify_quadric_decimation(target_number_of_triangles=args.max_tri)
        n_tri_dec = len(m.triangles)
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    m.compute_vertex_normals()

    verts = np.asarray(m.vertices); tris = np.asarray(m.triangles)
    cols = np.asarray(m.vertex_colors) if len(m.vertex_colors) == len(verts) and len(verts) else None

    import trimesh
    tm = trimesh.Trimesh(vertices=verts, faces=tris, process=False,
                         vertex_colors=(cols * 255).astype(np.uint8) if cols is not None else None)
    glb = out / "scene.glb"; tm.export(str(glb))
    usd = out / "scene.usd"
    if usd.exists():
        usd.unlink()
    write_usd(usd, verts, tris, cols, scene.split("__")[0])
    val = validate(usd)

    lo, hi = (verts.min(0), verts.max(0)) if len(verts) else (np.zeros(3), np.zeros(3))
    info = {
        "scene": scene, "source_mesh": str(mesh_file), "source_has_vertex_colors": has_color,
        # The prim the mesh actually landed under, which is not always the scene name -
        # a uuid-named scene gets a `scene_` prefix to be a legal USD identifier.
        "usd_prim_name": usd_prim_name(scene.split("__")[0]),
        # The number, not just the verdict: 1 is the grey signature, and a reader a month
        # from now should be able to tell 1 from 84,240 without re-opening the mesh.
        "source_unique_vertex_colors": n_unique_colors,
        "usd_unique_display_colors": (int(np.unique(cols, axis=0).shape[0])
                                      if cols is not None and len(cols) else 0),
        "transform_world_to_zup_floor0": T.tolist(),
        "floor_normal_up_world": fp["normal_up"], "floor_point_world": fp["point_on_plane"],
        "triangles_in": n_tri_in, "triangles_after_decimation": n_tri_dec,
        "triangles_out": int(len(tris)), "triangles_removed_as_small_components": removed,
        "vertices_out": int(len(verts)),
        "bbox_min_zup": [round(float(x), 3) for x in lo],
        "bbox_max_zup": [round(float(x), 3) for x in hi],
        "floor_z_check": {"p1": round(float(np.percentile(verts[:, 2], 1)), 3),
                          "median": round(float(np.median(verts[:, 2])), 3)} if len(verts) else None,
        "glb_bytes": glb.stat().st_size, "usd_bytes": usd.stat().st_size,
        "usd_validation": val, "wall_s": round(time.time() - t0, 1),
        "max_tri": args.max_tri, "min_component_tri": args.min_component_tri,
    }
    info["order"] = args.order
    (out / "transform.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--mesh-name", default="mesh.ply")
    ap.add_argument("--max-tri", type=int, default=200000)
    ap.add_argument("--min-component-tri", type=int, default=100)
    ap.add_argument("--order", choices=["filter_then_decimate", "decimate_then_filter"],
                    default="filter_then_decimate",
                    help="filter_then_decimate is the shipped order. The reverse loses most of "
                         "the triangle budget: quadric decimation shreds large surfaces into "
                         "sub-100-triangle pieces, which the component filter then deletes "
                         "(measured: 1.15M -> 36k triangles on own_0828_221137).")
    a = ap.parse_args()
    root, outroot = Path(a.scenes_root), Path(a.out_root)
    rows = []
    for s in a.scenes:
        print(f"=== {s} ===", flush=True)
        info = export_scene(s, root / s, outroot / s.replace("__optimal_step2", "").replace("__step2", ""), a)
        rows.append(info)
        print(json.dumps({k: info[k] for k in ["triangles_in", "triangles_out", "glb_bytes",
                                               "usd_bytes", "usd_validation", "floor_z_check"]},
                         indent=1), flush=True)
    (outroot / "export_summary.json").write_text(json.dumps(rows, indent=2))
    print("WROTE", outroot / "export_summary.json")


if __name__ == "__main__":
    main()
