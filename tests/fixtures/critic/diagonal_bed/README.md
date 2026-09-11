# Fixture: the 13d diagonal-bed regression

A frozen snapshot of the scene that produced `progress/13d_web_34.png` /
`progress/13d_web_top.png` (`var/scratch/run-20260906/`, 2026-09-07) - the
bed-orientation bug tracked as QUEUE.md M12 ("bed orientation bug").

**Root cause (per the geo-6 fix): the asset fitter squared only the mesh's
AABB while the TRELLIS-generated bed asset sits −39.9° inside its own
bounding box.** The collision hull (`objects.json`'s `angle_rad`,
`assets_placed.json`'s `yaw_deg`, both computed from the room-aligned hull
geometry) came out numerically correct either way - only the *placed visual
mesh* baked into the exported GLB is rotated off. That is exactly why
`scripts/audit/critic_rules.py`'s yaw check reads the GLB directly instead of
trusting `yaw_deg`/`angle_rad` alone - see that module's `_mesh_yaw_from_glb`
docstring for the full story.

## Do not regenerate this fixture from the current pipeline

**This is a deliberate snapshot of a state that is already fixed on the geo-6
branch.** Re-running bootstrap/assemble/export against the same source scene
today will NOT reproduce the bug (the bed comes out ~1 deg off the wall, not
~155 deg) - regenerating this fixture from a fresh pipeline run would silently
delete the regression coverage `tests/test_audit_critic_rules.py`'s
`TestDiagonalBedFixtureRegression` depends on. If this fixture ever needs to
change, it must be a deliberate, reviewed edit, not an automated regen.

## What's here and where it came from

All four JSON files are byte-for-byte copies of the real
`var/scratch/run-20260906/geo5_out/` files that fed `13d_web_34.png` (same
build, see `geo5_out/web_render.py`'s docstring/URL, which points at
`geo5_assets_textured.glb` - the file this GLB was trimmed from, at the
timestamp - `scene_assets_textured.glb` mtime 13:06:44, `web_render.py` run
~13:08 - that produced the screenshot):

- `objects.json` - all 15 objects, unmodified, real `room_polygon`/`hull_xz`/
  `angle_rad`/`size_uv` data. `bed_0.angle_rad` = 0.0191 rad (1.09 deg) -
  matches the wall direction, i.e. the *hull* was never wrong.
- `scene_meta.json` - unmodified (yaw-correction fields etc., not read by
  `critic_rules.py` today but kept for fidelity/future use).
- `assets_placed.json` - unmodified. `bed_0.yaw_deg` = 1.0920 - same story as
  `angle_rad` above, the recorded metadata is "correct".
- `report.json` - unmodified (support_keeps/support_drops log).

## What was trimmed, and why

`scene_assets_textured.glb` in the real out dir is 51 MB (two bed halves at
~172k vertices each, TRELLIS-textured) - far too large to check into the repo.
`scene_assets_textured.glb` in *this* fixture directory is a 20 KB stand-in
containing **only `bed_0`'s three GLB nodes**
(`bed_0/visual/part_0`, `bed_0/visual/part_1`, `bed_0/collision/hull`), each
replaced by its own 3D **convex hull** (234/234/40 vertices) at the same
already-baked world-space position the original mesh occupied (node
transforms in the real export are identity - vertex positions ARE world
positions, confirmed by inspecting the real GLB before trimming).

This is a lossless trim for the one thing this fixture exists to test:
`_mesh_yaw_from_glb` fits a **minimum-rotated-rectangle** over each node's
world-space XZ vertex projection, and a minimum-rotated-rectangle over a
point set depends only on that point set's convex hull - replacing a mesh
with its own convex hull cannot change the measured angle. Verified directly
(`_mesh_yaw_from_glb` on the real 51 MB file vs. this trimmed one both report
154.967... deg for `bed_0`'s visual mesh, vs. 1.092 deg for its collision
hull - a 25.03 deg mod-90 delta against the room's 0 deg wall direction,
past the 15 deg threshold).

No other object's GLB nodes are included - every other object in
`objects.json` still resolves through `check_yaw_vs_wall`'s metadata fallback
(`assets_placed.yaw_deg` / `objects.angle_rad`) since it can't find GLB nodes
for them, which does not raise and does not fabricate a finding.

## Regenerating (if ever explicitly needed)

The exact commands used to build the GLB (informational only - **do not run
this against a fresh pipeline export**, only against a preserved copy of the
original buggy `geo5_out/scene_assets_textured.glb` if one still exists):

```python
import trimesh, numpy as np
src = trimesh.load("<path to the original buggy scene_assets_textured.glb>", process=False)
g = src.graph
out = trimesh.Scene()
for src_node, dst_node in {
    "bed_0/visual/part_0_0": "bed_0/visual/part_0",
    "bed_0/visual/part_1_0": "bed_0/visual/part_1",
    "bed_0/collision/hull": "bed_0/collision/hull",
}.items():
    transform, geom_name = g.get(src_node)
    mesh = src.geometry[geom_name]
    verts_world = trimesh.transformations.transform_points(mesh.vertices, transform)
    hull = trimesh.Trimesh(vertices=verts_world, faces=mesh.faces, process=False).convex_hull
    out.add_geometry(hull, node_name=dst_node, transform=np.eye(4))
out.export("tests/fixtures/critic/diagonal_bed/scene_assets_textured.glb")
```
