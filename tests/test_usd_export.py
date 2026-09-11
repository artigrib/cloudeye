"""usd_export builds an Isaac Sim-ready stage from occupancy-grid data plus, per object,
either the convex hull of its own captured point cloud or (as a fallback) its bbox.

Tested against the real `scene_horizontal` fixture (like test_scene_ingest.py), plus
synthetic small grids for the box-merge algorithm itself where exact cell layout
matters more than realism. The fixture's objects all reference `.ply` files that
aren't checked into the repo (see conftest.py's `scene_horizontal_dir` docstring), so
building a stage from it always exercises the bbox-fallback path for every object -
real point clouds for the convex-hull path are instead synthesized here, the same way
test_ply_reader.py synthesizes its own `.ply` fixtures rather than checking in real
point-cloud binaries.
"""

import struct

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull

from app.services import scene_ingest, usd_export
from app.services.usd_export import UsdExportInput, UsdExportObject, merge_obstacle_boxes

FREE, OBSTACLE, UNKNOWN = 0, 1, 2


def _write_binary_ply(path, positions: np.ndarray) -> None:
    """Same minimal writer as test_ply_reader.py's, positions-only (no color - unused
    by build_convex_hull)."""
    n = len(positions)
    with open(path, "wb") as f:
        f.writelines([
            b"ply\n", b"format binary_little_endian 1.0\n", f"element vertex {n}\n".encode(),
            b"property double x\n", b"property double y\n", b"property double z\n",
            b"end_header\n",
        ])
        for i in range(n):
            f.write(struct.pack("<ddd", *positions[i]))


# --- merge_obstacle_boxes -------------------------------------------------------------


def test_merge_single_rectangle_becomes_one_box():
    cells = np.zeros((6, 6), dtype=np.uint8)
    cells[1:4, 1:4] = OBSTACLE
    boxes = merge_obstacle_boxes(cells)
    assert boxes == [(1, 1, 4, 4)]


def test_merge_covers_every_obstacle_cell_exactly_once():
    rng = np.random.default_rng(42)
    cells = (rng.random((30, 30)) < 0.3).astype(np.uint8)  # random OBSTACLE/FREE mix
    boxes = merge_obstacle_boxes(cells)

    covered = np.zeros_like(cells, dtype=bool)
    for ix0, iz0, ix1, iz1 in boxes:
        # no box may overlap another - the greedy algorithm marks cells visited as it
        # claims them, so any overlap would mean it re-claimed a cell
        assert not covered[ix0:ix1, iz0:iz1].any()
        covered[ix0:ix1, iz0:iz1] = True

    assert np.array_equal(covered, cells == OBSTACLE)


def test_merge_unknown_cells_are_not_boxed():
    cells = np.full((4, 4), UNKNOWN, dtype=np.uint8)
    assert merge_obstacle_boxes(cells) == []


def test_merge_reduces_box_count_vs_naive_one_per_cell():
    """The whole point of merging: a large solid obstacle block must not produce one
    box per cell."""
    cells = np.zeros((40, 40), dtype=np.uint8)
    cells[5:35, 5:35] = OBSTACLE  # a single big solid wall block, 900 cells
    boxes = merge_obstacle_boxes(cells)
    assert len(boxes) < 10  # a solid block merges into a handful of boxes, not 900


# --- to_isaac axis convention ----------------------------------------------------------


def test_to_isaac_is_a_proper_rotation_not_a_reflection():
    """to_isaac must have determinant +1 (a proper rotation). An earlier version used
    the naive axis swap (x, z, y), determinant -1 - a reflection that mirrored the
    whole exported scene. Build the 3x3 matrix from to_isaac's action on the standard
    basis vectors and check its determinant directly, rather than trusting the
    docstring's claim."""
    columns = [
        list(usd_export.to_isaac(1, 0, 0)),
        list(usd_export.to_isaac(0, 1, 0)),
        list(usd_export.to_isaac(0, 0, 1)),
    ]
    # 3x3 determinant via the standard cofactor expansion - avoids pulling in numpy
    # just for this, and keeps the check legible as a literal formula.
    (a, b, c), (d, e, f), (g, h, i) = columns
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    assert det == pytest.approx(1.0)


def test_to_isaac_preserves_left_right_handedness():
    """Regression guard for the mirroring bug, not a tautology: encode a source-frame
    (Y-up) observer facing +Z with up +Y, and an object at (1, 0, 2) - which is, by
    construction, on that observer's LEFT (left = up x forward = (1,0,0), so the object
    at x=+1 has a positive dot product with "left"). Map observer and object through
    to_isaac into Isaac's frame, recompute "left" there the same way, and check the
    object is still on the left.

    The old reflection (x, z, y) flips this sign (verified by hand) - an object on the
    left would come out measured as being on the right, i.e. the scene is mirrored.
    The new rotation (x, -z, y) preserves it, which is exactly the property a proper
    rotation must have and a reflection cannot."""

    def cross(u, v):
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    def dot(u, v):
        return sum(a * b for a, b in zip(u, v))

    forward_source, up_source, object_source = (0, 0, 1), (0, 1, 0), (1, 0, 2)
    left_source = cross(up_source, forward_source)
    assert dot(object_source, left_source) > 0  # sanity: the fixture is on the left

    forward_isaac = tuple(usd_export.to_isaac(*forward_source))
    up_isaac = tuple(usd_export.to_isaac(*up_source))
    object_isaac = tuple(usd_export.to_isaac(*object_source))
    left_isaac = cross(up_isaac, forward_isaac)

    assert dot(object_isaac, left_isaac) > 0  # still on the left after export


# --- Full stage build against the real fixture -----------------------------------------


@pytest.fixture
def fixture_export_input(scene_horizontal_dir) -> UsdExportInput:
    grid_meta = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    alignment = scene_ingest.read_alignment(scene_horizontal_dir)
    parsed = scene_ingest.parse_scene_objects(
        scene_horizontal_dir / "scene_objects" / "scene_objects.json", scene_horizontal_dir
    )
    cells = np.load(scene_horizontal_dir / "occupancy.npy")

    objects = [
        UsdExportObject(
            name=p.name,
            bbox_min=p.bbox_min,
            bbox_max=p.bbox_max,
            num_views=p.num_views,
            num_points=p.num_points,
            is_fragment=p.is_fragment,
            mesh_path=p.mesh_path,
        )
        for p in parsed
    ]
    return UsdExportInput(
        grid_cells=cells,
        grid_meta=grid_meta,
        ceiling_y=alignment.ceiling_y,
        robot_start=(0.3, -0.5),
        is_reflected=alignment.is_reflection,
        robot_radius_m=0.10,
        objects=objects,
        pointcloud_path=None,
        include_fragments=True,
    )


def test_stage_metadata(fixture_export_input):
    stage, _ = usd_export.build_stage(fixture_export_input)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath() == "/World"

    layer_data = stage.GetRootLayer().customLayerData
    assert layer_data["cloudeye:robotRadiusM"] == pytest.approx(0.10)
    assert layer_data["cloudeye:isReflected"] == fixture_export_input.is_reflected


def test_floor_is_static_collider_sized_to_grid(fixture_export_input):
    stage, _ = usd_export.build_stage(fixture_export_input)
    floor = stage.GetPrimAtPath("/World/Floor")
    assert floor.IsValid()
    assert floor.HasAPI(UsdPhysics.CollisionAPI)
    assert not floor.HasAPI(UsdPhysics.RigidBodyAPI)  # static, per spec

    meta = fixture_export_input.grid_meta
    scale = UsdGeom.Xformable(floor).GetOrderedXformOps()[1].Get()  # [translate, scale]
    assert scale[0] == pytest.approx(meta.width * meta.resolution)
    assert scale[1] == pytest.approx(meta.height * meta.resolution)


def test_floor_slab_is_a_thicker_static_collider_flush_below_the_floor(fixture_export_input):
    """isaac_validate.py's sphere_drop check tunnels through the 2cm-thick visible
    floor at its default physics_dt - this slab is a collision-only safety margin
    directly beneath it, same XY footprint, FLOOR_SLAB_THICKNESS_M thick."""
    stage, _ = usd_export.build_stage(fixture_export_input)
    floor = stage.GetPrimAtPath("/World/Floor")
    slab = stage.GetPrimAtPath("/World/FloorSlab")
    assert slab.IsValid()
    assert slab.HasAPI(UsdPhysics.CollisionAPI)
    assert not slab.HasAPI(UsdPhysics.RigidBodyAPI)  # static, per spec

    # purpose=guide (feat/isaac-demo-look: excludes the slab from RTX renders, which
    # otherwise showed its opaque default color through the now-translucent Floor
    # above it) - "guide" must be in the query purposes here or the bbox comes back
    # empty (a real collider is unaffected; only the geometric query needs updating).
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide"])
    floor_range = bbox_cache.ComputeWorldBound(floor).ComputeAlignedRange()
    slab_range = bbox_cache.ComputeWorldBound(slab).ComputeAlignedRange()

    assert slab_range.GetMax()[2] == pytest.approx(floor_range.GetMin()[2], abs=1e-9)
    assert slab_range.GetMax()[2] - slab_range.GetMin()[2] == pytest.approx(
        usd_export.FLOOR_SLAB_THICKNESS_M
    )
    for axis in (0, 1):  # same XY footprint as the floor
        assert slab_range.GetMin()[axis] == pytest.approx(floor_range.GetMin()[axis])
        assert slab_range.GetMax()[axis] == pytest.approx(floor_range.GetMax()[axis])


def test_dome_light_present(fixture_export_input):
    """Exported scenes used to carry zero UsdLux prims, leaving offscreen RTX renders
    (e.g. Isaac Sim) pitch black regardless of camera framing."""
    stage, _ = usd_export.build_stage(fixture_export_input)
    light = stage.GetPrimAtPath("/World/DomeLight")
    assert light.IsValid()
    assert light.HasAPI(UsdLux.LightAPI)
    assert UsdLux.DomeLight(light).GetIntensityAttr().Get() == usd_export.DOME_LIGHT_INTENSITY


def test_key_light_present(fixture_export_input):
    """feat/isaac-demo-look: a dome alone is flat - no directional shadow to read
    volume/depth by. Numbers are GPU-verified (see KEY_LIGHT_INTENSITY's comment)."""
    stage, _ = usd_export.build_stage(fixture_export_input)
    light = stage.GetPrimAtPath("/World/KeyLight")
    assert light.IsValid()
    assert light.HasAPI(UsdLux.LightAPI)
    assert UsdLux.DistantLight(light).GetIntensityAttr().Get() == usd_export.KEY_LIGHT_INTENSITY
    rotate_ops = [op for op in UsdGeom.Xformable(light).GetOrderedXformOps()]
    assert len(rotate_ops) == 1
    assert tuple(rotate_ops[0].Get()) == usd_export.KEY_LIGHT_ROTATE_XYZ_DEG


def test_structure_boxes_are_static_colliders(fixture_export_input):
    stage, stats = usd_export.build_stage(fixture_export_input)
    structure = stage.GetPrimAtPath("/World/Structure")
    children = list(structure.GetChildren())
    assert len(children) == stats.structure_box_count
    # sanity per spec: "hundreds, not thousands" for a real room
    assert stats.structure_box_count < 1000
    for child in children:
        assert child.HasAPI(UsdPhysics.CollisionAPI)
        assert not child.HasAPI(UsdPhysics.RigidBodyAPI)


def test_structure_boxes_are_fixed_low_height_not_ceiling_y(fixture_export_input):
    """feat/isaac-demo-look: Structure is exported at STRUCTURE_HEIGHT_M (low,
    translucent colliders so the point cloud above them stays visible) regardless of
    the room's actual ceiling_y - a regression here would silently bring back
    floor-to-ceiling walls that hide the point cloud."""
    assert fixture_export_input.ceiling_y != pytest.approx(usd_export.STRUCTURE_HEIGHT_M)
    stage, _ = usd_export.build_stage(fixture_export_input)
    structure = stage.GetPrimAtPath("/World/Structure")
    children = list(structure.GetChildren())
    assert children  # fixture has at least one obstacle box
    for child in children:
        scale = UsdGeom.Xformable(child).GetOrderedXformOps()[1].Get()  # [translate, scale]
        assert scale[2] == pytest.approx(usd_export.STRUCTURE_HEIGHT_M)


def test_floor_and_structure_display_color_and_opacity_match_2d_map_legend(fixture_export_input):
    """Floor/Structure colors are pinned to the 2D occupancy-map legend's
    free/obstacle colors (tokens.ts's OCC_FREE/OCC_OBSTACLE) so the Isaac export reads
    as a continuation of the app's own 2D view."""
    stage, _ = usd_export.build_stage(fixture_export_input)
    floor = UsdGeom.Cube(stage.GetPrimAtPath("/World/Floor"))
    assert tuple(floor.GetDisplayColorAttr().Get()[0]) == pytest.approx(usd_export.FLOOR_COLOR)
    assert floor.GetDisplayOpacityAttr().Get()[0] == pytest.approx(usd_export.FLOOR_OPACITY)

    box = UsdGeom.Cube(stage.GetPrimAtPath("/World/Structure").GetChildren()[0])
    assert tuple(box.GetDisplayColorAttr().Get()[0]) == pytest.approx(usd_export.STRUCTURE_COLOR)
    assert box.GetDisplayOpacityAttr().Get()[0] == pytest.approx(usd_export.STRUCTURE_OPACITY)

    # FLOOR_COLOR/STRUCTURE_COLOR must actually differ from each other (free vs.
    # obstacle are visually distinct on the 2D map, same must hold here).
    assert usd_export.FLOOR_COLOR != usd_export.STRUCTURE_COLOR


def test_object_class_color_is_deterministic_and_opacity_is_set(fixture_export_input):
    """One consistent color per object 'class' (= its name, see _object_class_color's
    docstring): same name -> same color across calls/objects; different names ->
    (almost certainly) different colors."""
    assert usd_export._object_class_color("chair") == usd_export._object_class_color("chair")
    assert usd_export._object_class_color("chair") != usd_export._object_class_color("table")

    stage, _ = usd_export.build_stage(fixture_export_input)
    objects_xform = stage.GetPrimAtPath("/World/Objects")
    children = [c for c in objects_xform.GetChildren() if c.GetName() != "_fragments"]
    assert children
    for child in children:
        gprim = UsdGeom.Gprim(child)
        assert gprim.GetDisplayOpacityAttr().Get()[0] == pytest.approx(usd_export.OBJECT_OPACITY)
        assert gprim.GetDisplayColorAttr().Get() is not None


def test_object_count_matches_fixture_and_fragments_are_split(fixture_export_input):
    stage, stats = usd_export.build_stage(fixture_export_input)
    total_objects = len(fixture_export_input.objects)
    fragment_total = sum(1 for o in fixture_export_input.objects if o.is_fragment)

    assert stats.fragment_count == fragment_total
    assert stats.object_count == total_objects  # include_fragments=True in the fixture

    main_children = {c.GetName() for c in stage.GetPrimAtPath("/World/Objects").GetChildren()}
    frag_children = {
        c.GetName() for c in stage.GetPrimAtPath("/World/Objects/_fragments").GetChildren()
    }
    assert "_fragments" not in (main_children - {"_fragments"})
    assert len(main_children - {"_fragments"}) + len(frag_children) == total_objects


def test_object_prims_are_static_collision_only(fixture_export_input):
    """Matches IsaacSimExport.tsx's promise ("static collision only, no...") - Objects
    briefly carried RigidBodyAPI/MassAPI during the initial feat/isaac-demo-look pass;
    a real GPU run found this scene's convex-hull colliders dynamically unstable
    against each other/Structure (docs/DECISIONS.md), so dynamics was dropped from
    Objects entirely rather than worked around downstream."""
    stage, _ = usd_export.build_stage(fixture_export_input)
    objects_root = stage.GetPrimAtPath("/World/Objects")
    non_fragment = [c for c in objects_root.GetChildren() if c.GetName() != "_fragments"]
    assert non_fragment  # fixture has at least one non-fragment object
    for prim in non_fragment:
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert prim.GetAttribute("cloudeye:name").IsValid()
        assert prim.GetAttribute("cloudeye:isFragment").Get() is False


def test_object_transform_matches_bbox_exactly(fixture_export_input):
    """Deterministic conversion, per spec: zero coordinate discrepancy, not
    within-tolerance. `_add_objects` creates one prim per non-fragment object in the
    same order as `fixture_export_input.objects`, and USD preserves prim definition
    order in GetChildren() - so pairing by that order (rather than re-deriving the
    prim name) is an exact, unambiguous match even when two objects share a name."""
    stage, _ = usd_export.build_stage(fixture_export_input)
    non_fragment_objects = [o for o in fixture_export_input.objects if not o.is_fragment]

    objects_root = stage.GetPrimAtPath("/World/Objects")
    prims = [c for c in objects_root.GetChildren() if c.GetName() != "_fragments"]
    assert len(prims) == len(non_fragment_objects)

    for obj, prim in zip(non_fragment_objects, prims):
        bmin, bmax = obj.bbox_min, obj.bbox_max
        expected_center = usd_export.to_isaac(
            (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2
        )
        expected_size = (bmax[0] - bmin[0], bmax[2] - bmin[2], bmax[1] - bmin[1])

        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        actual_translate, actual_scale = ops[0].Get(), ops[1].Get()  # [translate, scale]

        assert prim.GetAttribute("cloudeye:name").Get() == obj.name
        for i in range(3):
            assert actual_translate[i] == pytest.approx(expected_center[i], abs=1e-9)
            assert actual_scale[i] == pytest.approx(expected_size[i], abs=1e-9)


def test_floor_and_objects_occupy_the_same_footprint(tmp_path):
    """Regression test for the xformOpOrder bug: `_add_cube` (used for /World/Floor and
    /World/Structure/*) once added AddScaleOp() before AddTranslateOp(). Xformable's
    GetLocalTransformation() composes ops in *reverse* of listed order (see the
    comment in `_add_cube`), so [scale, translate] applied translate to the point
    before scale - silently scaling every cube's world position by its own size.
    Floor/Structure ended up meters away from Objects (which use `_add_hull_mesh`'s
    raw absolute coordinates when a real point cloud is available, so they were never
    affected) - two disjoint regions in one file. Both areas should footprint the same
    physical room, so their world-space XY bboxes must overlap.

    Uses a synthetic scene, not the `scene_horizontal` fixture: that fixture's grid
    origin is close enough to zero that the bug's self-scaling error is too small to
    break bbox overlap there, which would make this test pass whether or not the bug
    is present. Placing the grid origin far from zero (100m) makes the induced error
    (~ center * (scale - 1), see `_add_cube`) large relative to the room size, so the
    test fails deterministically without the fix and passes with it - not fixture-
    scale-dependent."""
    ply_path = tmp_path / "cube.ply"
    # a small non-degenerate point cloud (cube corners) positioned inside the floor's
    # footprint, given a real mesh_path so _add_object takes the convex-hull path
    # (unaffected by the _add_cube bug) rather than falling back to bbox (_add_cube).
    cube_points = np.array([
        [x, y, z] for x in (101.0, 102.0) for y in (0.0, 1.0) for z in (101.0, 102.0)
    ])
    _write_binary_ply(ply_path, cube_points)

    grid_meta = scene_ingest.GridMeta(resolution=1.0, origin_x=100.0, origin_z=100.0, width=4, height=4)
    cells = np.full((4, 4), FREE, dtype=np.uint8)
    export_input = UsdExportInput(
        grid_cells=cells,
        grid_meta=grid_meta,
        ceiling_y=2.5,
        robot_start=None,
        is_reflected=False,
        robot_radius_m=0.1,
        objects=[
            UsdExportObject(
                name="cube", bbox_min=tuple(cube_points.min(axis=0)),
                bbox_max=tuple(cube_points.max(axis=0)), num_views=3,
                num_points=len(cube_points), is_fragment=False, mesh_path=str(ply_path),
            )
        ],
        pointcloud_path=None,
        include_fragments=False,
    )

    stage, _ = usd_export.build_stage(export_input)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    floor = stage.GetPrimAtPath("/World/Floor")
    floor_range = cache.ComputeWorldBound(floor).ComputeAlignedRange()

    objects_range = None
    for prim in stage.GetPrimAtPath("/World/Objects").GetChildren():
        prim_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        objects_range = prim_range if objects_range is None else objects_range.UnionWith(prim_range)
    assert objects_range is not None  # fixture has one object

    floor_min, floor_max = floor_range.GetMin(), floor_range.GetMax()
    obj_min, obj_max = objects_range.GetMin(), objects_range.GetMax()

    # X and Z are the horizontal (floor-plane) axes in this Z-up, Isaac-convention stage.
    for axis in (0, 2):
        overlaps = floor_min[axis] <= obj_max[axis] and obj_min[axis] <= floor_max[axis]
        assert overlaps, (
            f"axis {axis}: floor range [{floor_min[axis]}, {floor_max[axis]}] does not "
            f"overlap objects range [{obj_min[axis]}, {obj_max[axis]}]"
        )


def test_robot_start_prim(fixture_export_input):
    stage, _ = usd_export.build_stage(fixture_export_input)
    prim = stage.GetPrimAtPath("/World/RobotStart")
    assert prim.IsValid()
    translate = UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()
    x, z = fixture_export_input.robot_start
    expected = usd_export.to_isaac(x, 0.0, z)
    for i in range(3):
        assert translate[i] == pytest.approx(expected[i], abs=1e-9)


def test_export_roundtrips_to_disk(fixture_export_input, tmp_path):
    out_path = tmp_path / "usd" / "scene.usd"
    stats = usd_export.export_scene_usd(fixture_export_input, out_path)
    assert out_path.is_file()
    assert stats.structure_box_count > 0

    reopened = Usd.Stage.Open(str(out_path))
    assert reopened.GetPrimAtPath("/World/Floor").IsValid()


def test_include_fragments_false_omits_fragment_prims(fixture_export_input):
    import dataclasses

    export_input = dataclasses.replace(fixture_export_input, include_fragments=False)
    stage, stats = usd_export.build_stage(export_input)
    assert not stage.GetPrimAtPath("/World/Objects/_fragments").IsValid()
    non_fragment_total = sum(1 for o in export_input.objects if not o.is_fragment)
    assert stats.object_count == non_fragment_total


def test_fixture_objects_all_fall_back_to_bbox(fixture_export_input):
    """The real fixture's mesh_path entries point at .ply files that aren't checked
    into the repo (see module docstring) - so every object collider should come back
    as a bbox fallback, never a convex hull, and the stats should say so."""
    stage, stats = usd_export.build_stage(fixture_export_input)
    assert stats.hull_object_count == 0
    assert stats.hull_decimated_object_count == 0
    assert stats.bbox_fallback_object_count == stats.object_count

    objects_root = stage.GetPrimAtPath("/World/Objects")
    non_fragment = [c for c in objects_root.GetChildren() if c.GetName() != "_fragments"]
    for prim in non_fragment:
        assert prim.GetTypeName() == "Cube"
        assert prim.GetAttribute("cloudeye:collisionSource").Get() == "bbox"


# --- build_convex_hull (pure) -----------------------------------------------------------


def _hull_signed_distances(hull: usd_export.HullBuildResult, query_points: np.ndarray) -> np.ndarray:
    """Signed distance of each query point from `hull` (positive = outside), via the
    hull's own re-derived facet equations. Used to check "does this hull actually
    contain these points" without depending on face winding."""
    recomputed = ConvexHull(hull.vertices)
    eqs = recomputed.equations  # (F, 4): a*x + b*y + c*z + d <= 0 inside
    dists = query_points @ eqs[:, :3].T + eqs[:, 3]
    return dists.max(axis=1)


CUBE_CORNERS = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])


def test_hull_of_cube_corners_contains_all_points_and_has_right_volume():
    hull = usd_export.build_convex_hull(CUBE_CORNERS)
    assert hull is not None
    assert not hull.decimated
    assert hull.vertices.shape[0] == 8  # every corner of a cube is a hull vertex
    assert hull.faces.shape == (12, 3)  # 6 square faces, triangulated
    assert hull.volume_m3 == pytest.approx(1.0)

    dists = _hull_signed_distances(hull, CUBE_CORNERS)
    assert dists.max() < 1e-9  # every input point on or inside its own hull


def test_hull_returns_none_for_fewer_than_four_points():
    assert usd_export.build_convex_hull(CUBE_CORNERS[:3]) is None


def test_hull_returns_none_for_coplanar_points():
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    assert usd_export.build_convex_hull(flat) is None


def test_hull_never_exceeds_bbox_of_its_own_input_points():
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, size=(200, 3))
    hull = usd_export.build_convex_hull(points)
    assert hull is not None
    assert not hull.decimated  # random interior points -> well under MAX_HULL_VERTICES

    bmin, bmax = points.min(axis=0), points.max(axis=0)
    eps = 1e-9
    assert (hull.vertices >= bmin - eps).all()
    assert (hull.vertices <= bmax + eps).all()


def test_hull_decimates_when_input_exceeds_vertex_cap():
    """Points spread over a sphere surface are all in convex position (none is
    interior to the others' hull), so a large enough sample reliably exceeds
    MAX_HULL_VERTICES and forces decimation - a Fibonacci sphere gives a deterministic,
    evenly-spread point set instead of relying on a lucky random draw."""
    n = 300
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    sphere_points = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )

    hull = usd_export.build_convex_hull(sphere_points)
    assert hull is not None
    assert hull.decimated
    assert hull.vertices.shape[0] <= usd_export.MAX_HULL_VERTICES

    # Decimation trades some containment for a small vertex count - not all 300
    # original points stay inside a 64-vertex approximation of a sphere, but the
    # violation should be bounded, not the hull collapsing to something unrelated to
    # the input (e.g. a sliver near one pole).
    dists = _hull_signed_distances(hull, sphere_points)
    assert dists.max() < 0.35  # sphere radius is 1.0, so this is a generous bound


# --- Object collider: convex hull vs bbox fallback ---------------------------------------

PYRAMID_POINTS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # base
    [0.5, 0.5, 1.0],  # apex
])


def _pyramid_object(ply_path) -> UsdExportObject:
    bmin = tuple(PYRAMID_POINTS.min(axis=0))
    bmax = tuple(PYRAMID_POINTS.max(axis=0))
    return UsdExportObject(
        name="pyramid",
        bbox_min=bmin,
        bbox_max=bmax,
        num_views=3,
        num_points=len(PYRAMID_POINTS),
        is_fragment=False,
        mesh_path=str(ply_path),
    )


def test_object_with_valid_cloud_gets_convex_hull_mesh(tmp_path):
    ply_path = tmp_path / "pyramid.ply"
    _write_binary_ply(ply_path, PYRAMID_POINTS)
    obj = _pyramid_object(ply_path)

    stage = Usd.Stage.CreateInMemory()
    source, decimated = usd_export._add_object(stage, "/World/pyramid", obj)
    assert source == "convex_hull"
    assert not decimated

    prim = stage.GetPrimAtPath("/World/pyramid")
    assert prim.GetTypeName() == "Mesh"
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)  # static collision only, see module docstring
    assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
    assert prim.GetAttribute("cloudeye:collisionSource").Get() == "convex_hull"


def test_object_hull_mesh_has_bound_translucent_material(tmp_path):
    """See _apply_translucent_material's docstring: Isaac's RTX renderer needs an
    actual bound material for opacity to render as translucent, not just the
    displayColor/displayOpacity primvars alone."""
    ply_path = tmp_path / "pyramid.ply"
    _write_binary_ply(ply_path, PYRAMID_POINTS)
    obj = _pyramid_object(ply_path)

    stage = Usd.Stage.CreateInMemory()
    usd_export._add_object(stage, "/World/pyramid", obj)
    prim = stage.GetPrimAtPath("/World/pyramid")

    binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    assert binding.GetPrim().IsValid()
    shader = UsdShade.Shader(binding.GetPrim().GetChild("Shader"))
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    opacity_input = shader.GetInput("opacity")
    assert opacity_input.Get() == pytest.approx(usd_export.OBJECT_OPACITY)


def test_hull_mesh_points_stay_within_object_bbox(tmp_path):
    ply_path = tmp_path / "pyramid.ply"
    _write_binary_ply(ply_path, PYRAMID_POINTS)
    obj = _pyramid_object(ply_path)

    stage = Usd.Stage.CreateInMemory()
    usd_export._add_object(stage, "/World/pyramid", obj)
    prim = stage.GetPrimAtPath("/World/pyramid")
    isaac_points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())

    bmin, bmax = np.array(obj.bbox_min), np.array(obj.bbox_max)
    isaac_min = usd_export.to_isaac(*bmin)
    isaac_max = usd_export.to_isaac(*bmax)
    eps = 1e-5
    for axis in range(3):
        lo, hi = sorted((isaac_min[axis], isaac_max[axis]))
        assert (isaac_points[:, axis] >= lo - eps).all()
        assert (isaac_points[:, axis] <= hi + eps).all()


def test_object_falls_back_to_bbox_without_mesh_path():
    obj = UsdExportObject(
        name="nomesh", bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
        num_views=1, num_points=0, is_fragment=False, mesh_path=None,
    )
    stage = Usd.Stage.CreateInMemory()
    source, decimated = usd_export._add_object(stage, "/World/nomesh", obj)
    assert source == "bbox"
    assert not decimated
    prim = stage.GetPrimAtPath("/World/nomesh")
    assert prim.GetTypeName() == "Cube"
    assert prim.GetAttribute("cloudeye:collisionSource").Get() == "bbox"


def test_object_falls_back_to_bbox_when_file_missing(tmp_path):
    obj = UsdExportObject(
        name="missing", bbox_min=(0, 0, 0), bbox_max=(1, 1, 1), num_views=1, num_points=10,
        is_fragment=False, mesh_path=str(tmp_path / "does_not_exist.ply"),
    )
    stage = Usd.Stage.CreateInMemory()
    source, _ = usd_export._add_object(stage, "/World/missing", obj)
    assert source == "bbox"


def test_object_falls_back_to_bbox_with_too_few_points(tmp_path):
    ply_path = tmp_path / "sparse.ply"
    _write_binary_ply(ply_path, PYRAMID_POINTS[:3])  # only 3 points
    obj = UsdExportObject(
        name="sparse", bbox_min=(0, 0, 0), bbox_max=(1, 1, 0), num_views=1, num_points=3,
        is_fragment=False, mesh_path=str(ply_path),
    )
    stage = Usd.Stage.CreateInMemory()
    source, _ = usd_export._add_object(stage, "/World/sparse", obj)
    assert source == "bbox"


def test_object_falls_back_to_bbox_with_coplanar_points(tmp_path):
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    ply_path = tmp_path / "flat.ply"
    _write_binary_ply(ply_path, flat)
    obj = UsdExportObject(
        name="flat", bbox_min=(0, 0, 0), bbox_max=(1, 1, 0), num_views=1, num_points=5,
        is_fragment=False, mesh_path=str(ply_path),
    )
    stage = Usd.Stage.CreateInMemory()
    source, _ = usd_export._add_object(stage, "/World/flat", obj)
    assert source == "bbox"


def test_decimated_hull_is_still_used_as_the_collider(tmp_path):
    """Decimation itself is covered directly against build_convex_hull
    (test_hull_decimates_when_input_exceeds_vertex_cap) - this just checks that a
    successful decimated hull still gets used as a convex_hull collider, not treated
    as a fallback, when reached through the full _add_object path."""
    n = 300
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    sphere_points = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )

    ply_path = tmp_path / "sphere.ply"
    _write_binary_ply(ply_path, sphere_points)
    obj = UsdExportObject(
        name="sphere",
        bbox_min=tuple(sphere_points.min(axis=0)),
        bbox_max=tuple(sphere_points.max(axis=0)),
        num_views=5,
        num_points=n,
        is_fragment=False,
        mesh_path=str(ply_path),
    )
    stage = Usd.Stage.CreateInMemory()
    source, decimated = usd_export._add_object(stage, "/World/sphere", obj)

    assert source == "convex_hull"
    assert decimated
    prim = stage.GetPrimAtPath("/World/sphere")
    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    assert points.shape[0] <= usd_export.MAX_HULL_VERTICES


# --- Isaac demo-look: point cloud purpose/instancer + schema version -----------------


def test_pointcloud_points_prim_is_guide_purpose(tmp_path):
    """purpose=guide (not render): Isaac's RTX Replicator pipeline was confirmed to not
    render UsdGeomPoints at all regardless of purpose (feat/isaac-demo-look, see
    docs/DECISIONS.md) - this Points prim is no longer the primary visible layer (see
    `_add_pointcloud_instancer`), just kept for other consumers. No CollisionAPI -
    visual only, per the module docstring."""
    points = np.array([[float(i), 0.0, float(i)] for i in range(10)])
    ply_path = tmp_path / "cloud.ply"
    _write_binary_ply(ply_path, points)
    positions, colors = usd_export._load_pointcloud(ply_path)

    stage = Usd.Stage.CreateInMemory()
    n = usd_export._add_pointcloud(stage, "/World/PointCloud", positions, colors)
    assert n == len(points)

    prim = stage.GetPrimAtPath("/World/PointCloud")
    assert prim.IsValid()
    assert prim.IsA(UsdGeom.Points)
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Points(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide


def test_pointcloud_extent_is_authored(tmp_path):
    """Without an authored extent, UsdGeom.BBoxCache returns an empty/sentinel range
    for a Points prim even with real points present - confirmed against a live Isaac
    Sim stage (feat/isaac-demo-look, see docs/DECISIONS.md's isaac-demo-look entry)."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 2.0]])
    ply_path = tmp_path / "cloud.ply"
    _write_binary_ply(ply_path, points)
    positions, colors = usd_export._load_pointcloud(ply_path)

    stage = Usd.Stage.CreateInMemory()
    usd_export._add_pointcloud(stage, "/World/PointCloud", positions, colors)

    prim = stage.GetPrimAtPath("/World/PointCloud")
    extent = UsdGeom.Points(prim).GetExtentAttr().Get()
    assert extent is not None and len(extent) == 2
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide"])
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    assert not rng.IsEmpty()


def test_voxel_downsample_caps_at_max_points_and_preserves_span():
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2.0, 2.0, size=(5000, 3))

    out_positions, out_colors = usd_export._voxel_downsample(
        positions, None, max_points=500, initial_voxel_m=0.01
    )
    assert out_colors is None
    assert 0 < out_positions.shape[0] <= 500
    # centroids stay within the input's own bounding box (never invent geometry
    # outside the real cloud's span)
    assert (out_positions.min(axis=0) >= positions.min(axis=0) - 1e-9).all()
    assert (out_positions.max(axis=0) <= positions.max(axis=0) + 1e-9).all()


def test_voxel_downsample_averages_colors_per_voxel():
    # two points in the same voxel (voxel_m=1.0 -> both floor to (0,0,0)), one in another
    positions = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]])
    colors = np.array([[0, 0, 0], [100, 100, 100], [255, 255, 255]], dtype=np.uint8)

    out_positions, out_colors = usd_export._voxel_downsample(
        positions, colors, max_points=100, initial_voxel_m=1.0
    )
    assert out_positions.shape[0] == 2
    # the merged voxel's color is the average of its two source colors
    merged_color = out_colors[np.argmin(out_positions[:, 0])]
    assert merged_color == pytest.approx([50, 50, 50])


def test_pointcloud_instancer_prim_exists_with_prototype_and_dedup(tmp_path):
    """PointInstancer of small cubes, voxel-downsampled - the actual visible-in-Isaac
    point cloud layer (feat/isaac-demo-look). No CollisionAPI - visual only."""
    points = np.array([[float(i) * 0.001, 0.0, 0.0] for i in range(1000)])  # tight cluster
    colors = np.tile(np.array([10, 20, 30], dtype=np.uint8), (1000, 1))
    ply_path = tmp_path / "cloud.ply"
    _write_binary_ply(ply_path, points)
    positions, _ = usd_export._load_pointcloud(ply_path)

    stage = Usd.Stage.CreateInMemory()
    n = usd_export._add_pointcloud_instancer(stage, "/World/PointCloudInstancer", positions, colors)

    instancer_prim = stage.GetPrimAtPath("/World/PointCloudInstancer")
    assert instancer_prim.IsValid()
    assert instancer_prim.IsA(UsdGeom.PointInstancer)
    assert not instancer_prim.HasAPI(UsdPhysics.CollisionAPI)

    instancer = UsdGeom.PointInstancer(instancer_prim)
    positions_attr = instancer.GetPositionsAttr().Get()
    proto_indices = instancer.GetProtoIndicesAttr().Get()
    assert len(positions_attr) == n == len(proto_indices)
    # a tight cluster downsamples to far fewer than 1000 instances (dedup actually happened)
    assert n < 1000

    proto_targets = instancer.GetPrototypesRel().GetTargets()
    assert len(proto_targets) == 1
    proto_prim = stage.GetPrimAtPath(proto_targets[0])
    assert proto_prim.IsA(UsdGeom.Cube)
    assert UsdGeom.Cube(proto_prim).GetSizeAttr().Get() == pytest.approx(
        usd_export.POINTCLOUD_INSTANCER_CUBE_SIZE_M
    )
    assert UsdGeom.Imageable(proto_prim).ComputeVisibility() == UsdGeom.Tokens.invisible

    color_primvar = UsdGeom.PrimvarsAPI(instancer_prim).GetPrimvar("displayColor")
    assert color_primvar.IsDefined()
    assert color_primvar.GetInterpolation() == UsdGeom.Tokens.vertex
    assert len(color_primvar.Get()) == n


def test_pointcloud_instancer_caps_at_module_max(tmp_path):
    rng = np.random.default_rng(1)
    points = rng.uniform(-5.0, 5.0, size=(5000, 3))
    ply_path = tmp_path / "cloud.ply"
    _write_binary_ply(ply_path, points)
    positions, _ = usd_export._load_pointcloud(ply_path)

    stage = Usd.Stage.CreateInMemory()
    n = usd_export._add_pointcloud_instancer(stage, "/World/PointCloudInstancer", positions, None)
    assert n <= usd_export.POINTCLOUD_INSTANCER_MAX_POINTS


def test_pointcloud_included_by_default_in_build_stage(fixture_export_input, tmp_path):
    """The Isaac export path (demo/record_isaac.py's resolve_standard_usd_path, and the
    /usd endpoint's own `include_pointcloud: bool = Query(True)`) already defaults to
    including the point cloud - this just pins that a pointcloud_path on UsdExportInput
    always produces both the guide-purpose Points prim and the visible PointInstancer."""
    import dataclasses

    points = np.array([[float(i), 0.0, float(i)] for i in range(10)])
    ply_path = tmp_path / "cloud.ply"
    _write_binary_ply(ply_path, points)

    export_input = dataclasses.replace(fixture_export_input, pointcloud_path=ply_path)
    stage, stats = usd_export.build_stage(export_input)
    assert stats.pointcloud_point_count == len(points)
    assert stats.pointcloud_instancer_count > 0
    assert stage.GetPrimAtPath("/World/PointCloud").IsValid()
    assert stage.GetPrimAtPath("/World/PointCloudInstancer").IsValid()


# --- Isaac demo-look: /World/Path and /World/Target -----------------------------------


def test_path_prim_is_a_polyline_through_the_waypoints(fixture_export_input):
    import dataclasses

    waypoints = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.5)]
    export_input = dataclasses.replace(fixture_export_input, path_points=waypoints)
    stage, _ = usd_export.build_stage(export_input)

    prim = stage.GetPrimAtPath("/World/Path")
    assert prim.IsValid()
    assert prim.IsA(UsdGeom.BasisCurves)
    curves = UsdGeom.BasisCurves(prim)
    assert list(curves.GetCurveVertexCountsAttr().Get()) == [len(waypoints)]
    points = curves.GetPointsAttr().Get()
    assert len(points) == len(waypoints)
    for (x, z), p in zip(waypoints, points):
        expected = usd_export.to_isaac(x, usd_export.PATH_HEIGHT_ABOVE_FLOOR_M, z)
        assert tuple(p) == pytest.approx(tuple(expected), abs=1e-5)


def test_path_omitted_when_fewer_than_two_points(fixture_export_input):
    import dataclasses

    export_input = dataclasses.replace(fixture_export_input, path_points=[(0.0, 0.0)])
    stage, _ = usd_export.build_stage(export_input)
    assert not stage.GetPrimAtPath("/World/Path").IsValid()


def test_target_prim_positioned_and_references_resolved_object(fixture_export_input):
    import dataclasses

    export_input = dataclasses.replace(
        fixture_export_input, target_point=(1.5, -0.5), target_object_index=0,
    )
    stage, _ = usd_export.build_stage(export_input)

    target = stage.GetPrimAtPath("/World/Target")
    assert target.IsValid()
    translate = UsdGeom.Xformable(target).GetOrderedXformOps()[0].Get()
    assert tuple(translate) == pytest.approx(tuple(usd_export.to_isaac(1.5, 0.0, -0.5)), abs=1e-6)

    rel = target.GetRelationship("cloudeye:targetObject")
    assert rel.IsValid()
    targets = rel.GetTargets()
    assert len(targets) == 1
    referenced = stage.GetPrimAtPath(targets[0])
    assert referenced.IsValid()
    assert referenced.GetAttribute("cloudeye:name").Get() == fixture_export_input.objects[0].name


def test_target_prim_without_relationship_when_no_object_index(fixture_export_input):
    import dataclasses

    export_input = dataclasses.replace(
        fixture_export_input, target_point=(1.5, -0.5), target_object_index=None,
    )
    stage, _ = usd_export.build_stage(export_input)
    target = stage.GetPrimAtPath("/World/Target")
    assert target.IsValid()
    assert not target.GetRelationship("cloudeye:targetObject").IsValid()


def test_export_schema_version_is_5():
    """Pins the current schema version so a future bump (a real, deliberate change) is
    a visible one-line diff here rather than a silent surprise - see
    EXPORT_SCHEMA_VERSION's comment for the full bump history/rationale."""
    assert usd_export.EXPORT_SCHEMA_VERSION == 5
    assert "scene_v5_" in usd_export.cache_filename(
        include_pointcloud=True, include_fragments=False, robot_radius_m=0.1
    )


# --- Isaac demo-look: exclude fully-elevated objects (ceiling fixtures etc) ---------


def test_elevated_object_entirely_above_threshold_is_excluded(fixture_export_input):
    """A real GPU render found two ceiling-mounted 'fan' objects rendering as
    disconnected hulls floating above the room's now-low Structure walls - an object
    entirely above OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M isn't a real navigation concern
    for a ground robot either way, so it's dropped from the export outright."""
    import dataclasses

    fan = dataclasses.replace(
        fixture_export_input.objects[0],
        name="ceiling fan",
        bbox_min=(1.0, 2.3, 1.0),
        bbox_max=(1.3, 2.6, 1.3),
    )
    export_input = dataclasses.replace(
        fixture_export_input, objects=[fan, *fixture_export_input.objects]
    )
    stage, stats = usd_export.build_stage(export_input)

    assert stats.elevated_excluded_object_count == 1
    objects_root = stage.GetPrimAtPath("/World/Objects")
    names = {
        c.GetAttribute("cloudeye:name").Get() for c in objects_root.GetChildren() if c.GetName() != "_fragments"
    }
    assert "ceiling fan" not in names


def test_object_merely_extending_above_threshold_is_kept(fixture_export_input):
    """Keyed on bbox_min (the object's LOWEST point), not its max - a tall shelf or a
    floor-to-ceiling curtain that starts near the floor and extends above the
    threshold is still a real obstacle at floor level and must not be dropped."""
    import dataclasses

    tall_shelf = dataclasses.replace(
        fixture_export_input.objects[0],
        name="tall shelf",
        bbox_min=(1.0, 0.1, 1.0),
        bbox_max=(1.3, 2.5, 1.3),
    )
    export_input = dataclasses.replace(
        fixture_export_input, objects=[tall_shelf, *fixture_export_input.objects]
    )
    stage, stats = usd_export.build_stage(export_input)

    assert stats.elevated_excluded_object_count == 0
    objects_root = stage.GetPrimAtPath("/World/Objects")
    names = {
        c.GetAttribute("cloudeye:name").Get() for c in objects_root.GetChildren() if c.GetName() != "_fragments"
    }
    assert "tall shelf" in names


def test_elevated_object_exactly_at_threshold_is_excluded(fixture_export_input):
    """>=, not >, per OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M's own comment ("at or
    above")."""
    import dataclasses

    at_threshold = dataclasses.replace(
        fixture_export_input.objects[0],
        name="at threshold",
        bbox_min=(1.0, usd_export.OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M, 1.0),
        bbox_max=(1.3, usd_export.OBJECT_EXCLUDE_IF_ENTIRELY_ABOVE_M + 0.2, 1.3),
    )
    export_input = dataclasses.replace(
        fixture_export_input, objects=[at_threshold, *fixture_export_input.objects]
    )
    _, stats = usd_export.build_stage(export_input)
    assert stats.elevated_excluded_object_count == 1
