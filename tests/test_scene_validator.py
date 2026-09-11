"""scene_validator is the mandatory gate that caught two real failures in this project
(inverted Y axis, RANSAC finding the ceiling). Tested against the real passing fixture,
plus a hand-mutated failing one to confirm the gate actually rejects bad data.
"""

import copy
import dataclasses

from app.services import scene_ingest
from app.services.scene_validator import find_oversized_objects, validate_scene


def _load_real(scene_horizontal_dir):
    objects = scene_ingest.parse_scene_objects(
        scene_horizontal_dir / "scene_objects" / "scene_objects.json", scene_horizontal_dir
    )
    alignment = scene_ingest.read_alignment(scene_horizontal_dir)
    grid = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    cameras = scene_ingest.read_camera_track(scene_horizontal_dir)
    return alignment, objects, grid, cameras


def test_real_fixture_passes_validation(scene_horizontal_dir):
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)
    report = validate_scene(alignment, objects, grid, cameras)
    assert report.passed, report.summary()


def test_real_fixture_reflection_flag_never_fails(scene_horizontal_dir):
    """det(R)=-1 (a reflection) must be flagged but never fail the scene - it's
    harmless for points/centroids/bboxes, only relevant for a future mesh export."""
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)
    mutated = dataclasses.replace(alignment, det_r=-1.0, is_reflection=True)
    report = validate_scene(mutated, objects, grid, cameras)
    assert report.passed
    reflection_check = next(c for c in report.checks if c.name == "reflection_flag")
    assert reflection_check.passed
    assert not reflection_check.fatal


def test_sink_above_mirror_fails_validation(scene_horizontal_dir):
    """The exact scenario that caught two real bugs this project hit: swap a sink's
    and a mirror's Y so the sink ends up taller - must be caught."""
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)

    sink_idx = next(i for i, o in enumerate(objects) if o.name == "sink" and not o.is_fragment)
    mirror_idx = next(i for i, o in enumerate(objects) if o.name == "mirror" and not o.is_fragment)

    mutated = copy.deepcopy(objects)
    sink, mirror = mutated[sink_idx], mutated[mirror_idx]
    # give the sink the mirror's height and vice versa
    mutated[sink_idx] = dataclasses.replace(sink, pos=(sink.pos[0], mirror.pos[1] + 0.3, sink.pos[2]))
    mutated[mirror_idx] = dataclasses.replace(mirror, pos=(mirror.pos[0], sink.pos[1], mirror.pos[2]))

    report = validate_scene(alignment, mutated, grid, cameras)
    assert not report.passed
    failure_names = {c.name for c in report.failures()}
    assert "relative_order_counter_below_wall" in failure_names
    assert "median counter-fixture y=" in report.summary()


def test_inverted_camera_track_fails_floor_check(scene_horizontal_dir):
    """Simulate the exact class of bug this project hit (floor not actually below the
    cameras) by pushing all cameras BELOW the floor (Y=0)."""
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)
    bad_cameras = [(x, -1.0, z) for x, y, z in cameras]
    report = validate_scene(alignment, objects, grid, bad_cameras)
    assert not report.passed
    failure_names = {c.name for c in report.failures()}
    assert "floor_below_cameras" in failure_names


def test_cameras_above_ceiling_fails_ceiling_check(scene_horizontal_dir):
    """The mirror-image case: cameras pushed above the ceiling."""
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)
    bad_cameras = [(x, alignment.ceiling_y + 1.0, z) for x, y, z in cameras]
    report = validate_scene(alignment, objects, grid, bad_cameras)
    assert not report.passed
    failure_names = {c.name for c in report.failures()}
    assert "ceiling_above_cameras" in failure_names


def test_degenerate_room_dimensions_fail(scene_horizontal_dir):
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)
    tiny_ceiling = dataclasses.replace(alignment, ceiling_y=0.5)  # below CEILING_MIN_M
    report = validate_scene(tiny_ceiling, objects, grid, cameras)
    assert not report.passed
    assert any(c.name == "room_dimensions" for c in report.failures())


def test_real_fixture_flags_its_one_borderline_door_but_nothing_else(scene_horizontal_dir):
    """The real fixture has 3 "door" objects (0.93m, 1.82m, 2.52m) - one genuinely just
    over threshold, most likely a doorway-opening segmentation picking up some
    surrounding wall, not a merge bug. Confirms the check is precise (flags exactly
    that one, not sink/mirror/switches) and, via test_real_fixture_passes_validation
    above, that a single flagged object still lets the scene pass overall."""
    _alignment, objects, _grid, _cameras = _load_real(scene_horizontal_dir)
    oversized = find_oversized_objects(objects)
    assert len(oversized) == 1
    assert "door (2.52m)" in oversized[0]


def test_oversized_object_flagged_but_non_fatal(scene_horizontal_dir):
    """Regression test for hotel-own-batch room 2 (2026-09-01): a "bed" object measured
    2.92x3.31m against a real ~1x2m single bed, almost certainly two objects merged by
    DBSCAN into one cluster. Must be flagged, but must NOT fail the scene - unlike
    height_prior_majority, one oversized object isn't strong evidence the whole
    reconstruction is wrong."""
    alignment, objects, grid, cameras = _load_real(scene_horizontal_dir)

    mutated = copy.deepcopy(objects)
    sink_idx = next(i for i, o in enumerate(mutated) if o.name == "sink" and not o.is_fragment)
    sink = mutated[sink_idx]
    oversized_bed = dataclasses.replace(
        sink,
        name="bed",
        bbox_min=(sink.pos[0] - 1.46, sink.bbox_min[1], sink.pos[2] - 1.655),
        bbox_max=(sink.pos[0] + 1.46, sink.bbox_max[1], sink.pos[2] + 1.655),
    )
    mutated[sink_idx] = oversized_bed

    oversized = find_oversized_objects(mutated)
    bed_flags = [o for o in oversized if o.startswith("bed ")]
    assert len(bed_flags) == 1
    assert "bed (3.31m)" in bed_flags[0]

    report = validate_scene(alignment, mutated, grid, cameras)
    extent_check = next(c for c in report.checks if c.name == "object_max_extent")
    assert not extent_check.passed
    assert not extent_check.fatal
    assert "bed (3.31m)" in extent_check.detail
    # A flagged object must not sink the whole scene.
    assert report.passed, report.summary()


def test_fragment_objects_exempt_from_oversized_check():
    from app.services.scene_ingest import ParsedObject

    huge_fragment = ParsedObject(
        name="curtain", description=None, pos=(0, 1, 0),
        bbox_min=(-5, 0, -5), bbox_max=(5, 2, 5),
        num_views=1, num_points=50, is_fragment=True, mesh_path=None,
    )
    assert find_oversized_objects([huge_fragment]) == []


def test_curtain_class_exempt_from_oversized_check_even_when_not_a_fragment():
    """M7b: a curtain legitimately spans a whole wall/window (the hero scene's real
    curtain measures 3.77m long) - that's not the "two objects merged" failure mode
    OBJECT_MAX_EXTENT_M targets, so curtain is exempted by class, independent of
    is_fragment (unlike test_fragment_objects_exempt_from_oversized_check above,
    which only proves the fragment exemption)."""
    from app.services.scene_ingest import ParsedObject

    huge_real_curtain = ParsedObject(
        name="curtain", description=None, pos=(0, 1, 0),
        bbox_min=(-2, 0, -0.1), bbox_max=(2, 3, 0.1),
        num_views=5, num_points=5000, is_fragment=False, mesh_path=None,
    )
    assert find_oversized_objects([huge_real_curtain]) == []

    # Sanity: a non-curtain object with the same huge extent IS still flagged, so
    # this isn't accidentally exempting everything.
    huge_bed = dataclasses.replace(huge_real_curtain, name="bed")
    oversized = find_oversized_objects([huge_bed])
    assert len(oversized) == 1
    assert oversized[0].startswith("bed (")


def test_classify_object():
    from app.services.scene_validator import classify_object

    assert classify_object("sink") == "counter"
    assert classify_object("pedestal sink") == "counter"
    assert classify_object("mirror") == "wall"
    assert classify_object("chair") == "furniture"
    assert classify_object("floor") == "floor"
    assert classify_object("door") is None  # spans floor to ceiling, deliberately unclassified
