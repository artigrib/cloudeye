"""scene_ingest parses real GPU pipeline artifacts - tested against an actual captured
scene, not synthetic JSON, since the whole point is catching real schema drift.
"""

import pytest

from app.services import scene_ingest
from app.services.scene_ingest import ArtifactParseError


def test_parse_real_scene_objects(scene_horizontal_dir):
    objects = scene_ingest.parse_scene_objects(
        scene_horizontal_dir / "scene_objects" / "scene_objects.json", scene_horizontal_dir
    )
    assert len(objects) > 0
    names = {o.name for o in objects}
    assert "sink" in names
    assert "mirror" in names

    sink = next(o for o in objects if o.name == "sink" and not o.is_fragment)
    mirror = next(o for o in objects if o.name == "mirror" and not o.is_fragment)
    # the exact check that caught two real failures in this project: a sink must not
    # be taller than a mirror
    assert sink.pos[1] < mirror.pos[1]


def test_object_paths_rebased_onto_scene_dir(scene_horizontal_dir):
    objects = scene_ingest.parse_scene_objects(
        scene_horizontal_dir / "scene_objects" / "scene_objects.json", scene_horizontal_dir
    )
    for obj in objects:
        if obj.mesh_path is not None:
            assert str(scene_horizontal_dir) in obj.mesh_path
            assert not obj.mesh_path.startswith("/workspace")  # never a raw GPU-host path


def test_missing_likely_fragment_defaults_false(tmp_path):
    """An older-schema payload without `likely_fragment` at all must not crash, and
    must default to not-a-fragment (per the ingest module's documented behavior)."""
    import json

    old_schema = [
        {
            "id": "sink_0",
            "label": "sink",
            "centroid_xyz": [0.1, 0.2, 0.3],
            "bbox_min": [0, 0, 0],
            "bbox_max": [1, 1, 1],
            "point_count": 100,
            "source_view_count": 5,
            "point_cloud_path": "scene_objects/sink_0.ply",
        }
    ]
    p = tmp_path / "scene_objects.json"
    p.write_text(json.dumps(old_schema))
    objects = scene_ingest.parse_scene_objects(p, tmp_path)
    assert objects[0].is_fragment is False


def test_absolute_legacy_path_rebased_by_filename(tmp_path):
    import json

    old_schema = [
        {
            "id": "sink_0",
            "label": "sink",
            "centroid_xyz": [0, 0, 0],
            "bbox_min": [0, 0, 0],
            "bbox_max": [1, 1, 1],
            "point_count": 1,
            "source_view_count": 1,
            "point_cloud_path": "/workspace/data/output_horizontal/run2/scene_objects/sink_0.ply",
        }
    ]
    p = tmp_path / "scene_objects.json"
    p.write_text(json.dumps(old_schema))
    objects = scene_ingest.parse_scene_objects(p, tmp_path)
    assert objects[0].mesh_path == str(tmp_path / "scene_objects" / "sink_0.ply")


def test_missing_file_raises_artifact_parse_error(tmp_path):
    with pytest.raises(ArtifactParseError):
        scene_ingest.parse_scene_objects(tmp_path / "does_not_exist.json", tmp_path)


def test_read_alignment_real_fixture(scene_horizontal_dir):
    alignment = scene_ingest.read_alignment(scene_horizontal_dir)
    assert alignment.ceiling_y > 2.0  # a real room, not a degenerate reconstruction
    assert alignment.ceiling_y < 5.0
    assert isinstance(alignment.is_reflection, bool)


def test_read_grid_metadata_real_fixture(scene_horizontal_dir):
    grid = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    assert grid.resolution == 0.05
    assert grid.width > 0
    assert grid.height > 0


def test_read_camera_track_real_fixture(scene_horizontal_dir):
    cameras = scene_ingest.read_camera_track(scene_horizontal_dir)
    assert len(cameras) > 0
    assert all(len(c) == 3 for c in cameras)


def test_choose_robot_start_uses_first_camera(scene_horizontal_dir):
    cameras = scene_ingest.read_camera_track(scene_horizontal_dir)
    grid = scene_ingest.read_grid_metadata(scene_horizontal_dir)
    x, z = scene_ingest.choose_robot_start(cameras, grid)
    assert x == pytest.approx(cameras[0][0])
    assert z == pytest.approx(cameras[0][2])


def test_choose_robot_start_falls_back_to_grid_center_with_no_cameras():
    from app.services.scene_ingest import GridMeta

    grid = GridMeta(resolution=0.1, origin_x=0.0, origin_z=0.0, width=10, height=10)
    x, z = scene_ingest.choose_robot_start([], grid)
    assert x == pytest.approx(0.5)
    assert z == pytest.approx(0.5)


def test_choose_robot_start_skips_camera_track_frames_before_the_room(scene_hotel_room2_dir):
    """Regression test for the `world_to_cell` "far outside the grid bounds" ValueError
    that crashed `resolve_robot_start` on this real scene (hotel-own-batch room 2,
    2026-09-01) after the GPU pipeline had already succeeded and all 31 validators had
    passed. Root cause: `choose_robot_start` picked `cameras[0]` unconditionally: the
    walkthrough's opening frames (indices 0-4, z from -0.001 to -0.605) were captured
    before the operator stepped into the room the occupancy grid was built from, so
    `cameras[0]` sat outside the grid entirely. Fixed by walking forward to the first
    camera pose that's actually inside the grid - verified here against the real
    recorded numbers, and end-to-end against `world_to_cell` so this stays a true
    regression test for the original crash, not just a `choose_robot_start` unit test."""
    from app.services.pathfinding import world_to_cell

    cameras = scene_ingest.read_camera_track(scene_hotel_room2_dir)
    grid = scene_ingest.read_grid_metadata(scene_hotel_room2_dir)

    # cameras[0] is the exact point that used to crash world_to_cell on this scene.
    assert cameras[0][0] == pytest.approx(1.6716119615276923e-05, abs=1e-6)
    assert cameras[0][2] == pytest.approx(-0.0009077778839105129, abs=1e-6)
    with pytest.raises(ValueError, match="far outside the grid bounds"):
        world_to_cell(cameras[0][0], cameras[0][2], grid)

    x, z = scene_ingest.choose_robot_start(cameras, grid)

    # Must not be the out-of-bounds first frame.
    assert (x, z) != pytest.approx((cameras[0][0], cameras[0][2]))
    # Must be camera index 4 specifically - the first one inside the grid's
    # margin-tolerant bounds (index 4: z=-0.605, within the 2-cell/0.1m margin past the
    # grid's z-max of -0.657; index 5, z=-0.895, is inside outright).
    assert (x, z) == pytest.approx((cameras[4][0], cameras[4][2]))

    # And the whole point: world_to_cell must now accept it.
    cell = world_to_cell(x, z, grid)
    assert 0 <= cell[0] < grid.width
    assert 0 <= cell[1] < grid.height


def test_choose_robot_start_falls_back_to_first_camera_if_track_never_enters_grid():
    """If literally no camera pose ever falls inside the grid, the track and the
    reconstructed room don't overlap at all - a genuinely broken scene, not something to
    paper over with a synthetic point. Keeps returning cameras[0] so it still fails
    loudly downstream in resolve_robot_start's world_to_cell call, per that function's
    own documented "must keep raising loudly on failure" contract."""
    from app.services.scene_ingest import GridMeta

    grid = GridMeta(resolution=0.05, origin_x=0.0, origin_z=0.0, width=10, height=10)
    cameras = [(50.0, 1.5, 50.0), (51.0, 1.5, 51.0)]  # nowhere near the 0.5x0.5m grid
    x, z = scene_ingest.choose_robot_start(cameras, grid)
    assert (x, z) == pytest.approx((cameras[0][0], cameras[0][2]))


def _write_keyframe_artifacts(tmp_path, *, fps, manifest_entries, keyframes_json_overrides=None):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps(manifest_entries))
    keyframes_json = {"fps": fps, **(keyframes_json_overrides or {})}
    (tmp_path / "keyframes.json").write_text(json.dumps(keyframes_json))


def test_read_keyframe_timestamps_missing_artifacts_returns_none(scene_horizontal_dir):
    """The real captured fixture predates manifest.json/keyframes.json - the honest
    degradation case this function exists for, not a synthetic edge case."""
    assert scene_ingest.read_keyframe_timestamps(scene_horizontal_dir, num_cameras=26) is None


def test_read_keyframe_timestamps_derives_from_fps_grid(tmp_path):
    # Mirrors a real capture: raw frames 1, 2, 4 extracted at fps=2.0 (so seconds
    # 0.0, 0.5, 1.5), frame 2 dropped as blurry - 2 kept keyframes survive.
    manifest = [
        {"source_frame": "frame_000001.jpg", "kept": True},
        {"source_frame": "frame_000002.jpg", "kept": False},
        {"source_frame": "frame_000004.jpg", "kept": True},
    ]
    _write_keyframe_artifacts(tmp_path, fps=2.0, manifest_entries=manifest)
    timestamps = scene_ingest.read_keyframe_timestamps(tmp_path, num_cameras=2)
    assert timestamps == [pytest.approx(0.0), pytest.approx(1.5)]


def test_read_keyframe_timestamps_count_mismatch_returns_none(tmp_path):
    """stage_keyframes.py's own VRAM-budget subsampling can delete already-`kept`
    keyframes after manifest.json was written, breaking the 1:1 correspondence with
    cameras_aligned.json - this function doesn't try to replicate that selection, it
    just detects the mismatch and backs off rather than mispair a timestamp."""
    manifest = [
        {"source_frame": "frame_000001.jpg", "kept": True},
        {"source_frame": "frame_000004.jpg", "kept": True},
    ]
    _write_keyframe_artifacts(tmp_path, fps=2.0, manifest_entries=manifest, keyframes_json_overrides={"subsampled": True})
    assert scene_ingest.read_keyframe_timestamps(tmp_path, num_cameras=1) is None


def test_read_keyframe_timestamps_missing_fps_returns_none(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps([{"source_frame": "frame_000001.jpg", "kept": True}]))
    (tmp_path / "keyframes.json").write_text(json.dumps({"kept_count": 1}))
    assert scene_ingest.read_keyframe_timestamps(tmp_path, num_cameras=1) is None


def test_read_keyframe_timestamps_malformed_manifest_returns_none(tmp_path):
    _write_keyframe_artifacts(tmp_path, fps=2.0, manifest_entries=[{"kept": True}])  # no source_frame
    assert scene_ingest.read_keyframe_timestamps(tmp_path, num_cameras=1) is None
