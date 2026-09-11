"""Smoke tests for scripts/msa/render_perspective.py's 3/4-view renderer.

Not a golden-image test (no pixel-level assertions - a software rasterizer's
exact pixels aren't a meaningful contract to freeze) - these check the module
doesn't crash on real geometry and actually produces image bytes, plus a
couple of targeted checks on the camera-framing math."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.msa.bootstrap import load_object_inputs_from_hulls_json, run_bootstrap
from scripts.msa.export_glb import build_scene
from scripts.msa.geometry import WallPolygon
from scripts.msa.render_perspective import (
    DEFAULT_IMAGE_SIZE,
    FOV_DEG,
    PITCH_SEARCH_RANGE_DEG,
    TARGET_FLOOR_SCREEN_FRAC,
    WALL_OPACITY,
    _bed_centroid_xz,
    _iter_scene_node_faces,
    _to_camera_space,
    _to_screen,
    compute_camera,
    render_hero_perspective,
    render_hero_perspective_from_glb,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "msa" / "02_modular_home"


class TestRenderHeroPerspectiveSimpleRoom:
    def test_produces_a_non_empty_png(self, tmp_path):
        wall = WallPolygon(vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], area_m2=12.0)
        floor_polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        objects = [
            {
                "id": "chair_0",
                "label": "chair",
                "center_xy": (2.0, 1.5),
                "size_uv": (0.5, 0.5),
                "angle_rad": 0.3,
                "height": 0.9,
                "bbox_min_y": 0.0,
                "color_rgb": (200, 100, 50),
            }
        ]
        out_path = tmp_path / "hero_34.png"

        render_hero_perspective([wall], floor_polygon, 0.0, 2.5, objects, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        with Image.open(out_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0

    def test_handles_no_floor_polygon_and_no_objects(self, tmp_path):
        wall = WallPolygon(vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], area_m2=12.0)
        out_path = tmp_path / "hero_34_walls_only.png"

        render_hero_perspective([wall], None, 0.0, 2.5, [], out_path)

        assert out_path.exists() and out_path.stat().st_size > 0

    def test_handles_totally_empty_scene_without_crashing(self, tmp_path):
        out_path = tmp_path / "hero_34_empty.png"

        render_hero_perspective([], None, 0.0, 2.5, [], out_path)

        assert out_path.exists() and out_path.stat().st_size > 0


class TestComputeCamera:
    """T3: the camera is derived entirely from the room's own geometry every
    time - no fixed back-offset/pitch any more. Rectangle room used
    throughout: vertices (0,0),(8,0),(8,4),(0,4), centroid (4,2), diagonal
    (the two most mutually-distant vertices, here the two opposite corners)
    = hypot(8,4)."""

    ROOM = np.array([(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)])
    ROOM_CENTROID = (4.0, 2.0)
    ROOM_DIAGONAL = math.hypot(8.0, 4.0)

    def test_height_is_relative_to_floor_y(self):
        camera_pos, _forward, _right, _up = compute_camera(self.ROOM, floor_y=1.0, camera_height_m=2.5)
        assert camera_pos[1] == pytest.approx(3.5)

    def test_distance_from_room_centroid_matches_the_diagonal_formula(self):
        expected_distance = max(0.7 * self.ROOM_DIAGONAL, 3.0)
        camera_pos, _f, _r, _u = compute_camera(self.ROOM, floor_y=0.0, bed_centroid_xz=(0.5, 0.5))
        cx, cz = self.ROOM_CENTROID
        actual_distance = math.hypot(camera_pos[0] - cx, camera_pos[2] - cz)
        assert actual_distance == pytest.approx(expected_distance, rel=1e-6)

    def test_min_distance_floor_applies_to_a_small_room(self):
        # A 1x1 room: 0.7 * diagonal (~0.99) is well under the 3.0m floor.
        small_room = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        camera_pos, _f, _r, _u = compute_camera(small_room, floor_y=0.0, bed_centroid_xz=(0.1, 0.1))
        actual_distance = math.hypot(camera_pos[0] - 0.5, camera_pos[2] - 0.5)
        assert actual_distance == pytest.approx(3.0, rel=1e-6)

    def test_direction_is_the_bed_to_farthest_corner_ray_continued_outward(self):
        # Bed anchored near corner (0,0): the room-polygon vertex farthest
        # from it is the opposite corner (8,4), so the camera should sit
        # beyond the centroid along the (0,0)->(8,4) ray.
        bed_xz = (0.5, 0.5)
        camera_pos, _f, _r, _u = compute_camera(self.ROOM, floor_y=0.0, bed_centroid_xz=bed_xz)
        cx, cz = self.ROOM_CENTROID
        expected_dir = np.array([8.0, 4.0]) - np.array(bed_xz)
        expected_dir /= np.linalg.norm(expected_dir)
        actual_dir = np.array([camera_pos[0] - cx, camera_pos[2] - cz])
        actual_dir /= np.linalg.norm(actual_dir)
        assert actual_dir == pytest.approx(expected_dir, abs=1e-6)

    def test_looks_at_the_room_centroid(self):
        camera_pos, forward, _r, _u = compute_camera(self.ROOM, floor_y=0.0, bed_centroid_xz=(0.5, 0.5))
        cx, cz = self.ROOM_CENTROID
        horiz_forward = np.array([forward[0], forward[2]])
        horiz_forward /= np.linalg.norm(horiz_forward)
        to_centroid = np.array([cx - camera_pos[0], cz - camera_pos[2]])
        to_centroid /= np.linalg.norm(to_centroid)
        assert horiz_forward == pytest.approx(to_centroid, abs=1e-6)

    def test_no_bed_falls_back_to_the_room_centroid_as_anchor(self):
        # Real scenario, not hypothetical: a scene with no bed. Omitting
        # bed_centroid_xz must behave exactly as if it had been passed
        # explicitly as the room's own centroid.
        camera_no_bed, forward_no_bed, _r1, _u1 = compute_camera(self.ROOM, floor_y=0.0)
        camera_explicit, forward_explicit, _r2, _u2 = compute_camera(self.ROOM, floor_y=0.0, bed_centroid_xz=self.ROOM_CENTROID)
        assert camera_no_bed == pytest.approx(camera_explicit)
        assert forward_no_bed == pytest.approx(forward_explicit)

    def test_pitch_is_solved_within_the_documented_search_range(self):
        _camera_pos, forward, _r, up = compute_camera(self.ROOM, floor_y=0.0, bed_centroid_xz=(0.5, 0.5))
        pitch_deg = math.degrees(math.asin(np.clip(forward[1], -1.0, 1.0)))
        lo, hi = PITCH_SEARCH_RANGE_DEG
        assert lo - 1e-6 <= pitch_deg <= hi + 1e-6
        assert up[1] > 0  # camera isn't upside-down

    def test_pitch_solve_puts_the_farthest_floor_point_near_85pct_of_frame_height(self):
        # A modestly-proportioned room (not the stretched-out 8x4 used above)
        # so the 85%-of-frame-height target is actually reachable within
        # PITCH_SEARCH_RANGE_DEG - see `_solve_pitch_for_target_screen_y`'s
        # docstring: a floor point's screen-Y has a finite ceiling as pitch
        # flattens, and that ceiling falls short of 85% for a large/elongated
        # room (the camera sits further back, so the far corner is very deep
        # relative to `camera_height_m`) - that's expected geometry, not
        # exercised by this particular test.
        room = np.array([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)])
        width, height = DEFAULT_IMAGE_SIZE
        camera_pos, forward, right, up = compute_camera(room, floor_y=0.0, bed_centroid_xz=(0.5, 0.5), image_height_px=height)
        # Recompute the same "farthest floor point along the view direction"
        # compute_camera itself targets, then re-project it with the module's
        # own _to_camera_space/_to_screen to check where it actually lands.
        camera_xz = np.array([camera_pos[0], camera_pos[2]])
        forward_xz = np.array([forward[0], forward[2]])
        forward_xz /= np.linalg.norm(forward_xz)
        depths = (room - camera_xz) @ forward_xz
        farthest_xz = room[int(np.argmax(depths))]
        target_world = np.array([farthest_xz[0], 0.0, farthest_xz[1]])

        focal_px = (height / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)
        cam_pt = _to_camera_space(target_world[None, :], camera_pos, right, up, forward)[0]
        screen_y = _to_screen(cam_pt[None, :], focal_px, width / 2.0, height / 2.0, 10.0 * max(width, height))[0][1]

        assert screen_y == pytest.approx(TARGET_FLOOR_SCREEN_FRAC * height, abs=3.0)  # within a few pixels

    def test_camera_may_land_outside_the_room_polygon(self):
        # Expected per task spec - not special-cased away. A bed anchored
        # deep in one corner pushes the camera out past the opposite corner.
        camera_pos, _f, _r, _u = compute_camera(self.ROOM, floor_y=0.0, bed_centroid_xz=(0.2, 0.2))
        assert camera_pos[0] > 8.0 or camera_pos[2] > 4.0


class TestBedAnchorFromScene:
    """`_bed_centroid_xz` is what `_render_scene` feeds `compute_camera` as
    the anchor - covers task 1's "find the bed object" step end to end
    against a real `build_scene()`-shaped Scene, including the no-bed
    fallback scenario."""

    WALL = WallPolygon(vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], area_m2=12.0)
    FLOOR_POLYGON = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

    def test_no_bed_object_returns_none(self):
        objects = [
            {
                "id": "chair_0",
                "label": "chair",
                "center_xy": (2.0, 1.5),
                "size_uv": (0.5, 0.5),
                "angle_rad": 0.0,
                "height": 0.9,
                "bbox_min_y": 0.0,
                "color_rgb": (200, 100, 50),
            }
        ]
        scene = build_scene([self.WALL], self.FLOOR_POLYGON, 0.0, 2.5, objects)
        assert _bed_centroid_xz(scene) is None

    def test_bed_object_footprint_centroid_is_found(self):
        objects = [
            {
                "id": "bed_0",
                "label": "bed",
                "center_xy": (1.0, 1.0),
                "size_uv": (1.5, 2.0),
                "angle_rad": 0.0,
                "height": 0.6,
                "bbox_min_y": 0.0,
                "color_rgb": (120, 80, 60),
            }
        ]
        scene = build_scene([self.WALL], self.FLOOR_POLYGON, 0.0, 2.5, objects)
        centroid = _bed_centroid_xz(scene)
        assert centroid is not None
        cx, cz = centroid
        assert cx == pytest.approx(1.0, abs=0.3)
        assert cz == pytest.approx(1.0, abs=0.3)


class TestNearWallCulling:
    """Task: a wall sitting between the camera and the room centroid should be
    culled entirely (not just drawn translucent) rather than reading as a
    near-opaque wall blocking the view into the room. Exercised directly at
    the `_iter_scene_node_faces` level (rather than pixel-diffing a full
    render) - simplest way to assert "which nodes get drawn"."""

    @staticmethod
    def _two_wall_scene():
        # A tiny wall segment sitting right between camera (0, -4) and room
        # centroid (0, 0) - vertex-mean midpoint (0, -1.0) - and another
        # sitting on the far side of the centroid - midpoint (0, 3.0).
        near_wall = WallPolygon(vertices=[(-1.0, -1.1), (1.0, -1.1), (1.0, -0.9), (-1.0, -0.9)], area_m2=0.4)
        far_wall = WallPolygon(vertices=[(-1.0, 2.9), (1.0, 2.9), (1.0, 3.1), (-1.0, 3.1)], area_m2=0.4)
        return build_scene([near_wall, far_wall], None, 0.0, 2.5, [])

    def test_wall_between_camera_and_centroid_is_excluded(self):
        scene = self._two_wall_scene()
        camera_pos_xz = (0.0, -4.0)
        room_centroid_xz = (0.0, 0.0)

        node_names = {name for name, _rgba, _tris in _iter_scene_node_faces(scene, camera_pos_xz, room_centroid_xz)}

        assert "wall_0" not in node_names  # near wall: culled
        assert "wall_1" in node_names  # far wall: still drawn

    def test_far_wall_keeps_its_normal_translucent_opacity(self):
        scene = self._two_wall_scene()
        camera_pos_xz = (0.0, -4.0)
        room_centroid_xz = (0.0, 0.0)

        by_name = {name: rgba for name, rgba, _tris in _iter_scene_node_faces(scene, camera_pos_xz, room_centroid_xz)}

        assert by_name["wall_1"][3] == int(round(0.30 * 255))  # unchanged 30% wall opacity

    def test_without_camera_info_no_culling_happens(self):
        # Omitting camera_pos_xz/room_centroid_xz (e.g. any other caller of
        # this generator) must preserve old behavior: every wall renders.
        scene = self._two_wall_scene()

        node_names = {name for name, _rgba, _tris in _iter_scene_node_faces(scene)}

        assert {"wall_0", "wall_1"} <= node_names


class TestTranslucentSoftFurnishings:
    """Task 2: curtain/drape/blind/mirror-classed objects render at the same
    30% opacity as walls (single-layer-per-node, same as `_composite_
    translucent_node` already gives walls - see `_iter_scene_node_faces`),
    while an ordinary object stays fully opaque. Class match is
    case-insensitive and derived from the id-derived node-name prefix (task
    2's derivation rule), not the `label` field."""

    WALL = WallPolygon(vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)], area_m2=12.0)
    FLOOR_POLYGON = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

    @classmethod
    def _scene(cls):
        objects = [
            {
                "id": "curtain_0",
                "label": "curtain",
                "center_xy": (0.2, 1.5),
                "size_uv": (0.1, 1.0),
                "angle_rad": 0.0,
                "height": 2.0,
                "bbox_min_y": 0.0,
                "color_rgb": (220, 220, 230),
            },
            {
                # Uppercase id prefix - class match must be case-insensitive.
                "id": "Mirror_2",
                "label": "mirror",
                "center_xy": (3.8, 1.5),
                "size_uv": (0.05, 0.8),
                "angle_rad": 0.0,
                "height": 1.2,
                "bbox_min_y": 0.5,
                "color_rgb": (200, 200, 210),
            },
            {
                "id": "chair_0",
                "label": "chair",
                "center_xy": (2.0, 1.5),
                "size_uv": (0.5, 0.5),
                "angle_rad": 0.0,
                "height": 0.9,
                "bbox_min_y": 0.0,
                "color_rgb": (200, 100, 50),
            },
        ]
        return build_scene([cls.WALL], cls.FLOOR_POLYGON, 0.0, 2.5, objects)

    def test_curtain_gets_wall_opacity_single_layer_treatment(self):
        scene = self._scene()
        alphas = [rgba[3] for name, rgba, _tris in _iter_scene_node_faces(scene) if name.startswith("curtain_0/visual/")]
        assert alphas  # the node actually exists in the scene
        assert all(a == int(round(WALL_OPACITY * 255)) for a in alphas)

    def test_mirror_class_match_is_case_insensitive(self):
        scene = self._scene()
        alphas = [rgba[3] for name, rgba, _tris in _iter_scene_node_faces(scene) if name.startswith("Mirror_2/visual/")]
        assert alphas
        assert all(a == int(round(WALL_OPACITY * 255)) for a in alphas)

    def test_ordinary_object_stays_fully_opaque(self):
        scene = self._scene()
        alphas = [rgba[3] for name, rgba, _tris in _iter_scene_node_faces(scene) if name.startswith("chair_0/visual/")]
        assert alphas
        assert all(a == 255 for a in alphas)

    def test_translucent_constants_come_from_the_shared_visibility_module(self):
        from scripts.msa import render_perspective as rp
        from scripts.msa import visibility as vis

        assert rp.TRANSLUCENT_OBJECT_CLASSES is vis.TRANSLUCENT_OBJECT_CLASSES
        assert rp.TRANSLUCENT_OPACITY == vis.TRANSLUCENT_OPACITY == WALL_OPACITY == 0.3


class TestOpaqueWallsMode:
    """T15i: `_render_scene(opaque_walls=True)` is the faithful preview of what Isaac
    renders of the presentation USD - stubs fully opaque, no near-wall culling here
    (the exporter already removed what it culled). A wall between the camera and a
    red object hides it in that mode and does not in the translucent default."""

    def _scene_with_wall_in_front(self):
        from scripts.msa.export_glb import build_scene

        wall = WallPolygon(vertices=[(1.0, -2.0), (1.1, -2.0), (1.1, 2.0), (1.0, 2.0)], area_m2=0.4)  # a stub across the view
        floor = [(0.0, -2.0), (6.0, -2.0), (6.0, 2.0), (0.0, 2.0)]
        obj = {"id": "box_0", "label": "box", "center_xy": (3.0, 0.0), "size_uv": (0.4, 0.4), "angle_rad": 0.0,
               "height": 0.4, "bbox_min_y": 0.0, "color_rgb": (255, 0, 0)}
        return build_scene([wall], floor, 0.0, 1.2, [obj]), np.asarray(floor, dtype=float)

    @staticmethod
    def _red_pixels(png):
        """Red-dominant pixels: the box itself (255, 0, 0) and the box seen through a
        30 % wall (~(235, 58, 61)); the wall colour (190, 195, 205) is not red-dominant."""
        a = np.asarray(Image.open(png).convert("RGB")).astype(int)
        return int(((a[..., 0] > 180) & (a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > 100)).sum())

    def test_opaque_walls_hide_what_a_translucent_wall_shows(self, tmp_path):
        from scripts.msa.render_perspective import _render_scene

        scene, floor = self._scene_with_wall_in_front()
        # eye 0.6 m up, 3 m before the wall, looking at the box: the ray crosses the 1.2 m wall
        eye = np.array([-2.0, 0.6, 0.0])
        fwd = np.array([5.0, -0.4, 0.0]); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        cam = (eye, fwd, right, up)
        _render_scene(scene, floor, 0.0, tmp_path / "opaque.png", image_size=(320, 180), camera=cam, opaque_walls=True)
        _render_scene(scene, floor, 0.0, tmp_path / "translucent.png", image_size=(320, 180), camera=cam, cull_near_walls=False)
        assert self._red_pixels(tmp_path / "opaque.png") == 0
        assert self._red_pixels(tmp_path / "translucent.png") > 0


@pytest.fixture(scope="module")
def bootstrap_glb(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("msa_render_perspective_fixture")
    inputs = load_object_inputs_from_hulls_json(FIXTURE_DIR / "scene_objects_hulls.json")
    run_bootstrap(FIXTURE_DIR, out_dir, object_inputs=inputs)
    return out_dir / "scene.glb"


class TestRenderHeroPerspectiveOnFixture:
    def test_renders_from_a_real_bootstrapped_scene_glb(self, bootstrap_glb, tmp_path):
        out_path = tmp_path / "modular_home_hero_34.png"

        render_hero_perspective_from_glb(bootstrap_glb, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        with Image.open(out_path) as img:
            width, height = img.size
        assert (width, height) == (1280, 960)
