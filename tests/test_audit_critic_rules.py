"""Unit tests for scripts/audit/critic_rules.py - one group per check, all against
synthetic fixtures (a real-data smoke test lives separately, run manually against
var/scratch since that data isn't checked into the repo). See critic_rules.py's
module docstring for the full design rationale.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.audit import critic_rules as cr

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _square_room(half: float = 3.0) -> list[list[float]]:
    """An axis-aligned square room polygon, walls exactly on the x/z axes directions
    (0 and 90 deg) - x in [-half, half], z in [-half, half]."""
    return [[-half, -half], [half, -half], [half, half], [-half, half]]


# --------------------------------------------------------------------------- check_yaw_vs_wall


class TestCheckYawVsWall:
    def _scene(self, objects: list[dict], assets_placed=None) -> cr.SceneData:
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=[tuple(p) for p in _square_room()],
            objects=objects,
            objects_source="objects.json",
            assets_placed_by_id={a["id"]: a for a in assets_placed} if assets_placed else None,
            support_reason_by_id=None,
            glb_path=None,
        )
        scene.__dict__["_wall_segments_fallback"] = None
        return scene

    def test_flags_an_asset_rotated_far_off_the_nearest_wall(self):
        # Elongated object (aspect 2:1, passes the confidence gate), sitting right
        # against the z=-3 wall (direction 0 deg) but yaw_deg says 40 deg - should
        # flag with delta 40 (well past the 15 deg threshold and the fold doesn't
        # touch anything under 45).
        obj = {"id": "bed_0", "label": "bed", "center_xy": [0.0, -2.9], "size_uv": [2.0, 1.0], "hull_xz": None}
        placed = [{"id": "bed_0", "yaw_deg": 40.0, "center_xy": [0.0, -2.9]}]
        scene = self._scene([obj], placed)

        findings = cr.check_yaw_vs_wall(scene)

        assert len(findings) == 1
        f = findings[0]
        assert f["check"] == "yaw_vs_wall"
        assert f["object_id"] == "bed_0"
        assert f["measured"] == pytest.approx(40.0, abs=0.5)
        assert f["severity"] == "high"

    def test_does_not_flag_an_asset_aligned_with_its_wall(self):
        obj = {"id": "desk_0", "label": "desk", "center_xy": [0.0, -2.9], "size_uv": [2.0, 1.0], "hull_xz": None}
        placed = [{"id": "desk_0", "yaw_deg": 2.0, "center_xy": [0.0, -2.9]}]
        scene = self._scene([obj], placed)

        assert cr.check_yaw_vs_wall(scene) == []

    def test_perpendicular_to_the_wall_also_counts_as_aligned(self):
        # mod-90 fold: 90 deg off one wall direction is 0 deg off the perpendicular
        # wall - a very common, legitimate placement (e.g. a nightstand's short edge
        # against the wall) that must not be flagged.
        obj = {"id": "nightstand_0", "label": "nightstand", "center_xy": [0.0, -2.9], "size_uv": [2.0, 1.0], "hull_xz": None}
        placed = [{"id": "nightstand_0", "yaw_deg": 90.0, "center_xy": [0.0, -2.9]}]
        scene = self._scene([obj], placed)

        assert cr.check_yaw_vs_wall(scene) == []

    def test_skips_a_near_square_footprint_as_low_confidence(self):
        # aspect ~1.05 - below YAW_ASPECT_CONFIDENCE_GATE (1.3), even though the raw
        # yaw number would otherwise trip the threshold.
        obj = {"id": "lamp_0", "label": "lamp", "center_xy": [0.0, -2.9], "size_uv": [0.42, 0.40], "hull_xz": None}
        placed = [{"id": "lamp_0", "yaw_deg": 45.0, "center_xy": [0.0, -2.9]}]
        scene = self._scene([obj], placed)

        assert cr.check_yaw_vs_wall(scene) == []

    def test_falls_back_to_objects_angle_rad_when_no_assets_placed(self):
        obj = {
            "id": "bed_0",
            "label": "bed",
            "center_xy": [0.0, -2.9],
            "size_uv": [2.0, 1.0],
            "hull_xz": None,
            "angle_rad": math.radians(40.0),
        }
        scene = self._scene([obj], assets_placed=None)

        findings = cr.check_yaw_vs_wall(scene)

        assert len(findings) == 1
        assert "objects.angle_rad" in findings[0]["detail"]

    def test_no_room_polygon_and_no_wall_fallback_yields_no_findings(self):
        obj = {"id": "bed_0", "label": "bed", "center_xy": [0.0, -2.9], "size_uv": [2.0, 1.0], "hull_xz": None}
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=None,
            objects=[obj],
            objects_source="objects.json",
            assets_placed_by_id={"bed_0": {"id": "bed_0", "yaw_deg": 40.0, "center_xy": [0.0, -2.9]}},
            support_reason_by_id=None,
            glb_path=None,
        )
        scene.__dict__["_wall_segments_fallback"] = None

        assert cr.check_yaw_vs_wall(scene) == []


class TestMeshYawFromGlb:
    """Exercises `_mesh_yaw_from_glb` directly against an in-memory trimesh Scene -
    this is the function the diagonal-bed acceptance criterion hinges on (see
    critic_rules.py's docstring for the real-data story: the bed's hull/metadata
    yaw stayed "correct" while its two visual mesh PARTS were each rotated ~150 deg
    off - a bug only visible by reading the GLB's actual baked vertex positions)."""

    def _box_node(self, scene: trimesh.Scene, name: str, extents, center_xz, yaw_deg: float) -> None:
        box = trimesh.creation.box(extents=extents)
        # Negated: trimesh's rotation_matrix(angle, [0, 1, 0]) rotates the opposite
        # way round from this module's atan2(z, x) convention (confirmed empirically
        # here) - negating keeps this test helper's `yaw_deg` meaning "what
        # _mesh_yaw_from_glb should measure", independent of that library-internal
        # sign choice (critic_rules.py itself never assumes a sign, only that hull
        # and visual meshes were built with the SAME convention, which real exports
        # are).
        transform = trimesh.transformations.rotation_matrix(math.radians(-yaw_deg), [0, 1, 0])
        transform[0, 3], transform[2, 3] = center_xz
        scene.add_geometry(box, node_name=name, transform=transform)

    def test_measures_a_single_rotated_part(self):
        scene = trimesh.Scene()
        self._box_node(scene, "bed_0/visual/part_0", extents=[2.0, 0.5, 1.0], center_xz=(0.0, 0.0), yaw_deg=40.0)

        yaw = cr._mesh_yaw_from_glb(scene, "bed_0")

        assert yaw == pytest.approx(40.0, abs=0.5)

    def test_two_parts_each_rotated_the_same_way_average_to_that_rotation(self):
        # Mirrors the real bug: two split-bed halves, each INTERNALLY rotated ~150
        # deg, offset from each other along the bed's own long (correct) axis - the
        # per-part circular mean must still report ~150 deg, not the ~0 deg you'd
        # get by (wrongly) pooling all vertices from both parts into one point set
        # and fitting a single rectangle through the pair's offset.
        scene = trimesh.Scene()
        self._box_node(scene, "bed_0/visual/part_0", extents=[2.0, 0.5, 1.0], center_xz=(-1.0, 0.0), yaw_deg=150.0)
        self._box_node(scene, "bed_0/visual/part_1", extents=[2.0, 0.5, 1.0], center_xz=(1.0, 0.0), yaw_deg=150.0)

        yaw = cr._mesh_yaw_from_glb(scene, "bed_0")

        assert yaw == pytest.approx(150.0, abs=1.0)

    def test_returns_none_when_object_has_no_visual_nodes(self):
        scene = trimesh.Scene()
        self._box_node(scene, "other_0/visual/part_0", extents=[1.0, 1.0, 1.0], center_xz=(0.0, 0.0), yaw_deg=0.0)

        assert cr._mesh_yaw_from_glb(scene, "bed_0") is None


# --------------------------------------------------------------------------- check_bbox_vs_class_limit


class TestCheckBboxVsClassLimit:
    def _scene(self, objects: list[dict]) -> cr.SceneData:
        return cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=None,
            objects=objects,
            objects_source="objects.json",
            assets_placed_by_id=None,
            support_reason_by_id=None,
            glb_path=None,
        )

    def test_flags_an_oversized_bed_without_split_from_blob(self):
        obj = {"id": "bed_0", "label": "bed", "size_uv": [2.6, 1.5]}
        findings = cr.check_bbox_vs_class_limit(self._scene([obj]))

        assert len(findings) == 1
        assert findings[0]["severity"] in ("high", "medium")
        assert "not exempt" in findings[0]["detail"]

    def test_split_from_blob_bed_is_reported_but_marked_exempt(self):
        obj = {"id": "bed_0", "label": "bed", "size_uv": [4.3, 2.3], "split_from_blob": True, "split_k": 2}
        findings = cr.check_bbox_vs_class_limit(self._scene([obj]))

        assert len(findings) == 1
        assert findings[0]["severity"] == "info"
        assert "exempt" in findings[0]["detail"]

    def test_oversized_tv_flagged(self):
        obj = {"id": "television_0", "label": "television", "size_uv": [1.8, 0.3]}
        findings = cr.check_bbox_vs_class_limit(self._scene([obj]))
        assert len(findings) == 1
        assert findings[0]["object_id"] == "television_0"

    def test_oversized_lamp_height_flagged(self):
        obj = {"id": "lamp_0", "label": "lamp", "size_uv": [0.4, 0.4], "height": 2.1}
        findings = cr.check_bbox_vs_class_limit(self._scene([obj]))
        assert len(findings) == 1
        assert findings[0]["measured"] == 2.1

    def test_object_under_the_limit_is_not_flagged(self):
        obj = {"id": "bed_0", "label": "bed", "size_uv": [2.0, 1.5]}
        assert cr.check_bbox_vs_class_limit(self._scene([obj])) == []

    def test_unlisted_class_is_never_checked(self):
        obj = {"id": "curtain_0", "label": "curtain", "size_uv": [5.0, 5.0]}
        assert cr.check_bbox_vs_class_limit(self._scene([obj])) == []


# --------------------------------------------------------------------------- check_centroid_outside_room


class TestCheckCentroidOutsideRoom:
    def _scene(self, objects: list[dict], room_polygon=None) -> cr.SceneData:
        return cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=room_polygon if room_polygon is not None else [tuple(p) for p in _square_room()],
            objects=objects,
            objects_source="objects.json",
            assets_placed_by_id=None,
            support_reason_by_id=None,
            glb_path=None,
        )

    def test_flags_a_centroid_outside_the_polygon(self):
        obj = {"id": "chair_0", "label": "chair", "center_xy": [10.0, 10.0]}
        findings = cr.check_centroid_outside_room(self._scene([obj]))

        assert len(findings) == 1
        assert findings[0]["severity"] == "high"
        assert findings[0]["measured"] > 0

    def test_does_not_flag_a_centroid_inside_the_polygon(self):
        obj = {"id": "chair_0", "label": "chair", "center_xy": [0.0, 0.0]}
        assert cr.check_centroid_outside_room(self._scene([obj])) == []

    def test_no_room_polygon_yields_no_findings(self):
        obj = {"id": "chair_0", "label": "chair", "center_xy": [10.0, 10.0]}
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=None,
            objects=[obj],
            objects_source="objects.json",
            assets_placed_by_id=None,
            support_reason_by_id=None,
            glb_path=None,
        )
        assert cr.check_centroid_outside_room(scene) == []


# --------------------------------------------------------------------------- check_asset_centroid_vs_hull


class TestCheckAssetCentroidVsHull:
    def _scene(self, objects: list[dict], assets_placed: list[dict]) -> cr.SceneData:
        return cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=None,
            objects=objects,
            objects_source="objects.json",
            assets_placed_by_id={a["id"]: a for a in assets_placed},
            support_reason_by_id=None,
            glb_path=None,
        )

    def test_flags_a_centroid_shift_past_threshold(self):
        obj = {"id": "bed_0", "label": "bed", "hull_xz": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}  # centroid (0, 0)
        placed = [{"id": "bed_0", "center_xy": [0.5, 0.0]}]  # 0.5 m away

        findings = cr.check_asset_centroid_vs_hull(self._scene([obj], placed))

        assert len(findings) == 1
        assert findings[0]["measured"] == pytest.approx(0.5, abs=1e-6)
        assert findings[0]["severity"] == "high"

    def test_small_shift_under_threshold_is_not_flagged(self):
        obj = {"id": "bed_0", "label": "bed", "hull_xz": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}
        placed = [{"id": "bed_0", "center_xy": [0.05, 0.0]}]

        assert cr.check_asset_centroid_vs_hull(self._scene([obj], placed)) == []

    def test_no_assets_placed_yields_no_findings(self):
        obj = {"id": "bed_0", "label": "bed", "hull_xz": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=None,
            objects=[obj],
            objects_source="objects.json",
            assets_placed_by_id=None,
            support_reason_by_id=None,
            glb_path=None,
        )
        assert cr.check_asset_centroid_vs_hull(scene) == []


# --------------------------------------------------------------------------- check_missing_support_reason


class TestCheckMissingSupportReason:
    def _scene(self, objects: list[dict], support_reason_by_id) -> cr.SceneData:
        return cr.SceneData(
            out_dir=Path("/nonexistent"),
            room_polygon=None,
            objects=objects,
            objects_source="objects.json",
            assets_placed_by_id=None,
            support_reason_by_id=support_reason_by_id,
            glb_path=None,
        )

    def test_flags_an_object_missing_from_the_support_log(self):
        objects = [{"id": "bed_0", "label": "bed"}, {"id": "chair_0", "label": "chair"}]
        findings = cr.check_missing_support_reason(self._scene(objects, {"bed_0": "floor"}))

        assert len(findings) == 1
        assert findings[0]["object_id"] == "chair_0"

    def test_object_present_in_the_log_is_not_flagged(self):
        objects = [{"id": "bed_0", "label": "bed"}]
        assert cr.check_missing_support_reason(self._scene(objects, {"bed_0": "floor"})) == []

    def test_no_support_log_at_all_yields_no_findings(self):
        objects = [{"id": "bed_0", "label": "bed"}]
        assert cr.check_missing_support_reason(self._scene(objects, None)) == []


# --------------------------------------------------------------------------- load_scene fallback path


class TestLoadSceneGlbFallback:
    """Covers the bootstrap-only out dir case (no objects.json/assets_placed.json/
    support log) - what the rejected own-scenes out dirs actually look like."""

    def _write_bootstrap_only_glb(self, tmp_path: Path) -> Path:
        scene = trimesh.Scene()
        # Two walls, axis-aligned in different directions.
        wall0 = trimesh.creation.box(extents=[4.0, 2.0, 0.1])
        scene.add_geometry(wall0, node_name="wall_0", transform=np.eye(4))
        wall1_t = np.eye(4)
        wall1_t[:3, :3] = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])[:3, :3]
        scene.add_geometry(trimesh.creation.box(extents=[3.0, 2.0, 0.1]), node_name="wall_1", transform=wall1_t)
        # One object: hull + visual, both simple axis-aligned boxes.
        obj_t = np.eye(4)
        obj_t[0, 3], obj_t[2, 3] = 0.5, 0.5
        scene.add_geometry(trimesh.creation.box(extents=[0.4, 0.9, 0.4]), node_name="sink_0/collision/hull", transform=obj_t)
        scene.add_geometry(trimesh.creation.box(extents=[0.4, 0.9, 0.4]), node_name="sink_0/visual/part_0", transform=obj_t)
        glb_path = tmp_path / "scene.glb"
        scene.export(str(glb_path))
        return glb_path

    def test_reconstructs_objects_and_walls_from_the_glb(self, tmp_path):
        self._write_bootstrap_only_glb(tmp_path)

        scene = cr.load_scene(tmp_path)

        assert scene.objects_source == "glb"
        assert len(scene.objects) == 1
        assert scene.objects[0]["id"] == "sink_0"
        assert scene.room_polygon is None
        assert scene.__dict__["_wall_segments_fallback"] is not None
        assert len(scene.__dict__["_wall_segments_fallback"]) == 2
        assert any("no objects.json" in n for n in scene.notes)
        assert any("no room_polygon" in n for n in scene.notes)

    def test_run_rules_does_not_crash_on_the_fallback_path(self, tmp_path):
        self._write_bootstrap_only_glb(tmp_path)

        result = cr.run_rules(tmp_path)

        assert result["objects_source"] == "glb"
        assert result["n_objects"] == 1


# --------------------------------------------------------------------------- 13d diagonal-bed regression fixture


class TestDiagonalBedFixtureRegression:
    """Permanent regression coverage for the QUEUE.md M12 "bed orientation bug" -
    see tests/fixtures/critic/diagonal_bed/README.md for exactly what this fixture
    is, where it came from, and why it must never be regenerated from a fresh
    pipeline run (the bug it captures is already fixed on the geo-6 branch, so a
    fresh export would no longer reproduce it). This test is the thing that would
    fail if the fixture - or `check_yaw_vs_wall`/`_mesh_yaw_from_glb` - ever
    regressed such that this exact, already-shipped bug stopped being caught."""

    FIXTURE_DIR = FIXTURES_ROOT / "critic" / "diagonal_bed"

    def test_fixture_files_present(self):
        for name in ("objects.json", "scene_meta.json", "assets_placed.json", "report.json", "scene_assets_textured.glb"):
            assert (self.FIXTURE_DIR / name).is_file(), f"missing fixture file: {name}"

    def test_bed_0_hull_yaw_matches_the_wall_metadata_says_it_is_fine(self):
        objects_json = json.loads((self.FIXTURE_DIR / "objects.json").read_text())
        bed = next(o for o in objects_json["objects"] if o["id"] == "bed_0")
        assert math.degrees(bed["angle_rad"]) == pytest.approx(1.09, abs=0.05)

    def test_rules_flag_bed_0_yaw_vs_wall(self):
        result = cr.run_rules(self.FIXTURE_DIR)

        yaw_findings = [f for f in result["findings"] if f["check"] == "yaw_vs_wall" and f["object_id"] == "bed_0"]

        assert len(yaw_findings) == 1, (
            "the diagonal-bed fixture must be flagged by check_yaw_vs_wall - if this "
            "fails, either the fixture was regenerated (see its README - don't) or "
            "check_yaw_vs_wall/_mesh_yaw_from_glb regressed."
        )
        finding = yaw_findings[0]
        assert finding["measured"] > cr.YAW_WALL_THRESHOLD_DEG
        assert finding["measured"] == pytest.approx(25.03, abs=0.5)
        assert finding["severity"] in ("high", "medium")

    def test_mesh_yaw_from_glb_reports_the_visual_mesh_rotated_off_the_hull(self):
        # Direct evidence for the fixture's headline claim: the placed VISUAL mesh
        # is diagonal (~155 deg) while the collision hull it's supposed to match is
        # essentially axis-aligned (~1 deg, matching objects.json's angle_rad) - the
        # gap check_yaw_vs_wall's GLB read exists to catch.
        glb_scene = trimesh.load(str(self.FIXTURE_DIR / "scene_assets_textured.glb"), process=False)

        visual_yaw = cr._mesh_yaw_from_glb(glb_scene, "bed_0")
        assert visual_yaw == pytest.approx(154.97, abs=0.1)

        transform, geom_name = glb_scene.graph.get("bed_0/collision/hull")
        verts_world = trimesh.transformations.transform_points(glb_scene.geometry[geom_name].vertices, transform)
        from shapely.geometry import MultiPoint

        rect = MultiPoint(verts_world[:, [0, 2]]).convex_hull.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)[:4]
        lengths_and_angles = [
            (
                math.hypot(coords[(i + 1) % 4][0] - coords[i][0], coords[(i + 1) % 4][1] - coords[i][1]),
                math.degrees(math.atan2(coords[(i + 1) % 4][1] - coords[i][1], coords[(i + 1) % 4][0] - coords[i][0])) % 180.0,
            )
            for i in range(4)
        ]
        collision_yaw = max(lengths_and_angles, key=lambda t: t[0])[1]

        assert min(collision_yaw % 90.0, 90.0 - (collision_yaw % 90.0)) < 5.0
        assert abs(visual_yaw - collision_yaw) > cr.YAW_WALL_THRESHOLD_DEG


# --------------------------------------------------------------------------- report-only guarantee


class TestFindingsAreTaggedRuleAndUnverifiedFalse:
    """PM decision (2026-09-07): every rule finding must carry `source: "rule"`,
    `unverified: False`, so nothing downstream can mistake a deterministic,
    threshold-checked measurement for a VLM's advisory-only free-text read (which
    is tagged the opposite way in critic_vlm.py) - the rules engine is the gate."""

    def test_every_check_tags_its_findings_source_rule_unverified_false(self):
        # Real, non-trivial fixture that exercises multiple checks at once (yaw,
        # bbox exemption, asset-vs-hull) - see TestDiagonalBedFixtureRegression.
        result = cr.run_rules(FIXTURES_ROOT / "critic" / "diagonal_bed")
        assert result["findings"], "fixture should produce at least one finding"
        for finding in result["findings"]:
            assert finding["source"] == "rule"
            assert finding["unverified"] is False

    def test_yaw_vs_wall_tags_its_finding(self):
        obj = {"id": "bed_0", "label": "bed", "center_xy": [0.0, -2.9], "size_uv": [2.0, 1.0], "hull_xz": None, "angle_rad": math.radians(40.0)}
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"), room_polygon=[tuple(p) for p in _square_room()], objects=[obj],
            objects_source="objects.json", assets_placed_by_id=None, support_reason_by_id=None, glb_path=None,
        )
        scene.__dict__["_wall_segments_fallback"] = None
        [finding] = cr.check_yaw_vs_wall(scene)
        assert finding["source"] == "rule"
        assert finding["unverified"] is False

    def test_bbox_vs_class_limit_tags_its_finding(self):
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"), room_polygon=None, objects=[{"id": "bed_0", "label": "bed", "size_uv": [2.6, 1.5]}],
            objects_source="objects.json", assets_placed_by_id=None, support_reason_by_id=None, glb_path=None,
        )
        [finding] = cr.check_bbox_vs_class_limit(scene)
        assert finding["source"] == "rule"
        assert finding["unverified"] is False

    def test_centroid_outside_room_tags_its_finding(self):
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"), room_polygon=[tuple(p) for p in _square_room()],
            objects=[{"id": "chair_0", "label": "chair", "center_xy": [10.0, 10.0]}],
            objects_source="objects.json", assets_placed_by_id=None, support_reason_by_id=None, glb_path=None,
        )
        [finding] = cr.check_centroid_outside_room(scene)
        assert finding["source"] == "rule"
        assert finding["unverified"] is False

    def test_asset_centroid_vs_hull_tags_its_finding(self):
        obj = {"id": "bed_0", "label": "bed", "hull_xz": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"), room_polygon=None, objects=[obj],
            objects_source="objects.json", assets_placed_by_id={"bed_0": {"id": "bed_0", "center_xy": [0.5, 0.0]}},
            support_reason_by_id=None, glb_path=None,
        )
        [finding] = cr.check_asset_centroid_vs_hull(scene)
        assert finding["source"] == "rule"
        assert finding["unverified"] is False

    def test_missing_support_reason_tags_its_finding(self):
        scene = cr.SceneData(
            out_dir=Path("/nonexistent"), room_polygon=None,
            objects=[{"id": "bed_0", "label": "bed"}, {"id": "chair_0", "label": "chair"}],
            objects_source="objects.json", assets_placed_by_id=None, support_reason_by_id={"bed_0": "floor"}, glb_path=None,
        )
        [finding] = cr.check_missing_support_reason(scene)
        assert finding["source"] == "rule"
        assert finding["unverified"] is False


class TestNeverTouchesInputFiles:
    """Hard constraint from the task: 'no code path may modify scene data' - the
    critic opens everything read-only and writes only its own JSON/markdown."""

    def _build_fixture(self, tmp_path: Path) -> Path:
        out_dir = tmp_path / "scene_out"
        out_dir.mkdir()
        objects = {
            "room_polygon": _square_room(),
            "objects": [
                {
                    "id": "bed_0",
                    "label": "bed",
                    "center_xy": [0.0, -2.9],
                    "size_uv": [2.6, 1.2],
                    "hull_xz": [[-1.3, -3.5], [1.3, -3.5], [1.3, -2.3], [-1.3, -2.3]],
                    "angle_rad": 0.0,
                    "height": 0.6,
                    "bbox_min_y": 0.0,
                },
            ],
        }
        (out_dir / "objects.json").write_text(json.dumps(objects))
        (out_dir / "assets_placed.json").write_text(json.dumps([{"id": "bed_0", "yaw_deg": 0.0, "center_xy": [0.0, -2.9]}]))
        (out_dir / "report.json").write_text(json.dumps({"support_keeps": [{"id": "bed_0", "reason": "floor"}], "support_drops": []}))

        scene = trimesh.Scene()
        box_t = np.eye(4)
        box_t[0, 3], box_t[2, 3] = 0.0, -2.9
        scene.add_geometry(trimesh.creation.box(extents=[2.6, 0.6, 1.2]), node_name="bed_0/visual/part_0", transform=box_t)
        scene.add_geometry(trimesh.creation.box(extents=[2.6, 0.6, 1.2]), node_name="bed_0/collision/hull", transform=box_t)
        scene.export(str(out_dir / "scene.glb"))
        return out_dir

    def _hash_all(self, out_dir: Path) -> dict[str, str]:
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out_dir.iterdir())
            if p.is_file()
        }

    def test_input_files_are_byte_identical_after_a_full_run(self, tmp_path):
        out_dir = self._build_fixture(tmp_path)
        before = self._hash_all(out_dir)

        dest = cr.write_critic_rules_json(out_dir)

        after_inputs = {k: v for k, v in self._hash_all(out_dir).items() if k != "critic_rules.json"}
        assert before == after_inputs
        assert dest.name == "critic_rules.json"
        assert dest.exists()
        result = json.loads(dest.read_text())
        assert result["n_objects"] == 1
