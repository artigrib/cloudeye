"""Unit tests for scripts/audit/critic_vlm.py. Network calls to OpenRouter are always
mocked (monkeypatching `critic_vlm.httpx.post`) - no test in this file makes a real
network request or needs OPENROUTER_API_KEY set. Rendering tests use small synthetic
GLBs (trimesh, in-memory) rather than the real (not-checked-in) hero scene."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.audit import critic_vlm as cv
from scripts.msa import render_perspective as rp


# --------------------------------------------------------------------------- sample_camera_indices


class TestSampleCameraIndices:
    def test_spreads_evenly_across_the_full_range(self):
        idx = cv.sample_camera_indices(74, 6)
        assert idx[0] == 0
        assert idx[-1] == 73
        assert len(idx) == 6
        assert idx == sorted(idx)

    def test_deduplicates_when_fewer_cameras_than_k(self):
        idx = cv.sample_camera_indices(3, 6)
        assert idx == [0, 1, 2]

    def test_zero_cameras_yields_empty(self):
        assert cv.sample_camera_indices(0, 6) == []

    def test_single_camera_yields_one_index(self):
        assert cv.sample_camera_indices(1, 6) == [0]


# --------------------------------------------------------------------------- load_prompt


class TestLoadPrompt:
    def test_splits_system_and_user_sections(self):
        system, user_template = cv.load_prompt()

        assert "discrepancies" in system.lower()
        assert "{view}" in user_template
        assert "## System" not in system
        assert "## User" not in user_template

    def test_user_template_formats_with_view(self):
        _system, user_template = cv.load_prompt()
        formatted = user_template.format(view="view_030")
        assert "view_030" in formatted
        assert "{view}" not in formatted


# --------------------------------------------------------------------------- camera_pose_from_extrinsics


class TestCameraPoseFromExtrinsics:
    def test_identity_extrinsics_no_yaw_correction(self):
        identity = np.eye(4)
        pos, forward, right, up = cv.camera_pose_from_extrinsics(identity, 0.0, (0.0, 0.0))

        # OpenCV convention: local Z (forward) -> world [0,0,1]; local X (right) ->
        # world [1,0,0]; local Y is down, so world up = -[0,1,0] = [0,-1,0].
        assert np.allclose(pos, [0, 0, 0])
        assert np.allclose(forward, [0, 0, 1])
        assert np.allclose(right, [1, 0, 0])
        assert np.allclose(up, [0, -1, 0])

    def test_yaw_correction_rotates_position_and_direction_vectors(self):
        identity = np.eye(4)
        identity[0, 3] = 1.0  # camera at world x=1
        pos, forward, right, up = cv.camera_pose_from_extrinsics(identity, math.pi / 2, (0.0, 0.0))

        # A 90 deg rotation about the origin sends world x=1 to roughly z=1 (sign
        # depends on rotate_point_xz's convention - just check magnitude/axis moved).
        assert pos[0] == pytest.approx(0.0, abs=1e-9)
        assert abs(pos[2]) == pytest.approx(1.0, abs=1e-9)
        # forward [0,0,1] rotates to roughly [+-1, 0, 0]
        assert forward[1] == pytest.approx(0.0, abs=1e-9)
        assert abs(forward[0]) == pytest.approx(1.0, abs=1e-9)

    def test_matches_real_hero_camera_composition(self):
        """Regression check for the pose-derivation formula itself (not a rendering
        test): reuses the exact numbers from view 30 of the real hero scan that were
        verified by eye against per_view_png/view_030.png during development (a bed
        with a pillow at the head, wall behind, floor lamp to the right) - pinned
        here as a numeric sanity check for the transform, not the visual match
        itself (which isn't repo-testable without the real 271 MB point cloud)."""
        extr = [
            [0.7843039098549134, -0.06016105165940092, 0.6174528686539908, 0.0009595644373545316],
            [0.0604365068999711, -0.9831432200228374, -0.17255975967479437, 1.2687681745364485],
            [0.617425965215627, 0.17265598798690476, -0.7674471380006237, -0.0011851997286024694],
            [0.0, 0.0, 0.0, 1.0],
        ]
        yaw_correction_rad = 0.9246458985738557
        yaw_center = (1.656572111295865, -3.2482216613091954)

        pos, forward, right, up = cv.camera_pose_from_extrinsics(extr, yaw_correction_rad, yaw_center)

        # Camera should end up inside the exported room bbox (x in [-2.4, 4.6], z in
        # [-5.4, -1.1] - see geo5_out/objects.json's room_polygon) and pointing
        # roughly toward +y-tilted-down / into the room, not off to arbitrary space.
        assert -3.0 < pos[0] < 5.0
        assert -6.0 < pos[2] < -0.5
        assert np.linalg.norm(forward) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- render_pairs_for_scene (fallback path)


class TestRenderPairsForSceneFallback:
    def _make_glb(self, tmp_path: Path) -> Path:
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(extents=[2, 1, 2]), node_name="floor", transform=np.eye(4))
        path = tmp_path / "scene.glb"
        scene.export(str(path))
        return path

    def test_falls_back_to_lookat_when_no_extrinsics(self, tmp_path):
        glb_path = self._make_glb(tmp_path)
        per_view_dir = tmp_path / "per_view_png"
        per_view_dir.mkdir()
        from PIL import Image

        Image.new("RGB", (100, 100)).save(per_view_dir / "view_000.png")
        Image.new("RGB", (100, 100)).save(per_view_dir / "view_001.png")

        cameras_aligned = {"cameras": [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]}  # positions only, no rotation

        pairs = cv.render_pairs_for_scene(
            glb_path=glb_path,
            cameras_aligned=cameras_aligned,
            per_view_dir=per_view_dir,
            yaw_correction_rad=0.0,
            yaw_rotation_center_xy=(0.0, 0.0),
            room_centroid_xy=(0.0, 0.0),
            out_dir=tmp_path / "renders",
            k=2,
        )

        assert len(pairs) == 2
        for p in pairs:
            assert "look-at-room-centroid fallback" in p.detail
            assert p.right_image.is_file()

    def test_skips_a_view_with_no_matching_real_frame(self, tmp_path):
        glb_path = self._make_glb(tmp_path)
        per_view_dir = tmp_path / "per_view_png"
        per_view_dir.mkdir()
        from PIL import Image

        Image.new("RGB", (100, 100)).save(per_view_dir / "view_000.png")
        # view_001.png deliberately missing

        cameras_aligned = {"cameras": [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]}

        pairs = cv.render_pairs_for_scene(
            glb_path=glb_path,
            cameras_aligned=cameras_aligned,
            per_view_dir=per_view_dir,
            yaw_correction_rad=0.0,
            yaw_rotation_center_xy=(0.0, 0.0),
            room_centroid_xy=(0.0, 0.0),
            out_dir=tmp_path / "renders",
            k=2,
        )

        assert len(pairs) == 1
        assert pairs[0].view == "view_000"


# --------------------------------------------------------------------------- render_cloud_vs_hull_pair


class TestRenderCloudVsHullPair:
    def _write_ply(self, path: Path, n: int = 500) -> None:
        rng = np.random.default_rng(0)
        xyz = rng.uniform(low=[-1, 0.2, -1], high=[1, 1.0, 1], size=(n, 3)).astype("<f8")
        rgb = rng.integers(0, 255, size=(n, 3)).astype("u1")
        with open(path, "wb") as f:
            f.write(
                (
                    "ply\nformat binary_little_endian 1.0\ncomment Created by Open3D\n"
                    f"element vertex {n}\nproperty double x\nproperty double y\nproperty double z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                ).encode("ascii")
            )
            dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec = np.zeros(n, dtype=dtype)
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())

    def test_writes_both_images_and_a_detail_string(self, tmp_path):
        ply_path = tmp_path / "aligned_room.ply"
        self._write_ply(ply_path)
        room_polygon = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        objects = [{"id": "bed_0", "hull_xz": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        pair = cv.render_cloud_vs_hull_pair(
            ply_path=ply_path,
            room_polygon=room_polygon,
            objects=objects,
            yaw_correction_rad=0.0,
            yaw_rotation_center_xy=(0.0, 0.0),
            floor_y=0.0,
            out_dir=tmp_path / "renders",
        )

        assert pair.view == "cloud_top_view_vs_hull_overlay"
        assert pair.left_image.is_file()
        assert pair.right_image.is_file()
        assert "1 object hull outline" in pair.detail


# --------------------------------------------------------------------------- call_vlm_pair (mocked network)


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json_body


class TestCallVlmPair:
    def _pair(self, tmp_path: Path) -> cv.VlmPair:
        from PIL import Image

        left = tmp_path / "left.png"
        right = tmp_path / "right.png"
        Image.new("RGB", (10, 10)).save(left)
        Image.new("RGB", (10, 10)).save(right)
        return cv.VlmPair(view="view_000", left_image=left, right_image=right)

    def test_successful_call_parses_findings_and_sets_seed_temperature_in_body(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["url"] = url
            captured["body"] = json
            captured["headers"] = headers
            content = '{"findings": [{"object_id_or_region": "bed_0", "issue": "missing pillow", "severity": "medium"}]}'
            return _FakeResponse({"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

        monkeypatch.setattr(cv.httpx, "post", fake_post)

        pair = self._pair(tmp_path)
        record = cv.call_vlm_pair(pair, system="sys", user_template="check {view}", model="google/gemma-4-31b-it", temperature=0.0, seed=42)

        assert captured["body"]["seed"] == 42
        assert captured["body"]["temperature"] == 0.0
        assert captured["body"]["model"] == "google/gemma-4-31b-it"
        assert "test-key-not-real" in captured["headers"]["Authorization"]
        assert record.error is None
        assert len(record.findings) == 1
        assert record.findings[0]["view"] == "view_000"  # filled in from the pair
        assert record.findings[0]["object_id_or_region"] == "bed_0"
        assert record.usage == {"prompt_tokens": 100, "completion_tokens": 20}
        # PM decision (2026-09-07): every VLM finding is tagged source=vlm,
        # unverified=True - advisory only, never gating (mirrors critic_rules.py's
        # source=rule/unverified=False on every rule finding).
        assert record.findings[0]["source"] == "vlm"
        assert record.findings[0]["unverified"] is True

    def test_model_cannot_override_the_unverified_tag(self, tmp_path, monkeypatch):
        # Even if the model's own JSON claims otherwise, the tag must be forced by
        # this module, not merely defaulted.
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

        def fake_post(url, json, headers, timeout):
            content = '{"findings": [{"object_id_or_region": "bed_0", "issue": "x", "severity": "high", "unverified": false, "source": "rule"}]}'
            return _FakeResponse({"choices": [{"message": {"content": content}}], "usage": None})

        monkeypatch.setattr(cv.httpx, "post", fake_post)

        pair = self._pair(tmp_path)
        record = cv.call_vlm_pair(pair, system="sys", user_template="check {view}")

        assert record.findings[0]["source"] == "vlm"
        assert record.findings[0]["unverified"] is True

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        pair = self._pair(tmp_path)

        from scripts.msa.agent_loop.openrouter_client import OpenRouterError

        with pytest.raises(OpenRouterError):
            cv.call_vlm_pair(pair, system="sys", user_template="check {view}")

    def test_unparseable_response_is_recorded_as_an_error_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

        def fake_post(url, json, headers, timeout):
            return _FakeResponse({"choices": [{"message": {"content": "not json at all"}}], "usage": None})

        monkeypatch.setattr(cv.httpx, "post", fake_post)

        pair = self._pair(tmp_path)
        record = cv.call_vlm_pair(pair, system="sys", user_template="check {view}")

        assert record.error is not None
        assert record.findings == []

    def test_api_error_body_is_recorded_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

        def fake_post(url, json, headers, timeout):
            return _FakeResponse({"error": {"message": "rate limited"}})

        monkeypatch.setattr(cv.httpx, "post", fake_post)

        pair = self._pair(tmp_path)
        record = cv.call_vlm_pair(pair, system="sys", user_template="check {view}")

        assert record.error is not None
        assert "rate limited" in record.error

    def test_never_logs_the_api_key(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-value-xyz")

        def fake_post(url, json, headers, timeout):
            content = '{"findings": []}'
            return _FakeResponse({"choices": [{"message": {"content": content}}], "usage": None})

        monkeypatch.setattr(cv.httpx, "post", fake_post)

        pair = self._pair(tmp_path)
        record = cv.call_vlm_pair(pair, system="sys", user_template="check {view}")

        out = json.dumps(
            {"view": record.view, "findings": record.findings, "usage": record.usage, "error": record.error}
        )
        assert "super-secret-value-xyz" not in out
        captured = capsys.readouterr()
        assert "super-secret-value-xyz" not in captured.out
        assert "super-secret-value-xyz" not in captured.err


# --------------------------------------------------------------------------- run_vlm never touches scene data


class TestRunVlmReportOnly:
    def test_scene_input_files_are_untouched(self, tmp_path, monkeypatch):
        import hashlib

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        objects = {"room_polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]], "objects": []}
        (out_dir / "objects.json").write_text(json.dumps(objects))
        (out_dir / "scene_meta.json").write_text(json.dumps({"yaw_correction_rad": 0.0, "yaw_rotation_center_xy": [0, 0], "floor_y": 0.0}))
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(extents=[2, 1, 2]), node_name="floor", transform=np.eye(4))
        scene.export(str(out_dir / "scene.glb"))

        source_scene_dir = tmp_path / "source"
        (source_scene_dir / "per_view_png").mkdir(parents=True)
        (source_scene_dir / "cameras_aligned.json").write_text(json.dumps({"cameras": []}))

        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out_dir.iterdir() if p.is_file()}

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
        result = cv.run_vlm(out_dir=out_dir, source_scene_dir=source_scene_dir, render_out_dir=tmp_path / "renders")

        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out_dir.iterdir() if p.is_file() and p.name != "critic_vlm.json"}
        assert before == after
        assert result["n_pairs"] == 0  # no cameras, no ply -> no pairs, no network calls made
