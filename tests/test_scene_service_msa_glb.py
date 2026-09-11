"""Pure-unit test for scene_service.msa_glb_path/resolve_msa_glb - no DB, no network,
matches this test suite's convention (see tests/conftest.py's module docstring). The
router endpoint that uses these (app.routers.scenes.get_scene_msa_glb) is a thin
FileResponse wrapper around resolve_msa_glb() and isn't separately covered here, since
this repo's scenes router has no existing request-level test harness to follow
(checked before writing this)."""

import uuid
from pathlib import Path

import pytest

from app.config import settings
from app.services.scene_service import MSA_GLB_CANDIDATES, msa_glb_path, resolve_msa_glb, scene_dir


def test_msa_glb_path_is_under_the_scenes_own_directory():
    scene_id = uuid.uuid4()
    path = msa_glb_path(scene_id)
    assert path == scene_dir(scene_id) / "msa" / "scene.glb"
    assert path.parent.parent == Path(settings.upload_dir) / "scenes" / str(scene_id)


def test_msa_glb_path_is_stable_for_the_same_scene_id():
    scene_id = uuid.uuid4()
    assert msa_glb_path(scene_id) == msa_glb_path(scene_id)


def test_msa_glb_path_differs_per_scene_id():
    assert msa_glb_path(uuid.uuid4()) != msa_glb_path(uuid.uuid4())


@pytest.fixture
def scene_id(monkeypatch, tmp_path):
    """A fresh scene id whose scene_dir() resolves under tmp_path, so tests can place
    real files on disk without touching var/uploads."""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return uuid.uuid4()


def _touch(scene_id: uuid.UUID, filename: str) -> Path:
    path = scene_dir(scene_id) / "msa" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"glb-bytes")
    return path


def test_resolve_msa_glb_prefers_the_plain_bootstrap_export_over_textured(scene_id):
    """Reversed on 2026-09-09. The textured dollhouse export is 44,195,740 bytes with 23
    embedded textures on the hero scene against `scene.glb`'s 99,716 bytes with 0
    images, and the main viewer parses whatever this route returns on the main thread -
    that pairing froze the tab for 19.9 s of longtask per scene open. The dollhouse
    viewer that wanted the textured file is off the product surface and does not use
    this route (MsaViewerPage takes `?glb=` or a static fixture)."""
    bootstrap = _touch(scene_id, "scene.glb")
    _touch(scene_id, "scene_assets_textured.glb")

    resolved = resolve_msa_glb(scene_id)

    assert resolved == (bootstrap, "scene.glb")


def test_resolve_msa_glb_still_serves_a_textured_export_when_it_is_the_only_one(scene_id):
    textured = _touch(scene_id, "scene_assets_textured.glb")

    resolved = resolve_msa_glb(scene_id)

    assert resolved == (textured, "scene_assets_textured.glb")


def test_resolve_msa_glb_serves_the_bootstrap_export_when_it_is_the_only_one(scene_id):
    bootstrap = _touch(scene_id, "scene.glb")

    resolved = resolve_msa_glb(scene_id)

    assert resolved == (bootstrap, "scene.glb")


def test_msa_glb_candidates_put_the_imageless_bootstrap_export_first(scene_id):
    """The order itself is the fix - assert it directly, so a future re-sort has to
    argue with this test rather than silently re-freeze the viewer."""
    assert MSA_GLB_CANDIDATES[0] == "scene.glb"
    assert MSA_GLB_CANDIDATES.index("scene.glb") < MSA_GLB_CANDIDATES.index("scene_assets_textured.glb")


def test_resolve_msa_glb_returns_none_when_no_candidate_exists(scene_id):
    assert resolve_msa_glb(scene_id) is None


def test_resolve_msa_glb_follows_full_candidate_order(scene_id):
    # Only the two middle candidates present - resolution must still prefer
    # scene_textured.glb over scene_assets.glb per MSA_GLB_CANDIDATES' order.
    assert MSA_GLB_CANDIDATES.index("scene_textured.glb") < MSA_GLB_CANDIDATES.index("scene_assets.glb")
    _touch(scene_id, "scene_assets.glb")
    textured = _touch(scene_id, "scene_textured.glb")

    resolved = resolve_msa_glb(scene_id)

    assert resolved == (textured, "scene_textured.glb")
