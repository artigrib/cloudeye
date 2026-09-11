"""Pure-unit test for scene_service.fit_prob_path - no DB, no network, same
convention as tests/test_scene_service_msa_glb.py (see that file's module
docstring for why the router endpoint itself isn't separately covered here)."""

import uuid
from pathlib import Path

import pytest

from app.config import settings
from app.services.scene_service import fit_prob_path, scene_dir


def test_fit_prob_path_is_under_the_scenes_own_msa_directory():
    scene_id = uuid.uuid4()
    path = fit_prob_path(scene_id)
    assert path == scene_dir(scene_id) / "msa" / "fit_prob.json"
    assert path.parent.parent == Path(settings.upload_dir) / "scenes" / str(scene_id)


def test_fit_prob_path_is_stable_for_the_same_scene_id():
    scene_id = uuid.uuid4()
    assert fit_prob_path(scene_id) == fit_prob_path(scene_id)


def test_fit_prob_path_differs_per_scene_id():
    assert fit_prob_path(uuid.uuid4()) != fit_prob_path(uuid.uuid4())


@pytest.fixture
def scene_id(monkeypatch, tmp_path):
    """A fresh scene id whose scene_dir() resolves under tmp_path, so tests can place
    real files on disk without touching var/uploads."""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return uuid.uuid4()


def test_fit_prob_path_not_a_file_when_absent(scene_id):
    assert not fit_prob_path(scene_id).is_file()


def test_fit_prob_path_is_a_file_once_written(scene_id):
    path = fit_prob_path(scene_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema": "cloudeye.audit.fit_prob/1"}')

    assert fit_prob_path(scene_id).is_file()
    assert fit_prob_path(scene_id).read_text().startswith('{"schema"')
