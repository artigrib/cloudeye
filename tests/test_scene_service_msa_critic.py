"""Pure-unit test for scene_service.msa_critic_json_path - no DB, no network, same
convention as tests/test_scene_service_msa_glb.py (see that file's module docstring
for why the router endpoint itself, app.routers.scenes.get_scene_critic, isn't
separately covered here: it's a thin FileResponse wrapper with no logic of its own
beyond the path lookup this file tests directly)."""

import uuid
from pathlib import Path

import pytest

from app.config import settings
from app.services.scene_service import msa_critic_json_path, scene_dir


def test_msa_critic_json_path_is_under_the_scenes_own_msa_directory():
    scene_id = uuid.uuid4()
    path = msa_critic_json_path(scene_id)
    assert path == scene_dir(scene_id) / "msa" / "critic.json"
    assert path.parent.parent == Path(settings.upload_dir) / "scenes" / str(scene_id)


def test_msa_critic_json_path_is_stable_for_the_same_scene_id():
    scene_id = uuid.uuid4()
    assert msa_critic_json_path(scene_id) == msa_critic_json_path(scene_id)


def test_msa_critic_json_path_differs_per_scene_id():
    assert msa_critic_json_path(uuid.uuid4()) != msa_critic_json_path(uuid.uuid4())


@pytest.fixture
def scene_id(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return uuid.uuid4()


def test_path_does_not_exist_until_a_critic_report_is_placed_there(scene_id):
    path = msa_critic_json_path(scene_id)
    assert not path.is_file()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"summary": {}}')

    assert path.is_file()
