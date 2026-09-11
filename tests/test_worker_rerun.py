"""process_video()'s rerun bypass: a plain re-enqueue of an already-`done` scene
short-circuits without doing any work (existing idempotency), but `rerun=True` must
bypass that short-circuit - otherwise POST /api/videos/{id}/rerun would be a no-op on
exactly the scenes it's meant for (the whole point of a rerun is reprocessing a scene
that's already `done`).

Pure-unit: fakes every collaborator process_video touches (DB session, scene_service,
pipeline_orchestrator) rather than hitting a real database, matching this suite's
no-DB convention (see conftest.py) while still exercising the real control flow in
app/worker.py.
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from app.models import Scene, Video
from app.worker import process_video


class _FakeSessionCM:
    """Stands in for `async_session_maker()`'s `async with ... as session:` usage -
    `session` itself is never touched beyond being passed through to the (also faked)
    collaborators, so a bare object is enough."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, video):
        self._video = video

    async def get(self, model, vid):
        return self._video


@pytest.fixture
def video_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def video(video_id) -> Video:
    v = Video(filename="room.mp4", filepath="/tmp/room.mp4", size_bytes=1, format="mp4")
    v.id = video_id
    v.project_id = uuid.uuid4()
    return v


def _patch_worker(monkeypatch, *, video, scene, created):
    session = _FakeSession(video)
    monkeypatch.setattr("app.worker.async_session_maker", lambda: _FakeSessionCM(session))

    get_or_create = AsyncMock(return_value=(scene, created))
    monkeypatch.setattr("app.worker.scene_service.get_or_create_scene", get_or_create)

    run_pipeline = AsyncMock(return_value=None)
    monkeypatch.setattr("app.worker.pipeline_orchestrator.run_pipeline_for_scene", run_pipeline)
    return run_pipeline


async def test_plain_reenqueue_of_done_scene_skips(monkeypatch, video, video_id):
    scene = Scene(id=uuid.uuid4(), status="done")
    run_pipeline = _patch_worker(monkeypatch, video=video, scene=scene, created=False)

    result = await process_video({"gpu": object()}, str(video_id))

    assert result == {"scene_id": str(scene.id), "status": "done", "skipped": True}
    run_pipeline.assert_not_called()


async def test_rerun_of_done_scene_does_not_skip(monkeypatch, video, video_id):
    scene = Scene(id=uuid.uuid4(), status="done")
    run_pipeline = _patch_worker(monkeypatch, video=video, scene=scene, created=False)

    result = await process_video({"gpu": object()}, str(video_id), rerun=True)

    run_pipeline.assert_awaited_once()
    args, kwargs = run_pipeline.await_args
    assert kwargs.get("rerun") is True
    assert result == {"scene_id": str(scene.id), "status": scene.status}


async def test_rerun_of_failed_scene_does_not_skip(monkeypatch, video, video_id):
    scene = Scene(id=uuid.uuid4(), status="failed")
    run_pipeline = _patch_worker(monkeypatch, video=video, scene=scene, created=False)

    await process_video({"gpu": object()}, str(video_id), rerun=True)

    run_pipeline.assert_awaited_once()


async def test_default_rerun_false_preserves_prior_behavior(monkeypatch, video, video_id):
    """Sanity check that the new `rerun` param defaults to False, so every existing
    non-rerun call site (upload-triggered enqueue) behaves exactly as before."""
    scene = Scene(id=uuid.uuid4(), status="queued")
    run_pipeline = _patch_worker(monkeypatch, video=video, scene=scene, created=True)

    await process_video({"gpu": object()}, str(video_id))

    run_pipeline.assert_awaited_once()
    _, kwargs = run_pipeline.await_args
    assert kwargs.get("rerun") is False
