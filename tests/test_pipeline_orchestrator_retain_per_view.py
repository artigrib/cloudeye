"""_retain_per_view_on_failure: fetch per_view/*.npz for a FAILED job before cleanup()
deletes them from the GPU box, regardless of settings.retain_per_view_artifacts (which
only governs the success path - see pipeline_orchestrator.py's module docstring on
this function and docs/DECISIONS.md's 2026-09-04 entry for the real jobs, room.mp4 and
street1.mp4, that couldn't be diagnosed after the fact because this data was already
gone).

Pure-unit, no DB, no GPU, no network - matches this test suite's existing convention
(see conftest.py). The real GpuClient is a typing.Protocol (structural), so a minimal
fake implementing just the one method under test is enough; Scene is a plain
SQLAlchemy model instantiated standalone (never touches a session) purely for its
`.id` attribute, which is all scene_service.scene_dir() and this function need.
"""

import uuid

import pytest

from app.models import Scene
from app.services.pipeline_orchestrator import _retain_per_view_on_failure


class _FakeLog:
    def warning(self, *args, **kwargs):
        pass


class _RecordingGpuClient:
    """Records every fetch_results call; raises if configured to, to exercise the
    best-effort/never-raises contract."""

    def __init__(self, *, raise_on_fetch: Exception | None = None):
        self.fetch_calls: list[tuple[str, object, bool]] = []
        self._raise_on_fetch = raise_on_fetch

    async def fetch_results(self, job_id, dest, *, include_per_view=False):
        self.fetch_calls.append((job_id, dest, include_per_view))
        if self._raise_on_fetch is not None:
            raise self._raise_on_fetch
        return dest


@pytest.fixture
def scene() -> Scene:
    return Scene(id=uuid.uuid4())


async def test_fetches_per_view_regardless_of_default_flag(scene):
    client = _RecordingGpuClient()
    await _retain_per_view_on_failure(client, str(scene.id), scene, _FakeLog())

    assert len(client.fetch_calls) == 1
    job_id, dest, include_per_view = client.fetch_calls[0]
    assert job_id == str(scene.id)
    assert include_per_view is True
    assert str(scene.id) in str(dest)


async def test_never_raises_even_if_fetch_fails(scene):
    """Best-effort, matching fetch_logs' own contract: called from failure-handling
    paths, must never itself raise and mask the real error (a job that failed before
    push_job ever ran has no remote job dir at all - that's an expected, not
    exceptional, case here)."""
    client = _RecordingGpuClient(raise_on_fetch=RuntimeError("rsync: no such file or directory"))
    await _retain_per_view_on_failure(client, str(scene.id), scene, _FakeLog())
    assert len(client.fetch_calls) == 1  # attempted, then swallowed the error
