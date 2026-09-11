"""enqueue_process_video()'s rerun handling: a normal enqueue keeps the deterministic
`process-video-{video_id}` job id (so arq's own dedup still refuses a concurrent
duplicate for a plain re-enqueue), while `rerun=True` always mints a fresh id instead -
deliberately sidestepping arq's `_job_id` dedup against the previous (already
completed) job, which is exactly the case a rerun needs to succeed despite (see
app.queue.enqueue_process_video's docstring on `keep_result`).

Pure-unit: a minimal fake in place of the real `ArqRedis` pool, no real Redis - matches
this suite's convention of faking the one collaborator method under test (see
test_pipeline_orchestrator_retain_per_view.py's `_RecordingGpuClient`).
"""

import uuid

import pytest

from app.queue import enqueue_process_video


class _FakeJob:
    def __init__(self, job_id: str):
        self.job_id = job_id


class _FakePool:
    """Records every enqueue_job call; `existing_job_id` simulates arq refusing a
    duplicate (returns None) when called with that id, matching real ArqRedis
    behavior for an already-queued/running job."""

    def __init__(self, *, existing_job_id: str | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self._existing_job_id = existing_job_id

    async def enqueue_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        job_id = kwargs.get("_job_id")
        if job_id == self._existing_job_id:
            return None
        return _FakeJob(job_id)


@pytest.fixture
def video_id() -> uuid.UUID:
    return uuid.uuid4()


async def test_normal_enqueue_uses_deterministic_job_id(video_id):
    pool = _FakePool()
    job_id = await enqueue_process_video(pool, video_id)
    assert job_id == f"process-video-{video_id}"
    (args, kwargs) = pool.calls[0]
    assert args == ("process_video", str(video_id), False)
    assert kwargs["_job_id"] == f"process-video-{video_id}"


async def test_rerun_enqueue_uses_a_fresh_job_id_each_time(video_id):
    pool = _FakePool()
    job_id_1 = await enqueue_process_video(pool, video_id, rerun=True)
    job_id_2 = await enqueue_process_video(pool, video_id, rerun=True)

    assert job_id_1 is not None and job_id_2 is not None
    assert job_id_1 != job_id_2
    for job_id in (job_id_1, job_id_2):
        assert job_id.startswith(f"process-video-{video_id}-rerun-")
    (args, kwargs) = pool.calls[0]
    assert args == ("process_video", str(video_id), True)


async def test_rerun_bypasses_dedup_that_would_block_a_normal_enqueue(video_id):
    """The exact scenario this exists for: a previous process_video job for this video
    already ran to completion under the deterministic id (still within arq's
    keep_result window) - a plain re-enqueue would be refused, but a rerun must not
    be."""
    pool = _FakePool(existing_job_id=f"process-video-{video_id}")

    blocked = await enqueue_process_video(pool, video_id)
    assert blocked is None  # normal enqueue: correctly refused, matches real arq dedup

    rerun_job_id = await enqueue_process_video(pool, video_id, rerun=True)
    assert rerun_job_id is not None  # rerun: not refused, fresh id sidesteps the dedup


async def test_already_in_flight_still_reported_for_normal_enqueue(video_id):
    pool = _FakePool(existing_job_id=f"process-video-{video_id}")
    assert await enqueue_process_video(pool, video_id) is None
