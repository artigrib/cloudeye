"""arq job queue: connection settings, pool creation, and the enqueue helper.

The API process holds one `ArqRedis` pool (created in `main.py`'s lifespan) purely for
enqueueing; the worker process (`app/worker.py`) is a separate `arq` invocation that
consumes the queue.
"""

import logging
import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import Request

from app.config import settings

logger = logging.getLogger(__name__)


def redis_settings() -> RedisSettings:
    """Build arq's RedisSettings from the app's REDIS_URL."""
    return RedisSettings.from_dsn(settings.redis_url)


async def create_pool_from_settings() -> ArqRedis:
    """Create a connection pool for enqueueing jobs. Called once at API startup."""
    return await create_pool(redis_settings())


def get_arq_pool(request: Request) -> ArqRedis:
    """FastAPI dependency: the enqueue pool created once in `main.py`'s lifespan."""
    return request.app.state.arq


async def enqueue_process_video(
    pool: ArqRedis, video_id: uuid.UUID, *, rerun: bool = False
) -> str | None:
    """Enqueue a `process_video` job for this video.

    A first-time enqueue uses a deterministic job id (`process-video-{video_id}`) so
    arq itself refuses a duplicate enqueue while one is already queued or running -
    this is idempotency layer 1; the `scenes.video_id` unique DB constraint is layer
    2, for the case where the first job already completed and this is a genuine
    re-upload attempt.

    `rerun=True` (POST /api/videos/{id}/rerun) deliberately does NOT reuse that same
    deterministic id: arq's own dedup also refuses a duplicate `_job_id` for as long
    as the previous job's result is kept (`WorkerSettings.keep_result`, 24h) - the
    exact case a rerun needs to work *because* the previous job already completed. A
    fresh uuid4 suffix per rerun attempt sidesteps that instead of reasoning about
    `keep_result`'s window (and stays unique even across two reruns enqueued in the
    same instant, unlike a timestamp-based suffix). Passed through to `process_video`'s
    `rerun` param, which is what actually makes the job reuse this scene's cached
    vocabulary (see app.services.pipeline_orchestrator.resolve_vocab_override) instead
    of resampling a new one.

    Returns the arq job id, or None if a job for this video was already queued/running
    (not an error - the caller should treat this as "already in flight"; only possible
    for `rerun=False`, since a rerun's job id is always fresh).
    """
    job_id = (
        f"process-video-{video_id}"
        if not rerun
        else f"process-video-{video_id}-rerun-{uuid.uuid4().hex}"
    )
    job = await pool.enqueue_job(
        "process_video", str(video_id), rerun, _job_id=job_id
    )
    if job is None:
        logger.info("process_video job for video %s already queued or running", video_id)
        return None
    return job.job_id
