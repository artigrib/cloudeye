"""arq worker process: `uv run arq app.worker.WorkerSettings`.

Runs as its own systemd unit (deploy/video-worker.service), separate from the API
process - a pipeline run takes minutes, not request-response time, and must survive an
API restart.
"""

import asyncio
import logging
import time
import uuid

from app.config import settings
from app.database import async_session_maker
from app.logging_setup import configure_logging, scene_logger
from app.models import Video
from app.queue import redis_settings
from app.services import pipeline_orchestrator, scene_service
from app.services.gpu_client import build_gpu_client

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    configure_logging("worker")
    ctx["gpu"] = build_gpu_client()
    logger.info("Worker started")


async def shutdown(ctx: dict) -> None:
    logger.info("Worker shutting down")


async def process_video(ctx: dict, video_id: str, rerun: bool = False) -> dict:
    """Run the pipeline for one video's scene, from `queued` through to `done`/`failed`.

    Idempotent: if a scene for this video already exists and is `done`, short-circuits
    without redoing any work (the deterministic arq job id already prevents two
    concurrent runs for the same video; this additionally covers a stale re-enqueue of a
    video that was already fully processed in the past). `rerun=True` (see
    app.queue.enqueue_process_video / POST /api/videos/{id}/rerun) bypasses that
    short-circuit deliberately - the whole point of a rerun is to reprocess a scene
    that's already `done` (or retry one that's `failed`), reusing its cached
    vocabulary rather than resampling a new one (see
    pipeline_orchestrator.resolve_vocab_override).
    """
    vid = uuid.UUID(video_id)
    log = scene_logger(__name__, vid)
    started_monotonic = time.monotonic()

    async with async_session_maker() as session:
        video = await session.get(Video, vid)
        if video is None:
            log.error("Video %s not found, cannot process", vid)
            return {"video_id": video_id, "status": "error", "detail": "video not found"}

        scene, created = await scene_service.get_or_create_scene(
            session, video_id=vid, project_id=video.project_id
        )
        if not rerun and not created and scene.status == "done":
            log.info("Scene %s already done, skipping", scene.id)
            return {"scene_id": str(scene.id), "status": "done", "skipped": True}

        try:
            await pipeline_orchestrator.run_pipeline_for_scene(session, scene, ctx["gpu"], rerun=rerun)
        except asyncio.CancelledError:
            # arq cancels the task on job_timeout; without this handler the scene would
            # stay stuck in "processing" forever, since mark_failed below never runs.
            #
            # A cancellation isn't necessarily a genuine job_timeout, though - it also
            # fires on an external SIGTERM (a service restart, a deploy), and the two
            # need different messages. Confirmed live: a video-worker restart cancelled
            # scene 2a9a882c at 291.62s while job_timeout_sec was 3600s, and this
            # handler used to unconditionally write "Job exceeded JOB_TIMEOUT_SEC" -
            # false, since it was nowhere near 3600s. Elapsed time since this function
            # started is the same clock arq's own job_timeout counts against, so a
            # cancellation arriving well short of that configured limit is not a timeout.
            elapsed = time.monotonic() - started_monotonic
            if elapsed >= settings.job_timeout_sec:
                message = f"Job exceeded JOB_TIMEOUT_SEC ({settings.job_timeout_sec}s)"
            else:
                message = f"job cancelled (worker shutdown or external interrupt) after {elapsed:.0f}s"
            async with async_session_maker() as fail_session:
                fresh = await scene_service.get_scene(fail_session, scene.id)
                if fresh is not None and fresh.status == "processing":
                    await scene_service.mark_failed(fail_session, fresh, message, completed=True)
            raise

    return {"scene_id": str(scene.id), "status": scene.status}


class WorkerSettings:
    functions = [process_video]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = redis_settings()
    job_timeout = settings.job_timeout_sec
    max_tries = 1  # no silent retries onto a busy single GPU
    max_jobs = 1  # one GPU instance, one job at a time
    keep_result = 86_400
