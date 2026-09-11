"""Drives one scene's GPU pipeline run from start to finish: push the job, poll status,
fetch results, ingest, validate, and mark the scene done or failed.

This is the real implementation (the step-2 stub that just marked scenes failed with a
"not implemented yet" message has been replaced). Order matters: nothing is written to
`scene_objects` or the scene's artifact fields until `scene_validator.validate_scene`
has passed - a scene that fails validation ends up `failed` with zero DB rows for it,
never a half-populated "done" scene.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.logging_setup import scene_file_log, scene_logger
from app.models import Scene, Video
from app.robots import resolve_radius_m
from app.services import (
    model_catalog,
    pathfinding,
    scene_ingest,
    scene_service,
    vertex_auth,
)
from app.services.gpu_client import (
    GpuClient,
    GpuCommandError,
    GpuJobStatus,
    GpuUnavailableError,
    RemoteJobFailedError,
)
from app.services.pathfinding import ObjectFootprint, OccupancyGrid
from app.services.scene_validator import find_oversized_objects, validate_scene
from app.services.stage_timings import refresh_stage_timings_from_log

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"done", "failed"}
_RECOVERABLE_ERRORS = (
    GpuUnavailableError,
    GpuCommandError,
    RemoteJobFailedError,
    scene_ingest.ArtifactParseError,
)


class PipelineError(Exception):
    """Raised for pipeline failures that aren't one of the more specific GPU/artifact
    exceptions (e.g. a deadline exceeded, or a failed validation report)."""


async def run_pipeline_for_scene(
    session: AsyncSession, scene: Scene, client: GpuClient, *, rerun: bool = False
) -> None:
    """`rerun=True` reprocesses a scene that has already run before (normally only
    called for a `done`/`failed` scene - a fresh `queued` one has no prior vocabulary
    to reuse anyway) and, unless the video has its own explicit `vocab_override`,
    reuses this scene's previously-resolved vocabulary instead of making a fresh
    OpenRouter/Vertex call - see resolve_vocab_override()."""
    log = scene_logger(__name__, scene.id)
    job_id = str(scene.id)

    video = await session.get(Video, scene.video_id)
    if video is None:
        await scene_service.mark_failed(session, scene, "video record missing for this scene")
        return

    await scene_service.mark_processing(session, scene)
    log.info("processing started for video %s%s", video.filename, " (rerun)" if rerun else "")

    with scene_file_log(scene.id):
        try:
            await _run(session, scene, video, client, job_id, log, rerun=rerun)
        except (*_RECOVERABLE_ERRORS, PipelineError) as exc:
            log.error("pipeline failed: %s", exc)
            try:
                await client.fetch_logs(job_id, scene_service.scene_dir(scene.id) / "logs")
            except Exception:
                log.warning("could not fetch remote logs after failure", exc_info=True)
            await _retain_per_view_on_failure(client, job_id, scene, log)
            await scene_service.mark_failed(session, scene, str(exc))
        except Exception as exc:
            # Catch-all for anything not already covered above - a bug in our own
            # post-processing (ingestion, robot-start resolution, artifact writing),
            # not a GPU/network/validation failure with its own clear exception type.
            # Without this, such an exception propagates past process_video's
            # CancelledError-only handler in worker.py and leaves the scene stuck in
            # "processing" forever - confirmed to actually happen (scene 5448cd08,
            # 2026-08-31: a ValueError in pathfinding.resolve_robot_start stranded a
            # scene whose GPU pipeline had already succeeded, with no terminal status
            # at all). Logged with a full traceback since reaching this branch means
            # something genuinely unanticipated happened. (asyncio.CancelledError is a
            # BaseException, not an Exception, so it's unaffected by this clause - arq's
            # own cancellation handling in worker.py still sees it.)
            log.error("pipeline failed with an unexpected %s: %s", type(exc).__name__, exc, exc_info=True)
            try:
                await client.fetch_logs(job_id, scene_service.scene_dir(scene.id) / "logs")
            except Exception:
                log.warning("could not fetch remote logs after failure", exc_info=True)
            await _retain_per_view_on_failure(client, job_id, scene, log)
            await scene_service.mark_failed(
                session, scene, f"unexpected {type(exc).__name__}: {exc}", completed=True
            )
        finally:
            try:
                await client.cleanup(job_id)
            except Exception:
                log.warning("remote cleanup failed (non-fatal)", exc_info=True)

    # Persist the final stage_timings.json (see app.services.stage_timings) now that
    # worker.log is complete - outside the `with scene_file_log` block above so the log
    # file has been flushed/closed first. Covers both success and every failure branch
    # above (none of them re-raise past this point). Best-effort/non-fatal by
    # construction (refresh_stage_timings_from_log never raises) - a scene that failed
    # before any stage line was ever written just ends up with an empty list, same as
    # scripts/nightly_batch.py's existing handling of that case.
    refresh_stage_timings_from_log(scene_service.scene_dir(scene.id))


async def _retain_per_view_on_failure(client: GpuClient, job_id: str, scene: Scene, log) -> None:
    """Best-effort: fetch per_view/*.npz + per_view_png/*.png for a FAILED job before
    the `finally` block's cleanup() deletes them from the GPU box, regardless of
    settings.retain_per_view_artifacts - that setting only governs the SUCCESS path's
    fetch_results() call in `_run` (kept False by default there since it's real disk
    cost most successful runs don't need). A failure is exactly when this data is
    needed most: the 2026-09-04 camera-convention/floor-selection investigations
    (docs/DECISIONS.md) both hit real jobs (room.mp4, street1.mp4) where per_view had
    already been deleted by this same cleanup() by the time anyone went looking,
    making the failure undiagnosable after the fact.

    Never raises - mirrors fetch_logs' own "called from failure-handling paths, must
    never itself mask the real error" contract. A job that failed before push_job ever
    ran (no remote job dir at all) is expected to fail this fetch; that's normal, not
    a bug in this function.
    """
    try:
        await client.fetch_results(job_id, scene_service.scene_dir(scene.id), include_per_view=True)
    except Exception:
        log.warning("could not fetch per_view artifacts after failure (non-fatal)", exc_info=True)


def resolve_vocab_override(video: Video, scene: Scene, *, rerun: bool) -> list | None:
    """What to put in this job's `vocab_override` param, if anything - the one thing
    that determines whether gpu/stage_vocab.py makes a fresh OpenRouter/Vertex call at
    all (see its module docstring; any truthy `vocab_override` skips the call
    entirely, regardless of `vocab_provider`).

    Precedence:
    1. `video.vocab_override` - an explicit fixed vocabulary set at upload time (see
       Video.vocab_override's docstring) always wins, rerun or not. A user who set
       this deliberately chose a specific vocabulary; a rerun shouldn't quietly
       replace it with whatever the scene happened to resolve to on its own last run.
    2. Otherwise, if this is a `rerun` and the scene already has a previously-resolved
       vocabulary cached (`scene.vocab_entries` - written by _run() after a prior
       successful run, from vocab.json's "objects" field via
       scene_ingest.read_vocab_entries()), reuse it. This is the actual "vocab cache"
       for reruns: without it, every rerun would independently re-sample the
       vocabulary (see gpu/stage_vocab.py's docstring on that call's own measured
       run-to-run non-determinism), making repeated runs of the same video
       incomparable - reusing the same vocabulary is what makes a rerun's other
       results (object counts/positions, occupancy, validation) actually comparable
       to the run before it.
    3. Otherwise None - a normal fresh vocabulary call (params["vocab_provider"]
       picks which provider, defaulting to Vertex primary with automatic OpenRouter
       fallback - see the vocab-provider dispatch below), same as before this existed.

    A `rerun` on a scene with no cached vocabulary yet (e.g. its only prior attempt
    failed before stage_vocab.py ever wrote vocab.json) falls through to a fresh call
    same as a first run - there's nothing to reuse.
    """
    if video.vocab_override:
        return video.vocab_override
    if rerun and scene.vocab_entries:
        return scene.vocab_entries
    return None


async def _run(
    session: AsyncSession,
    scene: Scene,
    video: Video,
    client: GpuClient,
    job_id: str,
    log,
    *,
    rerun: bool = False,
) -> None:
    secrets = {}
    if settings.openrouter_api_key:
        secrets["OPENROUTER_API_KEY"] = settings.openrouter_api_key

    params = {
        "keyframe_fps": settings.keyframe_fps,
        "max_keyframes": settings.max_keyframes,
        "openrouter_model": settings.openrouter_model_vision,
        "openrouter_base_url": settings.openrouter_base_url,
        "glb_max_points": settings.glb_max_points,
        "regularize_planes": settings.regularize_planes,
    }
    # Per-upload overrides (see Video.blur_threshold docstring). Omitted entirely when
    # unset, so stage_keyframes.py falls back to its own defaults unchanged.
    if video.blur_threshold is not None:
        params["blur_threshold"] = video.blur_threshold
    if video.similarity_threshold is not None:
        params["similarity_threshold"] = video.similarity_threshold

    vocab_override = resolve_vocab_override(video, scene, rerun=rerun)
    if vocab_override:
        params["vocab_override"] = vocab_override
        if rerun:
            log.info("rerun: reusing cached vocabulary (%d entries), skipping vocabulary call", len(vocab_override))
    elif video.vocab_provider == model_catalog.VOCAB_PROVIDER_OPENROUTER:
        # Explicit opt-out of Vertex (the Models selector) - honor it exactly, no
        # Vertex token minted, no fallback needed since there's nothing to fall back
        # from. Unchanged from before this feature.
        params["vocab_provider"] = model_catalog.VOCAB_PROVIDER_OPENROUTER
    else:
        # Vertex is the default vocab provider as of 2026-09-07 (video.vocab_provider
        # is None, or explicitly "vertex") - primary, with gpu/stage_vocab.py's
        # dispatch_vocab() automatically falling back to OpenRouter on any *Vertex API*
        # failure (docs/DECISIONS.md - supersedes the earlier "no fallback of any
        # kind" call, which was about not fabricating a vocabulary, not about refusing
        # a second real provider). If even minting the *local* ADC token fails, degrade
        # to a plain OpenRouter job up front instead of failing the whole scene over a
        # token-mint hiccup - logged either way, never silent.
        try:
            secrets["VERTEX_ACCESS_TOKEN"] = await vertex_auth.get_access_token()
            secrets["GCP_PROJECT_ID"] = settings.gcp_project_id or ""
            params["vocab_provider"] = model_catalog.VOCAB_PROVIDER_VERTEX
            params["vertex_vocab_model"] = model_catalog.VERTEX_VOCAB_MODEL
            log.info("vocab provider: vertex (primary), openrouter fallback available")
        except vertex_auth.VertexAuthError as exc:
            log.warning("vertex token mint failed (%s) - using openrouter for vocab", exc)
            params["vocab_provider"] = model_catalog.VOCAB_PROVIDER_OPENROUTER

    await client.ensure_pipeline_code()
    await client.push_job(job_id, Path(video.filepath), params, secrets)
    await client.start_pipeline(job_id)
    log.info("job pushed and started")

    status = await _poll_until_terminal(client, job_id, deadline_sec=settings.job_timeout_sec, log=log)
    if status.state != "done":
        raise RemoteJobFailedError(status)

    local_dir = scene_service.scene_dir(scene.id)
    await client.fetch_results(
        job_id, local_dir, include_per_view=settings.retain_per_view_artifacts
    )
    log.info("results fetched to %s", local_dir)

    objects = scene_ingest.parse_scene_objects(
        local_dir / "scene_objects" / "scene_objects.json", local_dir
    )
    alignment = scene_ingest.read_alignment(local_dir)
    grid = scene_ingest.read_grid_metadata(local_dir)
    cameras = scene_ingest.read_camera_track(local_dir)
    robot_start = scene_ingest.choose_robot_start(cameras, grid)
    vocab_source = scene_ingest.read_vocab_source(local_dir)
    vocab_model = scene_ingest.read_vocab_model(local_dir)
    # Cache this run's resolved vocabulary on the scene itself so a later `rerun` can
    # reuse it (resolve_vocab_override above) instead of making a fresh vocabulary
    # call - see Scene.vocab_entries' docstring. Written every run, including a rerun
    # that itself reused a cached vocabulary (vocab.json's "objects" field round-trips
    # unchanged in that case, source="override"), so the cache always reflects
    # whatever vocabulary this scene most recently actually ran with.
    vocab_entries = scene_ingest.read_vocab_entries(local_dir)

    report = validate_scene(alignment, objects, grid, cameras)
    if not report.passed:
        raise PipelineError(f"scene failed validation: {report.summary()}")
    log.info("validation passed (%d checks)", len(report.checks))

    oversized = find_oversized_objects(objects)
    oversized_object_warning = "; ".join(oversized) if oversized else None
    if oversized:
        log.warning("oversized object(s) flagged (non-fatal): %s", oversized_object_warning)

    occupancy_cells = np.load(local_dir / "occupancy.npy")
    footprints = [
        ObjectFootprint(
            x=o.pos[0], z=o.pos[2],
            bbox_min_x=o.bbox_min[0], bbox_min_z=o.bbox_min[2],
            bbox_max_x=o.bbox_max[0], bbox_max_z=o.bbox_max[2],
        )
        for o in objects
        if not o.is_fragment
    ]
    # robot_start is resolved once at scene creation, persisted, and reused by every
    # later command regardless of which platform is later selected in the UI - there's
    # no per-request platform selection at this point in the pipeline yet, so this uses
    # the registry's default platform (resolve_radius_m(None, None), same value as the
    # old settings.robot_radius_m read it replaces).
    start_resolution = pathfinding.resolve_robot_start(
        OccupancyGrid(cells=occupancy_cells, meta=grid),
        robot_start,
        footprints,
        robot_radius_m=resolve_radius_m(None, None),
    )
    if start_resolution.relocated:
        log.warning(
            "robot_start relocated from camera-track position %s to %s "
            "(reachability %d/%d -> %.0f%%)",
            robot_start, start_resolution.point,
            start_resolution.reachable_count, start_resolution.total_objects,
            start_resolution.reachable_fraction * 100,
        )
    else:
        log.info(
            "robot_start resolved at %s (snapped=%s, reachability %d/%d -> %.0f%%)",
            start_resolution.point, start_resolution.snapped,
            start_resolution.reachable_count, start_resolution.total_objects,
            start_resolution.reachable_fraction * 100,
        )
    robot_start = start_resolution.point

    await scene_service.apply_scene_artifacts(
        session,
        scene,
        mesh_path=str(local_dir / "scene_points.glb"),
        pointcloud_path=str(local_dir / "aligned_room.ply"),
        occupancy_path=str(local_dir / "occupancy.npy"),
        map_preview_path=str(local_dir / "map_preview.png"),
        transform_path=str(local_dir / "alignment_transform.npz"),
        grid_resolution=grid.resolution,
        grid_origin_x=grid.origin_x,
        grid_origin_z=grid.origin_z,
        grid_width=grid.width,
        grid_height=grid.height,
        # Always 0.0 by construction - the GPU pipeline normalizes the aligned cloud so
        # the floor sits at Y=0 (see scene_validator's module docstring). What varies is
        # ceiling_y, the ceiling's height above that floor.
        floor_y=0.0,
        ceiling_y=alignment.ceiling_y,
        is_reflected=alignment.is_reflection,
        points_below_floor_frac=alignment.points_below_floor_frac,
        reflective_floor_suspected=alignment.reflective_floor_suspected,
        oversized_object_warning=oversized_object_warning,
        vocab_source=vocab_source,
        vocab_model=vocab_model,
        vocab_entries=vocab_entries,
        robot_start_x=robot_start[0],
        robot_start_z=robot_start[1],
    )
    await scene_service.replace_scene_objects(
        session,
        scene,
        [
            dict(
                name=o.name,
                description=o.description,
                pos_x=o.pos[0], pos_y=o.pos[1], pos_z=o.pos[2],
                bbox_min_x=o.bbox_min[0], bbox_min_y=o.bbox_min[1], bbox_min_z=o.bbox_min[2],
                bbox_max_x=o.bbox_max[0], bbox_max_y=o.bbox_max[1], bbox_max_z=o.bbox_max[2],
                num_views=o.num_views,
                num_points=o.num_points,
                is_fragment=o.is_fragment,
                mesh_path=o.mesh_path,
            )
            for o in objects
        ],
    )
    await scene_service.mark_done(session, scene)
    log.info("scene done: %d objects (%d solid)", len(objects), sum(1 for o in objects if not o.is_fragment))


async def _poll_until_terminal(
    client: GpuClient, job_id: str, *, deadline_sec: int, log
) -> GpuJobStatus:
    """Poll status.json every GPU_POLL_INTERVAL_SEC until the pipeline reports a
    terminal state, or raise once `deadline_sec` has elapsed. This is a redundant
    safety net alongside arq's own `job_timeout` (worker.py handles that outer
    cancellation) - having both means a clean, descriptive failure message either way,
    not just a bare CancelledError."""
    start = time.monotonic()
    last_stage = None
    while True:
        status = await client.poll_status(job_id)
        if status.stage != last_stage:
            log.info(
                "stage=%s (%s/%s) state=%s message=%s",
                status.stage, status.stage_index, status.n_stages, status.state, status.message,
            )
            last_stage = status.stage
        if status.state in _TERMINAL_STATES:
            return status
        if time.monotonic() - start > deadline_sec:
            raise PipelineError(f"job exceeded JOB_TIMEOUT_SEC ({deadline_sec}s) at stage {status.stage!r}")
        await asyncio.sleep(settings.gpu_poll_interval_sec)
