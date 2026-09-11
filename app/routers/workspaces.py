"""The processing screen's one endpoint.

A "workspace" is a project - the upload auto-creates one per video (see
`video_service.handle_upload`), so the id the wizard redirects to is that project's id.
This router exists rather than living in `projects.py` because it answers a different
question: not "what is this project" but "what is the pipeline doing to it right now",
assembled from files on disk (`docs/JOB_SPEC.md`).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas import ProcessingStatusResponse, ProcessingStageResponse
from app.services import processing_status, project_service
from app.services.job_spec import read_job_spec

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


@router.get("/{workspace_id}/processing", response_model=ProcessingStatusResponse)
async def get_processing_status(
    workspace_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> ProcessingStatusResponse:
    """Everything the processing screen renders, on every poll.

    The screen keeps nothing of its own - no localStorage, no tab memory - so this is the
    single source of what it shows, and two tabs on the same URL necessarily agree.

    Stages come from `<scene_dir>/status.json` when it exists and from
    `<scene_dir>/logs/worker.log` otherwise, with `source` naming which, so a degraded
    feed is never mistaken for the authoritative one. Nothing is invented: with neither
    source present the stage list is empty and `worker_connected` is false, which is what
    the screen turns into "worker not connected" rather than a spinner.
    """
    detail = await project_service.get_project_detail(session, workspace_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    project, scenes, videos = detail

    # Newest scene: a workspace has one video today, but re-runs add scenes, and the
    # screen is about what is happening NOW.
    scene = max(scenes, key=lambda s: s.created_at) if scenes else None
    video = max(videos, key=lambda v: v.created_at) if videos else None
    spec = read_job_spec(video.filepath) if video and video.filepath else None

    result = processing_status.read_processing_status(workspace_id, scene.id if scene else None, spec=spec)

    return ProcessingStatusResponse(
        workspace_id=workspace_id,
        workspace_name=project.name,
        scene_id=result.scene_id,
        source=result.source,
        state=result.state,
        error=result.error,
        stages=[
            ProcessingStageResponse(
                stage=row.stage,
                state=row.state,
                started_at=row.started_at,
                finished_at=row.finished_at,
                duration_sec=row.duration_sec,
                error=row.error,
                substage=row.substage,
                detail=row.detail,
                since=row.since,
                attempt=row.attempt,
                next_retry=row.next_retry,
            )
            for row in result.stages
        ],
        seconds_since_last_record=result.seconds_since_last_record,
        worker_connected=result.worker_connected,
        is_done=result.is_done,
        is_failed=result.is_failed,
        fallback_notice=processing_status.fallback_notice(result),
        fallback_reason=result.fallback_reason,
        requested_provider=result.requested_provider,
        provider_used=result.provider_used,
    )
