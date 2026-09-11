"""Project CRUD and per-project video upload endpoints."""

import logging
import uuid

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.queue import enqueue_process_video, get_arq_pool
from app.schemas import (
    ProjectCreate,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
    SceneSummary,
    VideoResponse,
)
from app.services import project_service, video_service
from app.services.project_service import ProjectNotFoundError
from app.services.video_service import UploadTooLargeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    session: AsyncSession = Depends(get_session),
) -> ProjectResponse:
    """Create a new (initially empty) project."""
    project = await project_service.create_project(
        session, name=body.name, description=body.description
    )
    return ProjectResponse.model_validate(project)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    include_archived: bool = Query(False),
    session: AsyncSession = Depends(get_session),
) -> ProjectListResponse:
    """List projects with pagination, for a sidebar-style listing. Archived projects
    are excluded unless include_archived=true."""
    projects, total = await project_service.list_projects(
        session, skip=skip, limit=limit, include_archived=include_archived
    )
    return ProjectListResponse(
        total=total,
        skip=skip,
        limit=limit,
        items=[ProjectResponse.model_validate(p) for p in projects],
    )


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> ProjectDetailResponse:
    """Fetch a project along with its scenes and videos."""
    detail = await project_service.get_project_detail(session, project_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    project, scenes, videos = detail
    return ProjectDetailResponse(
        project=ProjectResponse.model_validate(project),
        scenes=[SceneSummary.model_validate(s) for s in scenes],
        videos=[VideoResponse.model_validate(v) for v in videos],
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdate,
    session: AsyncSession = Depends(get_session),
) -> ProjectResponse:
    """Rename a project, edit its description, change its primary scene, or
    archive/unarchive it."""
    project = await project_service.get_project(session, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    fields_set = body.model_fields_set
    project = await project_service.update_project(
        session,
        project,
        name=body.name,
        description=body.description,
        primary_scene_id=body.primary_scene_id,
        primary_scene_id_set="primary_scene_id" in fields_set,
        archived=body.archived,
    )
    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a project, cascading to its scenes, videos, and their files on disk."""
    project = await project_service.get_project(session, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    await project_service.delete_project(session, project)


@router.post("/{project_id}/videos", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def upload_video_to_project(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    blur_threshold: float | None = Form(None, ge=0),
    similarity_threshold: float | None = Form(None, ge=0, le=1),
    vocab_override: str | None = Form(None),
    vocab_provider: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> VideoResponse:
    """Upload a video into an existing project and enqueue its reconstruction pipeline.

    See `POST /api/videos/upload` for what `blur_threshold`/`similarity_threshold`/
    `vocab_override`/`vocab_provider` do.
    """
    try:
        video = await video_service.handle_upload(
            session,
            file,
            project_id=project_id,
            blur_threshold=blur_threshold,
            similarity_threshold=similarity_threshold,
            vocab_override=vocab_override,
            vocab_provider=vocab_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await enqueue_process_video(arq_pool, video.id)
    return VideoResponse.model_validate(video)
