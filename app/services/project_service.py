"""Project CRUD, cascade deletion, and default-project creation on bare upload."""

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models import Project, Scene, Video
from app.services import job_spec

logger = logging.getLogger(__name__)


class ProjectNotFoundError(Exception):
    """Raised when a project id doesn't exist."""


def default_project_name(filename: str, when: datetime | None = None) -> str:
    """Derive a default project name from an uploaded filename, or a dated fallback.

    Used when a video is uploaded without an explicit project_id, so it still lands
    somewhere sensible rather than failing.
    """
    stem = Path(filename).stem.strip()
    if stem:
        return stem
    when = when or datetime.now(timezone.utc)
    return f"Project {when.strftime('%Y-%m-%d %H:%M')}"


async def create_project(
    session: AsyncSession, *, name: str, description: str | None = None
) -> Project:
    """Insert a new project row and return it."""
    project = Project(name=name, description=description)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


async def ensure_project_for_upload(
    session: AsyncSession, project_id: uuid.UUID | None, filename: str
) -> Project:
    """Resolve the project a newly-uploaded video belongs to.

    If `project_id` is given, fetch and return that project (raising if it doesn't
    exist). If not, auto-create a new project named after the file, per the spec:
    "Загрузка видео без указания project_id создаёт новый проект автоматически".
    """
    if project_id is not None:
        project = await get_project(session, project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        return project
    return await create_project(session, name=default_project_name(filename))


async def list_projects(
    session: AsyncSession, *, skip: int, limit: int, include_archived: bool = False
) -> tuple[list[Project], int]:
    """Return a page of projects ordered by newest first, plus the total count.

    Archived projects are excluded unless `include_archived` - the sidebar's default
    view and the total count it shows should both reflect that."""
    query = select(Project)
    count_query = select(func.count()).select_from(Project)
    if not include_archived:
        query = query.where(~Project.archived)
        count_query = count_query.where(~Project.archived)
    total = await session.scalar(count_query)
    result = await session.execute(query.order_by(Project.created_at.desc()).offset(skip).limit(limit))
    return list(result.scalars().all()), total or 0


async def get_project(session: AsyncSession, project_id: uuid.UUID) -> Project | None:
    """Fetch a single project by id, or None if it doesn't exist."""
    return await session.get(Project, project_id)


async def get_project_detail(
    session: AsyncSession, project_id: uuid.UUID
) -> tuple[Project, list[Scene], list[Video]] | None:
    """Fetch a project with its scenes and videos eagerly loaded, or None if not found."""
    result = await session.execute(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.scenes), selectinload(Project.videos))
    )
    project = result.scalar_one_or_none()
    if project is None:
        return None
    scenes = sorted(project.scenes, key=lambda s: s.created_at, reverse=True)
    videos = sorted(project.videos, key=lambda v: v.created_at, reverse=True)
    return project, scenes, videos


async def update_project(
    session: AsyncSession,
    project: Project,
    *,
    name: str | None = None,
    description: str | None = None,
    primary_scene_id: uuid.UUID | None = None,
    primary_scene_id_set: bool = False,
    archived: bool | None = None,
) -> Project:
    """Apply a partial update to a project. Only non-None fields are changed, except
    `primary_scene_id` which can be explicitly cleared - `primary_scene_id_set` disambiguates
    "not provided" from "provided as null". `archived` has no such third state (True/False
    are its only meaningful values), so None simply means "leave it alone"."""
    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    if primary_scene_id_set:
        project.primary_scene_id = primary_scene_id
    if archived is not None:
        project.archived = archived
    await session.commit()
    await session.refresh(project)
    return project


def _scene_dir(scene_id: uuid.UUID) -> Path:
    return Path(settings.upload_dir) / "scenes" / str(scene_id)


async def delete_project(session: AsyncSession, project: Project) -> None:
    """Delete a project's scene directories and video files from disk, then the project
    row (DB cascade removes videos/scenes/objects/commands)."""
    detail = await get_project_detail(session, project.id)
    if detail is not None:
        _, scenes, videos = detail
        for scene in scenes:
            scene_dir = _scene_dir(scene.id)
            if scene_dir.exists():
                shutil.rmtree(scene_dir, ignore_errors=True)
                logger.info("Removed scene directory %s for project %s", scene_dir, project.id)
        for video in videos:
            Path(video.filepath).unlink(missing_ok=True)
            # The job spec written beside the video by the New-workspace wizard
            # (docs/JOB_SPEC.md). It was being left behind on every delete: measured on a
            # throwaway workspace, the .mp4 went and `<uuid>.job.json` stayed, so each
            # deleted workspace orphaned its spec in the upload dir forever.
            spec_path = job_spec.spec_path_for_video(video.filepath)
            spec_path.unlink(missing_ok=True)
    await session.delete(project)
    await session.commit()
