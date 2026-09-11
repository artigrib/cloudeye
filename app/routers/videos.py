"""Video upload and CRUD endpoints."""

import logging
import re
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from arq.connections import ArqRedis

from app.database import get_session
from app.queue import enqueue_process_video, get_arq_pool
from app.schemas import RerunResponse, SceneStatusResponse, VideoListResponse, VideoResponse
from app.services import job_spec as job_spec_service
from app.services import project_service, scene_service, video_service
from app.services.project_service import ProjectNotFoundError
from app.services.video_service import UploadTooLargeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/videos", tags=["videos"])

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB

CONTENT_TYPE_BY_FORMAT = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
}


@router.post("/upload", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    project_id: uuid.UUID | None = Form(None),
    blur_threshold: float | None = Form(None, ge=0),
    similarity_threshold: float | None = Form(None, ge=0, le=1),
    vocab_override: str | None = Form(None),
    vocab_provider: str | None = Form(None),
    job_spec: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> VideoResponse:
    """Accept a single mp4/mov file, stream it to disk, record its metadata, and enqueue
    the GPU reconstruction pipeline for it.

    If `project_id` is omitted, a new project is created automatically (named from the
    uploaded filename).

    `blur_threshold`/`similarity_threshold` override the keyframe-extraction stage's
    filter for this video only - leave both unset to use its defaults (100.0 / 0.98).
    Useful for footage the default blur check misjudges (e.g. sharp but low-texture
    rooms, which read as "blurry" under a plain Laplacian-variance check).

    `vocab_override` is a JSON array (e.g. '["door", {"name": "curtain", "size_class":
    "large"}]') fixing this video's object vocabulary and skipping the OpenRouter call
    entirely - leave unset for normal per-video vocabulary detection.

    `vocab_provider` picks which provider makes the scene-vocabulary call -
    "openrouter" or "vertex" (see GET /api/models/catalog) - leave unset for the
    default (OpenRouter). Ignored if `vocab_override` is also set.

    `job_spec` is the New-workspace wizard's choices as a JSON object (see
    docs/JOB_SPEC.md and docs/job_spec.schema.json). It is validated here and saved
    beside the video as `<video_uuid>.job.json`; `video_id` is filled in from the row
    this upload creates, so the client neither sends nor can forge it. Its `semantics`
    also supplies `vocab_provider` when that form field was not given separately, so the
    wizard's choice actually reaches the stage that makes the call rather than only being
    recorded for the worker. Absent is the normal case for every non-wizard upload.
    """
    try:
        requested_spec = job_spec_service.parse_request_spec(job_spec)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # The wizard's semantics choice IS the vocab provider - see docs/JOB_SPEC.md. An
    # explicit vocab_provider form field still wins, so the older callers that send it
    # are unaffected.
    if requested_spec is not None and vocab_provider is None:
        vocab_provider = requested_spec.vocab_provider()

    try:
        video = await video_service.handle_upload(
            session,
            file,
            project_id=project_id,
            blur_threshold=blur_threshold,
            similarity_threshold=similarity_threshold,
            vocab_provider=vocab_provider,
            vocab_override=vocab_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if requested_spec is not None:
        # After the row exists, so `video_id` is the real one. Written before the job is
        # enqueued: a worker that picks the job up must never race the spec onto disk.
        job_spec_service.write_job_spec(
            video.filepath,
            requested_spec.model_copy(update={"video_id": video.id, "created_at": job_spec_service.now_iso()}),
        )
    await enqueue_process_video(arq_pool, video.id)
    return VideoResponse.model_validate(video)


@router.get("/", response_model=VideoListResponse)
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> VideoListResponse:
    """List videos with pagination."""
    videos, total = await video_service.list_videos(session, skip=skip, limit=limit)
    return VideoListResponse(
        total=total,
        skip=skip,
        limit=limit,
        items=[VideoResponse.model_validate(v) for v in videos],
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> VideoResponse:
    """Fetch metadata for a single video."""
    video = await video_service.get_video(session, video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return VideoResponse.model_validate(video)


@router.get("/{video_id}/scene", response_model=SceneStatusResponse)
async def get_video_scene(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> SceneStatusResponse:
    """Fetch the status of the scene reconstruction for this video, including
    per-stage timing for the stage/ETA display (see
    app.services.scene_service.build_status_response)."""
    video = await video_service.get_video(session, video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    scene = await scene_service.get_scene_by_video(session, video_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No scene for this video yet"
        )
    return scene_service.build_status_response(scene)


@router.post("/{video_id}/rerun", response_model=RerunResponse, status_code=status.HTTP_202_ACCEPTED)
async def rerun_video(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> RerunResponse:
    """Re-run this video's scene reconstruction, reusing its previously-resolved
    object vocabulary instead of making a fresh OpenRouter/Vertex call - see
    app.services.pipeline_orchestrator.resolve_vocab_override. Makes repeated runs of
    the same video comparable: normal reprocessing re-samples a new vocabulary every
    time (see gpu/stage_vocab.py's docstring on that call's own run-to-run
    non-determinism), which changes which objects get found even on identical
    footage.

    Requires a scene to already exist for this video (any status - `done`, `failed`,
    even still `processing`, though that just re-queues behind/alongside it); a video
    that's never been processed at all has nothing to rerun, use the normal
    upload-triggered pipeline for that. `video.vocab_override`, if set at upload
    time, still takes precedence over the cached vocabulary either way - see
    resolve_vocab_override()'s precedence.
    """
    video = await video_service.get_video(session, video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    scene = await scene_service.get_scene_by_video(session, video_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No scene for this video yet - nothing to rerun"
        )
    job_id = await enqueue_process_video(arq_pool, video.id, rerun=True)
    return RerunResponse(video_id=video.id, scene_id=scene.id, job_id=job_id, enqueued=job_id is not None)


@router.get("/{video_id}/file")
async def stream_video_file(
    video_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Stream the video file, supporting HTTP Range requests for seeking."""
    video = await video_service.get_video(session, video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    filepath = Path(video.filepath)
    if not filepath.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found on disk")

    file_size = filepath.stat().st_size
    content_type = CONTENT_TYPE_BY_FORMAT.get(video.format, "application/octet-stream")

    range_header = request.headers.get("range")
    start, end = 0, file_size - 1

    if range_header:
        match = RANGE_RE.match(range_header)
        if not match:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Invalid Range header",
            )
        start_str, end_str = match.groups()
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        if start > end or end >= file_size:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Invalid Range header",
                headers={"Content-Range": f"bytes */{file_size}"},
            )

    content_length = end - start + 1

    async def file_stream():
        async with aiofiles.open(filepath, "rb") as f:
            await f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = await f.read(min(STREAM_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Disposition": f'inline; filename="{video.filename}"',
    }

    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        return StreamingResponse(
            file_stream(),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            media_type=content_type,
            headers=headers,
        )

    return StreamingResponse(file_stream(), media_type=content_type, headers=headers)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a video's file from disk (if present) and its DB row."""
    video = await video_service.get_video(session, video_id)
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    await video_service.delete_video(session, video)
