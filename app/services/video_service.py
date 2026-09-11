"""File storage, ffprobe metadata extraction, and DB operations for videos."""

import asyncio
import json
import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Video
from app.services import model_catalog, project_service

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB

ALLOWED_EXTENSIONS = {"mp4", "mov"}
ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime"}


class UploadTooLargeError(Exception):
    """Raised when an incoming upload exceeds the configured size limit."""


def validate_upload(filename: str, content_type: str | None) -> str:
    """Validate the file extension and content-type, returning the lowercase extension.

    Raises ValueError if either check fails.
    """
    extension = Path(filename).suffix.lstrip(".").lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: .{extension}")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")
    return extension


async def save_upload(file: UploadFile, extension: str) -> tuple[Path, int]:
    """Stream the upload to disk in chunks, enforcing the max size limit.

    Returns the destination path and total bytes written. Deletes the partial
    file and raises UploadTooLargeError if the limit is exceeded.
    """
    upload_dir = Path(settings.upload_dir)
    destination = upload_dir / f"{uuid.uuid4()}.{extension}"

    total_bytes = 0
    async with aiofiles.open(destination, "wb") as out_file:
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > settings.max_upload_bytes:
                await out_file.close()
                destination.unlink(missing_ok=True)
                raise UploadTooLargeError(
                    f"Upload exceeds max size of {settings.max_upload_bytes} bytes"
                )
            await out_file.write(chunk)

    return destination, total_bytes


async def probe_video(filepath: Path) -> tuple[float | None, str | None]:
    """Run ffprobe to extract (duration_sec, resolution). Returns (None, None) on any failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", filepath, stderr.decode(errors="replace"))
            return None, None

        data = json.loads(stdout)
        duration_sec = None
        if fmt := data.get("format"):
            if duration := fmt.get("duration"):
                duration_sec = float(duration)

        resolution = None
        for stream in data.get("streams", []):
            width, height = stream.get("width"), stream.get("height")
            if width and height:
                resolution = f"{width}x{height}"
                break

        return duration_sec, resolution
    except FileNotFoundError:
        logger.warning("ffprobe is not installed; skipping metadata extraction for %s", filepath)
        return None, None
    except Exception:
        logger.warning("ffprobe error while probing %s", filepath, exc_info=True)
        return None, None


def parse_vocab_override(raw: str | None) -> list | None:
    """Parse the `vocab_override` form field (a JSON-encoded string, since multipart
    form fields are always strings) into the list gpu/stage_vocab.py expects - same
    entry shape a real OpenRouter response must produce (a name string, or a
    {"name", "size_class"} dict). None/empty-string input (field omitted) returns None,
    preserving current behavior unchanged - only actually deferring to
    gpu/stage_vocab.py's own per-entry validation (build_vocab_entries()), not
    duplicating it here.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"vocab_override is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("vocab_override must be a JSON array")
    return parsed


def validate_vocab_provider(raw: str | None) -> str | None:
    """None/empty-string input (field omitted) returns None, meaning "use the
    default" (app/services/model_catalog.DEFAULT_VOCAB_PROVIDER) - preserving current
    behavior unchanged for every upload that doesn't set this."""
    if not raw:
        return None
    if raw not in model_catalog.VOCAB_PROVIDERS:
        raise ValueError(f"vocab_provider must be one of {sorted(model_catalog.VOCAB_PROVIDERS)}, got {raw!r}")
    return raw


async def handle_upload(
    session: AsyncSession,
    file: UploadFile,
    *,
    project_id: uuid.UUID | None,
    blur_threshold: float | None = None,
    similarity_threshold: float | None = None,
    vocab_override: str | None = None,
    vocab_provider: str | None = None,
) -> Video:
    """Validate, save, probe, and record a single uploaded video - the shared path behind
    both `POST /api/videos/upload` and `POST /api/projects/{id}/videos`.

    If `project_id` is None, a new project is auto-created (named after the filename),
    per the spec: "Загрузка видео без указания project_id создаёт новый проект автоматически".
    Raises ValueError (bad filename/content-type, or malformed vocab_override),
    UploadTooLargeError, or ProjectNotFoundError (explicit project_id that doesn't
    exist) - the router maps each to the appropriate HTTP status.

    `blur_threshold`/`similarity_threshold` override gpu/stage_keyframes.py's keyframe
    filter for this video only; leave both None to use that stage's own defaults
    (blur_threshold=100.0, similarity_threshold=0.98) - most uploads should.

    `vocab_override`, if set, skips this video's OpenRouter vocabulary call entirely
    (gpu/stage_vocab.py) - for reproducing a fixed vocabulary across repeated runs of
    the same footage, given that call's own run-to-run non-determinism (see
    docs/COMPARISON.md). Raw JSON string in, parsed here.

    `vocab_provider`, if set, picks which provider makes that call - "openrouter" or
    "vertex" (app/services/model_catalog.VOCAB_PROVIDERS). Independent of
    vocab_override: this picks who's asked, vocab_override skips asking anyone.
    """
    if not file.filename:
        raise ValueError("Filename is required")

    parsed_vocab_override = parse_vocab_override(vocab_override)
    validated_vocab_provider = validate_vocab_provider(vocab_provider)

    extension = validate_upload(file.filename, file.content_type)
    destination, size_bytes = await save_upload(file, extension)
    duration_sec, resolution = await probe_video(destination)

    project = await project_service.ensure_project_for_upload(session, project_id, file.filename)

    return await create_video(
        session,
        project_id=project.id,
        filename=file.filename,
        filepath=destination,
        size_bytes=size_bytes,
        fmt=extension,
        duration_sec=duration_sec,
        resolution=resolution,
        blur_threshold=blur_threshold,
        similarity_threshold=similarity_threshold,
        vocab_override=parsed_vocab_override,
        vocab_provider=validated_vocab_provider,
    )


async def create_video(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    filename: str,
    filepath: Path,
    size_bytes: int,
    fmt: str,
    duration_sec: float | None,
    resolution: str | None,
    blur_threshold: float | None = None,
    similarity_threshold: float | None = None,
    vocab_override: list | None = None,
    vocab_provider: str | None = None,
) -> Video:
    """Insert a new video row and return it."""
    video = Video(
        project_id=project_id,
        filename=filename,
        filepath=str(filepath),
        size_bytes=size_bytes,
        format=fmt,
        duration_sec=duration_sec,
        resolution=resolution,
        status="uploaded",
        blur_threshold=blur_threshold,
        similarity_threshold=similarity_threshold,
        vocab_override=vocab_override,
        vocab_provider=vocab_provider,
    )
    session.add(video)
    await session.commit()
    await session.refresh(video)
    return video


async def list_videos(session: AsyncSession, *, skip: int, limit: int) -> tuple[list[Video], int]:
    """Return a page of videos ordered by newest first, plus the total count."""
    total = await session.scalar(select(func.count()).select_from(Video))
    result = await session.execute(
        select(Video).order_by(Video.created_at.desc()).offset(skip).limit(limit)
    )
    return list(result.scalars().all()), total or 0


async def get_video(session: AsyncSession, video_id: uuid.UUID) -> Video | None:
    """Fetch a single video by id, or None if it doesn't exist."""
    return await session.get(Video, video_id)


async def delete_video(session: AsyncSession, video: Video) -> None:
    """Delete the video's file from disk (if present) and remove its DB row."""
    Path(video.filepath).unlink(missing_ok=True)
    await session.delete(video)
    await session.commit()
