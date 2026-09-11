"""FastAPI application factory and lifespan management."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import settings
from app.database import init_db
from app.logging_setup import configure_logging
from app.queue import create_pool_from_settings
from app.routers import health, models, projects, robots, scenes, videos, workspaces

configure_logging("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the upload directory exists, the database is reachable, and the arq
    enqueue pool is ready on startup."""
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    await init_db()
    app.state.arq = await create_pool_from_settings()
    yield
    await app.state.arq.aclose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="Video Upload Backend", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(videos.router)
    app.include_router(projects.router)
    app.include_router(scenes.router)
    app.include_router(robots.router)
    app.include_router(models.router)
    app.include_router(workspaces.router)
    return app


app = create_app()
