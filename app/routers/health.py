"""Liveness/readiness endpoint: checks the DB and the arq/Redis queue pool.

Used by deploy tooling and uptime checks to distinguish "process is up" from "process
can actually serve requests" - a process that's up but can't reach Postgres or Redis
should be pulled from rotation, hence the 503 on partial failure rather than always 200.
"""

import logging

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app.database import engine
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request, response: Response) -> HealthResponse:
    """Check DB (`SELECT 1` over the existing async engine) and Redis (`PING` over the
    arq enqueue pool created in `main.py`'s lifespan) connectivity.

    Returns 200 when both are reachable, 503 when either is not - the body always
    reports the per-dependency status either way so a caller can tell which one failed.
    """
    db_status = "ok"
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Health check: database ping failed")
        db_status = "error"

    redis_status = "ok"
    try:
        await request.app.state.arq.ping()
    except Exception:
        logger.exception("Health check: redis ping failed")
        redis_status = "error"

    overall_ok = db_status == "ok" and redis_status == "ok"
    response.status_code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if overall_ok else "error",
        db=db_status,
        redis=redis_status,
    )
