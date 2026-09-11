"""Model catalog and vocab-provider health, for the "Models" block shown when
creating a project / uploading a video - see docs/COMPARISON.md for how the Vertex
option was evaluated."""

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from app.schemas import ModelOptionResponse, ModelStageResponse, ProviderHealthResponse
from app.services import model_catalog, vocab_health

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/catalog", response_model=list[ModelStageResponse])
async def get_model_catalog() -> list[ModelStageResponse]:
    return [
        ModelStageResponse(
            stage=stage.stage,
            editable=stage.editable,
            default=stage.default,
            options=[ModelOptionResponse(id=o.id, provider=o.provider, model=o.model) for o in stage.options],
        )
        for stage in model_catalog.get_catalog()
    ]


@router.get("/vocab-health", response_model=dict[str, ProviderHealthResponse])
async def get_vocab_health(force: bool = Query(False)) -> dict[str, ProviderHealthResponse]:
    health = await vocab_health.get_health(force=force)
    return {
        provider_id: ProviderHealthResponse(
            available=h.available,
            latency_ms=h.latency_ms,
            checked_at=datetime.fromtimestamp(h.checked_at, tz=timezone.utc),
            detail=h.detail,
        )
        for provider_id, h in health.items()
    }
