"""Cheap liveness checks for the two scene-vocabulary providers - a minimal
text-only request (no images, no real prompt), not the real per-video call. Bounded to
a couple of seconds so opening the upload form never hangs waiting on a slow/dead
provider, and cached briefly so opening it repeatedly doesn't hammer either provider.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from app.config import settings
from app.services import model_catalog, vertex_auth

CHECK_TIMEOUT_SEC = 3.0
CACHE_TTL_SEC = 30.0


@dataclass(frozen=True)
class ProviderHealth:
    available: bool
    latency_ms: float | None
    checked_at: float  # unix epoch seconds
    detail: str | None = None  # short reason when unavailable - never a full traceback


_cache: dict[str, ProviderHealth] = {}


async def _check_openrouter() -> ProviderHealth:
    now = time.time()
    if not settings.openrouter_api_key:
        return ProviderHealth(available=False, latency_ms=None, checked_at=now, detail="no API key configured")
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=CHECK_TIMEOUT_SEC) as client:
            resp = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                json={
                    "model": settings.openrouter_model_vision,
                    "messages": [{"role": "user", "content": "OK"}],
                    "max_tokens": 1,
                },
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code >= 400:
            return ProviderHealth(False, latency_ms, now, detail=f"HTTP {resp.status_code}")
        return ProviderHealth(True, latency_ms, now)
    except httpx.TimeoutException:
        return ProviderHealth(False, (time.monotonic() - t0) * 1000, now, detail="timed out")
    except httpx.HTTPError as exc:
        return ProviderHealth(False, (time.monotonic() - t0) * 1000, now, detail=str(exc)[:200])


async def _vertex_probe() -> httpx.Response:
    token = await vertex_auth.get_access_token(timeout=CHECK_TIMEOUT_SEC)
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{settings.gcp_project_id}"
        "/locations/global/endpoints/openapi/chat/completions"
    )
    async with httpx.AsyncClient() as client:
        return await client.post(
            url,
            json={
                "model": model_catalog.VERTEX_VOCAB_MODEL,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 1,
            },
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        )


async def _check_vertex() -> ProviderHealth:
    now = time.time()
    if not settings.gcp_project_id:
        return ProviderHealth(available=False, latency_ms=None, checked_at=now, detail="GCP_PROJECT_ID not configured")
    t0 = time.monotonic()
    try:
        # One shared deadline for token-fetch + API call together (not one each,
        # which could add up to ~2x CHECK_TIMEOUT_SEC on a cold token cache) - see
        # vertex_auth.py's docstring for why the token fetch alone can be slow.
        resp = await asyncio.wait_for(_vertex_probe(), timeout=CHECK_TIMEOUT_SEC)
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code >= 400:
            return ProviderHealth(False, latency_ms, now, detail=f"HTTP {resp.status_code}")
        return ProviderHealth(True, latency_ms, now)
    except asyncio.TimeoutError:
        return ProviderHealth(False, (time.monotonic() - t0) * 1000, now, detail="timed out")
    except vertex_auth.VertexAuthError as exc:
        return ProviderHealth(False, (time.monotonic() - t0) * 1000, now, detail=f"auth failed: {exc}"[:200])
    except httpx.HTTPError as exc:
        return ProviderHealth(False, (time.monotonic() - t0) * 1000, now, detail=str(exc)[:200])


_CHECKERS = {
    model_catalog.VOCAB_PROVIDER_OPENROUTER: _check_openrouter,
    model_catalog.VOCAB_PROVIDER_VERTEX: _check_vertex,
}


async def get_health(*, force: bool = False) -> dict[str, ProviderHealth]:
    """One entry per vocab provider. Cached for CACHE_TTL_SEC - a form opened twice in
    a row doesn't re-probe either provider."""
    now = time.time()
    stale = [
        pid for pid in _CHECKERS
        if force or pid not in _cache or now - _cache[pid].checked_at > CACHE_TTL_SEC
    ]
    if stale:
        results = await asyncio.gather(*(_CHECKERS[pid]() for pid in stale))
        for pid, result in zip(stale, results):
            _cache[pid] = result
    return dict(_cache)
