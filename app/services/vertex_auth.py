"""Application Default Credentials access tokens for calling Vertex AI / Agent
Platform - no API key, no service-account key (org policy forbids the latter; see
app/config.py's gcp_project_id comment). Shells out to `gcloud` rather than adding a
google-auth dependency - this project has no other Google Cloud SDK usage, and a
prototype-scale feature doesn't warrant one for a single token-fetch call.

Tokens are cached in-memory for CACHE_TTL_SEC, refreshed only once stale - not because
the token itself needs refreshing that often (real access tokens last ~1 hour), but
because `gcloud`'s own CLI startup overhead alone measured ~1.7s on this box, which
would otherwise blow most of vocab_health.py's 2-3s check budget on every single
health-check call before the actual Vertex API request even starts. The per-job token
pushed to the GPU host (pipeline_orchestrator.py) benefits the same way - one gcloud
invocation serves many jobs within the cache window instead of one per job.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Comfortably under the ~1 hour real lifetime of a GCP access token - `gcloud` doesn't
# hand back an easy-to-parse expiry for `application-default print-access-token`, so
# this is a conservative fixed margin rather than tracking the real expiry.
CACHE_TTL_SEC = 45 * 60

_cached_token: str | None = None
_cached_at: float = 0.0


class VertexAuthError(Exception):
    """Raised when an ADC access token can't be obtained - gcloud missing, ADC not
    configured, or the command otherwise failed."""


async def get_access_token(timeout: float = 10.0) -> str:
    global _cached_token, _cached_at

    if _cached_token is not None and time.monotonic() - _cached_at < CACHE_TTL_SEC:
        return _cached_token

    proc = await asyncio.create_subprocess_exec(
        "gcloud", "auth", "application-default", "print-access-token",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        raise VertexAuthError("gcloud print-access-token timed out") from exc

    if proc.returncode != 0:
        raise VertexAuthError(
            f"gcloud print-access-token exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
        )
    token = stdout.decode().strip()
    if not token:
        raise VertexAuthError("gcloud print-access-token returned an empty token")

    _cached_token, _cached_at = token, time.monotonic()
    return token
