"""GET /health: checks the DB (SELECT 1 over the async engine) and Redis (PING over the
arq enqueue pool) that the API depends on. Runs against the real app with its real
lifespan (real Postgres/Redis in this dev environment), and monkeypatches each
dependency's call to force the failure branches - no mocking of the whole app.
"""

import httpx
import pytest
from httpx import ASGITransport

from app.main import app


@pytest.fixture
async def client():
    from app.database import engine

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    # The engine is a module-level singleton and pytest-asyncio gives each test
    # function its own event loop; without disposing here, the next test's loop
    # tries to tear down connections opened on this (now-closed) loop and blows up.
    await engine.dispose()


async def test_health_ok_when_db_and_redis_reachable(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok", "redis": "ok"}


async def test_health_503_when_db_unreachable(client, monkeypatch):
    from app.database import engine

    def _broken_connect(self, *args, **kwargs):
        raise RuntimeError("db down")

    # `connect` is defined on the class (AsyncEngine proxies through a facade), so
    # patching the instance attribute directly is rejected as read-only.
    monkeypatch.setattr(type(engine), "connect", _broken_connect)

    resp = await client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["db"] == "error"
    assert body["redis"] == "ok"


async def test_health_503_when_redis_unreachable(client, monkeypatch):
    async def _broken_ping(*args, **kwargs):
        raise RuntimeError("redis down")

    monkeypatch.setattr(app.state.arq, "ping", _broken_ping)

    resp = await client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["db"] == "ok"
    assert body["redis"] == "error"


async def test_health_503_when_both_unreachable(client, monkeypatch):
    from app.database import engine

    def _broken_connect(self, *args, **kwargs):
        raise RuntimeError("db down")

    async def _broken_ping(*args, **kwargs):
        raise RuntimeError("redis down")

    monkeypatch.setattr(type(engine), "connect", _broken_connect)
    monkeypatch.setattr(app.state.arq, "ping", _broken_ping)

    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "error", "db": "error", "redis": "error"}
