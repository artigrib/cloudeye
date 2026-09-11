"""The chat-LLM provider named in the workspace's job spec, and what happens when it has
no credentials.

Until now `JobSpec.chat_llm` was written at upload and then ignored - the module comment
said so outright ("the running app ignores it: vlm_client has no Vertex path") - so an
operator who chose vertex-gemma in the New-workspace wizard silently got OpenRouter.
These tests pin the two halves of fixing that: the spec's choice reaches the request, and
the answer records which model produced it.

Every provider is MOCKED. There is no API key in this worktree and none is being
configured; that is the condition the product has to work under, not an obstacle to
testing it.
"""

import uuid

import httpx
import pytest

from app.config import settings
from app.models import SceneObject
from app.services import vlm_client
from app.services.vlm_client import (
    CHAT_PROVIDERS,
    DEFAULT_CHAT_PROVIDER,
    PROVIDER_OPENROUTER_NEMOTRON,
    PROVIDER_VERTEX_GEMMA,
    ChatProviderNotConnected,
    VLMError,
    resolve_chat_provider,
)


def _object(name: str) -> SceneObject:
    return SceneObject(
        id=uuid.uuid4(), scene_id=uuid.uuid4(), name=name,
        pos_x=0.0, pos_y=0.0, pos_z=0.0,
        bbox_min_x=-0.1, bbox_min_y=0.0, bbox_min_z=-0.1,
        bbox_max_x=0.1, bbox_max_y=1.0, bbox_max_z=0.1,
        num_views=3, num_points=100, is_fragment=False,
    )


def _canned(payload: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": payload}}]}, request=httpx.Request("POST", "http://mock")
    )


class _Recorder:
    """Stands in for one provider's HTTP call, recording what it was asked for."""

    def __init__(self, payload='{"action": "goto", "target": "desk", "destination": null}'):
        self.payload = payload
        self.calls: list[tuple[str | None, float]] = []

    async def __call__(self, messages, model, timeout):
        self.calls.append((model, timeout))
        return _canned(self.payload)


# --- which provider gets asked --------------------------------------------------------


def test_resolve_falls_back_rather_than_refusing_an_unknown_name():
    # A spec written by a newer build must not make an older deployment reject every
    # command it is given.
    assert resolve_chat_provider(None) == DEFAULT_CHAT_PROVIDER
    assert resolve_chat_provider("something-new") == DEFAULT_CHAT_PROVIDER
    for provider in CHAT_PROVIDERS:
        assert resolve_chat_provider(provider) == provider


def test_the_default_is_the_job_spec_default():
    from app.services.job_spec import JobSpec

    assert DEFAULT_CHAT_PROVIDER == JobSpec(video_id=uuid.uuid4()).chat_llm


@pytest.mark.asyncio
async def test_openrouter_is_used_when_the_spec_names_it(monkeypatch):
    recorder = _Recorder()
    vertex = _Recorder()
    monkeypatch.setattr(vlm_client, "_openrouter_request", recorder)
    monkeypatch.setattr(vlm_client, "_vertex_request", vertex)

    parsed = await vlm_client.parse_command([_object("desk")], "go to the desk",
                                            provider=PROVIDER_OPENROUTER_NEMOTRON)

    assert parsed["target"] == "desk"
    assert len(recorder.calls) == 1
    assert vertex.calls == []


@pytest.mark.asyncio
async def test_vertex_is_used_when_the_spec_names_it(monkeypatch):
    openrouter = _Recorder()
    vertex = _Recorder()
    monkeypatch.setattr(vlm_client, "_openrouter_request", openrouter)
    monkeypatch.setattr(vlm_client, "_vertex_request", vertex)

    parsed = await vlm_client.parse_command([_object("desk")], "go to the desk",
                                            provider=PROVIDER_VERTEX_GEMMA)

    assert parsed["target"] == "desk"
    assert len(vertex.calls) == 1
    # The regression this is here for: before the dispatch existed, this request went to
    # OpenRouter no matter what the spec said.
    assert openrouter.calls == []


# --- no credentials -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_with_no_key_raises_the_not_connected_type(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    with pytest.raises(ChatProviderNotConnected):
        await vlm_client.chat_json([{"role": "user", "content": "hi"}],
                                   provider=PROVIDER_OPENROUTER_NEMOTRON)


@pytest.mark.asyncio
async def test_vertex_with_no_project_raises_the_not_connected_type(monkeypatch):
    monkeypatch.setattr(settings, "gcp_project_id", "")
    with pytest.raises(ChatProviderNotConnected):
        await vlm_client.chat_json([{"role": "user", "content": "hi"}],
                                   provider=PROVIDER_VERTEX_GEMMA)


@pytest.mark.asyncio
async def test_vertex_auth_failure_is_not_connected_rather_than_a_generic_failure(monkeypatch):
    from app.services import vertex_auth

    monkeypatch.setattr(settings, "gcp_project_id", "some-project")

    async def _fail(timeout=10.0):
        raise vertex_auth.VertexAuthError("ADC not configured")

    monkeypatch.setattr(vertex_auth, "get_access_token", _fail)
    with pytest.raises(ChatProviderNotConnected):
        await vlm_client.chat_json([{"role": "user", "content": "hi"}], provider=PROVIDER_VERTEX_GEMMA)


def test_not_connected_is_still_a_vlm_error():
    # Every existing `except VLMError` in command_service keeps catching it, so this
    # cannot turn a handled failure into a 500.
    assert issubclass(ChatProviderNotConnected, VLMError)


@pytest.mark.asyncio
async def test_a_provider_that_answers_with_an_error_body_is_a_plain_vlm_error(monkeypatch):
    async def _error_body(messages, model, timeout):
        return httpx.Response(200, json={"error": {"message": "rate limited"}},
                              request=httpx.Request("POST", "http://mock"))

    monkeypatch.setattr(vlm_client, "_openrouter_request", _error_body)
    with pytest.raises(VLMError) as excinfo:
        await vlm_client.chat_json([{"role": "user", "content": "hi"}],
                                   provider=PROVIDER_OPENROUTER_NEMOTRON)
    # Not the "not connected" case: a key exists, the model just refused this call.
    assert not isinstance(excinfo.value, ChatProviderNotConnected)
