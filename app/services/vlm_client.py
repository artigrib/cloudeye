"""OpenRouter VLM client for parsing natural-language robot commands into structured
actions. Text-only (unlike gpu/stage_vocab.py's image-carrying call) - same
OpenRouter endpoint, but its own model (settings.openrouter_model_command, distinct
from settings.openrouter_model_vision) since this is a pure text task. Reused via
httpx since this runs on the backend, not the GPU box (which deliberately stays
stdlib-only, see gpu/stage_vocab.py's docstring).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.models import SceneObject
from app.services import model_catalog, vertex_auth

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"take", "goto", "look"}

#: The chat-LLM providers a workspace's job spec can name (`JobSpec.chat_llm`, see
#: docs/JOB_SPEC.md), mapped to how this module reaches them. The spec used to be
#: recorded and ignored - "the running app ignores it: vlm_client has no Vertex path" -
#: which meant an operator who chose vertex-gemma in the wizard silently got OpenRouter.
#: Both are OpenAI-shaped /chat/completions endpoints; only the base URL, the auth and
#: the model id differ, which is why one client can serve both.
PROVIDER_OPENROUTER_NEMOTRON = "openrouter-nemotron"
PROVIDER_VERTEX_GEMMA = "vertex-gemma"
CHAT_PROVIDERS = (PROVIDER_OPENROUTER_NEMOTRON, PROVIDER_VERTEX_GEMMA)
DEFAULT_CHAT_PROVIDER = PROVIDER_OPENROUTER_NEMOTRON


class VLMError(Exception):
    """Raised when the VLM call fails outright, or its response can't be parsed into
    usable JSON even after stripping common wrapping (fences, prose)."""


class ChatProviderNotConnected(VLMError):
    """The configured chat-LLM provider has no credentials on this deployment.

    A distinct type because it is a fact about the DEPLOYMENT, not about the command:
    retyping the words cannot fix it and naming a different object cannot either, so the
    UI answers it with "Chat LLM not connected - templated commands still work" rather
    than "not understood". Every templated command (go to / can X reach Y / compare all /
    move) is parsed client-side and never reaches a provider at all, so this is a partial
    outage, not a dead screen.

    Subclasses VLMError so every existing `except VLMError` keeps catching it."""


def extract_json(text: str) -> dict:
    """Strip common LLM response wrapping - ```json fences, plain ``` fences, leading
    prose ("Here's the JSON:"), trailing prose - then parse the first balanced
    top-level `{...}` object. Raises VLMError (with the raw text attached, truncated)
    if nothing parseable is found."""
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start == -1:
        raise VLMError(f"no JSON object found in response: {text[:300]!r}")

    # Find the matching closing brace by depth-counting, rather than just rfind("}") -
    # a trailing explanation after the JSON (or a nested object) would otherwise grab
    # the wrong closing brace.
    depth = 0
    end = None
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise VLMError(f"unbalanced JSON object in response: {text[:300]!r}")

    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise VLMError(f"could not parse JSON object {candidate[:300]!r}: {exc}") from exc


def resolve_chat_provider(provider: str | None) -> str:
    """The provider id to actually use. An unknown or missing value falls back to the
    default rather than failing: a spec written by a newer build must not make an old
    deployment refuse every command, and `JobSpec.chat_llm` is already schema-validated
    where it is written."""
    return provider if provider in CHAT_PROVIDERS else DEFAULT_CHAT_PROVIDER


async def _openrouter_request(
    messages: list[dict[str, Any]], model: str | None, timeout: float
) -> httpx.Response:
    api_key = settings.openrouter_api_key
    if not api_key:
        raise ChatProviderNotConnected("OPENROUTER_API_KEY is not configured")
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            json={"model": model or settings.openrouter_model_command, "messages": messages, "temperature": 0},
            headers={"Authorization": f"Bearer {api_key}"},
        )


async def _vertex_request(
    messages: list[dict[str, Any]], model: str | None, timeout: float
) -> httpx.Response:
    """Vertex AI / Agent Platform's OpenAI-compatible endpoint - the same URL shape and
    the same body `vocab_health._vertex_probe` and `gpu/stage_vocab.py` already use, with
    an ADC access token instead of an API key (org policy forbids service-account keys;
    see app/config.py's gcp_project_id comment)."""
    if not settings.gcp_project_id:
        raise ChatProviderNotConnected("GCP_PROJECT_ID is not configured")
    try:
        token = await vertex_auth.get_access_token(timeout=timeout)
    except vertex_auth.VertexAuthError as exc:
        raise ChatProviderNotConnected(f"Vertex credentials are not configured: {exc}") from exc
    url = (
        f"https://aiplatform.googleapis.com/v1/projects/{settings.gcp_project_id}"
        "/locations/global/endpoints/openapi/chat/completions"
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(
            url,
            json={
                "model": model or model_catalog.VERTEX_VOCAB_MODEL,
                "messages": messages,
                "temperature": 0,
            },
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        )


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    provider: str | None = None,
    timeout: float = 60.0,
) -> dict:
    """One chat completion call against the named provider, returning the parsed JSON
    object from the model's text response. `provider` defaults to
    DEFAULT_CHAT_PROVIDER, which is what every pre-existing caller gets."""
    provider_id = resolve_chat_provider(provider)
    request = _vertex_request if provider_id == PROVIDER_VERTEX_GEMMA else _openrouter_request
    try:
        resp = await request(messages, model, timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise VLMError(f"{provider_id} request failed: {exc}") from exc

    result = resp.json()
    if "error" in result:
        raise VLMError(f"{provider_id} API error: {result['error']}")
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise VLMError(f"unexpected {provider_id} response shape: {result}") from exc

    return extract_json(text)


def build_command_messages(objects: list[SceneObject], text: str) -> list[dict[str, Any]]:
    names = sorted({o.name for o in objects})
    object_list = ", ".join(names) if names else "(no objects detected in this scene)"
    prompt = (
        "You control a mobile robot in a mapped room. The room contains exactly these "
        f"objects: {object_list}. A user gave this instruction: {text!r}\n\n"
        "Parse it into a strict JSON object with keys:\n"
        '  "action": one of "take" (pick an object up and bring it somewhere), '
        '"goto" (move to an object, nothing else), "look" (move to get a view of an object)\n'
        '  "target": the object name from the list above the action applies to\n'
        '  "destination": for "take" only, the object name to bring the target to '
        "(or null for goto/look)\n\n"
        "If the instruction refers to an object NOT in the list above, respond with "
        '{"error": "<short explanation>"} instead. '
        "Respond with ONLY the JSON object, no other text."
    )
    return [{"role": "user", "content": prompt}]


async def parse_command(objects: list[SceneObject], text: str, *, provider: str | None = None) -> dict:
    """Parse a natural-language command against this scene's actual object list.
    Returns either {"action", "target", "destination"} or {"error": "..."}.

    `provider` names the chat LLM from the workspace's job spec; None keeps the previous
    behaviour (DEFAULT_CHAT_PROVIDER)."""
    messages = build_command_messages(objects, text)
    parsed = await chat_json(messages, provider=provider)
    if "error" in parsed:
        return parsed
    if parsed.get("action") not in VALID_ACTIONS:
        raise VLMError(f"model returned an invalid action: {parsed.get('action')!r}")
    if not parsed.get("target"):
        raise VLMError("model response is missing 'target'")
    parsed.setdefault("destination", None)
    return parsed
