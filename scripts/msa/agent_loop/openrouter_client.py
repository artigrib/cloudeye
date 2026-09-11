"""Thin OpenRouter chat-completion client for Stage C1's Generator/Verifier
bake-off (SPEC.md §6). Mirrors `app/services/vlm_client.py`'s request/response
handling (this project's existing OpenRouter integration - JSON extraction,
error wrapping) but reads `OPENROUTER_API_KEY` directly from the environment
instead of `app.config.settings`, since the bake-off script runs standalone
(`uv run python -m scripts.msa.agent_loop.bakeoff`), not inside the FastAPI app.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(Exception):
    """Raised for a failed OpenRouter call, or a response with no parseable
    JSON object in its text content."""


@dataclass
class CallRecord:
    """One logged OpenRouter call - SPEC §6: 'every call logged with prompt
    hash, images, response, decision'. `decision` is filled in by the caller
    after it acts on `parsed` (e.g. accept/reject), not by this module."""

    role: str  # "generator" | "verifier"
    model: str
    prompt_hash: str
    had_image: bool
    response_text: str
    parsed: dict | None
    usage: dict | None
    decision: str | None = None

    def cost_usd(self, pricing: dict[str, float]) -> float:
        """pricing: {'prompt': <usd/token>, 'completion': <usd/token>} from
        OpenRouter's /models endpoint. Returns 0.0 if usage wasn't reported."""
        if not self.usage:
            return 0.0
        prompt_tok = self.usage.get("prompt_tokens", 0)
        completion_tok = self.usage.get("completion_tokens", 0)
        return prompt_tok * pricing.get("prompt", 0.0) + completion_tok * pricing.get("completion", 0.0)


def extract_json_object(text: str) -> dict | None:
    """Same fence-stripping + balanced-brace scan as
    `app/services/vlm_client.py:extract_json`, but returns None instead of
    raising - the bake-off treats an unparseable Generator response as a
    logged failure (a real robustness signal), not a fatal error."""
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
        return None
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
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def _image_data_url(png_path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png_path.read_bytes()).decode("ascii")


def call_vision_json(
    *,
    role: str,
    model: str,
    system: str,
    user_text: str,
    image_path: Path | None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> CallRecord:
    """One OpenRouter chat completion call with an optional image, returning a
    logged `CallRecord` (never raises for a parse failure - only for a
    transport/API-level failure, which IS fatal since it means the bake-off
    step produced no data at all)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY not set in environment")

    content: list[dict] = [{"type": "text", "text": user_text}]
    if image_path is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path)}})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    prompt_hash = hashlib.sha256(json.dumps({"system": system, "user_text": user_text, "has_image": image_path is not None}, sort_keys=True).encode()).hexdigest()[:16]

    body = {"model": model, "messages": messages, "temperature": temperature}
    try:
        resp = httpx.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OpenRouterError(f"OpenRouter request failed ({model}): {exc}") from exc

    result = resp.json()
    if "error" in result:
        raise OpenRouterError(f"OpenRouter API error ({model}): {result['error']}")
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"unexpected OpenRouter response shape ({model}): {result}") from exc

    return CallRecord(
        role=role,
        model=model,
        prompt_hash=prompt_hash,
        had_image=image_path is not None,
        response_text=text,
        parsed=extract_json_object(text),
        usage=result.get("usage"),
    )
