#!/usr/bin/env python3
"""Stage 2: derive a per-video object vocabulary via one Vertex AI or OpenRouter VLM
call on a handful of representative keyframes.

Runs under the vidmap venv (stdlib urllib only, no `requests`/`httpx` dependency needed
for one call). Deliberately does NOT fail soft with a *fabricated* vocabulary: it used
to catch any provider failure and fall back to a hardcoded 6-concept default,
continuing the job - but a scene built on a silently-substituted vocabulary looks
exactly like a real one and is not distinguishable without inspecting vocab_source -
the same class of silent corruption the "validate before writing to the DB" rule
already exists to prevent elsewhere in this pipeline. That fallback is gone for good;
see dispatch_vocab() below for what *is* allowed to happen on a provider failure now
(falling back to the other real, live provider - not to a fabricated list).

A job's params.json can set "vocab_override" (same entry shape a provider's own
response must produce - a name string, or a {"name", "size_class"} dict) to skip the
vocabulary call entirely and use that list as-is - e.g. for reproducing a fixed
vocabulary across repeated runs, given the call's own run-to-run non-determinism (see
docs/COMPARISON.md). Written to vocab.json as source="override", distinct from every
other source value, so a scene built this way is never mistaken for one built from a
real per-video call.

"vocab_provider" picks which provider makes that call - see dispatch_vocab() below.
Unset (the default) or explicitly "vertex" both mean **Vertex is primary, with
OpenRouter as the automatic, logged fallback on any Vertex failure** (docs/DECISIONS.md,
2026-09-07 - supersedes the 2026-08-31 "no fallback of any kind" call in
docs/COMPARISON.md, which was specifically about not fabricating a vocabulary, not
about refusing to try a second real provider). Explicitly "openrouter" opts out of
Vertex entirely - only that path is tried, no fallback needed. Ignored entirely if
"vocab_override" is set. Vertex auth is a short-lived ADC access token pushed per-job
as VERTEX_ACCESS_TOKEN (see app/services/vertex_auth.py) - no API key, no
service-account key, matching org policy; GCP_PROJECT_ID is pushed alongside it.
Vertex's MaaS endpoint caps multimodal requests at 4 images (see docs/COMPARISON.md),
below N_SAMPLE_FRAMES - VERTEX_MAX_FRAMES exists specifically for this.

The LLM only names objects and picks a rough size class - it never invents SAM3
segmentation thresholds. SIZE_CLASS_PARAMS maps each size class to the (score_threshold,
dbscan_eps, min_samples, min_cluster) tuple stage_objects.py needs, seeded from values
already proven to work in this project (see the run2 CONCEPTS table this was derived
from).
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_io import OCCUPANCY_GRID_RESOLUTION_M, job_paths, load_params  # noqa: E402


DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
N_SAMPLE_FRAMES = 5

# The one Gemma variant actually offered as serverless MaaS on Vertex/Agent Platform -
# see docs/COMPARISON.md. Matches app/services/model_catalog.VERTEX_VOCAB_MODEL - the
# backend normally passes this explicitly via params["vertex_vocab_model"], this is
# only the fallback if that key is somehow absent.
DEFAULT_VERTEX_MODEL = "google/gemma-4-26b-a4b-it-maas"
VERTEX_ENDPOINT = (
    "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/endpoints/openapi/chat/completions"
)
# Confirmed by direct testing (docs/COMPARISON.md): more than 4 images in one request
# gets rejected outright ("Expected a maximum of 4 images"), unlike OpenRouter's 5.
VERTEX_MAX_FRAMES = 4

# eps expressed as a multiple of OCCUPANCY_GRID_RESOLUTION_M (job_io.py) rather than an
# independent flat constant - single source of truth for the pipeline's spatial scale,
# and it's what stage_objects.py's per-concept voxel-downsample size is also derived
# from (voxel = min(OCCUPANCY_GRID_RESOLUTION_M, eps/3)), so the two stay consistent
# with each other by construction instead of two constants that can silently drift.
#
# "huge" was tightened 3x->2x (0.15m->0.10m) on 2026-09-01 after hotel room 2 reported a
# "bed" at 2.92x3.31m against a real ~1x2m bed, then REVERTED the same day: re-testing
# DBSCAN at both eps values against the actual saved point clouds of every real
# "huge"-class object in the three validated reference scenes (ikport, basement_apartment,
# modular_home - 12 objects total) showed the tightened value changed 4 of them,
# including ikport's own bed_0 (1 cluster -> 4, one of which is a real geometry change on
# a scene whose numbers are already in this project's documentation), while it did not
# fix the hotel-room merging it was meant to address (room 3's bed measured WORSE after
# tightening, 3.96x4.19m, than room 2's original 2.92x3.31m). A parameter change that
# regresses validated scenes without fixing the problem it targeted is not a trade worth
# making eight days from a deadline - reverted to the original value. Left at 3x here,
# not re-tuned to some other value: two attempts at a global eps have now both failed to
# fix the actual bug (adjacent real objects with a gap narrower than eps), which needs a
# different mechanism (e.g. per-object-pair geometric review), not a global threshold.
SIZE_CLASS_EPS_VOXEL_MULTIPLE = {
    "small": 0.6,  # e.g. dispenser, faucet, light switch, outlet
    "medium": 1.0,  # e.g. sink, toilet, small chair
    "large": 2.0,  # e.g. mirror, cabinet, table
    "huge": 3.0,  # e.g. door, window, large furniture - reverted, see above
}

# size_class -> (score_threshold, dbscan_eps_m, dbscan_min_samples, min_cluster_points)
SIZE_CLASS_PARAMS = {
    "small": (0.4, SIZE_CLASS_EPS_VOXEL_MULTIPLE["small"] * OCCUPANCY_GRID_RESOLUTION_M, 8, 15),
    "medium": (0.5, SIZE_CLASS_EPS_VOXEL_MULTIPLE["medium"] * OCCUPANCY_GRID_RESOLUTION_M, 15, 30),
    "large": (0.6, SIZE_CLASS_EPS_VOXEL_MULTIPLE["large"] * OCCUPANCY_GRID_RESOLUTION_M, 20, 60),
    "huge": (0.45, SIZE_CLASS_EPS_VOXEL_MULTIPLE["huge"] * OCCUPANCY_GRID_RESOLUTION_M, 15, 40),
}
DEFAULT_SIZE_CLASS = "medium"

PROMPT = (
    "These are representative frames from a continuous handheld walkthrough video of "
    "ONE ROOM. Some frames may show a doorway/window to an adjacent space (a different "
    "room, hallway, or outdoors) - that adjacent space is OUT OF SCOPE, a separate area "
    "the robot does not operate in. "
    "List the distinct physical objects INSIDE this room that a mobile robot operating "
    "in it could either PICK UP or need to NAVIGATE AROUND. Explicitly EXCLUDE anything "
    "visible only through a doorway/window into an adjacent space. "
    "Rules: single common nouns only (e.g. \"chair\", not \"chair leg\" or \"chair "
    "cushion\"); no nested sub-parts of a larger object already in the list; no room "
    "surfaces like wall/floor/ceiling unless they are actual obstacles; dedupe synonyms. "
    "For each object also estimate its rough physical size class: \"small\" (under "
    "~20cm, e.g. a switch or dispenser), \"medium\" (~20-60cm, e.g. a sink or small "
    "chair), \"large\" (~60cm-1.5m, e.g. a mirror, cabinet, or table), or \"huge\" "
    "(over 1.5m, e.g. a door, window, or large piece of furniture). "
    "Respond with ONLY a JSON array of objects, nothing else, e.g. "
    "[{\"name\": \"chair\", \"size_class\": \"medium\"}]."
)


def pick_sample_frames(keyframes_dir: Path, n: int) -> list[Path]:
    frames = sorted(keyframes_dir.glob("kf_*.jpg"))
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def call_openrouter(frames: list[Path], *, api_key: str, model: str, base_url: str) -> str:
    content = [{"type": "text", "text": PROMPT}]
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
    if "error" in result:
        raise RuntimeError(f"OpenRouter API error: {result['error']}")
    return result["choices"][0]["message"]["content"]


def call_vertex(frames: list[Path], *, token: str, project_id: str, model: str) -> str:
    """Same OpenAI-compatible chat-completions shape as call_openrouter() - Vertex's
    MaaS endpoint for open models speaks it too (see docs/COMPARISON.md). Caller must
    already have capped `frames` to VERTEX_MAX_FRAMES."""
    content = [{"type": "text", "text": PROMPT}]
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 1000}
    ).encode()
    req = urllib.request.Request(
        VERTEX_ENDPOINT.format(project_id=project_id),
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
    if "error" in result:
        raise RuntimeError(f"Vertex API error: {result['error']}")
    return result["choices"][0]["message"]["content"]


def extract_json_array(text: str) -> list:
    """Strip ```json fences / leading-trailing prose and parse the first JSON array."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array found in response: {text[:300]!r}")
    return json.loads(text[start : end + 1])


def build_vocab_entries(raw: list) -> list[dict]:
    entries = []
    seen = set()
    for item in raw:
        if isinstance(item, str):
            name, size_class = item, DEFAULT_SIZE_CLASS
        elif isinstance(item, dict):
            name = item.get("name")
            size_class = item.get("size_class", DEFAULT_SIZE_CLASS)
        else:
            continue
        if not name or not isinstance(name, str):
            continue
        name = name.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        if size_class not in SIZE_CLASS_PARAMS:
            size_class = DEFAULT_SIZE_CLASS
        entries.append({"name": name, "size_class": size_class})
    return entries


class VocabProviderError(Exception):
    """Raised when the selected provider can't produce a usable vocabulary - missing
    credentials, no keyframes, or the call itself failing. No fallback: see this
    module's docstring for why. Message always names the provider, so it survives
    into run_pipeline.sh's stage-failure log tail and, from there, into the scene's
    error_message (pipeline_orchestrator.py -> scene_service.mark_failed)."""


_CALL_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError, TimeoutError)


def run_openrouter(paths: dict, params: dict) -> tuple[list[dict], str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = params.get("openrouter_model", os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    base_url = params.get(
        "openrouter_base_url", os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    )

    if not api_key:
        raise VocabProviderError("openrouter: no OPENROUTER_API_KEY in environment")

    frames = pick_sample_frames(paths["keyframes_dir"], N_SAMPLE_FRAMES)
    if not frames:
        raise VocabProviderError("openrouter: no keyframes found")

    try:
        raw_text = call_openrouter(frames, api_key=api_key, model=model, base_url=base_url)
        print(f"raw model response: {raw_text[:500]!r}")
        raw = extract_json_array(raw_text)
        entries = build_vocab_entries(raw)
        if not entries:
            raise ValueError("model returned an empty/unusable object list")
    except _CALL_ERRORS as exc:
        raise VocabProviderError(f"openrouter: vocabulary call failed: {exc!r}") from exc

    return entries, "openrouter", model


def run_vertex(paths: dict, params: dict) -> tuple[list[dict], str, str]:
    token = os.environ.get("VERTEX_ACCESS_TOKEN", "")
    project_id = os.environ.get("GCP_PROJECT_ID", "")
    model = params.get("vertex_vocab_model", DEFAULT_VERTEX_MODEL)

    if not token or not project_id:
        raise VocabProviderError("vertex: no VERTEX_ACCESS_TOKEN/GCP_PROJECT_ID in environment")

    frames = pick_sample_frames(paths["keyframes_dir"], VERTEX_MAX_FRAMES)
    if not frames:
        raise VocabProviderError("vertex: no keyframes found")

    try:
        raw_text = call_vertex(frames, token=token, project_id=project_id, model=model)
        print(f"raw model response: {raw_text[:500]!r}")
        raw = extract_json_array(raw_text)
        entries = build_vocab_entries(raw)
        if not entries:
            raise ValueError("model returned an empty/unusable object list")
    except _CALL_ERRORS as exc:
        raise VocabProviderError(f"vertex: vocabulary call failed: {exc!r}") from exc

    return entries, "vertex", model


def dispatch_vocab(paths: dict, params: dict) -> tuple[list[dict], str, str | None]:
    """Picks the vocabulary source for this job, in order:

    1. `vocab_override` (see module docstring) wins outright - no provider call at all.
    2. An explicit `params["vocab_provider"] == "openrouter"` opts out of Vertex
       entirely - a person chose OpenRouter deliberately (the Models selector), so
       that's the only path tried, same as before this function existed.
    3. Otherwise (`vocab_provider` unset, or explicitly "vertex") - **Vertex is the
       primary provider**, with automatic, logged fallback to OpenRouter if the Vertex
       call raises `VocabProviderError` (docs/DECISIONS.md, 2026-09-07 "Vertex primary
       + automatic OpenRouter fallback for scene vocabulary" - supersedes the earlier
       2026-08-31 "no fallback of any kind" call recorded in docs/COMPARISON.md, which
       was about substituting a *fabricated* vocabulary on failure, not about trying a
       second real provider; a genuine second live call is not the same silent-
       corruption risk).

    Only case 3's fallback exists - OpenRouter failing never falls back to Vertex, and
    an explicit "openrouter" choice never attempts Vertex. Both providers still fail
    loudly (`VocabProviderError` propagates) if the one path tried (or, in case 3, both
    paths tried) can't produce a usable vocabulary - no fabricated substitute, ever.
    """
    vocab_override = params.get("vocab_override")
    if vocab_override:
        print(f"vocab_override present ({len(vocab_override)} entries) - skipping vocabulary call")
        return build_vocab_entries(vocab_override), "override", None

    if params.get("vocab_provider") == "openrouter":
        return run_openrouter(paths, params)

    try:
        entries, source, model_used = run_vertex(paths, params)
        print(f"vocab provider: vertex (primary) succeeded, model={model_used}")
        return entries, source, model_used
    except VocabProviderError as exc:
        print(f"vocab provider: vertex (primary) failed ({exc!r}) - falling back to openrouter")
        return run_openrouter(paths, params)


def main() -> None:
    job_dir = sys.argv[1]
    paths = job_paths(job_dir)
    params = load_params(job_dir)

    entries, source, model_used = dispatch_vocab(paths, params)

    for e in entries:
        thr, eps, min_samples, min_cluster = SIZE_CLASS_PARAMS[e["size_class"]]
        e.update(
            score_threshold=thr, dbscan_eps=eps, dbscan_min_samples=min_samples, min_cluster=min_cluster
        )

    # provider (`source`) and `model` are recorded for every run, not just
    # openrouter/vertex - `model` is `null` for "override" (no provider call was made)
    # rather than omitted, so every vocab.json has both keys.
    payload = {"source": source, "model": model_used, "objects": entries}
    paths["vocab_json"].write_text(json.dumps(payload, indent=2))
    print(f"DONE: vocabulary ({source}): {[e['name'] for e in entries]}")


if __name__ == "__main__":
    main()
