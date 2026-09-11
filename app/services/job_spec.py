"""The New-workspace wizard's job spec: validation, and the file it lives in.

Contract and rationale: `docs/JOB_SPEC.md`; machine-readable schema:
`docs/job_spec.schema.json`. This module is the Python side of the same contract - if the
three ever disagree, the schema file is the one to fix first, since it is what a
non-Python consumer reads.

Deliberately a FILE next to the uploaded video, not a table: the spec is written once at
upload, never mutated, and read by a worker that has the video path and nothing else. A
column would need a migration, and this needs zero.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SPEC_VERSION = 1

#: Suffix appended to the video's own stem, so the spec is discoverable from the video
#: path alone: `var/uploads/<uuid>.mp4` -> `var/uploads/<uuid>.job.json`.
SPEC_SUFFIX = ".job.json"

Semantics = Literal["openrouter-glm", "vertex-gemma"]
ChatLlm = Literal["openrouter-nemotron", "vertex-gemma"]

#: spec value -> the `vocab_provider` id the existing upload path and
#: `app/services/model_catalog.py` already use. See docs/JOB_SPEC.md section 1.
SEMANTICS_TO_VOCAB_PROVIDER: dict[str, str] = {
    "openrouter-glm": "openrouter",
    "vertex-gemma": "vertex",
}


class JobSpec(BaseModel):
    """What the operator chose, exactly as it is written to disk.

    `mapping` has one legal value and no UI control. It is a field anyway so that adding a
    second reconstruction backend is a named schema change rather than a silent
    reinterpretation of specs already on disk.
    """

    spec_version: Literal[1] = SPEC_VERSION
    video_id: uuid.UUID
    #: The measured knee on the hero footage (HANDOFF section 4), not a round number.
    frames_fps: float = Field(default=15.0, gt=0, le=30)
    mapping: Literal["mapanything"] = "mapanything"
    semantics: Semantics = "openrouter-glm"
    #: Recorded for the worker. The running app ignores it: the model catalog marks
    #: command parsing non-editable and vlm_client has no Vertex path (docs/JOB_SPEC.md).
    chat_llm: ChatLlm = "openrouter-nemotron"
    #: Platform ids from app/robots.py. Empty means "every registered platform", which is
    #: what the viewer already shows - it is not the same as "none".
    robots: list[str] = Field(default_factory=list)
    created_at: str | None = None

    def vocab_provider(self) -> str:
        """The `vocab_provider` id this spec's `semantics` corresponds to - the value the
        upload path and the scene-vocabulary stage already speak."""
        return SEMANTICS_TO_VOCAB_PROVIDER[self.semantics]


def now_iso() -> str:
    """Local ISO-8601 with seconds - the same shape pipeline-v1's `now()` writes into
    status.json, so timestamps from the two files sort against each other without a
    conversion step. Deliberately not UTC/Z: matching the file we have to read beats
    matching a preference."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def spec_path_for_video(video_path: str | Path) -> Path:
    """`var/uploads/<uuid>.mp4` -> `var/uploads/<uuid>.job.json`.

    Uses `with_suffix('')` once rather than splitting on '.', so a video whose name has no
    extension still produces a distinct spec path instead of overwriting the video.
    """
    p = Path(video_path)
    return p.with_name(f"{p.stem}{SPEC_SUFFIX}")


def write_job_spec(video_path: str | Path, spec: JobSpec) -> Path:
    """Write the spec beside the video, atomically (tmp + os.replace) for the same reason
    pipeline-v1 writes status.json that way: a reader must never see a half-written file.
    Returns the path written."""
    path = spec_path_for_video(video_path)
    payload = spec.model_dump(mode="json")
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_job_spec(video_path: str | Path) -> JobSpec | None:
    """The spec beside this video, or None when there is none (every video uploaded
    before the wizard existed, and any upload that did not carry one). Callers must treat
    None as "use the defaults", never as an error."""
    path = spec_path_for_video(video_path)
    if not path.is_file():
        return None
    return JobSpec.model_validate_json(path.read_text(encoding="utf-8"))


def parse_request_spec(raw: str | None) -> JobSpec | None:
    """Validate a `job_spec` form field from an upload request.

    `video_id` is NOT taken from the client: the caller fills it in from the row the
    upload creates, so a request cannot point its spec at somebody else's video. A
    placeholder is substituted here purely so the rest of the object can be validated
    before that row exists.

    Returns None for a missing/blank field (every non-wizard upload). Raises ValueError
    with a readable message on malformed JSON or a value outside the schema - the router
    maps that to 400.
    """
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"job_spec is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("job_spec must be a JSON object")
    payload = {**payload, "video_id": payload.get("video_id") or uuid.UUID(int=0)}
    try:
        return JobSpec.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - pydantic's message is the useful part
        raise ValueError(f"job_spec is not a valid job spec: {exc}") from exc
