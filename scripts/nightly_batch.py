#!/usr/bin/env python3
"""Nightly regression batch: re-runs the "named"/regression scene corpus through the
standard backend API (port 8000, video-api.service/video-worker.service - the same
services real uploads go through, nothing GPU-side or DB-side is bypassed) and writes
a human-readable report.md + machine-readable results.json to
`var/scratch/nightly/<YYYY-MM-DD>/`.

Scene selection ("сцены для прогона"): queried live from the `projects`/`videos` tables
(see `discover_regression_scenes`), not hardcoded - a project counts as a named
regression scene if its name matches the numbered external demo corpus
(`01_hotel_room` .. `15_sea_grill_restaurant`, see docs/DECISIONS.md) or is one of the
small set of hand-named scenes referenced across docs/DECISIONS.md
(room, street1, street2, ikport, 02_modular_home is covered by the numbered pattern).
One-off "-rerun-"/"-retest"/"_retest"/"-guard-fix" projects created by past manual
investigations are deliberately excluded - they're throwaway duplicates of a canonical
project, not additional scenes to regress nightly. For each kept project, the most
recently-uploaded video's already-on-disk file (`var/uploads/<uuid>.<ext>`) is
re-uploaded as this run's fresh copy, so every night genuinely exercises
POST /api/videos/upload -> arq enqueue -> GPU pipeline end to end, exactly like a real
user upload.

retain_per_view: `settings.retain_per_view_artifacts` (app/config.py) is a
process-wide setting read once at video-worker.service startup from
`<repo>/.env` / `EnvironmentFile=` - there is IS NO per-request/per-job API
parameter for it anywhere in app/routers/*.py or app/schemas.py (checked; the only
knob is that global boolean). This script therefore cannot force it on for a run
without restarting video-worker.service, which is out of scope here (prod restarts are
off limits for this script). What it does instead:
  - `detect_retain_per_view_setting()` best-effort reads the *currently configured*
    value straight out of .env (the same file the systemd unit's EnvironmentFile=
    points at) and records it in every report as `retain_per_view_artifacts_effective`,
    with a loud warning in report.md when it's False.
  - Regardless of that global flag, `_retain_per_view_on_failure` (pipeline_orchestrator.py,
    landed in fix/retain-per-view-on-failure, commit 2f2bf21) already fetches per_view
    artifacts for every FAILED scene unconditionally - so a nightly run's failures always
    get per-view diagnostics whether or not the global flag is on. Only *successful*
    scenes' per-view retention depends on the global setting.

Isaac validation: for every scene that reaches `done`, the exported USD
(GET /api/scenes/{id}/usd) is downloaded and run through
`tools/isaac_validate.py --usd-only --json-out` (pure-USD structural checks, no Isaac
Sim needed - see that script's own docstring), and its verdict is folded into this
scene's result.

Stage timing: read directly from this scene's own
`{upload_dir}/scenes/{scene_id}/logs/worker.log` (written by
`app.logging_setup.scene_file_log`, on the same box this script runs on - no SSH
needed), which carries one `stage=NAME (i/n) state=running message=starting NAME` line
per pipeline stage transition (`pipeline_orchestrator.py`). Consecutive lines' timestamp
deltas give each stage's wall-clock duration. Falls back to this script's own
client-side polling timestamps (coarser, `--poll-interval-sec` granularity) if the log
file can't be read (e.g. a scene that failed before any stage line was ever written).

Usage:
    uv run python scripts/nightly_batch.py [--api-base http://127.0.0.1:8000]
        [--out-dir var/scratch/nightly/<today>] [--date YYYY-MM-DD]
        [--poll-interval-sec 15] [--timeout-sec 5400] [--project NAME ...]

Exit code is 0 even if individual scenes failed (a failed regression scene is data, not
a script bug) - non-zero only for a setup/infra problem (DB unreachable, API
unreachable, no scenes discovered at all).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import settings  # noqa: E402
from app.database import async_session_maker  # noqa: E402
from app.models import Project, Scene, Video  # noqa: E402
from app.services import scene_service  # noqa: E402
from app.services.stage_timings import StageTiming, parse_stage_timings  # noqa: E402

logger = logging.getLogger("nightly_batch")

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_POLL_INTERVAL_SEC = 15
DEFAULT_TIMEOUT_SEC = 5400  # settings.job_timeout_sec (3600) + margin for queueing/upload
TERMINAL_STATES = {"done", "failed"}

# Numbered external demo corpus referenced across docs/DECISIONS.md, e.g. "01_hotel_room"
# .. "15_sea_grill_restaurant" (also covers "02_modular_home").
_NUMBERED_CORPUS_RE = re.compile(r"^\d{2}_[a-z0-9_]+$")
# Small set of hand-named scenes referenced repeatedly in docs/DECISIONS.md's
# investigations (room.mp4, street1.mp4, street2.mp4, ikport's own capture).
_CANONICAL_NAMES = {"room", "street1", "street2", "ikport"}
# Substrings marking a project as a one-off manual retest/rerun duplicate, not a
# canonical named scene of its own.
_EXCLUDE_SUBSTRINGS = ("-rerun-", "-retest", "_retest", "-guard-fix")


@dataclass(frozen=True)
class SceneSpec:
    """One regression scene to run: which project to upload into, and which
    already-on-disk video file to re-upload as this run's fresh copy."""

    name: str
    project_id: uuid.UUID
    source_video_path: Path
    # Set by --resume-map to skip re-uploading a scene that was already uploaded by a
    # previous (e.g. interrupted) run of this script - reuses that run's video_id and
    # just resumes polling/collection instead of creating a duplicate video+scene+GPU
    # job. None (the normal case) means "upload source_video_path as a fresh copy".
    existing_video_id: str | None = None


@dataclass
class SceneRunResult:
    name: str
    project_id: str
    video_id: str | None = None
    scene_id: str | None = None
    status: str = "not_run"  # "done" | "failed" | "timeout" | "error" | "not_run"
    validator_status: str = "unknown"  # "passed" | "failed" | "pipeline_error" | "unknown"
    error_message: str | None = None
    floor_y: float | None = None
    ceiling_y: float | None = None
    ceiling_height_m: float | None = None
    num_objects: int | None = None
    isaac_validate: dict[str, Any] | None = None
    stage_timings: list[StageTiming] = field(default_factory=list)
    total_duration_sec: float | None = None
    started_at: str | None = None
    finished_at: str | None = None


# --------------------------------------------------------------------------------------
# Pure formatting logic (unit-tested without GPU/DB/network - see
# tests/test_nightly_batch_report.py)
# --------------------------------------------------------------------------------------


def scene_result_to_dict(r: SceneRunResult) -> dict[str, Any]:
    d = {
        "name": r.name,
        "project_id": r.project_id,
        "video_id": r.video_id,
        "scene_id": r.scene_id,
        "status": r.status,
        "validator_status": r.validator_status,
        "error_message": r.error_message,
        "floor_y": r.floor_y,
        "ceiling_y": r.ceiling_y,
        "ceiling_height_m": r.ceiling_height_m,
        "num_objects": r.num_objects,
        "isaac_validate": r.isaac_validate,
        "stage_timings": [
            {"stage": t.stage, "started_at": t.started_at, "duration_sec": t.duration_sec}
            for t in r.stage_timings
        ],
        "total_duration_sec": r.total_duration_sec,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
    }
    return d


def build_results_json(
    *,
    run_date: str,
    generated_at: str,
    retain_per_view_artifacts_effective: bool,
    results: list[SceneRunResult],
) -> dict[str, Any]:
    """Machine-readable payload written to results.json. Pure function of its inputs -
    no I/O - so tests can feed synthetic SceneRunResult fixtures and assert on the
    returned dict directly."""
    n_done = sum(1 for r in results if r.status == "done")
    n_failed = sum(1 for r in results if r.status != "done")
    return {
        "schema": "cloudeye.nightly_batch/1",
        "run_date": run_date,
        "generated_at": generated_at,
        "retain_per_view_artifacts_effective": retain_per_view_artifacts_effective,
        "summary": {
            "total_scenes": len(results),
            "done": n_done,
            "failed_or_incomplete": n_failed,
        },
        "scenes": [scene_result_to_dict(r) for r in results],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report_md(
    *,
    run_date: str,
    generated_at: str,
    retain_per_view_artifacts_effective: bool,
    results: list[SceneRunResult],
) -> str:
    """Human-readable report.md. Pure function, mirrors build_results_json - both are
    exercised directly by tests against synthetic fixtures, no real pipeline run
    needed."""
    lines: list[str] = []
    lines.append(f"# Nightly batch report - {run_date}")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    if not retain_per_view_artifacts_effective:
        lines.append(
            "**WARNING:** `retain_per_view_artifacts` is currently **False** on the "
            "running video-worker.service (read from `<repo>/.env`). This is a "
            "process-wide setting with no per-request API override (see this script's "
            "module docstring) - successful scenes below did NOT retain per_view "
            "artifacts this run. Failed scenes still did (unconditional since "
            "fix/retain-per-view-on-failure, commit 2f2bf21)."
        )
        lines.append("")
    else:
        lines.append("`retain_per_view_artifacts` is currently **True** on video-worker.service.")
        lines.append("")

    n_done = sum(1 for r in results if r.status == "done")
    lines.append(f"**{n_done}/{len(results)} scenes done.**")
    lines.append("")

    lines.append("| Scene | Status | Validator | Floor Y | Ceiling Y | Height (m) | Objects | isaac_validate | Duration (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        isaac_cell = "-"
        if r.isaac_validate is not None:
            isaac_cell = str(r.isaac_validate.get("status", "?"))
        lines.append(
            "| {name} | {status} | {validator} | {floor} | {ceiling} | {height} | {objects} | {isaac} | {dur} |".format(
                name=r.name,
                status=r.status,
                validator=r.validator_status,
                floor=_fmt(r.floor_y),
                ceiling=_fmt(r.ceiling_y),
                height=_fmt(r.ceiling_height_m),
                objects=_fmt(r.num_objects, 0),
                isaac=isaac_cell,
                dur=_fmt(r.total_duration_sec, 1),
            )
        )
    lines.append("")

    lines.append("## Per-scene detail")
    lines.append("")
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append(f"- project_id: `{r.project_id}`")
        lines.append(f"- video_id: `{r.video_id}`")
        lines.append(f"- scene_id: `{r.scene_id}`")
        lines.append(f"- status: **{r.status}**")
        lines.append(f"- validator_status: {r.validator_status}")
        if r.error_message:
            lines.append(f"- error_message: {r.error_message}")
        lines.append(f"- floor_y: {_fmt(r.floor_y)}  ceiling_y: {_fmt(r.ceiling_y)}  height_m: {_fmt(r.ceiling_height_m)}")
        lines.append(f"- num_objects: {_fmt(r.num_objects, 0)}")
        if r.isaac_validate is not None:
            failures = r.isaac_validate.get("failures") or r.isaac_validate.get("checks_failed")
            lines.append(
                f"- isaac_validate --usd-only: status={r.isaac_validate.get('status')} "
                f"exit_code={r.isaac_validate.get('exit_code')}"
                + (f" failures={failures}" if failures else "")
            )
        else:
            lines.append("- isaac_validate --usd-only: not run (scene did not reach `done`)")
        if r.stage_timings:
            lines.append("- stage timings:")
            for t in r.stage_timings:
                dur = f"{t.duration_sec:.1f}s" if t.duration_sec is not None else "(in progress/last)"
                lines.append(f"  - {t.stage}: started {t.started_at}, duration {dur}")
        lines.append(f"- total_duration_sec: {_fmt(r.total_duration_sec, 1)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_report(
    out_dir: Path,
    *,
    run_date: str,
    generated_at: str,
    retain_per_view_artifacts_effective: bool,
    results: list[SceneRunResult],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_payload = build_results_json(
        run_date=run_date,
        generated_at=generated_at,
        retain_per_view_artifacts_effective=retain_per_view_artifacts_effective,
        results=results,
    )
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2) + "\n")
    report_md = build_report_md(
        run_date=run_date,
        generated_at=generated_at,
        retain_per_view_artifacts_effective=retain_per_view_artifacts_effective,
        results=results,
    )
    (out_dir / "report.md").write_text(report_md)


# --------------------------------------------------------------------------------------
# Stage-timing log parsing has moved to app.services.stage_timings (StageTiming,
# parse_stage_timings, imported above) - it's now shared with the backend's
# `GET /scenes/{id}/status` endpoint (feat/job-timeline), which persists the same
# parse's output to `{scene_dir}/stage_timings.json` for the frontend's ETA display.
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Scene discovery (DB read)
# --------------------------------------------------------------------------------------


def _is_regression_project_name(name: str) -> bool:
    if any(sub in name for sub in _EXCLUDE_SUBSTRINGS):
        return False
    if name in _CANONICAL_NAMES:
        return True
    return bool(_NUMBERED_CORPUS_RE.match(name))


async def discover_regression_scenes(session, *, only_names: set[str] | None = None) -> list[SceneSpec]:
    """The "actual list of visible/named projects" to regress nightly, queried live from
    the DB rather than hardcoded (see module docstring for the selection rule). For
    each kept project, picks the most-recently-created video and re-uses its
    already-on-disk file as the source to re-upload."""
    result = await session.execute(select(Project))
    projects = result.scalars().all()

    specs: list[SceneSpec] = []
    for project in projects:
        if not _is_regression_project_name(project.name):
            continue
        if only_names is not None and project.name not in only_names:
            continue
        video_result = await session.execute(
            select(Video).where(Video.project_id == project.id).order_by(Video.created_at.desc())
        )
        latest_video = video_result.scalars().first()
        if latest_video is None:
            continue
        source_path = Path(latest_video.filepath)
        if not source_path.is_file():
            logger.warning(
                "skipping %s: source file %s for latest video %s not found on disk",
                project.name, source_path, latest_video.id,
            )
            continue
        specs.append(SceneSpec(name=project.name, project_id=project.id, source_video_path=source_path))

    specs.sort(key=lambda s: s.name)
    return specs


# --------------------------------------------------------------------------------------
# retain_per_view detection (best-effort local .env read, no process restart)
# --------------------------------------------------------------------------------------


def detect_retain_per_view_setting(env_path: Path = REPO_ROOT / ".env") -> bool:
    """Best-effort read of the *currently configured* retain_per_view_artifacts value
    straight from .env (the same file video-worker.service's EnvironmentFile= points
    at) - see module docstring for why this can only be observed, not forced, from this
    script. Defaults to False (pydantic-settings' own default) if unset or unreadable."""
    try:
        text = env_path.read_text()
    except OSError:
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip().upper() == "RETAIN_PER_VIEW_ARTIFACTS":
            return value.strip().lower() in ("1", "true", "yes", "on")
    return False


# --------------------------------------------------------------------------------------
# Isaac USD validation (subprocess, --usd-only - no Isaac Sim needed)
# --------------------------------------------------------------------------------------


def run_isaac_validate_usd_only(usd_path: Path, json_out_path: Path) -> dict[str, Any]:
    """Runs `tools/isaac_validate.py --usd-only` against a downloaded scene USD and
    returns its parsed JSON report. Never raises for a validation *failure* (that's a
    legitimate result, folded into the report) - only for a script/process-launch
    problem, which callers should treat as an infra error for this scene."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "isaac_validate.py"),
        "--usd-only",
        str(usd_path),
        "--json-out",
        str(json_out_path),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    return json.loads(json_out_path.read_text())


# --------------------------------------------------------------------------------------
# HTTP-driven pipeline run (the real "прогоняет джобы через штатный backend API" path)
# --------------------------------------------------------------------------------------


async def upload_scene(client: httpx.AsyncClient, spec: SceneSpec, run_date: str) -> str:
    """POST /api/videos/upload for one scene spec. Returns the new video_id."""
    suffix = spec.source_video_path.suffix or ".mp4"
    upload_name = f"{spec.name}_nightly_{run_date}{suffix}"
    content_type = "video/quicktime" if suffix.lower() == ".mov" else "video/mp4"
    with spec.source_video_path.open("rb") as fh:
        files = {"file": (upload_name, fh, content_type)}
        data = {"project_id": str(spec.project_id)}
        resp = await client.post("/api/videos/upload", files=files, data=data)
    resp.raise_for_status()
    return resp.json()["id"]


async def poll_scene_status(
    client: httpx.AsyncClient, video_id: str, *, poll_interval_sec: float, timeout_sec: float
) -> dict[str, Any]:
    """Poll GET /api/videos/{id}/scene until a terminal status or timeout. Returns the
    last-seen SceneStatusResponse payload (with a synthetic status="timeout" merged in
    if the deadline is hit first).

    GET /api/videos/{id}/scene 404s ("No scene for this video yet") until the arq
    worker actually dequeues this job and `get_or_create_scene` runs - which can be
    seconds to (if a prior scene in the queue is still processing) a long time after
    the upload's 201 response, since only one GPU pipeline job runs at a time. That 404
    is expected/transient here, not a real error - it's retried like any other
    non-terminal status rather than raised, up to the same overall deadline."""
    deadline = asyncio.get_event_loop().time() + timeout_sec
    last: dict[str, Any] = {}
    while True:
        resp = await client.get(f"/api/videos/{video_id}/scene")
        if resp.status_code != 404:
            resp.raise_for_status()
            last = resp.json()
            if last.get("status") in TERMINAL_STATES:
                return last
        if asyncio.get_event_loop().time() >= deadline:
            last["status"] = "timeout"
            return last
        await asyncio.sleep(poll_interval_sec)


async def fetch_scene_detail(client: httpx.AsyncClient, scene_id: str) -> dict[str, Any]:
    resp = await client.get(f"/api/scenes/{scene_id}")
    resp.raise_for_status()
    return resp.json()


async def fetch_scene_usd(client: httpx.AsyncClient, scene_id: str, dest: Path) -> None:
    resp = await client.get(f"/api/scenes/{scene_id}/usd")
    resp.raise_for_status()
    dest.write_bytes(resp.content)


_VALIDATOR_FAILURE_MARKERS = (
    "ceiling_above_cameras",
    "height_prior_majority",
    "object_max_extent",
    "validate_scene",
    "ValidationError",
)


def derive_validator_status(status: str, error_message: str | None) -> str:
    if status == "done":
        return "passed"
    if status != "failed":
        return "unknown"
    if error_message and any(marker in error_message for marker in _VALIDATOR_FAILURE_MARKERS):
        return "failed"
    return "pipeline_error"


async def run_one_scene(
    client: httpx.AsyncClient,
    spec: SceneSpec,
    *,
    run_date: str,
    out_dir: Path,
    poll_interval_sec: float,
    timeout_sec: float,
) -> SceneRunResult:
    result = SceneRunResult(name=spec.name, project_id=str(spec.project_id))
    start = datetime.now(timezone.utc)
    result.started_at = start.isoformat()

    if spec.existing_video_id:
        result.video_id = spec.existing_video_id
    else:
        try:
            video_id = await upload_scene(client, spec, run_date)
            result.video_id = video_id
        except Exception as exc:
            result.status = "error"
            result.error_message = f"upload failed: {exc}"
            result.finished_at = datetime.now(timezone.utc).isoformat()
            return result

    try:
        status_payload = await poll_scene_status(
            client, result.video_id, poll_interval_sec=poll_interval_sec, timeout_sec=timeout_sec
        )
    except Exception as exc:
        result.status = "error"
        result.error_message = f"polling failed: {exc}"
        result.finished_at = datetime.now(timezone.utc).isoformat()
        return result

    result.scene_id = str(status_payload.get("id")) if status_payload.get("id") else None
    result.status = status_payload.get("status", "unknown")
    result.error_message = status_payload.get("error_message")

    if result.scene_id:
        try:
            detail = await fetch_scene_detail(client, result.scene_id)
            result.floor_y = detail.get("floor_y")
            result.ceiling_y = detail.get("ceiling_y")
            if result.floor_y is not None and result.ceiling_y is not None:
                result.ceiling_height_m = result.ceiling_y - result.floor_y
            objects = detail.get("objects")
            if objects is not None:
                result.num_objects = len(objects)
            if not result.error_message:
                result.error_message = detail.get("error_message")
        except Exception:
            logger.warning("could not fetch scene detail for %s", spec.name, exc_info=True)

        try:
            scene_uuid = uuid.UUID(result.scene_id)
            worker_log = scene_service.scene_dir(scene_uuid) / "logs" / "worker.log"
            result.stage_timings = parse_stage_timings(worker_log.read_text())
        except Exception:
            logger.info("no local worker.log stage timings available for %s", spec.name)

    result.validator_status = derive_validator_status(result.status, result.error_message)

    if result.status == "done" and result.scene_id:
        try:
            usd_dir = out_dir / "usd"
            usd_dir.mkdir(parents=True, exist_ok=True)
            usd_path = usd_dir / f"{spec.name}.usd"
            await fetch_scene_usd(client, result.scene_id, usd_path)
            json_out = usd_dir / f"{spec.name}_validate.json"
            result.isaac_validate = run_isaac_validate_usd_only(usd_path, json_out)
        except Exception as exc:
            logger.warning("isaac_validate --usd-only failed for %s: %s", spec.name, exc, exc_info=True)
            result.isaac_validate = {"status": "error", "exit_code": None, "error": str(exc)}

    finish = datetime.now(timezone.utc)
    result.finished_at = finish.isoformat()
    result.total_duration_sec = (finish - start).total_seconds()
    return result


# --------------------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--date", default=None, help="Run date (YYYY-MM-DD), defaults to today (UTC)")
    p.add_argument("--out-dir", default=None, help="Defaults to var/scratch/nightly/<date>")
    p.add_argument("--poll-interval-sec", type=float, default=DEFAULT_POLL_INTERVAL_SEC)
    p.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument(
        "--project", action="append", default=None,
        help="Restrict to this project name (repeatable); default is every discovered regression scene",
    )
    p.add_argument(
        "--resume-map", default=None,
        help=(
            "Path to a JSON file mapping project_id (string) -> already-uploaded video_id, "
            "for resuming a run that already uploaded some scenes (e.g. after this script "
            "crashed/was interrupted after upload but before collecting results) without "
            "re-uploading and creating a duplicate GPU job for them."
        ),
    )
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    run_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir) if args.out_dir else Path("var/scratch/nightly") / run_date
    only_names = set(args.project) if args.project else None

    resume_map: dict[str, str] = {}
    if args.resume_map:
        resume_map = json.loads(Path(args.resume_map).read_text())

    retain_per_view_effective = detect_retain_per_view_setting()
    if not retain_per_view_effective:
        logger.warning(
            "retain_per_view_artifacts is False on the running video-worker.service "
            "(process-wide, no per-request API override - see this script's module "
            "docstring). Successful scenes this run will NOT retain per_view artifacts."
        )

    async with async_session_maker() as session:
        specs = await discover_regression_scenes(session, only_names=only_names)

    if resume_map:
        specs = [
            spec if str(spec.project_id) not in resume_map
            else dataclasses.replace(spec, existing_video_id=resume_map[str(spec.project_id)])
            for spec in specs
        ]

    if not specs:
        logger.error("no regression scenes discovered - nothing to run")
        return 1

    logger.info("running %d regression scenes: %s", len(specs), [s.name for s in specs])

    results: list[SceneRunResult] = []
    async with httpx.AsyncClient(base_url=args.api_base, timeout=60.0) as client:
        for spec in specs:
            logger.info("=== scene: %s ===", spec.name)
            result = await run_one_scene(
                client,
                spec,
                run_date=run_date,
                out_dir=out_dir,
                poll_interval_sec=args.poll_interval_sec,
                timeout_sec=args.timeout_sec,
            )
            results.append(result)
            logger.info("=== scene %s finished: status=%s ===", spec.name, result.status)

    generated_at = datetime.now(timezone.utc).isoformat()
    write_report(
        out_dir,
        run_date=run_date,
        generated_at=generated_at,
        retain_per_view_artifacts_effective=retain_per_view_effective,
        results=results,
    )
    logger.info("report written to %s", out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
