"""Per-job pipeline stage timing: parsing, on-disk persistence, and lookup.

`pipeline_orchestrator.py`'s `_poll_until_terminal` already logs one
`stage=NAME (i/n) state=running message=starting NAME` line per stage transition
(picked up by `scene_file_log`, which writes it to this scene's own
`{scene_dir}/logs/worker.log`). `scripts/nightly_batch.py` was the first consumer of
this: it re-parses that log after a run to compute each stage's wall-clock duration
(consecutive transitions' timestamp deltas) for its report.md/results.json. This module
factors that parsing out to a shared, backend-importable home (`nightly_batch.py` now
just re-exports `StageTiming`/`parse_stage_timings` from here - no behavior change) and
adds a second consumer: a small `stage_timings.json` cache file written next to
`logs/worker.log` in the scene's own directory, so `GET /scenes/{id}/status` can return
per-stage timing (current stage, elapsed-in-stage, historical-style duration list)
without every poller re-reading and re-parsing the whole log file - see
`refresh_stage_timings_from_log`.

Written atomically (tmp file + `os.replace`), mirroring `gpu/job_io.py`'s
`write_status` pattern, so a concurrent `GET /status` read can never observe a
half-written file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StageTiming:
    stage: str
    started_at: str  # ISO8601
    duration_sec: float | None = None  # None for the still-running (last observed) stage


_STAGE_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*stage=(?P<stage>\S+) \(\d+/\d+\) state=(?P<state>\S+)"
)


def parse_stage_timings(worker_log_text: str) -> list[StageTiming]:
    """Parse `logs/worker.log`'s `stage=NAME (i/n) state=running ...` lines into a list
    of StageTiming, one per stage transition, with each stage's duration computed as the
    delta to the *next* transition's timestamp (the last stage's duration is left None -
    there is no next boundary to measure against; `state=done`'s own line, when present,
    is not itself a new stage - finalize's "message=pipeline complete" line already
    covers that)."""
    events: list[tuple[datetime, str]] = []
    for line in worker_log_text.splitlines():
        m = _STAGE_LINE_RE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
        events.append((ts, m.group("stage")))

    timings: list[StageTiming] = []
    for i, (ts, stage) in enumerate(events):
        duration = None
        if i + 1 < len(events):
            duration = (events[i + 1][0] - ts).total_seconds()
        timings.append(StageTiming(stage=stage, started_at=ts.isoformat(), duration_sec=duration))
    return timings


def stage_timings_path(scene_dir: Path) -> Path:
    return scene_dir / "stage_timings.json"


def write_stage_timings(scene_dir: Path, timings: list[StageTiming]) -> None:
    """Atomically overwrite stage_timings.json in `scene_dir`."""
    path = stage_timings_path(scene_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps([asdict(t) for t in timings], indent=2))
    os.replace(tmp_path, path)


def read_stage_timings(scene_dir: Path) -> list[StageTiming]:
    """Read the persisted stage_timings.json, or [] if it doesn't exist / is unreadable
    (e.g. a scene whose pipeline failed before any stage line was ever written)."""
    path = stage_timings_path(scene_dir)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [StageTiming(**t) for t in raw]


def refresh_stage_timings_from_log(scene_dir: Path) -> list[StageTiming]:
    """Re-parse `{scene_dir}/logs/worker.log` and persist the result to
    `stage_timings.json`. Best-effort: returns [] and never raises if the log can't be
    read (scene not yet processing, or removed) - called both while a job is still
    `processing` (from `GET /scenes/{id}/status`, so a live poll always reflects the
    current stage) and once more after the job reaches a terminal state
    (`pipeline_orchestrator.run_pipeline_for_scene`), so the persisted file always ends
    up complete."""
    log_path = scene_dir / "logs" / "worker.log"
    try:
        text = log_path.read_text()
    except OSError:
        return []
    timings = parse_stage_timings(text)
    try:
        write_stage_timings(scene_dir, timings)
    except OSError:
        logger.warning("could not persist stage_timings.json under %s", scene_dir, exc_info=True)
    return timings
