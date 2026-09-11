"""What the processing screen shows, assembled on the SERVER from files on disk.

Everything the screen renders comes from here on every poll. It holds no state of its
own and nothing is kept in the browser: two tabs on the same URL, and a reload at any
moment, must produce the same feed, which is the property the acceptance probe checks.

Two sources, in order of authority (see docs/JOB_SPEC.md section 2 and 3):

1. `<scene_dir>/status.json`, written by pipeline-v1 (READ-ONLY from here). Authoritative
   whenever it exists: per-step state, timing, and error.
2. `<scene_dir>/logs/worker.log`, parsed by the existing
   `stage_timings.parse_stage_timings`. A degraded source - it carries no per-step error,
   no artifacts and no terminal state - used only when there is no status.json.

Nothing is invented. When neither source has a record, the answer is "no records", and
the screen says so rather than spinning.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.services import scene_service
from app.services.job_spec import JobSpec
from app.services.stage_timings import parse_stage_timings

logger = logging.getLogger(__name__)

#: With no record for this long, the screen states "worker not connected" instead of
#: showing a spinner. A spinner claims progress that nothing has observed; on this box
#: nothing runs the pipeline at all, so that claim would be false every time.
WORKER_SILENT_AFTER_SEC = 60

#: pipeline-v1's terminal states (`run_pipeline.STATES` ends at DONE; FAILED is set by
#: Status.terminal and is deliberately not in that list).
TERMINAL_OK = "DONE"
TERMINAL_FAILED = "FAILED"


#: `substage`'s closed vocabulary, copied from pipeline-v1's `run_pipeline.SUBSTAGES`.
#: Kept here so a reader can tell a value the writer agreed to from a typo, but NOT used
#: to filter: an unknown substage is passed through and shown as itself, the same
#: convention `state` already follows. Silently dropping a value the worker wrote would
#: turn a spelling mistake into an invisible one.
SUBSTAGES = (
    "searching",
    "creating",
    "provisioning",
    "waiting_capacity",
    "transfer_gate",
    "refused",
    "waiting_manual",
    "skipped",
)

#: The two substages the screen must not let scroll past: one says a human has to do
#: something, the other says the run was turned away. Both get a banner, not a row.
SUBSTAGES_NEEDING_ATTENTION = ("refused", "waiting_manual")


@dataclass
class StageRow:
    stage: str
    #: "running" | "ok" | "failed" | whatever pipeline-v1 wrote. Passed through verbatim
    #: rather than mapped, so an unfamiliar value shows up as itself instead of silently
    #: becoming "unknown".
    state: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    #: What the step is doing INSIDE its state - see run_pipeline.SUBSTAGES. `state` says
    #: whether a step finished; this says why it has not, which is the difference between
    #: "GPU_UP_NV has been running 11 minutes because it is broken" and "...because vast
    #: has no 4090 free and it is on attempt 4 of 12".
    substage: str | None = None
    detail: str | None = None
    #: When the CURRENT SUBSTAGE began - not when the step did. The two differ exactly
    #: when a step waits, which is the case worth showing (run_pipeline.Status.enter).
    since: str | None = None
    #: 0 until a retry loop actually starts counting. Rendered only when > 0: "attempt 0"
    #: is the initial value, not an attempt.
    attempt: int | None = None
    next_retry: str | None = None

    @property
    def needs_attention(self) -> bool:
        return self.substage in SUBSTAGES_NEEDING_ATTENTION


@dataclass
class ProcessingStatus:
    workspace_id: uuid.UUID
    scene_id: uuid.UUID | None
    #: "status.json" | "worker.log" | "none" - which source the rows came from. Shown, so
    #: a degraded feed is never mistaken for the authoritative one.
    source: str
    state: str | None
    stages: list[StageRow] = field(default_factory=list)
    error: str | None = None
    #: Seconds since the most recent record from either source, or None when there has
    #: never been one.
    seconds_since_last_record: float | None = None
    worker_connected: bool = False
    requested_provider: str | None = None
    provider_used: str | None = None
    fallback_reason: str | None = None

    @property
    def is_done(self) -> bool:
        return self.state == TERMINAL_OK

    @property
    def is_failed(self) -> bool:
        return self.state == TERMINAL_FAILED


def _parse_ts(value: str | None) -> dt.datetime | None:
    """pipeline-v1 writes LOCAL time with an offset (see docs/JOB_SPEC.md). A naive
    fallback is assumed local too - worker.log's own timestamps have no offset at all."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.astimezone()


def _rows_from_status(doc: dict) -> list[StageRow]:
    """`steps` is a dict, and dict order is insertion order - which for pipeline-v1 is the
    order the states were entered, since `Status.enter` adds each one as it starts. So the
    feed is chronological without needing to sort by a timestamp that may be absent."""
    rows: list[StageRow] = []
    for name, step in (doc.get("steps") or {}).items():
        if not isinstance(step, dict):
            continue
        # Every substage field is optional and absent means absent. The recorded
        # own_0901_161054 run carries none of them - it predates the fields - and that is
        # the case the screen has to handle by omitting rows rather than printing "—".
        attempt = step.get("attempt")
        rows.append(
            StageRow(
                stage=str(name),
                state=str(step.get("state") or "unknown"),
                started_at=step.get("started"),
                finished_at=step.get("finished"),
                duration_sec=step.get("duration_s"),
                error=step.get("error"),
                substage=step.get("substage") or None,
                detail=step.get("detail") or None,
                since=step.get("since") or None,
                # `attempt` is an int, so `or None` would swallow a legitimate 0 - which
                # is exactly the value `Status.enter` initialises it to and which must not
                # reach the screen as an attempt. Kept as 0 here, filtered at the edge.
                attempt=attempt if isinstance(attempt, int) else None,
                next_retry=step.get("next_retry") or None,
            )
        )
    return rows


def _latest_record_at(rows: list[StageRow], doc: dict | None) -> dt.datetime | None:
    stamps = [
        _parse_ts(t)
        for row in rows
        for t in (row.finished_at, row.started_at)
    ]
    if doc:
        stamps.append(_parse_ts(doc.get("finished")))
        stamps.append(_parse_ts(doc.get("started")))
    real = [s for s in stamps if s is not None]
    return max(real) if real else None


def read_processing_status(
    workspace_id: uuid.UUID,
    scene_id: uuid.UUID | None,
    *,
    spec: JobSpec | None = None,
    now: dt.datetime | None = None,
) -> ProcessingStatus:
    """Assemble the screen's whole state. Never raises for a missing file: "there is
    nothing yet" is a normal answer here, not an error."""
    now = now or dt.datetime.now().astimezone()
    result = ProcessingStatus(
        workspace_id=workspace_id,
        scene_id=scene_id,
        source="none",
        state=None,
        requested_provider=spec.semantics if spec else None,
    )
    if scene_id is None:
        return result

    scene_dir = scene_service.scene_dir(scene_id)
    status_path = scene_dir / "status.json"
    doc: dict | None = None
    if status_path.is_file():
        try:
            doc = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # status.json is written atomically by pipeline-v1, so a parse failure is a
            # real problem rather than a torn read - logged, and treated as "no
            # status.json" so the worker.log fallback still gets its chance.
            logger.warning("could not read status.json for scene %s", scene_id, exc_info=True)
            doc = None

    if isinstance(doc, dict):
        result.source = "status.json"
        result.state = doc.get("state")
        result.error = doc.get("error")
        result.stages = _rows_from_status(doc)
        # Contract additions pipeline-v1 does not write yet (docs/JOB_SPEC.md section 2).
        # Absent must read as "no fallback happened", never as a notice.
        result.provider_used = doc.get("provider_used")
        result.fallback_reason = doc.get("fallback_reason")
    else:
        log_path = scene_dir / "logs" / "worker.log"
        if log_path.is_file():
            try:
                timings = parse_stage_timings(log_path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                timings = []
            if timings:
                result.source = "worker.log"
                result.stages = [
                    StageRow(
                        stage=t.stage,
                        # worker.log's transitions only ever say a stage STARTED; a stage
                        # with a measured duration is one a later transition closed.
                        state="ok" if t.duration_sec is not None else "running",
                        started_at=t.started_at,
                        duration_sec=t.duration_sec,
                    )
                    for t in timings
                ]
                result.state = result.stages[-1].stage

    last = _latest_record_at(result.stages, doc)
    if last is not None:
        result.seconds_since_last_record = max(0.0, (now - last).total_seconds())
        result.worker_connected = result.seconds_since_last_record < WORKER_SILENT_AFTER_SEC
    return result


def fallback_notice(status: ProcessingStatus) -> str | None:
    """The "your provider was unavailable" line, or None.

    Fires only when the pipeline actually reported a provider AND it differs from the one
    the job spec asked for. With `provider_used` absent - which is every run today, since
    pipeline-v1 writes neither field - this is None and the screen shows nothing.
    """
    used = status.provider_used
    if not used or not status.requested_provider or used == status.requested_provider:
        return None
    reason = status.fallback_reason or "reason not reported"
    return f"the chosen provider was unavailable: {reason}, used {used}"
