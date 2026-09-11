#!/usr/bin/env python3
"""The bridge between the arq worker and `run_pipeline.py`.

`app/worker.py` used to call `pipeline_orchestrator` + `gpu_client`, whose `gpu` ssh alias
resolves to instance **49271597 - a dead box** (exited, credentials shredded off it). The
production worker therefore could not process anything. This module points it at the driver
that has actually completed a live run.

What this module owns, and nothing else:

* reading `<uuid>.job.json` beside the video (`app/services/job_spec.py`, cherry-picked
  unmodified from the branch that wrote the contract - `docs/JOB_SPEC.md`);
* turning that spec into `run_pipeline.py` arguments;
* the worker-mode guards, which replace ARMED - see `worker_mode()`.

What it deliberately does NOT own: the DB. `app/services/scene_ingest.py` and
`scene_service.*` keep every row they wrote before; this module returns a result and the
worker does the writing exactly where it already did.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Env that replaces ARMED in worker mode. Absent means DRY RUN: the driver runs, writes
#: status.json, and makes no vast call and no spend. Renting is opt-in, never a default.
ENABLE_ENV = "PIPELINE_WORKER_ENABLED"
#: Daily ceiling in USD, checked against the same ledger `--max-usd` uses - create (or
#: start) to a confirmed `exited`. Absent means the per-run cap alone applies.
DAILY_CAP_ENV = "PIPELINE_DAILY_USD"
#: Where the day's spend is accumulated across jobs, so the cap survives a worker restart.
LEDGER = HERE / "state" / "daily_spend.json"

#: Where hand-driven MapAnything pass-A bundles live. Overridable so a test (and a future
#: move off this scratch path) does not need a code change.
PASS_A_ROOT_ENV = "PIPELINE_PASS_A_ROOT"
DEFAULT_PASS_A_ROOT = HERE.parent / "var" / "pass_a"

#: Colour. The driver's own default is ON; set this to "0" to run the colourless path the
#: pipeline produced before 2026-09-10. It is an env var rather than a spec field because
#: it is an operator decision about ONE session's risk appetite - "is the colour path
#: trusted on this box yet" - not a property of the job someone uploaded.
COLOR_ENV = "PIPELINE_COLOR"

#: Leave the GPU box running when the run ends, instead of stopping it in GPU_DOWN_NV.
#: Renting is cheap per hour and a REFUSED START is not recoverable in the same session -
#: that is what retired 50265868 and stalled the 2026-09-09 mvfilter run four times. When
#: several runs are queued back to back, holding one box across them trades a known hourly
#: cost against an unbounded "no capacity anywhere" risk. Off by default: an unattended
#: worker that leaks a running box is worse than one that pays a bring-up per job.
KEEP_BOX_ENV = "PIPELINE_KEEP_BOX_UP"


class JobRefused(RuntimeError):
    """The job cannot start. Distinct from a stage failing: nothing was attempted."""


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def worker_mode() -> tuple[bool, str]:
    """(live, reason). Live requires the enable env set to exactly "1".

    ARMED is a file an operator creates for one morning; a worker runs unattended, so the
    switch is an env var the unit file carries. The default is the safe one in both cases:
    without it the driver still runs and still writes status.json, it simply rents nothing.
    """
    v = os.environ.get(ENABLE_ENV)
    if v == "1":
        return True, f"{ENABLE_ENV}=1"
    if v is None:
        return False, f"{ENABLE_ENV} not set - dry run, no vast call and no spend"
    return False, f"{ENABLE_ENV}={v!r} (only \"1\" enables live) - dry run"


def pass_a_root() -> Path:
    return Path(os.environ.get(PASS_A_ROOT_ENV) or DEFAULT_PASS_A_ROOT)


def resolve_pass_a(video_filename: str, root: Path | None = None) -> Path | None:
    """The pass-A bundle for this video, matched by its ORIGINAL filename stem, or None.

    MapAnything pass A is not automated (`run_pipeline.py:1403-1418`): a B200 is driven by
    hand and the result lands in a directory under `pass_a_root()`. Without `--from-pulled`
    the driver parks MAPANYTHING in `waiting_manual` forever, so the worker was structurally
    unable to finish ANY job - it never passed one.

    The match is deliberately exact rather than fuzzy: `<stem>` or `<stem>__<variant>`.
    A bundle is 1-2 GB of someone's GPU hour; picking the wrong one silently would fuse the
    wrong room into this scene and nothing downstream would notice. So an ambiguous stem is
    a refusal that names every candidate, and no match at all returns None - which leaves
    MAPANYTHING parked by name, which is the correct behaviour for a genuinely new video.
    """
    root = Path(root) if root is not None else pass_a_root()
    stem = Path(str(video_filename or "")).stem.strip()
    if not stem or not root.is_dir():
        return None
    cands = sorted(p for p in root.iterdir()
                   if p.is_dir() and (p.name == stem or p.name.startswith(stem + "__")))
    if len(cands) > 1:
        raise JobRefused(
            f"{len(cands)} pass-A bundles match {stem!r} under {root}: "
            f"{[p.name for p in cands]}. Renaming the upload to one of these, or pruning "
            f"the others, is a human call - this will not guess which room to fuse.")
    return cands[0] if cands else None


def spend_today(ledger: Path = LEDGER) -> float:
    """USD already charged today, from the shared ledger. Missing file reads as 0.0."""
    try:
        doc = json.loads(ledger.read_text())
    except (OSError, ValueError):
        return 0.0
    return float(doc.get(dt.date.today().isoformat(), 0.0))


def add_spend(usd: float, ledger: Path = LEDGER) -> float:
    """Accumulate today's spend and return the new total. Days other than today are kept
    so a week can be read back, but only today's key is ever compared against the cap."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc = json.loads(ledger.read_text())
    except (OSError, ValueError):
        doc = {}
    key = dt.date.today().isoformat()
    doc[key] = round(float(doc.get(key, 0.0)) + float(usd), 6)
    tmp = ledger.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(ledger)
    return doc[key]


def check_daily_cap(ledger: Path = LEDGER) -> tuple[float, float | None]:
    """(spent_today, cap). Raises JobRefused when the cap is already reached.

    Checked BEFORE the driver starts, so an exhausted cap costs nothing: no search, no
    create, no start. That ordering is the whole point - a cap enforced after renting is
    an invoice, not a cap.
    """
    raw = os.environ.get(DAILY_CAP_ENV)
    if raw is None:
        return spend_today(ledger), None
    try:
        cap = float(raw)
    except ValueError:
        raise JobRefused(f"{DAILY_CAP_ENV}={raw!r} is not a number")
    spent = spend_today(ledger)
    if spent >= cap:
        raise JobRefused(
            f"daily cap reached before this job started: ${spent:.4f} of ${cap:.2f} "
            f"already spent today ({DAILY_CAP_ENV}); nothing was rented")
    return spent, cap


def build_argv(video: Path, scene_dir: Path, spec, *, scene_name: str,
               from_pulled: Path | None, live: bool, max_usd: float,
               step_timeout_s: int, extra: list[str] | None = None) -> list[str]:
    """The exact command line the worker runs. Returned rather than executed so a caller
    (and a test) can assert on it without spending anything."""
    argv = [sys.executable, "-u", str(HERE / "run_pipeline.py"), str(video),
            "--scene-dir", str(scene_dir),
            "--scene-name", scene_name,
            "--frames-fps", str(spec.frames_fps),
            "--semantics", str(spec.semantics),
            "--max-usd", str(max_usd),
            "--step-timeout-s", str(step_timeout_s),
            "--worker-mode"]
    if from_pulled is not None:
        argv += ["--from-pulled", str(from_pulled)]
    argv += ["--nvblox-source", "live" if live else "precomputed"]
    if os.environ.get(COLOR_ENV) == "0":
        argv.append("--no-color")
    if os.environ.get(KEEP_BOX_ENV) == "1":
        argv.append("--keep-box-up")
    return argv + (extra or [])


def run_job(video: Path, scene_dir: Path, spec, *, scene_name: str,
            from_pulled: Path | None = None, max_usd: float = 3.0,
            step_timeout_s: int = 900, ledger: Path = LEDGER,
            extra: list[str] | None = None) -> dict:
    """Run one job to completion. Returns the parsed status.json.

    Raises JobRefused before doing anything when the daily cap is already spent.
    """
    live, why = worker_mode()
    spent, cap = check_daily_cap(ledger)

    # The per-run cap is clamped to what is left of the day. Without this a $3 job started
    # with $2.90 of a $3.00 daily cap already spent would be allowed to run to $3 of its
    # own, i.e. to $5.90 for the day - a daily cap that only refuses the NEXT job is not a
    # daily cap. The driver's own supervisor enforces the clamped number mid-run by killing
    # the running stage, so this bound is live, not advisory.
    if cap is not None:
        max_usd = round(min(max_usd, max(0.0, cap - spent)), 4)

    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(video, scene_dir, spec, scene_name=scene_name,
                      from_pulled=from_pulled, live=live, max_usd=max_usd,
                      step_timeout_s=step_timeout_s, extra=extra)

    env = dict(os.environ)
    if not live:
        # The driver's own dry-run switch. Belt and braces with `worker_mode`: the driver
        # refuses a live vast call without it regardless of what this module intended.
        env["PIPELINE_DRY_RUN"] = "1"
    else:
        env.pop("PIPELINE_DRY_RUN", None)

    (scene_dir / "logs").mkdir(exist_ok=True)
    (scene_dir / "logs" / "worker.argv").write_text(" ".join(argv) + "\n")
    proc = subprocess.run(argv, capture_output=True, text=True, env=env,
                          timeout=step_timeout_s * 20)
    (scene_dir / "logs" / "worker.log").write_text(proc.stdout + proc.stderr)

    try:
        status = json.loads((scene_dir / "status.json").read_text())
    except (OSError, ValueError) as e:
        raise JobRefused(f"driver produced no readable status.json: {e}")

    cost = float(status.get("cost_estimate_usd") or 0.0)
    if cost > 0:
        status["daily_spend_usd"] = add_spend(cost, ledger)
    status["worker_mode"] = why
    status["daily_cap_usd"] = cap
    status["daily_spent_before_usd"] = round(spent, 6)
    status["max_usd_after_daily_clamp"] = max_usd
    return status
