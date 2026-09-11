#!/usr/bin/env python3
"""Deterministic `video -> scene catalogue` pipeline, assembled from the scripts that
already exist, as ONE state machine.

    PIPELINE_DRY_RUN=1 pipeline/run_pipeline.py <video.mp4> --scene-dir <dir>   # rehearsal
    env -u PIPELINE_DRY_RUN pipeline/run_pipeline.py ... --nvblox-source live   # live, needs ARMED

States, in fixed order - one box per GPU stage, each stopped in its own finally:

    QUEUED -> FRAMES -> GPU_UP_MA -> MAPANYTHING -> PULLED -> GPU_DOWN_MA
           -> SEMANTICS -> PACK -> MVFILTER
           -> GPU_UP_NV -> NVBLOX -> FLOOR -> GPU_DOWN_NV
           -> LAYERS -> REACH -> EXPORT -> INGEST -> DONE | FAILED

Each step also carries `substage` / `detail` / `since` / `attempt` / `next_retry` in
status.json, so a long-running step says WHY it is long: `provisioning` a box, sitting in
`waiting_capacity` because vast has none free, held at the `transfer_gate`, or parked in
`waiting_manual` for the one stage that is not automated. SEMANTICS is a declared slot with
no implementation - it reports `skipped` / "not implemented" rather than being left out.

PACK (per-view npz -> the uint16 depth stack nvblox actually eats) is not in the original
list of links but is mandatory: `nvblox_scenes.py` reads `depth_u16.npy`, never `per_view/`.

MVFILTER is optional and OFF by default (`--mvfilter`). It is the one place a depth pixel can
be dropped before nvblox fuses it - see `steps/mvfilter.py`. It sits before GPU_UP_NV on
purpose: it is CPU-only, so a filter that fails costs no box time.

Every step is idempotent: its `check()` runs FIRST, and if the output is already there and
passes (file count / size / JSON keys), the step is recorded as `skipped` and nothing runs.
A step whose output was produced by an earlier session elsewhere on disk is recorded as
`skipped_precomputed`, with the source path in `artifacts` - so the status file never
implies this run computed something it did not.

Refusals and safety:

* Either `PIPELINE_DRY_RUN=1` (rehearsal) or a valid `pipeline/ARMED` (live). With neither,
  the driver exits 2 before doing anything at all.
* vast.ai goes through `pipeline/vast_client.py`, which refuses a real call unless
  `PIPELINE_DRY_RUN` is absent AND `pipeline/ARMED` says `yes <today>`; every call is logged
  as RAN / WOULD_RUN / REFUSED either way.
* Every box brought up is stopped from its own `finally`, on success and on failure alike;
  a failure stopping one box does not prevent the other from being stopped. `destroy` is
  never issued by the driver - 50265868 carries the only working nvblox install.
* Cost is a measured ledger (`dph_total` x time up, per box, kept after the box stops), not
  an estimate; exceeding `--max-usd` kills the running stage rather than waiting for it.
* Monitoring is by PID file (`<scene_dir>/pipeline.pid`) + `<scene_dir>/heartbeat`, never by
  `pgrep -f <script name>` - that matches the harness's own wrapper shell and has made a dead
  job look alive for over an hour.
* A failing step records the FULL stderr in status.json and in `<scene_dir>/logs/<STATE>.stderr`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vast_client import (VastClient, VastNotArmed, VastStartUnavailable,  # noqa: E402
                         NVBLOX_BOX_ID, STATE_DIR)
from box_ops import BoxSession, select_gpu  # noqa: E402
from cpu_env import cpu_python, require_cpu_python  # noqa: E402

STATES = ["QUEUED", "FRAMES", "GPU_UP_MA", "MAPANYTHING", "PULLED", "GPU_DOWN_MA",
          "SEMANTICS", "PACK", "PACK_RGB", "MVFILTER",
          "GPU_UP_NV", "NVBLOX", "FLOOR", "GPU_DOWN_NV",
          "LAYERS", "REACH", "EXPORT", "INGEST", "DONE"]

# The vocabulary of `substage` in status.json. A step's `state` says whether it finished;
# `substage` says what it is doing *inside* that, which is the difference between "this run
# has been in GPU_UP_NV for 11 minutes because it is broken" and "...because vast has no
# 4090 free and it is on attempt 4 of 12". Closed set on purpose: a reader that switches on
# these must not meet a spelling nobody agreed to.
SUBSTAGES = ("searching", "creating", "provisioning", "waiting_capacity", "transfer_gate",
             "refused", "waiting_manual", "skipped")

HEARTBEAT_S = 30
BUDGET_POLL_S = 30

# The waiting_capacity ceiling. When a host has no resources free, `start_or_create` searches
# and rents a replacement - but on 2026-09-09 that path itself came up empty twice, and an
# unattended worker must not sit in that loop forever. Whichever limit is hit first ends the
# run as FAILED with the reason, rather than accumulating cost and silence.
WAITING_CAPACITY_MAX_ATTEMPTS = 12
WAITING_CAPACITY_MAX_S = 60 * 60
WAITING_CAPACITY_BACKOFF_S = 60
SHOW_FAILURES_BEFORE_UNREACHABLE = 3


class BudgetExceeded(Exception):
    pass


# Exactly what the downstream stages open out of an nvblox output directory. Verified by
# reading them, not guessed:
#   frontend_layers.py:168-173  esdf_slice_0.3m.npy, unobserved_slice.npy, floor_plane.json
#   reachability.py:179-197     the same three, plus mesh.ply
#   isaac_export.py:97-98       floor_plane.json, the mesh named by --mesh-name
#   check_nvblox (this file)    stats.json
# Everything else nvblox writes is left on the box. `map.nvblx` alone is 133.7 MB against
# 16.9 MB for this whole list, and on the 2026-09-09 run at 0.32 MB/s down it cost 483 s of
# a 567 s step for data nothing reads.
NVBLOX_PULL_FILES = ["esdf_slice_0.3m.npy", "unobserved_slice.npy", "floor_plane.json",
                     "mesh.ply", "stats.json"]


#: What PACK_RGB writes next to the PACK output, and what NVBLOX pushes so the remote
#: script can call add_color_frame. Must match steps/pack_rgb.py's --out default.
RGB_STACK = "rgb_u8.npy"

#: What MVFILTER writes next to the PACK output when it is enabled, and what NVBLOX then
#: pushes and fuses instead of `depth_u16.npy`. Must match steps/mvfilter.py's OUT_DEPTH.
MVFILTER_DEPTH = "depth_u16_mvfilter.npy"
MVFILTER_META = "mvfilter.json"
#: Parameters of the vote, in the order --mvfilter accepts them. Value types are the parser.
MVFILTER_KEYS = {"m": int, "tau": float, "neighbours": str, "K": int}


def mvfilter_slug(spec: dict) -> str:
    """A filename that says which vote made the stack. Two arms in one run dir differ only
    in these four numbers, so the numbers are the name."""
    return f"m{spec['m']}_t{spec['tau']:g}_{spec['neighbours']}_K{spec['K']}"


def parse_mvfilter(spec: str) -> dict:
    """`m=2,tau=0.10,neighbours=nearest,K=8` -> a dict, or raise ValueError saying why.

    Every key is required. A spec with a default for the ones you leave out would make two
    runs whose status.json says `--mvfilter` mean different filters, and the whole point of
    this step is that the run records exactly which vote produced its depth.
    """
    got = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{part!r} is not key=value")
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key not in MVFILTER_KEYS:
            raise ValueError(f"unknown key {key!r}; expected {', '.join(MVFILTER_KEYS)}")
        if key in got:
            raise ValueError(f"{key!r} given twice")
        try:
            got[key] = MVFILTER_KEYS[key](val)
        except ValueError:
            raise ValueError(f"{key}={val!r} is not a {MVFILTER_KEYS[key].__name__}") from None
    missing = [k for k in MVFILTER_KEYS if k not in got]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    if got["m"] < 1:
        raise ValueError(f"m={got['m']} keeps every pixel; leave --mvfilter off instead")
    if got["tau"] <= 0:
        raise ValueError(f"tau={got['tau']} must be positive")
    if got["K"] < got["m"]:
        raise ValueError(f"K={got['K']} is below m={got['m']}: no pixel could ever survive")
    if got["neighbours"] not in ("nearest", "baseline15", "framegap8"):
        raise ValueError(f"neighbours={got['neighbours']!r}; expected nearest, baseline15 "
                         f"or framegap8")
    return got


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class StepFailed(Exception):
    def __init__(self, state: str, message: str, stderr: str = "",
                 fail_reason: str | None = None):
        super().__init__(message)
        self.state = state
        self.message = message
        self.stderr = stderr
        # A short, machine-readable cause for status.json, distinct from the human message.
        self.fail_reason = fail_reason


# --------------------------------------------------------------------------------------
# status file
# --------------------------------------------------------------------------------------
class Status:
    """`<scene_dir>/status.json`, rewritten atomically (tmp + rename) after every change."""

    def __init__(self, path: Path, scene: str, video: str):
        self.path = Path(path)
        # The supervisor thread writes cost/box fields while the main thread writes steps.
        # Without this lock both threads race on the same tmp file and one rename loses.
        self.lock = threading.RLock()
        self.t0 = time.time()
        self.doc = {
            "scene": scene,
            "video": video,
            "pipeline_dry_run": True,
            "state": "QUEUED",
            "started": now(),
            "finished": None,
            "duration_s": None,
            "cost_estimate_usd": 0.0,
            "cost_basis": "measured: dph_total from `vastai show instance` x time up",
            "max_usd": None,
            "boxes": {},
            "error": None,
            "steps": {},
        }
        self.flush()

    def flush(self) -> None:
        with self.lock:
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.doc, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)

    def enter(self, state: str) -> None:
        with self.lock:
            self.doc["state"] = state
            self.doc["steps"][state] = {
                "state": "running", "started": now(), "finished": None,
                "duration_s": None, "cost_estimate_usd": 0.0,
                # substage/detail/since/attempt/next_retry: what the step is doing right now.
                # `since` is when the CURRENT substage began, not when the step did - the two
                # differ exactly when a step waits, which is the case worth showing.
                "substage": None, "detail": None, "since": now(),
                "attempt": 0, "next_retry": None,
                "error": None, "artifacts": {}}
            self.flush()

    def set_substage(self, state: str, substage: str | None, detail: str | None = None,
                     *, attempt: int | None = None, next_retry: str | None = None) -> None:
        """Record what a running step is doing. Safe to call from either thread."""
        if substage is not None and substage not in SUBSTAGES:
            raise ValueError(f"unknown substage {substage!r}; known: {SUBSTAGES}")
        with self.lock:
            s = self.doc["steps"].get(state)
            if s is None:
                return
            if s.get("substage") != substage:
                s["since"] = now()
            s["substage"] = substage
            if detail is not None:
                s["detail"] = detail
            if attempt is not None:
                s["attempt"] = attempt
            # next_retry is cleared by passing None only together with a new substage: a
            # step that is no longer waiting must not keep advertising a retry time.
            s["next_retry"] = next_retry
            self.flush()

    def set_provider(self, provider_used: str | None,
                     fallback_reason: str | None = None) -> None:
        """Top-level `provider_used` / `fallback_reason`, under JOB_SPEC 5b's rules.

        Both are optional and absent means "no fallback happened" - so this writes nothing
        when the provider is unknown. `fallback_reason` is written only when there is a real
        one: a `provider_used` with no reason renders as "reason not reported", and JOB_SPEC
        says in as many words not to invent a placeholder to avoid that.
        """
        if provider_used is None:
            return
        with self.lock:
            self.doc["provider_used"] = provider_used
            if fallback_reason:
                self.doc["fallback_reason"] = fallback_reason
            self.flush()

    def finish_step(self, state: str, outcome: str, artifacts: dict,
                    cost: float = 0.0, error: str | None = None,
                    substage: str | None = None, detail: str | None = None) -> None:
        with self.lock:
            s = self.doc["steps"][state]
            s["state"] = outcome
            s["finished"] = now()
            started = dt.datetime.fromisoformat(s["started"])
            s["duration_s"] = round(
                (dt.datetime.now().astimezone() - started).total_seconds(), 2)
            s["cost_estimate_usd"] = round(cost, 4)
            s["artifacts"] = artifacts
            s["error"] = error
            # Only overwrite when told to. A step that failed while waiting keeps the
            # substage that explains why - the failure path passes neither.
            if substage is not None:
                if substage not in SUBSTAGES:
                    raise ValueError(f"unknown substage {substage!r}")
                s["substage"] = substage
            if detail is not None:
                s["detail"] = detail
            self.flush()

    def set_box(self, instance_id: int, info: dict) -> None:
        with self.lock:
            self.doc["boxes"][str(instance_id)] = dict(info)
            self.flush()

    def set_cost(self, usd: float) -> None:
        with self.lock:
            self.doc["cost_estimate_usd"] = round(usd, 4)
            self.flush()

    def terminal(self, state: str, error: str | None = None,
                 fail_reason: str | None = None) -> None:
        self.doc["state"] = state
        self.doc["fail_reason"] = fail_reason
        self.doc["finished"] = now()
        self.doc["duration_s"] = round(time.time() - self.t0, 2)
        self.doc["error"] = error
        self.flush()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def run_cmd(argv: list[str], log_dir: Path, tag: str, timeout_s: int,
            cwd: Path | None = None, register=None) -> dict:
    """Run one existing script. Full argv, stdout and stderr are kept on disk.

    Popen rather than subprocess.run so the live handle can be registered with the
    supervisor thread - a budget overrun has to be able to cut a stage short, not wait
    for a 900 s timeout to expire.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{tag}.argv").write_text(" ".join(argv) + "\n", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(cwd) if cwd else None)
    if register:
        register(proc)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        (log_dir / f"{tag}.stderr").write_text(
            f"TIMEOUT after {timeout_s}s\n{err}", encoding="utf-8")
        raise StepFailed(tag, f"timeout after {timeout_s}s", err)
    finally:
        if register:
            register(None)
    (log_dir / f"{tag}.stdout").write_text(out, encoding="utf-8")
    (log_dir / f"{tag}.stderr").write_text(err, encoding="utf-8")
    if proc.returncode != 0:
        raise StepFailed(tag, f"{argv[0]} exited {proc.returncode}", err)
    return {"argv": argv, "returncode": proc.returncode,
            "wall_s": round(time.time() - t0, 2)}


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def need_keys(doc: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in doc]
    if missing:
        raise StepFailed(where, f"{where}: missing keys {missing} in {doc.keys()}")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


ESDF_DIFF_PROBE = (
    "import json,sys,numpy as np\n"
    "a,b=[np.load(x).astype(np.float64) for x in sys.argv[1:3]]\n"
    "if a.shape!=b.shape:\n"
    "    print(json.dumps({'shape_mismatch':[list(a.shape),list(b.shape)]}));raise SystemExit\n"
    "fa,fb=np.isfinite(a),np.isfinite(b);both=fa&fb\n"
    "d=np.abs(a[both]-b[both]) if both.any() else np.zeros(1)\n"
    "print(json.dumps({'shape':list(a.shape),'n_finite_both':int(both.sum()),"
    "'nan_pattern_mismatch_cells':int((fa!=fb).sum()),'max_abs_diff':float(d.max()),"
    "'mean_abs_diff':float(d.mean())}))\n"
)

NPY_PROBE = (
    "import json,sys,numpy as np\n"
    "out={}\n"
    "for name,path in json.loads(sys.argv[1]).items():\n"
    "    a=np.load(path,mmap_mode='r')\n"
    "    out[name]={'shape':list(a.shape),'dtype':str(a.dtype)}\n"
    "print(json.dumps(out))\n"
)


def probe_npy(python_cpu: str, paths: dict) -> dict:
    """Read shape/dtype of .npy files without importing numpy into this driver.

    The driver stays stdlib-only on purpose - same rule as `gpu/stage_*.py`, which are
    stdlib-only so they can be run from any of the venvs on the box. numpy lives only in
    the CPU venv that runs the actual steps, so the driver asks it.
    """
    p = subprocess.run([python_cpu, "-c", NPY_PROBE, json.dumps(paths)],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise StepFailed("probe_npy", f"npy probe failed: {p.stderr.strip()}", p.stderr)
    return json.loads(p.stdout)


# --------------------------------------------------------------------------------------
# supervisor: heartbeat + measured-cost budget
# --------------------------------------------------------------------------------------
class Supervisor(threading.Thread):
    """One background thread doing two jobs on the same 30 s tick.

    **Heartbeat** - rewrites `<scene_dir>/heartbeat` with time, pid and current state.
    Monitoring is by this file plus `<scene_dir>/pipeline.pid`, never by
    `pgrep -f run_pipeline.py`: that matches the harness's own wrapper shell and has made a
    dead job look alive for over an hour.

    **Budget** - for every box currently up, asks `vastai show instance <id> --raw` for
    `dph_total` and multiplies it by the measured time since that box came up. Nothing is
    estimated from a rate typed in advance; if the price was never actually read, no cost is
    claimed and no budget verdict is reached.

    A failed `show` is explicitly NOT a budget breach: it is logged, the failure counter for
    that box goes up, and the next tick tries again. Three consecutive failures raise
    `box_unreachable` in status.json as a *signal only* - the box is not stopped on it, and
    the running stage is left to finish.

    A real overrun sets `abort_reason` and kills the stage's subprocess so a 900 s stage
    cannot keep billing after the limit is crossed.
    """

    def __init__(self, pipeline, interval_s: int = HEARTBEAT_S):
        super().__init__(daemon=True)
        self.p = pipeline
        self.interval = interval_s
        self._stop = threading.Event()
        self.show_failures: dict[int, int] = {}
        self.ledger: dict[int, float] = {}   # instance_id -> last measured cost, USD
        self.last_cost = 0.0

    def beat(self) -> None:
        self.p.hb_path.write_text(
            f"{now()}\tpid={os.getpid()}\tstate={self.p.status.doc['state']}"
            f"\tcost_usd={self.last_cost:.4f}\n", encoding="utf-8")

    def poll_cost(self) -> None:
        """Cost is a LEDGER, not a snapshot of what is up right now.

        A box that has been stopped still cost what it cost: recomputing the total from the
        live sessions alone made the total fall back to zero the moment `GPU_DOWN_NV` ran,
        so a run could never exceed its budget after the box came down. Each box's last
        measured cost stays in `self.ledger` and the total is the sum over all boxes this
        run ever brought up.
        """
        for iid, sess in list(self.p.boxes_seen.items()):
            if sess.up_at is None and sess.billing_from is None:
                continue
            try:
                d = self.p.vast.show(iid, dry_run=self.p.dry_run)
                self.show_failures[iid] = 0
            except Exception as e:
                n = self.show_failures.get(iid, 0) + 1
                self.show_failures[iid] = n
                info = dict(sess.info)
                info["last_show_error"] = f"{type(e).__name__}: {e}"
                info["consecutive_show_failures"] = n
                if n >= SHOW_FAILURES_BEFORE_UNREACHABLE:
                    # A signal, not an action: no stop is issued on this, and the current
                    # stage is allowed to finish.
                    info["box_unreachable"] = True
                self.p.status.set_box(iid, info)
                print(f"[{now()}] show instance {iid} failed ({n} in a row), "
                      f"not a budget verdict: {e}", file=sys.stderr)
                continue
            dph = d.get("dph_total")
            if dph is None:
                continue
            # Charge from when the box STARTED COSTING MONEY, not from this session's own
            # start call. Measured on 2026-09-09: the ledger reported $0.0527 for the run
            # window while the box had actually been billing for 566 s - 252 s of
            # provisioning and 81 s of link probing before the run began - for $0.1293.
            # Undercounting by 2.5x in the direction of "cheaper than it is" is the wrong
            # way to be wrong about a budget.
            start = sess.billing_from or sess.up_at
            end = sess.stopped_at_monotonic or time.time()
            hours = (end - start) / 3600.0
            cost = float(dph) * hours
            self.ledger[iid] = cost
            info = dict(sess.info)
            info.update({"dph_total": dph, "hours_up": round(hours, 4),
                         "cost_usd_measured": round(cost, 4),
                         "consecutive_show_failures": 0, "box_unreachable": False})
            self.p.status.set_box(iid, info)
        total = sum(self.ledger.values())
        self.last_cost = total
        self.p.status.set_cost(total)
        if self.p.max_usd is not None and total > self.p.max_usd:
            self.p.abort_reason = (f"budget exceeded: measured ${total:.4f} > "
                                   f"--max-usd ${self.p.max_usd:.4f}")
            print(f"[{now()}] {self.p.abort_reason} - cutting the running stage",
                  file=sys.stderr)
            self.p.kill_current()

    def run(self) -> None:
        while not self._stop.is_set():
            self.beat()
            try:
                self.poll_cost()
            except Exception as e:      # the supervisor must never take the run down
                print(f"[{now()}] supervisor tick error: {e}", file=sys.stderr)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.beat()


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.scene_dir = Path(a.scene_dir).resolve()
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.logs = self.scene_dir / "logs"
        self.logs.mkdir(exist_ok=True)
        self.scene = a.scene_name
        self.short = self.scene.replace("__optimal_step2", "").replace("__step2", "")
        self.nvblox_root = Path(a.nvblox_root).resolve()
        self.pulled = Path(a.from_pulled).resolve() if a.from_pulled else None
        # Where pass-A output must be. `--from-pulled` names an existing run; without it
        # the worker looks in the scene's own dir, and MAPANYTHING waits there by name
        # rather than failing with "no producer" - see wait_for_manual_mapanything.
        self.pass_a = self.pulled if self.pulled else (self.scene_dir / "pass_a")
        self.packed_root = self.scene_dir / "packed"
        self.packed = self.packed_root / self.scene

        # MVFILTER. `mvfilter` is the parsed spec or None; `depth_file` is the single place
        # that decides which stack NVBLOX pushes and fuses, so the flag cannot half-apply -
        # a run that filtered but fused the unfiltered stack would look like a null result.
        # Colour. On by default: a grey scene is a defect, not a preference, and the cost is
        # one ffmpeg pass plus the push. `--no-color` exists for the case the colour path
        # itself is suspect - it reproduces exactly what every run before today produced.
        self.colour = bool(getattr(a, "color", True))
        self.rgb_stack = self.packed / RGB_STACK

        # Leave the box running at the end instead of stopping it. A refused START is not
        # recoverable in the same session - it retired 50265868 and stalled the 2026-09-09
        # mvfilter run four times - so across a queue of runs, holding one box trades a
        # known hourly cost against an unbounded "no capacity anywhere" risk. The operator
        # owns that trade and pays for it, which is why it is off unless asked for.
        self.keep_box_up = bool(getattr(a, "keep_box_up", False))

        self.mvfilter = getattr(a, "mvfilter_spec", None)
        self.depth_file = MVFILTER_DEPTH if self.mvfilter else "depth_u16.npy"
        # The A/B: fuse the unfiltered stack too, in the same box session, so obstacle cells
        # and compute seconds compare against something built by the same script on the same
        # box. Its output is a measurement, never the run's own scene.
        self.mvfilter_ab = bool(getattr(a, "mvfilter_ab", False)) and bool(self.mvfilter)
        self.ab_scene_dir = (self.scene_dir / "nvblox_unfiltered" / self.scene).resolve()
        # Extra filtered arms, each fused in the same session as its own control. The
        # neighbour rule changes what this filter does far more than tau does - `nearest`
        # keeps 74.0% of hero-15fps's pixels where `baseline15` keeps 57.0% - so comparing
        # one filtered arm against unfiltered cannot say whether a null result means "the
        # filter does not help" or "this arm barely filtered".
        self.mvfilter_also = [parse_mvfilter(s) for s in (getattr(a, "mvfilter_also", None)
                                                          or [])] if self.mvfilter else []
        self.alt_arms = [{"spec": s, "slug": mvfilter_slug(s),
                          "depth_file": f"depth_u16_mv_{mvfilter_slug(s)}.npy",
                          "meta_file": f"mvfilter_{mvfilter_slug(s)}.json",
                          "dest": (self.scene_dir / f"nvblox_{mvfilter_slug(s)}"
                                   / self.scene).resolve()}
                         for s in self.mvfilter_also]

        # Where FLOOR/LAYERS/REACH/EXPORT read the nvblox output from. When this run
        # computes nvblox itself (--nvblox-source live) that is our own directory, NOT the
        # published one - a live run must never silently read yesterday's answer.
        self.nvblox_live = (a.nvblox_source == "live")
        if self.nvblox_live:
            self.scenes_out = (self.scene_dir / "nvblox").resolve()
        else:
            self.scenes_out = Path(a.scenes_out_root).resolve()
        self.src_scene = self.scenes_out / self.scene

        self.dry_run = a.dry_run
        self.max_usd = a.max_usd
        self.abort_reason: str | None = None
        self.sessions: dict[int, BoxSession] = {}      # currently up
        self.boxes_seen: dict[int, BoxSession] = {}   # ever brought up, for the cost ledger
        self._proc = None
        self._proc_lock = threading.Lock()

        self.status = Status(self.scene_dir / "status.json", self.scene, str(a.video))
        self.status.doc["pipeline_dry_run"] = self.dry_run
        self.status.doc["max_usd"] = self.max_usd
        self.status.doc["nvblox_source"] = a.nvblox_source
        self.status.flush()
        self.vast = VastClient(self.scene_dir / "vast_calls.log", Path(a.armed_file))
        self.hb_path = self.scene_dir / "heartbeat"
        self.sup = Supervisor(self, interval_s=getattr(a, "heartbeat_s", HEARTBEAT_S))
        (self.scene_dir / "pipeline.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    # -- subprocess handle the supervisor may cut ----------------------------------------
    def register_proc(self, proc) -> None:
        with self._proc_lock:
            self._proc = proc

    def kill_current(self) -> None:
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()

    # -- generic step driver ------------------------------------------------------------
    def step(self, state: str, check, produce=None, kind: str = "real",
             skip_check_after: bool = False) -> None:
        """check() -> (ok, artifacts). Runs first; on ok the step is skipped."""
        # A tick can land badly - between two states, or while a box is only briefly up.
        # Checking at every boundary as well makes the budget deterministic rather than
        # dependent on when the 30 s tick happened to fire.
        if self.boxes_seen:
            try:
                self.sup.poll_cost()
            except Exception as e:
                print(f"[{now()}] boundary cost poll failed (not a verdict): {e}",
                      file=sys.stderr)
        if self.abort_reason:
            raise StepFailed(state, self.abort_reason)
        self.status.enter(state)
        self.sup.beat()
        try:
            ok, art = check()
            if ok:
                outcome = "skipped_precomputed" if kind == "precomputed" else (
                    "stubbed" if kind == "stub" else "skipped")
                sub, det = art.pop("_substage", None), art.pop("_detail", None)
                self.status.finish_step(state, outcome, art, substage=sub, detail=det)
                print(f"[{now()}] {state}: {outcome} ({art.get('note','')})")
                return
            if produce is None:
                raise StepFailed(state, f"{state}: output missing and no producer: {art}")
            t0 = time.time()
            art_run = produce()
            if self.abort_reason:
                raise StepFailed(state, self.abort_reason)
            if skip_check_after and self.dry_run:
                # A remote step in dry run logs the exact argv it would run and produces no
                # local output, so re-checking would fail for the wrong reason.
                art = dict(art_run or {})
                art["note"] = "dry run: remote commands logged, not executed"
                self.status.finish_step(state, "stubbed", art)
                print(f"[{now()}] {state}: stubbed (dry run, remote)")
                return
            ok, art = check()
            if not ok:
                raise StepFailed(state, f"{state}: producer ran but check failed: {art}")
            art.update(art_run or {})
            wall = time.time() - t0
            sub, det = art.pop("_substage", None), art.pop("_detail", None)
            self.status.finish_step(state, "stubbed" if kind == "stub" else "done", art,
                                    substage=sub, detail=det)
            print(f"[{now()}] {state}: done in {wall:.2f}s")
        except StepFailed:
            raise
        except Exception as e:  # any check/producer bug is a real failure, not a silent pass
            raise StepFailed(state, f"{state}: {type(e).__name__}: {e}")

    # -- individual checks --------------------------------------------------------------
    def check_frames(self):
        fj = self.pulled / "frames.json" if self.pulled else self.scene_dir / "frames.json"
        if not fj.exists():
            return False, {"missing": str(fj)}
        d = load_json(fj)
        need_keys(d, ["n_frames", "frames", "sampling"], "FRAMES")
        n = d["n_frames"]
        ok = n == len(d["frames"])
        return ok, {"source": str(fj), "n_frames": n, "len_frames_list": len(d["frames"]),
                    "sample_fps": d["sampling"].get("sample_fps"),
                    "step": d["sampling"].get("step"),
                    "note": f"{n} frames @ step {d['sampling'].get('step')}"}

    def check_mapanything(self):
        pj = self.pass_a / "poses.json"
        pv = self.pass_a / "per_view"
        if not pj.exists() or not pv.is_dir():
            return False, {"missing": str(pj if not pj.exists() else pv)}
        d = load_json(pj)
        need_keys(d, ["model", "n_views", "views"], "MAPANYTHING")
        n_npz = len(list(pv.glob("*.npz")))
        ok = d["n_views"] == len(d["views"]) == n_npz
        return ok, {"source": str(self.pass_a), "model": d["model"], "n_views": d["n_views"],
                    "n_per_view_npz": n_npz, "peak_vram_mib": d.get("peak_vram_mib"),
                    "infer_wall_clock_sec": d.get("wall_clock_sec"),
                    "note": f"{n_npz} npz == n_views {d['n_views']}"}

    def produce_frames(self):
        """Sample the video at the job spec's fps. Rotation is recorded, not assumed -
        frames_by_fps.probe_rotation reads the container's tag and states what the extracted
        frames actually are, per PIPELINE.md 4.4e. An unapplied rotation moves the cameras'
        "up" by a quarter turn and shows up much later as a floor fit that finds a wall."""
        fps = self.a.frames_fps
        if fps is None:
            raise StepFailed("FRAMES", "no frames.json and no --frames-fps: nothing says "
                                       "how densely to sample this video")
        out = self.scene_dir / "frames.json"
        # frames_by_fps.py is stdlib-only (it shells out to ffprobe), so this runs under the
        # driver's own interpreter rather than depending on the o3d venv existing.
        return run_cmd([sys.executable, str(HERE / "steps" / "frames_by_fps.py"),
                        "--video", str(self.a.video), "--fps", str(fps), "--out", str(out)],
                       self.logs, "FRAMES", self.a.step_timeout_s,
                       register=self.register_proc) | {"frames_json": str(out),
                                                       "requested_fps": fps}

    def wait_for_manual_mapanything(self):
        """MapAnything pass A is not automated in this build, and this says so by name.

        It needs a B200-class box driven by hand (PIPELINE.md 4). Rather than fail with
        "output missing and no producer", the step parks in `waiting_manual` and names the
        exact two paths. The job is resumable as it stands: every step's check() runs before
        its producer, so re-running this driver over the same scene dir picks the output up
        and carries on from PULLED.
        """
        pj, pv = self.pass_a / "poses.json", self.pass_a / "per_view"
        detail = (f"MapAnything pass A is manual: run it on a B200 box, then place "
                  f"poses.json at {pj} and the per-view .npz files under {pv}/ and re-run "
                  f"this driver over the same --scene-dir; it will resume from here")
        self.status.set_substage("MAPANYTHING", "waiting_manual", detail)
        raise StepFailed("MAPANYTHING", detail, fail_reason="waiting_manual")

    def check_pulled(self):
        ok, art = self.check_mapanything()
        if not ok:
            return ok, art
        art["bytes"] = dir_size(self.pass_a)
        art["note"] = f"{art['n_per_view_npz']} npz, {art['bytes']/1e9:.2f} GB"
        return True, art

    def check_pack(self):
        mj, dj = self.packed / "meta.json", self.packed / "depth_u16.npy"
        if not mj.exists() or not dj.exists():
            return False, {"missing": str(mj if not mj.exists() else dj)}
        m = load_json(mj)
        need_keys(m, ["source", "n_views", "shape_hw", "depth_units", "views"], "PACK")
        pr = probe_npy(self.a.python_cpu, {"depth": str(dj)})["depth"]
        h, w = m["shape_hw"]
        ok = (pr["dtype"] == "uint16" and pr["shape"] == [m["n_views"], h, w]
              and m["n_views"] == len(m["views"]))
        return ok, {"depth_u16": str(dj), "bytes": dj.stat().st_size, "dtype": pr["dtype"],
                    "shape": pr["shape"], "n_views": m["n_views"],
                    "note": f"depth {pr['shape']} {pr['dtype']}, {dj.stat().st_size} B"}

    def check_mvfilter(self):
        """Off -> a `skipped` step that says so. On -> the filtered stack must exist AND
        have been produced by the same vote this run asks for.

        The parameter comparison is the load-bearing half. The filtered stack has the same
        name, shape and dtype whatever (m, tau, neighbours, K) produced it, so a check that
        only looked at the file would silently reuse yesterday's m=4 stack for today's m=2
        run and report it as `skipped`.
        """
        if not self.mvfilter:
            return True, {"_substage": "skipped", "_detail": "--mvfilter not given",
                          "note": "MVFILTER is off; NVBLOX fuses depth_u16.npy unchanged"}
        meta = load_json(self.packed / "meta.json")
        h, w = meta["shape_hw"]
        arts, all_ok = {}, True
        for spec, depth_name, meta_name in self.mvfilter_arms():
            dj, mj = self.packed / depth_name, self.packed / meta_name
            key = f"{spec['neighbours']}_m{spec['m']}"
            if not dj.exists() or not mj.exists():
                arts[key] = {"missing": str(dj if not dj.exists() else mj),
                             "requested": spec}
                all_ok = False
                continue
            got = load_json(mj)
            need_keys(got, ["m", "tau", "neighbours", "k", "kept_pct", "valid_px_before",
                            "valid_px_after"], "MVFILTER")
            same = (got["m"] == spec["m"] and float(got["tau"]) == float(spec["tau"])
                    and got["neighbours"] == spec["neighbours"] and got["k"] == spec["K"])
            if not same:
                arts[key] = {"requested": spec,
                             "on_disk": {k: got[k] for k in ("m", "tau", "neighbours", "k")},
                             "note": "stack on disk was produced by a different vote"}
                all_ok = False
                continue
            pr = probe_npy(self.a.python_cpu, {"depth": str(dj)})["depth"]
            ok = pr["dtype"] == "uint16" and pr["shape"] == [meta["n_views"], h, w]
            all_ok = all_ok and ok
            arts[key] = {"depth": str(dj), "bytes": dj.stat().st_size,
                         "dtype": pr["dtype"], "shape": pr["shape"],
                         "m": got["m"], "tau": got["tau"], "neighbours": got["neighbours"],
                         "k": got["k"], "kept_pct": got["kept_pct"],
                         "valid_px_before": got["valid_px_before"],
                         "valid_px_after": got["valid_px_after"],
                         "k_eff": got.get("k_eff"), "wall_s": got.get("wall_s"),
                         "views_fully_dropped":
                             (got.get("per_view_kept_frac") or {}).get("views_fully_dropped")}
        note = ", ".join(f"{k} kept {v.get('kept_pct')}%" for k, v in arts.items()
                         if "kept_pct" in v)
        return all_ok, {"arms": arts, "note": note or "filtered stack(s) not present yet"}

    def mvfilter_arms(self):
        """(spec, depth filename, meta filename) for every stack MVFILTER must produce.

        The primary arm keeps the plain name so the common single-filter case reads simply;
        extra arms are named by their own parameters (mvfilter_slug).
        """
        out = [(self.mvfilter, MVFILTER_DEPTH, MVFILTER_META)] if self.mvfilter else []
        out += [(a["spec"], a["depth_file"], a["meta_file"]) for a in self.alt_arms]
        return out

    def produce_mvfilter(self):
        art, last = {}, None
        for i, (f, depth_name, meta_name) in enumerate(self.mvfilter_arms()):
            last = run_cmd([self.a.python_cpu, str(HERE / "steps" / "mvfilter.py"),
                            "--packed", str(self.packed), "--m", str(f["m"]),
                            "--tau", str(f["tau"]), "--neighbours", f["neighbours"],
                            "--k", str(f["K"]), "--out-depth", depth_name,
                            "--out-meta", meta_name],
                           self.logs, "MVFILTER" if i == 0 else f"MVFILTER_{i}",
                           self.a.step_timeout_s, cwd=self.nvblox_root,
                           register=self.register_proc)
            art[f"argv_{f['neighbours']}_m{f['m']}"] = last["argv"]
            art[f"wall_s_{f['neighbours']}_m{f['m']}"] = last["wall_s"]
        return art

    def esdf_control(self) -> dict:
        """Compare this run's own ESDF slice against the published one for the same scene.

        Reported, never gating: a difference here is a finding about determinism, not a
        reason to fail a run. Skipped silently when there is no published reference.
        """
        ref = Path(self.a.scenes_out_root) / self.scene / "esdf_slice_0.3m.npy"
        ours = self.src_scene / "esdf_slice_0.3m.npy"
        if not (ref.exists() and ours.exists()) or ref.resolve() == ours.resolve():
            return {"note": "no independent reference to compare against"}
        try:
            pr = subprocess.run([self.a.python_cpu, "-c", ESDF_DIFF_PROBE, str(ours),
                                 str(ref)], capture_output=True, text=True, timeout=300)
            out = json.loads(pr.stdout) if pr.returncode == 0 else {"error": pr.stderr[:200]}
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        out["reference"] = str(ref)
        return out

    def check_nvblox(self):
        need = ["mesh.ply", "esdf_slice_0.3m.npy", "unobserved_slice.npy", "stats.json"]
        missing = [f for f in need if not (self.src_scene / f).exists()]
        if missing:
            return False, {"missing": missing, "dir": str(self.src_scene)}
        st = load_json(self.src_scene / "stats.json")
        need_keys(st, ["mesh", "slice", "n_views"], "NVBLOX")
        control = self.esdf_control() if self.nvblox_live else {}
        return True, {"source": str(self.src_scene), "n_views": st["n_views"],
                      "esdf_control_vs_published": control,
                      "mesh_triangles": st["mesh"]["triangles"],
                      "mesh_vertices": st["mesh"]["vertices"],
                      "slice_shape": st["slice"]["shape"],
                      "integrate_wall_s": st.get("integrate_wall_s"),
                      "esdf_wall_s": st.get("esdf_wall_s"), "mesh_wall_s": st.get("mesh_wall_s"),
                      "note": f"precomputed on GPU box; {st['mesh']['triangles']} tri"}

    def check_floor(self):
        fp = self.src_scene / "floor_plane.json"
        if not fp.exists():
            return False, {"missing": str(fp)}
        d = load_json(fp)
        need_keys(d, ["normal_up", "point_on_plane", "slice_origin_world", "axis_u", "axis_v",
                      "u_range", "v_range", "slice_resolution_m"], "FLOOR")
        return True, {"source": str(fp), "normal_up": d["normal_up"],
                      "inlier_frac": d.get("inlier_frac"),
                      "slice_resolution_m": d["slice_resolution_m"],
                      "u_range": d["u_range"], "v_range": d["v_range"],
                      "note": "RANSAC floor is produced inside nvblox_scenes.py, same GPU pass"}

    def check_layers(self):
        """The band obstacle mask, not the ESDF slice.

        The slice is still written next to it for reference, but it is no longer what this
        step is FOR: capped at 0.30 m by construction (the floor is 0.30 m below every
        query point), it reports zero free cells for any robot wider than that - measured
        0.3421 m max on both hero and own_0901_161054, against a Husky radius of 0.5528.
        """
        out = self.scene_dir / "layers" / self.short
        need = ["obstacle_mask.npy", "occupancy.npy", "unobserved_mask.npy",
                "occupancy_meta.json", "grid_meta.json", "floor_plane.json"]
        missing = [f for f in need if not (out / f).exists()]
        if missing:
            return False, {"missing": missing, "dir": str(out)}
        pr = probe_npy(self.a.python_cpu,
                       {"obst": str(out / "obstacle_mask.npy"),
                        "occ": str(out / "occupancy.npy"),
                        "mask": str(out / "unobserved_mask.npy")})
        om = load_json(out / "occupancy_meta.json")
        ok = (pr["obst"]["shape"] == pr["mask"]["shape"] == pr["occ"]["shape"]
              and pr["obst"]["dtype"] == "bool" and pr["mask"]["dtype"] == "bool"
              and pr["occ"]["dtype"] == "uint8"
              and [om["width"], om["height"]] == pr["occ"]["shape"]
              # An unobserved cell that reads FREE is the one thing this layer must never
              # ship; band_mask applies the mask last so it cannot happen, and this is the
              # post-condition that says so.
              and om["n_free"] + om["n_obstacle"] + om["n_unknown"] == pr["occ"]["shape"][0]
                                                                      * pr["occ"]["shape"][1])
        return ok, {"dir": str(out), "grid": pr["occ"]["shape"],
                    "band_m": [om.get("band_min"), om.get("band_max")],
                    "obstacle_cells": om.get("obstacle_cells"),
                    "free": om.get("n_free"), "obstacle": om.get("n_obstacle"),
                    "unknown": om.get("n_unknown"),
                    "unobserved_frac": om.get("unobserved_frac"),
                    "note": f"band {om.get('band_min')}-{om.get('band_max')} m, "
                            f"{om.get('obstacle_cells')} obstacle cells, "
                            f"free/obst/unk {om.get('n_free')}/{om.get('n_obstacle')}/"
                            f"{om.get('n_unknown')}"}

    def check_reach(self):
        out = self.scene_dir / "reach"
        rj = out / "reachability.json"
        if not rj.exists():
            return False, {"missing": str(rj)}
        d = load_json(rj)
        need_keys(d, ["slice_shape", "obstacle_cells", "nav2d_clearance_max_m"], "REACH")
        return True, {"dir": str(out), "slice_shape": d["slice_shape"],
                      "obstacle_cells": d["obstacle_cells"],
                      "nav2d_clearance_max_m": d["nav2d_clearance_max_m"],
                      "observed_frac": d.get("observed_frac"),
                      "note": f"obstacle_cells {d['obstacle_cells']}, "
                              f"max clearance {d['nav2d_clearance_max_m']:.4f} m"}

    def check_export(self):
        out = self.scene_dir / "export"
        summ = out / "export_summary.json"
        usd = out / self.short / "scene.usd"
        if not summ.exists() or not usd.exists():
            return False, {"missing": str(summ if not summ.exists() else usd)}
        rows = load_json(summ)
        row = next((r for r in rows if r.get("scene") == self.scene), None)
        if row is None:
            return False, {"missing_row_for": self.scene}
        ok = (usd.stat().st_size > 0 and row.get("triangles_out", 0) > 0
              and row.get("usd_validation", {}).get("n_errors", 1) == 0)
        return ok, {"usd": str(usd), "usd_bytes": usd.stat().st_size,
                    "triangles_in": row.get("triangles_in"),
                    "triangles_out": row.get("triangles_out"),
                    "source_mesh": row.get("source_mesh"),
                    "usd_validation_errors": row.get("usd_validation", {}).get("n_errors"),
                    "wall_s": row.get("wall_s"),
                    "note": f"{row.get('triangles_out')} tri, {usd.stat().st_size} B USD"}

    def check_ingest(self):
        """The artifacts the app reads must exist AND describe this scene.

        The existence half alone would pass on a stale ingest from a previous run over a
        different scene - same filenames, same shapes - so the camera count is checked
        against this run's own pack. A camera track from another room lands off-grid and
        every /reachability call on the scene answers 500.
        """
        need = ["scene_objects/scene_objects.json", "cameras_aligned.json",
                "alignment.json", "vocab.json"]
        missing = [n for n in need if not (self.scene_dir / n).exists()]
        if missing:
            return False, {"missing": missing}
        al = load_json(self.scene_dir / "alignment.json")
        need_keys(al, ["floor_y", "ceiling_y", "bbox_min", "bbox_max",
                       "residual_tilt_deg", "det_r", "is_reflection"], "INGEST")
        cams = load_json(self.scene_dir / "cameras_aligned.json").get("cameras", [])
        n_views = load_json(self.packed / "meta.json")["n_views"]
        ok = len(cams) == n_views
        return ok, {"n_cameras": len(cams), "n_views": n_views,
                    "ceiling_y": al["ceiling_y"],
                    "scene_points_glb": (self.scene_dir / "scene_points.glb").exists(),
                    "note": f"{len(cams)} cameras, ceiling {al['ceiling_y']:.2f} m"
                            + ("" if ok else f" - MISMATCH, pack has {n_views} views")}

    # -- producers ----------------------------------------------------------------------
    def check_pack_rgb(self):
        """Off -> a `skipped` step that says so. On -> the stack must exist AND describe the
        same views as the depth stack it will be integrated alongside.

        The shape check is the load-bearing half. An rgb_u8.npy left over from a different
        scene has the same name and the same dtype, and pairing it with this scene's poses
        would paint one room's colour onto another room's geometry - a result that looks
        entirely plausible until someone recognises the wallpaper.
        """
        if not self.colour:
            return True, {"_substage": "skipped", "_detail": "--no-color",
                          "note": "colour off; nvblox will write nvblox's unset 127/127/127"}
        if not self.rgb_stack.exists():
            return False, {"missing": str(self.rgb_stack)}
        meta = load_json(self.packed / "meta.json")
        h, w = meta["shape_hw"]
        pr = probe_npy(self.a.python_cpu, {"rgb": str(self.rgb_stack)})["rgb"]
        want = [meta["n_views"], h, w, 3]
        ok = pr["dtype"] == "uint8" and pr["shape"] == want
        return ok, {"rgb_u8": str(self.rgb_stack), "dtype": pr["dtype"],
                    "shape": pr["shape"], "expected_shape": want,
                    "bytes": self.rgb_stack.stat().st_size,
                    "note": f"rgb {pr['shape']} {pr['dtype']}"}

    def produce_pack_rgb(self):
        return run_cmd([self.a.python_cpu, str(HERE / "steps" / "pack_rgb.py"),
                        "--video", str(self.a.video), "--packed", str(self.packed),
                        "--out", RGB_STACK],
                       self.logs, "PACK_RGB", self.a.step_timeout_s,
                       register=self.register_proc)

    def produce_pack(self):
        self.packed.parent.mkdir(parents=True, exist_ok=True)
        return run_cmd([self.a.python_cpu, str(self.nvblox_root / "pack_scenes.py"),
                        "--src", str(self.pass_a), "--out", str(self.packed)],
                       self.logs, "PACK", self.a.step_timeout_s, cwd=self.nvblox_root,
                       register=self.register_proc)

    def produce_layers(self):
        """frontend_layers.py for the pack and its frame metadata, then band_mask.py for
        the layer that actually decides where a robot can stand."""
        out = self.scene_dir / "layers"
        out.mkdir(parents=True, exist_ok=True)
        art = run_cmd([self.a.python_cpu, str(self.nvblox_root / "frontend_layers.py"),
                       "--scenes-root", str(self.scenes_out), "--out-root", str(out),
                       "--frontend-map", str(self.nvblox_root / "frontend_map.json"),
                       "--packed-root", str(self.packed_root), "--scenes", self.scene],
                      self.logs, "LAYERS", self.a.step_timeout_s, cwd=self.nvblox_root,
                      register=self.register_proc)
        band = run_cmd([self.a.python_cpu, str(HERE / "steps" / "layers" / "band_mask.py"),
                        "--scene-dir", str(self.src_scene),
                        "--out", str(out / self.short),
                        "--band", str(self.a.band_min), str(self.a.band_max),
                        "--min-component-tri", str(self.a.band_min_component_tri),
                        "--mesh-name", self.a.mesh_name],
                       self.logs, "LAYERS_BAND", self.a.step_timeout_s,
                       cwd=self.nvblox_root, register=self.register_proc)
        return art | {"band_argv": band["argv"], "band_wall_s": band["wall_s"],
                      "band_m": [self.a.band_min, self.a.band_max]}

    def produce_reach(self):
        """Reads the LAYERS output, not the nvblox scene dir.

        nvblox_v2/reachability.py rasterised its own obstacle band from mesh.ply every run,
        so the band rule lived in a script the front end never runs while LAYERS shipped a
        different layer to everyone else - two answers to "where can the robot stand" for
        one scene. steps/layers/reach.py loads obstacle_mask.npy instead, so the number
        reported here is the number the scene page shows.
        """
        out = self.scene_dir / "reach"
        out.mkdir(parents=True, exist_ok=True)
        layers = self.scene_dir / "layers" / self.short
        return run_cmd([self.a.python_cpu, str(HERE / "steps" / "layers" / "reach.py"),
                        "--layers-dir", str(layers), "--out", str(out),
                        "--radii", *[str(r) for r in self.a.reach_radii],
                        "--heights", *[str(h) for h in self.a.reach_heights],
                        "--height-margin", str(self.a.band_margin)],
                       self.logs, "REACH", self.a.step_timeout_s, cwd=self.nvblox_root,
                       register=self.register_proc) | {"layers_dir": str(layers)}

    def produce_export(self):
        # Guard the input BEFORE calling isaac_export.py, because that script does not fail
        # on a missing mesh: Open3D's PLY reader prints `RPly: Unable to open file` to stderr
        # and returns an EMPTY mesh, so the export runs on 0 vertices and only dies later
        # inside write_usd with `zero-size array to reduction operation minimum`. That
        # traceback names numpy, not the missing file - it cost real diagnosis time on the
        # 2026-09-09 live run. Fail here instead, naming the file.
        mesh = self.src_scene / self.a.mesh_name
        if not mesh.exists():
            present = sorted(f.name for f in self.src_scene.glob("*.ply")) \
                if self.src_scene.is_dir() else []
            raise StepFailed("EXPORT",
                             f"input mesh {mesh} does not exist "
                             f"(--mesh-name {self.a.mesh_name}); .ply files present in "
                             f"{self.src_scene}: {present or 'none'}")
        size = mesh.stat().st_size
        if size == 0:
            raise StepFailed("EXPORT", f"input mesh {mesh} is 0 bytes")
        out = self.scene_dir / "export"
        out.mkdir(parents=True, exist_ok=True)
        extra = (["--max-tri", str(self.a.export_max_tri)]
                 if self.a.export_max_tri is not None else [])
        # steps/isaac_export.py, not nvblox_root's: the USD writer is now under version
        # control here, because the `has_color` bug that shipped seven grey USDs lived in a
        # file no branch tracked. cwd stays nvblox_root - the script reads nothing from it,
        # but its sibling scripts do, and one difference at a time.
        return run_cmd([self.a.python_cpu, str(HERE / "steps" / "isaac_export.py"),
                        "--scenes-root", str(self.scenes_out), "--out-root", str(out),
                        "--scenes", self.scene, "--mesh-name", self.a.mesh_name, *extra],
                       self.logs, "EXPORT", self.a.step_timeout_s, cwd=self.nvblox_root,
                       register=self.register_proc) | {"input_mesh": str(mesh),
                                                       "input_mesh_bytes": size}

    def produce_ingest(self):
        """Write the artifacts `ingest_scene_directory` reads, from what this run measured.

        This was a stub - `ingest_stub.json`, `"implemented": false` - and the consequence was
        that a run could reach DONE with a complete scene on disk that the app then refused:
        `ArtifactParseError: required artifact missing: .../scene_objects/scene_objects.json`,
        with the scene row left `failed`. The driver produced everything a robot needs and
        nothing a reader could open.

        The layer is deliberately NOT copied to `layers/backfill/`. LAYERS already wrote
        `layers/<short>/`, and `nav_layer.layer_dir` prefers exactly that over `backfill`
        because it came from this scene's own mesh; a second copy under the backfill name
        would be the de-prioritised one and would drift the moment either is regenerated.
        """
        return run_cmd([self.a.python_cpu, str(HERE / "steps" / "ingest_scene.py"),
                        "--scene-dir", str(self.scene_dir),
                        "--nvblox-dir", str(self.src_scene),
                        "--layers-dir", str(self.scene_dir / "layers" / self.short),
                        "--packed", str(self.packed),
                        "--export-dir", str(self.scene_dir / "export" / self.short),
                        "--mesh-name", self.a.mesh_name],
                       self.logs, "INGEST", self.a.step_timeout_s,
                       register=self.register_proc)

    # -- gpu lifecycle: one session per box, each stopped in its own finally ------------
    def acquire_box(self, state: str, instance_id: int, name: str):
        """Worker-mode bring-up: start the cached box, rent a replacement if its host
        refuses, and give up rather than loop forever when there is no capacity at all.

        The operator path (`bring_up`) is a single start by id and is unchanged - an
        operator is standing there and can read the refusal. A worker is not, so the wait is
        bounded twice over: WAITING_CAPACITY_MAX_ATTEMPTS attempts or WAITING_CAPACITY_MAX_S
        wall seconds, whichever comes first. Every attempt is visible in status.json as
        substage=waiting_capacity with `attempt` and `next_retry`.
        """
        t_first = time.time()
        onstart = str(HERE / "box" / "onstart_nvblox.sh")
        max_attempts = self.a.capacity_max_attempts
        backoff = self.a.capacity_backoff_s
        for attempt in range(1, max_attempts + 1):
            self.status.set_substage(
                state, "searching",
                f"attempt {attempt}/{max_attempts}: starting instance "
                f"{instance_id}, or renting a replacement if its host refuses",
                attempt=attempt)
            try:
                res = self.vast.start_or_create(
                    dry_run=self.dry_run, label=f"pipeline-{self.short}"[:60],
                    onstart_file=onstart, instance_id=instance_id,
                    state_dir=self.a.state_dir)
            except VastStartUnavailable as e:
                left_attempts = max_attempts - attempt
                left_s = WAITING_CAPACITY_MAX_S - (time.time() - t_first)
                if left_attempts <= 0 or left_s <= backoff:
                    reason = (f"no capacity after {attempt} attempt(s) over "
                              f"{time.time() - t_first:.0f}s "
                              f"(ceiling {max_attempts} attempts / "
                              f"{WAITING_CAPACITY_MAX_S}s); vast's last words: {e}")
                    self.status.set_substage(state, "waiting_capacity", reason,
                                             attempt=attempt)
                    raise StepFailed(state, reason, fail_reason="waiting_capacity_exhausted")
                nxt = dt.datetime.now().astimezone() + dt.timedelta(seconds=backoff)
                self.status.set_substage(
                    state, "waiting_capacity",
                    f"no box available on attempt {attempt}: {e}",
                    attempt=attempt,
                    next_retry=nxt.isoformat(timespec="seconds"))
                print(f"[{now()}] {state}: no capacity (attempt {attempt}), retrying in "
                      f"{backoff}s: {e}", file=sys.stderr)
                time.sleep(backoff)
                continue

            iid = int(res["instance_id"])
            if res.get("created"):
                # A fresh box bills from the create call, not from when ssh answers: the
                # provisioning minutes in between are minutes this run caused.
                self.status.set_substage(
                    state, "creating",
                    f"rented {iid} from offer {(res.get('offer') or {}).get('id')}; "
                    f"provisioning runs {Path(onstart).name} on the box",
                    attempt=attempt)
                self.a.billing_from_epoch = self.a.billing_from_epoch or time.time()
                self.a.nvblox_instance_id = iid
            art = self.bring_up(state, iid, name, skip_start=True, attempt=attempt)
            if res.get("created"):
                # substage is what the step is doing NOW, and by here it is doing nothing -
                # so the fact that this box was rented mid-run lives in the artifacts, where
                # it is still there tomorrow.
                art["replaced_instance_id"] = instance_id
                art["created_from_offer"] = (res.get("offer") or {}).get("id")
                art["attempts_to_acquire"] = attempt
            return art
        raise StepFailed(state, "waiting_capacity loop fell through",
                         fail_reason="waiting_capacity_exhausted")

    def bring_up(self, state: str, instance_id: int, name: str,
                 skip_start: bool = False, attempt: int = 1):
        t0 = time.time()
        # A refusal has to surface in seconds with vast's own words, not as a 300 s ssh
        # timeout. BoxSession still issues the stop, cancelling the queued start.
        sess = BoxSession(self.vast, instance_id, dry_run=self.dry_run,
                          ssh_key=self.a.ssh_key, name=name,
                          deadline_s=self.a.ssh_wait_s, poll_s=self.a.ssh_poll_s,
                          probe_mb=self.a.probe_mb, probe_min_s=self.a.probe_min_s,
                          scratch=self.scene_dir / "probe",
                          log=print, skip_start=skip_start)
        if self.a.billing_from_epoch:
            sess.billing_from = float(self.a.billing_from_epoch)
        self.status.set_substage(
            state, "provisioning",
            f"starting instance {instance_id} and waiting for ssh "
            f"(up to {self.a.ssh_wait_s}s), then chmod, link probes and nvidia-smi",
            attempt=attempt)
        try:
            sess.__enter__()
        except VastStartUnavailable as e:
            self.status.set_substage(
                state, "refused",
                f"vast declined to start {instance_id} in its own words: {e.stdout.strip()}")
            raise StepFailed(state, f"vast refused to start instance {instance_id}: "
                                    f"{e.stdout}",
                             fail_reason=f"vast start refused: {e.stdout}")
        self.status.set_substage(state, None, f"instance {instance_id} is up and answering")
        sess.info["bring_up_total_s"] = round(time.time() - t0, 2)
        self.sessions[instance_id] = sess
        self.boxes_seen[instance_id] = sess
        self.status.set_box(instance_id, sess.info)
        return dict(sess.info)

    def take_down(self, instance_id: int):
        sess = self.sessions.pop(instance_id, None)
        if sess is None:
            return {"instance_id": instance_id, "note": "no live session to stop"}
        if self.keep_box_up:
            # Deliberately still billing. Said in the artifacts rather than only in a flag,
            # because "why is this box costing money" must be answerable from status.json
            # alone, by someone who did not start the run.
            sess.info["kept_up"] = True
            sess.info["kept_up_note"] = (
                "--keep-box-up: NOT stopped, still billing at "
                f"${sess.info.get('dph_total')}/h. Stop it by hand when the queue is done: "
                f"vastai stop instance {instance_id}")
            self.status.set_box(instance_id, sess.info)
            self.boxes_seen[instance_id] = sess
            return dict(sess.info)
        sess.stop(confirm_deadline_s=self.a.exit_confirm_s)
        self.status.set_box(instance_id, sess.info)
        return dict(sess.info)

    def check_box_up(self, instance_id: int):
        """A box is never 'already up' from a previous process: starting a running instance
        is harmless, and assuming it is up without proving ssh is how a run gets stuck."""
        sess = self.sessions.get(instance_id)
        if sess is None:
            return False, {"instance_id": instance_id, "note": "no live session yet"}
        return True, dict(sess.info)

    def check_box_down(self, instance_id: int):
        if instance_id in self.sessions:
            return False, {"instance_id": instance_id, "note": "session still open"}
        return True, {"instance_id": instance_id, "note": "no live session"}

    # -- nvblox on the box ---------------------------------------------------------------
    def produce_nvblox(self):
        """Push the depth stack, run nvblox_scenes.py in its own venv, pull the results.

        The remote script and venv are the ones HANDOFF_nvblox.md records as surviving a
        stop: /workspace/nvblox_v2/ + /workspace/venvs/nb. Nothing about them is modified
        from here.
        """
        sess = self.sessions.get(self.a.nvblox_instance_id)
        if sess is None and not self.dry_run:
            raise StepFailed("NVBLOX", "nvblox box session is not open")
        job = f"{self.a.remote_jobs}/{self.scene}"
        out = f"{self.a.remote_out}/{self.scene}"
        script_local = HERE / "steps" / "nvblox_scenes.py"
        script_remote = f"{self.a.remote_scripts}/nvblox_scenes.py"
        argv_log = []

        # Our own copy of the script is pushed and run, NOT /workspace/nvblox_v2's. The copy
        # differs only in seeding the floor RANSAC (see its header); nvblox_v2 stays read-only
        # on both sides.
        mk = f"mkdir -p {job} {self.a.remote_out} {self.a.remote_scripts}"

        def fuse_cmd(depth_file, suffix=""):
            """The remote invocation for one stack. `--depth-file` and `--suffix` are
            nvblox_scenes.py's own flags (:624, :625) - nothing about the remote script
            changes to support the filter."""
            return (f"{self.a.remote_venv}/bin/python {script_remote} "
                    f"--packed-root {self.a.remote_jobs} --out-root {self.a.remote_out} "
                    f"--scenes {self.scene} --esdf-3d "
                    f"--floor-seed {self.a.floor_seed} --depth-file {depth_file}"
                    + (f" --rgb-file {RGB_STACK}" if self.colour else "")
                    + (f" --suffix {suffix}" if suffix else ""))

        # Every fusion this step performs, in the order it performs them. The CONTROL arms
        # run first: if the box dies mid-step the run has lost a control, not the thing it
        # was asked to produce, and a floor gate that rejects the filtered stack still
        # leaves a fusion on disk to say whether the box itself was fine.
        arms = []
        if self.mvfilter_ab:
            arms.append({"label": "ab_", "depth_file": "depth_u16.npy",
                         "suffix": "__unfiltered", "dest": self.ab_scene_dir,
                         "log": "NVBLOX_ab", "spec": None, "key": "unfiltered"})
        for alt in self.alt_arms:
            arms.append({"label": f"{alt['slug']}_", "depth_file": alt["depth_file"],
                         "suffix": f"__{alt['slug']}", "dest": alt["dest"],
                         "log": f"NVBLOX_{alt['slug']}", "spec": alt["spec"],
                         "key": alt["slug"]})
        # The run's OWN scene last: it is the one LAYERS/REACH/EXPORT read.
        arms.append({"label": "", "depth_file": self.depth_file, "suffix": "",
                     "dest": self.src_scene, "log": "NVBLOX", "spec": self.mvfilter,
                     "key": "primary"})
        for arm in arms:
            arm["cmd"] = fuse_cmd(arm["depth_file"], arm["suffix"])
            arm["remote_dir"] = f"{self.a.remote_out}/{self.scene}{arm['suffix']}"

        # The stacks that must reach the box. meta.json is shared: every arm describes the
        # same views, and MVFILTER only ever zeroes pixels.
        push_files = ["meta.json"]
        if self.colour:
            # Bigger than every depth stack put together (313 MB for 686 views), so it goes
            # through the same transfer gate as everything else rather than around it.
            push_files.append(RGB_STACK)
        for arm in arms:
            if arm["depth_file"] not in push_files:
                push_files.append(arm["depth_file"])

        argv_log += [f"ssh: {mk}", f"push: {script_local} -> {script_remote}"]
        argv_log += [f"push: {self.packed}/{f} -> {job}/" for f in push_files]
        for arm in arms:
            argv_log += [f"ssh: {arm['cmd']}",
                         f"pull: {arm['remote_dir']}/ -> {arm['dest']}/"]
        if self.dry_run:
            for line in argv_log:
                print(f"[{now()}] NVBLOX would run -> {line}")
            return {"remote_commands": argv_log, "executed": False}

        # Per-phase timing: the first live run reported one 562 s number for the whole
        # step, which said nothing about where it went. Transfer dominates on this link,
        # and knowing by how much is what decides whether pulling map.nvblx is worth it.
        phases: dict[str, float] = {}

        def timed(label, fn):
            t0 = time.time()
            r = fn()
            phases[label] = round(time.time() - t0, 2)
            return r

        r = timed("mkdir_s", lambda: sess.ssh(mk, timeout=120))
        if r.returncode != 0:
            raise StepFailed("NVBLOX", f"remote mkdir failed rc={r.returncode}", r.stderr)
        r = timed("push_script_s",
                  lambda: sess.push(script_local, script_remote, timeout=300))
        if r.returncode != 0:
            raise StepFailed("NVBLOX", f"push of the seeded script failed rc={r.returncode}",
                             r.stderr)

        # --- transfer gate: refuse before spending the time, not after ---------------
        # A stage that cannot finish inside --step-timeout-s will be killed mid-transfer
        # and leave a partial job behind, having burned the whole timeout on the clock. The
        # link speed is already measured by then, and the byte counts are known on both
        # sides, so the decision can be made up front. Sizes on the box are read for the
        # pull because after a re-run the outputs may already be there.
        self.status.set_substage("NVBLOX", "transfer_gate",
                                 "sizing the push and the pull against the measured link "
                                 f"speed, before spending {self.a.step_timeout_s}s on it")
        up = sess.info.get("inet_up_mb_s") or 0.0
        down = sess.info.get("inet_down_mb_s") or 0.0
        push_bytes = sum((self.packed / f).stat().st_size for f in push_files)
        try:
            remote = sess.remote_sizes(out, NVBLOX_PULL_FILES)
        except Exception as e:      # a box that cannot answer is not a gate failure
            remote = {}
            print(f"[{now()}] NVBLOX: could not size the remote outputs ({e}); "
                  f"estimating the pull from this scene's last known sizes", file=sys.stderr)
        pull_bytes = sum(remote.values()) or self.a.assumed_pull_bytes
        # Every extra arm is another full pull; the gate has to see the whole step, not the
        # fraction of it that was here when there was only ever one fusion.
        pull_bytes *= len(arms)
        est = {"inet_up_mb_s": up, "inet_down_mb_s": down,
               "push_bytes": push_bytes, "pull_bytes": pull_bytes,
               "push_est_s": round(push_bytes / 1e6 / up, 1) if up > 0 else None,
               "pull_est_s": round(pull_bytes / 1e6 / down, 1) if down > 0 else None}
        art_extra = {"transfer_estimate": est}
        for label, secs, rate in (("push", est["push_est_s"], up),
                                  ("pull", est["pull_est_s"], down)):
            if secs is not None and secs > self.a.step_timeout_s:
                mb = (push_bytes if label == "push" else pull_bytes) / 1e6
                raise StepFailed(
                    "NVBLOX",
                    f"transfer would take {secs:.0f} s at {rate:.2f} MB/s, exceeds "
                    f"--step-timeout-s {self.a.step_timeout_s} ({label} of {mb:.0f} MB); "
                    f"refused before starting it, the box is stopped by its own session")
        print(f"[{now()}] NVBLOX: transfer estimate push {est['push_est_s']}s @ "
              f"{up:.2f} MB/s, pull {est['pull_est_s']}s @ {down:.2f} MB/s "
              f"(timeout {self.a.step_timeout_s}s)")
        self.status.set_substage(
            "NVBLOX", None,
            f"gate passed: push ~{est['push_est_s']}s, pull ~{est['pull_est_s']}s")
        t0 = time.time()
        pushed_bytes = 0
        for f in push_files:
            r = sess.push(self.packed / f, f"{job}/", timeout=self.a.step_timeout_s)
            if r.returncode != 0:
                raise StepFailed("NVBLOX", f"push {f} failed rc={r.returncode}", r.stderr)
            pushed_bytes += (self.packed / f).stat().st_size
        phases["push_data_s"] = round(time.time() - t0, 2)

        def fuse(label, cmd, remote_dir, dest, log_tag):
            """One remote fusion + its pull. Both arms go through here so the only thing
            that differs between them is the depth file - not the timeout, not the pull
            list, not how the logs are written."""
            r = timed(f"{label}compute_s",
                      lambda: sess.ssh(cmd, timeout=self.a.step_timeout_s))
            (self.logs / f"{log_tag}.stdout").write_text(r.stdout or "", encoding="utf-8")
            (self.logs / f"{log_tag}.stderr").write_text(r.stderr or "", encoding="utf-8")
            if r.returncode != 0:
                raise StepFailed("NVBLOX", f"nvblox_scenes.py ({log_tag}) exited "
                                           f"{r.returncode}", r.stderr)
            dest.mkdir(parents=True, exist_ok=True)
            r = timed(f"{label}pull_s",
                      lambda: sess.pull(f"{remote_dir}/", dest,
                                        timeout=self.a.step_timeout_s,
                                        include=NVBLOX_PULL_FILES))
            if r.returncode != 0:
                raise StepFailed("NVBLOX", f"pull ({log_tag}) failed rc={r.returncode}",
                                 r.stderr)
            return sum(f.stat().st_size for f in dest.glob("*") if f.is_file())

        pulled_bytes = 0
        for arm in arms:
            if arm["label"]:
                self.status.set_substage(
                    "NVBLOX", None,
                    f"control arm {arm['key']}: fusing {arm['depth_file']} on the same box, "
                    f"same script, same floor seed")
                print(f"[{now()}] NVBLOX: control arm {arm['key']} - fusing "
                      f"{arm['depth_file']} -> {arm['dest']}")
            arm["pulled_bytes"] = fuse(arm["label"], arm["cmd"], arm["remote_dir"],
                                       arm["dest"], arm["log"])
            if not arm["label"]:
                pulled_bytes = arm["pulled_bytes"]

        rate = lambda b, s: round(b / s / 1e6, 3) if s and s > 0 else None  # noqa: E731
        art = {"remote_commands": argv_log, "executed": True,
               **art_extra,
               "depth_file": self.depth_file,
               "mvfilter": self.mvfilter,
               "phases_s": phases,
               "remote_wall_s": round(sum(phases.values()), 2),
               "pushed_bytes": pushed_bytes, "pulled_bytes": pulled_bytes,
               "push_mb_s": rate(pushed_bytes, phases.get("push_data_s")),
               "pull_mb_s": rate(pulled_bytes, phases.get("pull_s"))}
        # Control arms keep their own numbers rather than being folded into the totals:
        # `compute_s` is the like-for-like NVBLOX seconds against each arm's, and a reader
        # who sums them gets the step's wall time, not the comparison.
        controls = {a["key"]: {"dir": str(a["dest"]), "depth_file": a["depth_file"],
                               "spec": a["spec"], "remote_dir": a["remote_dir"],
                               "pulled_bytes": a["pulled_bytes"],
                               "compute_s": phases.get(f"{a['label']}compute_s"),
                               "pull_s": phases.get(f"{a['label']}pull_s")}
                    for a in arms if a["label"]}
        if controls:
            art["control_arms"] = controls
            art["control_note"] = ("fused in this run's own box session, same script and "
                                   "floor seed; not this run's scene - LAYERS/REACH/EXPORT "
                                   "read the primary arm only")
        return art

    def required_binaries(self) -> dict[str, str]:
        """External programs this run will shell out to, and what needs each one.

        Checked at QUEUED, before a single stage runs, because the driver discovers these
        the hard way otherwise: the first live worker run reached GPU_UP_NV - past FRAMES,
        PACK and PACK_RGB, 24.5 s in - and died on
        `FileNotFoundError: [Errno 2] No such file or directory: 'vastai'`. The binary was
        on the operator's interactive PATH and not on the unit's, which is
        `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/snap/bin` by systemd default and
        does not contain `~/.local/bin`. Nothing about that is visible from inside the
        run until the call is made, and by then a box may already be billing.

        Only what THIS run will actually use: a dry run never calls vastai, a precomputed
        nvblox source never opens an ssh session, and a colourless run never runs ffmpeg.
        Claiming a missing binary the run would not have touched is its own kind of wrong.
        """
        need: dict[str, str] = {}
        if not self.dry_run:
            need["vastai"] = "renting, starting and stopping the GPU box (vast_client.py)"
        if self.nvblox_live:
            need["ssh"] = "driving the nvblox box (box_ops.py)"
            need["scp"] = "the link probe and script push (box_ops.py)"
            need["rsync"] = "pushing the depth stack and pulling the fusion (run_pipeline.py)"
        if self.colour or not self.a.from_pulled:
            need["ffmpeg"] = ("extracting frames" if not self.a.from_pulled else
                              "re-deriving colour from the source video (steps/pack_rgb.py)")
        return need

    def check_binaries(self) -> dict:
        need = self.required_binaries()
        found = {b: shutil.which(b) for b in need}
        missing = sorted(b for b, p in found.items() if not p)
        if missing:
            raise StepFailed(
                "QUEUED",
                f"required binar{'y is' if len(missing) == 1 else 'ies are'} not on PATH: "
                + "; ".join(f"{b} ({need[b]})" for b in missing)
                + f". PATH={os.environ.get('PATH', '')!r}. Under systemd the unit's PATH is "
                  f"not the operator's - add an Environment=PATH= line to "
                  f"deploy/video-worker.service covering the directory each one lives in. "
                  f"Nothing was rented and no stage ran.",
                fail_reason="missing_binaries")
        return {b: found[b] for b in sorted(found)}

    # -- main ---------------------------------------------------------------------------
    def run(self) -> int:
        self.sup.start()
        try:
            self.status.enter("QUEUED")
            # Before anything else: the external programs this run will call must exist.
            # Raises StepFailed("QUEUED") naming each one, so a PATH problem costs 0 stages
            # and $0 instead of surfacing at GPU_UP_NV.
            binaries = self.check_binaries()
            self.status.finish_step("QUEUED", "done", {
                "binaries": binaries,
                "scene_dir": str(self.scene_dir), "from_pulled": str(self.pulled),
                "nvblox_source": self.a.nvblox_source, "scenes_out": str(self.scenes_out),
                "dry_run": self.dry_run, "max_usd": self.max_usd,
                "armed": self.vast.armed_state()[1],
                "worker_mode": bool(self.a.worker_mode),
                # Recorded at QUEUED, not only at MVFILTER, so "which depth did nvblox
                # actually fuse" is answerable from the top of the file.
                "mvfilter": self.mvfilter,
                "mvfilter_ab": self.mvfilter_ab,
                "nvblox_depth_file": self.depth_file,
                # In worker mode ARMED is not the gate - PIPELINE_WORKER_ENABLED and
                # PIPELINE_DAILY_USD are, and worker_driver.py has already enforced both
                # before this process existed. Recording what they were makes the status
                # file answer "why did this rent / not rent" without a second file.
                "worker_guards": ({
                    "PIPELINE_WORKER_ENABLED": os.environ.get("PIPELINE_WORKER_ENABLED"),
                    "PIPELINE_DAILY_USD": os.environ.get("PIPELINE_DAILY_USD"),
                } if self.a.worker_mode else None),
                "waiting_capacity_ceiling": ({
                    "max_attempts": self.a.capacity_max_attempts,
                    "max_s": WAITING_CAPACITY_MAX_S,
                    "backoff_s": self.a.capacity_backoff_s,
                } if self.a.worker_mode else None),
                "note": "dry run" if self.dry_run else "LIVE"})

            precomp = "precomputed" if self.a.from_pulled else "real"
            self.step("FRAMES", self.check_frames,
                      produce=None if self.a.from_pulled else self.produce_frames,
                      kind=precomp)

            # --- MapAnything box ---------------------------------------------------------
            # `--from-pulled` is an existing pass-A run; worker mode has none and waits for
            # one by name. Either way no B200 is rented from here: pass A is a manual step.
            if self.a.from_pulled or self.a.worker_mode:
                # The B200 is not rented at all: the views already exist on disk. The
                # machine that *would* be needed is recorded for the report; nothing acts
                # on it - picking and renting a machine stays a human act.
                no_box = lambda: (True, {                                   # noqa: E731
                    "skipped": "MapAnything served from --from-pulled; no box rented",
                    "would_need": select_gpu(self.n_views_from_pulled()),
                    "note": "no box rented"})
                self.step("GPU_UP_MA", no_box, kind="precomputed")
                self.step("MAPANYTHING", self.check_mapanything,
                          produce=self.wait_for_manual_mapanything, kind=precomp)
                self.step("PULLED", self.check_pulled, kind=precomp)
                self.step("GPU_DOWN_MA", no_box, kind="precomputed")
            else:
                raise StepFailed("GPU_UP_MA", "live MapAnything is not wired in this build: "
                                 "rent and drive the B200 by hand, then pass --from-pulled")

            # A declared slot with no implementation, said out loud rather than left out of
            # the state list. This is where `provider_used` goes once a vocabulary provider
            # exists (JOB_SPEC 5b) - and until then nothing writes it, because an absent
            # provider_used means "no fallback happened" and a guessed one would be a lie.
            self.step("SEMANTICS", lambda: (True, {
                "_substage": "skipped", "_detail": "not implemented",
                "spec_semantics": getattr(self.a, "semantics", None),
                "note": "SEMANTICS is not implemented in this build"}), kind="real")

            self.step("PACK", self.check_pack, produce=self.produce_pack, kind="real")

            # Colour, re-derived from the source video, because nothing upstream carries it:
            # the per-view npz hold pts3d/mask/conf and FRAMES writes an index, not JPEGs.
            # Before GPU_UP_NV for the same reason MVFILTER is - it is a CPU pass, and a
            # colour extraction that fails should cost no box time.
            self.step("PACK_RGB", self.check_pack_rgb,
                      produce=self.produce_pack_rgb if self.colour else None, kind="real")

            # Before GPU_UP_NV deliberately: this is a CPU pass over the depth stack, and a
            # filter that fails should cost nothing. It is also the last chance to change
            # what nvblox sees - after this the stack is on the box.
            self.step("MVFILTER", self.check_mvfilter,
                      produce=self.produce_mvfilter if self.mvfilter else None, kind="real")

            # --- nvblox box --------------------------------------------------------------
            nv = self.a.nvblox_instance_id
            if self.nvblox_live:
                acquire = (self.acquire_box if self.a.worker_mode else self.bring_up)
                # Both the check and the take-down read the id lazily, never the `nv`
                # captured above: acquire_box may have rented a replacement, and a check
                # against the id that just refused reports "no live session yet" for a box
                # that is up and billing.
                self.step("GPU_UP_NV",
                          lambda: self.check_box_up(self.a.nvblox_instance_id),
                          produce=lambda: acquire("GPU_UP_NV", nv, "nvblox"),
                          kind="real", skip_check_after=self.dry_run)
                self.step("NVBLOX", self.check_nvblox, produce=self.produce_nvblox,
                          kind="real", skip_check_after=True)
                # In a dry run NVBLOX produced nothing locally, so there is no
                # floor_plane.json to check - say so rather than fail for the wrong reason.
                floor_check = (self.check_floor if not self.dry_run
                               else (lambda: (True, {"note": "dry run: floor plane would "
                                                     "come from this run's own nvblox "
                                                     "output"})))
                self.step("FLOOR", floor_check,
                          kind="real" if not self.dry_run else "stub")
                # Read the id lazily, not the `nv` captured above: acquire_box may have
                # rented a replacement, and stopping yesterday's id while today's keeps
                # billing is exactly the failure this whole ledger exists to prevent.
                self.step("GPU_DOWN_NV",
                          lambda: self.check_box_down(self.a.nvblox_instance_id),
                          produce=lambda: self.take_down(self.a.nvblox_instance_id),
                          kind="real", skip_check_after=False)
            else:
                self.step("GPU_UP_NV", lambda: (True, {
                    "skipped": "--nvblox-source precomputed; no box touched",
                    "note": "no box touched"}), kind="precomputed")
                self.step("NVBLOX", self.check_nvblox, kind="precomputed")
                self.step("FLOOR", self.check_floor, kind="precomputed")
                self.step("GPU_DOWN_NV", lambda: (True, {
                    "skipped": "--nvblox-source precomputed; no box touched",
                    "note": "no box touched"}), kind="precomputed")

            self.step("LAYERS", self.dry_gated(self.check_layers, "LAYERS"),
                      produce=self.produce_layers,
                      kind="stub" if self.dry_run and self.nvblox_live else "real")
            self.step("REACH", self.dry_gated(self.check_reach, "REACH"),
                      produce=self.produce_reach,
                      kind="stub" if self.dry_run and self.nvblox_live else "real")
            self.step("EXPORT", self.dry_gated(self.check_export, "EXPORT"),
                      produce=self.produce_export,
                      kind="stub" if self.dry_run and self.nvblox_live else "real")
            # `kind="real"` now, not "stub": INGEST writes files the app opens. It is gated
            # like LAYERS/REACH/EXPORT because in a dry run of the live path there is no
            # local nvblox output to measure a ceiling or a bbox from.
            self.step("INGEST", self.dry_gated(self.check_ingest, "INGEST"),
                      produce=self.produce_ingest,
                      kind="stub" if self.dry_run and self.nvblox_live else "real")
            self.status.terminal("DONE")
            print(f"[{now()}] DONE in {self.status.doc['duration_s']}s "
                  f"(measured cost ${self.status.doc['cost_estimate_usd']})")
            return 0
        except StepFailed as e:
            msg = e.message[len(e.state) + 2:] if e.message.startswith(e.state + ": ") \
                else e.message
            error = f"{e.state}: {msg}\n--- stderr ---\n{e.stderr}"
            if e.state not in self.status.doc["steps"]:
                # A budget abort fires at the boundary, before the state is entered. Record
                # it anyway: a status file that names a failing state in `error` but has no
                # entry for it under `steps` is a file that hides where the run stopped.
                self.status.enter(e.state)
                self.status.doc["steps"][e.state]["artifacts"] = {
                    "note": "aborted at the state boundary, before this step started"}
            self.status.finish_step(e.state, "failed",
                                    {"stderr_log": str(self.logs / f"{e.state}.stderr"),
                                     "fail_reason": e.fail_reason}, error=error)
            self.status.terminal("FAILED", error=error, fail_reason=e.fail_reason)
            print(f"[{now()}] FAILED at {e.state}: {msg}", file=sys.stderr)
            if e.fail_reason:
                print(f"[{now()}] fail_reason: {e.fail_reason}", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            return 1
        except VastNotArmed as e:
            self.status.terminal("FAILED", error=f"not armed: {e}")
            print(f"[{now()}] FAILED: {e}", file=sys.stderr)
            return 2
        finally:
            # Every box that is still up gets stopped, each in its own try - a failure
            # stopping one must not prevent the other from being stopped.
            for iid in list(self.sessions):
                try:
                    self.take_down(iid)
                except Exception as e:
                    print(f"[{now()}] stop {iid} failed: {e}", file=sys.stderr)
            self.sup.stop()

    def dry_gated(self, check, state: str):
        """In a dry run of the LIVE path, NVBLOX produced no local output, so every step
        downstream of it has no input to check. Report that plainly instead of failing on a
        missing file - a dry run of the morning command is a rehearsal, and its job is to
        print every command that would run. Outside a dry run this is a no-op wrapper."""
        if self.dry_run and self.nvblox_live and not (
                self.src_scene / "esdf_slice_0.3m.npy").exists():
            return lambda: (True, {
                "note": f"dry run: {state} input comes from this run's own NVBLOX output "
                        f"({self.src_scene}), which a dry run does not produce"})
        return check

    def n_views_from_pulled(self) -> int:
        try:
            return int(load_json(self.pass_a / "poses.json")["n_views"])
        except Exception:
            return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="video -> scene catalogue, one state machine")
    ap.add_argument("video")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-name", required=True,
                    help="run/scene id, e.g. own_0901_173903__step2")
    ap.add_argument("--from-pulled", default=None,
                    help="existing pulled MapAnything run; makes FRAMES..PULLED precomputed")
    ap.add_argument("--frames-fps", type=float, default=None,
                    help="sampling rate for FRAMES, from the job spec's frames_fps. "
                         "Default None keeps whatever the precomputed frames.json says.")
    ap.add_argument("--reach-radii", type=float, nargs="+",
                    default=[0.1, 0.2496, 0.5528],
                    help="disc radii REACH reports, in metres. The defaults are the "
                         "registry's TurtleBot burger, Unitree Go2 and Clearpath Husky "
                         "(app/robots.py), so the report names real platforms.")
    ap.add_argument("--reach-heights", type=float, nargs="+",
                    default=[0.192, 0.40, 0.3963],
                    help="robot HEIGHT per --reach-radii entry, from app/robots.py's "
                         "resolve_height_m: TurtleBot3 burger 0.192, Unitree Go2 0.40 "
                         "(VENDOR STANDING, 700x310x400mm - not the 0.1847 m trunk-only "
                         "mesh bbox posed at zero joint angles), Clearpath Husky 0.3963. A surface above this is driven "
                         "under, not around - measured on hero-74, a fixed 1.5 m band "
                         "turned a 0.694 m bed top into a wall for a 0.192 m robot.")
    ap.add_argument("--band-margin", type=float, default=0.05,
                    help="headroom over a robot's own height before a surface blocks it")
    ap.add_argument("--band-min", type=float, default=0.1,
                    help="floor of the obstacle band, metres above the fitted floor plane")
    ap.add_argument("--band-max", type=float, default=1.5,
                    help="ceiling of the obstacle band; above this is the room's ceiling, "
                         "which a ground robot drives under")
    ap.add_argument("--band-min-component-tri", type=int, default=100,
                    help="drop mesh components smaller than this before rasterising - an "
                         "isolated 10-triangle blob is reconstruction noise, and leaving it "
                         "in destroys the clearance field around it")
    ap.add_argument("--export-max-tri", type=int, default=None,
                    help="decimation cap handed to isaac_export.py --max-tri. Default None "
                         "leaves that script's own 200000 alone; the offline suite sets it "
                         "low because it asserts the export EXISTS, not how dense it is, "
                         "and 200k triangles is 8 MB of scene.usd + scene.glb per case.")
    ap.add_argument("--semantics", default=None,
                    help="the job spec's `semantics` provider, recorded on the SEMANTICS "
                         "step. Nothing runs it: SEMANTICS is a declared slot with no "
                         "implementation in this build.")
    ap.add_argument("--worker-mode", action="store_true",
                    help="driven by app/worker.py rather than an operator: the ARMED file "
                         "is replaced by PIPELINE_WORKER_ENABLED and PIPELINE_DAILY_USD, "
                         "both enforced in pipeline/worker_driver.py before this runs")
    ap.add_argument("--nvblox-source", choices=["precomputed", "live"],
                    default="precomputed",
                    help="'live' brings up the nvblox box and runs nvblox_scenes.py there")
    ap.add_argument("--nvblox-root", default="var/nvblox_v2")
    ap.add_argument("--scenes-out-root", default="var/nvblox_v2/results/scenes_out",
                    help="only used with --nvblox-source precomputed")
    ap.add_argument("--python-cpu", default=None,
                    help="interpreter for the CPU stages (PACK/LAYERS/REACH/EXPORT). "
                         "Defaults to $PIPELINE_CPU_PYTHON, else cpu_env.DEFAULT - see "
                         "pipeline/cpu_env.py. Checked once, before QUEUED, so a missing "
                         "venv is one line at the start rather than a traceback at PACK.")
    ap.add_argument("--mesh-name", default="mesh.ply",
                    help="the mesh EXPORT reads out of the nvblox output. `mesh.ply` is what "
                         "nvblox_scenes.py actually writes; the published Isaac export used "
                         "`mesh_color.ply`, which a different script (nvblox_hero_color.py) "
                         "produced and which a live run does not have.")
    ap.add_argument("--keep-box-up", action="store_true", default=False,
                    help="do NOT stop the nvblox box at the end of the run. It keeps "
                         "billing (check `dph_total` in status.json) and must be stopped by "
                         "hand: `vastai stop instance <id>`. Use it when several runs are "
                         "queued back to back - a refused START cannot be retried around, "
                         "and paying for an idle hour beats losing the queue to "
                         "'Required resources are currently unavailable'.")
    ap.add_argument("--color", dest="color", action="store_true", default=True,
                    help="integrate colour (default). PACK_RGB re-derives RGB from the "
                         "source video and NVBLOX calls add_color_frame, so mesh.ply and "
                         "the exported USD carry real vertex colours.")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="skip colour. Reproduces every run before 2026-09-10: the mesh "
                         "still has an RGB property, every vertex of it nvblox's unset "
                         "127/127/127, and the USD renders flat grey.")
    ap.add_argument("--mvfilter", default=None,
                    help="OPTIONAL, OFF BY DEFAULT. Run the multi-view consistency vote over "
                         "the packed depth stack before nvblox fuses it, and fuse the "
                         "filtered stack instead. Spec: `m=2,tau=0.10,neighbours=nearest,K=8` "
                         "- every key required, no defaults, so status.json records exactly "
                         "which vote made the depth. The rule and these values come from "
                         "var/scratch/mvfilter/REPORT.md; see pipeline/steps/mvfilter.py. "
                         "Omitting this leaves the default path untouched.")
    ap.add_argument("--mvfilter-ab", action="store_true",
                    help="with --mvfilter, ALSO fuse the unfiltered stack in the same box "
                         "session, into <scene-dir>/nvblox_unfiltered/. The control the "
                         "side-by-side needs: floor selection changed in 680dbae, so an "
                         "older run's obstacle cells are not comparable to today's. Costs a "
                         "second compute + pull; LAYERS onwards still read the filtered run.")
    ap.add_argument("--mvfilter-also", action="append", default=None, metavar="SPEC",
                    help="with --mvfilter, filter and fuse ANOTHER spec in the same session, "
                         "into <scene-dir>/nvblox_<slug>/. Repeatable. The neighbour rule "
                         "dominates this filter - on hero-15fps `nearest` keeps 74.0%% of "
                         "valid pixels where `baseline15` keeps 57.0%% - so one filtered arm "
                         "against unfiltered cannot separate 'the filter does not help' from "
                         "'this arm barely filtered'.")
    ap.add_argument("--nvblox-instance-id", type=int, default=None,
                    help="the nvblox box to start. Left unset it resolves to the CACHED id "
                         "in <state-dir>/nvblox_box_id, which is the id start_or_create "
                         "rewrites every time it replaces a box - so the default follows the "
                         "box that actually exists. NVBLOX_BOX_ID is only the last-resort "
                         "fallback when there is no cache at all: it is a historical "
                         "constant (50265868, retired 2026-09-09 and since destroyed), and "
                         "starting a destroyed id matches REFUSAL_MARKERS, which sends "
                         "start_or_create off to rent a NEW box while the cached one idles.")
    ap.add_argument("--capacity-max-attempts", type=int,
                    default=WAITING_CAPACITY_MAX_ATTEMPTS,
                    help="worker-mode waiting_capacity ceiling, in attempts")
    ap.add_argument("--capacity-backoff-s", type=int, default=WAITING_CAPACITY_BACKOFF_S,
                    help="seconds between waiting_capacity attempts")
    ap.add_argument("--state-dir", default=str(STATE_DIR),
                    help="where the cached box id and the retired list live. Overridable so "
                         "a test can exercise the create path without rewriting the real "
                         "cache - start_or_create rewrites it on every replacement.")
    ap.add_argument("--armed-file", default=str(HERE / "ARMED"),
                    help="the arming file; overridden only by the offline test harness")
    ap.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519_vast"))
    ap.add_argument("--ssh-wait-s", type=int, default=300)
    ap.add_argument("--ssh-poll-s", type=int, default=15)
    ap.add_argument("--probe-mb", type=int, default=50,
                    help="minimum link-probe size. 20 MB was measured to under-read by ~2x "
                         "(said 2.49 MB/s where the real 209.6 MB push ran at 6.20)")
    ap.add_argument("--probe-min-s", type=float, default=10.0,
                    help="minimum probe duration; a probe shorter than this is repeated "
                         "with a larger file, since TCP has not reached its stride")
    ap.add_argument("--assumed-pull-bytes", type=int, default=17_000_000,
                    help="pull size used by the transfer gate when the box cannot be asked "
                         "(the five files NVBLOX_PULL_FILES name are ~16.8 MB on hero)")
    ap.add_argument("--billing-from-epoch", type=float, default=None,
                    help="unix time the box began billing - pass the moment `create` "
                         "returned, so provisioning and link probes are charged to this run "
                         "rather than silently dropped from the ledger")
    ap.add_argument("--exit-confirm-s", type=int, default=120,
                    help="how long to wait for a stopped box to report `exited`; the cost "
                         "clock runs to that confirmation, not to the stop call")
    ap.add_argument("--remote-jobs", default="/workspace/pipeline_jobs")
    ap.add_argument("--remote-out", default="/workspace/pipeline_out")
    ap.add_argument("--remote-nvblox", default="/workspace/nvblox_v2",
                    help="the box's own read-only nvblox tree; not executed by this driver")
    ap.add_argument("--remote-scripts", default="/workspace/pipeline_scripts",
                    help="where this driver pushes its own copy of nvblox_scenes.py")
    ap.add_argument("--floor-seed", type=int, default=42)
    ap.add_argument("--remote-venv", default="/workspace/venvs/nb")
    ap.add_argument("--heartbeat-s", type=int, default=HEARTBEAT_S,
                    help="supervisor tick: heartbeat file + measured-cost budget poll")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--step-timeout-s", type=int, default=900)
    a = ap.parse_args()

    # --mvfilter is parsed HERE, with the other pre-flight checks, for the same reason the
    # CPU interpreter is: MVFILTER runs before GPU_UP_NV but after several minutes of PACK,
    # and a typo in the spec should cost a line at argv time, not a rerun.
    a.mvfilter_spec = None
    if a.mvfilter is not None:
        try:
            a.mvfilter_spec = parse_mvfilter(a.mvfilter)
        except ValueError as e:
            print(f"refusing to start: --mvfilter {a.mvfilter!r}: {e}\n"
                  "  expected e.g. --mvfilter m=2,tau=0.10,neighbours=nearest,K=8",
                  file=sys.stderr)
            return 2
        for extra in (a.mvfilter_also or []):
            try:
                parse_mvfilter(extra)
            except ValueError as e:
                print(f"refusing to start: --mvfilter-also {extra!r}: {e}", file=sys.stderr)
                return 2
    elif a.mvfilter_ab or a.mvfilter_also:
        flag = "--mvfilter-ab" if a.mvfilter_ab else "--mvfilter-also"
        print(f"refusing to start: {flag} is a control arm for --mvfilter and does nothing "
              f"without it; there is no filtered stack to compare against", file=sys.stderr)
        return 2

    # Run mode. A real vast call additionally needs the ARMED file (vast_client), so this
    # only decides whether the driver even offers to make one.
    dry_env = os.environ.get("PIPELINE_DRY_RUN")
    if dry_env == "1":
        a.dry_run = True
    elif dry_env is not None:
        print(f"refusing to start: PIPELINE_DRY_RUN is set to {dry_env!r}; "
              "set it to 1 for a dry run or unset it for a live run", file=sys.stderr)
        return 2
    else:
        probe = VastClient(Path(a.scene_dir) / "vast_calls.log", Path(a.armed_file))
        armed, reason = probe.armed_state()
        if not armed:
            print(f"refusing to start a live run: {reason}\n"
                  "Either set PIPELINE_DRY_RUN=1, or arm the run with:\n"
                  f"  printf 'yes %s\\n' \"$(date +%F)\" > {a.armed_file}",
                  file=sys.stderr)
            return 2
        a.dry_run = False
        print(f"[{now()}] LIVE RUN - {reason}; max_usd={a.max_usd}")

    # Resolve the nvblox box id the same way for both paths, before either of them runs.
    # `acquire_box` hands its id to start_or_create explicitly, and start_or_create only
    # falls back to the cache when that id is None - so an argparse default of NVBLOX_BOX_ID
    # meant the cache was never consulted, and the cache is the one thing that knows which
    # box currently exists. `bring_up` (the operator path) needs a real int either way.
    if a.nvblox_instance_id is None:
        cached = VastClient.cached_box_id(a.state_dir)
        a.nvblox_instance_id = cached if cached is not None else NVBLOX_BOX_ID
        src = f"cached id from {a.state_dir}" if cached is not None else \
            "NVBLOX_BOX_ID fallback (no cache found)"
        print(f"[{now()}] nvblox box: {a.nvblox_instance_id} ({src})")

    # Resolve and CHECK the CPU interpreter here, before any state is entered and before a
    # box can be rented. PACK is the first stage that needs it and it sits after GPU_UP_NV
    # on the live path - discovering a missing venv there means a box was started, billed
    # and then thrown away for a reason that was knowable at argv-parse time.
    a.python_cpu = a.python_cpu or cpu_python()
    try:
        require_cpu_python(a.python_cpu)
    except RuntimeError as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    return Pipeline(a).run()


if __name__ == "__main__":
    sys.exit(main())
