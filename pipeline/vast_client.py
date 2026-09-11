#!/usr/bin/env python3
"""Real vast.ai client over the `vastai` CLI — with two independent safeties in front of
every state-changing call.

Design rules, each one written because breaking it cost something on this project:

* **Id-only.** Every method takes a mandatory `instance_id`. There is no method that takes a
  filter, a label, or a listing, so "stop everything that matches X" cannot be expressed —
  a parallel session's box can appear in a listing mid-cleanup.
* **`search_offers` is report-only.** It returns offers for the record; nothing in this
  module or in `run_pipeline.py` feeds its result into a decision. Choosing a machine is a
  human act.
* **`dry_run` is keyword-only with no default.** Forgetting it is a TypeError, not a live call.
* **Two safeties for a real call**, both required (`_may_execute`):
    1. `PIPELINE_DRY_RUN` must be ABSENT from the environment (absent, not "!= 1"), and
    2. `pipeline/ARMED` must exist, contain `yes <YYYY-MM-DD>`, and that date must be TODAY.
  A stale ARMED left on disk from a previous day does not arm anything.
* **This module never reads the API key.** `vastai` resolves it itself from
  `~/.config/vastai/vast_api_key` (`vastai/cli/util.py:177`). There is no path here that
  opens, reads or passes the key — the guarantee is that the CLI is never executed at all
  unless both safeties pass.

Every call, executed or not, appends one line to the call log:

    <iso8601>\t(WOULD_RUN|RAN|REFUSED)\t<argv>\t-> <result>

CLI facts this module depends on, read from vastai 1.5.6's own source rather than assumed:

* `show instance ID --raw` returns JSON with `actual_status`, `cur_state`, `ssh_host`,
  `ssh_port`, `public_ipaddr`, `direct_port_start`, `gpu_name`, `inet_up`, `dph_total`
  (`vastai/data/instance.py:53-104`).
* `destroy instance ID -y` — `--yes` exists (`cli/commands/instances.py:228`); piping `yes |`
  is not needed.
* `start instance ID` accepts NO `--onstart-cmd` (`instances.py:292-294`); that flag belongs
  to `create instance` only (`:129-130`). Fixing the image's ssh key modes on a *started* box
  therefore happens over ssh after the first successful connect — see `box_ops.py`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

# The nvblox box. Named here so no caller retypes it.
NVBLOX_BOX_ID = 50265868   # historical default; the live value lives in pipeline/state

STATE_DIR = Path(__file__).resolve().parent / "state"

# Hosts that failed below our layer. 109808 (Taiwan) could not give its container a GPU
# through CDI on 2026-09-09 - see PIPELINE.md 4.4f. reliability2 does not predict this.
KNOWN_BAD_MACHINES = {109808}

EU_CC = {"SE","IS","NO","FI","DK","DE","NL","FR","PL","CZ","EE","LV","LT","ES","IT","AT",
         "CH","GB","UK","IE","RO","BG","HU","SK","SI","HR","GR","PT","BE","LU","UA","MD"}
US_CC = {"US","CA"}
CN_CC = {"CN","HK"}

OFFER_QUERY = ("gpu_name=RTX_4090 num_gpus=1 inet_down>=200 inet_up>=200 disk_space>=60 "
               "reliability>=0.98 verified=true rentable=true")

#: Hard ceiling on price, $/hour, applied in rank_offers alongside every other hard filter
#: rather than left to a caller's default argument. The box this pipeline actually works on
#: costs 0.80, so 1.00 is headroom over a known-good listing and not a guess. It is a HARD
#: filter, not a preference: an offer above it is dropped before ranking, so no amount of
#: being in the right region can bring an expensive box back.
MAX_DPH = 1.00

ARMED_RE = re.compile(r"^yes\s+(\d{4}-\d{2}-\d{2})\s*$")

# vastai's own default timeout behaviour is fine; this bounds a hung CLI.
CALL_TIMEOUT_S = 120


class VastNotArmed(RuntimeError):
    """A real call was requested but the safeties did not both pass."""


# Substrings that mean vast REFUSED, in stdout, on a call that still exits 0.
#
# Recorded from a real refusal on 2026-09-09: `vastai start instance 50265868` returned
# rc=0 with "Required resources are currently unavailable, state change queued." The box
# never started, the driver waited out its full 300 s ssh deadline, and only that text said
# why. rc is not the signal; the words are.
#
# The list is deliberately broad, per the operator's instruction. "failed" and "error" could
# in principle appear in a benign message, so this check is applied ONLY to `start` and
# `stop`, whose successful output is one short line ("starting instance N.", "stopping
# instance N."). A false positive costs one aborted run that says exactly what it matched;
# a false negative costs five minutes of waiting on a box that was never started.
REFUSAL_MARKERS = (
    "required resources are currently unavailable",
    "state change queued",
    "not enough",
    "failed",
    "error",
)


class VastStartUnavailable(RuntimeError):
    """vast declined a start/stop, in words, while exiting 0."""

    def __init__(self, argv: list[str], stdout: str):
        super().__init__(stdout.strip())
        self.argv, self.stdout = argv, stdout.strip()


class VastCallFailed(RuntimeError):
    def __init__(self, argv: list[str], rc: int, stderr: str):
        super().__init__(f"{' '.join(argv)} exited {rc}: {stderr.strip()[:400]}")
        self.argv, self.rc, self.stderr = argv, rc, stderr


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class VastClient:
    def __init__(self, log_path, armed_file, vastai_bin: str = "vastai"):
        self.log_path = Path(log_path)   # created lazily, on the first logged call
        self.armed_file = Path(armed_file)
        self.vastai_bin = vastai_bin
        self.calls: list[dict] = []

    # -- safeties -----------------------------------------------------------------------
    def armed_state(self) -> tuple[bool, str]:
        """(armed, reason). Never raises, never reads anything but the ARMED file."""
        if "PIPELINE_DRY_RUN" in os.environ:
            return False, ("PIPELINE_DRY_RUN is set in the environment "
                           f"({os.environ['PIPELINE_DRY_RUN']!r}) - dry run only")
        if not self.armed_file.exists():
            return False, f"no ARMED file at {self.armed_file}"
        try:
            text = self.armed_file.read_text(encoding="utf-8")
        except OSError as e:
            return False, f"ARMED file unreadable: {e}"
        m = ARMED_RE.match(text.strip() + "\n" if not text.endswith("\n") else text)
        if not m:
            return False, f"ARMED file does not match 'yes <YYYY-MM-DD>': {text.strip()!r}"
        today = dt.date.today().isoformat()
        if m.group(1) != today:
            return False, (f"ARMED file is dated {m.group(1)}, today is {today} - "
                           "a stale ARMED does not arm anything")
        return True, f"armed for {today}"

    #: Response fields that are credentials. `create` returns an `instance_api_key` this
    #: pipeline never uses (it authenticates by ssh key and account key). Stripping it from
    #: the RETURN value is not enough - the raw CLI output also goes to vast_calls.log, which
    #: is kept next to the scene and read in reports, so it is redacted on the way in too.
    SECRET_FIELDS = ("instance_api_key", "api_key", "ssh_key", "password", "token")

    @classmethod
    def redact(cls, text: str) -> str:
        """Replace the value of any secret field in a JSON-ish string with <redacted>."""
        out = str(text)
        for field in cls.SECRET_FIELDS:
            out = re.sub(rf'("{field}"\s*:\s*)"[^"]*"', r'\1"<redacted>"', out)
            out = re.sub(rf"({field}=)\S+", r"\1<redacted>", out)
        return out

    def _record(self, mode: str, argv: list[str], result) -> None:
        result = self.redact(result) if isinstance(result, str) else result
        entry = {"ts": now(), "mode": mode, "argv": argv, "result": result}
        self.calls.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{entry['ts']}\t{mode}\t{' '.join(argv)}\t"
                     f"-> {self.redact(result)}\n")

    @staticmethod
    def refusal_in(stdout: str) -> str | None:
        """The refusal marker that matched, or None. Case-insensitive."""
        low = (stdout or "").lower()
        for m in REFUSAL_MARKERS:
            if m in low:
                return m
        return None

    def _call(self, argv: list[str], *, dry_run: bool, stub_result, parse_json: bool = False,
              check_refusal: bool = False):
        """The single funnel every method goes through. Nothing else may run the CLI.

        `check_refusal` is set for the state-changing calls: vast can decline them in stdout
        while exiting 0, and that text must become an exception rather than a five-minute
        wait for a box nobody started.
        """
        argv = [self.vastai_bin] + argv
        if dry_run:
            self._record("WOULD_RUN", argv, f"{stub_result} (stub)")
            return stub_result
        armed, reason = self.armed_state()
        if not armed:
            self._record("REFUSED", argv, reason)
            raise VastNotArmed(f"refusing a real vast call: {reason}")
        p = subprocess.run(argv, capture_output=True, text=True, timeout=CALL_TIMEOUT_S)
        head = (p.stdout or "").strip().replace("\n", " ")[:200]
        self._record("RAN", argv, f"rc={p.returncode} {head}")
        if check_refusal:
            marker = self.refusal_in(p.stdout)
            if marker:
                # Checked BEFORE the return code: the whole point is that rc was 0.
                self._record("REFUSED_BY_VAST", argv, f"matched {marker!r}: {head}")
                raise VastStartUnavailable(argv, p.stdout)
        if p.returncode != 0:
            raise VastCallFailed(argv, p.returncode, p.stderr)
        if parse_json:
            try:
                return json.loads(p.stdout)
            except json.JSONDecodeError as e:
                raise VastCallFailed(argv, 0, f"could not parse --raw JSON: {e}")
        return p.stdout

    # -- read ---------------------------------------------------------------------------
    def show(self, instance_id: int, *, dry_run: bool) -> dict:
        """`vastai show instance <id> --raw`. Returns the instance dict."""
        return self._call(["show", "instance", str(int(instance_id)), "--raw"],
                          dry_run=dry_run, parse_json=True,
                          stub_result={"id": int(instance_id),
                                       "actual_status": "stub", "cur_state": "stub",
                                       "dph_total": 0.0, "gpu_name": "stub",
                                       "ssh_host": None, "ssh_port": None,
                                       "public_ipaddr": None, "direct_port_start": None})

    def search_offers(self, query: str, *, dry_run: bool) -> list:
        """REPORT ONLY. Nothing consumes this result to pick a machine — see the module
        docstring. It exists so a run can record what was on the market at the time."""
        return self._call(["search", "offers", query, "-o", "dph", "--raw"],
                          dry_run=dry_run, parse_json=True, stub_result=[])

    # -- the box cache: which instance to try first, and what was replaced ----------------
    @staticmethod
    def cached_box_id(state_dir: Path | str = STATE_DIR) -> int | None:
        f = Path(state_dir) / "nvblox_box_id"
        try:
            return int(f.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def set_cached_box_id(instance_id: int, state_dir: Path | str = STATE_DIR) -> None:
        d = Path(state_dir); d.mkdir(parents=True, exist_ok=True)
        (d / "nvblox_box_id").write_text(f"{int(instance_id)}\n")

    @staticmethod
    def retire_box_id(instance_id: int, reason: str,
                      state_dir: Path | str = STATE_DIR) -> None:
        """Append to the list the OWNER destroys. Nothing here is destroyed automatically -
        `destroy` is denied to the agent and stays a human act."""
        d = Path(state_dir); d.mkdir(parents=True, exist_ok=True)
        with (d / "retired_box_ids").open("a", encoding="utf-8") as fh:
            fh.write(f"{int(instance_id)}  # {now()} {reason}\n")

    def rank_offers(self, offers: list, max_dph: float = MAX_DPH) -> list:
        """Hard filter, then EU > US > other > CN, then price.

        Region beats price, and that ordering is measured rather than preferred: the
        binding constraint on this pipeline is the link to THIS VPS, not the headline
        bandwidth on the listing. The UK box delivered 4.38 MB/s where Taiwan gave 0.42, a
        10x difference that decides whether the NVBLOX pull fits inside the step timeout.
        A cheap box on a slow link is not a cheaper way to do the same work; it is a run
        that fails at the transfer gate.

        So a CN offer at $0.32/h ranks BELOW an EU offer at $0.80/h, and that is the whole
        point of the sort key being (region, price) and not (price, region).

        `max_dph` is a hard filter here, applied with the others, so an offer that is too
        expensive is gone before any of that ranking runs.
        """
        out = []
        for r in offers:
            try:
                drv = float(str(r.get("driver_version", "0")).split(".")[0])
            except ValueError:
                drv = 0.0
            if (r.get("gpu_name") != "RTX 4090" or (r.get("num_gpus") or 0) != 1
                    or r.get("verification") != "verified"
                    or (r.get("cuda_max_good") or 0) < 12.8 or drv < 580
                    or (r.get("inet_up") or 0) < 200 or (r.get("inet_down") or 0) < 200
                    or (r.get("reliability2") or 0) < 0.98
                    or (r.get("disk_space") or 0) < 60):
                continue
            if r.get("machine_id") in KNOWN_BAD_MACHINES:
                continue
            if (r.get("dph_total") or 9e9) > max_dph:
                continue
            cc = (r.get("geolocation") or "").split(",")[-1].strip().upper()
            rank = 0 if cc in EU_CC else 1 if cc in US_CC else 3 if cc in CN_CC else 2
            out.append((rank, r.get("dph_total", 9e9), r))
        out.sort(key=lambda t: (t[0], t[1]))
        return [r for _, _, r in out]

    def start_or_create(self, *, dry_run: bool, label: str, onstart_file: str,
                        instance_id: int | None = None, max_dph: float = MAX_DPH,
                        state_dir: Path | str = STATE_DIR) -> dict:
        """Start the cached box; if its host refuses, rent a replacement and cache that.

        Today's history is the whole argument for this method: 50265868's host refused
        `start` twice with "Required resources are currently unavailable, state change
        queued", and the answer each time was to search, rank and create by hand. A refusal
        is a fact about someone else's hardware, not about us, and it has exactly one sane
        response.

        Returns {"instance_id", "created": bool, "offer": <offer dict or None>}. On the
        create path the caller must wait for /workspace/PROVISIONED itself - that is a box
        concern, not a client one - and should pass the create time as the budget ledger's
        `--billing-from-epoch`.
        """
        iid = instance_id or self.cached_box_id(state_dir)
        if iid is None:
            raise VastNotArmed(f"no instance id given and no cache at {state_dir}")
        try:
            self.start(iid, dry_run=dry_run)
            return {"instance_id": iid, "created": False, "offer": None}
        except VastStartUnavailable as e:
            self._record("REPLACING_BOX", ["start_or_create", str(iid)],
                         f"host refused: {e.stdout[:120]}")

        offers = self.rank_offers(self.search_offers(OFFER_QUERY, dry_run=dry_run) or [],
                                  max_dph=max_dph)
        if not offers:
            raise VastStartUnavailable(["search", "offers"],
                                       "no offer passes the hard filter; nothing rented")
        best = offers[0]
        res = self.create(best["id"], dry_run=dry_run, label=label,
                          onstart_file=onstart_file)
        new_id = int(res.get("new_contract") or 0)
        if not new_id:
            raise VastStartUnavailable(["create", "instance", str(best["id"])],
                                       f"create returned no contract: {res}")
        self.retire_box_id(iid, f"replaced by {new_id} after its host refused start",
                           state_dir)
        self.set_cached_box_id(new_id, state_dir)
        return {"instance_id": new_id, "created": True, "offer": best}

    # -- renting: offer id in, instance id out -------------------------------------------
    def create(self, offer_id: int, *, dry_run: bool, label: str, onstart_file: str,
               image: str = "vastai/pytorch:cuda-12.8.1-auto", disk_gb: int = 80) -> dict:
        """Rent a machine, through the same `_may_execute` gate as everything else.

        This method exists so that renting is covered by ARMED. Calling `vastai create`
        straight from a shell bypasses the safety entirely - the gate lives here, not in the
        binary - so once a permission rule allows the CLI, every create MUST come through
        here, or "nothing rents without ARMED" is simply untrue.

        `--cancel-unavail` is not optional: without it a failed schedule leaves a stopped
        instance that still bills storage.
        """
        argv = ["create", "instance", str(int(offer_id)),
                "--image", image, "--disk", str(int(disk_gb)),
                "--ssh", "--direct", "--label", label, "--cancel-unavail",
                "--onstart", onstart_file, "--raw"]
        out = self._call(argv, dry_run=dry_run, parse_json=True, check_refusal=True,
                         stub_result={"success": True, "new_contract": 0, "stub": True})
        if isinstance(out, dict) and out.get("error"):
            raise VastStartUnavailable(argv, json.dumps(out))
        # The response also carries an instance_api_key: a credential this pipeline never
        # needs (it authenticates by ssh key and account key). Dropped here rather than
        # handed back to a caller that might log it.
        return {k: v for k, v in (out or {}).items() if k != "instance_api_key"}

    # -- state-changing, id only --------------------------------------------------------
    def start(self, instance_id: int, *, dry_run: bool):
        return self._call(["start", "instance", str(int(instance_id))],
                          dry_run=dry_run, stub_result=f"started {int(instance_id)}",
                          check_refusal=True)

    def stop(self, instance_id: int, *, dry_run: bool):
        return self._call(["stop", "instance", str(int(instance_id))],
                          dry_run=dry_run, stub_result=f"stopped {int(instance_id)}",
                          check_refusal=True)

    def destroy(self, instance_id: int, *, dry_run: bool):
        """Irreversible. `-y` skips the interactive prompt (without it the CLI silently
        aborts under a pipe)."""
        return self._call(["destroy", "instance", str(int(instance_id)), "-y"],
                          dry_run=dry_run, stub_result=f"destroyed {int(instance_id)}")
