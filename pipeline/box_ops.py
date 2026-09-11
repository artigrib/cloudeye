#!/usr/bin/env python3
"""Bringing a vast.ai box up, and everything that must happen before it is trusted.

`BoxSession` is a context manager: entering it starts the instance and does not return until
sshd actually answers; leaving it stops the instance **by id**, on success and on failure
alike. Each box gets its own session, so two boxes mean two independent `finally`s, not one
shared cleanup that a failure in the first can skip.

The bring-up sequence, in order, each step measured rather than assumed:

1. `start instance <id>` (id only).
2. **Wait for ssh by connecting**, never by sleeping: poll `show instance --raw` every
   `poll_s` for `ssh_host`/`ssh_port` and `public_ipaddr`/`direct_port_start`, and try a real
   `ssh … 'echo SSH_OK'` against each pair as soon as it appears. Deadline `deadline_s`
   (default 300 s). Same shape as `nvblox_v2/wait3.sh`, which is where this pattern was
   proven; that script is read-only from here.
3. **Fix the ssh key modes over ssh, immediately after the first successful connect.**
   `vastai/pytorch:cuda-12.8.1-auto` writes `$HOME/.ssh/authorized_keys` with ownership or
   modes sshd rejects, and instance 50254121 was lost to exactly that. It cannot be fixed
   from outside: `vastai execute` runs only on stopped instances and does not whitelist
   chmod, `start instance` accepts no `--onstart-cmd` (that flag exists only on `create
   instance`), and `update instance --onstart` *recreates* the instance from a template,
   which on the nvblox box would destroy the install. So the fix is applied through the
   connection we just proved works, and `--onstart-cmd` (ONSTART_CMD below) is what a
   `create` would carry.
4. **Measure the uplink for real**: scp a `probe_mb` file and report MB/s. A listing claiming
   736 Mbit/s delivered ~10.5 MB/s; another box gave 76 KB/s. The measured number decides
   whether the run is worth continuing, so it is measured, not read off the offer.
5. `nvidia-smi` for the name, total VRAM and driver.

Everything measured lands in `status.json` under `boxes[<id>]`.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import time
from pathlib import Path

# What a `create instance --onstart-cmd` would carry. Kept next to the ssh fallback so the
# two can never drift apart.
ONSTART_CMD = (
    "for i in $(seq 1 30); do "
    "chown -R root:root $HOME/.ssh 2>/dev/null; "
    "chmod 700 $HOME/.ssh 2>/dev/null; "
    "chmod 600 $HOME/.ssh/authorized_keys 2>/dev/null; "
    "sleep 10; done &"
)

SSH_FIX_CMD = ("chown -R root:root $HOME/.ssh 2>/dev/null || true; "
               "chmod 700 $HOME/.ssh 2>/dev/null || true; "
               "chmod 600 $HOME/.ssh/authorized_keys 2>/dev/null || true; "
               "ls -ld $HOME/.ssh; ls -l $HOME/.ssh/authorized_keys")

# Ceiling on the resized link probe. Beyond this the probe itself becomes the slow part of
# the bring-up, which is the problem it was meant to prevent.
PROBE_MAX_MB = 200

NVIDIA_SMI_CMD = ("nvidia-smi --query-gpu=name,memory.total,driver_version "
                  "--format=csv,noheader")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class BoxError(RuntimeError):
    pass


class BoxSession:
    def __init__(self, client, instance_id: int, *, dry_run: bool, ssh_key: str,
                 name: str = "box", deadline_s: int = 300, poll_s: int = 15,
                 probe_mb: int = 50, probe_min_s: float = 10.0,
                 scratch: Path | None = None, log=print, skip_start: bool = False):
        self.client = client
        self.instance_id = int(instance_id)
        # The caller already issued the start (worker mode goes through
        # VastClient.start_or_create, which starts the cached box or rents a replacement).
        # Issuing a second start against a box that is mid-state-change is how a healthy
        # bring-up gets reported as a refusal, so it is skipped rather than repeated.
        self.skip_start = bool(skip_start)
        self.dry_run = dry_run
        self.ssh_key = ssh_key
        self.name = name
        self.deadline_s = deadline_s
        self.poll_s = poll_s
        # 20 MB under-reads by about 2x: measured 2026-09-09, the 20 MB probe said
        # 2.49 MB/s up while the real 209.6 MB push ran at 6.20, and an external 50 MB
        # probe said 4.38. The transfer is too short for TCP to reach its stride. The probe
        # now grows until it has moved at least `probe_mb` MB AND taken `probe_min_s`
        # seconds, whichever demands more.
        self.probe_mb = probe_mb
        self.probe_min_s = probe_min_s
        self._probe_retried = False
        self.scratch = Path(scratch) if scratch else Path("/tmp")
        self.log = log
        self.host: str | None = None
        self.port: int | None = None
        self.info: dict = {"instance_id": self.instance_id, "name": name}
        # Billing starts at create/start, not when this session's own start call returns.
        # `billing_from` may be set by the caller to the moment the box began costing money
        # - after a `create`, that is minutes before this session ever touches it, and those
        # minutes are provisioning and link probes that the run genuinely caused.
        self.up_at: float | None = None
        self.billing_from: float | None = None
        self.stopped_at_monotonic: float | None = None

    # -- ssh plumbing --------------------------------------------------------------------
    def _ssh_argv(self, host: str, port: int, remote_cmd: str) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=30",
                "-i", self.ssh_key, "-p", str(port), f"root@{host}", remote_cmd]

    def ssh(self, remote_cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
        if self.dry_run:
            return subprocess.CompletedProcess([], 0, "DRY_RUN", "")
        if not self.host:
            raise BoxError(f"{self.name}: no ssh endpoint yet")
        return subprocess.run(self._ssh_argv(self.host, self.port, remote_cmd),
                              capture_output=True, text=True, timeout=timeout)

    def push(self, local: Path, remote: str, timeout: int = 3600) -> subprocess.CompletedProcess:
        """rsync up, always with --partial: an interrupted rsync without it deletes what it
        already transferred (an interrupted 280 MB pull once left 11 MB)."""
        return self._rsync([str(local)], f"root@{self.host}:{remote}", timeout)

    def pull(self, remote: str, local: Path, timeout: int = 3600,
             include: list[str] | None = None) -> subprocess.CompletedProcess:
        """rsync down. `include` names exactly the files to fetch; everything else is
        excluded. Pulling a whole output directory is how the 2026-09-09 run spent 483 s of
        its 567 s NVBLOX step dragging a 133.7 MB map.nvblx that no downstream stage reads.
        """
        Path(local).mkdir(parents=True, exist_ok=True)
        return self._rsync([f"root@{self.host}:{remote}"], str(local), timeout,
                           include=include)

    def _rsync(self, src: list[str], dst: str, timeout: int,
               include: list[str] | None = None) -> subprocess.CompletedProcess:
        if self.dry_run:
            return subprocess.CompletedProcess([], 0, "DRY_RUN", "")
        ssh_flag = (f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                    f"-o ConnectTimeout=8 -i {self.ssh_key} -p {self.port}")
        filt: list[str] = []
        if include:
            for f in include:
                filt += ["--include", f]
            filt += ["--exclude", "*"]
        argv = ["rsync", "-a", "--partial", "--info=stats2", *filt, "-e", ssh_flag, *src, dst]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    def remote_sizes(self, remote_dir: str, names: list[str]) -> dict:
        """Byte size of each named file on the box, 0 when absent. Used to predict transfer
        time before committing to it."""
        if self.dry_run:
            return {n: 0 for n in names}
        cmd = "; ".join(f"stat -c%s {remote_dir}/{n} 2>/dev/null || echo 0" for n in names)
        r = self.ssh(cmd, timeout=60)
        out = (r.stdout or "").split()
        return {n: int(out[i]) if i < len(out) and out[i].isdigit() else 0
                for i, n in enumerate(names)}

    # -- bring-up ------------------------------------------------------------------------
    def _endpoints(self, d: dict) -> list[tuple[str, int]]:
        out = []
        for host, port in ((d.get("ssh_host"), d.get("ssh_port")),
                           (d.get("public_ipaddr"), d.get("direct_port_start"))):
            if host and port and str(port) not in ("None", "-1"):
                out.append((str(host), int(port)))
        return out

    def wait_for_ssh(self) -> tuple[str, int]:
        deadline = time.time() + self.deadline_s
        last = "no status yet"
        attempts = 0
        while time.time() < deadline:
            d = self.client.show(self.instance_id, dry_run=self.dry_run)
            self.info["actual_status"] = d.get("actual_status")
            self.info["dph_total"] = d.get("dph_total")
            self.info["gpu_name"] = d.get("gpu_name")
            if self.dry_run:
                self.info["ssh_probe"] = "not attempted (dry run)"
                return "dry-run-host", 22
            for host, port in self._endpoints(d):
                attempts += 1
                p = subprocess.run(self._ssh_argv(host, port, "echo SSH_OK"),
                                   capture_output=True, text=True, timeout=30)
                if p.returncode == 0 and "SSH_OK" in p.stdout:
                    self.log(f"[{now()}] {self.name}: ssh up at {host}:{port} "
                             f"after {attempts} attempt(s)")
                    return host, port
                last = f"{host}:{port} rc={p.returncode} {p.stderr.strip()[:120]}"
            self.log(f"[{now()}] {self.name}: status={self.info.get('actual_status')} "
                     f"not reachable yet ({last})")
            time.sleep(self.poll_s)
        raise BoxError(f"{self.name}: ssh not reachable within {self.deadline_s}s "
                       f"(instance {self.instance_id}); last: {last}")

    def fix_ssh_key_modes(self) -> None:
        p = self.ssh(SSH_FIX_CMD, timeout=60)
        self.info["ssh_key_modes_fixed"] = (p.returncode == 0)
        self.info["ssh_key_modes_out"] = (p.stdout or "").strip()[:300]
        if p.returncode != 0:
            raise BoxError(f"{self.name}: could not fix ssh key modes: "
                           f"rc={p.returncode} {p.stderr.strip()[:200]}")

    def clear_probe_scratch(self) -> None:
        """Delete the local uplink probe once both directions are measured.

        measure_inet_down already unlinks what it pulled; the file measure_inet_up PUSHED
        was never removed, so every scene dir kept it - and it is not small. The resize
        path makes it up to PROBE_MAX_MB, and on the offline suite (a 1 MB probe against a
        fake scp that sleeps 0.2 s, so the duration floor forces one resize) it landed at
        50-55 MB per box brought up: 160 MB of the suite's 460 MB peak, for a file whose
        only purpose was to be timed.
        """
        for f in self.scratch.glob("_inet_probe_*.bin"):
            try:
                f.unlink()
            except OSError as e:      # a probe we cannot delete is not a run failure
                self.log(f"[{now()}] {self.name}: could not remove {f}: {e}")

    def measure_inet_down(self, remote_probe: str = "/tmp/_inet_probe.bin") -> float | None:
        """scp the probe file back DOWN and report MB/s.

        Measured separately from the uplink because the two are not symmetric on these
        boxes and it is the DOWNLINK that dominates: on 2026-09-09 the same box gave
        0.42 MB/s up and 0.32 MB/s down, and the pull was 85% of the NVBLOX step.
        """
        if self.dry_run:
            self.info["inet_down_mb_s"] = None
            return None
        dest = self.scratch / "_inet_probe_down.bin"
        dest.parent.mkdir(parents=True, exist_ok=True)
        argv = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=8", "-i", self.ssh_key, "-P", str(self.port),
                f"root@{self.host}:{remote_probe}", str(dest)]
        t0 = time.time()
        p = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        wall = time.time() - t0
        if p.returncode != 0:
            raise BoxError(f"{self.name}: inet_down probe failed rc={p.returncode}: "
                           f"{p.stderr.strip()[:200]}")
        mb = dest.stat().st_size / 1e6
        dest.unlink(missing_ok=True)
        mb_s = mb / wall if wall > 0 else 0.0
        self.info["inet_down_mb_s"] = round(mb_s, 3)
        self.info["inet_down_wall_s"] = round(wall, 2)
        self.log(f"[{now()}] {self.name}: inet_down {mb_s:.2f} MB/s "
                 f"({mb:.0f} MB in {wall:.1f}s)")
        return mb_s

    def measure_inet_up(self) -> float | None:
        """scp a probe file up and report MB/s. Returns None only in dry run."""
        if self.dry_run:
            self.info["inet_up_mb_s"] = None
            self.info["inet_up_note"] = "not measured (dry run)"
            return None
        probe = self.scratch / f"_inet_probe_{self.probe_mb}mb.bin"
        probe.parent.mkdir(parents=True, exist_ok=True)
        if not probe.exists() or probe.stat().st_size != self.probe_mb * 1024 * 1024:
            with probe.open("wb") as fh, open("/dev/urandom", "rb") as rnd:
                fh.write(rnd.read(self.probe_mb * 1024 * 1024))
        argv = ["scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=8", "-i", self.ssh_key, "-P", str(self.port),
                str(probe), f"root@{self.host}:/tmp/_inet_probe.bin"]
        t0 = time.time()
        p = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        wall = time.time() - t0
        if p.returncode != 0:
            raise BoxError(f"{self.name}: inet_up probe failed rc={p.returncode}: "
                           f"{p.stderr.strip()[:200]}")
        mb_s = self.probe_mb / wall if wall > 0 else 0.0
        if wall < self.probe_min_s and mb_s > 0 and not self._probe_retried:
            # Too quick to be trusted - resize so the next attempt spans probe_min_s at the
            # rate just seen, and measure once more. Exactly ONE retry, and the size is
            # capped: a link fast enough to finish even the resized probe early is fast
            # enough that the estimate cannot be the thing that fails a run.
            bigger = min(PROBE_MAX_MB, max(self.probe_mb,
                                           int(mb_s * self.probe_min_s * 1.2)))
            if bigger > self.probe_mb:
                self._probe_retried = True
                self.log(f"[{now()}] {self.name}: probe took {wall:.1f}s "
                         f"(< {self.probe_min_s}s) at {mb_s:.2f} MB/s - one repeat at "
                         f"{bigger} MB")
                self.probe_mb = bigger
                self.info["inet_up_probe_resized_to_mb"] = bigger
                return self.measure_inet_up()
        # The probe stays on the box; measure_inet_down pulls this same file back, then
        # __enter__ removes it. Uploading a second file just to measure the other direction
        # would double the slowest part of the bring-up.
        self.info["inet_up_mb_s"] = round(mb_s, 3)
        self.info["inet_up_probe_mb"] = self.probe_mb
        self.info["inet_up_wall_s"] = round(wall, 2)
        self.log(f"[{now()}] {self.name}: inet_up {mb_s:.2f} MB/s "
                 f"({self.probe_mb} MB in {wall:.1f}s)")
        return mb_s

    def read_gpu(self) -> dict:
        p = self.ssh(NVIDIA_SMI_CMD, timeout=60)
        if self.dry_run:
            self.info["gpu"] = {"note": "not queried (dry run)"}
            return self.info["gpu"]
        if p.returncode != 0:
            raise BoxError(f"{self.name}: nvidia-smi failed rc={p.returncode}: "
                           f"{p.stderr.strip()[:200]}")
        line = (p.stdout or "").strip().splitlines()[0] if p.stdout.strip() else ""
        parts = [x.strip() for x in line.split(",")]
        gpu = {"raw": line}
        if len(parts) >= 3:
            gpu["name"], gpu["memory_total"], gpu["driver_version"] = parts[0], parts[1], parts[2]
            try:
                gpu["vram_total_mib"] = int(parts[1].split()[0])
            except (ValueError, IndexError):
                gpu["vram_total_mib"] = None
        self.info["gpu"] = gpu
        self.log(f"[{now()}] {self.name}: gpu {line}")
        return gpu

    # -- context manager -----------------------------------------------------------------
    def __enter__(self) -> "BoxSession":
        # The start call is INSIDE the guarded block on purpose. vast can decline a start in
        # stdout while exiting 0 ("Required resources are currently unavailable, state change
        # queued") - the request is then sitting in a queue that must be cancelled, so the
        # stop below has to run even though the box never came up.
        try:
            if not self.skip_start:
                self.client.start(self.instance_id, dry_run=self.dry_run)
            self.up_at = time.time()
            if self.billing_from is None:
                self.billing_from = self.up_at
            self.info["up_at"] = now()
            self.info["billing_from"] = dt.datetime.fromtimestamp(
                self.billing_from).astimezone().isoformat(timespec="seconds")
            t = time.time()
            self.host, self.port = self.wait_for_ssh()
            self.info["ssh_wait_s"] = round(time.time() - t, 2)
            self.info["ssh_host"], self.info["ssh_port"] = self.host, self.port
            t = time.time()
            self.fix_ssh_key_modes()
            self.info["chmod_s"] = round(time.time() - t, 2)
            t = time.time()
            self.measure_inet_up()
            self.measure_inet_down()
            self.ssh("rm -f /tmp/_inet_probe.bin", timeout=30)
            self.clear_probe_scratch()
            self.info["inet_probe_s"] = round(time.time() - t, 2)
            t = time.time()
            self.read_gpu()
            self.info["nvidia_smi_s"] = round(time.time() - t, 2)
        except Exception:
            # __exit__ still runs for anything raised inside the `with` body, but a failure
            # *here* is before the body ever starts, so stop the box explicitly.
            self.stop()
            raise
        return self

    def stop(self, confirm_deadline_s: int = 120, poll_s: int = 10) -> None:
        """Stop by id, then WAIT for the instance to actually report `exited`.

        Billing does not end when the stop call returns. On the 2026-09-09 live run our
        ledger measured 612 s between the `start` call and the `stop` call ($0.0627) while
        vast's own `uptime_mins` for the same box moved 1237.3 -> 1249.6, i.e. 12.3 minutes
        (~$0.0756): about 20% more than we counted. So the clock we bill against runs from
        the start call to a CONFIRMED `exited`, not to the stop call.

        A stop that cannot be confirmed is recorded as such and the elapsed clock keeps
        running - reporting an unconfirmed stop as free would be the wrong direction to be
        wrong in.
        """
        try:
            self.client.stop(self.instance_id, dry_run=self.dry_run)
            self.info["stopped_at"] = now()
        except Exception as e:  # cleanup must never mask the original error
            self.info["stop_error"] = f"{type(e).__name__}: {e}"
            self.log(f"[{now()}] {self.name}: STOP FAILED: {e}")
            return
        if self.dry_run:
            self.stopped_at_monotonic = time.time()
            self.info["exited_confirmed"] = True
            return
        deadline = time.time() + confirm_deadline_s
        while time.time() < deadline:
            try:
                d = self.client.show(self.instance_id, dry_run=False)
            except Exception as e:
                self.info["confirm_show_error"] = f"{type(e).__name__}: {e}"
                time.sleep(poll_s)
                continue
            if d.get("actual_status") == "exited":
                self.stopped_at_monotonic = time.time()
                self.info["exited_confirmed"] = True
                self.info["exited_confirmed_at"] = now()
                self.log(f"[{now()}] {self.name}: exited confirmed")
                return
            time.sleep(poll_s)
        # Not confirmed: leave stopped_at_monotonic unset so the cost keeps accruing.
        self.info["exited_confirmed"] = False
        self.info["exited_confirm_note"] = (
            f"still not `exited` {confirm_deadline_s}s after the stop call; the cost clock "
            "keeps running rather than being cut short")
        self.log(f"[{now()}] {self.name}: WARNING exit not confirmed in "
                 f"{confirm_deadline_s}s")

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False


def required_vram_mib(n_views: int) -> float:
    """Measured rule over 22 plain MapAnything passes: allocated_MiB = 4704.7 + 67.0*n.
    `nvidia-smi` reads roughly 2853 MiB above this."""
    return 4704.7 + 67.0 * n_views


def select_gpu(n_views: int) -> dict:
    """REPORT ONLY - returns a recommendation and the search query that would confirm it.
    Nothing acts on this: renting a machine is a human decision (see vast_client's
    `search_offers` note). For nvblox the box is fixed and no search happens at all."""
    need_mib = required_vram_mib(n_views)
    need_gb = need_mib / 1024.0
    if need_gb <= 22:
        rec = "RTX 4090 (24 GB)"
    elif need_gb <= 44:
        rec = "A6000 (48 GB)"
    elif need_gb <= 76:
        rec = "H100 (80 GB)"
    else:
        rec = "B200 (180 GB)"
    return {
        "n_views": n_views,
        "required_vram_mib": round(need_mib, 1),
        "required_vram_gb": round(need_gb, 2),
        "recommended": rec,
        "search_query_for_report": (
            f"gpu_ram>={int(need_gb) + 2} num_gpus=1 rentable=true "
            "disk_space>=120 inet_up>=500"),
        "note": "recommendation only; the operator picks and rents the machine by hand",
    }
