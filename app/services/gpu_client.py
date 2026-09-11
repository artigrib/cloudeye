"""SSH-based client for the GPU pipeline: push a job, start it fire-and-forget, poll its
status file, fetch results back, clean up.

Everything goes through a `GpuClient` protocol behind `build_gpu_client()` so a future
Vast-SDK-automated client (spinning instances up/down) can be swapped in without
touching `pipeline_orchestrator.py` - per the explicit "don't automate instance
lifecycle yet, but don't block adding it later" constraint.

Every subprocess call uses `stdin=asyncio.subprocess.DEVNULL` except the deliberate
secret-push, matching the `stdin=subprocess.DEVNULL` pattern already established in
`video_service.probe_video` (the fix for a real ffmpeg-eats-stdin bug found earlier in
this project - the same class of bug can bite any subprocess sharing a pipe).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.config import settings

logger = logging.getLogger(__name__)

# Local repo path for the GPU pipeline source - rsync'd to the remote workspace before
# every job so the GPU always runs whatever's currently in this repo.
LOCAL_PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent / "gpu"

SSH_CONNECT_TIMEOUT = 15
SSH_RETRIES = 3
# OpenSSH's default cipher (chacha20-poly1305@openssh.com on this stack) measured at
# 1.46 MB/s pushing a 168MB video to the current GPU instance; aes128-gcm@openssh.com
# measured at 20.0 MB/s on the identical push, both via rsync - a reproduced 13.7x
# speedup, not host-CPU-bound (compression stayed on in both, CPU never saturated).
# Pinned here rather than left to ~/.ssh/config so a fresh clone (or a different GPU
# provider's host) gets the fast path automatically, without depending on that config
# file existing on whatever machine runs this code.
SSH_CIPHER = "aes128-gcm@openssh.com"
SSH_RETRY_BACKOFF_SEC = 3


class GpuUnavailableError(Exception):
    """SSH could not reach the GPU host at all (exit 255, timeout) after retries."""


class GpuCommandError(Exception):
    """A remote command ran but exited non-zero. Carries the command and stderr tail."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"command exited {returncode}: {' '.join(cmd)}\n{stderr[-2000:]}")


class RemoteJobFailedError(Exception):
    """The pipeline itself reported state=failed in status.json."""

    def __init__(self, status: "GpuJobStatus"):
        self.status = status
        super().__init__(f"pipeline failed at stage {status.stage!r}: {status.error}")


@dataclass(frozen=True)
class GpuJobStatus:
    state: str  # "running" | "done" | "failed" | "unknown" (status.json not written yet)
    stage: str | None
    stage_index: int | None
    n_stages: int | None
    message: str | None
    error: str | None
    updated_at: str | None
    raw: dict = field(default_factory=dict)


class GpuClient(Protocol):
    async def ensure_pipeline_code(self) -> None: ...
    async def push_job(
        self, job_id: str, video_path: Path, params: dict, secrets: dict[str, str]
    ) -> str: ...
    async def start_pipeline(self, job_id: str) -> None: ...
    async def poll_status(self, job_id: str) -> GpuJobStatus: ...
    async def fetch_results(
        self, job_id: str, dest: Path, *, include_per_view: bool = False
    ) -> Path: ...
    async def fetch_logs(self, job_id: str, dest: Path) -> Path: ...
    async def cleanup(self, job_id: str) -> None: ...


class SshGpuClient:
    """The only implementation today: talks to a manually-provisioned, always-on Vast.ai
    instance over SSH, using the `gpu` host alias already configured in `~/.ssh/config`
    when `gpu_ssh_port`/`gpu_ssh_key` are left unset."""

    def __init__(
        self,
        *,
        host: str,
        port: int | None,
        key_path: str | None,
        workspace_dir: str,
        connect_timeout: int = SSH_CONNECT_TIMEOUT,
        retries: int = SSH_RETRIES,
    ):
        self.host = host
        self.port = port
        self.key_path = key_path
        self.workspace_dir = workspace_dir.rstrip("/")
        self.connect_timeout = connect_timeout
        self.retries = retries

    def _job_dir(self, job_id: str) -> str:
        return f"{self.workspace_dir}/data/jobs/{job_id}"

    def _ssh_base_argv(self) -> list[str]:
        argv = [
            "ssh",
            "-c", SSH_CIPHER,
            "-o", "BatchMode=yes",
            f"-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "ServerAliveInterval=30",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if self.port is not None:
            argv += ["-p", str(self.port)]
        if self.key_path:
            argv += ["-i", self.key_path]
        return argv

    def _ssh_argv(self, remote_cmd: str) -> list[str]:
        return [*self._ssh_base_argv(), self.host, remote_cmd]

    def _ssh_argv_detached(self, remote_cmd: str) -> list[str]:
        """`-f`: request the SSH CLIENT itself fork to background right before running
        `remote_cmd`, then return control immediately - OpenSSH's own purpose-built
        flag for "launch a remote background job and detach", used here instead of a
        shell-level `cmd & disown` trick. (The actual hang this project hit and fixed
        turned out to be in `_run`'s use of `asyncio.subprocess.PIPE` + `communicate()`
        - see `_run`'s docstring - not in the backgrounding pattern itself; `-f` is kept
        regardless as the more robust, standard tool for this specific job.) `-f`
        implies stdin is /dev/null."""
        return [*self._ssh_base_argv(), "-f", self.host, remote_cmd]

    def _rsync_ssh_flag(self) -> str:
        # Mirrors _ssh_base_argv's StrictHostKeyChecking=accept-new: without it, the
        # first rsync to a host whose key isn't in known_hosts yet falls back to the
        # ssh client default (StrictHostKeyChecking=ask), which fails outright under
        # BatchMode=yes - a real risk for a freshly (re)provisioned GPU instance whose
        # host key has never been seen before.
        parts = [
            "ssh", "-c", SSH_CIPHER,
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if self.port is not None:
            parts += ["-p", str(self.port)]
        if self.key_path:
            parts += ["-i", self.key_path]
        return " ".join(parts)

    async def _run(
        self, argv: list[str], *, timeout: float, stdin_bytes: bytes | None = None
    ) -> str:
        """Run one subprocess with retries on transient SSH unavailability (exit 255).
        A real command failure (non-255, e.g. a remote `mkdir` failing) is NOT retried -
        only "couldn't even connect" is, so a 30-second network blip doesn't kill a
        6-minute job, but a genuine remote error surfaces immediately.

        stdout/stderr are captured via temp FILES, not `asyncio.subprocess.PIPE` +
        `communicate()`. This is a deliberate workaround for a real, reproduced bug:
        after a few `asyncio.create_subprocess_exec` calls have already run in this
        event loop (e.g. the rsyncs in `ensure_pipeline_code`/`push_job`), a later
        call's `proc.communicate()` reliably hangs forever reading its stdout pipe -
        even though the underlying command (verified directly, both from a plain shell
        and from an isolated asyncio script) completes in ~1 second. File-based
        redirection sidesteps asyncio's pipe-transport EOF handling entirely, which is
        exactly where the hang was isolated to.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *argv,
                        stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
                        stdout=out_f,
                        stderr=err_f,
                    )
                    if stdin_bytes is not None:
                        proc.stdin.write(stdin_bytes)
                        await proc.stdin.drain()
                        proc.stdin.write_eof()
                    await asyncio.wait_for(proc.wait(), timeout=timeout)

                    out_f.seek(0)
                    err_f.seek(0)
                    stdout = out_f.read()
                    stderr = err_f.read()

                    if proc.returncode == 255:
                        raise GpuUnavailableError(
                            f"ssh exit 255 (connection failed): {stderr.decode(errors='replace')[:500]}"
                        )
                    if proc.returncode != 0:
                        raise GpuCommandError(argv, proc.returncode, stderr.decode(errors="replace"))
                    return stdout.decode(errors="replace")
                except (GpuUnavailableError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                if attempt < self.retries:
                    logger.warning(
                        "GPU command attempt %d/%d failed (%s), retrying in %ds",
                        attempt, self.retries, last_exc, SSH_RETRY_BACKOFF_SEC,
                    )
                    await asyncio.sleep(SSH_RETRY_BACKOFF_SEC)
        raise GpuUnavailableError(f"GPU unreachable after {self.retries} attempts") from last_exc

    async def ensure_pipeline_code(self) -> None:
        """Sync the local gpu/ pipeline source to the remote workspace. Cheap (rsync
        deltas only) and idempotent - safe to call before every job so the GPU always
        runs whatever's currently in this repo, no separate deploy step to forget."""
        await self._run(
            self._ssh_argv(f"mkdir -p {self.workspace_dir}/pipeline {self.workspace_dir}/data/jobs"),
            timeout=30,
        )
        rsync_argv = [
            "rsync", "-az", "--delete",
            "--exclude", "_reference", "--exclude", "__pycache__",
            "-e", self._rsync_ssh_flag(),
            f"{LOCAL_PIPELINE_DIR}/",
            f"{self.host}:{self.workspace_dir}/pipeline/",
        ]
        await self._run(rsync_argv, timeout=60)
        logger.info("GPU pipeline code synced to %s:%s/pipeline", self.host, self.workspace_dir)

    async def push_job(
        self, job_id: str, video_path: Path, params: dict, secrets: dict[str, str]
    ) -> str:
        job_dir = self._job_dir(job_id)
        await self._run(self._ssh_argv(f"mkdir -p {job_dir}"), timeout=30)

        rsync_argv = [
            "rsync", "-az",
            "-e", self._rsync_ssh_flag(),
            str(video_path),
            f"{self.host}:{job_dir}/input.mp4",
        ]
        await self._run(rsync_argv, timeout=300)

        params_json = json.dumps(params).encode()
        await self._run(
            self._ssh_argv(f"cat > {job_dir}/params.json"), timeout=15, stdin_bytes=params_json
        )

        if secrets:
            # Mode-600, job-local only - never in argv/process list, deleted by
            # run_pipeline.sh on successful completion (and by our own cleanup() on
            # failure/deletion regardless).
            env_body = "\n".join(f"{k}={v}" for k, v in secrets.items()).encode()
            await self._run(
                self._ssh_argv(f"umask 077 && cat > {job_dir}/.env"), timeout=15, stdin_bytes=env_body
            )

        logger.info("Pushed job %s (video=%s)", job_id, video_path.name)
        return job_dir

    async def start_pipeline(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        # Fire-and-forget: setsid detaches the remote process from the SSH session, and
        # `-f` (see _ssh_argv_detached) detaches the LOCAL ssh client itself - together
        # these mean the pipeline keeps running for minutes with nothing holding this
        # connection open, and this call returns in ~1s regardless of how long the
        # pipeline itself takes.
        cmd = (
            f"cd {job_dir} && setsid bash {self.workspace_dir}/pipeline/run_pipeline.sh {job_dir} "
            f"> {job_dir}/logs_run_pipeline.log 2>&1 < /dev/null"
        )
        await self._run(self._ssh_argv_detached(cmd), timeout=30)
        logger.info("Started pipeline for job %s", job_id)

    async def poll_status(self, job_id: str) -> GpuJobStatus:
        job_dir = self._job_dir(job_id)
        try:
            raw_text = await self._run(
                self._ssh_argv(f"cat {job_dir}/status.json 2>/dev/null || echo '{{}}'"), timeout=20
            )
        except GpuCommandError as exc:
            raise GpuUnavailableError(f"could not read status.json for {job_id}") from exc

        try:
            raw = json.loads(raw_text) if raw_text.strip() else {}
        except json.JSONDecodeError:
            # status.json is written atomically (tmp + os.replace) specifically to avoid
            # ever observing a half-written file - a parse failure here means something
            # else is wrong, but don't crash the poll loop over one bad read, treat as
            # "not ready yet" and let the next poll retry.
            logger.warning("Could not parse status.json for job %s, treating as unknown", job_id)
            raw = {}

        return GpuJobStatus(
            state=raw.get("state", "unknown"),
            stage=raw.get("stage"),
            stage_index=raw.get("stage_index"),
            n_stages=raw.get("n_stages"),
            message=raw.get("message"),
            error=raw.get("error"),
            updated_at=raw.get("updated_at"),
            raw=raw,
        )

    async def fetch_results(
        self, job_id: str, dest: Path, *, include_per_view: bool = False
    ) -> Path:
        """Fetch the remote job dir into `dest` (the scene's local, persistent copy -
        the remote copy is deleted regardless once `cleanup()` runs in the
        orchestrator's `finally` block, unaffected by this flag).

        `include_per_view` (default False, matching `settings.retain_per_view_artifacts`)
        stops excluding `per_view/` (per-frame .npz: pts3d/mask/img_no_norm/camera_pose/
        intrinsics) and `per_view_png/` (per-frame preview images) from the rsync. Off
        by default - these are real disk cost (up to max_keyframes=200 frames of PNG+npz
        each) that most callers don't need; opt in only for scenes where a downstream
        consumer (multi-view TSDF fusion, 3DGS) actually needs per-frame pose/intrinsics
        or raw imagery. `keyframes/`, `input.mp4`, and `.env` stay excluded either way -
        out of scope for this flag.
        """
        job_dir = self._job_dir(job_id)
        dest.mkdir(parents=True, exist_ok=True)
        rsync_argv = ["rsync", "-az"]
        if not include_per_view:
            rsync_argv += ["--exclude", "per_view/", "--exclude", "per_view_png/"]
        rsync_argv += [
            "--exclude", "keyframes/", "--exclude", "input.mp4", "--exclude", ".env",
            "-e", self._rsync_ssh_flag(),
            f"{self.host}:{job_dir}/",
            f"{dest}/",
        ]
        await self._run(rsync_argv, timeout=300)
        logger.info(
            "Fetched results for job %s into %s (include_per_view=%s)",
            job_id, dest, include_per_view,
        )
        return dest

    async def fetch_logs(self, job_id: str, dest: Path) -> Path:
        job_dir = self._job_dir(job_id)
        dest.mkdir(parents=True, exist_ok=True)
        rsync_argv = [
            "rsync", "-az",
            "-e", self._rsync_ssh_flag(),
            f"{self.host}:{job_dir}/logs/",
            f"{dest}/",
        ]
        try:
            await self._run(rsync_argv, timeout=60)
        except (GpuCommandError, GpuUnavailableError) as exc:
            # Best-effort: called from failure-handling paths, must never itself raise
            # and mask the real error.
            logger.warning("Could not fetch logs for job %s: %s", job_id, exc)
        return dest

    async def cleanup(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        try:
            await self._run(self._ssh_argv(f"rm -rf {job_dir}"), timeout=30)
            logger.info("Cleaned up remote job dir for %s", job_id)
        except (GpuCommandError, GpuUnavailableError) as exc:
            logger.warning("Could not clean up job dir for %s: %s", job_id, exc)


def build_gpu_client() -> GpuClient:
    """The single factory for the active GpuClient implementation. A future
    Vast-SDK-automated client (create/destroy instances) swaps in here without
    `pipeline_orchestrator.py` changing at all."""
    return SshGpuClient(
        host=settings.gpu_ssh_host,
        port=settings.gpu_ssh_port,
        key_path=settings.gpu_ssh_key,
        workspace_dir=settings.gpu_workspace_dir,
    )
