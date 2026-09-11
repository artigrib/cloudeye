"""Structured logging setup: every line names its component, scene-related lines carry
a scene_id."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import settings

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] scene=%(scene_id)s %(message)s"

# Maps Python logging levels to syslog severity numbers (emerg=0 .. debug=7), per
# RFC 5424 / journald's SyslogLevelPrefix convention: a leading "<N>" on a line tells
# journald (and syslog) the priority of that line. Without this prefix, journald has
# no way to know a line's severity and stamps the unit's whole stdout/stderr stream at
# one default priority (info) -- which is why `journalctl -p err` was finding nothing
# even though ERROR-level lines were being emitted. Python has no direct equivalent of
# syslog's notice=5, so it's simply skipped here.
_SYSLOG_PRIORITY = {
    logging.CRITICAL: 2,  # crit
    logging.ERROR: 3,  # err
    logging.WARNING: 4,  # warning
    logging.INFO: 6,  # info
    logging.DEBUG: 7,  # debug
}


class _SyslogPrefixFormatter(logging.Formatter):
    """Prepends a syslog priority prefix ("<N>") to each formatted line, based on the
    record's level, so journald (which reads this from the unit's captured
    stdout/stderr) files the line at the right priority instead of defaulting every
    line to "info". Only meant for the handler that journald actually observes
    (`configure_logging`'s StreamHandler) -- not for the plain-text worker.log file
    written by `scene_file_log`, which has no journald reading it."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        priority = _SYSLOG_PRIORITY.get(record.levelno, 6)
        return f"<{priority}>{message}"


class _SceneIdFilter(logging.Filter):
    """Injects a default scene_id="-" onto any record that doesn't already carry one, so
    the shared format string works for every logger, not just ones going through
    `scene_logger`."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "scene_id"):
            record.scene_id = "-"
        return True


def configure_logging(component: str) -> None:
    """Configure root logging for this process ("api" or "worker"). Call once at startup,
    in place of the bare `logging.basicConfig(level=logging.INFO)` used before this existed."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(_SyslogPrefixFormatter(_FORMAT))
    handler.addFilter(_SceneIdFilter())
    root.addHandler(handler)
    logging.getLogger(__name__).info("Logging configured for component=%s", component)


def scene_logger(name: str, scene_id: object) -> logging.LoggerAdapter:
    """A logger adapter that stamps every message with `scene_id`, for the pipeline
    orchestration code where every log line should be traceable to one scene."""
    return logging.LoggerAdapter(logging.getLogger(name), {"scene_id": str(scene_id)})


@contextmanager
def scene_file_log(scene_id: object) -> Iterator[Path]:
    """Attach a temporary FileHandler that captures root-logger output to
    `{scene_dir}/logs/worker.log` for the duration of the `with` block, in addition to
    whatever handlers configure_logging already installed (journald via the systemd unit,
    typically). Yields the log file path."""
    from app.services.scene_service import scene_dir  # local import: avoids a cycle at module load

    log_dir = scene_dir(scene_id) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "worker.log"

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_SceneIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_path
    finally:
        root.removeHandler(handler)
        handler.close()
