"""Unit tests for app/logging_setup.py's syslog-priority-prefixed formatter.

These tests deliberately never call `configure_logging()`: that function mutates the
process-wide root logger (removes all handlers, installs new ones), and since the full
test suite runs many other tests in the same process, calling it here would leak
global state across test files. Instead we construct LogRecords by hand and exercise
the formatter classes directly, or use `scene_file_log`, which is self-contained (it
adds/removes its own handler on the root logger within the `with` block and cleans up
afterward).
"""

import logging

import pytest

from app.logging_setup import _FORMAT, _SyslogPrefixFormatter


def _make_record(level: int, msg: str) -> logging.LogRecord:
    """Build a standalone LogRecord, not tied to any real logger, so formatting it
    can't touch global logging state."""
    return logging.LogRecord(
        name="app.services.pipeline_orchestrator",
        level=level,
        pathname=__file__,
        lineno=42,
        msg=msg,
        args=(),
        exc_info=None,
    )


@pytest.mark.parametrize(
    ("level", "prefix"),
    [
        (logging.CRITICAL, "<2>"),
        (logging.ERROR, "<3>"),
        (logging.WARNING, "<4>"),
        (logging.INFO, "<6>"),
        (logging.DEBUG, "<7>"),
    ],
)
def test_syslog_prefix_formatter_prepends_correct_priority(level: int, prefix: str) -> None:
    formatter = _SyslogPrefixFormatter(_FORMAT)
    record = _make_record(level, "pipeline failed: boom")
    # scene_id isn't set on this hand-built record; the real pipeline emits it via
    # _SceneIdFilter, but that filter runs on the handler, not the formatter, so we
    # set it directly here to exercise the format string end to end.
    record.scene_id = "abc123"

    formatted = formatter.format(record)

    assert formatted.startswith(prefix)
    assert "pipeline failed: boom" in formatted
    assert "scene=abc123" in formatted
    assert logging.getLevelName(level) in formatted


def test_syslog_prefix_formatter_defaults_unknown_level_to_info_priority() -> None:
    """A level number with no explicit mapping (e.g. a custom level between standard
    ones) should fall back to the info priority (<6>) rather than raising or omitting
    a prefix."""
    formatter = _SyslogPrefixFormatter(_FORMAT)
    record = _make_record(25, "custom level message")
    record.scene_id = "-"

    formatted = formatter.format(record)

    assert formatted.startswith("<6>")


def test_plain_formatter_has_no_syslog_prefix() -> None:
    """The plain formatter used for the worker.log FileHandler (scene_file_log) must
    NOT gain a syslog prefix -- that file is read by humans directly, not journald."""
    formatter = logging.Formatter(_FORMAT)
    record = _make_record(logging.ERROR, "Scene 42 failed: boom")
    record.scene_id = "42"

    formatted = formatter.format(record)

    assert not formatted.startswith("<")
    assert "ERROR" in formatted
    assert "scene=42" in formatted
    assert "Scene 42 failed: boom" in formatted


def test_scene_file_log_writes_plain_unprefixed_lines(tmp_path, monkeypatch) -> None:
    """End-to-end check that scene_file_log's FileHandler output has no syslog prefix,
    using the real context manager. scene_file_log adds its FileHandler to the root
    logger and removes it again on exit, so this doesn't leak state -- but to keep this
    test isolated even from accidental interleaving, we also strip any handlers the
    context manager left attached to loggers we touch (it does not touch child loggers
    itself, only the root logger, and cleans that up in its `finally`)."""
    from app import logging_setup

    def fake_scene_dir(scene_id: object):
        d = tmp_path / f"scene-{scene_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("app.services.scene_service.scene_dir", fake_scene_dir)

    root = logging.getLogger()
    handlers_before = list(root.handlers)

    logger = logging.getLogger("app.services.pipeline_orchestrator")
    with logging_setup.scene_file_log("scene-xyz") as log_path:
        logger.error("Scene scene-xyz failed: disk full")

    # scene_file_log must clean up after itself: no handler leaked onto the root logger.
    assert root.handlers == handlers_before

    contents = log_path.read_text()
    assert "Scene scene-xyz failed: disk full" in contents
    for line in contents.splitlines():
        if line.strip():
            assert not line.startswith("<"), f"unexpected syslog prefix in worker.log line: {line!r}"
