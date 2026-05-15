"""Unit tests for Bundle 5 (2026-05-14) RotatingFileHandler logging sink.

Validates that ``pancakebot.log.configure_file_logging`` attaches a
``RotatingFileHandler`` to the namespaced ``pancakebot`` logger and that
the dual-write hook in ``_emit`` mirrors every structured log line into
the file. Stdout writer is preserved (verified indirectly: existing
log-formatting tests cover the stdout-emit contract).

Behaviors covered:
- File handler creates the target file under the requested path.
- Subsequent ``info()`` / ``warn()`` / ``error()`` calls land in the file.
- Idempotent: configure_file_logging called twice with the same path is
  a no-op (single handler, no duplicate writes).
- Re-configuring with a different path detaches the old handler and
  attaches the new one (no dual file outputs).
- Backup count cap is the requested 7.
- maxBytes is the requested 25 MiB.
- Format string carries time + level + logger name + message.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.log import (  # noqa: E402
    _FILE_LOGGER,
    configure_file_logging,
    error,
    info,
    warn,
)


@pytest.fixture(autouse=True)
def _reset_pancakebot_logger():
    """Detach any handlers attached by previous tests; restore after."""
    saved_handlers = list(_FILE_LOGGER.handlers)
    for h in saved_handlers:
        _FILE_LOGGER.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    yield
    for h in list(_FILE_LOGGER.handlers):
        _FILE_LOGGER.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    for h in saved_handlers:
        _FILE_LOGGER.addHandler(h)


def test_configure_file_logging_creates_handler(tmp_path):
    log_path = tmp_path / "runtime.log"
    handler = configure_file_logging(str(log_path))
    assert isinstance(handler, RotatingFileHandler)
    assert os.path.abspath(handler.baseFilename) == os.path.abspath(str(log_path))
    assert handler in _FILE_LOGGER.handlers


def test_configure_file_logging_25mb_threshold(tmp_path):
    handler = configure_file_logging(str(tmp_path / "runtime.log"))
    assert handler.maxBytes == 25 * 1024 * 1024


def test_configure_file_logging_seven_backups(tmp_path):
    handler = configure_file_logging(str(tmp_path / "runtime.log"))
    assert handler.backupCount == 7


def test_configure_file_logging_creates_parent_directory(tmp_path):
    nested = tmp_path / "deep" / "nested" / "subdir"
    log_path = nested / "runtime.log"
    assert not nested.exists()
    configure_file_logging(str(log_path))
    assert nested.exists()


def test_info_call_writes_to_file(tmp_path):
    log_path = tmp_path / "runtime.log"
    configure_file_logging(str(log_path))
    info("TEST", "SUB", "EVENT", msg="hello world")
    # Flush to ensure handler wrote before we read it back.
    for h in _FILE_LOGGER.handlers:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    assert "hello world" in content
    # The rendered ``_emit`` line carries the level column ("INFO");
    # the file format deliberately drops Python's %(levelname)s to
    # avoid the WARN/WARNING drift between stdout and file.
    assert "INFO" in content


def test_warn_call_writes_with_warning_level(tmp_path):
    log_path = tmp_path / "runtime.log"
    configure_file_logging(str(log_path))
    warn("TEST", "SUB", "EVENT", msg="something off")
    for h in _FILE_LOGGER.handlers:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    assert "something off" in content
    # File-side level string must match stdout (= "WARN", not Python's
    # "WARNING"). The dropped %(levelname)s in the formatter is the
    # mechanism that prevents the drift. Operators ``\bWARN\b`` grep
    # the file the same way they grep stdout.
    assert "WARN" in content
    assert "WARNING" not in content


def test_error_call_writes_with_error_level(tmp_path):
    log_path = tmp_path / "runtime.log"
    configure_file_logging(str(log_path))
    error("TEST", "SUB", "EVENT", msg="boom")
    for h in _FILE_LOGGER.handlers:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    assert "boom" in content
    assert "ERROR" in content


def test_idempotent_same_path_no_duplicate_handlers(tmp_path):
    log_path = tmp_path / "runtime.log"
    h1 = configure_file_logging(str(log_path))
    h2 = configure_file_logging(str(log_path))
    assert h1 is h2  # same handler returned
    file_handlers = [
        h for h in _FILE_LOGGER.handlers if isinstance(h, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1


def test_idempotent_no_duplicate_writes(tmp_path):
    """If the dual-write hook were attached twice, each info() call
    would produce two file lines. Verify only one."""
    log_path = tmp_path / "runtime.log"
    configure_file_logging(str(log_path))
    configure_file_logging(str(log_path))  # idempotent re-call
    info("TEST", "SUB", "EVENT", msg="unique-marker-string")
    for h in _FILE_LOGGER.handlers:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    assert content.count("unique-marker-string") == 1


def test_reconfigure_different_path_detaches_old_handler(tmp_path):
    path_a = tmp_path / "a.log"
    path_b = tmp_path / "b.log"
    h_a = configure_file_logging(str(path_a))
    h_b = configure_file_logging(str(path_b))
    assert h_a is not h_b
    file_handlers = [
        h for h in _FILE_LOGGER.handlers if isinstance(h, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    assert file_handlers[0] is h_b


def test_reconfigure_different_path_writes_only_to_new_file(tmp_path):
    path_a = tmp_path / "a.log"
    path_b = tmp_path / "b.log"
    configure_file_logging(str(path_a))
    configure_file_logging(str(path_b))
    info("TEST", "SUB", "EVENT", msg="second-config-marker")
    for h in _FILE_LOGGER.handlers:
        h.flush()
    assert "second-config-marker" in path_b.read_text(encoding="utf-8")
    # a.log either doesn't exist or doesn't contain the marker.
    if path_a.exists():
        assert "second-config-marker" not in path_a.read_text(encoding="utf-8")


def test_rotation_triggers_on_maxbytes_threshold(tmp_path):
    """End-to-end rotation: write enough bytes to cross the configured
    threshold and verify ``.1`` backup file is created. Uses a small
    threshold for the test (the production 25MB is too large to exercise
    in a unit-test budget); the test patches maxBytes on the handler
    directly after configuration."""
    log_path = tmp_path / "runtime.log"
    handler = configure_file_logging(str(log_path))
    # Patch threshold to 1KB to make rotation cheap.
    handler.maxBytes = 1024
    # Write more than 1KB of log lines.
    for i in range(200):
        info("TEST", "SUB", "EVENT", msg=f"line-{i}-padding-padding-padding")
    handler.flush()
    # After crossing threshold, .1 backup should exist.
    backup_path = Path(str(log_path) + ".1")
    assert backup_path.exists()


def test_backup_count_caps_at_seven(tmp_path):
    """Force many rotations and verify only 7 backups (plus the active
    file) are retained."""
    log_path = tmp_path / "runtime.log"
    handler = configure_file_logging(str(log_path))
    # Patch threshold tiny so each batch of writes rotates.
    handler.maxBytes = 256
    # Need to trigger > 7 rotations.
    for batch in range(15):
        for i in range(20):
            info("TEST", "SUB", "EVENT",
                 msg=f"batch{batch}-line{i}-paddingpaddingpadding")
        handler.flush()
    # Count rotated files: .1 through .7 may exist, but never .8.
    rotated = [
        p for p in tmp_path.iterdir()
        if p.name.startswith("runtime.log.") and p.suffix.lstrip(".").isdigit()
    ]
    assert len(rotated) <= 7
    assert not (tmp_path / "runtime.log.8").exists()


def test_format_includes_time_level_message(tmp_path):
    log_path = tmp_path / "runtime.log"
    configure_file_logging(str(log_path))
    info("TEST", "SUB", "EVENT", msg="format-probe")
    for h in _FILE_LOGGER.handlers:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    # Loose checks: contains a HH:MM:SS-style prefix, the level column
    # from the embedded ``_emit`` line ("INFO"), and the message. NB:
    # the file format intentionally drops Python's %(levelname)s and
    # %(name)s — the embedded ``_emit`` line already carries level +
    # sys_name columns, so the file format is "%(asctime)s.%(msecs)03d
    # %(message)s" only.
    assert "INFO" in content
    assert "format-probe" in content
    # First line should start with two digits (hour).
    first_line = content.splitlines()[0]
    assert first_line[:2].isdigit() and first_line[2] == ":"


def test_no_handler_attached_is_noop(tmp_path):
    """When no handler has been configured, ``info`` should still work
    (stdout path) — no exception from the missing handler."""
    # _reset_pancakebot_logger fixture already detached all handlers.
    assert len(_FILE_LOGGER.handlers) == 0
    # Should not raise.
    info("TEST", "SUB", "EVENT", msg="no-handler-attached")


def test_relative_path_resolves_against_repo_root(tmp_path, monkeypatch):
    """Relative log paths must resolve against the repo root anchor
    (= the dir containing the ``pancakebot/`` package), NOT
    ``os.getcwd()``. This prevents log misplacement when an operator
    launches from a non-repo-root cwd. Reviewer flag, Bundle 5."""
    import pancakebot.log as _log
    # Cross-check: simulate a non-repo-root cwd.
    monkeypatch.chdir(tmp_path)
    rel = "tmp_test_runtime.log"
    handler = configure_file_logging(rel)
    try:
        expected_repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(_log.__file__))
        )
        expected_path = os.path.abspath(
            os.path.join(expected_repo_root, rel)
        )
        assert os.path.abspath(handler.baseFilename) == expected_path
        # Must NOT have landed in the (non-repo-root) cwd.
        assert os.path.abspath(handler.baseFilename) != os.path.abspath(
            os.path.join(str(tmp_path), rel)
        )
    finally:
        # Cleanup: handler is the only reference to the file.
        handler.close()
        if os.path.exists(handler.baseFilename):
            try:
                os.remove(handler.baseFilename)
            except OSError:
                pass


def test_absolute_path_passes_through_unchanged(tmp_path):
    """Absolute log paths must pass through unchanged (no repo-root
    prefixing). The relative-path branch is for the canonical
    ``var/dry/runtime.log`` / ``var/live/runtime.log`` shorthand only."""
    abs_path = str(tmp_path / "absolute_runtime.log")
    handler = configure_file_logging(abs_path)
    assert os.path.abspath(handler.baseFilename) == os.path.abspath(abs_path)


def test_propagate_disabled_no_root_handler_writes(tmp_path, monkeypatch):
    """``pancakebot`` logger has propagate=False — a root-attached
    handler must NOT receive our log lines (avoids surprising third-
    party consumers of the root logger)."""
    log_path = tmp_path / "runtime.log"
    configure_file_logging(str(log_path))
    # Attach a probe handler to the root logger.
    probe_records: list[logging.LogRecord] = []

    class ProbeHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            probe_records.append(record)

    root_logger = logging.getLogger()
    probe = ProbeHandler(level=logging.DEBUG)
    root_logger.addHandler(probe)
    try:
        info("TEST", "SUB", "EVENT", msg="propagate-probe")
    finally:
        root_logger.removeHandler(probe)
    # No records should have reached the root logger.
    assert not any("propagate-probe" in r.getMessage() for r in probe_records)
