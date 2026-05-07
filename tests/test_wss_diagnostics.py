"""Tests for the WSS forensic diagnostics module.

Covers:
- ``RawWssLogger``: file lifecycle, daily roll, frame-write integrity,
  payload truncation, error swallowing.
- ``SystemStatsPoller``: non-Windows fallback (test-runner-friendly),
  start/stop idempotence.

The diagnostics module is failure-tolerant by design (its job is to
collect data, not to take down the bot). These tests pin that
contract.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import wss_diagnostics  # noqa: E402


# ---------------------------------------------------------------------------
# RawWssLogger
# ---------------------------------------------------------------------------

def test_raw_wss_logger_creates_log_dir(tmp_path: Path) -> None:
    """Constructor must create the log dir if it doesn't exist; no
    exception, no error log."""
    target = tmp_path / "logs_subdir" / "more"
    assert not target.exists()
    rl = wss_diagnostics.RawWssLogger(log_dir=target)
    assert target.exists()
    rl.close()


def test_raw_wss_logger_writes_frame_to_dated_file(tmp_path: Path) -> None:
    """A logged frame should land in a wss_raw_<date>.log file."""
    rl = wss_diagnostics.RawWssLogger(log_dir=tmp_path)
    rl.log_frame("recv", 12, '{"foo":"bar"}')
    rl.close()

    files = list(tmp_path.glob("wss_raw_*.log"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "recv" in content
    assert "len=12" in content
    assert '{"foo":"bar"}' in content


def test_raw_wss_logger_increments_frame_count(tmp_path: Path) -> None:
    """frame_count property tracks successful writes."""
    rl = wss_diagnostics.RawWssLogger(log_dir=tmp_path)
    assert rl.frame_count == 0
    rl.log_frame("recv", 5, "hello")
    rl.log_frame("recv", 7, "goodbye")
    assert rl.frame_count == 2
    rl.close()


def test_raw_wss_logger_truncates_long_payloads(tmp_path: Path) -> None:
    """Payloads larger than 8KB are truncated with a marker so the log
    file doesn't blow up on a single oversized frame."""
    rl = wss_diagnostics.RawWssLogger(log_dir=tmp_path)
    big_payload = "X" * 20_000
    rl.log_frame("recv", len(big_payload), big_payload)
    rl.close()

    files = list(tmp_path.glob("wss_raw_*.log"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "<truncated" in content
    # Unequal lengths covered: original len recorded, but written payload
    # truncated.
    assert f"len={20_000}" in content
    assert len(content) < 20_000  # truncation actually happened


def test_raw_wss_logger_no_payload_records_length_only(tmp_path: Path) -> None:
    """A None payload means metadata-only line (no body)."""
    rl = wss_diagnostics.RawWssLogger(log_dir=tmp_path)
    rl.log_frame("meta", 0, None)
    rl.close()

    files = list(tmp_path.glob("wss_raw_*.log"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "meta" in content
    assert "len=0" in content


def test_raw_wss_logger_close_is_idempotent(tmp_path: Path) -> None:
    """Repeated close() must not raise (graceful shutdown contract)."""
    rl = wss_diagnostics.RawWssLogger(log_dir=tmp_path)
    rl.log_frame("recv", 1, "x")
    rl.close()
    rl.close()  # second close is a no-op


def test_raw_wss_logger_swallows_write_errors_to_unwritable_dir(tmp_path: Path) -> None:
    """If the dir somehow becomes unwritable mid-life, errors are
    swallowed (we increment _dropped_count) -- the recv loop must not
    crash on disk-full or similar."""
    target = tmp_path / "logs"
    rl = wss_diagnostics.RawWssLogger(log_dir=target)
    # Force the file pointer to fail by closing it then trying to write
    # while the underlying file handle is dead. The implementation
    # should detect the failure and swallow.
    rl._fp = None  # simulate dead file pointer
    rl._current_date = "1999-01-01"  # forces re-open attempt; if open fails it sets _fp back to None
    # Make subsequent open fail by replacing log_dir with a path that
    # is a file (not a dir), so mkdir would fail / open(... , "a") would fail.
    file_path = tmp_path / "blocker.txt"
    file_path.write_text("blocking")
    rl._log_dir = file_path  # not a directory
    rl.log_frame("recv", 1, "x")  # should NOT raise
    rl.close()


# ---------------------------------------------------------------------------
# SystemStatsPoller
# ---------------------------------------------------------------------------

def test_system_stats_poller_constructs_cleanly(tmp_path: Path) -> None:
    """Construction must not raise, must create the log dir."""
    p = wss_diagnostics.SystemStatsPoller(
        log_dir=tmp_path / "stats",
        poll_interval_s=60.0,
    )
    assert (tmp_path / "stats").exists()
    p.stop()  # idempotent on un-started


def test_system_stats_poller_skips_on_non_windows(monkeypatch, tmp_path: Path) -> None:
    """On non-Windows OSes, start() must early-return without spawning
    a thread (PowerShell isn't available)."""
    # Force non-Windows.
    monkeypatch.setattr(os, "name", "posix")
    p = wss_diagnostics.SystemStatsPoller(
        log_dir=tmp_path,
        poll_interval_s=60.0,
    )
    p.start()
    assert p._thread is None  # no thread spawned
    p.stop()  # still idempotent


def test_system_stats_poller_stop_is_idempotent(tmp_path: Path) -> None:
    """Repeated stop() must not raise."""
    p = wss_diagnostics.SystemStatsPoller(
        log_dir=tmp_path,
        poll_interval_s=60.0,
    )
    p.stop()
    p.stop()


# ---------------------------------------------------------------------------
# Integration: PoolEventWatcher diagnostics initialization
# ---------------------------------------------------------------------------

def test_pool_watcher_diagnostics_default_on() -> None:
    """The watcher defaults to enable_diagnostics=True. Both raw logger
    and stats poller objects exist after construction."""
    from pancakebot.chain.pool_watcher import PoolEventWatcher

    pw = PoolEventWatcher(interval_seconds=300)
    assert pw._raw_wss_logger is not None
    assert pw._sys_stats_poller is not None
    assert pw._diagnostics_enabled is True


def test_pool_watcher_diagnostics_can_be_disabled() -> None:
    """Tests and lightweight scenarios can opt out via
    enable_diagnostics=False; both objects stay None."""
    from pancakebot.chain.pool_watcher import PoolEventWatcher

    pw = PoolEventWatcher(interval_seconds=300, enable_diagnostics=False)
    assert pw._raw_wss_logger is None
    assert pw._sys_stats_poller is None
    assert pw._diagnostics_enabled is False


def test_pool_watcher_diag_log_frame_no_op_when_disabled() -> None:
    """_diag_log_frame must be a safe no-op when diagnostics are off
    so the recv loop's call site doesn't need to know about diag state."""
    from pancakebot.chain.pool_watcher import PoolEventWatcher

    pw = PoolEventWatcher(interval_seconds=300, enable_diagnostics=False)
    # Should not raise even with diagnostics off.
    pw._diag_log_frame("recv", '{"foo":"bar"}')
    pw._diag_log_frame("meta", "")
