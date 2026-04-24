"""Tests for supervisor Option-C retry-once semantics.

Verifies that ``_safe_read_json`` and ``_pid_is_our_bot`` each retry once on
transient failures before returning the final answer. Added in response to a
2026-04-23 false-DOWN where two reads under the classifier returned bad data
simultaneously, firing a spurious Discord alert.

Retry contract:

    _safe_read_json
      - first read returns a dict    -> return it immediately (no retry, no log)
      - first read returns None AND second read returns a dict -> save-by-retry,
        return dict, emit ``safe_read_json_retry_recovered`` line on stderr
      - first and second both return None -> None (both failed; genuine miss)

    _pid_is_our_bot
      - psutil raises -> retry once after ``_TRANSIENT_READ_BACKOFF_S``
      - pid_exists returns False -> clean miss, NO retry
      - cmdline doesn't match needle -> clean miss, NO retry
      - both attempts raise -> False, emit retry_exhausted stderr line

Run:
    python -m pytest tests/test_supervisor_retry.py -v
    # or standalone (no pytest dependency required):
    python tests/test_supervisor_retry.py
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# scripts/ is not a package -- load supervisor.py by path.
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "supervisor_under_test", str(_REPO_ROOT / "scripts" / "supervisor.py")
)
supervisor = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(supervisor)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _safe_read_json
# ---------------------------------------------------------------------------

def test_safe_read_json_first_attempt_success_no_retry():
    """Healthy path: first read works, no retry, no stderr noise, no sleep."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "heartbeat.json"
        p.write_text(json.dumps({"state": "OPEN", "epoch": 42}), encoding="utf-8")

        stderr_buf = StringIO()
        with mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
             mock.patch.object(supervisor.time, "sleep") as mock_sleep:
            result = supervisor._safe_read_json(p)

        assert result == {"state": "OPEN", "epoch": 42}
        mock_sleep.assert_not_called()
        assert stderr_buf.getvalue() == "", "healthy path must not emit stderr"


def test_safe_read_json_retry_recovers():
    """First read None, second read succeeds -> returns dict + logs recovery."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "heartbeat.json"
        # File starts missing, then appears on retry. Use a side-effect to
        # materialize the file during the sleep.
        def fake_sleep(_s: float) -> None:
            p.write_text(json.dumps({"state": "OPEN"}), encoding="utf-8")

        stderr_buf = StringIO()
        with mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
             mock.patch.object(supervisor.time, "sleep", side_effect=fake_sleep) as mock_sleep:
            result = supervisor._safe_read_json(p)

        assert result == {"state": "OPEN"}
        mock_sleep.assert_called_once_with(supervisor._TRANSIENT_READ_BACKOFF_S)
        assert "safe_read_json_retry_recovered" in stderr_buf.getvalue()
        assert "heartbeat.json" in stderr_buf.getvalue()


def test_safe_read_json_both_fail_returns_none_no_recovery_log():
    """Genuine missing file: both attempts None, no recovery log emitted."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "heartbeat.json"
        # Never write the file -- both reads fail.

        stderr_buf = StringIO()
        with mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
             mock.patch.object(supervisor.time, "sleep") as mock_sleep:
            result = supervisor._safe_read_json(p)

        assert result is None
        mock_sleep.assert_called_once_with(supervisor._TRANSIENT_READ_BACKOFF_S)
        assert "safe_read_json_retry_recovered" not in stderr_buf.getvalue(), (
            "must not log recovery when retry also failed"
        )


def test_safe_read_json_malformed_then_valid_retries():
    """Mid-write (corrupt JSON) on first, valid on second -> recovers."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "heartbeat.json"
        p.write_text("{partial-write", encoding="utf-8")  # broken JSON

        def fake_sleep(_s: float) -> None:
            p.write_text(json.dumps({"ok": True}), encoding="utf-8")

        stderr_buf = StringIO()
        with mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
             mock.patch.object(supervisor.time, "sleep", side_effect=fake_sleep):
            result = supervisor._safe_read_json(p)

        assert result == {"ok": True}
        assert "safe_read_json_retry_recovered" in stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# _pid_is_our_bot
# ---------------------------------------------------------------------------

def test_pid_is_our_bot_success_first_attempt_no_retry():
    """Clean hit: cmdline matches on first try, no retry, no sleep."""
    fake_psutil = mock.MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_proc = mock.MagicMock()
    fake_proc.cmdline.return_value = ["python", "run.py", "--dry"]
    fake_psutil.Process.return_value = fake_proc

    with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
         mock.patch.object(supervisor.time, "sleep") as mock_sleep:
        result = supervisor._pid_is_our_bot(1234, "dry")

    assert result is True
    mock_sleep.assert_not_called()


def test_pid_is_our_bot_clean_miss_pid_not_exist_no_retry():
    """pid_exists=False is a CLEAN miss -- must NOT retry."""
    fake_psutil = mock.MagicMock()
    fake_psutil.pid_exists.return_value = False

    with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
         mock.patch.object(supervisor.time, "sleep") as mock_sleep:
        result = supervisor._pid_is_our_bot(1234, "dry")

    assert result is False
    mock_sleep.assert_not_called()
    # pid_exists should only be called once (no retry for clean miss)
    assert fake_psutil.pid_exists.call_count == 1


def test_pid_is_our_bot_clean_miss_wrong_cmdline_no_retry():
    """PID alive but wrong cmdline: clean miss, must NOT retry."""
    fake_psutil = mock.MagicMock()
    fake_psutil.pid_exists.return_value = True
    fake_proc = mock.MagicMock()
    fake_proc.cmdline.return_value = ["python", "something_else.py"]
    fake_psutil.Process.return_value = fake_proc

    with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
         mock.patch.object(supervisor.time, "sleep") as mock_sleep:
        result = supervisor._pid_is_our_bot(1234, "dry")

    assert result is False
    mock_sleep.assert_not_called()
    assert fake_psutil.Process.call_count == 1


def test_pid_is_our_bot_exception_then_success_retries():
    """psutil raises first, works second: retry recovers."""
    fake_psutil = mock.MagicMock()

    call_count = {"n": 0}

    def flaky_pid_exists(_pid: int) -> bool:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("transient psutil error")
        return True

    fake_psutil.pid_exists.side_effect = flaky_pid_exists
    fake_proc = mock.MagicMock()
    fake_proc.cmdline.return_value = ["python", "run.py", "--dry"]
    fake_psutil.Process.return_value = fake_proc

    with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
         mock.patch.object(supervisor.time, "sleep") as mock_sleep:
        result = supervisor._pid_is_our_bot(1234, "dry")

    assert result is True, "second attempt succeeded; should return True"
    mock_sleep.assert_called_once_with(supervisor._TRANSIENT_READ_BACKOFF_S)
    assert call_count["n"] == 2


def test_pid_is_our_bot_exception_both_attempts_returns_false_logs():
    """Both attempts raise: False with retry_exhausted log."""
    fake_psutil = mock.MagicMock()
    fake_psutil.pid_exists.side_effect = OSError("persistent psutil error")

    stderr_buf = StringIO()
    with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
         mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
         mock.patch.object(supervisor.time, "sleep") as mock_sleep:
        result = supervisor._pid_is_our_bot(1234, "dry")

    assert result is False
    mock_sleep.assert_called_once_with(supervisor._TRANSIENT_READ_BACKOFF_S)
    assert "pid_is_our_bot_retry_exhausted" in stderr_buf.getvalue()
    assert "pid=1234" in stderr_buf.getvalue()
    assert "mode=dry" in stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Retry diagnostic sink (Q5 fix: schtask discards stderr on exit 0)
# ---------------------------------------------------------------------------

def test_retry_diagnostic_appends_to_sink_when_configured():
    """When _RETRY_LOG_SINK is set, retry events also append to supervisor.log."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = Path(tmp) / "supervisor.log"
        hb = Path(tmp) / "heartbeat.json"

        def fake_sleep(_s: float) -> None:
            hb.write_text(json.dumps({"state": "OPEN"}), encoding="utf-8")

        stderr_buf = StringIO()
        original_sink = supervisor._RETRY_LOG_SINK
        try:
            supervisor._RETRY_LOG_SINK = sink
            with mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
                 mock.patch.object(supervisor.time, "sleep", side_effect=fake_sleep):
                result = supervisor._safe_read_json(hb)
        finally:
            supervisor._RETRY_LOG_SINK = original_sink

        assert result == {"state": "OPEN"}
        assert "safe_read_json_retry_recovered" in stderr_buf.getvalue()
        assert sink.exists(), "sink file must be created"
        sink_text = sink.read_text(encoding="utf-8")
        assert "DIAGNOSTIC safe_read_json_retry_recovered" in sink_text
        assert "heartbeat.json" in sink_text


def test_retry_diagnostic_sink_unset_only_stderr():
    """When _RETRY_LOG_SINK is None (default / tests), retry events stay on stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        hb = Path(tmp) / "heartbeat.json"

        def fake_sleep(_s: float) -> None:
            hb.write_text(json.dumps({"ok": True}), encoding="utf-8")

        stderr_buf = StringIO()
        original_sink = supervisor._RETRY_LOG_SINK
        try:
            supervisor._RETRY_LOG_SINK = None
            with mock.patch.object(supervisor.sys, "stderr", stderr_buf), \
                 mock.patch.object(supervisor.time, "sleep", side_effect=fake_sleep):
                result = supervisor._safe_read_json(hb)
        finally:
            supervisor._RETRY_LOG_SINK = original_sink

        assert result == {"ok": True}
        assert "safe_read_json_retry_recovered" in stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Wall-time budget sanity check
# ---------------------------------------------------------------------------

def test_backoff_within_budget():
    """Sanity: 2 reads * retry * backoff <= 1.5s, well inside 3-min budget."""
    assert supervisor._TRANSIENT_READ_BACKOFF_S <= 0.5
    # Two independent retry paths (heartbeat + psutil) each at most one sleep.
    worst_case_seconds = 2 * supervisor._TRANSIENT_READ_BACKOFF_S
    assert worst_case_seconds <= 1.0, (
        f"worst-case retry latency {worst_case_seconds}s breaches 1s margin"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
