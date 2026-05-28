"""Tests for the Windows-Service-based supervisor architecture.

Exercises the pure-logic surface of ``pancakebot/service/``:
  - ``supervision.classify_state`` for each state (UP / STARTING /
    CRASHED / DOWN / UNINSTRUMENTED). Heartbeat-staleness STALE
    classification removed 2026-05-27 (Step 27a).
  - ``supervision`` restart-history helpers (read / write / prune / count)
  - ``notifications.resolve_webhook_env`` channel routing per kind
  - ``notifications.rate_limit_ok`` cooldown enforcement
  - ``notifications.build_message`` basic structure

SCM-interaction code (win32service.OpenSCManager / ControlService etc.)
is intentionally NOT covered here — it's effectively e2e-only and
requires admin + an installed service. The service base class loads
``win32serviceutil`` at module import so we skip those imports in this
test file.

Run:
    python -m pytest tests/test_service_lifecycle.py -v
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from pancakebot.service import notifications, supervision  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mode_tree(tmp: Path, mode: str) -> dict[str, Path]:
    """Build a fake var/<mode>/ tree mirroring ``artifacts_for_mode`` paths.

    Returns the same dict shape as ``artifacts_for_mode`` but rooted under
    ``tmp`` instead of the repo root.
    """
    mode_dir = tmp / "var" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "logs").mkdir(exist_ok=True)
    return {
        "pid": mode_dir / "bot.pid",
        "crash": mode_dir / "crash.json",
        "supervisor_log": mode_dir / "supervisor.log",
        "trades": mode_dir / "trades.csv",
        "last_alert": mode_dir / "last_alert.json",
        "restart_history": mode_dir / "restart_history.jsonl",
        "logs_dir": mode_dir / "logs",
    }


def _backdate_mtime(path: Path, age_s: float) -> None:
    past = time.time() - age_s
    os.utime(str(path), (past, past))


def _write_pid_file(path: Path, pid: int, *, age_s: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")
    if age_s > 0:
        _backdate_mtime(path, age_s)


def _write_crash(path: Path, *, exc_type: str = "FakeError") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ts_wall": time.time(),
        "exc_type": exc_type,
        "exc_repr": f"{exc_type}('boom')",
        "traceback_str": "Traceback (most recent call last):\n  ...\n",
        "last_epoch": 100,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# supervision.classify_state — state-by-state coverage
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_artifacts(monkeypatch, tmp_path):
    """Redirect ``artifacts_for_mode`` to a tmp tree + stub pid-liveness."""
    art_live = _make_mode_tree(tmp_path, "live")
    art_dry = _make_mode_tree(tmp_path, "dry")
    monkeypatch.setattr(
        supervision, "artifacts_for_mode",
        lambda mode: art_live if mode == "live" else art_dry,
    )
    # Default: no pid is "our bot." Tests opt-in by overriding.
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: False)
    monkeypatch.setattr(supervision, "find_legacy_bot_pid", lambda mode: None)
    return {"live": art_live, "dry": art_dry}


def test_classify_down_when_no_artifacts(fake_artifacts):
    status, fields = supervision.classify_state("live")
    assert status == "DOWN"
    assert fields == {}


def test_classify_up_with_fresh_pid_past_grace(fake_artifacts, monkeypatch):
    """Step 27a: classify_state returns UP when PID file is past startup grace
    and points to a live process. Heartbeat infrastructure removed."""
    art = fake_artifacts["live"]
    _write_pid_file(art["pid"], 4242, age_s=120)  # past 90s startup grace
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: pid == 4242)
    status, fields = supervision.classify_state("live")
    assert status == "UP", f"unexpected: {status} {fields}"
    assert fields["pid"] == 4242


def test_classify_starting_with_fresh_pid(fake_artifacts, monkeypatch):
    """Step 27a: a fresh PID file (within startup grace) classifies as STARTING."""
    art = fake_artifacts["live"]
    _write_pid_file(art["pid"], 7777)
    monkeypatch.setattr(supervision, "_pid_is_our_bot", lambda pid, mode: pid == 7777)
    status, fields = supervision.classify_state("live")
    assert status == "STARTING"
    assert fields["pid"] == 7777


def test_classify_crashed_with_crash_json(fake_artifacts):
    art = fake_artifacts["live"]
    _write_crash(art["crash"], exc_type="ValueError")
    status, fields = supervision.classify_state("live")
    assert status == "CRASHED"
    assert fields["exc"] == "ValueError"


def test_classify_uninstrumented_with_legacy_bot(fake_artifacts, monkeypatch):
    monkeypatch.setattr(supervision, "find_legacy_bot_pid", lambda mode: 12345)
    status, fields = supervision.classify_state("live")
    assert status == "UNINSTRUMENTED"
    assert fields["pid"] == 12345
    assert fields["note"] == "legacy_no_instrumentation"


# ---------------------------------------------------------------------------
# Restart-history helpers
# ---------------------------------------------------------------------------

def test_restart_history_roundtrip(tmp_path):
    path = tmp_path / "restart_history.jsonl"
    entries = [
        {"ts_wall": 1000.0, "trigger": "DOWN", "new_pid": 1},
        {"ts_wall": 2000.0, "trigger": "CRASHED", "new_pid": 2},
    ]
    supervision.write_restart_history(path, entries)
    loaded = supervision.read_restart_history(path)
    assert loaded == entries


def test_restart_history_prune_drops_old_entries(tmp_path):
    entries = [
        {"ts_wall": 1000.0, "trigger": "DOWN"},
        {"ts_wall": 5000.0, "trigger": "CRASHED"},
    ]
    # now=6000, window=2000 → keep only entries with ts_wall >= 4000
    kept = supervision.prune_history(entries, now=6000.0, window_s=2000.0)
    assert len(kept) == 1
    assert kept[0]["ts_wall"] == 5000.0


def test_restart_history_count_within(tmp_path):
    entries = [
        {"ts_wall": 100.0},
        {"ts_wall": 500.0},
        {"ts_wall": 900.0},
    ]
    # now=1000, window=600 → count entries with ts_wall >= 400
    n = supervision.count_within(entries, now=1000.0, window_s=600.0)
    assert n == 2


def test_restart_history_drops_malformed_lines(tmp_path):
    path = tmp_path / "rh.jsonl"
    path.write_text(
        '{"ts_wall":1.0}\n'
        'not json at all\n'
        '{"ts_wall":2.0}\n'
        '\n',
        encoding="utf-8",
    )
    entries = supervision.read_restart_history(path)
    assert len(entries) == 2
    assert entries[0]["ts_wall"] == 1.0
    assert entries[1]["ts_wall"] == 2.0


# ---------------------------------------------------------------------------
# notifications.resolve_webhook_env — channel routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kind,expected", [
    ("live", "CRASHED", notifications.LIVE_WEBHOOK_ENV),
    ("dry",  "CRASHED", notifications.DRY_WEBHOOK_ENV),
    ("live", "UNINSTRUMENTED", notifications.GENERAL_WEBHOOK_ENV),
    ("dry",  "UNINSTRUMENTED", notifications.GENERAL_WEBHOOK_ENV),
    ("live", "STARTED", notifications.LIVE_WEBHOOK_ENV),
    ("live", "REBOOTED", notifications.LIVE_WEBHOOK_ENV),
    ("live", "STOPPED", notifications.LIVE_WEBHOOK_ENV),
    # MODE_TRANSITION always goes to live regardless of firing mode
    ("live", "MODE_TRANSITION", notifications.LIVE_WEBHOOK_ENV),
    ("dry",  "MODE_TRANSITION", notifications.LIVE_WEBHOOK_ENV),
    # MODE_TRANSITION_REFUSED always goes to dry (it's only the dry side
    # that ever fires this anyway, but we test the routing for completeness)
    ("dry",  "MODE_TRANSITION_REFUSED", notifications.DRY_WEBHOOK_ENV),
    ("live", "SERVICE_CRASHED", notifications.GENERAL_WEBHOOK_ENV),
])
def test_channel_routing(mode, kind, expected):
    assert notifications.resolve_webhook_env(mode, kind) == expected


# ---------------------------------------------------------------------------
# notifications.rate_limit_ok — cooldown enforcement
# ---------------------------------------------------------------------------

def test_rate_limit_first_call_passes(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "CRASHED", now=1000.0, cooldown_s=60.0)
    # Second immediate call is suppressed
    assert not notifications.rate_limit_ok(path, "CRASHED", now=1010.0, cooldown_s=60.0)


def test_rate_limit_clears_after_cooldown(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "DOWN", now=1000.0, cooldown_s=60.0)
    assert not notifications.rate_limit_ok(path, "DOWN", now=1030.0, cooldown_s=60.0)
    assert notifications.rate_limit_ok(path, "DOWN", now=1061.0, cooldown_s=60.0)


def test_rate_limit_per_key_independent(tmp_path):
    path = tmp_path / "last_alert.json"
    assert notifications.rate_limit_ok(path, "CRASHED", now=1000.0, cooldown_s=60.0)
    # Different kind: independent bucket
    assert notifications.rate_limit_ok(path, "DOWN", now=1000.0, cooldown_s=60.0)
    # Same kind, still in cooldown
    assert not notifications.rate_limit_ok(path, "CRASHED", now=1010.0, cooldown_s=60.0)


# ---------------------------------------------------------------------------
# notifications.build_message — sanity checks
# ---------------------------------------------------------------------------

def test_build_message_includes_kind_and_mode():
    msg = notifications.build_message(
        mode="live", kind="STARTED", fields={"pid": 9999},
    )
    assert "STARTED" in msg
    assert "PancakeBot-live" in msg
    assert "9999" in msg


def test_build_message_with_crash_artifact(tmp_path):
    art = _make_mode_tree(tmp_path, "live")
    _write_crash(art["crash"], exc_type="RuntimeError")
    msg = notifications.build_message(
        mode="live", kind="CRASHED",
        fields={"pid": 1, "last_epoch": 42},
        art=art,
    )
    assert "CRASHED" in msg
    assert "RuntimeError" in msg
    assert "traceback" in msg.lower() or "Traceback" in msg


def test_build_message_with_detail_string():
    msg = notifications.build_message(
        mode="live", kind="MODE_TRANSITION",
        detail="stopping PancakeBotDry to start live",
    )
    assert "MODE_TRANSITION" in msg
    assert "stopping PancakeBotDry" in msg


# ---------------------------------------------------------------------------
# notifications.notify — DISABLED path (no HTTP made)
# ---------------------------------------------------------------------------

def test_notify_disabled_when_env_var_unset(tmp_path, monkeypatch):
    """When the resolved webhook env var is unset, notify returns DISABLED
    without attempting any HTTP call."""
    monkeypatch.delenv(notifications.LIVE_WEBHOOK_ENV, raising=False)
    art = _make_mode_tree(tmp_path, "live")
    outcome = notifications.notify(mode="live", kind="CRASHED", art=art)
    assert outcome == "DISABLED"


# ---------------------------------------------------------------------------
# Module-import smoke test for the Win32-dependent code path
# ---------------------------------------------------------------------------

def test_common_module_imports():
    """``pancakebot.service.common`` imports pywin32; verify it loads cleanly.

    We don't exercise the ServiceFramework class here — that requires
    SCM context — but a successful import catches typos / wrong symbol
    references before install.
    """
    import importlib
    mod = importlib.import_module("pancakebot.service.common")
    assert hasattr(mod, "_PancakeBotServiceBase")
    assert hasattr(mod, "_query_service_state")
    assert hasattr(mod, "_stop_service_and_wait")


def test_live_and_dry_service_classes_importable():
    from pancakebot.service.live_service import PancakeBotLiveService
    from pancakebot.service.dry_service import PancakeBotDryService
    assert PancakeBotLiveService._MODE == "live"
    assert PancakeBotLiveService._OTHER_SERVICE == "PancakeBotDry"
    assert PancakeBotDryService._MODE == "dry"
    assert PancakeBotDryService._OTHER_SERVICE == "PancakeBotLive"


# ---------------------------------------------------------------------------
# Step 2 — classify_running_bot (Popen-based, no DOWN-race)
# ---------------------------------------------------------------------------

class _FakePopen:
    """Minimal Popen stand-in for testing classify_running_bot without
    actually spawning a process."""
    def __init__(self, pid: int, alive: bool, exit_code: int | None = 0):
        self.pid = pid
        self._alive = alive
        self._exit_code = exit_code

    def poll(self):
        return None if self._alive else self._exit_code

    def kill(self):
        self._alive = False


def test_classify_running_bot_just_spawned_is_starting_not_down(tmp_path):
    """The DOWN-race fix: a just-spawned Popen with no heartbeat yet must
    NOT classify as DOWN (the old bug). It should be STARTING (inside grace)."""
    art = _make_mode_tree(tmp_path, "live")
    proc = _FakePopen(pid=12345, alive=True)
    status, fields = supervision.classify_running_bot(
        proc, proc_started_at=time.time(), art=art,
        startup_grace_s=30.0,
    )
    assert status == "STARTING", f"expected STARTING, got {status} {fields}"
    assert fields["pid"] == 12345


def test_classify_running_bot_killed_process_is_down(tmp_path):
    """Popen.poll() returning exit code = process dead. No crash.json → DOWN."""
    art = _make_mode_tree(tmp_path, "live")
    proc = _FakePopen(pid=12345, alive=False, exit_code=1)
    status, fields = supervision.classify_running_bot(
        proc, proc_started_at=time.time() - 60.0, art=art,
    )
    assert status == "DOWN", f"expected DOWN, got {status} {fields}"


def test_classify_running_bot_killed_with_crash_json_is_crashed(tmp_path):
    """Dead Popen + crash.json present → CRASHED, not DOWN."""
    art = _make_mode_tree(tmp_path, "live")
    _write_crash(art["crash"], exc_type="ValueError")
    proc = _FakePopen(pid=12345, alive=False, exit_code=1)
    status, fields = supervision.classify_running_bot(
        proc, proc_started_at=time.time() - 60.0, art=art,
    )
    assert status == "CRASHED"
    assert fields["exc"] == "ValueError"


def test_classify_running_bot_alive_past_grace_is_up(tmp_path):
    """Step 27a: alive Popen handle past startup grace ⇒ UP. No heartbeat read."""
    art = _make_mode_tree(tmp_path, "live")
    proc = _FakePopen(pid=12345, alive=True)
    status, fields = supervision.classify_running_bot(
        proc, proc_started_at=time.time() - 60.0, art=art,
        startup_grace_s=30.0,
    )
    assert status == "UP", f"expected UP, got {status} {fields}"
    assert fields["pid"] == 12345
    assert "proc_uptime" in fields


def test_classify_running_bot_proc_none_is_down(tmp_path):
    """proc=None means we haven't spawned yet → DOWN."""
    art = _make_mode_tree(tmp_path, "live")
    status, _fields = supervision.classify_running_bot(
        None, proc_started_at=None, art=art,
    )
    assert status == "DOWN"


# ---------------------------------------------------------------------------
# Step 4 — safe_stderr_write None-guard regression test
# ---------------------------------------------------------------------------

def test_safe_stderr_write_no_attribute_error_when_stderr_is_none(monkeypatch):
    """When sys.stderr is None (service-hosted), writes must NOT raise
    AttributeError. This is the regression guard for the post-reboot
    crash 2026-05-23 where notify()'s SEND_FAILED path took down the
    service via sys.stderr.write on a NoneType."""
    monkeypatch.setattr(sys, "stderr", None)
    # Should not raise — that's the entire requirement.
    supervision.safe_stderr_write("test message that would have crashed")


def test_notify_send_failed_does_not_raise_with_stderr_none(tmp_path, monkeypatch):
    """End-to-end: a notify() call that hits SEND_FAILED with stderr=None
    must NOT crash. This is the exact scenario from the post-reboot incident."""
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setenv(notifications.LIVE_WEBHOOK_ENV, "https://invalid.example/webhook")
    # Make the HTTP call fail synchronously so we hit the SEND_FAILED path.
    monkeypatch.setattr(
        notifications, "_send_discord",
        lambda url, mode, msg: (False, "post_exception:simulated"),
    )
    art = _make_mode_tree(tmp_path, "live")
    # If safe_stderr_write didn't guard the None case, this would raise
    # AttributeError. Should return "SEND_FAILED" cleanly instead.
    outcome = notifications.notify(mode="live", kind="CRASHED", art=art)
    assert outcome == "SEND_FAILED"


def test_write_restart_history_does_not_raise_with_stderr_none(monkeypatch, tmp_path):
    """write_restart_history's error path also goes through safe_stderr_write."""
    monkeypatch.setattr(sys, "stderr", None)
    # Force the write to fail by pointing at a directory that doesn't
    # exist AND can't be created (use an invalid path).
    bad_path = tmp_path / "missing" / "subdir" / "rh.jsonl"
    # The function recovers via parent.mkdir(parents=True), so we need a
    # path where mkdir itself fails. Use a path containing a NUL byte —
    # invalid on Windows.
    bad_path_str = str(tmp_path / "rh\x00.jsonl")
    try:
        supervision.write_restart_history(Path(bad_path_str), [{"ts_wall": 1.0}])
    except ValueError:
        # NUL byte in path may raise ValueError before the function's
        # internal try/except. That's fine — it's not the AttributeError
        # we were guarding against.
        return
    # If we got here without exception, the safe_stderr_write guard worked
    # (the function's internal except triggered the stderr write).


# ---------------------------------------------------------------------------
# Step 3 — Job Object with KILL_ON_JOB_CLOSE
# ---------------------------------------------------------------------------

def test_job_object_kills_child_on_supervisor_death(tmp_path):
    """Spawn an outer 'supervisor' process that creates a kill-on-close job
    + spawns a long-sleeping child + assigns it to the job, then exits.
    The child MUST be killed when the supervisor exits (the job handle
    closes → KILL_ON_JOB_CLOSE → all members terminated)."""
    import subprocess as sp

    supervisor_script = r"""
import subprocess, sys, time
import win32job
job = win32job.CreateJobObject(None, "")
info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
win32job.AssignProcessToJobObject(job, int(proc._handle))
sys.stdout.write(f"{proc.pid}\n")
sys.stdout.flush()
# Brief sleep to make sure the child is fully running before we exit.
time.sleep(0.5)
# Outer process exits here — job handle is closed by the OS as part of
# process teardown — KILL_ON_JOB_CLOSE fires — child should die.
"""
    result = sp.run(
        [sys.executable, "-c", supervisor_script],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, f"supervisor script failed: {result.stderr}"
    child_pid_line = result.stdout.strip().splitlines()[-1]
    child_pid = int(child_pid_line)

    # Wait up to 2s for the child to be killed by job close.
    import psutil
    deadline = time.time() + 2.0
    killed = False
    while time.time() < deadline:
        if not psutil.pid_exists(child_pid):
            killed = True
            break
        time.sleep(0.05)

    if not killed:
        # Cleanup before failing
        try:
            psutil.Process(child_pid).kill()
        except Exception:
            pass
        raise AssertionError(
            f"child pid={child_pid} survived 2s after supervisor exit — "
            f"KILL_ON_JOB_CLOSE did not work"
        )


# ---------------------------------------------------------------------------
# _handle_unhealthy SCM state regression — 2026-05-24 StopPending bug
# ---------------------------------------------------------------------------
# Backstory: _stop_bot_child reports SERVICE_STOP_PENDING while reaping a
# dead bot (needed so SCM's 30s stop-deadline doesn't fire mid-reap).
# Without an explicit restore-to-RUNNING at the end of _handle_unhealthy,
# SCM was permanently stuck in StopPending after the first respawn —
# caught over the weekend when Get-Service showed StopPending
# despite the bot being alive + supervisor functional.
# Fix adds a try/finally in _handle_unhealthy that restores SERVICE_RUNNING
# on every exit path (suppressed/respawn-success/spawn-failed), guarded by
# self._stop_requested to avoid racing genuine SvcStop.

def _make_fake_service_instance(svc_name="PancakeBotTest", mode="live", stop_requested=False):
    """Build a _PancakeBotServiceBase without invoking pywin32 __init__
    (ServiceFramework needs SCM context which we don't have in tests)."""
    from pancakebot.service.common import _PancakeBotServiceBase
    svc = _PancakeBotServiceBase.__new__(_PancakeBotServiceBase)
    svc._MODE = mode
    svc._OTHER_SERVICE = "PancakeBotDry" if mode == "live" else "PancakeBotLive"
    svc._svc_name_ = svc_name
    svc._stop_requested = stop_requested
    svc._bot_proc = None
    svc._bot_started_at = None
    return svc


def _mock_supervision_and_notifications(monkeypatch, fast_count=0, slow_count=0):
    monkeypatch.setattr(supervision, "read_restart_history", lambda p: [])
    monkeypatch.setattr(supervision, "prune_history", lambda h, now, w: h)
    monkeypatch.setattr(supervision, "count_within",
                        lambda h, now, w: fast_count if w < 3600 else slow_count)
    monkeypatch.setattr(supervision, "write_restart_history", lambda p, h: None)
    monkeypatch.setattr(notifications, "notify", lambda **kw: "DISABLED")


def test_handle_unhealthy_restores_running_after_successful_respawn(monkeypatch, tmp_path):
    """The exact regression scenario: CRASHED detected -> _stop_bot_child
    (pushes STOP_PENDING) -> _spawn_bot_child (success) -> finally restores
    SERVICE_RUNNING. Last ReportServiceStatus must be SERVICE_RUNNING."""
    import win32service
    svc = _make_fake_service_instance()
    _mock_supervision_and_notifications(monkeypatch)

    status_log = []
    svc.ReportServiceStatus = lambda s: status_log.append(s)

    # _stop_bot_child stub: simulate what the real one does to SCM
    def stub_stop(reason):
        svc.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        svc._bot_proc = None
        svc._bot_started_at = None
    svc._stop_bot_child = stub_stop
    # _spawn_bot_child stub: simulate a successful spawn
    class FakeProc:
        pid = 12345
    def stub_spawn(art):
        svc._bot_proc = FakeProc()
        svc._bot_started_at = time.time()
    svc._spawn_bot_child = stub_spawn
    svc._archive_stale_crash = lambda crash: None

    art = _make_mode_tree(tmp_path, "live")
    svc._handle_unhealthy("CRASHED", {}, art)

    assert win32service.SERVICE_STOP_PENDING in status_log, \
        "_stop_bot_child stub should have logged STOP_PENDING"
    assert status_log[-1] == win32service.SERVICE_RUNNING, \
        f"last ReportServiceStatus must be SERVICE_RUNNING, got {status_log[-1]}; full: {status_log}"


def test_handle_unhealthy_restores_running_after_fast_crashloop_suppression(monkeypatch, tmp_path):
    """Even on the fast-crashloop suppression branch (no respawn attempted),
    the finally block must restore SERVICE_RUNNING so SCM doesn't drift if
    a prior respawn left it in STOP_PENDING."""
    import win32service
    svc = _make_fake_service_instance()
    _mock_supervision_and_notifications(monkeypatch, fast_count=10)  # well past _FAST_RESTART_MAX=3

    status_log = []
    svc.ReportServiceStatus = lambda s: status_log.append(s)
    svc._stop_bot_child = lambda reason: None
    svc._spawn_bot_child = lambda art: None
    svc._archive_stale_crash = lambda crash: None

    art = _make_mode_tree(tmp_path, "live")
    svc._handle_unhealthy("CRASHED", {}, art)

    assert status_log[-1] == win32service.SERVICE_RUNNING, \
        f"fast-suppress path must still end with SERVICE_RUNNING; got {status_log}"


def test_handle_unhealthy_restores_running_after_spawn_failure(monkeypatch, tmp_path):
    """If _spawn_bot_child raises, supervisor is still alive and will retry
    next tick — SCM must reflect that with SERVICE_RUNNING, not stay in
    the STOP_PENDING set by the preceding _stop_bot_child."""
    import win32service
    svc = _make_fake_service_instance()
    _mock_supervision_and_notifications(monkeypatch)

    status_log = []
    svc.ReportServiceStatus = lambda s: status_log.append(s)

    def stub_stop(reason):
        svc.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        svc._bot_proc = None
    svc._stop_bot_child = stub_stop
    def stub_spawn_fail(art):
        raise RuntimeError("simulated spawn failure")
    svc._spawn_bot_child = stub_spawn_fail
    svc._archive_stale_crash = lambda crash: None

    art = _make_mode_tree(tmp_path, "live")
    svc._handle_unhealthy("CRASHED", {}, art)  # must not raise on CRASHED unhealthy status

    assert win32service.SERVICE_STOP_PENDING in status_log
    assert status_log[-1] == win32service.SERVICE_RUNNING, \
        f"spawn-failed path must still end with SERVICE_RUNNING; got {status_log}"


def test_handle_unhealthy_does_NOT_restore_running_when_stop_requested(monkeypatch, tmp_path):
    """Race protection: if SvcStop was called (sets _stop_requested=True)
    concurrently with _handle_unhealthy, the finally block must NOT flip
    SCM back to SERVICE_RUNNING — the framework will report STOPPED when
    SvcDoRun returns, and we don't want to fight that."""
    import win32service
    svc = _make_fake_service_instance(stop_requested=True)
    _mock_supervision_and_notifications(monkeypatch)

    status_log = []
    svc.ReportServiceStatus = lambda s: status_log.append(s)

    def stub_stop(reason):
        svc.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        svc._bot_proc = None
    svc._stop_bot_child = stub_stop
    class FakeProc:
        pid = 12345
    def stub_spawn(art):
        svc._bot_proc = FakeProc()
        svc._bot_started_at = time.time()
    svc._spawn_bot_child = stub_spawn
    svc._archive_stale_crash = lambda crash: None

    art = _make_mode_tree(tmp_path, "live")
    svc._handle_unhealthy("CRASHED", {}, art)

    # The finally block's guard should have skipped the RUNNING report.
    # Last status is therefore the STOP_PENDING from _stop_bot_child.
    assert win32service.SERVICE_RUNNING not in status_log, \
        f"finally must not flip to RUNNING when _stop_requested=True; got {status_log}"


# ---------------------------------------------------------------------------
# Step 27c-B: tests for restart-pattern Discord aggregation (3-in-1h)
# ---------------------------------------------------------------------------
#
# Policy: Discord notifies on CRASHED/DOWN respawns ONLY when the
# `_handle_unhealthy` invocation is the 3rd+ within a 1-hour rolling window.
# First two restarts log to SCM only. Implementation:
#
#     recent_restarts_1h = supervision.count_within(history, now, 3600.0)
#     should_notify_status = recent_restarts_1h >= 2  # this would be the 3rd
#
# These tests intercept notifications.notify to capture every (kind, mode,
# fields) tuple, then verify only the status-restart aggregation behavior
# matches policy. SUPPRESSED_FAST_CRASHLOOP and SLOW_CRASHLOOP_WARNING are
# separate notification paths (independent of restart-pattern aggregation).

def _patch_for_aggregation(monkeypatch, recent_1h_count: int, slow_24h_count: int = 0,
                            fast_15m_count: int = 0):
    """Patch supervision/notifications and return a list of captured notify calls.

    The mocked ``count_within`` returns:
      - fast_15m_count   for window_s = 900   (_FAST_RESTART_WINDOW_S)
      - recent_1h_count  for window_s = 3600  (restart-pattern aggregation)
      - slow_24h_count   for window_s = 86400 (_SLOW_RESTART_WINDOW_S)

    Returns the captured-calls list. Each entry is a dict with the kwargs
    notifications.notify was called with.
    """
    captured: list[dict[str, object]] = []

    def fake_count_within(history, now, window_s):
        if window_s == 900.0 or window_s == 900:
            return fast_15m_count
        if window_s == 3600.0 or window_s == 3600:
            return recent_1h_count
        if window_s == 86400.0 or window_s == 86400:
            return slow_24h_count
        return 0

    def fake_notify(**kwargs):
        captured.append(dict(kwargs))
        return "DISABLED"

    monkeypatch.setattr(supervision, "read_restart_history", lambda p: [])
    monkeypatch.setattr(supervision, "prune_history", lambda h, now, w: h)
    monkeypatch.setattr(supervision, "count_within", fake_count_within)
    monkeypatch.setattr(supervision, "write_restart_history", lambda p, h: None)
    monkeypatch.setattr(notifications, "notify", fake_notify)

    return captured


def _make_svc_with_stubs():
    """Common service instance + stubs for aggregation tests."""
    import win32service
    svc = _make_fake_service_instance()
    svc.ReportServiceStatus = lambda s: None  # don't care about SCM here

    def stub_stop(reason):
        svc._bot_proc = None
    svc._stop_bot_child = stub_stop

    class FakeProc:
        pid = 12345
    def stub_spawn(art):
        svc._bot_proc = FakeProc()
        svc._bot_started_at = time.time()
    svc._spawn_bot_child = stub_spawn
    svc._archive_stale_crash = lambda crash: None
    return svc


def test_aggregation_first_restart_in_1h_does_NOT_fire_discord(monkeypatch, tmp_path):
    """1st restart in window (recent_1h=0): no CRASHED Discord notify."""
    captured = _patch_for_aggregation(monkeypatch, recent_1h_count=0)
    svc = _make_svc_with_stubs()
    art = _make_mode_tree(tmp_path, "live")

    svc._handle_unhealthy("CRASHED", {}, art)

    # No CRASHED-status notify should fire. (SLOW_CRASHLOOP_WARNING is gated
    # separately at slow_count>=8; fast-suppress at fast_count>=3. Neither
    # fires here.)
    crashed_calls = [c for c in captured if c.get("kind") == "CRASHED"]
    assert len(crashed_calls) == 0, \
        f"1st restart in 1h must not fire Discord CRASHED; got {captured}"


def test_aggregation_second_restart_in_1h_does_NOT_fire_discord(monkeypatch, tmp_path):
    """2nd restart (recent_1h=1, meaning 1 prior restart in last hour): still no Discord."""
    captured = _patch_for_aggregation(monkeypatch, recent_1h_count=1)
    svc = _make_svc_with_stubs()
    art = _make_mode_tree(tmp_path, "live")

    svc._handle_unhealthy("CRASHED", {}, art)

    crashed_calls = [c for c in captured if c.get("kind") == "CRASHED"]
    assert len(crashed_calls) == 0, \
        f"2nd restart in 1h must not fire Discord CRASHED; got {captured}"


def test_aggregation_third_restart_in_1h_FIRES_discord(monkeypatch, tmp_path):
    """3rd restart (recent_1h=2 prior + this one): Discord CRASHED fires."""
    captured = _patch_for_aggregation(monkeypatch, recent_1h_count=2)
    svc = _make_svc_with_stubs()
    art = _make_mode_tree(tmp_path, "live")

    svc._handle_unhealthy("CRASHED", {}, art)

    crashed_calls = [c for c in captured if c.get("kind") == "CRASHED"]
    assert len(crashed_calls) == 1, \
        f"3rd restart in 1h must fire Discord CRASHED exactly once; got {captured}"
    assert crashed_calls[0]["mode"] == "live"


def test_aggregation_fourth_restart_in_1h_ALSO_FIRES_discord(monkeypatch, tmp_path):
    """4th restart (recent_1h=3 prior): Discord continues to fire."""
    captured = _patch_for_aggregation(monkeypatch, recent_1h_count=3)
    svc = _make_svc_with_stubs()
    art = _make_mode_tree(tmp_path, "live")

    svc._handle_unhealthy("DOWN", {}, art)

    down_calls = [c for c in captured if c.get("kind") == "DOWN"]
    assert len(down_calls) == 1, \
        f"4th restart in 1h must continue to fire Discord; got {captured}"
    assert down_calls[0]["mode"] == "live"


def test_aggregation_independent_of_slow_crashloop_warning(monkeypatch, tmp_path):
    """SLOW_CRASHLOOP_WARNING (8+ restarts in 24h) fires regardless of the
    1h aggregation. With recent_1h=0 (no Discord for status) but slow_24h=8
    (escalate_slow=True), we should see SLOW_CRASHLOOP_WARNING but NOT
    CRASHED.
    """
    captured = _patch_for_aggregation(monkeypatch, recent_1h_count=0, slow_24h_count=8)
    svc = _make_svc_with_stubs()
    art = _make_mode_tree(tmp_path, "live")

    svc._handle_unhealthy("CRASHED", {}, art)

    crashed_calls = [c for c in captured if c.get("kind") == "CRASHED"]
    slow_calls = [c for c in captured if c.get("kind") == "SLOW_CRASHLOOP_WARNING"]
    assert len(crashed_calls) == 0, \
        f"recent_1h=0 must not fire CRASHED Discord; got {captured}"
    assert len(slow_calls) == 1, \
        f"slow_24h=8 must fire SLOW_CRASHLOOP_WARNING; got {captured}"
