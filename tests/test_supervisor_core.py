"""Unit tests for the OS-agnostic SupervisorCore (cross-platform supervision).

Uses a FakePlatform (no pywin32/systemd) + monkeypatched subprocess/notify so
the same logic that runs on Windows and Linux is exercised directly: mode
mutex, REBOOTED classification, crashloop limiter, and Discord alert dispatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.service import supervisor_core as sc  # noqa: E402
from pancakebot.service.platform_base import HealthState  # noqa: E402


# -- fakes -----------------------------------------------------------------


class FakeStopEvent:
    def __init__(self):
        self._set = False
        self.waits = 0

    def wait(self, _t):
        self.waits += 1
        return self._set

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


class FakeKillTree:
    def __init__(self):
        self.adopted = []

    def adopt(self, proc):
        self.adopted.append(proc)


class FakePlatform:
    def __init__(self, active=()):
        self.active = set(active)
        self.health = []
        self.stopped = []
        self.stop_event = FakeStopEvent()

    def create_stop_event(self):
        return self.stop_event

    def create_kill_tree(self):
        return FakeKillTree()

    def signal_health(self, state):
        self.health.append(state)

    def spawn_kwargs(self):
        return {}

    def is_service_active(self, name):
        return name in self.active

    def stop_service(self, name, timeout_s=30.0):
        self.stopped.append(name)
        self.active.discard(name)
        return True


class FakeProc:
    def __init__(self, pid=4321, poll_seq=None):
        self.pid = pid
        self._poll_seq = list(poll_seq or [None])
        self.terminated = False
        self.killed = False

    def poll(self):
        if self.terminated or self.killed:
            return 0  # dead after terminate/kill -> drain loop exits promptly
        return self._poll_seq.pop(0) if len(self._poll_seq) > 1 else self._poll_seq[0]

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def _make_core(tmp_path, monkeypatch, *, mode="dry", active=()):
    alerts = []
    monkeypatch.setattr(sc.notifications, "notify",
                        lambda **kw: alerts.append((kw.get("mode"), kw.get("kind"))))
    art = {
        "logs_dir": tmp_path / "logs",
        "crash": tmp_path / "crash.json",
        "restart_history": tmp_path / "restart_history.jsonl",
    }
    monkeypatch.setattr(sc.supervision, "artifacts_for_mode", lambda m: art)
    monkeypatch.setattr(sc, "subprocess",
                        type("S", (), {"Popen": lambda *a, **k: FakeProc()})())
    plat = FakePlatform(active=active)
    other = "pancakebot-live" if mode == "dry" else "pancakebot-dry"
    core = sc.SupervisorCore(
        mode=mode, other_service=other, platform=plat,
        repo_root=tmp_path, venv_python=tmp_path / "py", service_name=f"svc-{mode}",
        log=lambda lvl, m: None,
    )
    return core, plat, alerts, art


# -- mode mutex ------------------------------------------------------------


def test_dry_refuses_when_live_active(tmp_path, monkeypatch):
    core, plat, alerts, _ = _make_core(tmp_path, monkeypatch, mode="dry",
                                       active=("pancakebot-live",))
    assert core._enforce_mode_mutex() is False


def test_dry_proceeds_when_live_inactive(tmp_path, monkeypatch):
    core, plat, alerts, _ = _make_core(tmp_path, monkeypatch, mode="dry")
    assert core._enforce_mode_mutex() is True


def test_live_evicts_dry(tmp_path, monkeypatch):
    core, plat, alerts, _ = _make_core(tmp_path, monkeypatch, mode="live",
                                       active=("pancakebot-dry",))
    assert core._enforce_mode_mutex() is True
    assert "pancakebot-dry" in plat.stopped               # stop_service called
    assert ("live", "MODE_TRANSITION") in alerts          # eviction alert


# -- first-run classification ---------------------------------------------


def test_classify_rebooted_on_fresh_uptime(tmp_path, monkeypatch):
    core, _, _, art = _make_core(tmp_path, monkeypatch)
    monkeypatch.setattr("psutil.boot_time", lambda: __import__("time").time() - 60)  # 60s up
    assert core._classify_first_run(art) == "REBOOTED"


def test_classify_recovery_when_crash_present(tmp_path, monkeypatch):
    core, _, _, art = _make_core(tmp_path, monkeypatch)
    monkeypatch.setattr("psutil.boot_time", lambda: __import__("time").time() - 99999)  # warm
    art["crash"].write_text("{}", encoding="utf-8")
    assert core._classify_first_run(art) == "RECOVERY_AFTER_CRASH"


def test_classify_started_when_warm_no_crash(tmp_path, monkeypatch):
    core, _, _, art = _make_core(tmp_path, monkeypatch)
    monkeypatch.setattr("psutil.boot_time", lambda: __import__("time").time() - 99999)
    assert core._classify_first_run(art) == "STARTED"


# -- crashloop limiter -----------------------------------------------------


def test_fast_crashloop_suppressed(tmp_path, monkeypatch):
    core, plat, alerts, art = _make_core(tmp_path, monkeypatch)
    import time as _t
    now = _t.time()
    # Seed >= _FAST_RESTART_MAX recent restarts.
    hist = [{"ts_wall": now - 10, "trigger": "CRASHED"} for _ in range(sc._FAST_RESTART_MAX)]
    sc.supervision.write_restart_history(art["restart_history"], hist)
    core._handle_unhealthy("CRASHED", {}, art)
    assert any(k == "SUPPRESSED_FAST_CRASHLOOP" for _, k in alerts)
    # No respawn occurred (bot_proc stays None).
    assert core._bot_proc is None


def test_unhealthy_respawns_and_records(tmp_path, monkeypatch):
    core, plat, alerts, art = _make_core(tmp_path, monkeypatch)
    core._handle_unhealthy("DOWN", {}, art)
    assert core._bot_proc is not None                     # respawned
    hist = sc.supervision.read_restart_history(art["restart_history"])
    assert len(hist) == 1 and hist[0]["trigger"] == "DOWN"
    # SCM/systemd restored to RUNNING after the transient reap.
    assert HealthState.READY in plat.health


def test_slow_crashloop_warning(tmp_path, monkeypatch):
    core, plat, alerts, art = _make_core(tmp_path, monkeypatch)
    import time as _t
    now = _t.time()
    # >= _SLOW_RESTART_MAX over 24h but < _FAST_RESTART_MAX in 15min (spread out).
    hist = [{"ts_wall": now - (i + 1) * 3600, "trigger": "CRASHED"}
            for i in range(sc._SLOW_RESTART_MAX)]
    sc.supervision.write_restart_history(art["restart_history"], hist)
    core._handle_unhealthy("CRASHED", {}, art)
    assert any(k == "SLOW_CRASHLOOP_WARNING" for _, k in alerts)


# -- run loop + stop -------------------------------------------------------


def test_run_spawns_then_stops_clean(tmp_path, monkeypatch):
    core, plat, alerts, art = _make_core(tmp_path, monkeypatch, mode="dry")
    monkeypatch.setattr("psutil.boot_time", lambda: __import__("time").time() - 99999)
    monkeypatch.setattr(sc.supervision, "classify_running_bot",
                        lambda *a, **k: ("UP", {}))
    # Pre-set the stop event so the loop runs one iteration then exits.
    plat.stop_event.set()
    core.run()
    assert ("dry", "STARTED") in alerts                   # first-run alert
    assert ("dry", "STOPPED") in alerts                   # clean exit alert
    assert core._bot_proc is None                          # child drained on stop


def test_request_stop_is_minimal(tmp_path, monkeypatch):
    core, plat, alerts, _ = _make_core(tmp_path, monkeypatch)
    core.request_stop()
    assert core._stop_requested is True
    assert plat.stop_event.is_set() is True
    assert HealthState.STOPPING in plat.health
