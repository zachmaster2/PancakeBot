"""Tests for the cross-platform ServicePlatform abstraction + both adapters.

The Linux adapter is fully exercised via a mocked systemctl runner (runs on any
OS). The Windows adapter's pywin32-backed primitives run on Windows; its sc.exe
management is mocked. The lock-file primitives are OS-specific (skip the other).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.service.platform_base import (  # noqa: E402
    HealthState,
    ServiceSpec,
    ServiceState,
)
from pancakebot.service.linux_platform import LinuxServicePlatform  # noqa: E402

_IS_WIN = sys.platform == "win32"
_IS_LINUX = sys.platform.startswith("linux")


def _spec(name="pancakebot-live", conflicts="pancakebot-dry"):
    return ServiceSpec(
        name=name, description="PancakeBot Live", exe_path="/opt/pb/.venv/bin/python",
        args=("run.py", "--live"), working_dir="/opt/pb",
        env={"PB_X": "1"}, conflicts_with=conflicts,
        restart_max_attempts=3, restart_reset_window_s=86400, restart_delay_s=60,
    )


class _FakeRunner:
    """Records systemctl/sc.exe invocations; returns canned results by matcher."""
    def __init__(self, responses=None):
        self.calls: list[list[str]] = []
        self._responses = responses or {}

    def __call__(self, args):
        self.calls.append(list(args))
        for needle, (stdout, rc) in self._responses.items():
            if needle in args:
                return subprocess.CompletedProcess(args, rc, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


# --------------------------------------------------------------------------
# Linux adapter
# --------------------------------------------------------------------------


def test_linux_render_unit_has_key_directives():
    p = LinuxServicePlatform(runner=_FakeRunner())
    unit = p.render_unit(_spec())
    assert "Type=simple" in unit       # systemd supervises run.py directly
    assert "EnvironmentFile=-/etc/pancakebot/pancakebot.env" in unit
    assert "Environment=MALLOC_ARENA_MAX=2" in unit   # glibc RSS-ratchet fix
    assert "ExecStart=/opt/pb/.venv/bin/python run.py --live" in unit
    assert "KillMode=control-group" in unit                  # Job Object equivalent
    assert "Restart=on-failure" in unit
    assert "StartLimitBurst=3" in unit
    assert "Conflicts=pancakebot-dry.service" in unit         # mode mutex
    assert "After=network-online.target nss-lookup.target" in unit
    assert "WantedBy=multi-user.target" in unit
    assert "Environment=PB_X=1" in unit


def test_linux_install_writes_unit_and_reloads(tmp_path):
    r = _FakeRunner()
    p = LinuxServicePlatform(unit_dir=tmp_path, runner=r)
    p.install_service(_spec())
    assert (tmp_path / "pancakebot-live.service").exists()
    assert ["systemctl", "daemon-reload"] in r.calls


def test_linux_start_stop_status_mapping():
    r = _FakeRunner(responses={"show": ("active\n", 0)})
    p = LinuxServicePlatform(runner=r)
    p.start_service("pancakebot-live")
    assert ["systemctl", "start", "pancakebot-live.service"] in r.calls
    assert p.service_status("pancakebot-live") == ServiceState.RUNNING


def test_linux_status_states():
    for token, expected in [
        ("active", ServiceState.RUNNING), ("activating", ServiceState.STARTING),
        ("deactivating", ServiceState.STOPPING), ("inactive", ServiceState.STOPPED),
        ("failed", ServiceState.FAILED), ("garbage", ServiceState.UNKNOWN),
    ]:
        p = LinuxServicePlatform(runner=_FakeRunner(responses={"show": (token + "\n", 0)}))
        assert p.service_status("svc") == expected


def test_linux_is_service_active():
    p = LinuxServicePlatform(runner=_FakeRunner(responses={"is-active": ("active\n", 0)}))
    assert p.is_service_active("svc") is True
    p2 = LinuxServicePlatform(runner=_FakeRunner(responses={"is-active": ("inactive\n", 3)}))
    assert p2.is_service_active("svc") is False


def test_linux_enable_disable():
    r = _FakeRunner()
    p = LinuxServicePlatform(runner=r)
    p.enable_auto_start("pancakebot-live")
    p.disable_auto_start("pancakebot-live")
    assert ["systemctl", "enable", "pancakebot-live.service"] in r.calls
    assert ["systemctl", "disable", "pancakebot-live.service"] in r.calls


def test_linux_clear_restart_counter_calls_reset_failed():
    # E: intentional-restart handshake — Linux clears systemd's start counter.
    r = _FakeRunner()
    p = LinuxServicePlatform(runner=r)
    p.clear_restart_counter("pancakebot-live")
    assert ["systemctl", "reset-failed", "pancakebot-live.service"] in r.calls


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_clear_restart_counter_is_noop():
    # E: Windows inherits the base no-op — SCM `failureflag 1` already counts
    # only non-clean exits, so an intentional stop never increments the counter.
    from pancakebot.service.windows_platform import WindowsServicePlatform
    r = _FakeRunner()
    p = WindowsServicePlatform(runner=r)
    p.clear_restart_counter("PancakeBotLive")
    assert r.calls == []          # nothing to reset -> no sc.exe call


def test_linux_signal_health_payload_mapping():
    p = LinuxServicePlatform(runner=_FakeRunner())
    seen = []
    with mock.patch.object(p, "_sd_notify", side_effect=seen.append):
        p.signal_health(HealthState.READY)
        p.signal_health(HealthState.RUNNING)
        p.signal_health(HealthState.STOPPING)
        p.signal_health(HealthState.EXTEND)
    assert seen[0] == "READY=1"
    assert seen[1] == "WATCHDOG=1"
    assert seen[2] == "STOPPING=1"
    assert seen[3].startswith("EXTEND_TIMEOUT_USEC=")


def test_linux_sd_notify_noop_without_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    # Must not raise / touch sockets when NOTIFY_SOCKET is unset.
    LinuxServicePlatform._sd_notify("READY=1")


def test_linux_spawn_kwargs_and_killtree():
    p = LinuxServicePlatform(runner=_FakeRunner())
    assert p.spawn_kwargs() == {"start_new_session": True}
    kt = p.create_kill_tree()
    kt.adopt(mock.Mock(pid=123))   # no-op recorder under systemd cgroup
    assert kt.adopted_pids == [123]


def test_linux_restart_and_deps_dropins(tmp_path):
    r = _FakeRunner()
    p = LinuxServicePlatform(unit_dir=tmp_path, runner=r)
    p.set_restart_on_failure("pancakebot-live", max_attempts=5, reset_window_s=3600, delay_s=30)
    p.set_service_dependencies("pancakebot-live", requires_network=True)
    d = tmp_path / "pancakebot-live.service.d"
    assert (d / "restart.conf").read_text().count("StartLimitBurst=5") == 1
    assert "network-online.target" in (d / "deps.conf").read_text()


@pytest.mark.skipif(not _IS_LINUX, reason="fcntl lock is Linux-only")
def test_linux_exclusive_lock_mutex(tmp_path):
    p = LinuxServicePlatform(runner=_FakeRunner())
    lock = str(tmp_path / "mode.lock")
    with p.acquire_exclusive_lock(lock) as got_outer:
        assert got_outer is True
        with p.acquire_exclusive_lock(lock) as got_inner:
            assert got_inner is False   # already held


# --------------------------------------------------------------------------
# Windows adapter
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_signal_health_maps_to_scm():
    import win32service
    from pancakebot.service.windows_platform import WindowsServicePlatform
    reporter = mock.Mock()
    p = WindowsServicePlatform(status_reporter=reporter)
    p.signal_health(HealthState.READY)
    p.signal_health(HealthState.STOPPING)
    assert reporter.call_args_list[0].args[0] == win32service.SERVICE_RUNNING
    assert reporter.call_args_list[1].args[0] == win32service.SERVICE_STOP_PENDING


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_signal_health_noop_without_reporter():
    from pancakebot.service.windows_platform import WindowsServicePlatform
    WindowsServicePlatform().signal_health(HealthState.READY)  # must not raise


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_spawn_kwargs_has_creationflags():
    from pancakebot.service.windows_platform import WindowsServicePlatform
    kw = WindowsServicePlatform().spawn_kwargs()
    assert "creationflags" in kw
    assert kw["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_killtree_adopt_failure_is_nonfatal():
    from pancakebot.service.windows_platform import WindowsServicePlatform
    logs = []
    p = WindowsServicePlatform(log=logs.append)
    kt = p.create_kill_tree()       # real Job Object
    kt.adopt(mock.Mock(pid=999, _handle=0))   # bogus handle -> logged, no raise
    assert any("AssignProcessToJobObject failed" in m for m in logs)


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_management_uses_sc_exe():
    from pancakebot.service.windows_platform import WindowsServicePlatform
    r = _FakeRunner()
    p = WindowsServicePlatform(runner=r)
    p.enable_auto_start("PancakeBotLive")
    p.set_restart_on_failure("PancakeBotLive", max_attempts=3, reset_window_s=86400, delay_s=60)
    p.set_service_dependencies("PancakeBotLive", requires_network=True)
    flat = [" ".join(c) for c in r.calls]
    assert any("sc.exe config PancakeBotLive start= auto" in c for c in flat)
    assert any("sc.exe failure PancakeBotLive" in c for c in flat)
    assert any("depend= Dnscache/NlaSvc" in c for c in flat)


@pytest.mark.skipif(not _IS_WIN, reason="Windows adapter needs pywin32")
def test_windows_service_status_mapping(monkeypatch):
    import win32service
    from pancakebot.service import windows_platform as wp
    p = wp.WindowsServicePlatform(runner=_FakeRunner())
    monkeypatch.setattr(wp, "query_service_state", lambda n: win32service.SERVICE_RUNNING)
    assert p.service_status("X") == ServiceState.RUNNING
    monkeypatch.setattr(wp, "query_service_state", lambda n: None)
    assert p.service_status("X") == ServiceState.UNKNOWN


@pytest.mark.skipif(not _IS_WIN, reason="msvcrt lock is Windows-only")
def test_windows_exclusive_lock_mutex(tmp_path):
    from pancakebot.service.windows_platform import WindowsServicePlatform
    p = WindowsServicePlatform()
    lock = str(tmp_path / "mode.lock")
    with p.acquire_exclusive_lock(lock) as got_outer:
        assert got_outer is True
        with p.acquire_exclusive_lock(lock) as got_inner:
            assert got_inner is False


# --------------------------------------------------------------------------
# Factory + cross-adapter parity
# --------------------------------------------------------------------------


def test_factory_selects_by_os():
    from pancakebot.service import get_platform, reset_platform_cache
    reset_platform_cache()
    p = get_platform()
    assert p.name == ("windows" if _IS_WIN else "linux")
    assert get_platform() is p          # cached
    reset_platform_cache()


def test_cross_adapter_status_parity_normalized():
    """Both adapters normalize a 'running' service to ServiceState.RUNNING and
    an absent one to UNKNOWN — identical contract regardless of OS mechanism."""
    lin = LinuxServicePlatform(runner=_FakeRunner(responses={"show": ("active\n", 0)}))
    assert lin.service_status("svc") == ServiceState.RUNNING
    lin_absent = LinuxServicePlatform(runner=_FakeRunner(responses={"show": ("\n", 0)}))
    assert lin_absent.service_status("svc") == ServiceState.UNKNOWN
    if _IS_WIN:
        import win32service
        from pancakebot.service import windows_platform as wp
        p = wp.WindowsServicePlatform(runner=_FakeRunner())
        with mock.patch.object(wp, "query_service_state", lambda n: win32service.SERVICE_RUNNING):
            assert p.service_status("svc") == ServiceState.RUNNING
        with mock.patch.object(wp, "query_service_state", lambda n: None):
            assert p.service_status("svc") == ServiceState.UNKNOWN
