"""Windows (SCM / pywin32) implementation of ``ServicePlatform``.

The supervisor runtime primitives (health signaling, Job-Object kill-tree, SCM
mode-mutex query, detached-spawn flags) are extracted verbatim from the prior
inline code in ``common.py`` — behaviour-identical, so the supervisor delegates
to this adapter without changing what the running service does. Service
management (install/enable/restart-policy/deps/start/stop/status) is via
``sc.exe`` shell-outs, mirroring ``scripts/install_services.ps1``; these are
used by the bootstrap/control tooling, not the running supervision loop.

pywin32 is imported at module load — intentional, since this adapter is only
selected on win32 (``pancakebot/service/__init__.py``).
"""
from __future__ import annotations

import contextlib
import subprocess
import time
from typing import Callable

import win32job
import win32service

from pancakebot.service.platform_base import (
    HealthState,
    KillTree,
    ServicePlatform,
    ServiceSpec,
    ServiceState,
)

# SCM CurrentState -> normalized ServiceState.
_SCM_STATE_MAP = {
    win32service.SERVICE_RUNNING: ServiceState.RUNNING,
    win32service.SERVICE_START_PENDING: ServiceState.STARTING,
    win32service.SERVICE_STOP_PENDING: ServiceState.STOPPING,
    win32service.SERVICE_STOPPED: ServiceState.STOPPED,
    win32service.SERVICE_PAUSED: ServiceState.RUNNING,
}

# States that count as "active" for the mode mutex.
_ACTIVE_SCM_STATES = (
    win32service.SERVICE_RUNNING,
    win32service.SERVICE_START_PENDING,
    win32service.SERVICE_CONTINUE_PENDING,
    win32service.SERVICE_PAUSED,
    win32service.SERVICE_PAUSE_PENDING,
)


def query_service_state(svc_name: str) -> int | None:
    """Current SCM state (a SERVICE_* constant) or None if absent.

    Extracted verbatim from common.py:_query_service_state."""
    try:
        scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    except Exception:
        return None
    try:
        try:
            svc = win32service.OpenService(scm, svc_name, win32service.SERVICE_QUERY_STATUS)
        except Exception:
            return None
        try:
            status = win32service.QueryServiceStatusEx(svc)
            return int(status["CurrentState"])
        finally:
            win32service.CloseServiceHandle(svc)
    finally:
        win32service.CloseServiceHandle(scm)


def stop_service_and_wait(svc_name: str, timeout_s: float) -> bool:
    """ControlService(STOP) then poll until STOPPED or timeout.

    Extracted verbatim from common.py:_stop_service_and_wait."""
    try:
        scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    except Exception:
        return False
    try:
        try:
            svc = win32service.OpenService(
                scm, svc_name,
                win32service.SERVICE_STOP | win32service.SERVICE_QUERY_STATUS,
            )
        except Exception:
            return False
        try:
            try:
                win32service.ControlService(svc, win32service.SERVICE_CONTROL_STOP)
            except Exception:
                pass
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    status = win32service.QueryServiceStatusEx(svc)
                    if int(status["CurrentState"]) == win32service.SERVICE_STOPPED:
                        return True
                except Exception:
                    return False
                time.sleep(0.5)
            return False
        finally:
            win32service.CloseServiceHandle(svc)
    finally:
        win32service.CloseServiceHandle(scm)


class _JobObjectKillTree(KillTree):
    """Windows Job Object with KILL_ON_JOB_CLOSE — when the supervisor process
    exits for ANY reason, the kernel kills every enrolled child.

    Extracted verbatim from common.py:_create_kill_on_close_job + the
    AssignProcessToJobObject enrolment."""

    def __init__(self, *, log: Callable[[str], None] | None = None) -> None:
        self._log = log or (lambda _m: None)
        job = win32job.CreateJobObject(None, "")  # unnamed; held by us only
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation,
        )
        info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, info,
        )
        self._job = job

    @property
    def handle(self):
        return self._job

    def adopt(self, proc) -> None:  # noqa: ANN001
        try:
            win32job.AssignProcessToJobObject(self._job, int(proc._handle))
        except Exception as e:  # noqa: BLE001
            self._log(
                f"AssignProcessToJobObject failed for pid={getattr(proc, 'pid', '?')}: "
                f"{type(e).__name__}: {e} — child orphaned if supervisor crashes"
            )


class WindowsServicePlatform(ServicePlatform):
    name = "windows"

    def __init__(
        self,
        *,
        status_reporter: Callable[[int], None] | None = None,
        log: Callable[[str], None] | None = None,
        runner=None,
    ) -> None:
        # status_reporter = the ServiceFramework's bound ReportServiceStatus,
        # set when running inside the service host; None (no-op) otherwise.
        self._status_reporter = status_reporter
        self._log = log or (lambda _m: None)
        self._run = runner if runner is not None else self._default_run

    @staticmethod
    def _default_run(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def _sc(self, *args: str) -> subprocess.CompletedProcess:
        return self._run(["sc.exe", *args])

    # -- supervisor runtime primitives -------------------------------------

    def signal_health(self, state: HealthState) -> None:
        if self._status_reporter is None:
            return
        scm_state = {
            HealthState.READY: win32service.SERVICE_RUNNING,
            HealthState.RUNNING: win32service.SERVICE_RUNNING,
            HealthState.STOPPING: win32service.SERVICE_STOP_PENDING,
            HealthState.EXTEND: win32service.SERVICE_STOP_PENDING,
        }.get(state)
        if scm_state is None:
            return
        try:
            self._status_reporter(scm_state)
        except Exception:  # noqa: BLE001
            pass

    def create_kill_tree(self) -> KillTree:
        return _JobObjectKillTree(log=self._log)

    def spawn_kwargs(self) -> dict:
        return {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        }

    def is_service_active(self, service_name: str) -> bool:
        return query_service_state(service_name) in _ACTIVE_SCM_STATES

    # -- management (sc.exe; bootstrap/control tooling) --------------------

    def install_service(self, spec: ServiceSpec) -> None:
        bin_path = " ".join([spec.exe_path, *spec.args])
        self._sc("create", spec.name, f"binPath= {bin_path}", "start= demand",
                 f"DisplayName= {spec.description}")
        self.set_restart_on_failure(
            spec.name, max_attempts=spec.restart_max_attempts,
            reset_window_s=spec.restart_reset_window_s, delay_s=spec.restart_delay_s,
        )
        self.set_service_dependencies(spec.name, requires_network=spec.requires_network)

    def uninstall_service(self, service_name: str) -> None:
        self._sc("delete", service_name)

    def start_service(self, service_name: str) -> None:
        self._sc("start", service_name)

    def stop_service(self, service_name: str, timeout_s: float = 30.0) -> bool:
        return stop_service_and_wait(service_name, timeout_s)

    def service_status(self, service_name: str) -> ServiceState:
        st = query_service_state(service_name)
        if st is None:
            return ServiceState.UNKNOWN
        return _SCM_STATE_MAP.get(st, ServiceState.UNKNOWN)

    def enable_auto_start(self, service_name: str) -> None:
        self._sc("config", service_name, "start= auto")

    def disable_auto_start(self, service_name: str) -> None:
        self._sc("config", service_name, "start= disabled")

    def set_restart_on_failure(
        self, service_name: str, *, max_attempts: int, reset_window_s: int, delay_s: int = 60,
    ) -> None:
        actions = "/".join([f"restart/{delay_s * 1000}"] * max_attempts)
        self._sc("failure", service_name, f"reset= {reset_window_s}", f"actions= {actions}")
        self._sc("failureflag", service_name, "1")  # only non-clean exits count

    def set_service_dependencies(self, service_name: str, *, requires_network: bool) -> None:
        if not requires_network:
            return
        self._sc("config", service_name, "depend= Dnscache/NlaSvc")

    # -- lock-file mutex ---------------------------------------------------

    @contextlib.contextmanager
    def acquire_exclusive_lock(self, lock_path: str):
        import os
        import msvcrt
        from pathlib import Path
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        # Ensure the file has >=1 byte to lock; the byte-range lock on byte 0
        # is the mutex (we do NOT write under the lock — that would move the
        # file position and desync the lock/unlock byte range).
        if not os.path.exists(lock_path):
            with open(lock_path, "w", encoding="utf-8") as _init:
                _init.write("0")
        f = open(lock_path, "r+", encoding="utf-8")
        try:
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            f.close()
