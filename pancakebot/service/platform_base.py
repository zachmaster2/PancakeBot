"""Cross-platform service-supervision abstraction.

``ServicePlatform`` is the OS-agnostic contract the supervisor (and the
bootstrap tooling) call through, so the bot core never imports pywin32 /
systemd directly. Two adapters implement it:

- ``WindowsServicePlatform`` (pywin32 SCM / Job Object) — see windows_platform.py
- ``LinuxServicePlatform``   (systemd / systemctl / sd_notify) — see linux_platform.py

``get_platform()`` in ``pancakebot/service/__init__.py`` selects the adapter
by ``sys.platform`` at import time.

Scope note (Phase 1): this introduces the abstraction + both adapters +
factory. The Windows supervisor (``common.py``) delegates its
behaviour-identical primitives (Job-Object kill-tree, SCM mode-mutex query)
to the Windows adapter. The Linux adapter is complete but not yet deployed.
The service-management methods (install/enable/restart-policy/deps) are used
by the Phase-2 bootstrap scripts, not the running supervision loop.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum


class ServiceState(str, Enum):
    """Normalized service state, OS-independent."""
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class HealthState(str, Enum):
    """Supervisor → service-manager health signal.

    Maps to ``ReportServiceStatus`` (Windows SCM) and ``sd_notify`` (systemd):
      READY     -> SERVICE_RUNNING        / "READY=1"
      RUNNING   -> SERVICE_RUNNING        / "WATCHDOG=1" (liveness ping)
      STOPPING  -> SERVICE_STOP_PENDING   / "STOPPING=1"
      EXTEND    -> SERVICE_STOP_PENDING   / "EXTEND_TIMEOUT_USEC=..." (push the
                                            stop deadline back during reap)
    """
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    EXTEND = "EXTEND"


@dataclass(frozen=True)
class ServiceSpec:
    """Declarative description of a service to install. The adapter renders it
    to an SCM service definition (Windows) or a systemd unit file (Linux)."""
    name: str                       # e.g. "pancakebot-live" / "PancakeBotLive"
    description: str
    exe_path: str                   # interpreter (venv python)
    args: tuple[str, ...]           # e.g. ("run.py", "--live")
    working_dir: str
    env: dict[str, str] | None = None
    # Boot dependencies (network/DNS readiness). Adapter maps to the OS form:
    #   Windows: sc.exe depend= (Dnscache/NlaSvc)
    #   Linux:   After=/Wants= (network-online.target nss-lookup.target)
    requires_network: bool = True
    # Mutual exclusion: this service evicts/conflicts the named other service.
    conflicts_with: str | None = None
    # Restart-on-failure policy.
    restart_max_attempts: int = 3
    restart_reset_window_s: int = 86400
    restart_delay_s: int = 60


class KillTree(abc.ABC):
    """Process-group kill guarantee: every adopted child is killed when the
    supervisor process exits for ANY reason.

    Windows: a Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
    Linux:   the systemd unit cgroup (``KillMode=control-group``) does this for
             free; the standalone fallback is a new session / process group.
    """

    @abc.abstractmethod
    def adopt(self, proc) -> None:
        """Enroll a just-spawned ``subprocess.Popen`` into the kill tree.

        Best-effort: a failure means the child loses the auto-kill safety net
        but is otherwise functional; implementations log + continue.
        """


class ServicePlatform(abc.ABC):
    """OS-agnostic service-supervision contract. See module docstring."""

    name: str = "base"

    # -- service management (used by bootstrap / control tooling) ----------

    @abc.abstractmethod
    def install_service(self, spec: ServiceSpec) -> None: ...

    @abc.abstractmethod
    def uninstall_service(self, service_name: str) -> None: ...

    @abc.abstractmethod
    def start_service(self, service_name: str) -> None: ...

    @abc.abstractmethod
    def stop_service(self, service_name: str, timeout_s: float = 30.0) -> bool:
        """Graceful stop; returns True once STOPPED within timeout."""

    @abc.abstractmethod
    def service_status(self, service_name: str) -> ServiceState: ...

    @abc.abstractmethod
    def enable_auto_start(self, service_name: str) -> None: ...

    @abc.abstractmethod
    def disable_auto_start(self, service_name: str) -> None: ...

    @abc.abstractmethod
    def set_restart_on_failure(
        self, service_name: str, *, max_attempts: int, reset_window_s: int,
        delay_s: int = 60,
    ) -> None: ...

    @abc.abstractmethod
    def set_service_dependencies(self, service_name: str, *, requires_network: bool) -> None: ...

    # -- supervisor runtime primitives (used by the supervision loop) ------

    @abc.abstractmethod
    def signal_health(self, state: HealthState) -> None:
        """Notify the service manager of supervisor health (best-effort)."""

    @abc.abstractmethod
    def create_kill_tree(self) -> KillTree:
        """Construct the kill-all-children-on-supervisor-exit primitive."""

    @abc.abstractmethod
    def spawn_kwargs(self) -> dict:
        """OS-specific ``subprocess.Popen`` kwargs to detach the child into its
        own process group/session (so the kill tree can reap it)."""

    @abc.abstractmethod
    def is_service_active(self, service_name: str) -> bool:
        """True if the named service is RUNNING/STARTING (mode-mutex query)."""

    # -- lock-file mutex primitive (Live/Dry exclusion fallback) -----------

    @abc.abstractmethod
    def acquire_exclusive_lock(self, lock_path: str):
        """Return a context manager that holds an exclusive OS lock on
        ``lock_path`` for its lifetime (fcntl on Linux, msvcrt on Windows),
        releasing on exit. Yields True if acquired, False if already held by
        another process."""
