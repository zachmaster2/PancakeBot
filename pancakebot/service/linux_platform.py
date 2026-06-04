"""Linux (systemd) implementation of ``ServicePlatform``.

Service management is via ``systemctl`` shell-outs; health signaling is via
``sd_notify`` (raw ``AF_UNIX`` datagram to ``$NOTIFY_SOCKET`` — no
``systemd-python`` C-extension dependency). Process-tree kill is delegated to
the systemd unit cgroup (``KillMode=control-group``), so the ``KillTree`` is a
no-op under systemd; the standalone fallback is ``start_new_session=True``.
The Live/Dry mutex lock-file primitive uses ``fcntl.flock``.

Unit files render to ``/etc/systemd/system/<name>.service``; restart/dependency
policy is applied via systemd drop-in overrides
(``/etc/systemd/system/<name>.service.d/override.conf``) so it composes with a
re-rendered base unit.
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
from pathlib import Path

from pancakebot.service.platform_base import (
    HealthState,
    KillTree,
    ServicePlatform,
    ServiceSpec,
    ServiceState,
)

_UNIT_DIR = Path("/etc/systemd/system")

# systemctl ActiveState/SubState -> normalized ServiceState.
_ACTIVE_STATE_MAP = {
    "active": ServiceState.RUNNING,
    "activating": ServiceState.STARTING,
    "deactivating": ServiceState.STOPPING,
    "inactive": ServiceState.STOPPED,
    "failed": ServiceState.FAILED,
}


class _LinuxKillTree(KillTree):
    """No-op under systemd: the unit cgroup (KillMode=control-group) kills all
    unit processes on stop/crash. ``adopt`` records the pid for the standalone
    fallback only."""

    def __init__(self) -> None:
        self.adopted_pids: list[int] = []

    def adopt(self, proc) -> None:  # noqa: ANN001
        try:
            self.adopted_pids.append(int(proc.pid))
        except Exception:  # noqa: BLE001
            pass


class LinuxServicePlatform(ServicePlatform):
    name = "linux"

    def __init__(self, *, unit_dir: Path = _UNIT_DIR, runner=None) -> None:
        self._unit_dir = unit_dir
        # Single seam for testability: every systemctl invocation goes here.
        self._run = runner if runner is not None else self._default_run

    # -- subprocess seam ---------------------------------------------------

    @staticmethod
    def _default_run(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def _systemctl(self, *args: str) -> subprocess.CompletedProcess:
        return self._run(["systemctl", *args])

    def _unit_name(self, service_name: str) -> str:
        return service_name if service_name.endswith(".service") else f"{service_name}.service"

    # -- unit rendering ----------------------------------------------------

    def render_unit(self, spec: ServiceSpec) -> str:
        """Render a systemd unit file from a ServiceSpec (pure; no I/O)."""
        after = ["network-online.target", "nss-lookup.target"] if spec.requires_network else []
        wants = ["network-online.target"] if spec.requires_network else []
        exec_start = " ".join([spec.exe_path, *spec.args])
        lines = ["[Unit]", f"Description={spec.description}"]
        if after:
            lines.append(f"After={' '.join(after)}")
        if wants:
            lines.append(f"Wants={' '.join(wants)}")
        if spec.conflicts_with:
            lines.append(f"Conflicts={self._unit_name(spec.conflicts_with)}")
        lines += [
            "",
            "[Service]",
            "Type=notify",                       # supervisor uses sd_notify(READY=1)
            f"WorkingDirectory={spec.working_dir}",
            f"ExecStart={exec_start}",
            "Restart=on-failure",
            f"RestartSec={spec.restart_delay_s}",
            f"StartLimitIntervalSec={spec.restart_reset_window_s}",
            f"StartLimitBurst={spec.restart_max_attempts}",
            "KillMode=control-group",            # cgroup tree-kill (Job Object equiv)
            "TimeoutStopSec=25",
        ]
        if spec.env:
            for k, v in spec.env.items():
                lines.append(f"Environment={k}={v}")
        lines += ["", "[Install]", "WantedBy=multi-user.target", ""]
        return "\n".join(lines)

    def _unit_path(self, name: str) -> Path:
        return self._unit_dir / self._unit_name(name)

    # -- ServicePlatform: management ---------------------------------------

    def install_service(self, spec: ServiceSpec) -> None:
        self._unit_path(spec.name).write_text(self.render_unit(spec), encoding="utf-8")
        self._systemctl("daemon-reload")

    def uninstall_service(self, service_name: str) -> None:
        self._systemctl("disable", "--now", self._unit_name(service_name))
        with contextlib.suppress(FileNotFoundError):
            self._unit_path(service_name).unlink()
        self._systemctl("daemon-reload")

    def start_service(self, service_name: str) -> None:
        self._systemctl("start", self._unit_name(service_name))

    def stop_service(self, service_name: str, timeout_s: float = 30.0) -> bool:
        # systemctl stop blocks until inactive (or its own timeout).
        self._systemctl("stop", self._unit_name(service_name))
        return self.service_status(service_name) == ServiceState.STOPPED

    def service_status(self, service_name: str) -> ServiceState:
        res = self._systemctl("show", "-p", "ActiveState", "--value", self._unit_name(service_name))
        state = (res.stdout or "").strip().splitlines()
        token = state[0].strip() if state else ""
        return _ACTIVE_STATE_MAP.get(token, ServiceState.UNKNOWN)

    def enable_auto_start(self, service_name: str) -> None:
        self._systemctl("enable", self._unit_name(service_name))

    def disable_auto_start(self, service_name: str) -> None:
        self._systemctl("disable", self._unit_name(service_name))

    def set_restart_on_failure(
        self, service_name: str, *, max_attempts: int, reset_window_s: int, delay_s: int = 60,
    ) -> None:
        self._write_dropin(service_name, "restart", (
            "[Service]\n"
            "Restart=on-failure\n"
            f"RestartSec={delay_s}\n"
            f"StartLimitIntervalSec={reset_window_s}\n"
            f"StartLimitBurst={max_attempts}\n"
        ))

    def set_service_dependencies(self, service_name: str, *, requires_network: bool) -> None:
        if not requires_network:
            return
        self._write_dropin(service_name, "deps", (
            "[Unit]\n"
            "After=network-online.target nss-lookup.target\n"
            "Wants=network-online.target\n"
        ))

    def _write_dropin(self, service_name: str, key: str, content: str) -> None:
        d = self._unit_dir / f"{self._unit_name(service_name)}.d"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.conf").write_text(content, encoding="utf-8")
        self._systemctl("daemon-reload")

    # -- ServicePlatform: supervisor runtime -------------------------------

    def signal_health(self, state: HealthState) -> None:
        payload = {
            HealthState.READY: "READY=1",
            HealthState.RUNNING: "WATCHDOG=1",
            HealthState.STOPPING: "STOPPING=1",
            HealthState.EXTEND: "EXTEND_TIMEOUT_USEC=20000000",
        }.get(state)
        if payload is None:
            return
        self._sd_notify(payload)

    @staticmethod
    def _sd_notify(payload: str) -> None:
        """Send a datagram to $NOTIFY_SOCKET (best-effort; no-op if unset)."""
        addr = os.environ.get("NOTIFY_SOCKET")
        if not addr:
            return
        # Abstract namespace sockets start with '@' -> leading NUL.
        if addr.startswith("@"):
            addr = "\0" + addr[1:]
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                sock.sendto(payload.encode("utf-8"), addr)
            finally:
                sock.close()
        except OSError:
            pass

    def create_kill_tree(self) -> KillTree:
        return _LinuxKillTree()

    def spawn_kwargs(self) -> dict:
        # New session so the standalone (non-systemd) fallback can killpg;
        # under systemd the cgroup handles tree-kill regardless.
        return {"start_new_session": True}

    def is_service_active(self, service_name: str) -> bool:
        res = self._systemctl("is-active", self._unit_name(service_name))
        return (res.stdout or "").strip() in ("active", "activating")

    # -- lock-file mutex ---------------------------------------------------

    @contextlib.contextmanager
    def acquire_exclusive_lock(self, lock_path: str):
        import fcntl
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "w", encoding="utf-8")
        try:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                f.write(str(os.getpid()))
                f.flush()
                yield True
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()
