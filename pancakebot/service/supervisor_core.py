"""OS-agnostic supervision core, shared by the Windows Service host and the
Linux ``supervise`` entry point.

``SupervisorCore`` owns the entire supervision lifecycle — mode mutex,
first-run classification, bot-child spawn/monitor/restart, the fast/slow
crashloop limiter, and the full Discord alert taxonomy — and calls through a
``ServicePlatform`` for everything OS-specific (stop signal, health signaling,
process-tree kill, detached-spawn flags, cross-service query). The logic and
the alerts are byte-for-byte the same on both OSes.

Wiring:
  Windows:  pywin32 ServiceFramework (common.py) -> SupervisorCore -> run.py
  Linux:    systemd unit -> supervise.py -> SupervisorCore -> run.py
"""
from __future__ import annotations

import datetime
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from pancakebot.service import notifications, supervision
from pancakebot.service.platform_base import HealthState, ServicePlatform

# ---------------------------------------------------------------------------
# Tunables (constants; change here, redeploy)
# ---------------------------------------------------------------------------

_POLL_INTERVAL_S: float = 1.0
# Popen-based classifier grace (see supervision.DEFAULT_RUN_GRACE_S).
_STARTUP_GRACE_S: float = supervision.DEFAULT_RUN_GRACE_S
# Stop budget: give the bot time to flush in-flight atomic writes; hard kill
# after this. On Windows it stays under the 30s SCM SvcStop deadline.
_STOP_GRACE_S: float = 20.0
# Crashloop limiter (matches legacy supervisor defaults).
_FAST_RESTART_MAX: int = 3
_FAST_RESTART_WINDOW_S: float = 15 * 60.0
_SLOW_RESTART_MAX: int = 8
_SLOW_RESTART_WINDOW_S: float = 24 * 3600.0
# REBOOTED detection: system-uptime threshold.
_REBOOT_DETECT_S: float = 10 * 60.0
# Mode-mutex eviction timeout.
_MUTEX_EVICT_TIMEOUT_S: float = 30.0


class SupervisorCore:
    """Runs the supervision loop for one mode (``live``/``dry``)."""

    def __init__(
        self,
        *,
        mode: str,
        other_service: str,
        platform: ServicePlatform,
        repo_root: Path,
        venv_python: Path,
        service_name: str,
        log: Callable[[str, str], None],
    ) -> None:
        self._mode = mode
        self._other = other_service
        self._platform = platform
        self._repo_root = repo_root
        self._venv_python = venv_python
        self._svc_name = service_name
        self._log = log
        self._stop_event = platform.create_stop_event()
        self._kill_tree = platform.create_kill_tree()
        self._bot_proc: subprocess.Popen | None = None
        self._bot_started_at: float | None = None
        self._stop_requested = False

    # -- public lifecycle --------------------------------------------------

    def request_stop(self) -> None:
        """Signal the loop to stop. Minimal + re-entrancy-safe (callable from a
        SIGTERM handler): the child drain happens in ``run``'s exit path, not
        here. Called from the OS stop path (SvcStop / SIGTERM)."""
        self._platform.signal_health(HealthState.STOPPING)
        self._log("INFO", f"{self._svc_name}: stop requested")
        self._stop_requested = True
        self._stop_event.set()

    def run(self) -> None:
        """Supervision main loop. Returns on stop; raises on fatal spawn
        failure (so the OUTER supervisor — SCM / systemd — marks failed)."""
        art = supervision.artifacts_for_mode(self._mode)
        art["logs_dir"].mkdir(parents=True, exist_ok=True)

        if not self._enforce_mode_mutex():
            notifications.notify(
                mode=self._mode, kind="MODE_TRANSITION_REFUSED",
                detail=f"{self._other} is running", art=art,
            )
            return

        first_kind = self._classify_first_run(art)
        notifications.notify(mode=self._mode, kind=first_kind, art=art)

        self._archive_lingering_crash_file(art["crash"])

        try:
            self._spawn_bot_child(art)
        except Exception as e:  # noqa: BLE001
            self._log("ERROR", f"{self._svc_name}: initial spawn failed: {e!r}")
            notifications.notify(
                mode=self._mode, kind="SPAWN_FAILED",
                fields={"spawn_error": f"{type(e).__name__}: {e}"}, art=art,
            )
            raise

        # E: if the prior shutdown left an intentional-restart marker, this
        # start is a deploy / admin restart (not a crash recovery) — clear the
        # OUTER start-limit counter so intentional restarts don't exhaust it.
        self._consume_intentional_restart_marker(art)

        while not self._stop_requested:
            if self._stop_event.wait(_POLL_INTERVAL_S):
                break  # stop signaled
            if self._stop_requested:
                break
            try:
                status, fields = supervision.classify_running_bot(
                    self._bot_proc, self._bot_started_at, art,
                    startup_grace_s=_STARTUP_GRACE_S,
                )
            except Exception as e:  # noqa: BLE001
                self._log("ERROR", f"{self._svc_name}: classify_running_bot raised: {e!r}")
                continue

            if status in ("UP", "STARTING"):
                continue
            if status in ("CRASHED", "DOWN"):
                self._handle_unhealthy(status, fields, art)
                continue

        # Stop path: drain the child (kept here, not in request_stop, so a
        # SIGTERM handler stays minimal/re-entrancy-safe), then notify STOPPED.
        # This path is only reached when the run loop exits via _stop_requested
        # (SIGTERM/SCM stop / mode mutex / deploy) -> intentional (D4: INFO). A
        # child crash triggers CRASHED+restart, not STOPPED; a supervisor crash
        # is caught by the entrypoint wrapper as SERVICE_CRASHED.
        # E: mark this as an intentional stop so the NEXT start clears the
        # OUTER start-limit counter (deploys/admin restarts don't count toward
        # it; only crashes do, which leave no marker).
        self._write_intentional_restart_marker(art)
        self._stop_bot_child(reason="stop")
        notifications.notify(
            mode=self._mode, kind="STOPPED",
            fields={"intentional": self._stop_requested}, art=art,
        )
        self._log("INFO", f"{self._svc_name}: clean exit")

    # -- intentional-restart marker (start-limit handshake) ----------------

    def _restart_marker_path(self, art: dict) -> Path:
        # Sits beside the crash artifact in var/<mode>/.
        return art["crash"].parent / ".intentional_restart"

    def _write_intentional_restart_marker(self, art: dict) -> None:
        marker = self._restart_marker_path(art)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("intentional", encoding="utf-8")
        except OSError as e:  # noqa: BLE001
            self._log("WARN", f"{self._svc_name}: could not write restart marker: {e!r}")

    def _consume_intentional_restart_marker(self, art: dict) -> None:
        marker = self._restart_marker_path(art)
        if not marker.exists():
            return  # prior shutdown was a crash (or first run) -> count it
        try:
            marker.unlink()
        except OSError:  # noqa: BLE001
            pass
        try:
            self._platform.clear_restart_counter(self._svc_name)
            self._log("INFO", f"{self._svc_name}: intentional restart — start-limit counter cleared")
        except Exception as e:  # noqa: BLE001
            self._log("WARN", f"{self._svc_name}: clear_restart_counter failed: {e!r}")

    # -- mode mutex --------------------------------------------------------

    def _enforce_mode_mutex(self) -> bool:
        """Live evicts Dry; Dry yields to Live. Returns True to proceed."""
        if self._mode == "live":
            if self._platform.is_service_active(self._other):
                self._log("INFO", f"{self._svc_name}: stopping {self._other} (live priority)")
                art = supervision.artifacts_for_mode(self._mode)
                notifications.notify(
                    mode="live", kind="MODE_TRANSITION",
                    detail=f"stopping {self._other} to start live", art=art,
                )
                ok = self._platform.stop_service(self._other, _MUTEX_EVICT_TIMEOUT_S)
                if not ok:
                    self._log(
                        "ERROR",
                        f"{self._svc_name}: failed to stop {self._other} within "
                        f"{_MUTEX_EVICT_TIMEOUT_S}s; proceeding anyway",
                    )
            return True
        # Dry mode: refuse if Live is active.
        if self._platform.is_service_active(self._other):
            self._log("INFO", f"{self._svc_name}: refusing to start, {self._other} is running")
            return False
        return True

    # -- first-run classification -----------------------------------------

    def _classify_first_run(self, art: dict[str, Path]) -> str:
        try:
            import psutil
            uptime_s = time.time() - psutil.boot_time()
        except Exception:  # noqa: BLE001
            uptime_s = float("inf")  # conservative — assume warm system
        if uptime_s < _REBOOT_DETECT_S:
            return "REBOOTED"
        if art["crash"].exists():
            return "RECOVERY_AFTER_CRASH"
        return "STARTED"

    # -- bot child management ---------------------------------------------

    def _spawn_bot_child(self, art: dict[str, Path]) -> None:
        logs_dir = art["logs_dir"]
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_log = logs_dir / f"{self._mode}-svc-{ts}.log"
        err_log = logs_dir / f"{self._mode}-svc-{ts}_err.log"

        out_f = open(out_log, "w", encoding="utf-8")
        err_f = open(err_log, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [str(self._venv_python), "-u", "run.py", f"--{self._mode}"],
                cwd=str(self._repo_root),
                stdout=out_f,
                stderr=err_f,
                **self._platform.spawn_kwargs(),
            )
        finally:
            out_f.close()
            err_f.close()

        # Enroll in the kill-on-supervisor-exit tree (Job Object / cgroup).
        self._kill_tree.adopt(proc)

        self._bot_proc = proc
        self._bot_started_at = time.time()
        self._log("INFO", f"{self._svc_name}: spawned bot child pid={proc.pid} log={out_log.name}")

    def _stop_bot_child(self, reason: str) -> None:
        proc = self._bot_proc
        if proc is None or proc.poll() is not None:
            self._bot_proc = None
            self._bot_started_at = None
            return
        self._log("INFO", f"{self._svc_name}: stopping bot child pid={proc.pid} reason={reason}")
        try:
            proc.terminate()
        except Exception as e:  # noqa: BLE001
            self._log("ERROR", f"{self._svc_name}: terminate() raised: {e!r}")

        deadline = time.time() + _STOP_GRACE_S
        while time.time() < deadline:
            if proc.poll() is not None:
                self._bot_proc = None
                self._bot_started_at = None
                return
            # Keep the OUTER supervisor's stop deadline from firing while we
            # reap (Win: SCM STOP_PENDING pump; Linux: sd_notify EXTEND).
            self._platform.signal_health(HealthState.EXTEND)
            time.sleep(0.25)

        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._bot_proc = None
        self._bot_started_at = None

    # -- crashloop limiter -------------------------------------------------

    def _handle_unhealthy(self, status: str, fields: dict[str, Any], art: dict[str, Path]) -> None:
        try:
            now = time.time()
            history = supervision.read_restart_history(art["restart_history"])
            history = supervision.prune_history(history, now, _SLOW_RESTART_WINDOW_S)

            fast_count = supervision.count_within(history, now, _FAST_RESTART_WINDOW_S)
            if fast_count >= _FAST_RESTART_MAX:
                supervision.write_restart_history(art["restart_history"], history)
                notifications.notify(
                    mode=self._mode, kind="SUPPRESSED_FAST_CRASHLOOP", fields=fields, art=art,
                    detail=f"{fast_count} restarts in {_FAST_RESTART_WINDOW_S/60:.0f}min "
                    f">= {_FAST_RESTART_MAX}; not respawning",
                )
                return

            slow_count = supervision.count_within(history, now, _SLOW_RESTART_WINDOW_S)
            escalate_slow = slow_count >= _SLOW_RESTART_MAX

            recent_restarts_1h = supervision.count_within(history, now, 3600.0)
            should_notify_status = recent_restarts_1h >= 2
            if should_notify_status:
                notifications.notify(mode=self._mode, kind=status, fields=fields, art=art)
            else:
                self._log(
                    "INFO",
                    f"{self._svc_name}: {status} respawn "
                    f"(recent_restarts_1h={recent_restarts_1h + 1}/3, Discord suppressed)",
                )

            self._stop_bot_child(reason=f"unhealthy:{status}")
            self._archive_lingering_crash_file(art["crash"])
            try:
                self._spawn_bot_child(art)
                new_pid = self._bot_proc.pid if self._bot_proc else None
            except Exception as e:  # noqa: BLE001
                self._log("ERROR", f"{self._svc_name}: respawn failed: {e!r}")
                notifications.notify(
                    mode=self._mode, kind="SPAWN_FAILED",
                    fields={"spawn_error": f"{type(e).__name__}: {e}"}, art=art,
                )
                return

            history.append({
                "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ts_wall": now,
                "trigger": status,
                "new_pid": new_pid,
            })
            supervision.write_restart_history(art["restart_history"], history)

            if escalate_slow:
                notifications.notify(
                    mode=self._mode, kind="SLOW_CRASHLOOP_WARNING", fields=fields, art=art,
                    detail=f"{slow_count} restarts in {_SLOW_RESTART_WINDOW_S/3600:.0f}h "
                    f">= {_SLOW_RESTART_MAX}",
                )
        finally:
            # Restore the OUTER supervisor to RUNNING after _stop_bot_child's
            # transient STOP_PENDING, unless we're already shutting down.
            if not self._stop_requested:
                self._platform.signal_health(HealthState.READY)

    # -- crash artifact archival ------------------------------------------

    def _archive_lingering_crash_file(self, crash_path: Path) -> None:
        try:
            from pancakebot.runtime.supervisor_artifacts import archive_lingering_crash_file
            archive_lingering_crash_file(crash_path)
        except Exception as e:  # noqa: BLE001
            self._log("WARN", f"{self._svc_name}: archive_lingering_crash_file raised: {e!r}")
