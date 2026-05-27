"""Shared Windows Service base class for PancakeBot Live / Dry supervision.

``_PancakeBotServiceBase`` is the common ServiceFramework subclass; the
per-mode classes (``PancakeBotLiveService`` / ``PancakeBotDryService``) set
``_MODE`` and ``_OTHER_SERVICE`` and inherit everything else.

Key behaviors:
- 1-second supervision loop polling ``classify_running_bot`` (Popen-based,
  no filesystem race; replaced the legacy artifact-based ``classify_state``
  on 2026-05-23 after a post-reboot DOWN-race triggered spurious respawns).
- Bot child spawned with ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``
  AND immediately assigned to a Windows Job Object configured with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. When the service process exits
  for ANY reason (clean SvcStop, crash, kill -9), Windows automatically
  kills every process in the job — no orphaned bot children.
- Stop signaling: ``Popen.terminate()`` (Windows: ``TerminateProcess``).
  Services run console-less, so ``CTRL_BREAK_EVENT`` is not viable
  without ``AllocConsole`` ceremony. The bot's atomic-write state means
  nothing meaningful is lost by hard-kill: heartbeat, PID file, bankroll,
  trades.csv, crash.json all use tempfile + os.replace atomic rename.
- SCM stop grace: 20s (well under the 30s SCM default; bumps
  STOP_PENDING during the wait so SCM doesn't time out).
- Mode mutex: SCM-state-based. Live evicts Dry on start; Dry yields to
  Live (see ``_enforce_mode_mutex``).
- Crashloop limiter: same fast/slow tier semantics as legacy
  ``scripts/supervisor.py:_do_restart`` (3 fast/15min suppress, 8 slow/24h
  escalate alert).
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Win32 / pywin32 imports. Module-load failure on non-Windows is intentional
# — this module is only imported in service hosts.
import servicemanager
import win32event
import win32job
import win32service
import win32serviceutil

from pancakebot.service import notifications, supervision

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Path to the venv python interpreter used to spawn the bot child.
# CANNOT use ``sys.executable`` here — when the service is hosted by
# pythonservice.exe, ``sys.executable`` resolves to pythonservice.exe
# itself, which has its own arg parser and refuses to run run.py.
_VENV_PYTHON = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"


# ---------------------------------------------------------------------------
# Tunables (constants, not exposed via config — change here, redeploy service)
# ---------------------------------------------------------------------------

_POLL_INTERVAL_S: float = 1.0
_STALE_THRESHOLD_S: float = supervision.DEFAULT_STALE_THRESHOLD_S
# Popen-based classifier grace (30s default — tighter than the legacy 90s
# because we know exactly when the bot was spawned). See
# supervision.DEFAULT_RUN_GRACE_S for rationale.
_STARTUP_GRACE_S: float = supervision.DEFAULT_RUN_GRACE_S

# Stop budget: 20s gives the bot time to flush any in-flight atomic writes
# without exceeding SCM's 30s SvcStop deadline. Hard kill after this.
_STOP_GRACE_S: float = 20.0

# Crashloop limiter (matches legacy supervisor defaults).
_FAST_RESTART_MAX: int = 3
_FAST_RESTART_WINDOW_S: float = 15 * 60.0
_SLOW_RESTART_MAX: int = 8
_SLOW_RESTART_WINDOW_S: float = 24 * 3600.0

# REBOOTED detection: system uptime threshold. Below this on first
# SvcDoRun → REBOOTED notification; above → STARTED (manual enable).
_REBOOT_DETECT_S: float = 10 * 60.0

# Mode-mutex eviction timeout: how long to wait for the other service to
# transition to STOPPED after we ControlService(STOP) it.
_MUTEX_EVICT_TIMEOUT_S: float = 30.0


# ---------------------------------------------------------------------------
# SCM helpers (used by mode mutex)
# ---------------------------------------------------------------------------

def _query_service_state(svc_name: str) -> int | None:
    """Return the current state of ``svc_name`` (a SERVICE_* constant) or None if absent."""
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


def _stop_service_and_wait(svc_name: str, timeout_s: float) -> bool:
    """ControlService(STOP) on ``svc_name`` and poll until STOPPED or timeout."""
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
                # May already be stopping; fall through to the poll loop.
                pass
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    status = win32service.QueryServiceStatusEx(svc)
                    state = int(status["CurrentState"])
                    if state == win32service.SERVICE_STOPPED:
                        return True
                except Exception:
                    return False
                time.sleep(0.5)
            return False
        finally:
            win32service.CloseServiceHandle(svc)
    finally:
        win32service.CloseServiceHandle(scm)


# ---------------------------------------------------------------------------
# Service base class
# ---------------------------------------------------------------------------

class _PancakeBotServiceBase(win32serviceutil.ServiceFramework):
    """Abstract base. Subclasses set ``_svc_name_``, ``_svc_display_name_``,
    ``_MODE`` (``"live"`` / ``"dry"``), and ``_OTHER_SERVICE`` (the other
    mode's SCM service name, for the mutex check)."""

    # Set by subclasses:
    _MODE: str = ""
    _OTHER_SERVICE: str = ""

    def __init__(self, args):
        super().__init__(args)
        self._stop_event = win32event.CreateEvent(None, 0, 0, None)
        self._bot_proc: subprocess.Popen | None = None
        self._bot_started_at: float | None = None
        self._stop_requested: bool = False
        # Job Object with KILL_ON_JOB_CLOSE — when this service process
        # exits for ANY reason (clean SvcStop, crash, force-kill), Windows
        # closes the job handle and automatically kills every child
        # process in the job. This is what guarantees we never orphan a
        # bot subprocess if the supervisor dies (the symptom we hit on
        # 2026-05-23 when an unguarded sys.stderr.write crashed SvcDoRun
        # and left bot PID 7480 running detached).
        self._job = self._create_kill_on_close_job()

    @staticmethod
    def _create_kill_on_close_job():
        """Create a Job Object configured to kill all members on close."""
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
        return job

    # -- SCM entry points --------------------------------------------------

    def SvcStop(self):
        """SCM-initiated stop. Drain child gracefully, then signal main loop."""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        servicemanager.LogInfoMsg(f"{self._svc_name_}: SvcStop received")
        self._stop_requested = True
        # Begin tearing down the bot child *now* — don't wait for the main
        # loop to come around. The main loop will see _stop_requested and
        # exit cleanly.
        self._stop_bot_child(reason="SvcStop")
        win32event.SetEvent(self._stop_event)

    def SvcDoRun(self):
        """Service main entrypoint. Runs until SvcStop fires."""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        try:
            self._run_supervision_loop()
        except Exception as e:
            tb = traceback.format_exc()
            servicemanager.LogErrorMsg(
                f"{self._svc_name_}: SvcDoRun raised {type(e).__name__}: {e}\n{tb}"
            )
            notifications.notify_service_error(mode=self._MODE, exc=e)
            # Re-raise so SCM marks the service as failed and recovery actions kick in.
            raise

    # -- Supervision loop --------------------------------------------------

    def _run_supervision_loop(self) -> None:
        art = supervision.artifacts_for_mode(self._MODE)
        art["logs_dir"].mkdir(parents=True, exist_ok=True)

        # Mode mutex: enforce live-priority. Dry may exit early here.
        if not self._enforce_mode_mutex():
            # Dry refused to start (Live is running). Clean exit.
            notifications.notify(
                mode=self._MODE,
                kind="MODE_TRANSITION_REFUSED",
                detail=f"{self._OTHER_SERVICE} is running",
                art=art,
            )
            return

        # Determine first-run state for the initial notification.
        first_kind = self._classify_first_run(art)
        notifications.notify(mode=self._MODE, kind=first_kind, art=art)

        # Archive any stale crash.json from the previous bot generation
        # (matches run.py:archive_stale_crash policy — always archive).
        self._archive_stale_crash(art["crash"])

        # Spawn initial bot child.
        try:
            self._spawn_bot_child(art)
        except Exception as e:
            servicemanager.LogErrorMsg(
                f"{self._svc_name_}: initial spawn failed: {e!r}"
            )
            notifications.notify(
                mode=self._MODE, kind="SPAWN_FAILED",
                fields={"spawn_error": f"{type(e).__name__}: {e}"},
                art=art,
            )
            raise

        # Main loop: poll classify_running_bot every _POLL_INTERVAL_S until stop.
        # classify_running_bot uses Popen.poll() as the authoritative liveness
        # signal — no filesystem race between spawn and first heartbeat write
        # (the bug that caused the post-reboot DOWN cascade 2026-05-23).
        while not self._stop_requested:
            rc = win32event.WaitForSingleObject(
                self._stop_event, int(_POLL_INTERVAL_S * 1000),
            )
            if rc == win32event.WAIT_OBJECT_0:
                break  # SvcStop fired

            # Re-check stop flag — SvcStop sets both the flag and the event,
            # the event wait above can race ahead of the flag's visibility.
            if self._stop_requested:
                break

            try:
                status, fields = supervision.classify_running_bot(
                    self._bot_proc,
                    self._bot_started_at,
                    art,
                    stale_threshold_s=_STALE_THRESHOLD_S,
                    startup_grace_s=_STARTUP_GRACE_S,
                )
            except Exception as e:
                servicemanager.LogErrorMsg(
                    f"{self._svc_name_}: classify_running_bot raised: {e!r}"
                )
                continue

            if status in ("UP", "STARTING"):
                continue

            if status in ("CRASHED", "DOWN"):
                # STALE removed from trigger list 2026-05-27 (Step 27a).
                # Heartbeat-staleness no longer initiates restarts; only
                # actual process death (CRASHED with crash.json present, or
                # DOWN = process dead with no signal) does. classify_running_bot
                # never returns "STALE" anymore.
                self._handle_unhealthy(status, fields, art)
                continue

        # Loop exit path (SvcStop). Notify and report stopped to SCM.
        notifications.notify(mode=self._MODE, kind="STOPPED", art=art)
        servicemanager.LogInfoMsg(f"{self._svc_name_}: clean exit")

    # -- Mode mutex --------------------------------------------------------

    def _enforce_mode_mutex(self) -> bool:
        """Apply live-priority. Returns True if this service may proceed, False if it should exit.

        Live: stops Dry if running, then proceeds.
        Dry: refuses to proceed if Live is running.
        """
        if self._MODE == "live":
            other_state = _query_service_state(self._OTHER_SERVICE)
            if other_state in (
                win32service.SERVICE_RUNNING,
                win32service.SERVICE_START_PENDING,
                win32service.SERVICE_CONTINUE_PENDING,
                win32service.SERVICE_PAUSED,
                win32service.SERVICE_PAUSE_PENDING,
            ):
                servicemanager.LogInfoMsg(
                    f"{self._svc_name_}: stopping {self._OTHER_SERVICE} "
                    f"(live priority)"
                )
                art = supervision.artifacts_for_mode(self._MODE)
                notifications.notify(
                    mode="live",
                    kind="MODE_TRANSITION",
                    detail=f"stopping {self._OTHER_SERVICE} to start live",
                    art=art,
                )
                ok = _stop_service_and_wait(self._OTHER_SERVICE, _MUTEX_EVICT_TIMEOUT_S)
                if not ok:
                    servicemanager.LogErrorMsg(
                        f"{self._svc_name_}: failed to stop {self._OTHER_SERVICE} "
                        f"within {_MUTEX_EVICT_TIMEOUT_S}s; proceeding anyway"
                    )
            return True

        # Dry mode
        other_state = _query_service_state(self._OTHER_SERVICE)
        if other_state in (
            win32service.SERVICE_RUNNING,
            win32service.SERVICE_START_PENDING,
            win32service.SERVICE_CONTINUE_PENDING,
            win32service.SERVICE_PAUSED,
            win32service.SERVICE_PAUSE_PENDING,
        ):
            servicemanager.LogInfoMsg(
                f"{self._svc_name_}: refusing to start, {self._OTHER_SERVICE} is running"
            )
            return False
        return True

    # -- First-run classification (REBOOTED vs STARTED vs RECOVERY) --------

    def _classify_first_run(self, art: dict[str, Path]) -> str:
        """Decide what kind of first-message to send on service start."""
        try:
            import psutil
            uptime_s = time.time() - psutil.boot_time()
        except Exception:
            uptime_s = float("inf")  # be conservative — assume warm system

        if uptime_s < _REBOOT_DETECT_S:
            return "REBOOTED"
        if art["crash"].exists():
            return "RECOVERY_AFTER_CRASH"
        return "STARTED"

    # -- Bot child management ----------------------------------------------

    def _spawn_bot_child(self, art: dict[str, Path]) -> None:
        """Launch ``python -u run.py --<mode>`` as a detached subprocess.

        IMPORTANT: every new bot child MUST be assigned to ``self._job``
        immediately after Popen. The job has KILL_ON_JOB_CLOSE set, so
        if the service crashes after spawning but before the assignment,
        the child would be orphaned. The window between Popen and
        AssignProcessToJobObject is microseconds, but a sufficiently
        unlucky crash there still leaks a process. (Acceptable; the
        leak rate is dominated by the bug-free case.)
        """
        logs_dir = art["logs_dir"]
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_log = logs_dir / f"{self._MODE}-svc-{ts}.log"
        err_log = logs_dir / f"{self._MODE}-svc-{ts}_err.log"

        creationflags = 0
        if sys.platform == "win32":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )

        out_f = open(out_log, "w", encoding="utf-8")
        err_f = open(err_log, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [str(_VENV_PYTHON), "-u", "run.py", f"--{self._MODE}"],
                cwd=str(_REPO_ROOT),
                stdout=out_f,
                stderr=err_f,
                creationflags=creationflags,
            )
        finally:
            out_f.close()
            err_f.close()

        # Immediately enroll the child in the kill-on-close job. Uses
        # Popen._handle (a pywin32 PyHANDLE wrapping the process HANDLE
        # returned by CreateProcess) — converted to int because
        # AssignProcessToJobObject accepts either. Failure here is
        # logged but non-fatal: a child outside the job loses the
        # auto-kill safety net but is otherwise functional.
        try:
            win32job.AssignProcessToJobObject(self._job, int(proc._handle))
        except Exception as e:
            servicemanager.LogWarningMsg(
                f"{self._svc_name_}: AssignProcessToJobObject failed for "
                f"pid={proc.pid}: {type(e).__name__}: {e} — child will be "
                f"orphaned if supervisor crashes"
            )

        self._bot_proc = proc
        self._bot_started_at = time.time()
        servicemanager.LogInfoMsg(
            f"{self._svc_name_}: spawned bot child pid={proc.pid} log={out_log.name}"
        )

    def _stop_bot_child(self, reason: str) -> None:
        """Terminate the bot child. Hard-kill via TerminateProcess.

        Windows services run console-less, so CTRL_BREAK_EVENT requires an
        AllocConsole ceremony we choose not to do (added fragility for
        marginal gain). The bot's atomic-write state means hard-kill is
        safe — nothing meaningful is lost.

        Waits up to ``_STOP_GRACE_S`` for the child to fully reap, bumping
        SCM STOP_PENDING during the wait so the 30s SCM stop deadline
        doesn't trigger.
        """
        proc = self._bot_proc
        if proc is None or proc.poll() is not None:
            self._bot_proc = None
            self._bot_started_at = None
            return

        servicemanager.LogInfoMsg(
            f"{self._svc_name_}: stopping bot child pid={proc.pid} reason={reason}"
        )
        try:
            proc.terminate()  # Windows: TerminateProcess(hProcess, 1)
        except Exception as e:
            servicemanager.LogErrorMsg(
                f"{self._svc_name_}: terminate() raised: {e!r}"
            )

        # Poll for child reap with SCM heartbeating.
        deadline = time.time() + _STOP_GRACE_S
        while time.time() < deadline:
            if proc.poll() is not None:
                self._bot_proc = None
                self._bot_started_at = None
                return
            # Keep SCM alive — STOP_PENDING with a checkpoint advance prevents
            # the SCM stop deadline from firing while we wait.
            try:
                self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            except Exception:
                pass
            time.sleep(0.25)

        # Still alive after grace — issue a second kill (no-op on Windows
        # since terminate==kill, but defensive for completeness).
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        self._bot_proc = None
        self._bot_started_at = None

    # -- Restart-on-unhealthy (crashloop limiter) --------------------------

    def _handle_unhealthy(self, status: str, fields: dict[str, Any], art: dict[str, Path]) -> None:
        """Restart the bot, applying fast/slow crashloop limits.

        Wraps the whole body in try/finally so SCM is restored to
        ``SERVICE_RUNNING`` on EVERY exit path. ``_stop_bot_child`` pushes
        SCM to ``SERVICE_STOP_PENDING`` while it reaps the dead bot
        (necessary to keep SCM's stop-deadline from firing); without an
        explicit restore-to-RUNNING here, SCM would be permanently stuck
        in StopPending while the supervisor keeps happily supervising
        — caught 2026-05-24 weekend when 4 STALE-triggered respawns
        left ``Get-Service PancakeBotLive`` reporting StopPending despite
        the bot being alive and the supervisor functional.

        Guarded by ``not self._stop_requested`` so we don't race SvcStop
        when both fire simultaneously: if SvcStop has begun (it's already
        signaled the stop event), the supervisor is genuinely shutting
        down and we should let it stay in STOP_PENDING.
        """
        try:
            now = time.time()
            history = supervision.read_restart_history(art["restart_history"])
            history = supervision.prune_history(history, now, _SLOW_RESTART_WINDOW_S)

            fast_count = supervision.count_within(history, now, _FAST_RESTART_WINDOW_S)
            if fast_count >= _FAST_RESTART_MAX:
                supervision.write_restart_history(art["restart_history"], history)
                notifications.notify(
                    mode=self._MODE,
                    kind="SUPPRESSED_FAST_CRASHLOOP",
                    fields=fields,
                    art=art,
                    detail=f"{fast_count} restarts in {_FAST_RESTART_WINDOW_S/60:.0f}min "
                    f"≥ {_FAST_RESTART_MAX}; not respawning",
                )
                # Don't spawn. Loop will keep polling — eventually the fast window
                # clears and we'll retry.
                return

            slow_count = supervision.count_within(history, now, _SLOW_RESTART_WINDOW_S)
            escalate_slow = slow_count >= _SLOW_RESTART_MAX

            # Restart-pattern aggregation (Step 27a policy b, 2026-05-27):
            # Discord-notify the underlying status only when this is the 3rd+
            # restart in a 1-hour rolling window. Single isolated restarts
            # (process auto-recovered within seconds) go to log only; the
            # SLOW_CRASHLOOP_WARNING below still fires at the 8/24h pattern
            # for a separate severity signal.
            recent_restarts_1h = supervision.count_within(history, now, 3600.0)
            should_notify_status = recent_restarts_1h >= 2  # this would be the 3rd
            if should_notify_status:
                notifications.notify(mode=self._MODE, kind=status, fields=fields, art=art)
            else:
                servicemanager.LogInfoMsg(
                    f"{self._svc_name_}: {status} respawn "
                    f"(recent_restarts_1h={recent_restarts_1h + 1}/3, "
                    f"Discord suppressed below pattern threshold)"
                )

            # Drain dead child, archive crash, spawn new.
            self._stop_bot_child(reason=f"unhealthy:{status}")
            self._archive_stale_crash(art["crash"])

            try:
                self._spawn_bot_child(art)
                new_pid = self._bot_proc.pid if self._bot_proc else None
            except Exception as e:
                servicemanager.LogErrorMsg(
                    f"{self._svc_name_}: respawn failed: {e!r}"
                )
                notifications.notify(
                    mode=self._MODE, kind="SPAWN_FAILED",
                    fields={"spawn_error": f"{type(e).__name__}: {e}"},
                    art=art,
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
                    mode=self._MODE,
                    kind="SLOW_CRASHLOOP_WARNING",
                    fields=fields,
                    art=art,
                    detail=f"{slow_count} restarts in {_SLOW_RESTART_WINDOW_S/3600:.0f}h "
                    f"≥ {_SLOW_RESTART_MAX}",
                )
        finally:
            # Restore SERVICE_RUNNING after _stop_bot_child's transient
            # STOP_PENDING. Skipped when we're already in SvcStop (race
            # protection: if SvcStop was called, _stop_requested is True
            # and the loop should end with the framework reporting STOPPED).
            if not self._stop_requested:
                try:
                    self.ReportServiceStatus(win32service.SERVICE_RUNNING)
                except Exception:
                    pass

    # -- Crash artifact archival ------------------------------------------

    def _archive_stale_crash(self, crash_path: Path) -> None:
        """Same policy as run.py: always archive any existing crash.json on (re)spawn.

        Delegates to ``process_health.archive_stale_crash`` so the logic is
        single-source-of-truth with run.py.
        """
        try:
            from pancakebot.runtime.process_health import archive_stale_crash
            archive_stale_crash(crash_path)
        except Exception as e:
            # Never block spawn on a cleanup failure.
            servicemanager.LogWarningMsg(
                f"{self._svc_name_}: archive_stale_crash raised: {e!r}"
            )
